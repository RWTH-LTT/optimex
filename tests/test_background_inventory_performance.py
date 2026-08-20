"""Tests for the batched, cached background inventory calculation."""

from datetime import datetime
from pathlib import Path

import bw2calc as bc
import bw2data as bd
import numpy as np
import pyomo.environ as pyo
import pytest
from bw_temporalis import TemporalDistribution

from optimex import converter, lca_processor, optimizer
from optimex.lca_processor import (
    LCAConfig,
    LCADataProcessor,
    clear_lca_caches,
    compute_db_inventory_entries,
)


def _config(**background_inventory):
    """Config mirroring the fixture project, with overridable inventory settings.

    The on-disk cache is off unless a test asks for it, so that each test really
    runs the calculation it is about.
    """
    background_inventory.setdefault("use_disk_cache", False)
    years = range(2020, 2030)
    td_demand = TemporalDistribution(
        date=np.array(
            [datetime(year, 1, 1).isoformat() for year in years], dtype="datetime64[s]"
        ),
        amount=np.asarray([0, 0, 10, 5, 10, 5, 10, 5, 10, 5]),
    )
    return LCAConfig(
        demand={bd.get_node(database="foreground", code="R1"): td_demand},
        temporal={
            "start_date": datetime(2020, 1, 1),
            "temporal_resolution": "year",
            "time_horizon": 100,
        },
        characterization_methods=[
            {
                "category_name": "land_use",
                "brightway_method": ("land use", "example"),
            },
        ],
        background_inventory=background_inventory,
    )


def test_inventory_matches_bw2calc(setup_brightway_databases):
    """The aggregated inventory must equal what bw2calc computes for the same demand."""
    clear_lca_caches()
    entries = compute_db_inventory_entries(
        "db_2020",
        {
            "I1": {"name": "node I1", "reference product": "I1"},
            "I2": {"name": "node I2", "reference product": "I2"},
        },
        biosphere_db_name="biosphere3",
    )

    for code, identity in (
        ("I1", ("node I1", "I1", None)),
        ("I2", ("node I2", "I2", None)),
    ):
        lca = bc.LCA({bd.get_node(database="db_2020", code=code): 1})
        lca.lci()
        expected = {
            bd.get_node(id=lca.dicts.biosphere.reversed[row])["code"]: amount
            for row, amount in enumerate(np.asarray(lca.inventory.sum(axis=1)).ravel())
            if amount
        }
        computed = {flow: amount for flow, (_, amount) in entries[identity].items()}
        assert computed.keys() == expected.keys()
        for flow, amount in computed.items():
            assert amount == pytest.approx(expected[flow])


def test_repeated_processing_uses_cache(setup_brightway_databases, monkeypatch):
    """A second run in the same session must not recompute any inventory."""
    clear_lca_caches()
    LCADataProcessor(_config())

    calls = []
    monkeypatch.setattr(
        lca_processor,
        "compute_db_inventory_entries",
        lambda *args, **kwargs: calls.append(args) or {},
    )
    processor = LCADataProcessor(_config())

    assert calls == []
    assert processor.background_inventory


def test_parallel_matches_sequential(setup_brightway_databases):
    """Both calculation methods must produce the same tensor."""
    clear_lca_caches()
    sequential = LCADataProcessor(_config(calculation_method="sequential"))
    clear_lca_caches()
    parallel = LCADataProcessor(_config(calculation_method="parallel", n_jobs=2))

    assert parallel.background_inventory == sequential.background_inventory
    assert parallel.elementary_flows == sequential.elementary_flows


def test_cutoff_keeps_largest_flows(setup_brightway_databases):
    """A cutoff retains the requested number of flows per intermediate flow."""
    clear_lca_caches()
    entries = compute_db_inventory_entries(
        "db_2020",
        {"I1": {"name": "node I1", "reference product": "I1"}},
        cutoff=1,
        biosphere_db_name="biosphere3",
    )
    assert len(next(iter(entries.values()))) == 1


def test_uncharacterized_flows_are_dropped(setup_brightway_databases):
    """Flows without a characterization factor only inflate the model."""
    bd.Method(("only CO2", "example")).write([(("biosphere3", "CO2"), 1)])

    clear_lca_caches()
    config = _config()
    config.characterization_methods[0].brightway_method = ("only CO2", "example")
    processor = LCADataProcessor(config)

    assert "CO2" in processor.elementary_flows
    assert "CH4" not in processor.elementary_flows
    assert not [key for key in processor.background_inventory if key[2] == "CH4"]


def test_retain_flows_keeps_uncharacterized_flow(setup_brightway_databases):
    """Flows needed for flow limits can be kept explicitly."""
    bd.Method(("only CO2", "example")).write([(("biosphere3", "CO2"), 1)])

    clear_lca_caches()
    config = _config(retain_flows=["CH4"])
    config.characterization_methods[0].brightway_method = ("only CO2", "example")
    processor = LCADataProcessor(config)

    assert "CH4" in processor.elementary_flows

    clear_lca_caches()
    config = _config(restrict_to_characterized_flows=False)
    config.characterization_methods[0].brightway_method = ("only CO2", "example")
    processor = LCADataProcessor(config)

    assert "CH4" in processor.elementary_flows


def test_disk_cache_survives_a_cleared_session(setup_brightway_databases, monkeypatch):
    """A new session reads inventories from disk instead of recalculating them."""
    clear_lca_caches(include_disk=True)
    reference = LCADataProcessor(_config(use_disk_cache=True))

    clear_lca_caches()  # as if a new Python session started
    monkeypatch.setattr(
        lca_processor,
        "compute_db_inventory_entries",
        lambda *args, **kwargs: pytest.fail("recalculated despite the disk cache"),
    )
    processor = LCADataProcessor(_config(use_disk_cache=True))

    assert processor.background_inventory == reference.background_inventory
    assert processor.elementary_flows == reference.elementary_flows


def test_disk_cache_is_invalidated_by_a_database_edit(setup_brightway_databases):
    """Editing a background database must not serve stale inventories."""
    clear_lca_caches(include_disk=True)
    LCADataProcessor(_config(use_disk_cache=True))

    node = bd.get_node(database="db_2020", code="I1")
    for exchange in node.biosphere():
        exchange["amount"] = exchange["amount"] * 2
        exchange.save()
    bd.Database("db_2020").process()

    clear_lca_caches()
    processor = LCADataProcessor(_config(use_disk_cache=True))
    doubled = processor.background_inventory[("db_2020", "I1", "CO2")]
    assert doubled == pytest.approx(2.0)


def test_disk_cache_can_be_switched_off(setup_brightway_databases):
    """Nothing is written when the cache is disabled."""
    clear_lca_caches(include_disk=True)
    LCADataProcessor(_config(use_disk_cache=False))

    cache_dir = Path(bd.projects.dir) / "optimex-inventory-cache"
    assert not list(cache_dir.glob("*.pickle")) if cache_dir.exists() else True


def test_retained_flow_can_be_constrained_end_to_end(setup_brightway_databases):
    """A retained, uncharacterized background flow still drives a flow limit.

    Impacts are characterized per intermediate flow now, but the inventory is
    still tracked per elementary flow, so constraining one keeps working.
    """
    bd.Method(("only CO2", "example")).write([(("biosphere3", "CO2"), 1)])

    clear_lca_caches()
    config = _config(retain_flows=["CH4"])
    config.characterization_methods[0].category_name = "climate_change"
    config.characterization_methods[0].brightway_method = ("only CO2", "example")
    processor = LCADataProcessor(config)

    manager = converter.ModelInputManager()
    inputs = manager.parse_from_lca_processor(processor)
    assert "CH4" in inputs.ELEMENTARY_FLOW
    assert any(key[2] == "CH4" for key in inputs.background_inventory)

    model = optimizer.create_model(
        inputs, name="unconstrained", objective_category="climate_change"
    )
    solved, _, _ = optimizer.solve_model(model, solver_name="glpk", tee=False)
    fg_scale = solved.scales["foreground"]
    unconstrained = sum(
        pyo.value(solved.total_elementary_flow["CH4", t]) * fg_scale
        for t in solved.SYSTEM_TIME
    )
    assert unconstrained > 0, "CH4 only enters through the background inventory"

    limited = manager.override(cumulative_flow_limits_max={"CH4": unconstrained * 0.5})
    model = optimizer.create_model(
        limited, name="limited", objective_category="climate_change"
    )
    solved, _, _ = optimizer.solve_model(model, solver_name="glpk", tee=False)
    constrained = sum(
        pyo.value(solved.total_elementary_flow["CH4", t]) * fg_scale
        for t in solved.SYSTEM_TIME
    )
    assert constrained <= unconstrained * 0.5 * 1.0001


def test_dropped_flow_limit_error_points_at_retain_flows(setup_brightway_databases):
    """Constraining a dropped flow fails with an error that says what to do."""
    bd.Method(("only CO2", "example")).write([(("biosphere3", "CO2"), 1)])

    clear_lca_caches()
    config = _config()
    config.characterization_methods[0].brightway_method = ("only CO2", "example")
    processor = LCADataProcessor(config)

    manager = converter.ModelInputManager()
    manager.parse_from_lca_processor(processor)
    with pytest.raises(ValueError, match="retain_flows"):
        manager.override(cumulative_flow_limits_max={"CH4": 1.0})
