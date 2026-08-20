"""Tests for the batched, cached background inventory calculation."""

from datetime import datetime

import bw2calc as bc
import bw2data as bd
import numpy as np
import pytest
from bw_temporalis import TemporalDistribution

from optimex import lca_processor
from optimex.lca_processor import (
    LCAConfig,
    LCADataProcessor,
    clear_lca_caches,
    compute_db_inventory_entries,
)


def _config(**background_inventory):
    """Config mirroring the fixture project, with overridable inventory settings."""
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
