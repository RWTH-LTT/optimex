"""
Tests for multi-level temporal background inventory resolution.

Three-level background supply chain with cumulative time shifts:
    assembly → component → raw_material (leaf)

Each hop shifts time by -10 years via temporal distributions.
Demanding assembly at year Y triggers: component at Y-10, raw_material at Y-20.

Two databases (db_2020, db_2040):
    raw_material (leaf): db_2020 emits 20 CO2, db_2040 emits 2 CO2
    component/assembly: pass-through (0 direct CO2)

This tests that the traversal correctly chains through multiple levels,
selecting the time-appropriate database at each step.
"""
from datetime import datetime

import bw2data as bd
import numpy as np
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution

from optimex import converter, lca_processor, optimizer


@pytest.fixture(scope="module")
@bw2test
def setup_multilevel_background():
    """
    Three-level background chain: assembly → component → raw_material.

    Each level shifts time backward by 10 years:
        assembly  consumes 1 component     with TD at t=-10
        component consumes 1 raw_material  with TD at t=-10

    raw_material is the only emitter:
        db_2020: 20 CO2 (dirty extraction)
        db_2040:  2 CO2 (clean extraction)

    Foreground:
        Process P1: immediate production (operation at t=0)
            - produces 1 Gadget (operation)
            - consumes 1 assembly (installation)
            - emits 3 CO2 directly (operation)
    """
    bd.projects.set_current("__test_multilevel_bg__")

    bio_db = bd.Database("biosphere3")
    bio_db.write({
        ("biosphere3", "CO2"): {
            "type": "emission",
            "name": "carbon dioxide",
        },
    })
    bio_db.register()

    # ---- db_2020 ----
    bg_2020 = bd.Database("db_2020")
    bg_2020.write({
        ("db_2020", "raw_material"): {
            "name": "raw material extraction",
            "location": "GLO",
            "reference product": "raw_material",
            "exchanges": [
                {"amount": 1, "type": "production", "input": ("db_2020", "raw_material")},
                {"amount": 20.0, "type": "biosphere", "input": ("biosphere3", "CO2")},
            ],
        },
        ("db_2020", "component"): {
            "name": "component manufacturing",
            "location": "GLO",
            "reference product": "component",
            "exchanges": [
                {"amount": 1, "type": "production", "input": ("db_2020", "component")},
                {
                    "amount": 1.0,
                    "type": "technosphere",
                    "input": ("db_2020", "raw_material"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([-10], dtype="timedelta64[Y]"),
                        amount=np.array([1.0]),
                    ),
                },
            ],
        },
        ("db_2020", "assembly"): {
            "name": "assembly production",
            "location": "GLO",
            "reference product": "assembly",
            "exchanges": [
                {"amount": 1, "type": "production", "input": ("db_2020", "assembly")},
                {
                    "amount": 1.0,
                    "type": "technosphere",
                    "input": ("db_2020", "component"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([-10], dtype="timedelta64[Y]"),
                        amount=np.array([1.0]),
                    ),
                },
            ],
        },
    })
    bg_2020.metadata["representative_time"] = datetime(2020, 1, 1).isoformat()
    bg_2020.register()

    # ---- db_2040 ----
    bg_2040 = bd.Database("db_2040")
    bg_2040.write({
        ("db_2040", "raw_material"): {
            "name": "raw material extraction",
            "location": "GLO",
            "reference product": "raw_material",
            "exchanges": [
                {"amount": 1, "type": "production", "input": ("db_2040", "raw_material")},
                {"amount": 2.0, "type": "biosphere", "input": ("biosphere3", "CO2")},
            ],
        },
        ("db_2040", "component"): {
            "name": "component manufacturing",
            "location": "GLO",
            "reference product": "component",
            "exchanges": [
                {"amount": 1, "type": "production", "input": ("db_2040", "component")},
                {
                    "amount": 1.0,
                    "type": "technosphere",
                    "input": ("db_2040", "raw_material"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([-10], dtype="timedelta64[Y]"),
                        amount=np.array([1.0]),
                    ),
                },
            ],
        },
        ("db_2040", "assembly"): {
            "name": "assembly production",
            "location": "GLO",
            "reference product": "assembly",
            "exchanges": [
                {"amount": 1, "type": "production", "input": ("db_2040", "assembly")},
                {
                    "amount": 1.0,
                    "type": "technosphere",
                    "input": ("db_2040", "component"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([-10], dtype="timedelta64[Y]"),
                        amount=np.array([1.0]),
                    ),
                },
            ],
        },
    })
    bg_2040.metadata["representative_time"] = datetime(2040, 1, 1).isoformat()
    bg_2040.register()

    # ---- Foreground ----
    fg = bd.Database("foreground")
    fg.write({
        ("foreground", "Gadget"): {
            "name": "Gadget",
            "unit": "kg",
            "type": bd.labels.product_node_default,
        },
        ("foreground", "P1"): {
            "name": "Process P1",
            "location": "GLO",
            "type": bd.labels.process_node_default,
            "operation_time_limits": (0, 0),
            "exchanges": [
                {
                    "amount": 1,
                    "type": bd.labels.production_edge_default,
                    "input": ("foreground", "Gadget"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([0], dtype="timedelta64[Y]"),
                        amount=np.array([1.0]),
                    ),
                    "operation": True,
                },
                {
                    "amount": 1,
                    "type": bd.labels.consumption_edge_default,
                    "input": ("db_2020", "assembly"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([0], dtype="timedelta64[Y]"),
                        amount=np.array([1.0]),
                    ),
                },
                {
                    "amount": 3,
                    "type": bd.labels.biosphere_edge_default,
                    "input": ("biosphere3", "CO2"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([0], dtype="timedelta64[Y]"),
                        amount=np.array([1.0]),
                    ),
                    "operation": True,
                },
            ],
        },
    })
    fg.register()

    bd.Method(("GWP", "example")).write([
        (("biosphere3", "CO2"), 1),
    ])


def _make_config(demand_year, demand_amount, temporal=False):
    product = bd.get_node(database="foreground", name="Gadget")
    years = range(2020, demand_year + 1)
    amounts = [demand_amount if y == demand_year else 0 for y in years]
    td_demand = TemporalDistribution(
        date=np.array(
            [datetime(y, 1, 1).isoformat() for y in years], dtype="datetime64[s]"
        ),
        amount=np.asarray(amounts, dtype=float),
    )
    kwargs = {
        "demand": {product: td_demand},
        "temporal": {
            "start_date": datetime(2020, 1, 1),
            "temporal_resolution": "year",
            "time_horizon": 100,
        },
        "characterization_methods": [
            {"category_name": "climate_change", "brightway_method": ("GWP", "example")},
        ],
    }
    if temporal:
        kwargs["background_inventory"] = {"temporal": True}
    return lca_processor.LCAConfig(**kwargs)


# =============================================================================
# Tests: resolved inventory values (hand-computed)
# =============================================================================


def test_multilevel_resolved_inventory_at_2040(setup_multilevel_background):
    """
    assembly demanded at year 2040, 3-level traversal:

    Step 1: Pop (assembly, 2040, 1.0)
        weights: {db_2040: 1.0}
        db_2040 assembly: 0 CO2, 1 component with TD at t=-10
        → push (component, 2030, 1.0)

    Step 2: Pop (component, 2030, 1.0)
        weights: {db_2020: 0.5, db_2040: 0.5}
        db_2020 (0.5): push (raw_material, 2020, 0.5)
        db_2040 (0.5): push (raw_material, 2020, 0.5)

    Step 3: Pop (raw_material, 2020, 0.5)
        weights: {db_2020: 1.0}
        CO2: 20 × 1.0 × 0.5 = 10.0

    Step 4: Pop (raw_material, 2020, 0.5)
        CO2: 20 × 1.0 × 0.5 = 10.0

    Total = 20.0
    """
    config = _make_config(demand_year=2040, demand_amount=1, temporal=True)
    proc = lca_processor.LCADataProcessor(config)
    rbi = proc.resolved_background_inventory

    assert rbi is not None
    assembly_co2_2040 = rbi[("assembly", "CO2", 2040)]
    assert pytest.approx(assembly_co2_2040, abs=1e-6) == 20.0


def test_multilevel_nontemporal_per_db(setup_multilevel_background):
    """
    Non-temporal: matrix inversion within each db.

    db_2040 assembly inventory (A^{-1} resolves chain within same db):
        assembly → component → raw_material → 2 CO2
        Total = 0 + 0 + 2 = 2.0

    db_2020 assembly inventory:
        assembly → component → raw_material → 20 CO2
        Total = 0 + 0 + 20 = 20.0
    """
    config = _make_config(demand_year=2040, demand_amount=1, temporal=False)
    proc = lca_processor.LCADataProcessor(config)

    bi = proc.background_inventory
    assert pytest.approx(bi[("db_2040", "assembly", "CO2")], abs=1e-6) == 2.0
    assert pytest.approx(bi[("db_2020", "assembly", "CO2")], abs=1e-6) == 20.0


def test_multilevel_intermediate_year_2035(setup_multilevel_background):
    """
    assembly demanded at year 2035, with interpolation at each level.

    Step 1: assembly at 2035
        weights: {db_2020: 0.25, db_2040: 0.75}
        → push (component, 2025, 0.25) and (component, 2025, 0.75)

    Step 2a: component at 2025, amount=0.75
        weights: {db_2020: 0.75, db_2040: 0.25}
        → push (raw_material, 2015, 0.5625) and (raw_material, 2015, 0.1875)

    Step 2b: component at 2025, amount=0.25
        weights: {db_2020: 0.75, db_2040: 0.25}
        → push (raw_material, 2015, 0.1875) and (raw_material, 2015, 0.0625)

    Steps 3-6: raw_material at 2015, weights: {db_2020: 1.0} (clamped)
        CO2 = 20 × (0.5625 + 0.1875 + 0.1875 + 0.0625) = 20 × 1.0 = 20.0

    Total = 20.0 (same — all paths still land before db_2020)
    """
    config = _make_config(demand_year=2040, demand_amount=1, temporal=True)
    proc = lca_processor.LCADataProcessor(config)
    rbi = proc.resolved_background_inventory

    assert pytest.approx(rbi[("assembly", "CO2", 2035)], abs=1e-6) == 20.0


# =============================================================================
# Tests: full optimizer (hand-computed objectives)
# =============================================================================


def test_multilevel_optimizer_temporal(setup_multilevel_background):
    """
    Demand: 1 Gadget at 2040. P1 installed at 2040.
        Background: 1 assembly × resolved_bg = 20.0 CO2
        Direct:     3 CO2
        Objective = 23.0
    """
    config = _make_config(demand_year=2040, demand_amount=1, temporal=True)
    proc = lca_processor.LCADataProcessor(config)
    mgr = converter.ModelInputManager()
    inputs = mgr.parse_from_lca_processor(proc)
    model = optimizer.create_model(inputs, name="ml_t", objective_category="climate_change")
    _, obj, results = optimizer.solve_model(model, solver_name="glpk")

    assert results.solver.termination_condition.value == "optimal"
    assert pytest.approx(obj, rel=1e-3) == 23.0


def test_multilevel_optimizer_nontemporal(setup_multilevel_background):
    """
    Non-temporal: assembly at 2040 = 2.0 CO2 (db_2040 matrix inversion).
        Background: 2.0 CO2
        Direct:     3.0 CO2
        Objective = 5.0
    """
    config = _make_config(demand_year=2040, demand_amount=1, temporal=False)
    proc = lca_processor.LCADataProcessor(config)
    mgr = converter.ModelInputManager()
    inputs = mgr.parse_from_lca_processor(proc)
    model = optimizer.create_model(inputs, name="ml_s", objective_category="climate_change")
    _, obj, results = optimizer.solve_model(model, solver_name="glpk")

    assert results.solver.termination_condition.value == "optimal"
    assert pytest.approx(obj, rel=1e-3) == 5.0


def test_multilevel_10x_background_difference(setup_multilevel_background):
    """
    Background CO2 ratio between temporal and non-temporal = 20 / 2 = 10x.
    This confirms multi-level chaining correctly propagates the time shift
    through all three levels to reach the oldest, dirtiest database.
    """
    config_t = _make_config(demand_year=2040, demand_amount=1, temporal=True)
    proc_t = lca_processor.LCADataProcessor(config_t)
    mgr_t = converter.ModelInputManager()
    inp_t = mgr_t.parse_from_lca_processor(proc_t)
    model_t = optimizer.create_model(inp_t, name="ml_t2", objective_category="climate_change")
    _, obj_t, _ = optimizer.solve_model(model_t, solver_name="glpk")

    config_s = _make_config(demand_year=2040, demand_amount=1, temporal=False)
    proc_s = lca_processor.LCADataProcessor(config_s)
    mgr_s = converter.ModelInputManager()
    inp_s = mgr_s.parse_from_lca_processor(proc_s)
    model_s = optimizer.create_model(inp_s, name="ml_s2", objective_category="climate_change")
    _, obj_s, _ = optimizer.solve_model(model_s, solver_name="glpk")

    bg_temporal = obj_t - 3.0       # subtract direct foreground CO2
    bg_nontemporal = obj_s - 3.0
    assert pytest.approx(bg_temporal / bg_nontemporal, rel=1e-2) == 10.0
