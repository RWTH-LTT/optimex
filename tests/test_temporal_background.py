"""
Tests for temporal background inventory resolution.

Tests that the cross-database graph traversal with temporal distributions on
background exchanges produces correct, hand-computable results.

Key scenario:
    Two background databases 20 years apart (db_2020, db_2040).
    Electricity becomes 10x cleaner: 10 CO2/kWh → 1 CO2/kWh.
    Steel production consumes 1 kWh electricity with TD = 100% at t=-20
    (electricity was needed 20 years before steel delivery).

    Without temporal traversal: steel at 2040 uses db_2040 electricity → 1 CO2.
    With temporal traversal: steel at 2040 shifts electricity to 2020 → 10 CO2.
"""
from datetime import datetime

import bw2data as bd
import numpy as np
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution

from optimex import converter, lca_processor, optimizer


# =============================================================================
# Fixture
# =============================================================================


@pytest.fixture(scope="module")
@bw2test
def setup_temporal_background_system():
    """
    Set up a system with drastically different background databases.

    Background supply chain:
        electricity (leaf node):
            db_2020: 10.0 CO2/kWh (dirty coal grid)
            db_2040:  1.0 CO2/kWh (clean renewable grid)

        steel:
            Both dbs: 0 direct CO2, consumes 1 kWh electricity
            TD on electricity input: 100% at t=-20 (20 years before delivery)

    This means steel demanded at year Y sources electricity from year Y-20.

    Foreground:
        Process P1: immediate production (operation at t=0)
            - produces 1 Widget at t=0 (operation-dependent)
            - consumes 1 unit steel at t=0 (installation-dependent)
            - emits 5 kg CO2 directly at t=0 (operation-dependent)
    """
    bd.projects.set_current("__test_temporal_bg__")

    # Biosphere
    bio_db = bd.Database("biosphere3")
    bio_db.write({
        ("biosphere3", "CO2"): {
            "type": "emission",
            "name": "carbon dioxide",
        },
    })
    bio_db.register()

    # ---- Background database 2020 (dirty grid) ----
    bg_2020 = bd.Database("db_2020")
    bg_2020.write({
        ("db_2020", "electricity"): {
            "name": "electricity production",
            "location": "GLO",
            "reference product": "electricity",
            "exchanges": [
                {"amount": 1, "type": "production", "input": ("db_2020", "electricity")},
                {"amount": 10.0, "type": "biosphere", "input": ("biosphere3", "CO2")},
            ],
        },
        ("db_2020", "steel"): {
            "name": "steel production",
            "location": "GLO",
            "reference product": "steel",
            "exchanges": [
                {"amount": 1, "type": "production", "input": ("db_2020", "steel")},
                # No direct biosphere emissions from steel
                {
                    "amount": 1.0,
                    "type": "technosphere",
                    "input": ("db_2020", "electricity"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([-20], dtype="timedelta64[Y]"),
                        amount=np.array([1.0]),
                    ),
                },
            ],
        },
    })
    bg_2020.metadata["representative_time"] = datetime(2020, 1, 1).isoformat()
    bg_2020.register()

    # ---- Background database 2040 (clean grid) ----
    bg_2040 = bd.Database("db_2040")
    bg_2040.write({
        ("db_2040", "electricity"): {
            "name": "electricity production",
            "location": "GLO",
            "reference product": "electricity",
            "exchanges": [
                {"amount": 1, "type": "production", "input": ("db_2040", "electricity")},
                {"amount": 1.0, "type": "biosphere", "input": ("biosphere3", "CO2")},
            ],
        },
        ("db_2040", "steel"): {
            "name": "steel production",
            "location": "GLO",
            "reference product": "steel",
            "exchanges": [
                {"amount": 1, "type": "production", "input": ("db_2040", "steel")},
                # No direct biosphere emissions from steel
                {
                    "amount": 1.0,
                    "type": "technosphere",
                    "input": ("db_2040", "electricity"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([-20], dtype="timedelta64[Y]"),
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
        ("foreground", "Widget"): {
            "name": "Widget",
            "unit": "kg",
            "type": bd.labels.product_node_default,
        },
        ("foreground", "P1"): {
            "name": "Process P1",
            "location": "GLO",
            "type": bd.labels.process_node_default,
            "operation_time_limits": (0, 0),  # immediate: operation at t=0
            "exchanges": [
                {
                    "amount": 1,
                    "type": bd.labels.production_edge_default,
                    "input": ("foreground", "Widget"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([0], dtype="timedelta64[Y]"),
                        amount=np.array([1.0]),
                    ),
                    "operation": True,
                },
                {
                    "amount": 1,
                    "type": bd.labels.consumption_edge_default,
                    "input": ("db_2020", "steel"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([0], dtype="timedelta64[Y]"),
                        amount=np.array([1.0]),
                    ),
                    # No "operation" flag → installation-dependent
                },
                {
                    "amount": 5,
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


# =============================================================================
# Helper to build LCA config
# =============================================================================


def _make_config(demand_year, demand_amount, temporal=False):
    product = bd.get_node(database="foreground", name="Widget")
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
# Tests: resolved background inventory values
# =============================================================================


def test_resolved_inventory_steel_at_2040(setup_temporal_background_system):
    """
    Hand computation for steel demanded at year 2040 with temporal traversal:

    1. Pop (steel, 2040, 1.0)
       → mapping weights: {db_2040: 1.0}
       → db_2040 steel column:
         - biosphere: 0 CO2 (steel has no direct emissions)
         - technosphere: 1 kWh electricity, TD = [100% at t=-20]
         → push (electricity, year=2040-20=2020, amount=1.0)

    2. Pop (electricity, 2020, 1.0)
       → mapping weights: {db_2020: 1.0}
       → db_2020 electricity column:
         - biosphere: 10.0 CO2
         → accumulated CO2 = 10.0 × 1.0 × 1.0 = 10.0
         - technosphere: only diagonal (no upstream inputs)

    Result: resolved_background_inventory[("steel", "CO2", 2040)] = 10.0
    """
    config = _make_config(demand_year=2040, demand_amount=1, temporal=True)
    proc = lca_processor.LCADataProcessor(config)
    rbi = proc.resolved_background_inventory

    assert rbi is not None
    steel_co2_2040 = rbi[("steel", "CO2", 2040)]
    assert pytest.approx(steel_co2_2040, abs=1e-6) == 10.0


def test_resolved_inventory_steel_at_2020(setup_temporal_background_system):
    """
    Steel demanded at year 2020:

    1. Pop (steel, 2020, 1.0)
       → weights: {db_2020: 1.0}
       → db_2020 steel: 0 direct CO2, 1 elec with TD at t=-20
       → push (electricity, year=2020-20=2000, amount=1.0)

    2. Pop (electricity, 2000, 1.0)
       → weights: {db_2020: 1.0} (clamped to earliest db)
       → db_2020 electricity: 10.0 CO2
       → accumulated = 10.0

    Result: 10.0 (same as 2040, because electricity always goes to db_2020 era)
    """
    config = _make_config(demand_year=2040, demand_amount=1, temporal=True)
    proc = lca_processor.LCADataProcessor(config)
    rbi = proc.resolved_background_inventory

    steel_co2_2020 = rbi[("steel", "CO2", 2020)]
    assert pytest.approx(steel_co2_2020, abs=1e-6) == 10.0


def test_resolved_inventory_all_years_equal(setup_temporal_background_system):
    """
    With a 20-year backward shift between dbs that are 20 years apart,
    electricity ALWAYS gets resolved at db_2020 for any demand year in [2020, 2040]:

    - Year 2020: electricity at 2000 → clamped to db_2020 → 10.0
    - Year 2030: electricity at 2010 → clamped to db_2020 → 10.0
    - Year 2040: electricity at 2020 → exactly db_2020 → 10.0

    So temporal steel LCI = 10.0 for all system years.
    """
    config = _make_config(demand_year=2040, demand_amount=1, temporal=True)
    proc = lca_processor.LCADataProcessor(config)
    rbi = proc.resolved_background_inventory

    for year in proc.system_time:
        steel_co2 = rbi.get(("steel", "CO2", year), 0)
        assert pytest.approx(steel_co2, abs=1e-6) == 10.0, (
            f"Steel CO2 at {year} should be 10.0, got {steel_co2}"
        )


def test_nontemporal_inventory_varies_by_year(setup_temporal_background_system):
    """
    Without temporal traversal, background_inventory × mapping gives:

    Non-temporal steel LCI (via matrix inversion in each db):
        db_2020: 0 direct + 1 × 10.0 = 10.0 CO2
        db_2040: 0 direct + 1 ×  1.0 =  1.0 CO2

    At year 2020: mapping = {db_2020: 1.0}        → 10.0
    At year 2030: mapping = {db_2020: 0.5, db_2040: 0.5} → 5.5
    At year 2040: mapping = {db_2040: 1.0}        →  1.0
    """
    config = _make_config(demand_year=2040, demand_amount=1, temporal=False)
    proc = lca_processor.LCADataProcessor(config)

    # Verify background inventory per-database
    bi = proc.background_inventory
    steel_co2_db2020 = bi.get(("db_2020", "steel", "CO2"), 0)
    steel_co2_db2040 = bi.get(("db_2040", "steel", "CO2"), 0)
    assert pytest.approx(steel_co2_db2020, abs=1e-6) == 10.0
    assert pytest.approx(steel_co2_db2040, abs=1e-6) == 1.0

    # Verify mapping weights
    mapping = proc.mapping
    assert pytest.approx(mapping[("db_2020", 2020)]) == 1.0
    assert pytest.approx(mapping.get(("db_2020", 2030), 0)) == 0.5
    assert pytest.approx(mapping.get(("db_2040", 2030), 0)) == 0.5
    assert pytest.approx(mapping[("db_2040", 2040)]) == 1.0


# =============================================================================
# Tests: full optimizer integration with exact expected objectives
# =============================================================================


def test_optimizer_objective_temporal(setup_temporal_background_system):
    """
    Full pipeline with temporal=True, demand = 1 Widget at year 2040.

    Optimizer installs 1 unit of P1 at 2040:
        - Steel consumption (installation-dependent): 1 unit at 2040
          → background CO2 = 1 × resolved_bg[steel, CO2, 2040] = 1 × 10.0 = 10.0
        - Direct CO2 (operation-dependent): 5 × operation_level = 5 × 1 = 5.0

    Total CO2 = 10.0 + 5.0 = 15.0
    Characterization: CO2 × 1 = 15.0
    Objective = 15.0
    """
    config = _make_config(demand_year=2040, demand_amount=1, temporal=True)
    proc = lca_processor.LCADataProcessor(config)
    manager = converter.ModelInputManager()
    inputs = manager.parse_from_lca_processor(proc)
    model = optimizer.create_model(inputs, name="temporal", objective_category="climate_change")
    _, obj_temporal, results = optimizer.solve_model(model, solver_name="glpk")

    assert results.solver.termination_condition.value == "optimal"
    assert pytest.approx(obj_temporal, rel=1e-3) == 15.0, (
        f"Temporal objective should be 15.0 (10 bg + 5 direct), got {obj_temporal}"
    )


def test_optimizer_objective_nontemporal(setup_temporal_background_system):
    """
    Full pipeline with temporal=False, demand = 1 Widget at year 2040.

    Optimizer installs 1 unit of P1 at 2040:
        - Steel consumption: 1 unit at 2040
          → background CO2 = sum over dbs: bg_inv[db, steel, CO2] × mapping[db, 2040]
            = 10.0 × 0 + 1.0 × 1.0 = 1.0
        - Direct CO2: 5 × 1 = 5.0

    Total CO2 = 1.0 + 5.0 = 6.0
    Objective = 6.0
    """
    config = _make_config(demand_year=2040, demand_amount=1, temporal=False)
    proc = lca_processor.LCADataProcessor(config)
    manager = converter.ModelInputManager()
    inputs = manager.parse_from_lca_processor(proc)
    model = optimizer.create_model(inputs, name="nontemporal", objective_category="climate_change")
    _, obj_static, results = optimizer.solve_model(model, solver_name="glpk")

    assert results.solver.termination_condition.value == "optimal"
    assert pytest.approx(obj_static, rel=1e-3) == 6.0, (
        f"Non-temporal objective should be 6.0 (1 bg + 5 direct), got {obj_static}"
    )


def test_temporal_vs_nontemporal_10x_difference(setup_temporal_background_system):
    """
    The temporal approach reveals that steel's upstream electricity is sourced
    from 20 years ago (dirty grid), while non-temporal assumes it comes from
    the current (clean) grid.

    Background CO2 from steel at year 2040:
        Temporal:     10.0 (electricity at 2020, 10 CO2/kWh)
        Non-temporal:  1.0 (electricity at 2040,  1 CO2/kWh)

    This 10x difference in background impact propagates to the objective:
        Temporal objective:     10.0 + 5.0 = 15.0
        Non-temporal objective:  1.0 + 5.0 =  6.0
    """
    config_t = _make_config(demand_year=2040, demand_amount=1, temporal=True)
    proc_t = lca_processor.LCADataProcessor(config_t)
    mgr_t = converter.ModelInputManager()
    inp_t = mgr_t.parse_from_lca_processor(proc_t)
    model_t = optimizer.create_model(inp_t, name="t", objective_category="climate_change")
    _, obj_t, _ = optimizer.solve_model(model_t, solver_name="glpk")

    config_s = _make_config(demand_year=2040, demand_amount=1, temporal=False)
    proc_s = lca_processor.LCADataProcessor(config_s)
    mgr_s = converter.ModelInputManager()
    inp_s = mgr_s.parse_from_lca_processor(proc_s)
    model_s = optimizer.create_model(inp_s, name="s", objective_category="climate_change")
    _, obj_s, _ = optimizer.solve_model(model_s, solver_name="glpk")

    # Background contribution differs by 10x
    bg_temporal = obj_t - 5.0      # subtract direct foreground CO2
    bg_nontemporal = obj_s - 5.0
    assert pytest.approx(bg_temporal / bg_nontemporal, rel=1e-2) == 10.0


def test_optimizer_scales_with_demand(setup_temporal_background_system):
    """
    With 100 units of demand, the objective should scale linearly.

    Temporal:     100 × 15.0 = 1500.0
    Non-temporal: 100 ×  6.0 =  600.0
    """
    config = _make_config(demand_year=2040, demand_amount=100, temporal=True)
    proc = lca_processor.LCADataProcessor(config)
    manager = converter.ModelInputManager()
    inputs = manager.parse_from_lca_processor(proc)
    model = optimizer.create_model(inputs, name="scaled", objective_category="climate_change")
    _, obj, _ = optimizer.solve_model(model, solver_name="glpk")

    assert pytest.approx(obj, rel=1e-3) == 1500.0


# =============================================================================
# Tests: helper methods
# =============================================================================


def test_mapping_weights_interpolation(setup_temporal_background_system):
    """Test _get_mapping_weights_for_time returns correct interpolation weights."""
    config = _make_config(demand_year=2040, demand_amount=1)
    proc = lca_processor.LCADataProcessor(config)

    # At database years: 100% single db
    assert proc._get_mapping_weights_for_time(2020) == {"db_2020": 1.0}
    assert proc._get_mapping_weights_for_time(2040) == {"db_2040": 1.0}

    # Midpoint: 50/50
    w = proc._get_mapping_weights_for_time(2030)
    assert pytest.approx(w["db_2020"]) == 0.5
    assert pytest.approx(w["db_2040"]) == 0.5

    # Before earliest: clamp
    assert proc._get_mapping_weights_for_time(2000) == {"db_2020": 1.0}

    # After latest: clamp
    assert proc._get_mapping_weights_for_time(2060) == {"db_2040": 1.0}


def test_temporal_false_is_default(setup_temporal_background_system):
    """Verify temporal=False is the default and produces no resolved inventory."""
    config = _make_config(demand_year=2040, demand_amount=1, temporal=False)
    proc = lca_processor.LCADataProcessor(config)
    assert proc.resolved_background_inventory is None
    assert proc.config.background_inventory.temporal is False
