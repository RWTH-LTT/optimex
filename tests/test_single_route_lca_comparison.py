"""
Test for single-route system comparing optimex to standard LCA.

This is the simplest possible test: one product, one production route, no optimization.
Verifies that optimex produces the same results as bw2calc.LCA for a trivial case.
"""
from datetime import datetime

import bw2calc as bc
import bw2data as bd
import numpy as np
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution

from optimex import converter, lca_processor, optimizer, postprocessing


@pytest.fixture(scope="module")
@bw2test
def setup_single_route_system():
    """Set up the simplest possible system: one product, one route."""
    bd.projects.set_current("__test_single_route__")

    # Biosphere
    bio_db = bd.Database("biosphere3")
    bio_db.write({
        ("biosphere3", "CO2"): {
            "type": "emission",
            "name": "carbon dioxide",
        },
    })
    bio_db.register()

    # Background database
    bg_2020 = bd.Database("db_2020")
    bg_2020.write({
        ("db_2020", "electricity"): {
            "name": "electricity production",
            "location": "GLO",
            "reference product": "electricity",
            "exchanges": [
                {"amount": 1, "type": "production", "input": ("db_2020", "electricity")},
                {"amount": 0.5, "type": "biosphere", "input": ("biosphere3", "CO2")},
            ],
        },
    })
    bg_2020.metadata["representative_time"] = datetime(2020, 1, 1).isoformat()
    bg_2020.register()

    # Foreground with single product and single route
    fg = bd.Database("foreground")
    fg.write({
        # Product node
        ("foreground", "Widget"): {
            "name": "Widget",
            "unit": "kg",
            "type": bd.labels.product_node_default,
        },
        # Single production route
        ("foreground", "Widget_Route1"): {
            "name": "Widget production, Route 1",
            "location": "GLO",
            "type": bd.labels.process_node_default,
            "operation_time_limits": (1, 2),  # Operation phase at process times 1-2
            "exchanges": [
                {
                    "amount": 1,  # Produces 1 kg Widget
                    "type": bd.labels.production_edge_default,
                    "input": ("foreground", "Widget"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([0, 1, 2, 3], dtype="timedelta64[Y]"),
                        amount=np.array([0, 0.5, 0.5, 0]),  # Sums to 1.0
                    ),
                    "operation": True,
                },
                {
                    "amount": 10,  # Consumes 10 kWh electricity at construction
                    "type": bd.labels.consumption_edge_default,
                    "input": ("db_2020", "electricity"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([0, 1, 2, 3], dtype="timedelta64[Y]"),
                        amount=np.array([1, 0, 0, 0]),  # All at construction
                    ),
                },
                {
                    "amount": 5,  # Emits 5 kg CO2 during operation
                    "type": bd.labels.biosphere_edge_default,
                    "input": ("biosphere3", "CO2"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array([0, 1, 2, 3], dtype="timedelta64[Y]"),
                        amount=np.array([0, 0.5, 0.5, 0]),  # During operation
                    ),
                    "operation": True,
                },
            ],
        },
    })
    fg.register()

    # Impact method
    bd.Method(("GWP", "example")).write([
        (("biosphere3", "CO2"), 1),
    ])


def test_single_year_demand_builds_stranded_capacity(setup_single_route_system):
    """
    Demand in a single year, for a process whose operation window spans two years.

    One installed unit yields 1 kg Widget over its lifetime, delivered as 0.5 kg in
    each of its two operating years. Asking for 100 kg in 2022 alone therefore forces
    the model to install 200 units and leave the second operating year of each unit
    idle: the delivered amount is 100 kg but the built capacity is worth 200 kg.

    This is NOT expected to match `bc.LCA({widget: 100})`, because standard LCA
    implicitly assumes every unit is fully used. Expected here:
    - Construction electricity: 200 * 10 * 0.5   = 1000 kg CO2
    - Operation CO2 in 2022:    200 * 2.5 * 1.0  =  500 kg CO2
    - Total:                                      1500 kg CO2
    which is the LCA score of 100 kg (1000) plus the construction impact of the
    100 units' worth of capacity that is never used (500).

    See test_multi_period_demand_matches_standard_lca for the case where the demand
    profile allows full utilization and the two methods must agree exactly.
    """
    import pyomo.environ as pyo

    # Standard LCA calculation, for reference
    widget = bd.get_node(database="foreground", name="Widget")
    lca = bc.LCA({widget: 100}, method=("GWP", "example"))
    lca.lci()
    lca.lcia()
    lca_gwp = lca.score

    # Capacity worth 200 kg is built, so the impact is the LCA score of 200 kg
    # minus the operation emissions of the 100 kg that is never produced.
    expected_gwp = 1500.0

    print(f"\nStandard LCA GWP (100 kg, full utilization): {lca_gwp}")
    print(f"Expected optimex GWP (single-year demand): {expected_gwp}")

    # optimex calculation
    years = range(2020, 2030)
    td_demand = TemporalDistribution(
        date=np.array([datetime(year, 1, 1).isoformat() for year in years], dtype='datetime64[s]'),
        amount=np.asarray([0, 0, 100, 0, 0, 0, 0, 0, 0, 0]),  # 100 kg at year 2022
    )

    lca_config = lca_processor.LCAConfig(
        demand={widget: td_demand},
        temporal={
            "start_date": datetime(2020, 1, 1),
            "temporal_resolution": "year",
            "time_horizon": 100,
        },
        characterization_methods=[
            {
                "category_name": "climate_change",
                "brightway_method": ("GWP", "example"),
            },
        ],
    )

    lca_data_processor = lca_processor.LCADataProcessor(lca_config)
    manager = converter.ModelInputManager()
    optimization_model_inputs = manager.parse_from_lca_processor(lca_data_processor)

    model = optimizer.create_model(
        optimization_model_inputs,
        name="test_single_route",
        objective_category="climate_change",
    )

    solved_model, obj_real, results = optimizer.solve_model(model, solver_name="glpk")

    print(f"optimex GWP: {obj_real}")

    # 200 units must be installed to deliver 100 kg in a single year
    total_installed = sum(
        pyo.value(solved_model.var_installation[p, t])
        for p in solved_model.PROCESS
        for t in solved_model.SYSTEM_TIME
    )
    print(f"Total installed units: {total_installed}")
    assert pytest.approx(total_installed, rel=1e-3) == 200.0, (
        f"Delivering 100 kg in one year needs 200 units (0.5 kg per unit and year), "
        f"got {total_installed}"
    )

    assert pytest.approx(obj_real, rel=1e-3) == expected_gwp, (
        f"optimex result ({obj_real}) should be {expected_gwp}: the LCA score of "
        f"100 kg ({lca_gwp}) plus the construction impact of the stranded capacity"
    )

    # Additional check: verify postprocessing extracts correct unscaled values
    pp = postprocessing.PostProcessor(model)
    df_impacts = pp.get_impacts()

    # Sum all climate_change impacts across all processes and times
    if 'climate_change' in df_impacts.columns.get_level_values(0):
        climate_change_cols = [col for col in df_impacts.columns if col[0] == 'climate_change']
        total_cc_from_pp = df_impacts[climate_change_cols].sum().sum()

        print(f"\nPostprocessing climate_change total: {total_cc_from_pp}")

        assert pytest.approx(total_cc_from_pp, rel=1e-3) == expected_gwp, (
            f"Postprocessing climate_change sum ({total_cc_from_pp}) should match "
            f"the objective ({expected_gwp})"
        )


def test_multi_period_demand_matches_standard_lca(setup_single_route_system):
    """
    Same single-route system, but demand at MULTIPLE time steps.

    The point of this test is installation-impact accounting. The process has a
    2-year operation window (tau 1-2) and its production temporal distribution sums
    to 1 kg over that window (0.5 kg per year), so one installed unit delivers 1 kg
    of Widget over its lifetime. Serving 100 kg in 2022 AND 100 kg in 2023 therefore
    requires 200 units, all installed in 2021 and fully utilized in both years — the
    demand profile matches the production profile exactly, so this is a case where
    optimex must reproduce the standard LCA result.

    Note on test design: demand must line up with the operation window for LCA
    equivalence to hold. A single isolated demand year would force the model to build
    capacity whose second operating year is never used (legitimate stranded capacity),
    which is more impact than the standard LCA of the delivered amount.

    Demand: 100 kg in 2022, 100 kg in 2023.
    Standard LCA for 200 kg:
    - Direct CO2 from operation:   200 * 5 * 1.0 = 1000 kg CO2
    - Background electricity:      200 * 10 * 0.5 = 1000 kg CO2
    - Total:                       2000 kg CO2
    """
    import pyomo.environ as pyo

    demand_by_year = {2022: 100.0, 2023: 100.0}
    total_demand = sum(demand_by_year.values())

    # Standard LCA calculation for the same total amount
    widget = bd.get_node(database="foreground", name="Widget")
    lca = bc.LCA({widget: total_demand}, method=("GWP", "example"))
    lca.lci()
    lca.lcia()
    expected_gwp = lca.score

    print(f"\nStandard LCA GWP ({total_demand} kg): {expected_gwp}")

    years = list(range(2020, 2030))
    td_demand = TemporalDistribution(
        date=np.array(
            [datetime(year, 1, 1).isoformat() for year in years], dtype="datetime64[s]"
        ),
        amount=np.asarray([demand_by_year.get(year, 0.0) for year in years]),
    )

    lca_config = lca_processor.LCAConfig(
        demand={widget: td_demand},
        temporal={
            "start_date": datetime(2020, 1, 1),
            "temporal_resolution": "year",
            "time_horizon": 100,
        },
        characterization_methods=[
            {
                "category_name": "climate_change",
                "brightway_method": ("GWP", "example"),
            },
        ],
    )

    lca_data_processor = lca_processor.LCADataProcessor(lca_config)
    manager = converter.ModelInputManager()
    optimization_model_inputs = manager.parse_from_lca_processor(lca_data_processor)

    model = optimizer.create_model(
        optimization_model_inputs,
        name="test_multi_period_demand",
        objective_category="climate_change",
    )

    _, obj_real, _ = optimizer.solve_model(model, solver_name="glpk")

    # Diagnostics: how much was installed, and when
    installations = {
        (p, t): pyo.value(model.var_installation[p, t])
        for p in model.PROCESS
        for t in model.SYSTEM_TIME
        if pyo.value(model.var_installation[p, t]) > 1e-9
    }
    operations = {
        (p, v, t): pyo.value(model.var_operation[p, v, t])
        for (p, v, t) in model.ACTIVE_VINTAGE_TIME
        if pyo.value(model.var_operation[p, v, t]) > 1e-9
    }
    total_installed = sum(installations.values())

    print(f"optimex GWP: {obj_real}")
    print(f"Installations (process, year) -> units: {installations}")
    print(f"Total installed units: {total_installed}")
    print(f"Operation (process, vintage, year) -> level: {operations}")

    # One unit of the process delivers 1 kg Widget over its lifetime, so meeting
    # 300 kg of demand must require 300 installed units — no more, no less.
    assert pytest.approx(total_installed, rel=1e-3) == total_demand, (
        f"Total installed units ({total_installed}) should equal total demand "
        f"({total_demand}); a lower value means installation impacts are amortized "
        f"over more production than the process actually delivers"
    )

    assert pytest.approx(obj_real, rel=1e-3) == expected_gwp, (
        f"optimex result ({obj_real}) should match standard LCA ({expected_gwp})"
    )

    # Per-year impact breakdown from postprocessing
    pp = postprocessing.PostProcessor(model)
    df_impacts = pp.get_impacts()
    climate_change_cols = [c for c in df_impacts.columns if c[0] == "climate_change"]
    impacts_per_year = df_impacts[climate_change_cols].sum(axis=1)
    print("\nclimate_change impact per year:")
    print(impacts_per_year[impacts_per_year.abs() > 1e-9])

    total_cc_from_pp = df_impacts[climate_change_cols].sum().sum()
    print(f"Postprocessing climate_change total: {total_cc_from_pp}")

    assert pytest.approx(total_cc_from_pp, rel=1e-3) == expected_gwp, (
        f"Postprocessing climate_change sum ({total_cc_from_pp}) should match "
        f"standard LCA ({expected_gwp})"
    )

    # Construction electricity is consumed at tau=0, i.e. in the installation year.
    # Its impact must therefore show up in exactly the years where units were installed.
    install_years = {t for (_, t) in installations}
    construction_impact_per_year = {
        t: 0.5 * 10 * sum(v for (_, ty), v in installations.items() if ty == t)
        for t in install_years
    }
    print(f"Expected construction impact per year: {construction_impact_per_year}")
    for t, expected_construction in construction_impact_per_year.items():
        assert impacts_per_year.loc[t] >= expected_construction - 1e-6, (
            f"Impact in {t} ({impacts_per_year.loc[t]}) is below the construction "
            f"impact of the units installed that year ({expected_construction})"
        )
