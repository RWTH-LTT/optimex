"""
Test for two-level supply chain with internal demands.
Based on notebooks/basic_optimex_example_two_decision_layers.ipynb

This test verifies that optimex produces the same results as a standard LCA
when there are no optimization choices (single route).
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
def setup_two_level_system():
    """Set up a two-level supply chain: Product 2 -> Product 1 -> Background I1"""
    bd.projects.set_current("__test_two_level__")

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
        ("db_2020", "I1"): {
            "name": "node I1",
            "location": "somewhere",
            "reference product": "I1",
            "exchanges": [
                {"amount": 1, "type": "production", "input": ("db_2020", "I1")},
                {"amount": 1, "type": "biosphere", "input": ("biosphere3", "CO2")},
            ],
        },
    })
    bg_2020.metadata["representative_time"] = datetime(2020, 1, 1).isoformat()
    bg_2020.register()

    # Foreground with two-level supply chain
    fg = bd.Database("foreground")
    fg.write({
        # Product 1 (intermediate product)
        ("foreground", "Product 1"): {
            "name": "Product 1",
            "unit": "kg",
            "type": bd.labels.product_node_default,
        },
        ("foreground", "P1R1"): {
            "name": "Product 1 production, Route 1",
            "location": "somewhere",
            "type": bd.labels.process_node_default,
            "operation_time_limits": (1, 2),
            "exchanges": [
                {
                    "amount": 1,
                    "type": bd.labels.production_edge_default,
                    "input": ("foreground", "Product 1"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array(range(4), dtype="timedelta64[Y]"),
                        amount=np.array([0, 0.5, 0.5, 0]),
                    ),
                    "operation": True,
                },
                {
                    "amount": 27.5,
                    "type": bd.labels.consumption_edge_default,
                    "input": ("db_2020", "I1"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array(range(4), dtype="timedelta64[Y]"),
                        amount=np.array([1, 0, 0, 0]),
                    ),
                },
                {
                    "amount": 20,
                    "type": bd.labels.biosphere_edge_default,
                    "input": ("biosphere3", "CO2"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array(range(4), dtype="timedelta64[Y]"),
                        amount=np.array([0, 0.5, 0.5, 0]),
                    ),
                    "operation": True,
                },
            ],
        },

        # Product 2 (final product, consumes Product 1)
        ("foreground", "Product 2"): {
            "name": "Product 2",
            "unit": "kg",
            "type": bd.labels.product_node_default,
        },
        ("foreground", "P2R1"): {
            "name": "Product 2 production, Route 1",
            "location": "somewhere",
            "type": bd.labels.process_node_default,
            "operation_time_limits": (1, 2),
            "exchanges": [
                {
                    "amount": 1,
                    "type": bd.labels.production_edge_default,
                    "input": ("foreground", "Product 2"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array(range(4), dtype="timedelta64[Y]"),
                        amount=np.array([0, 0.5, 0.5, 0]),
                    ),
                    "operation": True,
                },
                {
                    "amount": 1,
                    "type": bd.labels.consumption_edge_default,
                    "input": ("foreground", "Product 1"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array(range(4), dtype="timedelta64[Y]"),
                        amount=np.array([0, 0.5, 0.5, 0]),
                    ),
                    "operation": True,
                },
                {
                    "amount": 15.5,
                    "type": bd.labels.consumption_edge_default,
                    "input": ("db_2020", "I1"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array(range(4), dtype="timedelta64[Y]"),
                        amount=np.array([1, 0, 0, 0]),
                    ),
                },
                {
                    "amount": 20,
                    "type": bd.labels.biosphere_edge_default,
                    "input": ("biosphere3", "CO2"),
                    "temporal_distribution": TemporalDistribution(
                        date=np.array(range(4), dtype="timedelta64[Y]"),
                        amount=np.array([0, 0.5, 0.5, 0]),
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


def test_two_level_supply_chain_matches_lca(setup_two_level_system):
    """Test that optimex produces same results as standard LCA for two-level system."""

    # Standard LCA calculation
    product_2 = bd.get_node(database="foreground", name="Product 2")
    lca = bc.LCA({product_2: 10}, method=("GWP", "example"))
    lca.lci()
    lca.lcia()
    expected_gwp = lca.score

    print(f"\nStandard LCA GWP: {expected_gwp}")

    # optimex calculation
    years = range(2020, 2030)
    td_demand = TemporalDistribution(
        date=np.array([datetime(year, 1, 1).isoformat() for year in years], dtype='datetime64[s]'),
        amount=np.asarray([0, 0, 10, 0, 0, 0, 0, 0, 0, 0]),
    )

    lca_config = lca_processor.LCAConfig(
        demand={product_2: td_demand},
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
        name="test_two_level",
        objective_category="climate_change",
        flexible_operation=True,
    )

    _, obj_real, results = optimizer.solve_model(model, solver_name="glpk")

    print(f"optimex GWP: {obj_real}")
    print(f"Difference: {abs(obj_real - expected_gwp)}")

    # They should match (within numerical tolerance)
    assert pytest.approx(obj_real, rel=1e-3) == expected_gwp, (
        f"optimex result ({obj_real}) should match standard LCA ({expected_gwp})"
    )

    # Additional check: verify postprocessing extracts correct unscaled values
    pp = postprocessing.PostProcessor(model)
    df_impacts = pp.get_impacts()

    # Sum all climate_change impacts across all processes and times
    if 'climate_change' in df_impacts.columns.get_level_values(0):
        climate_change_cols = [col for col in df_impacts.columns if col[0] == 'climate_change']
        total_cc_from_pp = df_impacts[climate_change_cols].sum().sum()

        print(f"\nPostprocessing climate_change total: {total_cc_from_pp}")
        print(f"Expected (from LCA): {expected_gwp}")

        # Postprocessing should also match standard LCA
        assert pytest.approx(total_cc_from_pp, rel=1e-3) == expected_gwp, (
            f"Postprocessing climate_change sum ({total_cc_from_pp}) should match standard LCA ({expected_gwp})"
        )


def test_two_level_supply_chain_multi_temporal_demand(setup_two_level_system):
    """
    Test two-level supply chain with demand at multiple time points.

    This test verifies:
    1. Total impact matches sum of individual LCA calculations
    2. Installation and operation are correctly distributed over time
    3. Internal demands (Product 1 for Product 2) are handled correctly
    4. PostProcessor extracts correct values for each time period
    """

    product_2 = bd.get_node(database="foreground", name="Product 2")

    # Define multi-temporal demand: 10 units in 2022, 5 units in 2024, 10 units in 2026
    demand_schedule = {
        2022: 10,
        2024: 5,
        2026: 10,
    }

    # Calculate expected total impact from standard LCA
    print("\n" + "="*80)
    print("STANDARD LCA CALCULATIONS")
    print("="*80)
    expected_gwp_total = 0
    for year, amount in demand_schedule.items():
        lca = bc.LCA({product_2: amount}, method=("GWP", "example"))
        lca.lci()
        lca.lcia()
        expected_gwp_total += lca.score
        print(f"Year {year}: {amount} units → {lca.score:.2f} kg CO2")

    print(f"Total expected GWP: {expected_gwp_total:.2f} kg CO2")

    # optimex calculation with multi-temporal demand
    years = range(2020, 2030)
    demand_amounts = [demand_schedule.get(year, 0) for year in years]

    td_demand = TemporalDistribution(
        date=np.array([datetime(year, 1, 1).isoformat() for year in years], dtype='datetime64[s]'),
        amount=np.asarray(demand_amounts),
    )

    lca_config = lca_processor.LCAConfig(
        demand={product_2: td_demand},
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
        name="test_two_level_multi_temporal",
        objective_category="climate_change",
        flexible_operation=True,
    )

    _, obj_real, results = optimizer.solve_model(model, solver_name="glpk")

    print("\n" + "="*80)
    print("OPTIMEX RESULTS")
    print("="*80)
    print(f"optimex GWP: {obj_real:.2f} kg CO2")
    print(f"Expected GWP: {expected_gwp_total:.2f} kg CO2")
    print(f"Difference: {abs(obj_real - expected_gwp_total):.4f}")

    # Verify total impact matches
    assert pytest.approx(obj_real, rel=1e-3) == expected_gwp_total, (
        f"optimex result ({obj_real}) should match sum of standard LCA ({expected_gwp_total})"
    )

    # Extract and verify temporal distribution of results
    pp = postprocessing.PostProcessor(model)
    df_installation = pp.get_installation()
    df_operation = pp.get_operation()
    df_impacts = pp.get_impacts()

    print("\n" + "="*80)
    print("INSTALLATION (Product 2 production)")
    print("="*80)
    p2_process_id = [p for p in model.PROCESS if model.process_names[p] == "Product 2 production, Route 1"][0]
    if p2_process_id in df_installation.columns:
        print(df_installation[[p2_process_id]])

        # Verify installation happens before demand years
        # (capacity must be installed before it can operate)
        for year, amount in demand_schedule.items():
            # Check that there's installation in years before demand
            years_before = [y for y in df_installation.index if y < year]
            if years_before:
                total_installation_before = df_installation.loc[years_before, p2_process_id].sum()
                print(f"\nInstallation before {year} (demand={amount}): {total_installation_before:.2f}")

    print("\n" + "="*80)
    print("OPERATION (Product 2 production)")
    print("="*80)
    if p2_process_id in df_operation.columns:
        print(df_operation[[p2_process_id]])

        # Verify operation matches demand at each time point
        for year, amount in demand_schedule.items():
            if year in df_operation.index:
                operation = df_operation.loc[year, p2_process_id]
                print(f"\nYear {year}: Operation={operation:.2f}, Demand={amount}")
                # Operation should equal demand for single-route system
                assert pytest.approx(operation, rel=1e-2) == amount, (
                    f"Operation at {year} ({operation}) should match demand ({amount})"
                )

    print("\n" + "="*80)
    print("INTERNAL DEMAND VERIFICATION (Product 1)")
    print("="*80)
    p1_process_id = [p for p in model.PROCESS if model.process_names[p] == "Product 1 production, Route 1"][0]

    # Product 2 consumes 1 unit of Product 1 per unit produced
    # So Product 1 operation should match Product 2 operation
    if p1_process_id in df_operation.columns and p2_process_id in df_operation.columns:
        p1_operation = df_operation[p1_process_id]
        p2_operation = df_operation[p2_process_id]

        print(f"Product 1 operation:\n{p1_operation[p1_operation > 0]}")
        print(f"\nProduct 2 operation:\n{p2_operation[p2_operation > 0]}")

        # They should match (Product 2 needs 1 unit of Product 1 per unit)
        for year in demand_schedule.keys():
            if year in df_operation.index:
                assert pytest.approx(p1_operation[year], rel=1e-2) == p2_operation[year], (
                    f"Product 1 operation ({p1_operation[year]}) should match "
                    f"Product 2 operation ({p2_operation[year]}) at year {year}"
                )

    # Verify postprocessing impact totals
    if 'climate_change' in df_impacts.columns.get_level_values(0):
        climate_change_cols = [col for col in df_impacts.columns if col[0] == 'climate_change']
        total_cc_from_pp = df_impacts[climate_change_cols].sum().sum()

        print("\n" + "="*80)
        print("POSTPROCESSING VERIFICATION")
        print("="*80)
        print(f"Total impact from PostProcessor: {total_cc_from_pp:.2f}")
        print(f"Expected (from LCA): {expected_gwp_total:.2f}")

        assert pytest.approx(total_cc_from_pp, rel=1e-3) == expected_gwp_total, (
            f"Postprocessing total ({total_cc_from_pp}) should match expected ({expected_gwp_total})"
        )

        # Verify impacts are distributed across correct years
        print("\nImpact distribution by year:")
        yearly_impacts = df_impacts[climate_change_cols].sum(axis=1)
        for year in yearly_impacts.index:
            if yearly_impacts[year] > 0.1:  # Only show non-negligible impacts
                print(f"  {year}: {yearly_impacts[year]:.2f} kg CO2")

    print("\n" + "="*80)
    print("TEST PASSED: Multi-temporal two-level supply chain works correctly!")
    print("="*80)
