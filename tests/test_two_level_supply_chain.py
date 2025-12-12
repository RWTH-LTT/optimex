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
