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


def test_single_route_matches_standard_lca(setup_single_route_system):
    """
    Test that optimex produces same results as standard LCA for a single-route system.

    Expected calculation for 100 kg Widget:
    - Direct CO2 from operation: 100 * 5 * 1.0 = 500 kg CO2
    - Background electricity at construction: 100 * 10 * 0.5 = 500 kg CO2
    - Total: 1000 kg CO2
    """

    # Standard LCA calculation
    widget = bd.get_node(database="foreground", name="Widget")
    lca = bc.LCA({widget: 100}, method=("GWP", "example"))
    lca.lci()
    lca.lcia()
    expected_gwp = lca.score

    print(f"\nStandard LCA GWP: {expected_gwp}")

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
