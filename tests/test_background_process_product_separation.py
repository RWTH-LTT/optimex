"""
Test for background database with separated process and product nodes.

This test verifies that optimex correctly handles background databases where
processes and products are separate nodes (not "chimera" processes where a
process produces itself). The foreground demands a background product, and
there is a separate background process that produces this product.

This structure mirrors how foreground databases work in optimex and tests
compatibility with standard Brightway LCA calculations.
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
def setup_separated_background_system():
    """
    Set up a system where background database has separate process and product nodes.
    
    Background structure:
    - Product node: ("bg_2020", "electricity") - the product
    - Process node: ("bg_2020", "electricity_production") - the process producing it
    
    Foreground structure:
    - Product node: ("foreground", "Widget")
    - Process node: ("foreground", "Widget_Route1") - consumes background electricity product
    """
    bd.projects.set_current("__test_background_separation__")

    # Biosphere
    bio_db = bd.Database("biosphere3")
    bio_db.write({
        ("biosphere3", "CO2"): {
            "type": "emission",
            "name": "carbon dioxide",
        },
    })
    bio_db.register()

    # Background database with SEPARATED process and product nodes
    bg_2020 = bd.Database("bg_2020")
    bg_2020.write({
        # Background PRODUCT node
        ("bg_2020", "electricity"): {
            "name": "electricity",
            "unit": "kWh",
            "type": bd.labels.product_node_default,
        },
        # Background PROCESS node that produces the electricity product
        ("bg_2020", "electricity_production"): {
            "name": "electricity production process",
            "location": "GLO",
            "type": bd.labels.process_node_default,
            "exchanges": [
                {
                    "amount": 1,
                    "type": "production",
                    "input": ("bg_2020", "electricity"),  # Produces the electricity PRODUCT
                },
                {
                    "amount": 0.5,
                    "type": "biosphere",
                    "input": ("biosphere3", "CO2"),
                },
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
                    "input": ("bg_2020", "electricity"),  # Demands the PRODUCT
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


def test_separated_background_matches_standard_lca(setup_separated_background_system):
    """
    Test that optimex produces same results as standard LCA when background has
    separate process and product nodes.

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
        name="test_separated_background",
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


def test_separated_background_process_linkage(setup_separated_background_system):
    """
    Verify that background product correctly links to background process.
    
    This test ensures that when the foreground demands a background product,
    the LCA processor correctly identifies and uses the background process
    that produces that product.
    """
    
    # Get the nodes
    electricity_product = bd.get_node(database="bg_2020", name="electricity")
    electricity_process = bd.get_node(database="bg_2020", name="electricity production process")
    
    # Verify they are correctly typed
    assert electricity_product.get('type') == bd.labels.product_node_default, (
        "Electricity should be a product node"
    )
    assert electricity_process.get('type') == bd.labels.process_node_default, (
        "Electricity production should be a process node"
    )
    
    # Verify the process produces the product
    production_exchanges = [
        exc for exc in electricity_process.exchanges()
        if exc.get('type') == "production"
    ]
    
    assert len(production_exchanges) == 1, (
        "Process should have exactly one production exchange"
    )
    
    assert production_exchanges[0]['input'] == electricity_product.key, (
        "Process should produce the electricity product"
    )
    
    # Verify standard LCA can handle this structure
    widget = bd.get_node(database="foreground", name="Widget")
    lca = bc.LCA({widget: 1}, method=("GWP", "example"))
    lca.lci()
    lca.lcia()
    
    # The LCA should complete successfully with a non-zero result
    assert lca.score > 0, "LCA should produce a positive impact score"
    
    print(f"\nBackground process-product linkage verified!")
    print(f"Product node: {electricity_product}")
    print(f"Process node: {electricity_process}")
    print(f"LCA score for 1 kg Widget: {lca.score}")
