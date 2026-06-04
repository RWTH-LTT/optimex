"""
Tests for economic helper utilities.
"""

from datetime import datetime

import bw2data as bd
import numpy as np
import pandas as pd
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution

from optimex import converter, lca_processor
from optimex.economics import set_market_prices


@pytest.fixture
@bw2test
def setup_economic_pipeline_databases():
    """Set up a small system with cap- and op-relevant background inputs."""
    bd.projects.set_current("__test_economic_pipeline__")

    bio_db = bd.Database("biosphere3")
    bio_db.write(
        {
            ("biosphere3", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
            },
        }
    )
    bio_db.register()

    for db_name, representative_year, co2_factor in (
        ("db_2020", 2020, 1.0),
        ("db_2030", 2030, 0.5),
    ):
        db = bd.Database(db_name)
        db.write(
            {
                (db_name, "I1"): {
                    "name": "node I1",
                    "location": "somewhere",
                    "reference product": "I1",
                    "exchanges": [
                        {
                            "amount": 1,
                            "type": "production",
                            "input": (db_name, "I1"),
                        },
                        {
                            "amount": co2_factor,
                            "type": "biosphere",
                            "input": ("biosphere3", "CO2"),
                        },
                    ],
                },
                (db_name, "I2"): {
                    "name": "node I2",
                    "location": "somewhere",
                    "reference product": "I2",
                    "exchanges": [
                        {
                            "amount": 1,
                            "type": "production",
                            "input": (db_name, "I2"),
                        },
                        {
                            "amount": co2_factor,
                            "type": "biosphere",
                            "input": ("biosphere3", "CO2"),
                        },
                    ],
                },
            }
        )
        db.metadata["representative_time"] = datetime(
            representative_year, 1, 1
        ).isoformat()
        db.register()

    foreground = bd.Database("foreground")
    foreground.write(
        {
            ("foreground", "R1"): {
                "name": "Product R1",
                "type": bd.labels.product_node_default,
                "unit": "kg",
            },
            ("foreground", "P1"): {
                "name": "process P1",
                "location": "somewhere",
                "type": bd.labels.process_node_default,
                "operation_time_limits": (0, 0),
                "exchanges": [
                    {
                        "amount": 1,
                        "type": bd.labels.production_edge_default,
                        "input": ("foreground", "R1"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([0], dtype="timedelta64[Y]"),
                            amount=np.array([1]),
                        ),
                        "operation": True,
                    },
                    {
                        "amount": 1,
                        "type": bd.labels.consumption_edge_default,
                        "input": ("db_2020", "I1"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([0], dtype="timedelta64[Y]"),
                            amount=np.array([1]),
                        ),
                    },
                    {
                        "amount": 1,
                        "type": bd.labels.consumption_edge_default,
                        "input": ("db_2020", "I2"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([0], dtype="timedelta64[Y]"),
                            amount=np.array([1]),
                        ),
                        "operation": True,
                    },
                ],
            },
        }
    )
    foreground.register()

    bd.Method(("GWP", "example")).write([(("biosphere3", "CO2"), 1)])


def _build_economic_pipeline_processor():
    years = range(2020, 2030)
    demand = TemporalDistribution(
        date=np.array(
            [datetime(year, 1, 1).isoformat() for year in years],
            dtype="datetime64[s]",
        ),
        amount=np.ones(len(years)),
    )
    product = bd.get_node(database="foreground", code="R1")
    config = lca_processor.LCAConfig(
        demand={product: demand},
        temporal={
            "start_date": datetime(2020, 1, 1),
            "temporal_resolution": "year",
            "time_horizon": 100,
        },
        characterization_methods=[
            {
                "category_name": "climate_change",
                "brightway_method": ("GWP", "example"),
            }
        ],
        background_inventory={
            "cutoff": 1e4,
            "calculation_method": "sequential",
        },
    )
    return lca_processor.LCADataProcessor(config)


def test_set_market_prices_from_list_of_dicts(setup_brightway_databases):
    """Market prices can be written from list-of-dict input."""
    set_market_prices(
        price_data=[
            {
                "name": "node I1",
                "location": "somewhere",
                "year": 2020,
                "price": 42.0,
            }
        ],
        background_databases={2020: "db_2020"},
    )

    node = bd.get_node(database="db_2020", name="node I1", location="somewhere")
    assert node["market_price"] == 42.0


def test_set_market_prices_from_dataframe(setup_brightway_databases):
    """Market prices can be written from pandas DataFrame input."""
    prices = pd.DataFrame(
        [
            {
                "name": "node I2",
                "location": "somewhere",
                "year": 2030,
                "price": 84.0,
            }
        ]
    )

    set_market_prices(
        price_data=prices,
        background_databases={2030: "db_2030"},
    )

    node = bd.get_node(database="db_2030", name="node I2", location="somewhere")
    assert node["market_price"] == 84.0


def test_set_market_prices_supports_custom_column_names(setup_brightway_databases):
    """Column/key names can be mapped to user-provided tabular schemas."""
    set_market_prices(
        price_data=[
            {
                "process": "node I1",
                "region": "somewhere",
                "scenario_year": 2030,
                "eur_per_unit": 21.0,
            }
        ],
        background_databases={2030: "db_2030"},
        name_col="process",
        location_col="region",
        year_col="scenario_year",
        price_col="eur_per_unit",
    )

    node = bd.get_node(database="db_2030", name="node I1", location="somewhere")
    assert node["market_price"] == 21.0


def test_set_market_prices_respects_overwrite_false(setup_brightway_databases):
    """Existing prices are preserved when overwrite is disabled."""
    node = bd.get_node(database="db_2020", name="node I1", location="somewhere")
    node["market_price"] = 100.0
    node.save()

    set_market_prices(
        price_data=[
            {
                "name": "node I1",
                "location": "somewhere",
                "year": 2020,
                "price": 50.0,
            }
        ],
        background_databases={2020: "db_2020"},
        overwrite=False,
    )

    node = bd.get_node(database="db_2020", name="node I1", location="somewhere")
    assert node["market_price"] == 100.0


def test_set_market_prices_strict_missing_year_raises(setup_brightway_databases):
    """Missing background database years fail in strict mode."""
    with pytest.raises(ValueError, match="No background database configured"):
        set_market_prices(
            price_data=[
                {
                    "name": "node I1",
                    "location": "somewhere",
                    "year": 2040,
                    "price": 42.0,
                }
            ],
            background_databases={2020: "db_2020"},
        )


def test_set_market_prices_strict_missing_node_raises(setup_brightway_databases):
    """Missing background nodes fail in strict mode."""
    with pytest.raises(ValueError, match="Could not find background node"):
        set_market_prices(
            price_data=[
                {
                    "name": "missing node",
                    "location": "somewhere",
                    "year": 2020,
                    "price": 42.0,
                }
            ],
            background_databases={2020: "db_2020"},
        )


def test_market_prices_flow_to_model_inputs_with_mapping(
    setup_economic_pipeline_databases,
):
    """Node prices are read, interpolated, and passed to OptimizationModelInputs."""
    set_market_prices(
        price_data=[
            {
                "name": "node I1",
                "location": "somewhere",
                "year": 2020,
                "price": 100.0,
            },
            {
                "name": "node I1",
                "location": "somewhere",
                "year": 2030,
                "price": 50.0,
            },
        ],
        background_databases={2020: "db_2020", 2030: "db_2030"},
    )

    processor = _build_economic_pipeline_processor()

    assert processor.background_costs[("db_2020", "I1")] == 100.0
    assert processor.background_costs[("db_2030", "I1")] == 50.0
    assert processor.intermediate_costs_cap[("I1", 2020)] == 100.0
    assert processor.intermediate_costs_cap[("I1", 2025)] == 75.0
    assert processor.intermediate_costs_cap[("I1", 2029)] == 55.0
    assert ("I1", 2025) not in processor.intermediate_costs_op

    manager = converter.ModelInputManager()
    model_inputs = manager.parse_from_lca_processor(processor)
    assert model_inputs.intermediate_costs_cap[("I1", 2025)] == 75.0


def test_operation_edge_prices_flow_to_operation_costs(
    setup_economic_pipeline_databases,
):
    """Prices for operation-edge inputs are written to op costs, not cap costs."""
    set_market_prices(
        price_data=[
            {
                "name": "node I2",
                "location": "somewhere",
                "year": 2020,
                "price": 20.0,
            },
            {
                "name": "node I2",
                "location": "somewhere",
                "year": 2030,
                "price": 10.0,
            },
        ],
        background_databases={2020: "db_2020", 2030: "db_2030"},
    )

    processor = _build_economic_pipeline_processor()

    assert processor.intermediate_costs_op[("I2", 2025)] == 15.0
    assert ("I2", 2025) not in processor.intermediate_costs_cap

    manager = converter.ModelInputManager()
    model_inputs = manager.parse_from_lca_processor(processor)
    assert model_inputs.intermediate_costs_op[("I2", 2025)] == 15.0


def test_missing_market_price_logs_warning(
    setup_economic_pipeline_databases,
    monkeypatch,
):
    """Missing market prices for cost-relevant flows are warned about."""
    messages = []

    def collect_warning(message, *args, **kwargs):
        messages.append(message.format(*args, **kwargs))

    monkeypatch.setattr(lca_processor.logger, "warning", collect_warning)
    set_market_prices(
        price_data=[
            {
                "name": "node I1",
                "location": "somewhere",
                "year": 2020,
                "price": 100.0,
            },
        ],
        background_databases={2020: "db_2020"},
    )

    processor = _build_economic_pipeline_processor()

    assert any(
        "Missing market_price" in message
        and "db_2030" in message
        and "I1" in message
        and "cap" in message
        for message in messages
    )
    assert processor.intermediate_costs_cap[("I1", 2025)] == 50.0
