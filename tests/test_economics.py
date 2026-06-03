"""
Tests for economic helper utilities.
"""

import bw2data as bd
import pandas as pd
import pytest

from optimex.economics import set_market_prices


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
