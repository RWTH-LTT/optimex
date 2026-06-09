"""
Economic helper utilities for annotating background databases with market prices.

This module provides user-facing helpers for writing price data to Brightway
background nodes. The optimizer later reads these node attributes and applies
them to first-level background purchases.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import bw2data as bd
from loguru import logger


def _records_from_price_data(price_data: Any) -> list[Mapping[str, Any]]:
    """Convert supported tabular inputs to a list of mapping-like records."""
    if hasattr(price_data, "to_dict"):
        records = price_data.to_dict(orient="records")
    elif isinstance(price_data, Sequence) and not isinstance(price_data, (str, bytes)):
        records = list(price_data)
    else:
        raise TypeError(
            "price_data must be a pandas DataFrame or a sequence of dictionaries."
        )

    if not all(isinstance(record, Mapping) for record in records):
        raise TypeError("price_data records must be dictionaries or mapping-like rows.")

    return records


def _warn_or_raise(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    logger.warning(message)


def set_market_prices(
    price_data: Any,
    background_databases: dict[int, str],
    name_col: str = "name",
    year_col: str = "year",
    price_col: str = "price",
    location_col: str | None = "location",
    product_col: str | None = None,
    unit_col: str | None = None,
    price_attribute: str = "market_price",
    overwrite: bool = True,
    strict: bool = True,
) -> None:
    """
    Write market price attributes to time-specific Brightway background nodes.

    Parameters
    ----------
    price_data : pandas.DataFrame or sequence of dict
        Tabular price data. Each row must identify a background process, a year,
        and a price. By default, rows use ``name``, ``location``, ``year``, and
        ``price`` keys/columns.
    background_databases : dict[int, str]
        Mapping from representative year to Brightway background database name.
        Example: ``{2020: "ei_2020", 2030: "ei_2030"}``.
    name_col : str
        Column/key containing the Brightway node name.
    year_col : str
        Column/key containing the representative background year.
    price_col : str
        Column/key containing the market price value to write.
    location_col : str or None
        Optional column/key containing the Brightway node location. Set to
        ``None`` to look up nodes without a location filter.
    product_col : str or None
        Optional column/key containing the Brightway product or reference
        product. Use this to disambiguate activities with the same name and
        location but different products or units.
    unit_col : str or None
        Optional column/key containing the Brightway unit. Use this to
        disambiguate activities with the same name, location, and product but
        different units.
    price_attribute : str
        Brightway node attribute used to store the price.
    overwrite : bool
        If False, existing non-None price attributes are left unchanged.
    strict : bool
        If True, missing columns, years, or nodes raise errors. If False, they
        are logged as warnings and skipped.
    """
    records = _records_from_price_data(price_data)
    required_cols = [name_col, year_col, price_col]
    if location_col is not None:
        required_cols.append(location_col)
    if product_col is not None:
        required_cols.append(product_col)
    if unit_col is not None:
        required_cols.append(unit_col)

    for index, record in enumerate(records):
        missing = [col for col in required_cols if col not in record]
        if missing:
            _warn_or_raise(
                f"Missing columns {missing} in price_data row {index}.",
                strict,
            )
            continue

        year = int(record[year_col])
        if year not in background_databases:
            _warn_or_raise(
                f"No background database configured for year {year}.",
                strict,
            )
            continue

        db_name = background_databases[year]
        lookup = {
            "database": db_name,
            "name": record[name_col],
        }
        if location_col is not None:
            lookup["location"] = record[location_col]
        if product_col is not None:
            lookup["product"] = record[product_col]
        if unit_col is not None:
            lookup["unit"] = record[unit_col]

        try:
            node = bd.get_node(**lookup)
        except Exception as exc:
            _warn_or_raise(
                f"Could not find background node for row {index} with lookup "
                f"{lookup}: {exc}",
                strict,
            )
            continue

        if not overwrite and node.get(price_attribute) is not None:
            logger.info(
                "Skipping existing {} on node {} in database {}.",
                price_attribute,
                node.get("name"),
                db_name,
            )
            continue

        node[price_attribute] = record[price_col]
        node.save()
