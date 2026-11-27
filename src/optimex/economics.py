"""
Helpers for the cost objective.

The cost objective prices the background products that the foreground purchases
directly. Prices are stored as an attribute of the background nodes, one value per
time-specific background database, so they evolve over time like the inventories:

1. `set_market_prices` writes the prices to the nodes.
2. `LCADataProcessor` reads them when `LCAConfig.price_attribute` is set and
   interpolates them to system time (`intermediate_flow_prices`).
3. `create_model(..., objective="cost")` minimizes the discounted cost.

Prices can carry an uncertainty distribution, in the same Brightway form as
exchanges, which `optimex.uncertainty` samples.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import bw2data as bd
import pandas as pd
from loguru import logger

PriceData = Union[pd.DataFrame, Sequence[Mapping[str, Any]]]


def set_market_prices(
    price_data: PriceData,
    background_databases: dict[int, str],
    name_col: str = "name",
    year_col: str = "year",
    price_col: str = "price",
    location_col: Optional[str] = "location",
    product_col: Optional[str] = None,
    unit_col: Optional[str] = None,
    price_attribute: str = "market_price",
    overwrite: bool = True,
    strict: bool = True,
) -> None:
    """
    Write market prices to the nodes of time-specific background databases.

    Each row of `price_data` names a background node, the year of the background
    database it belongs to, and its price per unit of the node's reference product.
    Nodes are identified by name and, optionally, location, reference product, and
    unit rather than by code, since premise gives the same activity a different code
    in every scenario database.

    No unit, currency, or inflation conversion is applied: use one currency and
    price basis for all prices.

    Parameters
    ----------
    price_data : pandas.DataFrame or sequence of dict
        One row per (node, year). By default, rows use the keys ``name``,
        ``location``, ``year``, and ``price``.
    background_databases : dict[int, str]
        Maps the representative year of each background database to its name,
        e.g. ``{2020: "ei_2020", 2030: "ei_2030"}``.
    name_col, year_col, price_col : str
        Columns holding the node name, database year, and price.
    location_col : str or None
        Column holding the node location. `None` matches nodes in any location.
    product_col : str or None
        Column holding the reference product, to tell apart nodes with the same
        name and location.
    unit_col : str or None
        Column holding the unit, to tell apart nodes with the same name, location,
        and reference product. Only used for matching; prices are not converted.
    price_attribute : str
        Node attribute the price is written to. Pass the same name as
        `LCAConfig.price_attribute`.
    overwrite : bool
        If False, nodes that already have a price keep it.
    strict : bool
        If True, rows with missing columns, unknown years, unresolvable nodes, or
        non-numeric prices raise a `ValueError`. If False, they are logged and
        skipped.

    Notes
    -----
    A row can also describe the uncertainty of its price, with the Brightway
    uncertainty columns ``uncertainty type`` (a `stats_arrays` id, e.g. 2 for
    lognormal, 3 for normal, 4 for uniform, 5 for triangular) and, as the
    distribution needs them, ``loc``, ``scale``, ``shape``, ``minimum``, and
    ``maximum``. A missing ``loc`` is derived from the price: its natural
    logarithm for a lognormal distribution, the price itself otherwise (the mode
    of a triangular distribution). The distribution is written to the attribute
    ``f"{price_attribute}_uncertainty"`` and sampled by `optimex.uncertainty`;
    the price itself stays the deterministic value. A row without an uncertainty
    type removes a distribution written earlier.

    Raises
    ------
    TypeError
        If `price_data` is neither a DataFrame nor a sequence of mappings.
    ValueError
        In strict mode, for the first row that cannot be applied.
    """
    records = _records_from_price_data(price_data)
    match_cols = {
        "name": name_col,
        "location": location_col,
        "product": product_col,  # bw2data's field for the reference product
        "unit": unit_col,
    }
    match_cols = {field: col for field, col in match_cols.items() if col is not None}
    required_cols = [*match_cols.values(), year_col, price_col]

    written = 0
    for index, record in enumerate(records):
        missing = [col for col in required_cols if col not in record]
        if missing:
            _warn_or_raise(f"Row {index} misses the columns {missing}.", strict)
            continue

        year = int(record[year_col])
        if year not in background_databases:
            _warn_or_raise(
                f"Row {index}: no background database configured for year {year}.",
                strict,
            )
            continue

        price = _as_price(record[price_col])
        if price is None:
            _warn_or_raise(
                f"Row {index}: price {record[price_col]!r} is not a finite number.",
                strict,
            )
            continue

        lookup = {"database": background_databases[year]}
        lookup.update({field: record[col] for field, col in match_cols.items()})
        try:
            node = bd.get_node(**lookup)
        except Exception as exc:  # unknown or ambiguous node
            _warn_or_raise(
                f"Row {index}: could not find background node {lookup}: {exc}",
                strict,
            )
            continue

        if not overwrite and node.get(price_attribute) is not None:
            logger.info(
                f"Keeping existing {price_attribute} of '{node['name']}' in "
                f"'{lookup['database']}'."
            )
            continue

        try:
            uncertainty = _price_uncertainty(record, price)
        except ValueError as exc:
            _warn_or_raise(f"Row {index}: {exc}", strict)
            continue

        node[price_attribute] = price
        if uncertainty is not None:
            node[f"{price_attribute}_uncertainty"] = uncertainty
        else:
            node._data.pop(f"{price_attribute}_uncertainty", None)
        node.save()
        written += 1

    logger.info(f"Wrote {written} market prices to '{price_attribute}'.")


def _records_from_price_data(price_data: PriceData) -> list[Mapping[str, Any]]:
    """Convert a DataFrame or a sequence of mappings to a list of mappings."""
    if isinstance(price_data, pd.DataFrame):
        records = price_data.to_dict(orient="records")
    elif isinstance(price_data, Sequence) and not isinstance(price_data, (str, bytes)):
        records = list(price_data)
    else:
        raise TypeError(
            "price_data must be a pandas DataFrame or a sequence of dictionaries."
        )

    if not all(isinstance(record, Mapping) for record in records):
        raise TypeError("price_data rows must be dictionaries or other mappings.")
    return records


def _as_price(value: Any) -> Optional[float]:
    """Return `value` as a finite float, or None if it is not one."""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) else None


def _price_uncertainty(record: Mapping[str, Any], price: float) -> Optional[dict]:
    """The uncertainty distribution described by a price row, or None."""
    from optimex.uncertainty import uncertainty_from_record

    return uncertainty_from_record(record, price)


def _warn_or_raise(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    logger.warning(message)
