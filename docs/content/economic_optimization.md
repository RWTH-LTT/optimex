---
icon: lucide/badge-euro
tags:
  - optimization
  - economics
  - costs
---

# Economic Optimization

`optimex` can be extended with economic cost data by assigning market prices to
time-specific background products. These prices are later applied to the
foreground system's direct first-level purchases from the background system.

!!! note "Current implementation status"

    Economic optimization is under active development. The helper below prepares
    Brightway background databases with `market_price` attributes. The optimizer
    integration reads these attributes in later implementation steps.

## Modeling Choice

Market prices are stored on background nodes, not on foreground edges.

This keeps the model structure consistent with the foreground-background
separation:

- foreground edges determine which background products are purchased and in what amount
- the `operation` edge attribute determines whether purchases are installation-related or operation-related
- background nodes provide the market price of the purchased product

This avoids duplicating the same market price on many foreground edges.

## Time-Specific Prices

When using prospective background databases, for example from `premise`, prices
are assigned to the corresponding node in each time-specific background database:

```text
background 2020 node -> market_price for 2020
background 2030 node -> market_price for 2030
background 2040 node -> market_price for 2040
```

`optimex` can then use the same background mapping logic already used for
environmental inventories to interpolate prices between representative years.

## Setting Market Prices

Use `set_market_prices()` to write market prices from tabular input data to the
correct Brightway background nodes.

```python
from optimex.economics import set_market_prices

price_data = [
    {
        "name": "market group for electricity, medium voltage",
        "location": "RER",
        "year": 2020,
        "price": 90.0,
    },
    {
        "name": "market group for electricity, medium voltage",
        "location": "RER",
        "year": 2030,
        "price": 70.0,
    },
]

set_market_prices(
    price_data=price_data,
    background_databases={
        2020: "ei312_REMIND-EU_SSP2_NDC_2020",
        2030: "ei312_REMIND-EU_SSP2_NDC_2030",
    },
)
```

This writes:

```python
node["market_price"] = price
node.save()
```

to each matching background node.

## Custom Column Names

`price_data` can also be a pandas DataFrame. If your columns use different
names, map them explicitly:

```python
set_market_prices(
    price_data=prices,
    background_databases=background_databases,
    name_col="process",
    location_col="region",
    year_col="scenario_year",
    price_col="eur_per_unit",
)
```

## Safety Options

Use `strict=True` to fail when a year, database, or node cannot be found.

Use `overwrite=False` to keep existing `market_price` attributes unchanged.

```python
set_market_prices(
    price_data=price_data,
    background_databases=background_databases,
    overwrite=False,
    strict=True,
)
```

## Cap/Opex Accounting

The price itself is a property of the background product. The cap/op split comes
from the foreground edge:

```text
operation=True  -> operation-related purchases
operation=False -> installation-related purchases
```

The same market price can therefore contribute to either cost account depending
on how the background product is used by the foreground system.

