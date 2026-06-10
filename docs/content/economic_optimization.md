---
icon: lucide/badge-euro
tags:
  - optimization
  - economics
  - costs
---

# Economic Optimization

`optimex` supports economic cost optimization by assigning time-specific market
prices to direct background products purchased by the foreground system. These
prices can be used to minimize total discounted costs instead of minimizing an
environmental impact category.

## Concept

Costs are applied only to first-level background purchases directly required by
the foreground system. They are not recursively applied to upstream products
inside the background inventory. The upstream background system remains relevant
for environmental impacts, while economic costs are represented by time-specific
market prices for direct background products.

Market prices are stored on background nodes, not on foreground edges. This keeps
the model structure consistent with the foreground-background separation:

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

`optimex` reads these prices and interpolates them to the model's `SYSTEM_TIME`
using the same background-to-system-time mapping that is used for time-specific
background data.

## Setting Market Prices

Use `set_market_prices()` to write market prices from tabular input data to the
correct Brightway background nodes. Nodes are identified by Brightway `name` and
`location`. If this is ambiguous, additional product and unit columns can be used.

```python
from optimex.economics import set_market_prices

price_data = [
    {
        "name": "market group for electricity, medium voltage",
        "location": "RER",
        "product": "electricity, medium voltage",
        "unit": "kilowatt hour",
        "year": 2020,
        "price": 90.0,
    },
    {
        "name": "market group for electricity, medium voltage",
        "location": "RER",
        "product": "electricity, medium voltage",
        "unit": "kilowatt hour",
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
    product_col="product",
    unit_col="unit",
)
```

This writes:

```python
node["market_price"] = price
node.save()
```

to each matching background node.

Brightway `code` is intentionally not used for this helper. In premise-generated
time-specific databases, equivalent activities can have different `code` and
`id` values across years. For example, the same electricity market group in 2020
and 2030 can share `name`, `reference product`, `unit`, and `location`, while
having different Brightway codes. Therefore, code is not a suitable identifier
for assigning one price time series across multiple premise databases.

Some Brightway databases contain multiple activities with the same name and
location but different products or units. In that case, include product and/or
unit columns and pass `product_col` and `unit_col`, matching the usual Brightway
lookup:

```python
set_market_prices(
    price_data=price_data,
    background_databases=background_databases,
    product_col="product",
    unit_col="unit",
)
```

## Custom Column Names

`price_data` can also be a pandas DataFrame. If your columns use different
names, map them explicitly:

```python
set_market_prices(
    price_data=prices,
    background_databases=background_databases,
    name_col="process",
    location_col="region",
    product_col="reference_product",
    unit_col="unit",
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

## Input Fields

Cost data enters the optimization model through `OptimizationModelInputs`:

| Field | Type | Description |
|-------|------|-------------|
| `intermediate_costs_cap` | `Dict[Tuple[str, int], float]` | Price for installation-related first-level background purchases, indexed by `(intermediate_flow, system_time)` |
| `intermediate_costs_op` | `Dict[Tuple[str, int], float]` | Price for operation-related first-level background purchases, indexed by `(intermediate_flow, system_time)` |
| `discount_rate` | `float` | Optional discount rate, for example `0.05` for 5% |
| `discount_reference_year` | `int` | Optional reference year for discounting. Defaults to the first system time |

When inputs are created from `LCADataProcessor`, `intermediate_costs_cap` and
`intermediate_costs_op` are filled from `market_price` attributes on the relevant
background nodes. They can also be overridden manually through
`ModelInputManager.override()`.

## CAPEX/OPEX Accounting

The price itself is a property of the background product. The cap/op split comes
from the foreground edge:

```text
operation=True  -> operation-related purchases
operation=False -> installation-related purchases
```

The same market price can therefore contribute to either cost account depending
on how the background product is used by the foreground system.

This is an accounting distinction. It does not require the background product to
have a different physical or market price when used during construction versus
operation.

## Cost Objective

The default objective is environmental:

```python
model = optimizer.create_model(
    inputs=model_inputs,
    name="environmental_model",
    objective_category="climate_change",
)
```

This is equivalent to:

```python
model = optimizer.create_model(
    inputs=model_inputs,
    name="environmental_model",
    objective_category="climate_change",
    objective="environmental",
)
```

To minimize total economic cost instead:

```python
model = optimizer.create_model(
    inputs=model_inputs,
    name="cost_model",
    objective_category="climate_change",
    objective="cost",
)

solved_model, objective_value, results = optimizer.solve_model(
    model,
    solver_name="highs",
)
```

`objective_category` is still required because environmental impacts and
environmental constraints may still be part of the model, even when the objective
is cost.

## Cost Expressions

The model calculates direct background purchases and applies prices to them:

```text
background_purchase_cap[i, t] = first-level installation-related purchase of i in t
background_purchase_op[i, t]  = first-level operation-related purchase of i in t

cost_cap[t] = sum_i intermediate_costs_cap[i, t] * background_purchase_cap[i, t]
cost_op[t]  = sum_i intermediate_costs_op[i, t]  * background_purchase_op[i, t]

total_cost = sum_t discount_factor[t] * (cost_cap[t] + cost_op[t])
```

After solving, the most relevant cost outputs are:

| Expression | Meaning |
|------------|---------|
| `model.cost_cap[t]` | Installation-related first-level background cost in year `t` |
| `model.cost_op[t]` | Operation-related first-level background cost in year `t` |
| `model.discount_factor[t]` | Discount factor applied to year `t` |
| `model.total_cost` | Discounted total cost minimized by `objective="cost"` |

## Environmental Constraints

Cost optimization can be combined with environmental constraints. For example,
minimize cost while respecting a cumulative climate budget:

```python
model_inputs = manager.override(
    cumulative_category_impact_limits={
        "climate_change": 1_000_000.0,
    },
)

model = optimizer.create_model(
    inputs=model_inputs,
    name="cost_with_climate_budget",
    objective_category="climate_change",
    objective="cost",
)
```

Time-specific environmental constraints can be added in the same way:

```python
model_inputs = manager.override(
    category_impact_limits={
        ("climate_change", 2030): 50_000.0,
    },
)
```

## Scaling

`optimex` scales LCA data internally for numerical stability. Cost inputs are
provided as real prices, and direct background purchases are converted back to
real units before costs are calculated. Therefore, the `objective_value` returned
by `solve_model()` for `objective="cost"` is already in the monetary unit used by
the input prices.
