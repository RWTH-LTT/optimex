---
icon: lucide/badge-euro
tags:
  - optimization
  - economics
  - costs
---

# Economic Optimization

The economic extension of `optimex` minimizes the discounted cost of background
products purchased directly by the foreground system. It uses the existing
installation and vintage-specific operation decisions; environmental impact
limits can constrain the cost-optimal pathway.

For a complete getting-started workflow, see the
[economic basic example](https://github.com/langeluca/optimex/blob/main/notebooks/basic_example_econ.ipynb).
The example illustrates model preparation, price assignment, discounting, and
economic and environmental results for a synthetic one-product system.

## Cost Boundary

The foreground comprises the explicitly modeled processes whose installation
and operation levels are decision variables. Background supply chains contain
no decision variables. Costs are assigned to the products purchased directly
across this boundary, referred to here as first-level background purchases.
The boundary depends on which processes are explicitly modeled.

Prices are not recursively applied to inputs within the background supply
chains: their upstream inventories are used for environmental assessment,
while their market prices represent the cost of the direct purchases.
Products exchanged internally between foreground processes receive no additional
market price. Their production costs enter through the background purchases
required by the modeled processes.

## Installation- and Operation-Related Costs

The cost split follows the decision that determines a purchase, not a separate
financial-accounting classification:

| Foreground exchange attribute | Purchase depends on |
|---|---|
| `operation=True` | Operation of an installation vintage at a given system time |
| `operation=False`, or omitted | Installed process units of a vintage |

Installation-related purchases can include construction, replacements, and
end-of-life activities, depending on the inventory and its temporal distributions.
They need not all occur in the installation year. These two groups are therefore
not automatically equivalent to financial CAPEX and OPEX.

### Purchase Time and Price Time

For an installation in year `v`, an installation-related exchange at process-time
offset `delta` occurs at system time `t = v + delta`. Its price and discount factor
are evaluated at that system time. For example, a replacement ten years after a
2030 installation is priced and discounted in 2040, not 2030.
Operation-related purchases use the price at their operation time.

Foreground exchanges define quantities and the installation/operation role.
The `market_price` attribute on a background node provides the price per unit
of the purchased product. The processor uses this price for either role in
which the product is required. Role-specific price inputs can also be supplied
manually through `ModelInputManager.override()`.

## Setting Market Prices

Use `set_market_prices()` to assign prices to background nodes before processing
the LCA data. Each price must refer to one unit of the purchased product.
Use a consistent currency and price basis across all inputs. For example,
90 EUR/MWh must be converted to 0.09 EUR/kWh for a product measured in kilowatt hours.
The helper performs no unit conversion, currency conversion, or inflation adjustment.
Its `unit_col` argument helps identify nodes; it does not convert prices.

The following illustrative prices are expressed in constant 2020 EUR/kWh.
Replace the database names and node identifiers with those of your project.

```python
from optimex.economics import set_market_prices

background_databases = {
    2020: "ei312_REMIND-EU_SSP2_NDC_2020",
    2030: "ei312_REMIND-EU_SSP2_NDC_2030",
}
price_data = [
    {
        "name": "market group for electricity, medium voltage",
        "location": "RER",
        "product": "electricity, medium voltage",
        "unit": "kilowatt hour",
        "year": 2020,
        "price": 0.09,
    },
    {
        "name": "market group for electricity, medium voltage",
        "location": "RER",
        "product": "electricity, medium voltage",
        "unit": "kilowatt hour",
        "year": 2030,
        "price": 0.07,
    },
]

set_market_prices(
    price_data=price_data,
    background_databases=background_databases,
    product_col="product",
    unit_col="unit",
    strict=True,
)
```

The helper writes `node["market_price"]` and saves each matching node. It matches
name and location, optionally narrowed by reference product and unit.
Equivalent nodes in different time-specific databases need not share Brightway
codes or IDs, so the helper does not use these as cross-database identifiers.

### Time-Specific Background Prices

Assign prices to the relevant nodes in each time-specific background database.
The processor maps them to system times using the background interpolation
weights: prices are interpolated between database years, with the nearest
available database used outside that range. Prospective inventories, for example
from `premise`, do not by themselves supply the required market prices.

### Custom Columns and Overwriting

Input can also be a pandas DataFrame. The helper accepts `name_col`,
`location_col`, `product_col`, `unit_col`, `year_col`, and `price_col` to map
custom column names. Use `overwrite=False` to preserve existing
`market_price` attributes.

### Validation and Missing Prices

`strict=True` validates the submitted price records and raises errors for
unresolved lookups. It does **not** check whether every purchase in the
optimization model has been priced.

The LCA processor warns about missing price attributes or unavailable nodes.
Missing prices contribute zero when prices are mapped across background years;
the remaining interpolation weights are not renormalized. For example, equal
weights on a price of 0.10 and a missing price produce 0.05, not 0.10.
Economic price parameters omitted from the optimization inputs also default to
zero. Review the warnings and price coverage: a zero contribution to the
objective is not evidence that a purchase is free.

## Model Inputs and Discounting

`OptimizationModelInputs` carries the economic data:

| Field | Meaning |
|---|---|
| `intermediate_costs_cap[(i, t)]` | Price per unit of installation-related background product `i` at system time `t` |
| `intermediate_costs_op[(i, t)]` | Price per unit of operation-related background product `i` at system time `t` |
| `discount_rate` | Nonnegative annual rate `r`; defaults to zero |
| `discount_reference_year` | Reference year `t0`; defaults to the first system time |

The processor supplies the price dictionaries from the node attributes.
Assuming `lca_data_processor` has been prepared, configure discounting as follows:

```python
from optimex import optimizer
from optimex.converter import ModelInputManager

manager = ModelInputManager()
manager.parse_from_lca_processor(lca_data_processor)
model_inputs = manager.override(
    discount_rate=0.05,
    discount_reference_year=2020,
)
```

The discount factor for system year `t` is `1 / (1 + r) ** (t - t0)`.
At 5%, a cost of 100 in 2030 contributes approximately 61.39 to the discounted
total referenced to 2020. Discounting changes monetary valuation, not purchase
quantities or environmental impacts.

Real prices, expressed in a constant purchasing-power basis, require a real
discount rate. Nominal prices, which include inflation, require a nominal rate.
The price base year specifies purchasing power; the discount reference year
specifies when costs are valued. These are distinct choices.

## Selecting the Objective and Environmental Limits

Set `objective="cost"` to minimize discounted cost. The default,
`objective="environmental"`, minimizes the selected impact category.

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

`objective_category` is required by the current function signature. With
`objective="cost"`, it neither selects the cost objective's coefficients nor
imposes an environmental limit. Environmental constraints are configured
separately in the model inputs.

For example, the following inputs combine a cumulative impact budget and a
limit for one system year. Values must use the impact units of the selected
category; the example assumes a climate-change category in kg CO2-equivalent.

```python
model_inputs = manager.override(
    cumulative_category_impact_limits={"climate_change": 1_000_000.0},
    category_impact_limits={("climate_change", 2030): 50_000.0},
)
model = optimizer.create_model(
    inputs=model_inputs,
    name="cost_with_climate_limits",
    objective_category="climate_change",
    objective="cost",
)
```

Rebuild the model after changing its inputs. These limits restrict feasible
pathways; they do not add a carbon price to the objective.
See [Constraints](constraints.md) for further constraint options.

## Cost Expressions and Results

Purchase quantities are derived from existing installation and operation
variables. The economic extension adds expressions, not independent purchase
decision variables.

| Mathematical quantity | Model component |
|---|---|
| Installed process units of vintage `v` | `var_installation[p, v]` |
| Operation of vintage `v` at system time `t` | `var_operation[p, v, t]` |
| Installation-related purchase of product `i` in `t` | `background_purchase_cap[i, t]` |
| Operation-related purchase of product `i` in `t` | `background_purchase_op[i, t]` |
| Undiscounted installation-related cost in `t` | `cost_cap[t]` |
| Undiscounted operation-related cost in `t` | `cost_op[t]` |
| Discount factor at system time `t` | `discount_factor[t]` |
| Discounted sum over all system times | `total_cost` |

```text
cost_cap[t] = sum_i intermediate_costs_cap[i, t] * background_purchase_cap[i, t]
cost_op[t]  = sum_i intermediate_costs_op[i, t]  * background_purchase_op[i, t]
total_cost = sum_t discount_factor[t] * (cost_cap[t] + cost_op[t])
```

Read the expressions from the solved model:

```python
import pyomo.environ as pyo

annual_costs = {
    t: {
        "installation": pyo.value(solved_model.cost_cap[t]),
        "operation": pyo.value(solved_model.cost_op[t]),
    }
    for t in solved_model.SYSTEM_TIME
}
discounted_total = pyo.value(solved_model.total_cost)
```

Annual costs are undiscounted; their simple sum equals the discounted total
only when discounting has no effect. Cost expressions can also be evaluated for
an environmentally optimized solution if economic inputs were supplied.

### Numerical Scaling

Prices are supplied **unscaled**. Before pricing, purchase quantities are restored
to their original product units from the internally scaled LCA coefficients.
Cost results therefore already use the monetary unit of the input prices.
Do not apply environmental impact scaling factors to them.
Here, unscaled refers to numerical normalization, not the distinction between
real and nominal prices.

## Modeling Assumptions and Limits

- **Allocation:** Cost assignment follows the LCI exchange quantities and thus
  inherits the environmental allocation represented in the inventory.
  Explicitly modeled processes or wrappers can change how economic requirements
  are represented; the extension does not introduce a separate allocation setting.
- **Linear costs:** At fixed prices, purchase costs scale proportionally with
  continuous installation and operation levels. The formulation does not
  represent economies of scale, indivisible plant sizes, or minimum plant
  capacities. Scaling a reference plant therefore keeps its specific costs constant.
- **Time horizon:** The objective sums costs over `SYSTEM_TIME`. Purchases
  outside those system times do not enter the total. Choose the horizon and
  temporal distributions to cover the costs required for the analysis.
- **Cost coverage:** The objective covers the modeled purchases with supplied
  prices. It is not automatically a complete financial account of a project.

The [economic basic example](https://github.com/langeluca/optimex/blob/main/notebooks/basic_example_econ.ipynb)
demonstrates the workflow; the [optimizer API](../api/optimizer.md) documents the
model components.
