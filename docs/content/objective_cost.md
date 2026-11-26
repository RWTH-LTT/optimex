---
icon: lucide/badge-euro
tags:
  - optimization
  - economics
  - costs
---

# Minimizing Costs

The cost objective minimizes the discounted cost of the background products that
the foreground system purchases directly. It uses the same installation and
operation decisions as the [environmental objective](objective_impact.md), so
impact limits still apply and can constrain the cost-optimal pathway. See
[Objectives](objectives.md) for how the two compare and how to trace the
trade-off between them.

The cost objective adds three steps to the workflow of
[Optimization Setup](optimization_setup.md):

1. Write market prices to the background nodes with `set_market_prices()`.
2. Read them by setting `LCAConfig.price_attribute`.
3. Build the model with `objective="cost"`.

The [basic cost example](examples/basic_example_econ.md) walks through all steps
with a synthetic one-product system that needs no ecoinvent data.

## Cost Boundary

The foreground comprises the explicitly modeled processes whose installation
and operation levels are decision variables. Background supply chains contain
no decision variables. Costs are assigned to the products purchased directly
across this boundary: the intermediate flows of the model. The boundary therefore
depends on which processes are modeled explicitly.

Prices are not propagated through the background supply chains. Their
inventories are used for the environmental assessment, while the market price of
a directly purchased product represents its full cost. Products exchanged
between foreground processes carry no price of their own: their production cost
enters through the background purchases of the processes producing them.

## Installation- and Operation-Driven Costs

A purchase is attributed to the decision driving it, following the same
installation/operation semantics as the environmental flows:

| Foreground exchange | Purchase scales with |
|---|---|
| `operation=True`, within the operation phase | Units of a vintage running in a given year (`var_operation`) |
| `operation=False`, or outside the operation phase | Installed units of a vintage (`var_installation`) |

Installation-driven purchases can include construction, replacements, and
end-of-life activities, depending on the temporal distributions. They need not
occur in the installation year, and the two groups are not equivalent to
financial CAPEX and OPEX.

A purchase is priced and discounted in the system year in which it occurs. For a
unit installed in `v`, an exchange at process time `tau` happens at `t = v + tau`:
a replacement ten years after a 2030 installation is priced and discounted in
2040, not in 2030.

A product has one price per year, whichever decision drives its purchase.

## Setting Market Prices

`set_market_prices()` writes a price to the background node of each
time-specific database. Each price refers to one unit of the node's reference
product. Use one currency and price basis throughout: 90 EUR/MWh becomes
0.09 EUR/kWh for a product measured in kilowatt hours. The helper performs no
unit conversion, currency conversion, or inflation adjustment.

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
        "year": year,
        "price": price,  # constant 2020 EUR/kWh
    }
    for year, price in [(2020, 0.09), (2030, 0.07)]
]

set_market_prices(
    price_data=price_data,
    background_databases=background_databases,
    product_col="product",
    unit_col="unit",
)
```

Rows identify nodes by name and location, optionally narrowed by reference
product and unit. Codes are not used, because premise assigns different codes to
the same activity in each scenario database. `price_data` can also be a pandas
DataFrame; `name_col`, `location_col`, `product_col`, `unit_col`, `year_col`, and
`price_col` map custom column names. `overwrite=False` keeps existing prices, and
`strict=False` turns unresolved rows into warnings instead of errors.

Prospective databases, for example from `premise`, do not supply market prices
themselves: set them for every database.

## Reading Prices

Prices are only read when `LCAConfig.price_attribute` names the node attribute
holding them, so impact-only workflows are unaffected:

```python
from optimex import lca_processor

config = lca_processor.LCAConfig(
    demand=...,
    temporal=...,
    characterization_methods=...,
    price_attribute="market_price",  # attribute written by set_market_prices
)
processor = lca_processor.LCADataProcessor(config)
processor.intermediate_flow_prices  # {(intermediate flow, year): price}
```

The processor resolves each directly purchased product in every background
database the same way it does for the inventories, and interpolates the prices to
system time with the same weights: linearly between database years, with the
nearest database used outside that range.

Missing prices are handled conservatively:

- A product without a price in **any** database is left out and treated as
  free. All such products are listed in a single warning: check it, since a free
  purchase is rarely what you want.
- A product priced in **some** databases but not in others raises a
  `ValueError`, because interpolating with the missing price would silently pull
  the price towards zero.

## Model Inputs and Discounting

`OptimizationModelInputs` carries the economic data:

| Field | Meaning |
|---|---|
| `intermediate_flow_prices[(i, t)]` | Price per unit of intermediate flow `i` in system year `t`; missing entries are free |
| `discount_rate` | Nonnegative annual rate `r`; defaults to 0 |
| `discount_reference_year` | Reference year `t0`; defaults to the first system year |

`ModelInputManager.parse_from_lca_processor()` fills the prices. Discounting, or
prices set by hand, are passed via `override()`:

```python
from optimex.converter import ModelInputManager

manager = ModelInputManager()
manager.parse_from_lca_processor(processor)
model_inputs = manager.override(discount_rate=0.05, discount_reference_year=2020)
```

The discount factor for year `t` is `(1 + r) ** -(t - t0)`. At 5%, a cost of 100
in 2030 contributes about 61.39 to a total referenced to 2020. Discounting changes
how costs are valued, not purchase quantities or impacts.

Real prices, expressed in a constant purchasing-power basis, require a real
discount rate; nominal prices, which include inflation, a nominal one. The price
base year fixes purchasing power, the discount reference year when costs are
valued: these are separate choices.

## Selecting the Objective

```python
from optimex import optimizer

model = optimizer.create_model(model_inputs, name="cost_model", objective="cost")
model, total_cost, results = optimizer.solve_model(model, solver_name="glpk")
```

`objective="environmental"` (the default) minimizes `objective_category`, which
the cost objective does not need. `solve_model()` returns the discounted total
cost in the currency of the input prices.

Impact limits are set in the model inputs, as for the environmental objective.
They restrict the feasible pathways; they do not add a carbon price to the
objective. Values use the units of the impact category:

```python
model_inputs = manager.override(
    cumulative_category_impact_limits={"climate_change": 1_000_000.0},
    category_impact_limits={("climate_change", 2030): 50_000.0},
)
model = optimizer.create_model(model_inputs, name="cost_with_limits", objective="cost")
```

Rebuild the model after changing its inputs. See [Constraints](constraints.md)
for further constraint options.

## Results

`PostProcessor.get_costs()` returns the costs in the currency of the input
prices, with one column per driver and process:

```python
from optimex.postprocessing import PostProcessor

pp = PostProcessor(model)
costs = pp.get_costs()                     # undiscounted, per year
costs["operation"]                         # operation-driven costs per process
pp.get_costs(discounted=True).sum().sum()  # equals the discounted total cost
```

Costs are built for every model with prices, so the cost of an impact-minimal
pathway can be read the same way.

The model components behind it are, like the impact expressions, in scaled units
(multiply by `model.scales["foreground"] * model.scales["cost"]`):

| Component | Meaning |
|---|---|
| `intermediate_flow_prices[i, t]` | Scaled price of intermediate flow `i` in year `t` |
| `discount_factor[t]` | Discount factor of year `t` |
| `installation_cost[p, t]` | Undiscounted cost of the installation-driven purchases of process `p` |
| `operation_cost[p, t]` | Undiscounted cost of the operation-driven purchases of process `p` |
| `total_cost` | Discounted sum over all processes and years |

## Modeling Assumptions and Limits

- **Allocation:** Costs follow the LCI exchange amounts and therefore inherit the
  allocation of the inventory. The extension adds no separate allocation setting.
- **Linear costs:** At fixed prices, costs scale proportionally with continuous
  installation and operation levels. Economies of scale, indivisible plant sizes,
  and minimum plant capacities are not represented.
- **Time horizon:** The objective sums costs over `SYSTEM_TIME`. Purchases
  outside it do not enter the total; choose the horizon and temporal
  distributions to cover the costs that matter.
- **Coverage:** The objective covers the priced purchases of the modeled
  processes. It is not a complete financial account: fixed operating costs, for
  example, are only included if modeled as purchases.
