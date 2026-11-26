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
operation decisions as the [environmental objective](minimizing_impacts.md), so
impact limits still apply and can constrain the cost-optimal pathway.

Compared with minimizing impacts, it needs three additions along the
[workflow](workflow.md):

1. **Data:** write market prices to the background nodes, see
   [Market Prices](background_data.md#market-prices).
2. **LCA processing:** read them by setting `LCAConfig.price_attribute`, see
   [LCA Processing](lca_processing.md#market-prices).
3. **Optimization:** build the model with `objective="cost"`, optionally with a
   discount rate (below).

This page explains what the cost objective computes and how to interpret it.

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

## Prices

Prices are data: they are written to the background nodes while
[setting up the background data](background_data.md#market-prices), one price per
node and background database. Each price refers to one unit of the node's
reference product, in one currency and price basis for all prices.
[LCA Processing](lca_processing.md#market-prices) reads them when
`LCAConfig.price_attribute` is set, interpolates them to every system year like
the inventories, and reports products without a price.

A product without a price is treated as free, so check the warning listing them.

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
for further constraint options, and [Trade-offs](trade_offs.md) for sweeping a
limit to trace how costs rise as the budget tightens.

## Results

[`PostProcessor.get_costs()`](postprocessing.md#costs) returns the costs in the currency of the input
prices, with one column per driver and process:

```python
from optimex.postprocessing import PostProcessor

pp = PostProcessor(model)
costs = pp.get_costs()                     # undiscounted, per year
costs["operation"]                         # operation-driven costs per process
pp.get_costs(discounted=True).sum().sum()  # equals the discounted total cost
```

Costs are built for every model with prices, so the cost of an impact-minimal
pathway can be read the same way, see
[Costs of the Impact Minimum](minimizing_impacts.md#costs-of-the-impact-minimum).

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
