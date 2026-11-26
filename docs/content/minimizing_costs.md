---
icon: lucide/badge-euro
tags:
  - optimization
  - objectives
  - costs
---

# Minimizing Costs

The cost objective minimizes the discounted cost of the background products the
foreground buys directly. It needs market prices, described in
[Cost Modeling](cost_modeling.md); everything else is the same as for
[minimizing impacts](minimizing_impacts.md).

```python
from optimex import optimizer

model = optimizer.create_model(model_inputs, name="cost_minimum", objective="cost")
solved_model, total_cost, results = optimizer.solve_model(model, solver_name="glpk")
```

`objective_category` is not needed. If the inputs carry no prices,
`create_model()` raises a `ValueError`. As on the other pages, `model_inputs` and
`manager` come from [LCA Processing](lca_processing.md).

---

## What Is Minimized

The objective is the cost of all priced background purchases, summed over the
system years of the time horizon and discounted to a reference year:

- **Purchases** are the amounts of background products the foreground buys,
  following the temporal distributions of its exchanges, the same amounts that
  drive the background impacts.
- **Prices** are taken for the year of purchase, interpolated between the
  background databases like the inventories.
- **Installation-driven** purchases (construction, end-of-life) scale with the
  installed units, **operation-driven** purchases with the units running, see
  [Cost Modeling](cost_modeling.md#installation-and-operation-driven-costs).

Upstream costs are not added on top: the market price of a purchased product
stands for its whole supply chain, see the
[cost boundary](cost_modeling.md#cost-boundary).

---

## Discounting

Discounting decides how much a future cost weighs against a present one, much like
dynamic characterization does for emissions. It is set in the model inputs:

| Field | Meaning |
|---|---|
| `discount_rate` | Nonnegative annual rate `r`; defaults to 0 (no discounting) |
| `discount_reference_year` | Reference year `t0`; defaults to the first system year |

```python
model_inputs = manager.override(discount_rate=0.05, discount_reference_year=2020)
```

The discount factor for year `t` is `(1 + r) ** -(t - t0)`. At 5%, a cost of 100
in 2030 contributes about 61.39 to a total referenced to 2020. Discounting changes
how costs are valued, not purchase quantities or impacts.

Real prices, expressed in a constant purchasing-power basis, require a real
discount rate; nominal prices, which include inflation, a nominal one. The price
base year fixes purchasing power, the discount reference year when costs are
valued: these are separate choices.

---

## Costs Under Impact Limits

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

This gives the cheapest pathway within the budget. Sweeping the limit traces how
costs rise as the budget tightens, see [Trade-offs](trade_offs.md).

---

## Results

[`PostProcessor.get_costs()`](postprocessing.md#costs) returns the costs in the currency of the input
prices, with one column per driver and process:

```python
from optimex.postprocessing import PostProcessor

pp = PostProcessor(solved_model)
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

---

## Pitfalls

- **Free purchases**: a background product without a price counts as free, which
  makes processes that buy it look cheaper than they are. Check the warning that
  lists unpriced products, see [Reading Prices](cost_modeling.md#reading-prices).
- **End of the horizon**: only purchases within the system years count, so late
  installations look cheap because their operating costs after the horizon are
  missing. Extend the horizon with `manager.extend_demand(years)` or judge late
  investments with that in mind.
- **Ties**: several pathways can reach the same minimal cost. The solver returns
  one of them, which is not necessarily the one with the lowest impact; limit the
  impact to break the tie.
- **Units**: costs are in the currency and price basis of the input prices; no
  conversion is applied.
