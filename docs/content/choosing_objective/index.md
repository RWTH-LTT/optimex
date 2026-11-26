---
icon: lucide/target
tags:
  - optimization
  - objectives
---

# Choosing an Objective

`optimex` decides which processes to install when, and how much of each vintage to
operate, so that the time-specific demand is met. What it minimizes while doing so
is the **objective**:

| Objective | Minimizes | Needs | Guide |
|---|---|---|---|
| Impacts (default) | Cumulative impact in one impact category over the time horizon | An LCIA method for the category | [Minimizing Impacts](../minimizing_impacts.md) |
| Costs | Discounted cost of the background products the foreground buys directly | Market prices on the background nodes | [Minimizing Costs](../minimizing_costs.md) |

Most studies minimize impacts, and the rest of the guide is written for that case.
If you are not interested in costs, read [Minimizing Impacts](../minimizing_impacts.md)
and skip [Minimizing Costs](../minimizing_costs.md), which collects everything
specific to the cost objective.

The objective is chosen when the model is built, from the same inputs:

```python
from optimex import optimizer

# Minimize climate change impacts
model = optimizer.create_model(
    model_inputs, name="impact_minimum", objective_category="climate_change"
)

# Minimize costs
model = optimizer.create_model(model_inputs, name="cost_minimum", objective="cost")
```

`objective` selects the kind of objective (`"environmental"`, the default, or
`"cost"`); `objective_category` selects which impact category, and is only used
for the environmental objective. Everything else is shared: the demand, the
process data, all [constraints](../constraints.md), and the
[results](../postprocessing.md). `solve_model()` returns the objective value in real
units: the unit of the impact category, or the currency of the prices.

---

## Which One to Use

**Minimize an impact** to find the pathway with the lowest environmental burden,
for example the lowest cumulative radiative forcing of a sector transition. This
is the classic Life Cycle Optimization question and needs no data beyond the LCA
model.

**Minimize costs** to find the cheapest pathway, typically within environmental
limits such as a carbon budget. This answers questions like "what does it cost to
stay within the budget, and which technologies does that take?". It needs market
prices for the directly purchased background products, see
[Setting Market Prices](../minimizing_costs.md#setting-market-prices).

Only one quantity is minimized at a time, but all impact categories, and the
costs if prices are available, are computed for every solution: you can minimize
an impact and still report its costs, see
[Costs of the Impact Minimum](../minimizing_impacts.md#costs-of-the-impact-minimum). To see how one
objective changes as the other is limited, trace the
[trade-offs](../trade_offs.md) between them.

!!! tip "Prepare for both"
    If you might need costs later, add the prices right away. Inputs with prices
    serve both objectives, and you can report the cost of an impact-minimal
    pathway or the impacts of a cost-minimal one.
