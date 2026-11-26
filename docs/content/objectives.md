---
icon: lucide/target
tags:
  - optimization
  - objectives
---

# Objectives

`optimex` decides which processes to install when, and how much of each vintage to
operate, so that the time-specific demand is met. What it minimizes while doing so
is the **objective**. You choose it with the `objective` argument of
`optimizer.create_model()`:

| Objective | Minimizes | Needs | Guide |
|---|---|---|---|
| `"environmental"` (default) | Cumulative impact in one impact category over the time horizon | `objective_category` | [Minimizing Impacts](objective_impact.md) |
| `"cost"` | Discounted cost of the background products the foreground buys directly | Market prices on the background nodes | [Minimizing Costs](objective_cost.md) |

```python
from optimex import optimizer

# Minimize climate change impacts
model = optimizer.create_model(
    model_inputs, name="impact_minimum", objective_category="climate_change"
)

# Minimize costs
model = optimizer.create_model(model_inputs, name="cost_minimum", objective="cost")

solved_model, objective_value, results = optimizer.solve_model(model, solver_name="glpk")
```

Everything else stays the same: the demand, the process data, and all
[constraints](constraints.md) apply to either objective, and the
[`PostProcessor`](postprocessing_guide.md) extracts the same results.
`solve_model()` returns the objective value in real units: the unit of the
impact category, or the currency of the prices.

---

## Choosing an Objective

**Minimize an impact** to find the pathway with the lowest environmental burden,
for example the lowest cumulative radiative forcing of a sector transition. This
is the classic Life Cycle Optimization question and needs no data beyond the LCA
model.

**Minimize costs** to find the cheapest pathway, typically under environmental
limits such as a carbon budget. This answers questions like "what does it cost to
stay within the budget, and which technologies does that take?". It needs market
prices for the directly purchased background products.

Only one quantity is minimized at a time. Other impact categories, and the costs,
are still computed for the solution, so you can always report them.

---

## Trade-offs Between Objectives

To see how impacts and costs trade off against each other, minimize one objective
while limiting the other. Impact limits are a standard constraint, so the common
pattern is to minimize costs under a tightening impact budget (the
epsilon-constraint method):

```python
import numpy as np
from optimex import optimizer
from optimex.postprocessing import PostProcessor

# `manager` is the ModelInputManager that produced `model_inputs`.

# The two ends of the trade-off
impact_minimum = optimizer.create_model(
    model_inputs, name="impact_minimum", objective_category="climate_change"
)
impact_minimum, lowest_impact, _ = optimizer.solve_model(impact_minimum, solver_name="glpk")

cost_minimum = optimizer.create_model(model_inputs, name="cost_minimum", objective="cost")
cost_minimum, lowest_cost, _ = optimizer.solve_model(cost_minimum, solver_name="glpk")
impact_of_cost_minimum = PostProcessor(cost_minimum).get_impacts()["climate_change"].sum().sum()

# Points in between: cheapest pathway for a range of carbon budgets
frontier = []
for budget in np.linspace(lowest_impact, impact_of_cost_minimum, 5):
    inputs = manager.override(cumulative_category_impact_limits={"climate_change": budget})
    model = optimizer.create_model(inputs, name=f"budget_{budget:.0f}", objective="cost")
    model, cost, _ = optimizer.solve_model(model, solver_name="glpk")
    frontier.append((budget, cost))
```

Each point is a cost-optimal pathway that respects its budget; together they form
the Pareto frontier between climate impact and cost. The
[basic cost example](examples/basic_example_econ.md) walks through one such
constrained case.

!!! note "Rebuild after changing inputs"
    `create_model()` copies the inputs into the Pyomo model. Build a new model for
    every set of inputs instead of changing the inputs of an existing one.

---

## Next Steps

- [Minimizing Impacts](objective_impact.md): impact categories, static and dynamic
  characterization, and impact results
- [Minimizing Costs](objective_cost.md): market prices, the cost boundary,
  discounting, and cost results
- [Constraints](constraints.md): deployment limits, impact budgets, and more
