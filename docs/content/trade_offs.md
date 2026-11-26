---
icon: lucide/scale
tags:
  - optimization
  - objectives
  - multi-objective
---

# Trade-offs

A model minimizes one [objective](choosing_objective/index.md) at a time. To see how
two objectives trade off against each other, for example costs and climate
impact, minimize one while limiting the other, and repeat for a range of limits.
Impact limits are a standard [constraint](constraints.md), so the common pattern
is to minimize costs under a tightening impact budget. This is the
epsilon-constraint method; each solution is Pareto-optimal.

---

## Costs Against an Impact

First solve the two extremes, then sweep the budget between them:

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

`frontier` holds the minimal cost for each carbon budget: the Pareto frontier
between climate impact and cost. The slope between two points is the abatement
cost of the tighter budget. The [basic cost example](examples/basic_example_econ.md#9-pareto-frontier-between-cost-and-climate-impact)
traces and plots such a frontier for a small system.

!!! tip "Pathways, not only totals"
    Keep the solved models of the sweep to compare how the technology mix shifts
    as the budget tightens, e.g. with `PostProcessor.get_installation()`.

---

## Two Impact Categories

The same pattern works between impact categories: minimize one category while
limiting another, for example climate change under a water budget:

```python
for water_budget in [3.0e6, 2.5e6, 2.0e6]:
    inputs = manager.override(cumulative_category_impact_limits={"water_use": water_budget})
    model = optimizer.create_model(
        inputs, name=f"water_{water_budget:.0e}", objective_category="climate_change"
    )
    model, climate_impact, _ = optimizer.solve_model(model, solver_name="glpk")
```

---

## Practical Notes

- **Endpoints first:** a budget below the minimum of that impact makes the model
  infeasible. Solving the extremes first gives the valid range.
- **Cumulative or annual:** `cumulative_category_impact_limits` limits the total
  over the horizon; `category_impact_limits` limits single years. Both can be
  swept.
- **Rebuild each model:** every point needs its own `create_model()` call from
  its own inputs.
- **Ties at the extremes:** the impact minimum need not be unique. If the
  cheapest among the impact-minimal pathways matters, minimize costs with the
  budget set to the minimal impact (plus a small tolerance) instead of reading
  the costs of the impact minimum.
