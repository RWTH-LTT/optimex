---
icon: lucide/scale
tags:
  - optimization
  - objectives
  - multi-objective
---

# Trade-offs

A model minimizes one [objective](choosing_objective/index.md) at a time. To see how
two objectives trade off against each other, for example climate change and water
use, minimize one while limiting the other, and repeat for a range of limits.
Impact limits are a standard [constraint](constraints.md), so this needs nothing
beyond the usual workflow. This is the epsilon-constraint method: each solution
is the best pathway for its limit, and together they trace the Pareto frontier.

The snippets continue from [LCA Processing](lca_processing.md): `model_inputs` are
the optimization inputs and `manager` the `ModelInputManager` that created them.

---

## Two Impact Categories

First solve the two extremes, then sweep the limit between them. For example,
climate change under a tightening water budget:

```python
import numpy as np
from optimex import optimizer
from optimex.postprocessing import PostProcessor

def total(model, category):
    return PostProcessor(model).get_impacts()[category].sum().sum()

# The two ends of the trade-off
water_minimum = optimizer.create_model(
    model_inputs, name="water_minimum", objective_category="water_use"
)
water_minimum, lowest_water, _ = optimizer.solve_model(water_minimum, solver_name="glpk")

climate_minimum = optimizer.create_model(
    model_inputs, name="climate_minimum", objective_category="climate_change"
)
climate_minimum, lowest_climate, _ = optimizer.solve_model(climate_minimum, solver_name="glpk")
water_of_climate_minimum = total(climate_minimum, "water_use")

# Points in between: lowest climate impact for a range of water budgets
frontier = []
for water_budget in np.linspace(lowest_water, water_of_climate_minimum, 5):
    inputs = manager.override(
        # A small tolerance keeps the tightest budget feasible despite rounding
        cumulative_category_impact_limits={"water_use": water_budget * (1 + 1e-6)}
    )
    model = optimizer.create_model(
        inputs, name=f"water_{water_budget:.0e}", objective_category="climate_change"
    )
    model, climate_impact, _ = optimizer.solve_model(model, solver_name="glpk")
    frontier.append((water_budget, climate_impact))
```

`frontier` holds the lowest climate impact for each water budget. The slope
between two points shows how much climate impact one unit of water saving costs.

!!! tip "Pathways, not only totals"
    Keep the solved models of the sweep to compare how the technology mix shifts
    as the budget tightens, e.g. with `PostProcessor.get_installation()`.

??? cost "Cost objective: costs against an impact"
    With market prices (see [Cost Modeling](cost_modeling.md)), the same pattern
    gives the cost of an environmental target: minimize costs under a tightening
    impact budget.

    ```python
    impact_minimum = optimizer.create_model(
        model_inputs, name="impact_minimum", objective_category="climate_change"
    )
    impact_minimum, lowest_impact, _ = optimizer.solve_model(impact_minimum, solver_name="glpk")

    cost_minimum = optimizer.create_model(model_inputs, name="cost_minimum", objective="cost")
    cost_minimum, lowest_cost, _ = optimizer.solve_model(cost_minimum, solver_name="glpk")
    impact_of_cost_minimum = total(cost_minimum, "climate_change")

    # Cheapest pathway for a range of carbon budgets
    cost_frontier = []
    for budget in np.linspace(lowest_impact, impact_of_cost_minimum, 5):
        inputs = manager.override(
            cumulative_category_impact_limits={"climate_change": budget * (1 + 1e-6)}
        )
        model = optimizer.create_model(inputs, name=f"budget_{budget:.0f}", objective="cost")
        model, cost, _ = optimizer.solve_model(model, solver_name="glpk")
        cost_frontier.append((budget, cost))
    ```

    The slope between two points is the abatement cost of the tighter budget. The
    [basic cost example](examples/basic_example_econ.md#9-pareto-frontier-between-cost-and-climate-impact)
    traces and plots such a frontier for a small system.

---

## Practical Notes

- **Endpoints first:** a budget below the minimum of that impact makes the model
  infeasible. Solving the extremes first gives the valid range, and a small
  tolerance on the tightest budget avoids infeasibility from rounding.
- **Cumulative or annual:** `cumulative_category_impact_limits` limits the total
  over the horizon; `category_impact_limits` limits single years. Both can be
  swept.
- **Rebuild each model:** every point needs its own `create_model()` call from
  its own inputs.
- **Ties at the extremes:** the minimum of one objective need not be unique, and
  the solver returns one of the tied pathways, which can be worse in the other
  objective. Strictly, each point is therefore only *weakly* Pareto-optimal. To
  get the best end point, minimize the other objective with the budget set to
  the minimum (plus the small tolerance), as the sweep above does at its tight
  end.
