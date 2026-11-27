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

First solve the two extremes, then sweep the limit between them with
`optimizer.solve_series`. For example, climate change under a tightening water
budget:

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
water_minimum, lowest_water, _ = optimizer.solve_model(water_minimum, solver_name="highs")

climate_minimum = optimizer.create_model(
    model_inputs, name="climate_minimum", objective_category="climate_change"
)
climate_minimum, lowest_climate, _ = optimizer.solve_model(climate_minimum, solver_name="highs")
water_of_climate_minimum = total(climate_minimum, "water_use")

# Points in between: lowest climate impact for a range of water budgets.
# A small tolerance keeps the tightest budget feasible despite rounding.
budgets = np.linspace(lowest_water, water_of_climate_minimum, 10) * (1 + 1e-6)
frontier = optimizer.solve_series(
    {
        budget: model_inputs.model_copy(
            update={"cumulative_category_impact_limits": {"water_use": budget}}
        )
        for budget in budgets
    },
    base=model_inputs,
    objective_category="climate_change",
    solver_name="highs",
)
frontier.objective          # lowest climate impact per water budget
frontier.impacts            # all categories per budget
frontier.installation       # installed units per budget and year
```

The results are indexed by the budget. The slope between two points shows how
much climate impact one unit of water saving costs.

!!! tip "Why `solve_series`"
    The points of a sweep differ only in one limit. With HiGHS or Gurobi,
    `solve_series` builds the model once and assembles each point from it, which
    is much faster than building and solving every point on its own. It also
    collects objectives, impacts, and installations in one table, and can solve
    points in parallel (`n_jobs`). Make the points with
    `model_inputs.model_copy(update=...)`, as above: what they share with
    `model_inputs` then does not have to be rebuilt. See
    [Sweeps and Scenarios](series.md).

!!! tip "Pathways, not only totals"
    `frontier.installation` shows how the technology mix shifts as the budget
    tightens. Pass `outputs=("installation", "operation", "production")` for
    more tables, or `keep_models=True` to keep every solved model.

??? cost "Cost objective: costs against an impact"
    With market prices (see [Cost Modeling](cost_modeling.md)), the same pattern
    gives the cost of an environmental target: minimize costs under a tightening
    impact budget.

    ```python
    impact_minimum = optimizer.create_model(
        model_inputs, name="impact_minimum", objective_category="climate_change"
    )
    impact_minimum, lowest_impact, _ = optimizer.solve_model(impact_minimum, solver_name="highs")

    cost_minimum = optimizer.create_model(model_inputs, name="cost_minimum", objective="cost")
    cost_minimum, lowest_cost, _ = optimizer.solve_model(cost_minimum, solver_name="highs")
    impact_of_cost_minimum = total(cost_minimum, "climate_change")

    # Cheapest pathway for a range of carbon budgets
    budgets = np.linspace(lowest_impact, impact_of_cost_minimum, 10) * (1 + 1e-6)
    cost_frontier = optimizer.solve_series(
        {
            budget: model_inputs.model_copy(
                update={"cumulative_category_impact_limits": {"climate_change": budget}}
            )
            for budget in budgets
        },
        base=model_inputs,
        objective="cost",
        solver_name="highs",
    )
    cost_frontier.objective  # lowest discounted cost per budget
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
- **Infeasible points:** a point without a feasible pathway does not stop the
  sweep; its `status` names the solver's termination condition, and its
  objective is NaN.
- **Other sweeps:** the same works for any limit, price, or demand assumption,
  and for comparing scenarios, see
  [Sweeps and Scenarios](series.md).
- **Ties at the extremes:** the minimum of one objective need not be unique, and
  the solver returns one of the tied pathways, which can be worse in the other
  objective. Strictly, each point is therefore only *weakly* Pareto-optimal. To
  get the best end point, minimize the other objective with the budget set to
  the minimum (plus the small tolerance), as the sweep above does at its tight
  end.
