---
icon: lucide/leaf
tags:
  - optimization
  - objectives
  - impact assessment
---

# Minimizing Impacts

The environmental objective, the default of `create_model()`, minimizes the
cumulative impact in one impact category. It needs nothing beyond the LCA model
described in [Optimization Setup](optimization_setup.md).

```python
from optimex import optimizer

model = optimizer.create_model(
    model_inputs,
    name="impact_minimum",
    objective_category="climate_change",  # a category_name from LCAConfig
)
solved_model, total_impact, results = optimizer.solve_model(model, solver_name="glpk")
```

`objective="environmental"` is the default and can be left out.
`objective_category` must be one of the `category_name`s configured in
`LCAConfig.characterization_methods`; any other value raises a `ValueError`.

---

## What Is Minimized

The objective is the impact of all processes, summed over the system years of the
time horizon:

- **Foreground emissions** of each process at the time they occur, following the
  temporal distributions of its exchanges.
- **Background emissions** of the products the foreground buys, taken from the
  background database matching the year of purchase (interpolated between the
  database years).
- **Installation-driven** flows (construction, end-of-life) scale with the
  installed units, **operation-driven** flows with the units running, so idle
  capacity still carries its construction burden.

Each elementary flow is weighted with the characterization factor of the category
in the year it occurs, so the result depends on the characterization method.

---

## Static or Dynamic Characterization

The characterization method of the objective category decides how the timing of
emissions counts:

| `metric` in the method config | Characterization | Typical use |
|---|---|---|
| none | Static factors, equal in every year (e.g. GWP100) | Categories where timing does not matter, e.g. land or water use |
| `"GWP"` | Dynamic Global Warming Potential, in CO~2~-equivalents, integrated over the time horizon | Climate change, comparable to GWP-based budgets |
| `"CRF"` | Cumulative Radiative Forcing, integrated over the time horizon | Climate change, physically accumulated warming effect |

With `TemporalConfig.fixed_time_horizon=True` (the default), the time horizon ends
at a fixed date counted from the demand, so an emission late in the transition is
integrated over a shorter period than an early one and weighs less. The optimizer
thus sees an incentive to defer emissions or to remove CO~2~ early. See
[Characterization Methods](optimization_setup.md#characterization-methods) for the
configuration and `time_horizon`.

---

## Several Impact Categories

The objective minimizes one category. Further categories are configured the same
way and can be used as **limits**, so the optimum for one category respects
budgets in the others:

```python
model_inputs = manager.override(
    cumulative_category_impact_limits={"water_use": 2.0e6},
    category_impact_limits={("land_use", 2040): 5.0e4},
)
model = optimizer.create_model(
    model_inputs, name="climate_under_water_budget", objective_category="climate_change"
)
```

To explore the trade-off between two categories, solve the model once per category
for the two extremes and then sweep a limit on one while minimizing the other, as
shown for costs in [Objectives](objectives.md#trade-offs-between-objectives).

---

## Results

`solve_model()` returns the minimal cumulative impact in the unit of the category's
characterization method. The `PostProcessor` breaks it down by year and process:

```python
from optimex.postprocessing import PostProcessor

pp = PostProcessor(solved_model)
impacts = pp.get_impacts()            # columns: (category, process)
impacts["climate_change"].sum(axis=1) # climate change impact per year
pp.plot_impacts()
```

All configured categories are evaluated for the solution, not only the objective
category. If market prices are available (see [Minimizing Costs](objective_cost.md)),
`pp.get_costs()` also reports what the impact-minimal pathway costs.

---

## Pitfalls

- **Unbounded models**: if an installation-driven flow has a net negative impact
  (for example, a credit at end-of-life), installing more units always lowers the
  objective. Set finite `process_deployment_limits_max` for such processes or
  check the sign of the credit. `solve_model()` points this out when it happens.
- **Ties**: several pathways can reach the same minimal impact, for example when a
  process has no impact in the objective category. The solver returns one of them,
  which is not necessarily the cheapest or otherwise preferable one.
- **Units**: impact limits are given in the unit of the category, the same unit in
  which the objective is reported.
