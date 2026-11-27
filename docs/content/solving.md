---
icon: lucide/rocket
tags:
  - optimization
  - solver
---

# Solving

With the [optimization inputs](lca_processing.md) ready, the
[objective](choosing_objective/index.md) chosen, and any [constraints](constraints.md)
added, build the Pyomo model and solve it.

---

## Building the Model

```python
from optimex import optimizer

model = optimizer.create_model(
    inputs=model_inputs,
    name="my_optimization",
    objective_category="climate_change",
)
```

| Parameter | Description |
|-----------|-------------|
| `inputs` | The `OptimizationModelInputs` object |
| `name` | Identifier for the model |
| `objective` | What to minimize: `"environmental"` (default) or `"cost"`, see [Choosing an Objective](choosing_objective/index.md) |
| `objective_category` | Impact category to minimize; required for `objective="environmental"`, ignored for `"cost"` |
| `debug_path` | Optional path of an LP file to write the model to |
| `backend` | `"pyomo"` (default) or `"linopy"`, see [Choosing a Backend](#choosing-a-backend) |
| `aggregate_vintages` | Share one operation variable between identical vintages, see [Large Models](#large-models) |

??? cost "Cost objective: building the model"
    To minimize costs instead, pass `objective="cost"` and leave out the category. This needs market prices in the inputs, see [Minimizing Costs](minimizing_costs.md).

    ```python
    model = optimizer.create_model(model_inputs, name="cost_minimum", objective="cost")
    ```

`create_model()` checks the objective first: an unknown objective, a category
that is not in the inputs, or the cost objective without prices raise a
`ValueError` before anything is built.

!!! note "Rebuild after changing inputs"
    `create_model()` copies the inputs into the model. Build a new model for
    every set of inputs instead of changing the inputs of an existing one.

### Choosing a Backend

The problem is formulated once and built by one of two backends. Both give the
same optimum, and `solve_model()` and the [PostProcessor](postprocessing.md)
work with either.

| Backend | Builds | Use it for |
|---------|--------|------------|
| `"pyomo"` (default) | A Pyomo `ConcreteModel` | Working on the model itself: fixing variables, adding constraints, IIS files, Pyomo's solver interfaces |
| `"linopy"` | A linopy model, from arrays | Large systems (thousands of processes), where it builds several times faster. Needs `pip install optimex[linopy]` |

```python
model = optimizer.create_model(
    inputs=model_inputs,
    name="my_optimization",
    objective_category="climate_change",
    backend="linopy",
)
```

### Inspecting the Formulation

Write the model as an LP file to inspect it:

```python
model = optimizer.create_model(
    inputs=model_inputs,
    name="debug_model",
    objective_category="climate_change",
    debug_path="model_debug.lp",
)
```

---

## Solving the Model

```python
solved_model, objective_value, results = optimizer.solve_model(
    model,
    solver_name="glpk",  # or "highs", "gurobi", "cplex"
    tee=False,           # True prints the solver output
)
```

`objective_value` is in real units: the unit of the impact category, or the
currency of the prices for the cost objective. Internally, the model is scaled
for numerical stability; `solve_model()` and the [PostProcessor](postprocessing.md)
undo the scaling.

### Solvers

| Solver | License | Notes |
|--------|---------|-------|
| `glpk` | Open source | Good for small to medium problems |
| `highs` | Open source | Fast, good for large problems; supports the fast path of [series](#solving-a-series-of-models) |
| `gurobi` | Commercial (free academic) | Very fast, robust; supports the fast path of [series](#solving-a-series-of-models) |
| `cplex` | Commercial | Enterprise-grade |

Solver-specific options, e.g. a time limit, are passed as `solver_options`:

```python
optimizer.solve_model(model, solver_name="gurobi", solver_options={"TimeLimit": 600})
```

See [Installation](installation.md#solver-requirements) for installing a solver.

### Large Models

For systems with thousands of processes, build the model with
[linopy](https://linopy.readthedocs.io) instead of Pyomo (`pip install
optimex[linopy]`). The formulation is the same; linopy receives every constraint
block as arrays, which builds and hands over large models several times faster:

```python
model = optimizer.create_model(
    inputs,
    name="large",
    objective_category="climate_change",
    aggregate_vintages=True,
    backend="linopy",
)
solved_model, objective_value, results = optimizer.solve_model(
    model, solver_name="gurobi"  # or "highs", "glpk", "cplex", ...
)
```

`solve_model()` stores the solution on the model under the same names as on a
Pyomo model (`var_installation`, `total_impact`, `specific_impact`, ...), as
plain numbers, so the [PostProcessor](postprocessing.md),
`optimizer.operation_value()` and `save_solved_model()` work unchanged.
`results.solver.termination_condition` holds Pyomo's termination condition.
What needs Pyomo are its own tools: fixing variables, adding constraints to the
model, IIS files, and Pyomo solver interfaces. The linopy model itself is
`model.model`. Gurobi and HiGHS are called in memory, other solvers through an
LP file.

With many processes and long horizons, most of the model is the per-vintage
operation: one variable per process, vintage and year. If the vintages of a
process all run identically (no vintage-dependent operation flows, operation
flows constant over the operation window), they can share one operation
variable per year:

```python
model = optimizer.create_model(
    inputs,
    name="large",
    objective_category="climate_change",
    aggregate_vintages=True,
)
```

The optimum is the same. Processes whose vintages differ keep per-vintage
dispatch. Aggregated processes have `var_operation_total[p, t]` instead of
`var_operation[p, v, t]`, and their capacity is carried forward year by year
rather than summed over all active vintages, which keeps the constraint matrix
sparse. The [PostProcessor](postprocessing.md) works as usual: it splits the
operation onto the vintages in proportion to their capacity. To read the
per-vintage operation yourself, use `optimizer.operation_value(model, p, v, t)`,
which works with and without aggregation. To fix per-vintage operation (e.g. to
evaluate a given design), build without aggregation.

Pyomo's default `gurobi` interface writes the model to an LP file and reads the
solution back from disk. With `gurobipy` installed, `gurobi_direct_v2` passes
the model to Gurobi in memory instead, which saves several seconds on large
models:

```python
optimizer.solve_model(model, solver_name="gurobi_direct_v2")
```

For large LPs, the barrier method without crossover is often faster than
Gurobi's default. The solution is optimal, but it may not be a vertex of the
feasible region:

```python
optimizer.solve_model(
    model, solver_name="gurobi", solver_options={"Method": 2, "Crossover": 0}
)
```

---

## Solving a Series of Models

Sweeps over a limit (e.g. a [Pareto frontier](trade_offs.md)), scenario
comparisons, and sensitivity studies solve the same product system many times
with slightly different data. Solve them together with
`optimizer.solve_series` instead of a loop over `create_model` and
`solve_model`:

```python
budgets = [4.0e5, 4.5e5, 5.0e5, 5.5e5]
sweep = optimizer.solve_series(
    {
        budget: model_inputs.model_copy(
            update={"cumulative_category_impact_limits": {"climate_change": budget}}
        )
        for budget in budgets
    },
    base=model_inputs,
    objective="cost",
    solver_name="highs",  # or "gurobi"
)
sweep.objective      # one value per variant, indexed by the dict keys
sweep.status         # "optimal", or why a variant has no solution
sweep.impacts        # cumulative impact per category
sweep.costs          # discounted total cost, with prices
sweep.installation   # installed units, indexed by (variant, year)
```

With HiGHS or Gurobi, `solve_series` builds the model of `base` once and
assembles every variant from it with NumPy, then loads it straight into the
solver. This applies to variants that differ in exchange amounts, background
inventories, characterization factors, prices, discounting, demand, impact
limits, or deployment and operation limits; other variants are built in full
automatically. With other solvers, each variant is built and solved on its own,
and a warning says so.

| Option | Effect |
|--------|--------|
| `base` | The inputs the variants were made from; defaults to the first variant |
| `outputs` | PostProcessor tables to keep: `"installation"` (default), `"operation"`, `"production"` |
| `extract` | A function `extract(model)` whose result is kept per variant |
| `keep_models` | Keep every solved model, e.g. for a full PostProcessor analysis |
| `n_jobs` | Solve variants in parallel processes (`-1`: all cores) |
| `persistent` | `"auto"` (default), `True` (fail instead of falling back), `False` (always build full models) |

!!! tip "Make variants with `model_copy`"
    `model_inputs.model_copy(update={...})` creates a variant that shares
    everything else with `model_inputs`, which is what lets `solve_series`
    skip the rebuild. `manager.override(...)` works too, but validates and
    copies all inputs for every variant.

The fast path needs full models only for what cannot be computed from the
variables: the `"production"` table, `extract`, and `keep_models` build each
variant's model in full (still solved in one solver session). Monte Carlo
analyses use the same machinery through
[`uncertainty.solve_samples`](uncertainty.md).

---

## Handling Infeasibility

If no pathway satisfies all constraints, `solve_model()` raises a `RuntimeError`
with the solver's termination condition. To find the conflicting constraints,
compute an Irreducible Infeasible Set (IIS):

```python
solved_model, objective_value, results = optimizer.solve_model(
    model,
    solver_name="gurobi",
    compute_iis=True,  # writes model_iis.ilp
)
```

`model_iis.ilp` lists a minimal set of constraints that cannot hold together.

!!! tip "Common causes of infeasibility"
    - Demand cannot be met with the available capacity limits
    - Impact limits are below the minimum achievable impact; check it by
      [minimizing that impact](minimizing_impacts.md) first
    - Conflicting minimum and maximum constraints
    - Insufficient existing capacity combined with restrictive deployment limits

!!! warning "Unbounded models"
    If an installation-driven flow has a net negative impact (for example, a
    credit at end-of-life), installing more units always lowers an impact
    objective, and the model is unbounded. Set finite
    `process_deployment_limits_max` for such processes. `solve_model()` points
    this out when it happens.

---

## Saving Solved Models

```python
# Save a solved model for later analysis
optimizer.save_solved_model(
    solved_model,
    "solved_model.pkl",
    objective_value=objective_value,
)

# Load it later
loaded_model = optimizer.load_solved_model("solved_model.pkl")
```

---

## Complete Example

```python
from datetime import datetime
import numpy as np
import bw2data as bd
from bw_temporalis import TemporalDistribution
from optimex import lca_processor, converter, optimizer

# LCA processing
product = bd.get_node(database="foreground", name="hydrogen")
years = range(2020, 2050)
td_demand = TemporalDistribution(
    date=np.array([datetime(y, 1, 1).isoformat() for y in years], dtype="datetime64[s]"),
    amount=np.array([0] * 5 + [1000] * 25, dtype=float),
)
config = lca_processor.LCAConfig(
    demand={product: td_demand},
    temporal={
        "start_date": datetime(2020, 1, 1),
        "temporal_resolution": "year",
        "time_horizon": 100,
    },
    characterization_methods=[
        # Static GWP100, so impacts and limits are in kg CO2-eq
        {"category_name": "climate_change", "brightway_method": ("IPCC 2021", "GWP 100a")},
    ],
)
lca_data = lca_processor.LCADataProcessor(config)
manager = converter.ModelInputManager()
model_inputs = manager.parse_from_lca_processor(lca_data)

# Constraints (optional): at most 500 t CO2-eq in total
model_inputs = manager.override(
    cumulative_category_impact_limits={"climate_change": 5.0e5}
)

# Build and solve
model = optimizer.create_model(
    model_inputs, name="hydrogen_transition", objective_category="climate_change"
)
solved, objective, results = optimizer.solve_model(model, solver_name="glpk")
print(f"Optimal impact: {objective:.2f}")
```

---

## Next Steps

- [Postprocessing](postprocessing.md): Analyze and visualize the results
- [Trade-offs](trade_offs.md): Sweep a limit to trace the trade-off between objectives
- [Uncertainty](uncertainty.md): Monte Carlo analysis of the uncertain data
