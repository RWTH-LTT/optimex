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

??? cost "Cost objective: building the model"
    To minimize costs instead, pass `objective="cost"` and leave out the category. This needs market prices in the inputs, see [Minimizing Costs](minimizing_costs.md).

    ```python
    model = optimizer.create_model(model_inputs, name="cost_minimum", objective="cost")
    ```

`create_model()` checks the objective first: an unknown objective, a category
that is not in the inputs, or the cost objective without prices raise a
`ValueError` before anything is built.

!!! note "Rebuild after changing inputs"
    `create_model()` copies the inputs into the Pyomo model. Build a new model for
    every set of inputs instead of changing the inputs of an existing one.

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
| `highs` | Open source | Fast, good for large problems |
| `gurobi` | Commercial (free academic) | Very fast, robust |
| `cplex` | Commercial | Enterprise-grade |

Solver-specific options, e.g. a time limit, are passed as `solver_options`:

```python
optimizer.solve_model(model, solver_name="gurobi", solver_options={"TimeLimit": 600})
```

See [Installation](installation.md#solver-requirements) for installing a solver.

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
- [Trade-offs](trade_offs.md): Solve a series of models to compare objectives
