---
icon: lucide/sliders-horizontal
tags:
  - optimization
  - setup
---

# Optimization Setup

Once your Brightway databases are configured, you need to set up the optimization problem: define the time-specific demand $\mathbf{f}_t$, configure characterization methods (static or [dynamic](theory.md#dynamic-characterization)), process the LCA data into time-explicit tensors, and create and solve the optimization model.

---

## Overview

The optimization setup follows this pipeline:

```
Demand + Config → LCADataProcessor → ModelInputManager → create_model → solve_model
```

---

## Defining Demand

Demand specifies **how much** of each product is needed **when**. Use `TemporalDistribution` from `bw_temporalis`:

```python
from datetime import datetime
import numpy as np
from bw_temporalis import TemporalDistribution
import bw2data as bd

# Get the product node from your foreground database
product = bd.get_node(database="foreground", name="hydrogen")

# Define demand over time
years = range(2020, 2040)
demand_values = [0, 0, 100, 100, 150, 150, 200, 200, 250, 250,
                 300, 300, 350, 350, 400, 400, 450, 450, 500, 500]

td_demand = TemporalDistribution(
    date=np.array([datetime(y, 1, 1).isoformat() for y in years], dtype='datetime64[s]'),
    amount=np.array(demand_values, dtype=float),
)

demand = {product: td_demand}
```

### Demand Patterns

**Constant demand:**
```python
amount=np.array([100] * 20)  # 100 units every year
```

**Growing demand:**
```python
amount=np.array([100 + i*10 for i in range(20)])  # Linear growth
```

**Step change:**
```python
amount=np.array([100]*10 + [200]*10)  # Double after 10 years
```

**Delayed start:**
```python
amount=np.array([0, 0, 0, 100, 100, 100, ...])  # Start in year 4
```

---

## LCA Configuration

The `LCAConfig` object defines temporal settings and characterization methods:

```python
from optimex import lca_processor

config = lca_processor.LCAConfig(
    demand=demand,
    temporal={
        "start_date": datetime(2020, 1, 1),
        "temporal_resolution": "year",
        "time_horizon": 100,
    },
    characterization_methods=[
        {
            "category_name": "climate_change",
            "brightway_method": ("IPCC 2021", "climate change", "GWP 100a"),
        },
    ],
)
```

### Temporal Settings

| Parameter | Description | Example |
|-----------|-------------|---------|
| `start_date` | Beginning of the optimization horizon | `datetime(2020, 1, 1)` |
| `temporal_resolution` | Time step granularity | `"year"`, `"month"`, `"day"` |
| `time_horizon` | Years for impact accumulation | `100` (for GWP100) |

### Characterization Methods

Each method requires:

| Field | Required | Description |
|-------|----------|-------------|
| `category_name` | Yes | Your name for this impact category |
| `brightway_method` | Yes | Tuple identifying the Brightway method |
| `metric` | No | Dynamic characterization: `"CRF"` or `"GWP"` |

**Static characterization** (default):
```python
{
    "category_name": "land_use",
    "brightway_method": ("ReCiPe", "land use"),
}
```

**Dynamic characterization** (for climate change):
```python
{
    "category_name": "climate_change",
    "brightway_method": ("IPCC 2021", "GWP 100a"),
    "metric": "CRF",  # Cumulative Radiative Forcing
}
```

!!! info "Dynamic Metrics"
    - **CRF** (Cumulative Radiative Forcing): Integrates radiative forcing over time
    - **GWP** (Global Warming Potential): Time-dependent GWP factors

    Dynamic metrics account for when emissions occur, not just how much.

### Multiple Impact Categories

```python
characterization_methods=[
    {
        "category_name": "climate_change",
        "brightway_method": ("IPCC 2021", "GWP 100a"),
        "metric": "CRF",
    },
    {
        "category_name": "water_use",
        "brightway_method": ("ReCiPe", "water consumption"),
    },
    {
        "category_name": "land_use",
        "brightway_method": ("ReCiPe", "land use"),
    },
]
```

### Background Inventory

`background_inventory` controls how the inventories of the background databases are
calculated:

```python
config = lca_processor.LCAConfig(
    ...,
    background_inventory={
        "calculation_method": "parallel",  # default; one process per background database
        "retain_flows": [iridium["code"]],  # keep flows that carry no impact
    },
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `calculation_method` | `"parallel"` | `"parallel"` calculates each background database in its own process, `"sequential"` one after another |
| `n_jobs` | `None` | Worker processes for `"parallel"`; defaults to one per background database, capped by the CPU count |
| `restrict_to_characterized_flows` | `True` | Drop elementary flows that have no characterization factor in any configured category |
| `retain_flows` | `[]` | Elementary flow codes to keep even when they carry no characterization factor |
| `cutoff` | `None` | Optional number of largest elementary flows to keep per intermediate flow. `None` keeps all |
| `use_disk_cache` | `True` | Keep calculated inventories in a cache inside the Brightway project, so a new session does not rebuild them |
| `disk_cache_dir` | `None` | Where that cache lives; defaults to `optimex-inventory-cache` in the current project |
| `path_to_save` / `path_to_load` | `None` | Save or load the whole inventory tensor as a pickle |

!!! warning "Uncharacterized flows are dropped"
    By default, an elementary flow that has **no characterization factor in any of your
    configured categories** is removed from the model. Such a flow cannot affect any
    impact, and keeping it would only inflate the model (the inventory is expressed per
    process, elementary flow and year).

    This matters when you constrain a flow directly, e.g. a resource limit through
    [`cumulative_flow_limits_max`](constraints.md). Iridium has no climate change or
    water use factor, so it is dropped unless you ask for it:

    ```python
    background_inventory={"retain_flows": [iridium["code"]]}
    ```

    Set `restrict_to_characterized_flows=False` to keep every flow instead. The
    calculation log reports how many flows were dropped.

!!! note "Parallel calculation from a script"
    Worker processes are spawned, so a **script** using the default
    `calculation_method="parallel"` must guard its entry point:

    ```python
    if __name__ == "__main__":
        main()
    ```

    Notebooks need no guard. Use `"sequential"` if a guard is not an option.

---

## Processing LCA Data

The `LCADataProcessor` extracts all necessary data from your Brightway databases:

```python
lca_data = lca_processor.LCADataProcessor(config)
```

This step:

- Identifies foreground processes and their temporal distributions
- Calculates background inventories for each database
- Constructs characterization factors (static or dynamic)
- Builds interpolation weights between background databases

!!! note "Computation Time"
    Most of this step is building and factorizing the technosphere matrix of each
    background database. Results are cached per (project, database, activity), in
    memory for the running session and on disk for the next one, so only the first
    run pays for it. On the methanol & iron case study that is ~37 s for the first
    run and ~0.3 s afterwards.

    Cache entries carry the database's `modified` token, so editing a background
    database invalidates them automatically. To clear them by hand:

    ```python
    from optimex import lca_processor

    lca_processor.clear_lca_caches()                     # this session
    lca_processor.clear_lca_caches(include_disk=True)    # and the files
    ```

---

## Converting to Optimization Inputs

The `ModelInputManager` converts LCA data to optimization-ready format:

```python
from optimex import converter

manager = converter.ModelInputManager()
model_inputs = manager.parse_from_lca_processor(lca_data)
```

### Saving and Loading Model Inputs

Avoid reprocessing by saving:

```python
# Save to file
manager.save_inputs("model_inputs.json")  # Human-readable
manager.save_inputs("model_inputs.pkl")   # Faster, binary

# Load later
manager = converter.ModelInputManager()
manager.load_inputs("model_inputs.json")
model_inputs = manager.model_inputs
```

### Adding Constraints

Before creating the model, add any constraints:

```python
# See Constraints guide for all options
model_inputs.cumulative_category_impact_limits = {"climate_change": 1000000}
model_inputs.process_deployment_limits_max = {("coal_plant", 2025): 0}
```

---

## Creating the Optimization Model

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
| `objective_category` | Impact category to minimize |

### Debug Output

To inspect the model formulation:

```python
model = optimizer.create_model(
    inputs=model_inputs,
    name="debug_model",
    objective_category="climate_change",
    debug_path="model_debug.lp",  # Write LP file
)
```

---

## Solving the Model

```python
solved_model, objective_value, results = optimizer.solve_model(
    model,
    solver_name="glpk",  # or "gurobi", "cplex", "highs"
    tee=False,           # Set True for solver output
)
```

### Solver Options

| Solver | License | Notes |
|--------|---------|-------|
| `glpk` | Open source | Good for small-medium problems |
| `highs` | Open source | Fast, good for large problems |
| `gurobi` | Commercial (free academic) | Very fast, robust |
| `cplex` | Commercial | Enterprise-grade |

### Handling Infeasibility

If the problem is infeasible:

```python
solved_model, objective_value, results = optimizer.solve_model(
    model,
    solver_name="gurobi",
    compute_iis=True,  # Generate Irreducible Infeasible Set
)
# Check model_iis.ilp for conflicting constraints
```

---

## Saving Solved Models

```python
# Save solved model for later analysis
optimizer.save_solved_model(
    solved_model,
    "solved_model.pkl",
    objective_value=objective_value,
)

# Load later
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

# 1. Define demand
product = bd.get_node(database="foreground", name="hydrogen")
years = range(2020, 2050)
td_demand = TemporalDistribution(
    date=np.array([datetime(y, 1, 1).isoformat() for y in years], dtype='datetime64[s]'),
    amount=np.array([0]*5 + [1000]*25, dtype=float),
)

# 2. Configure LCA
config = lca_processor.LCAConfig(
    demand={product: td_demand},
    temporal={
        "start_date": datetime(2020, 1, 1),
        "temporal_resolution": "year",
        "time_horizon": 100,
    },
    characterization_methods=[
        {
            "category_name": "climate_change",
            "brightway_method": ("IPCC 2021", "GWP 100a"),
            "metric": "CRF",
        },
    ],
)

# 3. Process LCA data
lca_data = lca_processor.LCADataProcessor(config)

# 4. Convert to optimization inputs
manager = converter.ModelInputManager()
model_inputs = manager.parse_from_lca_processor(lca_data)

# 5. Add constraints (optional)
model_inputs.cumulative_category_impact_limits = {"climate_change": 500000}

# 6. Create and solve
model = optimizer.create_model(
    model_inputs,
    name="hydrogen_transition",
    objective_category="climate_change",
)
solved, objective, results = optimizer.solve_model(model, solver_name="glpk")

print(f"Optimal impact: {objective:.2f}")
```

---

## Next Steps

- [Constraints](constraints.md): Add deployment limits, impact budgets, and more
- [Postprocessing](postprocessing_guide.md): Analyze and visualize results
