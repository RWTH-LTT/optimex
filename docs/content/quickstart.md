---
icon: lucide/zap
tags:
  - getting started
  - tutorial
---

# Quick Start

A condensed reference for using `optimex`. For the underlying framework, see the [Theory](theory_how_it_works.md) page. For detailed explanations, see the [User Guide](workflow.md), [Examples](examples/index.md), or the [API Reference](../api/index.md).

!!! tip "New to optimex? Start with the Basic Example"
    The snippets on this page are a reference and assume an existing Brightway project. For a complete example that runs from scratch, without ecoinvent, work through the [Basic Example](examples/basic_example.md) (notebook: `notebooks/basic_example.ipynb`).

---

## Install `optimex` and a Solver

```bash
pip install optimex
conda install -c conda-forge glpk  # or: brew install glpk (macOS)
```

`optimex` needs a linear programming solver; the examples use the open-source GLPK. See the [Installation guide](installation.md) for other installation methods, other solvers, and platform-specific notes.

---

## Workflow at a Glance

This skeleton shows the steps of a study. It assumes a Brightway project with background and foreground databases as described below.

```python
from datetime import datetime
import numpy as np
import bw2data as bd
from bw_temporalis import TemporalDistribution
from optimex import lca_processor, converter, optimizer, postprocessing

# 1. Open the Brightway project
bd.projects.set_current("my_project")

# 2. Define the demand for a foreground product over time
years = range(2020, 2030)
td_demand = TemporalDistribution(
    date=np.array([datetime(y, 1, 1).isoformat() for y in years], dtype="datetime64[s]"),
    amount=np.array([0, 0, 10, 10, 10, 10, 10, 10, 10, 10], dtype=float),
)
demand = {bd.get_node(database="foreground", code="my_product"): td_demand}

# 3. Configure LCA processing
config = lca_processor.LCAConfig(
    demand=demand,
    temporal={"start_date": datetime(2020, 1, 1)},
    characterization_methods=[
        # Any method from bd.methods; static by default
        {"category_name": "climate_change", "brightway_method": ("GWP", "100 years")},
    ],
)

# 4. Process the LCA data
lca_data = lca_processor.LCADataProcessor(config)

# 5. Convert to optimization inputs, optionally with constraints
manager = converter.ModelInputManager()
model_inputs = manager.parse_from_lca_processor(lca_data)

# 6. Build and solve the model (minimizing an impact; objective="cost" minimizes costs)
model = optimizer.create_model(model_inputs, name="my_model", objective_category="climate_change")
solved, objective, results = optimizer.solve_model(model, solver_name="glpk")

# 7. Analyze the results
pp = postprocessing.PostProcessor(solved)
pp.plot_impacts()
pp.plot_installation()
pp.plot_capacity_balance()
```

---

## Database Structure

```
Brightway Project
├── biosphere3          # Elementary flows (emissions)
├── db_2020             # Background database (year 2020)
├── db_2030             # Background database (year 2030)
└── foreground          # Your products and processes
```

The foreground database is named `"foreground"` by default; set `LCAConfig(foreground_db_name=...)` to use another name.

---

## Foreground Template

The foreground contains **product nodes** (what is demanded and produced) and **process nodes** (the candidate technologies, which produce the products):

```python
from bw_temporalis import TemporalDistribution
import numpy as np

foreground_data = {
    ("foreground", "my_product"): {
        "name": "My Product",
        "type": "product",
        "unit": "kg",
    },
    ("foreground", "my_process"): {
        "name": "Process Name",
        "location": "GLO",
        "type": "process",
        "operation_time_limits": (1, 2),  # operates at process times 1 and 2
        "exchanges": [
            {   # 1 unit produces 1 kg over its lifetime: 0.5 kg in each operating year
                "amount": 1,
                "type": "production",
                "input": ("foreground", "my_product"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array([1, 2], dtype="timedelta64[Y]"),
                    amount=np.array([0.5, 0.5]),
                ),
                "operation": True,
            },
            {   # construction input, bought once when the unit is installed
                "amount": 10,
                "type": "technosphere",
                "input": ("db_2020", "electricity"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array([0], dtype="timedelta64[Y]"),
                    amount=np.array([1.0]),
                ),
            },
            {   # operational emission, scales with the units running
                "amount": 5,
                "type": "biosphere",
                "input": ("biosphere3", "CO2"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array([1, 2], dtype="timedelta64[Y]"),
                    amount=np.array([0.5, 0.5]),
                ),
                "operation": True,
            },
        ],
    },
}
bd.Database("foreground").write(foreground_data)
```

Temporal distribution amounts are fractions of the exchange `amount` and sum to 1. See [Foreground Modeling](foreground_modeling.md) for the details.

---

## Key optimex Additions to Brightway

| Element | Field | Description |
|---------|-------|-------------|
| Foreground | separate product and process nodes | Demand is defined on products; processes produce them |
| Process | `operation_time_limits` | `(start, end)` process times of the operation phase |
| Exchange | `temporal_distribution` | When the exchange occurs, in years after installation |
| Exchange | `operation` | `True` if the exchange scales with the units running |
| Background database | `representative_time` | The year the database represents |

---

## LCAConfig Quick Reference

```python
lca_processor.LCAConfig(
    demand={product_node: temporal_distribution},
    temporal={
        "start_date": datetime(2020, 1, 1),  # first year of the study
        "time_horizon": 100,                  # years for dynamic impact assessment
    },
    characterization_methods=[
        {
            "category_name": "climate_change",
            "brightway_method": ("GWP", "100 years"),  # a method from bd.methods
            "metric": "CRF",  # optional: "CRF" or "GWP" for dynamic characterization
        },
    ],
)
```

---

## Common Constraints

Constraints are added to the model inputs with `override()`, which validates them:

```python
model_inputs = manager.override(
    cumulative_process_limits_max={"process_id": 100.0},          # units in total
    process_deployment_limits_max={("process_id", 2025): 50.0},  # units installed in 2025
    cumulative_category_impact_limits={"climate_change": 1.0e6},  # impact budget
    category_impact_limits={("climate_change", 2030): 5.0e4},    # impact limit in 2030
    existing_capacity={("old_plant", 2010): 500.0},              # units built before the study
)
```

Limits are in process units and in the unit of the impact category; see [Constraints](constraints.md).

---

## Objectives

| Minimize | Call | Extra data | Guide |
|----------|------|------------|-------|
| Impact in one category | `create_model(inputs, name, objective_category="climate_change")` | — | [Minimizing Impacts](minimizing_impacts.md) |
| Discounted cost | `create_model(inputs, name, objective="cost")` | Prices via `set_market_prices()` and `LCAConfig(price_attribute="market_price")` | [Minimizing Costs](minimizing_costs.md) |

---

## PostProcessor Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_impacts()` | DataFrame | Impact by category, process, and year |
| `get_installation()` | DataFrame | Process units installed per year (a lifetime quantity, not an annual capacity) |
| `get_operation()` | DataFrame | Units running per year |
| `get_production()` | DataFrame | Production by process, product, and year |
| `get_production_capacity()` | DataFrame | Maximum annual production available per year |
| `get_demand()` | DataFrame | Demand by product and year |
| `get_costs()` | DataFrame | Cost by driver (installation/operation), process, and year; needs prices |
| `plot_impacts()` | — (shows a plot) | Stacked bars of impacts |
| `plot_installation()` | — (shows a plot) | Bar chart of installations |
| `plot_capacity_balance()` | — (shows a plot) | Production vs. capacity |

---

## Saving & Loading

```python
# Save model inputs
manager.save_inputs("model_inputs.json")  # or .pkl

# Load model inputs
model_inputs = manager.load_inputs("model_inputs.json")

# Save solved model
optimizer.save_solved_model(solved, "solved.pkl", objective_value=objective)

# Load solved model
loaded = optimizer.load_solved_model("solved.pkl")
```
