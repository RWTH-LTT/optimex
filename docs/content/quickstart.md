# Quick Start

A condensed reference for using `optimex`. For the underlying framework, see the [Theory](theory.md) page. For detailed explanations, see the [User Guide](brightway_setup.md), [Examples](examples/index.md), or the [API Reference](../api/index.md).

---

## Install `optimex`

```bash
pip install optimex
```

For other installation methods (uv, conda) and platform-specific notes, see the [Installation guide](installation.md).

---

## Minimal Working Example

```python
from datetime import datetime
import numpy as np
import bw2data as bd
from bw_temporalis import TemporalDistribution
from optimex import lca_processor, converter, optimizer, postprocessing

# 1. Set up Brightway project
bd.projects.set_current("my_project")

# 2. Define temporal demand
years = range(2020, 2030)
td_demand = TemporalDistribution(
    date=np.array([datetime(y, 1, 1).isoformat() for y in years], dtype='datetime64[s]'),
    amount=np.array([0, 0, 10, 10, 10, 10, 10, 10, 10, 10]),
)
demand = {bd.get_node(code="my_product"): td_demand}

# 3. Configure LCA processing
config = lca_processor.LCAConfig(
    demand=demand,
    temporal={
        "start_date": datetime(2020, 1, 1),
        "temporal_resolution": "year",
        "time_horizon": 100,
    },
    characterization_methods=[
        {"category_name": "climate_change", "brightway_method": ("IPCC", "GWP100")},
    ],
)

# 4. Process LCA data
lca_data = lca_processor.LCADataProcessor(config)

# 5. Convert to optimization inputs
manager = converter.ModelInputManager()
model_inputs = manager.parse_from_lca_processor(lca_data)

# 6. Create and solve model
model = optimizer.create_model(model_inputs, name="my_model", objective_category="climate_change")
solved, objective, results = optimizer.solve_model(model, solver_name="glpk")

# 7. Analyze results
pp = postprocessing.PostProcessor(solved)
pp.plot_impacts()
pp.plot_installation()
pp.plot_production_and_demand()
```

---

## Database Structure

```
Brightway Project
├── biosphere3          # Elementary flows (emissions)
├── db_2020             # Background database (year 2020)
├── db_2030             # Background database (year 2030)
└── foreground          # Your processes (required name)
```

---

## Foreground Process Template

```python
{
    ("foreground", "my_process"): {
        "name": "Process Name",
        "location": "GLO",
        "operation_time_limits": (1, 2),  # (start_tau, end_tau)
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "input": ("foreground", "my_product"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array([0, 1, 2], dtype="timedelta64[Y]"),
                    amount=np.array([0, 0.5, 0.5]),
                ),
                "operation": True,
            },
            {
                "amount": 10,
                "type": "technosphere",
                "input": ("db_2020", "electricity"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array([0], dtype="timedelta64[Y]"),
                    amount=np.array([1]),
                ),
            },
            {
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
    }
}
```

---

## Key optimex Additions to Brightway

| Element | Field | Description |
|---------|-------|-------------|
| Process | `operation_time_limits` | `(start, end)` tuple defining operation phase |
| Exchange | `temporal_distribution` | When the exchange occurs (relative years) |
| Exchange | `operation` | `True` if scales with operation level |
| Database | `representative_time` | Metadata for background DB timestamp |

---

## LCAConfig Quick Reference

```python
lca_processor.LCAConfig(
    demand={product_node: temporal_distribution},
    temporal={
        "start_date": datetime(2020, 1, 1),
        "temporal_resolution": "year",  # or "month", "day"
        "time_horizon": 100,            # years for impact assessment
    },
    characterization_methods=[
        {
            "category_name": "climate_change",
            "brightway_method": ("IPCC", "GWP100"),
            "metric": "CRF",  # Optional: "CRF" or "GWP" for dynamic
        },
    ],
)
```

---

## Common Constraints

```python
# Limit total capacity
model_inputs.cumulative_process_limits_max = {"process_id": 100.0}

# Limit yearly installation
model_inputs.process_deployment_limits_max = {("process_id", 2025): 50.0}

# Carbon budget (cumulative)
model_inputs.cumulative_category_impact_limits = {"climate_change": 1000000.0}

# Annual carbon limit
model_inputs.category_impact_limits = {("climate_change", 2030): 50000.0}

# Pre-existing capacity
model_inputs.existing_capacity = {("old_plant", 2010): 500.0}
```

---

## PostProcessor Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_impacts()` | DataFrame | Impact by process, category, time |
| `get_installation()` | DataFrame | Capacity installed per time |
| `get_operation()` | DataFrame | Operation level per time |
| `get_production()` | DataFrame | Production by process and product |
| `get_demand()` | DataFrame | Demand fulfillment over time |
| `plot_impacts()` | Figure | Stacked area plot of impacts |
| `plot_installation()` | Figure | Bar chart of installations |
| `plot_production_and_demand()` | Figure | Production vs demand comparison |

---

## Saving & Loading

```python
# Save model inputs
manager.save_inputs("model_inputs.json")  # or .pkl

# Load model inputs
manager.load_inputs("model_inputs.json")
model_inputs = manager.model_inputs

# Save solved model
optimizer.save_solved_model(solved, "solved.pkl", objective_value=objective)

# Load solved model
loaded = optimizer.load_solved_model("solved.pkl")
```
