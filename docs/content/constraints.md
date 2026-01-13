# Optimization Constraints

By default, `optimex` minimizes the environmental impact of a specified category while meeting the demand for products over time. However, real-world optimization problems often require additional constraints to reflect practical limitations, policy requirements, or scenario-specific conditions.

This page documents all available constraint options in `optimex` and how to use them.

## Overview

Constraints in `optimex` are specified through the `OptimizationModelInputs` object. After creating your model inputs from the LCA processor, you can add constraints before creating the optimization model:

```python
from optimex import converter, optimizer

# Parse LCA data
manager = converter.ModelInputManager()
model_inputs = manager.parse_from_lca_processor(lca_data_processor)

# Add constraints (examples below)
model_inputs.cumulative_process_limits_max = {"ProcessA": 100.0}
model_inputs.category_impact_limits = {("climate_change", 2030): 1000.0}

# Create and solve the model
model = optimizer.create_model(model_inputs, name="my_model", objective_category="climate_change")
solved_model, objective, results = optimizer.solve_model(model)
```

---

## Process Deployment Limits

Control how much capacity can be installed for each process.

### Time-Specific Deployment Limits

Limit the capacity installed for a specific process at a specific year.

| Field | Type | Description |
|-------|------|-------------|
| `process_deployment_limits_max` | `Dict[Tuple[str, int], float]` | Upper bound on `(process, year)` installation |
| `process_deployment_limits_min` | `Dict[Tuple[str, int], float]` | Lower bound on `(process, year)` installation |

**Example: Limit new solar capacity to 50 MW in 2025**
```python
model_inputs.process_deployment_limits_max = {
    ("solar_pv", 2025): 50.0,
    ("solar_pv", 2026): 75.0,
    ("solar_pv", 2027): 100.0,
}
```

**Example: Require at least 10 MW of wind installation in 2030**
```python
model_inputs.process_deployment_limits_min = {
    ("wind_turbine", 2030): 10.0,
}
```

### Cumulative Deployment Limits

Limit the total capacity installed for a process across all time periods.

| Field | Type | Description |
|-------|------|-------------|
| `cumulative_process_limits_max` | `Dict[str, float]` | Maximum total installation for a process |
| `cumulative_process_limits_min` | `Dict[str, float]` | Minimum total installation for a process |

**Example: Cap total nuclear capacity at 500 MW**
```python
model_inputs.cumulative_process_limits_max = {
    "nuclear_plant": 500.0,
}
```

**Example: Require at least 200 MW total renewable capacity**
```python
model_inputs.cumulative_process_limits_min = {
    "solar_pv": 100.0,
    "wind_turbine": 100.0,
}
```

---

## Process Operation Limits

Control how much a process can operate in each time period.

| Field | Type | Description |
|-------|------|-------------|
| `process_operation_limits_max` | `Dict[Tuple[str, int], float]` | Upper bound on `(process, year)` operation |
| `process_operation_limits_min` | `Dict[Tuple[str, int], float]` | Lower bound on `(process, year)` operation |

!!! note "Operation vs Installation"
    Operation limits constrain how much of the installed capacity is actually used in each period. A process can only operate up to its installed capacity, but operation limits can further restrict this.

**Example: Limit coal plant operation during phase-out**
```python
model_inputs.process_operation_limits_max = {
    ("coal_plant", 2025): 80.0,   # Max 80% of capacity
    ("coal_plant", 2030): 50.0,   # Max 50% of capacity
    ("coal_plant", 2035): 0.0,    # Complete phase-out
}
```

**Example: Ensure minimum baseload operation**
```python
model_inputs.process_operation_limits_min = {
    ("nuclear_plant", 2025): 50.0,  # Minimum 50 MW operation
    ("nuclear_plant", 2026): 50.0,
}
```

---

## Category Impact Limits

Constrain the environmental impact in specific categories, either at specific times or cumulatively.

### Time-Specific Impact Limits

Limit the impact in a category at a specific year.

| Field | Type | Description |
|-------|------|-------------|
| `category_impact_limits` | `Dict[Tuple[str, int], float]` | Upper bound on `(category, year)` impact |

**Example: Annual carbon budget**
```python
model_inputs.category_impact_limits = {
    ("climate_change", 2025): 1000000.0,  # 1 Mt CO2-eq in 2025
    ("climate_change", 2030): 500000.0,   # 0.5 Mt CO2-eq in 2030
    ("climate_change", 2035): 100000.0,   # 0.1 Mt CO2-eq in 2035
}
```

**Example: Limit land use in specific years**
```python
model_inputs.category_impact_limits = {
    ("land_use", 2025): 5000.0,  # Max 5000 m² in 2025
    ("land_use", 2030): 4000.0,  # Max 4000 m² in 2030
}
```

### Cumulative Impact Limits

Limit the total impact across all time periods.

| Field | Type | Description |
|-------|------|-------------|
| `cumulative_category_impact_limits` | `Dict[str, float]` | Upper bound on total impact in a category |

**Example: Total carbon budget**
```python
model_inputs.cumulative_category_impact_limits = {
    "climate_change": 5000000.0,  # 5 Mt CO2-eq total budget
}
```

**Example: Multi-category constraints**
```python
model_inputs.cumulative_category_impact_limits = {
    "climate_change": 5000000.0,
    "water_use": 1000000.0,
    "land_use": 50000.0,
}
```

!!! tip "Combining Time-Specific and Cumulative Limits"
    You can use both types of limits simultaneously. For example, set annual carbon budgets with `category_impact_limits` while also enforcing a total budget with `cumulative_category_impact_limits`.

---

## Flow Limits

Constrain specific material or energy flows in the system.

### Time-Specific Flow Limits

Limit flows at specific times. Flows can be products, intermediate flows, or elementary flows.

| Field | Type | Description |
|-------|------|-------------|
| `flow_limits_max` | `Dict[Tuple[str, int], float]` | Upper bound on `(flow, year)` |
| `flow_limits_min` | `Dict[Tuple[str, int], float]` | Lower bound on `(flow, year)` |

**Example: Limit natural gas consumption**
```python
model_inputs.flow_limits_max = {
    ("natural_gas", 2025): 1000.0,  # Max 1000 units in 2025
    ("natural_gas", 2030): 500.0,   # Max 500 units in 2030
}
```

**Example: Require minimum hydrogen production**
```python
model_inputs.flow_limits_min = {
    ("hydrogen", 2030): 100.0,  # At least 100 units in 2030
}
```

**Example: Limit CO2 emissions per year**
```python
model_inputs.flow_limits_max = {
    ("CO2", 2025): 50000.0,
    ("CO2", 2030): 25000.0,
}
```

### Cumulative Flow Limits

Limit the total flow across all time periods.

| Field | Type | Description |
|-------|------|-------------|
| `cumulative_flow_limits_max` | `Dict[str, float]` | Upper bound on total flow |
| `cumulative_flow_limits_min` | `Dict[str, float]` | Lower bound on total flow |

**Example: Total resource budget**
```python
model_inputs.cumulative_flow_limits_max = {
    "rare_earth_elements": 10000.0,  # Limited total availability
}
```

---

## Process Coupling

Link the deployment of related processes with fixed ratios.

| Field | Type | Description |
|-------|------|-------------|
| `process_coupling` | `Dict[Tuple[str, str], float]` | Ratio constraint: `process1 = ratio * process2` |

**Example: Coupled storage and generation**
```python
# For every 4 MW of solar, require 1 MW of battery storage
model_inputs.process_coupling = {
    ("battery_storage", "solar_pv"): 0.25,  # storage = 0.25 * solar
}
```

**Example: Backup capacity requirements**
```python
# Gas backup must be at least 50% of wind capacity
model_inputs.process_coupling = {
    ("gas_peaker", "wind_turbine"): 0.5,
}
```

!!! warning "Coupling Direction"
    The coupling constraint enforces `installation[process1] = ratio * installation[process2]`. Make sure to set up the ratio in the correct direction.

---

## Existing Capacity (Brownfield)

Model systems with pre-existing infrastructure that was installed before the optimization horizon.

| Field | Type | Description |
|-------|------|-------------|
| `existing_capacity` | `Dict[Tuple[str, int], float]` | Capacity at `(process, installation_year)` |

!!! info "Brownfield vs Greenfield"
    - **Greenfield**: Optimization starts from scratch with no existing capacity
    - **Brownfield**: Some capacity already exists from previous investments

**Example: Existing power plant fleet**
```python
model_inputs.existing_capacity = {
    ("coal_plant", 2010): 500.0,    # 500 MW installed in 2010
    ("gas_plant", 2015): 300.0,     # 300 MW installed in 2015
    ("nuclear_plant", 2005): 1000.0, # 1000 MW installed in 2005
}
```

Key characteristics of existing capacity:

1. **Installation year must be before the optimization horizon**: If your `SYSTEM_TIME` starts at 2020, existing capacity must have installation years before 2020.

2. **Installation impacts are excluded**: The environmental impacts of building existing capacity are considered "sunk costs" and not counted in the optimization.

3. **Operation impacts are included**: When existing capacity operates during the optimization horizon, its operational emissions are counted.

4. **Capacity retirement**: Existing capacity is subject to the same lifetime constraints as new capacity. It will retire based on the process's operation time limits.

---

## Combining Constraints

Multiple constraints can be combined to model complex scenarios:

```python
from optimex import converter, optimizer

# Parse LCA data
manager = converter.ModelInputManager()
model_inputs = manager.parse_from_lca_processor(lca_data_processor)

# Scenario: Rapid decarbonization with practical limits
model_inputs.cumulative_category_impact_limits = {
    "climate_change": 5000000.0,  # Total carbon budget
}
model_inputs.category_impact_limits = {
    ("climate_change", 2030): 800000.0,  # Annual limit in 2030
    ("climate_change", 2040): 200000.0,  # Annual limit in 2040
}
model_inputs.process_deployment_limits_max = {
    ("nuclear_plant", 2025): 0.0,  # No new nuclear until 2026
    ("nuclear_plant", 2026): 100.0,
    ("nuclear_plant", 2027): 200.0,
}
model_inputs.cumulative_process_limits_max = {
    "coal_plant": 0.0,  # No new coal plants
}
model_inputs.existing_capacity = {
    ("coal_plant", 2000): 2000.0,  # Existing coal to phase out
    ("gas_plant", 2010): 1000.0,
}

# Create and solve
model = optimizer.create_model(
    model_inputs,
    name="decarbonization_scenario",
    objective_category="climate_change"
)
solved_model, objective, results = optimizer.solve_model(model)
```

---

## Constraint Summary Table

| Constraint Type | Time-Specific | Cumulative |
|-----------------|---------------|------------|
| Process Deployment | `process_deployment_limits_max/min` | `cumulative_process_limits_max/min` |
| Process Operation | `process_operation_limits_max/min` | — |
| Category Impact | `category_impact_limits` | `cumulative_category_impact_limits` |
| Flow Limits | `flow_limits_max/min` | `cumulative_flow_limits_max/min` |
| Process Coupling | `process_coupling` | — |
| Existing Capacity | `existing_capacity` | — |

---

## Handling Infeasibility

If constraints are too restrictive, the optimization may become infeasible (no solution exists). To debug:

```python
# Enable IIS (Irreducible Infeasible Set) generation
solved_model, objective, results = optimizer.solve_model(
    model,
    compute_iis=True
)
# Check model_iis.ilp for the conflicting constraints
```

!!! tip "Common Causes of Infeasibility"
    - Demand cannot be met with available capacity limits
    - Impact limits are below the minimum achievable impact
    - Conflicting min/max constraints
    - Insufficient existing capacity with restrictive deployment limits
