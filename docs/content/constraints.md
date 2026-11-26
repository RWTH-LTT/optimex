---
icon: lucide/lock
tags:
  - optimization
  - constraints
---

# Optimization Constraints

`optimex` minimizes the [objective](choosing_objective/index.md) of your choice, an impact category or costs, while meeting the demand in every year (see [Theory: How It Works](theory_how_it_works.md)). Real-world optimization problems often require additional constraints to reflect practical limitations, policy requirements, or scenario-specific conditions — such as existing brownfield capacities, resource bottlenecks, or cumulative environmental budgets.

This page documents all available constraint options in `optimex`.

## Overview

Constraints are fields of the `OptimizationModelInputs`. After creating the model inputs from the LCA processor, add constraints with `ModelInputManager.override()` before creating the optimization model:

```python
from optimex import converter, optimizer

# Parse LCA data
manager = converter.ModelInputManager()
model_inputs = manager.parse_from_lca_processor(lca_data_processor)

# Add constraints (examples below)
model_inputs = manager.override(
    cumulative_process_limits_max={"ProcessA": 100.0},
    category_impact_limits={("climate_change", 2030): 1000.0},
)

# Create and solve the model
model = optimizer.create_model(model_inputs, name="my_model", objective_category="climate_change")
solved_model, objective, results = optimizer.solve_model(model, solver_name="glpk")
```

`override()` validates the new inputs: unknown process, flow, or category codes and years outside the system time raise an error right away. It replaces each field you pass as a whole and keeps all others, so you can add constraints in several calls.

!!! warning "Don't assign fields directly"
    `model_inputs.process_deployment_limits_max = {...}` also changes the inputs, but skips the validation. A typo in a process code then silently creates a constraint that never applies, and the saved inputs fail to load again. Use `override()`.

Processes, products, and flows are identified by their **codes** in the Brightway databases, not by their names. For ecoinvent flows, these are long identifiers; look them up with `bd.get_node(...)["code"]`.

---

## Process Deployment Limits

Control how many units of a process can be installed.

!!! note "Limits are in process units"
    Deployment limits bound `var_installation`, which counts process **units**. One unit delivers the production of its production exchange over its whole lifetime, so a limit of 50 allows a lifetime output of 50 times that production, not an annual output of 50. See [Foreground Modeling](foreground_modeling.md#what-one-installed-unit-means).

    If your data is an annual capacity, e.g. in MW or t/yr, convert it to units by dividing by the annual output of one unit, the per-year entry of its production exchange: 50 MW of a plant whose unit produces 8760 MWh per year, i.e. 1 MW, is 50 units.

### Time-Specific Deployment Limits

Limit the units of a process installed in a specific year.

| Field | Type | Description |
|-------|------|-------------|
| `process_deployment_limits_max` | `Dict[Tuple[str, int], float]` | Upper bound on units installed for `(process, year)` |
| `process_deployment_limits_min` | `Dict[Tuple[str, int], float]` | Lower bound on units installed for `(process, year)` |

**Example: Limit the solar ramp-up**
```python
model_inputs = manager.override(
    process_deployment_limits_max={
        ("solar_pv", 2025): 50.0,   # at most 50 units installed in 2025
        ("solar_pv", 2026): 75.0,
        ("solar_pv", 2027): 100.0,
    },
)
```

**Example: Require a minimum wind installation in 2030**
```python
model_inputs = manager.override(
    process_deployment_limits_min={("wind_turbine", 2030): 10.0},  # at least 10 units
)
```

### Cumulative Deployment Limits

Limit the units of a process installed over all years together.

| Field | Type | Description |
|-------|------|-------------|
| `cumulative_process_limits_max` | `Dict[str, float]` | Maximum total units installed for a process |
| `cumulative_process_limits_min` | `Dict[str, float]` | Minimum total units installed for a process |

**Example: Cap the total nuclear build-out**
```python
model_inputs = manager.override(
    cumulative_process_limits_max={"nuclear_plant": 500.0},  # at most 500 units in total
)
```

**Example: Require a minimum renewable build-out**
```python
model_inputs = manager.override(
    cumulative_process_limits_min={"solar_pv": 100.0, "wind_turbine": 100.0},
)
```

---

## Process Operation Limits

Control how many units of a process run in each year.

| Field | Type | Description |
|-------|------|-------------|
| `process_operation_limits_max` | `Dict[Tuple[str, int], float]` | Upper bound on units running for `(process, year)` |
| `process_operation_limits_min` | `Dict[Tuple[str, int], float]` | Lower bound on units running for `(process, year)` |

!!! note "Operation vs Installation"
    Operation limits constrain how much of the installed capacity is actually used in each year. A process can only operate up to its installed capacity, but operation limits can further restrict this.

    Both variables are counted in process units: `var_operation[p, v, t]` is how many units of vintage `v` run in year `t`, bounded by `var_installation[p, v]`. An operation limit caps the units running in that year, summed over all vintages. It is an absolute number of units, not a share of the capacity. Multiply by the output per unit and year to express it as a production limit, or use [flow limits](#flow-limits) to bound production directly.

**Example: Phase out coal**
```python
model_inputs = manager.override(
    process_operation_limits_max={
        ("coal_plant", 2025): 80.0,  # at most 80 units running in 2025
        ("coal_plant", 2030): 50.0,  # at most 50 units running in 2030
        ("coal_plant", 2035): 0.0,   # complete phase-out
    },
)
```

**Example: Ensure a minimum baseload**
```python
model_inputs = manager.override(
    process_operation_limits_min={
        ("nuclear_plant", 2025): 50.0,  # at least 50 units running
        ("nuclear_plant", 2026): 50.0,
    },
)
```

---

## Category Impact Limits

Constrain the environmental impact in specific categories, either in specific years or cumulatively. Impact limits apply with either objective. Combined with the [cost objective](minimizing_costs.md), they give the cheapest pathway within an environmental budget, and sweeping the limit traces the [trade-offs](trade_offs.md).

!!! note "Units of impact limits"
    Limits use the unit of the category's characterization: the unit of the Brightway method for static methods (e.g. kg CO2-eq for GWP100), kg CO2-eq for dynamic GWP, and W·m⁻²·yr for CRF. A budget in tonnes of CO2-eq never binds on a CRF category. See [Characterization Methods](lca_processing.md#characterization-methods).

### Time-Specific Impact Limits

Limit the impact in a category in a specific year.

| Field | Type | Description |
|-------|------|-------------|
| `category_impact_limits` | `Dict[Tuple[str, int], float]` | Upper bound on `(category, year)` impact |

**Example: Annual carbon limits** (static GWP100 category, in kg CO2-eq)
```python
model_inputs = manager.override(
    category_impact_limits={
        ("climate_change", 2025): 1.0e9,  # 1 Mt CO2-eq in 2025
        ("climate_change", 2030): 5.0e8,  # 0.5 Mt CO2-eq in 2030
        ("climate_change", 2035): 1.0e8,  # 0.1 Mt CO2-eq in 2035
    },
)
```

**Example: Limit land use in specific years**
```python
model_inputs = manager.override(
    category_impact_limits={
        ("land_use", 2025): 5000.0,  # in the unit of the land use method
        ("land_use", 2030): 4000.0,
    },
)
```

### Cumulative Impact Limits

Limit the total impact over all years.

| Field | Type | Description |
|-------|------|-------------|
| `cumulative_category_impact_limits` | `Dict[str, float]` | Upper bound on total impact in a category |

**Example: Total carbon budget**
```python
model_inputs = manager.override(
    cumulative_category_impact_limits={"climate_change": 5.0e9},  # 5 Mt CO2-eq in total
)
```

**Example: Multi-category constraints**
```python
model_inputs = manager.override(
    cumulative_category_impact_limits={
        "climate_change": 5.0e9,
        "water_use": 1.0e6,
        "land_use": 5.0e4,
    },
)
```

!!! tip "Combining Time-Specific and Cumulative Limits"
    You can use both types of limits simultaneously. For example, set annual carbon budgets with `category_impact_limits` while also enforcing a total budget with `cumulative_category_impact_limits`.

---

## Flow Limits

Constrain specific material or energy flows in the system, in the units of the flow.

### Time-Specific Flow Limits

Limit flows in specific years. Flows can be foreground products, intermediate flows (background products bought by the foreground), or elementary flows.

| Field | Type | Description |
|-------|------|-------------|
| `flow_limits_max` | `Dict[Tuple[str, int], float]` | Upper bound on `(flow, year)` |
| `flow_limits_min` | `Dict[Tuple[str, int], float]` | Lower bound on `(flow, year)` |

**Example: Limit natural gas consumption**
```python
model_inputs = manager.override(
    flow_limits_max={
        ("natural_gas", 2025): 1000.0,  # at most 1000 units of the flow in 2025
        ("natural_gas", 2030): 500.0,
    },
)
```

**Example: Require a minimum hydrogen production**
```python
model_inputs = manager.override(
    flow_limits_min={("hydrogen", 2030): 100.0},  # at least 100 units in 2030
)
```

**Example: Limit CO2 emissions per year**
```python
co2 = bd.get_node(database="biosphere3", name="Carbon dioxide, fossil", categories=("air",))
model_inputs = manager.override(
    flow_limits_max={(co2["code"], 2025): 50000.0, (co2["code"], 2030): 25000.0},
)
```

### Cumulative Flow Limits

Limit the total flow over all years.

| Field | Type | Description |
|-------|------|-------------|
| `cumulative_flow_limits_max` | `Dict[str, float]` | Upper bound on total flow |
| `cumulative_flow_limits_min` | `Dict[str, float]` | Lower bound on total flow |

**Example: Total resource budget**
```python
model_inputs = manager.override(
    cumulative_flow_limits_max={"rare_earth_elements": 10000.0},  # limited availability
)
```

!!! warning "Constraining an elementary flow"
    Elementary flows without a characterization factor in any of your impact
    categories are dropped during LCA processing, since they cannot affect any
    impact. To constrain such a flow, list it in
    `LCAConfig.background_inventory.retain_flows`. See
    [Background Inventory](lca_processing.md#background-inventory).
    `override()` raises an error if a limited flow is not in the model.

---

## Process Coupling

Tie the installations of two processes to a fixed ratio.

| Field | Type | Description |
|-------|------|-------------|
| `process_coupling` | `Dict[Tuple[str, str], float]` | `(process1, process2): ratio` enforces `installation[process1] = ratio × installation[process2]` |

The coupling is an **exact equality**, applied in every year: the units of `process1` installed in a year always equal `ratio` times the units of `process2` installed in that year. It is not a minimum.

**Example: Storage installed together with solar**
```python
# 1 unit of battery storage for every 4 units of solar, in every year
model_inputs = manager.override(
    process_coupling={("battery_storage", "solar_pv"): 0.25},
)
```

!!! warning "Coupling Direction"
    The coupling constraint enforces `installation[process1] = ratio × installation[process2]`. Make sure to set up the ratio in the correct direction.

For a minimum or maximum share instead of a fixed ratio, combine deployment limits, or model the dependent process as an input of the other.

---

## Existing Capacity (Brownfield)

Model systems with pre-existing infrastructure that was installed before the optimization horizon.

| Field | Type | Description |
|-------|------|-------------|
| `existing_capacity` | `Dict[Tuple[str, int], float]` | Units at `(process, installation_year)` |

Existing capacity is given in the same process units as `var_installation`, and the entries behave like vintages of that installation year: they can run while their lifecycle stage falls inside the operation window, and they carry no installation impacts.

!!! info "Brownfield vs Greenfield"
    - **Greenfield**: Optimization starts from scratch with no existing capacity
    - **Brownfield**: Some capacity already exists from previous investments

**Example: Existing power plant fleet**
```python
model_inputs = manager.override(
    existing_capacity={
        ("coal_plant", 2010): 500.0,      # 500 units installed in 2010
        ("gas_plant", 2015): 300.0,       # 300 units installed in 2015
        ("nuclear_plant", 2005): 1000.0,  # 1000 units installed in 2005
    },
)
```

Key characteristics of existing capacity:

1. **Installation year must be before the optimization horizon**: If your system time starts in 2020, existing capacity must have installation years before 2020.

2. **Installation impacts are excluded**: The environmental impacts of building existing capacity are considered "sunk" and not counted in the optimization.

3. **Operation impacts are included**: When existing capacity operates during the optimization horizon, its operational emissions are counted.

4. **Capacity retirement**: Existing capacity is subject to the same lifetime as new capacity. It retires when its lifecycle stage leaves the process's operation window.

---

## Combining Constraints

Multiple constraints can be combined to model complex scenarios:

```python
from optimex import converter, optimizer

# Parse LCA data
manager = converter.ModelInputManager()
model_inputs = manager.parse_from_lca_processor(lca_data_processor)

# Scenario: Rapid decarbonization with practical limits
model_inputs = manager.override(
    cumulative_category_impact_limits={"climate_change": 5.0e9},  # total carbon budget
    category_impact_limits={
        ("climate_change", 2030): 8.0e8,  # annual limit in 2030
        ("climate_change", 2040): 2.0e8,  # annual limit in 2040
    },
    process_deployment_limits_max={
        ("nuclear_plant", 2025): 0.0,  # no new nuclear until 2026
        ("nuclear_plant", 2026): 100.0,
        ("nuclear_plant", 2027): 200.0,
    },
    cumulative_process_limits_max={"coal_plant": 0.0},  # no new coal plants
    existing_capacity={
        ("coal_plant", 2000): 2000.0,  # existing coal to phase out
        ("gas_plant", 2010): 1000.0,
    },
)

# Create and solve
model = optimizer.create_model(
    model_inputs, name="decarbonization_scenario", objective_category="climate_change"
)
solved_model, objective, results = optimizer.solve_model(model, solver_name="glpk")
```

---

## Constraint Summary Table

| Constraint Type | Time-Specific | Cumulative | Unit |
|-----------------|---------------|------------|------|
| Process Deployment | `process_deployment_limits_max/min` | `cumulative_process_limits_max/min` | process units |
| Process Operation | `process_operation_limits_max/min` | — | process units running |
| Category Impact | `category_impact_limits` | `cumulative_category_impact_limits` | unit of the category |
| Flow Limits | `flow_limits_max/min` | `cumulative_flow_limits_max/min` | unit of the flow |
| Process Coupling | `process_coupling` | — | ratio of installations |
| Existing Capacity | `existing_capacity` | — | process units |

---

## Handling Infeasibility

Constraints that are too restrictive make the problem infeasible. See [Solving: Handling Infeasibility](solving.md#handling-infeasibility) for finding the conflicting constraints and for common causes.
