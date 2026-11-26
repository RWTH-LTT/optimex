---
icon: lucide/refresh-cw
tags:
  - optimization
  - solver
---

# Optimizer

Optimization model construction and solving for temporal LCA-based pathway optimization.

This module creates and solves Pyomo optimization models that minimize either an environmental impact category or the discounted cost of background purchases over time, while meeting demand constraints and respecting process limits.

## Key Functions

- **`create_model()`**: Constructs a Pyomo ConcreteModel from optimization inputs
- **`solve_model()`**: Solves the model and returns denormalized results

## Objectives

| Objective | Minimizes |
|-----------|-----------|
| `objective="environmental"` (default) | `total_impact[objective_category]` |
| `objective="cost"` | `total_cost`, the discounted cost of the background purchases priced with `intermediate_flow_prices` |

Impact limits apply with either objective. See
[Choosing an Objective](../content/choosing_objective/index.md).

## Scaling Convention

The optimization uses a two-tier scaling system for numerical stability:

### Decision Variables (Real Units)

- `var_installation[p, t]`: Number of process units installed (dimensionless)
- `var_operation[p, t]`: Operation level (dimensionless, 0 to capacity)

### Parameters (Scaled Units)

**Foreground parameters** (scaled by `fg_scale`):

- `foreground_production[p, r, tau]`: kg product per process unit
- `foreground_biosphere[p, e, tau]`: kg emission per process unit
- `foreground_technosphere[p, i, tau]`: kg intermediate per process unit

**Characterization parameters** (scaled by `cat_scales[category]`):

- `characterization[c, e, t]`: impact per kg emission
- `category_impact_limit[c]`: maximum impact allowed

**Price parameters** (scaled by `cost_scale`):

- `intermediate_flow_prices[i, t]`: price per unit of intermediate flow

Cost expressions (`installation_cost`, `operation_cost`, `total_cost`) are in units
of `fg_scale * cost_scale`. `solve_model()` and `PostProcessor.get_costs()` return
them in the currency of the input prices.

## Module Reference

::: optimex.optimizer
    options:
      show_root_heading: false
      show_root_toc_entry: false
