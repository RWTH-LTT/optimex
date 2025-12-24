---
icon: lucide/refresh-cw
---

# Optimizer

Optimization model construction and solving for temporal LCA-based pathway optimization.

This module creates and solves Pyomo optimization models that minimize environmental impacts over time while meeting demand constraints and respecting process limits.

## Key Functions

- **`create_model()`**: Constructs a Pyomo ConcreteModel from optimization inputs
- **`solve_model()`**: Solves the model and returns denormalized results

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

## Module Reference

::: optimex.optimizer
    options:
      show_root_heading: false
      show_root_toc_entry: false
