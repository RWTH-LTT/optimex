---
icon: lucide/refresh-cw
tags:
  - optimization
  - solver
---

# Optimizer

Optimization model construction and solving for temporal LCA-based pathway optimization.

This module creates and solves Pyomo optimization models that minimize
environmental impacts or economic costs over time while meeting demand
constraints and respecting process limits.

## Key Functions

- **`create_model()`**: Constructs a Pyomo ConcreteModel from optimization inputs
- **`solve_model()`**: Solves the model and returns denormalized results

## Objectives

`create_model()` supports two objective modes:

| Objective | Description |
|-----------|-------------|
| `objective="environmental"` | Default. Minimize `total_impact[objective_category]` |
| `objective="cost"` | Minimize `total_cost` from first-level background purchases |

For cost optimization, `objective_category` is still required because
environmental impacts and environmental constraints may remain part of the model.

```python
model = optimizer.create_model(
    inputs=model_inputs,
    name="cost_model",
    objective_category="climate_change",
    objective="cost",
)
```

Cost-related expressions include:

| Expression | Description |
|------------|-------------|
| `background_purchase_cap[i, t]` | Installation-related first-level background purchase |
| `background_purchase_op[i, t]` | Operation-related first-level background purchase |
| `cost_cap[t]` | Installation-related cost in year `t` |
| `cost_op[t]` | Operation-related cost in year `t` |
| `discount_factor[t]` | Discount factor for year `t` |
| `total_cost` | Discounted total cost |

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

**Economic parameters** are provided as real prices:

- `intermediate_costs_cap[i, t]`: price for installation-related purchases
- `intermediate_costs_op[i, t]`: price for operation-related purchases
- `discount_rate`: optional discount rate for `objective="cost"`
- `discount_reference_year`: optional reference year for discounting

Direct background purchases are converted back to real units before costs are
calculated, so the cost objective returned by `solve_model()` is already in the
monetary unit used by the input prices.

## Module Reference

::: optimex.optimizer
    options:
      show_root_heading: false
      show_root_toc_entry: false
