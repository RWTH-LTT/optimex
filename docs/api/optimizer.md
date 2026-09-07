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
- **`solve_model()`**: Returns the solved model, objective value in original units, and solver results

## Objectives

`create_model()` supports two objective modes:

| Objective | Description |
|-----------|-------------|
| `objective="environmental"` | Default. Minimize `total_impact[objective_category]` |
| `objective="cost"` | Minimize `total_cost` from first-level background purchases |

For cost optimization, `objective_category` is required by the current function
signature but does not determine the cost objective or impose environmental
limits. Configure those limits separately in the model inputs.

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
| `cost_cap[t]` | Undiscounted installation-related cost in year `t` |
| `cost_op[t]` | Undiscounted operation-related cost in year `t` |
| `discount_factor[t]` | Discount factor for year `t` |
| `total_cost` | Discounted total cost |

## Scaling Convention

The optimization uses a two-tier scaling system for numerical stability:

### Decision Variables (Unscaled Process Units)

- `var_installation[p, v]`: Continuous number of process units installed in vintage `v`
- `var_operation[p, v, t]`: Operation level of process `p`, vintage `v`, at system time `t`, constrained by available installed capacity

### Parameters (Scaled Units)

**Foreground parameters** (scaled by `fg_scale`):

- `foreground_production[p, r, tau]`: kg product per process unit
- `foreground_biosphere[p, e, tau]`: kg emission per process unit
- `foreground_technosphere[p, i, tau]`: kg intermediate per process unit

**Characterization parameters** (scaled by `cat_scales[category]`):

- `characterization[c, e, t]`: impact per kg emission
- `category_impact_limits[(c, t)]` (input field): maximum impact allowed at system time `t`

**Economic parameters** include unscaled prices per product unit:

- `intermediate_costs_cap[i, t]`: price for installation-related purchases
- `intermediate_costs_op[i, t]`: price for operation-related purchases
- `discount_rate`: nonnegative annual discount rate; defaults to zero
- `discount_reference_year`: reference year for discounting; defaults to the first system time

Direct background purchases are converted back to original product units before costs are
calculated, so the cost objective returned by `solve_model()` is already in the
monetary unit used by the input prices.

Installation-related purchases need not occur in the installation year.
Prices and discount factors apply at the system time of each purchase.
See [Economic Optimization](../content/economic_optimization.md) for the cost
boundary, price coverage, discounting, and modeling assumptions.

## Module Reference

::: optimex.optimizer
    options:
      show_root_heading: false
      show_root_toc_entry: false
