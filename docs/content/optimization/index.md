---
icon: lucide/sliders-horizontal
tags:
  - optimization
  - setup
---

# Optimization

With the product system in place, three steps lead from the Brightway model to an
optimal pathway:

```
LCAConfig → LCADataProcessor → OptimizationModelInputs → (+ constraints) → create_model → solve_model
```

<div class="grid cards" markdown>

-   :lucide-cpu:{ style="margin-right: 0.25rem;" } **[LCA Processing](../lca_processing.md)**

    ---

    Define the demand, configure the time-explicit LCA, and turn the product
    system into optimization inputs. The slowest step, so the inputs can be
    saved and reused.

-   :lucide-lock:{ style="margin-right: 0.25rem;" } **[Constraints](../constraints.md)**

    ---

    Add deployment and operation limits, impact budgets, flow limits, process
    coupling, and existing capacity to the inputs.

-   :lucide-rocket:{ style="margin-right: 0.25rem;" } **[Solving](../solving.md)**

    ---

    Build the model for the chosen [objective](../choosing_objective/index.md),
    solve it, handle infeasibility, and save the solution.

</div>

Once the inputs exist, constraints and solving are fast. To solve many variants,
such as a range of limits, scenarios, or Monte Carlo samples, see
[Analyses](../analyses/index.md).
