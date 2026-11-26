---
icon: lucide/layers
tags:
  - optimization
  - solver
---

# linopy Backend

`create_model(..., backend="linopy")` builds the optimization model with
[linopy](https://linopy.readthedocs.io) instead of Pyomo. Both backends are
emitted from the same backend-independent formulation
(`optimex.formulation.build_formulation`), so they have the same variables,
constraints and optimum. linopy receives every constraint block as arrays,
which makes it the faster choice for systems with thousands of processes.

`solve_model()` stores the solution on the `LinopyModel` under the attribute
names of a solved Pyomo model, so the PostProcessor and `save_solved_model()`
work unchanged. See [Solving: Large Models](../content/solving.md#large-models).

## Module Reference

::: optimex.linopy_backend

## Shared Formulation

::: optimex.formulation
