---
icon: lucide/box
tags:
  - optimization
  - solver
---

# Pyomo Backend

`create_model(..., backend="pyomo")`, the default, builds the optimization model
as a Pyomo `ConcreteModel` from the backend-independent formulation
(`optimex.formulation`). Use it for anything that works on the model itself:
fixing variables (e.g. to evaluate a given design), adding constraints, writing
IIS files, or Pyomo's direct and persistent solver interfaces. For systems with
thousands of processes, the [linopy backend](linopy_backend.md) builds the same
model faster. See [Solving: Large Models](../content/solving.md#large-models).

## Module Reference

::: optimex.pyomo_backend
