---
icon: lucide/list-ordered
tags:
  - optimization
  - analysis
---

# Sweeps and Scenarios


Sweeps over a limit (e.g. a [Pareto frontier](trade_offs.md)), scenario
comparisons, and sensitivity studies solve the same product system many times
with slightly different data. Solve them together with
`optimizer.solve_series` instead of a loop over `create_model` and
`solve_model`:

```python
budgets = [4.0e5, 4.5e5, 5.0e5, 5.5e5]
sweep = optimizer.solve_series(
    {
        budget: model_inputs.model_copy(
            update={"cumulative_category_impact_limits": {"climate_change": budget}}
        )
        for budget in budgets
    },
    base=model_inputs,
    objective="cost",
    solver_name="highs",  # or "gurobi"
)
sweep.objective      # one value per variant, indexed by the dict keys
sweep.status         # "optimal", or why a variant has no solution
sweep.impacts        # cumulative impact per category
sweep.costs          # discounted total cost, with prices
sweep.installation   # installed units, indexed by (variant, year)
```

With HiGHS or Gurobi, `solve_series` builds the model of `base` once and
assembles every variant from it with NumPy, then loads it straight into the
solver. This applies to variants that differ in exchange amounts, background
inventories, characterization factors, prices, discounting, demand, impact
limits, or deployment and operation limits; other variants are built in full
automatically. With other solvers, each variant is built and solved on its own,
and a warning says so.

| Option | Effect |
|--------|--------|
| `base` | The inputs the variants were made from; defaults to the first variant |
| `outputs` | PostProcessor tables to keep: `"installation"` (default), `"operation"`, `"production"` |
| `extract` | A function `extract(model)` whose result is kept per variant |
| `keep_models` | Keep every solved model, e.g. for a full PostProcessor analysis |
| `n_jobs` | Solve variants in parallel processes (`-1`: all cores) |
| `persistent` | `"auto"` (default), `True` (fail instead of falling back), `False` (always build full models) |

!!! tip "Make variants with `model_copy`"
    `model_inputs.model_copy(update={...})` creates a variant that shares
    everything else with `model_inputs`, which is what lets `solve_series`
    skip the rebuild. `manager.override(...)` works too, but validates and
    copies all inputs for every variant.

The fast path needs full models only for what cannot be computed from the
variables: the `"production"` table, `extract`, and `keep_models` build each
variant's model in full (still solved in one solver session). Monte Carlo
analyses use the same machinery through
[`uncertainty.solve_samples`](uncertainty.md).

---

## Next Steps

- [Trade-offs](trade_offs.md): sweep an impact budget to trace a Pareto frontier
- [Uncertainty](uncertainty.md): Monte Carlo analysis of the uncertain data
- [Comparing Scenarios](postprocessing.md#comparing-scenarios): analyze the solved scenarios
