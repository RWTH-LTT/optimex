# `optimex` Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
* Added a choice of modeling backend: `create_model(..., backend="pyomo")`
  (default) builds a Pyomo `ConcreteModel` as before, `backend="linopy"` builds
  the same model from arrays with [linopy](https://linopy.readthedocs.io)
  (`pip install optimex[linopy]`), which is faster for large systems. Both are
  built from one backend-independent formulation (`optimex.formulation`) and
  live in their own modules (`optimex.pyomo_backend`,
  `optimex.linopy_backend`). `solve_model()` solves either and stores the
  solution under the same names, so the `PostProcessor` and
  `save_solved_model()` work with both
  ([#74](https://github.com/RWTH-LTT/optimex/pull/74))
* Added `create_model(..., aggregate_vintages=True)`: processes whose vintages
  operate identically get one operation variable per year
  (`var_operation_total`) and a capacity stock carried forward year by year
  (`var_capacity_total`) instead of one operation variable per vintage and year.
  The optimum is the same with far fewer variables and constraints. These
  processes have no `var_operation[p, v, t]`; read per-vintage operation with
  the new `optimizer.operation_value()`. Off by default
  ([#74](https://github.com/RWTH-LTT/optimex/pull/74))
* Model construction no longer slows down disproportionately for large systems:
  coupling constraints are only built for coupled pairs, product balances only
  visit the processes that produce or consume the product, and all expressions
  are flat linear expressions instead of nested ones. Building a hydrogen trade
  system with 816 processes went from 195 s to 11 s at the same optimum
  ([#74](https://github.com/RWTH-LTT/optimex/pull/74))
* `PostProcessor.get_production()`, `get_operation()`,
  `get_production_capacity()`, `plot_capacity_balance()` and
  `plot_utilization_heatmap()` no longer scan all active vintages for every
  process and year, which was quadratic in the model size (`get_production()`
  for 216 processes: 71 s to 0.1 s). Their results are unchanged
  ([#74](https://github.com/RWTH-LTT/optimex/pull/74))
* The objective is normalized by its largest coefficient, which kept impact
  coefficients from falling below the solvers' optimality tolerances.
  `solve_model()` still returns the objective in real units, and the factor is
  stored as `model.scales["objective"]`
  ([#74](https://github.com/RWTH-LTT/optimex/pull/74))
* Changed model components ([#74](https://github.com/RWTH-LTT/optimex/pull/74)):
  * Deployment limits are bounds of `var_installation` instead of the
    `ProcessDeploymentLimitMax`/`Min` constraints, which are removed.
    Operation, cumulative and cumulative impact limits that cannot bind (infinite
    maximum, non-positive minimum) no longer get a constraint. This also fixes
    `appsi_gurobi`, which rejected constraints against infinite limits
  * `ProcessCouplingConstraint`, `FlowLimitMax`/`Min`,
    `CumulativeFlowLimitMax`/`Min` and `CategoryImpactLimits` are indexed only
    over the limited or coupled entries
  * `scaled_technosphere_flow`, `scaled_biosphere_flow`,
    `scaled_internal_demand_flow` and `specific_impact` are stored as
    coefficients and only become Pyomo expressions when indexed, and
    `scaled_inventory`, `total_elementary_flow` and `total_intermediate_flow` are
    built on access. Indexing them and `pyo.value()` work as before; they are no
    longer Pyomo `Expression` components. `optimizer.expression_value()` reads
    their solved values
  * Variables that appear in no constraint get their lower bound (or 0) after
    solving instead of no value
* `OptimizationModelInputs.get_scaled_copy()` copies the inputs one level deep
  instead of deep-copying them ([#74](https://github.com/RWTH-LTT/optimex/pull/74))

## [0.8.0] - 2026-11-26
* Added a cost objective: `create_model(..., objective="cost")` minimizes the
  discounted economic cost of the background products the foreground buys, instead
  of an impact category. Impact limits still apply, so the cheapest pathway within a
  carbon budget, and how its cost rises as the budget tightens, can be traced.
  Market prices are written to background nodes with
  `optimex.economics.set_market_prices()`, read by setting
  `LCAConfig.price_attribute`, and discounted with `discount_rate` and
  `discount_reference_year`. `objective_category` is now optional
  ([#72](https://github.com/RWTH-LTT/optimex/pull/72), based on
  [#71](https://github.com/RWTH-LTT/optimex/pull/71) by Luca Lange)
* Added `PostProcessor.get_costs()`, which reports costs per year by process and by
  installation- or operation-driven purchases for any solved model with prices, also
  when it minimized an impact ([#72](https://github.com/RWTH-LTT/optimex/pull/72))
* Fixed the interpolation of vintage-dependent parameters. Each exchange is now
  interpolated between its own vintage years; before, a vintage year defined only on
  another exchange silently became an amount of 0 (`vintage_amounts`) or a factor of
  1 (`vintage_improvements`). `vintage_amounts` now take precedence over
  `vintage_improvements` per exchange instead of for all exchanges of the same kind.
  Optimization results change for processes whose exchanges use different vintage
  years, or that combine both kinds of parameters
  ([#72](https://github.com/RWTH-LTT/optimex/pull/72))
* A vintage-year key in `vintage_amounts` (e.g. `{2020: 60}`) is now the exchange
  total for that vintage, distributed over process time like `amount`, instead of
  the amount at every process time. Results change for such keys on exchanges that
  span more than one process time; `(process_time, vintage_year)` keys are unchanged
  ([#72](https://github.com/RWTH-LTT/optimex/pull/72))
* Fixed `ModelInputManager.extend_demand()`, which failed with an `IndexError`
  ([#72](https://github.com/RWTH-LTT/optimex/pull/72))
* Fixed a race in the parallel background inventory calculation: the Brightway
  database registry is cleaned once before the workers start, so they no longer
  write it concurrently ([#69](https://github.com/RWTH-LTT/optimex/pull/69))
* Restructured the documentation along the workflow (choosing an objective, product
  system modeling, optimization, results), with new pages on minimizing impacts,
  minimizing costs, cost modeling, trade-offs, and solving, a basic cost example,
  and fixes across the guide found in a newcomer review. Old page URLs redirect to
  the new ones ([#72](https://github.com/RWTH-LTT/optimex/pull/72))
* Updated the citation to the published paper in *Environmental Science &
  Technology* ([#70](https://github.com/RWTH-LTT/optimex/pull/70))

## [0.7.0] - 2026-08-20
* Fixed under-counting of installation impacts. One installed unit now delivers its
  production temporal distribution over its whole lifetime instead of in every
  operating year, so a process with an n-year operation window no longer amortizes
  its construction and end-of-life impacts over n times the output it actually
  delivers. Optimization results change for any process whose operation window spans
  more than one time step ([#61](https://github.com/RWTH-LTT/optimex/pull/61))
* `var_installation` and `var_operation` both count process units now, so
  `OperationCapacity` compares them directly. `existing_capacity` and the deployment
  and operation limits are in the same units: values calibrated as an annual capacity
  need multiplying by the number of operating years
  ([#61](https://github.com/RWTH-LTT/optimex/pull/61))
* Postprocessing capacity outputs (`get_production_capacity`, `plot_capacity_balance`,
  `plot_utilization_heatmap`) now report annual capacity, directly comparable with
  production. `get_installation` reports installed units, a lifetime quantity
  ([#61](https://github.com/RWTH-LTT/optimex/pull/61))

## [0.6.0] - 2026-08-20
* Sped up the whole pipeline a lot: the `notebooks/methanol_and_iron.ipynb` case
  study goes from 613 s to 44 s end to end
  ([#64](https://github.com/RWTH-LTT/optimex/pull/64))
* **Changed results**: `background_inventory.cutoff` now defaults to `None`. The old
  default truncated background inventories before aggregating them, biasing flow
  amounts low by up to 30% ([#64](https://github.com/RWTH-LTT/optimex/pull/64))
* Background databases are now calculated in parallel by default, and elementary
  flows without a characterization factor are dropped from the optimization model.
  See `background_inventory.retain_flows` and `restrict_to_characterized_flows`
  ([#64](https://github.com/RWTH-LTT/optimex/pull/64))
* Building and solving the model is much faster too: on the same case study, model
  build 103.6 s to 1.7 s, solve wall clock 55.7 s to 0.1 s and
  `PostProcessor.get_dynamic_inventory()` 19.8 s to 0.2 s, with identical results
  ([#64](https://github.com/RWTH-LTT/optimex/pull/64))
* `get_dynamic_inventory()` no longer returns rows whose amount is exactly zero
  ([#64](https://github.com/RWTH-LTT/optimex/pull/64))
* Flow limits that cannot take effect are now reported: naming a flow that is not in
  the model raises in `ModelInputManager.override()` (pointing at `retain_flows`), and
  `create_model()` warns about limits it has to ignore or that can never bind
  ([#64](https://github.com/RWTH-LTT/optimex/pull/64))
* Background inventories are cached on disk per (project, database, activity), so a
  new session reuses them instead of rebuilding every technosphere matrix. Switch it
  off with `background_inventory.use_disk_cache` and clear it with
  `lca_processor.clear_lca_caches(include_disk=True)`
  ([#64](https://github.com/RWTH-LTT/optimex/pull/64))

## [0.5.0] - 2026-08-18
* Added foreground_db_name argument to LCAConfig
* Fixed `LCAConfig.foreground_db_name` being ignored by `LCADataProcessor`, which
  silently fell back to the database named "foreground". A config naming a
  non-existent database now raises instead of falling back.

## [0.4.2] - 2026-04-27
* Fix conda publish workflow

## [0.4.1] - 2026-04-27
* Convert to src package layout and use absolute imports

## [0.4.0] - 2026-04-21
* Added Vintage-dependent foreground parameters: Model how process characteristics change based on installation year (vintage). Supports two approaches:
  - Explicit values per vintage via `foreground_*_vintages` fields
  - Scaling factors via `vintage_improvements` field

## [0.3.0] - 2025-07-04
* Fixed an issue with process installation scaling

## [0.2.0] - 2025-05-28
* Introduced automatic testing
* Differentiation between capacity installment and actual operation
* Improved user-facing API

## [0.1.0] - 2025-02-27
* Initial release.
