# `optimex` Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
* Added `create_model(..., backend="linopy")`, which builds the model with linopy
  instead of Pyomo (`pip install optimex[linopy]`) and is faster for large
  systems. `solve_model()` and the `PostProcessor` work with both backends
  ([#74](https://github.com/RWTH-LTT/optimex/pull/74))
* Added `create_model(..., aggregate_vintages=True)`, which gives processes with
  identically operating vintages one operation variable per year instead of one
  per vintage. Same optimum, much smaller model; read per-vintage operation with
  `optimizer.operation_value()` ([#74](https://github.com/RWTH-LTT/optimex/pull/74))
* Model construction and `PostProcessor` methods no longer slow down
  disproportionately with the number of processes; results are unchanged
  ([#74](https://github.com/RWTH-LTT/optimex/pull/74))
* Deployment limits are variable bounds instead of the removed
  `ProcessDeploymentLimitMax`/`Min` constraints, limits that cannot bind create
  no constraint, and the objective is normalized by its largest coefficient
  (`model.scales["objective"]`) ([#74](https://github.com/RWTH-LTT/optimex/pull/74))
* The per-process flow and impact expressions (`scaled_*_flow`,
  `specific_impact`, `scaled_inventory`, ...) are no longer Pyomo `Expression`
  components; indexing them and `pyo.value()` work as before
  ([#74](https://github.com/RWTH-LTT/optimex/pull/74))
* Added `optimex.uncertainty` for Monte Carlo analysis: `UncertaintySampler`
  draws optimization inputs from the uncertainty of foreground exchanges,
  Brightway parameters, background databases, and prices; `solve_samples()`
  optimizes every sample, `evaluate_design()` evaluates a fixed design in every
  sample. New example: `notebooks/uncertainty_example.ipynb` ([#76](https://github.com/RWTH-LTT/optimex/pull/76))
* `set_market_prices()` reads price distributions from the Brightway uncertainty
  columns (`uncertainty type`, `loc`, `scale`, ...) ([#76](https://github.com/RWTH-LTT/optimex/pull/76))
* Added `optimizer.solve_series()` for sweeps, Pareto frontiers, and scenario
  comparisons: it solves a dict or list of input variants and returns one table
  of objectives, impacts, costs, and installations ([#76](https://github.com/RWTH-LTT/optimex/pull/76))
* With Gurobi or HiGHS, `solve_series()` and the Monte Carlo functions build the
  base model once and assemble each variant from it, about 45 times faster than
  a rebuild for large models; `n_jobs` solves variants in parallel ([#76](https://github.com/RWTH-LTT/optimex/pull/76))
* New documentation pages on uncertainty and on solving series of models; the
  trade-off, scenario, and Pareto examples use `solve_series()` ([#76](https://github.com/RWTH-LTT/optimex/pull/76))

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
