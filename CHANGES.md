# `optimex` Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
* **Changed results**: fixed under-counting of installation impacts. One installed
  unit now delivers its production temporal distribution over its whole lifetime
  instead of in every operating year, so a process with an n-year operation window
  no longer amortizes its construction and end-of-life impacts over n times the
  output it actually delivers
  ([#61](https://github.com/RWTH-LTT/optimex/pull/61))
* `var_installation` and `var_operation` are both unit counts now, so
  `OperationCapacity` compares them directly. `existing_capacity` and the deployment
  and operation limits are in the same process units: values calibrated as an annual
  capacity need multiplying by the number of operating years
  ([#61](https://github.com/RWTH-LTT/optimex/pull/61))
* Postprocessing capacity outputs (`get_production_capacity`,
  `plot_capacity_balance`, `plot_utilization_heatmap`) now report ANNUAL capacity,
  comparable with production; `get_installation` reports installed units, a lifetime
  quantity ([#61](https://github.com/RWTH-LTT/optimex/pull/61))

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
