# `optimex` Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
* Sped up LCA processing a lot: the `notebooks/methanol_and_iron.ipynb` case study
  goes from 613 s to 85 s end to end ([#64](https://github.com/RWTH-LTT/optimex/pull/64))
* **Changed results**: `background_inventory.cutoff` now defaults to `None`. The old
  default truncated background inventories before aggregating them, biasing flow
  amounts low by up to 30% ([#64](https://github.com/RWTH-LTT/optimex/pull/64))
* Background databases are now calculated in parallel by default, and elementary
  flows without a characterization factor are dropped from the optimization model.
  See `background_inventory.retain_flows` and `restrict_to_characterized_flows`
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
