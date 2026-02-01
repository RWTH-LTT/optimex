# `optimex` Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
* **Flexible temporal resolution**: Support for `"year"`, `"month"`, and `"day"` resolutions via `temporal_resolution` config
  - Monthly resolution enables seasonal modeling (e.g., renewable intermittency, water scarcity)
  - Daily resolution for fine-grained operations and storage optimization
  - Automatic conversion between resolutions (yearly→monthly→daily)
* **Mixed temporal resolutions**: Processes can use different resolutions in the same model
  - `detect_temporal_resolution()` infers resolution from TemporalDistribution values
  - Coarser resolutions automatically expanded to match configured resolution
* **User-provided characterization factors**: New `characterization_factors` field for custom time-varying LCIA
  - Maps `(flow_code, time_index)` tuples directly to characterization factor values
  - No Brightway method required - full flexibility for seasonal/regionalized factors
  - Example: seasonal water scarcity factors varying by month
* **Vintage-dependent foreground parameters**: Model how process characteristics change based on installation year (vintage). Supports two approaches:
  - Explicit values per vintage via `foreground_*_vintages` fields
  - Scaling factors via `technology_evolution` field
* Linear interpolation between reference vintages for intermediate installation years
* Full backward compatibility - existing models without vintage parameters work unchanged

## [0.3.0] - 2025-07-04
* Fixed an issue with process installation scaling

## [0.2.0] - 2025-05-28
* Introduced automatic testing
* Differentiation between capacity installment and actual operation
* Improved user-facing API

## [0.1.0] - 2025-02-27
* Initial release. 
