# `optimex` Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
* **Vintage-dependent foreground parameters**: Model how process characteristics change based on installation year (vintage). Supports two approaches:
  - Explicit values per vintage via `foreground_*_vintages` fields
  - Scaling factors via `vintage_improvements` field
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
