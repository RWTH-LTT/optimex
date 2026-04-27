# `optimex` Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
