---
icon: lucide/bar-chart-3
---

# Postprocessing

Post-processing and visualization of optimization results.

This module provides tools to extract, process, and visualize results from solved optimization models. The `PostProcessor` class handles denormalization of scaled results, data extraction into DataFrames, and creation of publication-quality plots.

## Key Classes

- **`PostProcessor`**: Extract and visualize optimization results

## Available Methods

### Data Extraction

- `get_impacts()`: Extract impact results as DataFrame
- `get_installation()`: Extract installation schedules
- `get_operation()`: Extract operation profiles
- `get_production()`: Extract production quantities
- `get_demand()`: Extract demand values

### Visualization

- `plot_impacts()`: Stacked bar chart of impacts over time
- `plot_installation()`: Installation schedule visualization
- `plot_production_and_demand()`: Production vs demand comparison

## Module Reference

::: optimex.postprocessing
    options:
      show_root_heading: false
      show_root_toc_entry: false
