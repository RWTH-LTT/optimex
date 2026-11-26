---
icon: lucide/badge-euro
tags:
  - economics
  - costs
---

# Economics

Helpers for the cost objective.

This module writes market prices to the time-specific background databases. The
`LCADataProcessor` reads them when `LCAConfig.price_attribute` is set, and the
optimizer prices the background purchases with them. See
[Minimizing Costs](../content/objective_cost.md) for the workflow.

## Key Functions

- **`set_market_prices()`**: Writes market prices from a table to background nodes

## Module Reference

::: optimex.economics
    options:
      show_root_heading: false
      show_root_toc_entry: false
