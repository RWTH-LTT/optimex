---
icon: lucide/scan
tags:
  - background
  - methodology
---

# Scope and Limitations

## Dynamic Characterization

`optimex` supports both static and dynamic Life Cycle Impact Assessment (LCIA). For impacts where timing matters — particularly climate change — dynamic metrics like Cumulative Radiative Forcing (CRF) or dynamic Global Warming Potential (GWP) capture how different greenhouse gases accumulate and decay in the atmosphere over time. Static characterization (e.g., GWP100) can still be used for impact categories where timing is less critical.

## Economic Costs

The cost objective prices only the background products the foreground buys directly, at given market prices; costs are not propagated through background supply chains, and prices do not react to the decisions taken. Costs scale linearly with installed and operated units, so economies of scale and indivisible plant sizes are not represented. See [Cost Modeling](cost_modeling.md#modeling-assumptions-and-limits).

## Uncertainty

A single optimization uses deterministic data. The uncertainty of exchanges,
parameters, background databases, and prices, as described in Brightway, is
propagated by Monte Carlo sampling: every sample is optimized, or a given
pathway is evaluated, on its own data ([Uncertainty](uncertainty.md)). This
shows how often a technology is optimal and how robust a pathway is; it does not
compute a single pathway that is optimal under uncertainty, such as one that
minimizes expected impact plus a risk penalty. Characterization factors,
temporal distributions, and demand are not sampled.

## Data Requirements

Time-explicit LCO requires data on temporal distributions of processes, temporal evolution of production systems, and (optionally) time-specific characterization factors. When temporal information is unavailable, `optimex` defaults to static values, allowing analyses to proceed with partial temporal coverage.
