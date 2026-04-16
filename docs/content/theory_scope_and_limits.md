---
icon: lucide/scan
tags:
  - background
  - methodology
---

# Scope and Limitations

## Single-Tier Convolution

Unlike dynamic LCA approaches that propagate temporal distributions through entire multi-tier supply chains, `optimex` applies convolution only at the foreground tier. Upstream supply chains in the background are treated as temporally aggregated at the time of the direct exchange. This simplification is necessary because multi-tier convolution creates a circular dependency: upstream timings depend on installation timing, which is the decision variable being optimized.

**Workarounds** where finer upstream resolution is needed:

- Move critical upstream processes into the foreground, allowing explicit temporal distributions on their exchanges.
- Pre-compute time-resolved emission profiles using dynamic LCA and map them as temporal distributions on elementary flows.

## Dynamic Characterization

`optimex` supports both static and dynamic Life Cycle Impact Assessment (LCIA). For impacts where timing matters — particularly climate change — dynamic metrics like Cumulative Radiative Forcing (CRF) or dynamic Global Warming Potential (GWP) capture how different greenhouse gases accumulate and decay in the atmosphere over time. Static characterization (e.g., GWP100) can still be used for impact categories where timing is less critical.

## Data Requirements

Time-explicit LCO requires data on temporal distributions of processes, temporal evolution of production systems, and (optionally) time-specific characterization factors. When temporal information is unavailable, `optimex` defaults to static values, allowing analyses to proceed with partial temporal coverage.
