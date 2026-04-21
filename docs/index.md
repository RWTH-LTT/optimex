---
title: Documentation
icon: lucide/house
tags:
  - introduction
  - getting started
---

# Time-Explicit Life Cycle Optimization with optimex

`optimex` is an open-source Python package for **time-explicit Life Cycle Optimization (LCO)**. It finds optimal technology transition pathways while accounting for *when* emissions occur and *how* product systems evolve over time.

Standard LCA evaluates predefined product systems. But transition planning asks a different question: *which technologies should be deployed, when, and at what scale*? That is the domain of Life Cycle Optimization. In fast-changing energy and industrial systems, static formulations can be misleading because construction, operation, and end-of-life happen at different times, while background supply chains and technology performance evolve in parallel.

`optimex` closes this gap by combining temporalized LCA with optimization in one workflow, so pathways can be assessed against both time-specific and cumulative environmental constraints.

[:lucide-rocket: Get Started](content/quickstart.md){ .md-button .md-button--primary } [:fontawesome-brands-github: View on GitHub](https://github.com/RWTH-LTT/optimex){ .md-button }

---

## Why Go Time-Explicit?

Making optimization time-explicit unlocks insights that static approaches miss:

- **Temporal distribution of flows** — Construction, operation, and end-of-life occur over years or decades and should be modeled where they actually happen in time
- **Temporal evolution of technologies** — Process performance improves over time; `optimex` tracks vintages so technologies are evaluated with the parameters available at their installation date
- **Time-varying background systems** — Foreground demands are linked to time-specific background databases so future activity is assessed against future supply chains
- **Flexible operation and dispatch** — Installation and operation are separate decisions, enabling vintage-specific utilization where cleaner cohorts are preferred
- **Dynamic impact assessment** — Time-dependent characterization (e.g., radiative forcing and dynamic climate metrics) captures how impacts accumulate over the planning horizon

---

## Key Capabilities

<div class="grid cards" markdown>

-   :lucide-hourglass:{ style="color: #4dabf7" } **Temporal Distribution**

    ---

    Maps life cycle exchanges across their actual timeframes via convolution, capturing time lags between construction, operation, and end-of-life.

-   :lucide-settings:{ style="color: #69db7c" } **Technology Evolution**

    ---

    Tracks vintage-dependent foreground improvements and links to prospective background databases reflecting supply chain decarbonization.

-   :lucide-arrow-up-down:{ style="color: #ffa94d" } **Flexible Operation**

    ---

    Separates capacity installation from operational dispatch, enabling vintage-specific merit order where cleaner cohorts are utilized first.

-   :lucide-trending-up:{ style="color: #da77f2" } **Dynamic Characterization**

    ---

    Retains emission timing for dynamic LCIA (e.g., Radiative Forcing, dynamic GWP), capturing how impacts accumulate over time.

</div>

---

## What This Enables

Time-explicit LCO reveals transition strategies that are invisible to static approaches:

- **Strategic overcapacity** — Early investment in clean technologies can offset stranded fossil assets when the net emission savings outweigh the embodied impacts of idle infrastructure
- **Vintage-specific dispatch** — When multiple cohorts of the same technology coexist, the optimizer preferentially utilizes cleaner vintages, creating an emissions-aware merit order
- **Resource bottleneck navigation** — Time-specific constraints on water use, critical minerals, or other resources force technology diversification, revealing realistic pathways through transient scarcity
- **Cumulative budget compliance** — By tracking exact emission timing alongside dynamic characterization, pathways can be verified against carbon budgets and other absolute limits

---

## Built on Brightway

`optimex` is deeply integrated with [Brightway](https://brightway.dev). You can model foreground systems with familiar Brightway workflows, add temporal metadata, and convert those models directly into optimization problems.

This means:

- **No lock-in** — Use Brightway-compatible databases, including custom inventories
- **Familiar modeling workflow** — Build and maintain product systems with tools you already use
- **Reuse existing models** — Temporalize and optimize product systems created for conventional LCA

For optimization, `optimex` uses [Pyomo](https://www.pyomo.org), a powerful open-source algebraic modeling framework.

For time-explicit LCA without optimization, see [`bw_timex`](https://docs.brightway.dev/projects/bw-timex/en/latest/).

`optimex` is free and open source software, published under the [BSD 3-Clause License](content/license.md).

---

## Use Cases

`optimex` is broadly applicable across sectors where temporal dynamics are decisive for sustainability:

- **Evolving supply chains** — Systems depending on electricity, steel, or hydrogen that undergo rapid decarbonization
- **Early-stage technologies** — Processes with significant vintage-dependent performance improvements (e.g., electrolyzers, DAC)
- **Circular economy planning** — Long material residence times create temporal mismatches between primary demand and secondary supply
- **Time-resolved carbon accounting** — Biogenic feedstocks, temporary carbon storage, or CO~2~ removal with varying temporal profiles
- **Multi-regional supply chains** — Sourcing decisions across regions with divergent decarbonization trajectories

---

## Support

If you have any questions or need help, do not hesitate to contact us:

- Timo Diepers ([timo.diepers@ltt.rwth-aachen.de](mailto:timo.diepers@ltt.rwth-aachen.de))
- Jan Tautorus ([jan.tautorus@rwth-aachen.de](mailto:jan.tautorus@rwth-aachen.de))
