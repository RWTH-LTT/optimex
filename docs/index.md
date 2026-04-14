---
title: optimex
icon: lucide/house
---

# Time-Explicit Life Cycle Optimization

Transition pathway optimization typically collapses all life cycle stages and emissions to a single point in time, hiding critical temporal interdependencies: life cycles are distributed across years or decades (*temporal distribution*), and the production systems behind them are evolving (*temporal evolution*). Ignoring this interplay means pathways may appear sustainable while violating time-specific limits during intensive infrastructure buildup, or miscounting cumulative impacts when decommissioning happens in a fundamentally different future.

`optimex` jointly models both dimensions — *when* exchanges occur and *how* their magnitudes change over time — to design pathways that respect time-specific and cumulative environmental constraints.

[Get Started](content/quickstart.md){ .md-button .md-button--primary } [View on GitHub](https://github.com/RWTH-LTT/optimex){ .md-button }

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

## Use Cases

`optimex` is broadly applicable across sectors where temporal dynamics are decisive for sustainability:

- **Evolving supply chains** — Systems depending on electricity, steel, or hydrogen that undergo rapid decarbonization
- **Early-stage technologies** — Processes with significant vintage-dependent performance improvements (e.g., electrolyzers, DAC)
- **Circular economy planning** — Long material residence times create temporal mismatches between primary demand and secondary supply
- **Time-resolved carbon accounting** — Biogenic feedstocks, temporary carbon storage, or CO~2~ removal with varying temporal profiles
- **Multi-regional supply chains** — Sourcing decisions across regions with divergent decarbonization trajectories

---

## Built On

`optimex` integrates with established open-source tools:

- [Pyomo](https://github.com/Pyomo/pyomo) for mathematical optimization
- [Brightway](https://docs.brightway.dev/en/latest) for life cycle assessment
- [bw_temporalis](https://github.com/brightway-lca/bw_temporalis) for temporal distributions
- [dynamic_characterization](https://github.com/brightway-lca/dynamic_characterization) for dynamic impact assessment
- [premise](https://github.com/polca/premise) for prospective background databases

For time-explicit LCA without optimization, see [`bw_timex`](https://docs.brightway.dev/projects/bw-timex/en/latest/).

`optimex` is free and open source software, published under the [BSD 3-Clause License](content/license.md).

## Support

If you have any questions or need help, do not hesitate to contact us:

- Timo Diepers ([timo.diepers@ltt.rwth-aachen.de](mailto:timo.diepers@ltt.rwth-aachen.de))
- Jan Tautorus ([jan.tautorus@rwth-aachen.de](mailto:jan.tautorus@rwth-aachen.de))
