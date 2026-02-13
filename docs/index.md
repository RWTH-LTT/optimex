# Time-Explicit Life Cycle Optimization

Transition pathway optimization typically collapses all life cycle stages and emissions to a single point in time, hiding critical temporal interdependencies: life cycles are distributed across years or decades (*temporal distribution*), and the production systems behind them are evolving (*temporal evolution*). Ignoring this interplay means pathways may appear sustainable while violating time-specific limits during intensive infrastructure buildup, or miscounting cumulative impacts when decommissioning happens in a fundamentally different future.

`optimex` jointly models both dimensions — *when* exchanges occur and *how* their magnitudes change over time — to design pathways that respect time-specific and cumulative environmental constraints.

<p class="home-links">
<a href="content/quickstart" class="primary">Get Started<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-rocket"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg></a>
<a href="https://github.com/RWTH-LTT/optimex">View on GitHub<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-github"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg></a>
</p>

---

## Key Capabilities

<div class="grid cards" markdown>

-   :lucide-hourglass:{ style="color: #4dabf7" } <span style="color: var(--md-primary-bg-color); font-weight: 700;">Temporal Distribution</span>

    ---

    Maps life cycle exchanges across their actual timeframes via convolution, capturing time lags between construction, operation, and end-of-life.

-   :lucide-settings:{ style="color: #69db7c" } <span style="color: var(--md-primary-bg-color); font-weight: 700;">Technology Evolution</span>

    ---

    Tracks vintage-dependent foreground improvements and links to prospective background databases reflecting supply chain decarbonization.

-   :lucide-arrow-up-down:{ style="color: #ffa94d" } <span style="color: var(--md-primary-bg-color); font-weight: 700;">Flexible Operation</span>

    ---

    Separates capacity installation from operational dispatch, enabling vintage-specific merit order where cleaner cohorts are utilized first.

-   :lucide-trending-up:{ style="color: #da77f2" } <span style="color: var(--md-primary-bg-color); font-weight: 700;">Dynamic Characterization</span>

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
