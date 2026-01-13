# Time-explicit Life Cycle Optimization with `optimex`

This is a Python package for time-explicit Life Cycle Optimization for transition pathways. `optimex` helps identify optimal process portfolios and deployment timings in systems with multiple processes producing the same product, aiming to minimize dynamically accumulating environmental impacts over time.

`optimex` builds on top of the optimization framework [pyomo](https://github.com/Pyomo/pyomo) and the LCA framework [Brightway](https://docs.brightway.dev/en/latest). If you are looking for a time-explicit LCA rather than an optimization tool, make sure to check out [`bw_timex`](https://docs.brightway.dev/projects/bw-timex/en/latest/).

<p class="home-links">
<a href="content/installation" class="primary"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg> Installation</a>
<a href="content/quickstart"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg> Quick Start</a>
</p>

## Features

Like other transition pathway optimization tools, `optimex` identifies the optimal timing and scale of process deployments to minimize environmental impacts over a transition period. What sets `optimex` apart is its integration of three additional, temporal considerations for environmental impacts:

<div class="grid cards" markdown>

-   :lucide-hourglass:{ style="color: #4dabf7" } **Life Cycle Timing**

    ---

    The processes within a product's life cycle occur in sequence rather than all at once: production comes first, the use phase and end-of-life follow. As a result, emissions are also spread over time.

    `optimex` captures this by distributing process inputs and outputs over time.

-   :lucide-settings:{ style="color: #69db7c" } **Technology Evolution**

    ---

    In the future, processes will (hopefully) reduce their emissions. So, the time at which a process happens affects its impacts, with later occurence often resulting in lower emissions.

    `optimex` reflects this by allowing process inventories to change over time.

-   :lucide-trending-up:{ style="color: #ffa94d" } **Emission Accumulation**

    ---

    Most impacts come from emission accumulation, but are typically modeled as separate pulses. Emission timing affects how they accumulate, influencing environmental impacts.

    `optimex` considers the emission timing, enabling characterization via dynamic LCIA.

</div>

During the transition pathway optimization, `optimex` simultaneously accounts for these temporal considerations, identifying the environmentally optimal process deployment over the transition period.

## Support

If you have any questions or need help, do not hesitate to contact us:

- Timo Diepers ([timo.diepers@ltt.rwth-aachen.de](mailto:timo.diepers@ltt.rwth-aachen.de))
- Jan Tautorus ([jan.tautorus@rwth-aachen.de](mailto:jan.tautorus@rwth-aachen.de))
