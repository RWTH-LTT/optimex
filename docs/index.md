# Time-explicit Life Cycle Optimization with `optimex`

This is a Python package for time-explicit Life Cycle Optimization for transition pathways. `optimex` helps identify optimal process portfolios and deployment timings in systems with multiple processes producing the same product, aiming to minimize dynamically accumulating environmental impacts over time.

`optimex` builds on top of the optimization framework [pyomo](https://github.com/Pyomo/pyomo) and the LCA framework [Brightway](https://docs.brightway.dev/en/latest). If you are looking for a time-explicit LCA rather than an optimization tool, make sure to check out [`bw_timex`](https://docs.brightway.dev/projects/bw-timex/en/latest/).

<p class="home-links">
<a href="content/quickstart" class="primary">Get Started<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-rocket"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg></a>
<a href="https://github.com/RWTH-LTT/optimex">View on GitHub<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-github"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg></a>
</p>

---

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
