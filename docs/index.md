# Time-explicit Transition Pathway Optimization with `optimex`

This is a Python package for transition pathway optimization based on time-explicit Life Cycle Assessment (LCA). `optimex` helps identify optimal process portfolios and deployment timing in systems with multiple processes producing the same product, aiming to minimize dynamically accumulating environmental impacts over time. 

`optimex` builds on top of the optimization framework [pyomo](https://github.com/Pyomo/pyomo) and the LCA framework [Brightway](https://docs.brightway.dev/en/latest). If you are looking for a time-explicit LCA rather than an optimization tool, make sure to check out [`bw_timex`](https://docs.brightway.dev/projects/bw-timex/en/latest/).

## ✨ Features

Like other transition pathway optimization tools, `optimex` identifies the optimal timing and scale of process deployments to minimize environmental impacts over a transition period. What sets `optimex` apart is its integration of three additional, temporal considerations for environmental impacts:

1. **Timing within Process Life Cycles:** Environmental impacts are spread across a process’s life cycle: construction happens first, use comes later, and end-of-life impacts follow. `optimex` captures this by distributing process inputs and outputs over time.

2. **Technology Evolution:** Future technologies may become more sustainable, reducing the environmental impacts later in the expansion period. `optimex` reflects this by allowing process inventories to evolve over time.

3.	**Accumulation of Emissions and Impacts:** Most impacts arise from the accumulation of emissions, but are typically modeled as discrete and independent pulses. `optimex` retains the timing of emissions during inventory calculations and applies dynamic characterization to account for impact accumulation

During the transition pathway optimization, `optimex` simultaneously accounts for these temporal considerations, identifying the environmentally optimal process deployment over the transition period.

## 💬 Support

If you have any questions or need help, do not hesitate to contact us:
- Timo Diepers ([timo.diepers@ltt.rwth-aachen.de](mailto:timo.diepers@ltt.rwth-aachen.de))
- Jan Tautorus ([jan.tautorus@rwth-aachen.de](mailto:jan.tautorus@rwth-aachen.de))

```{toctree}
---
hidden:
maxdepth: 1
---
Installation <content/installation>
Getting Started <content/getting_started>
Theory <content/theory>
Examples <content/examples/index>
API <content/api/index>
Code of Conduct <content/codeofconduct>
Contributing <content/contributing>
License <content/license>
Changelog <content/changelog>
```
