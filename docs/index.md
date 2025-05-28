# Time-explicit Transition Pathway Optimization with `optimex`

This is a Python package for transition pathway optimization based on time-explicit Life Cycle Assessment (LCA). `optimex` helps identify optimal process portfolios and deployment timing in systems with multiple processes producing the same product, aiming to minimize dynamically accumulating environmental impacts over time. 

`optimex` builds on top of the [Brightway LCA framework](https://docs.brightway.dev/en/latest). If you are looking for a time-explicit assessment rather than an optimization tool, make sure to check out our time-explicit LCA package [`bw_timex`](https://docs.brightway.dev/projects/bw-timex/en/latest/).

## ✨ Features
This package enables you to:
- Optimize the **timing and scale** of process deployments over a transition period
- Jointly consider the **temporal distribution and evolution** of processes (e.g., electricity consumption over a 20-year use phase dynamically chooses the appropriate electricity mix based on the actual time of consumption)
- Account for the **timing and accumulation of emissions** using dynamic Life Cycle Impact Assessment

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
