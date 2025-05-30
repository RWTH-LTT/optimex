<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/optimex_dark_nomargins.svg" height="50">
    <img alt="optimex logo" src="docs/_static/optimex_light_nomargins.svg" height="50">
  </picture>
</h1>

[![Read the Docs](https://img.shields.io/readthedocs/optimex?label=documentation)](https://optimex.readthedocs.io/)
[![PyPI - Version](https://img.shields.io/pypi/v/optimex?color=%2300549f)](https://pypi.org/project/optimex/)
[![Conda Version](https://img.shields.io/conda/v/diepers/optimex?label=conda)](https://anaconda.org/diepers/optimex)
[![Conda - License](https://img.shields.io/conda/l/diepers/optimex)](https://github.com/TimoDiepers/optimex/blob/main/LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TimoDiepers/optimex/main?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2Fbasic_optimex_example.ipynb)

This is a Python package for transition pathway optimization based on time-explicit Life Cycle Assessment (LCA). `optimex` helps identify optimal process portfolios and deployment timing in systems with multiple processes producing the same product, aiming to minimize dynamically accumulating environmental impacts over time. 

`optimex` builds on top of the optimization framework [pyomo](https://github.com/Pyomo/pyomo) and the LCA framework [Brightway](https://docs.brightway.dev/en/latest). If you are looking for a time-explicit LCA rather than an optimization tool, make sure to check out [`bw_timex`](https://docs.brightway.dev/projects/bw-timex/en/latest/).

## Features

Like other transition pathway optimization tools, `optimex` identifies the optimal timing and scale of process deployments to minimize environmental impacts over a transition period. What sets `optimex` apart is its integration of three additional, temporal considerations for environmental impacts:

1. **Timing within Process Life Cycles:** Environmental impacts are spread across a process’s life cycle: construction happens first, use comes later, and end-of-life impacts follow. `optimex` captures this by distributing process inputs and outputs over time.

2. **Technology Evolution:** Future technologies may become more sustainable, reducing the environmental impacts later in the expansion period. `optimex` reflects this by allowing process inventories to evolve over time.

3.	**Accumulation of Emissions and Impacts:** Most impacts arise from the accumulation of emissions, but are typically modeled as discrete and independent pulses. `optimex` retains the timing of emissions during inventory calculations and applies dynamic characterization to account for impact accumulation.

During the transition pathway optimization, `optimex` simultaneously accounts for these temporal considerations, identifying the environmentally optimal process deployment over the transition period.
  
## Getting Started

- [Installation Guide](https://optimex.readthedocs.io/en/latest/content/installation.html)
- [Getting Started Tutorial](https://optimex.readthedocs.io/en/latest/content/getting_started.html)
- [Example Collection](https://optimex.readthedocs.io/en/latest/content/examples/index.html)

## Support

If you have any questions or need help, do not hesitate to contact us:
- Jan Tautorus ([jan.tautorus@rwth-aachen.de](mailto:jan.tautorus@rwth-aachen.de))
- Timo Diepers ([timo.diepers@ltt.rwth-aachen.de](mailto:timo.diepers@ltt.rwth-aachen.de))

## Contributing

We welcome contributions! If you have suggestions or want to fix a bug, please:
- [Open an Issue](https://github.com/TimoDiepers/optimex/issues)
- [Send a Pull Request](https://github.com/TimoDiepers/optimex/pulls)

## License

Distributed under the terms of the [BSD 3 Clause license][License], `optimex` is free and open source software.

<!-- github-only -->

[command-line reference]: https://optimex.readthedocs.io/en/latest/usage.html
[License]: https://github.com/TimoDiepers/optimex/blob/main/LICENSE
[Contributor Guide]: https://github.com/TimoDiepers/optimex/blob/main/CONTRIBUTING.md
[Issue Tracker]: https://github.com/TimoDiepers/optimex/issues

