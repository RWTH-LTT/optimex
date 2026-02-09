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

**Time-explicit life cycle optimization for transition pathways.**

Current life cycle optimization tools collapse all emissions to a single point in time, hiding critical temporal interdependencies: life cycles are distributed across years or decades, and the production systems behind them are evolving. `optimex` jointly models both dimensions — *when* exchanges occur and *how* they change over time — to design pathways that respect time-specific and cumulative environmental constraints.

## Key Capabilities

- **Temporal Distribution** — Maps life cycle exchanges across their actual timeframes via convolution, capturing time lags between construction, operation, and end-of-life
- **Technology Evolution** — Tracks vintage-dependent foreground improvements and links to prospective background databases reflecting supply chain decarbonization
- **Flexible Operation** — Separates capacity installation from operational dispatch, enabling vintage-specific merit order where cleaner cohorts are utilized first
- **Dynamic Characterization** — Retains emission timing for dynamic LCIA (e.g., Radiative Forcing, dynamic GWP), capturing how impacts accumulate over time

## What This Enables

Time-explicit LCO reveals transition strategies invisible to static approaches:

- **Strategic overcapacity** — Early clean technology investment that offsets stranded fossil assets when net emission savings outweigh embodied impacts
- **Vintage-specific dispatch** — Emissions-aware merit order that preferentially utilizes cleaner technology cohorts
- **Resource bottleneck navigation** — Technology diversification driven by time-specific constraints on water, critical minerals, or other resources
- **Cumulative budget compliance** — Pathway verification against carbon budgets and absolute limits through exact emission timing

## Use Cases

`optimex` is broadly applicable across sectors where temporal dynamics are decisive for sustainability:

- **Evolving supply chains** — Systems depending on electricity, steel, or hydrogen undergoing rapid decarbonization
- **Early-stage technologies** — Processes with significant vintage-dependent performance improvements (e.g., electrolyzers, DAC)
- **Circular economy planning** — Temporal mismatches between primary demand and secondary supply from long material residence times
- **Time-resolved carbon accounting** — Biogenic feedstocks, temporary carbon storage, or CO2 removal with varying temporal profiles
- **Multi-regional supply chains** — Sourcing across regions with divergent decarbonization trajectories

## Installation

```bash
pip install optimex
```

## Documentation

Full documentation, tutorials, and examples are available at **[optimex.readthedocs.io](https://optimex.readthedocs.io/)**.

- [Getting Started](https://optimex.readthedocs.io/en/latest/content/quickstart/)
- [Examples](https://optimex.readthedocs.io/en/latest/content/examples/)
- [API Reference](https://optimex.readthedocs.io/en/latest/api/overview/)

`optimex` builds on [Pyomo](https://github.com/Pyomo/pyomo) and [Brightway](https://docs.brightway.dev/en/latest). For time-explicit LCA without optimization, see [`bw_timex`](https://docs.brightway.dev/projects/bw-timex/en/latest/).

## Support

- Timo Diepers ([timo.diepers@ltt.rwth-aachen.de](mailto:timo.diepers@ltt.rwth-aachen.de))
- Jan Tautorus ([jan.tautorus@rwth-aachen.de](mailto:jan.tautorus@rwth-aachen.de))

## Contributing

[Open an Issue](https://github.com/TimoDiepers/optimex/issues) or [Send a Pull Request](https://github.com/TimoDiepers/optimex/pulls) — contributions are welcome.

## License

[BSD 3-Clause License](https://github.com/TimoDiepers/optimex/blob/main/LICENSE)
