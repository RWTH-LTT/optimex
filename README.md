<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/optimex_dark_nomargins.svg" height="50">
    <img alt="optimex logo" src="docs/_static/optimex_light_nomargins.svg" height="50">
  </picture>
</h1>

[![Read the Docs](https://img.shields.io/readthedocs/optimex?label=documentation)](https://optimex.readthedocs.io/)
[![tests](https://img.shields.io/github/actions/workflow/status/RWTH-LTT/optimex/python-test.yml?label=tests)](https://github.com/RWTH-LTT/optimex/actions/workflows/python-test.yml)
[![codecov](https://codecov.io/gh/RWTH-LTT/optimex/graph/badge.svg)](https://codecov.io/gh/RWTH-LTT/optimex)
[![PyPI - Version](https://img.shields.io/pypi/v/optimex?color=%2300549f)](https://pypi.org/project/optimex/)
[![Conda Version](https://img.shields.io/conda/v/diepers/optimex?label=conda)](https://anaconda.org/diepers/optimex)
[![Conda - License](https://img.shields.io/conda/l/diepers/optimex)](https://github.com/TimoDiepers/optimex/blob/main/LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TimoDiepers/optimex/main?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2Fbasic_optimex_example.ipynb)

## What is optimex?

`optimex` is an open-source Python package for **time-explicit Life Cycle Optimization (LCO)** — a framework that finds optimal technology transition pathways while fully accounting for *when* emissions occur and *how* the product systems and technologies evolve over time.

If you already do Life Cycle Assessment (LCA) with [Brightway](https://brightway.dev), `optimex` lets you take your existing product system models, temporalize them, and turn them into fully-fledged optimization problems — without having to rebuild anything from scratch.

## Why LCA users need this

Standard LCA tells you the environmental impact of a predefined product system. But what if you want to choose *between* competing technologies, or find the best deployment schedule for a set of processes over a 25-year horizon? That's the domain of **Life Cycle Optimization**. LCO extends LCA by treating technology selection and capacity planning as decision variables, and minimizes an environmental objective subject to system constraints.

However, both LCA and LCO are traditionally *static*: all flows are collapsed to a single point in time, background supply chains are fixed, and the timing of real-world activities — construction, multi-year operation, end-of-life — is ignored. This matters enormously in a rapidly decarbonizing world where the same technology installed in 2025 versus 2035 carries very different lifecycle impacts.

`optimex` solves this by making both the assessment and the optimization **time-explicit**.

## Why go time-explicit?

Making your optimization time-explicit unlocks insights that static approaches simply cannot provide:

- **Temporal distribution of flows** — Construction, operation, and end-of-life happen at different points in time. An electric vehicle built today will consume electricity over the next, say, 15 years; the underlying electricity mix might change a lot over this time period, affecting environmental impacts.
- **Temporal evolution of technologies** — A process installed in 2030 will be more efficient than one installed today. `optimex` locks in technology parameters at the time of installation (the *vintage*), so improving technologies are correctly credited when they are actually deployed.
- **Time-varying background systems** — Upstream supply chains decarbonize. `optimex` links foreground demands to time-specific background databases (e.g., generated with [premise](https://premise.readthedocs.io)), so that future electricity, hydrogen, or steel inputs are assessed against future grid mixes rather than today's.
- **Flexible process operation** - Because `optimex` can differentiate process installation from operation, it can also scale their operation independent of installation. Together with the vintage-tracking abilities, this enables vintage-specific dispatch preferring more efficient vintages.
- **Dynamic impact assessment** — Characterization factors can vary over time. For climate change, `optimex` directly integrates dynamic LCIA methods from [dynamic_characterization](https://dynamic-characterization.readthedocs.io), enabling radiative forcing, AGWP, or AGTP as objective metrics.
- **Novel transition strategies** — Time-explicit LCO reveals strategies invisible to static models: strategic overcapacity that accepts stranded fossil assets to accelerate clean deployment, preferential dispatch of cleaner vintages, and technology diversification to navigate transient resource bottlenecks.

## Use Cases

`optimex` is broadly applicable across sectors where temporal dynamics are decisive for sustainability:

- **Evolving supply chains** — Systems depending on electricity, steel, or hydrogen undergoing rapid decarbonization
- **Early-stage technologies** — Processes with significant vintage-dependent performance improvements (e.g., electrolyzers, DAC)
- **Circular economy planning** — Temporal mismatches between primary demand and secondary supply from long material residence times
- **Time-resolved carbon accounting** — Biogenic feedstocks, temporary carbon storage, or CO2 removal with varying temporal profiles
- **Multi-regional supply chains** — Sourcing across regions with divergent decarbonization trajectories

## Built on Brightway

`optimex` is deeply integrated with the [Brightway](https://brightway.dev) LCA ecosystem. You model your foreground system exactly as you would for a standard LCA — defining products, processes, and exchanges. You then add temporal metadata (relative temporal distributions, vintage-dependent scaling factors, operation-vs-installation flow classifications) as flow-level attributes. The rest is handled by `optimex`.

This means:
- **No lock-in** — Use any Brightway-compatible inventory database (ecoinvent, custom databases, etc.).
- **Familiar workflows** — If you know Brightway, you already know how to build foreground systems for `optimex`.
- **Reuse existing models** — Temporalize and optimize a product system you have already built for standard LCA.

> **Tip:** you can also directly re-use temporalized product system models made with [`bw_timex`](https://github.com/brightway-lca/bw_timex), our time-explicit *assessment* framework.

For optimization, `optimex` uses [Pyomo](https://www.pyomo.org), a powerful open-source algebraic modeling language for mathematical programming.

## Installation

```bash
pip install optimex
```

More complete installation instructions are available [here](https://optimex.readthedocs.io/en/latest/content/installation/).

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

[Open an Issue](https://github.com/TimoDiepers/optimex/issues) or [Send a Pull Request](https://github.com/TimoDiepers/optimex/pulls) — contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor setup and workflow.

## License

[BSD 3-Clause License](https://github.com/TimoDiepers/optimex/blob/main/LICENSE)
