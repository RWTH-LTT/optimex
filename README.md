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

This is a Python package for transition pathway optimization based on time-explicit Life Cycle Assessment (LCA). `optimex` helps identify optimal process portfolios and deployment timing in systems with multiple processes producing the same product, aiming to minimize dynamically accumulating environmental impacts over time. 

`optimex` builds on top of the [Brightway LCA framework](https://docs.brightway.dev/en/latest). If you are looking for a time-explicit assessment rather than an optimization tool, make sure to check out our time-explicit LCA package [`bw_timex`](https://docs.brightway.dev/projects/bw-timex/en/latest/).

## Installation

You can install _optimex_ via [pip] from [PyPI]:

```console
$ pip install optimex
```

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide][Contributor Guide].

## License

Distributed under the terms of the [BSD 3 Clause license][License],
_optimex_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue][Issue Tracker] along with a detailed description.


<!-- github-only -->

[command-line reference]: https://optimex.readthedocs.io/en/latest/usage.html
[License]: https://github.com/TimoDiepers/optimex/blob/main/LICENSE
[Contributor Guide]: https://github.com/TimoDiepers/optimex/blob/main/CONTRIBUTING.md
[Issue Tracker]: https://github.com/TimoDiepers/optimex/issues


## Acknowledgments

We’d like to thank the authors and contributors of the following key packages that _optimex_ is based on:

- [**pyomo**](https://github.com/Pyomo/pyomo)
- [**brightway2.5**](https://github.com/brightway-lca/brightway25)

Additionally, we want to give a shoutout to the pioneering ideas and contributions from the following works:

- [**bw_timex**](https://github.com/brightway-lca/bw_timex)
- [**pulpo**](https://github.com/flechtenberg/pulpo)
- [**premise**](https://github.com/polca/premise)

## Support
If you have any questions or need help, do not hesitate to contact us:
- Jan Tautorus ([jan.tautorus@rwth-aachen.de](mailto:jan.tautorus@rwth-aachen.de))
- Timo Diepers ([timo.diepers@ltt.rwth-aachen.de](mailto:timo.diepers@ltt.rwth-aachen.de))
