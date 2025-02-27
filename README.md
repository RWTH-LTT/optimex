<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/optimex_dark_nomargins.svg" height="50">
    <img alt="optimex logo" src="docs/_static/optimex_light_nomargins.svg" height="50">
  </picture>
</h1>

> *Please note that this is an early access version developed during the master thesis of [@JanTautorus](https://github.com/JanTautorus). While it's functional, it’s not fully configured to handle all use cases yet.*

[![Read the Docs](https://img.shields.io/readthedocs/optimex?label=documentation)](https://optimex.readthedocs.io/)
[![PyPI - Version](https://img.shields.io/pypi/v/optimex?color=%2300549f)](https://pypi.org/project/optimex/)
[![Conda Version](https://img.shields.io/conda/v/diepers/optimex?label=conda)](https://anaconda.org/diepers/optimex)
[![Conda - License](https://img.shields.io/conda/l/diepers/optimex)](https://github.com/TimoDiepers/optimex/blob/main/LICENSE)

This is a python package for time-explicit Life Cylce Optimization that helps you identify transition pathways of systems with minimal environmental impacts.

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


## Building the Documentation

You can build the documentation locally by installing the documentation Conda environment:

```bash
conda env create -f docs/environment.yml
```

activating the environment

```bash
conda activate sphinx_optimex
```

and [running the build command](https://www.sphinx-doc.org/en/master/man/sphinx-build.html#sphinx-build):

```bash
sphinx-build docs _build/html --builder=html --jobs=auto --write-all; open _build/html/index.html
```

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
