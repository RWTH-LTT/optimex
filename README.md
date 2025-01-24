# optimex

[![PyPI](https://img.shields.io/pypi/v/optimex.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/optimex.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/optimex)][pypi status]
[![License](https://img.shields.io/pypi/l/optimex)][license]

[![Read the documentation at https://optimex.readthedocs.io/](https://img.shields.io/readthedocs/optimex/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/TimoDiepers/optimex/actions/workflows/python-test.yml/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/TimoDiepers/optimex/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/optimex/
[read the docs]: https://optimex.readthedocs.io/
[tests]: https://github.com/TimoDiepers/optimex/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/TimoDiepers/optimex
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Overview

This is a python package for time-explicit Life Cylce Optimization that helps you identify transition pathways of systems with minimal environmental impacts.

*Please note that this is an early access version developed during my master thesis. While it's functional, it’s not fully configured to handle all use cases yet.*

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
