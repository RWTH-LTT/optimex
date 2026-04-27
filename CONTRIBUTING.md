# Contributor Guide

Thank you for your interest in improving this project.
This project is open-source under the [BSD 3 Clause][License] and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- [Source Code][Source Code]
- [Documentation][Documentation]
- [Issue Tracker][Issue Tracker]
- [Code of Conduct][Code of Conduct]

[License]: https://opensource.org/licenses/BSD-3-Clause
[Source Code]: https://github.com/TimoDiepers/optimex
[Documentation]: https://optimex.readthedocs.io/
[Issue Tracker]: https://github.com/TimoDiepers/optimex/issues
[Code of Conduct]: https://github.com/TimoDiepers/optimex/blob/main/CODE_OF_CONDUCT.md

## How to report a bug

Report bugs on the [Issue Tracker][Issue Tracker].

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker][Issue Tracker].

## How to set up your development environment

We recommend using [uv](https://docs.astral.sh/uv/) to manage the project environment.

1. Install `uv` (if needed), see [uv installation instructions.](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

2. Sync the project with development, docs, and testing dependencies:

```console
$ uv sync --extra dev --extra docs --extra testing
```

## How to build the documentation locally

Serve the documentation locally with live reload:

```console
$ uv run zensical serve
```

The documentation will be available at `http://127.0.0.1:8000/`.

To build static documentation files:

```console
$ uv run zensical build
```

The built documentation will be in the `site` directory.

## How to test the project


1. Install the package with development requirements:

```console
$ uv sync --extra testing
```

2. Run the full test suite:

```console
$ uv run pytest
```


Unit tests are located in the _tests_ directory,
and are written using the [pytest][pytest] testing framework.

[pytest]: https://pytest.readthedocs.io/

## How to submit changes

Open a [pull request] to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The test suite must pass without errors and warnings.
- Include unit tests.
- If your changes add functionality, update the documentation accordingly.

To run linting and code formatting checks before committing your change, install pre-commit hooks and run checks before pushing:

```console
$ uv run pre-commit install
$ uv run pre-commit run --all-files
```


It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

[pytest]: https://pytest.readthedocs.io/
[pull request]: https://github.com/TimoDiepers/optimex/pulls
