---
icon: lucide/download
tags:
  - installation
  - setup
---

# Installation

`optimex` is available on [PyPI](https://pypi.org/project/optimex/) and [conda](https://anaconda.org/channels/diepers/packages/optimex/overview). Choose your preferred package manager below. In case you have an Apple Silicon Chip, you can directly jump to the [platform-specific notes](#apple-silicon-m1m2m3m4).

## Quick Install

If you just want to get started quickly:

=== "uv"

    ```bash
    uv pip install optimex
    ```

=== "pip"

    ```bash
    pip install optimex
    ```

=== "conda / mamba"

    ```bash
    conda install -c conda-forge -c cmutel -c diepers optimex
    ```

---

## Detailed Installation

### Using uv

[uv](https://docs.astral.sh/uv/) is a fast, modern Python package manager written in Rust. It's significantly faster than pip and handles dependency resolution more reliably. Installing uv is a [one-liner](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

=== "New Project"

    Create a new project with uv:

    ```bash
    uv init my-optimex-project
    cd my-optimex-project
    uv add optimex
    ```

    Run your scripts with:

    ```bash
    uv run python my_script.py
    ```

    Or directly get started in [JupyterLab](https://jupyter.org):

    ```bash
    uv run --with jupyter jupyter lab
    ```

    By default, `jupyter lab` will start the server at [http://localhost:8888/lab](http://localhost:8888/lab).

    !!! tip "One-line `optimex` setup"
        For your convenience, we've prepared a [setup script](https://raw.githubusercontent.com/RWTH-LTT/optimex/refs/heads/main/scripts/uv_setup.py) that initiates a new uv project and fetches our `optimex` example notebook, so you can directly get started. Just run:
        ```bash
        uv run https://raw.githubusercontent.com/RWTH-LTT/optimex/refs/heads/main/scripts/uv_setup.py my-optimex-example
        ```
        You can of course customize the project name, just replace the "my-optimex-example" at the end.

=== "New Environment"

    Create and activate a new virtual environment:

    ```bash
    uv venv .venv
    source .venv/bin/activate  # Linux/macOS
    # or: .venv\Scripts\activate  # Windows
    ```

    Install optimex:

    ```bash
    uv pip install optimex
    ```

=== "Existing Environment"

    If you already have an activated virtual environment:

    ```bash
    uv pip install optimex
    ```

---

### Using conda / mamba

[Conda](https://docs.conda.io/) and [mamba](https://mamba.readthedocs.io/) are popular in scientific computing. Mamba is a faster drop-in replacement for conda.

=== "New Environment"

    Create a new environment with optimex:

    ```bash
    conda create -n optimex -c conda-forge -c cmutel -c diepers optimex
    conda activate optimex
    ```

    Or with mamba:

    ```bash
    mamba create -n optimex -c conda-forge -c cmutel -c diepers optimex
    mamba activate optimex
    ```

=== "Existing Environment"

    Activate your environment and install:

    ```bash
    conda activate myenv
    conda install -c conda-forge -c cmutel -c diepers optimex
    ```

---

### Using pip

The standard Python package manager.

=== "New Environment"

    Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # or: .venv\Scripts\activate  # Windows
    ```

    Install optimex:

    ```bash
    pip install optimex
    ```

=== "Existing Environment"

    If you already have an activated virtual environment:

    ```bash
    pip install optimex
    ```

---
## Platform-Specific Notes

### Apple Silicon (M1/M2/M3/M4)

Brightway and `optimex` run natively on Apple Silicon. However, the `pypardiso` sparse solver used on Windows and Linux is not available for ARM processors. Without it, Brightway falls back to SciPy's slower SuperLU solver. For fast linear algebra, install `scikit-umfpack` (available as `pip install "optimex[umfpack]"`), as described below.

=== "uv"

    `uv` manages your Python environment efficiently, but you must first install the system-level `SuiteSparse` dependencies via [Homebrew](https://brew.sh).

    Install system dependencies:
    ```bash
    brew install swig suite-sparse
    ```

    Create project & add dependencies:
    ```bash
    uv init my-project
    cd my-project
    # Adding optimex along with the ARM-compatible solver
    uv add optimex scikit-umfpack
    ```

    Run Jupyter (without adding to project)
    ```bash
    uv run --with jupyter jupyter lab
    ```

=== "pip"

    When using `pip`, you must first install the system-level `SuiteSparse` dependencies via [Homebrew](https://brew.sh).

    Install system dependencies:
    ```bash
    brew install swig suite-sparse
    ```

    Install packages:
    ```bash
    # Ensure your virtual environment is activated
    pip install optimex scikit-umfpack
    ```

=== "conda / mamba"

    Conda is the most robust method for Apple Silicon because it handles the complex C-libraries (`SuiteSparse`) automatically without requiring [Homebrew](https://brew.sh).

    ```bash
    conda create -n optimex -c conda-forge -c cmutel -c diepers \
        optimex \
        "scikit-umfpack>=0.4.2" \
        "numpy>=2"
    
    conda activate optimex
    ```
---

## Optional Dependencies

### Jupyter Support

For interactive notebooks:

=== "uv"

    ```bash
    uv pip install jupyterlab
    ```

=== "pip"

    ```bash
    pip install jupyterlab
    ```

=== "conda / mamba"

    ```bash
    conda install -c conda-forge jupyterlab
    ```

### Development Installation

To install from source for development:

```bash
git clone https://github.com/RWTH-LTT/optimex.git
cd optimex
pip install -e ".[dev,docs]"
```

---

## Verifying Installation

Test that optimex is installed correctly:

```python
import optimex
print(optimex.__version__)
```

## Solver Requirements

`optimex` builds its models with [Pyomo](https://www.pyomo.org/), which needs a separate linear programming solver. **No solver is installed with `optimex`**, so install one of these:

| Solver | Install | `solver_name` |
|---|---|---|
| [HiGHS](https://highs.dev/) (open source) | `pip install highspy` or `pip install "optimex[highs]"` | `"highs"` |
| [GLPK](https://www.gnu.org/software/glpk/) (open source) | `conda install -c conda-forge glpk`, or `brew install glpk` (macOS), or `apt install glpk-utils` (Debian/Ubuntu) | `"glpk"` |
| [Gurobi](https://www.gurobi.com/) (commercial, free for academics) | `pip install "optimex[gurobi]"`, plus a license | `"gurobi"` |

Most examples in this documentation use GLPK, which is enough for single
models. For [series of models](series.md) (sweeps,
scenarios) and [Monte Carlo analyses](uncertainty.md), use HiGHS or Gurobi:
`optimex` then builds the model once and only updates what changes, which is
much faster for large systems. Pass the solver to every solve:

```python
optimizer.solve_model(model, solver_name="glpk")
```

`solve_model()` uses Gurobi when no `solver_name` is given. To check that a solver is found:

```python
import pyomo.environ as pyo
print(pyo.SolverFactory("glpk").available())  # True if GLPK is installed
```

!!! info "Gurobi License"
    Gurobi offers free academic licenses. Visit [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/) to obtain one.

---

## Next Steps

- [Basic Example](examples/basic_example.md): A complete example that runs from scratch, without ecoinvent
- [Quick Start](quickstart.md): A condensed reference of the workflow
- [Workflow Overview](workflow.md): How the user guide is organized
