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

[uv](https://docs.astral.sh/uv/) is a fast, modern Python package manager written in Rust. It's significantly faster than pip and handles dependency resolution more reliably.

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

    By default, `juypter lab` will start the server at [http://localhost:8888/lab](http://localhost:8888/lab).

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

Brightway and `optimex` run natively on Apple Silicon. However, the standard `pypardiso` solver used on Windows and Linux is not compatible with ARM processors. To achieve fast linear algebra calculations, you must use `scikit-umfpack`.

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
git clone https://github.com/TimoDiepers/optimex.git
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

`optimex` uses [Pyomo](https://www.pyomo.org/) for optimization, which requires a solver. By default, it uses [Gurobi](https://www.gurobi.com/), but you can configure other solvers like GLPK, CBC, or HiGHS.

!!! info "Gurobi License"
    Gurobi offers free academic licenses. Visit [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/) to obtain one.
