"""
Optimization model construction and solving for temporal LCA-based pathway optimization.

This module creates and solves optimization models that minimize either an
environmental impact category or the discounted cost of background purchases over
time, while meeting demand constraints and respecting process limits.

## Backends

The problem is formulated once, independently of any modeling framework
(`optimex.formulation`), and built by one of two backends, chosen with
`create_model(..., backend=...)`:

- `"pyomo"` (default, `optimex.pyomo_backend`): a Pyomo `ConcreteModel`, for
  anything that works on the model itself (fixing variables, extra constraints,
  IIS files, Pyomo's solver interfaces).
- `"linopy"` (`optimex.linopy_backend`, needs `pip install optimex[linopy]`):
  built from arrays, much faster for systems with thousands of processes.

`solve_model()` solves either and stores the solution on the model under the same
names, so post-processing does not depend on the backend.

## Unit Convention

One "unit" of a process is one reference-flow unit of the underlying LCA process:
installing 1 unit delivers exactly the production stated by its production temporal
distribution, i.e. `sum_tau foreground_production[p, r, tau]`, spread over the whole
operation window. The per-tau entry is therefore the unit's output *per year of
operation*, and the sum over the window is its *lifetime* output.

Consequences:
- `var_installation[p, v]` counts units of vintage v; its installation-driven
  flows (construction, end-of-life) are incurred once per unit.
- `var_operation[p, v, t]` counts how many of those units are running in year t,
  so the capacity bound is `var_operation[p, v, t] <= var_installation[p, v]`.
- Annual output at time t of vintage v is
  `foreground_production[p, r, t - v] * var_operation[p, v, t]`, never the sum over
  the operation window (that would let one unit deliver its lifetime output every
  year and under-count installation impacts by the number of operating years).
- To convert installed units into an annual production capacity for reporting or
  plotting, multiply by the same per-tau rate — see
  `PostProcessor.get_production_capacity()`.

## Scaling Convention

The optimization uses a two-tier scaling system for numerical stability:

### Decision Variables (REAL UNITS)
- `var_installation[p, t]`: Number of process units installed (dimensionless)
- `var_operation[p, v, t]`: Units of vintage v running at time t (dimensionless,
  0 to the units available from that vintage)

Both decision variables remain in REAL (unscaled) units to:
1. Maintain physical interpretability
2. Allow direct comparison with process limits
3. Ensure correct background inventory calculations

### Parameters (SCALED UNITS)

**Foreground parameters** (scaled by `fg_scale`):
- `foreground_production[p, r, tau]`: kg product per process unit [SCALED]
- `foreground_biosphere[p, e, tau]`: kg emission per process unit [SCALED]
- `foreground_technosphere[p, i, tau]`: kg intermediate per process unit [SCALED]
- `internal_demand_technosphere[p, r, tau]`: kg product per process unit [SCALED]
- `demand[r, t]`: kg product demanded [SCALED]

**Characterization parameters** (scaled by `cat_scales[category]`):
- `characterization[c, e, t]`: impact per kg emission [SCALED]
- `category_impact_limits[(c, t)]`: time-specific maximum impact allowed [SCALED]
- `cumulative_category_impact_limits[c]`: cumulative maximum impact allowed [SCALED]

**Price parameters** (scaled by `cost_scale`):
- `intermediate_flow_prices[i, t]`: price per kg intermediate [SCALED]
- Cost expressions are therefore in units of `fg_scale * cost_scale`, the same way
  impact expressions are in units of `fg_scale * cat_scales[category]`.

**Unscaled parameters**:
- `background_inventory[bkg, i, e]`: kg emission / kg intermediate [UNSCALED]
- `mapping[bkg, t]`: interpolation weights [UNSCALED, dimensionless]
- `process_deployment_limits_*`: deployment limits [UNSCALED, matches var_installation]
- `process_operation_limits_*`: operation limits [UNSCALED, matches var_operation]

### Dimensional Consistency

When SCALED parameters are multiplied by REAL decision variables:
```
scaled_param [kg SCALED/process] × var_real [# processes] = result [kg SCALED]
```

To convert back to REAL units:
```
result [kg SCALED] × fg_scale [REAL/SCALED] = result [kg REAL]
```

Example constraint dimensional analysis:
```
ProductDemandFulfillment:
    production[p, r, t-v] [kg SCALED/(unit·year)] × var_operation [# units running]
        = demand [kg SCALED] ✓

OperationCapacity:
    var_operation [# units running] <= var_installation [# units installed] ✓
    (both REAL unit counts, so no scaling factor appears)
```
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import dill
import pyomo.environ as pyo
from loguru import logger
from pyomo.opt.results.results_ import SolverResults

from optimex.converter import OptimizationModelInputs

# Pyomo backend; its helpers are re-exported here for backward compatibility
from optimex.pyomo_backend import (  # noqa: F401
    LazyExpressionMap,
    LinearExpressionMap,
    create_pyomo_model,
    linear_sum,
    solve_pyomo_model,
)


def expression_value(container, key) -> float:
    """Solved value of `container[key]` for Pyomo Expressions and expression maps."""
    if isinstance(container, LinearExpressionMap):
        return container.value(key)
    return pyo.value(container[key])


def create_model(
    inputs: OptimizationModelInputs,
    name: str,
    objective_category: Optional[str] = None,
    debug_path: str = None,
    objective: str = "environmental",
    aggregate_vintages: bool = False,
    backend: str = "pyomo",
):
    """
    Build the optimization model for the provided inputs.

    It uses flexible operation mode where processes can operate between 0 and
    their maximum installed capacity.

    Parameters
    ----------
    inputs : OptimizationModelInputs
        Structured input data containing all flows, mappings, and constraints
        required for model construction.
    name : str
        Name of the model instance.
    objective_category : str, optional
        The impact category to minimize. Required for `objective="environmental"`,
        ignored for `objective="cost"`.
    debug_path : str, optional
        If provided, the LP formulation is written to this path.
    objective : {"environmental", "cost"}, optional
        What to minimize. "environmental" (default) minimizes the cumulative impact in
        `objective_category`. "cost" minimizes the discounted cost of the background
        products the foreground purchases directly, priced with
        `inputs.intermediate_flow_prices`. Impact limits apply in both cases.
    aggregate_vintages : bool, optional
        If True, processes whose vintages all operate identically (no
        vintage-dependent operation flows, operation flows constant over the
        operation window) get one operation variable per year
        (`var_operation_total[p, t]`) instead of one per vintage and year
        (`var_operation[p, v, t]`), with their capacity carried forward year by
        year. The optimum is the same and the model is several times smaller,
        which matters for large systems. Read per-vintage operation with
        `operation_value()`; the PostProcessor does so. Keep the default (False)
        to fix or inspect per-vintage operation variables directly.
    backend : {"pyomo", "linopy"}, optional
        "pyomo" (default) returns a Pyomo `ConcreteModel`. "linopy" returns a
        `optimex.linopy_backend.LinopyModel`, which builds much faster for large
        systems; `solve_model()` and the PostProcessor accept both. The linopy
        backend needs the `linopy` package.

    Returns
    -------
    pyo.ConcreteModel or LinopyModel
        A fully constructed model ready for optimization.
    """
    if backend == "linopy":
        from optimex.linopy_backend import create_linopy_model

        create = create_linopy_model
    elif backend == "pyomo":
        create = create_pyomo_model
    else:
        raise ValueError(f"Unknown backend {backend!r}. Expected 'pyomo' or 'linopy'.")
    return create(
        inputs,
        name=name,
        objective_category=objective_category,
        objective=objective,
        aggregate_vintages=aggregate_vintages,
        debug_path=debug_path,
    )


def solve_model(
    model: pyo.ConcreteModel,
    solver_name: str = "gurobi",
    solver_args: Dict[str, Any] = None,
    solver_options: Dict[str, Any] = None,
    tee: bool = True,
    compute_iis: bool = False,
    **solve_kwargs: Any,
) -> Tuple[pyo.ConcreteModel, float, SolverResults]:
    """
    Solve a model from `create_model()` with the specified solver and
    denormalize the objective using the stored scales. Dispatches to the
    backend the model was built with.

    Parameters
    ----------
    model : pyo.ConcreteModel or LinopyModel
        The model to be solved, from `create_model()`. Must have attribute
        `scales` with keys 'foreground', 'characterization' and, for the cost
        objective, 'cost'.
    solver_name : str, optional
        Name of the solver (default: "gurobi"). For a linopy model, Pyomo names
        like "gurobi_direct_v2" or "appsi_highs" are mapped to the linopy solver.
    solver_args : dict, optional
        Args to pass to SolverFactory (Pyomo models only).
    solver_options : dict, optional
        Solver-specific options, e.g. timelimit, mipgap.
    tee : bool, optional
        If True, prints solver output.
    compute_iis : bool, optional
        If True and infeasible, writes IIS to file.
    **solve_kwargs
        Additional kwargs for solver.solve().

    Returns
    -------
    model : pyo.ConcreteModel or LinopyModel
        The model passed in, now holding the solution (with original scaling
        preserved). A linopy model holds it as plain numbers under the attribute
        names of the Pyomo model.
    true_obj : float
        The denormalized objective value: the impact in the units of the objective
        category, or the discounted cost in the currency of the input prices.
    results : SolverResults
        The raw Pyomo solver results object; for a linopy model, a namespace
        with `status` and `termination_condition`.
    """
    from optimex.linopy_backend import LinopyModel, solve_linopy_model

    if isinstance(model, LinopyModel):
        model, results = solve_linopy_model(
            model,
            solver_name=solver_name,
            solver_options=solver_options,
            tee=tee,
            compute_iis=compute_iis,
            **solve_kwargs,
        )
    else:
        model, results = solve_pyomo_model(
            model,
            solver_name=solver_name,
            solver_args=solver_args,
            solver_options=solver_options,
            tee=tee,
            compute_iis=compute_iis,
            **solve_kwargs,
        )
    return model, _real_objective(model), results


def _real_objective(model) -> float:
    """The objective of a solved model in real units."""
    scales = getattr(model, "scales", {})
    if getattr(model, "_objective", "environmental") == "cost":
        objective_scale = scales.get("cost", 1.0)
    else:
        objective_scale = scales.get("characterization", {}).get(
            model._objective_category, 1.0
        )
    # OBJ is additionally normalized by its largest coefficient
    scaled_obj = pyo.value(model.OBJ) * scales.get("objective", 1.0)
    true_obj = scaled_obj * scales.get("foreground", 1.0) * objective_scale
    logger.info(f"Objective (scaled): {scaled_obj:.6g}")
    logger.info(f"Objective (real):   {true_obj:.6g}")
    return true_obj


def _numeric(value) -> float:
    """Value of a Pyomo variable or of a plain number (solved linopy models)."""
    value = getattr(value, "value", value)
    return 0.0 if value is None else float(value)


def _split_operation(model: pyo.ConcreteModel, p, t) -> dict:
    """
    Per-vintage operation of an aggregated process p in year t.

    Each active vintage runs the same share of its capacity, so the per-vintage
    values satisfy the per-vintage capacity bounds and reproduce every flow of the
    total (all vintages of an aggregated process have the same rates).
    """
    cache = getattr(model, "_disaggregated_operation", None)
    if cache is None:
        cache = model._disaggregated_operation = {}
    if (p, t) not in cache:
        total = max(_numeric(model.var_operation_total[p, t]), 0.0)
        vintages = model._active_vintages_by_process_time[p, t]
        capacities = []
        for v in vintages:
            if (p, v) in model.var_installation:
                capacities.append(max(_numeric(model.var_installation[p, v]), 0.0))
            else:
                capacities.append(model._existing_capacity_dict.get((p, v), 0.0))
        available = sum(capacities)
        cache[p, t] = {
            v: total * capacity / available if available > 0 else 0.0
            for v, capacity in zip(vintages, capacities)
        }
    return cache[p, t]


def operation_value(model: pyo.ConcreteModel, p, v, t) -> float:
    """
    Solved number of vintage-v units of process p running in year t.

    Reads `var_operation[p, v, t]`, or for processes built with
    `aggregate_vintages=True` the share of `var_operation_total[p, t]` assigned
    to vintage v.
    """
    if (p, v, t) in model.var_operation:
        return pyo.value(model.var_operation[p, v, t])
    totals = getattr(model, "var_operation_total", None)
    if totals is not None and (p, t) in totals:
        return _split_operation(model, p, t).get(v, 0.0)
    return 0.0


def validate_operation_bounds(
    model: pyo.ConcreteModel, tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Validate that operation levels respect capacity constraints.

    This function performs post-solve validation to ensure that var_operation[p, v, t]
    does not exceed the units available from that vintage. Both variables count
    process units, so the comparison needs no production rate.

    With 3D operation variables:
    - Greenfield: var_operation[p, v, t] <= var_installation[p, v]
    - Brownfield: var_operation[p, v, t] <= existing_capacity[p, v]

    Parameters
    ----------
    model : pyo.ConcreteModel
        A solved Pyomo model.
    tolerance : float, optional
        Relative tolerance for validation (default: 1e-6).

    Returns
    -------
    dict
        Validation results with keys:
        - "valid": bool, True if all operation levels respect capacity
        - "violations": list of tuples (process, vintage, time, operation, capacity, violation_type)
        - "max_violation": float, maximum violation found
        - "summary": str, human-readable summary

    Raises
    ------
    ValueError
        If model is not solved.
    """
    if not hasattr(model, "var_operation"):
        raise ValueError("Model must have var_operation")

    violations = []
    max_violation = 0.0

    existing_cap_dict = getattr(model, "_existing_capacity_dict", {})

    # Validate per-vintage operation bounds
    for p, v, t in model.ACTIVE_VINTAGE_TIME:
        operation_level = operation_value(model, p, v, t)

        # Units available from this vintage (operation and installation are both
        # unit counts, so no production rate enters here)
        if v in model.SYSTEM_TIME:
            max_capacity = pyo.value(model.var_installation[p, v])
        else:
            max_capacity = existing_cap_dict.get((p, v), 0)

        # Check if operation exceeds capacity
        if operation_level < -tolerance:
            violation = abs(operation_level)
            violations.append((p, v, t, operation_level, max_capacity, "negative"))
            max_violation = max(max_violation, violation)
        elif max_capacity > 0 and operation_level > max_capacity * (1.0 + tolerance):
            violation = operation_level - max_capacity
            violations.append(
                (p, v, t, operation_level, max_capacity, "exceeds_capacity")
            )
            max_violation = max(max_violation, violation)

    # Generate summary
    is_valid = len(violations) == 0
    if is_valid:
        summary = "✓ All per-vintage operation levels respect capacity constraints"
    else:
        summary = (
            f"✗ Found {len(violations)} per-vintage operation bound violations. "
            f"Max violation: {max_violation:.2e}"
        )

    return {
        "valid": is_valid,
        "violations": violations,
        "max_violation": max_violation,
        "summary": summary,
    }


def solve_series(variants, base=None, solver_name: str = "glpk", **kwargs):
    """
    Solve a series of related models: a sweep, a Pareto frontier, or scenarios.

    Much faster than calling `create_model` and `solve_model` in a loop when
    the variants differ only in their data (limits, prices, background data,
    exchange amounts): with Gurobi or HiGHS, the formulation is built once and
    each variant is assembled from it. See `optimex.series.solve_series` for all
    arguments.

    Examples
    --------
    >>> sweep = optimizer.solve_series(
    ...     {b: inputs.model_copy(update={
    ...         "cumulative_category_impact_limits": {"climate_change": b}})
    ...      for b in budgets},
    ...     base=inputs, objective="cost", solver_name="highs",
    ... )
    """
    from optimex.series import solve_series as _solve_series

    return _solve_series(variants, base=base, solver_name=solver_name, **kwargs)


def save_solved_model(
    model: pyo.ConcreteModel,
    path: Union[str, Path],
    objective_value: float = None,
) -> None:
    """
    Save a solved Pyomo model to disk for later use.

    This function saves the model's solution state (variable values, scales,
    and metadata) to a pickle file, allowing you to reload it later and use
    it with PostProcessor without re-running the optimization.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The solved Pyomo model to save.
    path : str or Path
        File path to save the model. Should have .pkl extension.
    objective_value : float, optional
        The denormalized objective value from solve_model().
        If provided, it will be stored with the model.

    Examples
    --------
    >>> model, obj, results = solve_model(model)
    >>> save_solved_model(model, "solved_model.pkl", objective_value=obj)

    Notes
    -----
    The saved file contains:
    - The complete Pyomo model with all variable values
    - Model scales for denormalization
    - Optionally: objective value

    Warning: Only load files from trusted sources.
    """
    path = Path(path)

    # Store additional metadata on the model
    if objective_value is not None:
        model._saved_objective_value = objective_value

    with open(path, "wb") as f:
        dill.dump(model, f, protocol=dill.HIGHEST_PROTOCOL)

    logger.info(f"Saved solved model to: {path}")


# Alias for backward compatibility
save_model = save_solved_model


def load_solved_model(
    path: Union[str, Path],
) -> Tuple[pyo.ConcreteModel, float]:
    """
    Load a previously saved solved Pyomo model from disk.

    This function deserializes a model saved with save_solved_model(), restoring
    the complete solved state for use with PostProcessor.

    Parameters
    ----------
    path : str or Path
        File path to the saved model (.pkl file).

    Returns
    -------
    model : pyo.ConcreteModel
        The loaded solved model, ready for use with PostProcessor.
    objective_value : float or None
        The denormalized objective value, if it was saved.

    Examples
    --------
    >>> model, obj = load_solved_model("solved_model.pkl")
    >>> pp = PostProcessor(model)
    >>> pp.plot_impacts()

    Notes
    -----
    Warning: Only load files from trusted sources, as dill
    can execute arbitrary code during deserialization.
    """
    path = Path(path)

    with open(path, "rb") as f:
        model = dill.load(f)

    # Retrieve stored metadata
    objective_value = getattr(model, "_saved_objective_value", None)

    logger.info(f"Loaded solved model from: {path}")

    return model, objective_value


# Alias for backward compatibility
load_model = load_solved_model
