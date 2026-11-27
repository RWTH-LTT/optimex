"""
Solving a series of related models: sweeps, Pareto frontiers, scenarios.

Many studies solve the same product system many times with slightly different
data: a range of impact budgets to trace a Pareto frontier, a set of price or
background scenarios, or Monte Carlo samples (`optimex.uncertainty`).
`solve_series` solves such a series and collects the results in one table:

- With Gurobi or HiGHS, the formulation of the first (or `base`) inputs is built
  once and every variant is assembled from it with NumPy
  (`optimex.incremental`), then loaded straight into the solver
  (`optimex.persistent`). This is much faster than building and solving every
  model on its own.
- Variants can be solved in parallel worker processes (``n_jobs``).
- Only what is needed is kept per variant: objective, cumulative impacts, and
  cost always, PostProcessor tables on request, or the result of an
  ``extract`` function.

Variants are best made with ``inputs.model_copy(update={...})``, which shares
everything that does not change with the base inputs; that is what the fast
path relies on.
"""

import re
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from optimex import optimizer
from optimex.converter import OptimizationModelInputs
from optimex.postprocessing import PostProcessor


@dataclass
class SeriesResults:
    """
    Results of solving a series of models: one entry per variant (or sample).

    The index holds the variant labels (``"Variant"``), or the sample numbers
    (``"Sample"``) for Monte Carlo results.

    Attributes
    ----------
    objective : pandas.Series
        Objective value per variant, in real units; NaN where the solve failed.
    status : pandas.Series
        ``"optimal"``, or the solver's termination condition when a variant has
        no solution (e.g. ``"infeasible"``; GLPK reports ``"other"``).
    impacts : pandas.DataFrame
        Cumulative impact per variant (rows) and category (columns).
    costs : pandas.Series or None
        Discounted total cost per variant, when the inputs have prices.
    installation, operation, production : pandas.DataFrame
        Installed units, units running, and production, as from the
        PostProcessor, indexed by (variant, Time). Empty unless requested with
        `outputs` (installation is by default).
    models : list or None
        The solved models, if they were kept.
    extracted : list or None
        What the `extract` function returned for each variant (None where the
        variant has no solution).
    """

    objective: pd.Series
    status: pd.Series
    impacts: pd.DataFrame
    costs: Optional[pd.Series]
    installation: pd.DataFrame
    operation: pd.DataFrame
    production: pd.DataFrame
    models: Optional[list] = field(default=None, repr=False)
    extracted: Optional[list] = field(default=None, repr=False)

    @property
    def feasible(self) -> pd.Series:
        """Whether each variant has a solution."""
        return self.status == "optimal"

    def summary(self, quantiles: Sequence[float] = (0.05, 0.5, 0.95)) -> pd.DataFrame:
        """Mean, standard deviation, and quantiles of objective, impacts, and costs."""
        data = pd.concat(
            {"objective": self.objective, **{f"impact: {c}": v for c, v in self.impacts.items()}},
            axis=1,
        )
        if self.costs is not None:
            data["cost (discounted)"] = self.costs
        data = data[self.feasible]
        table = pd.DataFrame({"mean": data.mean(), "std": data.std()})
        for q in quantiles:
            table[f"q{q:g}"] = data.quantile(q)
        table["feasible"] = len(data)
        return table


def _results(rows, models, labels=None, index_name="Variant") -> SeriesResults:
    labels = list(range(len(rows))) if labels is None else list(labels)

    def stack(name):
        frames = {labels[i]: row[name] for i, row in enumerate(rows) if row[name] is not None}
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, names=[index_name])

    index = pd.Index(labels, name=index_name)
    impacts = pd.DataFrame([row["impacts"] for row in rows], index=index)
    costs = None
    if any(row["cost"] is not None for row in rows):
        costs = pd.Series([row["cost"] for row in rows], index=index, name="cost", dtype=float)
    return SeriesResults(
        objective=pd.Series([row["objective"] for row in rows], index=index, name="objective", dtype=float),
        status=pd.Series([row["status"] for row in rows], index=index, name="status"),
        impacts=impacts,
        costs=costs,
        installation=stack("installation"),
        operation=stack("operation"),
        production=stack("production"),
        models=models,
        extracted=(
            [row["extracted"] for row in rows]
            if any(row["extracted"] is not None for row in rows) else None
        ),
    )


try:  # Pyomo's newer solver interfaces (e.g. "highs") raise their own error
    from pyomo.contrib.solver.common.util import NoFeasibleSolutionError

    _NO_SOLUTION_ERRORS = (RuntimeError, NoFeasibleSolutionError)
except ImportError:  # pragma: no cover - older Pyomo
    _NO_SOLUTION_ERRORS = (RuntimeError,)

_FORMULATION_KWARGS = {"objective_category", "objective", "aggregate_vintages"}
OUTPUTS = ("installation", "operation", "production")


def _use_persistent(solver_name, persistent, model_kwargs) -> bool:
    """Whether the persistent mode applies; warns when "auto" falls back."""
    from optimex.persistent import persistent_solver

    if persistent is False:
        return False
    if persistent not in (True, "auto"):
        raise ValueError('persistent must be "auto", True, or False.')
    solver, reason = persistent_solver(solver_name)
    other = set(model_kwargs) - _FORMULATION_KWARGS - {"backend"}
    if solver is not None and other:
        solver, reason = None, f"the options {sorted(other)} need a new model per sample"
    if solver is None:
        if persistent is True:
            raise RuntimeError(f"persistent=True, but {reason}.")
        logger.warning(
            f"Building and solving every sample from scratch, because {reason}. "
            "With Gurobi or HiGHS, optimex updates one solver model in place, which "
            "is faster for large models. Pass persistent=False to silence this "
            "warning."
        )
        return False
    return True


def _totals(model, inputs) -> Tuple[dict, Optional[float]]:
    """Cumulative impact per category and discounted total cost, in real units."""
    import pyomo.environ as pyo

    scales = getattr(model, "scales", {})
    foreground = scales.get("foreground", 1.0)
    categories = scales.get("characterization", {})
    impacts = {}
    for category in model.CATEGORY:
        try:
            value = pyo.value(model.total_impact[category])
        except KeyError:  # no flow is characterized in this category
            value = 0.0
        impacts[category] = float(value) * foreground * categories.get(category, 1.0)
    cost = None
    if inputs.intermediate_flow_prices:
        cost = float(pyo.value(model.total_cost)) * foreground * scales.get("cost", 1.0)
    return impacts, cost


_FAST_OUTPUTS = {"installation", "operation"}


def _solve_chunk(start, inputs_list, config, progress=None):
    """
    Solve consecutive samples with one persistent solver session.

    With a persistent solver, samples are assembled by an
    `IncrementalFormulation` of the base inputs when possible (only results
    computed from the variables are requested), else from their full
    formulation.
    """
    from optimex.formulation import build_formulation
    from optimex.incremental import IncrementalFormulation
    from optimex.persistent import PersistentModel

    if config.get("worker"):
        import sys

        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    model_kwargs = config["model_kwargs"]
    formulation_kwargs = {k: v for k, v in model_kwargs.items() if k in _FORMULATION_KWARGS}
    persistent_model = incremental = scales = None
    if config["persistent"]:
        persistent_model = PersistentModel(config["solver_name"], config["solver_options"])
        reference = config["reference"]
        fast = (
            set(config["outputs"]) <= _FAST_OUTPUTS
            and config["extract"] is None
            and not config["keep_models"]
        )
        if fast:
            incremental = IncrementalFormulation(reference, **formulation_kwargs)
            scales = incremental.scales
        else:
            scales = build_formulation(reference, **formulation_kwargs).scales

    rows, models = [], []
    for offset, inputs in enumerate(inputs_list):
        index = start + offset
        row = {
            "objective": np.nan, "status": "optimal", "impacts": {}, "cost": None,
            **{name: None for name in OUTPUTS}, "extracted": None,
        }
        model = None
        sample_lp = incremental.build(inputs) if incremental is not None else None
        if sample_lp is not None:
            termination, x = persistent_model.solve_arrays(sample_lp.lp)
            if termination != "optimal":
                row["status"] = termination
            else:
                impacts, cost = incremental.totals(
                    sample_lp, x, priced=bool(inputs.intermediate_flow_prices)
                )
                row.update(
                    objective=incremental.objective_value(sample_lp, x),
                    impacts=impacts,
                    cost=cost,
                )
                if "installation" in config["outputs"]:
                    row["installation"] = incremental.installation(x)
                if "operation" in config["outputs"]:
                    row["operation"] = PostProcessor(incremental.model(x)).get_operation()
            rows.append(row)
            models.append(None)
            if progress is not None:
                progress.update(1)
            continue
        if persistent_model is not None:
            formulation = build_formulation(inputs, scales=scales, **formulation_kwargs)
            model, termination = persistent_model.solve(formulation, name=f"sample_{index}")
            if model is None:
                row["status"] = termination
            else:
                objective = optimizer._real_objective(model)
        else:
            try:
                model = optimizer.create_model(inputs, name=f"sample_{index}", **model_kwargs)
                model, objective, _ = optimizer.solve_model(
                    model,
                    solver_name=config["solver_name"],
                    solver_options=config["solver_options"],
                    tee=False,
                )
            except _NO_SOLUTION_ERRORS as exc:
                match = re.search(r"termination: (\w+)", str(exc))
                if match:
                    row["status"] = match.group(1)
                elif type(exc).__name__ == "NoFeasibleSolutionError":
                    row["status"] = "infeasible"
                else:
                    row["status"] = str(exc).split("\n")[0]
                model = None
        if model is not None:
            impacts, cost = _totals(model, inputs)
            row.update(objective=objective, impacts=impacts, cost=cost)
            if config["outputs"]:
                pp = PostProcessor(model)
                for name in config["outputs"]:
                    row[name] = getattr(pp, f"get_{name}")()
            if config["extract"] is not None:
                row["extracted"] = config["extract"](model)
        rows.append(row)
        models.append(model if config["keep_models"] else None)
        if progress is not None:
            progress.update(1)
    return start, rows, models


def _solve_all(
    inputs_list,
    solver_name,
    solver_options,
    keep_models,
    progress,
    description,
    model_kwargs,
    persistent="auto",
    base=None,
    outputs=("installation",),
    extract=None,
    n_jobs=1,
    labels=None,
    index_name="Variant",
) -> SeriesResults:
    import os

    outputs = tuple(outputs or ())
    unknown = set(outputs) - set(OUTPUTS)
    if unknown:
        raise ValueError(f"Unknown outputs {sorted(unknown)}; use {OUTPUTS}.")
    n_jobs = os.cpu_count() if n_jobs in (None, -1) else int(n_jobs)
    n_jobs = max(1, min(n_jobs, len(inputs_list)))
    if n_jobs > 1 and keep_models:
        raise ValueError("keep_models=True needs n_jobs=1: solved models stay in their worker.")

    use_persistent = _use_persistent(solver_name, persistent, model_kwargs)

    solver_options = dict(solver_options or {})
    if n_jobs > 1 and "gurobi" in solver_name and "Threads" not in solver_options:
        # Share the cores between the workers instead of oversubscribing them
        solver_options["Threads"] = max(1, (os.cpu_count() or 1) // n_jobs)
    config = {
        "solver_name": solver_name,
        "solver_options": solver_options,
        "persistent": use_persistent,
        # All variants share the scaling of this reference
        "reference": base if base is not None else inputs_list[0],
        "model_kwargs": model_kwargs,
        "outputs": outputs,
        "extract": extract,
        "keep_models": keep_models,
    }

    bar = tqdm(total=len(inputs_list), desc=description) if progress else None
    try:
        if n_jobs == 1:
            _, rows, models = _solve_chunk(0, inputs_list, config, bar)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            # One chunk per worker: each sets up its solver session (and, for the
            # incremental path, the base formulation) once
            bounds = np.linspace(0, len(inputs_list), n_jobs + 1)
            bounds = np.unique(bounds.astype(int))
            worker_config = {**config, "worker": True}
            parts = {}
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(_solve_chunk, int(a), inputs_list[a:b], worker_config): b - a
                    for a, b in zip(bounds[:-1], bounds[1:])
                }
                for future in as_completed(futures):
                    start, chunk_rows, chunk_models = future.result()
                    parts[start] = (chunk_rows, chunk_models)
                    if bar is not None:
                        bar.update(futures[future])
            rows = [row for start in sorted(parts) for row in parts[start][0]]
            models = [m for start in sorted(parts) for m in parts[start][1]]
    finally:
        if bar is not None:
            bar.close()
    return _results(rows, models if keep_models else None, labels, index_name)


def solve_series(
    variants: Union[Mapping[object, OptimizationModelInputs], Sequence[OptimizationModelInputs]],
    base: Optional[OptimizationModelInputs] = None,
    solver_name: str = "glpk",
    solver_options: Optional[dict] = None,
    keep_models: bool = False,
    progress: bool = True,
    persistent="auto",
    outputs: Sequence[str] = ("installation",),
    extract=None,
    n_jobs: int = 1,
    **model_kwargs,
) -> SeriesResults:
    """
    Solve a series of related models, e.g. a sweep or a set of scenarios.

    Parameters
    ----------
    variants : dict or sequence of OptimizationModelInputs
        The inputs to solve. With a dict, its keys label the results (e.g. the
        budget of each point of a sweep); with a sequence, results are numbered.
        Make variants with ``base.model_copy(update={...})``: what they share
        with the base inputs does not have to be rebuilt.
    base : OptimizationModelInputs, optional
        The inputs the variants were derived from. Its formulation is built once
        and all variants share its scaling. Defaults to the first variant.
    solver_name, solver_options :
        As in `optimizer.solve_model`.
    keep_models : bool
        Keep the solved models in the results. Default False saves memory.
    progress : bool
        Show a progress bar.
    persistent : {"auto", True, False}
        Assemble the variants from the base formulation and load them straight
        into Gurobi or HiGHS, instead of building and solving each model on its
        own (see the module description). ``"auto"`` (default) does so when
        `solver_name` is Gurobi or HiGHS and warns otherwise; True raises instead
        of falling back; False always builds full models. HiGHS also starts each
        solve from the previous solution; when several optima exist, this can
        return a different one with the same objective.
    outputs : sequence of str
        Which PostProcessor tables to keep per variant: any of
        ``"installation"`` (default), ``"operation"``, and ``"production"``.
        Objective, cumulative impacts, and total cost are always kept. For large
        models, extracting tables can take as long as solving, so keep only what
        is needed.
    extract : callable, optional
        ``extract(model)`` is called with every solved model, and its return
        values are collected in `SeriesResults.extracted`. With ``n_jobs > 1``
        it must be a module-level function, so it can be sent to the workers.
    n_jobs : int
        Number of worker processes; -1 uses all cores. Each worker holds its own
        model, so memory use grows with `n_jobs`. Gurobi's threads are divided
        between the workers unless ``Threads`` is set.
    **model_kwargs :
        Passed to `optimizer.create_model`, e.g. ``objective_category``,
        ``objective``, ``backend``, or ``aggregate_vintages``.

    Returns
    -------
    SeriesResults

    Notes
    -----
    The fast path applies to variants that differ from `base` in exchange
    amounts (by one factor per exchange), background inventories,
    characterization factors, prices, discounting, demand, impact limits, or
    deployment and operation limits, when only ``"installation"`` and
    ``"operation"`` tables are requested and neither `extract` nor
    `keep_models` is used. Other variants are built in full automatically.

    Examples
    --------
    Trace the cost of a tightening carbon budget:

    >>> budgets = np.linspace(lowest, highest, 10)
    >>> sweep = optimizer.solve_series(
    ...     {b: inputs.model_copy(update={
    ...         "cumulative_category_impact_limits": {"climate_change": b}})
    ...      for b in budgets},
    ...     base=inputs, objective="cost", solver_name="highs",
    ... )
    >>> sweep.objective  # lowest cost per budget
    """
    if isinstance(variants, Mapping):
        labels, inputs_list = list(variants.keys()), list(variants.values())
    else:
        inputs_list = list(variants)
        labels = list(range(len(inputs_list)))
    if not inputs_list:
        raise ValueError("variants is empty.")
    return _solve_all(
        inputs_list, solver_name, solver_options, keep_models, progress,
        "Solving series", model_kwargs, persistent, base, outputs, extract, n_jobs,
        labels=labels, index_name="Variant",
    )
