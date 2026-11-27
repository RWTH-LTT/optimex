"""
linopy backend: the optimization model built from arrays instead of Pyomo objects.

`create_model(..., backend="linopy")` returns a `LinopyModel`. It holds the same
formulation as the Pyomo model (`optimex.formulation`), with every constraint
block passed to linopy as coefficient and variable arrays, which avoids one
Python object per term and builds large models several times faster. Solving
it with `optimizer.solve_model()` stores the solution on the model as plain
Python data under the attribute names of the Pyomo model, so the
`PostProcessor`, `operation_value()` and `save_solved_model()` work the same
way.

linopy supports HiGHS, Gurobi, CPLEX, Xpress, GLPK, CBC, SCIP, MOSEK and COPT.
"""

import math
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from optimex.converter import OptimizationModelInputs
from optimex.formulation import VARIABLE_FAMILIES, Formulation, build_formulation

# Pyomo solver names that map onto a linopy solver
_SOLVER_ALIASES = {
    "gurobi_direct": "gurobi",
    "gurobi_direct_v2": "gurobi",
    "gurobi_persistent": "gurobi",
    "gurobi_persistent_v2": "gurobi",
    "appsi_gurobi": "gurobi",
    "appsi_highs": "highs",
    "cplex_direct": "cplex",
    "appsi_cplex": "cplex",
    "xpress_direct": "xpress",
}
# Solvers linopy can talk to in memory instead of through an LP file
_DIRECT_API_SOLVERS = {"gurobi", "highs", "mosek", "xpress", "copt"}
_SIGNS = {"<=": "<=", ">=": ">=", "==": "="}


def _set_model_data(target, formulation: Formulation) -> None:
    """Sets, parameters and post-processing data, named as on the Pyomo model."""
    inputs = formulation.inputs
    target.scales = formulation.scales
    target._objective = formulation.objective
    target._objective_category = formulation.objective_category

    target.PROCESS = list(inputs.PROCESS)
    target.PRODUCT = list(inputs.PRODUCT)
    target.INTERMEDIATE_FLOW = list(inputs.INTERMEDIATE_FLOW)
    target.ELEMENTARY_FLOW = list(inputs.ELEMENTARY_FLOW)
    target.FLOW = target.PRODUCT + target.INTERMEDIATE_FLOW + target.ELEMENTARY_FLOW
    target.CATEGORY = list(inputs.CATEGORY)
    target.BACKGROUND_ID = list(inputs.BACKGROUND_ID)
    target.PROCESS_TIME = list(inputs.PROCESS_TIME)
    target.SYSTEM_TIME = list(inputs.SYSTEM_TIME)
    target.ACTIVE_VINTAGE_TIME = list(formulation.active_vintage_time)
    target.OPERATION_VINTAGE_TIME = list(formulation.operation_vintage_time)
    target.COST_PROCESS = list(formulation.cost_processes)

    target.process_names = _ValueMap(inputs.process_names or {}, None)
    target.demand = _ValueMap(inputs.demand or {})
    target.foreground_technosphere = _ValueMap(inputs.foreground_technosphere)
    target.foreground_biosphere = _ValueMap(inputs.foreground_biosphere)
    target.foreground_production = _ValueMap(inputs.foreground_production)
    target.internal_demand_technosphere = _ValueMap(inputs.internal_demand_technosphere)
    target.background_inventory = _ValueMap(inputs.background_inventory)
    target.mapping = _ValueMap(inputs.mapping)
    target.characterization = _ValueMap(inputs.characterization)
    target.operation_flow = _ValueMap(inputs.operation_flow, 0)
    target.process_operation_start = _ValueMap(
        {p: phase[0] for p, phase in formulation.operation_phase.items()}, 0
    )
    target.process_operation_end = _ValueMap(
        {p: phase[1] for p, phase in formulation.operation_phase.items()}, 0
    )
    target.existing_capacity = _ValueMap(formulation.existing_capacity)
    target.intermediate_flow_prices = _ValueMap(formulation.prices)
    target.discount_factor = _ValueMap(formulation.discount_factor)

    target._existing_capacity_dict = formulation.existing_capacity
    target._mapped_inventory_by_flow = formulation.mapped_inventory_by_flow
    target._technosphere_vintage_overrides = formulation.technosphere_vintage_overrides
    target._biosphere_vintage_overrides = formulation.biosphere_vintage_overrides
    target._production_vintage_overrides = formulation.production_vintage_overrides
    target._production_overrides_index = frozenset(
        (k[0], k[1]) for k in formulation.production_vintage_overrides
    )
    target._aggregated_processes = formulation.aggregated_processes
    target._active_vintages_by_process_time = formulation.active_vintages_by_process_time
    target._category_impact_limits = inputs.category_impact_limits or {}


class LinopyModel:
    """
    An optimex model built with linopy, ready for `optimizer.solve_model()`.

    Sets and parameters (`PROCESS`, `SYSTEM_TIME`, `demand`, ...) are available
    under the same names as on the Pyomo model.

    Attributes
    ----------
    name : str
        The model name.
    model : linopy.Model
        The linopy model, e.g. for inspection or writing it to a file.
    formulation : Formulation
        The backend-independent formulation it was built from.
    scales : dict
        The scaling factors, as on the Pyomo model.
    """

    def __init__(self, name: str, model, formulation: Formulation, infeasible_rows):
        self.name = name
        self.model = model
        self.formulation = formulation
        self._infeasible_rows = infeasible_rows
        self.solved = False
        _set_model_data(self, formulation)

    def load_solution(self, values: np.ndarray) -> None:
        """
        Store solved values (in formulation order) on the model, laid out like a
        solved Pyomo model: variables and expressions map keys to numbers, so
        `pyo.value(model.total_impact[c])` and the PostProcessor work as usual.
        """
        formulation = self.formulation
        # Variables
        for family in VARIABLE_FAMILIES:
            setattr(
                self,
                family,
                _ValueMap(
                    {
                        key: float(values[index])
                        for key, index in formulation.variable_index[family].items()
                    }
                ),
            )

        # Expressions
        expressions = formulation.expressions
        for expression_name, terms_by_key in expressions.items():
            if expression_name != "total_cost":
                setattr(self, expression_name, SolvedExpressionMap(terms_by_key, values))
        self.total_cost = SolvedExpressionMap(expressions["total_cost"], values).value(
            None
        )
        for expression_name, function in formulation.lazy_expressions.items():
            setattr(self, expression_name, _SolvedLazyMap(function, values))
        objective = SolvedExpressionMap({None: formulation.objective_terms}, values)
        self.OBJ = objective.value(None)

        self._disaggregated_operation = {}
        self.solved = True


    def __getstate__(self):
        # The linopy model and the formulation are only needed to build and
        # solve; a saved, solved model keeps the results.
        state = dict(self.__dict__)
        state["model"] = None
        state["formulation"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"LinopyModel({self.name!r}, {self.formulation.n_variables} variables, "
            f"{sum(len(c.rows) for c in self.formulation.constraints.values())} "
            "constraints)"
        )


def create_linopy_model(
    inputs: OptimizationModelInputs,
    name: str,
    objective_category: Optional[str] = None,
    objective: str = "environmental",
    aggregate_vintages: bool = False,
    debug_path: Optional[str] = None,
) -> LinopyModel:
    """Build the linopy model; see `optimizer.create_model` for the arguments."""
    try:
        import linopy
        import xarray as xr
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "backend='linopy' needs the linopy package: pip install linopy"
        ) from error

    formulation = build_formulation(
        inputs,
        objective_category=objective_category,
        objective=objective,
        aggregate_vintages=aggregate_vintages,
    )
    model = linopy.Model()

    logger.info("Creating variables")
    lower = np.array(
        [0.0 if lb is None else lb for lb in formulation.lower], dtype=float
    )
    upper = np.array(
        [np.inf if ub is None else ub for ub in formulation.upper], dtype=float
    )
    for family in VARIABLE_FAMILIES:
        indices = list(formulation.variable_index[family].values())
        if not indices:
            continue
        start, stop = indices[0], indices[-1] + 1
        variable = model.add_variables(
            lower=lower[start:stop],
            upper=upper[start:stop],
            coords=[np.arange(stop - start)],
            dims=[f"{family}_index"],
            name=family,
        )
        # linopy numbers variables consecutively in the order they are added,
        # which is the order of the formulation's variable indices
        first = int(variable.labels.values.flat[0])
        if first != start:  # pragma: no cover - would be a linopy change
            raise RuntimeError(
                f"linopy labelled {family} from {first}, expected {start}."
            )

    logger.info("Creating constraints")
    infeasible_rows = []
    for constraint_name, constraint in formulation.constraints.items():
        rows = []
        for key, (terms, rhs) in constraint.rows.items():
            if terms:
                rows.append((terms, rhs))
            else:
                infeasible_rows.append((constraint_name, key, constraint.sense, rhs))
        for bucket, bucket_rows in enumerate(_bucket_by_length(rows)):
            bucket_name = constraint_name if bucket == 0 else f"{constraint_name}#{bucket}"
            expression, rhs = _expression_arrays(bucket_rows, bucket_name, xr)
            model.add_constraints(
                linopy.LinearExpression(expression, model),
                _SIGNS[constraint.sense],
                rhs,
                name=bucket_name,
            )

    objective_terms = formulation.objective_terms
    model.add_objective(
        linopy.LinearExpression(
            xr.Dataset(
                {
                    "coeffs": ("_term", np.fromiter(objective_terms.values(), float)),
                    "vars": ("_term", np.fromiter(objective_terms.keys(), np.int64)),
                    "const": 0.0,
                }
            ),
            model,
        ),
        sense="min",
    )

    linopy_model = LinopyModel(name, model, formulation, infeasible_rows)
    if debug_path is not None:
        model.to_file(debug_path)
    return linopy_model


def _bucket_by_length(rows):
    """
    Group rows by number of terms (powers of two), so a few long rows (e.g. a
    cumulative impact limit) do not pad every short row of their block.
    """
    buckets = {}
    for row in rows:
        buckets.setdefault(max(len(row[0]), 1).bit_length(), []).append(row)
    return [buckets[size] for size in sorted(buckets)]


def _expression_arrays(rows, name, xr):
    """Coefficient/variable arrays (rows x terms, padded with -1) and rhs."""
    width = max(len(terms) for terms, _ in rows)
    coefficients = np.zeros((len(rows), width))
    variables = np.full((len(rows), width), -1, dtype=np.int64)
    rhs = np.empty(len(rows))
    for row, (terms, value) in enumerate(rows):
        n = len(terms)
        variables[row, :n] = np.fromiter(terms.keys(), np.int64, n)
        coefficients[row, :n] = np.fromiter(terms.values(), float, n)
        rhs[row] = value
    dimension = f"{name}_row"
    coords = {dimension: np.arange(len(rows))}
    expression = xr.Dataset(
        {
            "coeffs": ((dimension, "_term"), coefficients),
            "vars": ((dimension, "_term"), variables),
            "const": ((dimension,), np.zeros(len(rows))),
        },
        coords=coords,
    )
    return expression, xr.DataArray(rhs, coords=coords, dims=[dimension])


def solve_linopy_model(
    linopy_model: LinopyModel,
    solver_name: str = "gurobi",
    solver_options: Optional[Dict[str, Any]] = None,
    tee: bool = True,
    compute_iis: bool = False,
    **solve_kwargs: Any,
):
    """Solve a `LinopyModel`; see `optimizer.solve_model` for the arguments."""
    solver = _SOLVER_ALIASES.get(solver_name, solver_name)
    options = dict(solver_options or {})
    if not tee:
        options.update(_quiet_options(solver))
    if "io_api" not in solve_kwargs and solver in _DIRECT_API_SOLVERS:
        solve_kwargs["io_api"] = "direct"

    if linopy_model._infeasible_rows:
        name, key, sense, rhs = linopy_model._infeasible_rows[0]
        raise RuntimeError(
            f"Solver [{solver_name}] termination: infeasible — constraint "
            f"{name}[{key}] reads 0 {sense} {rhs}: no process can contribute to it."
        )

    model = linopy_model.model
    status, termination = model.solve(
        solver_name=solver, progress=False, **solve_kwargs, **options
    )
    results = _results(status, termination, solver)
    if termination != "optimal":
        msg = f"Solver [{solver_name}] termination: {termination}"
        if termination in ("infeasible", "infeasible_or_unbounded") and compute_iis:
            try:
                labels = model.compute_infeasibilities()
                msg += f" — irreducible infeasible set: {model.constraints.print_labels(labels) if hasattr(model.constraints, 'print_labels') else labels}"
            except Exception as error:  # pragma: no cover - solver dependent
                msg += f" — IIS computation failed: {error}"
        elif termination in ("infeasible", "infeasible_or_unbounded"):
            msg += " — rerun with compute_iis=True (Gurobi) to diagnose"
        if "unbounded" in termination:
            msg += (
                " — the model is unbounded. Common cause: installation-driven"
                " flows with a net-negative impact let var_installation grow without"
                " bound. Set finite process_deployment_limits_max for those processes."
            )
        raise RuntimeError(msg)

    logger.info(f"Solver [{solver_name}] termination: {termination}")
    linopy_model.load_solution(_solution_values(linopy_model))
    return linopy_model, results


def _results(status: str, termination: str, solver: str):
    """
    Solver results with the structure of Pyomo's: `results.solver.status` and
    `results.solver.termination_condition` hold Pyomo's enum values.
    """
    import pyomo.environ as pyo

    condition = getattr(pyo.TerminationCondition, termination, None)
    if condition is None:
        condition = {
            "infeasible_or_unbounded": pyo.TerminationCondition.infeasibleOrUnbounded,
            "time_limit": pyo.TerminationCondition.maxTimeLimit,
            "iteration_limit": pyo.TerminationCondition.maxIterations,
        }.get(termination, pyo.TerminationCondition.other)
    solver_status = pyo.SolverStatus.ok if status == "ok" else pyo.SolverStatus.warning
    return SimpleNamespace(
        solver=SimpleNamespace(
            status=solver_status, termination_condition=condition, name=solver
        ),
        status=status,
        termination_condition=termination,
    )


def _quiet_options(solver: str) -> dict:
    return {
        "gurobi": {"OutputFlag": 0},
        "highs": {"output_flag": False},
        "cplex": {},
        "xpress": {"outputlog": 0},
        "copt": {"Logging": 0},
    }.get(solver, {})


def _solution_values(linopy_model: LinopyModel) -> np.ndarray:
    """Solved values in formulation order; unused variables get their lower bound."""
    formulation = linopy_model.formulation
    values = np.array(
        [0.0 if lb is None else lb for lb in formulation.lower], dtype=float
    )
    for family in VARIABLE_FAMILIES:
        indices = list(formulation.variable_index[family].values())
        if not indices:
            continue
        solution = np.asarray(
            linopy_model.model.variables[family].solution.values, dtype=float
        )
        start = indices[0]
        segment = values[start : start + len(solution)]
        found = ~np.isnan(solution)
        segment[found] = solution[found]
    return values


class _ValueMap(dict):
    """Solved or input data by key, with a default for missing keys."""

    def __init__(self, data=(), default=0.0):
        super().__init__(data)
        self._default = default

    def __missing__(self, key):
        return self._default


class SolvedExpressionMap:
    """Solved values of linear expressions stored as coefficients."""

    def __init__(self, terms_by_key: dict, values: np.ndarray):
        self._terms = terms_by_key
        self._values = values

    def value(self, key) -> float:
        values = self._values
        return float(
            math.fsum(c * values[k] for k, c in self._terms[key].items())
        )

    __getitem__ = value

    def __contains__(self, key) -> bool:
        return key in self._terms

    def __iter__(self):
        return iter(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def keys(self):
        return self._terms.keys()

    def items(self):
        return ((key, self.value(key)) for key in self._terms)


class _SolvedLazyMap:
    """Solved values of the large reporting expressions, computed on request."""

    def __init__(self, function, values: np.ndarray):
        self._function = function
        self._values = values
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._cache:
            terms = self._function(*key)
            self._cache[key] = float(
                math.fsum(c * self._values[k] for k, c in terms.items())
            )
        return self._cache[key]

    def __getstate__(self):
        # The function is a closure over the formulation; keep computed values
        return {"_function": None, "_values": None, "_cache": self._cache}

    def __setstate__(self, state):
        self.__dict__.update(state)
