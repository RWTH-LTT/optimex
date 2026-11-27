"""
Pyomo backend: the optimization model as a Pyomo `ConcreteModel`.

`create_model(..., backend="pyomo")` (the default) builds the model from the
backend-independent formulation (`optimex.formulation`) with Pyomo, and
`solve_model()` solves it through Pyomo's solver interfaces. Pyomo is the
backend to use for anything that works on the model itself: fixing variables,
adding constraints, IIS files, or Pyomo's persistent and direct solvers.

Component names (`var_installation`, `ProductDemandFulfillment`,
`total_impact`, ...) and the scaling conventions are described in
`optimex.optimizer`.
"""

from typing import Any, Dict, Optional, Tuple

import pyomo.environ as pyo
from loguru import logger
from pyomo.contrib.iis import write_iis
from pyomo.core.expr import LinearExpression
from pyomo.opt import ProblemFormat
from pyomo.opt.results.results_ import SolverResults

from optimex.converter import OptimizationModelInputs
from optimex.formulation import VARIABLE_FAMILIES, Formulation, build_formulation


def linear_sum(coefficients, variables):
    """
    Build `sum(c * x)` as a single flat LinearExpression.

    Accumulating the terms with `+=` creates one intermediate expression object
    per term; handing Pyomo the coefficient and variable lists directly does not.
    """
    if not variables:
        return 0
    return LinearExpression(
        constant=0, linear_coefs=list(coefficients), linear_vars=list(variables)
    )


class LinearExpressionMap:
    """
    Linear expressions stored as coefficients, indexed like a Pyomo `Expression`.

    Used for the reporting expressions (per-process flows and impacts) that no
    constraint references: there are about a million of them in large models,
    and creating a Pyomo expression object for each one dominated model
    construction. `model.name[key]` builds the Pyomo expression on request, so
    `pyo.value(model.name[key])` works as before; `model.name.value(key)`
    evaluates the solved value directly from the coefficients.
    """

    def __init__(self, terms_by_key: dict, variables: list):
        self._terms = terms_by_key
        self._variables = variables

    def __getitem__(self, key):
        terms = self._terms[key]
        return linear_sum(terms.values(), [self._variables[k] for k in terms])

    def value(self, key) -> float:
        """Value of the expression at `key` for the current variable values."""
        variables = self._variables
        return sum(
            coefficient * (variables[k].value or 0.0)
            for k, coefficient in self._terms[key].items()
        )

    def __contains__(self, key) -> bool:
        return key in self._terms

    def __iter__(self):
        return iter(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def keys(self):
        return self._terms.keys()

    def items(self):
        return ((key, self[key]) for key in self._terms)


class LazyExpressionMap:
    """
    Expressions indexed like a Pyomo `Expression`, built on first access.

    Used for the reporting totals whose dense index (e.g. process x elementary
    flow x time) dwarfs the rest of the model but that only flow limits and
    post-processing read. `model.name[key]` returns the expression, so
    `pyo.value(model.name[key])` works as for an indexed `Expression`.
    """

    def __init__(self, rule):
        self._rule = rule
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = self._rule(*key)
        return self._cache[key]

    def __getstate__(self):
        # The rule is a closure over model-building locals; built expressions
        # are kept so that a pickled, solved model still evaluates them.
        return {"_rule": None, "_cache": self._cache}

    def __setstate__(self, state):
        self.__dict__.update(state)


def create_pyomo_model(
    inputs: OptimizationModelInputs,
    name: str,
    objective_category: Optional[str] = None,
    objective: str = "environmental",
    aggregate_vintages: bool = False,
    debug_path: Optional[str] = None,
) -> pyo.ConcreteModel:
    """Build the Pyomo model; see `optimizer.create_model` for the arguments."""
    formulation = build_formulation(
        inputs,
        objective_category=objective_category,
        objective=objective,
        aggregate_vintages=aggregate_vintages,
    )
    model = _pyomo_model(formulation, name)
    if debug_path is not None:
        model.write(
            debug_path,
            io_options={"symbolic_solver_labels": True},
            format=ProblemFormat.cpxlp,
        )
    return model


def _pyomo_model(formulation: Formulation, name: str) -> pyo.ConcreteModel:
    """Emit the Pyomo model of a formulation."""
    scaled_inputs = formulation.inputs
    model = pyo.ConcreteModel(name=name)
    model._objective = formulation.objective
    model._objective_category = formulation.objective_category
    model.scales = formulation.scales

    logger.info("Creating sets")
    model.PROCESS = pyo.Set(
        doc="Set of processes (or activities), indexed by p",
        initialize=scaled_inputs.PROCESS,
    )
    model.PRODUCT = pyo.Set(
        doc="Set of foreground products, indexed by r",
        initialize=scaled_inputs.PRODUCT,
    )
    model.INTERMEDIATE_FLOW = pyo.Set(
        doc="Set of background products (intermediate flows), indexed by i",
        initialize=scaled_inputs.INTERMEDIATE_FLOW,
    )
    model.ELEMENTARY_FLOW = pyo.Set(
        doc="Set of elementary flows, indexed by e",
        initialize=scaled_inputs.ELEMENTARY_FLOW,
    )
    model.FLOW = pyo.Set(
        initialize=lambda m: m.PRODUCT | m.INTERMEDIATE_FLOW | m.ELEMENTARY_FLOW,
        doc="Set of all flows, indexed by f",
    )
    model.CATEGORY = pyo.Set(
        doc="Set of impact categories, indexed by c", initialize=scaled_inputs.CATEGORY
    )
    model.BACKGROUND_ID = pyo.Set(
        doc="Set of identifiers of the prospective background databases, indexed by b",
        initialize=scaled_inputs.BACKGROUND_ID,
    )
    model.PROCESS_TIME = pyo.Set(
        doc="Set of process time points, indexed by tau",
        initialize=scaled_inputs.PROCESS_TIME,
    )
    model.SYSTEM_TIME = pyo.Set(
        doc="Set of system time points, indexed by t",
        initialize=scaled_inputs.SYSTEM_TIME,
    )

    # Parameters: the data the model was built from, kept for inspection and
    # post-processing (the constraints themselves are built from plain numbers).
    logger.info("Creating parameters")
    model.process_names = pyo.Param(
        model.PROCESS,
        within=pyo.Any,
        doc="Names of the processes",
        default=None,
        initialize=scaled_inputs.process_names,
    )
    model.demand = pyo.Param(
        model.PRODUCT,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="time-explicit external demand vector d",
        default=0,
        initialize=scaled_inputs.demand,
    )
    model.foreground_technosphere = pyo.Param(
        model.PROCESS,
        model.INTERMEDIATE_FLOW,
        model.PROCESS_TIME,
        within=pyo.Reals,
        doc="time-explicit foreground technosphere tensor A (background flows)",
        default=0,
        initialize=scaled_inputs.foreground_technosphere,
    )
    model.foreground_biosphere = pyo.Param(
        model.PROCESS,
        model.ELEMENTARY_FLOW,
        model.PROCESS_TIME,
        within=pyo.Reals,
        doc="time-explicit foreground biosphere tensor B",
        default=0,
        initialize=scaled_inputs.foreground_biosphere,
    )
    model.foreground_production = pyo.Param(
        model.PROCESS,
        model.PRODUCT,
        model.PROCESS_TIME,
        within=pyo.Reals,
        doc="time-explicit foreground production tensor F",
        default=0,
        initialize=scaled_inputs.foreground_production,
    )
    model.internal_demand_technosphere = pyo.Param(
        model.PROCESS,
        model.PRODUCT,
        model.PROCESS_TIME,
        within=pyo.Reals,
        doc="time-explicit internal demand tensor A^{internal}",
        default=0,
        initialize=scaled_inputs.internal_demand_technosphere,
    )
    model.background_inventory = pyo.Param(
        model.BACKGROUND_ID,
        model.INTERMEDIATE_FLOW,
        model.ELEMENTARY_FLOW,
        within=pyo.Reals,
        doc="prospective background inventory tensor G",
        default=0,
        initialize=scaled_inputs.background_inventory,
    )
    model.mapping = pyo.Param(
        model.BACKGROUND_ID,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="time-explicit background mapping tensor M",
        default=0,
        initialize=scaled_inputs.mapping,
    )
    model.characterization = pyo.Param(
        model.CATEGORY,
        model.ELEMENTARY_FLOW,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="time-explicit characterization tensor Q",
        default=0,
        initialize=scaled_inputs.characterization,
    )
    model.operation_flow = pyo.Param(
        model.PROCESS,
        model.FLOW,
        within=pyo.Binary,
        doc="operation flow matrix",
        default=0,
        initialize=scaled_inputs.operation_flow,
    )
    model.process_operation_start = pyo.Param(
        model.PROCESS,
        within=pyo.NonNegativeIntegers,
        doc="start time of process operation",
        default=0,
        initialize={k: v[0] for k, v in scaled_inputs.operation_time_limits.items()},
    )
    model.process_operation_end = pyo.Param(
        model.PROCESS,
        within=pyo.NonNegativeIntegers,
        doc="end time of process operation",
        default=0,
        initialize={k: v[1] for k, v in scaled_inputs.operation_time_limits.items()},
    )
    for param, default_name, doc in (
        ("process_deployment_limits_max", None, "maximum time specific process deployment limit S_max"),
        ("process_deployment_limits_min", None, "minimum time specific process deployment limit S_min"),
        ("process_operation_limits_max", None, "maximum time specific process operation limit O_max"),
        ("process_operation_limits_min", None, "minimum time specific process operation limit O_min"),
    ):
        model.add_component(
            param,
            pyo.Param(
                model.PROCESS,
                model.SYSTEM_TIME,
                within=pyo.Reals,
                doc=doc,
                default=getattr(scaled_inputs, f"{param}_default"),
                initialize=getattr(scaled_inputs, param) or {},
            ),
        )
    for param, doc in (
        ("cumulative_process_limits_max", "maximum cumulative process limit S_max,cum"),
        ("cumulative_process_limits_min", "minimum cumulative process limit S_min,cum"),
    ):
        model.add_component(
            param,
            pyo.Param(
                model.PROCESS,
                within=pyo.Reals,
                doc=doc,
                default=getattr(scaled_inputs, f"{param}_default"),
                initialize=getattr(scaled_inputs, param) or {},
            ),
        )
    model.process_coupling = pyo.Param(
        model.PROCESS,
        model.PROCESS,
        within=pyo.NonNegativeReals,
        doc="coupling matrix",
        initialize=scaled_inputs.process_coupling or {},
        default=0,
    )
    model.existing_capacity = pyo.Param(
        model.PROCESS,
        pyo.Any,  # Installation year (any year before SYSTEM_TIME)
        within=pyo.NonNegativeReals,
        doc="Existing capacity (process, installation_year) -> amount",
        initialize=scaled_inputs.existing_capacity or {},
        default=0,
    )
    model.cumulative_category_impact_limits = pyo.Param(
        model.CATEGORY,
        within=pyo.Reals,
        doc="cumulative maximum impact limit per category",
        default=float("inf"),
        initialize=scaled_inputs.cumulative_category_impact_limits or {},
    )

    # Plain data used by post-processing
    model._existing_capacity_dict = formulation.existing_capacity
    model._mapped_inventory_by_flow = formulation.mapped_inventory_by_flow
    model._technosphere_vintage_overrides = formulation.technosphere_vintage_overrides
    model._biosphere_vintage_overrides = formulation.biosphere_vintage_overrides
    model._production_vintage_overrides = formulation.production_vintage_overrides
    model._technosphere_overrides_index = frozenset(
        (k[0], k[1]) for k in formulation.technosphere_vintage_overrides
    )
    model._biosphere_overrides_index = frozenset(
        (k[0], k[1]) for k in formulation.biosphere_vintage_overrides
    )
    model._production_overrides_index = frozenset(
        (k[0], k[1]) for k in formulation.production_vintage_overrides
    )
    model._category_impact_limits = scaled_inputs.category_impact_limits or {}
    model._flow_limits_max = scaled_inputs.flow_limits_max or {}
    model._flow_limits_min = scaled_inputs.flow_limits_min or {}
    model._cumulative_flow_limits_max = scaled_inputs.cumulative_flow_limits_max or {}
    model._cumulative_flow_limits_min = scaled_inputs.cumulative_flow_limits_min or {}
    model._aggregated_processes = formulation.aggregated_processes
    model._active_vintages_by_process_time = formulation.active_vintages_by_process_time

    model.ACTIVE_VINTAGE_TIME = pyo.Set(
        dimen=3,
        initialize=formulation.active_vintage_time,
        doc="Set of (process, vintage, time) tuples where vintage is in operation phase at time t",
    )
    model.OPERATION_VINTAGE_TIME = pyo.Set(
        dimen=3,
        initialize=formulation.operation_vintage_time,
        doc="Active (process, vintage, time) tuples with per-vintage dispatch",
    )

    # Variables
    logger.info("Creating variables")
    lower, upper = formulation.lower, formulation.upper
    installation_index = formulation.variable_index["var_installation"]
    model.var_installation = pyo.Var(
        model.PROCESS,
        model.SYSTEM_TIME,
        within=pyo.NonNegativeReals,
        bounds=lambda m, p, t: (
            lower[installation_index[p, t]],
            upper[installation_index[p, t]],
        ),
        doc="Installation of the process",
    )
    model.var_operation = pyo.Var(
        model.OPERATION_VINTAGE_TIME,
        within=pyo.NonNegativeReals,
        doc="Operational activity level per vintage (process, vintage, time)",
    )
    model.var_operation_total = pyo.Var(
        formulation.keys_of("var_operation_total"),
        within=pyo.NonNegativeReals,
        doc="Total operation of processes whose vintages operate identically",
    )
    model.var_capacity_total = pyo.Var(
        formulation.keys_of("var_capacity_total"),
        within=pyo.NonNegativeReals,
        doc="Units available of processes with aggregated vintages",
    )

    # Pyomo variables in formulation order, so that a coefficient dict's keys
    # index straight into this list
    variables = [None] * formulation.n_variables
    for family in VARIABLE_FAMILIES:
        component = getattr(model, family)
        for key, index in formulation.variable_index[family].items():
            variables[index] = component[key]

    def from_terms(terms: dict):
        """Flat LinearExpression from a {variable index: coefficient} dict."""
        return linear_sum(terms.values(), [variables[k] for k in terms])

    # Constraints
    logger.info("Creating constraints")
    for constraint_name, constraint in formulation.constraints.items():
        rows = constraint.rows
        sense = constraint.sense

        def rule(m, *index, rows=rows, sense=sense):
            terms, rhs = rows[index[0] if len(index) == 1 else index]
            if not terms:
                # A constant row that cannot hold (see build_formulation)
                return pyo.Constraint.Infeasible
            body = from_terms(terms)
            if sense == "<=":
                return body <= rhs
            if sense == ">=":
                return body >= rhs
            return body == rhs

        model.add_component(constraint_name, pyo.Constraint(list(rows), rule=rule))

    # Reporting expressions. The large per-process ones are kept as coefficients
    # and only become Pyomo expressions when indexed.
    expressions = formulation.expressions
    for expression_name in (
        "scaled_technosphere_flow",
        "scaled_biosphere_flow",
        "scaled_internal_demand_flow",
        "specific_impact",
    ):
        setattr(
            model, expression_name, LinearExpressionMap(expressions[expression_name], variables)
        )
    model.total_impact = pyo.Expression(
        model.CATEGORY, rule=lambda m, c: from_terms(expressions["total_impact"][c])
    )
    model.time_specific_impact = pyo.Expression(
        model.CATEGORY,
        model.SYSTEM_TIME,
        rule=lambda m, c, t: from_terms(expressions["time_specific_impact"][c, t]),
    )
    model.total_product_flow = pyo.Expression(
        model.PRODUCT,
        model.SYSTEM_TIME,
        rule=lambda m, r, t: from_terms(expressions["total_product_flow"][r, t]),
    )
    for expression_name, function in formulation.lazy_expressions.items():
        setattr(
            model,
            expression_name,
            LazyExpressionMap(lambda *key, f=function: from_terms(f(*key))),
        )

    # Costs of the background purchases
    model.intermediate_flow_prices = pyo.Param(
        model.INTERMEDIATE_FLOW,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="time-specific market price of intermediate flows [SCALED]",
        default=0,
        initialize=formulation.prices,
    )
    model.discount_factor = pyo.Param(
        model.SYSTEM_TIME,
        within=pyo.NonNegativeReals,
        doc="discount factor of costs at time t",
        initialize=formulation.discount_factor,
    )
    model.COST_PROCESS = pyo.Set(
        initialize=formulation.cost_processes,
        doc="Processes purchasing at least one priced intermediate flow",
    )
    model.installation_cost = pyo.Expression(
        model.COST_PROCESS,
        model.SYSTEM_TIME,
        rule=lambda m, p, t: from_terms(expressions["installation_cost"][p, t]),
        doc="undiscounted cost of installation-driven purchases [SCALED]",
    )
    model.operation_cost = pyo.Expression(
        model.COST_PROCESS,
        model.SYSTEM_TIME,
        rule=lambda m, p, t: from_terms(expressions["operation_cost"][p, t]),
        doc="undiscounted cost of operation-driven purchases [SCALED]",
    )
    model.total_cost = pyo.Expression(
        expr=from_terms(expressions["total_cost"][None]),
        doc="discounted total cost [SCALED]",
    )

    model.OBJ = pyo.Objective(
        sense=pyo.minimize, expr=from_terms(formulation.objective_terms)
    )
    return model


def solve_pyomo_model(
    model: pyo.ConcreteModel,
    solver_name: str = "gurobi",
    solver_args: Optional[Dict[str, Any]] = None,
    solver_options: Optional[Dict[str, Any]] = None,
    tee: bool = True,
    compute_iis: bool = False,
    **solve_kwargs: Any,
) -> Tuple[pyo.ConcreteModel, SolverResults]:
    """Solve a Pyomo model; see `optimizer.solve_model` for the arguments."""
    # 1) Instantiate solver
    solver_args = solver_args or {}
    solver = pyo.SolverFactory(solver_name, **solver_args)
    if solver_options:
        for opt, val in solver_options.items():
            solver.options[opt] = val

    # 2) Solve model
    results = solver.solve(model, tee=tee, **solve_kwargs)

    termination = results.solver.termination_condition

    # 3) Check termination and handle non-optimal outcomes
    if termination != pyo.TerminationCondition.optimal:
        msg = f"Solver [{solver_name}] termination: {termination}"

        if termination == pyo.TerminationCondition.infeasible:
            if compute_iis:
                try:
                    write_iis(model, iis_file_name="model_iis.ilp", solver=solver)
                    msg += " — IIS written to model_iis.ilp"
                except Exception as e:
                    msg += f" — IIS generation failed: {e}"
            else:
                msg += " — rerun with compute_iis=True to diagnose"
        elif termination == pyo.TerminationCondition.unbounded or (
            termination == pyo.TerminationCondition.other
            and "unbounded" in str(results).lower()
        ):
            msg += (
                " — the model is unbounded. Common cause: installation-driven"
                " flows (non-operation technosphere/biosphere exchanges) that produce"
                " a net-negative impact allow var_installation to grow without bound."
                " Fix by setting finite process_deployment_limits_max for affected"
                " processes, or ensure installation-driven flows do not create"
                " a negative-impact incentive for over-installation."
            )
        elif termination == pyo.TerminationCondition.other:
            msg += (
                " — solver returned non-standard termination. Run with tee=True"
                " for detailed solver output. Common causes: unbounded model"
                " (set finite process_deployment_limits_max) or numerical issues."
            )

        # Append full solver details
        msg += f"\n\nFull solver results:\n{results}"

        raise RuntimeError(msg)

    model.solutions.load_from(results)
    # Per-vintage split of aggregated operation is recomputed on request
    model._disaggregated_operation = {}
    _fill_unreferenced_variables(model)
    logger.info(f"Solver [{solver_name}] termination: {termination}")

    # 4) (Optional) Denormalize duals
    scales = getattr(model, "scales", {})
    fg_scale = scales.get("foreground", 1.0)
    catscales = scales.get("characterization", {})

    if hasattr(model, "dual"):
        denorm_duals: Dict[Any, float] = {}
        # Example: demand constraint duals
        for idx, con in getattr(model, "demand_constraint", {}).items():
            λ = model.dual.get(con, None)
            if λ is not None:
                denorm_duals[f"demand_{idx}"] = λ * fg_scale
        # Example: impact constraint duals
        for c, con in getattr(model, "category_impact_constraint", {}).items():
            μ = model.dual.get(con, None)
            if μ is not None:
                denorm_duals[f"impact_{c}"] = μ * catscales.get(c, 1.0)
        logger.info(f"Denormalized duals: {denorm_duals}")

    return model, results


def _fill_unreferenced_variables(model: pyo.ConcreteModel) -> None:
    """
    Give a value to variables that appear in no constraint or objective.

    Solvers only receive referenced variables, so e.g. the installation of a process
    in a year in which none of its flows fall inside the horizon comes back without
    a value. It influences nothing, so its lower bound (or zero) is optimal.
    """
    for component in model.component_objects(pyo.Var, descend_into=True):
        for var in component.values():
            if var.value is None:
                var.set_value(
                    var.lb if var.lb is not None else 0.0, skip_validation=True
                )
