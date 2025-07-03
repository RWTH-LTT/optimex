"""
This module contains the optimizer for the Optimex project.
It provides functionality to perform optimization using Pyomo.
"""

from typing import Any, Dict, Tuple

import pyomo.environ as pyo
from loguru import logger
from pyomo.contrib.iis import write_iis
from pyomo.opt import ProblemFormat
from pyomo.opt.results.results_ import SolverResults

from optimex.converter import OptimizationModelInputs


def create_model(
    inputs: OptimizationModelInputs,
    name: str,
    objective_category: str,
    flexible_operation: bool = True,
    debug_path: str = None,
) -> pyo.ConcreteModel:
    """
    Build a Pyomo ConcreteModel for the optimization problem based on the provided
    inputs.

    This function constructs a fully defined Pyomo model using data from a `OptimizationModelInputs`
    instance. It optionally supports flexible operation of processes and can save
    intermediate data to a specified path.

    Parameters
    ----------
    inputs : OptimizationModelInputs
        Structured input data containing all flows, mappings, and constraints
        required for model construction.
    name : str
        Name of the Pyomo model instance.
    objective_category : str
        The category of impact to be minimized in the optimization problem.
    flexible_operation : bool, optional
        Enables flexible operation mode for processes. When set to True, the model
        introduces additional variables that allow processes to operate between 0 and
        their maximum installed capacity during their designated process time. This
        allows partial operation of a process rather than enforcing full capacity usage
        at all times.

        Flexible operation is based on scaling the inventory associated with the first
        time step of operation. In contrast, fixed operation (when `flexible_operation`
        is False) assumes that processes always run at full capacity once deployed.
    debug_path : str, optional
        If provided, specifies the directory path where intermediate model data (such as
        the LP formulation) or diagnostics may be stored.

    Returns
    -------
    pyo.ConcreteModel
        A fully constructed Pyomo model ready for optimization.
    """

    model = pyo.ConcreteModel(name=name)
    model._objective_category = objective_category
    scaled_inputs, scales = inputs.get_scaled_copy()
    model.scales = scales  # Store scales for denormalization later
    model.flexible_operation = flexible_operation

    logger.info("Creating sets")
    # Sets
    model.PROCESS = pyo.Set(
        doc="Set of processes (or activities), indexed by p",
        initialize=scaled_inputs.PROCESS,
    )
    model.REFERENCE_PRODUCT = pyo.Set(
        doc="Set of reference products, indexed by r",
        initialize=scaled_inputs.REFERENCE_PRODUCT,
    )
    model.INTERMEDIATE_FLOW = pyo.Set(
        doc="Set of intermediate flows, indexed by i",
        initialize=scaled_inputs.INTERMEDIATE_FLOW,
    )
    model.ELEMENTARY_FLOW = pyo.Set(
        doc="Set of elementary flows, indexed by e",
        initialize=scaled_inputs.ELEMENTARY_FLOW,
    )
    model.FLOW = pyo.Set(
        initialize=lambda m: m.REFERENCE_PRODUCT
        | m.INTERMEDIATE_FLOW
        | m.ELEMENTARY_FLOW,
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

    # Parameters
    logger.info("Creating parameters")
    model.process_names = pyo.Param(
        model.PROCESS,
        within=pyo.Any,
        doc="Names of the processes",
        default=None,
        initialize=scaled_inputs.process_names,
    )
    model.demand = pyo.Param(
        model.REFERENCE_PRODUCT,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="time-explicit demand vector d",
        default=0,
        initialize=scaled_inputs.demand,
    )
    model.foreground_technosphere = pyo.Param(
        model.PROCESS,
        model.INTERMEDIATE_FLOW,
        model.PROCESS_TIME,
        within=pyo.Reals,
        doc="time-explicit foreground technosphere tensor A",
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
        model.REFERENCE_PRODUCT,
        model.PROCESS_TIME,
        within=pyo.Reals,
        doc="time-explicit foreground production tensor F",
        default=0,
        initialize=scaled_inputs.foreground_production,
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
    model.process_limits_max = pyo.Param(
        model.PROCESS,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="maximum time specific process limit S_max",
        default=scaled_inputs.process_limits_max_default,
        initialize=(
            scaled_inputs.process_limits_max
            if scaled_inputs.process_limits_max is not None
            else {}
        ),
    )
    model.process_limits_min = pyo.Param(
        model.PROCESS,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="minimum time specific process limit S_min",
        default=scaled_inputs.process_limits_min_default,
        initialize=(
            scaled_inputs.process_limits_min
            if scaled_inputs.process_limits_min is not None
            else {}
        ),
    )
    model.cumulative_process_limits_max = pyo.Param(
        model.PROCESS,
        within=pyo.Reals,
        doc="maximum cumulatative process limit S_max,cum",
        default=scaled_inputs.cumulative_process_limits_max_default,
        initialize=(
            scaled_inputs.cumulative_process_limits_max
            if scaled_inputs.cumulative_process_limits_max is not None
            else {}
        ),
    )
    model.cumulative_process_limits_min = pyo.Param(
        model.PROCESS,
        within=pyo.Reals,
        doc="minimum cumulatative process limit S_min,cum",
        default=scaled_inputs.cumulative_process_limits_min_default,
        initialize=(
            scaled_inputs.cumulative_process_limits_min
            if scaled_inputs.cumulative_process_limits_min is not None
            else {}
        ),
    )
    model.process_coupling = pyo.Param(
        model.PROCESS,
        model.PROCESS,
        within=pyo.NonNegativeReals,
        doc="coupling matrix",
        initialize=(
            scaled_inputs.process_coupling
            if scaled_inputs.process_coupling is not None
            else {}
        ),
        default=0,  # Set default coupling value to 0 if not defined
    )

    model.category_impact_limit = pyo.Param(
        model.CATEGORY,
        within=pyo.Reals,
        doc="maximum impact limit",
        default=float("inf"),
        initialize=(
            scaled_inputs.category_impact_limit
            if scaled_inputs.category_impact_limit is not None
            else {}
        ),
    )

    # Variables
    logger.info("Creating variables")
    model.var_installation = pyo.Var(
        model.PROCESS,
        model.SYSTEM_TIME,
        within=pyo.NonNegativeReals,
        doc="Installation of the process",
    )

    # Process limits
    model.ProcessLimitMax = pyo.Constraint(
        model.PROCESS,
        model.SYSTEM_TIME,
        rule=lambda m, p, t: m.var_installation[p, t] <= m.process_limits_max[p, t],
    )

    model.ProcessLimitMin = pyo.Constraint(
        model.PROCESS,
        model.SYSTEM_TIME,
        rule=lambda m, p, t: m.var_installation[p, t] >= m.process_limits_min[p, t],
    )
    model.CumulativeProcessLimitMax = pyo.Constraint(
        model.PROCESS,
        rule=lambda m, p: sum(m.var_installation[p, t] for t in m.SYSTEM_TIME)
        <= m.cumulative_process_limits_max[p],
    )
    model.CumulativeProcessLimitMin = pyo.Constraint(
        model.PROCESS,
        rule=lambda m, p: sum(m.var_installation[p, t] for t in m.SYSTEM_TIME)
        >= m.cumulative_process_limits_min[p],
    )

    # Process coupling
    def process_coupling_rule(model, p1, p2, t):
        if (
            model.process_coupling[p1, p2] > 0
        ):  # only create constraint for non-zero coupling
            return (
                model.var_installation[p1, t]
                == model.process_coupling[p1, p2] * model.var_installation[p2, t]
            )
        else:
            return pyo.Constraint.Skip

    model.ProcessCouplingConstraint = pyo.Constraint(
        model.PROCESS, model.PROCESS, model.SYSTEM_TIME, rule=process_coupling_rule
    )

    def in_operation_phase(p, tau):
        return model.process_operation_start[p] <= tau <= model.process_operation_end[p]

    model.var_operation = pyo.Var(
        model.PROCESS,
        model.SYSTEM_TIME,
        within=pyo.NonNegativeReals,
        doc="Operational level",
    )
    
    if flexible_operation:
        # Expressions builder
        def scale_tensor_by_installation(tensor: pyo.Param, flow_set):
            def expr(m, p, x, t):
                return sum(
                    tensor[p, x, tau] * m.var_installation[p, t - tau]
                    for tau in m.PROCESS_TIME
                    if (t - tau in m.SYSTEM_TIME)
                    and (
                        not flexible_operation
                        or not in_operation_phase(p, tau)
                        or not m.operation_flow[p, x]
                    )
                )

            return pyo.Expression(
                model.PROCESS, getattr(model, flow_set), model.SYSTEM_TIME, rule=expr
            )

        def scale_tensor_by_operation(tensor: pyo.Param, flow_set):
            def expr(m, p, x, t):
                tau0 = m.process_operation_start[p]
                return tensor[p, x, tau0] * m.var_operation[p, t]

            return pyo.Expression(
                model.PROCESS, getattr(model, flow_set), model.SYSTEM_TIME, rule=expr
            )

        model.scaled_technosphere_dependent_on_installation = (
            scale_tensor_by_installation(
                model.foreground_technosphere, "INTERMEDIATE_FLOW"
            )
        )
        model.scaled_biosphere_dependent_on_installation = scale_tensor_by_installation(
            model.foreground_biosphere, "ELEMENTARY_FLOW"
        )
        model.scaled_technosphere_dependent_on_operation = scale_tensor_by_operation(
            model.foreground_technosphere, "INTERMEDIATE_FLOW"
        )
        model.scaled_biosphere_dependent_on_operation = scale_tensor_by_operation(
            model.foreground_biosphere, "ELEMENTARY_FLOW"
        )

        def scaled_inventory_tensor(model, p, e, t):
            """
            Returns a Pyomo expression for the total inventory impact for a given
            process p, elementary flow e, and time step t.
            """

            return sum(
                (
                    model.scaled_technosphere_dependent_on_installation[p, i, t]
                    + model.scaled_technosphere_dependent_on_operation[p, i, t]
                )
                * sum(
                    model.background_inventory[bkg, i, e] * model.mapping[bkg, t]
                    for bkg in model.BACKGROUND_ID
                )
                for i in model.INTERMEDIATE_FLOW
            ) + (
                model.scaled_biosphere_dependent_on_installation[p, e, t]
                + model.scaled_biosphere_dependent_on_operation[p, e, t]
            )

        model.scaled_inventory = pyo.Expression(
            model.PROCESS,
            model.ELEMENTARY_FLOW,
            model.SYSTEM_TIME,
            rule=scaled_inventory_tensor,
        )

        def operation_limited_by_installation_rule(model, p, f, t):
            return model.var_operation[p, t] * model.foreground_production[
                p, f, model.process_operation_start[p]
            ] <= sum(
                (1 if model.foreground_production[p, f, tau] != 0 else 0)
                * model.var_installation[p, t - tau]
                for tau in model.PROCESS_TIME
                if (t - tau in model.SYSTEM_TIME)
            )

        model.OperationLimit = pyo.Constraint(
            model.PROCESS,
            model.REFERENCE_PRODUCT,
            model.SYSTEM_TIME,
            rule=operation_limited_by_installation_rule,
        )

        def fulfill_demand_rule(model, f, t):
            return model.demand[f, t] == sum(
                model.foreground_production[p, f, model.process_operation_start[p]]
                * model.var_operation[p, t]
                for p in model.PROCESS
            )

        model.DemandConstraint = pyo.Constraint(
            model.REFERENCE_PRODUCT, model.SYSTEM_TIME, rule=fulfill_demand_rule
        )

    else:
        # broken for now, but shouldnt be used anyway
        raise NotImplementedError(
            "Fixed operation doesn't work right now, but why would you use it anyway? Please set flexible_operation=True for now."
        )
        
        def operation_at_full_capacity(model, p, f, t):
            return model.var_operation[p, t] * model.foreground_production[
                p, f, model.process_operation_start[p]
            ] == sum(
                (1 if model.foreground_production[p, f, tau] != 0 else 0)
                * model.var_installation[p, t - tau]
                for tau in model.PROCESS_TIME
                if (t - tau in model.SYSTEM_TIME)
            )
            
        model.OperationLimit = pyo.Constraint( # Always at full capacity
            model.PROCESS,
            model.REFERENCE_PRODUCT,
            model.SYSTEM_TIME,
            rule=operation_at_full_capacity,
        )

        def scale_tensor_by_installation(tensor: pyo.Param, flow_set: str):
            def expr(m, p, x, t):
                return sum(
                    tensor[p, x, tau] * m.var_installation[p, t - tau]
                    for tau in m.PROCESS_TIME
                    if (t - tau in m.SYSTEM_TIME)
                )

            return pyo.Expression(
                model.PROCESS, getattr(model, flow_set), model.SYSTEM_TIME, rule=expr
            )

        model.scaled_technosphere = scale_tensor_by_installation(
            model.foreground_technosphere, "INTERMEDIATE_FLOW"
        )
        model.scaled_biosphere = scale_tensor_by_installation(
            model.foreground_biosphere, "ELEMENTARY_FLOW"
        )

        def scaled_inventory_tensor(model, p, e, t):
            """
            Returns a Pyomo expression for the total inventory impact for a given
            process p, elementary flow e, and time step t.
            """

            return (
                sum(
                    model.scaled_technosphere[p, i, t]
                    * sum(
                        model.background_inventory[bkg, i, e] * model.mapping[bkg, t]
                        for bkg in model.BACKGROUND_ID
                    )
                    for i in model.INTERMEDIATE_FLOW
                )
                + model.scaled_biosphere[p, e, t]
            )

        model.scaled_inventory = pyo.Expression(
            model.PROCESS,
            model.ELEMENTARY_FLOW,
            model.SYSTEM_TIME,
            rule=scaled_inventory_tensor,
        )

        def fulfill_demand_rule(model, f, t):
            return (
                sum(
                    model.foreground_production[p, f, tau]
                    * model.var_operation[p, t - tau]
                    for p in model.PROCESS
                    for tau in model.PROCESS_TIME
                    if (t - tau in model.SYSTEM_TIME)
                )
                >= model.demand[f, t]
            )

        model.DemandConstraint = pyo.Constraint(
            model.REFERENCE_PRODUCT, model.SYSTEM_TIME, rule=fulfill_demand_rule
        )

    def category_process_time_specific_impact(model, c, p, t):
        return sum(
            model.characterization[c, e, t]
            * (model.scaled_inventory[p, e, t])  # Total inventory impact
            for e in model.ELEMENTARY_FLOW
        )

    # impact of process p at time t in category c
    model.specific_impact = pyo.Expression(
        model.CATEGORY,
        model.PROCESS,
        model.SYSTEM_TIME,
        rule=category_process_time_specific_impact,
    )

    # Total impact
    def total_impact_in_category(model, c):
        return sum(
            model.specific_impact[c, p, t]
            for p in model.PROCESS
            for t in model.SYSTEM_TIME
        )

    model.total_impact = pyo.Expression(model.CATEGORY, rule=total_impact_in_category)

    # Category impact limit
    def category_impact_limit_rule(model, c):
        return model.total_impact[c] <= model.category_impact_limit[c]

    model.CategoryImpactLimit = pyo.Constraint(
        model.CATEGORY, rule=category_impact_limit_rule
    )

    def objective_function(model):
        return model.total_impact[model._objective_category]

    model.OBJ = pyo.Objective(sense=pyo.minimize, rule=objective_function)

    if debug_path is not None:
        model.write(
            debug_path,
            io_options={"symbolic_solver_labels": True},
            format=ProblemFormat.cpxlp,
        )
    return model


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
    Solve a Pyomo optimization model using a specified solver and
    denormalize the objective (and optional duals) using stored scales.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The Pyomo model to be solved. Must have attribute `scales` with keys
        'foreground' and 'characterization'.
    solver_name : str, optional
        Name of the solver (default: "gurobi").
    solver_args : dict, optional
        Args to pass to SolverFactory.
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
    model : pyo.ConcreteModel
        The solved model (with original scaling preserved).
    true_obj : float
        The denormalized objective value.
    results : SolverResults
        The raw Pyomo solver results object.
    """
    # 1) Instantiate solver
    solver_args = solver_args or {}
    solver = pyo.SolverFactory(solver_name, **solver_args)
    if solver_options:
        for opt, val in solver_options.items():
            solver.options[opt] = val

    # 2) Solve model
    results = solver.solve(model, tee=tee, **solve_kwargs)
    model.solutions.load_from(results)
    logger.info(
        f"Solver [{solver_name}] termination: {results.solver.termination_condition}"
    )

    # 3) Handle infeasibility and optional IIS
    if (
        results.solver.termination_condition == pyo.TerminationCondition.infeasible
        and compute_iis
    ):
        try:
            write_iis(model, iis_file_name="model_iis.ilp", solver=solver)
            logger.info("IIS written to model_iis.ilp")
        except Exception as e:
            logger.warning(f"IIS generation failed: {e}")

    # 4) Denormalize objective
    scaled_obj = pyo.value(model.OBJ)
    fg_scale = getattr(model, "scales", {}).get("foreground", 1.0)
    catscales = getattr(model, "scales", {}).get("characterization", {})
    if model._objective_category and model._objective_category in catscales:
        cat_scale = catscales[model._objective_category]
    else:
        cat_scale = 1.0

    true_obj = scaled_obj * fg_scale * cat_scale
    logger.info(f"Objective (scaled): {scaled_obj:.6g}")
    logger.info(f"Objective (real):   {true_obj:.6g}")

    # 5) (Optional) Denormalize duals
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

    return model, true_obj, results
