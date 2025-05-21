"""
This module contains the optimizer for the Optimex project.
It provides functionality to perform optimization using Pyomo.
"""

import logging
from typing import Any, Dict, Tuple

import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.opt import ProblemFormat
from pyomo.opt.results.results_ import SolverResults

from optimex.converter import ModelInputs


def create_model(
    inputs: ModelInputs,
    name: str,
    objective_category: str,
    scales: Dict[str, Any] = None,
    flexible_operation: bool = True,
    debug_path: str = None,
) -> pyo.ConcreteModel:
    """
    Build a Pyomo ConcreteModel for the optimization problem based on the provided
    inputs.

    This function constructs a fully defined Pyomo model using data from a `ModelInputs`
    instance. It optionally supports flexible operation of processes and can save
    intermediate data to a specified path.

    Parameters
    ----------
    inputs : ModelInputs
        Structured input data containing all flows, mappings, and constraints
        required for model construction.
    name : str
        Name of the Pyomo model instance.
    objective_category : str
        The category of impact to be minimized in the optimization problem.
    scales : dict, optional
        Dictionary containing scaling factors for a scaled model. The keys should
        include 'foreground' and 'characterization'. The values should be the
        corresponding scaling factors. This is used to denormalize the objective
        function and duals.
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
    model._scales = scales or {}
    model._objective_category = objective_category

    logging.info("Creating sets")
    # Sets
    model.PROCESS = pyo.Set(
        doc="Set of processes (or activities), indexed by p", initialize=inputs.PROCESS
    )
    model.FUNCTIONAL_FLOW = pyo.Set(
        doc="Set of functional flows (or products), indexed by r",
        initialize=inputs.FUNCTIONAL_FLOW,
    )
    model.INTERMEDIATE_FLOW = pyo.Set(
        doc="Set of intermediate flows, indexed by i",
        initialize=inputs.INTERMEDIATE_FLOW,
    )
    model.ELEMENTARY_FLOW = pyo.Set(
        doc="Set of elementary flows, indexed by e", initialize=inputs.ELEMENTARY_FLOW
    )
    model.CATEGORY = pyo.Set(
        doc="Set of impact categories, indexed by c", initialize=inputs.CATEGORY
    )

    model.BACKGROUND_ID = pyo.Set(
        doc="Set of identifiers of the prospective background databases, indexed by b",
        initialize=inputs.BACKGROUND_ID,
    )
    model.PROCESS_TIME = pyo.Set(
        doc="Set of process time points, indexed by tau", initialize=inputs.PROCESS_TIME
    )
    model.SYSTEM_TIME = pyo.Set(
        doc="Set of system time points, indexed by t", initialize=inputs.SYSTEM_TIME
    )

    # Parameters
    logging.info("Creating parameters")
    model.process_names = pyo.Param(
        model.PROCESS,
        within=pyo.Any,
        doc="Names of the processes",
        default=None,
        initialize=inputs.process_names,
    )
    model.demand = pyo.Param(
        model.FUNCTIONAL_FLOW,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="time-explicit demand vector d",
        default=0,
        initialize=inputs.demand,
    )
    model.process_operation_start = pyo.Param(
        model.PROCESS,
        within=model.PROCESS_TIME,
        initialize={p: inputs.process_operation_time[p][0] for p in inputs.PROCESS},
        doc="start of operation phase",
    )
    model.process_operation_end = pyo.Param(
        model.PROCESS,
        within=model.PROCESS_TIME,
        initialize={p: inputs.process_operation_time[p][1] for p in inputs.PROCESS},
        doc="end of operation phase",
    )

    model.foreground_technosphere = pyo.Param(
        model.PROCESS,
        model.INTERMEDIATE_FLOW,
        model.PROCESS_TIME,
        within=pyo.Reals,
        doc="time-explicit foreground technosphere tensor A",
        default=0,
        initialize=inputs.foreground_technosphere,
    )
    model.foreground_biosphere = pyo.Param(
        model.PROCESS,
        model.ELEMENTARY_FLOW,
        model.PROCESS_TIME,
        within=pyo.Reals,
        doc="time-explicit foreground biosphere tensor B",
        default=0,
        initialize=inputs.foreground_biosphere,
    )
    model.foreground_production = pyo.Param(
        model.PROCESS,
        model.FUNCTIONAL_FLOW,
        model.PROCESS_TIME,
        within=pyo.Reals,
        doc="time-explicit foreground production tensor F",
        default=0,
        initialize=inputs.foreground_production,
    )
    # TODO: check that first year production is 1
    model.background_inventory = pyo.Param(
        model.BACKGROUND_ID,
        model.INTERMEDIATE_FLOW,
        model.ELEMENTARY_FLOW,
        within=pyo.Reals,
        doc="prospective background inventory tensor G",
        default=0,
        initialize=inputs.background_inventory,
    )
    model.mapping = pyo.Param(
        model.BACKGROUND_ID,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="time-explicit background mapping tensor M",
        default=0,
        initialize=inputs.mapping,
    )
    model.characterization = pyo.Param(
        model.CATEGORY,
        model.ELEMENTARY_FLOW,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="time-explicit characterization tensor Q",
        default=0,
        initialize=inputs.characterization,
    )
    model.process_limits_max = pyo.Param(
        model.PROCESS,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="maximum time specific process limit S_max",
        default=inputs.process_limits_max_default,
        initialize=(
            inputs.process_limits_max if inputs.process_limits_max is not None else {}
        ),
    )
    model.process_limits_min = pyo.Param(
        model.PROCESS,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="minimum time specific process limit S_min",
        default=inputs.process_limits_min_default,
        initialize=(
            inputs.process_limits_min if inputs.process_limits_min is not None else {}
        ),
    )
    model.cumulative_process_limits_max = pyo.Param(
        model.PROCESS,
        within=pyo.Reals,
        doc="maximum cumulatative process limit S_max,cum",
        default=inputs.cumulative_process_limits_max_default,
        initialize=(
            inputs.cumulative_process_limits_max
            if inputs.cumulative_process_limits_max is not None
            else {}
        ),
    )
    model.cumulative_process_limits_min = pyo.Param(
        model.PROCESS,
        within=pyo.Reals,
        doc="minimum cumulatative process limit S_min,cum",
        default=inputs.cumulative_process_limits_min_default,
        initialize=(
            inputs.cumulative_process_limits_min
            if inputs.cumulative_process_limits_min is not None
            else {}
        ),
    )
    model.process_coupling = pyo.Param(
        model.PROCESS,
        model.PROCESS,
        within=pyo.NonNegativeReals,
        doc="coupling matrix",
        initialize=(
            inputs.process_coupling if inputs.process_coupling is not None else {}
        ),
        default=0,  # Set default coupling value to 0 if not defined
    )

    model.category_impact_limit = pyo.Param(
        model.CATEGORY,
        within=pyo.Reals,
        doc="maximum impact limit",
        default=float("inf"),
        initialize=(
            inputs.category_impact_limit
            if inputs.category_impact_limit is not None
            else {}
        ),
    )

    # Variables
    logging.info("Creating variables")
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

    if flexible_operation:
        # Operation variable
        model.var_operation = pyo.Var(
            model.PROCESS,
            model.SYSTEM_TIME,
            within=pyo.NonNegativeReals,
            doc="Operation of the process",
        )

        def in_operation_phase(p, tau):
            return (
                model.process_operation_start[p]
                <= tau
                <= model.process_operation_end[p]
            )

        # Scaled technosphere (capacity and operation)
        def tech_cap(model, p, i, t):
            return sum(
                model.foreground_technosphere[p, i, tau]
                * model.var_installation[p, t - tau]
                for tau in model.PROCESS_TIME
                if (t - tau in model.SYSTEM_TIME) and not in_operation_phase(p, tau)
            )

        model.scaled_technosphere_cap = pyo.Expression(
            model.PROCESS, model.INTERMEDIATE_FLOW, model.SYSTEM_TIME, rule=tech_cap
        )

        def tech_op(model, p, i, t):
            tau0 = model.process_operation_start[p]
            return model.foreground_technosphere[p, i, tau0] * model.var_operation[p, t]

        model.scaled_technosphere_op = pyo.Expression(
            model.PROCESS, model.INTERMEDIATE_FLOW, model.SYSTEM_TIME, rule=tech_op
        )

        # Scaled biosphere (capacity and operation)
        def bio_cap(model, p, e, t):
            return sum(
                model.foreground_biosphere[p, e, tau]
                * model.var_installation[p, t - tau]
                for tau in model.PROCESS_TIME
                if (t - tau in model.SYSTEM_TIME) and not in_operation_phase(p, tau)
            )

        model.scaled_biosphere_cap = pyo.Expression(
            model.PROCESS, model.ELEMENTARY_FLOW, model.SYSTEM_TIME, rule=bio_cap
        )

        def bio_op(model, p, e, t):
            tau0 = model.process_operation_start[p]
            return model.foreground_biosphere[p, e, tau0] * model.var_operation[p, t]

        model.scaled_biosphere_op = pyo.Expression(
            model.PROCESS, model.ELEMENTARY_FLOW, model.SYSTEM_TIME, rule=bio_op
        )

        # Time process-specific impact
        def impact_op(model, c, p, t):
            return sum(
                model.characterization[c, e, t]
                * (
                    sum(
                        (
                            model.scaled_technosphere_cap[p, i, t]
                            + model.scaled_technosphere_op[p, i, t]
                        )
                        * sum(
                            model.background_inventory[bkg, i, e]
                            * model.mapping[bkg, t]
                            for bkg in model.BACKGROUND_ID
                        )
                        for i in model.INTERMEDIATE_FLOW
                    )
                    + (
                        model.scaled_biosphere_cap[p, e, t]
                        + model.scaled_biosphere_op[p, e, t]
                    )
                )
                for e in model.ELEMENTARY_FLOW
            )

        # impact of process p at time t in category c
        model.specific_impact = pyo.Expression(
            model.CATEGORY, model.PROCESS, model.SYSTEM_TIME, rule=impact_op
        )

        # Operation limit
        def op_limit(model, p, f, t):
            return model.var_operation[p, t] * model.foreground_production[
                p, f, model.process_operation_start[p]
            ] <= sum(
                model.foreground_production[p, f, tau]
                * model.var_installation[p, t - tau]
                for tau in model.PROCESS_TIME
                if (t - tau in model.SYSTEM_TIME)
            )

        model.OperationLimit = pyo.Constraint(
            model.PROCESS, model.FUNCTIONAL_FLOW, model.SYSTEM_TIME, rule=op_limit
        )

        # Demand driven by operation
        def demand_op(model, f, t):
            return model.demand[f, t] == sum(
                model.foreground_production[p, f, model.process_operation_start[p]]
                * model.var_operation[p, t]
                for p in model.PROCESS
            )

        model.DemandConstraint = pyo.Constraint(
            model.FUNCTIONAL_FLOW, model.SYSTEM_TIME, rule=demand_op
        )
    else:
        # Scaled technosphere
        def scaled_tech_orig(model, p, i, t):
            return sum(
                model.foreground_technosphere[p, i, tau]
                * model.var_installation[p, t - tau]
                for tau in model.PROCESS_TIME
                if (t - tau in model.SYSTEM_TIME)
            )

        model.scaled_technosphere = pyo.Expression(
            model.PROCESS,
            model.INTERMEDIATE_FLOW,
            model.SYSTEM_TIME,
            rule=scaled_tech_orig,
        )

        # Scaled biosphere
        def scaled_bio_orig(model, p, e, t):
            return sum(
                model.foreground_biosphere[p, e, tau]
                * model.var_installation[p, t - tau]
                for tau in model.PROCESS_TIME
                if (t - tau in model.SYSTEM_TIME)
            )

        model.scaled_biosphere = pyo.Expression(
            model.PROCESS,
            model.ELEMENTARY_FLOW,
            model.SYSTEM_TIME,
            rule=scaled_bio_orig,
        )

        # Time process-specific impact
        def impact_orig(model, c, p, t):
            return sum(
                model.characterization[c, e, t]
                * (
                    sum(
                        model.scaled_technosphere[p, i, t]
                        * sum(
                            model.background_inventory[bkg, i, e]
                            * model.mapping[bkg, t]
                            for bkg in model.BACKGROUND_ID
                        )
                        for i in model.INTERMEDIATE_FLOW
                    )
                    + model.scaled_biosphere[p, e, t]
                )
                for e in model.ELEMENTARY_FLOW
            )

        model.specific_impact = pyo.Expression(
            model.CATEGORY, model.PROCESS, model.SYSTEM_TIME, rule=impact_orig
        )

        # Demand constraint
        def demand_orig(model, f, t):
            return (
                sum(
                    model.foreground_production[p, f, tau]
                    * model.var_installation[p, t - tau]
                    for p in model.PROCESS
                    for tau in model.PROCESS_TIME
                    if (t - tau in model.SYSTEM_TIME)
                )
                >= model.demand[f, t]
            )

        model.DemandConstraint = pyo.Constraint(
            model.FUNCTIONAL_FLOW, model.SYSTEM_TIME, rule=demand_orig
        )

    # Category impact limit
    def category_impact_limit_rule(model, c):
        return (
            sum(
                model.specific_impact[c, p, t]
                for p in model.PROCESS
                for t in model.SYSTEM_TIME
            )
            <= model.category_impact_limit[c]
        )

    model.CategoryImpactLimit = pyo.Constraint(
        model.CATEGORY, rule=category_impact_limit_rule
    )

    # Objective: Direct computation
    logging.info("Creating objective function")

    def expression_objective_function(model):
        return sum(
            model.specific_impact[objective_category, p, t]
            for p in model.PROCESS
            for t in model.SYSTEM_TIME
        )

    model.OBJ = pyo.Objective(sense=pyo.minimize, rule=expression_objective_function)

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
        The Pyomo model to be solved. Must have attribute `_scales` with keys
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
    logging.info(
        f"Solver [{solver_name}] termination: {results.solver.termination_condition}"
    )

    # 3) Handle infeasibility and optional IIS
    if (
        results.solver.termination_condition == pyo.TerminationCondition.infeasible
        and compute_iis
    ):
        try:
            write_iis(model, iis_file_name="model_iis.ilp", solver=solver)
            logging.info("IIS written to model_iis.ilp")
        except Exception as e:
            logging.warning(f"IIS generation failed: {e}")

    # 4) Denormalize objective
    scaled_obj = pyo.value(model.OBJ)
    fg_scale = getattr(model, "_scales", {}).get("foreground", 1.0)
    cat_scales = getattr(model, "_scales", {}).get("characterization", {})
    if model._objective_category and model._objective_category in cat_scales:
        cat_scale = cat_scales[model._objective_category]
    else:
        cat_scale = 1.0

    true_obj = scaled_obj * fg_scale * cat_scale
    logging.info(f"Objective (scaled): {scaled_obj:.6g}")
    logging.info(f"Objective (real):   {true_obj:.6g}")

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
                denorm_duals[f"impact_{c}"] = μ * cat_scales.get(c, 1.0)
        logging.info(f"Denormalized duals: {denorm_duals}")

    return model, true_obj, results
