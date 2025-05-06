"""
This module contains the optimizer for the Optimex project.
It provides functionality to perform optimization using Pyomo.
"""

import logging

import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.opt import ProblemFormat

from optimex.converter import ModelInputs


def create_model(
    inputs: ModelInputs, name: str, flexible_operation: bool = False, path: str = None
) -> pyo.ConcreteModel:
    """
    Build a concrete model with all elements required to solve the optimization
    problem.

    Returns:
        pyo.ConcreteModel: Concrete model for optimization problem
    """
    model = pyo.ConcreteModel(name=name)

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
    # model.INDICATOR = pyo.Set(doc="Set of environmental indicators,
    # indexed by ind", initialize=inputs.INDICATOR)

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
        #    model.INDICATOR,
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
        def impact_op(model, p, t):
            return sum(
                model.characterization[e, t]
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

        model.time_process_specific_impact = pyo.Expression(
            model.PROCESS, model.SYSTEM_TIME, rule=impact_op
        )

        # Operation limit
        def op_limit(model, p, t):
            return model.var_operation[p, t] <= sum(
                model.foreground_production[p, f, tau]
                * model.var_installation[p, t - tau]
                for f in model.FUNCTIONAL_FLOW
                for tau in model.PROCESS_TIME
                if (t - tau in model.SYSTEM_TIME)
            )

        model.OperationLimit = pyo.Constraint(
            model.PROCESS, model.SYSTEM_TIME, rule=op_limit
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
        def impact_orig(model, p, t):
            return sum(
                model.characterization[e, t]
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

        model.time_process_specific_impact = pyo.Expression(
            model.PROCESS, model.SYSTEM_TIME, rule=impact_orig
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

    # Objective: Direct computation
    logging.info("Creating objective function")

    def expression_objective_function(model):
        return sum(
            model.time_process_specific_impact[p, t]
            for p in model.PROCESS
            for t in model.SYSTEM_TIME
        )

    model.OBJ = pyo.Objective(sense=pyo.minimize, rule=expression_objective_function)

    if path is not None:
        model.write(path, format=ProblemFormat.cpxlp)
    return model


def solve_model(model: pyo.ConcreteModel, tee: bool = True, compute_iis: bool = False):
    """
    Solve the provided model.

    Args:
        model (pyo.ConcreteModel): Model to solve
        tee (bool, optional): Print solver output.
        compute_iis (bool, optional): Compute Irreducible Infeasible Set.

    Returns:
        Tuple[pyo.ConcreteModel, pyo.SolverResults]: Solved model and results
    """
    solver = pyo.SolverFactory("gurobi")
    solver.options["logfile"] = "gurobi.log"

    results = solver.solve(model, tee=tee)
    logging.info(f"Solver status: {results.solver.termination_condition}")

    if (
        results.solver.termination_condition == pyo.TerminationCondition.infeasible
        and compute_iis
    ):
        try:
            write_iis(model, iis_file_name="model_iis.ilp", solver=solver)
        except Exception as e:
            logging.info(f"Failed to compute IIS: {e}")

    return model, results
