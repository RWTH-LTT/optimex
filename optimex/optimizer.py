"""
This module contains the optimizer for the Optimex project.
It provides functionality to perform optimization using Pyomo.
"""

import logging

import pyomo.contrib.iis as iis
import pyomo.environ as pyo

from optimex.converter import ModelInputs


def create_model(inputs: ModelInputs, name: str) -> pyo.ConcreteModel:
    """
    Build an abstract model with all elements required to solve the optimization
    problem.

    Returns:
        pyo.AbstractModel: Abstract model for optimization problem
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
    model.scaling = pyo.Var(
        model.PROCESS, model.SYSTEM_TIME, within=pyo.Reals, doc="scaling matrix S"
    )

    # Expressions - easier for readability
    logging.info("Creating expressions")

    def scaled_technosphere_rule(model, p, i, t):
        return sum(
            model.foreground_technosphere[p, i, tau] * model.scaling[p, t - tau]
            for tau in model.PROCESS_TIME
            if t - tau in model.SYSTEM_TIME
        )

    model.scaled_technosphere = pyo.Expression(
        model.PROCESS,
        model.INTERMEDIATE_FLOW,
        model.SYSTEM_TIME,
        rule=scaled_technosphere_rule,
    )

    def scaled_biosphere_rule(model, p, e, t):
        return sum(
            model.foreground_biosphere[p, e, tau] * model.scaling[p, t - tau]
            for tau in model.PROCESS_TIME
            if t - tau in model.SYSTEM_TIME
        )

    model.scaled_biosphere = pyo.Expression(
        model.PROCESS,
        model.ELEMENTARY_FLOW,
        model.SYSTEM_TIME,
        rule=scaled_biosphere_rule,
    )

    def time_process_specific_impact_rule(model, p, t):
        return sum(
            model.characterization[e, t]
            * (
                sum(
                    model.scaled_technosphere[p, i, t]
                    * sum(
                        model.background_inventory[b, i, e] * model.mapping[b, t]
                        for b in model.BACKGROUND_ID
                    )
                    for i in model.INTERMEDIATE_FLOW
                )
                + model.scaled_biosphere[p, e, t]
            )
            for e in model.ELEMENTARY_FLOW
        )

    model.time_process_specific_impact = pyo.Expression(
        model.PROCESS, model.SYSTEM_TIME, rule=time_process_specific_impact_rule
    )

    logging.info("Creating constraints")

    # Demand constraints
    def demand_rule(model, f, t):
        return (
            sum(
                model.foreground_production[p, f, tau] * model.scaling[p, t - tau]
                for p in model.PROCESS
                for tau in model.PROCESS_TIME
                if t - tau in model.SYSTEM_TIME
            )
            >= model.demand[f, t]
        )

    model.DemandConstraint = pyo.Constraint(
        model.FUNCTIONAL_FLOW, model.SYSTEM_TIME, rule=demand_rule
    )

    # Process limits
    model.ProcessLimitMax = pyo.Constraint(
        model.PROCESS,
        model.SYSTEM_TIME,
        rule=lambda m, p, t: m.scaling[p, t] <= m.process_limits_max[p, t],
    )

    model.ProcessLimitMin = pyo.Constraint(
        model.PROCESS,
        model.SYSTEM_TIME,
        rule=lambda m, p, t: m.scaling[p, t] >= m.process_limits_min[p, t],
    )
    model.CumulativeProcessLimitMax = pyo.Constraint(
        model.PROCESS,
        rule=lambda m, p: sum(m.scaling[p, t] for t in m.SYSTEM_TIME)
        <= m.cumulative_process_limits_max[p],
    )
    model.CumulativeProcessLimitMin = pyo.Constraint(
        model.PROCESS,
        rule=lambda m, p: sum(m.scaling[p, t] for t in m.SYSTEM_TIME)
        >= m.cumulative_process_limits_min[p],
    )

    # Process coupling
    def process_coupling_rule(model, p1, p2, t):
        if (
            model.process_coupling[p1, p2] > 0
        ):  # only create constraint for non-zero coupling
            return (
                model.scaling[p1, t]
                == model.process_coupling[p1, p2] * model.scaling[p2, t]
            )
        else:
            return pyo.Constraint.Skip

    model.ProcessCouplingConstraint = pyo.Constraint(
        model.PROCESS, model.PROCESS, model.SYSTEM_TIME, rule=process_coupling_rule
    )

    # Objective: Direct computation
    logging.info("Creating objective function")

    def direct_objective_function(model):
        return sum(
            sum(
                model.characterization[e, t]
                * (
                    sum(
                        model.foreground_technosphere[p, i, tau]
                        * model.scaling[p, t - tau]
                        * sum(
                            model.background_inventory[b, i, e] * model.mapping[b, t]
                            for b in model.BACKGROUND_ID
                        )
                        for i in model.INTERMEDIATE_FLOW
                        for tau in model.PROCESS_TIME
                        if t - tau in model.SYSTEM_TIME
                    )
                    + sum(
                        model.foreground_biosphere[p, e, tau]
                        * model.scaling[p, t - tau]
                        for tau in model.PROCESS_TIME
                        if t - tau in model.SYSTEM_TIME
                    )
                )
                for e in model.ELEMENTARY_FLOW
            )
            for p in model.PROCESS
            for t in model.SYSTEM_TIME
        )

    def expression_objective_function(model):
        return sum(
            model.time_process_specific_impact[p, t]
            for p in model.PROCESS
            for t in model.SYSTEM_TIME
        )

    model.OBJ = pyo.Objective(sense=pyo.minimize, rule=expression_objective_function)
    return model


def solve_model(model: pyo.ConcreteModel, tee=True, gap=0.01, compute_iis=False):
    """
    Solve the provided model.

    Args:
        model (pyo.ConcreteModel): Model to solve
        tee (bool, optional): Print solver output. Defaults to True.
        gap (float, optional): MIP gap tolerance. Defaults to 0.01.
        compute_iis (bool, optional): Compute Irreducible Infeasible Set.
        Defaults to False.

    Returns:
        pyo.SolverResults: Results of the optimization
    """
    solver = pyo.SolverFactory("gurobi")
    solver.options["MIPGap"] = gap

    results = solver.solve(model, tee=tee)
    if (
        results.solver.termination_condition == pyo.TerminationCondition.infeasible
        and compute_iis
    ):
        try:
            iis.write_iis(model, iis_file_name="model_iis.ilp", solver="gurobi")
        except Exception as e:
            logging.info(f"Failed to compute IIS: {e}")
    return model, results
