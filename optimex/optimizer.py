"""
Optimization model construction and solving for temporal LCA-based pathway optimization.

This module creates and solves Pyomo optimization models that minimize environmental
impacts over time while meeting demand constraints and respecting process limits.

## Scaling Convention

The optimization uses a two-tier scaling system for numerical stability:

### Decision Variables (REAL UNITS)
- `var_installation[p, t]`: Number of process units installed (dimensionless)
- `var_operation[p, t]`: Operation level (dimensionless, 0 to capacity)

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
- `category_impact_limit[c]`: maximum impact allowed [SCALED]

**Unscaled parameters**:
- `background_inventory[bkg, i, e]`: kg emission / kg intermediate [UNSCALED]
- `mapping[bkg, t]`: interpolation weights [UNSCALED, dimensionless]
- `process_limits_*`: capacity limits [UNSCALED, matches var_installation]

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
    production [kg SCALED/operation] × var_operation [#] = demand [kg SCALED] ✓

OperationLimit (LHS):
    var_operation [#] × production [kg SCALED/operation] × fg_scale = [kg REAL]
OperationLimit (RHS):
    production [kg SCALED/process] × fg_scale × var_installation [#] = [kg REAL]
```
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
        initialize=lambda m: m.PRODUCT
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
    model.internal_demand_technosphere = pyo.Param(
        model.PROCESS,
        model.PRODUCT,
        model.PROCESS_TIME,
        within=pyo.Reals,
        doc="time-explicit internal demand tensor A^{internal}",
        default=0,
        initialize=scaled_inputs.internal_demand_technosphere,
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

    # Existing (brownfield) capacity: capacity installed before SYSTEM_TIME
    # These contribute to operation capacity but NOT to installation-phase impacts
    model.existing_capacity = pyo.Param(
        model.PROCESS,
        pyo.Any,  # Installation year (can be any year before SYSTEM_TIME)
        within=pyo.NonNegativeReals,
        doc="Existing capacity (process, installation_year) -> amount",
        initialize=(
            scaled_inputs.existing_capacity
            if scaled_inputs.existing_capacity is not None
            else {}
        ),
        default=0,
    )
    # Store the existing capacity dict for iteration
    model._existing_capacity_dict = (
        scaled_inputs.existing_capacity
        if scaled_inputs.existing_capacity is not None
        else {}
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
        within=pyo.NonNegativeReals,  # Non-negative activity level
        doc="Operational activity level (production amounts at each time)",
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
                # Only apply operational scaling to flows marked as operational
                # Check the value explicitly since operation_flow is a Param
                if pyo.value(m.operation_flow[p, x]) == 0:
                    return 0

                # For operational flows: use TOTAL across operation phase
                # (same treatment as production flows)
                # o_t represents production amount, so operational flows scale proportionally
                # This is LINEAR: parameter × variable
                total_flow_per_installation = sum(
                    tensor[p, x, tau]
                    for tau in m.PROCESS_TIME
                    if m.process_operation_start[p] <= tau <= m.process_operation_end[p]
                )

                # Total flow = sum(flow[tau]) × o_t
                return total_flow_per_installation * m.var_operation[p, t]

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
        model.scaled_internal_demand_dependent_on_installation = (
            scale_tensor_by_installation(
                model.internal_demand_technosphere, "PRODUCT"
            )
        )
        model.scaled_internal_demand_dependent_on_operation = (
            scale_tensor_by_operation(
                model.internal_demand_technosphere, "PRODUCT"
            )
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

        def operation_capacity_constraint_rule(model, p, r, t):
            """
            Capacity constraint: o_t ≤ (total production per installation) × (installations in operation)

            This constraint ensures operation level cannot exceed the total production
            capacity provided by active installations. The formulation matches the
            demand constraint which also sums production over the operation phase.

            Note: foreground_production is SCALED, so we multiply by fg_scale to get
            real capacity. var_operation is in REAL units (dimensionless activity level).

            Only applied when process p produces product r (total_production > 0).

            Brownfield (existing) capacity:
            Existing capacity installed before SYSTEM_TIME is included in installations_in_operation
            if it is still within its operation phase at time t. This allows brownfield capacity
            to contribute to production without adding installation-phase impacts.
            """
            fg_scale = model.scales["foreground"]

            # Total production per installation during operation phase (SCALED)
            # This matches the demand constraint which also sums over operation phase
            total_production_scaled = sum(
                model.foreground_production[p, r, tau]
                for tau in model.PROCESS_TIME
                if model.process_operation_start[p] <= tau <= model.process_operation_end[p]
            )

            # Skip constraint if process doesn't produce this product
            # (otherwise constraint becomes var_operation <= 0, which is overly restrictive)
            if total_production_scaled == 0:
                return pyo.Constraint.Skip

            # Count installations currently in their operation phase (REAL units)
            # This includes both new installations (var_installation) and existing capacity
            installations_in_operation = sum(
                model.var_installation[p, t - tau]
                for tau in model.PROCESS_TIME
                if (t - tau in model.SYSTEM_TIME)
                and model.process_operation_start[p] <= tau <= model.process_operation_end[p]
            )

            # Add existing (brownfield) capacity that is still in operation phase
            # For existing capacity installed at inst_year, tau = t - inst_year
            # It's in operation if: process_operation_start <= tau <= process_operation_end
            op_start = pyo.value(model.process_operation_start[p])
            op_end = pyo.value(model.process_operation_end[p])
            for (proc, inst_year), capacity in model._existing_capacity_dict.items():
                if proc == p:
                    tau_existing = t - inst_year
                    if op_start <= tau_existing <= op_end:
                        installations_in_operation += capacity

            # Capacity = total_production (SCALED) × fg_scale × installations (REAL) = (REAL)
            capacity = total_production_scaled * fg_scale * installations_in_operation
            return model.var_operation[p, t] <= capacity

        model.OperationCapacity = pyo.Constraint(
            model.PROCESS,
            model.PRODUCT,
            model.SYSTEM_TIME,
            rule=operation_capacity_constraint_rule,
        )

        def product_demand_fulfillment_rule(model, r, t):
            """
            Demand constraint: total_production × o_t ≥ f_t

            For production flows: use total production per installation (sum across operation phase).
            o_t represents production amount, bounded by installed capacity.

            Total production must meet:
            - External demand (from demand parameter)
            - Internal consumption by other foreground processes

            This is LINEAR: parameter × variable
            """
            # Total production of product r at time t
            # Sum of production across operation phase × operation level
            total_production = sum(
                sum(
                    model.foreground_production[p, r, tau]
                    for tau in model.PROCESS_TIME
                    if model.process_operation_start[p] <= tau <= model.process_operation_end[p]
                )
                * model.var_operation[p, t]
                for p in model.PROCESS
            )

            # External demand (from demand parameter)
            external_demand = model.demand[r, t]

            # Internal consumption (sum over all processes consuming product r)
            internal_consumption = sum(
                model.scaled_internal_demand_dependent_on_installation[p, r, t]
                + model.scaled_internal_demand_dependent_on_operation[p, r, t]
                for p in model.PROCESS
            )

            # Exact fulfillment (= constraint)
            return total_production == external_demand + internal_consumption

        model.ProductDemandFulfillment = pyo.Constraint(
            model.PRODUCT, model.SYSTEM_TIME, rule=product_demand_fulfillment_rule
        )

    else:
        # Fixed operation mode is not currently implemented
        # The previous implementation had dimensional consistency issues and
        # did not properly handle the temporal structure of processes.
        #
        # To implement fixed operation mode properly, the following would be needed:
        # 1. Force var_operation to equal installed capacity (no partial operation)
        # 2. Adjust demand fulfillment to account for forced full-capacity operation
        # 3. Update inventory calculations to remove operation-dependent flows
        # 4. Ensure all installations operate at full capacity throughout their lifecycle
        #
        # For now, flexible operation mode is recommended and fully supported.
        raise NotImplementedError(
            "Fixed operation mode is not currently implemented. "
            "Please use flexible_operation=True. "
            "See optimizer.py documentation for details on implementing fixed mode."
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


def validate_operation_bounds(model: pyo.ConcreteModel, tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Validate that operation levels respect capacity constraints.

    This function performs post-solve validation to ensure that var_operation
    does not exceed the production capacity from installations in operation phase.

    The capacity constraint is: var_operation <= total_production * fg_scale * installations

    Parameters
    ----------
    model : pyo.ConcreteModel
        A solved Pyomo model with flexible operation mode.
    tolerance : float, optional
        Relative tolerance for validation (default: 1e-6).

    Returns
    -------
    dict
        Validation results with keys:
        - "valid": bool, True if all operation levels respect capacity
        - "violations": list of tuples (process, time, operation, capacity, violation_type)
        - "max_violation": float, maximum violation found
        - "summary": str, human-readable summary

    Raises
    ------
    ValueError
        If model is not in flexible operation mode or not solved.
    """
    if not model.flexible_operation:
        raise ValueError("Validation only applies to flexible operation mode")

    if not hasattr(model, "var_operation"):
        raise ValueError("Model must have var_operation")

    violations = []
    max_violation = 0.0
    fg_scale = model.scales["foreground"]

    for p in model.PROCESS:
        for t in model.SYSTEM_TIME:
            operation_value = pyo.value(model.var_operation[p, t])

            # Calculate capacity for each product this process produces
            # The binding capacity constraint uses the product with maximum capacity
            max_capacity = 0.0
            for r in model.PRODUCT:
                # Total production per installation during operation phase (SCALED)
                total_production_scaled = sum(
                    pyo.value(model.foreground_production[p, r, tau])
                    for tau in model.PROCESS_TIME
                    if pyo.value(model.process_operation_start[p]) <= tau <= pyo.value(model.process_operation_end[p])
                )

                # Skip if process doesn't produce this product
                if total_production_scaled == 0:
                    continue

                # Calculate installations in operation phase
                installations_in_operation = sum(
                    pyo.value(model.var_installation[p, t - tau])
                    for tau in model.PROCESS_TIME
                    if (t - tau in model.SYSTEM_TIME)
                    and pyo.value(model.process_operation_start[p]) <= tau <= pyo.value(model.process_operation_end[p])
                )

                # Add existing (brownfield) capacity that is still in operation phase
                op_start = pyo.value(model.process_operation_start[p])
                op_end = pyo.value(model.process_operation_end[p])
                existing_cap_dict = getattr(model, "_existing_capacity_dict", {})
                for (proc, inst_year), cap in existing_cap_dict.items():
                    if proc == p:
                        tau_existing = t - inst_year
                        if op_start <= tau_existing <= op_end:
                            installations_in_operation += cap

                # Capacity = total_production (SCALED) × fg_scale × installations (REAL)
                capacity = total_production_scaled * fg_scale * installations_in_operation
                max_capacity = max(max_capacity, capacity)

            # Check if operation exceeds capacity
            if operation_value < -tolerance:
                violation = abs(operation_value)
                violations.append((p, t, operation_value, max_capacity, "negative"))
                max_violation = max(max_violation, violation)
            elif max_capacity > 0 and operation_value > max_capacity * (1.0 + tolerance):
                violation = operation_value - max_capacity
                violations.append((p, t, operation_value, max_capacity, "exceeds_capacity"))
                max_violation = max(max_violation, violation)

    # Generate summary
    is_valid = len(violations) == 0
    if is_valid:
        summary = "✓ All operation levels respect capacity constraints"
    else:
        summary = (
            f"✗ Found {len(violations)} operation bound violations. "
            f"Max violation: {max_violation:.2e}"
        )

    return {
        "valid": is_valid,
        "violations": violations,
        "max_violation": max_violation,
        "summary": summary,
    }
