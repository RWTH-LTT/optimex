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
- `category_impact_limits[(c, t)]`: time-specific maximum impact allowed [SCALED]
- `cumulative_category_impact_limits[c]`: cumulative maximum impact allowed [SCALED]

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
    production [kg SCALED/operation] × var_operation [#] = demand [kg SCALED] ✓

OperationLimit (LHS):
    var_operation [#] × production [kg SCALED/operation] × fg_scale = [kg REAL]
OperationLimit (RHS):
    production [kg SCALED/process] × fg_scale × var_installation [#] = [kg REAL]
```
"""

import dill
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, Union

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
    debug_path: str = None,
) -> pyo.ConcreteModel:
    """
    Build a Pyomo ConcreteModel for the optimization problem based on the provided
    inputs.

    This function constructs a fully defined Pyomo model using data from a `OptimizationModelInputs`
    instance. It uses flexible operation mode where processes can operate between 0 and
    their maximum installed capacity.

    Parameters
    ----------
    inputs : OptimizationModelInputs
        Structured input data containing all flows, mappings, and constraints
        required for model construction.
    name : str
        Name of the Pyomo model instance.
    objective_category : str
        The category of impact to be minimized in the optimization problem.
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
    # Always use 3D base tensors - vintage overrides are applied via sparse lookup
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

    # Store sparse vintage overrides as Python dicts (not Pyomo params)
    # These are looked up at expression construction time
    model._technosphere_vintage_overrides = getattr(
        scaled_inputs, 'foreground_technosphere_vintage_overrides', None
    ) or {}
    model._biosphere_vintage_overrides = getattr(
        scaled_inputs, 'foreground_biosphere_vintage_overrides', None
    ) or {}
    model._production_vintage_overrides = getattr(
        scaled_inputs, 'foreground_production_vintage_overrides', None
    ) or {}

    # Precompute sets of (process, flow) pairs that have overrides for O(1) lookup
    model._technosphere_overrides_index = frozenset(
        (k[0], k[1]) for k in model._technosphere_vintage_overrides
    )
    model._biosphere_overrides_index = frozenset(
        (k[0], k[1]) for k in model._biosphere_vintage_overrides
    )
    model._production_overrides_index = frozenset(
        (k[0], k[1]) for k in model._production_vintage_overrides
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
    model.process_deployment_limits_max = pyo.Param(
        model.PROCESS,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="maximum time specific process deployment limit S_max",
        default=scaled_inputs.process_deployment_limits_max_default,
        initialize=(
            scaled_inputs.process_deployment_limits_max
            if scaled_inputs.process_deployment_limits_max is not None
            else {}
        ),
    )
    model.process_deployment_limits_min = pyo.Param(
        model.PROCESS,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="minimum time specific process deployment limit S_min",
        default=scaled_inputs.process_deployment_limits_min_default,
        initialize=(
            scaled_inputs.process_deployment_limits_min
            if scaled_inputs.process_deployment_limits_min is not None
            else {}
        ),
    )
    model.process_operation_limits_max = pyo.Param(
        model.PROCESS,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="maximum time specific process operation limit O_max",
        default=scaled_inputs.process_operation_limits_max_default,
        initialize=(
            scaled_inputs.process_operation_limits_max
            if scaled_inputs.process_operation_limits_max is not None
            else {}
        ),
    )
    model.process_operation_limits_min = pyo.Param(
        model.PROCESS,
        model.SYSTEM_TIME,
        within=pyo.Reals,
        doc="minimum time specific process operation limit O_min",
        default=scaled_inputs.process_operation_limits_min_default,
        initialize=(
            scaled_inputs.process_operation_limits_min
            if scaled_inputs.process_operation_limits_min is not None
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

    # Store category impact limit data for constraint generation
    model._category_impact_limits = (
        scaled_inputs.category_impact_limits
        if scaled_inputs.category_impact_limits is not None
        else {}
    )

    model.cumulative_category_impact_limits = pyo.Param(
        model.CATEGORY,
        within=pyo.Reals,
        doc="cumulative maximum impact limit per category",
        default=float("inf"),
        initialize=(
            scaled_inputs.cumulative_category_impact_limits
            if scaled_inputs.cumulative_category_impact_limits is not None
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

    # Deployment limits
    model.ProcessDeploymentLimitMax = pyo.Constraint(
        model.PROCESS,
        model.SYSTEM_TIME,
        rule=lambda m, p, t: m.var_installation[p, t] <= m.process_deployment_limits_max[p, t],
    )

    model.ProcessDeploymentLimitMin = pyo.Constraint(
        model.PROCESS,
        model.SYSTEM_TIME,
        rule=lambda m, p, t: m.var_installation[p, t] >= m.process_deployment_limits_min[p, t],
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

    # Operation limits
    model.ProcessOperationLimitMax = pyo.Constraint(
        model.PROCESS,
        model.SYSTEM_TIME,
        rule=lambda m, p, t: m.var_operation[p, t] <= m.process_operation_limits_max[p, t],
    )

    model.ProcessOperationLimitMin = pyo.Constraint(
        model.PROCESS,
        model.SYSTEM_TIME,
        rule=lambda m, p, t: m.var_operation[p, t] >= m.process_operation_limits_min[p, t],
    )

    # Expression builders using sparse vintage override lookup
    # Base tensor is always 3D; overrides are checked first for vintage-specific values
    def scale_tensor_by_installation(tensor: pyo.Param, flow_set: str, overrides: dict, overrides_index: frozenset):
        def expr(m, p, x, t):
            result = 0
            for tau in m.PROCESS_TIME:
                vintage = t - tau
                if (vintage in m.SYSTEM_TIME) and (
                    not in_operation_phase(p, tau) or not m.operation_flow[p, x]
                ):
                    # Check sparse override first, fall back to base 3D tensor
                    key = (p, x, tau, vintage)
                    if key in overrides:
                        flow_value = overrides[key]
                    else:
                        flow_value = tensor[p, x, tau]
                    result += flow_value * m.var_installation[p, vintage]
            return result

        return pyo.Expression(
            model.PROCESS, getattr(model, flow_set), model.SYSTEM_TIME, rule=expr
        )

    def scale_tensor_by_operation(tensor: pyo.Param, flow_set: str, overrides: dict, overrides_index: frozenset):
        def expr(m, p, x, t):
            # Only apply operational scaling to flows marked as operational
            if pyo.value(m.operation_flow[p, x]) == 0:
                return 0

            op_start = pyo.value(m.process_operation_start[p])
            op_end = pyo.value(m.process_operation_end[p])

            # O(1) check if overrides exist for this (process, flow) combination
            if (p, x) in overrides_index:
                # Vintage-aware: sum flow rates across ALL operating taus
                # No SYSTEM_TIME filter - include all taus (handles brownfield naturally)
                flow_rate = sum(
                    overrides.get((p, x, tau, t - tau), tensor[p, x, tau])
                    for tau in m.PROCESS_TIME
                    if op_start <= tau <= op_end
                )

                return flow_rate * m.var_operation[p, t]
            else:
                # No overrides: use efficient 3D computation with var_operation
                total_flow_per_installation = sum(
                    tensor[p, x, tau]
                    for tau in m.PROCESS_TIME
                    if op_start <= tau <= op_end
                )
                return total_flow_per_installation * m.var_operation[p, t]

        return pyo.Expression(
            model.PROCESS, getattr(model, flow_set), model.SYSTEM_TIME, rule=expr
        )

    # Create expressions with sparse vintage override lookup
    model.scaled_technosphere_dependent_on_installation = (
        scale_tensor_by_installation(
            model.foreground_technosphere,
            "INTERMEDIATE_FLOW",
            model._technosphere_vintage_overrides,
            model._technosphere_overrides_index,
        )
    )
    model.scaled_biosphere_dependent_on_installation = scale_tensor_by_installation(
        model.foreground_biosphere,
        "ELEMENTARY_FLOW",
        model._biosphere_vintage_overrides,
        model._biosphere_overrides_index,
    )
    model.scaled_technosphere_dependent_on_operation = scale_tensor_by_operation(
        model.foreground_technosphere,
        "INTERMEDIATE_FLOW",
        model._technosphere_vintage_overrides,
        model._technosphere_overrides_index,
    )
    model.scaled_biosphere_dependent_on_operation = scale_tensor_by_operation(
        model.foreground_biosphere,
        "ELEMENTARY_FLOW",
        model._biosphere_vintage_overrides,
        model._biosphere_overrides_index,
    )
    # Internal demand has no vintage overrides
    _empty_index = frozenset()
    model.scaled_internal_demand_dependent_on_installation = (
        scale_tensor_by_installation(
            model.internal_demand_technosphere, "PRODUCT", {}, _empty_index
        )
    )
    model.scaled_internal_demand_dependent_on_operation = (
        scale_tensor_by_operation(
            model.internal_demand_technosphere, "PRODUCT", {}, _empty_index
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

    # Helper to get production value with sparse override lookup
    def get_production_value(p, r, tau, vintage):
        """Get production value, checking sparse overrides first."""
        key = (p, r, tau, vintage)
        if key in model._production_vintage_overrides:
            return model._production_vintage_overrides[key]
        return model.foreground_production[p, r, tau]

    # O(1) check if production overrides exist for a given process/product
    def has_production_overrides(p, r):
        """Check if any vintage overrides exist for this process/product."""
        return (p, r) in model._production_overrides_index

    def operation_capacity_constraint_rule(model, p, r, t):
        """
        Capacity constraint: o_t ≤ total_capacity_at_time_t

        This constraint ensures operation level cannot exceed the total production
        capacity provided by active installations.

        With vintage overrides: different installation cohorts may have different production rates.
        Without overrides: all installations have the same production rate (efficient 3D path).
        """
        fg_scale = model.scales["foreground"]
        op_start = pyo.value(model.process_operation_start[p])
        op_end = pyo.value(model.process_operation_end[p])

        if has_production_overrides(p, r):
            # Vintage-aware capacity calculation with sparse override lookup
            has_production = any(
                pyo.value(get_production_value(p, r, tau, min(model.SYSTEM_TIME))) != 0
                for tau in model.PROCESS_TIME
                if op_start <= tau <= op_end
            )
            if not has_production:
                return pyo.Constraint.Skip

            # Total capacity from all vintage cohorts
            total_capacity = 0
            for tau in model.PROCESS_TIME:
                vintage = t - tau
                if vintage in model.SYSTEM_TIME and op_start <= tau <= op_end:
                    production_per_unit = sum(
                        get_production_value(p, r, tau_op, vintage)
                        for tau_op in model.PROCESS_TIME
                        if op_start <= tau_op <= op_end
                    )
                    total_capacity += production_per_unit * model.var_installation[p, vintage]

            # Add brownfield capacity
            for (proc, inst_year), cap in model._existing_capacity_dict.items():
                if proc == p:
                    tau_existing = t - inst_year
                    if op_start <= tau_existing <= op_end:
                        nearest_vintage = min(model.SYSTEM_TIME)
                        production_per_unit = sum(
                            pyo.value(get_production_value(p, r, tau_op, nearest_vintage))
                            for tau_op in model.PROCESS_TIME
                            if op_start <= tau_op <= op_end
                        )
                        total_capacity += production_per_unit * cap

            capacity = total_capacity * fg_scale
        else:
            # 3D efficient path: no overrides, all vintages have same production rate
            total_production_scaled = sum(
                model.foreground_production[p, r, tau]
                for tau in model.PROCESS_TIME
                if op_start <= tau <= op_end
            )
            if total_production_scaled == 0:
                return pyo.Constraint.Skip

            installations_in_operation = sum(
                model.var_installation[p, t - tau]
                for tau in model.PROCESS_TIME
                if (t - tau in model.SYSTEM_TIME) and op_start <= tau <= op_end
            )

            for (proc, inst_year), cap in model._existing_capacity_dict.items():
                if proc == p:
                    tau_existing = t - inst_year
                    if op_start <= tau_existing <= op_end:
                        installations_in_operation += cap

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
        Demand constraint: total_production == external_demand + internal_consumption

        With vintage overrides: production from each vintage cohort may differ.
        Without overrides: all installations have the same production rate (efficient 3D path).
        """
        # Check if ANY process has production overrides for this product
        any_overrides = any(has_production_overrides(p, r) for p in model.PROCESS)

        if any_overrides:
            # Vintage-aware production calculation using var_operation
            # The OperationCapacity constraint bounds var_operation by vintage-specific
            # production capacity. For each operating tau, the vintage is (t - tau).
            total_production = 0
            for p in model.PROCESS:
                op_start = pyo.value(model.process_operation_start[p])
                op_end = pyo.value(model.process_operation_end[p])

                # Check if THIS specific process has production overrides
                if has_production_overrides(p, r):
                    # 4D path: sum production rates for taus where vintage is in SYSTEM_TIME
                    production_rate = sum(
                        get_production_value(p, r, tau, t - tau)
                        for tau in model.PROCESS_TIME
                        if op_start <= tau <= op_end and (t - tau) in model.SYSTEM_TIME
                    )

                    # Add contribution from existing capacity (brownfield)
                    # Only add ONCE per process if ANY existing capacity is operating
                    has_operating_existing = any(
                        proc == p and op_start <= (t - inst_year) <= op_end
                        for (proc, inst_year), cap in model._existing_capacity_dict.items()
                    )
                    if has_operating_existing:
                        nearest_vintage = min(model.SYSTEM_TIME)
                        existing_prod_rate = sum(
                            pyo.value(get_production_value(p, r, tau_op, nearest_vintage))
                            for tau_op in model.PROCESS_TIME
                            if op_start <= tau_op <= op_end
                        )
                        production_rate += existing_prod_rate
                else:
                    # 3D path for this process: no overrides, use efficient calculation
                    # Sum over ALL operating taus (no vintage filter needed)
                    production_rate = sum(
                        model.foreground_production[p, r, tau]
                        for tau in model.PROCESS_TIME
                        if op_start <= tau <= op_end
                    )

                total_production += production_rate * model.var_operation[p, t]
        else:
            # 3D efficient path: no overrides
            total_production = sum(
                sum(
                    model.foreground_production[p, r, tau]
                    for tau in model.PROCESS_TIME
                    if model.process_operation_start[p] <= tau <= model.process_operation_end[p]
                )
                * model.var_operation[p, t]
                for p in model.PROCESS
            )

        external_demand = model.demand[r, t]
        internal_consumption = sum(
            model.scaled_internal_demand_dependent_on_installation[p, r, t]
            + model.scaled_internal_demand_dependent_on_operation[p, r, t]
            for p in model.PROCESS
        )
        return total_production == external_demand + internal_consumption

    model.ProductDemandFulfillment = pyo.Constraint(
        model.PRODUCT, model.SYSTEM_TIME, rule=product_demand_fulfillment_rule
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

    # Time-specific impact (impact at a specific time across all processes)
    def time_specific_impact_rule(model, c, t):
        return sum(model.specific_impact[c, p, t] for p in model.PROCESS)

    model.time_specific_impact = pyo.Expression(
        model.CATEGORY, model.SYSTEM_TIME, rule=time_specific_impact_rule
    )

    # Time-specific category impact limits
    def category_impact_limits_rule(model, c, t):
        if (c, t) in model._category_impact_limits:
            return model.time_specific_impact[c, t] <= model._category_impact_limits[(c, t)]
        return pyo.Constraint.Skip

    model.CategoryImpactLimits = pyo.Constraint(
        model.CATEGORY, model.SYSTEM_TIME, rule=category_impact_limits_rule
    )

    # Cumulative category impact limit
    def cumulative_category_impact_limit_rule(model, c):
        return model.total_impact[c] <= model.cumulative_category_impact_limits[c]

    model.CumulativeCategoryImpactLimits = pyo.Constraint(
        model.CATEGORY, rule=cumulative_category_impact_limit_rule
    )

    # Flow limits
    # Store flow limits data for constraint generation
    model._flow_limits_max = (
        scaled_inputs.flow_limits_max
        if scaled_inputs.flow_limits_max is not None
        else {}
    )
    model._flow_limits_min = (
        scaled_inputs.flow_limits_min
        if scaled_inputs.flow_limits_min is not None
        else {}
    )
    model._cumulative_flow_limits_max = (
        scaled_inputs.cumulative_flow_limits_max
        if scaled_inputs.cumulative_flow_limits_max is not None
        else {}
    )
    model._cumulative_flow_limits_min = (
        scaled_inputs.cumulative_flow_limits_min
        if scaled_inputs.cumulative_flow_limits_min is not None
        else {}
    )

    # Expression for total product output at time t (in SCALED units)
    def total_product_flow_rule(model, r, t):
        # Check if ANY process has production overrides for this product
        any_overrides = any(has_production_overrides(p, r) for p in model.PROCESS)

        if any_overrides:
            # Vintage-aware: use var_operation with vintage-specific production rates
            # (consistent with demand fulfillment constraint)
            total = 0
            for p in model.PROCESS:
                op_start = pyo.value(model.process_operation_start[p])
                op_end = pyo.value(model.process_operation_end[p])

                # Sum production rates across all operating taus/vintages
                production_rate = sum(
                    get_production_value(p, r, tau, t - tau)
                    for tau in model.PROCESS_TIME
                    if op_start <= tau <= op_end and (t - tau) in model.SYSTEM_TIME
                )

                total += production_rate * model.var_operation[p, t]
            return total
        else:
            # 3D efficient path: no overrides
            return sum(
                sum(
                    model.foreground_production[p, r, tau]
                    for tau in model.PROCESS_TIME
                    if model.process_operation_start[p] <= tau <= model.process_operation_end[p]
                )
                * model.var_operation[p, t]
                for p in model.PROCESS
            )

    model.total_product_flow = pyo.Expression(
        model.PRODUCT, model.SYSTEM_TIME, rule=total_product_flow_rule
    )

    # Expression for total intermediate flow consumed at time t (in SCALED units)
    def total_intermediate_flow_rule(model, i, t):
        return sum(
            model.scaled_technosphere_dependent_on_installation[p, i, t]
            + model.scaled_technosphere_dependent_on_operation[p, i, t]
            for p in model.PROCESS
        )

    model.total_intermediate_flow = pyo.Expression(
        model.INTERMEDIATE_FLOW, model.SYSTEM_TIME, rule=total_intermediate_flow_rule
    )

    # Expression for total elementary flow at time t (in SCALED units)
    def total_elementary_flow_rule(model, e, t):
        return sum(
            model.scaled_biosphere_dependent_on_installation[p, e, t]
            + model.scaled_biosphere_dependent_on_operation[p, e, t]
            for p in model.PROCESS
        )

    model.total_elementary_flow = pyo.Expression(
        model.ELEMENTARY_FLOW, model.SYSTEM_TIME, rule=total_elementary_flow_rule
    )

    # Helper function to get total flow for any flow type
    def get_total_flow(model, f, t):
        if f in model.PRODUCT:
            return model.total_product_flow[f, t]
        elif f in model.INTERMEDIATE_FLOW:
            return model.total_intermediate_flow[f, t]
        elif f in model.ELEMENTARY_FLOW:
            return model.total_elementary_flow[f, t]
        else:
            return 0

    # Time-specific flow limit constraints (max)
    def flow_limit_max_rule(model, f, t):
        if (f, t) not in model._flow_limits_max:
            return pyo.Constraint.Skip
        fg_scale = model.scales["foreground"]
        limit = model._flow_limits_max[(f, t)]
        total_flow = get_total_flow(model, f, t)
        # total_flow is SCALED, convert limit to scaled units
        return total_flow <= limit / fg_scale

    model.FlowLimitMax = pyo.Constraint(
        model.FLOW, model.SYSTEM_TIME, rule=flow_limit_max_rule
    )

    # Time-specific flow limit constraints (min)
    def flow_limit_min_rule(model, f, t):
        if (f, t) not in model._flow_limits_min:
            return pyo.Constraint.Skip
        fg_scale = model.scales["foreground"]
        limit = model._flow_limits_min[(f, t)]
        total_flow = get_total_flow(model, f, t)
        # total_flow is SCALED, convert limit to scaled units
        return total_flow >= limit / fg_scale

    model.FlowLimitMin = pyo.Constraint(
        model.FLOW, model.SYSTEM_TIME, rule=flow_limit_min_rule
    )

    # Cumulative flow limit constraints (max)
    def cumulative_flow_limit_max_rule(model, f):
        if f not in model._cumulative_flow_limits_max:
            return pyo.Constraint.Skip
        fg_scale = model.scales["foreground"]
        limit = model._cumulative_flow_limits_max[f]
        total_flow = sum(get_total_flow(model, f, t) for t in model.SYSTEM_TIME)
        # total_flow is SCALED, convert limit to scaled units
        return total_flow <= limit / fg_scale

    model.CumulativeFlowLimitMax = pyo.Constraint(
        model.FLOW, rule=cumulative_flow_limit_max_rule
    )

    # Cumulative flow limit constraints (min)
    def cumulative_flow_limit_min_rule(model, f):
        if f not in model._cumulative_flow_limits_min:
            return pyo.Constraint.Skip
        fg_scale = model.scales["foreground"]
        limit = model._cumulative_flow_limits_min[f]
        total_flow = sum(get_total_flow(model, f, t) for t in model.SYSTEM_TIME)
        # total_flow is SCALED, convert limit to scaled units
        return total_flow >= limit / fg_scale

    model.CumulativeFlowLimitMin = pyo.Constraint(
        model.FLOW, rule=cumulative_flow_limit_min_rule
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
        A solved Pyomo model.
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
        If model is not solved.
    """
    if not hasattr(model, "var_operation"):
        raise ValueError("Model must have var_operation")

    violations = []
    max_violation = 0.0
    fg_scale = model.scales["foreground"]

    # Get sparse overrides for production with precomputed index for O(1) lookup
    production_overrides = getattr(model, "_production_vintage_overrides", {}) or {}
    production_overrides_index = getattr(model, "_production_overrides_index", frozenset())

    def get_prod_value(p, r, tau, vintage):
        """Get production value, checking sparse overrides first."""
        key = (p, r, tau, vintage)
        if key in production_overrides:
            return production_overrides[key]
        return pyo.value(model.foreground_production[p, r, tau])

    def has_prod_overrides(p, r):
        """O(1) check if any vintage overrides exist for this process/product."""
        return (p, r) in production_overrides_index

    for p in model.PROCESS:
        for t in model.SYSTEM_TIME:
            operation_value = pyo.value(model.var_operation[p, t])
            op_start = pyo.value(model.process_operation_start[p])
            op_end = pyo.value(model.process_operation_end[p])

            max_capacity = 0.0
            for r in model.PRODUCT:
                if has_prod_overrides(p, r):
                    # Vintage-aware capacity calculation with sparse override lookup
                    total_capacity = 0.0
                    for tau in model.PROCESS_TIME:
                        vintage = t - tau
                        if vintage in model.SYSTEM_TIME and op_start <= tau <= op_end:
                            production_rate = sum(
                                get_prod_value(p, r, tau_op, vintage)
                                for tau_op in model.PROCESS_TIME
                                if op_start <= tau_op <= op_end
                            )
                            installed = pyo.value(model.var_installation[p, vintage])
                            total_capacity += production_rate * installed

                    existing_cap_dict = getattr(model, "_existing_capacity_dict", {})
                    for (proc, inst_year), cap in existing_cap_dict.items():
                        if proc == p:
                            tau_existing = t - inst_year
                            if op_start <= tau_existing <= op_end:
                                nearest_vintage = min(model.SYSTEM_TIME)
                                production_rate = sum(
                                    get_prod_value(p, r, tau_op, nearest_vintage)
                                    for tau_op in model.PROCESS_TIME
                                    if op_start <= tau_op <= op_end
                                )
                                total_capacity += production_rate * cap

                    capacity = total_capacity * fg_scale
                else:
                    # 3D efficient path: no overrides
                    total_production_scaled = sum(
                        pyo.value(model.foreground_production[p, r, tau])
                        for tau in model.PROCESS_TIME
                        if op_start <= tau <= op_end
                    )
                    if total_production_scaled == 0:
                        continue

                    installations_in_operation = sum(
                        pyo.value(model.var_installation[p, t - tau])
                        for tau in model.PROCESS_TIME
                        if (t - tau in model.SYSTEM_TIME) and op_start <= tau <= op_end
                    )

                    existing_cap_dict = getattr(model, "_existing_capacity_dict", {})
                    for (proc, inst_year), cap in existing_cap_dict.items():
                        if proc == p:
                            tau_existing = t - inst_year
                            if op_start <= tau_existing <= op_end:
                                installations_in_operation += cap

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
