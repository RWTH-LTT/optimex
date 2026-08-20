"""
Optimization model construction and solving for temporal LCA-based pathway optimization.

This module creates and solves Pyomo optimization models that minimize environmental
impacts over time while meeting demand constraints and respecting process limits.

## Unit Convention

One "unit" of a process is one reference-flow unit of the underlying LCA process:
installing 1 unit delivers exactly the production stated by its production temporal
distribution, i.e. `sum_tau foreground_production[p, r, tau]`, spread over the whole
operation window. The per-tau entry is therefore the unit's output *per year of
operation*, and the sum over the window is its *lifetime* output.

Consequences:
- `var_installation[p, v]` counts units of vintage v; its installation-dependent
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
from typing import Any, Dict, Tuple, Union

import dill
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
    model._technosphere_vintage_overrides = (
        getattr(scaled_inputs, "foreground_technosphere_vintage_overrides", None) or {}
    )
    model._biosphere_vintage_overrides = (
        getattr(scaled_inputs, "foreground_biosphere_vintage_overrides", None) or {}
    )
    model._production_vintage_overrides = (
        getattr(scaled_inputs, "foreground_production_vintage_overrides", None) or {}
    )

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

    # Build ACTIVE_VINTAGE_TIME index: valid (process, vintage, time) tuples
    # where vintage is active (in operation phase) at time t
    def _build_active_vintage_index(m):
        """
        Build set of valid (process, vintage, time) tuples where vintage is active at time t.

        A vintage v is active at time t if:
        - tau = t - v (process-relative time)
        - op_start <= tau <= op_end (within operation phase)

        This includes both:
        - Greenfield vintages: v in SYSTEM_TIME (new installations)
        - Brownfield vintages: v from existing_capacity (pre-existing installations)
        """
        indices = []
        for p in m.PROCESS:
            op_start = pyo.value(m.process_operation_start[p])
            op_end = pyo.value(m.process_operation_end[p])

            # Greenfield vintages (new installations within system time horizon)
            for v in m.SYSTEM_TIME:
                for t in m.SYSTEM_TIME:
                    tau = t - v  # Process-relative time
                    if v <= t and op_start <= tau <= op_end:
                        indices.append((p, v, t))

            # Brownfield vintages (existing capacity installed before system time)
            for (proc, inst_year), _ in model._existing_capacity_dict.items():
                if proc == p:
                    for t in m.SYSTEM_TIME:
                        tau = t - inst_year
                        if op_start <= tau <= op_end:
                            indices.append((p, inst_year, t))
        return indices

    model.ACTIVE_VINTAGE_TIME = pyo.Set(
        dimen=3,
        initialize=_build_active_vintage_index,
        doc="Set of (process, vintage, time) tuples where vintage is in operation phase at time t",
    )

    # ------------------------------------------------------------------
    # Precomputed lookups
    #
    # The tensors below carry no decision variables, so they are combined here
    # as plain Python numbers. Reading them through Pyomo `Param` objects inside
    # the expression rules costs one indexed lookup with full index validation
    # per scalar, which added up to tens of millions of lookups and dominated
    # model construction.
    # ------------------------------------------------------------------
    active_vintages_by_process_time = {}
    active_processes_by_time = {}
    for _process, _vintage, _time in model.ACTIVE_VINTAGE_TIME:
        active_vintages_by_process_time.setdefault((_process, _time), []).append(
            _vintage
        )
        active_processes_by_time.setdefault(_time, []).append((_process, _vintage))

    system_time_set = set(scaled_inputs.SYSTEM_TIME)
    process_times = list(scaled_inputs.PROCESS_TIME)
    operation_flow_data = dict(scaled_inputs.operation_flow)
    operation_phase = {
        process: (
            pyo.value(model.process_operation_start[process]),
            pyo.value(model.process_operation_end[process]),
        )
        for process in model.PROCESS
    }

    # Background inventory of an intermediate flow at a given time, collapsed
    # over the background databases: sum_bkg G[bkg, i, e] * M[bkg, t].
    inventory_by_background = {}
    for (_bkg, _flow, _emission), _amount in scaled_inputs.background_inventory.items():
        if _amount:
            inventory_by_background.setdefault(_bkg, []).append(
                ((_flow, _emission), _amount)
            )

    mapped_inventory = {}
    for (_bkg, _time), _weight in scaled_inputs.mapping.items():
        if not _weight:
            continue
        for (_flow, _emission), _amount in inventory_by_background.get(_bkg, ()):
            _key = (_flow, _emission, _time)
            mapped_inventory[_key] = mapped_inventory.get(_key, 0.0) + _weight * _amount

    # (elementary flow, time) -> [(intermediate flow, amount), ...]
    mapped_inventory_by_emission = {}
    # (intermediate flow, time) -> [(elementary flow, amount), ...], kept on the
    # model so post-processing can rebuild inventories numerically.
    mapped_inventory_by_flow = {}
    for (_flow, _emission, _time), _amount in mapped_inventory.items():
        mapped_inventory_by_emission.setdefault((_emission, _time), []).append(
            (_flow, _amount)
        )
        mapped_inventory_by_flow.setdefault((_flow, _time), []).append(
            (_emission, _amount)
        )
    model._mapped_inventory_by_flow = mapped_inventory_by_flow
    # The flat form held the same numbers a third time; the groupings cover every
    # use from here on.
    del mapped_inventory, inventory_by_background

    # Characterized impact per unit of an intermediate flow, i.e. the background
    # inventory already multiplied by the characterization factors:
    # (category, time) -> [(intermediate flow, factor), ...]. This collapses the
    # sum over elementary flows out of the impact expressions entirely.
    characterization_by_emission = {}
    for (
        _category,
        _emission,
        _time,
    ), _factor in scaled_inputs.characterization.items():
        if _factor:
            characterization_by_emission.setdefault((_emission, _time), []).append(
                (_category, _factor)
            )

    _impact_factors = {}
    for (_flow, _time), _emissions in mapped_inventory_by_flow.items():
        for _emission, _amount in _emissions:
            for _category, _factor in characterization_by_emission.get(
                (_emission, _time), ()
            ):
                _key = (_category, _time, _flow)
                _impact_factors[_key] = (
                    _impact_factors.get(_key, 0.0) + _factor * _amount
                )

    impact_factors_by_category_time = {}
    for (_category, _time, _flow), _factor in _impact_factors.items():
        if _factor:
            impact_factors_by_category_time.setdefault((_category, _time), []).append(
                (_flow, _factor)
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
        rule=lambda m, p, t: m.var_installation[p, t]
        <= m.process_deployment_limits_max[p, t],
    )

    model.ProcessDeploymentLimitMin = pyo.Constraint(
        model.PROCESS,
        model.SYSTEM_TIME,
        rule=lambda m, p, t: m.var_installation[p, t]
        >= m.process_deployment_limits_min[p, t],
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

    # 3D operation variable: (process, vintage, time)
    # Each vintage cohort has its own operation variable for merit-order dispatch
    model.var_operation = pyo.Var(
        model.ACTIVE_VINTAGE_TIME,
        within=pyo.NonNegativeReals,
        doc="Operational activity level per vintage (process, vintage, time)",
    )

    # Operation limits apply to TOTAL operation across all vintages
    def process_operation_limit_max_rule(m, p, t):
        active_vintages = active_vintages_by_process_time.get((p, t))
        if not active_vintages:
            return pyo.Constraint.Skip
        total_op = sum(m.var_operation[p, v, t] for v in active_vintages)
        return total_op <= m.process_operation_limits_max[p, t]

    model.ProcessOperationLimitMax = pyo.Constraint(
        model.PROCESS,
        model.SYSTEM_TIME,
        rule=process_operation_limit_max_rule,
    )

    def process_operation_limit_min_rule(m, p, t):
        active_vintages = active_vintages_by_process_time.get((p, t))
        if not active_vintages:
            return pyo.Constraint.Skip
        total_op = sum(m.var_operation[p, v, t] for v in active_vintages)
        return total_op >= m.process_operation_limits_min[p, t]

    model.ProcessOperationLimitMin = pyo.Constraint(
        model.PROCESS,
        model.SYSTEM_TIME,
        rule=process_operation_limit_min_rule,
    )

    # Expression builders using sparse vintage override lookup
    #
    # Installation- and operation-driven contributions are always used together,
    # so they are combined into a single Expression per (process, flow, time).
    # The index only covers (process, flow) pairs that carry data: a process
    # consumes a handful of the intermediate flows and emits a handful of the
    # elementary flows, so a dense index would be almost entirely zeros.
    def scaled_flow_expression(
        tensor: dict, overrides: dict, overrides_index: frozenset
    ):
        pairs = {(p, x) for (p, x, _) in tensor} | set(overrides_index)
        rates = {}

        def flow_rate(p, x, v, tau):
            """
            Flow per operating unit and year, at lifecycle stage tau of a vintage-v
            unit. Cached; equal for all vintages without overrides.

            This is the per-tau entry, NOT the sum over the operation window: the
            latter is what one unit emits over its whole lifetime, and using it as
            an annual rate lets a single unit deliver its lifetime flows every year.
            """
            key = (p, x, v, tau) if (p, x) in overrides_index else (p, x, tau)
            if key not in rates:
                if (p, x) in overrides_index:
                    # Vintage-aware: this vintage's rate at this lifecycle stage
                    rate = overrides.get((p, x, tau, v), tensor.get((p, x, tau), 0))
                else:
                    # No overrides: all vintages have the same rate
                    rate = tensor.get((p, x, tau), 0)
                rates[key] = rate
            return rates[key]

        def expr(m, p, x, t):
            is_operation_flow = bool(operation_flow_data.get((p, x)))
            op_start, op_end = operation_phase[p]

            # Installation-driven part: flows outside the operation phase scale
            # with the capacity installed in vintage t - tau.
            result = 0
            for tau in process_times:
                vintage = t - tau
                if vintage not in system_time_set:
                    continue
                if is_operation_flow and op_start <= tau <= op_end:
                    continue
                # Check sparse override first, fall back to base 3D tensor
                flow_value = overrides.get((p, x, tau, vintage))
                if flow_value is None:
                    flow_value = tensor.get((p, x, tau), 0)
                if flow_value:
                    result += flow_value * m.var_installation[p, vintage]

            # Operation-driven part: summed over the vintages operating at t,
            # each at its own lifecycle stage tau = t - v.
            if is_operation_flow:
                for v in active_vintages_by_process_time.get((p, t), ()):
                    rate = flow_rate(p, x, v, t - v)
                    if rate:
                        result += rate * m.var_operation[p, v, t]

            return result

        index = [
            (p, x, t) for (p, x) in sorted(pairs) for t in scaled_inputs.SYSTEM_TIME
        ]
        return pairs, pyo.Expression(index, rule=expr)

    technosphere_pairs, model.scaled_technosphere_flow = scaled_flow_expression(
        dict(scaled_inputs.foreground_technosphere),
        model._technosphere_vintage_overrides,
        model._technosphere_overrides_index,
    )
    biosphere_pairs, model.scaled_biosphere_flow = scaled_flow_expression(
        dict(scaled_inputs.foreground_biosphere),
        model._biosphere_vintage_overrides,
        model._biosphere_overrides_index,
    )
    # Internal demand has no vintage overrides
    internal_demand_pairs, model.scaled_internal_demand_flow = scaled_flow_expression(
        dict(scaled_inputs.internal_demand_technosphere), {}, frozenset()
    )

    # Which intermediate/elementary flows a process actually has, so the sums
    # below skip the pairs whose expression would be zero.
    technosphere_flows_by_process = {}
    for _process, _flow in technosphere_pairs:
        technosphere_flows_by_process.setdefault(_process, set()).add(_flow)
    biosphere_flows_by_process = {}
    for _process, _emission in biosphere_pairs:
        biosphere_flows_by_process.setdefault(_process, set()).add(_emission)

    def scaled_inventory_tensor(model, p, e, t):
        """
        Returns a Pyomo expression for the total inventory impact for a given
        process p, elementary flow e, and time step t.
        """

        process_flows = technosphere_flows_by_process.get(p, ())
        background = sum(
            amount * model.scaled_technosphere_flow[p, i, t]
            for i, amount in mapped_inventory_by_emission.get((e, t), ())
            if i in process_flows
        )
        if e in biosphere_flows_by_process.get(p, ()):
            return background + model.scaled_biosphere_flow[p, e, t]
        return background

    model.scaled_inventory = pyo.Expression(
        model.PROCESS,
        model.ELEMENTARY_FLOW,
        model.SYSTEM_TIME,
        rule=scaled_inventory_tensor,
    )

    # O(1) check if production overrides exist for a given process/product
    def has_production_overrides(p, r):
        """Check if any vintage overrides exist for this process/product."""
        return (p, r) in model._production_overrides_index

    production_data = dict(scaled_inputs.foreground_production)
    _production_rates = {}

    def production_rate(p, r, v, tau):
        """
        Annual production of r by one running vintage-v unit of p at lifecycle
        stage tau, cached.

        This is the per-tau entry, not the sum over the operation window: the sum
        is the unit's LIFETIME output, and using it as an annual rate would let one
        unit deliver its whole lifetime output in every operating year.

        Without overrides the rate is the same for every vintage, so it is cached
        per (process, product, tau) instead.
        """
        key = (p, r, v, tau) if has_production_overrides(p, r) else (p, r, tau)
        if key not in _production_rates:
            if has_production_overrides(p, r):
                rate = model._production_vintage_overrides.get(
                    (p, r, tau, v), production_data.get((p, r, tau), 0)
                )
            else:
                rate = production_data.get((p, r, tau), 0)
            _production_rates[key] = rate
        return _production_rates[key]

    def operation_capacity_constraint_rule(model, p, v, t):
        """
        Per-vintage capacity constraint: var_operation[p, v, t] ≤ units of vintage v.

        Both variables count process UNITS: var_installation[p, v] is how many units
        of vintage v exist, var_operation[p, v, t] is how many of them are running in
        year t. A unit's production temporal distribution states what one unit yields
        over its whole lifetime, so the annual output of a running unit is the
        per-tau entry, and the bound here is a plain unit count comparison.

        For greenfield (v in SYSTEM_TIME): bound is var_installation[p, v]
        For brownfield (v not in SYSTEM_TIME): bound is existing_capacity[p, v]
        """
        if v in model.SYSTEM_TIME:
            # Greenfield: units available come from var_installation
            return model.var_operation[p, v, t] <= model.var_installation[p, v]

        # Brownfield: units available come from the existing_capacity dict
        existing_cap = model._existing_capacity_dict.get((p, v), 0)
        if existing_cap == 0:
            return pyo.Constraint.Skip
        return model.var_operation[p, v, t] <= existing_cap

    model.OperationCapacity = pyo.Constraint(
        model.ACTIVE_VINTAGE_TIME,
        rule=operation_capacity_constraint_rule,
    )

    def as_constraint(expression):
        """
        Turn a rule result into something `pyo.Constraint` accepts.

        Sparse expressions can collapse to plain numbers (a process that has no
        flows of the requested kind), in which case the comparison evaluates to a
        Python bool rather than a Pyomo object.
        """
        if expression is True:
            return pyo.Constraint.Feasible
        if expression is False:
            return pyo.Constraint.Infeasible
        return expression

    def product_demand_fulfillment_rule(model, r, t):
        """
        Demand constraint: total_production == external_demand + internal_consumption

        With 3D var_operation[p, v, t], sum production across all active vintages.
        The rate used is the vintage's annual output at its current lifecycle stage
        tau = t - v; the sum over the whole operation window is the unit's lifetime
        output, not its annual output.
        """
        total_production = 0

        # Sum production across all active vintages at time t
        for p, v in active_processes_by_time.get(t, ()):
            rate = production_rate(p, r, v, t - v)
            if rate:
                total_production += rate * model.var_operation[p, v, t]

        external_demand = model.demand[r, t]
        internal_consumption = sum(
            model.scaled_internal_demand_flow[p, r, t]
            for p in model.PROCESS
            if (p, r) in internal_demand_pairs
        )
        return as_constraint(total_production == external_demand + internal_consumption)

    model.ProductDemandFulfillment = pyo.Constraint(
        model.PRODUCT, model.SYSTEM_TIME, rule=product_demand_fulfillment_rule
    )

    def category_process_time_specific_impact(model, c, p, t):
        """
        Impact of process p at time t in category c.

        The sum over elementary flows is folded into the background inventory
        beforehand (`impact_factors_by_category_time`), so only the intermediate
        flows and the process's own emissions remain here. Writing it as
        `sum_e Q[c, e, t] * scaled_inventory[p, e, t]` is mathematically the same
        but builds an expression tree that is orders of magnitude larger.
        """
        process_flows = technosphere_flows_by_process.get(p, ())
        background = sum(
            factor * model.scaled_technosphere_flow[p, i, t]
            for i, factor in impact_factors_by_category_time.get((c, t), ())
            if i in process_flows
        )
        foreground = 0
        for e in biosphere_flows_by_process.get(p, ()):
            factor = scaled_inputs.characterization.get((c, e, t))
            if factor:
                foreground += factor * model.scaled_biosphere_flow[p, e, t]
        return background + foreground

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
            return (
                model.time_specific_impact[c, t]
                <= model._category_impact_limits[(c, t)]
            )
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

    def warn_about_unusable_flow_limits():
        """
        Report flow limits that cannot do anything.

        `OptimizationModelInputs` rejects unknown flows when it is constructed,
        but assigning a limit to an already-built instance bypasses that. The
        constraint rules below are indexed over the model's own sets, so such a
        limit would otherwise be dropped without a word.
        """
        known_flows = (
            set(scaled_inputs.PRODUCT)
            | set(scaled_inputs.INTERMEDIATE_FLOW)
            | set(scaled_inputs.ELEMENTARY_FLOW)
        )
        flows_with_data = (
            {key[1] for key in scaled_inputs.foreground_technosphere}
            | {key[1] for key in scaled_inputs.foreground_biosphere}
            | {key[1] for key in scaled_inputs.foreground_production}
            | {key[1] for key in scaled_inputs.internal_demand_technosphere}
            | {key[2] for key in scaled_inputs.background_inventory}
        )

        limited_flows = set(model._cumulative_flow_limits_max)
        limited_flows |= set(model._cumulative_flow_limits_min)
        limited_flows |= {key[0] for key in model._flow_limits_max}
        limited_flows |= {key[0] for key in model._flow_limits_min}

        unknown = sorted(limited_flows - known_flows)
        if unknown:
            logger.warning(
                f"Ignoring flow limits for {unknown}: these flows are not in the "
                "model. Elementary flows without a characterization factor in any "
                "category are dropped during LCA processing; list them in "
                "`LCAConfig.background_inventory.retain_flows` to keep them."
            )

        without_data = sorted((limited_flows & known_flows) - flows_with_data)
        if without_data:
            logger.warning(
                f"Flow limits for {without_data} can never bind: no process "
                "exchanges these flows, so the limited amount is always zero."
            )

        limited_times = {key[1] for key in model._flow_limits_max}
        limited_times |= {key[1] for key in model._flow_limits_min}
        unknown_times = sorted(limited_times - set(scaled_inputs.SYSTEM_TIME))
        if unknown_times:
            logger.warning(
                f"Ignoring flow limits for years {unknown_times}: outside the "
                "system time horizon."
            )

    warn_about_unusable_flow_limits()

    # Expression for total product output at time t (in SCALED units)
    def total_product_flow_rule(model, r, t):
        """
        Calculate total product output at time t, summing across all active vintages.
        Uses each vintage's annual output at its lifecycle stage tau = t - v.
        """
        total = 0
        for p, v in active_processes_by_time.get(t, ()):
            rate = production_rate(p, r, v, t - v)
            if rate:
                total += rate * model.var_operation[p, v, t]
        return total

    model.total_product_flow = pyo.Expression(
        model.PRODUCT, model.SYSTEM_TIME, rule=total_product_flow_rule
    )

    # Expression for total intermediate flow consumed at time t (in SCALED units)
    def total_intermediate_flow_rule(model, i, t):
        return sum(
            model.scaled_technosphere_flow[p, i, t]
            for p in model.PROCESS
            if (p, i) in technosphere_pairs
        )

    model.total_intermediate_flow = pyo.Expression(
        model.INTERMEDIATE_FLOW, model.SYSTEM_TIME, rule=total_intermediate_flow_rule
    )

    # Expression for total elementary flow at time t (in SCALED units)
    # This includes both foreground biosphere flows AND background inventory flows
    # (flows from intermediate flows going through background databases)
    def total_elementary_flow_rule(model, e, t):
        # Identical to summing the per-process inventory, which already covers
        # both the foreground biosphere flows and the background contribution.
        return sum(model.scaled_inventory[p, e, t] for p in model.PROCESS)

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
                " — the model is unbounded. Common cause: installation-dependent"
                " flows (non-operation technosphere/biosphere exchanges) that produce"
                " a net-negative impact allow var_installation to grow without bound."
                " Fix by setting finite process_deployment_limits_max for affected"
                " processes, or ensure installation-dependent flows do not create"
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
    logger.info(f"Solver [{solver_name}] termination: {termination}")

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
        operation_value = pyo.value(model.var_operation[p, v, t])

        # Units available from this vintage (operation and installation are both
        # unit counts, so no production rate enters here)
        if v in model.SYSTEM_TIME:
            max_capacity = pyo.value(model.var_installation[p, v])
        else:
            max_capacity = existing_cap_dict.get((p, v), 0)

        # Check if operation exceeds capacity
        if operation_value < -tolerance:
            violation = abs(operation_value)
            violations.append((p, v, t, operation_value, max_capacity, "negative"))
            max_violation = max(max_violation, violation)
        elif max_capacity > 0 and operation_value > max_capacity * (1.0 + tolerance):
            violation = operation_value - max_capacity
            violations.append(
                (p, v, t, operation_value, max_capacity, "exceeds_capacity")
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
