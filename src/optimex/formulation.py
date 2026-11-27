"""
Backend-independent formulation of the optimization problem.

`build_formulation()` turns `OptimizationModelInputs` into a linear program in
plain Python data: variables with bounds, constraint rows as
{variable index: coefficient} dicts with a sense and right-hand side, the
objective, and the coefficients of the reporting expressions (flows, impacts,
costs). The Pyomo model (`optimizer.create_model`) and the linopy model
(`optimex.linopy_backend`) are both emitted from it, so they cannot drift apart.

See `optimex.optimizer` for the unit and scaling conventions.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

from optimex.converter import OptimizationModelInputs

INFINITY = float("inf")

# Variable families, in the order their indices are assigned
VARIABLE_FAMILIES = (
    "var_installation",
    "var_operation",
    "var_operation_total",
    "var_capacity_total",
)


def accumulate(target: dict, terms: dict, factor: float = 1.0) -> None:
    """Add `factor * terms` to the {variable index: coefficient} dict `target`."""
    for var, coefficient in terms.items():
        target[var] = target.get(var, 0.0) + factor * coefficient


@dataclass
class ConstraintBlock:
    """Rows of one named constraint: key -> (terms, rhs), all with one sense."""

    sense: str  # "<=", ">=" or "=="
    rows: Dict[object, Tuple[dict, float]] = field(default_factory=dict)


@dataclass
class Formulation:
    """A linear program in plain Python data, plus what reporting needs."""

    inputs: OptimizationModelInputs  # scaled copy
    scales: dict
    objective: str
    objective_category: Optional[str]

    # Structure
    active_vintage_time: List[tuple]
    active_vintages_by_process_time: Dict[tuple, list]
    aggregated_processes: frozenset
    operation_vintage_time: List[tuple]
    existing_capacity: dict
    operation_phase: Dict[str, Tuple[int, int]]

    # Variables: family -> {key: index}; bounds per index (None = unbounded)
    variable_index: Dict[str, Dict[object, int]]
    lower: List[Optional[float]]
    upper: List[Optional[float]]

    constraints: Dict[str, ConstraintBlock]
    objective_terms: dict  # already divided by objective_scale
    objective_scale: float

    # Reporting expressions as coefficients: name -> {key: terms}
    expressions: Dict[str, Dict[object, dict]]
    # Built on request (large, dense indices): name -> function(*key) -> terms
    lazy_expressions: Dict[str, Callable]

    # Data kept for post-processing
    mapped_inventory_by_flow: dict
    technosphere_vintage_overrides: dict
    biosphere_vintage_overrides: dict
    production_vintage_overrides: dict
    prices: dict
    discount_factor: Dict[int, float]
    cost_processes: List[str]

    @property
    def n_variables(self) -> int:
        return len(self.lower)

    def keys_of(self, family: str) -> list:
        return list(self.variable_index[family])


def build_formulation(
    inputs: OptimizationModelInputs,
    objective_category: Optional[str] = None,
    objective: str = "environmental",
    aggregate_vintages: bool = False,
    scales: Optional[dict] = None,
) -> Formulation:
    """
    Build the backend-independent linear program for `inputs`.

    `scales` reuses the scaling factors of another formulation (its `scales`,
    including the objective normalization) instead of computing them.
    """
    validate_objective(inputs, objective, objective_category)
    fixed_scales = scales
    scaled_inputs, scales = inputs.get_scaled_copy(fixed_scales)

    system_times = list(scaled_inputs.SYSTEM_TIME)
    system_time_set = set(system_times)
    process_times = list(scaled_inputs.PROCESS_TIME)
    operation_flow_data = dict(scaled_inputs.operation_flow)
    operation_phase = {
        p: tuple(scaled_inputs.operation_time_limits.get(p, (0, 0)))
        for p in scaled_inputs.PROCESS
    }
    existing_capacity = dict(scaled_inputs.existing_capacity or {})

    technosphere_overrides = (
        getattr(scaled_inputs, "foreground_technosphere_vintage_overrides", None) or {}
    )
    biosphere_overrides = (
        getattr(scaled_inputs, "foreground_biosphere_vintage_overrides", None) or {}
    )
    production_overrides = (
        getattr(scaled_inputs, "foreground_production_vintage_overrides", None) or {}
    )
    technosphere_overrides_index = frozenset((k[0], k[1]) for k in technosphere_overrides)
    biosphere_overrides_index = frozenset((k[0], k[1]) for k in biosphere_overrides)
    production_overrides_index = frozenset((k[0], k[1]) for k in production_overrides)

    # ------------------------------------------------------------------
    # Active vintages: vintage v of p is active at t if
    # op_start <= t - v <= op_end. Greenfield vintages are SYSTEM_TIME years,
    # brownfield vintages the installation years of existing capacity.
    # ------------------------------------------------------------------
    active_vintage_time = []
    for p in scaled_inputs.PROCESS:
        op_start, op_end = operation_phase[p]
        for v in system_times:
            for t in system_times:
                if v <= t and op_start <= t - v <= op_end:
                    active_vintage_time.append((p, v, t))
        for (proc, inst_year) in existing_capacity:
            if proc == p:
                for t in system_times:
                    if op_start <= t - inst_year <= op_end:
                        active_vintage_time.append((p, inst_year, t))

    active_vintages_by_process_time = {}
    for p, v, t in active_vintage_time:
        active_vintages_by_process_time.setdefault((p, t), []).append(v)

    # ------------------------------------------------------------------
    # Background inventory and characterization, combined numerically
    # ------------------------------------------------------------------
    inventory_by_background = {}
    for (bkg, flow, emission), amount in scaled_inputs.background_inventory.items():
        if amount:
            inventory_by_background.setdefault(bkg, []).append(((flow, emission), amount))

    mapped_inventory = {}
    for (bkg, time), weight in scaled_inputs.mapping.items():
        if not weight:
            continue
        for (flow, emission), amount in inventory_by_background.get(bkg, ()):
            key = (flow, emission, time)
            mapped_inventory[key] = mapped_inventory.get(key, 0.0) + weight * amount

    # (elementary flow, time) -> [(intermediate flow, amount)]
    mapped_inventory_by_emission = {}
    # (intermediate flow, time) -> [(elementary flow, amount)]
    mapped_inventory_by_flow = {}
    for (flow, emission, time), amount in mapped_inventory.items():
        mapped_inventory_by_emission.setdefault((emission, time), []).append(
            (flow, amount)
        )
        mapped_inventory_by_flow.setdefault((flow, time), []).append((emission, amount))
    del mapped_inventory, inventory_by_background

    # Characterized impact per unit of intermediate flow:
    # (category, time) -> {intermediate flow: factor}
    characterization = dict(scaled_inputs.characterization)
    characterization_by_emission = {}
    for (category, emission, time), factor in characterization.items():
        if factor:
            characterization_by_emission.setdefault((emission, time), []).append(
                (category, factor)
            )
    impact_factors = {}
    for (flow, time), emissions in mapped_inventory_by_flow.items():
        for emission, amount in emissions:
            for category, factor in characterization_by_emission.get(
                (emission, time), ()
            ):
                key = (category, time, flow)
                impact_factors[key] = impact_factors.get(key, 0.0) + factor * amount
    impact_factors_by_category_time = {}
    for (category, time, flow), factor in impact_factors.items():
        if factor:
            impact_factors_by_category_time.setdefault((category, time), {})[
                flow
            ] = factor
    del impact_factors, characterization_by_emission

    # ------------------------------------------------------------------
    # Vintage aggregation: processes whose vintages all operate identically
    # get one operation variable per year (see `create_model`).
    # ------------------------------------------------------------------
    operation_tensors = (
        (dict(scaled_inputs.foreground_technosphere), technosphere_overrides_index, False),
        (dict(scaled_inputs.foreground_biosphere), biosphere_overrides_index, False),
        (dict(scaled_inputs.internal_demand_technosphere), frozenset(), False),
        # Production always scales with operation, flagged or not
        (dict(scaled_inputs.foreground_production), production_overrides_index, True),
    )

    def vintages_operate_identically():
        flows_by_process = {}
        for number, (tensor, _, _) in enumerate(operation_tensors):
            for key in tensor:
                flows_by_process.setdefault(key[0], set()).add((key[1], number))
        identical = set()
        for p in scaled_inputs.PROCESS:
            op_start, op_end = operation_phase[p]
            window = range(op_start, op_end + 1)
            ok = True
            for x, number in flows_by_process.get(p, ()):
                tensor, overrides_index, always = operation_tensors[number]
                if not (always or operation_flow_data.get((p, x))):
                    continue
                if (p, x) in overrides_index or (
                    len({tensor.get((p, x, tau), 0) for tau in window}) > 1
                ):
                    ok = False
                    break
            # Overrides on flows that have no base tensor entry
            for _, overrides_index, always in operation_tensors:
                for q, x in overrides_index:
                    if q == p and (always or operation_flow_data.get((p, x))):
                        ok = False
            if ok:
                identical.add(p)
        return identical

    aggregated_processes = frozenset(
        vintages_operate_identically() if aggregate_vintages else ()
    )

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    variable_index = {family: {} for family in VARIABLE_FAMILIES}
    lower, upper = [], []

    def add_variable(family, key, lb=0.0, ub=None):
        variable_index[family][key] = len(lower)
        lower.append(lb)
        upper.append(ub)

    # Deployment limits are bounds of the installation variables. Infinite
    # limits mean no bound; limits below zero are dominated by non-negativity.
    deployment_max = scaled_inputs.process_deployment_limits_max or {}
    deployment_min = scaled_inputs.process_deployment_limits_min or {}
    for p in scaled_inputs.PROCESS:
        for t in system_times:
            ub = deployment_max.get((p, t), scaled_inputs.process_deployment_limits_max_default)
            lb = deployment_min.get((p, t), scaled_inputs.process_deployment_limits_min_default)
            add_variable(
                "var_installation",
                (p, t),
                max(lb, 0.0),
                ub if ub is not None and ub < INFINITY else None,
            )

    operation_vintage_time = [
        key for key in active_vintage_time if key[0] not in aggregated_processes
    ]
    for key in operation_vintage_time:
        add_variable("var_operation", key)
    capacity_index = [
        (p, t) for (p, t) in active_vintages_by_process_time if p in aggregated_processes
    ]
    for key in capacity_index:
        add_variable("var_operation_total", key)
    for key in capacity_index:
        add_variable("var_capacity_total", key)

    installation_vars = variable_index["var_installation"]
    operation_vars = variable_index["var_operation"]
    operation_total_vars = variable_index["var_operation_total"]
    capacity_vars = variable_index["var_capacity_total"]

    constraints: Dict[str, ConstraintBlock] = {}

    def block(name, sense):
        constraints[name] = ConstraintBlock(sense)
        return constraints[name].rows

    # Cumulative deployment limits, only where they can bind
    cumulative_max = scaled_inputs.cumulative_process_limits_max or {}
    cumulative_min = scaled_inputs.cumulative_process_limits_min or {}
    rows = block("CumulativeProcessLimitMax", "<=")
    for p in scaled_inputs.PROCESS:
        limit = cumulative_max.get(p, scaled_inputs.cumulative_process_limits_max_default)
        if limit < INFINITY:
            rows[p] = ({installation_vars[p, t]: 1.0 for t in system_times}, limit)
    rows = block("CumulativeProcessLimitMin", ">=")
    for p in scaled_inputs.PROCESS:
        limit = cumulative_min.get(p, scaled_inputs.cumulative_process_limits_min_default)
        if limit > 0:
            rows[p] = ({installation_vars[p, t]: 1.0 for t in system_times}, limit)

    # Process coupling, only for coupled pairs
    coupling = {
        pair: factor
        for pair, factor in (scaled_inputs.process_coupling or {}).items()
        if factor > 0
    }
    rows = block("ProcessCouplingConstraint", "==")
    for p1, p2 in sorted(coupling):
        for t in system_times:
            terms = {installation_vars[p1, t]: 1.0}
            accumulate(terms, {installation_vars[p2, t]: 1.0}, -coupling[p1, p2])
            rows[p1, p2, t] = (terms, 0.0)

    def total_operation_terms(p, t):
        if p in aggregated_processes:
            return {operation_total_vars[p, t]: 1.0}
        return {
            operation_vars[p, v, t]: 1.0 for v in active_vintages_by_process_time[p, t]
        }

    # Operation limits on the total over vintages, only where they can bind
    operation_max = scaled_inputs.process_operation_limits_max or {}
    operation_min = scaled_inputs.process_operation_limits_min or {}
    rows = block("ProcessOperationLimitMax", "<=")
    for p, t in active_vintages_by_process_time:
        limit = operation_max.get((p, t), scaled_inputs.process_operation_limits_max_default)
        if limit < INFINITY:
            rows[p, t] = (total_operation_terms(p, t), limit)
    rows = block("ProcessOperationLimitMin", ">=")
    for p, t in active_vintages_by_process_time:
        limit = operation_min.get((p, t), scaled_inputs.process_operation_limits_min_default)
        if limit > 0:
            rows[p, t] = (total_operation_terms(p, t), limit)

    # ------------------------------------------------------------------
    # Flows
    # ------------------------------------------------------------------
    def operation_terms(p, t, rate_of_vintage):
        """(variable, rate) pairs of an operation-driven flow of p at t."""
        vintages = active_vintages_by_process_time.get((p, t), ())
        if not vintages:
            return ()
        if p in aggregated_processes:
            rate = rate_of_vintage(vintages[0])
            return ((operation_total_vars[p, t], rate),) if rate else ()
        return [
            (operation_vars[p, v, t], rate)
            for v in vintages
            if (rate := rate_of_vintage(v))
        ]

    def flow_terms(tensor, overrides, overrides_index, installation=True, operation=True):
        """
        Coefficients of flow x of process p at time t. The installation-driven
        part (steps outside the operation phase) scales with the installation
        of vintage t - tau, the operation-driven part with the operation of the
        vintages running at t, each at its annual rate (the per-tau entry, not
        the lifetime sum).
        """
        rates = {}

        def flow_rate(p, x, v, tau):
            key = (p, x, v, tau) if (p, x) in overrides_index else (p, x, tau)
            if key not in rates:
                if (p, x) in overrides_index:
                    rates[key] = overrides.get((p, x, tau, v), tensor.get((p, x, tau), 0))
                else:
                    rates[key] = tensor.get((p, x, tau), 0)
            return rates[key]

        installation_steps = {}

        def installation_pattern(p, x):
            if (p, x) not in installation_steps:
                is_operation_flow = bool(operation_flow_data.get((p, x)))
                op_start, op_end = operation_phase[p]
                steps = [
                    tau
                    for tau in process_times
                    if not (is_operation_flow and op_start <= tau <= op_end)
                ]
                if (p, x) in overrides_index:
                    installation_steps[p, x] = (steps, None)
                else:
                    installation_steps[p, x] = (
                        steps,
                        [(tau, tensor[p, x, tau]) for tau in steps if tensor.get((p, x, tau))],
                    )
            return installation_steps[p, x]

        def terms(p, x, t):
            result = {}
            if installation:
                steps, constant = installation_pattern(p, x)
                if constant is not None:
                    for tau, value in constant:
                        vintage = t - tau
                        if vintage in system_time_set:
                            var = installation_vars[p, vintage]
                            result[var] = result.get(var, 0.0) + value
                else:
                    for tau in steps:
                        vintage = t - tau
                        if vintage not in system_time_set:
                            continue
                        value = overrides.get((p, x, tau, vintage))
                        if value is None:
                            value = tensor.get((p, x, tau), 0)
                        if value:
                            var = installation_vars[p, vintage]
                            result[var] = result.get(var, 0.0) + value
            if operation and operation_flow_data.get((p, x)):
                for var, rate in operation_terms(p, t, lambda v: flow_rate(p, x, v, t - v)):
                    result[var] = result.get(var, 0.0) + rate
            return result

        return terms

    def scaled_flow_terms(tensor, overrides, overrides_index):
        pairs = {(p, x) for (p, x, _) in tensor} | set(overrides_index)
        rule = flow_terms(tensor, overrides, overrides_index)
        return pairs, {
            (p, x, t): rule(p, x, t) for (p, x) in sorted(pairs) for t in system_times
        }

    technosphere_tensor = dict(scaled_inputs.foreground_technosphere)
    technosphere_pairs, technosphere_terms = scaled_flow_terms(
        technosphere_tensor, technosphere_overrides, technosphere_overrides_index
    )
    biosphere_pairs, biosphere_terms = scaled_flow_terms(
        dict(scaled_inputs.foreground_biosphere),
        biosphere_overrides,
        biosphere_overrides_index,
    )
    internal_demand_pairs, internal_demand_terms = scaled_flow_terms(
        dict(scaled_inputs.internal_demand_technosphere), {}, frozenset()
    )

    technosphere_flows_by_process = {}
    consumers_by_intermediate_flow = {}
    for p, i in sorted(technosphere_pairs):
        technosphere_flows_by_process.setdefault(p, []).append(i)
        consumers_by_intermediate_flow.setdefault(i, []).append(p)
    biosphere_flows_by_process = {}
    emitters_by_elementary_flow = {}
    for p, e in sorted(biosphere_pairs):
        biosphere_flows_by_process.setdefault(p, []).append(e)
        emitters_by_elementary_flow.setdefault(e, set()).add(p)

    def inventory_terms(p, e, t):
        """Inventory of e caused by p at t: own emissions plus background."""
        result = {}
        process_flows = set(technosphere_flows_by_process.get(p, ()))
        for i, amount in mapped_inventory_by_emission.get((e, t), ()):
            if i in process_flows:
                accumulate(result, technosphere_terms[p, i, t], amount)
        if (p, e, t) in biosphere_terms:
            accumulate(result, biosphere_terms[p, e, t])
        return result

    def total_intermediate_flow_terms(i, t):
        result = {}
        for p in consumers_by_intermediate_flow.get(i, ()):
            accumulate(result, technosphere_terms[p, i, t])
        return result

    def total_elementary_flow_terms(e, t):
        processes = set(emitters_by_elementary_flow.get(e, ()))
        for i, _ in mapped_inventory_by_emission.get((e, t), ()):
            processes.update(consumers_by_intermediate_flow.get(i, ()))
        result = {}
        for p in sorted(processes):
            accumulate(result, inventory_terms(p, e, t))
        return result

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------
    rows = block("OperationCapacity", "<=")
    for p, v, t in operation_vintage_time:
        if v in system_time_set:
            rows[p, v, t] = (
                {operation_vars[p, v, t]: 1.0, installation_vars[p, v]: -1.0},
                0.0,
            )
        elif existing_capacity.get((p, v), 0):
            rows[p, v, t] = ({operation_vars[p, v, t]: 1.0}, existing_capacity[p, v])

    # Aggregated processes: a capacity stock carried forward year by year,
    # capacity[t] = capacity[t_prev] + vintages entering - vintages leaving.
    def vintage_capacity_terms(p, vintages):
        terms, existing = {}, 0.0
        for v in vintages:
            if v in system_time_set:
                terms[installation_vars[p, v]] = 1.0
            else:
                existing += existing_capacity.get((p, v), 0)
        return terms, existing

    previous_time, last = {}, {}
    for p, t in sorted(capacity_index):
        if p in last:
            previous_time[p, t] = last[p]
        last[p] = t

    balance_rows = block("CapacityBalanceTotal", "==")
    capacity_rows = block("OperationCapacityTotal", "<=")
    for p, t in capacity_index:
        active = set(active_vintages_by_process_time[p, t])
        terms = {capacity_vars[p, t]: 1.0}
        prev = previous_time.get((p, t))
        if prev is None:
            entering, leaving = active, set()
        else:
            before = set(active_vintages_by_process_time[p, prev])
            entering, leaving = active - before, before - active
            terms[capacity_vars[p, prev]] = -1.0
        added, existing_added = vintage_capacity_terms(p, entering)
        removed, existing_removed = vintage_capacity_terms(p, leaving)
        accumulate(terms, added, -1.0)
        accumulate(terms, removed, 1.0)
        balance_rows[p, t] = (terms, existing_added - existing_removed)
        capacity_rows[p, t] = (
            {operation_total_vars[p, t]: 1.0, capacity_vars[p, t]: -1.0},
            0.0,
        )

    # ------------------------------------------------------------------
    # Products
    # ------------------------------------------------------------------
    production_data = dict(scaled_inputs.foreground_production)
    production_rates = {}

    def production_rate(p, r, v, tau):
        """Annual output of r by one running vintage-v unit at stage tau."""
        has_overrides = (p, r) in production_overrides_index
        key = (p, r, v, tau) if has_overrides else (p, r, tau)
        if key not in production_rates:
            if has_overrides:
                production_rates[key] = production_overrides.get(
                    (p, r, tau, v), production_data.get((p, r, tau), 0)
                )
            else:
                production_rates[key] = production_data.get((p, r, tau), 0)
        return production_rates[key]

    producers_by_product = {}
    for p, r in sorted({(k[0], k[1]) for k in production_data} | set(production_overrides_index)):
        producers_by_product.setdefault(r, []).append(p)
    consumers_by_product = {}
    for p, r in sorted(internal_demand_pairs):
        consumers_by_product.setdefault(r, []).append(p)

    # Production of r by p at t, per producer
    production_flow = {}
    for r, producers in producers_by_product.items():
        for p in producers:
            for t in system_times:
                production_flow[p, r, t] = dict(
                    operation_terms(p, t, lambda v: production_rate(p, r, v, t - v))
                )

    def production_terms(r, t):
        result = {}
        for p in producers_by_product.get(r, ()):
            accumulate(result, production_flow[p, r, t])
        return result

    total_product_flow = {
        (r, t): production_terms(r, t) for r in scaled_inputs.PRODUCT for t in system_times
    }

    demand = scaled_inputs.demand or {}
    rows = block("ProductDemandFulfillment", "==")
    for r in scaled_inputs.PRODUCT:
        for t in system_times:
            terms = dict(total_product_flow[r, t])
            for p in consumers_by_product.get(r, ()):
                accumulate(terms, internal_demand_terms[p, r, t], -1.0)
            rows[r, t] = (terms, demand.get((r, t), 0))

    # ------------------------------------------------------------------
    # Impacts
    # ------------------------------------------------------------------
    specific_impact, time_specific_impact, total_impact = {}, {}, {}
    for c in scaled_inputs.CATEGORY:
        category_terms = total_impact.setdefault(c, {})
        for t in system_times:
            factors = impact_factors_by_category_time.get((c, t), {})
            time_terms = time_specific_impact.setdefault((c, t), {})
            for p in scaled_inputs.PROCESS:
                terms = {}
                for i in technosphere_flows_by_process.get(p, ()):
                    factor = factors.get(i)
                    if factor:
                        accumulate(terms, technosphere_terms[p, i, t], factor)
                for e in biosphere_flows_by_process.get(p, ()):
                    factor = characterization.get((c, e, t))
                    if factor:
                        accumulate(terms, biosphere_terms[p, e, t], factor)
                specific_impact[c, p, t] = terms
                accumulate(time_terms, terms)
            accumulate(category_terms, time_terms)

    rows = block("CategoryImpactLimits", "<=")
    for (c, t), limit in (scaled_inputs.category_impact_limits or {}).items():
        if (c, t) in time_specific_impact:
            rows[c, t] = (time_specific_impact[c, t], limit)
    rows = block("CumulativeCategoryImpactLimits", "<=")
    for c, limit in (scaled_inputs.cumulative_category_impact_limits or {}).items():
        if c in total_impact and limit < INFINITY:
            rows[c] = (total_impact[c], limit)

    # ------------------------------------------------------------------
    # Flow limits
    # ------------------------------------------------------------------
    flow_limits_max = scaled_inputs.flow_limits_max or {}
    flow_limits_min = scaled_inputs.flow_limits_min or {}
    cumulative_flow_limits_max = scaled_inputs.cumulative_flow_limits_max or {}
    cumulative_flow_limits_min = scaled_inputs.cumulative_flow_limits_min or {}
    _warn_about_unusable_flow_limits(
        scaled_inputs,
        flow_limits_max,
        flow_limits_min,
        cumulative_flow_limits_max,
        cumulative_flow_limits_min,
    )
    products = set(scaled_inputs.PRODUCT)
    intermediate_flows = set(scaled_inputs.INTERMEDIATE_FLOW)
    elementary_flows = set(scaled_inputs.ELEMENTARY_FLOW)

    def total_flow_terms(f, t):
        if f in products:
            return total_product_flow[f, t]
        if f in intermediate_flows:
            return total_intermediate_flow_terms(f, t)
        if f in elementary_flows:
            return total_elementary_flow_terms(f, t)
        return None

    # Flow limits are in real units; the flows are in scaled units
    fg_scale = scales["foreground"]
    for name, limits, sense in (
        ("FlowLimitMax", flow_limits_max, "<="),
        ("FlowLimitMin", flow_limits_min, ">="),
    ):
        rows = block(name, sense)
        for (f, t), limit in limits.items():
            if t not in system_time_set:
                continue
            terms = total_flow_terms(f, t)
            if terms is not None:
                rows[f, t] = (terms, limit / fg_scale)
    for name, limits, sense in (
        ("CumulativeFlowLimitMax", cumulative_flow_limits_max, "<="),
        ("CumulativeFlowLimitMin", cumulative_flow_limits_min, ">="),
    ):
        rows = block(name, sense)
        for f, limit in limits.items():
            if total_flow_terms(f, system_times[0]) is None:
                continue
            terms = {}
            for t in system_times:
                accumulate(terms, total_flow_terms(f, t))
            rows[f] = (terms, limit / fg_scale)

    # ------------------------------------------------------------------
    # Costs of the background purchases (built for every model, so the cost
    # of an environmentally optimal pathway can be reported as well)
    # ------------------------------------------------------------------
    prices = scaled_inputs.intermediate_flow_prices or {}
    rate = scaled_inputs.discount_rate
    reference_year = scaled_inputs.discount_reference_year
    if reference_year is None:
        reference_year = min(system_times)
    discount_factor = {t: (1 + rate) ** -(t - reference_year) for t in system_times}

    priced_flows = {i for (i, _), price in prices.items() if price}
    priced_flows_by_process = {}
    for p, i in sorted(technosphere_pairs):
        if i in priced_flows:
            priced_flows_by_process.setdefault(p, []).append(i)

    technosphere_args = (technosphere_tensor, technosphere_overrides, technosphere_overrides_index)
    installation_flow = flow_terms(*technosphere_args, operation=False)
    operation_flow = flow_terms(*technosphere_args, installation=False)

    def cost_terms(flow):
        costs = {}
        for p in priced_flows_by_process:
            for t in system_times:
                terms = {}
                for i in priced_flows_by_process[p]:
                    price = prices.get((i, t))
                    if price:
                        accumulate(terms, flow(p, i, t), price)
                costs[p, t] = terms
        return costs

    installation_cost = cost_terms(installation_flow)
    operation_cost = cost_terms(operation_flow)
    total_cost = {}
    for costs in (installation_cost, operation_cost):
        for (p, t), terms in costs.items():
            accumulate(total_cost, terms, discount_factor[t])

    # ------------------------------------------------------------------
    # Objective, normalized by its largest coefficient: impact coefficients
    # easily end up around 1e-8, below the solvers' optimality tolerances.
    # ------------------------------------------------------------------
    objective_raw = total_cost if objective == "cost" else total_impact[objective_category]
    objective_scale = max((abs(c) for c in objective_raw.values()), default=0.0)
    if not objective_scale or objective_scale == INFINITY:
        objective_scale = 1.0
    if fixed_scales is not None and "objective" in fixed_scales:
        objective_scale = fixed_scales["objective"]
    scales["objective"] = objective_scale
    objective_terms = {k: c / objective_scale for k, c in objective_raw.items()}

    # Constant rows (nothing can enter them) that hold are dropped. Those that
    # cannot hold are kept: the model is infeasible, which the solve reports.
    for constraint in constraints.values():
        for key in [k for k, (terms, _) in constraint.rows.items() if not terms]:
            rhs = constraint.rows[key][1]
            if {"<=": 0 <= rhs, ">=": 0 >= rhs, "==": rhs == 0}[constraint.sense]:
                del constraint.rows[key]

    return Formulation(
        inputs=scaled_inputs,
        scales=scales,
        objective=objective,
        objective_category=objective_category if objective == "environmental" else None,
        active_vintage_time=active_vintage_time,
        active_vintages_by_process_time=active_vintages_by_process_time,
        aggregated_processes=aggregated_processes,
        operation_vintage_time=operation_vintage_time,
        existing_capacity=existing_capacity,
        operation_phase=operation_phase,
        variable_index=variable_index,
        lower=lower,
        upper=upper,
        constraints=constraints,
        objective_terms=objective_terms,
        objective_scale=objective_scale,
        expressions={
            "scaled_technosphere_flow": technosphere_terms,
            "scaled_biosphere_flow": biosphere_terms,
            "scaled_internal_demand_flow": internal_demand_terms,
            "scaled_production_flow": production_flow,
            "specific_impact": specific_impact,
            "total_impact": total_impact,
            "time_specific_impact": time_specific_impact,
            "total_product_flow": total_product_flow,
            "installation_cost": installation_cost,
            "operation_cost": operation_cost,
            "total_cost": {None: total_cost},
        },
        lazy_expressions={
            "scaled_inventory": inventory_terms,
            "total_intermediate_flow": total_intermediate_flow_terms,
            "total_elementary_flow": total_elementary_flow_terms,
        },
        mapped_inventory_by_flow=mapped_inventory_by_flow,
        technosphere_vintage_overrides=technosphere_overrides,
        biosphere_vintage_overrides=biosphere_overrides,
        production_vintage_overrides=production_overrides,
        prices=prices,
        discount_factor=discount_factor,
        cost_processes=sorted(priced_flows_by_process),
    )


def validate_objective(
    inputs: OptimizationModelInputs, objective: str, objective_category: Optional[str]
) -> None:
    """Check that the requested objective can be built from `inputs`."""
    if objective == "environmental":
        if objective_category not in inputs.CATEGORY:
            raise ValueError(
                f"objective='environmental' needs an objective_category from "
                f"{sorted(inputs.CATEGORY)}, got {objective_category!r}."
            )
    elif objective == "cost":
        if not any((inputs.intermediate_flow_prices or {}).values()):
            raise ValueError(
                "objective='cost' needs non-zero intermediate_flow_prices. Set prices "
                "on the background nodes (optimex.economics.set_market_prices) and "
                "LCAConfig.price_attribute, or pass them with "
                "ModelInputManager.override(intermediate_flow_prices=...)."
            )
    else:
        raise ValueError(
            f"Unknown objective {objective!r}. Expected 'environmental' or 'cost'."
        )


def _warn_about_unusable_flow_limits(
    scaled_inputs, flow_limits_max, flow_limits_min, cumulative_max, cumulative_min
) -> None:
    """
    Report flow limits that cannot do anything.

    `OptimizationModelInputs` rejects unknown flows when it is constructed, but
    assigning a limit to an already-built instance bypasses that.
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
    limited_flows = set(cumulative_max) | set(cumulative_min)
    limited_flows |= {key[0] for key in flow_limits_max}
    limited_flows |= {key[0] for key in flow_limits_min}

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
    limited_times = {key[1] for key in flow_limits_max} | {
        key[1] for key in flow_limits_min
    }
    unknown_times = sorted(limited_times - set(scaled_inputs.SYSTEM_TIME))
    if unknown_times:
        logger.warning(
            f"Ignoring flow limits for years {unknown_times}: outside the "
            "system time horizon."
        )
