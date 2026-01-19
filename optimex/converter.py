"""
Model input conversion and validation for optimization.

This module bridges LCA data processing and optimization by converting outputs from
LCADataProcessor into structured OptimizationModelInputs. It provides validation,
scaling, serialization, and constraint management for optimization model inputs.

Key classes:
    - OptimizationModelInputs: Validated data structure for optimization inputs
    - ModelInputManager: Handles conversion, serialization, and constraint overrides
"""
import copy
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from optimex.lca_processor import LCADataProcessor


class OptimizationModelInputs(BaseModel):
    """
    Interface data structure for linking LCA-based outputs with optimization inputs.

    This class organizes all relevant inputs needed to build a temporal, process-based
    life cycle model suitable for linear optimization, including foreground and
    background exchanges, temporal system information, and optional process constraints.
    """

    PROCESS: List[str] = Field(
        ..., description="Identifiers for all modeled processes."
    )
    PRODUCT: List[str] = Field(
        ..., description="Identifiers for all foreground products."
    )
    INTERMEDIATE_FLOW: List[str] = Field(
        ..., description="Identifiers for background products (from background databases)."
    )
    ELEMENTARY_FLOW: List[str] = Field(
        ...,
        description=(
            "Identifiers for flows representing exchanges with the environment."
        ),
    )
    BACKGROUND_ID: List[str] = Field(
        ..., description="Identifiers for background system databases."
    )
    PROCESS_TIME: List[int] = Field(
        ...,
        description=(
            "Relative time steps representing the operation timeline of each process."
        ),
    )
    SYSTEM_TIME: List[int] = Field(
        ..., description="Absolute time steps representing actual years in the system."
    )
    CATEGORY: List[str] = Field(
        ..., description="Impact categories for each elementary flow."
    )

    demand: Dict[Tuple[str, int], float] = Field(
        ..., description=("Maps (product, system_time) to external demand amount.")
    )
    operation_flow: Dict[Tuple[str, str], bool] = Field(
        ...,
        description=(
            "Maps (process, flow) to boolean indicating if the flow is occuring during "
            "the operation phase of the process."
        ),
    )

    foreground_technosphere: Dict[Tuple[str, str, int], float] = Field(
        ...,
        description=("Maps (process, intermediate_flow, process_time) to background flow amount."),
    )
    internal_demand_technosphere: Dict[Tuple[str, str, int], float] = Field(
        ...,
        description=("Maps (process, product, process_time) to internal product consumption amount."),
    )
    foreground_biosphere: Dict[Tuple[str, str, int], float] = Field(
        ...,
        description=(
            "Maps (process, elementary_flow, process_time) to emission or resource "
            "amount."
        ),
    )
    foreground_production: Dict[Tuple[str, str, int], float] = Field(
        ...,
        description=(
            "Maps (process, product, process_time) to produced amount."
        ),
    )

    background_inventory: Dict[Tuple[str, str, str], float] = Field(
        ...,
        description=(
            "Maps (background_id, intermediate_flow, environmental_flow) to "
            "exchange amount."
        ),
    )
    mapping: Dict[Tuple[str, int], float] = Field(
        ...,
        description=(
            "Maps (background_id, system_time) to scaling or availability factor."
        ),
    )
    characterization: Dict[Tuple[str, str, int], float] = Field(
        ...,
        description=(
            "Maps (impact_category, elementary_flow, system_time) to impact "
            "characterization factor."
        ),
    )

    operation_time_limits: Dict[str, Tuple[int, int]] = Field(
        None,
        description=(
            "Maps process identifiers to tuples of (min_time, max_time) for operation "
            "time limits."
        ),
    )

    category_impact_limits: Optional[Dict[Tuple[str, int], float]] = Field(
        None,
        description=(
            "Time-specific upper bounds on impact categories. Maps (category, system_time) "
            "to maximum allowed impact at that time."
        ),
    )
    cumulative_category_impact_limits: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Cumulative upper bounds on total impact per category across all time periods."
        ),
    )
    process_deployment_limits_max: Optional[Dict[Tuple[str, int], float]] = Field(
        None, description="Upper bounds on (process, system_time) deployment."
    )
    process_deployment_limits_min: Optional[Dict[Tuple[str, int], float]] = Field(
        None, description="Lower bounds on (process, system_time) deployment."
    )
    process_operation_limits_max: Optional[Dict[Tuple[str, int], float]] = Field(
        None, description="Upper bounds on (process, system_time) operation."
    )
    process_operation_limits_min: Optional[Dict[Tuple[str, int], float]] = Field(
        None, description="Lower bounds on (process, system_time) operation."
    )
    cumulative_process_limits_max: Optional[Dict[str, float]] = Field(
        None, description=("Global upper bound on cumulative deployment for a process.")
    )
    cumulative_process_limits_min: Optional[Dict[str, float]] = Field(
        None, description=("Global lower bound on cumulative deployment for a process.")
    )

    process_coupling: Optional[Dict[Tuple[str, str], float]] = Field(
        None,
        description=(
            "Coupling of process deployment, constraining deployment of one "
            "process as multiplier of another."
        ),
    )
    existing_capacity: Optional[Dict[Tuple[str, int], float]] = Field(
        None,
        description=(
            "Existing (brownfield) capacity installed before the optimization horizon. "
            "Maps (process, installation_year) to capacity amount. Installation years "
            "must be before min(SYSTEM_TIME). These capacities contribute to operation "
            "and production but their installation impacts are excluded (sunk costs)."
        ),
    )

    flow_limits_max: Optional[Dict[Tuple[str, int], float]] = Field(
        None,
        description=(
            "Time-specific upper bounds on flows. Maps (flow, system_time) to maximum "
            "allowed flow amount at that time. Flows can be from PRODUCT, INTERMEDIATE_FLOW, "
            "or ELEMENTARY_FLOW sets."
        ),
    )
    flow_limits_min: Optional[Dict[Tuple[str, int], float]] = Field(
        None,
        description=(
            "Time-specific lower bounds on flows. Maps (flow, system_time) to minimum "
            "required flow amount at that time. Flows can be from PRODUCT, INTERMEDIATE_FLOW, "
            "or ELEMENTARY_FLOW sets."
        ),
    )
    cumulative_flow_limits_max: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Cumulative upper bounds on flows across all system times. Maps flow identifier "
            "to maximum total flow amount. Flows can be from PRODUCT, INTERMEDIATE_FLOW, "
            "or ELEMENTARY_FLOW sets."
        ),
    )
    cumulative_flow_limits_min: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Cumulative lower bounds on flows across all system times. Maps flow identifier "
            "to minimum total flow amount. Flows can be from PRODUCT, INTERMEDIATE_FLOW, "
            "or ELEMENTARY_FLOW sets."
        ),
    )

    process_names: Optional[Dict[str, str]] = Field(
        None, description="Maps process identifiers to human-readable names."
    )

    process_deployment_limits_max_default: float = Field(
        default=float("inf"),
        description=(
            "Default upper bound for annual process deployment if not explicitly "
            "specified."
        ),
    )
    process_deployment_limits_min_default: float = Field(
        default=0.0,
        description=(
            "Default lower bound for annual process deployment if not explicitly "
            "specified."
        ),
    )
    process_operation_limits_max_default: float = Field(
        default=float("inf"),
        description=(
            "Default upper bound for process operation if not explicitly "
            "specified."
        ),
    )
    process_operation_limits_min_default: float = Field(
        default=0.0,
        description=(
            "Default lower bound for process operation if not explicitly "
            "specified."
        ),
    )
    cumulative_process_limits_max_default: float = Field(
        default=float("inf"),
        description=(
            "Default global upper bound for total process deployment if not explicitly "
            "specified."
        ),
    )
    cumulative_process_limits_min_default: float = Field(
        default=0.0,
        description=(
            "Default global lower bound for total process deployment if not explicitly "
            "specified."
        ),
    )

    @model_validator(mode="before")
    def check_all_keys(cls, data):
        """
        Validate that all dictionary keys reference valid set elements.

        This validator ensures that all keys in the input dictionaries (e.g., demand,
        foreground_technosphere) reference elements that exist in the corresponding
        sets (e.g., PROCESS, PRODUCT, SYSTEM_TIME). This prevents runtime errors
        from invalid references.

        Parameters
        ----------
        data : dict
            The raw data dictionary before model instantiation.

        Returns
        -------
        dict
            The validated data dictionary.

        Raises
        ------
        ValueError
            If any dictionary key references an element not in the corresponding set.
        """
        # Convert lists to sets for fast lookup
        processes = set(data.get("PROCESS", []))
        products = set(data.get("PRODUCT", []))
        intermediate_flows = set(data.get("INTERMEDIATE_FLOW", []))
        elementary_flows = set(data.get("ELEMENTARY_FLOW", []))
        background_ids = set(data.get("BACKGROUND_ID", []))
        process_times = set(data.get("PROCESS_TIME", []))
        system_times = set(data.get("SYSTEM_TIME", []))
        categories = set(data.get("CATEGORY", []))

        def validate_keys(keys, valid_set, context):
            invalid = [k for k in keys if k not in valid_set]
            if invalid:
                raise ValueError(
                    f"Invalid keys {invalid} in {context}. "
                    f"Valid keys: {sorted(valid_set)}"
                )

        # Now validate keys in all dict fields similarly

        for key in data.get("demand", {}).keys():
            validate_keys([key[0]], products, "demand products")
            validate_keys([key[1]], system_times, "demand system times")

        for key in data.get("foreground_technosphere", {}).keys():
            validate_keys([key[0]], processes, "foreground_technosphere processes")
            validate_keys(
                [key[1]],
                intermediate_flows,
                "foreground_technosphere intermediate flows",
            )
            validate_keys(
                [key[2]], process_times, "foreground_technosphere process times"
            )

        for key in data.get("internal_demand_technosphere", {}).keys():
            validate_keys([key[0]], processes, "internal_demand_technosphere processes")
            validate_keys(
                [key[1]], products, "internal_demand_technosphere products"
            )
            validate_keys(
                [key[2]], process_times, "internal_demand_technosphere process times"
            )

        for key in data.get("foreground_biosphere", {}).keys():
            validate_keys([key[0]], processes, "foreground_biosphere processes")
            validate_keys(
                [key[1]], elementary_flows, "foreground_biosphere elementary flows"
            )
            validate_keys([key[2]], process_times, "foreground_biosphere process times")

        for key in data.get("foreground_production", {}).keys():
            validate_keys([key[0]], processes, "foreground_production processes")
            validate_keys(
                [key[1]], products, "foreground_production products"
            )
            validate_keys(
                [key[2]], process_times, "foreground_production process times"
            )

        for key in data.get("background_inventory", {}).keys():
            validate_keys(
                [key[0]], background_ids, "background_inventory background IDs"
            )
            validate_keys(
                [key[1]], intermediate_flows, "background_inventory intermediate flows"
            )
            validate_keys(
                [key[2]], elementary_flows, "background_inventory environmental flows"
            )

        for key in data.get("mapping", {}).keys():
            validate_keys([key[0]], background_ids, "mapping background IDs")
            validate_keys([key[1]], system_times, "mapping system times")

        for key in data.get("characterization", {}).keys():
            validate_keys([key[0]], categories, "characterization categories")
            validate_keys(
                [key[1]], elementary_flows, "characterization elementary flows"
            )
            validate_keys([key[2]], system_times, "characterization system times")

        if data.get("category_impact_limits") is not None:
            for key in data["category_impact_limits"].keys():
                validate_keys([key[0]], categories, "category_impact_limits categories")
                validate_keys([key[1]], system_times, "category_impact_limits system times")

        if data.get("cumulative_category_impact_limits") is not None:
            for key in data["cumulative_category_impact_limits"].keys():
                validate_keys([key], categories, "cumulative_category_impact_limits")

        if data.get("process_deployment_limits_max") is not None:
            for key in data["process_deployment_limits_max"].keys():
                validate_keys([key[0]], processes, "process_deployment_limits_max processes")
                validate_keys([key[1]], system_times, "process_deployment_limits_max system times")

        if data.get("process_deployment_limits_min") is not None:
            for key in data["process_deployment_limits_min"].keys():
                validate_keys([key[0]], processes, "process_deployment_limits_min processes")
                validate_keys([key[1]], system_times, "process_deployment_limits_min system times")

        if data.get("process_operation_limits_max") is not None:
            for key in data["process_operation_limits_max"].keys():
                validate_keys([key[0]], processes, "process_operation_limits_max processes")
                validate_keys([key[1]], system_times, "process_operation_limits_max system times")

        if data.get("process_operation_limits_min") is not None:
            for key in data["process_operation_limits_min"].keys():
                validate_keys([key[0]], processes, "process_operation_limits_min processes")
                validate_keys([key[1]], system_times, "process_operation_limits_min system times")

        if data.get("cumulative_process_limits_max") is not None:
            for key in data["cumulative_process_limits_max"].keys():
                validate_keys(
                    [key], processes, "cumulative_process_limits_max processes"
                )

        if data.get("cumulative_process_limits_min") is not None:
            for key in data["cumulative_process_limits_min"].keys():
                validate_keys(
                    [key], processes, "cumulative_process_limits_min processes"
                )

        if data.get("process_coupling") is not None:
            for (p1, p2), val in data["process_coupling"].items():
                validate_keys([p1, p2], processes, "process_coupling processes")
                if val <= 0:
                    raise ValueError(
                        f"Coupling value for ({p1}, {p2}) must be positive, got {val}"
                    )

        if data.get("existing_capacity") is not None:
            min_system_time = min(system_times) if system_times else None
            for (proc, inst_year), capacity in data["existing_capacity"].items():
                validate_keys([proc], processes, "existing_capacity processes")
                if min_system_time is not None and inst_year >= min_system_time:
                    raise ValueError(
                        f"Existing capacity installation year ({inst_year}) for process "
                        f"'{proc}' must be before min(SYSTEM_TIME) ({min_system_time}). "
                        "Brownfield capacity represents pre-existing installations."
                    )
                if capacity < 0:
                    raise ValueError(
                        f"Existing capacity for ({proc}, {inst_year}) must be "
                        f"non-negative, got {capacity}"
                    )

        # Flow limits validation - flows can be from PRODUCT, INTERMEDIATE_FLOW, or ELEMENTARY_FLOW
        all_flows = products | intermediate_flows | elementary_flows

        if data.get("flow_limits_max") is not None:
            for key in data["flow_limits_max"].keys():
                validate_keys([key[0]], all_flows, "flow_limits_max flows")
                validate_keys([key[1]], system_times, "flow_limits_max system times")

        if data.get("flow_limits_min") is not None:
            for key in data["flow_limits_min"].keys():
                validate_keys([key[0]], all_flows, "flow_limits_min flows")
                validate_keys([key[1]], system_times, "flow_limits_min system times")

        if data.get("cumulative_flow_limits_max") is not None:
            for key in data["cumulative_flow_limits_max"].keys():
                validate_keys([key], all_flows, "cumulative_flow_limits_max flows")

        if data.get("cumulative_flow_limits_min") is not None:
            for key in data["cumulative_flow_limits_min"].keys():
                validate_keys([key], all_flows, "cumulative_flow_limits_min flows")

        return data

    # For flexible operation: all non-zero flow values must remain constant over time
    # to ensure that their ratio to the reference flow is well-defined and consistent
    # for scaling.
    @model_validator(mode="after")
    def validate_constant_operation_flows(self) -> "OptimizationModelInputs":
        """
        Validate that flows marked as operational are constant over process time.

        For flexible operation mode, flows that occur during the operation phase must
        have constant values across process time steps. This is because the optimization
        scales these flows linearly with the operation variable. Time-varying operational
        flows would require fixed operation mode instead.

        Returns
        -------
        OptimizationModelInputs
            Self, after validation.

        Raises
        ------
        ValueError
            If any operational flow has varying values across process time.
        """
        def check_constancy(
            flow_data: Dict[Tuple[str, str, int], float], flow_type: str
        ):
            grouped: Dict[Tuple[str, str], List[float]] = {}

            for (proc, flow, t), val in flow_data.items():
                grouped.setdefault((proc, flow), []).append((t, val))

            for (proc, flow), tv_pairs in grouped.items():
                if not self.operation_flow.get((proc, flow), False):
                    continue  # Skip non-operational flows

                # Sort by time, filter out zeros
                values = [v for _, v in sorted(tv_pairs) if v != 0]

                if len(set(values)) > 1:
                    raise ValueError(
                        f"{flow_type} ({proc}, {flow}) is not constant over time: "
                        f"values = {values}. If you want to conduct an optimization "
                        "with these values, don't flag any as operational and use "
                        "fixed operation in the optimization later."
                    )

        check_constancy(self.foreground_technosphere, "intermediate flow")
        check_constancy(self.foreground_biosphere, "elementary flow")

        return self

    @model_validator(mode="after")
    def validate_process_limits_consistency(self) -> "OptimizationModelInputs":
        """
        Validate that min limits are not greater than max limits for process bounds.

        This ensures logical consistency of the bounds - having min > max would
        create an infeasible constraint.
        """
        # Check per-period deployment limits
        if self.process_deployment_limits_min and self.process_deployment_limits_max:
            for key in self.process_deployment_limits_min:
                if key in self.process_deployment_limits_max:
                    min_val = self.process_deployment_limits_min[key]
                    max_val = self.process_deployment_limits_max[key]
                    if min_val > max_val:
                        raise ValueError(
                            f"Process deployment limit min ({min_val}) > max ({max_val}) for {key}. "
                            "Constraints would be infeasible."
                        )

        # Check per-period operation limits
        if self.process_operation_limits_min and self.process_operation_limits_max:
            for key in self.process_operation_limits_min:
                if key in self.process_operation_limits_max:
                    min_val = self.process_operation_limits_min[key]
                    max_val = self.process_operation_limits_max[key]
                    if min_val > max_val:
                        raise ValueError(
                            f"Process operation limit min ({min_val}) > max ({max_val}) for {key}. "
                            "Constraints would be infeasible."
                        )

        # Check cumulative process limits
        if self.cumulative_process_limits_min and self.cumulative_process_limits_max:
            for proc in self.cumulative_process_limits_min:
                if proc in self.cumulative_process_limits_max:
                    min_val = self.cumulative_process_limits_min[proc]
                    max_val = self.cumulative_process_limits_max[proc]
                    if min_val > max_val:
                        raise ValueError(
                            f"Cumulative process limit min ({min_val}) > max ({max_val}) "
                            f"for {proc}. Constraints would be infeasible."
                        )
                    
        # Cross-check cumulative vs. per-period limits
        if self.process_deployment_limits_max and self.cumulative_process_limits_min:
            for key in self.cumulative_process_limits_min:
                total_max = sum(
                    self.process_deployment_limits_max.get((key, t), 0.0)
                    for t in self.SYSTEM_TIME
                )
                min_cum = self.cumulative_process_limits_min[key]
                if min_cum > total_max:
                    raise ValueError(
                        f"Cumulative process limit min ({min_cum}) > sum of per-period "
                        f"max ({total_max}) for {key}. Constraints would be infeasible."
                    )
        if self.process_deployment_limits_min and self.cumulative_process_limits_max:
            for key in self.cumulative_process_limits_max:
                total_min = sum(
                    self.process_deployment_limits_min.get((key, t), 0.0)
                    for t in self.SYSTEM_TIME
                )
                max_cum = self.cumulative_process_limits_max[key]
                if total_min > max_cum:
                    raise ValueError(
                        f"Sum of per-period process limit min ({total_min}) > cumulative "
                        f"process limit max ({max_cum}) for {key}. Constraints would be infeasible."
                    )


        # Check defaults consistency
        if self.process_deployment_limits_min_default > self.process_deployment_limits_max_default:
            raise ValueError(
                f"process_deployment_limits_min_default ({self.process_deployment_limits_min_default}) > "
                f"process_deployment_limits_max_default ({self.process_deployment_limits_max_default}). "
                "Constraints would be infeasible."
            )

        if self.process_operation_limits_min_default > self.process_operation_limits_max_default:
            raise ValueError(
                f"process_operation_limits_min_default ({self.process_operation_limits_min_default}) > "
                f"process_operation_limits_max_default ({self.process_operation_limits_max_default}). "
                "Constraints would be infeasible."
            )

        if self.cumulative_process_limits_min_default > self.cumulative_process_limits_max_default:
            raise ValueError(
                f"cumulative_process_limits_min_default ({self.cumulative_process_limits_min_default}) > "
                f"cumulative_process_limits_max_default ({self.cumulative_process_limits_max_default}). "
                "Constraints would be infeasible."
            )

        # Check per-period flow limits
        if self.flow_limits_min and self.flow_limits_max:
            for key in self.flow_limits_min:
                if key in self.flow_limits_max:
                    min_val = self.flow_limits_min[key]
                    max_val = self.flow_limits_max[key]
                    if min_val > max_val:
                        raise ValueError(
                            f"Flow limit min ({min_val}) > max ({max_val}) for {key}. "
                            "Constraints would be infeasible."
                        )

        # Check cumulative flow limits
        if self.cumulative_flow_limits_min and self.cumulative_flow_limits_max:
            for flow in self.cumulative_flow_limits_min:
                if flow in self.cumulative_flow_limits_max:
                    min_val = self.cumulative_flow_limits_min[flow]
                    max_val = self.cumulative_flow_limits_max[flow]
                    if min_val > max_val:
                        raise ValueError(
                            f"Cumulative flow limit min ({min_val}) > max ({max_val}) "
                            f"for {flow}. Constraints would be infeasible."
                        )

        # Cross-check cumulative vs. per-period flow limits
        if self.flow_limits_max and self.cumulative_flow_limits_min:
            for flow in self.cumulative_flow_limits_min:
                total_max = sum(
                    self.flow_limits_max.get((flow, t), float("inf"))
                    for t in self.SYSTEM_TIME
                )
                min_cum = self.cumulative_flow_limits_min[flow]
                # Only check if there are actual per-period limits for this flow
                has_period_limits = any(
                    (flow, t) in self.flow_limits_max for t in self.SYSTEM_TIME
                )
                if has_period_limits and min_cum > total_max:
                    raise ValueError(
                        f"Cumulative flow limit min ({min_cum}) > sum of per-period "
                        f"max ({total_max}) for {flow}. Constraints would be infeasible."
                    )

        if self.flow_limits_min and self.cumulative_flow_limits_max:
            for flow in self.cumulative_flow_limits_max:
                total_min = sum(
                    self.flow_limits_min.get((flow, t), 0.0)
                    for t in self.SYSTEM_TIME
                )
                max_cum = self.cumulative_flow_limits_max[flow]
                if total_min > max_cum:
                    raise ValueError(
                        f"Sum of per-period flow limit min ({total_min}) > cumulative "
                        f"flow limit max ({max_cum}) for {flow}. Constraints would be infeasible."
                    )

        return self

    @model_validator(mode="after")
    def warn_negative_tau_boundary(self) -> "OptimizationModelInputs":
        """
        Warn about negative process times that may fall outside SYSTEM_TIME.

        When tau < 0 (e.g., construction before deployment), the contribution
        appears at system time (t - tau). If min(SYSTEM_TIME) - tau < min(SYSTEM_TIME),
        those contributions are lost for early installations.

        Example: With SYSTEM_TIME starting at 2020 and tau=-1:
        - Installation at 2020 has construction at t=2019 (NOT in SYSTEM_TIME)
        - These emissions are silently ignored

        This validator warns users about this boundary condition.
        """
        from loguru import logger

        if not self.PROCESS_TIME or not self.SYSTEM_TIME:
            return self

        min_tau = min(self.PROCESS_TIME)
        min_system_time = min(self.SYSTEM_TIME)

        if min_tau < 0:
            # Check which tensors have non-zero values at negative tau
            affected_flows = []

            for (proc, flow, tau), val in self.foreground_biosphere.items():
                if tau < 0 and val != 0:
                    affected_flows.append(f"biosphere ({proc}, {flow}, tau={tau})")

            for (proc, flow, tau), val in self.foreground_technosphere.items():
                if tau < 0 and val != 0:
                    affected_flows.append(f"technosphere ({proc}, {flow}, tau={tau})")

            if affected_flows:
                affected_years = abs(min_tau)
                logger.warning(
                    f"Process time includes negative values (min tau = {min_tau}). "
                    f"Flows at negative tau for installations in the first {affected_years} "
                    f"year(s) of SYSTEM_TIME ({min_system_time}) will NOT be counted "
                    f"because they fall before SYSTEM_TIME starts. "
                    f"Affected flows: {affected_flows[:5]}{'...' if len(affected_flows) > 5 else ''}"
                )

        return self

    def get_scaled_copy(self) -> Tuple["OptimizationModelInputs", Dict[str, Any]]:
        """
        Create a scaled copy of inputs for numerical stability in optimization.

        Scaling improves solver performance by normalizing values to similar magnitudes.
        The method scales foreground tensors, characterization factors, demand, and
        limits while preserving the original data structure. Scaling factors are returned
        for denormalizing results.

        Returns
        -------
        tuple[OptimizationModelInputs, dict]
            - Scaled copy of the model inputs
            - Dictionary of scaling factors used:
                - "foreground": Scale factor for all foreground tensors and demand
                - "characterization": Dict mapping each category to its scale factor
        """
        # Deep copy to preserve raw data
        scaled = copy.deepcopy(self)
        scaling_factors: Dict[str, Any] = {}

        # 1. Compute shared foreground scale
        fg_vals = list(self.foreground_production.values())
        fg_vals += list(self.foreground_biosphere.values())
        fg_vals += list(self.foreground_technosphere.values())
        fg_vals += list(self.internal_demand_technosphere.values())
        fg_scale = max((abs(v) for v in fg_vals), default=1.0)
        if fg_scale == 0:
            fg_scale = 1.0
        scaling_factors["foreground"] = fg_scale

        # Apply foreground scaling
        for d in (
            "foreground_production",
            "foreground_biosphere",
            "foreground_technosphere",
            "internal_demand_technosphere",
        ):
            orig: Dict = getattr(self, d)
            scaled_dict = {k: orig[k] / fg_scale for k in orig}
            setattr(scaled, d, scaled_dict)

        # 2. Compute per-category characterization scales
        cat_scales: Dict[str, float] = {}
        for cat in self.CATEGORY:
            vals = [v for (c, *_), v in self.characterization.items() if c == cat]
            scale = max((abs(v) for v in vals), default=1.0)
            if scale == 0:
                scale = 1.0
            cat_scales[cat] = scale

        scaling_factors["characterization"] = cat_scales

        # Apply characterization scaling
        scaled_char: Dict = {}
        for key, v in self.characterization.items():
            cat, *_ = key
            scale = cat_scales.get(cat, 1.0)
            scaled_char[key] = v / scale
        scaled.characterization = scaled_char

        # 3. Scale demand by foreground scale
        if self.demand is not None:
            scaled.demand = {k: v / fg_scale for k, v in self.demand.items()}

        # 4. Scale category impact limits (if provided)
        # Impact is computed as: (biosphere/fg_scale) * (characterization/cat_scale) * installation
        # So scaled_impact = real_impact / (fg_scale * cat_scale)
        # Therefore, the limit must also be divided by both scales
        if self.category_impact_limits is not None:
            scaled.category_impact_limits = {
                (cat, t): lim / (fg_scale * cat_scales.get(cat, 1.0))
                for (cat, t), lim in self.category_impact_limits.items()
            }

        if self.cumulative_category_impact_limits is not None:
            scaled.cumulative_category_impact_limits = {
                cat: lim / (fg_scale * cat_scales.get(cat, 1.0))
                for cat, lim in self.cumulative_category_impact_limits.items()
            }

        # NOTE: Process limits are NOT scaled because var_installation is in real units
        # (it must be in real units for background inventory calculations to work correctly)

        return scaled, scaling_factors

    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": False,
        "extra": "forbid",  # Reject unknown fields to catch typos
    }


class ModelInputManager:
    """
    Interface between LCA data processing and optimization modeling.

    The `ModelInputManager` is responsible for transforming, validating, and managing
    structured data inputs for optimization models derived from an `LCADataProcessor`.

    Responsibilities:

    - Extracts raw structural and quantitative data from an `LCADataProcessor` instance.
    - Constructs and validates a `OptimizationModelInputs` Pydantic model, ensuring all necessary
      fields are populated and internally consistent.
    - Allows for user-defined overrides of any input fields to enable customization,
      correction, or scenario-specific tuning.
    - Supports serialization and deserialization of `OptimizationModelInputs` for reproducibility,
      sharing, or caching via `.json` or `.pickle`.
    - Provides access to scaled versions of the model inputs (e.g., for numerical
      stability in optimization solvers), with metadata on scaling transformations.

    This class is intended to serve as the main interface between upstream life cycle
    assessment (LCA) data and downstream optimization workflows, abstracting away
    validation, preprocessing, and I/O concerns from both ends.

    Example
    -------
    >>> # Initialize
    >>> manager = ModelInputManager()
    >>>
    >>> # Parse data from LCA data processor
    >>> inputs = manager.parse_from_lca_processor(lca_data_processor)
    >>>
    >>> # Optionally override fields
    >>> inputs = manager.override_inputs(PROCESS=["P1", "P2"], demand={...})
    >>>
    >>> # Save to disk
    >>> manager.save("inputs.json")
    >>>
    >>> # Load from disk
    >>> manager.load("inputs.json")
    >>>
    >>> # Get a numerically scaled version
    >>> scaled_inputs, scale_factors = inputs.get_scaled_copy()
    """

    def __init__(self):
        """
        Initialize a new ModelInputManager with empty model inputs.

        The manager starts with no model inputs. Use `parse_from_lca_processor()`
        to populate inputs from an LCADataProcessor, or use `load()` to load
        previously saved inputs from disk.
        """
        self.model_inputs = None

    def parse_from_lca_processor(
        self, lca_processor: LCADataProcessor
    ) -> OptimizationModelInputs:
        """
        Extracts data from the LCADataProcessor and constructs OptimizationModelInputs.
        """
        # Extract data
        data = {
            "PROCESS": list(lca_processor.processes.keys()),
            "process_names": lca_processor.processes,
            "PRODUCT": list(lca_processor.products.keys()),
            "INTERMEDIATE_FLOW": list(lca_processor.intermediate_flows.keys()),
            "ELEMENTARY_FLOW": list(lca_processor.elementary_flows.keys()),
            "BACKGROUND_ID": list(lca_processor.background_dbs.keys()),
            "PROCESS_TIME": list(lca_processor.process_time),
            "SYSTEM_TIME": list(lca_processor.system_time),
            "CATEGORY": list(lca_processor.category),
            "demand": lca_processor.demand,
            "operation_flow": lca_processor.operation_flow,
            "foreground_technosphere": lca_processor.foreground_technosphere,
            "internal_demand_technosphere": lca_processor.internal_demand_technosphere,
            "foreground_biosphere": lca_processor.foreground_biosphere,
            "foreground_production": lca_processor.foreground_production,
            "background_inventory": lca_processor.background_inventory,
            "mapping": lca_processor.mapping,
            "characterization": lca_processor.characterization,
            "operation_time_limits": lca_processor.operation_time_limits,
            # Optional constraints not populated by default
            "category_impact_limits": None,
            "cumulative_category_impact_limits": None,
            "process_deployment_limits_max": None,
            "process_deployment_limits_min": None,
            "process_operation_limits_max": None,
            "process_operation_limits_min": None,
            "cumulative_process_limits_max": None,
            "cumulative_process_limits_min": None,
            "process_coupling": None,
            "existing_capacity": None,
            "flow_limits_max": None,
            "flow_limits_min": None,
            "cumulative_flow_limits_max": None,
            "cumulative_flow_limits_min": None,
        }
        self.model_inputs = OptimizationModelInputs(**data)
        return self.model_inputs

    def override(self, **overrides) -> OptimizationModelInputs:
        """
        Override fields of the current OptimizationModelInputs instance and re-validate.

        Parameters:
            overrides: Keyword arguments matching OptimizationModelInputs fields to override.
        """
        data = self.model_inputs.model_dump()
        data.update(overrides)
        self.model_inputs = OptimizationModelInputs(**data)
        return self.model_inputs

    def extend_demand(self, years: int) -> OptimizationModelInputs:
        """
        Extend demand beyond the current horizon by repeating last known values.

        This addresses the "end-of-horizon" effect where the optimizer doesn't
        build capacity near the end because there's no future demand. By extending
        demand, the model accounts for ongoing production requirements.

        Also extends all time-indexed tensors (foreground_technosphere,
        foreground_biosphere, background_inventory, characterization) by copying
        the last year's values to the extended years.

        Parameters
        ----------
        years : int
            Number of additional years to extend beyond current SYSTEM_TIME.

        Returns
        -------
        OptimizationModelInputs
            Updated model inputs with extended time horizon and demand.
        """
        if self.model_inputs is None:
            raise ValueError("No OptimizationModelInputs to extend.")

        if years <= 0:
            return self.model_inputs

        # Determine time extension
        current_times = list(self.model_inputs.SYSTEM_TIME)
        last_year = max(current_times)
        extended_times = list(range(last_year + 1, last_year + 1 + years))
        new_system_time = current_times + extended_times

        # Extend demand: repeat last value per product
        extended_demand = dict(self.model_inputs.demand)
        for product in self.model_inputs.PRODUCT:
            last_demand = self.model_inputs.demand.get((product, last_year), 0)
            for t in extended_times:
                extended_demand[(product, t)] = last_demand

        # Extend foreground_technosphere: (process, flow, tau, system_time)
        extended_fg_tech = dict(self.model_inputs.foreground_technosphere)
        for key, value in self.model_inputs.foreground_technosphere.items():
            if key[3] == last_year:  # key = (p, f, tau, t)
                for t in extended_times:
                    extended_fg_tech[(key[0], key[1], key[2], t)] = value

        # Extend foreground_biosphere: (process, elem_flow, tau, system_time)
        extended_fg_bio = dict(self.model_inputs.foreground_biosphere)
        for key, value in self.model_inputs.foreground_biosphere.items():
            if key[3] == last_year:  # key = (p, e, tau, t)
                for t in extended_times:
                    extended_fg_bio[(key[0], key[1], key[2], t)] = value

        # Extend background_inventory: (process, elem_flow, tau, system_time)
        extended_bg_inv = dict(self.model_inputs.background_inventory)
        for key, value in self.model_inputs.background_inventory.items():
            if key[3] == last_year:  # key = (p, e, tau, t)
                for t in extended_times:
                    extended_bg_inv[(key[0], key[1], key[2], t)] = value

        # Extend characterization: (category, elem_flow, system_time)
        extended_char = dict(self.model_inputs.characterization)
        for key, value in self.model_inputs.characterization.items():
            if key[2] == last_year:  # key = (c, e, t)
                for t in extended_times:
                    extended_char[(key[0], key[1], t)] = value

        # Use override to create new model inputs with extended data
        return self.override(
            SYSTEM_TIME=new_system_time,
            demand=extended_demand,
            foreground_technosphere=extended_fg_tech,
            foreground_biosphere=extended_fg_bio,
            background_inventory=extended_bg_inv,
            characterization=extended_char,
        )

    @staticmethod
    def _tuple_key_to_str(key: Tuple) -> str:
        """Convert tuple key to JSON-serializable string."""
        return json.dumps(key)

    @staticmethod
    def _str_to_tuple_key(key_str: str) -> Tuple:
        """Convert JSON string back to tuple key."""
        return tuple(json.loads(key_str))

    @staticmethod
    def _serialize_dict_with_tuple_keys(d: Optional[Dict]) -> Optional[Dict]:
        """Convert dictionary with tuple keys to dictionary with string keys."""
        if d is None:
            return None
        return {ModelInputManager._tuple_key_to_str(k): v for k, v in d.items()}

    @staticmethod
    def _deserialize_dict_with_tuple_keys(d: Optional[Dict]) -> Optional[Dict]:
        """Convert dictionary with string keys back to dictionary with tuple keys."""
        if d is None:
            return None
        return {ModelInputManager._str_to_tuple_key(k): v for k, v in d.items()}

    def save_inputs(self, path: str) -> None:
        """
        Save the current OptimizationModelInputs to a JSON or pickle file.

        Use this to save model inputs so you can recreate the optimization model
        without re-running LCA processing.

        Parameters
        ----------
        path : str
            File path with .json or .pkl extension.
            - .json: Human-readable, good for inspection and version control
            - .pkl: Faster, preserves exact Python types

        Examples
        --------
        >>> manager.save_inputs("model_inputs.json")
        >>> # Later:
        >>> manager.load_inputs("model_inputs.json")
        >>> model = optimizer.create_model(manager.model_inputs, ...)
        """
        if self.model_inputs is None:
            raise ValueError("No OptimizationModelInputs to save.")
        if path.endswith(".json"):
            # Get model data
            data = self.model_inputs.model_dump()

            # Convert tuple keys to string keys for JSON serialization
            tuple_key_fields = [
                "demand",
                "operation_flow",
                "foreground_technosphere",
                "internal_demand_technosphere",
                "foreground_biosphere",
                "foreground_production",
                "background_inventory",
                "mapping",
                "characterization",
                "process_deployment_limits_max",
                "process_deployment_limits_min",
                "process_operation_limits_max",
                "process_operation_limits_min",
                "process_coupling",
                "existing_capacity",
                "flow_limits_max",
                "flow_limits_min",
                "category_impact_limits",
            ]

            for field in tuple_key_fields:
                if field in data:
                    data[field] = self._serialize_dict_with_tuple_keys(data[field])

            # Special handling for operation_time_limits (values are tuples, not keys)
            if "operation_time_limits" in data and data["operation_time_limits"] is not None:
                data["operation_time_limits"] = {
                    k: list(v) for k, v in data["operation_time_limits"].items()
                }

            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif path.endswith(".pkl"):
            with open(path, "wb") as f:
                pickle.dump(self.model_inputs, f)
        else:
            raise ValueError("Unsupported file extension; use .json or .pkl")

    # Alias for backward compatibility
    save = save_inputs

    def load_inputs(self, path: str) -> OptimizationModelInputs:
        """
        Load OptimizationModelInputs from a JSON or pickle file.

        Use this to load previously saved model inputs, allowing you to
        recreate the optimization model without re-running LCA processing.

        Parameters
        ----------
        path : str
            File path with .json or .pkl extension.

        Returns
        -------
        OptimizationModelInputs
            The loaded model inputs, also stored in self.model_inputs.

        Examples
        --------
        >>> manager = ModelInputManager()
        >>> manager.load_inputs("model_inputs.json")
        >>> model = optimizer.create_model(manager.model_inputs, ...)
        """
        if path.endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)

            # Convert string keys back to tuple keys
            tuple_key_fields = [
                "demand",
                "operation_flow",
                "foreground_technosphere",
                "internal_demand_technosphere",
                "foreground_biosphere",
                "foreground_production",
                "background_inventory",
                "mapping",
                "characterization",
                "process_deployment_limits_max",
                "process_deployment_limits_min",
                "process_operation_limits_max",
                "process_operation_limits_min",
                "process_coupling",
                "existing_capacity",
                "flow_limits_max",
                "flow_limits_min",
                "category_impact_limits",
            ]

            for field in tuple_key_fields:
                if field in data:
                    data[field] = self._deserialize_dict_with_tuple_keys(data[field])

            # Special handling for operation_time_limits (convert lists back to tuples)
            if "operation_time_limits" in data and data["operation_time_limits"] is not None:
                data["operation_time_limits"] = {
                    k: tuple(v) for k, v in data["operation_time_limits"].items()
                }

            self.model_inputs = OptimizationModelInputs(**data)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                self.model_inputs = pickle.load(f)
        else:
            raise ValueError("Unsupported file extension; use .json or .pkl")
        return self.model_inputs

    # Alias for backward compatibility
    load = load_inputs
