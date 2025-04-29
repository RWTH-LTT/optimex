import pickle
from dataclasses import dataclass
from typing import Dict, Tuple

import yaml

from optimex.optimex import Optimex


@dataclass
class ModelInputs:
    PROCESS: list[str]
    FUNCTIONAL_FLOW: list[str]
    INTERMEDIATE_FLOW: list[str]
    ELEMENTARY_FLOW: list[str]
    # INDICATOR: list[str]
    BACKGROUND_ID: list[str]
    PROCESS_TIME: list[int]
    SYSTEM_TIME: list[int]
    demand: Dict[Tuple[str, int], float]  # (functional_flow, system_time): amount
    process_operation_time: Dict[str, Tuple[int, int]]  # process: (start, end)
    foreground_technosphere: Dict[
        Tuple[str, str, int], float
    ]  # (process, intermediate_flow, process_time): amount
    foreground_biosphere: Dict[
        Tuple[str, str, int], float
    ]  # (process, elementary_flow, process_time): amount
    foreground_production: Dict[
        Tuple[str, str, int], float
    ]  # (process, functional_flow, process_time): amount
    background_inventory: Dict[
        Tuple[str, str, str], float
    ]  # (background_id, intermediate_flow, environmental_flow): amount
    mapping: Dict[Tuple[str, int], float]  # (background_id, system_time): amount
    characterization: Dict[
        Tuple[str, int], float
    ]  # (elementary flow, system_time): amount
    process_limits_max: Dict[Tuple[str, int], float] = None
    process_limits_min: Dict[Tuple[str, int], float] = None
    cumulative_process_limits_max: Dict[str, float] = None
    cumulative_process_limits_min: Dict[str, float] = None
    process_coupling: Dict[str, str] = None
    process_names: Dict[str, str] = None
    process_limits_max_default: float = float("inf")
    process_limits_min_default: float = 0.0
    cumulative_process_limits_max_default: float = float("inf")
    cumulative_process_limits_min_default: float = 0.0


class Converter:
    def __init__(self, optimex: Optimex):
        self.optimex = optimex
        self.model_inputs = None

    def combine_and_check(self, **kwargs) -> ModelInputs:
        process = kwargs.get("process", list(self.optimex.processes.keys()))
        process_names = kwargs.get("process_names", self.optimex.processes)
        functional_flow = kwargs.get(
            "functional_flow", list(self.optimex.functional_flows)
        )
        intermediate_flow = kwargs.get(
            "intermediate_flow", list(self.optimex.intermediate_flows.keys())
        )
        elementary_flow = kwargs.get(
            "elementary_flow", list(self.optimex.elementary_flows.keys())
        )
        background_id = kwargs.get(
            "background_id", list(self.optimex.background_dbs.keys())
        )
        process_time = kwargs.get("process_time", list(self.optimex.process_time))
        system_time = kwargs.get("system_time", list(self.optimex.system_time))

        demand = kwargs.get("demand", self.optimex.demand)
        process_operation_time = kwargs.get(
            "process_operation_time", self.optimex.process_operation_time
        )
        foreground_technosphere = kwargs.get(
            "foreground_technosphere", self.optimex.foreground_technosphere
        )
        foreground_biosphere = kwargs.get(
            "foreground_biosphere", self.optimex.foreground_biosphere
        )
        foreground_production = kwargs.get(
            "foreground_production", self.optimex.foreground_production
        )
        background_inventory = kwargs.get(
            "background_inventory", self.optimex._background_inventory
        )
        mapping = kwargs.get("mapping", self.optimex.mapping)
        characterization = kwargs.get("characterization", self.optimex.characterization)
        process_limits_max = kwargs.get("process_limits_max")
        process_limits_min = kwargs.get("process_limits_min")
        cumulative_process_limits_max = kwargs.get("cumulative_process_limits_max")
        cumulative_process_limits_min = kwargs.get("cumulative_process_limits_min")
        process_coupling = kwargs.get("process_coupling")
        process_limits_max_default = kwargs.get(
            "process_limits_max_default", float("inf")
        )
        process_limits_min_default = kwargs.get("process_limits_min_default", 0)
        cumulative_process_limits_max_default = kwargs.get(
            "cumulative_process_limits_max_default", float("inf")
        )
        cumulative_process_limits_min_default = kwargs.get(
            "cumulative_process_limits_min_default", 0.0
        )

        def assert_keys_in_set(keys, valid_set, dict_name):
            for key in keys:
                if key not in valid_set:
                    raise ValueError(f"Invalid key {key} found in {dict_name}")

        for ref_flow, sys_time in demand.keys():
            assert_keys_in_set([ref_flow], functional_flow, "demand")
            assert_keys_in_set([sys_time], system_time, "demand")

        for proc, int_flow, proc_time in foreground_technosphere.keys():
            assert_keys_in_set([proc], process, "foreground_technosphere")
            assert_keys_in_set([int_flow], intermediate_flow, "foreground_technosphere")
            assert_keys_in_set([proc_time], process_time, "foreground_technosphere")

        for proc, elem_flow, proc_time in foreground_biosphere.keys():
            assert_keys_in_set([proc], process, "foreground_biosphere")
            assert_keys_in_set([elem_flow], elementary_flow, "foreground_biosphere")
            assert_keys_in_set([proc_time], process_time, "foreground_biosphere")

        for proc, ref_flow, proc_time in foreground_production.keys():
            assert_keys_in_set([proc], process, "foreground_production")
            assert_keys_in_set([ref_flow], functional_flow, "foreground_production")
            assert_keys_in_set([proc_time], process_time, "foreground_production")

        for bg_id, int_flow, env_flow in background_inventory.keys():
            assert_keys_in_set([bg_id], background_id, "background_inventory")
            assert_keys_in_set([int_flow], intermediate_flow, "background_inventory")
            assert_keys_in_set([env_flow], elementary_flow, "background_inventory")

        for bg_id, sys_time in mapping.keys():
            assert_keys_in_set([bg_id], background_id, "mapping")
            assert_keys_in_set([sys_time], system_time, "mapping")

        for elem_flow, sys_time in characterization.keys():
            assert_keys_in_set([elem_flow], elementary_flow, "characterization")
            assert_keys_in_set([sys_time], system_time, "characterization")

        if process_limits_max is not None:
            for proc, sys_time in process_limits_max.keys():
                assert_keys_in_set([proc], process, "process_limits_max")
                assert_keys_in_set([sys_time], system_time, "process_limits_max")

        if process_limits_min is not None:
            for proc, sys_time in process_limits_min.keys():
                assert_keys_in_set([proc], process, "process_limits_min")
                assert_keys_in_set([sys_time], system_time, "process_limits_min")

        if cumulative_process_limits_max is not None:
            for proc in cumulative_process_limits_max.keys():
                assert_keys_in_set([proc], process, "cumulative_process_limits_max")

        if cumulative_process_limits_min is not None:
            for proc in cumulative_process_limits_min.keys():
                assert_keys_in_set([proc], process, "cumulative_process_limits_min")

        if process_coupling is not None:
            for proc1, proc2 in process_coupling.keys():
                assert_keys_in_set([proc1], process, "process_coupling")
                assert_keys_in_set([proc2], process, "process_coupling")
            for v in process_coupling.values():
                if v <= 0:
                    raise ValueError("Coupling values must be positive")

        # TODO: which tensors have clear properties that can be checked?
        # self._check_mapping_sums(mapping, system_time)

        self.model_inputs = ModelInputs(
            PROCESS=process,
            process_names=process_names,
            FUNCTIONAL_FLOW=functional_flow,
            INTERMEDIATE_FLOW=intermediate_flow,
            ELEMENTARY_FLOW=elementary_flow,
            BACKGROUND_ID=background_id,
            PROCESS_TIME=process_time,
            SYSTEM_TIME=system_time,
            demand=demand,
            process_operation_time=process_operation_time,
            foreground_technosphere=foreground_technosphere,
            foreground_biosphere=foreground_biosphere,
            foreground_production=foreground_production,
            background_inventory=background_inventory,
            mapping=mapping,
            characterization=characterization,
            process_limits_max=process_limits_max,
            process_limits_min=process_limits_min,
            cumulative_process_limits_max=cumulative_process_limits_max,
            cumulative_process_limits_min=cumulative_process_limits_min,
            process_coupling=process_coupling,
            process_limits_max_default=process_limits_max_default,
            process_limits_min_default=process_limits_min_default,
            cumulative_process_limits_max_default=cumulative_process_limits_max_default,
            cumulative_process_limits_min_default=cumulative_process_limits_min_default,
        )
        return self.model_inputs

    def _check_mapping_sums(self, mapping, system_time):
        for sys_time in system_time:
            if not (
                0.99
                <= sum(mapping.get((bg_id, sys_time), 0) for bg_id in mapping.keys())
                <= 1.01
            ):
                raise ValueError(
                    f"Mapping for system time {sys_time} does not sum to 1"
                )
        return True

    def save_model_inputs(self, filename="model_inputs.yaml"):
        with open(filename, "w") as f:
            yaml.dump(self.model_inputs, f)

    def load_model_inputs(self, filename="model_inputs.yaml"):
        with open(filename, "r") as f:
            self.model_inputs = yaml.load(f, Loader=yaml.FullLoader)
        return self.model_inputs

    def pickle_model_inputs(self, filename="model_inputs.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.model_inputs, f)

    def unpickle_model_inputs(self, filename="model_inputs.pkl"):
        with open(filename, "rb") as f:
            self.model_inputs = pickle.load(f)
        return self.model_inputs

    def model_inputs_to_dict(self, model_inputs: ModelInputs) -> dict:
        if model_inputs is None:
            raise ValueError("Model inputs have not been set.")
        return {
            "PROCESS": model_inputs.PROCESS,
            "process_names": model_inputs.process_names,
            "FUNCTIONAL_FLOW": model_inputs.FUNCTIONAL_FLOW,
            "INTERMEDIATE_FLOW": model_inputs.INTERMEDIATE_FLOW,
            "ELEMENTARY_FLOW": model_inputs.ELEMENTARY_FLOW,
            "BACKGROUND_ID": model_inputs.BACKGROUND_ID,
            "PROCESS_TIME": model_inputs.PROCESS_TIME,
            "SYSTEM_TIME": model_inputs.SYSTEM_TIME,
            "demand": model_inputs.demand,
            "process_operation_time": model_inputs.process_operation_time,
            "foreground_technosphere": model_inputs.foreground_technosphere,
            "foreground_biosphere": model_inputs.foreground_biosphere,
            "foreground_production": model_inputs.foreground_production,
            "background_inventory": model_inputs.background_inventory,
            "mapping": model_inputs.mapping,
            "characterization": model_inputs.characterization,
            "process_limits_max": model_inputs.process_limits_max,
            "process_limits_min": model_inputs.process_limits_min,
            "cumulative_process_limits_max": model_inputs.cumulative_process_limits_max,
            "cumulative_process_limits_min": model_inputs.cumulative_process_limits_min,
            "process_coupling": model_inputs.process_coupling,
        }
