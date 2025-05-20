import pickle
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import yaml

from optimex.lca_processor import LCADataProcessor


@dataclass
class ModelInputs:
    """
    Interface data structure for linking LCA-based outputs with optimization inputs.

    This class organizes all relevant inputs needed to build a temporal, process-based
    life cycle model suitable for linear optimization, including foreground and
    background exchanges, temporal system information, and optional process constraints.

    Attributes
    ----------
    PROCESS : list of str
        Identifiers for all modeled processes.
    FUNCTIONAL_FLOW : list of str
        Identifiers for flows delivering functional output.
    INTERMEDIATE_FLOW : list of str
        Identifiers for flows exchanged between processes.
    ELEMENTARY_FLOW : list of str
        Identifiers for flows representing exchanges with the environment.
    BACKGROUND_ID : list of str
        Identifiers for background system databases.
    PROCESS_TIME : list of int
        Relative time steps representing the operation timeline of each process.
        Time is measured from process deployment (e.g., year offsets like 0, 1, 2).
    SYSTEM_TIME : list of int
        Absolute time steps representing actual years in the system (e.g., 2025, 2026).
        Used for defining system-wide constraints, demand, and impact assessments.
    CATEGORY : list of str
        Impact categories for each elementary flow.
    demand : dict of (str, int) to float
        Maps (functional_flow, system_time) to demand amount.
    process_operation_time : dict of str to (int, int)
        Maps each process to its (start_time, end_time) operation interval.
    foreground_technosphere : dict of (str, str, int) to float
        Maps (process, intermediate_flow, process_time) to flow amount.
    foreground_biosphere : dict of (str, str, int) to float
        Maps (process, elementary_flow, process_time) to emission or resource amount.
    foreground_production : dict of (str, str, int) to float
        Maps (process, functional_flow, process_time) to produced amount.
    background_inventory : dict of (str, str, str) to float
        Maps (background_id, intermediate_flow, environmental_flow) to exchange amount.
    mapping : dict of (str, int) to float
        Maps (background_id, system_time) to scaling or availability factor.
    characterization : dict of (str, int) to float
        Maps (elementary_flow, system_time) to impact characterization factor.
    process_limits_max : dict of (str, int) to float, optional
        Upper bounds on (process, system_time) deployment.
    process_limits_min : dict of (str, int) to float, optional
        Lower bounds on (process, system_time) deployment.
    cumulative_process_limits_max : dict of str to float, optional
        Global upper bound on cumulative deployment for a process.
    cumulative_process_limits_min : dict of str to float, optional
        Global lower bound on cumulative deployment for a process.
    process_coupling : dict of (str, str) to float, optional
        Coupling of process deplyoment. Each key is a tuple of two process identifiers
        `(P1, P2)`, and the corresponding value is a multiplier `k` such that the
        deplyoment of `P1` is constrained to be `k * P2`
    process_names : dict of str to str, optional
        Maps process identifiers to human-readable names.
    process_limits_max_default : float, default: inf
        Default upper bound for annual process deployment of all process if not
        specified explicitly in process_limits_max.
    process_limits_min_default : float, default: 0.0
        Default lower bound for annual process deployment of all process if not
        specified explicitly in process_limits_min.
    cumulative_process_limits_max_default : float, default: inf
        Default global upper bound for total process deployment of all proscceses if
        not specified explicitly in cumulative_process_limits_max.
    cumulative_process_limits_min_default : float, default: 0.0
        Default global lower bound for total process deployment of all processes if not
        explicitly specified in cumulative_process_limits_min.
    """

    PROCESS: List[str]
    FUNCTIONAL_FLOW: List[str]
    INTERMEDIATE_FLOW: List[str]
    ELEMENTARY_FLOW: List[str]
    BACKGROUND_ID: List[str]
    PROCESS_TIME: List[int]
    SYSTEM_TIME: List[int]
    CATEGORY: List[str]

    demand: Dict[Tuple[str, int], float]
    process_operation_time: Dict[str, Tuple[int, int]]

    foreground_technosphere: Dict[Tuple[str, str, int], float]
    foreground_biosphere: Dict[Tuple[str, str, int], float]
    foreground_production: Dict[Tuple[str, str, int], float]

    background_inventory: Dict[Tuple[str, str, str], float]
    mapping: Dict[Tuple[str, int], float]
    characterization: Dict[Tuple[str, int], float]

    category_impact_limit: Dict[str, float] = None
    process_limits_max: Dict[Tuple[str, int], float] = None
    process_limits_min: Dict[Tuple[str, int], float] = None
    cumulative_process_limits_max: Dict[str, float] = None
    cumulative_process_limits_min: Dict[str, float] = None

    process_coupling: Dict[Tuple[str, str], float] = None
    process_names: Dict[str, str] = None

    process_limits_max_default: float = float("inf")
    process_limits_min_default: float = 0.0
    cumulative_process_limits_max_default: float = float("inf")
    cumulative_process_limits_min_default: float = 0.0


class Converter:
    """
    Converts and validates inputs from an LCADataProcessor instance for use in
    optimization.

    This class extracts relevant structural data from an `LCADataProcessor` object
    and constructs a `ModelInputs` object, ensuring format compatibility
    and optionally filtering or renaming components.

    Parameters
    ----------
    lca_data_processor : LCADataProcessor
        The source object containing LCA data structures such as processes,
        flows, and background information.
    """

    def __init__(self, lca_data_processor: LCADataProcessor):
        self.lca_data_processor = lca_data_processor
        self.model_inputs = None

    def combine_and_check(self, **kwargs) -> ModelInputs:
        """
        Combine data from `LCADataProcessor` and additional user-provided parameters,
        validate the input structure, and return a validated `ModelInputs` object.

        This method acts as an interface between the LCA-based data in
        `LCADataProcessor` and the structured input required for the optimization
        process. It collects data from the `LCADataProcessor` object, formats it
        according to the `ModelInputs` dataclass, and allows the user to override any
        default values with additional parameters passed through `kwargs`.
        The method performs validation to ensure that the data adheres to the expected
        structure and consistency between different input types, such as processes,
        flows, and time periods.
        If any inconsistencies are detected, a `ValueError` is raised.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments that override the default values from the LCADataProcessor
            instance. These correspond to the fields in `ModelInputs.
        """

        # STEP 1: Retrieve all data from kwargs or default to self.lca_data_processor
        process = kwargs.get("PROCESS", list(self.lca_data_processor.processes.keys()))
        process_names = kwargs.get("process_names", self.lca_data_processor.processes)
        functional_flow = kwargs.get(
            "FUNCTIONAL_FLOW", list(self.lca_data_processor.functional_flows)
        )
        intermediate_flow = kwargs.get(
            "INTERMEDIATE_FLOW", list(self.lca_data_processor.intermediate_flows.keys())
        )
        elementary_flow = kwargs.get(
            "ELEMENTARY_FLOW", list(self.lca_data_processor.elementary_flows.keys())
        )
        background_id = kwargs.get(
            "BACKGROUND_ID", list(self.lca_data_processor.background_dbs.keys())
        )
        process_time = kwargs.get(
            "PROCESS_TIME", list(self.lca_data_processor.process_time)
        )
        system_time = kwargs.get(
            "SYSTEM_TIME", list(self.lca_data_processor.system_time)
        )
        category = kwargs.get("CATEGORY", list(self.lca_data_processor.category))
        demand = kwargs.get("demand", self.lca_data_processor.demand)
        process_operation_time = kwargs.get(
            "process_operation_time", self.lca_data_processor.process_operation_time
        )
        foreground_technosphere = kwargs.get(
            "foreground_technosphere", self.lca_data_processor.foreground_technosphere
        )
        foreground_biosphere = kwargs.get(
            "foreground_biosphere", self.lca_data_processor.foreground_biosphere
        )
        foreground_production = kwargs.get(
            "foreground_production", self.lca_data_processor.foreground_production
        )
        background_inventory = kwargs.get(
            "background_inventory", self.lca_data_processor.background_inventory
        )
        mapping = kwargs.get("mapping", self.lca_data_processor.mapping)
        characterization = kwargs.get(
            "characterization", self.lca_data_processor.characterization
        )
        category_impact_limit = kwargs.get("category_impact_limit")
        process_limits_max = kwargs.get("process_limits_max")
        process_limits_min = kwargs.get("process_limits_min")
        cumulative_process_limits_max = kwargs.get("cumulative_process_limits_max")
        cumulative_process_limits_min = kwargs.get("cumulative_process_limits_min")
        process_coupling = kwargs.get("process_coupling")
        process_limits_max_default = kwargs.get(
            "process_limits_max_default", float("inf")
        )
        process_limits_min_default = kwargs.get("process_limits_min_default", 0.0)
        cumulative_process_limits_max_default = kwargs.get(
            "cumulative_process_limits_max_default", float("inf")
        )
        cumulative_process_limits_min_default = kwargs.get(
            "cumulative_process_limits_min_default", 0.0
        )

        # STEP 2: Define helper for validation
        def assert_keys_in_set(keys, valid_set, dict_name):
            for key in keys:
                if key not in valid_set:
                    raise ValueError(f"Invalid key {key} found in {dict_name}")

        # STEP 3: Validate structure of all key-based tensors
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

        for cat, elem_flow, sys_time in characterization.keys():
            assert_keys_in_set([elem_flow], elementary_flow, "characterization")
            assert_keys_in_set([sys_time], system_time, "characterization")
            assert_keys_in_set([cat], category, "characterization")

        if category_impact_limit is not None:
            for cat, limit in category_impact_limit.items():
                assert_keys_in_set([cat], category, "category_impact_limit")

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

        # STEP 4: Build and store model input object
        self.model_inputs = ModelInputs(
            PROCESS=process,
            process_names=process_names,
            FUNCTIONAL_FLOW=functional_flow,
            INTERMEDIATE_FLOW=intermediate_flow,
            ELEMENTARY_FLOW=elementary_flow,
            BACKGROUND_ID=background_id,
            PROCESS_TIME=process_time,
            SYSTEM_TIME=system_time,
            CATEGORY=category,
            demand=demand,
            process_operation_time=process_operation_time,
            foreground_technosphere=foreground_technosphere,
            foreground_biosphere=foreground_biosphere,
            foreground_production=foreground_production,
            background_inventory=background_inventory,
            mapping=mapping,
            characterization=characterization,
            category_impact_limit=category_impact_limit,
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
        """
        Save the current model inputs to a YAML file.

        Parameters
        ----------
        filename : str, optional
            Path to the YAML file to write to. Default is "model_inputs.yaml".
        """

        with open(filename, "w") as f:
            yaml.dump(self.model_inputs, f)

    def load_model_inputs(self, filename="model_inputs.yaml"):
        """
        Load model inputs from a YAML file.

        Parameters
        ----------
        filename : str, optional
            Path to the YAML file to load from. Default is "model_inputs.yaml".

        Returns
        -------
        ModelInputs
            The loaded model input data.
        """
        with open(filename, "r") as f:
            self.model_inputs = yaml.load(f, Loader=yaml.FullLoader)
        return self.model_inputs

    def pickle_model_inputs(self, filename="model_inputs.pkl"):
        """
        Serialize and save the model inputs to a binary pickle file.

        Parameters
        ----------
        filename : str, optional
            Path to the pickle file to write to. Default is "model_inputs.pkl".
        """
        with open(filename, "wb") as f:
            pickle.dump(self.model_inputs, f)

    def unpickle_model_inputs(self, filename="model_inputs.pkl"):
        """
        Load model inputs from a pickle file.

            .. warning::
        Only unpickle data you trust. Loading pickle files from untrusted
        sources can be insecure.

        Parameters
        ----------
        filename : str, optional
            Path to the pickle file to load from. Default is "model_inputs.pkl".

        Returns
        -------
        ModelInputs
            The deserialized model input data.
        """
        with open(filename, "rb") as f:
            self.model_inputs = pickle.load(f)
        return self.model_inputs

    def model_inputs_to_dict(self, model_inputs: ModelInputs) -> dict:
        """
        Convert a ModelInputs object into a dictionary.

        Parameters
        ----------
        model_inputs : ModelInputs
            The model inputs dataclass instance to convert.

        Returns
        -------
        dict
            A dictionary representation of the model inputs.
        """
        if model_inputs is None:
            raise ValueError("Model inputs have not been set.")
        return asdict(model_inputs)
