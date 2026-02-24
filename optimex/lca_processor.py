"""
Time-explicit LCA data processing for optimization.

This module provides classes and utilities for performing time-explicit Life Cycle
Assessment (LCA) computations using Brightway. It processes
temporal distributions of product demands, constructs foreground and background
inventory tensors, and prepares characterization factors for optimization.

Key classes:
    - LCAConfig: Configuration for LCA computations
    - LCADataProcessor: Main class for time-explicit LCA processing
"""
import heapq
import pickle
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import bw2calc as bc
import bw2data as bd
import numpy as np
import pandas as pd
from bw_temporalis import TemporalDistribution, easy_timedelta_distribution
from dynamic_characterization import characterize
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm


class MetricEnum(str, Enum):
    """
    Supported metrics for dynamic impact characterization.

    Attributes:
        GWP: Global Warming Potential - time-dependent radiative forcing metric
        CRF: Cumulative Radiative Forcing - integrated radiative forcing over time horizon
    """

    GWP = "GWP"
    CRF = "CRF"


class TemporalResolutionEnum(str, Enum):
    """
    Supported temporal resolutions for the optimization model.

    Attributes:
        year: Annual time steps (currently the only supported resolution)
    """

    year = "year"


class CharacterizationMethodConfig(BaseModel):
    """
    Configuration for a single LCIA characterization method.

    Attributes:
        category_name: User-defined identifier for the impact category
            (e.g., 'climate_change_dynamic_gwp').
        brightway_method: Brightway method identifier tuple, either 2 or 3 elements
            (e.g., ('GWP', 'example') or ('IPCC', 'climate change', 'GWP 100a')).
        metric: Impact metric used for dynamic characterization.
            None implies static method.
            Supported values: 'GWP', 'CRF'.
    """

    category_name: str = Field(
        ...,
        description="User-defined name for the impact category "
        "(e.g., 'climate_change_dynamic_gwp').",
    )
    brightway_method: Union[
        Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]
    ] = Field(
        ...,
        description=(
            "The Brightway method tuple with 2 to 4 elements "
            "(e.g., ('IPCC', 'climate change', 'GWP 100a'))."
        ),
    )
    metric: Optional[MetricEnum] = Field(
        None,
        description="Impact metric for dynamic characterization. "
        "Use None for static methods.",
    )

    @property
    def dynamic(self) -> bool:
        """Indicates whether this is a dynamic characterization method."""
        return self.metric is not None


class TemporalConfig(BaseModel):
    """
    Configuration related to temporal aspects of the model.

    Attributes:
        start_date: The start date of the time horizon.
        temporal_resolution: Temporal resolution for the model.
            Options: 'year', 'month', 'day'.
        time_horizon: Length of the time horizon (in units of `temporal_resolution`).
        fixed_time_horizon: If True, the time horizon is calculated from the time of the functional 
            unit (FU) instead of the time of emission
        database_dates: Mapping from database names to their respective reference dates.
    """

    start_date: datetime = Field(
        ..., description="The start date for the time horizon."
    )
    temporal_resolution: TemporalResolutionEnum = Field(
        TemporalResolutionEnum.year,
        description="Temporal resolution for the model (e.g., 'year').",
    )
    time_horizon: int = Field(
        100, description="Length of the time horizon in units of temporal resolution."
    )
    fixed_time_horizon: bool = Field(
        True,
        description="If True, the time horizon is calculated from the time of the functional unit (FU) "
        "instead of the time of emission.",
    )
    database_dates: Optional[Dict[str, Union[datetime, str]]] = Field(
        None,
        description="Mapping from database names to their respective reference dates.",
    )


class BackgroundInventoryConfig(BaseModel):
    """
    Configuration for background inventory data.

    Attributes:
        cutoff: Cutoff threshold for the number of top elementary flows to retain based on impact magnitude.
        calculation_method: Method for calculating the inventory tensor. Options: 'sequential', 'parallel'.
        temporal: If True, perform cross-database graph traversal of background supply chains
            using temporal distributions on exchanges. At each traversal step, the absolute time
            determines which database's coefficients to use, producing a pre-computed
            resolved_background_inventory tensor that replaces background_inventory + mapping.
        path_to_save: Optional path to save the inventory tensor.
        path_to_load: Optional path to load the inventory tensor.
    """

    cutoff: float = Field(
        1e4,
        description="Cutoff threshold for the number of top elementary flows to retain "
        "based on impact magnitude.",
    )
    calculation_method: str = Field(
        "sequential",
        description="Method for calculating the inventory tensor. "
        "Options: 'sequential', 'parallel'.",
    )
    temporal: bool = Field(
        False,
        description="If True, perform cross-database graph traversal of background "
        "supply chains using temporal distributions on exchanges.",
    )
    path_to_save: Optional[str] = Field(
        None, description="Optional path to save the inventory tensor."
    )
    path_to_load: Optional[str] = Field(
        None,
        description="Optional path to load the inventory tensor. "
        "If provided, the tensor will be loaded instead of calculated.",
    )


class LCAConfig(BaseModel):
    """
    Configuration class for Life Cycle Assessment (LCA) data processing.

    Attributes:
        demand: Dictionary {product_node: temporal_distribution} containing time-explicit demands for each product.
            Keys must be Brightway product node objects (bd.get_node(...)).
        temporal: Temporal configuration for model time behavior.
        characterization_methods: List of characterization method configurations.
        background_inventory: Configuration for background inventory data calculation.
    """

    demand: Dict[bd.backends.proxies.Activity, TemporalDistribution]
    temporal: TemporalConfig
    characterization_methods: List[CharacterizationMethodConfig]
    background_inventory: Optional[BackgroundInventoryConfig] = Field(
        default_factory=BackgroundInventoryConfig
    )

    class Config:
        arbitrary_types_allowed = True


class LCADataProcessor:
    """
    Class to perform time-explicit Life Cycle Assessment (LCA)
    computations and gather necessary data for building an optimization model.

    This class is primarily responsible for executing the LCA-based computations
    required to collect all the data needed for building `OptimizationModelInputs`. It is reliant on
    Brightway2, an open-source framework for Life Cycle Assessment, to perform the
    calculations and retrieve LCA results.
    """

    def __init__(self, config: LCAConfig) -> None:
        """
        Initialize the LCADataProcessor with the LCA configuration.

        Parameters
        ----------
        config : LCAConfig
            The configuration object containing all settings for demand,
            temporal parameters, characterization methods, and background inventory.
        """
        self.config = config
        if "foreground" not in bd.databases:
            raise ValueError("Foreground database 'foreground' is not defined.")
        self.foreground_db = bd.Database("foreground")
        self.background_dbs = {}
        if config.temporal.database_dates is not None:
            self.background_dbs = {
                db: date
                for db, date in config.temporal.database_dates.items()
                if db != self.foreground_db.name
            }
        else:
            for db_name in bd.databases:
                db = bd.Database(db_name)
                if (date := db.metadata.get("representative_time")) is not None:
                    self.background_dbs[db.name] = datetime.fromisoformat(date)

        self.biosphere_db = bd.Database(bd.config.biosphere)

        self._demand = {}
        self._processes = {}
        self._products = {}  # Maps product codes to product names
        self._intermediate_flows = {}
        self._elementary_flows = {}

        self._reference_products = set()
        self._system_time = set()
        self._process_time = set()
        self._category = set()

        self._foreground_technosphere = {}
        self._internal_demand_technosphere = {}  # (process, product, year) -> amount
        self._foreground_biosphere = {}
        self._foreground_production = {}
        self._background_inventory = {}
        self._mapping = {}
        self._characterization = {}
        self._operation_flow = {}
        self._operation_time_limits = {}
        
        # Vintage-dependent parameters extracted from exchange attributes
        self._foreground_technosphere_vintages = {}
        self._foreground_biosphere_vintages = {}
        self._foreground_production_vintages = {}
        self._vintage_improvements = {}
        self._reference_vintages = set()

        # Resolved background inventory from temporal graph traversal
        self._resolved_background_inventory = None

        # Cached data for temporal background traversal
        self._bg_lca_objects = None
        self._bg_reverse_activity = None
        self._bg_reverse_biosphere = None
        self._bg_td_lookup = None

        self._parse_demand()
        self._construct_foreground_tensors()
        self._prepare_background_inventory()
        self._construct_mapping_matrix()
        self._construct_characterization_tensor()

        # Resolve background inventory temporally if configured
        if self.config.background_inventory.temporal:
            self._resolve_background_inventory()

    @property
    def processes(self) -> dict:
        """Read-only access to the processes dictionary."""
        return self._processes

    @property
    def intermediate_flows(self) -> dict:
        """Read-only access to the intermediate flows dictionary."""
        return self._intermediate_flows

    @property
    def elementary_flows(self) -> dict:
        """Read-only access to the elementary flows dictionary."""
        return self._elementary_flows

    @property
    def reference_products(self) -> set:
        """Read-only access to the functional flows list."""
        return self._reference_products

    @property
    def system_time(self) -> set:
        """Read-only access to the system time list."""
        return self._system_time

    @property
    def category(self) -> set:
        """Read-only access to the impact categories list."""
        return self._category

    @property
    def process_time(self) -> set:
        """Read-only access to the process time list."""
        return self._process_time

    @property
    def foreground_technosphere(self) -> dict:
        """Read-only access to the foreground technosphere tensor."""
        return self._foreground_technosphere

    @property
    def foreground_biosphere(self) -> dict:
        """Read-only access to the foreground biosphere tensor."""
        return self._foreground_biosphere

    @property
    def foreground_production(self) -> dict:
        """Read-only access to the foreground production tensor."""
        return self._foreground_production

    @property
    def background_inventory(self) -> dict:
        """Read-only access to the inventory tensor."""
        return self._background_inventory

    @property
    def mapping(self) -> dict:
        """Read-only access to the mapping matrix."""
        return self._mapping

    @property
    def characterization(self) -> dict:
        """Read-only access to the characterization matrix."""
        return self._characterization

    @property
    def demand(self) -> dict:
        """Read-only access to the parsed demand dictionary."""
        return self._demand

    @property
    def operation_flow(self) -> dict:
        """Read-only access to the operation flow dictionary."""
        return self._operation_flow

    @property
    def operation_time_limits(self) -> dict:
        """Read-only access to the operation time limits dictionary."""
        return self._operation_time_limits

    @property
    def products(self) -> dict:
        """Read-only access to the products dictionary."""
        return self._products

    @property
    def internal_demand_technosphere(self) -> dict:
        """Read-only access to the internal demand technosphere tensor."""
        return self._internal_demand_technosphere

    @property
    def foreground_technosphere_vintages(self) -> Optional[dict]:
        """Read-only access to vintage-specific technosphere values."""
        return self._foreground_technosphere_vintages if self._foreground_technosphere_vintages else None

    @property
    def foreground_biosphere_vintages(self) -> Optional[dict]:
        """Read-only access to vintage-specific biosphere values."""
        return self._foreground_biosphere_vintages if self._foreground_biosphere_vintages else None

    @property
    def foreground_production_vintages(self) -> Optional[dict]:
        """Read-only access to vintage-specific production values."""
        return self._foreground_production_vintages if self._foreground_production_vintages else None

    @property
    def vintage_improvements(self) -> Optional[dict]:
        """Read-only access to vintage improvement scaling factors."""
        return self._vintage_improvements if self._vintage_improvements else None

    @property
    def reference_vintages(self) -> Optional[list]:
        """Read-only access to reference vintage years."""
        return sorted(list(self._reference_vintages)) if self._reference_vintages else None

    @property
    def resolved_background_inventory(self) -> Optional[Dict[Tuple[str, str, int], float]]:
        """Read-only access to the resolved background inventory tensor.

        Only populated when background_inventory.temporal is True.
        Maps (intermediate_flow_code, elementary_flow_code, system_time) to amount.
        """
        return self._resolved_background_inventory

    def _parse_demand(self) -> None:
        """
        Parse and process the demand dictionary from the configuration.

        This method transforms the demand data into a dictionary mapping (product_code, year)
        tuples to their corresponding amounts. It validates that demand is specified on
        foreground product nodes.

        Side Effects
        ------------
        Updates the following instance attributes:
            - self._demand: dict with keys (product_code, year) and values as amounts.
            - self._products: dict mapping product codes to product names.
            - self._system_time: range of years covering the longest demand interval.
        """
        raw_demand = self.config.demand
        start_year = self.config.temporal.start_date.year
        longest_demand_interval = 0

        for product_node, td in raw_demand.items():
            # Validate demand is on product nodes
            if not hasattr(product_node, 'key'):
                raise ValueError(
                    f"Demand must be on Brightway Node objects, got {type(product_node)}"
                )

            if product_node.get('type') != bd.labels.product_node_default:
                raise ValueError(
                    f"Demand must be on product nodes. "
                    f"Node {product_node['name']} has type {product_node.get('type')}"
                )

            product_code = product_node['code']
            years = td.date.astype("datetime64[Y]").astype(int) + 1970
            if years[-1] - start_year > longest_demand_interval:
                longest_demand_interval = years[-1] - start_year
            amounts = td.amount

            self._demand.update(
                {(product_code, year): amount for year, amount in zip(years, amounts)}
            )

            # Store product information
            self._products[product_code] = product_node['name']

        self._system_time = range(start_year, start_year + longest_demand_interval + 1)
        logger.info(
            "Identified demand in system time range of %s for products %s",
            self._system_time,
            set(product_code for product_code, _ in self._demand.keys()),
        )

    def _construct_foreground_tensors(self) -> None:
        """
        Construct foreground technosphere, biosphere, and production tensors with
        time-explicit structure, supporting explicit product nodes.

        This method constructs tensors based on explicit process and product nodes.
        It processes only process nodes (type=process_node_default) and handles
        three types of edges: production edges (to product nodes), consumption edges
        (from background or foreground products), and biosphere edges (emissions).
        
        Additionally, this method extracts vintage-dependent parameters from exchange
        attributes when present:
        - vintage_improvements: Dict mapping vintage years to scaling factors
        - vintage_amounts: Dict mapping vintage years or (process_time, vintage_year) 
          tuples to amounts

        Side Effects
        -----------
        Updates the following instance attributes:
            - self._foreground_technosphere: dict mapping (process_code, flow_code, year)
              to amount for external intermediate flows (background consumption).
            - self._internal_demand_technosphere: dict mapping (process_code, product_code, year)
              to amount for internal product consumption (foreground products).
            - self._foreground_biosphere: dict mapping (process_code, flow_code, year)
              to amount for biosphere flows (emissions).
            - self._foreground_production: dict mapping (process_code, product_code, year)
              to amount for product production.
            - self._products: dict mapping product codes to their names.
            - self._intermediate_flows: dict mapping background intermediate flow codes
              to their names.
            - self._elementary_flows: dict mapping elementary flow codes to their names.
            - self._processes: dict mapping process codes to their names.
            - self._operation_flow: dict mapping (process_code, flow_code) to boolean
              indicating if the flow occurs during the operation phase.
            - self._operation_time_limits: dict mapping process codes to their
              operation time limits, if defined.
            - self._foreground_technosphere_vintages: dict mapping (process_code, 
              flow_code, process_time, vintage_year) to vintage-specific amounts.
            - self._foreground_biosphere_vintages: dict mapping (process_code, 
              flow_code, process_time, vintage_year) to vintage-specific amounts.
            - self._foreground_production_vintages: dict mapping (process_code, 
              product_code, process_time, vintage_year) to vintage-specific amounts.
            - self._vintage_improvements: dict mapping (process_code, flow_code, 
              vintage_year) to scaling factors.
            - self._reference_vintages: set of reference vintage years.
        """
        technosphere_tensor = {}
        internal_demand_technosphere = {}
        production_tensor = {}
        biosphere_tensor = {}

        for act in self.foreground_db:
            # Only process nodes (not product nodes)
            if act.get('type') != bd.labels.process_node_default:
                continue

            # Store process information
            self._processes.setdefault(act["code"], act["name"])
            if (limits := act.get("operation_time_limits")) is not None:
                self._operation_time_limits[act["code"]] = limits

            for exc in act.exchanges():
                # Extract temporal distribution
                temporal_dist = exc.get(
                    "temporal_distribution",
                    TemporalDistribution(
                        date=np.array([0], dtype="timedelta64[Y]"), amount=np.array([1])
                    ),
                )                
                years = temporal_dist.date.astype("timedelta64[Y]").astype(int)
                # Ensure all years are included in process time
                self._process_time.update(
                    year for year in years if year not in self._process_time
                )
                temporal_factor = temporal_dist.amount

                # Skip if temporal distribution is missing or invalid (empty arrays)
                if years.size == 0 or temporal_factor.size == 0:
                    logger.debug(
                        f"Skipping exchange {exc.input} due to missing or invalid temporal distribution.")
                    continue

                edge_type = exc["type"]
                input_code = exc.input["code"]
                input_name = exc.input["name"]
                input_db = exc.input["database"]
                
                # ========== Extract Vintage Parameters from Exchange Attributes ==========
                # Vintage parameters allow foreground exchanges to vary based on installation year.
                # Two attributes are supported on exchanges:
                #
                # 1. vintage_improvements: Dict mapping vintage years to scaling factors
                #    Format: {vintage_year: scaling_factor}
                #    Example: {2020: 1.0, 2030: 0.75}
                #
                # 2. vintage_amounts: Dict mapping vintage years to amounts
                #    Format: {vintage_year: amount} OR {(process_time, vintage_year): amount}
                #    Example: {2020: 60, 2030: 45} or {(1, 2020): 60, (1, 2030): 45}
                # ==========================================================================
                
                vintage_amounts = exc.get("vintage_amounts")
                vintage_improvements = exc.get("vintage_improvements")
                
                # Process vintage_improvements attribute if present
                if vintage_improvements is not None:
                    if not isinstance(vintage_improvements, dict):
                        logger.warning(
                            f"vintage_improvements on exchange {exc.input} must be a dict, "
                            f"got {type(vintage_improvements).__name__}. Skipping."
                        )
                    else:
                        for vintage_year, scaling_factor in vintage_improvements.items():
                            self._reference_vintages.add(vintage_year)
                            self._vintage_improvements[(act["code"], input_code, vintage_year)] = scaling_factor
                
                # Process vintage_amounts attribute if present
                if vintage_amounts is not None:
                    if not isinstance(vintage_amounts, dict):
                        logger.warning(
                            f"vintage_amounts on exchange {exc.input} must be a dict, "
                            f"got {type(vintage_amounts).__name__}. Skipping vintage extraction."
                        )
                    else:
                        for vintage_key, vintage_amount in vintage_amounts.items():
                            if isinstance(vintage_key, tuple):
                                # Explicit (process_time, vintage_year) format
                                process_time_vintage, vintage_year = vintage_key
                            elif isinstance(vintage_key, int):
                                # Just vintage year - apply to all process times from temporal distribution
                                vintage_year = vintage_key
                                process_time_vintage = None  # Will be expanded for all years
                            else:
                                logger.warning(
                                    f"Invalid vintage_amounts key {vintage_key} on exchange {exc.input}. "
                                    f"Must be int (vintage year) or tuple (process_time, vintage_year)."
                                )
                                continue
                            
                            self._reference_vintages.add(vintage_year)
                            
                            # Determine which process times to apply this vintage value to
                            if process_time_vintage is not None:
                                process_times_to_update = [process_time_vintage]
                            else:
                                # Apply to all process times in temporal distribution
                                process_times_to_update = years
                            
                            for tau in process_times_to_update:
                                # Store in appropriate vintage dictionary based on edge type
                                if edge_type == bd.labels.production_edge_default:
                                    self._foreground_production_vintages[(act["code"], input_code, tau, vintage_year)] = vintage_amount
                                elif edge_type == bd.labels.consumption_edge_default:
                                    if input_db != self.foreground_db.name:
                                        # Only for background consumption (technosphere)
                                        self._foreground_technosphere_vintages[(act["code"], input_code, tau, vintage_year)] = vintage_amount
                                elif edge_type == bd.labels.biosphere_edge_default:
                                    self._foreground_biosphere_vintages[(act["code"], input_code, tau, vintage_year)] = vintage_amount

                # Handle production edges
                if edge_type == bd.labels.production_edge_default:
                    product_code = input_code
                    production_tensor.update({
                        (act["code"], product_code, year): exc["amount"] * factor
                        for year, factor in zip(years, temporal_factor)
                    })
                    if exc.get("operation"):
                        self._operation_flow.update({(act["code"], product_code): True})
                    self._products.setdefault(product_code, input_name)

                # Handle consumption edges
                elif edge_type == bd.labels.consumption_edge_default:
                    if input_db == self.foreground_db.name:
                        # Internal demand: foreground product consumed
                        internal_demand_technosphere.update({
                            (act["code"], input_code, year): exc["amount"] * factor
                            for year, factor in zip(years, temporal_factor)
                        })
                        if exc.get("operation"):
                            self._operation_flow.update({(act["code"], input_code): True})
                        self._products.setdefault(input_code, input_name)
                    else:
                        # External intermediate: background consumption
                        technosphere_tensor.update({
                            (act["code"], input_code, year): exc["amount"] * factor
                            for year, factor in zip(years, temporal_factor)
                        })
                        if exc.get("operation"):
                            self._operation_flow.update({(act["code"], input_code): True})
                        self._intermediate_flows.setdefault(input_code, input_name)

                # Handle biosphere edges
                elif edge_type == bd.labels.biosphere_edge_default:
                    biosphere_tensor.update({
                        (act["code"], input_code, year): exc["amount"] * factor
                        for year, factor in zip(years, temporal_factor)
                    })
                    if exc.get("operation"):
                        self._operation_flow.update({(act["code"], input_code): True})
                    self._elementary_flows.setdefault(input_code, input_name)

        # Store the tensors as protected variables
        self._foreground_technosphere = technosphere_tensor
        self._internal_demand_technosphere = internal_demand_technosphere
        self._foreground_biosphere = biosphere_tensor
        self._foreground_production = production_tensor

        # Compute and log tensor shapes
        def log_tensor_dimensions(tensor, name):
            processes = {k[0] for k in tensor}
            flows = {k[1] for k in tensor}
            years = {k[2] for k in tensor}
            logger.info(
                f"{name} shape: ({len(processes)} processes, {len(flows)} flows, "
                f"{len(years)} years) with {len(tensor)} total entries."
            )

        logger.info("Constructed foreground tensors.")
        log_tensor_dimensions(technosphere_tensor, "Technosphere (external)")
        log_tensor_dimensions(internal_demand_technosphere, "Internal demand")
        log_tensor_dimensions(biosphere_tensor, "Biosphere")
        log_tensor_dimensions(production_tensor, "Production")

    def _calculate_inventory_of_db(
        self, db_name: str, intermediate_flows: dict, methods: list, cutoff: float
    ) -> Tuple[dict, dict]:
        """
        Calculate the life cycle inventory for a specified background database.

        Performs an LCA for each intermediate flow exchanged with the given database
        using the specified LCIA method. Intermediate flows are mapped to resulting
        elementary flows to construct an inventory tensor. A cutoff threshold is
        applied to filter insignificant results.

        Parameters
        ----------
        db_name : str
            Name of the background database to analyze.
        intermediate_flows : dict
            Dictionary mapping intermediate flow codes to flow names.
        methods : List[tuple]
            A List of LCIA methods represented by a tuple (e.g.,
            `("EF v3.1", "climate change", "global warming potential (GWP100)")`).
        cutoff : float
            Number of top elementary flows (per intermediate flow) to retain based on
            impact magnitude. Used to reduce computational complexity.

        Returns
        -------
        inventory_tensor : dict
            Dictionary with keys as (db_name, intermediate_flow_code,
            elementary_flow_code) and values as flow amounts.
        elementary_flows : dict
            Dictionary mapping elementary flow codes to their names.
        """

        logger.info(f"Calculating inventory for database: {db_name}")
        db = bd.Database(name=db_name)
        inventory_tensor = {}
        elementary_flows = {}
        activity_cache = {}

        # Cache activity objects by looking up intermediate flows in the database
        for key in intermediate_flows.keys():
            try:
                activity_cache[key] = db.get(code=key)
            except Exception as e:  # Catch exceptions (e.g., if key is not valid)
                logger.warning(f"Failed to get activity for key '{key}': {e}")
        function_unit_dict = {activity: 1 for activity in activity_cache.values()}

        lca = bc.LCA(function_unit_dict, next(iter(methods)))
        lca.lci(factorize=len(function_unit_dict) > 10)  # factorize if many activities
        logger.info(f"Factorized LCI for database: {db_name}")
        for intermediate_flow_code, activity in tqdm(activity_cache.items()):
            # logger.info(f"Calculating inventory for activity: {activity}")
            for method in methods:
                lca.switch_method(method)
                lca.lci(demand={activity.id: 1})
                if lca.inventory.nnz == 0:
                    logger.warning(
                        f"Skipping activity {activity} as it has no non-zero inventory."
                    )
                    continue
                raw_inventory_df = lca.to_dataframe(
                    matrix_label="inventory", cutoff=cutoff
                )

                inventory_df = (
                    raw_inventory_df.groupby("row_code", as_index=False)
                    .agg({"amount": "sum"})
                    .merge(
                        raw_inventory_df[["row_code", "row_name"]].drop_duplicates(
                            "row_code"
                        ),
                        on="row_code",
                    )
                )

                # Vectorized updates to `inventory_tensor`
                inventory_tensor.update(
                    {
                        (db_name, intermediate_flow_code, elementary_flow_code): amount
                        for elementary_flow_code, amount in zip(
                            inventory_df["row_code"], inventory_df["amount"]
                        )
                    }
                )

                # Vectorized updates to `elementary_flows`
                elementary_flows.update(
                    dict(zip(inventory_df["row_code"], inventory_df["row_name"]))
                )
        logger.info(f"Finished calculating inventory for database: {db_name}")
        return inventory_tensor, elementary_flows

    def parallel_inventory_tensor_calculation(self, cutoff=1e4, n_jobs=None) -> dict:
        """
        Not yet implemented. Could improve performance significantly by parallelizing
        """
        raise NotImplementedError("This method is not yet functionally implemented.")

    def _sequential_inventory_tensor_calculation(self) -> None:
        """
        Compute the background inventory tensor for all background databases
        sequentially.

        This method performs time-explicit LCA calculations for each background
        database listed in `self.background_dbs`. For each intermediate flow in the
        foreground system, it calculates associated elementary flows using the
        configured characterization methods and applies a cutoff to retain only the
        most relevant contributions.

        The results are stored in a sparse tensor structure that maps:
            (database name, intermediate flow code, elementary flow code) → amount

        Errors during database processing are logged, and processing continues for
        remaining databases.

        Side Effects
        ------------
        Updates internal tensors and flow mappings used in downstream modeling.
            - self._background_inventory: Combined inventory tensor for all
              background databases.
            - self._elementary_flows: Updated dictionary of all observed elementary
              flows.
        """
        results = []

        # Iterate over each database in self.background_dbs sequentially
        cutoff = self.config.background_inventory.cutoff
        brightway_methods = [
            char.brightway_method for char in self.config.characterization_methods
        ]
        for db_name in self.background_dbs:
            try:
                # Directly call the _calculate_inventory_of_db method for each db
                inventory_tensor, elementary_flows = self._calculate_inventory_of_db(
                    db_name, self._intermediate_flows, brightway_methods, cutoff
                )
                # Store the result in the results list
                results.append((inventory_tensor, elementary_flows))

            except Exception as e:
                logger.error(
                    f"Error occurred while processing database {db_name}: {str(e)}",
                )
                raise

        # Combine results from all databases
        for inventory_tensor, elementary_flows in results:
            self._background_inventory.update(inventory_tensor)
            self._elementary_flows.update(elementary_flows)

    def _prepare_background_inventory(self) -> None:
        """
        Prepare the background inventory tensor, either by loading from a file or
        computing it.

        If a file path is provided in the configuration (`path_to_load`), the
        inventory tensor is loaded from that pickle file. Otherwise, it is computed
        based on the specified method (`sequential` or `parallel`). After computation
        or loading, the tensor may be saved to disk if `path_to_save` is provided.

        The background inventory tensor maps (database, intermediate flow, elementary
        flow) to amount. It updates internal state:
            - self._background_inventory
            - self._elementary_flows

        .. warning::
            Only unpickle data you trust. Loading pickle files from untrusted sources
            can be insecure.
        """
        load_path = self.config.background_inventory.path_to_load
        save_path = self.config.background_inventory.path_to_save
        method = self.config.background_inventory.calculation_method

        if load_path:
            # Load from file
            with open(load_path, "rb") as file:
                self._background_inventory = pickle.load(file)

            # Populate missing elementary flow names from biosphere database
            for _, _, ef_code in self._background_inventory.keys():
                if ef_code not in self._elementary_flows:
                    self._elementary_flows[ef_code] = self.biosphere_db.get(
                        code=ef_code
                    )["name"]
            logger.info(f"Loaded background inventory from: {load_path}")

        else:
            # Compute the background inventory
            if method == "sequential":
                self._sequential_inventory_tensor_calculation()
            elif method == "parallel":
                self.parallel_inventory_tensor_calculation()
            else:
                raise ValueError(
                    f"Unsupported background inventory calculation method: {method}"
                )
            logger.info(f"Computed background inventory using method: {method}")

            # Optionally save the computed tensor
            if save_path:
                with open(save_path, "wb") as file:
                    pickle.dump(self._background_inventory, file)
                logger.info(f"Saved background inventory to: {save_path}")

    def _construct_mapping_matrix(self) -> None:
        """
        Construct a linear interpolation-based mapping matrix between system time points
        and background databases, based on their associated reference years.

        For each year in the system timeline, this method computes interpolation weights
        for each background database based on their configured reference dates. The
        result is stored in `self._mapping`, mapping (db_name, year) tuples to
        interpolation weights.

        The weights sum to 1 for each year and are linearly interpolated between the
        closest two databases. If the year is outside the range of database reference
        years, all weight  is assigned to the nearest boundary database.

        Side Effects
        ------------
        Updates
            - `self._mapping`: dict with keys (db_name, year) and float values
        representing weights.
        """
        years = sorted(self._system_time)  # Ensure chronological order

        # Sort background DBs by year and extract mapping
        db_year_map = {db: self.background_dbs[db].year for db in self.background_dbs}
        db_names_sorted = sorted(db_year_map, key=lambda db: db_year_map[db])
        db_years_sorted = [db_year_map[db] for db in db_names_sorted]

        mapping_matrix = {}

        for year in years:
            if year <= db_years_sorted[0]:
                mapping_matrix.update({(db_names_sorted[0], year): 1.0})
            elif year >= db_years_sorted[-1]:
                mapping_matrix.update({(db_names_sorted[-1], year): 1.0})
            else:
                for i in range(len(db_years_sorted) - 1):
                    y0, y1 = db_years_sorted[i], db_years_sorted[i + 1]
                    if y0 <= year <= y1:
                        db0, db1 = db_names_sorted[i], db_names_sorted[i + 1]
                        weight1 = (year - y0) / (y1 - y0)
                        weight0 = 1.0 - weight1
                        mapping_matrix[(db0, year)] = weight0
                        mapping_matrix[(db1, year)] = weight1
                        break

        self._mapping = mapping_matrix
        logger.info(
            "Constructed mapping matrix for background databases "
            "based on linear interpolation."
        )

    def _get_mapping_weights_for_time(self, abs_time: int) -> Dict[str, float]:
        """
        Get database interpolation weights for an arbitrary absolute time.

        Uses the same logic as _construct_mapping_matrix but for any year,
        not just system times.

        Parameters
        ----------
        abs_time : int
            The absolute year to get weights for.

        Returns
        -------
        Dict[str, float]
            Mapping from database name to interpolation weight.
        """
        db_year_map = {db: self.background_dbs[db].year for db in self.background_dbs}
        db_names_sorted = sorted(db_year_map, key=lambda db: db_year_map[db])
        db_years_sorted = [db_year_map[db] for db in db_names_sorted]

        weights = {}
        if abs_time <= db_years_sorted[0]:
            weights[db_names_sorted[0]] = 1.0
        elif abs_time >= db_years_sorted[-1]:
            weights[db_names_sorted[-1]] = 1.0
        else:
            for idx in range(len(db_years_sorted) - 1):
                y0, y1 = db_years_sorted[idx], db_years_sorted[idx + 1]
                if y0 <= abs_time <= y1:
                    w1 = (abs_time - y0) / (y1 - y0)
                    w0 = 1.0 - w1
                    weights[db_names_sorted[idx]] = w0
                    weights[db_names_sorted[idx + 1]] = w1
                    break
        return weights

    def _init_temporal_traversal_cache(self) -> None:
        """
        Initialize cached data structures needed for temporal background traversal.

        Creates per-database LCA objects, reverse index lookups, and temporal
        distribution lookups. Only called once on first use.

        Side Effects
        ------------
        Populates:
            - self._bg_lca_objects: {db_name: bw2calc.LCA}
            - self._bg_reverse_activity: {db_name: {matrix_row_index: activity_code}}
            - self._bg_reverse_biosphere: {matrix_row_index: elementary_flow_code}
            - self._bg_td_lookup: {(input_code, output_code, db_name): TemporalDistribution}
        """
        if self._bg_lca_objects is not None:
            return  # Already initialized

        self._bg_lca_objects = {}
        self._bg_reverse_activity = {}
        self._bg_reverse_biosphere = {}
        self._bg_td_lookup = {}

        for db_name in self.background_dbs:
            db = bd.Database(db_name)
            activities = list(db)
            if not activities:
                continue

            # Create LCA object with a dummy demand
            fu = {activities[0]: 1}
            lca = bc.LCA(fu)
            lca.lci()
            self._bg_lca_objects[db_name] = lca

            # Build reverse mapping: matrix index -> activity code
            reverse_act = {}
            activity_dict = lca.dicts.activity
            for act in activities:
                if act.id in activity_dict:
                    matrix_idx = activity_dict[act.id]
                    reverse_act[matrix_idx] = act["code"]
            self._bg_reverse_activity[db_name] = reverse_act

            # Build biosphere reverse mapping (same across databases)
            if not self._bg_reverse_biosphere:
                biosphere_dict = lca.dicts.biosphere
                for bio_act in self.biosphere_db:
                    if bio_act.id in biosphere_dict:
                        matrix_idx = biosphere_dict[bio_act.id]
                        self._bg_reverse_biosphere[matrix_idx] = bio_act["code"]

            # Build TD lookup for exchanges in this database
            for act in activities:
                for exc in act.exchanges():
                    td = exc.get("temporal_distribution")
                    if td is not None:
                        input_code = exc.input["code"]
                        output_code = act["code"]
                        self._bg_td_lookup[(input_code, output_code, db_name)] = td

        logger.info(
            "Initialized temporal traversal cache for %d background databases.",
            len(self._bg_lca_objects),
        )

    def _temporal_inventory_of_intermediate_flow(
        self,
        intermediate_flow_code: str,
        system_time: int,
        cutoff: float = 1e-9,
        max_calc: int = 2000,
    ) -> Dict[str, float]:
        """
        Compute the elementary flow inventory for one unit of an intermediate flow
        demanded at a specific system time, using cross-database graph traversal
        with temporal distributions.

        At each node in the background supply chain, the absolute time determines
        which database's coefficients to use (via the mapping matrix). Temporal
        distributions on exchanges shift the time at which upstream demands occur.

        Parameters
        ----------
        intermediate_flow_code : str
            Code of the intermediate flow to traverse.
        system_time : int
            The absolute year at which the intermediate flow is demanded.
        cutoff : float, optional
            Minimum absolute demand amount to continue traversal (default: 1e-9).
        max_calc : int, optional
            Maximum number of heap iterations (default: 2000).

        Returns
        -------
        Dict[str, float]
            Maps elementary_flow_code to total accumulated amount.
        """
        self._init_temporal_traversal_cache()

        elementary_flows: Dict[str, float] = defaultdict(float)
        heap: list = []
        calc_count = 0

        # Initialize: push the demanded intermediate flow
        # Priority: negative absolute demand (so largest demands are processed first)
        heapq.heappush(heap, (-1.0, intermediate_flow_code, system_time, 1.0))

        while heap and calc_count < max_calc:
            neg_priority, activity_code, abs_time, demand_amount = heapq.heappop(heap)
            calc_count += 1

            if abs(demand_amount) < cutoff:
                continue

            # Get mapping weights for this absolute time
            weights = self._get_mapping_weights_for_time(abs_time)

            for db_name, weight in weights.items():
                if weight == 0 or db_name not in self._bg_lca_objects:
                    continue

                lca = self._bg_lca_objects[db_name]
                db = bd.Database(db_name)

                # Look up activity in this database
                try:
                    act = db.get(code=activity_code)
                except Exception:
                    continue

                if act.id not in lca.dicts.activity:
                    continue

                col_idx = lca.dicts.activity[act.id]

                # Extract biosphere column: B[:, j]
                bio_matrix = lca.biosphere_matrix
                bio_col = bio_matrix[:, col_idx]
                if hasattr(bio_col, 'toarray'):
                    bio_col = bio_col.toarray().ravel()
                else:
                    bio_col = np.asarray(bio_col).ravel()

                for row_idx in np.nonzero(bio_col)[0]:
                    bio_amount = bio_col[row_idx]
                    ef_code = self._bg_reverse_biosphere.get(row_idx)
                    if ef_code is None:
                        continue
                    elementary_flows[ef_code] += bio_amount * weight * demand_amount

                # Extract technosphere column: A[:, j]
                tech_matrix = lca.technosphere_matrix
                tech_col = tech_matrix[:, col_idx]
                if hasattr(tech_col, 'toarray'):
                    tech_col = tech_col.toarray().ravel()
                else:
                    tech_col = np.asarray(tech_col).ravel()

                # Diagonal element (production of this activity)
                diag = tech_col[col_idx]
                if diag == 0:
                    continue

                reverse_act = self._bg_reverse_activity[db_name]

                for row_idx in np.nonzero(tech_col)[0]:
                    if row_idx == col_idx:
                        continue  # Skip diagonal (self-production)

                    input_amount = -tech_col[row_idx] / diag  # Normalize by production
                    input_code = reverse_act.get(row_idx)
                    if input_code is None:
                        continue

                    child_demand = input_amount * weight * demand_amount

                    # Check for temporal distribution on this exchange
                    td = self._bg_td_lookup.get((input_code, activity_code, db_name))
                    if td is not None:
                        # Spread demand across time according to TD
                        years = td.date.astype("timedelta64[Y]").astype(int)
                        amounts = td.amount
                        for delta, share in zip(years, amounts):
                            child_time = abs_time + int(delta)
                            child_amount = child_demand * share
                            if abs(child_amount) >= cutoff:
                                heapq.heappush(
                                    heap,
                                    (-abs(child_amount), input_code, child_time, child_amount),
                                )
                    else:
                        # No TD: upstream demand at same time
                        if abs(child_demand) >= cutoff:
                            heapq.heappush(
                                heap,
                                (-abs(child_demand), input_code, abs_time, child_demand),
                            )

        if calc_count >= max_calc:
            logger.warning(
                "Temporal traversal for intermediate flow '%s' at t=%d hit max_calc=%d. "
                "Results may be incomplete.",
                intermediate_flow_code,
                system_time,
                max_calc,
            )

        return dict(elementary_flows)

    def _resolve_background_inventory(self) -> None:
        """
        Resolve the background inventory using temporal graph traversal.

        For each (intermediate_flow, system_time) pair, performs cross-database
        graph traversal and stores the resulting elementary flow amounts in
        self._resolved_background_inventory.

        Side Effects
        ------------
        Populates:
            - self._resolved_background_inventory: Dict[(i_code, e_code, t), float]
            - self._elementary_flows: Updated with any newly discovered flows
        """
        logger.info("Resolving background inventory with temporal graph traversal...")
        self._resolved_background_inventory = {}

        cutoff = self.config.background_inventory.cutoff
        # For temporal traversal, cutoff is a demand amount threshold, not a count
        # Use a small absolute cutoff
        traversal_cutoff = 1e-9

        total_pairs = len(self._intermediate_flows) * len(self._system_time)
        pair_count = 0

        for i_code in tqdm(
            self._intermediate_flows,
            desc="Resolving background inventory (temporal)",
        ):
            for t in self._system_time:
                pair_count += 1
                ef_amounts = self._temporal_inventory_of_intermediate_flow(
                    intermediate_flow_code=i_code,
                    system_time=t,
                    cutoff=traversal_cutoff,
                )

                for ef_code, amount in ef_amounts.items():
                    self._resolved_background_inventory[(i_code, ef_code, t)] = amount

                    # Register newly discovered elementary flows
                    if ef_code not in self._elementary_flows:
                        try:
                            self._elementary_flows[ef_code] = self.biosphere_db.get(
                                code=ef_code
                            )["name"]
                        except Exception:
                            self._elementary_flows[ef_code] = ef_code

        logger.info(
            "Resolved background inventory: %d entries for %d (flow, time) pairs.",
            len(self._resolved_background_inventory),
            pair_count,
        )

    def _construct_characterization_tensor(self) -> None:
        """
        Construct the characterization tensor for LCIA methods over system time points.

        This method computes characterization factors for elementary flows across all
        system years, supporting both static and dynamic methods. It handles metrics
        like Global Warming Potential (GWP) and Cumulative Radiative Forcing (CRF)
        when dynamic characterization is requested.

        Side Effects
        -----------
        Updates the following instance attribute:
            - self._characterization: dict mapping (method_name, elementary_flow_code,
            system_year) to characterization factor values.
        """
        start_date = self.config.temporal.start_date
        time_horizon = self.config.temporal.time_horizon
        dates = pd.date_range(
            start=start_date, periods=len(self._system_time), freq="YE"
        )
        flow_codes = list(self.elementary_flows.keys())

        # Pre-map flow codes to Brightway flow IDs
        flow_df = pd.DataFrame({"code": flow_codes})
        flow_df["flow"] = flow_df["code"].map(
            lambda code: self.biosphere_db.get(code=code).id
        )

        characterization_tensor = {}

        for config in self.config.characterization_methods:
            category_name = config.category_name
            self._category.add(category_name)
            method = config.brightway_method
            metric = config.metric

            df = flow_df.copy()
            df["amount"] = 1
            df["activity"] = np.nan

            if metric is None:
                # Static LCIA
                method_data = bd.Method(method).load()
                method_dict = {flow: value for flow, value in method_data if value != 0}

                for _, row in df.iterrows():
                    flow_code, flow_id = row["code"], row["flow"]
                    if flow_id in method_dict:
                        for year in dates.year:
                            characterization_tensor[
                                (category_name, flow_code, year)
                            ] = method_dict[flow_id]
                logger.info(
                    f"Static characterization for method {category_name} completed."
                )

            elif metric == "GWP":
                # Dynamic GWP (year-specific values)
                df = df.loc[np.repeat(df.index, len(dates))].reset_index(drop=True)
                df["date"] = np.tile(dates, len(flow_codes))
                df["date"] = df["date"].astype("datetime64[s]")

                df_char = characterize(
                    df,
                    metric="GWP",
                    fixed_time_horizon=self.config.temporal.fixed_time_horizon,
                    base_lcia_method=method,
                    time_horizon=time_horizon,
                )
                df_char["date"] = df_char["date"].dt.year

                for _, row in df_char.iterrows():
                    flow_code = df.loc[df["flow"] == row["flow"], "code"].values[0]
                    characterization_tensor[(category_name, flow_code, row["date"])] = (
                        row["amount"]
                    )
                logger.info(
                    f"Dynamic GWP characterization for {category_name} completed."
                )

            elif metric == "CRF":
                # Dynamic CRF (cumulative RF over time horizon)
                df["date"] = pd.Timestamp(self.config.temporal.start_date)

                for _, row in df.iterrows():
                    flow_code = row["code"]
                    flow_id = row["flow"]
                    df_row = row[["date", "flow", "amount", "activity"]].to_frame().T

                    df_char = characterize(
                        df_row,
                        metric="radiative_forcing",
                        fixed_time_horizon=self.config.temporal.fixed_time_horizon,
                        base_lcia_method=method,
                        time_horizon=time_horizon,
                        time_horizon_start=pd.Timestamp(start_date),
                    )
                    rf_series = df_char["amount"].values

                    for year in self.system_time:
                        cutoff = start_date.year + time_horizon - year - 1
                        cumulative_rf = rf_series[:cutoff].sum()
                        characterization_tensor[(category_name, flow_code, year)] = (
                            cumulative_rf
                        )
                logger.info(
                    f"Dynamic CRF characterization for {category_name} completed."
                )

            else:
                raise ValueError(f"Unsupported dynamic metric: {metric}")

        self._characterization.update(characterization_tensor)
