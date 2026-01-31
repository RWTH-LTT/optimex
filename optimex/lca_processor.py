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
import pickle
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
        SEASONAL: Seasonal/time-varying characterization factors provided by user
    """

    GWP = "GWP"
    CRF = "CRF"
    SEASONAL = "SEASONAL"


class TemporalResolutionEnum(str, Enum):
    """
    Supported temporal resolutions for the optimization model.

    Attributes:
        year: Annual time steps
        month: Monthly time steps
        day: Daily time steps
    """

    year = "year"
    month = "month"
    day = "day"

    @property
    def numpy_unit(self) -> str:
        """Return the numpy timedelta64/datetime64 unit code for this resolution."""
        mapping = {
            "year": "Y",
            "month": "M",
            "day": "D",
        }
        return mapping[self.value]

    @property
    def pandas_freq(self) -> str:
        """Return the pandas frequency string for date_range with this resolution."""
        mapping = {
            "year": "YE",
            "month": "ME",
            "day": "D",
        }
        return mapping[self.value]

    @property
    def pandas_offset(self) -> str:
        """Return the pandas DateOffset attribute name for this resolution."""
        mapping = {
            "year": "year",
            "month": "month",
            "day": "day",
        }
        return mapping[self.value]

    @property
    def priority(self) -> int:
        """Return priority for resolution comparison (higher = finer granularity)."""
        mapping = {
            "year": 1,
            "month": 2,
            "day": 3,
        }
        return mapping[self.value]

    @staticmethod
    def from_numpy_unit(unit: str) -> "TemporalResolutionEnum":
        """Convert numpy timedelta64 unit code to TemporalResolutionEnum."""
        mapping = {
            "Y": TemporalResolutionEnum.year,
            "M": TemporalResolutionEnum.month,
            "D": TemporalResolutionEnum.day,
        }
        if unit not in mapping:
            raise ValueError(f"Unsupported numpy unit: {unit}. Supported: Y, M, D")
        return mapping[unit]

    @staticmethod
    def get_finest(*resolutions: "TemporalResolutionEnum") -> "TemporalResolutionEnum":
        """Return the finest (most granular) resolution from a list of resolutions."""
        return max(resolutions, key=lambda r: r.priority)


# Conversion constants used throughout the module for consistency
# These are approximations - exact calendar calculations are not used for simplicity
DAYS_PER_YEAR = 365  # Approximate (ignores leap years)
DAYS_PER_MONTH = 30  # Approximate average
MONTHS_PER_YEAR = 12  # Exact


def detect_temporal_resolution(td: TemporalDistribution) -> TemporalResolutionEnum:
    """
    Detect the temporal resolution of a TemporalDistribution based on its date values.
    
    Since bw_temporalis converts all timedelta64 to seconds internally, we infer
    the resolution from the value patterns:
    - 1 year ≈ 31,556,952 seconds
    - 1 month ≈ 2,629,746 seconds
    - 1 day = 86,400 seconds
    
    Parameters
    ----------
    td : TemporalDistribution
        The temporal distribution to analyze.
    
    Returns
    -------
    TemporalResolutionEnum
        The detected resolution (year, month, or day).
    """
    if td.date.size == 0:
        return TemporalResolutionEnum.year  # Default
    
    # Constants for time conversions (in seconds)
    SECONDS_PER_DAY = 86400
    SECONDS_PER_MONTH = 2629746  # ~30.44 days
    SECONDS_PER_YEAR = 31556952  # 365.2425 days
    
    # Get the values in seconds
    values = td.date.astype('timedelta64[s]').astype(int)
    
    # If only one value (possibly 0), check the dtype string as fallback
    dtype_str = str(td.date.dtype)
    if "[Y]" in dtype_str:
        return TemporalResolutionEnum.year
    elif "[M]" in dtype_str:
        return TemporalResolutionEnum.month
    elif "[D]" in dtype_str:
        return TemporalResolutionEnum.day
    
    # Get non-zero values to analyze the step size
    non_zero = values[values != 0]
    if len(non_zero) == 0:
        # All zeros or single zero, try to infer from dtype or default
        return TemporalResolutionEnum.year
    
    # Get the minimum non-zero value or the GCD of differences
    if len(values) > 1:
        # Calculate differences between consecutive values
        diffs = np.diff(values)
        non_zero_diffs = diffs[diffs != 0]
        if len(non_zero_diffs) > 0:
            step = np.min(np.abs(non_zero_diffs))
        else:
            step = np.min(np.abs(non_zero))
    else:
        step = np.min(np.abs(non_zero))
    
    # Determine resolution based on step size
    # Allow 10% tolerance for rounding
    if step >= SECONDS_PER_YEAR * 0.9:
        return TemporalResolutionEnum.year
    elif step >= SECONDS_PER_MONTH * 0.9:
        return TemporalResolutionEnum.month
    elif step >= SECONDS_PER_DAY * 0.9:
        return TemporalResolutionEnum.day
    else:
        # Sub-day resolution not supported, default to day
        return TemporalResolutionEnum.day


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
            Supported values: 'GWP', 'CRF', 'SEASONAL'.
        seasonal_factors: Optional dict mapping month number (1-12) to characterization 
            factor multipliers. Required when metric='SEASONAL'. Applied multiplicatively 
            to base characterization factors from brightway_method.
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
        "Use None for static methods. 'SEASONAL' for user-provided time-varying factors.",
    )
    seasonal_factors: Optional[Dict[int, float]] = Field(
        None,
        description="Optional dict mapping month number (1-12) to characterization "
        "factor multipliers. Required when metric='SEASONAL'. Applied multiplicatively "
        "to base characterization factors from brightway_method.",
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

        self._parse_demand()
        self._construct_foreground_tensors()
        self._prepare_background_inventory()
        self._construct_characterization_tensor()
        self._construct_mapping_matrix()

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

    def _convert_temporal_resolution(
        self,
        time_indices: np.ndarray,
        amounts: np.ndarray,
        source_resolution: TemporalResolutionEnum,
        target_resolution: TemporalResolutionEnum,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert time indices and amounts from one temporal resolution to another.
        
        When converting from coarser to finer resolution (e.g., year to month),
        each time point is expanded. The amount is distributed uniformly across
        the expanded time points.
        
        When converting from finer to coarser resolution (e.g., month to year),
        time points are aggregated. This scenario is less common but supported.
        
        Parameters
        ----------
        time_indices : np.ndarray
            Array of time indices in source resolution.
        amounts : np.ndarray
            Array of amounts corresponding to each time index.
        source_resolution : TemporalResolutionEnum
            The resolution of the input time indices.
        target_resolution : TemporalResolutionEnum
            The desired output resolution.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Converted time indices and amounts in target resolution.
        
        Notes
        -----
        Conversion uses approximate factors defined in module constants:
        - MONTHS_PER_YEAR = 12 (exact)
        - DAYS_PER_MONTH = 30 (approximate average)
        - DAYS_PER_YEAR = 365 (approximate, ignores leap years)
        """
        if source_resolution == target_resolution:
            return time_indices, amounts
        
        # Define conversion factors using module constants
        conversion_factors = {
            (TemporalResolutionEnum.year, TemporalResolutionEnum.month): MONTHS_PER_YEAR,
            (TemporalResolutionEnum.year, TemporalResolutionEnum.day): DAYS_PER_YEAR,
            (TemporalResolutionEnum.month, TemporalResolutionEnum.day): DAYS_PER_MONTH,
            (TemporalResolutionEnum.month, TemporalResolutionEnum.year): 1 / MONTHS_PER_YEAR,
            (TemporalResolutionEnum.day, TemporalResolutionEnum.month): 1 / DAYS_PER_MONTH,
            (TemporalResolutionEnum.day, TemporalResolutionEnum.year): 1 / DAYS_PER_YEAR,
        }
        
        factor = conversion_factors.get((source_resolution, target_resolution))
        if factor is None:
            raise ValueError(
                f"Unsupported resolution conversion: {source_resolution} -> {target_resolution}"
            )
        
        if factor >= 1:
            # Expanding: coarser to finer resolution
            factor = int(factor)
            new_time_indices = []
            new_amounts = []
            
            for idx, amount in zip(time_indices, amounts):
                # Create expanded time points
                base_idx = int(idx * factor)
                expanded_indices = np.arange(base_idx, base_idx + factor)
                # Distribute amount uniformly across expanded time points
                expanded_amounts = np.full(factor, amount / factor)
                new_time_indices.extend(expanded_indices)
                new_amounts.extend(expanded_amounts)
            
            return np.array(new_time_indices), np.array(new_amounts)
        else:
            # Aggregating: finer to coarser resolution (less common)
            inverse_factor = int(1 / factor)
            aggregated = {}
            
            for idx, amount in zip(time_indices, amounts):
                target_idx = int(idx // inverse_factor)
                aggregated[target_idx] = aggregated.get(target_idx, 0) + amount
            
            return np.array(list(aggregated.keys())), np.array(list(aggregated.values()))

    def _parse_demand(self) -> None:
        """
        Parse and process the demand dictionary from the configuration.

        This method transforms the demand data into a dictionary mapping (product_code, time_index)
        tuples to their corresponding amounts. It validates that demand is specified on
        foreground product nodes.

        Side Effects
        ------------
        Updates the following instance attributes:
            - self._demand: dict with keys (product_code, time_index) and values as amounts.
            - self._products: dict mapping product codes to product names.
            - self._system_time: range of time indices covering the longest demand interval.
        
        Notes
        -----
        Time indexing schemes vary by resolution:
            - year: Uses calendar year (e.g., 2020, 2021, ...)
            - month: Uses months since year 0 (e.g., 2020*12 + 0 = 24240 for Jan 2020)
            - day: Uses days since Unix epoch (1970-01-01), consistent with numpy datetime64[D]
        """
        raw_demand = self.config.demand
        resolution = self.config.temporal.temporal_resolution
        start_date = self.config.temporal.start_date
        longest_demand_interval = 0

        # Compute the start index based on resolution
        # Year: calendar year, Month: months since year 0, Day: days since Unix epoch
        if resolution == TemporalResolutionEnum.year:
            start_index = start_date.year
        elif resolution == TemporalResolutionEnum.month:
            start_index = start_date.year * 12 + start_date.month - 1
        else:  # day - uses days since Unix epoch (1970-01-01)
            start_index = int(np.datetime64(start_date, 'D').astype(int))

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
            
            # Convert datetime64 to time indices based on resolution
            numpy_unit = resolution.numpy_unit
            if resolution == TemporalResolutionEnum.year:
                time_indices = td.date.astype(f"datetime64[{numpy_unit}]").astype(int) + 1970
            elif resolution == TemporalResolutionEnum.month:
                # Convert to months since epoch, then add offset to get months since year 0
                dates_as_months = td.date.astype("datetime64[M]").astype(int)
                time_indices = dates_as_months + 1970 * 12
            else:  # day - uses days since Unix epoch, consistent with numpy datetime64[D]
                time_indices = td.date.astype("datetime64[D]").astype(int)
            
            if time_indices[-1] - start_index > longest_demand_interval:
                longest_demand_interval = time_indices[-1] - start_index
            amounts = td.amount

            self._demand.update(
                {(product_code, idx): amount for idx, amount in zip(time_indices, amounts)}
            )

            # Store product information
            self._products[product_code] = product_node['name']

        self._system_time = range(start_index, start_index + longest_demand_interval + 1)
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

        Mixed temporal resolutions are supported: exchanges can have different temporal
        resolutions (year, month, day), and the system will convert them to the configured
        resolution. When an exchange uses a coarser resolution than configured, its values
        are expanded to the finer resolution.

        Side Effects
        -----------
        Updates the following instance attributes:
            - self._foreground_technosphere: dict mapping (process_code, flow_code, time_index)
              to amount for external intermediate flows (background consumption).
            - self._internal_demand_technosphere: dict mapping (process_code, product_code, time_index)
              to amount for internal product consumption (foreground products).
            - self._foreground_biosphere: dict mapping (process_code, flow_code, time_index)
              to amount for biosphere flows (emissions).
            - self._foreground_production: dict mapping (process_code, product_code, time_index)
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
        """
        technosphere_tensor = {}
        internal_demand_technosphere = {}
        production_tensor = {}
        biosphere_tensor = {}
        
        target_resolution = self.config.temporal.temporal_resolution
        target_numpy_unit = target_resolution.numpy_unit

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
                        date=np.array([0], dtype=f"timedelta64[{target_numpy_unit}]"), amount=np.array([1])
                    ),
                )
                
                # Detect the resolution of this temporal distribution
                source_resolution = detect_temporal_resolution(temporal_dist)
                source_numpy_unit = source_resolution.numpy_unit
                
                # Convert timedelta to process time indices
                # First convert to source resolution, then to target if needed
                raw_time_indices = temporal_dist.date.astype(f"timedelta64[{source_numpy_unit}]").astype(int)
                temporal_factor = temporal_dist.amount
                
                # Convert to target resolution if different
                if source_resolution != target_resolution:
                    time_indices, temporal_factor = self._convert_temporal_resolution(
                        raw_time_indices, temporal_factor, source_resolution, target_resolution
                    )
                else:
                    time_indices = raw_time_indices
                
                # Ensure all time indices are included in process time
                self._process_time.update(
                    idx for idx in time_indices if idx not in self._process_time
                )

                # Skip if temporal distribution is missing or invalid (empty arrays)
                if time_indices.size == 0 or temporal_factor.size == 0:
                    logger.debug(
                        f"Skipping exchange {exc.input} due to missing or invalid temporal distribution.")
                    continue

                edge_type = exc["type"]
                input_code = exc.input["code"]
                input_name = exc.input["name"]
                input_db = exc.input["database"]

                # Handle production edges
                if edge_type == bd.labels.production_edge_default:
                    product_code = input_code
                    production_tensor.update({
                        (act["code"], product_code, idx): exc["amount"] * factor
                        for idx, factor in zip(time_indices, temporal_factor)
                    })
                    if exc.get("operation"):
                        self._operation_flow.update({(act["code"], product_code): True})
                    self._products.setdefault(product_code, input_name)

                # Handle consumption edges
                elif edge_type == bd.labels.consumption_edge_default:
                    if input_db == self.foreground_db.name:
                        # Internal demand: foreground product consumed
                        internal_demand_technosphere.update({
                            (act["code"], input_code, idx): exc["amount"] * factor
                            for idx, factor in zip(time_indices, temporal_factor)
                        })
                        if exc.get("operation"):
                            self._operation_flow.update({(act["code"], input_code): True})
                        self._products.setdefault(input_code, input_name)
                    else:
                        # External intermediate: background consumption
                        technosphere_tensor.update({
                            (act["code"], input_code, idx): exc["amount"] * factor
                            for idx, factor in zip(time_indices, temporal_factor)
                        })
                        if exc.get("operation"):
                            self._operation_flow.update({(act["code"], input_code): True})
                        self._intermediate_flows.setdefault(input_code, input_name)

                # Handle biosphere edges
                elif edge_type == bd.labels.biosphere_edge_default:
                    biosphere_tensor.update({
                        (act["code"], input_code, idx): exc["amount"] * factor
                        for idx, factor in zip(time_indices, temporal_factor)
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
            time_points = {k[2] for k in tensor}
            logger.info(
                f"{name} shape: ({len(processes)} processes, {len(flows)} flows, "
                f"{len(time_points)} time points) with {len(tensor)} total entries."
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

    def _construct_characterization_tensor(self) -> None:
        """
        Construct the characterization tensor for LCIA methods over system time points.

        This method computes characterization factors for elementary flows across all
        system time points, supporting both static and dynamic methods. It handles metrics
        like Global Warming Potential (GWP) and Cumulative Radiative Forcing (CRF)
        when dynamic characterization is requested.

        Side Effects
        -----------
        Updates the following instance attribute:
            - self._characterization: dict mapping (method_name, elementary_flow_code,
            system_time_index) to characterization factor values.
        """
        start_date = self.config.temporal.start_date
        time_horizon = self.config.temporal.time_horizon
        resolution = self.config.temporal.temporal_resolution
        pandas_freq = resolution.pandas_freq
        
        dates = pd.date_range(
            start=start_date, periods=len(self._system_time), freq=pandas_freq
        )
        
        # Create mapping from system time indices to dates
        system_time_list = sorted(self._system_time)
        time_index_to_date = dict(zip(system_time_list, dates))
        
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
                        for time_idx in system_time_list:
                            characterization_tensor[
                                (category_name, flow_code, time_idx)
                            ] = method_dict[flow_id]
                logger.info(
                    f"Static characterization for method {category_name} completed."
                )

            elif metric == "GWP":
                # Dynamic GWP (time-specific values)
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
                
                # Map dates back to system time indices based on resolution
                if resolution == TemporalResolutionEnum.year:
                    df_char["time_idx"] = df_char["date"].dt.year
                elif resolution == TemporalResolutionEnum.month:
                    df_char["time_idx"] = df_char["date"].dt.year * 12 + df_char["date"].dt.month - 1
                else:  # day
                    df_char["time_idx"] = (df_char["date"] - pd.Timestamp("1970-01-01")).dt.days

                for _, row in df_char.iterrows():
                    flow_code = df.loc[df["flow"] == row["flow"], "code"].values[0]
                    characterization_tensor[(category_name, flow_code, row["time_idx"])] = (
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

                    for time_idx in system_time_list:
                        # Compute cutoff based on resolution
                        # The cutoff determines how many years of radiative forcing to sum
                        if resolution == TemporalResolutionEnum.year:
                            cutoff = start_date.year + time_horizon - time_idx - 1
                        elif resolution == TemporalResolutionEnum.month:
                            # Convert time_idx to elapsed years for cutoff calculation
                            start_month_idx = start_date.year * 12 + start_date.month - 1
                            months_elapsed = time_idx - start_month_idx
                            # Use integer division to avoid floating-point precision issues
                            years_elapsed = months_elapsed // 12
                            cutoff = time_horizon - years_elapsed - 1
                        else:  # day
                            # Note: 365.25 is used as an approximation for average days per year
                            # This may accumulate small errors over very long time horizons
                            start_day_idx = int(np.datetime64(start_date, 'D').astype(int))
                            days_elapsed = time_idx - start_day_idx
                            years_elapsed = days_elapsed // 365  # Use integer division
                            cutoff = time_horizon - years_elapsed - 1
                        
                        cutoff = max(0, cutoff)  # Ensure non-negative
                        cumulative_rf = rf_series[:cutoff].sum()
                        characterization_tensor[(category_name, flow_code, time_idx)] = (
                            cumulative_rf
                        )
                logger.info(
                    f"Dynamic CRF characterization for {category_name} completed."
                )

            elif metric == MetricEnum.SEASONAL:
                # Seasonal/time-varying characterization factors
                # Get base characterization factors from the method
                method_data = bd.Method(method).load()
                method_dict = {flow: value for flow, value in method_data if value != 0}
                
                # Get seasonal factors from config
                seasonal_factors = config.seasonal_factors
                if seasonal_factors is None:
                    raise ValueError(
                        f"seasonal_factors must be provided when metric='SEASONAL' "
                        f"for category '{category_name}'"
                    )
                
                for _, row in df.iterrows():
                    flow_code, flow_id = row["code"], row["flow"]
                    if flow_id in method_dict:
                        base_cf = method_dict[flow_id]
                        
                        for time_idx in system_time_list:
                            # Get the corresponding date for this time index
                            date = time_index_to_date.get(time_idx)
                            if date is not None:
                                # Extract month (1-12) from the date
                                month = date.month
                                # Get seasonal factor, default to 1.0 if not specified
                                seasonal_factor = seasonal_factors.get(month, 1.0)
                                cf = base_cf * seasonal_factor
                            else:
                                cf = base_cf
                            
                            characterization_tensor[
                                (category_name, flow_code, time_idx)
                            ] = cf
                
                logger.info(
                    f"Seasonal characterization for {category_name} completed "
                    f"with {len(seasonal_factors)} monthly factors."
                )

            else:
                raise ValueError(f"Unsupported dynamic metric: {metric}")

        self._characterization.update(characterization_tensor)
