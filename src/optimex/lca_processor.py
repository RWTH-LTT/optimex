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

import hashlib
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import bw2calc as bc
import bw2data as bd
import numpy as np
import pandas as pd
from bw_temporalis import TemporalDistribution, easy_timedelta_distribution
from dynamic_characterization import (
    characterize,
    create_characterization_functions_from_method,
)
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

# Module-level caches, shared by all `LCADataProcessor` instances in a session.
#
# Background inventories are expensive to compute mostly because building and
# factorizing the technosphere matrix of a full background database takes tens of
# seconds. Keys carry the database's `modified` token, so editing a database
# invalidates its entries automatically instead of silently reusing stale ones.
_BACKGROUND_INVENTORY_CACHE = {}

# {(project, biosphere db, modified): {flow id: (code, name)}}
_BIOSPHERE_METADATA_CACHE = {}

# {(project, db, modified): {name: [node, ...]}} for identity-based lookups
_NODE_INDEX_CACHE = {}

# {(project, method tuple): {flow id: dynamic characterization function}}
_CHARACTERIZATION_FUNCTION_CACHE = {}


def clear_lca_caches(include_disk: bool = False, cache_dir=None) -> None:
    """
    Clear the module-level background inventory and metadata caches.

    Parameters
    ----------
    include_disk : bool, optional
        Also delete the on-disk inventory cache of the current project.
    cache_dir : str or Path, optional
        Directory of the on-disk cache, if it is not in the default location.
    """
    _BACKGROUND_INVENTORY_CACHE.clear()
    _BIOSPHERE_METADATA_CACHE.clear()
    _NODE_INDEX_CACHE.clear()
    _CHARACTERIZATION_FUNCTION_CACHE.clear()

    if include_disk:
        directory = Path(
            cache_dir
            if cache_dir is not None
            else Path(bd.projects.dir) / "optimex-inventory-cache"
        )
        for path in directory.glob("*.pickle"):
            path.unlink(missing_ok=True)


def _cache_token(db_name: str, cutoff: Optional[float] = None) -> tuple:
    """Cache key prefix that invalidates itself when the database is edited."""
    return (
        bd.projects.current,
        db_name,
        bd.Database(name=db_name).metadata.get("modified"),
        cutoff,
    )


def _disk_cache_path(db_name: str, cutoff: Optional[float], cache_dir=None):
    """
    File holding the cached inventories of one background database.

    The `modified` token of the database and the cutoff are part of the file
    name, so an edited database or a different cutoff simply misses instead of
    reading stale numbers.
    """
    if cache_dir is None:
        cache_dir = Path(bd.projects.dir) / "optimex-inventory-cache"
    cache_dir = Path(cache_dir)
    token = _cache_token(db_name, cutoff)
    digest = hashlib.sha256(repr(token).encode()).hexdigest()[:16]
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in db_name)
    return cache_dir / f"{stem[:80]}.{digest}.pickle"


def _load_disk_cache(db_name: str, cutoff: Optional[float], cache_dir=None) -> int:
    """
    Populate the in-memory cache from disk. Returns the number of entries read.

    .. warning::
        The cache is read with `pickle`. It is written by optimex itself inside
        the Brightway project directory; point `disk_cache_dir` somewhere else
        only if you trust its contents.
    """
    path = _disk_cache_path(db_name, cutoff, cache_dir)
    if not path.is_file():
        return 0
    try:
        with open(path, "rb") as file:
            entries = pickle.load(file)
    except Exception as e:  # A truncated or unreadable cache is not fatal.
        logger.warning(f"Ignoring unreadable inventory cache {path}: {e}")
        return 0

    cache_token = _cache_token(db_name, cutoff)
    _BACKGROUND_INVENTORY_CACHE.update(
        {cache_token + identity: entry for identity, entry in entries.items()}
    )
    logger.info(f"Loaded {len(entries)} cached inventories from {path}")
    return len(entries)


def _store_disk_cache(
    db_name: str, cutoff: Optional[float], entries: dict, cache_dir=None
) -> None:
    """Merge `entries` into the on-disk cache of a database, atomically."""
    if not entries:
        return
    path = _disk_cache_path(db_name, cutoff, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    stored = {}
    if path.is_file():
        try:
            with open(path, "rb") as file:
                stored = pickle.load(file)
        except Exception:  # Overwrite a cache we cannot read.
            stored = {}
    stored.update(entries)

    temporary = path.with_suffix(".pickle.tmp")
    with open(temporary, "wb") as file:
        pickle.dump(stored, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)

    # Drop caches of older versions of the same database.
    for other in path.parent.glob(f"{path.name.split('.')[0]}.*.pickle"):
        if other != path:
            other.unlink(missing_ok=True)

    logger.info(f"Cached {len(entries)} inventories in {path}")


def _flow_identity(key: str, meta: Union[dict, str]) -> tuple:
    """
    Identity of an intermediate flow, independent of the background database.

    premise assigns a different code to the same activity in each scenario
    database, so activities are resolved by (name, reference product, location).
    Legacy inputs (e.g. old pickles) carry no metadata and fall back to the code.
    """
    if isinstance(meta, dict):
        return (meta["name"], meta.get("reference product"), meta.get("location"))
    return ("__code__", key, None)


def _biosphere_metadata(biosphere_db_name: str) -> dict:
    """
    Map biosphere flow ids to their (code, name), loading the database once.

    Returns
    -------
    dict
        ``{flow id: (code, name)}`` for every node in the biosphere database.
    """
    db = bd.Database(name=biosphere_db_name)
    key = (bd.projects.current, biosphere_db_name, db.metadata.get("modified"))
    if key not in _BIOSPHERE_METADATA_CACHE:
        _BIOSPHERE_METADATA_CACHE[key] = {
            node["id"]: (node["code"], node["name"]) for node in db
        }
    return _BIOSPHERE_METADATA_CACHE[key]


def _node_index(db_name: str) -> dict:
    """
    Index a database's nodes by name, loading it in a single pass.

    `bd.get_node` costs a few hundred ms per call, while iterating a full
    background database takes about a second, so identity lookups are served from
    this index instead.

    Returns
    -------
    dict
        ``{node name: [node, ...]}`` for every node in the database.
    """
    db = bd.Database(name=db_name)
    key = (bd.projects.current, db_name, db.metadata.get("modified"))
    if key not in _NODE_INDEX_CACHE:
        index = {}
        for node in db:
            index.setdefault(node["name"], []).append(node)
        _NODE_INDEX_CACHE[key] = index
    return _NODE_INDEX_CACHE[key]


def _resolve_node(db_name: str, meta: dict):
    """
    Resolve an activity in `db_name` by (name, reference product, location).

    Mirrors `bd.get_node` semantics: fields left out of `meta` are not filtered on,
    and an ambiguous identity is an error.
    """
    candidates = _node_index(db_name).get(meta["name"], [])
    for field in ("reference product", "location"):
        if meta.get(field) is None:
            continue
        candidates = [node for node in candidates if node.get(field) == meta[field]]
    if not candidates:
        raise KeyError(f"No node found for {meta!r} in '{db_name}'")
    if len(candidates) > 1:
        raise ValueError(f"Multiple nodes found for {meta!r} in '{db_name}'")
    return candidates[0]


def compute_db_inventory_entries(
    db_name: str,
    intermediate_flows: dict,
    cutoff: Optional[float] = None,
    biosphere_db_name: Optional[str] = None,
    project: Optional[str] = None,
    base_dirs: Optional[Tuple[str, str]] = None,
) -> dict:
    """
    Compute aggregated background inventories for the given intermediate flows.

    All flows are solved against a single factorization of the database's
    technosphere matrix. For an intermediate flow :math:`j` with unit demand, the
    aggregated elementary flow vector is :math:`g_j = B x_j`, i.e. the column of
    :math:`B A^{-1}` belonging to that flow. The per-background-process breakdown
    that `LCA.lci()` builds (B times diag(x_j)) is never needed here and is skipped,
    since only the aggregate enters the optimization.

    This is a module-level function so that it can also run in a worker process.

    Parameters
    ----------
    db_name : str
        Name of the background database to analyze.
    intermediate_flows : dict
        Dictionary mapping intermediate flow codes (foreground reference codes) to
        identity metadata dicts with keys "name", "reference product", and
        "location".
    cutoff : float, optional
        If given, keep only the ``cutoff`` largest elementary flows (by absolute
        amount) per intermediate flow. Default ``None`` keeps every non-zero flow,
        since a small flow can still carry a large characterized impact.
    biosphere_db_name : str, optional
        Biosphere database to read flow codes and names from. Defaults to the
        project's configured biosphere database.
    project : str, optional
        Brightway project to activate first. Needed when running in a worker
        process, which starts without an active project.
    base_dirs : tuple of str, optional
        ``(data directory, logs directory)`` of the Brightway installation, for
        worker processes that would otherwise fall back to the default location.

    Returns
    -------
    dict
        ``{flow identity: {elementary flow code: (name, amount)}}``.
    """
    if base_dirs is not None and str(bd.projects._base_data_dir) != base_dirs[0]:
        bd.projects.change_base_directories(
            Path(base_dirs[0]), Path(base_dirs[1]), project_name=project
        )
    elif project is not None and bd.projects.current != project:
        bd.projects.set_current(project)
    if biosphere_db_name is None:
        biosphere_db_name = bd.config.biosphere

    logger.info(f"Calculating inventory for database: {db_name}")
    db = bd.Database(name=db_name)

    activities = {}
    for key, meta in intermediate_flows.items():
        try:
            if isinstance(meta, dict):
                activities[key] = _resolve_node(db_name, meta)
            else:
                activities[key] = db.get(code=key)
        except Exception as e:  # Catch exceptions (e.g., if activity not found)
            logger.warning(
                f"Failed to resolve intermediate flow {meta!r} (code '{key}') "
                f"in '{db_name}': {e}"
            )

    if not activities:
        return {}

    # No LCIA method is needed: the inventory does not depend on it, and the
    # characterization factors are applied later, per system year.
    lca = bc.LCA({activity: 1 for activity in activities.values()})
    lca.lci(factorize=len(activities) > 1)
    logger.info(f"Built and factorized technosphere matrix for: {db_name}")

    bio_meta = _biosphere_metadata(biosphere_db_name)
    reversed_biosphere = lca.dicts.biosphere.reversed
    row_codes = []
    row_names = []
    for row in range(lca.biosphere_matrix.shape[0]):
        flow_id = reversed_biosphere[row]
        if flow_id in bio_meta:
            code, name = bio_meta[flow_id]
        else:
            node = bd.get_node(id=flow_id)
            code, name = node["code"], node["name"]
        row_codes.append(code)
        row_names.append(name)

    entries = {}
    for key, activity in tqdm(activities.items()):
        # `lci()` is bypassed on purpose: it would build the full
        # (elementary flow x background process) inventory matrix, of which only
        # the row sums are used below.
        lca.build_demand_array({activity.id: 1})
        aggregated = lca.biosphere_matrix @ lca.solve_linear_system()

        rows = np.flatnonzero(aggregated)
        if cutoff is not None and len(rows) > int(cutoff):
            largest = np.argpartition(np.abs(aggregated[rows]), -int(cutoff))
            rows = rows[largest[-int(cutoff) :]]

        if not len(rows):
            logger.warning(
                f"Activity {activity} has no non-zero inventory in '{db_name}'."
            )

        entries[_flow_identity(key, intermediate_flows[key])] = {
            row_codes[row]: (row_names[row], float(aggregated[row])) for row in rows
        }

    logger.info(f"Finished calculating inventory for database: {db_name}")
    return entries


def _assemble_inventory_tensor(
    db_name: str, intermediate_flows: dict, cutoff: Optional[float] = None
) -> Tuple[dict, dict]:
    """
    Build the inventory tensor of a database from the cached per-flow inventories.

    Returns
    -------
    inventory_tensor : dict
        ``{(db_name, intermediate flow code, elementary flow code): amount}``.
    elementary_flows : dict
        ``{elementary flow code: name}``.
    """
    cache_token = _cache_token(db_name, cutoff)
    inventory_tensor = {}
    elementary_flows = {}
    for key, meta in intermediate_flows.items():
        entry = _BACKGROUND_INVENTORY_CACHE.get(cache_token + _flow_identity(key, meta))
        if entry is None:
            continue
        for ef_code, (ef_name, amount) in entry.items():
            inventory_tensor[(db_name, key, ef_code)] = amount
            elementary_flows[ef_code] = ef_name
    return inventory_tensor, elementary_flows


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
        cutoff: Optional number of top elementary flows to retain per intermediate flow, ranked by absolute inventory amount. `None` (default) keeps all non-zero flows.
        restrict_to_characterized_flows: Drop elementary flows without a characterization factor in any category.
        retain_flows: Elementary flow codes to keep regardless of characterization.
        calculation_method: Method for calculating the inventory tensor. Options: 'sequential', 'parallel'.
        n_jobs: Number of worker processes used by the 'parallel' method.
        use_disk_cache: Whether calculated inventories are cached on disk between sessions.
        disk_cache_dir: Directory for the on-disk cache; defaults to a folder in the Brightway project.
        path_to_save: Optional path to save the inventory tensor.
        path_to_load: Optional path to load the inventory tensor.
    """

    cutoff: Optional[float] = Field(
        None,
        description="Optional number of top elementary flows to retain per "
        "intermediate flow, ranked by absolute inventory amount. Default `None` "
        "keeps every non-zero flow: a small flow can still carry a large "
        "characterized impact, and dropping it would silently bias the result.",
    )
    restrict_to_characterized_flows: bool = Field(
        True,
        description="Drop elementary flows that have no characterization factor in "
        "any configured category. Such flows contribute exactly zero impact, so "
        "removing them only shrinks the optimization model. Set to False (or list "
        "the flow in `retain_flows`) when a flow is needed for a flow limit.",
    )
    retain_flows: List[str] = Field(
        default_factory=list,
        description="Codes of elementary flows to keep even when they have no "
        "characterization factor, e.g. flows used in flow limit constraints.",
    )
    calculation_method: str = Field(
        "parallel",
        description="Method for calculating the inventory tensor. Options: "
        "'parallel' (default; one worker process per background database) and "
        "'sequential'. Scripts using 'parallel' must guard their entry point with "
        '`if __name__ == "__main__":`; notebooks need no guard.',
    )
    n_jobs: Optional[int] = Field(
        None,
        description="Number of worker processes for the 'parallel' calculation "
        "method. Defaults to one per background database, capped by the CPU count.",
    )
    use_disk_cache: bool = Field(
        True,
        description="Cache calculated background inventories on disk, so that a "
        "new session does not have to rebuild and factorize the technosphere "
        "matrix of every background database again. Entries are keyed by the "
        "database's `modified` token, so editing a database invalidates them.",
    )
    disk_cache_dir: Optional[str] = Field(
        None,
        description="Directory for the on-disk inventory cache. Defaults to "
        "`optimex-inventory-cache` inside the current Brightway project.",
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
        foreground_db_name: Name of the foreground Brightway database.
    """

    demand: Dict[bd.backends.proxies.Activity, TemporalDistribution]
    temporal: TemporalConfig
    characterization_methods: List[CharacterizationMethodConfig]
    background_inventory: Optional[BackgroundInventoryConfig] = Field(
        default_factory=BackgroundInventoryConfig
    )
    foreground_db_name: str = Field(
        "foreground",
        description="Name of the foreground Brightway database.",
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

    def __init__(
        self, config: LCAConfig, foreground_db_name: Optional[str] = None
    ) -> None:
        """
        Initialize the LCADataProcessor with the LCA configuration.

        Parameters
        ----------
        config : LCAConfig
            The configuration object containing all settings for demand,
            temporal parameters, characterization methods, and background inventory.
        foreground_db_name : str, optional
            The name of the foreground Brightway database. Defaults to
            `config.foreground_db_name`, which is itself "foreground" unless set.
            Passing a name here overrides the one on the config.
        """
        self.config = config
        if foreground_db_name is None:
            foreground_db_name = config.foreground_db_name
        if foreground_db_name not in bd.databases:
            raise ValueError(
                f"Foreground database '{foreground_db_name}' is not defined."
            )
        self.foreground_db = bd.Database(foreground_db_name)
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

        self._parse_demand()
        self._construct_foreground_tensors()
        self._prepare_background_inventory()
        self._construct_characterization_tensor()
        self._prune_uncharacterized_flows()
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

    @property
    def foreground_technosphere_vintages(self) -> Optional[dict]:
        """Read-only access to vintage-specific technosphere values."""
        return (
            self._foreground_technosphere_vintages
            if self._foreground_technosphere_vintages
            else None
        )

    @property
    def foreground_biosphere_vintages(self) -> Optional[dict]:
        """Read-only access to vintage-specific biosphere values."""
        return (
            self._foreground_biosphere_vintages
            if self._foreground_biosphere_vintages
            else None
        )

    @property
    def foreground_production_vintages(self) -> Optional[dict]:
        """Read-only access to vintage-specific production values."""
        return (
            self._foreground_production_vintages
            if self._foreground_production_vintages
            else None
        )

    @property
    def vintage_improvements(self) -> Optional[dict]:
        """Read-only access to vintage improvement scaling factors."""
        return self._vintage_improvements if self._vintage_improvements else None

    @property
    def reference_vintages(self) -> Optional[list]:
        """Read-only access to reference vintage years."""
        return (
            sorted(list(self._reference_vintages)) if self._reference_vintages else None
        )

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
            if not hasattr(product_node, "key"):
                raise ValueError(
                    f"Demand must be on Brightway Node objects, got {type(product_node)}"
                )

            if product_node.get("type") != bd.labels.product_node_default:
                raise ValueError(
                    f"Demand must be on product nodes. "
                    f"Node {product_node['name']} has type {product_node.get('type')}"
                )

            product_code = product_node["code"]
            years = td.date.astype("datetime64[Y]").astype(int) + 1970
            if years[-1] - start_year > longest_demand_interval:
                longest_demand_interval = years[-1] - start_year
            amounts = td.amount

            self._demand.update(
                {(product_code, year): amount for year, amount in zip(years, amounts)}
            )

            # Store product information
            self._products[product_code] = product_node["name"]

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
            if act.get("type") != bd.labels.process_node_default:
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
                        f"Skipping exchange {exc.input} due to missing or invalid temporal distribution."
                    )
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
                        for (
                            vintage_year,
                            scaling_factor,
                        ) in vintage_improvements.items():
                            self._reference_vintages.add(vintage_year)
                            self._vintage_improvements[
                                (act["code"], input_code, vintage_year)
                            ] = scaling_factor

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
                                process_time_vintage = (
                                    None  # Will be expanded for all years
                                )
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
                                    self._foreground_production_vintages[
                                        (act["code"], input_code, tau, vintage_year)
                                    ] = vintage_amount
                                elif edge_type == bd.labels.consumption_edge_default:
                                    if input_db != self.foreground_db.name:
                                        # Only for background consumption (technosphere)
                                        self._foreground_technosphere_vintages[
                                            (act["code"], input_code, tau, vintage_year)
                                        ] = vintage_amount
                                elif edge_type == bd.labels.biosphere_edge_default:
                                    self._foreground_biosphere_vintages[
                                        (act["code"], input_code, tau, vintage_year)
                                    ] = vintage_amount

                # Handle production edges
                if edge_type == bd.labels.production_edge_default:
                    product_code = input_code
                    production_tensor.update(
                        {
                            (act["code"], product_code, year): exc["amount"] * factor
                            for year, factor in zip(years, temporal_factor)
                        }
                    )
                    if exc.get("operation"):
                        self._operation_flow.update({(act["code"], product_code): True})
                    self._products.setdefault(product_code, input_name)

                # Handle consumption edges
                elif edge_type == bd.labels.consumption_edge_default:
                    if input_db == self.foreground_db.name:
                        # Internal demand: foreground product consumed
                        internal_demand_technosphere.update(
                            {
                                (act["code"], input_code, year): exc["amount"] * factor
                                for year, factor in zip(years, temporal_factor)
                            }
                        )
                        if exc.get("operation"):
                            self._operation_flow.update(
                                {(act["code"], input_code): True}
                            )
                        self._products.setdefault(input_code, input_name)
                    else:
                        # External intermediate: background consumption
                        technosphere_tensor.update(
                            {
                                (act["code"], input_code, year): exc["amount"] * factor
                                for year, factor in zip(years, temporal_factor)
                            }
                        )
                        if exc.get("operation"):
                            self._operation_flow.update(
                                {(act["code"], input_code): True}
                            )
                        # Store identity attributes, not just the code: premise assigns a
                        # different code to the same activity in each scenario database, so
                        # background activities are resolved across databases by
                        # (name, reference product, location), not by code.
                        self._intermediate_flows.setdefault(
                            input_code,
                            {
                                "name": input_name,
                                "reference product": exc.input.get("reference product"),
                                "location": exc.input.get("location"),
                            },
                        )

                # Handle biosphere edges
                elif edge_type == bd.labels.biosphere_edge_default:
                    biosphere_tensor.update(
                        {
                            (act["code"], input_code, year): exc["amount"] * factor
                            for year, factor in zip(years, temporal_factor)
                        }
                    )
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

    def _disk_cache_dir(self):
        """
        Where the on-disk inventory cache lives.

        Returns `False` when caching is switched off, `None` for the default
        location inside the Brightway project, or a configured directory.
        """
        config = self.config.background_inventory
        if not config.use_disk_cache:
            return False
        return config.disk_cache_dir

    def _pending_flows(
        self, db_name: str, intermediate_flows: dict, cutoff: Optional[float]
    ) -> dict:
        """
        Intermediate flows of a database that still have to be calculated.

        Reads the on-disk cache into memory on the first miss, so a fresh session
        skips rebuilding and factorizing the technosphere matrix.
        """
        cache_token = _cache_token(db_name, cutoff)

        def missing():
            return {
                key: meta
                for key, meta in intermediate_flows.items()
                if cache_token + _flow_identity(key, meta)
                not in _BACKGROUND_INVENTORY_CACHE
            }

        pending = missing()
        cache_dir = self._disk_cache_dir()
        if pending and cache_dir is not False:
            if _load_disk_cache(db_name, cutoff, cache_dir):
                pending = missing()
        return pending

    def _store_on_disk(
        self, db_name: str, cutoff: Optional[float], entries: dict
    ) -> None:
        """Write freshly calculated inventories to the on-disk cache."""
        cache_dir = self._disk_cache_dir()
        if cache_dir is not False:
            _store_disk_cache(db_name, cutoff, entries, cache_dir)

    def _calculate_inventory_of_db(
        self, db_name: str, intermediate_flows: dict, cutoff: Optional[float] = None
    ) -> Tuple[dict, dict]:
        """
        Calculate the life cycle inventory for a specified background database.

        See `compute_db_inventory_entries` for the actual calculation. This wrapper
        serves cached databases from memory and assembles the tensor.

        Parameters
        ----------
        db_name : str
            Name of the background database to analyze.
        intermediate_flows : dict
            Dictionary mapping intermediate flow codes (foreground reference codes)
            to identity metadata dicts with keys "name", "reference product", and
            "location", used to resolve the activity in each background database.
        cutoff : float, optional
            If given, keep only the ``cutoff`` largest elementary flows (by absolute
            amount) per intermediate flow. Default ``None`` keeps every non-zero
            flow, since a small flow can still carry a large characterized impact.

        Returns
        -------
        inventory_tensor : dict
            Dictionary with keys as (db_name, intermediate_flow_code,
            elementary_flow_code) and values as flow amounts.
        elementary_flows : dict
            Dictionary mapping elementary flow codes to their names.
        """
        cache_token = _cache_token(db_name, cutoff)
        pending = self._pending_flows(db_name, intermediate_flows, cutoff)

        if pending:
            entries = compute_db_inventory_entries(
                db_name,
                pending,
                cutoff=cutoff,
                biosphere_db_name=self.biosphere_db.name,
            )
            _BACKGROUND_INVENTORY_CACHE.update(
                {cache_token + identity: entry for identity, entry in entries.items()}
            )
            self._store_on_disk(db_name, cutoff, entries)
        else:
            logger.info(f"Reused cached inventory for database: {db_name}")

        return _assemble_inventory_tensor(db_name, intermediate_flows, cutoff)

    def parallel_inventory_tensor_calculation(
        self, n_jobs: Optional[int] = None
    ) -> None:
        """
        Compute the background inventory tensor for all background databases in
        parallel, one process per database.

        Each database needs its own technosphere matrix built and factorized, which
        is the bulk of the work and is independent between databases. Results are
        merged into the module-level cache of the parent process, so a rerun in the
        same session is served from memory.

        Worker processes are spawned, so a plain script calling this must guard its
        entry point with ``if __name__ == "__main__":``. Notebooks need no guard.

        Parameters
        ----------
        n_jobs : int, optional
            Number of worker processes. Defaults to one per background database,
            capped by the CPU count.

        Side Effects
        ------------
            - self._background_inventory: Combined inventory tensor for all
              background databases.
            - self._elementary_flows: Updated dictionary of all observed elementary
              flows.
        """
        cutoff = self.config.background_inventory.cutoff
        project = bd.projects.current
        biosphere_db_name = self.biosphere_db.name
        base_dirs = (
            str(bd.projects._base_data_dir),
            str(bd.projects._base_logs_dir),
        )

        pending_per_db = {}
        for db_name in self.background_dbs:
            pending = self._pending_flows(db_name, self._intermediate_flows, cutoff)
            if pending:
                pending_per_db[db_name] = pending

        if len(pending_per_db) == 1:
            # A single database gains nothing from a worker process, and staying
            # in-process avoids the spawn requirements entirely.
            db_name, pending = next(iter(pending_per_db.items()))
            entries = compute_db_inventory_entries(
                db_name, pending, cutoff, biosphere_db_name
            )
            cache_token = _cache_token(db_name, cutoff)
            _BACKGROUND_INVENTORY_CACHE.update(
                {cache_token + identity: entry for identity, entry in entries.items()}
            )
            self._store_on_disk(db_name, cutoff, entries)
        elif pending_per_db:
            n_jobs = min(
                n_jobs or len(pending_per_db),
                len(pending_per_db),
                os.cpu_count() or 1,
            )
            logger.info(
                f"Calculating inventories of {len(pending_per_db)} databases "
                f"in {n_jobs} processes."
            )
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(
                        compute_db_inventory_entries,
                        db_name,
                        pending,
                        cutoff,
                        biosphere_db_name,
                        project,
                        base_dirs,
                    ): db_name
                    for db_name, pending in pending_per_db.items()
                }
                for future in as_completed(futures):
                    db_name = futures[future]
                    cache_token = _cache_token(db_name, cutoff)
                    entries = future.result()
                    _BACKGROUND_INVENTORY_CACHE.update(
                        {
                            cache_token + identity: entry
                            for identity, entry in entries.items()
                        }
                    )
                    # Written from the parent so that workers never contend for
                    # the same cache file.
                    self._store_on_disk(db_name, cutoff, entries)

        for db_name in self.background_dbs:
            inventory_tensor, elementary_flows = _assemble_inventory_tensor(
                db_name, self._intermediate_flows, cutoff
            )
            self._background_inventory.update(inventory_tensor)
            self._elementary_flows.update(elementary_flows)

    def _sequential_inventory_tensor_calculation(self) -> None:
        """
        Compute the background inventory tensor for all background databases
        sequentially.

        This method performs LCA calculations for each background database listed in
        `self.background_dbs`. All intermediate flows of a database are solved
        against a single factorization of its technosphere matrix, yielding the
        aggregated elementary flow vector per intermediate flow.

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
        for db_name in self.background_dbs:
            try:
                # Directly call the _calculate_inventory_of_db method for each db
                inventory_tensor, elementary_flows = self._calculate_inventory_of_db(
                    db_name, self._intermediate_flows, cutoff
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

            # Populate missing elementary flow names from biosphere database,
            # read in a single pass rather than one query per flow.
            names = {
                code: name
                for code, name in _biosphere_metadata(self.biosphere_db.name).values()
            }
            for _, _, ef_code in self._background_inventory.keys():
                if ef_code not in self._elementary_flows:
                    self._elementary_flows[ef_code] = names[ef_code]
            logger.info(f"Loaded background inventory from: {load_path}")

        else:
            # Compute the background inventory
            if method == "sequential":
                self._sequential_inventory_tensor_calculation()
            elif method == "parallel":
                self.parallel_inventory_tensor_calculation(
                    n_jobs=self.config.background_inventory.n_jobs
                )
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

    def _prune_uncharacterized_flows(self) -> None:
        """
        Drop elementary flows that carry no characterization factor.

        The background inventory keeps every non-zero elementary flow, because a
        tiny flow can still dominate an impact category. A flow without a
        characterization factor in *any* configured category, however, contributes
        exactly zero to every impact, and only inflates the optimization model,
        where the inventory is expressed per (process, elementary flow, year).

        Flows listed in `config.background_inventory.retain_flows` are kept, as are
        all flows when `restrict_to_characterized_flows` is False. Foreground
        biosphere flows are never dropped.

        Side Effects
        ------------
            - self._background_inventory: entries of dropped flows are removed.
            - self._elementary_flows: dropped flows are removed.
        """
        if not self.config.background_inventory.restrict_to_characterized_flows:
            return

        keep = {code for _, code, _ in self._characterization}
        keep.update(self.config.background_inventory.retain_flows)
        keep.update(code for _, code, _ in self._foreground_biosphere)

        dropped = set(self._elementary_flows) - keep
        if not dropped:
            return

        self._background_inventory = {
            key: value
            for key, value in self._background_inventory.items()
            if key[2] in keep
        }
        for code in dropped:
            del self._elementary_flows[code]

        logger.info(
            f"Dropped {len(dropped)} elementary flows without characterization "
            f"factors; {len(self._elementary_flows)} flows remain. Use "
            "`retain_flows` to keep specific flows (e.g. for flow limits)."
        )

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

    def _characterization_functions(self, method: tuple) -> dict:
        """
        Return the dynamic characterization functions of an LCIA method, cached.

        Deriving them from the method costs ~0.12 s, which dominates the actual
        characterization when it is repeated per elementary flow.
        """
        key = (bd.projects.current, tuple(method))
        if key not in _CHARACTERIZATION_FUNCTION_CACHE:
            _CHARACTERIZATION_FUNCTION_CACHE[key] = (
                create_characterization_functions_from_method(method)
            )
        return _CHARACTERIZATION_FUNCTION_CACHE[key]

    def _construct_characterization_tensor(self) -> None:
        """
        Construct the characterization tensor for LCIA methods over system time points.

        This method computes characterization factors for elementary flows across all
        system years, supporting both static and dynamic methods. It handles metrics
        like Global Warming Potential (GWP) and Cumulative Radiative Forcing (CRF)
        when dynamic characterization is requested. Dynamic metrics are characterized
        in a single call covering all elementary flows, with the method's
        characterization functions built once and reused.

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
        years = list(dates.year)
        flow_codes = list(self.elementary_flows.keys())

        # Pre-map flow codes to Brightway flow IDs from a single pass over the
        # biosphere database instead of one query per flow.
        code_to_id = {
            code: flow_id
            for flow_id, (code, _) in _biosphere_metadata(
                self.biosphere_db.name
            ).items()
        }
        flow_ids = {}
        for code in flow_codes:
            flow_id = code_to_id.get(code)
            if flow_id is None:
                flow_id = self.biosphere_db.get(code=code).id
            flow_ids[code] = flow_id
        id_to_code = {flow_id: code for code, flow_id in flow_ids.items()}

        characterization_tensor = {}

        for config in self.config.characterization_methods:
            category_name = config.category_name
            self._category.add(category_name)
            method = config.brightway_method
            metric = config.metric

            if metric is None:
                # Static LCIA
                method_data = bd.Method(method).load()
                method_dict = {flow: value for flow, value in method_data if value != 0}

                for flow_code, flow_id in flow_ids.items():
                    if flow_id not in method_dict:
                        continue
                    value = method_dict[flow_id]
                    for year in years:
                        characterization_tensor[(category_name, flow_code, year)] = (
                            value
                        )
                logger.info(
                    f"Static characterization for method {category_name} completed."
                )
                continue

            characterization_functions = self._characterization_functions(method)
            # Flows without a characterization function are skipped inside
            # `characterize` anyway; dropping them here keeps the frame small.
            characterized_ids = [
                flow_id
                for flow_id in flow_ids.values()
                if flow_id in characterization_functions
            ]

            if not characterized_ids:
                logger.warning(
                    f"No dynamically characterizable flows for {category_name}."
                )
                continue

            if metric == "GWP":
                # Dynamic GWP (year-specific values)
                df = pd.DataFrame(
                    {
                        "flow": np.repeat(characterized_ids, len(dates)),
                        "date": np.tile(
                            dates.values.astype("datetime64[s]"),
                            len(characterized_ids),
                        ),
                    }
                )
                df["amount"] = 1
                df["activity"] = np.nan

                df_char = characterize(
                    df,
                    metric="GWP",
                    characterization_functions=characterization_functions,
                    fixed_time_horizon=self.config.temporal.fixed_time_horizon,
                    base_lcia_method=method,
                    time_horizon=time_horizon,
                )
                df_char["date"] = df_char["date"].dt.year

                for flow_id, year, amount in zip(
                    df_char["flow"], df_char["date"], df_char["amount"]
                ):
                    characterization_tensor[
                        (category_name, id_to_code[flow_id], year)
                    ] = amount
                logger.info(
                    f"Dynamic GWP characterization for {category_name} completed."
                )

            elif metric == "CRF":
                # Dynamic CRF (cumulative RF over time horizon)
                df = pd.DataFrame({"flow": characterized_ids})
                df["date"] = pd.Timestamp(start_date)
                df["amount"] = 1
                df["activity"] = np.nan

                df_char = characterize(
                    df,
                    metric="radiative_forcing",
                    characterization_functions=characterization_functions,
                    fixed_time_horizon=self.config.temporal.fixed_time_horizon,
                    base_lcia_method=method,
                    time_horizon=time_horizon,
                    time_horizon_start=pd.Timestamp(start_date),
                )

                for flow_id, group in df_char.groupby("flow", sort=False):
                    flow_code = id_to_code[flow_id]
                    rf_series = group.sort_values("date")["amount"].values
                    cumulative = np.cumsum(rf_series)
                    for year in self.system_time:
                        cutoff = start_date.year + time_horizon - year - 1
                        if cutoff <= 0:
                            cumulative_rf = 0.0
                        else:
                            cumulative_rf = cumulative[min(cutoff, len(cumulative)) - 1]
                        characterization_tensor[(category_name, flow_code, year)] = (
                            cumulative_rf
                        )
                logger.info(
                    f"Dynamic CRF characterization for {category_name} completed."
                )

            else:
                raise ValueError(f"Unsupported dynamic metric: {metric}")

        self._characterization.update(characterization_tensor)
