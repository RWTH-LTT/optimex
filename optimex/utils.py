"""Utility functions for optimex-specific setup tasks.

This module provides helper functions for common optimex tasks like:
- Setting operation time limits on processes
- Adding temporal distributions to exchanges
- Marking exchanges as operational
- Setting up temporal metadata for databases
- Creating temporal demand patterns
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

import bw2data as bd
import numpy as np
from bw_temporalis import TemporalDistribution, easy_timedelta_distribution


def set_operation_time_limits(
    process: bd.Node,
    start: int,
    end: int,
    save: bool = True,
) -> bd.Node:
    """
    Set operation time limits for a process.

    Operation time limits define the start and end years of the operation phase
    for a process in optimex optimization.

    Parameters
    ----------
    process : bd.Node
        The process node to set operation time limits for
    start : int
        Start year of operation phase
    end : int
        End year of operation phase
    save : bool, optional
        Whether to save the process after modification (default: True)

    Returns
    -------
    bd.Node
        The modified process node (for chaining)

    Examples
    --------
    >>> process = bd.get_node(database="foreground", code="h2_process")
    >>> set_operation_time_limits(process, start=0, end=10)
    """
    process["operation_time_limits"] = (start, end)
    if save:
        process.save()
    return process


def add_temporal_distribution_to_exchanges(
    process_or_exchanges: Union[bd.Node, List],
    start: int = 0,
    end: int = 10,
    steps: Optional[int] = None,
    kind: str = "uniform",
    resolution: str = "Y",
    save: bool = True,
) -> Union[bd.Node, List]:
    """
    Add temporal distributions to exchanges of a process or a list of exchanges.

    This applies temporal distributions to all exchanges, which is useful for
    spreading impacts across the operation lifetime in optimex.

    Parameters
    ----------
    process_or_exchanges : bd.Node or list
        The process node or list of exchanges to add temporal distributions to
    start : int, optional
        Start time in years (default: 0)
    end : int, optional
        End time in years (default: 10)
    steps : int, optional
        Number of time steps (defaults to end - start + 1)
    kind : str, optional
        Type of distribution (default: "uniform")
    resolution : str, optional
        Temporal resolution, e.g., "Y" for years (default: "Y")
    save : bool, optional
        Whether to save exchanges after modification (default: True)

    Returns
    -------
    bd.Node or list
        The modified process node or list of exchanges (for chaining)

    Examples
    --------
    >>> # Apply to all exchanges of a process
    >>> add_temporal_distribution_to_exchanges(process, start=0, end=8)
    >>>
    >>> # Apply to specific exchanges
    >>> exchanges = list(process.technosphere())
    >>> add_temporal_distribution_to_exchanges(exchanges, start=0, end=10, steps=11)
    """
    if steps is None:
        steps = end - start + 1

    # Handle both process nodes and exchange lists
    if isinstance(process_or_exchanges, bd.Node):
        exchanges = list(process_or_exchanges.exchanges())
        is_process = True
    else:
        exchanges = process_or_exchanges
        is_process = False

    for exc in exchanges:
        exc["temporal_distribution"] = easy_timedelta_distribution(
            start=start, end=end, steps=steps, kind=kind, resolution=resolution
        )
        if save:
            exc.save()

    if is_process and save:
        process_or_exchanges.save()

    return process_or_exchanges


def mark_exchanges_as_operation(
    process_or_exchanges: Union[bd.Node, List],
    save: bool = True,
) -> Union[bd.Node, List]:
    """
    Mark exchanges as operational exchanges.

    This sets the 'operation' flag on exchanges, which is used by optimex
    to identify exchanges that occur during the operation phase.

    Parameters
    ----------
    process_or_exchanges : bd.Node or list
        The process node or list of exchanges to mark
    save : bool, optional
        Whether to save exchanges after modification (default: True)

    Returns
    -------
    bd.Node or list
        The modified process node or list of exchanges (for chaining)

    Examples
    --------
    >>> # Mark all exchanges of a process
    >>> mark_exchanges_as_operation(process)
    >>>
    >>> # Mark specific exchanges only
    >>> tech_exchanges = list(process.technosphere())
    >>> mark_exchanges_as_operation(tech_exchanges)
    """
    # Handle both process nodes and exchange lists
    if isinstance(process_or_exchanges, bd.Node):
        exchanges = list(process_or_exchanges.exchanges())
        is_process = True
    else:
        exchanges = process_or_exchanges
        is_process = False

    for exc in exchanges:
        exc["operation"] = True
        if save:
            exc.save()

    if is_process and save:
        process_or_exchanges.save()

    return process_or_exchanges


def setup_optimex_process(
    process: bd.Node,
    operation_time_limits: tuple,
    temporal_distribution_params: Optional[Dict] = None,
    mark_as_operation: bool = True,
    save: bool = True,
) -> bd.Node:
    """
    Configure a process with all optimex-specific settings in one call.

    This is a convenience function that combines setting operation time limits,
    adding temporal distributions, and marking exchanges as operational.

    Parameters
    ----------
    process : bd.Node
        The process node to configure
    operation_time_limits : tuple of (int, int)
        Start and end time for operation phase in years (e.g., (0, 10))
    temporal_distribution_params : dict, optional
        Parameters for temporal distribution (start, end, steps, kind, resolution).
        If None, uses operation_time_limits for start/end.
    mark_as_operation : bool, optional
        Whether to mark exchanges as operation=True (default: True)
    save : bool, optional
        Whether to save after modifications (default: True)

    Returns
    -------
    bd.Node
        The configured process node

    Examples
    --------
    >>> # Basic setup with operation time limits
    >>> setup_optimex_process(process, operation_time_limits=(0, 10))
    >>>
    >>> # With custom temporal distribution
    >>> setup_optimex_process(
    ...     process,
    ...     operation_time_limits=(0, 8),
    ...     temporal_distribution_params={'start': 0, 'end': 8, 'steps': 9}
    ... )
    >>>
    >>> # Without marking as operation
    >>> setup_optimex_process(
    ...     process,
    ...     operation_time_limits=(0, 10),
    ...     mark_as_operation=False
    ... )
    """
    # Set operation time limits
    process["operation_time_limits"] = operation_time_limits

    # Setup temporal distribution parameters
    if temporal_distribution_params is None:
        temporal_distribution_params = {
            "start": operation_time_limits[0],
            "end": operation_time_limits[1],
        }

    # Get default values for optional params
    start = temporal_distribution_params.get("start", 0)
    end = temporal_distribution_params.get("end", 10)
    steps = temporal_distribution_params.get("steps", None)
    kind = temporal_distribution_params.get("kind", "uniform")
    resolution = temporal_distribution_params.get("resolution", "Y")

    if steps is None:
        steps = end - start + 1

    # Modify all exchanges directly
    for exc in process.exchanges():
        exc["temporal_distribution"] = easy_timedelta_distribution(
            start=start, end=end, steps=steps, kind=kind, resolution=resolution
        )
        if mark_as_operation:
            exc["operation"] = True
        if save:
            exc.save()

    # Save process at the end if requested
    if save:
        process.save()

    return process


def setup_database_temporal_metadata(
    databases: Dict[int, bd.Database],
    month: int = 1,
    day: int = 1,
) -> None:
    """
    Set representative_time metadata for multiple databases.

    This sets the temporal metadata for background databases that represent
    different time periods (e.g., 2020, 2030, 2040, 2050).

    Parameters
    ----------
    databases : dict
        Dictionary mapping years (int) to Database objects
    month : int, optional
        Month for the representative date (default: 1)
    day : int, optional
        Day for the representative date (default: 1)

    Examples
    --------
    >>> dbs = {
    ...     2020: bd.Database("ei311_2020"),
    ...     2030: bd.Database("ei311_2030"),
    ...     2050: bd.Database("ei311_2050"),
    ... }
    >>> setup_database_temporal_metadata(dbs)
    """
    for year, db in databases.items():
        db.metadata["representative_time"] = datetime(year, month, day).isoformat()


def create_temporal_demand(
    product: bd.Node,
    years: range,
    amounts: Optional[np.ndarray] = None,
    trend_start: float = 10.0,
    trend_end: float = 20.0,
    noise_std: float = 0.0,
    random_seed: Optional[int] = None,
) -> Dict[bd.Node, TemporalDistribution]:
    """
    Create a temporal distribution for functional demand.

    This creates a temporal demand pattern with optional trend and noise,
    which is useful for modeling demand growth over time.

    Parameters
    ----------
    product : bd.Node
        The product node for which to create demand
    years : range
        Range of years for the demand (e.g., range(2025, 2075))
    amounts : np.ndarray, optional
        Custom amounts for each year. If None, uses trend and noise.
    trend_start : float, optional
        Starting value for linear trend (default: 10.0)
    trend_end : float, optional
        Ending value for linear trend (default: 20.0)
    noise_std : float, optional
        Standard deviation of Gaussian noise (default: 0.0)
    random_seed : int, optional
        Seed for random number generator for reproducibility

    Returns
    -------
    dict
        Dictionary with product node as key and TemporalDistribution as value

    Examples
    --------
    >>> # Simple linear trend
    >>> demand = create_temporal_demand(methanol, range(2025, 2075))
    >>>
    >>> # With noise
    >>> demand = create_temporal_demand(
    ...     methanol, range(2025, 2075),
    ...     noise_std=4.0, random_seed=42
    ... )
    >>>
    >>> # Custom amounts
    >>> custom_amounts = np.array([10, 12, 15, 18, 20])
    >>> demand = create_temporal_demand(
    ...     methanol, range(2025, 2030), amounts=custom_amounts
    ... )
    """
    years_list = list(years)

    if amounts is None:
        rng = (
            np.random.default_rng(random_seed)
            if random_seed is not None
            else np.random.default_rng()
        )
        trend = np.linspace(trend_start, trend_end, len(years_list))
        noise = rng.normal(0, noise_std, len(years_list))
        amounts = trend + noise

    td_demand = TemporalDistribution(
        date=np.array(
            [datetime(year, 1, 1).isoformat() for year in years_list],
            dtype="datetime64[s]",
        ),
        amount=amounts,
    )

    return {product: td_demand}
