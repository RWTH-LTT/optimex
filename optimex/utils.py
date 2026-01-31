"""
Utility functions for optimex.
"""

import numpy as np
import bw2data as bd

from bw_temporalis import TemporalDistribution, easy_timedelta_distribution


def infer_operation_td_from_limits(
    node: bd.backends.proxies.Activity, 
    resolution: str = "Y"
):
    """
    Infer a temporal distribution for the operation of a process node based on its
    operation_time_limits.
    
    Parameters
    ----------
    node : bd.backends.proxies.Activity
        The Brightway activity node with operation_time_limits attribute.
    resolution : str, optional
        Temporal resolution for the distribution. Options: "Y" (year), "M" (month), "D" (day).
        Default is "Y".
    
    Returns
    -------
    TemporalDistribution or None
        A temporal distribution object, or None if limits are not defined.
    """
    limits = node.get("operation_time_limits")

    if not limits or len(limits) != 2:
        return None

    start, end = limits

    # Calculate steps: e.g., 2020 to 2022 inclusive is 3 steps (2020, 2021, 2022)
    num_steps = int(end - start + 1)

    return easy_timedelta_distribution(
        start=start,
        end=end,
        steps=num_steps,
        kind="uniform",
        resolution=resolution,
    )


def infer_eol_td_from_limits(
    node: bd.backends.proxies.Activity,
    resolution: str = "Y"
):
    """
    Infer a temporal distribution for the end-of-life of a process node based on its
    operation_time_limits, assuming EOL occurs one time unit after operation ends.
    
    Parameters
    ----------
    node : bd.backends.proxies.Activity
        The Brightway activity node with operation_time_limits attribute.
    resolution : str, optional
        Temporal resolution for the distribution. Options: "Y" (year), "M" (month), "D" (day).
        Default is "Y".
    
    Returns
    -------
    TemporalDistribution or None
        A temporal distribution object, or None if limits are not defined.
    """
    limits = node.get("operation_time_limits")

    if not limits or len(limits) != 2:
        return None

    _, end = limits

    return TemporalDistribution(
        date=np.array([end + 1], dtype=f"timedelta64[{resolution}]"), amount=np.array([1])
    )


def infer_construction_td_from_limits(
    node: bd.backends.proxies.Activity,
    resolution: str = "Y"
):
    """
    Infer a temporal distribution for the construction of a process node based on its
    operation_time_limits, assuming construction occurs at the start of operation.
    
    Parameters
    ----------
    node : bd.backends.proxies.Activity
        The Brightway activity node with operation_time_limits attribute.
    resolution : str, optional
        Temporal resolution for the distribution. Options: "Y" (year), "M" (month), "D" (day).
        Default is "Y".
    
    Returns
    -------
    TemporalDistribution or None
        A temporal distribution object, or None if limits are not defined.
    """
    limits = node.get("operation_time_limits")

    if not limits or len(limits) != 2:
        return None

    start, _ = limits

    return TemporalDistribution(
        date=np.array([start], dtype=f"timedelta64[{resolution}]"), amount=np.array([1])
    )
