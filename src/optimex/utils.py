"""
Utility functions for optimex.
"""

import numpy as np
import bw2data as bd

from bw_temporalis import TemporalDistribution, easy_timedelta_distribution


def infer_operation_td_from_limits(node: bd.backends.proxies.Activity):
    """
    Infer a temporal distribution for the operation of a process node based on its
    operation_time_limits.
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
        resolution="Y",
    )


def infer_eol_td_from_limits(node: bd.backends.proxies.Activity):
    """
    Infer a temporal distribution for the end-of-life of a process node based on its
    operation_time_limits, assuming EOL occurs one year after operation ends.
    """
    limits = node.get("operation_time_limits")

    if not limits or len(limits) != 2:
        return None

    _, end = limits

    return TemporalDistribution(
        date=np.array([end + 1], dtype="timedelta64[Y]"), amount=np.array([1])
    )


def infer_construction_td_from_limits(node: bd.backends.proxies.Activity):
    """
    Infer a temporal distribution for the construction of a process node based on its
    operation_time_limits, assuming construction occurs one year before operation starts.
    """
    limits = node.get("operation_time_limits")

    if not limits or len(limits) != 2:
        return None

    start, _ = limits

    return TemporalDistribution(
        date=np.array([start], dtype="timedelta64[Y]"), amount=np.array([1])
    )
