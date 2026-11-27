"""The incremental rebuild must give exactly the linear program of a full rebuild."""

import math

import numpy as np
import pytest

from optimex import converter
from optimex.formulation import build_formulation
from optimex.incremental import IncrementalFormulation
from optimex.persistent import formulation_arrays

YEARS = list(range(2020, 2027))


def _system(**overrides):
    """Three processes, two products (one consumed internally), vintages, existing capacity."""
    data = {
        "PROCESS": ["P1", "P2", "P3"],
        "PRODUCT": ["R1", "R2"],
        "INTERMEDIATE_FLOW": ["I1", "I2"],
        "ELEMENTARY_FLOW": ["E1", "E2"],
        "BACKGROUND_ID": ["db_2020", "db_2026"],
        "PROCESS_TIME": [0, 1, 2, 3],
        "SYSTEM_TIME": YEARS,
        "CATEGORY": ["climate", "water"],
        "operation_time_limits": {"P1": (1, 2), "P2": (1, 3), "P3": (0, 2)},
        "demand": {("R1", t): 5.0 + (t - 2020) for t in YEARS} | {("R2", t): 2.0 for t in YEARS},
        "foreground_production": {
            ("P1", "R1", 1): 2.0, ("P1", "R1", 2): 2.0,
            ("P2", "R1", 1): 1.5, ("P2", "R1", 2): 1.5, ("P2", "R1", 3): 1.5,
            ("P3", "R2", 0): 1.0, ("P3", "R2", 1): 1.0, ("P3", "R2", 2): 1.0,
        },
        "internal_demand_technosphere": {
            ("P3", "R1", 0): 0.5, ("P3", "R1", 1): 0.5, ("P3", "R1", 2): 0.5,
        },
        "foreground_technosphere": {
            ("P1", "I1", 0): 4.0,
            ("P1", "I2", 1): 1.0, ("P1", "I2", 2): 1.0,
            ("P2", "I1", 0): 8.0,
            ("P2", "I2", 1): 0.4, ("P2", "I2", 2): 0.4, ("P2", "I2", 3): 0.4,
            ("P3", "I2", 0): 0.2, ("P3", "I2", 1): 0.2, ("P3", "I2", 2): 0.2,
        },
        "foreground_biosphere": {
            ("P1", "E1", 1): 3.0, ("P1", "E1", 2): 3.0,
            ("P2", "E1", 1): 0.5, ("P2", "E1", 2): 0.5, ("P2", "E1", 3): 0.5,
            ("P2", "E2", 0): 2.0,
            ("P3", "E2", 0): 0.1, ("P3", "E2", 1): 0.1, ("P3", "E2", 2): 0.1,
        },
        "foreground_technosphere_vintages": {
            ("P2", "I2", 1, 2020): 0.4, ("P2", "I2", 1, 2026): 0.2,
            ("P2", "I2", 2, 2020): 0.4, ("P2", "I2", 2, 2026): 0.2,
            ("P2", "I2", 3, 2020): 0.4, ("P2", "I2", 3, 2026): 0.2,
        },
        "vintage_improvements": {("P1", "E1", 2020): 1.0, ("P1", "E1", 2026): 0.6},
        "operation_flow": {
            ("P1", "R1"): True, ("P2", "R1"): True, ("P3", "R2"): True,
            ("P3", "R1"): True, ("P1", "I2"): True, ("P2", "I2"): True,
            ("P3", "I2"): True, ("P1", "E1"): True, ("P2", "E1"): True, ("P3", "E2"): True,
        },
        "background_inventory": {
            ("db_2020", "I1", "E1"): 2.0, ("db_2020", "I1", "E2"): 1.0,
            ("db_2020", "I2", "E1"): 0.5, ("db_2020", "I2", "E2"): 3.0,
            ("db_2026", "I1", "E1"): 1.0, ("db_2026", "I1", "E2"): 1.0,
            ("db_2026", "I2", "E1"): 0.2, ("db_2026", "I2", "E2"): 2.0,
        },
        "mapping": {
            (db, t): (1 - (t - 2020) / 6 if db == "db_2020" else (t - 2020) / 6)
            for db in ("db_2020", "db_2026") for t in YEARS
        },
        "characterization": {
            ("climate", "E1", t): 1.0 + 0.01 * (t - 2020) for t in YEARS
        } | {("water", "E2", t): 1.0 for t in YEARS},
        "intermediate_flow_prices": {("I1", t): 10.0 for t in YEARS}
        | {("I2", t): 2.0 - 0.1 * (t - 2020) for t in YEARS},
        "discount_rate": 0.03,
        "existing_capacity": {("P1", 2018): 3.0},
    }
    data.update(overrides)
    return converter.OptimizationModelInputs(**data)


def _scale_pairs(tensor, factors):
    return {k: v * factors.get(k[:2], 1.0) for k, v in tensor.items()}


def _sample(base, rng, background=True, prices=True):
    """Like a Monte Carlo sample: one factor per exchange, new background and prices."""
    update = {}
    for tensor, vintages in (
        ("foreground_technosphere", "foreground_technosphere_vintages"),
        ("foreground_biosphere", "foreground_biosphere_vintages"),
        ("foreground_production", "foreground_production_vintages"),
        ("internal_demand_technosphere", None),
    ):
        pairs = {k[:2] for k in getattr(base, tensor)}
        if vintages and getattr(base, vintages):
            pairs |= {k[:2] for k in getattr(base, vintages)}
        factors = {pair: rng.lognormal(0, 0.3) for pair in pairs}
        update[tensor] = _scale_pairs(getattr(base, tensor), factors)
        if vintages and getattr(base, vintages):
            update[vintages] = _scale_pairs(getattr(base, vintages), factors)
    if background:
        update["background_inventory"] = {
            k: v * rng.lognormal(0, 0.3) for k, v in base.background_inventory.items()
        }
    if prices:
        update["intermediate_flow_prices"] = {
            k: v * rng.lognormal(0, 0.2) for k, v in base.intermediate_flow_prices.items()
        }
    return base.model_copy(update=update)


def _canonical(lp):
    rows = []
    A = lp.A.tocsr()
    for i in range(A.shape[0]):
        start, end = A.indptr[i], A.indptr[i + 1]
        terms = tuple(sorted(
            (int(col), float(f"{val:.10g}"))
            for col, val in zip(A.indices[start:end], A.data[start:end])
            if abs(val) > 1e-14
        ))
        rows.append((int(lp.senses[i]), float(f"{lp.rhs[i]:.10g}"), terms))
    return sorted(rows)


def _assert_same_lp(incremental, inputs, **kwargs):
    sample_lp = incremental.build(inputs)
    assert sample_lp is not None, incremental.unsupported(inputs)
    full = formulation_arrays(build_formulation(inputs, scales=incremental.scales, **kwargs))
    lp = sample_lp.lp
    assert lp.infeasible == full.infeasible
    np.testing.assert_allclose(lp.c, full.c, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(lp.lb, full.lb)
    np.testing.assert_allclose(lp.ub, full.ub)
    assert _canonical(lp) == _canonical(full)


CASES = [
    dict(objective_category="climate"),
    dict(objective_category="water", aggregate_vintages=True),
    dict(objective="cost"),
    dict(objective="cost", aggregate_vintages=True),
]


@pytest.mark.parametrize("kwargs", CASES)
def test_samples_give_the_full_rebuild(kwargs):
    base = _system()
    incremental = IncrementalFormulation(base, **kwargs)
    rng = np.random.default_rng(1)
    _assert_same_lp(incremental, base, **kwargs)
    for _ in range(4):
        _assert_same_lp(incremental, _sample(base, rng), **kwargs)
    _assert_same_lp(incremental, _sample(base, rng, background=False, prices=False), **kwargs)


@pytest.mark.parametrize("kwargs", CASES)
def test_limits_demand_and_bounds(kwargs):
    base = _system()
    incremental = IncrementalFormulation(base, **kwargs)
    sample = _sample(base, np.random.default_rng(2))
    variants = [
        {"cumulative_category_impact_limits": {"climate": 500.0, "water": math.inf}},
        {"category_impact_limits": {("climate", 2023): 40.0, ("water", 2025): 30.0}},
        {"demand": {k: v * 1.3 for k, v in base.demand.items()}},
        {"process_deployment_limits_max": {("P1", 2022): 2.0, ("P2", 2024): 0.0},
         "process_deployment_limits_min": {("P3", 2021): 1.0}},
        {"process_operation_limits_max": {("P1", 2023): 1.5},
         "process_operation_limits_min": {("P2", 2025): 0.5}},
        {"discount_rate": 0.07, "discount_reference_year": 2022},
    ]
    for update in variants:
        _assert_same_lp(incremental, sample.model_copy(update=update), **kwargs)


def test_infeasible_demand_is_detected():
    base = _system(PRODUCT=["R1", "R2", "R3"])
    base = base.model_copy(update={"demand": {}})
    incremental = IncrementalFormulation(base, objective_category="climate")
    sample = base.model_copy(update={"demand": {("R3", 2022): 1.0}})
    assert incremental.build(sample).lp.infeasible


def test_unsupported_changes_fall_back():
    base = _system()
    incremental = IncrementalFormulation(base, objective_category="climate")
    # One exchange changed by different factors at different process times
    tensor = dict(base.foreground_biosphere)
    tensor["P2", "E1", 1] *= 2.0
    assert incremental.build(base.model_copy(update={"foreground_biosphere": tensor})) is None
    # A structural change
    assert incremental.build(base.model_copy(update={"existing_capacity": {("P1", 2018): 1.0}})) is None
    assert incremental.build(base.model_copy(update={"mapping": {k: 0.5 for k in base.mapping}})) is None
    # A new key
    tensor = dict(base.foreground_biosphere)
    tensor["P1", "E2", 1] = 1.0
    assert incremental.build(base.model_copy(update={"foreground_biosphere": tensor})) is None
