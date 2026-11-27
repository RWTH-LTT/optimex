"""`optimizer.solve_series`: sweeps and scenarios give the same results as a loop."""

import numpy as np
import pandas as pd
import pytest

from optimex import optimizer
from optimex.persistent import persistent_solver
from optimex.postprocessing import PostProcessor
from tests.test_incremental import _system

SOLVERS = ["glpk"] + [
    pytest.param(s, marks=pytest.mark.skipif(persistent_solver(s)[0] is None, reason=s))
    for s in ("highs", "gurobi")
]


def _loop(variants, solver, **kwargs):
    objectives, impacts = {}, {}
    for label, inputs in variants.items():
        model = optimizer.create_model(inputs, name=str(label), **kwargs)
        try:
            model, objective, _ = optimizer.solve_model(model, solver_name="glpk", tee=False)
        except RuntimeError:
            objectives[label], impacts[label] = np.nan, np.nan
            continue
        objectives[label] = objective
        impacts[label] = PostProcessor(model).get_impacts()["climate"].to_numpy().sum()
    return pd.Series(objectives), pd.Series(impacts)


def _budget_sweep(base):
    climate_min = optimizer.create_model(base, name="min", objective_category="climate")
    _, lowest, _ = optimizer.solve_model(climate_min, solver_name="glpk", tee=False)
    cost_min = optimizer.create_model(base, name="cost", objective="cost")
    cost_min, _, _ = optimizer.solve_model(cost_min, solver_name="glpk", tee=False)
    highest = PostProcessor(cost_min).get_impacts()["climate"].to_numpy().sum()
    budgets = [lowest * 0.9] + list(np.linspace(lowest * (1 + 1e-6), highest, 5))
    return {
        round(b, 6): base.model_copy(
            update={"cumulative_category_impact_limits": {"climate": b}}
        )
        for b in budgets
    }


@pytest.mark.parametrize("solver", SOLVERS)
def test_budget_sweep_matches_a_loop(solver):
    base = _system()
    variants = _budget_sweep(base)
    sweep = optimizer.solve_series(
        variants, base=base, solver_name=solver, objective="cost", progress=False
    )
    expected_objective, expected_impact = _loop(variants, solver, objective="cost")
    assert list(sweep.objective.index) == list(variants)
    assert sweep.objective.index.name == "Variant"
    assert sweep.status.iloc[0] != "optimal"  # below the climate minimum
    assert sweep.feasible.iloc[1:].all()
    np.testing.assert_allclose(sweep.objective, expected_objective.values, rtol=1e-6)
    np.testing.assert_allclose(
        sweep.impacts["climate"].iloc[1:], expected_impact.values[1:], rtol=1e-6
    )
    # Cost falls as the budget loosens
    assert (np.diff(sweep.objective.iloc[1:].to_numpy()) <= 1e-6).all()
    assert set(sweep.installation.index.get_level_values("Variant")) == set(list(variants)[1:])


@pytest.mark.parametrize("solver", SOLVERS)
def test_price_scenarios_and_numbered_variants(solver):
    base = _system()
    scenarios = {
        "reference": base,
        "cheap I2": base.model_copy(update={"intermediate_flow_prices": {
            k: v * (0.2 if k[0] == "I2" else 1.0) for k, v in base.intermediate_flow_prices.items()
        }}),
        "expensive I1": base.model_copy(update={"intermediate_flow_prices": {
            k: v * (5.0 if k[0] == "I1" else 1.0) for k, v in base.intermediate_flow_prices.items()
        }}),
        "new structure": base.model_copy(update={"existing_capacity": {}}),  # full rebuild
    }
    results = optimizer.solve_series(
        scenarios, solver_name=solver, objective="cost",
        outputs=("installation", "operation"), progress=False,
    )
    expected, _ = _loop(scenarios, solver, objective="cost")
    np.testing.assert_allclose(results.objective, expected.values, rtol=1e-6)
    assert list(results.costs.index) == list(scenarios)

    numbered = optimizer.solve_series(
        list(scenarios.values()), solver_name=solver, objective="cost", progress=False
    )
    assert list(numbered.objective.index) == [0, 1, 2, 3]
    np.testing.assert_allclose(numbered.objective, expected.values, rtol=1e-6)


def test_empty_series():
    with pytest.raises(ValueError, match="empty"):
        optimizer.solve_series({}, objective="cost")
