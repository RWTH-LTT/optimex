"""The linopy backend must reproduce the Pyomo model's optimum and results."""

import pandas as pd
import pyomo.environ as pyo
import pytest

from optimex import converter, optimizer
from optimex.postprocessing import PostProcessor

pytest.importorskip("linopy")

SOLVER = "glpk"


def _inputs(abstract_system_model_inputs, **overrides):
    inputs = converter.OptimizationModelInputs(**abstract_system_model_inputs)
    for name, value in overrides.items():
        setattr(inputs, name, value)
    return inputs


def _solve_both(inputs, **kwargs):
    results = {}
    for backend in ("pyomo", "linopy"):
        model = optimizer.create_model(inputs, name=backend, backend=backend, **kwargs)
        results[backend] = optimizer.solve_model(model, solver_name=SOLVER, tee=False)
    return results["pyomo"], results["linopy"]


def _assert_same_results(pyomo_model, linopy_model):
    pyomo_pp, linopy_pp = PostProcessor(pyomo_model), PostProcessor(linopy_model)
    for method in ("get_impacts", "get_installation", "get_production", "get_demand"):
        expected = getattr(pyomo_pp, method)()
        actual = getattr(linopy_pp, method)()
        pd.testing.assert_frame_equal(
            actual.sort_index(axis=1), expected.sort_index(axis=1), rtol=1e-7, atol=1e-9
        )
    # The split between identical vintages may differ; totals per year may not
    pd.testing.assert_series_equal(
        linopy_pp.get_operation().sum(axis=1),
        pyomo_pp.get_operation().sum(axis=1),
        rtol=1e-7,
        atol=1e-9,
    )


@pytest.mark.parametrize("aggregate", [False, True])
def test_same_optimum_and_results(abstract_system_model_inputs, aggregate):
    (pyomo_model, pyomo_obj, _), (linopy_model, linopy_obj, results) = _solve_both(
        _inputs(abstract_system_model_inputs),
        objective_category="climate_change",
        aggregate_vintages=aggregate,
    )
    assert linopy_obj == pytest.approx(pyomo_obj, rel=1e-9)
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    assert linopy_model.solved
    assert pyo.value(linopy_model.total_impact["climate_change"]) == pytest.approx(
        pyo.value(pyomo_model.total_impact["climate_change"]), rel=1e-9
    )
    _assert_same_results(pyomo_model, linopy_model)


def test_same_optimum_with_limits(abstract_system_model_inputs):
    inputs = _inputs(
        abstract_system_model_inputs,
        cumulative_process_limits_max={"P1": 10.0},
        category_impact_limits={("land_use", 2025): 1e3},
        flow_limits_max={("CO2", 2026): 1e3},
        cumulative_flow_limits_max={"CH4": 1e4},
        process_operation_limits_max={("P2", 2027): 15.0},
    )
    (pyomo_model, pyomo_obj, _), (linopy_model, linopy_obj, _) = _solve_both(
        inputs, objective_category="climate_change"
    )
    assert linopy_obj == pytest.approx(pyomo_obj, rel=1e-9)
    _assert_same_results(pyomo_model, linopy_model)


def test_cost_objective():
    from tests.test_economics import _two_route_inputs

    (pyomo_model, pyomo_obj, _), (linopy_model, linopy_obj, _) = _solve_both(
        _two_route_inputs(), objective="cost"
    )
    assert linopy_obj == pytest.approx(pyomo_obj, rel=1e-9)
    pd.testing.assert_frame_equal(
        PostProcessor(linopy_model).get_costs(),
        PostProcessor(pyomo_model).get_costs(),
        rtol=1e-7,
    )


def test_operation_value_and_bounds(abstract_system_model_inputs):
    model = optimizer.create_model(
        _inputs(abstract_system_model_inputs),
        name="linopy",
        objective_category="climate_change",
        aggregate_vintages=True,
        backend="linopy",
    )
    model, _, _ = optimizer.solve_model(model, solver_name=SOLVER, tee=False)
    assert optimizer.validate_operation_bounds(model)["valid"]
    for (p, t), total in model.var_operation_total.items():
        split = sum(
            optimizer.operation_value(model, p, v, t)
            for v in model._active_vintages_by_process_time[p, t]
        )
        assert split == pytest.approx(total, abs=1e-9)


def test_infeasible_constant_row_is_reported(abstract_system_model_inputs):
    data = dict(abstract_system_model_inputs)
    data["PRODUCT"] = ["R1", "R2"]
    data["demand"] = {**data["demand"], ("R2", 2025): 1.0}  # nothing produces R2
    inputs = converter.OptimizationModelInputs(**data)
    model = optimizer.create_model(
        inputs, name="infeasible", objective_category="climate_change", backend="linopy"
    )
    with pytest.raises(RuntimeError, match="infeasible"):
        optimizer.solve_model(model, solver_name=SOLVER, tee=False)


def test_save_and_load(abstract_system_model_inputs, tmp_path):
    model = optimizer.create_model(
        _inputs(abstract_system_model_inputs),
        name="linopy",
        objective_category="climate_change",
        backend="linopy",
    )
    model, objective, _ = optimizer.solve_model(model, solver_name=SOLVER, tee=False)
    path = tmp_path / "solved.pkl"
    optimizer.save_solved_model(model, path, objective_value=objective)
    loaded, loaded_objective = optimizer.load_solved_model(path)
    assert loaded_objective == objective
    pd.testing.assert_frame_equal(
        PostProcessor(loaded).get_impacts(), PostProcessor(model).get_impacts()
    )


def test_debug_path_writes_lp_file(abstract_system_model_inputs, tmp_path):
    path = tmp_path / "model.lp"
    optimizer.create_model(
        _inputs(abstract_system_model_inputs),
        name="linopy",
        objective_category="climate_change",
        backend="linopy",
        debug_path=str(path),
    )
    assert path.read_text().strip()


def test_unknown_backend(abstract_system_model_inputs):
    with pytest.raises(ValueError, match="backend"):
        optimizer.create_model(
            _inputs(abstract_system_model_inputs),
            name="x",
            objective_category="climate_change",
            backend="cvxpy",
        )
