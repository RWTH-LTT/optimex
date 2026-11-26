"""Aggregating identical vintages must not change the optimum or the results."""

import pyomo.environ as pyo
import pytest
from pyomo.repn import generate_standard_repn

from optimex import converter, optimizer


def _operation_variables(model):
    """Operation variables the solver sees (those appearing in constraints)."""
    seen = set()
    for con in model.component_data_objects(pyo.Constraint, active=True):
        for var in generate_standard_repn(con.body).linear_vars:
            if var.parent_component().name.startswith("var_operation"):
                seen.add(id(var))
    return len(seen)


def _solve(inputs, aggregate):
    model = optimizer.create_model(
        inputs,
        name="agg",
        objective_category="climate_change",
        aggregate_vintages=aggregate,
    )
    return optimizer.solve_model(model, solver_name="glpk", tee=False)


def test_aggregation_matches_per_vintage(abstract_system_model_inputs):
    inputs = converter.OptimizationModelInputs(**abstract_system_model_inputs)
    per_vintage, obj_ref, _ = _solve(inputs, aggregate=False)
    aggregated, obj_agg, _ = _solve(inputs, aggregate=True)

    assert obj_agg == pytest.approx(obj_ref, rel=1e-9)
    assert aggregated._aggregated_processes == {"P1", "P2"}
    assert _operation_variables(aggregated) < _operation_variables(per_vintage)

    # Disaggregated per-vintage operation respects capacity and sums to the total
    assert optimizer.validate_operation_bounds(aggregated)["valid"]
    for (p, t), var in aggregated.var_operation_total.items():
        split = sum(
            optimizer.operation_value(aggregated, p, v, t)
            for v in aggregated._active_vintages_by_process_time[p, t]
        )
        assert split == pytest.approx(pyo.value(var), abs=1e-9)


def test_vintage_dependent_process_is_not_aggregated(abstract_system_model_inputs):
    data = dict(abstract_system_model_inputs)
    data["foreground_production"] = {
        ("P1", "R1", 1): 0.6,  # rate changes over the operation window
        ("P1", "R1", 2): 0.4,
        ("P2", "R1", 1): 0.5,
        ("P2", "R1", 2): 0.5,
    }
    inputs = converter.OptimizationModelInputs(**data)
    model, obj_agg, _ = _solve(inputs, aggregate=True)
    _, obj_ref, _ = _solve(inputs, aggregate=False)
    assert model._aggregated_processes == {"P2"}
    assert obj_agg == pytest.approx(obj_ref, rel=1e-9)


def test_aggregated_processes_have_no_per_vintage_variables(
    abstract_system_model_inputs,
):
    inputs = converter.OptimizationModelInputs(**abstract_system_model_inputs)
    model, _, _ = _solve(inputs, aggregate=True)
    assert len(model.var_operation) == 0
    assert len(model.var_operation_total) > 0


def test_postprocessing_matches_per_vintage(abstract_system_model_inputs):
    from optimex.postprocessing import PostProcessor

    inputs = converter.OptimizationModelInputs(**abstract_system_model_inputs)
    per_vintage, _, _ = _solve(inputs, aggregate=False)
    aggregated, _, _ = _solve(inputs, aggregate=True)
    for method in ("get_impacts", "get_production", "get_installation"):
        expected = getattr(PostProcessor(per_vintage), method)()
        actual = getattr(PostProcessor(aggregated), method)()
        assert actual.sum().sum() == pytest.approx(expected.sum().sum(), rel=1e-9)
    # Totals over vintages agree; the split between identical vintages may not
    expected = PostProcessor(per_vintage).get_operation().sum(axis=1)
    actual = PostProcessor(aggregated).get_operation().sum(axis=1)
    assert actual.to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-9, abs=1e-9)
