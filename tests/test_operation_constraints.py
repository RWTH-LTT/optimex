"""
Tests for operation capacity constraints and flexible operation mode.

These tests verify that the OperationLimit constraint correctly bounds
operation levels by available installed capacity across multiple time periods.
"""
import pyomo.environ as pyo
import pytest

from optimex import converter, optimizer


def get_total_operation(model, p, t):
    """Get total operation for a process at a time, summed across all vintages."""
    return sum(
        pyo.value(model.var_operation[proc, v, time])
        for (proc, v, time) in model.ACTIVE_VINTAGE_TIME
        if proc == p and time == t
    )


def test_operation_capacity_single_installation():
    """
    Test that operation is correctly bounded by a single installation.

    Setup:
    - Single process producing 0.5 kg per unit and operating year (tau=1 and tau=2),
      i.e. 1.0 kg over a unit's lifetime
    - Demand of 5 kg in each of the two operating years 2021 and 2022

    Expected:
    - 10 units installed at 2020: 10 units * 0.5 kg = 5 kg per year
    - All 10 units run in both years, so operation equals the installed units
    """
    model_inputs_dict = {
        "PROCESS": ["P1"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": [],
        "ELEMENTARY_FLOW": ["CO2"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0, 1, 2, 3],  # Construction at 0, operation at 1-2, decommission at 3
        "SYSTEM_TIME": [2020, 2021, 2022, 2023, 2024],
        "CATEGORY": ["climate_change"],
        "operation_time_limits": {"P1": (1, 2)},
        "demand": {
            ("product", 2021): 5,  # Installation at 2020, tau=1 → t=2021
            ("product", 2022): 5,  # Installation at 2020, tau=2 → t=2022
        },
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        "foreground_biosphere": {("P1", "CO2", 1): 10, ("P1", "CO2", 2): 10},
        "foreground_production": {
            ("P1", "product", 1): 0.5,  # Each process produces 0.5 at tau=1
            ("P1", "product", 2): 0.5,  # and 0.5 at tau=2
        },
        "operation_flow": {
            ("P1", "product"): True,
            ("P1", "CO2"): True,
        },
        "background_inventory": {},
        "mapping": {
            ("db_2020", 2020): 1.0,
            ("db_2020", 2021): 1.0,
            ("db_2020", 2022): 1.0,
            ("db_2020", 2023): 1.0,
            ("db_2020", 2024): 1.0,
        },
        "characterization": {
            ("climate_change", "CO2", t): 1.0
            for t in [2020, 2021, 2022, 2023, 2024]
        },
    }

    # Create and solve model
    model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
    model = optimizer.create_model(
        inputs=model_inputs,
        objective_category="climate_change",
        name="test_single_installation",
    )

    solved_model, objective, results = optimizer.solve_model(
        model, solver_name="glpk", tee=False
    )

    # Verify solution is optimal
    assert results.solver.status == pyo.SolverStatus.ok
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    # Check that 10 units were installed at 2020 to meet demand
    # Each unit delivers 0.5 kg per operating year, so 5 kg/year needs 10 units.
    # Their combined lifetime output is 10 kg, matching the total demand of 10 kg.
    installed_2020 = pyo.value(solved_model.var_installation["P1", 2020])
    assert pytest.approx(10.0, rel=0.01) == installed_2020

    # Check that operation levels are bounded correctly
    # var_operation counts the units running in that year
    # At 2021: 10 units running * 0.5 kg/unit = 5 kg (demand)
    # At 2022: 10 units running * 0.5 kg/unit = 5 kg (demand)
    operation_2021 = get_total_operation(solved_model, "P1", 2021)
    operation_2022 = get_total_operation(solved_model, "P1", 2022)

    assert pytest.approx(10.0, rel=0.01) == operation_2021
    assert pytest.approx(10.0, rel=0.01) == operation_2022


def test_operation_capacity_multiple_installations():
    """
    Test that operation capacity correctly sums across multiple installations.

    Setup:
    - Single process with operation phase at tau=1,2
    - Install units at multiple years (2020, 2021)
    - High demand requiring multiple installations to operate

    Expected:
    - At year 2022, the units able to run are those installed in 2021 (at tau=1)
      and in 2020 (at tau=2)
    - Each running unit delivers 0.5 kg, so a demand of 15 needs 30 units running,
      and therefore at least 30 units installed across 2020 and 2021
    """
    model_inputs_dict = {
        "PROCESS": ["P1"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": [],
        "ELEMENTARY_FLOW": ["CO2"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0, 1, 2, 3],
        "SYSTEM_TIME": [2020, 2021, 2022, 2023],
        "CATEGORY": ["climate_change"],
        "operation_time_limits": {"P1": (1, 2)},
        "demand": {
            ("product", 2022): 15,  # High demand requiring multiple installations
        },
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        "foreground_biosphere": {("P1", "CO2", 1): 10, ("P1", "CO2", 2): 10},
        "foreground_production": {
            ("P1", "product", 1): 0.5,
            ("P1", "product", 2): 0.5,
        },
        "operation_flow": {
            ("P1", "product"): True,
            ("P1", "CO2"): True,
        },
        "background_inventory": {},
        "mapping": {("db_2020", t): 1.0 for t in [2020, 2021, 2022, 2023]},
        "characterization": {
            ("climate_change", "CO2", t): 1.0 for t in [2020, 2021, 2022, 2023]
        },
    }

    model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
    model = optimizer.create_model(
        inputs=model_inputs,
        objective_category="climate_change",
        name="test_multiple_installations",
    )

    solved_model, objective, results = optimizer.solve_model(
        model, solver_name="glpk", tee=False
    )

    assert results.solver.status == pyo.SolverStatus.ok
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    # Get installations
    install_2020 = pyo.value(solved_model.var_installation["P1", 2020])
    install_2021 = pyo.value(solved_model.var_installation["P1", 2021])

    # At 2022:
    # - Installations from 2021 are at tau=1, delivering 0.5 kg each
    # - Installations from 2020 are at tau=2, delivering 0.5 kg each
    # Total annual capacity = (install_2020 + install_2021) * 0.5

    # Get operation at 2022
    operation_2022 = get_total_operation(solved_model, "P1", 2022)

    # 30 units must run to deliver the demand of 15 kg
    assert pytest.approx(30.0, rel=0.01) == operation_2022

    # Total installations must cover those 30 running units
    total_installations = install_2020 + install_2021
    assert total_installations >= 29.9  # Allow small tolerance


def test_operation_capacity_with_varying_demand():
    """
    Test that operation adjusts to demand while respecting capacity constraints.

    Setup:
    - Install capacity upfront
    - Varying demand across time periods

    Expected:
    - Operation should match demand when demand < capacity
    - Operation should be bounded by capacity when demand > capacity

    With the capacity constraint:
    - A running unit delivers 1.0 (production at tau=1 and tau=2 is 1.0 each)
    - Units able to run at t=2021: those installed in 2020 (at tau=1)
    - Units able to run at t=2022: install_2020 (at tau=2) + install_2021 (at tau=1)
    """
    model_inputs_dict = {
        "PROCESS": ["P1"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": [],
        "ELEMENTARY_FLOW": ["CO2"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0, 1, 2],
        "SYSTEM_TIME": [2020, 2021, 2022, 2023],
        "CATEGORY": ["climate_change"],
        "operation_time_limits": {"P1": (1, 2)},
        "demand": {
            ("product", 2021): 3,   # Low demand
            ("product", 2022): 8,   # High demand
        },
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        "foreground_biosphere": {("P1", "CO2", 1): 10, ("P1", "CO2", 2): 10},
        "foreground_production": {
            ("P1", "product", 1): 1.0,  # Simpler: 1.0 per process time
            ("P1", "product", 2): 1.0,
        },
        "operation_flow": {
            ("P1", "product"): True,
            ("P1", "CO2"): True,
        },
        "background_inventory": {},
        "mapping": {("db_2020", t): 1.0 for t in [2020, 2021, 2022, 2023]},
        "characterization": {
            ("climate_change", "CO2", t): 1.0 for t in [2020, 2021, 2022, 2023]
        },
    }

    model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
    model = optimizer.create_model(
        inputs=model_inputs,
        objective_category="climate_change",
        name="test_varying_demand",
    )

    solved_model, objective, results = optimizer.solve_model(
        model, solver_name="glpk", tee=False
    )

    assert results.solver.status == pyo.SolverStatus.ok

    # Check that operations match demands
    # With production of 1.0 per unit and operating year:
    # - To produce 3.0, 3 units must run
    # - To produce 8.0, 8 units must run
    operation_2021 = get_total_operation(solved_model, "P1", 2021)
    operation_2022 = get_total_operation(solved_model, "P1", 2022)

    assert pytest.approx(3.0, rel=0.01) == operation_2021  # 3 units * 1.0 = 3.0
    assert pytest.approx(8.0, rel=0.01) == operation_2022  # 8 units * 1.0 = 8.0

    # Verify capacity constraint was respected
    install_2020 = pyo.value(solved_model.var_installation["P1", 2020])
    install_2021 = pyo.value(solved_model.var_installation["P1", 2021])

    # Capacity constraint: var_operation <= installed units of that vintage
    # At 2021: only the 2020 vintage can run -> install_2020 >= 3.0
    # At 2022: both vintages can run -> install_2020 + install_2021 >= 8.0

    total_installations = install_2020 + install_2021
    assert total_installations >= 7.9  # At least 8 units total needed
    assert install_2020 >= 2.9  # At least 3 units to serve the 2021 demand


def test_operation_capacity_constraint_violation_prevented():
    """
    Test that the model prevents operation from exceeding installed capacity.

    Setup:
    - Install capacity at 2020 and 2021
    - Demand at 2022 that requires multiple installations

    Expected:
    - Model should be feasible by installing at both years
    - Operation should be bounded by total capacity from both installations

    With the capacity constraint:
    - A running unit delivers 1.0 per year (production 1.0 at tau=1 and tau=2)
    - At t=2022 the runnable units are install_2020 (at tau=2) + install_2021 (at tau=1)
    - To meet a demand of 8, 8 units must run at 2022
    - Per-year deployment is capped at 5, so neither vintage can cover it alone and
      the model must install in both years
    """
    model_inputs_dict = {
        "PROCESS": ["P1"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": [],
        "ELEMENTARY_FLOW": ["CO2"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0, 1, 2],
        "SYSTEM_TIME": [2020, 2021, 2022],
        "CATEGORY": ["climate_change"],
        "operation_time_limits": {"P1": (1, 2)},
        "demand": {
            ("product", 2022): 8,  # Demand that can be met with both installations
        },
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        "foreground_biosphere": {("P1", "CO2", 1): 10, ("P1", "CO2", 2): 10},
        "foreground_production": {
            ("P1", "product", 1): 1.0,
            ("P1", "product", 2): 1.0,
        },
        "operation_flow": {
            ("P1", "product"): True,
            ("P1", "CO2"): True,
        },
        "background_inventory": {},
        "mapping": {("db_2020", t): 1.0 for t in [2020, 2021, 2022]},
        "characterization": {
            ("climate_change", "CO2", t): 1.0 for t in [2020, 2021, 2022]
        },
        # Limit each year's installation to force distribution
        "process_deployment_limits_max": {
            ("P1", 2020): 5,  # Can install max 5 at 2020
            ("P1", 2021): 5,  # Can install max 5 at 2021
        },
    }

    model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
    model = optimizer.create_model(
        inputs=model_inputs,
        objective_category="climate_change",
        name="test_capacity_limit",
    )

    solved_model, objective, results = optimizer.solve_model(
        model, solver_name="glpk", tee=False
    )

    assert results.solver.status == pyo.SolverStatus.ok

    # Get installations
    install_2020 = pyo.value(solved_model.var_installation["P1", 2020])
    install_2021 = pyo.value(solved_model.var_installation["P1", 2021])

    # At 2022:
    # - Installations from 2021 are at tau=1
    # - Installations from 2020 are at tau=2
    # - Each running unit delivers 1.0, so 8 units must run to produce 8

    operation_2022 = get_total_operation(solved_model, "P1", 2022)

    assert pytest.approx(8.0, rel=0.01) == operation_2022

    # 8 units must be installed, and with a cap of 5 per year both vintages are used
    total_installations = install_2020 + install_2021
    assert total_installations >= 7.9
    assert install_2020 > 0 and install_2021 > 0, (
        "The per-year deployment cap of 5 forces installation in both years, "
        f"got 2020={install_2020}, 2021={install_2021}"
    )


def test_operation_capacity_with_non_constant_production():
    """
    Test operation capacity when production varies across process times.

    NOTE: This currently will fail validation due to validate_constant_operation_flows,
    but tests that the constraint logic would work correctly if that validation were removed.
    """
    pytest.skip("Skipping: non-constant operation flows not currently supported")

    # This test would verify that the fixed constraint correctly handles
    # varying production amounts across process times


def test_operation_capacity_edge_case_zero_installation():
    """
    Test that operation correctly handles zero installations.

    When no capacity is installed, operation should be forced to zero.
    """
    model_inputs_dict = {
        "PROCESS": ["P1"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": [],
        "ELEMENTARY_FLOW": ["CO2"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0, 1, 2],
        "SYSTEM_TIME": [2020, 2021, 2022],
        "CATEGORY": ["climate_change"],
        "operation_time_limits": {"P1": (1, 2)},
        "demand": {
            ("product", 2021): 0,  # Zero demand
        },
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        "foreground_biosphere": {("P1", "CO2", 1): 10, ("P1", "CO2", 2): 10},
        "foreground_production": {
            ("P1", "product", 1): 1.0,
            ("P1", "product", 2): 1.0,
        },
        "operation_flow": {
            ("P1", "product"): True,
            ("P1", "CO2"): True,
        },
        "background_inventory": {},
        "mapping": {("db_2020", t): 1.0 for t in [2020, 2021, 2022]},
        "characterization": {
            ("climate_change", "CO2", t): 1.0 for t in [2020, 2021, 2022]
        },
        # Force no installations
        "process_deployment_limits_max": {("P1", 2020): 0},
    }

    model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
    model = optimizer.create_model(
        inputs=model_inputs,
        objective_category="climate_change",
        name="test_zero_installation",
    )

    solved_model, objective, results = optimizer.solve_model(
        model, solver_name="glpk", tee=False
    )

    assert results.solver.status == pyo.SolverStatus.ok

    # No installations should occur
    install_2020 = pyo.value(solved_model.var_installation["P1", 2020])
    assert pytest.approx(0.0, abs=1e-6) == install_2020

    # Operation should be zero (no capacity)
    operation_2021 = get_total_operation(solved_model, "P1", 2021)
    assert pytest.approx(0.0, abs=1e-6) == operation_2021


def test_operation_bounds_validation_function():
    """
    Test the validate_operation_bounds function.

    This ensures that the validation function correctly identifies
    when operation levels respect installed capacity.
    """
    model_inputs_dict = {
        "PROCESS": ["P1"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": [],
        "ELEMENTARY_FLOW": ["CO2"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0, 1, 2],
        "SYSTEM_TIME": [2020, 2021, 2022],
        "CATEGORY": ["climate_change"],
        "operation_time_limits": {"P1": (1, 2)},
        "demand": {
            ("product", 2021): 5,
        },
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        "foreground_biosphere": {("P1", "CO2", 1): 10, ("P1", "CO2", 2): 10},
        "foreground_production": {
            ("P1", "product", 1): 1.0,
            ("P1", "product", 2): 1.0,
        },
        "operation_flow": {
            ("P1", "product"): True,
            ("P1", "CO2"): True,
        },
        "background_inventory": {},
        "mapping": {("db_2020", t): 1.0 for t in [2020, 2021, 2022]},
        "characterization": {
            ("climate_change", "CO2", t): 1.0 for t in [2020, 2021, 2022]
        },
    }

    model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
    model = optimizer.create_model(
        inputs=model_inputs,
        objective_category="climate_change",
        name="test_validation",
    )

    solved_model, objective, results = optimizer.solve_model(
        model, solver_name="glpk", tee=False
    )

    # Run validation
    validation_result = optimizer.validate_operation_bounds(solved_model)

    # Should be valid (operation fractions within [0, 1])
    assert validation_result["valid"], validation_result["summary"]
    assert validation_result["max_violation"] == 0.0
    assert len(validation_result["violations"]) == 0
