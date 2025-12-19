import pyomo.environ as pyo
import pytest

from optimex import converter, optimizer


def test_dict_converts_to_modelinputs(abstract_system_model_inputs):
    model_inputs = converter.OptimizationModelInputs(**abstract_system_model_inputs)
    assert isinstance(model_inputs, converter.OptimizationModelInputs)


def test_pyomo_model_generation(abstract_system_model):
    assert isinstance(abstract_system_model, pyo.ConcreteModel)


def test_all_sets_init(abstract_system_model, abstract_system_model_inputs):
    # List of sets to test
    sets_to_test = [
        ("PROCESS", "PROCESS"),
        ("PRODUCT", "PRODUCT"),
        ("INTERMEDIATE_FLOW", "INTERMEDIATE_FLOW"),
        ("ELEMENTARY_FLOW", "ELEMENTARY_FLOW"),
        ("BACKGROUND_ID", "BACKGROUND_ID"),
        ("PROCESS_TIME", "PROCESS_TIME"),
        ("SYSTEM_TIME", "SYSTEM_TIME"),
        ("CATEGORY", "CATEGORY"),
    ]

    for model_set_name, input_set_name in sets_to_test:
        model_set = getattr(abstract_system_model, model_set_name)
        input_set = abstract_system_model_inputs[input_set_name]

        assert set(model_set) == set(
            input_set
        ), f"Set {model_set_name} does not match expected input {input_set_name}"


def test_all_params_scaled(abstract_system_model_inputs):
    # 1) Prepare scaled inputs exactly as your fixture does
    raw = converter.OptimizationModelInputs(**abstract_system_model_inputs)
    scaled_inputs, scales = raw.get_scaled_copy()

    # 2) Build the model using scaled_inputs
    model = optimizer.create_model(
        inputs=raw,
        objective_category="climate_change",
        name="test_model",
        flexible_operation=True,
    )

    # 3) Now assert model.param == scaled_inputs.param
    param_names = [
        "demand",
        "foreground_technosphere",
        "foreground_biosphere",
        "foreground_production",
        "background_inventory",
        "mapping",
        "characterization",
        "operation_flow",
    ]
    for name in param_names:
        model_param = getattr(model, name)
        expected_dict = getattr(scaled_inputs, name) or {}
        for key, exp in expected_dict.items():
            obs = pyo.value(model_param[key])
            assert (
                pytest.approx(obs, rel=1e-9) == exp
            ), f"Scaled param '{name}'[{key}] was {obs}, expected {exp}"


def test_model_solution_is_optimal(solved_system_model):
    _, _, results = solved_system_model
    assert results.solver.status == pyo.SolverStatus.ok, (
        f"Solver status is '{results.solver.status}', expected 'ok'. "
        "The solver did not exit normally."
    )
    assert results.solver.termination_condition in [
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.unknown,
    ], (
        f"Solver termination condition is '{results.solver.termination_condition}', "
        "expected 'optimal' or 'unknown'."
    )


@pytest.mark.parametrize(
    "model_type, expected_value",
    [
        # ("fixed", 3.15417e-10),  # Expected value for the fixed model
        ("flex", 1.9172462799736082e-10),  # Expected value for the flexible model
        ("constrained", 1.9197314619763485e-10),  # Constrained by limiting P1 to 10 installations (0.13% higher)
    ],
    ids=["flex_result", "constrained_process_limit"], # "fixed_result",
)
def test_system_model(model_type, expected_value, solved_system_model):
    # Get the model from the solved system model fixture
    model, objective, _ = solved_system_model

    model_name = model.name
    expected_name = f"abstract_system_model_{model_type}"
    # Only run the test if the model type matches the result type
    if model_name != expected_name:
        pytest.skip()
    # Assert that the objective value is approximately equal to the expected value
    assert pytest.approx(expected_value, rel=1e-4) == objective, (
        f"Objective value for {model_type} model should be {expected_value} "
        f"but was {objective}."
    )


def test_model_scaling_values_within_tolerance(solved_system_model):
    model, _, _ = solved_system_model

    if (
        model.name == "abstract_system_model_fixed"
        or model.name == "abstract_system_model_flex"
    ):
        expected_values = {
            ("P1", 2025): 10.0,
            ("P1", 2027): 10.0,
            ("P2", 2021): 10.0,
            ("P2", 2023): 10.0,
        }
    elif model.name == "abstract_system_model_constrained":
        # With P1 limited to 10 total, optimizer shifts significantly more to P2
        expected_values = {
            ("P1", 2027): 10.0,
            ("P2", 2021): 10.0,
            ("P2", 2023): 10.0,
            ("P2", 2025): 10.0,
        }
    else:
        pytest.skip(f"Unknown model name: {model.name}")

    # var_installation is in real units, so compare directly
    for (process, start_time), expected in expected_values.items():
        actual = pyo.value(model.var_installation[process, start_time])
        assert pytest.approx(expected, rel=1e-2) == actual, (
            f"Installation value for {process} at {start_time} "
            f"should be {expected} but was {actual}."
        )

    # Check all other values are close to zero
    for process in model.PROCESS:
        for time in model.SYSTEM_TIME:
            if (process, time) not in expected_values:
                actual = pyo.value(model.var_installation[process, time])
                assert pytest.approx(0, abs=1e-2) == actual, (
                    f"Installation value for {process} at {time} "
                    f"should be 0 but was {actual}."
                )


def test_cumulative_process_limits_respected():
    """
    Test that cumulative process limits are correctly applied and respected.

    Creates a model with two processes:
    - P1 (low emissions): preferred by optimizer
    - P2 (high emissions): avoided unless necessary

    Sets a cumulative limit on P1 to force P2 usage.
    Verifies that actual cumulative installation matches the limit.
    """
    # Create model inputs with two processes with different emission levels
    model_inputs_dict = {
        "PROCESS": ["P1_low_emission", "P2_high_emission"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": [],
        "ELEMENTARY_FLOW": ["CO2"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0],  # Simplified to single time step
        "SYSTEM_TIME": [2020, 2021, 2022, 2023, 2024],
        "CATEGORY": ["climate_change"],
        "operation_time_limits": {
            "P1_low_emission": (0, 0),
            "P2_high_emission": (0, 0),
        },
        "demand": {
            ("product", 2020): 10,
            ("product", 2021): 10,
            ("product", 2022): 10,
            ("product", 2023): 10,
            ("product", 2024): 10,
        },
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        # P1 has low emissions (5 kg CO2), P2 has high emissions (20 kg CO2)
        "foreground_biosphere": {
            ("P1_low_emission", "CO2", 0): 5,
            ("P2_high_emission", "CO2", 0): 20,
        },
        # Both processes produce 1 unit of product per unit of operation
        "foreground_production": {
            ("P1_low_emission", "product", 0): 1.0,
            ("P2_high_emission", "product", 0): 1.0,
        },
        "operation_flow": {
            ("P1_low_emission", "product"): True,
            ("P1_low_emission", "CO2"): True,
            ("P2_high_emission", "product"): True,
            ("P2_high_emission", "CO2"): True,
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
            ("climate_change", "CO2", 2020): 1.0,
            ("climate_change", "CO2", 2021): 1.0,
            ("climate_change", "CO2", 2022): 1.0,
            ("climate_change", "CO2", 2023): 1.0,
            ("climate_change", "CO2", 2024): 1.0,
        },
    }

    # Create model inputs
    model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)

    # Set cumulative limit on low-emission process (30 units total across all years)
    # Total demand is 50, so P2 must provide at least 20
    cumulative_limit = 30.0
    model_inputs.cumulative_process_limits_max = {
        "P1_low_emission": cumulative_limit
    }

    # Create and solve model
    model = optimizer.create_model(
        inputs=model_inputs,
        objective_category="climate_change",
        name="test_cumulative_limits",
        flexible_operation=True,
    )

    solved_model, objective, results = optimizer.solve_model(
        model,
        solver_name="glpk",
        tee=False
    )

    # Verify solution is optimal
    assert results.solver.status == pyo.SolverStatus.ok
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    # Calculate cumulative installation for P1 (already in real units)
    cumulative_p1 = sum(
        pyo.value(solved_model.var_installation["P1_low_emission", t])
        for t in solved_model.SYSTEM_TIME
    )

    # Calculate cumulative installation for P2 (already in real units)
    cumulative_p2 = sum(
        pyo.value(solved_model.var_installation["P2_high_emission", t])
        for t in solved_model.SYSTEM_TIME
    )

    # Verify P1 respects the cumulative limit (within 1% tolerance)
    assert cumulative_p1 <= cumulative_limit * 1.01, (
        f"P1 cumulative installation ({cumulative_p1:.2f}) exceeds "
        f"limit ({cumulative_limit})"
    )

    # Verify P1 uses the full limit (should be at the limit since it's cheaper)
    assert cumulative_p1 >= cumulative_limit * 0.99, (
        f"P1 cumulative installation ({cumulative_p1:.2f}) is significantly below "
        f"limit ({cumulative_limit}), suggesting constraint is not binding"
    )

    # Verify P2 is used to meet remaining demand
    total_demand = sum(model_inputs_dict["demand"].values())
    assert cumulative_p2 > 0, (
        "P2 should be used to meet demand when P1 is limited"
    )

    # Verify total capacity roughly equals demand (within 1% tolerance)
    total_capacity = cumulative_p1 + cumulative_p2
    assert pytest.approx(total_demand, rel=0.01) == total_capacity, (
        f"Total capacity ({total_capacity:.2f}) should equal "
        f"demand ({total_demand})"
    )


def test_capacity_constraint_with_high_production():
    """
    Test that capacity constraint correctly accounts for production coefficients > 1.

    When production per installation > 1, the optimizer should be able to use
    fewer installations than the demand, since each installation produces more
    than 1 unit per operation.

    This test verifies the fix for the bug where production coefficients were
    not being multiplied in the capacity constraint, leading to suboptimal
    solutions with more installations than necessary.
    """
    model_inputs_dict = {
        "PROCESS": ["P1"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": [],
        "ELEMENTARY_FLOW": ["CO2"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0],
        "SYSTEM_TIME": [2020],
        "CATEGORY": ["climate_change"],
        "operation_time_limits": {"P1": (0, 0)},
        "demand": {("product", 2020): 10},
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        # Each installation emits 10 kg CO2 (capacity-dependent)
        "foreground_biosphere": {("P1", "CO2", 0): 10},
        # Each installation produces 5 units of product (high production)
        "foreground_production": {("P1", "product", 0): 5.0},
        # Only production is operation-dependent, emissions are capacity-dependent
        "operation_flow": {("P1", "product"): True},
        "background_inventory": {},
        "mapping": {("db_2020", 2020): 1.0},
        "characterization": {("climate_change", "CO2", 2020): 1.0},
    }

    model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
    model = optimizer.create_model(
        inputs=model_inputs,
        objective_category="climate_change",
        name="test_high_production",
        flexible_operation=True,
    )

    solved_model, objective, results = optimizer.solve_model(
        model, solver_name="glpk", tee=False
    )

    # Verify solution is optimal
    assert results.solver.status == pyo.SolverStatus.ok
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    # Get solution values
    inst = pyo.value(solved_model.var_installation["P1", 2020])
    oper = pyo.value(solved_model.var_operation["P1", 2020])

    # With production = 5.0 and demand = 10:
    # - Need var_operation = 10/5 = 2 to produce 10 units
    # - Capacity = 5.0 * installations, so need 5.0 * inst >= 2
    # - Minimum installations = 2/5 = 0.4
    assert pytest.approx(0.4, rel=0.01) == inst, (
        f"With production=5.0, only 0.4 installations needed, got {inst}"
    )
    assert pytest.approx(2.0, rel=0.01) == oper, (
        f"Operation should be 2.0 to produce 10 units, got {oper}"
    )

    # Verify emissions (capacity-dependent) = 0.4 * 10 = 4 kg CO2
    assert pytest.approx(4.0, rel=0.01) == objective, (
        f"Emissions should be 4.0 kg CO2, got {objective}"
    )


def test_operation_limits_respected():
    """
    Test that operation limits correctly cap operation when it would otherwise be higher.

    Creates a model with two processes:
    - P1 (low emissions): preferred by optimizer
    - P2 (high emissions): fallback when P1 is limited

    Sets an operation limit on P1 that forces the optimizer to use P2 to meet
    the remaining demand. Without the operation limit, P1 would be used exclusively.

    This verifies that:
    1. Operation is capped at the specified limit
    2. The limit is binding (operation reaches the limit)
    3. Remaining demand is fulfilled by the fallback process
    """
    model_inputs_dict = {
        "PROCESS": ["P1_low_emission", "P2_high_emission"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": [],
        "ELEMENTARY_FLOW": ["CO2"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0],
        "SYSTEM_TIME": [2020, 2021, 2022],
        "CATEGORY": ["climate_change"],
        "operation_time_limits": {
            "P1_low_emission": (0, 0),
            "P2_high_emission": (0, 0),
        },
        # Demand of 100 units per year
        "demand": {
            ("product", 2020): 100,
            ("product", 2021): 100,
            ("product", 2022): 100,
        },
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        # P1 has low emissions (1 kg CO2 per operation), P2 has high emissions (10 kg CO2)
        "foreground_biosphere": {
            ("P1_low_emission", "CO2", 0): 1,
            ("P2_high_emission", "CO2", 0): 10,
        },
        # Both processes produce 1 unit of product per unit of operation
        "foreground_production": {
            ("P1_low_emission", "product", 0): 1.0,
            ("P2_high_emission", "product", 0): 1.0,
        },
        "operation_flow": {
            ("P1_low_emission", "product"): True,
            ("P1_low_emission", "CO2"): True,
            ("P2_high_emission", "product"): True,
            ("P2_high_emission", "CO2"): True,
        },
        "background_inventory": {},
        "mapping": {
            ("db_2020", 2020): 1.0,
            ("db_2020", 2021): 1.0,
            ("db_2020", 2022): 1.0,
        },
        "characterization": {
            ("climate_change", "CO2", 2020): 1.0,
            ("climate_change", "CO2", 2021): 1.0,
            ("climate_change", "CO2", 2022): 1.0,
        },
    }

    # First solve WITHOUT operation limits to establish baseline
    model_inputs_baseline = converter.OptimizationModelInputs(**model_inputs_dict)
    model_baseline = optimizer.create_model(
        inputs=model_inputs_baseline,
        objective_category="climate_change",
        name="test_operation_limits_baseline",
        flexible_operation=True,
    )
    solved_baseline, obj_baseline, _ = optimizer.solve_model(
        model_baseline, solver_name="glpk", tee=False
    )

    # Without limits, P1 should handle all demand (300 total)
    p1_operation_baseline = sum(
        pyo.value(solved_baseline.var_operation["P1_low_emission", t])
        for t in solved_baseline.SYSTEM_TIME
    )
    p2_operation_baseline = sum(
        pyo.value(solved_baseline.var_operation["P2_high_emission", t])
        for t in solved_baseline.SYSTEM_TIME
    )
    assert pytest.approx(300, rel=0.01) == p1_operation_baseline, (
        f"Without limits, P1 should handle all demand (300), got {p1_operation_baseline}"
    )
    assert pytest.approx(0, abs=0.01) == p2_operation_baseline, (
        f"Without limits, P2 should not be used, got {p2_operation_baseline}"
    )

    # Now solve WITH operation limits on P1
    # Limit P1 operation to 50 per year (150 total), forcing P2 to handle the rest
    operation_limit = 50.0
    model_inputs_limited = converter.OptimizationModelInputs(**model_inputs_dict)
    model_inputs_limited.process_operation_limits_max = {
        ("P1_low_emission", 2020): operation_limit,
        ("P1_low_emission", 2021): operation_limit,
        ("P1_low_emission", 2022): operation_limit,
    }

    model_limited = optimizer.create_model(
        inputs=model_inputs_limited,
        objective_category="climate_change",
        name="test_operation_limits_constrained",
        flexible_operation=True,
    )
    solved_limited, obj_limited, results = optimizer.solve_model(
        model_limited, solver_name="glpk", tee=False
    )

    # Verify solution is optimal
    assert results.solver.status == pyo.SolverStatus.ok
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    # Check operation values for each year
    for t in [2020, 2021, 2022]:
        p1_op = pyo.value(solved_limited.var_operation["P1_low_emission", t])
        p2_op = pyo.value(solved_limited.var_operation["P2_high_emission", t])

        # P1 should be at the limit (binding constraint)
        assert p1_op <= operation_limit * 1.001, (
            f"P1 operation at {t} ({p1_op:.2f}) exceeds limit ({operation_limit})"
        )
        assert p1_op >= operation_limit * 0.999, (
            f"P1 operation at {t} ({p1_op:.2f}) should be at limit ({operation_limit}), "
            "constraint may not be binding"
        )

        # P2 should handle the remaining demand (100 - 50 = 50)
        expected_p2 = 100 - operation_limit
        assert pytest.approx(expected_p2, rel=0.01) == p2_op, (
            f"P2 operation at {t} should be {expected_p2}, got {p2_op}"
        )

    # Verify that the limited solution has higher emissions
    # Baseline: 300 * 1 = 300 kg CO2
    # Limited: 150 * 1 + 150 * 10 = 150 + 1500 = 1650 kg CO2
    assert obj_limited > obj_baseline, (
        f"Limited objective ({obj_limited}) should be higher than baseline ({obj_baseline})"
    )
    assert pytest.approx(1650, rel=0.01) == obj_limited, (
        f"Limited objective should be 1650 kg CO2, got {obj_limited}"
    )
