import pyomo.environ as pyo

from optimex import converter


def assert_relative_error(actual, expected, tolerance=1e-2):
    epsilon = 1e-5
    if abs(expected) < epsilon:
        # Near-zero expected value, fall back to absolute error check
        assert abs(actual) < tolerance, (
            f"Expected near zero (|{expected:.5e}| < {epsilon}), "
            f"but actual value {actual:.5f} exceeds absolute tolerance {tolerance:.5f}."
        )
    else:
        # Use relative error check otherwise
        relative_error = abs(actual - expected) / abs(expected)
        assert relative_error < tolerance, (
            f"Relative error {relative_error:.5%} exceeds tolerance {tolerance:.5%}. "
            f"Expected {expected:.5f}, got {actual:.5f}."
        )


def test_dict_converts_to_modelinputs(abstract_system_model_inputs):
    model_inputs = converter.ModelInputs(**abstract_system_model_inputs)
    assert isinstance(model_inputs, converter.ModelInputs)


def test_pyomo_model_generation(abstract_system_model):
    assert isinstance(abstract_system_model, pyo.ConcreteModel)


def test_all_sets_init(abstract_system_model, abstract_system_model_inputs):
    # List of sets to test
    sets_to_test = [
        ("PROCESS", "PROCESS"),
        ("FUNCTIONAL_FLOW", "FUNCTIONAL_FLOW"),
        ("INTERMEDIATE_FLOW", "INTERMEDIATE_FLOW"),
        ("ELEMENTARY_FLOW", "ELEMENTARY_FLOW"),
        ("BACKGROUND_ID", "BACKGROUND_ID"),
        ("PROCESS_TIME", "PROCESS_TIME"),
        ("SYSTEM_TIME", "SYSTEM_TIME"),
    ]

    for model_set_name, input_set_name in sets_to_test:
        model_set = getattr(abstract_system_model, model_set_name)
        input_set = abstract_system_model_inputs[input_set_name]

        assert set(model_set) == set(
            input_set
        ), f"Set {model_set_name} does not match expected input {input_set_name}"


def test_all_params(abstract_system_model, abstract_system_model_inputs):
    model = abstract_system_model
    inputs = abstract_system_model_inputs

    # List of param names that should be checked
    param_names = [
        "demand",
        "foreground_technosphere",
        "foreground_biosphere",
        "foreground_production",
        "background_inventory",
        "mapping",
        "characterization",
        "process_limits_max",
        "process_limits_min",
        "cumulative_process_limits_max",
        "cumulative_process_limits_min",
        "process_coupling",
    ]

    for name in param_names:
        param = getattr(model, name)
        input_data = inputs.get(name, {})

        # Check initialized values
        if input_data:
            for key, value in input_data.items():
                assert (
                    pyo.value(param[key]) == value
                ), f"Param '{name}' at {key} does not match input value."

        # Check if default zero values are properly filled for other entries
        if not input_data:  # No data in input for this parameter
            if name in [
                "process_limits_max",
                "cumulative_process_limits_max",
            ]:
                for key in param:
                    assert pyo.value(param[key]) == float("inf"), (
                        f"Param '{name}' at {key} should be 'inf' but was "
                        f"{pyo.value(param[key])}."
                    )
            else:
                for key in param:
                    assert pyo.value(param[key]) == 0, (
                        f"Param '{name}' at {key} should be 0 but was "
                        f"{pyo.value(param[key])}."
                    )


def test_model_solution_is_optimal(solved_system_model):
    _, results = solved_system_model
    assert results.solver.status == pyo.SolverStatus.ok, "Solver did not exit normally."
    assert (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ), "Solution is not optimal."


def test_model_objective_in_tolerance(solved_system_model):
    model, _ = solved_system_model

    expected_objective = 8.99602e02
    actual_objective = pyo.value(model.OBJ)

    assert_relative_error(actual_objective, expected_objective)


def test_model_scaling_values_within_tolerance(solved_system_model):
    model, _ = solved_system_model

    expected_values = {
        ("P1", 2025): 10.00,
        ("P1", 2027): 10.00,
        ("P2", 2021): 10.00,
        ("P2", 2023): 10.00,
    }

    # Check non-zero expected values are within tolerance
    for (process, start_time), expected in expected_values.items():
        actual = pyo.value(model.var_installation[process, start_time])
        assert_relative_error(actual, expected)

    # Check all other values are close to zero
    for process in model.PROCESS:
        for time in model.SYSTEM_TIME:
            if (process, time) not in expected_values:
                actual = pyo.value(model.var_installation[process, time])
                assert_relative_error(actual, 0)
