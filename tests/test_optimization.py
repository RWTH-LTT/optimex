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
        ("REFERENCE_PRODUCT", "REFERENCE_PRODUCT"),
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
    scaled_inputs, _ = raw.get_scaled_copy()

    # 2) Build the model using scaled_inputs
    model = optimizer.create_model(
        inputs=raw,
        objective_category="climate_change",
        name="test_model",
        flexible_operation=False,
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
        ("flex", 2.81685e-10),  # Expected value for the flexible model
        ("constrained", 2.83062e-10),  # Expected value for the constrained model
    ],
    ids=["flex_result", "constrained_result"], # "fixed_result", 
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
            ("P1", 2025): 10.00,
            ("P1", 2027): 10.00,
            ("P2", 2021): 10.00,
            ("P2", 2023): 10.00,
        }
    elif model.name == "abstract_system_model_constrained":
        expected_values = {
            ("P1", 2027): 5.44/2,
            ("P2", 2021): 20.00/2,
            ("P2", 2023): 20.00/2,
            ("P2", 2025): 20.00/2,
            ("P2", 2027): 14.56/2,
        }
    else:
        pytest.skip(f"Unknown model name: {model.name}")
        
    fg_scale = getattr(model, "scales", {}).get("foreground", 1.0)
    
    # Check non-zero expected values are within tolerance
    for (process, start_time), expected in expected_values.items():
        actual = pyo.value(model.var_installation[process, start_time])
        assert pytest.approx(expected / fg_scale, rel=1e-2) == actual, (
            f"Installation value for {process} at {start_time} "
            f"should be {expected / fg_scale} but was {actual}."
        )

    # Check all other values are close to zero
    for process in model.PROCESS:
        for time in model.SYSTEM_TIME:
            if (process, time) not in expected_values:
                actual = pyo.value(model.var_installation[process, time])
                assert pytest.approx(0, rel=1e-2) == actual, (
                    f"Installation value for {process} at {time} "
                    f"should be 0 but was {actual}."
                )
