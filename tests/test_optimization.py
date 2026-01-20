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
    )

    solved_model, objective, results = optimizer.solve_model(
        model, solver_name="glpk", tee=False
    )

    # Verify solution is optimal
    assert results.solver.status == pyo.SolverStatus.ok
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    # Get solution values
    inst = pyo.value(solved_model.var_installation["P1", 2020])
    oper = get_total_operation(solved_model, "P1", 2020)

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
    )
    solved_baseline, obj_baseline, _ = optimizer.solve_model(
        model_baseline, solver_name="glpk", tee=False
    )

    # Without limits, P1 should handle all demand (300 total)
    p1_operation_baseline = sum(
        get_total_operation(solved_baseline, "P1_low_emission", t)
        for t in solved_baseline.SYSTEM_TIME
    )
    p2_operation_baseline = sum(
        get_total_operation(solved_baseline, "P2_high_emission", t)
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
    )
    solved_limited, obj_limited, results = optimizer.solve_model(
        model_limited, solver_name="glpk", tee=False
    )

    # Verify solution is optimal
    assert results.solver.status == pyo.SolverStatus.ok
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    # Check operation values for each year
    for t in [2020, 2021, 2022]:
        p1_op = get_total_operation(solved_limited, "P1_low_emission", t)
        p2_op = get_total_operation(solved_limited, "P2_high_emission", t)

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


def test_cumulative_category_impact_limit_respected():
    """
    Test that cumulative category impact limits are correctly applied and respected.

    Creates a model with two processes:
    - P1 (low CO2, high land use): preferred for climate, bad for land
    - P2 (high CO2, low land use): worse for climate, better for land

    Optimizes for climate_change but limits cumulative land_use.
    Verifies that the land_use limit forces P2 usage even though it has higher CO2.
    """
    model_inputs_dict = {
        "PROCESS": ["P1_low_co2", "P2_high_co2"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": [],
        "ELEMENTARY_FLOW": ["CO2", "land"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0],
        "SYSTEM_TIME": [2020, 2021, 2022, 2023, 2024],
        "CATEGORY": ["climate_change", "land_use"],
        "operation_time_limits": {
            "P1_low_co2": (0, 0),
            "P2_high_co2": (0, 0),
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
        # P1: low CO2 (1), high land use (10)
        # P2: high CO2 (5), low land use (1)
        "foreground_biosphere": {
            ("P1_low_co2", "CO2", 0): 1,
            ("P1_low_co2", "land", 0): 10,
            ("P2_high_co2", "CO2", 0): 5,
            ("P2_high_co2", "land", 0): 1,
        },
        "foreground_production": {
            ("P1_low_co2", "product", 0): 1.0,
            ("P2_high_co2", "product", 0): 1.0,
        },
        "operation_flow": {
            ("P1_low_co2", "product"): True,
            ("P1_low_co2", "CO2"): True,
            ("P1_low_co2", "land"): True,
            ("P2_high_co2", "product"): True,
            ("P2_high_co2", "CO2"): True,
            ("P2_high_co2", "land"): True,
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
            ("climate_change", "land", 2020): 0.0,
            ("climate_change", "land", 2021): 0.0,
            ("climate_change", "land", 2022): 0.0,
            ("climate_change", "land", 2023): 0.0,
            ("climate_change", "land", 2024): 0.0,
            ("land_use", "CO2", 2020): 0.0,
            ("land_use", "CO2", 2021): 0.0,
            ("land_use", "CO2", 2022): 0.0,
            ("land_use", "CO2", 2023): 0.0,
            ("land_use", "CO2", 2024): 0.0,
            ("land_use", "land", 2020): 1.0,
            ("land_use", "land", 2021): 1.0,
            ("land_use", "land", 2022): 1.0,
            ("land_use", "land", 2023): 1.0,
            ("land_use", "land", 2024): 1.0,
        },
    }

    # First solve WITHOUT category limit to establish baseline
    model_inputs_baseline = converter.OptimizationModelInputs(**model_inputs_dict)
    model_baseline = optimizer.create_model(
        inputs=model_inputs_baseline,
        objective_category="climate_change",
        name="test_category_limit_baseline",
    )
    solved_baseline, obj_baseline, _ = optimizer.solve_model(
        model_baseline, solver_name="glpk", tee=False
    )

    # Without limits, P1 should handle all demand (lowest CO2)
    # Total demand = 50, all from P1 -> CO2 = 50, land_use = 500
    p1_op_baseline = sum(
        get_total_operation(solved_baseline, "P1_low_co2", t)
        for t in solved_baseline.SYSTEM_TIME
    )
    assert pytest.approx(50, rel=0.01) == p1_op_baseline, (
        f"Without limits, P1 should handle all demand (50), got {p1_op_baseline}"
    )
    assert pytest.approx(50, rel=0.01) == obj_baseline, (
        f"Baseline climate impact should be 50, got {obj_baseline}"
    )

    # Calculate baseline land_use impact
    baseline_land_use = (
        pyo.value(solved_baseline.total_impact["land_use"])
        * solved_baseline.scales["foreground"]
        * solved_baseline.scales["characterization"]["land_use"]
    )
    assert pytest.approx(500, rel=0.01) == baseline_land_use, (
        f"Baseline land_use should be 500, got {baseline_land_use}"
    )

    # Now solve WITH category impact limit on land_use
    # Limit land_use to 200 (requires 30 units from P2)
    # P1 produces 10 land per unit, P2 produces 1 land per unit
    # If P1 handles x units and P2 handles (50-x) units:
    # land_use = 10*x + 1*(50-x) = 9x + 50 <= 200
    # => x <= 150/9 = 16.67
    # So P1 max ~16.67 units, P2 handles ~33.33 units
    land_use_limit = 200.0

    model_inputs_limited = converter.OptimizationModelInputs(**model_inputs_dict)
    model_inputs_limited.cumulative_category_impact_limits = {"land_use": land_use_limit}

    model_limited = optimizer.create_model(
        inputs=model_inputs_limited,
        objective_category="climate_change",
        name="test_category_limit_constrained",
    )
    solved_limited, obj_limited, results = optimizer.solve_model(
        model_limited, solver_name="glpk", tee=False
    )

    # Verify solution is optimal
    assert results.solver.status == pyo.SolverStatus.ok
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    # Calculate actual land_use impact (denormalize)
    actual_land_use = (
        pyo.value(solved_limited.total_impact["land_use"])
        * solved_limited.scales["foreground"]
        * solved_limited.scales["characterization"]["land_use"]
    )

    # Verify land_use respects the limit (within 1% tolerance)
    assert actual_land_use <= land_use_limit * 1.01, (
        f"Actual land_use ({actual_land_use:.2f}) exceeds limit ({land_use_limit})"
    )

    # Verify land_use is at or near the limit (constraint should be binding)
    assert actual_land_use >= land_use_limit * 0.99, (
        f"Actual land_use ({actual_land_use:.2f}) is significantly below "
        f"limit ({land_use_limit}), suggesting constraint is not binding"
    )

    # Verify the limited solution has higher climate impact (forced to use P2)
    assert obj_limited > obj_baseline, (
        f"Limited objective ({obj_limited}) should be higher than baseline ({obj_baseline})"
    )

    # Expected: P1 uses ~16.67, P2 uses ~33.33
    # With land_use constraint: 10*x + 1*(50-x) <= 200  =>  x <= 150/9 = 16.67
    # Climate impact = 1*x + 5*(50-x) = 250 - 4x = 250 - 4*(150/9) = 183.33
    expected_climate = 250 - 4 * (150 / 9)  # ~183.33
    assert pytest.approx(expected_climate, rel=0.01) == obj_limited, (
        f"Limited climate impact should be ~{expected_climate:.1f}, got {obj_limited}"
    )


def test_time_specific_category_impact_limit_respected():
    """
    Test that time-specific category impact limits constrain impact at specific times.

    Creates a model with two processes:
    - P1 (low CO2, high land use): preferred for climate, bad for land
    - P2 (high CO2, low land use): worse for climate, better for land

    Optimizes for climate_change but limits land_use at specific years.
    Verifies that the time-specific limits are respected independently.
    """
    model_inputs_dict = {
        "PROCESS": ["P1_low_co2", "P2_high_co2"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": [],
        "ELEMENTARY_FLOW": ["CO2", "land"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0],
        "SYSTEM_TIME": [2020, 2021, 2022],
        "CATEGORY": ["climate_change", "land_use"],
        "operation_time_limits": {
            "P1_low_co2": (0, 0),
            "P2_high_co2": (0, 0),
        },
        "demand": {
            ("product", 2020): 10,
            ("product", 2021): 10,
            ("product", 2022): 10,
        },
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        # P1: low CO2 (1), high land use (10)
        # P2: high CO2 (5), low land use (1)
        "foreground_biosphere": {
            ("P1_low_co2", "CO2", 0): 1,
            ("P1_low_co2", "land", 0): 10,
            ("P2_high_co2", "CO2", 0): 5,
            ("P2_high_co2", "land", 0): 1,
        },
        "foreground_production": {
            ("P1_low_co2", "product", 0): 1.0,
            ("P2_high_co2", "product", 0): 1.0,
        },
        "operation_flow": {
            ("P1_low_co2", "product"): True,
            ("P1_low_co2", "CO2"): True,
            ("P1_low_co2", "land"): True,
            ("P2_high_co2", "product"): True,
            ("P2_high_co2", "CO2"): True,
            ("P2_high_co2", "land"): True,
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
            ("climate_change", "land", 2020): 0.0,
            ("climate_change", "land", 2021): 0.0,
            ("climate_change", "land", 2022): 0.0,
            ("land_use", "CO2", 2020): 0.0,
            ("land_use", "CO2", 2021): 0.0,
            ("land_use", "CO2", 2022): 0.0,
            ("land_use", "land", 2020): 1.0,
            ("land_use", "land", 2021): 1.0,
            ("land_use", "land", 2022): 1.0,
        },
    }

    # Set time-specific limit on land_use for year 2021 only
    # Limit land_use in 2021 to 50, which requires mixing P1 and P2
    # Without limits, P1 handles all 10 units -> land_use = 100 in each year
    # With limit of 50: need x from P1, (10-x) from P2
    # 10*x + 1*(10-x) = 50 => 9x = 40 => x = 40/9 = 4.44
    land_use_limit_2021 = 50.0

    model_inputs_limited = converter.OptimizationModelInputs(**model_inputs_dict)
    model_inputs_limited.category_impact_limits = {
        ("land_use", 2021): land_use_limit_2021
    }

    model_limited = optimizer.create_model(
        inputs=model_inputs_limited,
        objective_category="climate_change",
        name="test_time_specific_category_limit",
    )
    solved_limited, obj_limited, results = optimizer.solve_model(
        model_limited, solver_name="glpk", tee=False
    )

    # Verify solution is optimal
    assert results.solver.status == pyo.SolverStatus.ok
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    # Calculate actual land_use impact for 2021 (denormalize)
    fg_scale = solved_limited.scales["foreground"]
    cat_scale = solved_limited.scales["characterization"]["land_use"]
    actual_land_use_2021 = (
        pyo.value(solved_limited.time_specific_impact["land_use", 2021])
        * fg_scale
        * cat_scale
    )

    # Verify land_use in 2021 respects the limit
    assert actual_land_use_2021 <= land_use_limit_2021 * 1.01, (
        f"Land_use in 2021 ({actual_land_use_2021:.2f}) exceeds limit ({land_use_limit_2021})"
    )

    # Verify constraint is binding (at the limit)
    assert actual_land_use_2021 >= land_use_limit_2021 * 0.99, (
        f"Land_use in 2021 ({actual_land_use_2021:.2f}) is below limit, constraint not binding"
    )

    # Verify years without limits are not constrained (should be at 100)
    actual_land_use_2020 = (
        pyo.value(solved_limited.time_specific_impact["land_use", 2020])
        * fg_scale
        * cat_scale
    )
    assert pytest.approx(100, rel=0.01) == actual_land_use_2020, (
        f"Land_use in 2020 should be 100 (all P1), got {actual_land_use_2020}"
    )

    actual_land_use_2022 = (
        pyo.value(solved_limited.time_specific_impact["land_use", 2022])
        * fg_scale
        * cat_scale
    )
    assert pytest.approx(100, rel=0.01) == actual_land_use_2022, (
        f"Land_use in 2022 should be 100 (all P1), got {actual_land_use_2022}"
    )


def test_time_specific_flow_limits_respected():
    """
    Test that time-specific flow limits correctly constrain flows at each time step.

    Tests both max and min flow limits to verify constraints are properly enforced.
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
        "demand": {
            ("product", 2020): 10,
            ("product", 2021): 10,
            ("product", 2022): 10,
        },
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        # P1 has low emissions (1 kg CO2 per unit), P2 has high emissions (5 kg CO2)
        "foreground_biosphere": {
            ("P1_low_emission", "CO2", 0): 1,
            ("P2_high_emission", "CO2", 0): 5,
        },
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

    # First solve WITHOUT flow limits to establish baseline
    model_inputs_baseline = converter.OptimizationModelInputs(**model_inputs_dict)
    model_baseline = optimizer.create_model(
        inputs=model_inputs_baseline,
        objective_category="climate_change",
        name="test_flow_limit_baseline",
    )
    solved_baseline, obj_baseline, _ = optimizer.solve_model(
        model_baseline, solver_name="glpk", tee=False
    )

    # Without limits, P1 should handle all demand (lowest emissions)
    # Total CO2 = 30 * 1 = 30 kg
    assert pytest.approx(30, rel=0.01) == obj_baseline, (
        f"Baseline emissions should be 30, got {obj_baseline}"
    )

    # Test 1: MINIMUM flow limit - force higher emissions in 2022
    # Require CO2 >= 20 in 2022 (higher than optimal of 10)
    # With demand of 10, this requires some P2 usage:
    # x + y = 10 (demand), x + 5y >= 20 (min emission)
    # y = 10 - x, so x + 5(10-x) >= 20 => -4x >= -30 => x <= 7.5
    # So P1 max 7.5 units, P2 min 2.5 units
    # Optimal: P1=7.5, P2=2.5 => CO2 = 7.5 + 12.5 = 20 (binding)
    co2_min_2022 = 20.0

    model_inputs_min = converter.OptimizationModelInputs(**model_inputs_dict)
    model_inputs_min.flow_limits_min = {
        ("CO2", 2022): co2_min_2022,
    }

    model_min = optimizer.create_model(
        inputs=model_inputs_min,
        objective_category="climate_change",
        name="test_flow_limit_min",
    )
    solved_min, obj_min, results_min = optimizer.solve_model(
        model_min, solver_name="glpk", tee=False
    )

    # Verify solution is optimal
    assert results_min.solver.status == pyo.SolverStatus.ok
    assert results_min.solver.termination_condition == pyo.TerminationCondition.optimal

    # Calculate actual CO2 emissions in 2022 (denormalize)
    fg_scale = solved_min.scales["foreground"]
    actual_co2_2022 = pyo.value(solved_min.total_elementary_flow["CO2", 2022]) * fg_scale

    # Verify CO2 meets the minimum limit
    assert actual_co2_2022 >= co2_min_2022 * 0.99, (
        f"Actual CO2 in 2022 ({actual_co2_2022:.2f}) is below min limit ({co2_min_2022})"
    )

    # Verify constraint is binding (optimizer should hit exactly the min)
    assert actual_co2_2022 <= co2_min_2022 * 1.01, (
        f"Actual CO2 in 2022 ({actual_co2_2022:.2f}) is above min limit ({co2_min_2022}), "
        "constraint should be binding"
    )

    # Verify objective is higher than baseline (forced suboptimal in one year)
    # Expected: 2 years at 10 kg + 1 year at 20 kg = 40 kg
    expected_obj = 2 * 10 + 20  # 40 kg
    assert pytest.approx(expected_obj, rel=0.01) == obj_min, (
        f"Limited objective should be {expected_obj}, got {obj_min}"
    )

    # Test 2: MAX flow limit below achievable minimum should be infeasible
    # Optimal CO2 per year is 10 kg (all P1). Limit to 8 kg should be infeasible.
    model_inputs_infeasible = converter.OptimizationModelInputs(**model_inputs_dict)
    model_inputs_infeasible.flow_limits_max = {
        ("CO2", 2022): 8.0,  # Below optimal of 10, forces impossible constraint
    }

    model_infeasible = optimizer.create_model(
        inputs=model_inputs_infeasible,
        objective_category="climate_change",
        name="test_flow_limit_infeasible",
    )
    solver = pyo.SolverFactory("glpk")
    results_infeasible = solver.solve(model_infeasible, tee=False)

    # Model should be infeasible since min achievable CO2 (10) > limit (8)
    assert results_infeasible.solver.termination_condition in [
        pyo.TerminationCondition.infeasible,
        pyo.TerminationCondition.other,  # GLPK sometimes reports as "other"
    ], (
        f"Model should be infeasible when CO2 limit < achievable, "
        f"got {results_infeasible.solver.termination_condition}"
    )


def test_cumulative_flow_limits_respected():
    """
    Test that cumulative flow limits correctly constrain total flows across all years.

    Tests both max and min cumulative flow limits to verify constraints are properly enforced.
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
        "demand": {
            ("product", 2020): 10,
            ("product", 2021): 10,
            ("product", 2022): 10,
        },
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        # P1 has low emissions (1 kg CO2 per unit), P2 has high emissions (5 kg CO2)
        "foreground_biosphere": {
            ("P1_low_emission", "CO2", 0): 1,
            ("P2_high_emission", "CO2", 0): 5,
        },
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

    # Baseline: all from P1, total CO2 = 30 kg
    model_inputs_baseline = converter.OptimizationModelInputs(**model_inputs_dict)
    model_baseline = optimizer.create_model(
        inputs=model_inputs_baseline,
        objective_category="climate_change",
        name="test_cumulative_flow_baseline",
    )
    solved_baseline, obj_baseline, _ = optimizer.solve_model(
        model_baseline, solver_name="glpk", tee=False
    )
    assert pytest.approx(30, rel=0.01) == obj_baseline

    # Test 1: MINIMUM cumulative flow limit - force higher total emissions
    # Require total CO2 >= 50 (higher than optimal of 30)
    # With total demand of 30, this requires some P2 usage:
    # x + y = 30 (demand), x + 5y >= 50 (min emission)
    # y = 30 - x, so x + 5(30-x) >= 50 => -4x >= -100 => x <= 25
    # So P1 max 25 units, P2 min 5 units
    # Optimal: P1=25, P2=5 => CO2 = 25 + 25 = 50 (binding)
    cumulative_min = 50.0

    model_inputs_min = converter.OptimizationModelInputs(**model_inputs_dict)
    model_inputs_min.cumulative_flow_limits_min = {
        "CO2": cumulative_min,
    }

    model_min = optimizer.create_model(
        inputs=model_inputs_min,
        objective_category="climate_change",
        name="test_cumulative_flow_min",
    )
    solved_min, obj_min, results_min = optimizer.solve_model(
        model_min, solver_name="glpk", tee=False
    )

    # Verify solution is optimal
    assert results_min.solver.status == pyo.SolverStatus.ok
    assert results_min.solver.termination_condition == pyo.TerminationCondition.optimal

    # Calculate total CO2 emissions (denormalize)
    fg_scale = solved_min.scales["foreground"]
    total_co2 = sum(
        pyo.value(solved_min.total_elementary_flow["CO2", t]) * fg_scale
        for t in solved_min.SYSTEM_TIME
    )

    # Verify total CO2 meets the minimum limit
    assert total_co2 >= cumulative_min * 0.99, (
        f"Total CO2 ({total_co2:.2f}) is below min limit ({cumulative_min})"
    )

    # Verify constraint is binding
    assert total_co2 <= cumulative_min * 1.01, (
        f"Total CO2 ({total_co2:.2f}) is above min limit ({cumulative_min}), "
        "constraint should be binding"
    )

    # Verify objective matches expected (50 kg)
    assert pytest.approx(cumulative_min, rel=0.01) == obj_min, (
        f"Limited objective should be {cumulative_min}, got {obj_min}"
    )

    # Test 2: MAX cumulative flow limit below achievable should be infeasible
    # Optimal total CO2 is 30 kg. Limit to 25 kg should be infeasible.
    model_inputs_infeasible = converter.OptimizationModelInputs(**model_inputs_dict)
    model_inputs_infeasible.cumulative_flow_limits_max = {
        "CO2": 25.0,  # Below optimal of 30, forces impossible constraint
    }

    model_infeasible = optimizer.create_model(
        inputs=model_inputs_infeasible,
        objective_category="climate_change",
        name="test_cumulative_flow_infeasible",
    )
    solver = pyo.SolverFactory("glpk")
    results_infeasible = solver.solve(model_infeasible, tee=False)

    # Model should be infeasible since min achievable CO2 (30) > limit (25)
    assert results_infeasible.solver.termination_condition in [
        pyo.TerminationCondition.infeasible,
        pyo.TerminationCondition.other,  # GLPK sometimes reports as "other"
    ], (
        f"Model should be infeasible when cumulative CO2 limit < optimal, "
        f"got {results_infeasible.solver.termination_condition}"
    )


def test_product_flow_limits_respected():
    """
    Test that flow limits on products (not just elementary flows) are respected.

    Creates a model where production of a product is limited.
    """
    model_inputs_dict = {
        "PROCESS": ["P1"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": [],
        "ELEMENTARY_FLOW": ["CO2"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0],
        "SYSTEM_TIME": [2020, 2021, 2022],
        "CATEGORY": ["climate_change"],
        "operation_time_limits": {"P1": (0, 0)},
        "demand": {
            ("product", 2020): 10,
            ("product", 2021): 10,
            ("product", 2022): 10,
        },
        "foreground_technosphere": {},
        "internal_demand_technosphere": {},
        "foreground_biosphere": {("P1", "CO2", 0): 1},
        "foreground_production": {("P1", "product", 0): 1.0},
        "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
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

    # Test time-specific product limit - limit product output in 2021 to 5 units
    # This should make the model infeasible since demand is 10
    model_inputs_limited = converter.OptimizationModelInputs(**model_inputs_dict)
    model_inputs_limited.flow_limits_max = {
        ("product", 2021): 5.0,  # Can only produce 5 units in 2021
    }

    model_limited = optimizer.create_model(
        inputs=model_inputs_limited,
        objective_category="climate_change",
        name="test_product_flow_limit",
    )
    # Use solver directly without relying on solve_model's value extraction
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model_limited, tee=False)

    # Model should be infeasible since demand (10) > production limit (5)
    assert results.solver.termination_condition == pyo.TerminationCondition.infeasible, (
        f"Model should be infeasible when product limit < demand, "
        f"got {results.solver.termination_condition}"
    )

    # Now test with cumulative limit of 25 (less than total demand of 30)
    model_inputs_cumulative = converter.OptimizationModelInputs(**model_inputs_dict)
    model_inputs_cumulative.cumulative_flow_limits_max = {
        "product": 25.0,  # Total production limited to 25 units
    }

    model_cumulative = optimizer.create_model(
        inputs=model_inputs_cumulative,
        objective_category="climate_change",
        name="test_cumulative_product_limit",
    )
    results_cum = solver.solve(model_cumulative, tee=False)

    # Model should be infeasible since total demand (30) > cumulative limit (25)
    assert results_cum.solver.termination_condition == pyo.TerminationCondition.infeasible, (
        f"Model should be infeasible when cumulative product limit < total demand, "
        f"got {results_cum.solver.termination_condition}"
    )


def test_cumulative_flow_limits_background_inventory():
    """
    Test that cumulative flow limits correctly constrain flows from background inventory.

    This tests the bug where cumulative_flow_limits_max for elementary flows
    only considered foreground biosphere flows but not background inventory flows.

    The test creates a scenario where:
    - An elementary flow (e.g., "iridium") comes ONLY from background inventory
      (via intermediate flows), not from foreground biosphere
    - A cumulative flow limit is set on this elementary flow
    - The constraint should be respected when checking total inventory flows
    """
    # Create a model where the constrained elementary flow comes from background inventory
    # P1 uses intermediate flow "electricity" which produces "rare_element" in background
    model_inputs_dict = {
        "PROCESS": ["P1"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": ["electricity"],
        "ELEMENTARY_FLOW": ["rare_element"],  # e.g., iridium - comes from background
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0],
        "SYSTEM_TIME": [2020, 2021, 2022],
        "CATEGORY": ["climate_change"],
        "operation_time_limits": {"P1": (0, 0)},
        "demand": {
            ("product", 2020): 10,
            ("product", 2021): 10,
            ("product", 2022): 10,
        },
        # P1 requires 1 unit of electricity per product
        "foreground_technosphere": {
            ("P1", "electricity", 0): 1.0,
        },
        "internal_demand_technosphere": {},
        # NO foreground biosphere for rare_element - it only comes from background
        "foreground_biosphere": {},
        "foreground_production": {
            ("P1", "product", 0): 1.0,
        },
        "operation_flow": {
            ("P1", "product"): True,
            ("P1", "electricity"): True,
        },
        # Background inventory: each unit of electricity produces 0.5 kg of rare_element
        "background_inventory": {
            ("db_2020", "electricity", "rare_element"): 0.5,
        },
        "mapping": {
            ("db_2020", 2020): 1.0,
            ("db_2020", 2021): 1.0,
            ("db_2020", 2022): 1.0,
        },
        # Characterization for climate impact (use rare_element for simplicity)
        "characterization": {
            ("climate_change", "rare_element", 2020): 1.0,
            ("climate_change", "rare_element", 2021): 1.0,
            ("climate_change", "rare_element", 2022): 1.0,
        },
    }

    # First solve without constraint to get baseline
    model_inputs_baseline = converter.OptimizationModelInputs(**model_inputs_dict)
    model_baseline = optimizer.create_model(
        inputs=model_inputs_baseline,
        objective_category="climate_change",
        name="test_background_flow_baseline",
    )
    solved_baseline, obj_baseline, _ = optimizer.solve_model(
        model_baseline, solver_name="glpk", tee=False
    )

    # Calculate total rare_element from inventory (including background)
    # Total demand = 30, electricity per product = 1, so total electricity = 30
    # Background inventory: 0.5 rare_element per electricity
    # Total rare_element = 30 * 0.5 = 15
    fg_scale = solved_baseline.scales["foreground"]
    total_inventory = sum(
        pyo.value(solved_baseline.scaled_inventory["P1", "rare_element", t]) * fg_scale
        for t in solved_baseline.SYSTEM_TIME
    )
    assert pytest.approx(15, rel=0.01) == total_inventory, (
        f"Baseline total inventory should be 15, got {total_inventory}"
    )

    # Now test with cumulative limit of 10 (less than the 15 needed)
    # This should make the model infeasible since rare_element comes from background
    cumulative_limit = 10.0
    model_inputs_limited = converter.OptimizationModelInputs(**model_inputs_dict)
    model_inputs_limited.cumulative_flow_limits_max = {
        "rare_element": cumulative_limit,
    }

    model_limited = optimizer.create_model(
        inputs=model_inputs_limited,
        objective_category="climate_change",
        name="test_background_flow_limited",
    )
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model_limited, tee=False)

    # The model SHOULD be infeasible because:
    # - Total demand requires 30 units of product
    # - This requires 30 units of electricity (foreground_technosphere)
    # - This generates 15 kg of rare_element from background (30 * 0.5)
    # - But cumulative limit is only 10 kg
    # Therefore, the constraint cannot be satisfied.
    assert results.solver.termination_condition == pyo.TerminationCondition.infeasible, (
        f"Model should be infeasible when cumulative rare_element limit (10) < "
        f"required background flows (15), got {results.solver.termination_condition}. "
        "This indicates the cumulative flow constraint is not considering background inventory flows."
    )
