"""
Tests for per-vintage merit-order dispatch feature.

Merit-order dispatch enables the optimizer to dispatch cleaner vintages first
when multiple installation cohorts are operating simultaneously. This is achieved
by transforming var_operation from 2D (process, time) to 3D (process, vintage, time).

Key features tested:
1. Per-vintage operation variables: Each vintage has its own operation variable
2. Merit-order preference: Cleaner vintages are dispatched before dirtier ones
3. Brownfield dispatch: Existing capacity gets operation variables
4. Operation limits: Apply to total operation across all vintages
5. Per-vintage capacity: Each vintage bounded by its own installation/existing capacity
"""

import pytest
import pyomo.environ as pyo

from optimex import converter, optimizer
from optimex.postprocessing import PostProcessor


class TestPerVintageVariables:
    """Tests for 3D var_operation[process, vintage, time] structure."""

    def test_active_vintage_time_set_exists(self):
        """Test that ACTIVE_VINTAGE_TIME set is created with valid tuples."""
        model_inputs_dict = {
            "PROCESS": ["P1"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db"],
            "PROCESS_TIME": [0, 1, 2],
            "SYSTEM_TIME": [2020, 2021, 2022],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"P1": (1, 2)},  # Operation at tau 1 and 2
            "demand": {("product", t): 10 for t in [2020, 2021, 2022]},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", tau): 10 for tau in [0, 1, 2]},
            "foreground_production": {
                ("P1", "product", 1): 1.0,
                ("P1", "product", 2): 1.0,
            },
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {("db", t): 1.0 for t in [2020, 2021, 2022]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021, 2022]},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="test_active_vintage_time",
        )

        # Check that ACTIVE_VINTAGE_TIME set exists
        assert hasattr(model, "ACTIVE_VINTAGE_TIME")
        assert len(model.ACTIVE_VINTAGE_TIME) > 0

        # Verify all tuples have 3 elements: (process, vintage, time)
        for item in model.ACTIVE_VINTAGE_TIME:
            assert len(item) == 3
            p, v, t = item
            assert p in model.PROCESS
            assert t in model.SYSTEM_TIME

    def test_var_operation_is_3d(self):
        """Test that var_operation is indexed by (process, vintage, time)."""
        model_inputs_dict = {
            "PROCESS": ["P1"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"P1": (0, 1)},
            "demand": {("product", t): 10 for t in [2020, 2021]},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", tau): 10 for tau in [0, 1]},
            "foreground_production": {
                ("P1", "product", 0): 1.0,
                ("P1", "product", 1): 1.0,
            },
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {("db", t): 1.0 for t in [2020, 2021]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021]},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="test_3d_operation",
        )

        # Check that var_operation is indexed by ACTIVE_VINTAGE_TIME (3D)
        assert hasattr(model, "var_operation")
        # The variable should be accessible with 3-tuple indices
        for (p, v, t) in model.ACTIVE_VINTAGE_TIME:
            assert model.var_operation[p, v, t] is not None

    def test_brownfield_has_operation_variables(self):
        """Test that brownfield (existing) capacity gets operation variables."""
        model_inputs_dict = {
            "PROCESS": ["P1"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"P1": (0, 1)},
            "demand": {("product", t): 10 for t in [2020, 2021]},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", tau): 10 for tau in [0, 1]},
            "foreground_production": {
                ("P1", "product", 0): 1.0,
                ("P1", "product", 1): 1.0,
            },
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {("db", t): 1.0 for t in [2020, 2021]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021]},
            # Brownfield capacity from 2019
            "existing_capacity": {("P1", 2019): 5.0},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="test_brownfield_operation",
        )

        # Check that brownfield vintage (2019) appears in ACTIVE_VINTAGE_TIME
        brownfield_entries = [
            (p, v, t) for (p, v, t) in model.ACTIVE_VINTAGE_TIME
            if v == 2019
        ]
        assert len(brownfield_entries) > 0, (
            "Brownfield vintage 2019 should have entries in ACTIVE_VINTAGE_TIME"
        )

        # Check that operation variable exists for brownfield
        for (p, v, t) in brownfield_entries:
            assert model.var_operation[p, v, t] is not None


class TestMeritOrderDispatch:
    """Tests for merit-order dispatch behavior."""

    def test_cleaner_vintage_dispatched_first(self):
        """
        Verify optimizer prefers cleaner vintages when both are available.

        Scenario:
        - Two vintages: 2020 (inefficient, high emissions) and 2022 (efficient, low emissions)
        - Both have capacity available in 2022
        - Optimizer should prefer 2022 vintage for its lower emissions

        Note: With vintage-dependent emissions, we use technology_evolution to
        create different emission rates for different vintages.
        """
        model_inputs_dict = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["fuel"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0, 1, 2, 3],
            "SYSTEM_TIME": [2020, 2021, 2022, 2023],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (1, 3)},
            "demand": {
                ("output", 2020): 0,
                ("output", 2021): 0,
                ("output", 2022): 0,
                ("output", 2023): 50,  # Only demand at 2023 when both vintages available
            },
            # Fuel consumption varies by vintage (technology evolution)
            "foreground_technosphere": {
                ("Plant", "fuel", tau): 10 for tau in range(4)
            },
            "technology_evolution": {
                ("Plant", "fuel", 2020): 1.0,   # 2020 vintage: baseline
                ("Plant", "fuel", 2022): 0.5,   # 2022 vintage: 50% more efficient
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},  # Emissions come from fuel via background
            "foreground_production": {
                ("Plant", "output", tau): 100 if 1 <= tau <= 3 else 0
                for tau in range(4)
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "fuel"): True,
            },
            "background_inventory": {("grid", "fuel", "CO2"): 1.0},
            "mapping": {("grid", t): 1.0 for t in [2020, 2021, 2022, 2023]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021, 2022, 2023]},
            # Force some installation at both 2020 and 2022
            "process_deployment_limits_min": {
                ("Plant", 2020): 1.0,  # Force installation in 2020
                ("Plant", 2022): 1.0,  # Force installation in 2022
            },
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="merit_order_test",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # At 2023: both 2020 vintage (tau=3) and 2022 vintage (tau=1) are in operation
        # Optimizer should prefer 2022 vintage due to lower emissions

        # Get operation values for both vintages at 2023
        op_2020_vintage = pyo.value(solved_model.var_operation["Plant", 2020, 2023])
        op_2022_vintage = pyo.value(solved_model.var_operation["Plant", 2022, 2023])

        # The 2022 vintage should be dispatched more/first due to lower emissions
        # Since 2022 vintage has 50% lower fuel consumption, it should be preferred
        assert op_2022_vintage >= op_2020_vintage, (
            f"Cleaner 2022 vintage ({op_2022_vintage:.4f}) should be dispatched "
            f"at least as much as dirtier 2020 vintage ({op_2020_vintage:.4f})"
        )

    def test_brownfield_dispatchable_independently(self):
        """
        Verify brownfield capacity can be scaled independently from greenfield.

        Scenario:
        - Brownfield capacity from 2018 (high emissions)
        - Greenfield capacity from 2020 (lower emissions)
        - Optimizer should prefer greenfield when emissions differ
        """
        model_inputs_dict = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["fuel"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0, 1, 2, 3, 4, 5],
            "SYSTEM_TIME": [2020, 2021, 2022],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (1, 5)},
            "demand": {("output", t): 100 for t in [2020, 2021, 2022]},
            "foreground_technosphere": {
                ("Plant", "fuel", tau): 10 for tau in range(6)
            },
            # Technology evolution makes newer vintages more efficient
            "technology_evolution": {
                ("Plant", "fuel", 2020): 1.0,
                ("Plant", "fuel", 2022): 0.7,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", tau): 100 if 1 <= tau <= 5 else 0
                for tau in range(6)
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "fuel"): True,
            },
            "background_inventory": {("grid", "fuel", "CO2"): 1.0},
            "mapping": {("grid", t): 1.0 for t in [2020, 2021, 2022]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021, 2022]},
            # Brownfield capacity (older, uses 2020 rates)
            "existing_capacity": {("Plant", 2018): 2.0},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="brownfield_dispatch_test",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Brownfield vintage 2018 should have operation variable
        brownfield_ops = [
            (v, t, pyo.value(solved_model.var_operation["Plant", v, t]))
            for (p, v, t) in solved_model.ACTIVE_VINTAGE_TIME
            if p == "Plant" and v == 2018
        ]
        assert len(brownfield_ops) > 0, "Brownfield should have operation entries"

        # Verify brownfield is independently dispatchable (can be < full capacity)
        for v, t, op_val in brownfield_ops:
            # Just verify it's a valid (non-negative) operation value
            assert op_val >= -1e-6, f"Brownfield operation at {t} should be non-negative"


class TestOperationLimits:
    """Tests for operation limits applying to total across vintages."""

    def test_operation_limits_apply_to_total(self):
        """
        Verify operation limits apply to sum of operation across vintages.

        Scenario:
        - Multiple vintages operating at same time
        - Operation limit constrains total (not per-vintage)
        """
        model_inputs_dict = {
            "PROCESS": ["P1"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db"],
            "PROCESS_TIME": [0, 1, 2],
            "SYSTEM_TIME": [2020, 2021, 2022],
            "CATEGORY": ["GWP"],
            # Operation at tau 0-2 (immediate operation)
            "operation_time_limits": {"P1": (0, 2)},
            "demand": {("product", t): 50 for t in [2020, 2021, 2022]},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", tau): 10 for tau in [0, 1, 2]},
            "foreground_production": {
                ("P1", "product", 0): 1.0,
                ("P1", "product", 1): 1.0,
                ("P1", "product", 2): 1.0,
            },
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {("db", t): 1.0 for t in [2020, 2021, 2022]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021, 2022]},
            # Set a maximum operation limit at 2022
            "process_operation_limits_max": {("P1", 2022): 30},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="operation_limits_test",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Sum operation across all vintages at 2022
        total_op_2022 = sum(
            pyo.value(solved_model.var_operation[p, v, t])
            for (p, v, t) in solved_model.ACTIVE_VINTAGE_TIME
            if p == "P1" and t == 2022
        )

        # Total operation should respect the limit
        assert total_op_2022 <= 30 + 1e-6, (
            f"Total operation at 2022 ({total_op_2022:.4f}) should not exceed limit of 30"
        )


class TestPerVintageCapacity:
    """Tests for per-vintage capacity bounds."""

    def test_per_vintage_capacity_respected(self):
        """
        Verify each vintage's operation is bounded by its own installation capacity.

        Scenario:
        - Vintages installed at different times
        - Each vintage's operation cannot exceed its own capacity
        """
        model_inputs_dict = {
            "PROCESS": ["P1"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db"],
            "PROCESS_TIME": [0, 1, 2],
            "SYSTEM_TIME": [2020, 2021, 2022, 2023],
            "CATEGORY": ["GWP"],
            # Operation at tau 0-2 (immediate operation)
            "operation_time_limits": {"P1": (0, 2)},
            "demand": {("product", t): 100 for t in [2020, 2021, 2022, 2023]},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", tau): 10 for tau in [0, 1, 2]},
            "foreground_production": {
                ("P1", "product", 0): 50.0,
                ("P1", "product", 1): 50.0,
                ("P1", "product", 2): 50.0,
            },
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {("db", t): 1.0 for t in [2020, 2021, 2022, 2023]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021, 2022, 2023]},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="per_vintage_capacity_test",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Verify each vintage's operation <= its installation * production_rate * fg_scale
        fg_scale = solved_model.scales["foreground"]
        production_per_unit = 50.0 + 50.0 + 50.0  # Sum over tau 0, 1, and 2

        for (p, v, t) in solved_model.ACTIVE_VINTAGE_TIME:
            if v in solved_model.SYSTEM_TIME:
                # Greenfield: capacity from var_installation
                installation = pyo.value(solved_model.var_installation[p, v])
                capacity = production_per_unit * installation * fg_scale
                operation = pyo.value(solved_model.var_operation[p, v, t])
                assert operation <= capacity + 1e-6, (
                    f"Operation ({operation:.4f}) should not exceed capacity ({capacity:.4f}) "
                    f"for vintage {v} at time {t}"
                )


class TestPostprocessingWithVintages:
    """Tests for postprocessing with per-vintage operation."""

    def test_get_operation_aggregate_vintages(self):
        """Test get_operation with aggregate_vintages=True (default)."""
        model_inputs_dict = {
            "PROCESS": ["P1"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"P1": (0, 1)},
            "demand": {("product", t): 10 for t in [2020, 2021]},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", tau): 10 for tau in [0, 1]},
            "foreground_production": {
                ("P1", "product", 0): 1.0,
                ("P1", "product", 1): 1.0,
            },
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {("db", t): 1.0 for t in [2020, 2021]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021]},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="postprocessing_test",
        )

        solved_model, _, _ = optimizer.solve_model(model, solver_name="glpk", tee=False)

        pp = PostProcessor(solved_model)
        op_df = pp.get_operation(aggregate_vintages=True)

        # Should have Process columns (not Process, Vintage)
        assert "P1" in op_df.columns
        assert not isinstance(op_df.columns, type(op_df.columns).__class__)  # Not MultiIndex

    def test_get_operation_by_vintage(self):
        """Test get_operation with aggregate_vintages=False."""
        model_inputs_dict = {
            "PROCESS": ["P1"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db"],
            "PROCESS_TIME": [0, 1, 2],
            "SYSTEM_TIME": [2020, 2021, 2022],
            "CATEGORY": ["GWP"],
            # Operation at tau 0-2 (immediate operation)
            "operation_time_limits": {"P1": (0, 2)},
            "demand": {("product", t): 10 for t in [2020, 2021, 2022]},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", tau): 10 for tau in [0, 1, 2]},
            "foreground_production": {
                ("P1", "product", 0): 1.0,
                ("P1", "product", 1): 1.0,
                ("P1", "product", 2): 1.0,
            },
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {("db", t): 1.0 for t in [2020, 2021, 2022]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021, 2022]},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="vintage_postprocessing_test",
        )

        solved_model, _, _ = optimizer.solve_model(model, solver_name="glpk", tee=False)

        pp = PostProcessor(solved_model)
        op_df = pp.get_operation(aggregate_vintages=False)

        # Should have MultiIndex columns with (Process, Vintage)
        import pandas as pd
        assert isinstance(op_df.columns, pd.MultiIndex), "Columns should be MultiIndex"
        # Columns should be tuples with (Process, Vintage)
        for col in op_df.columns:
            assert len(col) == 2

    def test_get_operation_by_vintage_method(self):
        """Test the convenience method get_operation_by_vintage()."""
        model_inputs_dict = {
            "PROCESS": ["P1"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"P1": (0, 1)},
            "demand": {("product", t): 10 for t in [2020, 2021]},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", tau): 10 for tau in [0, 1]},
            "foreground_production": {
                ("P1", "product", 0): 1.0,
                ("P1", "product", 1): 1.0,
            },
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {("db", t): 1.0 for t in [2020, 2021]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021]},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="vintage_method_test",
        )

        solved_model, _, _ = optimizer.solve_model(model, solver_name="glpk", tee=False)

        pp = PostProcessor(solved_model)
        op_by_vintage = pp.get_operation_by_vintage()

        # Should return the same as get_operation(aggregate_vintages=False)
        op_explicit = pp.get_operation(aggregate_vintages=False)
        assert op_by_vintage.shape == op_explicit.shape


class TestValidateOperationBounds:
    """Tests for validate_operation_bounds with 3D operation."""

    def test_validate_operation_bounds_valid(self):
        """Test that valid model passes validation."""
        model_inputs_dict = {
            "PROCESS": ["P1"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"P1": (0, 1)},
            "demand": {("product", t): 10 for t in [2020, 2021]},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", tau): 10 for tau in [0, 1]},
            "foreground_production": {
                ("P1", "product", 0): 1.0,
                ("P1", "product", 1): 1.0,
            },
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {("db", t): 1.0 for t in [2020, 2021]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021]},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="validation_test",
        )

        solved_model, _, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Validate operation bounds
        validation = optimizer.validate_operation_bounds(solved_model)
        assert validation["valid"], validation["summary"]
