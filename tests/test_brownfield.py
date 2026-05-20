"""
Tests for brownfield (existing capacity) optimization feature.

Brownfield optimization allows users to specify existing process capacities
that were installed before the optimization horizon. These existing capacities:
- Contribute to production capacity (can meet demand)
- Do NOT contribute to installation-phase impacts (sunk costs)
- Only contribute operational impacts during their remaining lifetime
"""

import pytest
import pyomo.environ as pyo

from optimex import converter, optimizer


def get_total_operation(model, p, t):
    """Get total operation for a process at a time, summed across all vintages."""
    return sum(
        pyo.value(model.var_operation[proc, v, time])
        for (proc, v, time) in model.ACTIVE_VINTAGE_TIME
        if proc == p and time == t
    )


class TestBrownfieldValidation:
    """Tests for validation of existing_capacity parameter."""

    def test_existing_capacity_valid(self):
        """Test that valid existing capacity is accepted."""
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
            "demand": {("product", 2020): 10, ("product", 2021): 10, ("product", 2022): 10},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", 0): 10},
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
            # Valid: installation year 2018 is before min(SYSTEM_TIME)=2020
            "existing_capacity": {("P1", 2018): 5.0},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        assert model_inputs.existing_capacity == {("P1", 2018): 5.0}

    def test_existing_capacity_invalid_year(self):
        """Test that existing capacity with year >= min(SYSTEM_TIME) is rejected."""
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
            "demand": {("product", 2020): 10},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", 0): 10},
            "foreground_production": {("P1", "product", 0): 1.0},
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {("db_2020", 2020): 1.0},
            "characterization": {("climate_change", "CO2", 2020): 1.0},
            # Invalid: installation year 2020 is not before min(SYSTEM_TIME)=2020
            "existing_capacity": {("P1", 2020): 5.0},
        }

        with pytest.raises(ValueError, match="must be before min\\(SYSTEM_TIME\\)"):
            converter.OptimizationModelInputs(**model_inputs_dict)

    def test_existing_capacity_invalid_process(self):
        """Test that existing capacity with invalid process ID is rejected."""
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
            "demand": {("product", 2020): 10},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", 0): 10},
            "foreground_production": {("P1", "product", 0): 1.0},
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {("db_2020", 2020): 1.0},
            "characterization": {("climate_change", "CO2", 2020): 1.0},
            # Invalid: P_INVALID is not in PROCESS
            "existing_capacity": {("P_INVALID", 2018): 5.0},
        }

        with pytest.raises(ValueError, match="Invalid keys"):
            converter.OptimizationModelInputs(**model_inputs_dict)

    def test_existing_capacity_negative_rejected(self):
        """Test that negative existing capacity is rejected."""
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
            "demand": {("product", 2020): 10},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", 0): 10},
            "foreground_production": {("P1", "product", 0): 1.0},
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {("db_2020", 2020): 1.0},
            "characterization": {("climate_change", "CO2", 2020): 1.0},
            # Invalid: negative capacity
            "existing_capacity": {("P1", 2018): -5.0},
        }

        with pytest.raises(ValueError, match="must be non-negative"):
            converter.OptimizationModelInputs(**model_inputs_dict)


class TestBrownfieldOptimization:
    """Tests for brownfield optimization behavior."""

    def test_existing_capacity_meets_demand(self):
        """
        Test that existing capacity can meet demand with significant installation penalty.

        Scenario:
        - Demand: 5 units in year 2025 only
        - Existing capacity: 10 units installed in 2024 (operation at tau=1 covers 2025)
        - Operation phase: tau = 1 only
        - Installation penalty: HIGH emissions at tau=0 (via CO2_install, NOT operational)
        - Operation emissions: LOW (via CO2_operate, operational)
        - System time: 2025 only

        With high installation penalty, optimizer should prefer using existing capacity.
        Note: We use separate elementary flows for installation vs operation emissions
        because operational flows must be constant over time.
        """
        model_inputs_dict = {
            "PROCESS": ["P1"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2_install", "CO2_operate"],
            "BACKGROUND_ID": ["db_2020"],
            "PROCESS_TIME": [0, 1],  # tau=0: installation, tau=1: operation
            "SYSTEM_TIME": [2025],
            "CATEGORY": ["climate_change"],
            "operation_time_limits": {"P1": (1, 1)},  # Only tau=1 is operation
            "demand": {
                ("product", 2025): 5,
            },
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            # Separate flows for installation vs operation emissions
            "foreground_biosphere": {
                ("P1", "CO2_install", 0): 1000,  # HIGH installation emissions (NOT operational)
                ("P1", "CO2_operate", 1): 1,    # LOW operation emissions
            },
            "foreground_production": {
                ("P1", "product", 1): 1.0,
            },
            "operation_flow": {
                ("P1", "product"): True,
                ("P1", "CO2_install"): False,  # NOT operational (scaled by installation)
                ("P1", "CO2_operate"): True,   # Operational (scaled by operation)
            },
            "background_inventory": {},
            "mapping": {
                ("db_2020", 2025): 1.0,
            },
            "characterization": {
                ("climate_change", "CO2_install", 2025): 1.0,
                ("climate_change", "CO2_operate", 2025): 1.0,
            },
            # Existing capacity: 10 units installed in 2024
            # At 2025: tau = 2025-2024 = 1, in operation (1 <= 1 <= 1)
            "existing_capacity": {("P1", 2024): 10.0},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="climate_change",
            name="test_brownfield_meets_demand",

        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        # Verify solution is optimal
        assert results.solver.status == pyo.SolverStatus.ok
        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Existing capacity of 10 units at tau=1 provides capacity
        # Should be enough to meet demand of 5 with no new installations
        inst_2025 = pyo.value(solved_model.var_installation["P1", 2025])
        assert pytest.approx(0, abs=1e-4) == inst_2025, (
            f"No new installation expected at 2025 (existing capacity should suffice), got {inst_2025}"
        )

    def test_existing_capacity_reduces_emissions(self):
        """
        Test that existing capacity's installation impacts are excluded (sunk costs).

        Compare two scenarios with same demand:
        1. Greenfield: All capacity must be installed new
        2. Brownfield: All capacity already exists

        Brownfield should have lower total impact because existing capacity's
        installation emissions are excluded (they're sunk costs from the past).

        Note: We use separate elementary flows for installation vs operation emissions
        because operational flows must be constant over time.
        """
        base_config = {
            "PROCESS": ["P1"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2_install", "CO2_operate"],
            "BACKGROUND_ID": ["db_2020"],
            "PROCESS_TIME": [0, 1, 2],
            "SYSTEM_TIME": [2022, 2023],
            "CATEGORY": ["climate_change"],
            "operation_time_limits": {"P1": (1, 2)},  # tau 1 and 2 are operation
            "demand": {("product", 2023): 10},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            # Separate flows for installation vs operation emissions
            "foreground_biosphere": {
                ("P1", "CO2_install", 0): 100,  # Installation emissions (NOT operational)
                ("P1", "CO2_operate", 1): 1,    # Operation emissions (same at tau=1 and 2)
                ("P1", "CO2_operate", 2): 1,
            },
            "foreground_production": {
                ("P1", "product", 1): 1.0,
                ("P1", "product", 2): 1.0,
            },
            "operation_flow": {
                ("P1", "product"): True,
                ("P1", "CO2_install"): False,  # NOT operational (scaled by installation)
                ("P1", "CO2_operate"): True,   # Operational (scaled by operation)
            },
            "background_inventory": {},
            "mapping": {("db_2020", 2022): 1.0, ("db_2020", 2023): 1.0},
            "characterization": {
                ("climate_change", "CO2_install", 2022): 1.0,
                ("climate_change", "CO2_install", 2023): 1.0,
                ("climate_change", "CO2_operate", 2022): 1.0,
                ("climate_change", "CO2_operate", 2023): 1.0,
            },
        }

        # Greenfield: must install new capacity
        # Install at 2022: tau=0 at 2022 (100 kg install emissions)
        # Then operation at tau=1,2 which falls at 2023, 2024 (but 2024 outside SYSTEM_TIME)
        greenfield_config = dict(base_config)
        greenfield_inputs = converter.OptimizationModelInputs(**greenfield_config)
        greenfield_model = optimizer.create_model(
            inputs=greenfield_inputs,
            objective_category="climate_change",
            name="greenfield",

        )
        _, greenfield_obj, _ = optimizer.solve_model(
            greenfield_model, solver_name="glpk", tee=False
        )

        # Brownfield: all capacity already exists
        # Existing capacity at 2021: at 2023, tau = 2023-2021 = 2, in operation!
        brownfield_config = dict(base_config)
        brownfield_config["existing_capacity"] = {("P1", 2021): 10.0}  # Plenty of capacity

        brownfield_inputs = converter.OptimizationModelInputs(**brownfield_config)
        brownfield_model = optimizer.create_model(
            inputs=brownfield_inputs,
            objective_category="climate_change",
            name="brownfield",

        )
        _, brownfield_obj, _ = optimizer.solve_model(
            brownfield_model, solver_name="glpk", tee=False
        )

        # Brownfield should have lower emissions:
        # - No installation emissions (existing capacity's tau=0 was in 2021, before SYSTEM_TIME)
        # - Only operation emissions
        assert brownfield_obj < greenfield_obj, (
            f"Brownfield ({brownfield_obj}) should have lower emissions than "
            f"greenfield ({greenfield_obj}) because installation impacts are excluded"
        )

    def test_existing_capacity_retirement(self):
        """
        Test that existing capacity correctly retires when reaching end of life.

        Scenario:
        - Process with 3-year operation phase (tau = 0, 1, 2)
        - Existing capacity installed in 2018
        - System time: 2020-2023

        Timeline for existing capacity:
        - 2020: tau = 2, still in operation (capacity available)
        - 2021: tau = 3, retired (capacity = 0)
        - 2022: tau = 4, retired (capacity = 0)
        - 2023: tau = 5, retired (capacity = 0)

        After retirement, new capacity must be installed to meet demand.
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
            "operation_time_limits": {"P1": (0, 2)},
            "demand": {
                ("product", 2020): 5,
                ("product", 2021): 5,
                ("product", 2022): 5,
                ("product", 2023): 5,
            },
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {
                ("P1", "CO2", 0): 10,
                ("P1", "CO2", 1): 10,
                ("P1", "CO2", 2): 10,
            },
            "foreground_production": {
                ("P1", "product", 0): 1.0,
                ("P1", "product", 1): 1.0,
                ("P1", "product", 2): 1.0,
            },
            "operation_flow": {
                ("P1", "product"): True,
                ("P1", "CO2"): True,
            },
            "background_inventory": {},
            "mapping": {
                ("db_2020", t): 1.0 for t in [2020, 2021, 2022, 2023]
            },
            "characterization": {
                ("climate_change", "CO2", t): 1.0 for t in [2020, 2021, 2022, 2023]
            },
            # Existing capacity retires after 2020
            "existing_capacity": {("P1", 2018): 10.0},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="climate_change",
            name="test_retirement",

        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Check that new installations are needed after existing capacity retires
        total_new_installations = sum(
            pyo.value(solved_model.var_installation["P1", t])
            for t in [2020, 2021, 2022, 2023]
        )

        # Some new installations should be needed to replace retired capacity
        assert total_new_installations > 0, (
            "New installations should be needed after existing capacity retires"
        )


class TestBrownfieldPostprocessing:
    """Tests for postprocessing with brownfield capacity."""

    def test_get_existing_capacity(self):
        """Test that get_existing_capacity returns correct data."""
        from optimex.postprocessing import PostProcessor

        model_inputs_dict = {
            "PROCESS": ["P1", "P2"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db_2020"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021],
            "CATEGORY": ["climate_change"],
            "operation_time_limits": {"P1": (0, 1), "P2": (0, 1)},
            "demand": {
                ("product", 2020): 10,
                ("product", 2021): 10,
            },
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {
                ("P1", "CO2", 0): 10,
                ("P1", "CO2", 1): 10,
                ("P2", "CO2", 0): 20,
                ("P2", "CO2", 1): 20,
            },
            "foreground_production": {
                ("P1", "product", 0): 1.0,
                ("P1", "product", 1): 1.0,
                ("P2", "product", 0): 1.0,
                ("P2", "product", 1): 1.0,
            },
            "operation_flow": {
                ("P1", "product"): True,
                ("P1", "CO2"): True,
                ("P2", "product"): True,
                ("P2", "CO2"): True,
            },
            "background_inventory": {},
            "mapping": {
                ("db_2020", 2020): 1.0,
                ("db_2020", 2021): 1.0,
            },
            "characterization": {
                ("climate_change", "CO2", 2020): 1.0,
                ("climate_change", "CO2", 2021): 1.0,
            },
            # P1 has existing capacity, P2 does not
            "existing_capacity": {("P1", 2019): 5.0},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="climate_change",
            name="test_postprocessing",

        )

        solved_model, _, _ = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        pp = PostProcessor(solved_model)
        existing_df = pp.get_existing_capacity()

        # Should have data for P1, not P2
        assert not existing_df.empty
        assert ("P1", "existing_capacity") in existing_df.columns
        assert ("P1", "existing_operating") in existing_df.columns

        # At 2020: tau = 2020-2019 = 1, in operation (0 <= 1 <= 1)
        # At 2021: tau = 2021-2019 = 2, NOT in operation (2 > 1)
        assert existing_df.loc[2020, ("P1", "existing_operating")] == 5.0
        assert existing_df.loc[2021, ("P1", "existing_operating")] == 0.0

    def test_production_capacity_includes_existing(self):
        """Test that get_production_capacity includes existing capacity."""
        from optimex.postprocessing import PostProcessor

        model_inputs_dict = {
            "PROCESS": ["P1"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db_2020"],
            "PROCESS_TIME": [0],
            "SYSTEM_TIME": [2020, 2021],
            "CATEGORY": ["climate_change"],
            "operation_time_limits": {"P1": (0, 0)},
            "demand": {
                ("product", 2020): 5,
                ("product", 2021): 5,
            },
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("P1", "CO2", 0): 10},
            "foreground_production": {("P1", "product", 0): 1.0},
            "operation_flow": {("P1", "product"): True, ("P1", "CO2"): True},
            "background_inventory": {},
            "mapping": {
                ("db_2020", 2020): 1.0,
                ("db_2020", 2021): 1.0,
            },
            "characterization": {
                ("climate_change", "CO2", 2020): 1.0,
                ("climate_change", "CO2", 2021): 1.0,
            },
            # Existing capacity of 10 units installed in 2019
            # At 2020: tau = 1, NOT in operation (0 <= 0 <= 0 means only tau=0)
            # So we need installation year = 2020 to have tau = 0
            "existing_capacity": {("P1", 2020 - 0): 10.0},  # This won't work due to validation
        }

        # Actually, the validation requires inst_year < min(SYSTEM_TIME)
        # So let's test with a different setup where existing capacity IS in operation
        model_inputs_dict["existing_capacity"] = {("P1", 2019): 10.0}
        # With operation_time_limits (0, 0), at 2020: tau = 2020-2019 = 1, NOT in operation

        # Let's fix by extending operation time
        model_inputs_dict["PROCESS_TIME"] = [0, 1]
        model_inputs_dict["operation_time_limits"] = {"P1": (0, 1)}
        model_inputs_dict["foreground_biosphere"] = {
            ("P1", "CO2", 0): 10,
            ("P1", "CO2", 1): 10,
        }
        model_inputs_dict["foreground_production"] = {
            ("P1", "product", 0): 1.0,
            ("P1", "product", 1): 1.0,
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="climate_change",
            name="test_capacity",

        )

        solved_model, _, _ = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        pp = PostProcessor(solved_model)
        capacity_df = pp.get_production_capacity()

        # At 2020: existing capacity (tau=1) is in operation
        # Capacity should include the existing 10 units * 2 (production per unit) = 20
        assert capacity_df.loc[2020, "product"] >= 20.0, (
            f"Production capacity at 2020 should include existing capacity, "
            f"got {capacity_df.loc[2020, 'product']}"
        )


class TestBrownfieldWithVintageParameters:
    """Tests for brownfield optimization combined with vintage-dependent parameters."""

    def test_brownfield_with_vintage_improvements(self):
        """
        Test that brownfield + vintage_improvements works correctly.

        This tests the 4D code path triggered by vintage_improvements when
        existing capacity is installed before SYSTEM_TIME.

        Scenario:
        - Existing capacity from 2015 (before SYSTEM_TIME 2025-2030)
        - vintage_improvements triggers 4D code path
        - Model should be feasible and use existing capacity
        """
        model_inputs_dict = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": list(range(21)),  # 20-year operation
            "SYSTEM_TIME": list(range(2025, 2031)),  # 2025-2030
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (1, 20)},
            "demand": {("output", t): 100 for t in range(2025, 2031)},
            "foreground_technosphere": {
                ("Plant", "electricity", tau): 10 for tau in range(21)
            },
            # vintage_improvements triggers 4D code path
            "vintage_improvements": {
                ("Plant", "electricity", 2025): 1.0,
                ("Plant", "electricity", 2030): 0.8,  # 20% efficiency improvement
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {
                ("Plant", "CO2", tau): 5 for tau in range(21)
            },
            "foreground_production": {
                ("Plant", "output", tau): 50 if 1 <= tau <= 20 else 0
                for tau in range(21)
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
                ("Plant", "CO2"): True,
            },
            "background_inventory": {("grid", "electricity", "CO2"): 0.5},
            "mapping": {("grid", t): 1.0 for t in range(2025, 2031)},
            "characterization": {("GWP", "CO2", t): 1.0 for t in range(2025, 2031)},
            # Existing capacity from 2015 - before SYSTEM_TIME
            "existing_capacity": {("Plant", 2015): 3.0},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="brownfield_vintage_test",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        # Model should be feasible
        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Existing capacity should contribute to meeting demand
        # At 2025, existing capacity from 2015 is at tau = 2025-2015 = 10 (in operation)
        operation_2025 = get_total_operation(solved_model, "Plant", 2025)
        assert operation_2025 > 0, "Operation should be positive when existing capacity exists"

    def test_brownfield_with_vintage_production_overrides(self):
        """
        Test brownfield with explicit vintage production overrides.

        Scenario:
        - Existing capacity from before SYSTEM_TIME
        - foreground_production_vintages provides vintage-specific rates
        - Model should use nearest vintage rate for existing capacity
        """
        model_inputs_dict = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2_install", "CO2_operate"],
            "BACKGROUND_ID": ["db"],
            "PROCESS_TIME": [0, 1, 2],
            "SYSTEM_TIME": [2025, 2026, 2027],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (1, 2)},
            "demand": {("output", t): 50 for t in [2025, 2026, 2027]},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            # Separate flows for installation vs operation (operational must be constant)
            "foreground_biosphere": {
                ("Plant", "CO2_install", 0): 100,  # Installation emissions
                ("Plant", "CO2_operate", 1): 10,   # Operation emissions (constant)
                ("Plant", "CO2_operate", 2): 10,
            },
            # Base production (will be overridden by vintages)
            "foreground_production": {
                ("Plant", "output", 0): 0,
                ("Plant", "output", 1): 100,
                ("Plant", "output", 2): 100,
            },
            # Vintage-specific production rates
            "foreground_production_vintages": {
                ("Plant", "output", 1, 2025): 100,
                ("Plant", "output", 2, 2025): 100,
                ("Plant", "output", 1, 2027): 120,  # 20% more efficient
                ("Plant", "output", 2, 2027): 120,
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "CO2_install"): False,  # NOT operational
                ("Plant", "CO2_operate"): True,   # Operational
            },
            "background_inventory": {},
            "mapping": {("db", t): 1.0 for t in [2025, 2026, 2027]},
            "characterization": {
                ("GWP", "CO2_install", t): 1.0 for t in [2025, 2026, 2027]
            } | {
                ("GWP", "CO2_operate", t): 1.0 for t in [2025, 2026, 2027]
            },
            # Existing capacity from 2023 (before min SYSTEM_TIME 2025)
            # At 2025: tau = 2025-2023 = 2, in operation (1 <= 2 <= 2)
            # At 2026: tau = 2026-2023 = 3, NOT in operation (3 > 2)
            "existing_capacity": {("Plant", 2023): 1.0},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="brownfield_production_vintage",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Existing capacity should contribute at 2025 (tau=2 is in operation)
        operation_2025 = get_total_operation(solved_model, "Plant", 2025)
        assert operation_2025 > 0, "Existing capacity should contribute at 2025"

    def test_brownfield_with_mixed_vintage_processes(self):
        """
        Test brownfield with mixed processes - some with vintage evolution, some without.

        Scenario:
        - OldPlant: No vintage parameters, existing capacity
        - NewPlant: Has vintage_improvements, no existing capacity

        Both should work together correctly.
        """
        model_inputs_dict = {
            "PROCESS": ["OldPlant", "NewPlant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": list(range(11)),  # 10-year operation
            "SYSTEM_TIME": [2025, 2026, 2027],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {
                "OldPlant": (1, 10),
                "NewPlant": (1, 10),
            },
            "demand": {("output", t): 200 for t in [2025, 2026, 2027]},
            "foreground_technosphere": {
                ("OldPlant", "electricity", tau): 20 for tau in range(11)
            } | {
                ("NewPlant", "electricity", tau): 15 for tau in range(11)
            },
            # vintage_improvements only for NewPlant
            "vintage_improvements": {
                ("NewPlant", "electricity", 2025): 1.0,
                ("NewPlant", "electricity", 2027): 0.8,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {
                ("OldPlant", "CO2", tau): 10 for tau in range(11)
            } | {
                ("NewPlant", "CO2", tau): 5 for tau in range(11)
            },
            "foreground_production": {
                ("OldPlant", "output", tau): 100 if 1 <= tau <= 10 else 0
                for tau in range(11)
            } | {
                ("NewPlant", "output", tau): 100 if 1 <= tau <= 10 else 0
                for tau in range(11)
            },
            "operation_flow": {
                ("OldPlant", "output"): True,
                ("OldPlant", "electricity"): True,
                ("OldPlant", "CO2"): True,
                ("NewPlant", "output"): True,
                ("NewPlant", "electricity"): True,
                ("NewPlant", "CO2"): True,
            },
            "background_inventory": {("grid", "electricity", "CO2"): 0.5},
            "mapping": {("grid", t): 1.0 for t in [2025, 2026, 2027]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2025, 2026, 2027]},
            # Existing capacity only for OldPlant
            "existing_capacity": {("OldPlant", 2020): 2.0},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="mixed_brownfield_vintage",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # OldPlant should use existing capacity
        old_plant_operation_2025 = get_total_operation(solved_model, "OldPlant", 2025)

        # NewPlant may or may not be used depending on optimization
        # The key is that the model is feasible and optimal
        total_operation = sum(
            get_total_operation(solved_model, p, t)
            for p in ["OldPlant", "NewPlant"]
            for t in [2025, 2026, 2027]
        )
        assert total_operation > 0, "Total operation should be positive to meet demand"

    def test_brownfield_vintage_impact_calculation(self):
        """
        Test that impacts are correctly calculated with brownfield + vintage.

        Verifies that:
        1. Installation impacts from existing capacity are excluded (sunk costs)
        2. Operational impacts use correct vintage-specific rates
        """
        base_config = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2_install", "CO2_operate"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0, 1, 2],
            "SYSTEM_TIME": [2025, 2026],
            "CATEGORY": ["GWP"],
            # Use (0, 2) so installations produce immediately at tau=0
            "operation_time_limits": {"Plant": (0, 2)},
            "demand": {("output", 2025): 100, ("output", 2026): 100},
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 10,
                ("Plant", "electricity", 1): 10,
                ("Plant", "electricity", 2): 10,
            },
            "vintage_improvements": {
                ("Plant", "electricity", 2025): 1.0,
                ("Plant", "electricity", 2026): 0.9,
            },
            "internal_demand_technosphere": {},
            # Separate flows for installation vs operation (operational must be constant)
            # Installation emissions are NOT operational (scaled by installation)
            "foreground_biosphere": {
                ("Plant", "CO2_install", 0): 500,  # High installation emissions at construction
            },
            "foreground_production": {
                ("Plant", "output", 0): 100,
                ("Plant", "output", 1): 100,
                ("Plant", "output", 2): 100,
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
                ("Plant", "CO2_install"): False,  # NOT operational - scales with installation
            },
            "background_inventory": {("grid", "electricity", "CO2_operate"): 1.0},
            "mapping": {("grid", 2025): 1.0, ("grid", 2026): 1.0},
            "characterization": {
                ("GWP", "CO2_install", 2025): 1.0,
                ("GWP", "CO2_install", 2026): 1.0,
                ("GWP", "CO2_operate", 2025): 1.0,
                ("GWP", "CO2_operate", 2026): 1.0,
            },
        }

        # Greenfield scenario (no existing capacity)
        greenfield_inputs = converter.OptimizationModelInputs(**base_config)
        greenfield_model = optimizer.create_model(
            inputs=greenfield_inputs,
            objective_category="GWP",
            name="greenfield_vintage",
        )
        _, greenfield_obj, greenfield_results = optimizer.solve_model(
            greenfield_model, solver_name="glpk", tee=False
        )
        assert greenfield_results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Brownfield scenario (existing capacity covers demand)
        brownfield_config = dict(base_config)
        brownfield_config["existing_capacity"] = {("Plant", 2024): 2.0}

        brownfield_inputs = converter.OptimizationModelInputs(**brownfield_config)
        brownfield_model = optimizer.create_model(
            inputs=brownfield_inputs,
            objective_category="GWP",
            name="brownfield_vintage",
        )
        _, brownfield_obj, brownfield_results = optimizer.solve_model(
            brownfield_model, solver_name="glpk", tee=False
        )
        assert brownfield_results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Brownfield should have lower impact (no installation emissions)
        assert brownfield_obj < greenfield_obj, (
            f"Brownfield ({brownfield_obj:.2f}) should have lower impact than "
            f"greenfield ({greenfield_obj:.2f}) due to excluded installation emissions"
        )

    def test_brownfield_multiple_existing_entries_same_process(self):
        """
        Test that multiple existing_capacity entries for the same process work correctly.

        This tests a regression where having multiple existing_capacity entries
        (e.g., installations from 2005 AND 2015) incorrectly doubled the production
        rate in the demand constraint, causing var_operation to be too low.

        The production rate should only be counted ONCE per process, regardless
        of how many existing capacity entries that process has.
        """
        model_inputs_dict = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": list(range(31)),  # 30-year operation
            "SYSTEM_TIME": list(range(2025, 2036)),  # 2025-2035
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (1, 30)},
            # Demand of 1000 per year
            "demand": {("output", t): 1000 for t in range(2025, 2036)},
            "foreground_technosphere": {
                ("Plant", "electricity", tau): 10 for tau in range(31)
            },
            # vintage_improvements triggers 4D path
            "vintage_improvements": {
                ("Plant", "electricity", 2025): 1.0,
                ("Plant", "electricity", 2035): 0.9,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", tau): 100 if 1 <= tau <= 30 else 0
                for tau in range(31)
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            "background_inventory": {("grid", "electricity", "CO2"): 1.0},
            "mapping": {("grid", t): 1.0 for t in range(2025, 2036)},
            "characterization": {("GWP", "CO2", t): 1.0 for t in range(2025, 2036)},
            # MULTIPLE existing capacity entries for the same process
            # This should NOT double the production rate
            "existing_capacity": {
                ("Plant", 2005): 0.5,  # 0.5 units from 2005
                ("Plant", 2015): 0.5,  # 0.5 units from 2015
            },
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="multiple_existing_test",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Check that operation is reasonable (should be ~demand/production_rate = 1000/3000 ≈ 0.33)
        # If the bug exists, operation would be ~0.16 (half of expected) due to doubled rate
        operation_2025 = get_total_operation(solved_model, "Plant", 2025)

        # Production per unit of operation = sum of production rates for operating taus
        # With operation_time_limits (1, 30), there are 30 operating taus
        # Each tau has production = 100, so total = 3000 per unit of operation
        # To meet demand of 1000, we need operation = 1000/3000 ≈ 0.333
        expected_operation = 1000 / 3000  # ≈ 0.333
        assert operation_2025 == pytest.approx(expected_operation, rel=0.01), (
            f"Operation at 2025 should be ~{expected_operation:.4f}, got {operation_2025:.4f}. "
            "If operation is half of expected, the production rate is being doubled incorrectly."
        )
