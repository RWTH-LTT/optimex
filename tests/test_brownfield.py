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
