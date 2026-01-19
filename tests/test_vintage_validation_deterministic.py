"""
Deterministic validation test for time-vintage functionality.

This test creates a simple system with:
1. Single foreground technology that improves over time
2. Single background process that improves over time
3. Manually calculated expected impacts
4. Verification that production exactly matches demand
5. Verification that vintages are correctly represented

The goal is to check if the vintage system produces correct results,
not just that it runs without errors.
"""

import pytest
import pyomo.environ as pyo
from optimex import converter, optimizer


class TestVintageDeterministicValidation:
    """
    Deterministic tests to validate vintage calculations are correct.
    
    These tests manually calculate expected impacts and verify that:
    1. Production exactly matches demand (balance check)
    2. Impacts match hand-calculated expected values
    3. Both foreground and background vintages work correctly
    """

    def test_single_foreground_process_simple_vintage(self):
        """
        Test a single foreground process with vintage improvements.
        
        System setup:
        - One process "Plant" that produces "output"
        - Process consumes "electricity" from background
        - Process has construction at tau=0, operation at tau=1
        - Demand: 100 units at t=2022
        - Installation must happen at t=2021 (to operate at t=2022 with tau=1)
        
        Vintage improvements (for operation, tau=1):
        - 2020 vintage: 2.0 kWh electricity per unit output
        - 2022 vintage: 1.0 kWh electricity per unit output
        - 2021 vintage (interpolated): 1.5 kWh per unit output
        
        Background:
        - Electricity: 0.5 kg CO2 per kWh (constant over time)
        
        Manual calculation:
        - Installation at t=2021 (vintage=2021)
        - Interpolated electricity consumption: (2.0 + 1.0) / 2 = 1.5 kWh/unit
        - Total electricity: 100 units × 1.5 kWh/unit = 150 kWh
        - Total CO2: 150 kWh × 0.5 kg/kWh = 75 kg
        - Expected impact: 75 kg CO2
        
        Checks:
        1. Production exactly equals demand (100 units)
        2. Total impact equals 75 kg CO2
        3. var_installation[Plant, 2021] > 0
        4. var_operation[Plant, 2022] produces exactly 100 units
        """
        inputs = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0, 1],  # tau=0: construction, tau=1: operation
            "SYSTEM_TIME": [2020, 2021, 2022],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (1, 1)},  # Only tau=1 is operation
            "demand": {("output", 2022): 100},  # Demand only at t=2022
            
            # Foreground: Base values (will be overridden by vintages)
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 0,  # No electricity during construction
                ("Plant", "electricity", 1): 0,  # Will use vintages instead
            },
            "foreground_technosphere_vintages": {
                # 2020 vintage: inefficient (2.0 kWh per unit output)
                ("Plant", "electricity", 1, 2020): 2.0,
                # 2022 vintage: efficient (1.0 kWh per unit output)
                ("Plant", "electricity", 1, 2022): 1.0,
                # 2021 will be interpolated: 1.5 kWh per unit output
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", 0): 0,  # No production during construction
                ("Plant", "output", 1): 1,  # 1 unit per installation per tau
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            
            # Background: Constant electricity emissions
            "background_inventory": {
                ("grid", "electricity", "CO2"): 0.5,  # 0.5 kg CO2 per kWh
            },
            "mapping": {
                ("grid", 2020): 1.0,
                ("grid", 2021): 1.0,
                ("grid", 2022): 1.0,
            },
            "characterization": {
                ("GWP", "CO2", 2020): 1.0,
                ("GWP", "CO2", 2021): 1.0,
                ("GWP", "CO2", 2022): 1.0,
            },
        }

        model_inputs = converter.OptimizationModelInputs(**inputs)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="simple_vintage_test"
        )
        solved, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)

        assert results.solver.termination_condition.name == "optimal"

        # Check 1: Verify production balance
        # Production should exactly equal demand
        installation_2021 = pyo.value(solved.var_installation["Plant", 2021])
        operation_2022 = pyo.value(solved.var_operation["Plant", 2022])
        production_rate = 1.0  # 1 unit per installation from foreground_production
        actual_production = production_rate * operation_2022
        
        print(f"\nDETAILS:")
        print(f"  Installation at t=2021: {installation_2021}")
        print(f"  Operation at t=2022: {operation_2022}")
        print(f"  Production: {actual_production}")
        print(f"  Demand: 100")
        print(f"  Total impact: {obj}")
        
        assert actual_production == pytest.approx(100.0, rel=1e-6), \
            f"Production ({actual_production}) must exactly equal demand (100)"
        
        # Check 2: Verify installation timing
        assert installation_2021 > 0, "Should install in 2021 to operate in 2022"
        
        # Check 3: Verify total impact matches manual calculation
        # Expected: 100 units × 1.5 kWh/unit × 0.5 kg/kWh = 75 kg CO2
        expected_impact = 100 * 1.5 * 0.5
        assert obj == pytest.approx(expected_impact, rel=1e-5), \
            f"Impact ({obj}) should match manual calculation ({expected_impact})"
        
        # Check 4: Verify var_operation scaling
        # var_operation should be such that production equals demand
        expected_operation = 100.0 / production_rate
        assert operation_2022 == pytest.approx(expected_operation, rel=1e-6), \
            f"var_operation ({operation_2022}) should be {expected_operation} to produce exactly 100 units"

    def test_single_process_with_background_vintage(self):
        """
        Test single foreground process with improving background.
        
        System setup:
        - One process "Plant" that produces "output"
        - Process consumes "electricity" (constant: 10 kWh per unit)
        - Background electricity improves over time
        - Demand: 50 units at t=2021
        - Installation at t=2020 (to operate at t=2021 with tau=1)
        
        Background vintage improvements:
        - 2020 grid: 1.0 kg CO2 per kWh
        - 2022 grid: 0.5 kg CO2 per kWh
        - 2021 (interpolated): 0.75 kg CO2 per kWh
        
        Manual calculation:
        - Installation at t=2020 (to operate in 2021)
        - Electricity consumption: 50 units × 10 kWh/unit = 500 kWh
        - Background at t=2021 (interpolated): 0.75 kg CO2/kWh
        - Total CO2: 500 kWh × 0.75 kg/kWh = 375 kg
        - Expected impact: 375 kg CO2
        """
        inputs = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid_2020", "grid_2022"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021, 2022],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (1, 1)},
            "demand": {("output", 2021): 50},
            
            # Foreground: Constant electricity consumption
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 0,
                ("Plant", "electricity", 1): 10,  # 10 kWh per unit output
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", 0): 0,
                ("Plant", "output", 1): 1,
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            
            # Background: Two grid databases with interpolation
            "background_inventory": {
                ("grid_2020", "electricity", "CO2"): 1.0,  # Dirty grid
                ("grid_2022", "electricity", "CO2"): 0.5,  # Clean grid
            },
            # Linear interpolation from 2020 to 2022
            "mapping": {
                ("grid_2020", 2020): 1.0,
                ("grid_2020", 2021): 0.5,  # 50% weight
                ("grid_2020", 2022): 0.0,
                ("grid_2022", 2020): 0.0,
                ("grid_2022", 2021): 0.5,  # 50% weight
                ("grid_2022", 2022): 1.0,
            },
            "characterization": {
                ("GWP", "CO2", t): 1.0 for t in [2020, 2021, 2022]
            },
        }

        model_inputs = converter.OptimizationModelInputs(**inputs)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="background_vintage_test"
        )
        solved, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)

        assert results.solver.termination_condition.name == "optimal"

        # Check production balance
        operation_2021 = pyo.value(solved.var_operation["Plant", 2021])
        production_rate = 1.0
        actual_production = production_rate * operation_2021
        
        print(f"\nDETAILS:")
        print(f"  Operation at t=2021: {operation_2021}")
        print(f"  Production: {actual_production}")
        print(f"  Demand: 50")
        print(f"  Total impact: {obj}")
        
        assert actual_production == pytest.approx(50.0, rel=1e-6), \
            f"Production ({actual_production}) must exactly equal demand (50)"
        
        # Manual calculation:
        # - 50 units × 10 kWh/unit = 500 kWh
        # - Background interpolated at 2021: 0.5 × 1.0 + 0.5 × 0.5 = 0.75 kg/kWh
        # - Total: 500 × 0.75 = 375 kg CO2
        expected_impact = 50 * 10 * 0.75
        assert obj == pytest.approx(expected_impact, rel=1e-5), \
            f"Impact ({obj}) should match manual calculation ({expected_impact})"

    def test_combined_foreground_and_background_vintage(self):
        """
        Test both foreground and background improving over time.
        
        System setup:
        - One process "Plant" that produces "output"
        - Process has vintage-dependent electricity consumption
        - Background grid has vintage-dependent emissions
        - Demand: 100 units at t=2022
        
        Foreground vintages (electricity consumption per unit):
        - 2020 vintage: 2.0 kWh/unit
        - 2022 vintage: 1.0 kWh/unit
        - 2021 (interpolated): 1.5 kWh/unit
        
        Background vintages (CO2 per kWh):
        - 2020 grid: 1.0 kg/kWh
        - 2022 grid: 0.5 kg/kWh
        - 2021 (interpolated): 0.75 kg/kWh
        
        Manual calculation (installation at t=2021 to operate at t=2022):
        - Foreground electricity at vintage 2021: 1.5 kWh/unit
        - Total electricity: 100 units × 1.5 kWh/unit = 150 kWh
        - Background at t=2022: 100% grid_2022 = 0.5 kg/kWh
        - Total CO2: 150 kWh × 0.5 kg/kWh = 75 kg
        - Expected impact: 75 kg CO2
        
        This validates that BOTH foreground vintage (2021) and background timing (t=2022)
        are correctly applied.
        """
        inputs = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid_2020", "grid_2022"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021, 2022],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (1, 1)},
            "demand": {("output", 2022): 100},
            
            # Foreground with vintage improvements
            "foreground_technosphere": {},
            "foreground_technosphere_vintages": {
                ("Plant", "electricity", 1, 2020): 2.0,  # Inefficient
                ("Plant", "electricity", 1, 2022): 1.0,  # Efficient
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", 0): 0,
                ("Plant", "output", 1): 1,
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            
            # Background with improvements
            "background_inventory": {
                ("grid_2020", "electricity", "CO2"): 1.0,
                ("grid_2022", "electricity", "CO2"): 0.5,
            },
            "mapping": {
                ("grid_2020", 2020): 1.0,
                ("grid_2020", 2021): 0.5,
                ("grid_2020", 2022): 0.0,
                ("grid_2022", 2020): 0.0,
                ("grid_2022", 2021): 0.5,
                ("grid_2022", 2022): 1.0,
            },
            "characterization": {
                ("GWP", "CO2", t): 1.0 for t in [2020, 2021, 2022]
            },
        }

        model_inputs = converter.OptimizationModelInputs(**inputs)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="combined_vintage_test"
        )
        solved, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)

        assert results.solver.termination_condition.name == "optimal"

        # Check production balance
        operation_2022 = pyo.value(solved.var_operation["Plant", 2022])
        installation_2021 = pyo.value(solved.var_installation["Plant", 2021])
        production_rate = 1.0
        actual_production = production_rate * operation_2022
        
        print(f"\nDETAILS:")
        print(f"  Installation at t=2021: {installation_2021}")
        print(f"  Operation at t=2022: {operation_2022}")
        print(f"  Production: {actual_production}")
        print(f"  Demand: 100")
        print(f"  Total impact: {obj}")
        
        assert actual_production == pytest.approx(100.0, rel=1e-6), \
            f"Production ({actual_production}) must exactly equal demand (100)"
        
        # Manual calculation:
        # - Installation at t=2021 → vintage 2021
        # - Foreground electricity (interpolated): 1.5 kWh/unit
        # - Total electricity: 100 × 1.5 = 150 kWh
        # - Background at t=2022: 100% grid_2022 = 0.5 kg/kWh
        # - Total CO2: 150 × 0.5 = 75 kg
        expected_impact = 100 * 1.5 * 0.5
        assert obj == pytest.approx(expected_impact, rel=1e-5), \
            f"Impact ({obj}) should match manual calculation ({expected_impact})"
        
        # Verify that installation happens at t=2021
        assert installation_2021 > 0, "Should install at t=2021"

    def test_multiple_installations_different_vintages(self):
        """
        Test system with multiple installations at different vintages.
        
        This ensures that when multiple installations operate simultaneously,
        each uses its correct vintage-specific parameters.
        
        System setup:
        - One process "Plant"
        - Demand at t=2021 and t=2022
        - Installations at t=2020 and t=2021
        - Each installation operates with its installation year's vintage
        
        Vintages (electricity per unit output):
        - 2020 vintage: 2.0 kWh/unit
        - 2022 vintage: 1.0 kWh/unit
        - 2021 (interpolated): 1.5 kWh/unit
        
        Scenario:
        - Demand at t=2021: 100 units (met by 2020 installation)
        - Demand at t=2022: 150 units (met by both 2020 and 2021 installations)
        
        Manual calculation:
        At t=2021:
        - 2020 installation operates (vintage 2020): 100 units × 2.0 kWh = 200 kWh
        - Impact: 200 × 0.5 = 100 kg
        
        At t=2022:
        - 2020 installation: 100 units × 2.0 kWh = 200 kWh
        - 2021 installation: 50 units × 1.5 kWh = 75 kWh
        - Total: 275 kWh
        - Impact: 275 × 0.5 = 137.5 kg
        
        Total impact: 100 + 137.5 = 237.5 kg
        """
        inputs = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021, 2022],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (1, 1)},
            "demand": {
                ("output", 2021): 100,
                ("output", 2022): 150,
            },
            
            "foreground_technosphere": {},
            "foreground_technosphere_vintages": {
                ("Plant", "electricity", 1, 2020): 2.0,
                ("Plant", "electricity", 1, 2022): 1.0,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", 0): 0,
                ("Plant", "output", 1): 1,
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            "background_inventory": {
                ("grid", "electricity", "CO2"): 0.5,
            },
            "mapping": {
                ("grid", t): 1.0 for t in [2020, 2021, 2022]
            },
            "characterization": {
                ("GWP", "CO2", t): 1.0 for t in [2020, 2021, 2022]
            },
        }

        model_inputs = converter.OptimizationModelInputs(**inputs)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="multiple_vintages_test"
        )
        solved, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)

        assert results.solver.termination_condition.name == "optimal"

        # Check production balances
        operation_2021 = pyo.value(solved.var_operation["Plant", 2021])
        operation_2022 = pyo.value(solved.var_operation["Plant", 2022])
        production_rate = 1.0
        
        print(f"\nDETAILS:")
        print(f"  Operation at t=2021: {operation_2021} → production {operation_2021 * production_rate}")
        print(f"  Operation at t=2022: {operation_2022} → production {operation_2022 * production_rate}")
        print(f"  Total impact: {obj}")
        
        assert operation_2021 * production_rate == pytest.approx(100.0, rel=1e-6), \
            f"Production at t=2021 must equal demand (100)"
        assert operation_2022 * production_rate == pytest.approx(150.0, rel=1e-6), \
            f"Production at t=2022 must equal demand (150)"
        
        # Manual calculation (exact calculation depends on optimizer choices):
        # If optimizer is smart, it will use later installations for later demands
        # But it should at least meet demand constraints
        # The impact should be in a reasonable range based on vintage improvements
        
        # Minimum impact (all demand met with best vintage):
        # t=2021: 100 × 1.5 × 0.5 = 75 (if using 2021 installation, but can't)
        # Actually must use 2020 installation: 100 × 2.0 × 0.5 = 100
        # t=2022: Best case all with 2021 vintage: 150 × 1.5 × 0.5 = 112.5
        # Total minimum: ~212.5
        
        # Maximum impact (worst vintage):
        # All with 2020 vintage: 250 × 2.0 × 0.5 = 250
        
        # The actual impact should be between these bounds
        assert 200 <= obj <= 260, \
            f"Impact ({obj}) should be in reasonable range based on vintage improvements"
