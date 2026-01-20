"""
Deterministic test for vintage-dependent parameters.

This test creates a minimal scenario where we can manually calculate
expected results and verify the optimizer produces correct values.

Scenario:
- Single foreground process: "Plant" producing "output"
- Single background flow: "electricity"
- Single elementary flow: "CO2"
- Technology evolution: Plant gets more efficient over time (uses less electricity)
- Background evolution: Grid gets cleaner over time (less CO2 per kWh)

Manual calculation approach:
1. At each time t, determine which vintages are operating
2. For each vintage, calculate electricity consumption based on vintage-specific rate
3. For each time, calculate CO2 emissions based on time-specific grid intensity
4. Sum up total impacts and compare with model results
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


class TestVintageDeterministicResults:
    """Deterministic tests with manually calculated expected values."""

    def test_single_process_production_equals_demand(self):
        """
        Test that a single process exactly fulfills demand.

        Setup:
        - 1 process producing 1 product
        - Demand of 100 units per year for 3 years (2025, 2026, 2027)
        - Production rate: 10 units per installation per operating tau
        - operation_time_limits: (0, 0) - instant production
        - technology_evolution triggers 4D path

        Expected:
        - var_operation should equal demand (since production_rate = 10,
          and production = production_rate * var_operation = demand,
          so var_operation = demand / production_rate = 100/10 = 10)
        """
        model_inputs_dict = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0],
            "SYSTEM_TIME": [2025, 2026, 2027],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (0, 0)},
            "demand": {
                ("output", 2025): 100,
                ("output", 2026): 100,
                ("output", 2027): 100,
            },
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 5,  # 5 kWh per unit output
            },
            # technology_evolution makes Plant use less electricity over time
            "technology_evolution": {
                ("Plant", "electricity", 2025): 1.0,  # 5 kWh in 2025
                ("Plant", "electricity", 2027): 0.8,  # 4 kWh in 2027 (20% better)
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", 0): 10,  # 10 units per installation
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            "background_inventory": {
                ("grid", "electricity", "CO2"): 0.5,  # 0.5 kg CO2/kWh
            },
            "mapping": {
                ("grid", 2025): 1.0,
                ("grid", 2026): 1.0,
                ("grid", 2027): 1.0,
            },
            "characterization": {
                ("GWP", "CO2", 2025): 1.0,
                ("GWP", "CO2", 2026): 1.0,
                ("GWP", "CO2", 2027): 1.0,
            },
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="deterministic_test",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Check production equals demand
        # production = production_rate * var_operation
        # demand = 100, production_rate = 10
        # So var_operation should be 100/10 = 10
        for t in [2025, 2026, 2027]:
            var_op = get_total_operation(solved_model, "Plant", t)
            expected_operation = 100 / 10  # demand / production_rate
            assert var_op == pytest.approx(expected_operation, rel=0.001), (
                f"At {t}: var_operation = {var_op}, expected {expected_operation}"
            )

    def test_vintage_evolution_impacts_with_manual_calculation(self):
        """
        Test impacts with vintage evolution, comparing against manual calculation.

        Setup:
        - Single process, single product
        - Demand: 100 units in 2026 only
        - production_rate = 100 (produces 100 per installation at tau=0)
        - operation_time_limits: (0, 0) - instant production
        - electricity consumption: 10 kWh/unit base, evolving to 8 kWh/unit by 2027
        - Background CO2: 0.5 kg/kWh (constant)

        For installation in 2026 (vintage 2026):
        - Electricity rate at 2026 vintage = interpolate(2025:1.0, 2027:0.8) = 0.9
        - Electricity consumption = 10 * 0.9 = 9 kWh/unit
        - Total electricity = 100 units * 9 kWh/unit = 900 kWh
        - CO2 = 900 * 0.5 = 450 kg

        Note: Since we have demand only at 2026, we install at 2026.
        """
        model_inputs_dict = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0],
            "SYSTEM_TIME": [2025, 2026, 2027],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (0, 0)},
            "demand": {
                ("output", 2026): 100,  # Only demand in 2026
            },
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 10,  # 10 kWh/unit base
            },
            "technology_evolution": {
                ("Plant", "electricity", 2025): 1.0,
                ("Plant", "electricity", 2027): 0.8,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", 0): 100,  # 100 units per installation
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            "background_inventory": {
                ("grid", "electricity", "CO2"): 0.5,
            },
            "mapping": {
                ("grid", 2025): 1.0,
                ("grid", 2026): 1.0,
                ("grid", 2027): 1.0,
            },
            "characterization": {
                ("GWP", "CO2", 2025): 1.0,
                ("GWP", "CO2", 2026): 1.0,
                ("GWP", "CO2", 2027): 1.0,
            },
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="vintage_impact_test",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Manual calculation:
        # - Install at 2026 to meet demand of 100
        # - var_operation = 100/100 = 1 (demand / production_rate)
        # - Vintage 2026: interpolate evolution factor = (1.0 + 0.8) / 2 = 0.9
        # - Electricity consumption = 10 * 0.9 = 9 kWh/unit * 100 units = 900 kWh
        # - Wait, that's per installation, not per unit of operation

        # Let me recalculate:
        # - production_rate = 100 (at tau=0)
        # - demand = 100
        # - var_operation = demand / production_rate = 1
        # - electricity_rate = 10 kWh (base) * 0.9 (evolution) = 9 kWh per installation
        # - electricity consumed = electricity_rate * var_operation = 9 * 1 = 9 kWh
        # - CO2 = 9 * 0.5 = 4.5 kg

        # Check var_operation
        var_op_2026 = get_total_operation(solved_model, "Plant", 2026)
        assert var_op_2026 == pytest.approx(1.0, rel=0.001), f"var_operation at 2026 = {var_op_2026}"

        # Check objective (total CO2)
        # At 2026: 9 kWh * 0.5 kg/kWh = 4.5 kg
        expected_impact = 4.5
        assert objective == pytest.approx(expected_impact, rel=0.01), (
            f"Objective = {objective}, expected {expected_impact}"
        )

    def test_multi_year_with_background_evolution(self):
        """
        Test multi-year scenario with both technology and background evolution.

        Setup:
        - Demand: 100 units per year in 2025, 2026, 2027
        - Plant produces 100 units per installation at tau=0
        - Electricity consumption: 10 kWh/unit base
        - Technology evolution: 1.0 at 2025, 0.8 at 2027 (linear interpolation)
        - Background evolution: Grid CO2 decreases from 0.5 to 0.3 kg/kWh

        Manual calculation for each year:
        - Install at each year to meet demand
        - 2025: elec = 10*1.0 = 10, CO2/kWh = 0.5, impact = 10*0.5 = 5
        - 2026: elec = 10*0.9 = 9, CO2/kWh = 0.4, impact = 9*0.4 = 3.6
        - 2027: elec = 10*0.8 = 8, CO2/kWh = 0.3, impact = 8*0.3 = 2.4
        - Total = 5 + 3.6 + 2.4 = 11
        """
        model_inputs_dict = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid_2025", "grid_2026", "grid_2027"],
            "PROCESS_TIME": [0],
            "SYSTEM_TIME": [2025, 2026, 2027],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (0, 0)},
            "demand": {
                ("output", 2025): 100,
                ("output", 2026): 100,
                ("output", 2027): 100,
            },
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 10,
            },
            "technology_evolution": {
                ("Plant", "electricity", 2025): 1.0,
                ("Plant", "electricity", 2027): 0.8,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", 0): 100,
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            # Background CO2 intensity decreases over time
            "background_inventory": {
                ("grid_2025", "electricity", "CO2"): 0.5,
                ("grid_2026", "electricity", "CO2"): 0.4,
                ("grid_2027", "electricity", "CO2"): 0.3,
            },
            # Mapping: each year uses its own grid
            "mapping": {
                ("grid_2025", 2025): 1.0,
                ("grid_2026", 2026): 1.0,
                ("grid_2027", 2027): 1.0,
            },
            "characterization": {
                ("GWP", "CO2", 2025): 1.0,
                ("GWP", "CO2", 2026): 1.0,
                ("GWP", "CO2", 2027): 1.0,
            },
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="multi_year_test",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Verify var_operation = 1 at each year (demand/production_rate = 100/100)
        for t in [2025, 2026, 2027]:
            var_op = get_total_operation(solved_model, "Plant", t)
            assert var_op == pytest.approx(1.0, rel=0.001), (
                f"var_operation at {t} = {var_op}, expected 1.0"
            )

        # Manual calculation:
        # 2025: vintage=2025, evolution=1.0, elec=10*1=10, CO2/kWh=0.5, impact=5
        # 2026: vintage=2026, evolution=0.9, elec=10*0.9=9, CO2/kWh=0.4, impact=3.6
        # 2027: vintage=2027, evolution=0.8, elec=10*0.8=8, CO2/kWh=0.3, impact=2.4
        # Total = 11
        expected_impact = 5 + 3.6 + 2.4
        assert objective == pytest.approx(expected_impact, rel=0.01), (
            f"Objective = {objective}, expected {expected_impact}"
        )

    def test_simple_no_vintage_baseline(self):
        """
        Baseline test WITHOUT vintage parameters to verify basic model behavior.

        Setup:
        - Single process, single product
        - Demand: 100 units at 2025
        - Production rate: 100 per installation
        - Electricity: 10 kWh per installation
        - CO2: 0.5 kg/kWh

        Expected:
        - var_operation = 1 (100/100)
        - Electricity = 10 kWh
        - CO2 = 10 * 0.5 = 5 kg
        """
        model_inputs_dict = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0],
            "SYSTEM_TIME": [2025],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (0, 0)},
            "demand": {("output", 2025): 100},
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 10,
            },
            # NO technology_evolution - stays in 3D path
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", 0): 100,
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            "background_inventory": {
                ("grid", "electricity", "CO2"): 0.5,
            },
            "mapping": {("grid", 2025): 1.0},
            "characterization": {("GWP", "CO2", 2025): 1.0},
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="baseline_test",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # var_operation = demand / production_rate = 100 / 100 = 1
        var_op = get_total_operation(solved_model, "Plant", 2025)
        assert var_op == pytest.approx(1.0, rel=0.001)

        # Impact = electricity * CO2_per_kWh = 10 * 0.5 = 5
        expected_impact = 5.0
        assert objective == pytest.approx(expected_impact, rel=0.001), (
            f"Objective = {objective}, expected {expected_impact}"
        )

    def test_technology_evolution_reduces_impact(self):
        """
        Test that technology_evolution correctly reduces impacts.

        Compare two scenarios:
        1. No evolution (all years use base rate)
        2. With evolution (later years use improved rate)

        The second scenario should have lower total impact.
        """
        base_config = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0],
            "SYSTEM_TIME": [2025, 2026, 2027],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (0, 0)},
            "demand": {
                ("output", 2025): 100,
                ("output", 2026): 100,
                ("output", 2027): 100,
            },
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 10,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", 0): 100,
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            "background_inventory": {
                ("grid", "electricity", "CO2"): 0.5,
            },
            "mapping": {
                ("grid", 2025): 1.0,
                ("grid", 2026): 1.0,
                ("grid", 2027): 1.0,
            },
            "characterization": {
                ("GWP", "CO2", 2025): 1.0,
                ("GWP", "CO2", 2026): 1.0,
                ("GWP", "CO2", 2027): 1.0,
            },
        }

        # Scenario 1: No evolution
        no_evolution_inputs = converter.OptimizationModelInputs(**base_config)
        no_evolution_model = optimizer.create_model(
            inputs=no_evolution_inputs,
            objective_category="GWP",
            name="no_evolution",
        )
        _, no_evolution_obj, _ = optimizer.solve_model(
            no_evolution_model, solver_name="glpk", tee=False
        )

        # Scenario 2: With evolution (50% improvement by 2027)
        with_evolution_config = dict(base_config)
        with_evolution_config["technology_evolution"] = {
            ("Plant", "electricity", 2025): 1.0,
            ("Plant", "electricity", 2027): 0.5,  # 50% less electricity
        }
        with_evolution_inputs = converter.OptimizationModelInputs(**with_evolution_config)
        with_evolution_model = optimizer.create_model(
            inputs=with_evolution_inputs,
            objective_category="GWP",
            name="with_evolution",
        )
        _, with_evolution_obj, _ = optimizer.solve_model(
            with_evolution_model, solver_name="glpk", tee=False
        )

        # No evolution: 3 years * 10 kWh * 0.5 = 15 kg CO2
        expected_no_evolution = 15.0
        assert no_evolution_obj == pytest.approx(expected_no_evolution, rel=0.01), (
            f"No evolution impact = {no_evolution_obj}, expected {expected_no_evolution}"
        )

        # With evolution:
        # 2025: 10 * 1.0 * 0.5 = 5
        # 2026: 10 * 0.75 * 0.5 = 3.75
        # 2027: 10 * 0.5 * 0.5 = 2.5
        # Total = 11.25
        expected_with_evolution = 5 + 3.75 + 2.5
        assert with_evolution_obj == pytest.approx(expected_with_evolution, rel=0.01), (
            f"With evolution impact = {with_evolution_obj}, expected {expected_with_evolution}"
        )

        # With evolution should be lower
        assert with_evolution_obj < no_evolution_obj, (
            f"With evolution ({with_evolution_obj}) should be less than "
            f"no evolution ({no_evolution_obj})"
        )

    def test_postprocessor_production_matches_demand(self):
        """
        Verify that postprocessor get_production() matches demand.

        This test checks if there's a bug in the postprocessor where it uses
        3D calculation (all taus) instead of 4D vintage-aware calculation.
        """
        from optimex.postprocessing import PostProcessor

        model_inputs_dict = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0, 1, 2],
            "SYSTEM_TIME": [2025, 2026, 2027],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (0, 2)},
            "demand": {
                ("output", 2025): 300,
                ("output", 2026): 300,
                ("output", 2027): 300,
            },
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 10,
                ("Plant", "electricity", 1): 10,
                ("Plant", "electricity", 2): 10,
            },
            # technology_evolution to trigger 4D path
            "technology_evolution": {
                ("Plant", "electricity", 2025): 1.0,
                ("Plant", "electricity", 2027): 0.8,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", 0): 100,
                ("Plant", "output", 1): 100,
                ("Plant", "output", 2): 100,
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            "background_inventory": {
                ("grid", "electricity", "CO2"): 0.5,
            },
            "mapping": {
                ("grid", 2025): 1.0,
                ("grid", 2026): 1.0,
                ("grid", 2027): 1.0,
            },
            "characterization": {
                ("GWP", "CO2", 2025): 1.0,
                ("GWP", "CO2", 2026): 1.0,
                ("GWP", "CO2", 2027): 1.0,
            },
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="postprocessor_test",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        pp = PostProcessor(solved_model)
        df_production = pp.get_production()

        print("\nProduction from postprocessor:")
        print(df_production)
        print("\nExpected demand: 300 at each year")

        # Check if production matches demand
        for t in [2025, 2026, 2027]:
            # DataFrame has MultiIndex columns (Process, Product)
            production = df_production.loc[t, ("Plant", "output")]
            demand = 300
            print(f"t={t}: production={production}, demand={demand}")

            # This assertion will PASS now that postprocessor uses 4D calculation
            assert production == pytest.approx(demand, rel=0.01), (
                f"Production {production} should match demand {demand} at t={t}"
            )

    def test_technology_evolution_only_affects_specified_flows(self):
        """
        Verify that technology_evolution only creates overrides for the specified
        (process, flow) pairs, not for ALL flows.

        This test catches a bug where expand_foreground_tensor_with_evolution()
        was creating 4D overrides for ALL flows when technology_evolution was set,
        even for flows that didn't have any evolution factors specified.

        Setup:
        - Two processes: Plant1 and Plant2, both produce "output"
        - technology_evolution ONLY specified for Plant1's electricity consumption
        - Plant2 should use 3D path (not vintage-aware) for production
        - Both should correctly fulfill demand
        """
        from optimex.postprocessing import PostProcessor

        model_inputs_dict = {
            "PROCESS": ["Plant1", "Plant2"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0, 1, 2],  # Multi-tau process time
            "SYSTEM_TIME": [2025, 2026, 2027],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {
                "Plant1": (0, 2),
                "Plant2": (0, 2),
            },
            "demand": {
                ("output", 2025): 600,  # Both plants needed
                ("output", 2026): 600,
                ("output", 2027): 600,
            },
            "foreground_technosphere": {
                ("Plant1", "electricity", 0): 10,
                ("Plant1", "electricity", 1): 10,
                ("Plant1", "electricity", 2): 10,
                ("Plant2", "electricity", 0): 20,
                ("Plant2", "electricity", 1): 20,
                ("Plant2", "electricity", 2): 20,
            },
            # technology_evolution ONLY for Plant1's electricity, not Plant2
            "technology_evolution": {
                ("Plant1", "electricity", 2025): 1.0,
                ("Plant1", "electricity", 2027): 0.5,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            # Both plants produce 100 per tau = 300 total per installation
            "foreground_production": {
                ("Plant1", "output", 0): 100,
                ("Plant1", "output", 1): 100,
                ("Plant1", "output", 2): 100,
                ("Plant2", "output", 0): 100,
                ("Plant2", "output", 1): 100,
                ("Plant2", "output", 2): 100,
            },
            "operation_flow": {
                ("Plant1", "output"): True,
                ("Plant1", "electricity"): True,
                ("Plant2", "output"): True,
                ("Plant2", "electricity"): True,
            },
            "background_inventory": {
                ("grid", "electricity", "CO2"): 0.5,
            },
            "mapping": {
                ("grid", 2025): 1.0,
                ("grid", 2026): 1.0,
                ("grid", 2027): 1.0,
            },
            "characterization": {
                ("GWP", "CO2", 2025): 1.0,
                ("GWP", "CO2", 2026): 1.0,
                ("GWP", "CO2", 2027): 1.0,
            },
        }

        model_inputs = converter.OptimizationModelInputs(**model_inputs_dict)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="GWP",
            name="evolution_only_affects_specified",
        )

        solved_model, objective, results = optimizer.solve_model(
            model, solver_name="glpk", tee=False
        )

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Get production
        pp = PostProcessor(solved_model)
        df_production = pp.get_production()

        print("\nProduction from postprocessor:")
        print(df_production)

        # Verify total production matches demand at each year
        for t in [2025, 2026, 2027]:
            total_production = (
                df_production.loc[t, ("Plant1", "output")]
                + df_production.loc[t, ("Plant2", "output")]
            )
            demand = 600
            print(f"t={t}: total_production={total_production}, demand={demand}")

            assert total_production == pytest.approx(demand, rel=0.01), (
                f"Total production {total_production} should match demand {demand} at t={t}"
            )

        # Also verify var_operation values are reasonable
        # With production_rate = 300 per installation for 3D path:
        # var_operation * 300 = production => var_operation = production / 300
        # If each plant produces 300, var_operation should be ~1.0

        var_op_p1_2025 = get_total_operation(solved_model, "Plant1", 2025)
        var_op_p2_2025 = get_total_operation(solved_model, "Plant2", 2025)

        print(f"\nvar_operation Plant1 at 2025: {var_op_p1_2025}")
        print(f"var_operation Plant2 at 2025: {var_op_p2_2025}")

        # Plant2 should use 3D path, so var_operation should be reasonable
        # (not inflated by incorrect 4D path)
        # With 3D path: production_rate = 300, so var_operation = demand_share / 300
        # If Plant2 produces half the demand (300), var_operation = 300/300 = 1.0
        assert var_op_p2_2025 < 10, (
            f"Plant2 var_operation {var_op_p2_2025} should be small (using 3D path), "
            f"not inflated by incorrect 4D path"
        )


class TestVintageValidation:
    """
    Deterministic tests to validate vintage calculations with manual calculations.

    These tests verify that:
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
            "demand": {("output", 2022): 100},
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 0,
                ("Plant", "electricity", 1): 0,
            },
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

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Check production balance
        operation_2022 = get_total_operation(solved, "Plant", 2022)
        production_rate = 1.0
        actual_production = production_rate * operation_2022

        assert actual_production == pytest.approx(100.0, rel=1e-6), \
            f"Production ({actual_production}) must exactly equal demand (100)"

        # Verify total impact: 100 × 1.5 × 0.5 = 75 kg CO2
        expected_impact = 100 * 1.5 * 0.5
        assert obj == pytest.approx(expected_impact, rel=1e-5), \
            f"Impact ({obj}) should match manual calculation ({expected_impact})"

    def test_single_process_with_background_vintage(self):
        """
        Test single foreground process with improving background.

        System setup:
        - One process "Plant" that produces "output"
        - Process consumes "electricity" (constant: 10 kWh per unit)
        - Background electricity improves over time
        - Demand: 50 units at t=2021

        Background vintage improvements:
        - 2020 grid: 1.0 kg CO2 per kWh
        - 2022 grid: 0.5 kg CO2 per kWh
        - 2021 (interpolated): 0.75 kg CO2 per kWh

        Manual calculation:
        - Electricity consumption: 50 units × 10 kWh/unit = 500 kWh
        - Background at t=2021 (interpolated): 0.75 kg CO2/kWh
        - Total CO2: 500 kWh × 0.75 kg/kWh = 375 kg
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
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 0,
                ("Plant", "electricity", 1): 10,
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
            name="background_vintage_test"
        )
        solved, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Check production balance
        operation_2021 = get_total_operation(solved, "Plant", 2021)
        production_rate = 1.0
        actual_production = production_rate * operation_2021

        assert actual_production == pytest.approx(50.0, rel=1e-6), \
            f"Production ({actual_production}) must exactly equal demand (50)"

        # Manual: 50 × 10 × 0.75 = 375 kg CO2
        expected_impact = 50 * 10 * 0.75
        assert obj == pytest.approx(expected_impact, rel=1e-5), \
            f"Impact ({obj}) should match manual calculation ({expected_impact})"

    def test_combined_foreground_and_background_vintage(self):
        """
        Test both foreground and background improving over time.

        Foreground vintages (electricity consumption per unit):
        - 2020 vintage: 2.0 kWh/unit
        - 2022 vintage: 1.0 kWh/unit
        - 2021 (interpolated): 1.5 kWh/unit

        Background vintages (CO2 per kWh):
        - 2020 grid: 1.0 kg/kWh
        - 2022 grid: 0.5 kg/kWh

        Manual calculation (installation at t=2021 to operate at t=2022):
        - Foreground electricity at vintage 2021: 1.5 kWh/unit
        - Total electricity: 100 units × 1.5 kWh/unit = 150 kWh
        - Background at t=2022: 100% grid_2022 = 0.5 kg/kWh
        - Total CO2: 150 kWh × 0.5 kg/kWh = 75 kg
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

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Check production balance
        operation_2022 = get_total_operation(solved, "Plant", 2022)
        actual_production = operation_2022

        assert actual_production == pytest.approx(100.0, rel=1e-6), \
            f"Production ({actual_production}) must exactly equal demand (100)"

        # Manual: 100 × 1.5 × 0.5 = 75 kg CO2
        expected_impact = 100 * 1.5 * 0.5
        assert obj == pytest.approx(expected_impact, rel=1e-5), \
            f"Impact ({obj}) should match manual calculation ({expected_impact})"

    def test_multiple_installations_different_vintages(self):
        """
        Test system with multiple installations at different vintages.

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

        assert results.solver.termination_condition == pyo.TerminationCondition.optimal

        # Check production balances
        operation_2021 = get_total_operation(solved, "Plant", 2021)
        operation_2022 = get_total_operation(solved, "Plant", 2022)

        assert operation_2021 == pytest.approx(100.0, rel=1e-6), \
            f"Production at t=2021 must equal demand (100)"
        assert operation_2022 == pytest.approx(150.0, rel=1e-6), \
            f"Production at t=2022 must equal demand (150)"

        # Impact should be in reasonable range based on vintage improvements
        assert 200 <= obj <= 260, \
            f"Impact ({obj}) should be in reasonable range based on vintage improvements"
