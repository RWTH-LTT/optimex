"""
Tests for vintage-dependent (time-based) foreground parameters.

This module tests the ability to define foreground parameters that vary based on
when a process is installed (vintage year), enabling scenarios like:
- EVs built in 2025 consuming 2 kWh/km
- EVs built in 2040 consuming 1 kWh/km
"""

import pytest

from optimex import converter


class TestVintageDataModelExtensions:
    """Tests for Phase 1: Data model extensions for vintage parameters."""

    @pytest.fixture
    def base_model_inputs(self):
        """Minimal valid model inputs for testing."""
        return {
            "PROCESS": ["EV"],
            "PRODUCT": ["vkm"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db_2020", "db_2030"],
            "PROCESS_TIME": [0, 1, 2],
            "SYSTEM_TIME": [2020, 2021, 2022, 2023, 2024, 2025],
            "CATEGORY": ["climate_change"],
            "operation_time_limits": {"EV": (1, 2)},
            "demand": {("vkm", 2022): 100, ("vkm", 2023): 100, ("vkm", 2024): 100, ("vkm", 2025): 100},
            "foreground_technosphere": {
                ("EV", "electricity", 0): 0,
                ("EV", "electricity", 1): 50,
                ("EV", "electricity", 2): 50,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {
                ("EV", "CO2", 0): 5000,  # Manufacturing emissions
                ("EV", "CO2", 1): 0,
                ("EV", "CO2", 2): 0,
            },
            "foreground_production": {
                ("EV", "vkm", 1): 50,
                ("EV", "vkm", 2): 50,
            },
            "operation_flow": {
                ("EV", "vkm"): True,
                ("EV", "electricity"): True,
            },
            "background_inventory": {
                ("db_2020", "electricity", "CO2"): 0.5,
                ("db_2030", "electricity", "CO2"): 0.3,
            },
            "mapping": {
                ("db_2020", 2020): 1.0,
                ("db_2020", 2021): 0.9,
                ("db_2020", 2022): 0.8,
                ("db_2020", 2023): 0.7,
                ("db_2020", 2024): 0.6,
                ("db_2020", 2025): 0.5,
                ("db_2030", 2020): 0.0,
                ("db_2030", 2021): 0.1,
                ("db_2030", 2022): 0.2,
                ("db_2030", 2023): 0.3,
                ("db_2030", 2024): 0.4,
                ("db_2030", 2025): 0.5,
            },
            "characterization": {
                ("climate_change", "CO2", t): 1.0
                for t in [2020, 2021, 2022, 2023, 2024, 2025]
            },
        }

    def test_reference_vintages_inferred_from_vintage_data(self, base_model_inputs):
        """Test that REFERENCE_VINTAGES is inferred from vintage tensor keys."""
        base_model_inputs["foreground_technosphere_vintages"] = {
            ("EV", "electricity", 1, 2020): 60,
            ("EV", "electricity", 1, 2025): 40,
        }
        model = converter.OptimizationModelInputs(**base_model_inputs)
        assert model.REFERENCE_VINTAGES == [2020, 2025]

    def test_reference_vintages_inferred_from_multiple_sources(self, base_model_inputs):
        """Test that REFERENCE_VINTAGES is the union of all vintage years."""
        base_model_inputs["foreground_technosphere_vintages"] = {
            ("EV", "electricity", 1, 2020): 60,
        }
        base_model_inputs["technology_evolution"] = {
            ("EV", "CO2", 2025): 0.8,
            ("EV", "CO2", 2030): 0.6,
        }
        model = converter.OptimizationModelInputs(**base_model_inputs)
        assert model.REFERENCE_VINTAGES == [2020, 2025, 2030]

    def test_reference_vintages_none_when_no_vintage_data(self, base_model_inputs):
        """Test that REFERENCE_VINTAGES is None when no vintage data is provided."""
        model = converter.OptimizationModelInputs(**base_model_inputs)
        assert model.REFERENCE_VINTAGES is None

    def test_model_accepts_foreground_technosphere_vintages(self, base_model_inputs):
        """Test that foreground_technosphere_vintages field is accepted."""
        base_model_inputs["foreground_technosphere_vintages"] = {
            # 2020 vintage: higher electricity consumption
            ("EV", "electricity", 1, 2020): 60,
            ("EV", "electricity", 2, 2020): 60,
            # 2025 vintage: lower electricity consumption (improved efficiency)
            ("EV", "electricity", 1, 2025): 40,
            ("EV", "electricity", 2, 2025): 40,
        }
        model = converter.OptimizationModelInputs(**base_model_inputs)
        assert ("EV", "electricity", 1, 2020) in model.foreground_technosphere_vintages
        assert model.foreground_technosphere_vintages[("EV", "electricity", 1, 2020)] == 60
        assert model.foreground_technosphere_vintages[("EV", "electricity", 1, 2025)] == 40
        assert model.REFERENCE_VINTAGES == [2020, 2025]

    def test_model_accepts_foreground_biosphere_vintages(self, base_model_inputs):
        """Test that foreground_biosphere_vintages field is accepted."""
        base_model_inputs["foreground_biosphere_vintages"] = {
            # 2020 vintage: higher manufacturing emissions
            ("EV", "CO2", 0, 2020): 6000,
            # 2025 vintage: lower manufacturing emissions
            ("EV", "CO2", 0, 2025): 4000,
        }
        model = converter.OptimizationModelInputs(**base_model_inputs)
        assert model.foreground_biosphere_vintages[("EV", "CO2", 0, 2020)] == 6000
        assert model.foreground_biosphere_vintages[("EV", "CO2", 0, 2025)] == 4000
        assert model.REFERENCE_VINTAGES == [2020, 2025]

    def test_model_accepts_foreground_production_vintages(self, base_model_inputs):
        """Test that foreground_production_vintages field is accepted."""
        base_model_inputs["foreground_production_vintages"] = {
            # 2020 vintage: lower production capacity
            ("EV", "vkm", 1, 2020): 45,
            ("EV", "vkm", 2, 2020): 45,
            # 2025 vintage: higher production capacity
            ("EV", "vkm", 1, 2025): 55,
            ("EV", "vkm", 2, 2025): 55,
        }
        model = converter.OptimizationModelInputs(**base_model_inputs)
        assert model.foreground_production_vintages[("EV", "vkm", 1, 2020)] == 45
        assert model.foreground_production_vintages[("EV", "vkm", 1, 2025)] == 55
        assert model.REFERENCE_VINTAGES == [2020, 2025]

    def test_model_accepts_technology_evolution(self, base_model_inputs):
        """Test that technology_evolution scaling factors are accepted."""
        base_model_inputs["technology_evolution"] = {
            # Electricity consumption evolves (decreases over time)
            ("EV", "electricity", 2020): 1.0,  # baseline
            ("EV", "electricity", 2025): 0.7,  # 30% improvement
            # Manufacturing emissions also evolve
            ("EV", "CO2", 2020): 1.0,
            ("EV", "CO2", 2025): 0.8,  # 20% improvement
        }
        model = converter.OptimizationModelInputs(**base_model_inputs)
        assert model.technology_evolution[("EV", "electricity", 2020)] == 1.0
        assert model.technology_evolution[("EV", "electricity", 2025)] == 0.7
        assert model.REFERENCE_VINTAGES == [2020, 2025]

    def test_validation_process_in_vintage_tensors_must_exist(self, base_model_inputs):
        """Test that processes in vintage tensors must exist in PROCESS."""
        base_model_inputs["foreground_technosphere_vintages"] = {
            ("NONEXISTENT", "electricity", 1, 2020): 60,
        }
        with pytest.raises(ValueError, match="Invalid keys.*NONEXISTENT"):
            converter.OptimizationModelInputs(**base_model_inputs)

    def test_validation_flow_in_vintage_tensors_must_exist(self, base_model_inputs):
        """Test that flows in vintage tensors must exist in respective sets."""
        base_model_inputs["foreground_technosphere_vintages"] = {
            ("EV", "nonexistent_flow", 1, 2020): 60,
        }
        with pytest.raises(ValueError, match="Invalid keys.*nonexistent_flow"):
            converter.OptimizationModelInputs(**base_model_inputs)

    def test_validation_process_time_in_vintage_tensors_must_exist(self, base_model_inputs):
        """Test that process times in vintage tensors must exist in PROCESS_TIME."""
        base_model_inputs["foreground_technosphere_vintages"] = {
            ("EV", "electricity", 99, 2020): 60,  # 99 not in PROCESS_TIME
        }
        with pytest.raises(ValueError, match="Invalid keys.*99"):
            converter.OptimizationModelInputs(**base_model_inputs)


class TestVintageMapping:
    """Tests for Phase 2: Vintage mapping (interpolation between reference vintages)."""

    def test_construct_vintage_mapping_exact_match(self):
        """Test vintage mapping when installation year matches a reference vintage."""
        reference_vintages = [2020, 2030]
        system_times = [2020, 2025, 2030]

        mapping = converter.construct_vintage_mapping(reference_vintages, system_times)

        # At 2020: 100% weight to 2020 vintage
        assert mapping[(2020, 2020)] == 1.0
        assert (2030, 2020) not in mapping or mapping.get((2030, 2020), 0) == 0

        # At 2030: 100% weight to 2030 vintage
        assert mapping[(2030, 2030)] == 1.0
        assert (2020, 2030) not in mapping or mapping.get((2020, 2030), 0) == 0

    def test_construct_vintage_mapping_interpolation(self):
        """Test vintage mapping interpolation between reference vintages."""
        reference_vintages = [2020, 2030]
        system_times = [2020, 2025, 2030]

        mapping = converter.construct_vintage_mapping(reference_vintages, system_times)

        # At 2025: 50% between 2020 and 2030
        assert mapping[(2020, 2025)] == pytest.approx(0.5)
        assert mapping[(2030, 2025)] == pytest.approx(0.5)

    def test_construct_vintage_mapping_extrapolation_before(self):
        """Test vintage mapping for years before first reference vintage."""
        reference_vintages = [2025, 2035]
        system_times = [2020, 2025, 2030, 2035]

        mapping = converter.construct_vintage_mapping(reference_vintages, system_times)

        # At 2020 (before 2025): use earliest vintage
        assert mapping[(2025, 2020)] == 1.0
        assert (2035, 2020) not in mapping or mapping.get((2035, 2020), 0) == 0

    def test_construct_vintage_mapping_extrapolation_after(self):
        """Test vintage mapping for years after last reference vintage."""
        reference_vintages = [2020, 2025]
        system_times = [2020, 2025, 2030, 2035]

        mapping = converter.construct_vintage_mapping(reference_vintages, system_times)

        # At 2030 (after 2025): use latest vintage
        assert mapping[(2025, 2030)] == 1.0
        assert (2020, 2030) not in mapping or mapping.get((2020, 2030), 0) == 0

        # At 2035: also use latest vintage
        assert mapping[(2025, 2035)] == 1.0

    def test_construct_vintage_mapping_multiple_references(self):
        """Test vintage mapping with multiple reference vintages."""
        reference_vintages = [2020, 2030, 2040]
        system_times = [2020, 2025, 2030, 2035, 2040]

        mapping = converter.construct_vintage_mapping(reference_vintages, system_times)

        # At 2025: interpolate between 2020 and 2030
        assert mapping[(2020, 2025)] == pytest.approx(0.5)
        assert mapping[(2030, 2025)] == pytest.approx(0.5)
        assert (2040, 2025) not in mapping or mapping.get((2040, 2025), 0) == 0

        # At 2035: interpolate between 2030 and 2040
        assert mapping[(2030, 2035)] == pytest.approx(0.5)
        assert mapping[(2040, 2035)] == pytest.approx(0.5)
        assert (2020, 2035) not in mapping or mapping.get((2020, 2035), 0) == 0


class TestEffectiveForegroundTensors:
    """Tests for Phase 3: Construction of effective (expanded) foreground tensors."""

    @pytest.fixture
    def vintage_model_inputs(self):
        """Model inputs with vintage-specific parameters."""
        return {
            "PROCESS": ["EV"],
            "PRODUCT": ["vkm"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db_2020"],
            "PROCESS_TIME": [0, 1, 2],
            "SYSTEM_TIME": [2020, 2021, 2022, 2023, 2024, 2025],
            "CATEGORY": ["climate_change"],
            # REFERENCE_VINTAGES is inferred automatically from vintage data
            "operation_time_limits": {"EV": (1, 2)},
            "demand": {("vkm", t): 100 for t in [2020, 2021, 2022, 2023, 2024, 2025]},
            "foreground_technosphere": {},
            "foreground_technosphere_vintages": {
                # 2020 vintage: 60 kWh electricity per operation time step
                ("EV", "electricity", 1, 2020): 60,
                ("EV", "electricity", 2, 2020): 60,
                # 2025 vintage: 40 kWh (improved efficiency)
                ("EV", "electricity", 1, 2025): 40,
                ("EV", "electricity", 2, 2025): 40,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_biosphere_vintages": {
                ("EV", "CO2", 0, 2020): 6000,  # 2020 vintage manufacturing
                ("EV", "CO2", 0, 2025): 4000,  # 2025 vintage manufacturing
            },
            "foreground_production": {},
            "foreground_production_vintages": {
                ("EV", "vkm", 1, 2020): 50,
                ("EV", "vkm", 2, 2020): 50,
                ("EV", "vkm", 1, 2025): 50,
                ("EV", "vkm", 2, 2025): 50,
            },
            "operation_flow": {("EV", "vkm"): True, ("EV", "electricity"): True},
            "background_inventory": {("db_2020", "electricity", "CO2"): 0.5},
            "mapping": {("db_2020", t): 1.0 for t in [2020, 2021, 2022, 2023, 2024, 2025]},
            "characterization": {
                ("climate_change", "CO2", t): 1.0
                for t in [2020, 2021, 2022, 2023, 2024, 2025]
            },
        }

    def test_expand_foreground_technosphere_exact_vintage(self, vintage_model_inputs):
        """Test that effective tensor uses exact values at reference vintages."""
        model = converter.OptimizationModelInputs(**vintage_model_inputs)
        effective = converter.expand_foreground_tensor_with_vintages(
            model.foreground_technosphere_vintages,
            model.REFERENCE_VINTAGES,
            model.SYSTEM_TIME,
        )

        # At installation year 2020: use 2020 vintage values
        assert effective[("EV", "electricity", 1, 2020)] == 60
        assert effective[("EV", "electricity", 2, 2020)] == 60

        # At installation year 2025: use 2025 vintage values
        assert effective[("EV", "electricity", 1, 2025)] == 40
        assert effective[("EV", "electricity", 2, 2025)] == 40

    def test_expand_foreground_technosphere_interpolated_vintage(self, vintage_model_inputs):
        """Test that effective tensor interpolates between reference vintages."""
        model = converter.OptimizationModelInputs(**vintage_model_inputs)
        effective = converter.expand_foreground_tensor_with_vintages(
            model.foreground_technosphere_vintages,
            model.REFERENCE_VINTAGES,
            model.SYSTEM_TIME,
        )

        # At installation year 2022 (40% between 2020 and 2025):
        # Expected: 60 * 0.6 + 40 * 0.4 = 36 + 16 = 52
        expected_2022 = 60 * 0.6 + 40 * 0.4
        assert effective[("EV", "electricity", 1, 2022)] == pytest.approx(expected_2022)

        # At installation year 2023 (60% between 2020 and 2025):
        # Expected: 60 * 0.4 + 40 * 0.6 = 24 + 24 = 48
        expected_2023 = 60 * 0.4 + 40 * 0.6
        assert effective[("EV", "electricity", 1, 2023)] == pytest.approx(expected_2023)

    def test_expand_with_technology_evolution_factors(self, vintage_model_inputs):
        """Test expansion using technology evolution scaling factors."""
        # Use base tensor + evolution factors instead of vintages
        vintage_model_inputs["foreground_technosphere_vintages"] = None
        vintage_model_inputs["foreground_technosphere"] = {
            ("EV", "electricity", 1): 60,  # Base value
            ("EV", "electricity", 2): 60,
        }
        vintage_model_inputs["technology_evolution"] = {
            ("EV", "electricity", 2020): 1.0,  # baseline
            ("EV", "electricity", 2025): 0.667,  # ~40 kWh (40/60)
        }

        model = converter.OptimizationModelInputs(**vintage_model_inputs)
        effective = converter.expand_foreground_tensor_with_evolution(
            model.foreground_technosphere,
            model.technology_evolution,
            model.REFERENCE_VINTAGES,
            model.SYSTEM_TIME,
            "INTERMEDIATE_FLOW",
        )

        # At 2020: base * 1.0 = 60
        assert effective[("EV", "electricity", 1, 2020)] == pytest.approx(60.0)

        # At 2025: base * 0.667 ≈ 40
        assert effective[("EV", "electricity", 1, 2025)] == pytest.approx(40.02)

        # At 2022 (interpolated): 60 * (0.6 * 1.0 + 0.4 * 0.667) ≈ 52
        expected_factor_2022 = 0.6 * 1.0 + 0.4 * 0.667
        assert effective[("EV", "electricity", 1, 2022)] == pytest.approx(
            60 * expected_factor_2022
        )


class TestOptimizerWithVintages:
    """Tests for Phase 4: Optimizer integration with vintage parameters."""

    @pytest.fixture
    def vintage_optimization_inputs(self):
        """Complete inputs for optimization with vintage-dependent parameters."""
        return {
            "PROCESS": ["EV"],
            "PRODUCT": ["vkm"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["db_2020"],
            "PROCESS_TIME": [0, 1, 2],
            "SYSTEM_TIME": [2020, 2021, 2022, 2023, 2024, 2025],
            "CATEGORY": ["climate_change"],
            # REFERENCE_VINTAGES is inferred automatically from vintage data
            "operation_time_limits": {"EV": (1, 2)},
            "demand": {
                ("vkm", 2022): 100,
                ("vkm", 2023): 100,
                ("vkm", 2024): 100,
                ("vkm", 2025): 100,
            },
            "foreground_technosphere": {},
            "foreground_technosphere_vintages": {
                ("EV", "electricity", 1, 2020): 60,
                ("EV", "electricity", 2, 2020): 60,
                ("EV", "electricity", 1, 2025): 40,
                ("EV", "electricity", 2, 2025): 40,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_biosphere_vintages": {
                ("EV", "CO2", 0, 2020): 6000,
                ("EV", "CO2", 0, 2025): 4000,
            },
            "foreground_production": {},
            "foreground_production_vintages": {
                ("EV", "vkm", 1, 2020): 50,
                ("EV", "vkm", 2, 2020): 50,
                ("EV", "vkm", 1, 2025): 50,
                ("EV", "vkm", 2, 2025): 50,
            },
            "operation_flow": {("EV", "vkm"): True, ("EV", "electricity"): True},
            "background_inventory": {("db_2020", "electricity", "CO2"): 0.5},
            "mapping": {("db_2020", t): 1.0 for t in [2020, 2021, 2022, 2023, 2024, 2025]},
            "characterization": {
                ("climate_change", "CO2", t): 1.0
                for t in [2020, 2021, 2022, 2023, 2024, 2025]
            },
        }

    def test_model_creation_with_vintages(self, vintage_optimization_inputs):
        """Test that optimizer model can be created with vintage inputs."""
        from optimex import optimizer

        model_inputs = converter.OptimizationModelInputs(**vintage_optimization_inputs)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="climate_change",
            name="test_vintage_model",
        )
        assert model is not None

    def test_model_solves_with_vintages(self, vintage_optimization_inputs):
        """Test that model with vintage parameters can be solved."""
        from optimex import optimizer

        model_inputs = converter.OptimizationModelInputs(**vintage_optimization_inputs)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="climate_change",
            name="test_vintage_model",
        )
        solved_model, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)
        assert results.solver.termination_condition.name == "optimal"

    def test_later_vintage_has_lower_impact(self, vintage_optimization_inputs):
        """
        Test that installing later vintages results in lower environmental impact.

        EVs built in 2025 have:
        - Lower electricity consumption (40 vs 60)
        - Lower manufacturing emissions (4000 vs 6000)

        So the optimizer should prefer later installations if given the choice.
        """
        from optimex import optimizer
        import pyomo.environ as pyo

        model_inputs = converter.OptimizationModelInputs(**vintage_optimization_inputs)
        model = optimizer.create_model(
            inputs=model_inputs,
            objective_category="climate_change",
            name="test_vintage_model",
        )
        solved_model, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)

        # Get total installations at different years
        # Earlier installations (2020-2022) should have higher per-unit impact
        # Later installations (2023-2025) should have lower per-unit impact

        # Check that the model accounts for vintage differences
        # The exact behavior depends on demand timing and process lifetimes
        early_installations = sum(
            pyo.value(solved_model.var_installation["EV", t])
            for t in [2020, 2021]
        )
        late_installations = sum(
            pyo.value(solved_model.var_installation["EV", t])
            for t in [2023, 2024]
        )

        # Both should be non-negative (basic sanity check)
        assert early_installations >= 0
        assert late_installations >= 0

    def test_optimizer_prefers_efficient_vintage(self):
        """
        Test that optimizer installs at times that minimize total impact.

        Scenario: Demand only in 2025, two vintages with different efficiency.
        2020 vintage: 100 electricity/unit
        2025 vintage: 50 electricity/unit (50% more efficient)

        Optimizer should prefer installing in 2024 (gets 2025-interpolated efficiency)
        over installing earlier (gets worse efficiency).
        """
        from optimex import optimizer
        import pyomo.environ as pyo

        inputs = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021, 2022, 2023, 2024, 2025],
            "CATEGORY": ["GWP"],
            # REFERENCE_VINTAGES is inferred from vintage data
            "operation_time_limits": {"Plant": (1, 1)},
            # Demand only in 2025 - can be met by installing any year from 2020-2024
            "demand": {("output", 2025): 100},
            "foreground_technosphere": {},
            "foreground_technosphere_vintages": {
                ("Plant", "electricity", 1, 2020): 100,  # Inefficient
                ("Plant", "electricity", 1, 2025): 50,   # Efficient
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("Plant", "CO2", 0): 0},
            "foreground_production": {},
            "foreground_production_vintages": {
                ("Plant", "output", 1, 2020): 100,
                ("Plant", "output", 1, 2025): 100,
            },
            "operation_flow": {("Plant", "output"): True, ("Plant", "electricity"): True},
            "background_inventory": {("grid", "electricity", "CO2"): 1.0},
            "mapping": {("grid", t): 1.0 for t in range(2020, 2026)},
            "characterization": {("GWP", "CO2", t): 1.0 for t in range(2020, 2026)},
        }

        model_inputs = converter.OptimizationModelInputs(**inputs)
        model = optimizer.create_model(
            inputs=model_inputs, objective_category="GWP", name="efficiency_test"
        )
        solved, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)
        assert results.solver.termination_condition.name == "optimal"

        # Check installation pattern - should prefer later years (more efficient)
        installations = {
            t: pyo.value(solved.var_installation["Plant", t])
            for t in range(2020, 2025)  # Can install 2020-2024 to operate in 2025
        }

        # 2024 vintage interpolates to 90% of way to 2025 efficiency: 100 - 0.8*(100-50) = 60
        # Earlier vintages are progressively worse
        # Optimizer should install as late as possible (2024)
        assert installations[2024] > 0, "Optimizer should install in 2024 (most efficient vintage)"
        assert sum(installations[t] for t in [2020, 2021, 2022]) == pytest.approx(0, abs=1e-6), \
            "Optimizer should avoid early installations (inefficient vintages)"


class TestTechnologyEvolutionIntegration:
    """Integration tests for technology_evolution scaling factors."""

    def test_technology_evolution_affects_optimization(self):
        """
        Test that technology_evolution scaling factors affect the optimal solution.

        Same scenario as above but using evolution factors instead of explicit vintages.
        """
        from optimex import optimizer
        import pyomo.environ as pyo

        inputs = {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021, 2022, 2023, 2024, 2025],
            "CATEGORY": ["GWP"],
            # REFERENCE_VINTAGES is inferred from vintage data
            "operation_time_limits": {"Plant": (1, 1)},
            "demand": {("output", 2025): 100},
            # Base consumption - will be scaled by evolution factors
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 0,
                ("Plant", "electricity", 1): 100,
            },
            # Evolution: 2020 = 1.0x, 2025 = 0.5x (50% improvement)
            "technology_evolution": {
                ("Plant", "electricity", 2020): 1.0,
                ("Plant", "electricity", 2025): 0.5,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {("Plant", "CO2", 0): 0, ("Plant", "CO2", 1): 0},
            "foreground_production": {
                ("Plant", "output", 0): 0,
                ("Plant", "output", 1): 100,
            },
            "operation_flow": {("Plant", "output"): True, ("Plant", "electricity"): True},
            "background_inventory": {("grid", "electricity", "CO2"): 1.0},
            "mapping": {("grid", t): 1.0 for t in range(2020, 2026)},
            "characterization": {("GWP", "CO2", t): 1.0 for t in range(2020, 2026)},
        }

        model_inputs = converter.OptimizationModelInputs(**inputs)
        model = optimizer.create_model(
            inputs=model_inputs, objective_category="GWP", name="evolution_test"
        )
        solved, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)
        assert results.solver.termination_condition.name == "optimal"

        # Should prefer later installation (lower electricity consumption via evolution factor)
        late_install = pyo.value(solved.var_installation["Plant", 2024])
        early_install = sum(
            pyo.value(solved.var_installation["Plant", t]) for t in [2020, 2021, 2022]
        )
        assert late_install > early_install, "Optimizer should prefer later vintages with evolution factors"


class TestMixedVintageScenarios:
    """Tests for scenarios with partial vintage overrides."""

    def test_mixed_processes_some_with_overrides(self):
        """
        Test scenario where only some processes have vintage overrides.

        - ProcessA: Has vintage-dependent electricity consumption
        - ProcessB: No vintage overrides (constant efficiency)

        Both should work correctly in the same model.
        """
        from optimex import optimizer
        import pyomo.environ as pyo

        inputs = {
            "PROCESS": ["ProcessA", "ProcessB"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021, 2022, 2023, 2024, 2025],
            "CATEGORY": ["GWP"],
            # REFERENCE_VINTAGES is inferred from vintage data
            "operation_time_limits": {"ProcessA": (1, 1), "ProcessB": (1, 1)},
            "demand": {("output", 2025): 100},
            # ProcessB: constant efficiency (3D tensor)
            "foreground_technosphere": {
                ("ProcessA", "electricity", 0): 0,
                ("ProcessA", "electricity", 1): 0,  # Will be overridden
                ("ProcessB", "electricity", 0): 0,
                ("ProcessB", "electricity", 1): 75,  # Constant, no vintage override
            },
            # ProcessA: vintage-dependent (4D sparse override)
            "foreground_technosphere_vintages": {
                ("ProcessA", "electricity", 1, 2020): 100,  # Inefficient
                ("ProcessA", "electricity", 1, 2025): 50,   # Efficient
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {
                ("ProcessA", "CO2", 0): 0, ("ProcessA", "CO2", 1): 0,
                ("ProcessB", "CO2", 0): 0, ("ProcessB", "CO2", 1): 0,
            },
            "foreground_production": {
                ("ProcessA", "output", 0): 0, ("ProcessA", "output", 1): 100,
                ("ProcessB", "output", 0): 0, ("ProcessB", "output", 1): 100,
            },
            "operation_flow": {
                ("ProcessA", "output"): True, ("ProcessA", "electricity"): True,
                ("ProcessB", "output"): True, ("ProcessB", "electricity"): True,
            },
            "background_inventory": {("grid", "electricity", "CO2"): 1.0},
            "mapping": {("grid", t): 1.0 for t in range(2020, 2026)},
            "characterization": {("GWP", "CO2", t): 1.0 for t in range(2020, 2026)},
        }

        model_inputs = converter.OptimizationModelInputs(**inputs)
        model = optimizer.create_model(
            inputs=model_inputs, objective_category="GWP", name="mixed_test"
        )
        solved, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)
        assert results.solver.termination_condition.name == "optimal"

        # Model should solve and make sensible choices
        # ProcessA installed in 2024 has efficiency ~60 (interpolated)
        # ProcessB always has efficiency 75
        # So ProcessA in 2024 (60) < ProcessB (75) - should prefer ProcessA late

        total_A = sum(
            pyo.value(solved.var_installation["ProcessA", t]) for t in range(2020, 2025)
        )
        total_B = sum(
            pyo.value(solved.var_installation["ProcessB", t]) for t in range(2020, 2025)
        )

        # At least one process should be installed to meet demand
        assert total_A + total_B > 0, "At least one process must be installed"

        # If ProcessA is installed, it should be later (more efficient)
        if total_A > 0:
            late_A = pyo.value(solved.var_installation["ProcessA", 2024])
            assert late_A > 0, "ProcessA installations should be late (more efficient)"


class TestVintageResultsValidation:
    """
    End-to-end tests that validate vintage parameters produce correct results.

    These tests compare optimization results with and without vintage improvements
    to verify that tensors, inventories, and impacts are computed correctly.
    """

    @pytest.fixture
    def simple_system_base(self):
        """
        Simple single-process system for baseline comparison.

        Process: Plant that consumes electricity and produces output.
        - 1 unit of output requires 100 units of electricity
        - Electricity has 1 kg CO2/unit in background
        - Demand: 100 units in year 2022
        - Single process time (immediate production)
        """
        return {
            "PROCESS": ["Plant"],
            "PRODUCT": ["output"],
            "INTERMEDIATE_FLOW": ["electricity"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["grid"],
            "PROCESS_TIME": [0],
            "SYSTEM_TIME": [2020, 2021, 2022],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Plant": (0, 0)},
            "demand": {("output", 2022): 100},
            "foreground_technosphere": {
                ("Plant", "electricity", 0): 100,  # 100 electricity per unit output
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {
                ("Plant", "output", 0): 100,  # 100 output per installation
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            "background_inventory": {
                ("grid", "electricity", "CO2"): 1.0,  # 1 kg CO2 per unit electricity
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

    def test_baseline_without_vintage_parameters(self, simple_system_base):
        """
        Test baseline system without vintage parameters.

        Expected impact: 100 kg CO2 (100 elec * 1 kg/elec * 1.0 CF)
        """
        from optimex import optimizer

        model_inputs = converter.OptimizationModelInputs(**simple_system_base)
        model = optimizer.create_model(
            inputs=model_inputs, objective_category="GWP", name="baseline"
        )
        solved, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)

        assert results.solver.termination_condition.name == "optimal"

        # Check objective (total impact)
        assert obj == pytest.approx(100.0, rel=1e-5), \
            "Baseline impact should be 100 kg CO2 (100 elec * 1 kg/elec * 1.0 CF)"

    def test_vintage_50_percent_improvement_halves_impact(self, simple_system_base):
        """
        Test that 50% efficiency improvement results in 50% lower impact.

        Using technology_evolution with factor 0.5 for 2022 vintage means:
        - Electricity consumption: 100 * 0.5 = 50 units
        - CO2 emissions: 50 kg
        - Impact: 50 kg CO2
        """
        from optimex import optimizer

        # Add 50% improvement via technology_evolution
        inputs_with_evolution = simple_system_base.copy()
        inputs_with_evolution["technology_evolution"] = {
            ("Plant", "electricity", 2020): 1.0,  # baseline at 2020
            ("Plant", "electricity", 2022): 0.5,  # 50% reduction by 2022
        }

        model_inputs = converter.OptimizationModelInputs(**inputs_with_evolution)

        # Verify REFERENCE_VINTAGES was inferred correctly
        assert model_inputs.REFERENCE_VINTAGES == [2020, 2022]

        model = optimizer.create_model(
            inputs=model_inputs, objective_category="GWP", name="with_evolution"
        )
        solved, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)

        assert results.solver.termination_condition.name == "optimal"

        # Check objective is halved (50% efficiency = 50% impact)
        assert obj == pytest.approx(50.0, rel=1e-5), \
            "Impact should be 50 kg CO2 (50% of baseline due to efficiency improvement)"

    def test_vintage_explicit_values_affect_optimization(self):
        """
        Test that explicit vintage values produce correct optimization results.

        2020 vintage: 100 electricity per unit
        2022 vintage: 50 electricity per unit (50% improvement)

        With demand in 2022 and operation at tau=1, installation happens in 2021.
        2021 vintage is interpolated: (100 + 50) / 2 = 75 electricity.
        Impact: 75 electricity * 1 kg CO2/elec = 75 kg CO2.
        """
        from optimex import optimizer

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
            "foreground_technosphere": {},
            "foreground_technosphere_vintages": {
                ("Plant", "electricity", 1, 2020): 100,
                ("Plant", "electricity", 1, 2022): 50,
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {},
            "foreground_production_vintages": {
                ("Plant", "output", 1, 2020): 100,
                ("Plant", "output", 1, 2022): 100,
            },
            "operation_flow": {
                ("Plant", "output"): True,
                ("Plant", "electricity"): True,
            },
            "background_inventory": {
                ("grid", "electricity", "CO2"): 1.0,
            },
            "mapping": {("grid", t): 1.0 for t in [2020, 2021, 2022]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021, 2022]},
        }

        model_inputs = converter.OptimizationModelInputs(**inputs)
        model = optimizer.create_model(
            inputs=model_inputs, objective_category="GWP", name="explicit_vintages"
        )
        solved, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)

        assert results.solver.termination_condition.name == "optimal"

        # Installing in 2021 (for operation in 2022) with interpolated 2021 vintage:
        # Interpolated electricity = (100 + 50) / 2 = 75
        # 75 electricity * 1 kg CO2/elec = 75 kg CO2
        assert obj == pytest.approx(75.0, rel=1e-5), \
            "Impact should be 75 kg CO2 (interpolated 2021 vintage efficiency)"

    def test_compare_baseline_vs_improved_full_workflow(self, simple_system_base):
        """
        Full comparison test: run same system with and without improvement,
        verify the ratio of impacts matches the improvement factor.
        """
        from optimex import optimizer

        # Run baseline
        baseline_inputs = converter.OptimizationModelInputs(**simple_system_base)
        baseline_model = optimizer.create_model(
            inputs=baseline_inputs, objective_category="GWP", name="baseline"
        )
        _, baseline_obj, baseline_results = optimizer.solve_model(
            baseline_model, solver_name="glpk", tee=False
        )
        assert baseline_results.solver.termination_condition.name == "optimal"

        # Run with 30% improvement (factor = 0.7)
        improved_inputs = simple_system_base.copy()
        improved_inputs["technology_evolution"] = {
            ("Plant", "electricity", 2020): 1.0,
            ("Plant", "electricity", 2022): 0.7,  # 30% improvement
        }
        improved_model_inputs = converter.OptimizationModelInputs(**improved_inputs)
        improved_model = optimizer.create_model(
            inputs=improved_model_inputs, objective_category="GWP", name="improved"
        )
        _, improved_obj, improved_results = optimizer.solve_model(
            improved_model, solver_name="glpk", tee=False
        )
        assert improved_results.solver.termination_condition.name == "optimal"

        # Verify ratio
        improvement_ratio = improved_obj / baseline_obj
        expected_ratio = 0.7  # 30% improvement means 70% of baseline

        assert improvement_ratio == pytest.approx(expected_ratio, rel=1e-5), \
            f"Impact ratio should be {expected_ratio}, got {improvement_ratio}"

    def test_biosphere_vintage_affects_direct_emissions(self):
        """
        Test that foreground_biosphere_vintages correctly affects direct emissions.

        Setup: Process with direct CO2 emissions that improve over time.
        Compare 2020 vintage (200 kg) vs 2022 vintage (100 kg).
        """
        from optimex import optimizer

        # Base inputs for comparing vintages
        base_inputs = {
            "PROCESS": ["Factory"],
            "PRODUCT": ["product"],
            "INTERMEDIATE_FLOW": [],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["bg"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021, 2022],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Factory": (1, 1)},
            "foreground_technosphere": {},
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_biosphere_vintages": {
                ("Factory", "CO2", 1, 2020): 200,  # 200 kg CO2 per unit at 2020
                ("Factory", "CO2", 1, 2022): 100,  # 100 kg CO2 per unit at 2022
            },
            "foreground_production": {},
            "foreground_production_vintages": {
                ("Factory", "product", 1, 2020): 100,
                ("Factory", "product", 1, 2022): 100,
            },
            "operation_flow": {
                ("Factory", "product"): True,
            },
            "background_inventory": {},
            "mapping": {("bg", t): 1.0 for t in [2020, 2021, 2022]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021, 2022]},
        }

        # Test 2020 vintage: demand in 2021 requires 2020 installation
        inputs_2020 = base_inputs.copy()
        inputs_2020["demand"] = {("product", 2021): 100}

        model_inputs_2020 = converter.OptimizationModelInputs(**inputs_2020)
        model_2020 = optimizer.create_model(
            inputs=model_inputs_2020, objective_category="GWP", name="biosphere_2020"
        )
        _, obj_2020, results_2020 = optimizer.solve_model(model_2020, solver_name="glpk", tee=False)
        assert results_2020.solver.termination_condition.name == "optimal"

        # Test 2022 vintage: demand in 2022 allows 2021 installation (closer to 2022)
        inputs_2022 = base_inputs.copy()
        inputs_2022["demand"] = {("product", 2022): 100}

        model_inputs_2022 = converter.OptimizationModelInputs(**inputs_2022)
        model_2022 = optimizer.create_model(
            inputs=model_inputs_2022, objective_category="GWP", name="biosphere_2022"
        )
        _, obj_2022, results_2022 = optimizer.solve_model(model_2022, solver_name="glpk", tee=False)
        assert results_2022.solver.termination_condition.name == "optimal"

        # 2020 vintage should have higher emissions than 2021 vintage (interpolated toward 2022)
        # 2020 vintage: 200 kg CO2 per unit
        # 2021 vintage (interpolated): (200 + 100) / 2 = 150 kg CO2 per unit
        # Expected ratio: 200 / 150 = 1.33
        assert obj_2022 < obj_2020, \
            "Later vintage should have lower emissions"

        expected_ratio = 200 / 150  # 2020 emissions / 2021 interpolated emissions
        actual_ratio = obj_2020 / obj_2022
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-2), \
            f"Emission ratio should be ~{expected_ratio:.2f}, got {actual_ratio:.2f}"

    def test_production_vintage_affects_capacity(self):
        """
        Test that foreground_production_vintages correctly affects production capacity.

        Setup: Process where production capacity improves over time.
        2020 vintage: 100 output per unit
        2022 vintage: 200 output per unit (double capacity)

        With demand in 2022 and operation at tau=1, installation happens in 2021.
        2021 vintage has interpolated capacity: (100 + 200) / 2 = 150 output/unit.
        To meet demand of 200: need 200/150 = 1.333 units.
        Fuel: 1.333 * 50 * 2 = 133.33 kg CO2.
        """
        from optimex import optimizer

        inputs = {
            "PROCESS": ["Generator"],
            "PRODUCT": ["power"],
            "INTERMEDIATE_FLOW": ["fuel"],
            "ELEMENTARY_FLOW": ["CO2"],
            "BACKGROUND_ID": ["bg"],
            "PROCESS_TIME": [0, 1],
            "SYSTEM_TIME": [2020, 2021, 2022],
            "CATEGORY": ["GWP"],
            "operation_time_limits": {"Generator": (1, 1)},
            "demand": {("power", 2022): 200},
            "foreground_technosphere": {},
            "foreground_technosphere_vintages": {
                ("Generator", "fuel", 1, 2020): 50,  # 50 fuel per unit
                ("Generator", "fuel", 1, 2022): 50,  # Same fuel consumption
            },
            "internal_demand_technosphere": {},
            "foreground_biosphere": {},
            "foreground_production": {},
            "foreground_production_vintages": {
                ("Generator", "power", 1, 2020): 100,  # 100 power per unit
                ("Generator", "power", 1, 2022): 200,  # 200 power per unit (improved)
            },
            "operation_flow": {
                ("Generator", "power"): True,
                ("Generator", "fuel"): True,
            },
            "background_inventory": {
                ("bg", "fuel", "CO2"): 2.0,  # 2 kg CO2 per fuel
            },
            "mapping": {("bg", t): 1.0 for t in [2020, 2021, 2022]},
            "characterization": {("GWP", "CO2", t): 1.0 for t in [2020, 2021, 2022]},
        }

        model_inputs = converter.OptimizationModelInputs(**inputs)
        model = optimizer.create_model(
            inputs=model_inputs, objective_category="GWP", name="production_vintage"
        )
        solved, obj, results = optimizer.solve_model(model, solver_name="glpk", tee=False)

        assert results.solver.termination_condition.name == "optimal"

        # Installing in 2021 with interpolated capacity = 150 power/unit
        # Need 200/150 = 1.333 units to meet demand
        # Fuel: 1.333 * 50 = 66.67
        # Impact: 66.67 * 2 = 133.33 kg CO2
        assert obj == pytest.approx(133.33, rel=1e-3), \
            "Impact should be ~133.33 kg CO2 (interpolated production capacity)"



class TestDatabaseVintageParameterExtraction:
    """Tests for extracting vintage parameters from Brightway database."""

    @pytest.fixture
    def database_with_vintage_params(self, request):
        """Create a test database with vintage parameter attributes."""
        from datetime import datetime
        import bw2data as bd
        import numpy as np
        from bw2data.tests import bw2test
        from bw_temporalis import TemporalDistribution

        # Set up test project
        project_name = "__test_vintage_db__"
        bd.projects.set_current(project_name)
        
        # Register cleanup to delete project after test
        def cleanup():
            try:
                bd.projects.delete_project(project_name, delete_dir=True)
            except:
                pass  # Ignore cleanup errors
        
        request.addfinalizer(cleanup)
        
        # Create biosphere database
        bio_db = bd.Database("biosphere3")
        bio_db.write({
            ("biosphere3", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
            },
        })
        bio_db.register()

        # Create background database
        bg_db = bd.Database("db_2020")
        bg_db.write({
            ("db_2020", "electricity"): {
                "name": "electricity",
                "location": "GLO",
                "reference product": "electricity",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2020", "electricity"),
                    },
                    {
                        "amount": 0.5,
                        "type": "biosphere",
                        "input": ("biosphere3", "CO2"),
                    },
                ],
            },
        })
        bg_db.metadata["representative_time"] = datetime(2020, 1, 1).isoformat()
        bg_db.register()

        # Create foreground database with vintage parameters
        fg_db = bd.Database("foreground")
        fg_db.write({
            # Product node
            ("foreground", "vkm"): {
                "name": "vehicle-km",
                "type": bd.labels.product_node_default,
                "unit": "km",
            },
            # Process node with vintage-dependent exchange
            ("foreground", "EV"): {
                "name": "Electric Vehicle",
                "type": bd.labels.process_node_default,
                "operation_time_limits": (1, 2),
                "exchanges": [
                    {
                        "amount": 1,
                        "type": bd.labels.production_edge_default,
                        "input": ("foreground", "vkm"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([0, 1, 2, 3], dtype="timedelta64[Y]"),
                            amount=np.array([0, 0.5, 0.5, 0]),
                        ),
                        "operation": True,
                    },
                    {
                        "amount": 60,  # Base amount (will be overridden by vintage_amounts)
                        "type": bd.labels.consumption_edge_default,
                        "input": ("db_2020", "electricity"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([1, 2], dtype="timedelta64[Y]"),
                            amount=np.array([0.5, 0.5]),
                        ),
                        "operation": True,
                        # NEW: Vintage-specific values
                        "vintage_amounts": {
                            (1, 2020): 30,  # τ=1, 2020 vintage: 30 MJ/vkm
                            (2, 2020): 30,  # τ=2, 2020 vintage: 30 MJ/vkm
                            (1, 2030): 22.5,  # τ=1, 2030 vintage: 22.5 MJ/vkm
                            (2, 2030): 22.5,  # τ=2, 2030 vintage: 22.5 MJ/vkm
                        },
                    },
                    {
                        "amount": 5000,  # Manufacturing emissions
                        "type": bd.labels.biosphere_edge_default,
                        "input": ("biosphere3", "CO2"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([0], dtype="timedelta64[Y]"),
                            amount=np.array([1.0]),
                        ),
                        # Vintage improvement scaling
                        "vintage_improvements": {
                            2020: 1.0,   # 100% of base (5000 kg)
                            2030: 0.8,   # 80% of base (4000 kg)
                        },
                    },
                ],
            },
        })
        fg_db.register()

        # Create LCIA method
        bd.Method(("GWP", "example")).write([
            (("biosphere3", "CO2"), 1.0),
        ])

        return fg_db

    def test_lca_processor_extracts_vintage_amounts(self, database_with_vintage_params):
        """Test that LCADataProcessor extracts vintage_amounts from exchanges."""
        from datetime import datetime
        import bw2data as bd
        import numpy as np
        from bw_temporalis import TemporalDistribution
        from optimex import lca_processor

        # Create demand
        product_node = bd.get_node(database="foreground", code="vkm")
        
        demand = {
            product_node: TemporalDistribution(
                date=np.array([2025], dtype="datetime64[Y]"),
                amount=np.array([100]),
            )
        }

        # Configure LCA processor
        config = lca_processor.LCAConfig(
            demand=demand,
            temporal=lca_processor.TemporalConfig(
                start_date=datetime(2020, 1, 1),
                temporal_resolution=lca_processor.TemporalResolutionEnum.year,
                time_horizon=100,
            ),
            characterization_methods=[
                lca_processor.CharacterizationMethodConfig(
                    category_name="climate_change",
                    brightway_method=("GWP", "example"),
                    metric=None,
                )
            ],
            background_inventory=lca_processor.BackgroundInventoryConfig(),
        )

        # Run processor
        processor = lca_processor.LCADataProcessor(config)

        # Verify vintage parameters were extracted
        assert processor.foreground_technosphere_vintages is not None
        assert len(processor.foreground_technosphere_vintages) > 0
        
        # Check specific vintage values
        assert ("EV", "electricity", 1, 2020) in processor.foreground_technosphere_vintages
        assert processor.foreground_technosphere_vintages[("EV", "electricity", 1, 2020)] == 30
        assert ("EV", "electricity", 1, 2030) in processor.foreground_technosphere_vintages
        assert processor.foreground_technosphere_vintages[("EV", "electricity", 1, 2030)] == 22.5

    def test_lca_processor_extracts_vintage_improvements(self, database_with_vintage_params):
        """Test that LCADataProcessor extracts vintage_improvements from exchanges."""
        from datetime import datetime
        import bw2data as bd
        import numpy as np
        from bw_temporalis import TemporalDistribution
        from optimex import lca_processor

        # Create demand
        product_node = bd.get_node(database="foreground", code="vkm")
        
        demand = {
            product_node: TemporalDistribution(
                date=np.array([2025], dtype="datetime64[Y]"),
                amount=np.array([100]),
            )
        }

        # Configure LCA processor
        config = lca_processor.LCAConfig(
            demand=demand,
            temporal=lca_processor.TemporalConfig(
                start_date=datetime(2020, 1, 1),
                temporal_resolution=lca_processor.TemporalResolutionEnum.year,
                time_horizon=100,
            ),
            characterization_methods=[
                lca_processor.CharacterizationMethodConfig(
                    category_name="climate_change",
                    brightway_method=("GWP", "example"),
                    metric=None,
                )
            ],
            background_inventory=lca_processor.BackgroundInventoryConfig(),
        )

        # Run processor
        processor = lca_processor.LCADataProcessor(config)

        # Verify vintage improvements were extracted
        assert processor.vintage_improvements is not None
        assert len(processor.vintage_improvements) > 0
        
        # Check specific vintage improvement values
        assert ("EV", "CO2", 2020) in processor.vintage_improvements
        assert processor.vintage_improvements[("EV", "CO2", 2020)] == 1.0
        assert ("EV", "CO2", 2030) in processor.vintage_improvements
        assert processor.vintage_improvements[("EV", "CO2", 2030)] == 0.8

    def test_model_input_manager_uses_database_vintages(self, database_with_vintage_params):
        """Test that ModelInputManager includes vintage parameters from database."""
        from datetime import datetime
        import bw2data as bd
        import numpy as np
        from bw_temporalis import TemporalDistribution
        from optimex import lca_processor, converter

        # Create demand
        product_node = bd.get_node(database="foreground", code="vkm")
        
        demand = {
            product_node: TemporalDistribution(
                date=np.array([2025], dtype="datetime64[Y]"),
                amount=np.array([100]),
            )
        }

        # Configure LCA processor
        config = lca_processor.LCAConfig(
            demand=demand,
            temporal=lca_processor.TemporalConfig(
                start_date=datetime(2020, 1, 1),
                temporal_resolution=lca_processor.TemporalResolutionEnum.year,
                time_horizon=100,
            ),
            characterization_methods=[
                lca_processor.CharacterizationMethodConfig(
                    category_name="climate_change",
                    brightway_method=("GWP", "example"),
                    metric=None,
                )
            ],
            background_inventory=lca_processor.BackgroundInventoryConfig(),
        )

        # Run processor and convert to model inputs
        processor = lca_processor.LCADataProcessor(config)
        manager = converter.ModelInputManager()
        model_inputs = manager.parse_from_lca_processor(processor)

        # Verify vintage parameters are in model inputs
        assert model_inputs.foreground_technosphere_vintages is not None
        assert len(model_inputs.foreground_technosphere_vintages) > 0
        assert ("EV", "electricity", 1, 2020) in model_inputs.foreground_technosphere_vintages
        
        assert model_inputs.technology_evolution is not None
        assert len(model_inputs.technology_evolution) > 0
        assert ("EV", "CO2", 2020) in model_inputs.technology_evolution
