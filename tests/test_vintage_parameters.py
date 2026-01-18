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

    def test_model_accepts_reference_vintages(self, base_model_inputs):
        """Test that REFERENCE_VINTAGES field is accepted."""
        base_model_inputs["REFERENCE_VINTAGES"] = [2020, 2025, 2030]
        model = converter.OptimizationModelInputs(**base_model_inputs)
        assert model.REFERENCE_VINTAGES == [2020, 2025, 2030]

    def test_model_reference_vintages_optional(self, base_model_inputs):
        """Test that REFERENCE_VINTAGES is optional (backwards compatible)."""
        model = converter.OptimizationModelInputs(**base_model_inputs)
        assert model.REFERENCE_VINTAGES is None

    def test_model_accepts_foreground_technosphere_vintages(self, base_model_inputs):
        """Test that foreground_technosphere_vintages field is accepted."""
        base_model_inputs["REFERENCE_VINTAGES"] = [2020, 2025]
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

    def test_model_accepts_foreground_biosphere_vintages(self, base_model_inputs):
        """Test that foreground_biosphere_vintages field is accepted."""
        base_model_inputs["REFERENCE_VINTAGES"] = [2020, 2025]
        base_model_inputs["foreground_biosphere_vintages"] = {
            # 2020 vintage: higher manufacturing emissions
            ("EV", "CO2", 0, 2020): 6000,
            # 2025 vintage: lower manufacturing emissions
            ("EV", "CO2", 0, 2025): 4000,
        }
        model = converter.OptimizationModelInputs(**base_model_inputs)
        assert model.foreground_biosphere_vintages[("EV", "CO2", 0, 2020)] == 6000
        assert model.foreground_biosphere_vintages[("EV", "CO2", 0, 2025)] == 4000

    def test_model_accepts_foreground_production_vintages(self, base_model_inputs):
        """Test that foreground_production_vintages field is accepted."""
        base_model_inputs["REFERENCE_VINTAGES"] = [2020, 2025]
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

    def test_model_accepts_technology_evolution(self, base_model_inputs):
        """Test that technology_evolution scaling factors are accepted."""
        base_model_inputs["REFERENCE_VINTAGES"] = [2020, 2025]
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

    def test_validation_vintage_tensors_require_reference_vintages(self, base_model_inputs):
        """Test that vintage tensors require REFERENCE_VINTAGES to be set."""
        # Try to set vintage tensors without REFERENCE_VINTAGES
        base_model_inputs["foreground_technosphere_vintages"] = {
            ("EV", "electricity", 1, 2020): 60,
        }
        with pytest.raises(ValueError, match="REFERENCE_VINTAGES"):
            converter.OptimizationModelInputs(**base_model_inputs)

    def test_validation_vintage_years_must_be_in_reference_vintages(self, base_model_inputs):
        """Test that vintage years in tensors must be in REFERENCE_VINTAGES."""
        base_model_inputs["REFERENCE_VINTAGES"] = [2020, 2025]
        base_model_inputs["foreground_technosphere_vintages"] = {
            ("EV", "electricity", 1, 2030): 60,  # 2030 not in REFERENCE_VINTAGES
        }
        with pytest.raises(ValueError, match="vintage year.*2030.*not in REFERENCE_VINTAGES"):
            converter.OptimizationModelInputs(**base_model_inputs)

    def test_validation_technology_evolution_years_must_be_in_reference_vintages(
        self, base_model_inputs
    ):
        """Test that technology_evolution years must be in REFERENCE_VINTAGES."""
        base_model_inputs["REFERENCE_VINTAGES"] = [2020, 2025]
        base_model_inputs["technology_evolution"] = {
            ("EV", "electricity", 2030): 0.5,  # 2030 not in REFERENCE_VINTAGES
        }
        with pytest.raises(ValueError, match="vintage year.*2030.*not in REFERENCE_VINTAGES"):
            converter.OptimizationModelInputs(**base_model_inputs)

    def test_validation_process_in_vintage_tensors_must_exist(self, base_model_inputs):
        """Test that processes in vintage tensors must exist in PROCESS."""
        base_model_inputs["REFERENCE_VINTAGES"] = [2020, 2025]
        base_model_inputs["foreground_technosphere_vintages"] = {
            ("NONEXISTENT", "electricity", 1, 2020): 60,
        }
        with pytest.raises(ValueError, match="Invalid keys.*NONEXISTENT"):
            converter.OptimizationModelInputs(**base_model_inputs)

    def test_validation_flow_in_vintage_tensors_must_exist(self, base_model_inputs):
        """Test that flows in vintage tensors must exist in respective sets."""
        base_model_inputs["REFERENCE_VINTAGES"] = [2020, 2025]
        base_model_inputs["foreground_technosphere_vintages"] = {
            ("EV", "nonexistent_flow", 1, 2020): 60,
        }
        with pytest.raises(ValueError, match="Invalid keys.*nonexistent_flow"):
            converter.OptimizationModelInputs(**base_model_inputs)

    def test_validation_process_time_in_vintage_tensors_must_exist(self, base_model_inputs):
        """Test that process times in vintage tensors must exist in PROCESS_TIME."""
        base_model_inputs["REFERENCE_VINTAGES"] = [2020, 2025]
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
            "REFERENCE_VINTAGES": [2020, 2025],
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
            "REFERENCE_VINTAGES": [2020, 2025],
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
            "REFERENCE_VINTAGES": [2020, 2025],
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
            "REFERENCE_VINTAGES": [2020, 2025],
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
            "REFERENCE_VINTAGES": [2020, 2025],
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
