"""
Test serialization and deserialization of OptimizationModelInputs with tuple keys.
"""
import json
import tempfile
from pathlib import Path

import pytest

from optimex import converter


def test_json_serialization_round_trip(abstract_system_model_inputs):
    """Test that saving and loading to JSON preserves tuple keys correctly."""
    # Create manager and load inputs
    manager = converter.ModelInputManager()
    original_inputs = converter.OptimizationModelInputs(**abstract_system_model_inputs)
    manager.model_inputs = original_inputs

    # Save to temporary JSON file
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_inputs.json"
        manager.save(str(json_path))

        # Verify file was created and is valid JSON
        assert json_path.exists()
        with open(json_path, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)

        # Load back from JSON
        manager_loaded = converter.ModelInputManager()
        loaded_inputs = manager_loaded.load(str(json_path))

        # Verify all tuple-key dictionaries are preserved
        tuple_key_fields = [
            "demand",
            "operation_flow",
            "foreground_technosphere",
            "internal_demand_technosphere",
            "foreground_biosphere",
            "foreground_production",
            "background_inventory",
            "mapping",
            "characterization",
        ]

        for field in tuple_key_fields:
            original_dict = getattr(original_inputs, field)
            loaded_dict = getattr(loaded_inputs, field)

            # Check that all keys are tuples in loaded data
            for key in loaded_dict.keys():
                assert isinstance(key, tuple), (
                    f"Key in {field} should be tuple, got {type(key)}"
                )

            # Check that the dictionaries are equal
            assert loaded_dict == original_dict, (
                f"Field {field} not preserved after JSON round-trip"
            )

        # Verify operation_time_limits (values are tuples)
        if original_inputs.operation_time_limits is not None:
            for key, value in loaded_inputs.operation_time_limits.items():
                assert isinstance(value, tuple), (
                    f"Value in operation_time_limits should be tuple, got {type(value)}"
                )
            assert loaded_inputs.operation_time_limits == original_inputs.operation_time_limits

        # Verify other fields
        assert loaded_inputs.PROCESS == original_inputs.PROCESS
        assert loaded_inputs.PRODUCT == original_inputs.PRODUCT
        assert loaded_inputs.CATEGORY == original_inputs.CATEGORY


def test_pickle_serialization_round_trip(abstract_system_model_inputs):
    """Test that saving and loading to pickle works (no tuple conversion needed)."""
    # Create manager and load inputs
    manager = converter.ModelInputManager()
    original_inputs = converter.OptimizationModelInputs(**abstract_system_model_inputs)
    manager.model_inputs = original_inputs

    # Save to temporary pickle file
    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = Path(tmpdir) / "test_inputs.pkl"
        manager.save(str(pkl_path))

        # Verify file was created
        assert pkl_path.exists()

        # Load back from pickle
        manager_loaded = converter.ModelInputManager()
        loaded_inputs = manager_loaded.load(str(pkl_path))

        # Verify all fields are preserved
        assert loaded_inputs.model_dump() == original_inputs.model_dump()


def test_json_with_optional_fields():
    """Test JSON serialization with optional constraint fields containing tuple keys."""
    minimal_inputs = {
        "PROCESS": ["P1"],
        "PRODUCT": ["R1"],
        "INTERMEDIATE_FLOW": ["I1"],
        "ELEMENTARY_FLOW": ["CO2"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0, 1],
        "SYSTEM_TIME": [2020, 2021],
        "CATEGORY": ["climate_change"],
        "demand": {("R1", 2020): 10.0},
        "operation_flow": {("P1", "R1"): True},
        "foreground_technosphere": {("P1", "I1", 0): 5.0},
        "internal_demand_technosphere": {},
        "foreground_biosphere": {("P1", "CO2", 1): 2.0},
        "foreground_production": {("P1", "R1", 1): 1.0},
        "background_inventory": {("db_2020", "I1", "CO2"): 1.0},
        "mapping": {("db_2020", 2020): 1.0},
        "characterization": {("climate_change", "CO2", 2020): 1e-12},
        "operation_time_limits": {"P1": (1, 1)},
        # Optional fields with tuple keys
        "process_limits_max": {("P1", 2020): 100.0},
        "process_limits_min": {("P1", 2020): 0.0},
        "process_coupling": {("P1", "P1"): 1.0},
    }

    manager = converter.ModelInputManager()
    original_inputs = converter.OptimizationModelInputs(**minimal_inputs)
    manager.model_inputs = original_inputs

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_optional.json"
        manager.save(str(json_path))

        manager_loaded = converter.ModelInputManager()
        loaded_inputs = manager_loaded.load(str(json_path))

        # Verify optional fields with tuple keys are preserved
        assert loaded_inputs.process_limits_max == original_inputs.process_limits_max
        assert loaded_inputs.process_limits_min == original_inputs.process_limits_min
        assert loaded_inputs.process_coupling == original_inputs.process_coupling


def test_invalid_file_extension():
    """Test that invalid file extensions raise an error."""
    manager = converter.ModelInputManager()
    manager.model_inputs = converter.OptimizationModelInputs(
        PROCESS=["P1"],
        PRODUCT=["R1"],
        INTERMEDIATE_FLOW=["I1"],
        ELEMENTARY_FLOW=["CO2"],
        BACKGROUND_ID=["db_2020"],
        PROCESS_TIME=[0],
        SYSTEM_TIME=[2020],
        CATEGORY=["climate_change"],
        demand={("R1", 2020): 10.0},
        operation_flow={("P1", "R1"): True},
        foreground_technosphere={},
        internal_demand_technosphere={},
        foreground_biosphere={},
        foreground_production={("P1", "R1", 0): 1.0},
        background_inventory={},
        mapping={("db_2020", 2020): 1.0},
        characterization={("climate_change", "CO2", 2020): 1e-12},
        operation_time_limits={"P1": (0, 0)},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_path = Path(tmpdir) / "test.txt"

        with pytest.raises(ValueError, match="Unsupported file extension"):
            manager.save(str(invalid_path))

        with pytest.raises(ValueError, match="Unsupported file extension"):
            manager.load(str(invalid_path))
