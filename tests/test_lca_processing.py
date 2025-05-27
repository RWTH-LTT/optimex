import numpy as np
import pytest  # noqa: F401

from optimex import lca_processor


def assert_dicts_equal_allowing_zeros(dict_a, dict_b):
    """
    Assert that two dictionaries with tuple keys are equal, allowing keys with
    zero value to be missing from either dict.

    Raises AssertionError with descriptive message if inequality found.
    """

    # Combine all keys from both dicts
    all_keys = set(dict_a) | set(dict_b)

    for key in all_keys:
        val_a = dict_a.get(key, 0)
        val_b = dict_b.get(key, 0)

        # Allow missing keys only if value is zero in either
        if val_a == 0 and key not in dict_b:
            continue
        if val_b == 0 and key not in dict_a:
            continue

        # Otherwise, they must match exactly
        if not np.isclose(val_a, val_b, atol=1e-5):
            raise AssertionError(
                f"Mismatch at key {key}: dict_a has {val_a}, dict_b has {val_b}"
            )


def test_lca_data_processor_initialization(mock_lca_data_processor):
    assert isinstance(mock_lca_data_processor, lca_processor.LCADataProcessor)


def test_demands(mock_lca_data_processor, abstract_system_model_inputs):
    demand_generated = mock_lca_data_processor.demand
    demand_expected = abstract_system_model_inputs["demand"]
    assert_dicts_equal_allowing_zeros(demand_generated, demand_expected)


def test_foreground_tensors(mock_lca_data_processor, abstract_system_model_inputs):
    fg_technosphere = mock_lca_data_processor.foreground_technosphere
    fg_biosphere = mock_lca_data_processor.foreground_biosphere
    fg_production = mock_lca_data_processor.foreground_production
    operation_time_limits = mock_lca_data_processor.operation_time_limits

    expected = abstract_system_model_inputs
    fg_technosphere_expected = expected["foreground_technosphere"]
    fg_biosphere_expected = expected["foreground_biosphere"]
    fg_production_expected = expected["foreground_production"]
    operation_time_limits_expected = expected["operation_time_limits"]

    assert_dicts_equal_allowing_zeros(fg_technosphere, fg_technosphere_expected)
    assert_dicts_equal_allowing_zeros(fg_biosphere, fg_biosphere_expected)
    assert_dicts_equal_allowing_zeros(fg_production, fg_production_expected)
    assert operation_time_limits_expected == operation_time_limits


def test_sequential_inventory_tensor_calculation(
    mock_lca_data_processor, abstract_system_model_inputs
):
    sequential_inventory_tensor_generated = mock_lca_data_processor.background_inventory
    sequential_inventory_tensor_expected = abstract_system_model_inputs[
        "background_inventory"
    ]

    assert_dicts_equal_allowing_zeros(
        sequential_inventory_tensor_generated, sequential_inventory_tensor_expected
    )


def test_mapping(mock_lca_data_processor, abstract_system_model_inputs):
    mapping_generated = mock_lca_data_processor.mapping
    mapping_expected = abstract_system_model_inputs["mapping"]
    assert_dicts_equal_allowing_zeros(mapping_generated, mapping_expected)


def test_characterization_tensor(mock_lca_data_processor, abstract_system_model_inputs):
    characterization_tensor_generated = mock_lca_data_processor.characterization
    characterization_tensor_expected = abstract_system_model_inputs["characterization"]
    assert_dicts_equal_allowing_zeros(
        characterization_tensor_generated, characterization_tensor_expected
    )
