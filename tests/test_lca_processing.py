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


def test_parse_demand(mock_lca_data_processor, abstract_system_model_inputs):
    demand_generated = mock_lca_data_processor.parse_demand()
    demand_expected = abstract_system_model_inputs["demand"]
    assert_dicts_equal_allowing_zeros(demand_generated, demand_expected)


def test_foreground_tensors(mock_lca_data_processor, abstract_system_model_inputs):
    mock_lca_data_processor.parse_demand()
    mock_lca_data_processor.construct_foreground_tensors()
    foreground_technosphere_generated = mock_lca_data_processor.foreground_technosphere
    foreground_biosphere_generated = mock_lca_data_processor.foreground_biosphere
    foreground_production_generated = mock_lca_data_processor.foreground_production

    foreground_technosphere_expected = abstract_system_model_inputs[
        "foreground_technosphere"
    ]
    foreground_biosphere_expected = abstract_system_model_inputs["foreground_biosphere"]
    foreground_production_expected = abstract_system_model_inputs[
        "foreground_production"
    ]

    assert_dicts_equal_allowing_zeros(
        foreground_technosphere_generated, foreground_technosphere_expected
    )
    assert_dicts_equal_allowing_zeros(
        foreground_biosphere_generated, foreground_biosphere_expected
    )
    assert_dicts_equal_allowing_zeros(
        foreground_production_generated, foreground_production_expected
    )


def test_set_process_operation_time(
    mock_lca_data_processor, abstract_system_model_inputs
):
    mock_lca_data_processor.parse_demand()
    mock_lca_data_processor.construct_foreground_tensors()
    process_operation_time_expected = abstract_system_model_inputs[
        "process_operation_time"
    ]
    mock_lca_data_processor.set_process_operation_time(process_operation_time_expected)
    process_operation_time_generated = mock_lca_data_processor.process_operation_time
    assert process_operation_time_generated == process_operation_time_expected


def test_sequential_inventory_tensor_calculation(
    mock_lca_data_processor, abstract_system_model_inputs
):
    mock_lca_data_processor.parse_demand
    mock_lca_data_processor.construct_foreground_tensors()
    mock_lca_data_processor.sequential_inventory_tensor_calculation()
    sequential_inventory_tensor_generated = mock_lca_data_processor.background_inventory
    sequential_inventory_tensor_expected = abstract_system_model_inputs[
        "background_inventory"
    ]
    assert_dicts_equal_allowing_zeros(
        sequential_inventory_tensor_generated, sequential_inventory_tensor_expected
    )


def test_characterization_tensor_calculation(
    mock_lca_data_processor, abstract_system_model_inputs
):
    raise NotImplementedError()
