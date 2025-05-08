import pytest  # noqa: F401

from optimex import lca_processor


def test_lca_data_processor_initialization(mock_lca_data_processor):
    assert isinstance(mock_lca_data_processor, lca_processor.LCADataProcessor)
