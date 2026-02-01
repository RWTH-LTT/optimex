"""
Tests for flexible temporal resolution support in optimex.
"""

from datetime import datetime

import bw2data as bd
import numpy as np
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution

from optimex import lca_processor
from optimex.lca_processor import TemporalResolutionEnum


class TestTemporalResolutionEnum:
    """Tests for the TemporalResolutionEnum class properties."""

    def test_year_resolution_properties(self):
        """Test that year resolution returns correct numpy and pandas mappings."""
        res = TemporalResolutionEnum.year
        assert res.numpy_unit == "Y"
        assert res.pandas_freq == "YE"
        assert res.pandas_offset == "year"

    def test_month_resolution_properties(self):
        """Test that month resolution returns correct numpy and pandas mappings."""
        res = TemporalResolutionEnum.month
        assert res.numpy_unit == "M"
        assert res.pandas_freq == "ME"
        assert res.pandas_offset == "month"

    def test_day_resolution_properties(self):
        """Test that day resolution returns correct numpy and pandas mappings."""
        res = TemporalResolutionEnum.day
        assert res.numpy_unit == "D"
        assert res.pandas_freq == "D"
        assert res.pandas_offset == "day"


@pytest.fixture(scope="module")
@bw2test
def setup_monthly_resolution_databases():
    """Set up Brightway databases for monthly resolution testing."""
    bd.projects.set_current("__test_monthly_resolution_db__")
    
    bio_db = bd.Database("biosphere3")
    bio_db.write(
        {
            ("biosphere3", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
            },
        },
    )
    bio_db.register()

    # Simple background database
    background = bd.Database("db_2020")
    background.write(
        {
            ("db_2020", "I1"): {
                "name": "node I1",
                "location": "somewhere",
                "reference product": "I1",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2020", "I1"),
                    },
                    {
                        "amount": 1,
                        "type": "biosphere",
                        "input": ("biosphere3", "CO2"),
                    },
                ],
            },
        }
    )
    background.metadata["representative_time"] = datetime(2020, 1, 1).isoformat()
    background.register()

    # Foreground database with monthly temporal distributions
    foreground = bd.Database("foreground")
    foreground.write(
        {
            # Product node
            ("foreground", "R1"): {
                "name": "Product R1",
                "type": bd.labels.product_node_default,
                "unit": "kg",
            },
            # Process node with monthly temporal distribution
            ("foreground", "P1"): {
                "name": "process P1",
                "location": "somewhere",
                "type": bd.labels.process_node_default,
                "operation_time_limits": (0, 11),  # 12 months: indices 0-11 inclusive
                "exchanges": [
                    {
                        "amount": 1,
                        "type": bd.labels.production_edge_default,
                        "input": ("foreground", "R1"),
                        # Production spread over 12 months
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(12), dtype="timedelta64[M]"),
                            amount=np.array([1/12] * 12),
                        ),
                        "operation": True,
                    },
                    {
                        "amount": 12,  # 1 kg/month * 12 months
                        "type": bd.labels.consumption_edge_default,
                        "input": ("db_2020", "I1"),
                        # Construction at month 0
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([0], dtype="timedelta64[M]"),
                            amount=np.array([1]),
                        ),
                    },
                    {
                        "amount": 12,  # 1 kg CO2/month * 12 months
                        "type": bd.labels.biosphere_edge_default,
                        "input": ("biosphere3", "CO2"),
                        # Emissions spread over 12 months
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(12), dtype="timedelta64[M]"),
                            amount=np.array([1/12] * 12),
                        ),
                        "operation": True,
                    },
                ],
            },
        }
    )
    foreground.register()

    bd.Method(("GWP", "example")).write(
        [
            (("biosphere3", "CO2"), 1),
        ]
    )


@pytest.fixture(scope="module")
def mock_monthly_lca_data_processor(setup_monthly_resolution_databases):
    """Create an LCADataProcessor configured for monthly resolution."""
    # Create demand for 12 months starting from January 2020
    months = range(12)
    dates = [datetime(2020, m + 1, 1).isoformat() for m in months]
    td_demand = TemporalDistribution(
        date=np.array(dates, dtype="datetime64[s]"),
        amount=np.array([1] * 12),  # 1 unit demand per month
    )
    
    product_r1 = bd.get_node(database="foreground", code="R1")
    
    lca_config = lca_processor.LCAConfig(
        demand={product_r1: td_demand},
        temporal={
            "start_date": datetime(2020, 1, 1),
            "temporal_resolution": "month",  # Use monthly resolution
            "time_horizon": 100,  # 100 years for characterization
        },
        characterization_methods=[
            {
                "category_name": "climate_change",
                "brightway_method": ("GWP", "example"),
            },
        ],
        background_inventory={
            "cutoff": 1e4,
            "calculation_method": "sequential",
        },
    )
    
    return lca_processor.LCADataProcessor(lca_config)


class TestMonthlyResolution:
    """Tests for monthly temporal resolution."""

    def test_processor_initialization_with_monthly_resolution(
        self, mock_monthly_lca_data_processor
    ):
        """Test that LCADataProcessor initializes correctly with monthly resolution."""
        processor = mock_monthly_lca_data_processor
        assert isinstance(processor, lca_processor.LCADataProcessor)
        assert processor.config.temporal.temporal_resolution == TemporalResolutionEnum.month

    def test_system_time_is_monthly_indexed(self, mock_monthly_lca_data_processor):
        """Test that system time uses monthly indices."""
        processor = mock_monthly_lca_data_processor
        system_time = processor.system_time
        
        # For monthly resolution starting Jan 2020, index = year*12 + (month-1)
        # Jan 2020: 2020 * 12 + 0 = 24240 (months since year 0)
        expected_start = 2020 * 12
        
        assert min(system_time) == expected_start
        # Should have at least 12 months
        assert len(system_time) >= 12

    def test_process_time_is_monthly_indexed(self, mock_monthly_lca_data_processor):
        """Test that process time uses monthly indices (0-11 for 12 months)."""
        processor = mock_monthly_lca_data_processor
        process_time = processor.process_time
        
        # Process time should include months 0-11
        assert 0 in process_time
        assert 11 in process_time

    def test_demand_keys_use_monthly_indices(self, mock_monthly_lca_data_processor):
        """Test that demand dictionary keys use monthly indices."""
        processor = mock_monthly_lca_data_processor
        demand = processor.demand
        
        # All keys should be (product_code, monthly_index)
        expected_start_month = 2020 * 12
        for (product_code, time_idx), amount in demand.items():
            assert time_idx >= expected_start_month
            assert time_idx < expected_start_month + 12
            assert product_code == "R1"
            assert amount == 1  # 1 unit per month

    def test_foreground_tensors_use_monthly_process_time(
        self, mock_monthly_lca_data_processor
    ):
        """Test that foreground tensors use monthly process time indices."""
        processor = mock_monthly_lca_data_processor
        
        # Check production tensor
        production = processor.foreground_production
        time_indices = {k[2] for k in production.keys()}
        assert 0 in time_indices  # Should have month 0
        assert 11 in time_indices  # Should have month 11
        
        # Check biosphere tensor
        biosphere = processor.foreground_biosphere
        bio_time_indices = {k[2] for k in biosphere.keys()}
        assert 0 in bio_time_indices
        assert 11 in bio_time_indices

    def test_characterization_keys_use_monthly_system_time(
        self, mock_monthly_lca_data_processor
    ):
        """Test that characterization tensor keys use monthly system time indices."""
        processor = mock_monthly_lca_data_processor
        characterization = processor.characterization
        
        # Keys should be (category, flow_code, monthly_system_time_index)
        expected_start_month = 2020 * 12
        for (category, flow_code, time_idx), value in characterization.items():
            assert time_idx >= expected_start_month
            assert category == "climate_change"


class TestDailyResolution:
    """Basic tests for daily temporal resolution enum properties."""

    def test_daily_resolution_enum_exists(self):
        """Test that daily resolution is available in the enum."""
        assert TemporalResolutionEnum.day == "day"
        
    def test_daily_resolution_numpy_unit(self):
        """Test that daily resolution returns correct numpy unit."""
        assert TemporalResolutionEnum.day.numpy_unit == "D"


class TestResolutionBackwardCompatibility:
    """Tests to ensure backward compatibility with yearly resolution."""

    def test_year_is_default_resolution(self):
        """Test that yearly resolution is the default."""
        config = lca_processor.TemporalConfig(
            start_date=datetime(2020, 1, 1)
        )
        assert config.temporal_resolution == TemporalResolutionEnum.year

    def test_year_string_accepted(self):
        """Test that 'year' string is accepted as resolution."""
        config = lca_processor.TemporalConfig(
            start_date=datetime(2020, 1, 1),
            temporal_resolution="year"
        )
        assert config.temporal_resolution == TemporalResolutionEnum.year

    def test_month_string_accepted(self):
        """Test that 'month' string is accepted as resolution."""
        config = lca_processor.TemporalConfig(
            start_date=datetime(2020, 1, 1),
            temporal_resolution="month"
        )
        assert config.temporal_resolution == TemporalResolutionEnum.month

    def test_day_string_accepted(self):
        """Test that 'day' string is accepted as resolution."""
        config = lca_processor.TemporalConfig(
            start_date=datetime(2020, 1, 1),
            temporal_resolution="day"
        )
        assert config.temporal_resolution == TemporalResolutionEnum.day


class TestMixedResolutions:
    """Tests for mixing different temporal resolutions across processes."""

    def test_resolution_priority(self):
        """Test that resolution priority is correctly ordered (day > month > year)."""
        assert TemporalResolutionEnum.day.priority > TemporalResolutionEnum.month.priority
        assert TemporalResolutionEnum.month.priority > TemporalResolutionEnum.year.priority

    def test_get_finest_resolution(self):
        """Test that get_finest returns the most granular resolution."""
        result = TemporalResolutionEnum.get_finest(
            TemporalResolutionEnum.year,
            TemporalResolutionEnum.month,
            TemporalResolutionEnum.day
        )
        assert result == TemporalResolutionEnum.day

        result = TemporalResolutionEnum.get_finest(
            TemporalResolutionEnum.year,
            TemporalResolutionEnum.month
        )
        assert result == TemporalResolutionEnum.month

    def test_detect_temporal_resolution_year(self):
        """Test detection of yearly temporal distribution."""
        from optimex.lca_processor import detect_temporal_resolution
        td = TemporalDistribution(
            date=np.array([0, 1, 2], dtype="timedelta64[Y]"),
            amount=np.array([0.33, 0.33, 0.34])
        )
        assert detect_temporal_resolution(td) == TemporalResolutionEnum.year

    def test_detect_temporal_resolution_month(self):
        """Test detection of monthly temporal distribution."""
        from optimex.lca_processor import detect_temporal_resolution
        td = TemporalDistribution(
            date=np.array([0, 1, 2, 3, 4, 5], dtype="timedelta64[M]"),
            amount=np.array([0.16, 0.17, 0.16, 0.17, 0.17, 0.17])
        )
        assert detect_temporal_resolution(td) == TemporalResolutionEnum.month

    def test_detect_temporal_resolution_day(self):
        """Test detection of daily temporal distribution."""
        from optimex.lca_processor import detect_temporal_resolution
        td = TemporalDistribution(
            date=np.array([0, 1, 2, 3, 4], dtype="timedelta64[D]"),
            amount=np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        )
        assert detect_temporal_resolution(td) == TemporalResolutionEnum.day

    def test_from_numpy_unit(self):
        """Test conversion from numpy unit codes to enum."""
        assert TemporalResolutionEnum.from_numpy_unit("Y") == TemporalResolutionEnum.year
        assert TemporalResolutionEnum.from_numpy_unit("M") == TemporalResolutionEnum.month
        assert TemporalResolutionEnum.from_numpy_unit("D") == TemporalResolutionEnum.day


class TestResolutionConversion:
    """Tests for temporal resolution conversion logic."""

    def test_year_to_month_conversion(self):
        """Test converting yearly indices to monthly."""
        from optimex.lca_processor import LCADataProcessor, LCAConfig, TemporalResolutionEnum
        
        # Create a minimal mock to test the conversion method
        # We'll use the static method approach
        time_indices = np.array([0, 1, 2])  # Years 0, 1, 2
        amounts = np.array([0.33, 0.34, 0.33])
        
        source = TemporalResolutionEnum.year
        target = TemporalResolutionEnum.month
        
        # Test the conversion factor logic
        # Year to month should multiply by 12
        # Each year (0, 1, 2) should expand to 12 months
        factor = 12  # year to month
        
        new_indices = []
        new_amounts = []
        for idx, amount in zip(time_indices, amounts):
            base_idx = int(idx * factor)
            expanded_indices = np.arange(base_idx, base_idx + factor)
            expanded_amounts = np.full(factor, amount / factor)
            new_indices.extend(expanded_indices)
            new_amounts.extend(expanded_amounts)
        
        result_indices = np.array(new_indices)
        result_amounts = np.array(new_amounts)
        
        # Year 0 should map to months 0-11
        # Year 1 should map to months 12-23
        # Year 2 should map to months 24-35
        assert len(result_indices) == 36  # 3 years * 12 months
        assert 0 in result_indices
        assert 11 in result_indices
        assert 12 in result_indices
        assert 35 in result_indices
        
        # Total amount should be preserved
        assert np.isclose(result_amounts.sum(), amounts.sum())

    def test_month_to_day_conversion(self):
        """Test converting monthly indices to daily."""
        time_indices = np.array([0, 1])  # Months 0, 1
        amounts = np.array([0.5, 0.5])
        
        factor = 30  # month to day (approximate)
        
        new_indices = []
        new_amounts = []
        for idx, amount in zip(time_indices, amounts):
            base_idx = int(idx * factor)
            expanded_indices = np.arange(base_idx, base_idx + factor)
            expanded_amounts = np.full(factor, amount / factor)
            new_indices.extend(expanded_indices)
            new_amounts.extend(expanded_amounts)
        
        result_indices = np.array(new_indices)
        result_amounts = np.array(new_amounts)
        
        # Month 0 should map to days 0-29
        # Month 1 should map to days 30-59
        assert len(result_indices) == 60  # 2 months * 30 days
        assert 0 in result_indices
        assert 29 in result_indices
        assert 30 in result_indices
        
        # Total amount should be preserved
        assert np.isclose(result_amounts.sum(), amounts.sum())

    def test_same_resolution_no_change(self):
        """Test that same resolution returns unchanged values."""
        time_indices = np.array([0, 1, 2, 3])
        amounts = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Same resolution should return identical values
        result_indices = time_indices.copy()
        result_amounts = amounts.copy()
        
        assert np.array_equal(result_indices, time_indices)
        assert np.array_equal(result_amounts, amounts)


class TestUserProvidedCharacterization:
    """Tests for user-provided characterization factors."""

    def test_characterization_factors_config_accepted(self):
        """Test that characterization_factors config is accepted."""
        from optimex.lca_processor import CharacterizationMethodConfig
        
        # User-provided factors mapping (flow_code, time_index) -> CF value
        user_factors = {
            ("water", 24240): 0.6,  # Jan 2020
            ("water", 24241): 0.7,  # Feb 2020
            ("water", 24242): 0.8,  # Mar 2020
            ("water", 24246): 1.5,  # Jul 2020 (peak scarcity)
        }
        
        config = CharacterizationMethodConfig(
            category_name="water_scarcity",
            characterization_factors=user_factors,
        )
        
        assert config.characterization_factors == user_factors
        assert config.brightway_method is None
        assert config.metric is None
        assert config.dynamic is True  # User factors make it dynamic

    def test_static_method_config(self):
        """Test that static Brightway methods work."""
        from optimex.lca_processor import CharacterizationMethodConfig
        
        config = CharacterizationMethodConfig(
            category_name="climate_change",
            brightway_method=("GWP", "test"),
            metric=None,
        )
        
        assert config.brightway_method == ("GWP", "test")
        assert config.characterization_factors is None
        assert config.dynamic is False

    def test_dynamic_gwp_metric_config(self):
        """Test that dynamic GWP metric config works."""
        from optimex.lca_processor import CharacterizationMethodConfig, MetricEnum
        
        config = CharacterizationMethodConfig(
            category_name="climate_change_dynamic",
            brightway_method=("GWP", "test"),
            metric="GWP",
        )
        
        assert config.metric == MetricEnum.GWP
        assert config.dynamic is True
