"""Tests for optimex.utils module - optimex-specific utilities."""

import bw2data as bd
import numpy as np
import pytest
from bw_temporalis import TemporalDistribution
from datetime import datetime

from optimex import utils


@pytest.fixture
def clean_project():
    """Create a clean test project."""
    project_name = "test_utils_project"
    if project_name in bd.projects:
        bd.projects.delete_project(project_name, delete_dir=True)
    bd.projects.set_current(project_name)

    # Create a minimal biosphere
    biosphere = bd.Database("biosphere3")
    biosphere.write(
        {
            ("biosphere3", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
            },
        }
    )

    # Create a simple foreground database
    foreground = bd.Database("foreground")
    foreground.register()

    yield foreground

    # Cleanup
    if project_name in bd.projects:
        bd.projects.delete_project(project_name, delete_dir=True)


class TestSetOperationTimeLimits:
    """Tests for set_operation_time_limits function."""

    def test_basic_operation_time_limits(self, clean_project):
        """Test setting basic operation time limits."""
        fg = clean_project
        process = fg.new_node(
            name="test_process", code="proc1", type=bd.labels.process_node_default
        )
        process.save()

        utils.set_operation_time_limits(process, start=0, end=10)

        assert process["operation_time_limits"] == (0, 10)

    def test_custom_time_limits(self, clean_project):
        """Test with different time limits."""
        fg = clean_project
        process = fg.new_node(
            name="test_process", code="proc1", type=bd.labels.process_node_default
        )
        process.save()

        utils.set_operation_time_limits(process, start=5, end=15)

        assert process["operation_time_limits"] == (5, 15)

    def test_save_parameter(self, clean_project):
        """Test save parameter."""
        fg = clean_project
        process = fg.new_node(
            name="test_process", code="proc1", type=bd.labels.process_node_default
        )
        process.save()

        # With save=False
        utils.set_operation_time_limits(process, start=0, end=10, save=False)
        # Manually save
        process.save()

        assert process["operation_time_limits"] == (0, 10)

    def test_chaining(self, clean_project):
        """Test that function returns process for chaining."""
        fg = clean_project
        process = fg.new_node(
            name="test_process", code="proc1", type=bd.labels.process_node_default
        )
        process.save()

        result = utils.set_operation_time_limits(process, start=0, end=10)

        assert result == process


class TestAddTemporalDistributionToExchanges:
    """Tests for add_temporal_distribution_to_exchanges function."""

    def test_add_to_process(self, clean_project):
        """Test adding temporal distribution to all process exchanges."""
        fg = clean_project
        product = fg.new_node(
            name="product", code="prod1", type=bd.labels.product_node_default
        )
        product.save()

        process = fg.new_node(
            name="process", code="proc1", type=bd.labels.process_node_default
        )
        process.save()

        process.new_edge(
            input=product, amount=1.0, type=bd.labels.production_edge_default
        ).save()
        process.new_edge(
            input=bd.get_node(database="biosphere3", code="CO2"),
            amount=2.0,
            type="biosphere",
        ).save()

        utils.add_temporal_distribution_to_exchanges(process, start=0, end=5)

        # Check all exchanges have temporal distribution
        for exc in process.exchanges():
            assert "temporal_distribution" in exc
            assert isinstance(exc["temporal_distribution"], TemporalDistribution)

    def test_add_to_exchange_list(self, clean_project):
        """Test adding temporal distribution to a list of exchanges."""
        fg = clean_project
        product = fg.new_node(
            name="product", code="prod1", type=bd.labels.product_node_default
        )
        product.save()

        process = fg.new_node(
            name="process", code="proc1", type=bd.labels.process_node_default
        )
        process.save()

        process.new_edge(
            input=product, amount=1.0, type=bd.labels.production_edge_default
        ).save()

        # Get only technosphere exchanges
        tech_exchanges = list(process.technosphere())

        utils.add_temporal_distribution_to_exchanges(tech_exchanges, start=0, end=10)

        for exc in tech_exchanges:
            assert "temporal_distribution" in exc

    def test_custom_steps(self, clean_project):
        """Test with custom steps parameter."""
        fg = clean_project
        product = fg.new_node(
            name="product", code="prod1", type=bd.labels.product_node_default
        )
        product.save()

        process = fg.new_node(
            name="process", code="proc1", type=bd.labels.process_node_default
        )
        process.save()
        process.new_edge(
            input=product, amount=1.0, type=bd.labels.production_edge_default
        ).save()

        utils.add_temporal_distribution_to_exchanges(
            process, start=0, end=10, steps=11
        )

        for exc in process.exchanges():
            td = exc["temporal_distribution"]
            assert isinstance(td, TemporalDistribution)
            assert len(td.amount) == 11


class TestMarkExchangesAsOperation:
    """Tests for mark_exchanges_as_operation function."""

    def test_mark_process_exchanges(self, clean_project):
        """Test marking all exchanges of a process as operation."""
        fg = clean_project
        product = fg.new_node(
            name="product", code="prod1", type=bd.labels.product_node_default
        )
        product.save()

        process = fg.new_node(
            name="process", code="proc1", type=bd.labels.process_node_default
        )
        process.save()
        process.new_edge(
            input=product, amount=1.0, type=bd.labels.production_edge_default
        ).save()

        utils.mark_exchanges_as_operation(process)

        for exc in process.exchanges():
            assert exc["operation"] is True

    def test_mark_exchange_list(self, clean_project):
        """Test marking a list of exchanges."""
        fg = clean_project
        product = fg.new_node(
            name="product", code="prod1", type=bd.labels.product_node_default
        )
        product.save()

        process = fg.new_node(
            name="process", code="proc1", type=bd.labels.process_node_default
        )
        process.save()
        process.new_edge(
            input=product, amount=1.0, type=bd.labels.production_edge_default
        ).save()
        process.new_edge(
            input=bd.get_node(database="biosphere3", code="CO2"),
            amount=2.0,
            type="biosphere",
        ).save()

        # Mark only technosphere exchanges
        tech_exchanges = list(process.technosphere())
        utils.mark_exchanges_as_operation(tech_exchanges)

        # Check technosphere is marked
        for exc in process.technosphere():
            assert exc["operation"] is True

        # Biosphere might not be marked
        # (depends on whether it was in the list)


class TestSetupOptimexProcess:
    """Tests for setup_optimex_process function."""

    def test_basic_setup(self, clean_project):
        """Test basic optimex process setup."""
        fg = clean_project
        product = fg.new_node(
            name="product", code="prod1", type=bd.labels.product_node_default
        )
        product.save()

        process = fg.new_node(
            name="process", code="proc1", type=bd.labels.process_node_default
        )
        process.save()
        process.new_edge(
            input=product, amount=1.0, type=bd.labels.production_edge_default
        ).save()
        process.new_edge(
            input=bd.get_node(database="biosphere3", code="CO2"),
            amount=2.0,
            type="biosphere",
        ).save()

        utils.setup_optimex_process(process, operation_time_limits=(0, 10))

        # Check operation time limits
        assert process["operation_time_limits"] == (0, 10)

        # Check temporal distributions
        for exc in process.exchanges():
            assert "temporal_distribution" in exc
            assert exc["operation"] is True

    def test_custom_temporal_params(self, clean_project):
        """Test with custom temporal distribution parameters."""
        fg = clean_project
        product = fg.new_node(
            name="product", code="prod1", type=bd.labels.product_node_default
        )
        product.save()

        process = fg.new_node(
            name="process", code="proc1", type=bd.labels.process_node_default
        )
        process.save()
        process.new_edge(
            input=product, amount=1.0, type=bd.labels.production_edge_default
        ).save()

        utils.setup_optimex_process(
            process,
            operation_time_limits=(0, 8),
            temporal_distribution_params={"start": 0, "end": 8, "steps": 9},
        )

        assert process["operation_time_limits"] == (0, 8)
        for exc in process.exchanges():
            assert "temporal_distribution" in exc
            assert len(exc["temporal_distribution"].amount) == 9

    def test_without_operation_marking(self, clean_project):
        """Test setup without marking as operation."""
        fg = clean_project
        product = fg.new_node(
            name="product", code="prod1", type=bd.labels.product_node_default
        )
        product.save()

        process = fg.new_node(
            name="process", code="proc1", type=bd.labels.process_node_default
        )
        process.save()
        process.new_edge(
            input=product, amount=1.0, type=bd.labels.production_edge_default
        ).save()

        utils.setup_optimex_process(
            process, operation_time_limits=(0, 10), mark_as_operation=False
        )

        # Check operation time limits are set
        assert process["operation_time_limits"] == (0, 10)
        # Check temporal distributions are set
        for exc in process.exchanges():
            assert "temporal_distribution" in exc
        # But operation flag should not be set
        for exc in process.exchanges():
            assert exc.get("operation") is not True


class TestSetupDatabaseTemporalMetadata:
    """Tests for setup_database_temporal_metadata function."""

    def test_basic_metadata_setup(self, clean_project):
        """Test basic temporal metadata setup."""
        db_2020 = bd.Database("db_2020")
        db_2020.register()
        db_2030 = bd.Database("db_2030")
        db_2030.register()

        dbs = {2020: db_2020, 2030: db_2030}

        utils.setup_database_temporal_metadata(dbs)

        assert db_2020.metadata["representative_time"] == "2020-01-01T00:00:00"
        assert db_2030.metadata["representative_time"] == "2030-01-01T00:00:00"

    def test_custom_date(self, clean_project):
        """Test with custom month and day."""
        db_2020 = bd.Database("db_2020")
        db_2020.register()

        utils.setup_database_temporal_metadata({2020: db_2020}, month=6, day=15)

        assert db_2020.metadata["representative_time"] == "2020-06-15T00:00:00"


class TestCreateTemporalDemand:
    """Tests for create_temporal_demand function."""

    def test_basic_demand_creation(self, clean_project):
        """Test basic temporal demand creation."""
        fg = clean_project
        product = fg.new_node(
            name="product", code="prod1", type=bd.labels.product_node_default
        )
        product.save()

        demand = utils.create_temporal_demand(product, range(2025, 2030))

        assert product in demand
        assert isinstance(demand[product], TemporalDistribution)
        assert len(demand[product].amount) == 5

    def test_demand_with_noise(self, clean_project):
        """Test temporal demand with noise."""
        fg = clean_project
        product = fg.new_node(
            name="product", code="prod1", type=bd.labels.product_node_default
        )
        product.save()

        demand = utils.create_temporal_demand(
            product, range(2025, 2030), noise_std=2.0, random_seed=42
        )

        # With random seed, results should be reproducible
        amounts1 = demand[product].amount

        demand2 = utils.create_temporal_demand(
            product, range(2025, 2030), noise_std=2.0, random_seed=42
        )
        amounts2 = demand2[product].amount

        np.testing.assert_array_equal(amounts1, amounts2)

    def test_custom_amounts(self, clean_project):
        """Test with custom amounts array."""
        fg = clean_project
        product = fg.new_node(
            name="product", code="prod1", type=bd.labels.product_node_default
        )
        product.save()

        custom_amounts = np.array([10, 15, 20, 25, 30])
        demand = utils.create_temporal_demand(
            product, range(2025, 2030), amounts=custom_amounts
        )

        np.testing.assert_array_equal(demand[product].amount, custom_amounts)

    def test_trend_parameters(self, clean_project):
        """Test with custom trend parameters."""
        fg = clean_project
        product = fg.new_node(
            name="product", code="prod1", type=bd.labels.product_node_default
        )
        product.save()

        demand = utils.create_temporal_demand(
            product, range(2025, 2030), trend_start=5.0, trend_end=15.0, noise_std=0.0
        )

        # Without noise, should be a linear trend
        amounts = demand[product].amount
        assert amounts[0] == 5.0
        assert amounts[-1] == 15.0
        # Check it's roughly linear
        assert np.allclose(np.diff(amounts), np.diff(amounts)[0])

