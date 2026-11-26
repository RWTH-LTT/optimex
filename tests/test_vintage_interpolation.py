"""
Regression tests for interpolating vintage-dependent parameters.

Each exchange is interpolated between the vintage years it defines itself, and
explicit vintage amounts take precedence over improvement factors per
(process, flow), not per tensor.
"""

from datetime import datetime

import bw2data as bd
import numpy as np
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution

from optimex import converter, lca_processor
from optimex.converter import (
    expand_foreground_tensor_with_evolution,
    expand_foreground_tensor_with_vintages,
)

SYSTEM_TIMES = [2020, 2025, 2030, 2035, 2040]


def test_vintage_amounts_use_own_vintage_years():
    """An exchange without a value at another exchange's vintage year is unaffected."""
    vintage_tensor = {
        ("P", "A", 1, 2020): 60.0,
        ("P", "A", 1, 2040): 40.0,
        ("P", "B", 1, 2030): 10.0,
    }
    expanded = expand_foreground_tensor_with_vintages(
        vintage_tensor, [2020, 2030, 2040], SYSTEM_TIMES
    )

    assert [expanded[("P", "A", 1, t)] for t in SYSTEM_TIMES] == pytest.approx(
        [60, 55, 50, 45, 40]
    )
    assert [expanded[("P", "B", 1, t)] for t in SYSTEM_TIMES] == pytest.approx([10] * 5)


def test_vintage_improvements_use_own_vintage_years():
    base = {("P", "A", 1): 100.0, ("P", "B", 1): 10.0}
    improvements = {
        ("P", "A", 2020): 1.0,
        ("P", "A", 2040): 0.5,
        ("P", "B", 2030): 0.8,
    }
    expanded = expand_foreground_tensor_with_evolution(
        base, improvements, [2020, 2030, 2040], SYSTEM_TIMES, "INTERMEDIATE_FLOW"
    )

    assert expanded[("P", "A", 1, 2030)] == pytest.approx(75.0)
    assert [expanded[("P", "B", 1, t)] for t in SYSTEM_TIMES] == pytest.approx(
        [8.0] * 5
    )


def _inputs(**vintage_data):
    return converter.OptimizationModelInputs(
        PROCESS=["P"],
        PRODUCT=["R"],
        INTERMEDIATE_FLOW=["A", "B"],
        ELEMENTARY_FLOW=["CO2"],
        BACKGROUND_ID=["db"],
        PROCESS_TIME=[0],
        SYSTEM_TIME=SYSTEM_TIMES,
        CATEGORY=["climate"],
        demand={("R", t): 1.0 for t in SYSTEM_TIMES},
        operation_flow={("P", "R"): True},
        foreground_technosphere={("P", "A", 0): 100.0, ("P", "B", 0): 10.0},
        internal_demand_technosphere={},
        foreground_biosphere={},
        foreground_production={("P", "R", 0): 1.0},
        background_inventory={},
        mapping={("db", t): 1.0 for t in SYSTEM_TIMES},
        characterization={("climate", "CO2", t): 1.0 for t in SYSTEM_TIMES},
        operation_time_limits={"P": (0, 0)},
        **vintage_data,
    )


def test_amounts_and_improvements_combine_per_flow():
    """Vintage amounts on one flow do not disable improvements on another flow."""
    scaled, _ = _inputs(
        foreground_technosphere_vintages={
            ("P", "A", 0, 2020): 100.0,
            ("P", "A", 0, 2040): 60.0,
        },
        vintage_improvements={("P", "B", 2020): 1.0, ("P", "B", 2040): 0.5},
    ).get_scaled_copy()

    overrides = scaled.foreground_technosphere_vintage_overrides
    fg_scale = max(100.0, 10.0, 1.0)
    assert overrides[("P", "A", 0, 2030)] * fg_scale == pytest.approx(80.0)
    assert overrides[("P", "B", 0, 2030)] * fg_scale == pytest.approx(7.5)


@bw2test
def test_vintage_amounts_year_key_is_exchange_total():
    """A vintage-year key is the total, split over process time like `amount`."""
    bd.projects.set_current("__test_vintage_amount_total__")
    bd.Database("biosphere3").write(
        {("biosphere3", "CO2"): {"type": "emission", "name": "carbon dioxide"}}
    )
    db = bd.Database("db_2020")
    db.write(
        {
            ("db_2020", "I1"): {
                "name": "I1",
                "location": "GLO",
                "reference product": "I1",
                "exchanges": [
                    {"amount": 1, "type": "production", "input": ("db_2020", "I1")},
                    {"amount": 1, "type": "biosphere", "input": ("biosphere3", "CO2")},
                ],
            }
        }
    )
    db.metadata["representative_time"] = datetime(2020, 1, 1).isoformat()
    db.register()

    two_years = TemporalDistribution(
        date=np.array([1, 2], dtype="timedelta64[Y]"), amount=np.array([0.5, 0.5])
    )
    bd.Database("foreground").write(
        {
            ("foreground", "R"): {
                "name": "R",
                "type": bd.labels.product_node_default,
                "unit": "kg",
            },
            ("foreground", "P"): {
                "name": "P",
                "type": bd.labels.process_node_default,
                "operation_time_limits": (1, 2),
                "exchanges": [
                    {
                        "amount": 1,
                        "type": bd.labels.production_edge_default,
                        "input": ("foreground", "R"),
                        "temporal_distribution": two_years,
                        "operation": True,
                    },
                    {
                        "amount": 60,
                        "type": bd.labels.consumption_edge_default,
                        "input": ("db_2020", "I1"),
                        "temporal_distribution": two_years,
                        "operation": True,
                        "vintage_amounts": {2020: 60, 2030: 40},
                    },
                ],
            },
        }
    )
    bd.Method(("GWP", "test")).write([(("biosphere3", "CO2"), 1)])

    processor = lca_processor.LCADataProcessor(
        lca_processor.LCAConfig(
            demand={
                bd.get_node(database="foreground", code="R"): TemporalDistribution(
                    date=np.array(["2022-01-01"], dtype="datetime64[s]"),
                    amount=np.array([1.0]),
                )
            },
            temporal={"start_date": datetime(2020, 1, 1)},
            characterization_methods=[
                {"category_name": "climate", "brightway_method": ("GWP", "test")}
            ],
            background_inventory={"calculation_method": "sequential"},
        )
    )

    vintages = processor.foreground_technosphere_vintages
    assert vintages[("P", "I1", 1, 2020)] == pytest.approx(30.0)
    assert vintages[("P", "I1", 2, 2020)] == pytest.approx(30.0)
    assert vintages[("P", "I1", 1, 2030)] == pytest.approx(20.0)
