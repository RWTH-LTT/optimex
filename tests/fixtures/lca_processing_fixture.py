from datetime import datetime

import bw2data as bd
import numpy as np
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution

from optimex import lca_processor


@pytest.fixture(scope="module")
@bw2test
def setup_brightway_databases():
    bd.projects.set_current("__test_standalone_db__")
    bd.Database("biosphere3").write(
        {
            ("biosphere3", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
            },
            ("biosphere3", "CH4"): {
                "type": "emission",
                "name": "methane",
            },
        },
    )

    bd.Database("db_2020").write(
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
            ("db_2020", "I2"): {
                "name": "node I2",
                "location": "somewhere",
                "reference product": "I2",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2020", "I2"),
                    },
                    {
                        "amount": 1,
                        "type": "biosphere",
                        "input": ("biosphere3", "CH4"),
                    },
                ],
            },
        }
    )
    bd.Database("db_2030").write(
        {
            ("db_2030", "I1"): {
                "name": "node I1",
                "location": "somewhere",
                "reference product": "I1",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2030", "I1"),
                    },
                    {
                        "amount": 0.9,
                        "type": "biosphere",
                        "input": ("biosphere3", "CO2"),
                    },
                ],
            },
            ("db_2030", "I2"): {
                "name": "node I2",
                "location": "somewhere",
                "reference product": "I2",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2030", "I2"),
                    },
                    {
                        "amount": 0.9,
                        "type": "biosphere",
                        "input": ("biosphere3", "CH4"),
                    },
                ],
            },
        }
    )

    bd.Database("foreground").write(
        {
            ("foreground", "P1"): {
                "name": "process P1",
                "location": "somewhere",
                "reference product": "F1",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("foreground", "P1"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([0, 0.5, 0.5, 0]),
                        ),
                    },
                    {
                        "amount": 27.5,
                        "type": "technosphere",
                        "input": ("db_2020", "I1"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([1, 0, 0, 0]),
                        ),
                    },
                    {
                        "amount": 20,
                        "type": "biosphere",
                        "input": ("biosphere3", "CO2"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([0, 0.5, 0.5, 0]),
                        ),
                    },
                ],
            },
            ("foreground", "P2"): {
                "name": "process P2",
                "location": "somewhere",
                "reference product": "F1",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("foreground", "P2"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([0, 0.5, 0.5, 0]),
                        ),
                    },
                    {
                        "amount": 1,
                        "type": "technosphere",
                        "input": ("db_2020", "I2"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([1, 0, 0, 0]),
                        ),
                    },
                    {
                        "amount": 20,
                        "type": "biosphere",
                        "input": ("biosphere3", "CO2"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([0, 0.5, 0.5, 0]),
                        ),
                    },
                ],
            },
        }
    )

    bd.Method(("GWP", "example")).write(
        [
            (("biosphere3", "CO2"), 1),
            (("biosphere3", "CH4"), 27),
        ]
    )


@pytest.fixture(scope="module")
def mock_lca_data_processor(setup_brightway_databases):
    td_demand = TemporalDistribution(
        date=np.arange(2020 - 1970, 10, dtype="datetime64[Y]"),
        amount=np.asarray([0, 0, 10, 5, 10, 5, 10, 5, 10, 5]),
    )

    # Define functional flows for optimex
    foreground = bd.Database("foreground")
    for act in foreground:
        act["functional flow"] = "F1"
        act.save()

    lca_data_processor = lca_processor.LCADataProcessor(
        demand={"F1": td_demand},
        start_date=datetime.strptime("2020", "%Y"),
        method=("GWP", "example"),
        database_date_dict={
            "db_2020": datetime.strptime("2020", "%Y"),
            "db_2030": datetime.strptime("2030", "%Y"),
            "foreground": "dynamic",
        },
        temporal_resolution="year",
        timehorizon=100,
    )
    return lca_data_processor
