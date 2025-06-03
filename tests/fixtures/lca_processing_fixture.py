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
    bio_db = bd.Database("biosphere3")
    bio_db.write(
        {
            ("biosphere3", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
            },
            ("biosphere3", "CH4"): {
                "type": "emission",
                "name": "methane, fossil",
            },
        },
    )
    bio_db.register()

    background_2020 = bd.Database("db_2020")
    background_2020.write(
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
    background_2020.metadata["representative_time"] = datetime(2020, 1, 1).isoformat()
    background_2020.register()

    background_2030 = bd.Database("db_2030")
    background_2030.write(
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
    background_2030.metadata["representative_time"] = datetime(2030, 1, 1).isoformat()
    background_2030.register()

    foreground = bd.Database("foreground")
    foreground.write(
        {
            ("foreground", "P1"): {
                "name": "process P1",
                "location": "somewhere",
                "reference product": "R1",
                "operation_time_limits": (
                    1,
                    2,
                ),  # start and end year of operation phase
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("foreground", "P1"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([0, 0.5, 0.5, 0]),
                        ),
                        "operation": True,
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
                        "operation": True,
                    },
                ],
            },
            ("foreground", "P2"): {
                "name": "process P2",
                "location": "somewhere",
                "reference product": "R1",
                "operation_time_limits": (
                    1,
                    2,
                ),  # start and end year of operation phase
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("foreground", "P2"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([0, 0.5, 0.5, 0]),
                        ),
                        "operation": True,
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
            (("biosphere3", "CH4"), 27),
        ]
    )

    bd.Method(("land use", "example")).write(
        [
            (("biosphere3", "CO2"), 2),
            (("biosphere3", "CH4"), 1),
        ]
    )


@pytest.fixture(scope="module")
def mock_lca_data_processor(setup_brightway_databases):
    years = range(2020, 2030)
    td_demand = TemporalDistribution(
        date=np.array(
            [datetime(year, 1, 1).isoformat() for year in years], dtype="datetime64[s]"
        ),
        amount=np.asarray([0, 0, 10, 5, 10, 5, 10, 5, 10, 5]),
    )
    lca_config = lca_processor.LCAConfig(
        demand={"R1": td_demand},
        temporal={
            "start_date": datetime(2020, 1, 1),
            "temporal_resolution": "year",
            "time_horizon": 100,
        },
        characterization_methods=[
            {
                "category_name": "climate_change",
                "brightway_method": ("GWP", "example"),
                "metric": "CRF",
            },
            {
                "category_name": "land_use",
                "brightway_method": ("land use", "example"),
            },
        ],
        background_inventory={
            "cutoff": 1e4,
            "calculation_method": "sequential",
        },
    )
    lca_data_processor = lca_processor.LCADataProcessor(lca_config)
    return lca_data_processor
