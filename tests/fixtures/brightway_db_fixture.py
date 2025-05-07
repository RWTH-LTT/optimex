import bw2data as bd
import numpy as np
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution


@pytest.fixture(scope="module")
@bw2test
def standalone_db():
    bd.projects.set_current("__test_standalone_db__")
    bd.Database("bio").write(
        {
            ("bio", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
            },
        },
    )
    bd.Database("bio").write(
        {
            ("bio", "CH4"): {
                "type": "emission",
                "name": "methane",
            },
        },
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
                        "input": ("foreground", "F1"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([0, 0.5, 0.5, 0]),
                        ),
                    },
                    {
                        "amount": 27.5,
                        "type": "technosphere",
                        "input": ("foreground", "I1"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([1, 0, 0, 0]),
                        ),
                    },
                    {
                        "amount": 20,
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([0, 0.5, 0.5, 0]),
                        ),
                    },
                ],
            },
        }
    )
    bd.Database("foreground").write(
        {
            ("foreground", "P2"): {
                "name": "process P2",
                "location": "somewhere",
                "reference product": "F1",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("foreground", "F1"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([0, 0.5, 0.5, 0]),
                        ),
                    },
                    {
                        "amount": 1,
                        "type": "technosphere",
                        "input": ("foreground", "I2"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([0, 0, 0, 1]),
                        ),
                    },
                    {
                        "amount": 20,
                        "type": "biosphere",
                        "input": ("bio", "CO2"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array(range(4), dtype="timedelta64[Y]"),
                            amount=np.array([0, 0.5, 0.5, 0]),
                        ),
                    },
                ],
            },
        }
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
                        "input": ("bio", "CO2"),
                    },
                ],
            },
        }
    )
    bd.Database("db_2020").write(
        {
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
                        "input": ("bio", "CH4"),
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
                        "input": ("bio", "CO2"),
                    },
                ],
            },
        }
    )
    bd.Database("db_2030").write(
        {
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
                        "input": ("bio", "CH4"),
                    },
                ],
            },
        }
    )
