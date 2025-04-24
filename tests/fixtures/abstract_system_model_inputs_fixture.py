import pytest

from optimex import converter, optimizer


@pytest.fixture(scope="module")
def abstract_system_model_inputs():
    """Viable model inputs for optimization of transition pathway
    of an abstract system."""
    return {
        "PROCESS": ["P1", "P2"],
        "FUNCTIONAL_FLOW": ["F1"],
        "INTERMEDIATE_FLOW": ["I1", "I2"],
        "ELEMENTARY_FLOW": ["CO2", "CH4"],
        "BACKGROUND_ID": ["db_2020", "db_2030"],
        "PROCESS_TIME": [0, 1, 2, 3],
        "SYSTEM_TIME": [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029],
        # "process_phaseend_map":
        #     {
        #         ("P1", "pre-production"): 0,
        #         ("P1", "production"): 2,
        #         ("P1", "post-production"): 3,
        #         ("P2", "pre-production"): 0,
        #         ("P2", "production"): 2,
        #         ("P2", "post-production"): 3
        #     },
        "demand": {
            ("F1", 2020): 0,
            ("F1", 2021): 0,
            ("F1", 2022): 10,
            ("F1", 2023): 10,
            ("F1", 2024): 10,
            ("F1", 2025): 10,
            ("F1", 2026): 10,
            ("F1", 2027): 10,
            ("F1", 2028): 10,
            ("F1", 2029): 10,
        },
        "foreground_technosphere": {
            ("P1", "I1", 0): 27.5,
            ("P2", "I2", 0): 1,
        },
        "foreground_biosphere": {
            ("P1", "CO2", 1): 0.1,
            ("P1", "CO2", 2): 0.1,
            ("P2", "CO2", 1): 0.1,
            ("P2", "CO2", 2): 0.1,
        },
        "foreground_production": {
            ("P1", "F1", 1): 0.5,
            ("P1", "F1", 2): 0.5,
            ("P2", "F1", 1): 0.5,
            ("P2", "F1", 2): 0.5,
        },
        "background_inventory": {
            ("db_2020", "I1", "CO2"): 1,
            ("db_2020", "I2", "CH4"): 1,
            ("db_2030", "I1", "CO2"): 0.9,
            ("db_2030", "I2", "CH4"): 0.9,
        },
        "mapping": {
            ("db_2020", 2020): 1.0,
            ("db_2020", 2021): 0.9,
            ("db_2020", 2022): 0.8,
            ("db_2020", 2023): 0.7,
            ("db_2020", 2024): 0.6,
            ("db_2020", 2025): 0.5,
            ("db_2020", 2026): 0.4,
            ("db_2020", 2027): 0.3,
            ("db_2020", 2028): 0.2,
            ("db_2020", 2029): 0.1,
            ("db_2030", 2020): 0,
            ("db_2030", 2021): 0.1,
            ("db_2030", 2022): 0.2,
            ("db_2030", 2023): 0.3,
            ("db_2030", 2024): 0.4,
            ("db_2030", 2025): 0.5,
            ("db_2030", 2026): 0.6,
            ("db_2030", 2027): 0.7,
            ("db_2030", 2028): 0.8,
            ("db_2030", 2029): 0.9,
        },
        # cumulative radiative forcing for timehorizon from 2020 till 2120
        "characterization": {
            key: val * 1e13
            for key, val in {  # scaling for numerical stability
                ("CO2", 2020): 8.856378067710995e-14,
                ("CO2", 2021): 8.78632948322376e-14,
                ("CO2", 2022): 8.716115201983699e-14,
                ("CO2", 2023): 8.645732530491629e-14,
                ("CO2", 2024): 8.575178705349817e-14,
                ("CO2", 2025): 8.50445089133486e-14,
                ("CO2", 2026): 8.433546179417147e-14,
                ("CO2", 2027): 8.362461584725384e-14,
                ("CO2", 2028): 8.291194044454685e-14,
                ("CO2", 2029): 8.219740415716608e-14,
                ("CH4", 2020): 2.3651673669270527e-12,
                ("CH4", 2021): 2.3651198384711042e-12,
                ("CH4", 2022): 2.3650681065838066e-12,
                ("CH4", 2023): 2.36501179951239e-12,
                ("CH4", 2024): 2.3649505126261558e-12,
                ("CH4", 2025): 2.3648838055087402e-12,
                ("CH4", 2026): 2.364811198793221e-12,
                ("CH4", 2027): 2.3647321707173162e-12,
                ("CH4", 2028): 2.3646461533739266e-12,
                ("CH4", 2029): 2.364552528630073e-12,
            }.items()
        },
    }


@pytest.fixture(scope="module")
def abstract_system_model(abstract_system_model_inputs):
    """Fixture to generate a model from the abstract system model inputs."""
    model_inputs = converter.ModelInputs(**abstract_system_model_inputs)
    model = optimizer.create_model(
        inputs=model_inputs,
        name="abstract_system_model",
    )
    return model


@pytest.fixture(scope="module")
def solved_system_model(abstract_system_model):
    """Fixture to generate a solved model from the abstract system model."""
    return optimizer.solve_model(abstract_system_model)
