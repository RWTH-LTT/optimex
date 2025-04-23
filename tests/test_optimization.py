import pyomo.environ as pyo

from optimex import converter


def test_dict_converts_to_modelinputs(abstract_system_model_inputs):
    model_inputs = converter.ModelInputs(**abstract_system_model_inputs)
    assert isinstance(model_inputs, converter.ModelInputs)


def test_pyomo_model_generation(abstract_system_model):
    assert isinstance(abstract_system_model, pyo.ConcreteModel)
