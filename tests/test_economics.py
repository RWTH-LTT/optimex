"""
Tests for economic helper utilities.
"""

from datetime import datetime

import bw2data as bd
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution

from optimex import converter, lca_processor, optimizer
from optimex.economics import set_market_prices


@pytest.fixture
@bw2test
def setup_economic_pipeline_databases():
    """Set up a small system with cap- and op-relevant background inputs."""
    bd.projects.set_current("__test_economic_pipeline__")

    bio_db = bd.Database("biosphere3")
    bio_db.write(
        {
            ("biosphere3", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
            },
        }
    )
    bio_db.register()

    for db_name, representative_year, co2_factor in (
        ("db_2020", 2020, 1.0),
        ("db_2030", 2030, 0.5),
    ):
        db = bd.Database(db_name)
        db.write(
            {
                (db_name, "I1"): {
                    "name": "node I1",
                    "location": "somewhere",
                    "reference product": "I1",
                    "exchanges": [
                        {
                            "amount": 1,
                            "type": "production",
                            "input": (db_name, "I1"),
                        },
                        {
                            "amount": co2_factor,
                            "type": "biosphere",
                            "input": ("biosphere3", "CO2"),
                        },
                    ],
                },
                (db_name, "I2"): {
                    "name": "node I2",
                    "location": "somewhere",
                    "reference product": "I2",
                    "exchanges": [
                        {
                            "amount": 1,
                            "type": "production",
                            "input": (db_name, "I2"),
                        },
                        {
                            "amount": co2_factor,
                            "type": "biosphere",
                            "input": ("biosphere3", "CO2"),
                        },
                    ],
                },
            }
        )
        db.metadata["representative_time"] = datetime(
            representative_year, 1, 1
        ).isoformat()
        db.register()

    foreground = bd.Database("foreground")
    foreground.write(
        {
            ("foreground", "R1"): {
                "name": "Product R1",
                "type": bd.labels.product_node_default,
                "unit": "kg",
            },
            ("foreground", "P1"): {
                "name": "process P1",
                "location": "somewhere",
                "type": bd.labels.process_node_default,
                "operation_time_limits": (0, 0),
                "exchanges": [
                    {
                        "amount": 1,
                        "type": bd.labels.production_edge_default,
                        "input": ("foreground", "R1"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([0], dtype="timedelta64[Y]"),
                            amount=np.array([1]),
                        ),
                        "operation": True,
                    },
                    {
                        "amount": 1,
                        "type": bd.labels.consumption_edge_default,
                        "input": ("db_2020", "I1"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([0], dtype="timedelta64[Y]"),
                            amount=np.array([1]),
                        ),
                    },
                    {
                        "amount": 1,
                        "type": bd.labels.consumption_edge_default,
                        "input": ("db_2020", "I2"),
                        "temporal_distribution": TemporalDistribution(
                            date=np.array([0], dtype="timedelta64[Y]"),
                            amount=np.array([1]),
                        ),
                        "operation": True,
                    },
                ],
            },
        }
    )
    foreground.register()

    bd.Method(("GWP", "example")).write([(("biosphere3", "CO2"), 1)])


def _build_economic_pipeline_processor():
    years = range(2020, 2030)
    demand = TemporalDistribution(
        date=np.array(
            [datetime(year, 1, 1).isoformat() for year in years],
            dtype="datetime64[s]",
        ),
        amount=np.ones(len(years)),
    )
    product = bd.get_node(database="foreground", code="R1")
    config = lca_processor.LCAConfig(
        demand={product: demand},
        temporal={
            "start_date": datetime(2020, 1, 1),
            "temporal_resolution": "year",
            "time_horizon": 100,
        },
        characterization_methods=[
            {
                "category_name": "climate_change",
                "brightway_method": ("GWP", "example"),
            }
        ],
        background_inventory={
            "cutoff": 1e4,
            "calculation_method": "sequential",
        },
    )
    return lca_processor.LCADataProcessor(config)


def _build_cost_expression_inputs(discount_rate=None, discount_reference_year=None):
    return converter.OptimizationModelInputs(
        PROCESS=["P"],
        PRODUCT=["R"],
        INTERMEDIATE_FLOW=["Icap", "Iop"],
        ELEMENTARY_FLOW=["CO2"],
        BACKGROUND_ID=["db_2020"],
        PROCESS_TIME=[0],
        SYSTEM_TIME=[2020, 2021],
        CATEGORY=["climate_change"],
        demand={("R", 2020): 0.0, ("R", 2021): 0.0},
        operation_flow={("P", "R"): True, ("P", "Iop"): True},
        foreground_technosphere={
            ("P", "Icap", 0): 2.0,
            ("P", "Iop", 0): 3.0,
        },
        internal_demand_technosphere={},
        foreground_biosphere={},
        foreground_production={("P", "R", 0): 1.0},
        background_inventory={},
        mapping={("db_2020", 2020): 1.0, ("db_2020", 2021): 1.0},
        characterization={("climate_change", "CO2", 2020): 1.0},
        operation_time_limits={"P": (0, 0)},
        intermediate_costs_cap={
            ("Icap", 2020): 10.0,
            ("Icap", 2021): 10.0,
        },
        intermediate_costs_op={
            ("Iop", 2020): 2.0,
            ("Iop", 2021): 2.0,
        },
        discount_rate=discount_rate,
        discount_reference_year=discount_reference_year,
    )


def _set_cost_expression_variables(model):
    model.var_installation["P", 2020].set_value(5.0)
    model.var_installation["P", 2021].set_value(0.0)
    model.var_operation["P", 2020, 2020].set_value(4.0)
    model.var_operation["P", 2021, 2021].set_value(0.0)


def test_set_market_prices_from_list_of_dicts(setup_brightway_databases):
    """Market prices can be written from list-of-dict input."""
    set_market_prices(
        price_data=[
            {
                "name": "node I1",
                "location": "somewhere",
                "year": 2020,
                "price": 42.0,
            }
        ],
        background_databases={2020: "db_2020"},
    )

    node = bd.get_node(database="db_2020", name="node I1", location="somewhere")
    assert node["market_price"] == 42.0


def test_set_market_prices_from_dataframe(setup_brightway_databases):
    """Market prices can be written from pandas DataFrame input."""
    prices = pd.DataFrame(
        [
            {
                "name": "node I2",
                "location": "somewhere",
                "year": 2030,
                "price": 84.0,
            }
        ]
    )

    set_market_prices(
        price_data=prices,
        background_databases={2030: "db_2030"},
    )

    node = bd.get_node(database="db_2030", name="node I2", location="somewhere")
    assert node["market_price"] == 84.0


def test_set_market_prices_supports_custom_column_names(setup_brightway_databases):
    """Column/key names can be mapped to user-provided tabular schemas."""
    set_market_prices(
        price_data=[
            {
                "process": "node I1",
                "region": "somewhere",
                "scenario_year": 2030,
                "eur_per_unit": 21.0,
            }
        ],
        background_databases={2030: "db_2030"},
        name_col="process",
        location_col="region",
        year_col="scenario_year",
        price_col="eur_per_unit",
    )

    node = bd.get_node(database="db_2030", name="node I1", location="somewhere")
    assert node["market_price"] == 21.0


def test_set_market_prices_supports_product_and_unit_disambiguation(
    setup_brightway_databases,
):
    """Product and unit can disambiguate nodes with same name and location."""
    db = bd.Database("db_2020")
    db.new_node(
        code="gas_kg",
        name="gas production",
        location="DE",
        **{"reference product": "natural gas, high pressure"},
        unit="kilogram",
    ).save()
    db.new_node(
        code="gas_m3",
        name="gas production",
        location="DE",
        **{"reference product": "natural gas, high pressure"},
        unit="cubic meter",
    ).save()

    with pytest.raises(ValueError, match="Found 2 results"):
        set_market_prices(
            price_data=[
                {
                    "name": "gas production",
                    "location": "DE",
                    "year": 2020,
                    "price": 0.3,
                }
            ],
            background_databases={2020: "db_2020"},
        )

    set_market_prices(
        price_data=[
            {
                "name": "gas production",
                "location": "DE",
                "product": "natural gas, high pressure",
                "unit": "cubic meter",
                "year": 2020,
                "price": 0.3,
            }
        ],
        background_databases={2020: "db_2020"},
        product_col="product",
        unit_col="unit",
    )

    node = bd.get_node(
        database="db_2020",
        name="gas production",
        location="DE",
        product="natural gas, high pressure",
        unit="cubic meter",
    )
    assert node["market_price"] == 0.3


def test_set_market_prices_respects_overwrite_false(setup_brightway_databases):
    """Existing prices are preserved when overwrite is disabled."""
    node = bd.get_node(database="db_2020", name="node I1", location="somewhere")
    node["market_price"] = 100.0
    node.save()

    set_market_prices(
        price_data=[
            {
                "name": "node I1",
                "location": "somewhere",
                "year": 2020,
                "price": 50.0,
            }
        ],
        background_databases={2020: "db_2020"},
        overwrite=False,
    )

    node = bd.get_node(database="db_2020", name="node I1", location="somewhere")
    assert node["market_price"] == 100.0


def test_set_market_prices_strict_missing_year_raises(setup_brightway_databases):
    """Missing background database years fail in strict mode."""
    with pytest.raises(ValueError, match="No background database configured"):
        set_market_prices(
            price_data=[
                {
                    "name": "node I1",
                    "location": "somewhere",
                    "year": 2040,
                    "price": 42.0,
                }
            ],
            background_databases={2020: "db_2020"},
        )


def test_set_market_prices_strict_missing_node_raises(setup_brightway_databases):
    """Missing background nodes fail in strict mode."""
    with pytest.raises(ValueError, match="Could not find background node"):
        set_market_prices(
            price_data=[
                {
                    "name": "missing node",
                    "location": "somewhere",
                    "year": 2020,
                    "price": 42.0,
                }
            ],
            background_databases={2020: "db_2020"},
        )


def test_market_prices_flow_to_model_inputs_with_mapping(
    setup_economic_pipeline_databases,
):
    """Node prices are read, interpolated, and passed to OptimizationModelInputs."""
    set_market_prices(
        price_data=[
            {
                "name": "node I1",
                "location": "somewhere",
                "year": 2020,
                "price": 100.0,
            },
            {
                "name": "node I1",
                "location": "somewhere",
                "year": 2030,
                "price": 50.0,
            },
        ],
        background_databases={2020: "db_2020", 2030: "db_2030"},
    )

    processor = _build_economic_pipeline_processor()

    assert processor.background_costs[("db_2020", "I1")] == 100.0
    assert processor.background_costs[("db_2030", "I1")] == 50.0
    assert processor.intermediate_costs_cap[("I1", 2020)] == 100.0
    assert processor.intermediate_costs_cap[("I1", 2025)] == 75.0
    assert processor.intermediate_costs_cap[("I1", 2029)] == 55.0
    assert ("I1", 2025) not in processor.intermediate_costs_op

    manager = converter.ModelInputManager()
    model_inputs = manager.parse_from_lca_processor(processor)
    assert model_inputs.intermediate_costs_cap[("I1", 2025)] == 75.0


def test_operation_edge_prices_flow_to_operation_costs(
    setup_economic_pipeline_databases,
):
    """Prices for operation-edge inputs are written to op costs, not cap costs."""
    set_market_prices(
        price_data=[
            {
                "name": "node I2",
                "location": "somewhere",
                "year": 2020,
                "price": 20.0,
            },
            {
                "name": "node I2",
                "location": "somewhere",
                "year": 2030,
                "price": 10.0,
            },
        ],
        background_databases={2020: "db_2020", 2030: "db_2030"},
    )

    processor = _build_economic_pipeline_processor()

    assert processor.intermediate_costs_op[("I2", 2025)] == 15.0
    assert ("I2", 2025) not in processor.intermediate_costs_cap

    manager = converter.ModelInputManager()
    model_inputs = manager.parse_from_lca_processor(processor)
    assert model_inputs.intermediate_costs_op[("I2", 2025)] == 15.0


@bw2test
def test_background_cost_lookup_falls_back_to_metadata_when_codes_differ():
    """Cost lookup can find equivalent premise nodes with different codes."""
    bd.projects.set_current("__test_economic_cost_metadata_lookup__")

    db_2020 = bd.Database("db_2020")
    db_2020.write(
        {
            ("db_2020", "I1"): {
                "name": "node I1",
                "location": "somewhere",
                "reference product": "I1",
                "unit": "kg",
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2020", "I1"),
                    },
                ],
            },
        }
    )
    db_2020.register()

    db_2030 = bd.Database("db_2030")
    db_2030.write(
        {
            ("db_2030", "I1_2030"): {
                "name": "node I1",
                "location": "somewhere",
                "reference product": "I1",
                "unit": "kg",
                "market_price": 50.0,
                "exchanges": [
                    {
                        "amount": 1,
                        "type": "production",
                        "input": ("db_2030", "I1_2030"),
                    },
                ],
            },
        }
    )
    db_2030.register()

    processor = object.__new__(lca_processor.LCADataProcessor)
    processor.background_dbs = {"db_2030": datetime(2030, 1, 1)}
    processor._intermediate_flows = {"I1": "node I1"}
    processor._intermediate_flow_metadata = {
        "I1": {
            "name": "node I1",
            "location": "somewhere",
            "product": "I1",
            "unit": "kg",
        }
    }
    processor._cost_relevant_cap_flows = {"I1"}
    processor._cost_relevant_op_flows = set()
    processor._background_costs = {}

    processor._construct_background_costs()

    assert processor.background_costs[("db_2030", "I1")] == 50.0


def test_missing_market_price_logs_warning(
    setup_economic_pipeline_databases,
    monkeypatch,
):
    """Missing market prices for cost-relevant flows are warned about."""
    messages = []

    def collect_warning(message, *args, **kwargs):
        messages.append(message.format(*args, **kwargs))

    monkeypatch.setattr(lca_processor.logger, "warning", collect_warning)
    set_market_prices(
        price_data=[
            {
                "name": "node I1",
                "location": "somewhere",
                "year": 2020,
                "price": 100.0,
            },
        ],
        background_databases={2020: "db_2020"},
    )

    processor = _build_economic_pipeline_processor()

    assert any(
        "Missing market_price" in message
        and "db_2030" in message
        and "I1" in message
        and "cap" in message
        for message in messages
    )
    assert processor.intermediate_costs_cap[("I1", 2025)] == 50.0


def test_cost_expressions_without_discounting():
    """Cost expressions combine real background purchases with market prices."""
    model = optimizer.create_model(
        _build_cost_expression_inputs(),
        name="cost_expression_test",
        objective_category="climate_change",
    )
    _set_cost_expression_variables(model)

    assert pyo.value(model.background_purchase_cap["Icap", 2020]) == 10.0
    assert pyo.value(model.background_purchase_op["Iop", 2020]) == 12.0
    assert pyo.value(model.cost_cap[2020]) == 100.0
    assert pyo.value(model.cost_op[2020]) == 24.0
    assert pyo.value(model.discount_factor[2020]) == 1.0
    assert pyo.value(model.total_cost) == 124.0


def test_cost_expressions_with_discounting():
    """Discount factors are applied to total costs by system year."""
    model = optimizer.create_model(
        _build_cost_expression_inputs(
            discount_rate=0.1,
            discount_reference_year=2020,
        ),
        name="discounted_cost_expression_test",
        objective_category="climate_change",
    )
    _set_cost_expression_variables(model)

    assert pyo.value(model.discount_factor[2020]) == pytest.approx(1.0)
    assert pyo.value(model.discount_factor[2021]) == pytest.approx(1 / 1.1)
    assert pyo.value(model.total_cost) == pytest.approx(124.0)


def _require_highs():
    """Skip solver-based economics tests when HiGHS is not available."""
    try:
        available = pyo.SolverFactory("highs").available(exception_flag=False)
    except Exception as exc:
        pytest.skip(f"HiGHS solver not available: {exc}")
    if not available:
        pytest.skip("HiGHS solver not available")
    return "highs"


def _build_two_route_cost_inputs(cumulative_impact_limit=None):
    data = {
        "PROCESS": ["cheap_dirty", "expensive_clean"],
        "PRODUCT": ["product"],
        "INTERMEDIATE_FLOW": ["dirty_input", "clean_input"],
        "ELEMENTARY_FLOW": ["CO2"],
        "BACKGROUND_ID": ["db_2020"],
        "PROCESS_TIME": [0],
        "SYSTEM_TIME": [2020],
        "CATEGORY": ["climate_change"],
        "operation_time_limits": {
            "cheap_dirty": (0, 0),
            "expensive_clean": (0, 0),
        },
        "demand": {("product", 2020): 1.0},
        "foreground_technosphere": {
            ("cheap_dirty", "dirty_input", 0): 1.0,
            ("expensive_clean", "clean_input", 0): 1.0,
        },
        "internal_demand_technosphere": {},
        "foreground_biosphere": {},
        "foreground_production": {
            ("cheap_dirty", "product", 0): 1.0,
            ("expensive_clean", "product", 0): 1.0,
        },
        "operation_flow": {
            ("cheap_dirty", "product"): True,
            ("cheap_dirty", "dirty_input"): True,
            ("expensive_clean", "product"): True,
            ("expensive_clean", "clean_input"): True,
        },
        "background_inventory": {
            ("db_2020", "dirty_input", "CO2"): 10.0,
            ("db_2020", "clean_input", "CO2"): 1.0,
        },
        "mapping": {("db_2020", 2020): 1.0},
        "characterization": {("climate_change", "CO2", 2020): 1.0},
        "intermediate_costs_op": {
            ("dirty_input", 2020): 1.0,
            ("clean_input", 2020): 10.0,
        },
    }
    if cumulative_impact_limit is not None:
        data["cumulative_category_impact_limits"] = {
            "climate_change": cumulative_impact_limit
        }
    return converter.OptimizationModelInputs(**data)


def _total_operation(model, process):
    return sum(
        pyo.value(model.var_operation[p, v, t])
        for (p, v, t) in model.ACTIVE_VINTAGE_TIME
        if p == process
    )


def test_cost_objective_switch_and_invalid_objective():
    """Objective selection stays backward-compatible and validates bad input."""
    inputs = _build_cost_expression_inputs()
    environmental_model = optimizer.create_model(
        inputs,
        name="default_environmental_objective_test",
        objective_category="climate_change",
    )
    cost_model = optimizer.create_model(
        inputs,
        name="cost_objective_switch_test",
        objective_category="climate_change",
        objective="cost",
    )
    _set_cost_expression_variables(cost_model)

    assert environmental_model._objective == "environmental"
    assert cost_model._objective == "cost"
    assert pyo.value(cost_model.OBJ) == pytest.approx(
        pyo.value(cost_model.total_cost)
    )

    with pytest.raises(ValueError, match="Unknown objective"):
        optimizer.create_model(
            inputs,
            name="invalid_objective_test",
            objective_category="climate_change",
            objective="not_a_real_objective",
        )


def test_cost_objective_chooses_cheaper_route():
    """Cost optimization selects the lower-price direct background purchase."""
    solver_name = _require_highs()
    model = optimizer.create_model(
        _build_two_route_cost_inputs(),
        name="least_cost_two_route_test",
        objective_category="climate_change",
        objective="cost",
    )

    solved_model, objective, results = optimizer.solve_model(
        model,
        solver_name=solver_name,
        tee=False,
    )

    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    assert _total_operation(solved_model, "cheap_dirty") == pytest.approx(1.0)
    assert _total_operation(solved_model, "expensive_clean") == pytest.approx(0.0)
    assert objective == pytest.approx(1.0)
    assert pyo.value(solved_model.total_cost) == pytest.approx(1.0)


def test_cost_objective_respects_cumulative_environmental_budget():
    """Environmental constraints remain active when minimizing total cost."""
    solver_name = _require_highs()
    model = optimizer.create_model(
        _build_two_route_cost_inputs(cumulative_impact_limit=1.0),
        name="least_cost_with_environmental_budget_test",
        objective_category="climate_change",
        objective="cost",
    )

    solved_model, objective, results = optimizer.solve_model(
        model,
        solver_name=solver_name,
        tee=False,
    )

    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    assert _total_operation(solved_model, "cheap_dirty") == pytest.approx(0.0)
    assert _total_operation(solved_model, "expensive_clean") == pytest.approx(1.0)
    assert objective == pytest.approx(10.0)
    assert pyo.value(solved_model.total_impact["climate_change"]) == pytest.approx(1.0)


def test_solve_model_denormalizes_environmental_but_not_cost_objective():
    """Cost objectives stay real while environmental objectives are denormalized."""
    solver_name = _require_highs()
    cost_model = optimizer.create_model(
        _build_two_route_cost_inputs(),
        name="cost_denormalization_test",
        objective_category="climate_change",
        objective="cost",
    )
    solved_cost_model, cost_objective, _ = optimizer.solve_model(
        cost_model,
        solver_name=solver_name,
        tee=False,
    )

    environmental_model = optimizer.create_model(
        _build_two_route_cost_inputs(),
        name="environmental_denormalization_test",
        objective_category="climate_change",
    )
    solved_environmental_model, environmental_objective, _ = optimizer.solve_model(
        environmental_model,
        solver_name=solver_name,
        tee=False,
    )

    assert cost_objective == pytest.approx(pyo.value(solved_cost_model.total_cost))
    assert cost_objective == pytest.approx(1.0)
    assert environmental_objective == pytest.approx(1.0)
    assert pyo.value(
        solved_environmental_model.total_impact["climate_change"]
    ) == pytest.approx(1.0)
