"""
Tests for the cost objective: writing market prices to background nodes, reading
them in the LCA processor, and the cost components of the optimization model.
"""

from datetime import datetime

import bw2data as bd
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution

from optimex import converter, lca_processor, optimizer, postprocessing
from optimex.economics import set_market_prices

SOLVER = "glpk"


def _td(years, amounts=None):
    amounts = np.ones(len(years)) if amounts is None else np.asarray(amounts)
    return TemporalDistribution(
        date=np.array(years, dtype="timedelta64[Y]"), amount=amounts
    )


# --------------------------------------------------------------------------
# Brightway pipeline fixture
# --------------------------------------------------------------------------


@pytest.fixture
@bw2test
def economic_pipeline_databases():
    """
    One process buying I1 at installation and I2 during operation.

    The 2030 database uses different codes than the 2020 one, like premise does,
    so prices must be resolved by node identity rather than by code.
    """
    bd.projects.set_current("__test_economic_pipeline__")

    bio_db = bd.Database("biosphere3")
    bio_db.write(
        {("biosphere3", "CO2"): {"type": "emission", "name": "carbon dioxide"}}
    )
    bio_db.register()

    for db_name, year, suffix in (("db_2020", 2020, ""), ("db_2030", 2030, "_2030")):
        db = bd.Database(db_name)
        db.write(
            {
                (db_name, f"{flow}{suffix}"): {
                    "name": f"node {flow}",
                    "location": "somewhere",
                    "reference product": flow,
                    "exchanges": [
                        {
                            "amount": 1,
                            "type": "production",
                            "input": (db_name, f"{flow}{suffix}"),
                        },
                        {
                            "amount": 1,
                            "type": "biosphere",
                            "input": ("biosphere3", "CO2"),
                        },
                    ],
                }
                for flow in ("I1", "I2")
            }
        )
        db.metadata["representative_time"] = datetime(year, 1, 1).isoformat()
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
                        "temporal_distribution": _td([0]),
                        "operation": True,
                    },
                    {
                        "amount": 1,
                        "type": bd.labels.consumption_edge_default,
                        "input": ("db_2020", "I1"),
                        "temporal_distribution": _td([0]),
                    },
                    {
                        "amount": 1,
                        "type": bd.labels.consumption_edge_default,
                        "input": ("db_2020", "I2"),
                        "temporal_distribution": _td([0]),
                        "operation": True,
                    },
                ],
            },
        }
    )
    foreground.register()

    bd.Method(("GWP", "example")).write([(("biosphere3", "CO2"), 1)])


def _price_rows(name, prices_by_year):
    return [
        {"name": name, "location": "somewhere", "year": year, "price": price}
        for year, price in prices_by_year.items()
    ]


def _build_processor(price_attribute="market_price"):
    years = range(2020, 2030)
    demand = TemporalDistribution(
        date=np.array(
            [datetime(year, 1, 1).isoformat() for year in years],
            dtype="datetime64[s]",
        ),
        amount=np.ones(len(years)),
    )
    config = lca_processor.LCAConfig(
        demand={bd.get_node(database="foreground", code="R1"): demand},
        temporal={"start_date": datetime(2020, 1, 1), "temporal_resolution": "year"},
        characterization_methods=[
            {"category_name": "climate_change", "brightway_method": ("GWP", "example")}
        ],
        background_inventory={
            "calculation_method": "sequential",
            "use_disk_cache": False,
        },
        price_attribute=price_attribute,
    )
    return lca_processor.LCADataProcessor(config)


BACKGROUND_DATABASES = {2020: "db_2020", 2030: "db_2030"}


# --------------------------------------------------------------------------
# set_market_prices
# --------------------------------------------------------------------------


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
    with pytest.raises(ValueError, match="no background database configured"):
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
    with pytest.raises(ValueError, match="could not find background node"):
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


def test_set_market_prices_rejects_non_numeric_prices(setup_brightway_databases):
    """Missing prices in a DataFrame (NaN) are not written to the node."""
    prices = pd.DataFrame(
        [{"name": "node I1", "location": "somewhere", "year": 2020, "price": None}]
    )
    with pytest.raises(ValueError, match="not a finite number"):
        set_market_prices(prices, background_databases={2020: "db_2020"})


# --------------------------------------------------------------------------
# Reading prices in the LCA processor
# --------------------------------------------------------------------------


def test_prices_are_interpolated_and_passed_to_model_inputs(
    economic_pipeline_databases,
):
    """Prices are resolved across databases by identity and interpolated in time."""
    set_market_prices(
        _price_rows("node I1", {2020: 100.0, 2030: 50.0})
        + _price_rows("node I2", {2020: 20.0, 2030: 10.0}),
        background_databases=BACKGROUND_DATABASES,
    )

    processor = _build_processor()
    prices = processor.intermediate_flow_prices
    # Installation- and operation-driven flows are priced the same way.
    assert prices[("I1", 2020)] == pytest.approx(100.0)
    assert prices[("I1", 2025)] == pytest.approx(75.0)
    assert prices[("I1", 2029)] == pytest.approx(55.0)
    assert prices[("I2", 2025)] == pytest.approx(15.0)

    model_inputs = converter.ModelInputManager().parse_from_lca_processor(processor)
    assert model_inputs.intermediate_flow_prices == prices


def test_prices_are_not_read_without_price_attribute(economic_pipeline_databases):
    """Reading prices is opt-in, so impact-only users get no warnings or lookups."""
    set_market_prices(
        _price_rows("node I1", {2020: 100.0, 2030: 50.0}),
        background_databases=BACKGROUND_DATABASES,
    )

    processor = _build_processor(price_attribute=None)

    assert processor.intermediate_flow_prices == {}
    model_inputs = converter.ModelInputManager().parse_from_lca_processor(processor)
    assert model_inputs.intermediate_flow_prices is None


def test_unpriced_flows_are_reported_once(economic_pipeline_databases, monkeypatch):
    """Flows without any price are left out and reported in a single warning."""
    messages = []
    monkeypatch.setattr(lca_processor.logger, "warning", messages.append)
    set_market_prices(
        _price_rows("node I1", {2020: 100.0, 2030: 50.0}),
        background_databases=BACKGROUND_DATABASES,
    )

    processor = _build_processor()

    assert {flow for flow, _ in processor.intermediate_flow_prices} == {"I1"}
    price_warnings = [m for m in messages if "market_price" in m]
    assert len(price_warnings) == 1
    assert "node I2" in price_warnings[0]


def test_partially_priced_flow_raises(economic_pipeline_databases):
    """A price missing in one database would bias the interpolation, so it fails."""
    set_market_prices(
        _price_rows("node I1", {2020: 100.0}),
        background_databases=BACKGROUND_DATABASES,
    )

    with pytest.raises(ValueError, match=r"node I1.*db_2030"):
        _build_processor()


# --------------------------------------------------------------------------
# Cost components of the optimization model
# --------------------------------------------------------------------------


def _single_process_inputs(**overrides):
    """
    Process P, operating at tau 1 only. Icap is bought at installation (tau 0);
    Iop is an operation flow that is also bought at tau 0, outside the operation
    phase, so part of it is installation-driven.
    """
    data = dict(
        PROCESS=["P"],
        PRODUCT=["R"],
        INTERMEDIATE_FLOW=["Icap", "Iop"],
        ELEMENTARY_FLOW=["CO2"],
        BACKGROUND_ID=["db"],
        PROCESS_TIME=[0, 1],
        SYSTEM_TIME=[2020, 2021, 2022],
        CATEGORY=["climate_change"],
        demand={("R", 2021): 1.0},
        operation_flow={("P", "R"): True, ("P", "Iop"): True},
        foreground_technosphere={
            ("P", "Icap", 0): 2.0,
            ("P", "Iop", 0): 3.0,
            ("P", "Iop", 1): 3.0,
        },
        internal_demand_technosphere={},
        foreground_biosphere={},
        foreground_production={("P", "R", 1): 1.0},
        background_inventory={("db", "Icap", "CO2"): 1.0, ("db", "Iop", "CO2"): 1.0},
        mapping={("db", t): 1.0 for t in (2020, 2021, 2022)},
        characterization={
            ("climate_change", "CO2", t): 1.0 for t in (2020, 2021, 2022)
        },
        operation_time_limits={"P": (1, 1)},
        intermediate_flow_prices={
            **{("Icap", t): 10.0 for t in (2020, 2021, 2022)},
            **{("Iop", t): 2.0 for t in (2020, 2021, 2022)},
        },
    )
    data.update(overrides)
    return converter.OptimizationModelInputs(**data)


def _set_variables(model, installed=1.0, running=1.0):
    for t in model.SYSTEM_TIME:
        model.var_installation["P", t].set_value(installed if t == 2020 else 0.0)
    for idx in model.var_operation:
        model.var_operation[idx].set_value(running if idx == ("P", 2020, 2021) else 0.0)


def _real(model, value):
    return value * model.scales["foreground"] * model.scales["cost"]


def test_cost_components_split_by_driver():
    """Purchases are priced by driver, including operation flows outside the window."""
    model = optimizer.create_model(
        _single_process_inputs(), name="split", objective="cost"
    )
    _set_variables(model, installed=4.0, running=1.0)

    # 2020: 4 units installed -> 4 * (2 Icap * 10 + 3 Iop * 2) = 104
    assert _real(model, pyo.value(model.installation_cost["P", 2020])) == pytest.approx(
        104.0
    )
    assert _real(model, pyo.value(model.operation_cost["P", 2020])) == pytest.approx(
        0.0
    )
    # 2021: 1 unit running -> 3 Iop * 2 = 6
    assert _real(model, pyo.value(model.operation_cost["P", 2021])) == pytest.approx(
        6.0
    )
    assert _real(model, pyo.value(model.installation_cost["P", 2021])) == pytest.approx(
        0.0
    )
    assert _real(model, pyo.value(model.total_cost)) == pytest.approx(110.0)


def test_discounting_applies_in_purchase_year():
    """Purchases are discounted from the year they occur, not the installation year."""
    model = optimizer.create_model(
        _single_process_inputs(discount_rate=0.1, discount_reference_year=2020),
        name="discounted",
        objective="cost",
    )
    _set_variables(model, installed=4.0, running=1.0)

    assert pyo.value(model.discount_factor[2020]) == pytest.approx(1.0)
    assert pyo.value(model.discount_factor[2021]) == pytest.approx(1 / 1.1)
    assert _real(model, pyo.value(model.total_cost)) == pytest.approx(104.0 + 6.0 / 1.1)


def test_objective_validation():
    inputs = _single_process_inputs()
    with pytest.raises(ValueError, match="Unknown objective"):
        optimizer.create_model(inputs, name="bad", objective="profit")
    with pytest.raises(ValueError, match="objective_category"):
        optimizer.create_model(inputs, name="no_category")
    with pytest.raises(ValueError, match="intermediate_flow_prices"):
        optimizer.create_model(
            _single_process_inputs(intermediate_flow_prices=None),
            name="no_prices",
            objective="cost",
        )


def test_inputs_reject_negative_discount_rate_and_unknown_flows():
    with pytest.raises(ValueError):
        _single_process_inputs(discount_rate=-0.01)
    with pytest.raises(ValueError, match="intermediate_flow_prices"):
        _single_process_inputs(intermediate_flow_prices={("unknown", 2020): 1.0})


def test_prices_are_scaled_for_the_solver():
    """Large prices are normalized; scaled prices stay within [-1, 1]."""
    inputs = _single_process_inputs(
        intermediate_flow_prices={("Icap", 2020): 4e6, ("Iop", 2021): 1e3}
    )
    scaled, scales = inputs.get_scaled_copy()
    assert scales["cost"] == 4e6
    assert max(scaled.intermediate_flow_prices.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Solving with the cost objective
# --------------------------------------------------------------------------


def _two_route_inputs(**overrides):
    data = dict(
        PROCESS=["cheap_dirty", "expensive_clean"],
        PRODUCT=["product"],
        INTERMEDIATE_FLOW=["dirty_input", "clean_input"],
        ELEMENTARY_FLOW=["CO2"],
        BACKGROUND_ID=["db"],
        PROCESS_TIME=[0],
        SYSTEM_TIME=[2020],
        CATEGORY=["climate_change"],
        operation_time_limits={"cheap_dirty": (0, 0), "expensive_clean": (0, 0)},
        demand={("product", 2020): 1.0},
        foreground_technosphere={
            ("cheap_dirty", "dirty_input", 0): 1.0,
            ("expensive_clean", "clean_input", 0): 1.0,
        },
        internal_demand_technosphere={},
        foreground_biosphere={},
        foreground_production={
            ("cheap_dirty", "product", 0): 1.0,
            ("expensive_clean", "product", 0): 1.0,
        },
        operation_flow={
            ("cheap_dirty", "product"): True,
            ("cheap_dirty", "dirty_input"): True,
            ("expensive_clean", "product"): True,
            ("expensive_clean", "clean_input"): True,
        },
        background_inventory={
            ("db", "dirty_input", "CO2"): 10.0,
            ("db", "clean_input", "CO2"): 1.0,
        },
        mapping={("db", 2020): 1.0},
        characterization={("climate_change", "CO2", 2020): 1.0},
        intermediate_flow_prices={
            ("dirty_input", 2020): 1e5,
            ("clean_input", 2020): 1e6,
        },
    )
    data.update(overrides)
    return converter.OptimizationModelInputs(**data)


def _total_operation(model, process):
    return sum(
        pyo.value(model.var_operation[p, v, t])
        for (p, v, t) in model.ACTIVE_VINTAGE_TIME
        if p == process
    )


def test_cost_objective_chooses_cheaper_route():
    model = optimizer.create_model(
        _two_route_inputs(), name="least_cost", objective="cost"
    )
    model, objective, _ = optimizer.solve_model(model, solver_name=SOLVER, tee=False)

    assert _total_operation(model, "cheap_dirty") == pytest.approx(1.0)
    assert _total_operation(model, "expensive_clean") == pytest.approx(0.0)
    assert objective == pytest.approx(1e5)

    costs = postprocessing.PostProcessor(model).get_costs(discounted=True)
    assert costs.to_numpy().sum() == pytest.approx(objective)
    assert costs[("operation", "cheap_dirty")].loc[2020] == pytest.approx(1e5)


def test_cost_objective_respects_impact_limit():
    model = optimizer.create_model(
        _two_route_inputs(cumulative_category_impact_limits={"climate_change": 1.0}),
        name="least_cost_with_budget",
        objective="cost",
    )
    model, objective, _ = optimizer.solve_model(model, solver_name=SOLVER, tee=False)

    assert _total_operation(model, "expensive_clean") == pytest.approx(1.0)
    assert objective == pytest.approx(1e6)
    impacts = postprocessing.PostProcessor(model).get_impacts()
    assert impacts["climate_change"].to_numpy().sum() == pytest.approx(1.0)


def test_costs_of_environmental_optimum_are_reported():
    """The cost of an impact-minimal pathway can be read from the same model."""
    model = optimizer.create_model(
        _two_route_inputs(), name="impact_minimum", objective_category="climate_change"
    )
    model, objective, _ = optimizer.solve_model(model, solver_name=SOLVER, tee=False)

    assert objective == pytest.approx(1.0)
    costs = postprocessing.PostProcessor(model).get_costs()
    assert costs.to_numpy().sum() == pytest.approx(1e6)


def test_get_costs_without_prices_is_empty():
    model = optimizer.create_model(
        _two_route_inputs(intermediate_flow_prices=None),
        name="no_prices",
        objective_category="climate_change",
    )
    model, _, _ = optimizer.solve_model(model, solver_name=SOLVER, tee=False)
    costs = postprocessing.PostProcessor(model).get_costs()
    assert costs.empty
    assert list(costs.index) == [2020]


def test_extend_demand_extends_year_indexed_inputs():
    manager = converter.ModelInputManager()
    manager.model_inputs = _single_process_inputs()

    extended = manager.extend_demand(2)

    assert list(extended.SYSTEM_TIME) == [2020, 2021, 2022, 2023, 2024]
    # Demand of the last year (none in this fixture) is repeated
    assert extended.demand.get(("R", 2024), 0) == extended.demand.get(("R", 2022), 0)
    assert extended.mapping[("db", 2024)] == 1.0
    assert extended.characterization[("climate_change", "CO2", 2024)] == 1.0
    assert extended.intermediate_flow_prices[("Iop", 2024)] == 2.0
    optimizer.create_model(extended, name="extended", objective="cost")
