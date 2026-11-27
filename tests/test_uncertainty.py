"""Monte Carlo sampling of the optimization inputs (`optimex.uncertainty`)."""

import math
from datetime import datetime

import bw2data as bd
import numpy as np
import pandas as pd
import pytest
from bw2data.tests import bw2test
from bw_temporalis import TemporalDistribution

from optimex import converter, lca_processor, optimizer, uncertainty
from optimex.economics import set_market_prices

SOLVER = "glpk"
YEARS = range(2025, 2031)
LIFETIME = 3


def _td(years, amounts):
    return TemporalDistribution(
        date=np.array(years, dtype="timedelta64[Y]"), amount=np.asarray(amounts, float)
    )


@pytest.fixture
@bw2test
def uncertain_system():
    """Two routes; uncertain exchanges, parameters, background, and prices."""
    bd.projects.set_current("__test_uncertainty__")
    bd.Database("biosphere3").write(
        {("biosphere3", "CO2"): {"name": "carbon dioxide", "type": "emission"}}
    )
    method = ("test", "climate")
    bd.Method(method).register()
    bd.Method(method).write([(("biosphere3", "CO2"), 1.0)])

    databases = {2025: "bg_2025", 2030: "bg_2030"}
    for year, db_name in databases.items():
        factor = 1.0 if year == 2025 else 0.5
        bd.Database(db_name).write(
            {
                (db_name, flow): {
                    "name": flow,
                    "reference product": flow,
                    "location": "GLO",
                    "exchanges": [
                        {"input": (db_name, flow), "amount": 1.0, "type": "production"},
                        {
                            "input": ("biosphere3", "CO2"),
                            "amount": co2 * factor,
                            "type": "biosphere",
                            "uncertainty type": 2,
                            "loc": math.log(co2 * factor),
                            "scale": 0.3,
                        },
                    ],
                }
                for flow, co2 in (("steel", 2.0), ("power", 1.0))
            }
        )
        bd.Database(db_name).metadata["representative_time"] = datetime(year, 1, 1).isoformat()
        bd.Database(db_name).register()

    operation = _td(range(LIFETIME), np.full(LIFETIME, 1 / LIFETIME))
    installation = _td([0], [1.0])
    exchanges = {
        "A": [
            ("steel", 1.0, installation, False, {}),
            ("power", 3.0, operation, True, {"formula": "power_demand"}),
            ("CO2", 6.0, operation, True, {"uncertainty type": 3, "loc": 6.0, "scale": 0.6}),
        ],
        "B": [
            ("steel", 2.0, installation, False, {}),
            ("power", 1.5, operation, True, {}),
            (
                "CO2",
                3.0,
                operation,
                True,
                {"uncertainty type": 2, "loc": math.log(3.0), "scale": math.log(1.5)},
            ),
        ],
    }
    foreground = {("foreground", "product"): {"name": "product", "type": "product"}}
    for process, rows in exchanges.items():
        data = [
            {
                "input": ("foreground", "product"),
                "amount": float(LIFETIME),
                "type": "production",
                "temporal_distribution": operation,
                "operation": True,
            }
        ]
        for flow, amount, td, is_operation, extra in rows:
            db = "biosphere3" if flow == "CO2" else "bg_2025"
            data.append(
                {
                    "input": (db, flow),
                    "amount": amount,
                    "type": "biosphere" if flow == "CO2" else "technosphere",
                    "temporal_distribution": td,
                    "operation": is_operation,
                    **extra,
                }
            )
        foreground[("foreground", process)] = {
            "name": process,
            "type": "process",
            "operation_time_limits": (0, LIFETIME - 1),
            "exchanges": data,
        }
    bd.Database("foreground").write(foreground)

    bd.parameters.new_project_parameters(
        [
            {"name": "efficiency", "amount": 0.5, "uncertainty type": 4, "minimum": 0.4, "maximum": 0.6},
            {"name": "lifetime", "amount": LIFETIME},
        ]
    )
    bd.parameters.new_activity_parameters(
        [
            {
                "name": "power_demand",
                "formula": "lifetime * 0.5 / efficiency",
                "database": "foreground",
                "code": "A",
            }
        ],
        group="A",
    )
    bd.parameters.add_exchanges_to_group("A", bd.get_node(database="foreground", code="A"))
    bd.parameters.recalculate()

    set_market_prices(
        [
            {"name": "steel", "location": "GLO", "year": year, "price": 10.0,
             "uncertainty type": 3, "scale": 1.0}
            for year in databases
        ]
        + [
            {"name": "power", "location": "GLO", "year": 2025, "price": 2.0,
             "uncertainty type": 2, "scale": 0.2},
            {"name": "power", "location": "GLO", "year": 2030, "price": 1.0,
             "uncertainty type": 2, "scale": 0.4},
        ],
        databases,
    )

    demand = TemporalDistribution(
        date=np.array([f"{y}-01-01" for y in YEARS], dtype="datetime64[s]"),
        amount=np.full(len(YEARS), 5.0),
    )
    config = lca_processor.LCAConfig(
        demand={bd.get_node(database="foreground", code="product"): demand},
        temporal={
            "start_date": datetime(2025, 1, 1),
            "temporal_resolution": "year",
            "database_dates": {name: datetime(y, 1, 1) for y, name in databases.items()},
        },
        characterization_methods=[{"category_name": "climate", "brightway_method": method}],
        background_inventory={"calculation_method": "sequential", "use_disk_cache": False},
        price_attribute="market_price",
    )
    processor = lca_processor.LCADataProcessor(config)
    inputs = converter.ModelInputManager().parse_from_lca_processor(processor)
    return processor, inputs


# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------


def test_sample_distribution_quantiles():
    normal = {"uncertainty type": 3, "loc": 2.0, "scale": 0.5}
    assert uncertainty.sample_distribution(normal, 2.0, [0.5])[0] == pytest.approx(2.0)
    lognormal = {"uncertainty type": 2, "scale": 0.3}  # loc from the amount
    assert uncertainty.sample_distribution(lognormal, 4.0, [0.5])[0] == pytest.approx(4.0)
    negative = {"uncertainty type": 2, "loc": math.log(4.0), "scale": 0.3, "negative": True}
    values = uncertainty.sample_distribution(negative, -4.0, np.linspace(0.01, 0.99, 50))
    assert (values < 0).all() and np.median(values) == pytest.approx(-4.0, rel=0.05)
    assert uncertainty.sample_distribution({}, 7.0, [0.1, 0.9]).tolist() == [7.0, 7.0]


def test_sample_distribution_truncates_at_bounds():
    data = {"uncertainty type": 3, "loc": 1.0, "scale": 1.0, "minimum": 0.5, "maximum": 1.2}
    values = uncertainty.sample_distribution(data, 1.0, np.linspace(0, 1, 101))
    assert values.min() >= 0.5 and values.max() <= 1.2
    assert values.min() == pytest.approx(0.5, abs=1e-6)
    assert values.max() == pytest.approx(1.2, abs=1e-6)
    # Truncation keeps the distribution's shape inside the bounds: no point masses
    assert len(np.unique(np.round(values, 9))) == len(values)


def test_uncertainty_from_record():
    assert uncertainty.uncertainty_from_record({"price": 3.0}, 3.0) is None
    assert uncertainty.uncertainty_from_record({"uncertainty type": 1}, 3.0) is None
    record = {"uncertainty type": 2, "scale": 0.1, "minimum": float("nan")}
    assert uncertainty.uncertainty_from_record(record, 3.0) == {
        "uncertainty type": 2,
        "loc": pytest.approx(math.log(3.0)),
        "scale": 0.1,
    }
    with pytest.raises(ValueError, match="invalid uncertainty"):
        uncertainty.uncertainty_from_record({"uncertainty type": 3}, 3.0)  # no scale


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def test_summary_lists_all_sources(uncertain_system):
    processor, inputs = uncertain_system
    summary = uncertainty.UncertaintySampler(processor, inputs).summary()
    found = set(zip(summary["source"], summary["quantity"]))
    assert ("exchanges", "A <- CO2") in found
    assert ("exchanges", "B <- CO2") in found
    assert ("parameters", "project:efficiency") in found
    assert ("parameterized exchanges", "A <- power") in found
    assert ("prices", "power @ bg_2030") in found
    assert {"bg_2025", "bg_2030"} <= set(summary.loc[summary.source == "background", "quantity"])


def test_exchange_samples_scale_every_process_time(uncertain_system):
    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 20, seed=1, sources=["exchanges"])
    for sample, amount in zip(samples, samples.values["exchanges", "B <- CO2"]):
        factor = amount / 3.0
        for tau in range(LIFETIME):
            key = ("B", "CO2", tau)
            assert sample.foreground_biosphere[key] == pytest.approx(
                inputs.foreground_biosphere[key] * factor
            )
        # Other tensors are the same objects as in the deterministic inputs
        assert sample.foreground_technosphere is inputs.foreground_technosphere
        assert sample.background_inventory is inputs.background_inventory
    assert samples.values["exchanges", "B <- CO2"].std() > 0


def test_parameters_match_brightway_recalculation(uncertain_system):
    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 5, seed=3, sources=["parameters"])
    efficiency = samples.values["parameters", "project:efficiency"]
    assert ((efficiency >= 0.4) & (efficiency <= 0.6)).all()
    expected = LIFETIME * 0.5 / efficiency
    np.testing.assert_allclose(samples.values["parameterized exchanges", "A <- power"], expected)

    # Brightway itself gives the same exchange amount for the sampled value
    value = float(efficiency.iloc[0])
    bd.parameters.new_project_parameters([{"name": "efficiency", "amount": value}])
    bd.parameters.recalculate()
    power = next(
        exc for exc in bd.get_node(database="foreground", code="A").technosphere()
        if exc.input["code"] == "power"
    )
    assert power["amount"] == pytest.approx(expected.iloc[0])
    for tau in range(LIFETIME):
        key = ("A", "power", tau)
        assert samples[0].foreground_technosphere[key] == pytest.approx(
            inputs.foreground_technosphere[key] * expected.iloc[0] / 3.0
        )


def test_background_samples_are_correlated_across_databases(uncertain_system):
    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 30, seed=5, sources=["background"])
    early = np.array([s.background_inventory["bg_2025", "power", "CO2"] for s in samples])
    late = np.array([s.background_inventory["bg_2030", "power", "CO2"] for s in samples])
    assert early.std() > 0
    np.testing.assert_allclose(late, 0.5 * early)  # same structure: same quantile
    assert np.median(early) == pytest.approx(1.0, rel=0.25)


def test_prices_share_one_quantile_and_are_interpolated(uncertain_system):
    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 30, seed=7, sources=["prices"])
    values = samples.values["prices"]
    np.testing.assert_allclose(values["steel @ bg_2025"], values["steel @ bg_2030"])
    assert values["power @ bg_2025"].rank().equals(values["power @ bg_2030"].rank())
    for sample, (_, row) in zip(samples, values.iterrows()):
        assert sample.intermediate_flow_prices["power", 2025] == pytest.approx(row["power @ bg_2025"])
        assert sample.intermediate_flow_prices["power", 2030] == pytest.approx(row["power @ bg_2030"])
        midway = 0.6 * row["power @ bg_2025"] + 0.4 * row["power @ bg_2030"]
        assert sample.intermediate_flow_prices["power", 2027] == pytest.approx(midway)


def test_sampling_is_reproducible_and_sources_are_independent(uncertain_system):
    processor, inputs = uncertain_system
    first = uncertainty.sample_inputs(processor, inputs, 10, seed=11)
    again = uncertainty.sample_inputs(processor, inputs, 10, seed=11)
    pd.testing.assert_frame_equal(first.values, again.values)
    for a, b in zip(first, again):
        assert a.background_inventory == b.background_inventory
        assert a.foreground_biosphere == b.foreground_biosphere
    exchanges_only = uncertainty.sample_inputs(processor, inputs, 10, seed=11, sources=["exchanges"])
    pd.testing.assert_frame_equal(exchanges_only.values["exchanges"], first.values["exchanges"])
    other = uncertainty.sample_inputs(processor, inputs, 10, seed=12)
    assert not first.values.equals(other.values)


def test_unknown_source(uncertain_system):
    processor, inputs = uncertain_system
    with pytest.raises(ValueError, match="Unknown uncertainty sources"):
        uncertainty.UncertaintySampler(processor, inputs, sources=["characterization"])


def test_price_uncertainty_is_removed_when_a_row_has_none(uncertain_system):
    set_market_prices(
        [{"name": "steel", "location": "GLO", "year": 2025, "price": 11.0}],
        {2025: "bg_2025"},
    )
    node = bd.get_node(database="bg_2025", name="steel")
    assert node["market_price"] == 11.0
    assert "market_price_uncertainty" not in node
    assert "market_price_uncertainty" in bd.get_node(database="bg_2030", name="steel")


# --------------------------------------------------------------------------
# Solving and evaluating
# --------------------------------------------------------------------------


def test_solve_samples_and_evaluate_design(uncertain_system):
    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 8, seed=2)
    model = optimizer.create_model(inputs, name="det", objective_category="climate")
    model, objective, _ = optimizer.solve_model(model, solver_name=SOLVER, tee=False)

    optima = uncertainty.solve_samples(samples, objective_category="climate", progress=False)
    assert optima.feasible.all()
    assert optima.objective.std() > 0
    np.testing.assert_allclose(optima.objective, optima.impacts["climate"], rtol=1e-6)
    assert optima.costs is not None and (optima.costs > 0).all()
    assert set(optima.installation.index.get_level_values("Sample")) == set(range(8))

    for fix in ("installation", "pathway"):
        evaluated = uncertainty.evaluate_design(
            model, samples, fix=fix, objective_category="climate", progress=False
        )
        assert evaluated.feasible.all()
        # No design does better than the optimum of the sample itself
        assert (evaluated.objective >= optima.objective - 1e-6).all()
        np.testing.assert_allclose(
            evaluated.installation.xs(0, level="Sample"),
            optimizer_installation(model),
            atol=1e-6,
        )

    # The design reproduces its own objective in the deterministic inputs
    same = uncertainty.evaluate_design(
        model, [inputs], fix="pathway", objective_category="climate", progress=False
    )
    assert same.objective[0] == pytest.approx(objective, rel=1e-6)

    summary = optima.summary()
    assert {"mean", "std", "q0.05", "q0.5", "q0.95"} <= set(summary.columns)


def test_infeasible_samples_are_reported(uncertain_system):
    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 3, seed=2, sources=["exchanges"])
    impossible = [
        sample.model_copy(update={"cumulative_category_impact_limits": {"climate": 1e-3}})
        for sample in samples
    ]
    results = uncertainty.solve_samples(impossible, objective_category="climate", progress=False)
    assert not results.feasible.any()
    assert results.objective.isna().all()
    assert results.summary()["feasible"].eq(0).all()


def optimizer_installation(model):
    from optimex.postprocessing import PostProcessor

    return PostProcessor(model).get_installation()


# --------------------------------------------------------------------------
# Brightway machinery: distributions and datapackages
# --------------------------------------------------------------------------


def test_exchange_draws_follow_their_distributions(uncertain_system):
    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 400, seed=4, sources=["exchanges"])
    lognormal = samples.values["exchanges", "B <- CO2"]
    assert np.median(lognormal) == pytest.approx(3.0, rel=0.08)
    assert np.log(lognormal).std() == pytest.approx(math.log(1.5), rel=0.15)
    normal = samples.values["exchanges", "A <- CO2"]
    assert normal.mean() == pytest.approx(6.0, rel=0.03)
    assert normal.std() == pytest.approx(0.6, rel=0.15)


def _presampled(matrix, row, col, values, flip=False):
    import bw_processing as bwp

    package = bwp.create_datapackage(sequential=True)
    package.add_persistent_array(
        matrix=matrix,
        indices_array=np.array([(row, col)], dtype=bwp.INDICES_DTYPE),
        data_array=np.asarray(values, float).reshape(1, -1),
        flip_array=np.array([flip]),
    )
    return package


def test_foreground_datapackages_override_exchanges(uncertain_system):
    processor, inputs = uncertain_system
    co2 = bd.get_node(database="biosphere3", code="CO2")
    process_b = bd.get_node(database="foreground", code="B")
    presampled = [2.0, 4.0, 6.0]
    package = _presampled("biosphere_matrix", co2.id, process_b.id, presampled)
    samples = uncertainty.sample_inputs(
        processor, inputs, 3, seed=1, sources=["exchanges"],
        foreground_datapackages=[package],
    )
    np.testing.assert_allclose(samples.values["exchanges", "B <- CO2"], presampled)
    for sample, value in zip(samples, presampled):
        assert sample.foreground_biosphere["B", "CO2", 0] == pytest.approx(
            inputs.foreground_biosphere["B", "CO2", 0] * value / 3.0
        )


def test_background_datapackages_override_exchanges(uncertain_system):
    processor, inputs = uncertain_system
    co2 = bd.get_node(database="biosphere3", code="CO2")
    power = bd.get_node(database="bg_2025", code="power")
    presampled = [0.1, 0.2, 0.3]
    package = _presampled("biosphere_matrix", co2.id, power.id, presampled)
    samples = uncertainty.sample_inputs(
        processor, inputs, 3, seed=1, sources=["background"],
        background_datapackages=[package],
    )
    early = [s.background_inventory["bg_2025", "power", "CO2"] for s in samples]
    np.testing.assert_allclose(early, presampled)
    late = [s.background_inventory["bg_2030", "power", "CO2"] for s in samples]
    assert np.std(late) > 0  # other databases still sample their distributions


# --------------------------------------------------------------------------
# Persistent solver models
# --------------------------------------------------------------------------


def _available(solver):
    from optimex.persistent import persistent_solver

    return persistent_solver(solver)[0] is not None


PERSISTENT_SOLVERS = [
    pytest.param(s, marks=pytest.mark.skipif(not _available(s), reason=f"{s} not available"))
    for s in ("highs", "gurobi")
]


@pytest.mark.parametrize("solver", PERSISTENT_SOLVERS)
def test_persistent_matches_fresh_models(uncertain_system, solver):
    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 12, seed=8)
    for kwargs in ({"objective_category": "climate"}, {"objective": "cost"}):
        fresh = uncertainty.solve_samples(
            samples, solver_name=solver, persistent=False, progress=False, **kwargs
        )
        warm = uncertainty.solve_samples(
            samples, solver_name=solver, persistent=True, progress=False, **kwargs
        )
        np.testing.assert_allclose(warm.objective, fresh.objective, rtol=1e-6)
        np.testing.assert_allclose(warm.impacts, fresh.impacts, rtol=1e-6)
        np.testing.assert_allclose(warm.costs, fresh.costs, rtol=1e-6)


@pytest.mark.parametrize("solver", PERSISTENT_SOLVERS)
def test_persistent_handles_rows_that_come_and_go(uncertain_system, solver):
    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 6, seed=9)
    free = optimizer.create_model(inputs, name="free", objective="cost")
    _, unconstrained, _ = optimizer.solve_model(free, solver_name=solver, tee=False)
    climate = uncertainty.solve_samples(
        samples, solver_name=solver, objective_category="climate", progress=False
    ).impacts["climate"]
    # Alternate between no budget, a binding budget, and an impossible one
    mixed = []
    for i, sample in enumerate(samples):
        if i % 3 == 1:
            sample = sample.model_copy(
                update={"cumulative_category_impact_limits": {"climate": climate[i] * 1.05}}
            )
        elif i % 3 == 2:
            sample = sample.model_copy(
                update={"cumulative_category_impact_limits": {"climate": climate[i] * 0.5}}
            )
        mixed.append(sample)
    fresh = uncertainty.solve_samples(
        mixed, solver_name=solver, persistent=False, objective="cost", progress=False
    )
    warm = uncertainty.solve_samples(
        mixed, solver_name=solver, persistent=True, objective="cost", progress=False
    )
    assert warm.feasible.tolist() == fresh.feasible.tolist()
    assert warm.feasible.tolist() == [True, True, False] * 2
    np.testing.assert_allclose(warm.objective, fresh.objective, rtol=1e-6)


@pytest.mark.parametrize("solver", PERSISTENT_SOLVERS)
def test_persistent_design_evaluation(uncertain_system, solver):
    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 8, seed=10)
    model = optimizer.create_model(inputs, name="det", objective_category="climate")
    model, _, _ = optimizer.solve_model(model, solver_name=solver, tee=False)
    for fix in ("installation", "pathway"):
        results = [
            uncertainty.evaluate_design(
                model, samples, fix=fix, solver_name=solver, persistent=persistent,
                objective_category="climate", progress=False,
            )
            for persistent in (False, True)
        ]
        np.testing.assert_allclose(results[1].objective, results[0].objective, rtol=1e-6)


def test_fallback_without_persistent_solver(uncertain_system):
    from loguru import logger

    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 2, seed=1, sources=["exchanges"])
    messages = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        results = uncertainty.solve_samples(
            samples, solver_name=SOLVER, objective_category="climate", progress=False
        )
    finally:
        logger.remove(sink)
    assert results.feasible.all()
    assert any("from scratch" in str(m) for m in messages)
    with pytest.raises(RuntimeError, match="persistent=True"):
        uncertainty.solve_samples(
            samples, solver_name=SOLVER, persistent=True,
            objective_category="climate", progress=False,
        )


# --------------------------------------------------------------------------
# Outputs and parallel runs
# --------------------------------------------------------------------------


def _advanced_share(model):
    """Module level, so worker processes can receive it."""
    from optimex.postprocessing import PostProcessor

    installation = PostProcessor(model).get_installation()
    return float(installation["B"].sum() / installation.to_numpy().sum())


@pytest.mark.parametrize("persistent", [False, "auto"])
def test_totals_match_postprocessor(uncertain_system, persistent):
    from optimex.postprocessing import PostProcessor

    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 4, seed=6)
    solver = "highs" if persistent and _available("highs") else SOLVER
    results = uncertainty.solve_samples(
        samples, solver_name=solver, persistent=persistent, keep_models=True,
        objective="cost", progress=False,
    )
    for i, model in enumerate(results.models):
        pp = PostProcessor(model)
        assert results.impacts.loc[i, "climate"] == pytest.approx(
            pp.get_impacts()["climate"].to_numpy().sum(), rel=1e-9
        )
        assert results.costs[i] == pytest.approx(
            pp.get_costs(discounted=True).to_numpy().sum(), rel=1e-9
        )
        assert results.costs[i] == pytest.approx(results.objective[i], rel=1e-6)


def test_outputs_and_extract(uncertain_system):
    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 3, seed=6, sources=["exchanges"])
    kwargs = dict(solver_name=SOLVER, persistent=False, objective_category="climate", progress=False)
    default = uncertainty.solve_samples(samples, **kwargs)
    assert not default.installation.empty
    assert default.operation.empty and default.production.empty
    assert default.extracted is None

    everything = uncertainty.solve_samples(
        samples, outputs=uncertainty.OUTPUTS, extract=_advanced_share, **kwargs
    )
    assert not everything.operation.empty and not everything.production.empty
    assert len(everything.extracted) == 3
    assert all(0 <= share <= 1 for share in everything.extracted)

    nothing = uncertainty.solve_samples(samples, outputs=(), **kwargs)
    assert nothing.installation.empty
    pd.testing.assert_series_equal(nothing.objective, default.objective)

    with pytest.raises(ValueError, match="Unknown outputs"):
        uncertainty.solve_samples(samples, outputs=("costs",), **kwargs)


@pytest.mark.parametrize(
    "solver,persistent",
    [(SOLVER, False)]
    + [pytest.param("highs", True, marks=pytest.mark.skipif(not _available("highs"), reason="highs"))],
)
def test_parallel_matches_sequential(uncertain_system, solver, persistent):
    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 6, seed=12)
    kwargs = dict(
        solver_name=solver, persistent=persistent, objective_category="climate",
        extract=_advanced_share, progress=False,
    )
    sequential = uncertainty.solve_samples(samples, **kwargs)
    parallel = uncertainty.solve_samples(samples, n_jobs=2, **kwargs)
    np.testing.assert_allclose(parallel.objective, sequential.objective, rtol=1e-6)
    pd.testing.assert_frame_equal(parallel.installation, sequential.installation, atol=1e-6)
    np.testing.assert_allclose(parallel.extracted, sequential.extracted, atol=1e-6)
    with pytest.raises(ValueError, match="keep_models"):
        uncertainty.solve_samples(samples, n_jobs=2, keep_models=True, **kwargs)


@pytest.mark.parametrize("solver", PERSISTENT_SOLVERS)
def test_incremental_path_builds_the_formulation_once(uncertain_system, solver, monkeypatch):
    from optimex import formulation, incremental

    processor, inputs = uncertain_system
    samples = uncertainty.sample_inputs(processor, inputs, 6, seed=13)
    calls = []
    original = formulation.build_formulation

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(formulation, "build_formulation", counting)
    monkeypatch.setattr(incremental, "build_formulation", counting)
    fast = uncertainty.solve_samples(
        samples, solver_name=solver, objective="cost", outputs=("installation", "operation"),
        progress=False,
    )
    assert len(calls) == 1  # the base inputs only
    calls.clear()
    full = uncertainty.solve_samples(
        samples, solver_name=solver, objective="cost", outputs=uncertainty.OUTPUTS,
        progress=False,
    )
    assert len(calls) == 1 + len(samples)  # production needs full models
    np.testing.assert_allclose(fast.objective, full.objective, rtol=1e-6)
    np.testing.assert_allclose(fast.impacts, full.impacts, rtol=1e-6)
    pd.testing.assert_frame_equal(fast.installation, full.installation, atol=1e-6)
    pd.testing.assert_frame_equal(fast.operation, full.operation, atol=1e-6)
