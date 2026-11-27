---
icon: lucide/dices
tags:
  - optimization
  - uncertainty
---

# Uncertainty

LCA data are uncertain, and Brightway stores that uncertainty as distributions on
exchanges, parameters, and background data. `optimex.uncertainty` carries those
distributions into the optimization with Monte Carlo sampling. It draws many
complete sets of optimization inputs and then either optimizes or evaluates each
of them.

```python
from optimex import uncertainty

sampler = uncertainty.UncertaintySampler(processor, inputs)
samples = sampler.sample(200, seed=42)

optima = uncertainty.solve_samples(samples, objective_category="climate_change")
robustness = uncertainty.evaluate_design(
    model, samples, fix="pathway", objective_category="climate_change"
)
```

The [uncertainty example](examples/uncertainty_example.md) walks through the
whole workflow, first for an impact objective and then with uncertain prices and
the cost objective.

---

## What Is Sampled

`UncertaintySampler(processor, inputs, sources=...)` takes the `LCADataProcessor`
the inputs were built from and the deterministic inputs, including any limits or
discounting. Each sample is a copy of `inputs` in which only the uncertain values
differ. `sampler.summary()` lists every uncertain quantity it found.

| Source | Defined by | Drawn with |
|--------|------------|------------|
| `"exchanges"` | Uncertainty fields on foreground exchanges | `matrix_utils` on the foreground datapackage, `use_distributions=True` |
| `"parameters"` | Uncertain project, database, and activity parameters used by foreground exchange formulas | `stats_arrays`, propagated with `bw2parameters`, passed on as a `bw_processing` array datapackage |
| `"background"` | Uncertainty fields on background exchanges | A `bw2calc` Monte Carlo LCA per background database |
| `"prices"` | Uncertainty columns in [`set_market_prices`](cost_modeling.md) | `stats_arrays` quantile functions |

Brightway describes a distribution with the fields `uncertainty type` (a
`stats_arrays` id: 2 lognormal, 3 normal, 4 uniform, 5 triangular), `loc`,
`scale`, `shape`, `minimum`, and `maximum`:

```python
{"input": ("biosphere3", "CO2"), "amount": 15.0, "type": "biosphere",
 "uncertainty type": 2, "loc": math.log(15.0), "scale": math.log(1.5)}
```

Price rows take the same columns. A missing `loc` is derived from the price: its
logarithm for a lognormal distribution, the price itself otherwise:

```python
set_market_prices(
    [{"name": "electricity", "location": "DE", "year": 2030, "price": 0.1,
      "uncertainty type": 2, "scale": 0.3}],
    background_databases,
)
```

Characterization factors, temporal distributions, and demand stay
deterministic. Parameterized exchanges in background databases keep their
amounts.

### Correlation

Quantities that describe the same thing move together:

- An exchange or parameter is drawn once per sample and applies at every process
  time, in every vintage, and in every year. Uncertainty is applied as a relative
  change of the deterministic value, so vintage-specific amounts change with it,
  and an exchange with an amount of 0 cannot be varied.
- A parameter propagates through every formula that uses it.
- All background databases are sampled with the same seed (common random
  numbers). Databases with the same structure, such as the time-specific
  databases of one premise scenario, therefore draw the same quantile for the
  same exchange.
- A price takes one quantile per product and sample, shared by all background
  databases: a product that is expensive in 2030 is also expensive in 2050.

Other correlations, e.g. between a fuel input and its CO2 emission, can be
pre-sampled and layered on top as `bw_processing` datapackages, see
[Pre-sampled Datapackages](#pre-sampled-datapackages).

### Seeds

Every source draws from its own random stream derived from `seed`. The same seed
gives the same samples, and switching a source on or off leaves the draws of the
others unchanged. This means that a cost study with prices added shares its
environmental draws with an impact study run with the same seed.

---

## Optimizing Every Sample

`solve_samples(samples, **model_kwargs)` optimizes every sample on its own.
Keyword arguments go to `create_model`, so the objective, backend, and
`aggregate_vintages` are chosen as usual. The result, `MonteCarloResults`, holds
per sample:

| Attribute | Content |
|-----------|---------|
| `objective` | Objective value in real units |
| `status` | `"optimal"`, or the solver's termination condition |
| `impacts` | Cumulative impact per category |
| `costs` | Discounted total cost, when the inputs have prices |
| `installation`, `operation`, `production` | As from the PostProcessor, indexed by (Sample, Time), if requested with `outputs` |
| `extracted` | What an `extract(model)` function returned per sample |

Only the installation table is kept by default (`outputs=("installation",)`).
For large models, extracting tables can take as long as solving, so request
only what is needed, or pass a function that keeps a single number per sample:

```python
def hydrogen_from_electrolysis(model):
    production = PostProcessor(model).get_production()
    return production.filter(like="electrolysis").to_numpy().sum()

optima = uncertainty.solve_samples(
    samples, objective_category="climate_change",
    outputs=(), extract=hydrogen_from_electrolysis,
)
```

`summary()` gives the mean, standard deviation, and quantiles. A linear program
puts its solution on a vertex, so in each sample one technology usually takes
all of a cohort. How often each technology is chosen is therefore more
meaningful than the mean installation.

Limits can be added to existing samples without drawing again:

```python
budget = {"cumulative_category_impact_limits": {"climate_change": 520.0}}
with_budget = [sample.model_copy(update=budget) for sample in samples]
```

---

## Evaluating a Design

`evaluate_design(design, samples, fix=...)` fixes a design in every sample,
through deployment and operation limits, and optimizes the rest:

| `fix` | Fixed | Use it for |
|-------|-------|------------|
| `"pathway"` | Installation and operation | Uncertainty that is never observed, such as most LCA data. Demand must still be met exactly, so the production per unit must not be uncertain. |
| `"installation"` | Installation only; operation is re-optimized per sample | Uncertainty the operator sees before deciding how to run the plants, such as prices |

The design is a solved model or a dict with the DataFrames
`installation` (and `operation`) as returned by the PostProcessor. A sample in
which the design cannot meet demand or a limit is reported as infeasible.

Comparing the evaluated design with the optimum of each sample gives its
**regret**. The mean regret is the expected value of perfect information.

```python
regret = robustness.objective - optima.objective
```

---

## Pre-sampled Datapackages

Datapackages layered on top override the database values at the coordinates
they define, exactly as in `bw2calc`. Use arrays with `sequential=True` so that
column `i` is sample `i`:

```python
import bw_processing as bwp

ng = np.random.default_rng(1).triangular(200, 237, 300, size=200)
co2 = 26.6 / 237 * ng

package = bwp.create_datapackage(sequential=True)
package.add_persistent_array(
    matrix="technosphere_matrix",
    indices_array=np.array([(gas.id, plant.id)], dtype=bwp.INDICES_DTYPE),
    data_array=ng.reshape(1, -1),
    flip_array=np.array([True]),
)
package.add_persistent_array(
    matrix="biosphere_matrix",
    indices_array=np.array([(co2_flow.id, plant.id)], dtype=bwp.INDICES_DTYPE),
    data_array=co2.reshape(1, -1),
)

sampler = uncertainty.UncertaintySampler(
    processor, inputs, foreground_datapackages=[package]
)
```

`background_datapackages=` does the same for the background databases. Each
package is added to every background database; coordinates a database does not
contain are ignored.

---

## Run Time and Scope

Samples differ only in their data, not in the structure of the optimization
problem. With Gurobi or HiGHS (`pip install highspy`, free), `solve_samples` and
`evaluate_design` therefore load one solver model and, for every sample, change
only the coefficients that differ (`persistent="auto"`, the default). HiGHS
also starts each solve from the previous solution; Gurobi solves from scratch,
because its default concurrent barrier was faster on large models than a
warm-started simplex. This bypasses Pyomo and linopy, so the
`backend` option does not matter here. All samples share the scaling of the
deterministic inputs.

On top of that, the formulation itself is not rebuilt for every sample. It is
built once for the deterministic inputs and kept as sparse matrices; a sample
only rescales the rows of its uncertain exchanges and recomputes the impact and
cost weights from its background data and prices (`optimex.incremental`). At the
scale of a 52-region hydrogen model, this takes 0.25 s per sample instead of
11 s, so the solve is nearly all that remains. It applies when only
`installation` and `operation` tables are requested, without `extract` or
`keep_models`, and to samples that differ from the deterministic inputs in
exchange amounts (by one factor per exchange), background inventories,
characterization factors, prices, discounting, demand, impact limits, or
deployment and operation limits. Other samples are built in full automatically.

With other solvers, every sample is built and solved from scratch, and a warning
says so. `persistent=True` raises instead of falling back, `persistent=False`
always builds new models.

!!! note "Ties between optimal pathways"
    When several pathways are equally good, a warm start (HiGHS) can end on a
    different one than a solve from scratch: the same objective, but e.g. another split of
    installations. Shares such as "how often is a technology chosen" can then
    differ slightly. `persistent=False` reproduces the from-scratch results
    exactly.

`n_jobs` solves the samples in parallel worker processes (`-1` uses all
cores). Every worker holds its own model, so memory sets the limit for large
systems; Gurobi's threads are divided between the workers. Gurobi already uses
several cores per solve, so workers help most with single-threaded solvers such
as HiGHS's simplex. An `extract` function
must then be defined at module level, so it can be sent to the workers.

Sampling itself is cheap: the background databases are solved once per sample,
all purchased products against the same factorized matrix.

Monte Carlo describes how the optimum and a given pathway respond to the data.
For a design that directly minimizes a risk measure, such as the expected impact
plus a penalty on the worst samples, the samples are the input for a
sample-average or CVaR formulation.

??? info "Why not two-stage stochastic programming?"
    Two-stage stochastic programming assumes the uncertainty is revealed after
    the first decisions: build now, learn the true values, then adapt operation.
    Most LCA uncertainty is never revealed: nobody observes the true emission
    factor of a background process. Letting operation adapt to it would make
    the result look better than it can be. Stochastic programming fits
    uncertainty that is resolved over time, such as which demand or scenario
    materializes.

## API

See [`optimex.uncertainty`](../api/uncertainty.md).
