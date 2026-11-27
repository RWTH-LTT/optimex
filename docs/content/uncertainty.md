---
icon: lucide/dices
tags:
  - optimization
  - uncertainty
---

# Uncertainty

LCA data are uncertain. Brightway stores that uncertainty as probability
distributions on exchanges and parameters, and `optimex` carries them into the
optimization with Monte Carlo sampling: it draws many complete sets of
optimization inputs and optimizes, or evaluates a given pathway, for each of
them. The result is not one optimal pathway but answers to the questions that
matter for decisions: how often is a technology the best choice, how much does
the optimum vary, and how likely does a chosen pathway keep a limit?

!!! tip "Distributions you already have are used as they are"
    Uncertainty in your Brightway project needs no extra modeling for
    `optimex`. Distributions on foreground exchanges, uncertain Brightway
    parameters and the formulas that use them, and the distributions of
    background databases such as ecoinvent are all sampled. Only prices, which
    are `optimex` attributes, get their distributions through
    `set_market_prices`.

```python
from optimex import uncertainty

# `processor` and `model_inputs` as in the deterministic workflow
sampler = uncertainty.UncertaintySampler(processor, model_inputs)
sampler.summary()                      # what is uncertain
samples = sampler.sample(200, seed=42)

optima = uncertainty.solve_samples(
    samples, objective_category="climate_change", solver_name="highs"
)
optima.summary()                       # spread of the optimum
```

The [uncertainty example](examples/uncertainty_example.md) walks through the
whole workflow: defining the uncertainty, the spread of the climate optimum, the
robustness of the deterministic design, and, with uncertain prices, the cost
objective under a climate budget.

---

## Defining Uncertainty

Brightway describes a distribution with the same fields everywhere: on
exchanges, on parameters, and, in `optimex`, on prices. `uncertainty type` is a
[`stats_arrays`](https://stats-arrays.readthedocs.io) id; the other fields
depend on the distribution:

| Distribution | `uncertainty type` | Fields | Typical use |
|---|---|---|---|
| Lognormal | 2 | `loc` = ln(median), `scale` = ln(geometric standard deviation) | Emission factors and amounts, the ecoinvent default |
| Normal | 3 | `loc` = mean, `scale` = standard deviation | Measured values with symmetric error |
| Uniform | 4 | `minimum`, `maximum` | Only a range is known |
| Triangular | 5 | `loc` = mode, `minimum`, `maximum` | A best guess within a range |

`minimum` and `maximum` also truncate the other distributions. For a lognormal
distribution of a negative amount, set `"negative": True`. The deterministic
amount stays what `optimex` optimizes without uncertainty; for a lognormal
distribution it is the median, so the mean of the samples is higher.

### Exchanges

Add the fields to an exchange, in the foreground or in a background database you
build yourself:

```python
import math

{"input": ("biosphere3", "CO2"), "amount": 15.0, "type": "biosphere",
 "temporal_distribution": operation_td, "operation": True,
 "uncertainty type": 2, "loc": math.log(15.0), "scale": math.log(1.5)}
```

To add a distribution to an existing exchange:

```python
exchange["uncertainty type"] = 3
exchange["loc"], exchange["scale"] = exchange["amount"], 0.1 * exchange["amount"]
exchange.save()
```

A sampled exchange changes by the same factor at every process time and in every
vintage, including its `vintage_amounts`: the distribution describes the
exchange, the temporal distribution and vintage parameters its shape.

### Parameters

Brightway parameters describe quantities that several exchanges share, such as
an efficiency or a lifetime. Uncertain parameters propagate through every
formula that uses them, so the exchanges move together:

```python
import bw2data as bd

bd.parameters.new_project_parameters([
    {"name": "efficiency", "amount": 0.8,
     "uncertainty type": 5, "loc": 0.8, "minimum": 0.6, "maximum": 0.9},
])
bd.parameters.new_activity_parameters([
    {"name": "fuel_demand", "formula": "5 / efficiency",
     "database": "foreground", "code": "plant"},
], group="plant")

# The exchange refers to the parameter through its formula
plant = bd.get_node(database="foreground", code="plant")
fuel = next(exc for exc in plant.technosphere() if exc.input["name"] == "fuel")
fuel["formula"] = "fuel_demand"
fuel.save()
bd.parameters.add_exchanges_to_group("plant", plant)
bd.parameters.recalculate()  # writes the deterministic amounts
```

Project, database, and activity parameters are evaluated with Brightway's
scoping. Formulas are evaluated for foreground exchanges; parameterized
exchanges in background databases keep their amounts. An exchange with a
formula is sampled through its parameters, not through its own distribution.

### Background Databases

Background databases are sampled with a regular Brightway Monte Carlo LCA, so
whatever distributions they contain are used. ecoinvent provides distributions
for most of its exchanges. Prospective databases from premise keep them for the
exchanges premise leaves unchanged; exchanges it creates or rescales may carry
none, so check your databases if the background matters. All time-specific
databases are sampled with the same random numbers, so databases with the same
structure stay correlated: a sample with a high emission factor in 2030 has it
in 2040 too.

Sampling the background rebuilds and factorizes the technosphere matrix of
every database once per sample, and solves all purchased products against it.
For ecoinvent-sized databases this takes seconds per database and sample, so
200 samples of five time-specific databases mean about 1,000 factorizations.
Leave the background out (`sources=["exchanges", "parameters", "prices"]`) for a
first, quick analysis of the foreground.

### Prices

Prices are attributes of the background nodes, written with
[`set_market_prices`](cost_modeling.md#setting-market-prices). The same function
takes the uncertainty columns, one distribution per product and database year:

```python
from optimex.economics import set_market_prices

set_market_prices(
    [
        {"name": "electricity", "location": "DE", "year": 2030, "price": 0.10,
         "uncertainty type": 2, "scale": 0.2},
        {"name": "electricity", "location": "DE", "year": 2050, "price": 0.08,
         "uncertainty type": 2, "scale": 0.4},  # less certain further ahead
    ],
    background_databases,
)
```

A missing `loc` is derived from the price: its logarithm for a lognormal
distribution, the price itself otherwise (the mode of a triangular one). A
price takes one quantile per sample in all databases, so a product that is
expensive in 2030 is also expensive in 2050, with the spread of each year. The
processor must read the prices (`LCAConfig.price_attribute`), as for the
deterministic [cost objective](minimizing_costs.md).

### Checking What Is Found

`sampler.summary()` lists every uncertain quantity the sampler found, with its
distribution: foreground exchanges, parameters, the exchanges that use them,
prices per database, and the background databases it samples. A quantity
missing from the list is treated as certain.

---

## Drawing Samples

`UncertaintySampler(processor, inputs, sources=...)` takes the `LCADataProcessor`
the inputs were built from and the deterministic inputs, including any limits or
discounting. `sample(n, seed)` returns a `SampleSet`: one copy of `inputs` per
sample in which only the uncertain values differ, plus a table of the sampled
values (`samples.values`). `sources` restricts the sampling, e.g. to
`["exchanges", "parameters"]` to leave the background deterministic.

| Source | Defined by | Drawn with |
|--------|------------|------------|
| `"exchanges"` | Uncertainty fields on foreground exchanges | `matrix_utils` on the foreground datapackage, `use_distributions=True` |
| `"parameters"` | Uncertain project, database, and activity parameters used by foreground exchange formulas | `stats_arrays`, propagated with `bw2parameters`, passed on as a `bw_processing` array datapackage |
| `"background"` | Uncertainty fields on background exchanges | A `bw2calc` Monte Carlo LCA per background database |
| `"prices"` | Uncertainty columns in `set_market_prices` | `stats_arrays` quantile functions |

Characterization factors, temporal distributions, and demand stay
deterministic.

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

## How Many Samples

Monte Carlo results are estimates; their precision grows with the square root
of the number of samples. As a rule of thumb:

- **100 to 200 samples** give means, medians, and how often a technology is
  chosen to within a few percentage points.
- **500 to 1,000 samples** are needed for tails, such as a 95th percentile or a
  small probability of missing a limit.

To check, draw a second set with another seed: if the conclusions change, draw
more. The run time grows linearly with the number of samples, see
[Run Time and Scope](#run-time-and-scope).

---

## Reading the Results

- **Shares, not mean pathways.** A linear program puts each solution on a vertex,
  so in each sample one technology usually takes a whole cohort. Report how often
  each option is chosen, and when, rather than the mean installation, which can
  describe a mix that is optimal in no sample.
- **Spread of the optimum vs. robustness of a design.** `solve_samples` shows
  what is achievable if the data were known; `evaluate_design` shows what a
  pathway decided now achieves. The gap between the two, the mean regret, is
  what better data would be worth.
- **Limits under uncertainty.** A pathway that exactly uses up a budget with the
  deterministic data misses it in roughly half of the samples. To keep a limit
  with high probability, plan with a tighter one and evaluate the resulting
  design, as in the [uncertainty example](examples/uncertainty_example.md).
- **Medians and means differ.** With lognormal data, the deterministic result
  corresponds to the medians; the mean of the samples is usually higher.

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
