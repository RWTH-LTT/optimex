---
icon: lucide/pencil
tags:
  - modeling
  - supply chain
---

# Foreground Modeling

The foreground database contains the candidate processes that `optimex` will optimize — these are the processes whose capacity installation and operational dispatch are the decision variables. This guide explains the `optimex`-specific attributes you need to add to standard Brightway processes. For the underlying concepts (temporal distribution, process time vs system time, convolution), see the [Theory](theory_how_it_works.md) page.

---

## Products and Processes

The foreground has two kinds of nodes:

- **Product nodes** (`"type": "product"`): what is demanded and produced, e.g. hydrogen. Demand is always defined on a product.
- **Process nodes** (`"type": "process"`, the default): the candidate technologies that produce the products. Each has a production exchange pointing to its product.

A product can be produced by several processes, which is what the optimizer chooses between. A process that produces itself, as in Brightway's combined "process with reference product" style, is not picked up by `optimex`: model the product as its own node.

A Brightway process becomes an `optimex` process by adding:

1. **`operation_time_limits`** on the process
2. **`temporal_distribution`** on exchanges
3. **`operation`** flag on operation-phase exchanges

```python
foreground_data = {
    # Product node
    ("foreground", "my_product"): {
        "name": "My Product",
        "type": "product",
        "unit": "kg",
    },
    # Process node producing it
    ("foreground", "my_process"): {
        "name": "My Process",
        "type": "process",
        "operation_time_limits": (1, 2),  # NEW: operation phase timing
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "input": ("foreground", "my_product"),
                "temporal_distribution": TemporalDistribution(...),  # NEW
                "operation": True,  # NEW
            },
            {
                "amount": 10,
                "type": "technosphere",
                "input": ("db_2020", "electricity"),
                "temporal_distribution": TemporalDistribution(...),  # NEW
            },
        ],
    },
}
```

Exchanges without a `temporal_distribution` occur entirely at τ = 0, the installation year.

---

## Process Lifecycle Phases

`optimex` models processes with distinct lifecycle phases using **process time** τ: the number of years after the installation year of a unit. τ = 0 is the installation year itself. A unit installed in 2030 (its *vintage*) is at τ = 3 in 2033.

```
τ=0          τ=1          τ=2          τ=3
 |            |            |            |
 v            v            v            v
[Construction][  Operation  ][Decommission]
```

This guide uses the convention above: construction at τ = 0, operation from τ = 1, end-of-life after the operation phase. Operation can also start at τ = 0, e.g. for processes without a construction delay.

- **Construction (pre-operation)**: Initial investment, equipment manufacturing
- **Operation**: Active production phase where output scales with utilization
- **Decommissioning (post-operation)**: End-of-life treatment

---

## Operation Time Limits

The `operation_time_limits` attribute defines when the operation phase occurs:

```python
"operation_time_limits": (start_tau, end_tau)
```

| Example | Meaning |
|---------|---------|
| `(1, 2)` | Operation at τ=1 and τ=2 |
| `(0, 0)` | Immediate production (no construction delay) |
| `(2, 5)` | Operation from τ=2 through τ=5 |

Without `operation_time_limits`, the operation phase is `(0, 0)`: the unit operates only in its installation year. Production and other operation exchanges only count inside the operation phase, so make sure the window covers their temporal distributions.

```python
{
    ("foreground", "solar_pv"): {
        "name": "Solar PV Plant",
        "operation_time_limits": (1, 25),  # 1 year construction, 25 years operation
        ...
    }
}
```

---

## Temporal Distributions

Temporal distributions specify **when** an exchange occurs relative to process installation, using `bw_temporalis.TemporalDistribution`:

```python
from bw_temporalis import TemporalDistribution
import numpy as np

# Exchange at construction (τ=0 only)
TemporalDistribution(
    date=np.array([0], dtype="timedelta64[Y]"),
    amount=np.array([1.0]),  # 100% at τ=0
)

# Exchange spread over operation (τ=1 and τ=2)
TemporalDistribution(
    date=np.array([1, 2], dtype="timedelta64[Y]"),
    amount=np.array([0.5, 0.5]),  # 50% each year
)

# Exchange at end-of-life (τ=3)
TemporalDistribution(
    date=np.array([3], dtype="timedelta64[Y]"),
    amount=np.array([1.0]),
)
```

!!! warning "Amounts are fractions that sum to 1"
    The exchange occurs as `amount × fraction` at each process time, so the temporal distribution amounts are fractions of the exchange `amount` and should sum to 1. A production exchange with `amount=10` and fractions `[0.5, 0.5]` produces 5 in each of the two years.

---

## What One Installed Unit Means

This convention decides how installation impacts are amortized, so it is worth stating explicitly:

**One unit of a process delivers the production stated by its production exchange — over its entire lifetime, not per year.**

Take a production exchange with `amount=1`, spread as `[0, 0.5, 0.5, 0]` over τ = 0 to 3, and a two-year operation window `(1, 2)`:

| Quantity | Value |
|----------|-------|
| Lifetime output of one unit | 1 kg (the sum of the temporal distribution) |
| Output of one running unit per year | 0.5 kg (the per-τ entry) |
| Units needed to deliver 10 kg in a single year | 20 |
| Units needed to deliver 10 kg in each of two consecutive years | 20 (one cohort, fully used) |

Consequences for the optimization:

- `var_installation[p, v]` counts **units**, and installation-driven exchanges (construction, end-of-life) are incurred once per unit.
- `var_operation[p, v, t]` counts how many of those units **run in year t**, bounded by the units installed: `var_operation[p, v, t] ≤ var_installation[p, v]`.
- Annual output is `production[τ = t - v] × var_operation[p, v, t]`.

This is what makes optimex agree with a standard LCA of the delivered amount whenever every unit is fully utilized. If the demand profile does not allow full utilization — for instance a single demand year for a process with a multi-year operation window — the model builds capacity that is partly idle, and the resulting impact is legitimately *higher* than the standard LCA score of the delivered amount.

!!! tip "Reporting an annual capacity"
    Because installed units cover a whole lifetime, `PostProcessor.get_installation()` is not comparable with per-year production. Use `PostProcessor.get_production_capacity()`, which multiplies unit counts by the output per unit and year.

---

## The Operation Flag

Exchanges marked with `"operation": True` **scale with the operational level**, the number of units running in a year (`var_operation`). This enables flexible operation where a process can run below its installed capacity (`var_installation`; see [Theory: Flexible Operation](theory_how_it_works.md#step-5-flexible-operation)).

```python
"exchanges": [
    # Production scales with operation
    {
        "amount": 1,
        "type": "production",
        "input": ("foreground", "product"),
        "temporal_distribution": ...,
        "operation": True,  # Output depends on how much we operate
    },
    # Construction inputs do NOT scale with operation
    {
        "amount": 100,
        "type": "technosphere",
        "input": ("db_2020", "steel"),
        "temporal_distribution": TemporalDistribution(
            date=np.array([0], dtype="timedelta64[Y]"),
            amount=np.array([1.0]),
        ),
        # No "operation" flag - this is a fixed construction cost
    },
    # Operational emissions DO scale with operation
    {
        "amount": 5,
        "type": "biosphere",
        "input": ("biosphere3", "CO2"),
        "temporal_distribution": ...,
        "operation": True,  # Emissions depend on operation level
    },
]
```

**Rule of thumb:**
- Construction/decommissioning exchanges: No `operation` flag
- Production outputs: `"operation": True`
- Operational inputs/emissions: `"operation": True`

!!! warning "Operation exchanges must be constant over process time"
    A running unit uses the same amount in every year it runs, so an exchange with `"operation": True` must have the same amount in every process-time step it occurs in, e.g. `[0.5, 0.5]`, not `[0.7, 0.3]`. Inputs whose amount changes over the lifetime (e.g. degradation) need to be split into a constant operation part and a non-operation part. `OptimizationModelInputs` raises a `ValueError` otherwise.

??? cost "Cost objective: the flag also splits costs"
    With the [cost objective](minimizing_costs.md), the same flag decides how a background purchase is costed (see [Cost Modeling](cost_modeling.md#installation-and-operation-driven-costs)): purchases on operation exchanges within the operation phase are **operation-driven** and scale with the units running; all others are **installation-driven** and scale with the installed units, wherever they fall in the lifecycle. `PostProcessor.get_costs()` reports the two separately.

---

## Complete Process Example

A solar PV plant with 1-year construction and 2-year operation:

```python
from bw_temporalis import TemporalDistribution
import numpy as np

solar_pv = {
    ("foreground", "electricity_solar"): {
        "name": "Electricity from solar PV",
        "type": "product",
        "unit": "megawatt hour",
    },
    ("foreground", "solar_pv"): {
        "name": "Solar PV Installation",
        "location": "DE",
        "type": "process",
        "operation_time_limits": (1, 2),  # Operation at τ=1 and τ=2
        "exchanges": [
            # Production: 1 MWh total, split across operation years
            {
                "amount": 1,
                "type": "production",
                "input": ("foreground", "electricity_solar"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array([0, 1, 2, 3], dtype="timedelta64[Y]"),
                    amount=np.array([0, 0.5, 0.5, 0]),
                ),
                "operation": True,
            },
            # Construction: PV panels (at τ=0)
            {
                "amount": 0.01,
                "type": "technosphere",
                "input": ("db_2020", "pv_panel"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array([0], dtype="timedelta64[Y]"),
                    amount=np.array([1.0]),
                ),
                # No operation flag - construction is fixed
            },
            # Operational input: maintenance (during operation)
            {
                "amount": 0.001,
                "type": "technosphere",
                "input": ("db_2020", "maintenance"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array([1, 2], dtype="timedelta64[Y]"),
                    amount=np.array([0.5, 0.5]),
                ),
                "operation": True,
            },
            # End-of-life: recycling (at τ=3)
            {
                "amount": 0.01,
                "type": "technosphere",
                "input": ("db_2020", "pv_recycling"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array([3], dtype="timedelta64[Y]"),
                    amount=np.array([1.0]),
                ),
                # No operation flag - decommissioning is fixed
            },
        ],
    }
}
```

---

## Multiple Processes Producing the Same Product

`optimex` optimizes which processes to use when multiple can produce the same product:

```python
foreground_data = {
    # Product node
    ("foreground", "hydrogen"): {
        "name": "Hydrogen",
        "type": "product",
    },
    # Process 1: Green hydrogen (electrolysis)
    ("foreground", "electrolysis"): {
        "name": "PEM Electrolysis",
        "type": "process",
        "operation_time_limits": (1, 10),
        "exchanges": [
            {"type": "production", "input": ("foreground", "hydrogen"), ...},
            # Low emissions, high cost
        ],
    },
    # Process 2: Grey hydrogen (SMR)
    ("foreground", "smr"): {
        "name": "Steam Methane Reforming",
        "type": "process",
        "operation_time_limits": (2, 15),
        "exchanges": [
            {"type": "production", "input": ("foreground", "hydrogen"), ...},
            # High emissions, low cost
        ],
    },
}
```

The optimizer will choose the mix of processes that minimizes environmental impact while meeting demand.

---

## Vintage-Dependent Parameters

Real-world technologies improve over time. An EV manufactured in 2025 will have different characteristics than one manufactured in 2040. `optimex` supports **vintage-dependent parameters** to model how foreground exchanges change based on when a process is installed (see [Theory: Foreground Evolution](theory_how_it_works.md#step-4-modeling-temporal-evolution)).

### The Concept

- **Vintage** (or installation year): The system time when a process unit is built
- **Process time** (τ): The lifecycle stage of that unit (construction, operation, end-of-life)

These are independent dimensions:

| Unit | Installed (Vintage) | Operating in 2035 (τ=10) | Electricity Consumption |
|------|---------------------|--------------------------|------------------------|
| EV A | 2025 | τ=10 | 2.0 kWh/km (2025 technology) |
| EV B | 2030 | τ=5 | 1.5 kWh/km (2030 technology) |

Both are operating in 2035, but have different efficiency based on when they were built.

### Defining Vintage Parameters in the Database

Define vintage parameters directly in the Brightway database as exchange attributes. This approach keeps all process-specific data in one place.

#### Using `vintage_improvements`

For uniform scaling across all process times:

```python
from bw_temporalis import TemporalDistribution
import numpy as np

foreground_data = {
    ("foreground", "EV"): {
        "name": "Electric Vehicle",
        "operation_time_limits": (1, 2),
        "exchanges": [
            # Production exchange (no vintage variation)
            {
                "amount": 1,
                "type": "production",
                "input": ("foreground", "vkm"),
                "temporal_distribution": TemporalDistribution(...),
                "operation": True,
            },
            # Electricity consumption with vintage-dependent efficiency
            {
                "amount": 60,  # Base amount (2020 technology)
                "type": "technosphere",
                "input": ("db_2020", "electricity"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array([1, 2], dtype="timedelta64[Y]"),
                    amount=np.array([0.5, 0.5]),
                ),
                "operation": True,
                # Vintage improvement scaling factors
                "vintage_improvements": {
                    2020: 1.0,   # 100% of base (60 MJ)
                    2030: 0.75,  # 75% of base (45 MJ) - 25% improvement
                    2040: 0.6,   # 60% of base (36 MJ) - 40% improvement
                },
            },
        ],
    }
}
```

#### Using `vintage_amounts`

For explicit values at specific process times and vintages:

```python
{
    "amount": 60,  # Base amount (optional when using vintage_amounts)
    "type": "technosphere",
    "input": ("db_2020", "electricity"),
    "temporal_distribution": TemporalDistribution(
        date=np.array([1, 2], dtype="timedelta64[Y]"),
        amount=np.array([0.5, 0.5]),
    ),
    "operation": True,
    # Explicit vintage-specific values
    "vintage_amounts": {
        # Format: {vintage_year: amount} OR {(process_time, vintage_year): amount}
        (1, 2020): 30,  # τ=1, 2020 vintage: 30 MJ/vkm
        (2, 2020): 30,  # τ=2, 2020 vintage: 30 MJ/vkm
        (1, 2030): 22.5,  # τ=1, 2030 vintage: 22.5 MJ/vkm (25% improvement)
        (2, 2030): 22.5,  # τ=2, 2030 vintage: 22.5 MJ/vkm
        (1, 2040): 18,  # τ=1, 2040 vintage: 18 MJ/vkm (40% improvement)
        (2, 2040): 18,  # τ=2, 2040 vintage: 18 MJ/vkm
    },
}
```

**Key points:**
- Each exchange is interpolated linearly between the vintage years it defines itself. Installation years before the first or after the last of them use the nearest defined value. Different exchanges can use different vintage years.
- `vintage_improvements` scales the exchange `amount` by a factor per vintage.
- `vintage_amounts` replaces the amount. A vintage-year key (`{2020: 60}`) gives the exchange total for that vintage, distributed over process time by the temporal distribution, like `amount`. A `(process_time, vintage_year)` key gives the amount at that process time directly, as in the example above (30 at τ = 1 plus 30 at τ = 2 is 60 in total).
- If an exchange has both, `vintage_amounts` takes precedence for that exchange.
- Works for production, technosphere, and biosphere exchanges, and is extracted automatically by `LCADataProcessor`.

### Optimizer Behavior

With vintage-dependent parameters, the optimizer considers:

1. **Installation timing trade-offs**: Later installations are more efficient, but may have capacity constraints
2. **Mixed vintages**: Different installation cohorts operating simultaneously with different efficiencies
3. **Background evolution**: Combined with time-varying background databases for full temporal LCA

!!! tip "Sparse Implementation"
    Vintage parameters only affect processes/flows where they're specified. Processes without vintage overrides use the standard base tensors efficiently.

---

## Uncertain Data

Exchanges and Brightway parameters can carry uncertainty distributions, in the
usual Brightway fields (`uncertainty type`, `loc`, `scale`, `minimum`,
`maximum`). The optimization itself uses the deterministic amounts; a
[Monte Carlo analysis](uncertainty.md) samples the distributions and shows how
robust the optimal pathway is. Parameters with formulas are sampled as well, so
exchanges that share a parameter, such as an efficiency, move together. See
[Defining Uncertainty](uncertainty.md#defining-uncertainty) for examples.

---

## Common Patterns

### Immediate Production (No Construction Delay)

```python
"operation_time_limits": (0, 0),
"exchanges": [
    {
        "amount": 1,
        "type": "production",
        "temporal_distribution": TemporalDistribution(
            date=np.array([0], dtype="timedelta64[Y]"),
            amount=np.array([1.0]),
        ),
        "operation": True,
    },
]
```

### Long-Lived Infrastructure

```python
"operation_time_limits": (2, 30),  # 2 years construction, 30 years operation
```

### Seasonal/Variable Production

```python
# Production varies by year (e.g., degradation)
"temporal_distribution": TemporalDistribution(
    date=np.array([1, 2, 3, 4, 5], dtype="timedelta64[Y]"),
    amount=np.array([0.22, 0.21, 0.20, 0.19, 0.18]),  # Declining output
),
```

---

## Writing the Foreground Database

```python
import bw2data as bd

fg = bd.Database("foreground")
fg.write(foreground_data)
fg.register()
```

---

## Next Steps

- [LCA Processing](lca_processing.md): Define the demand and build the optimization inputs
- [Constraints](constraints.md): Add deployment limits and other constraints
