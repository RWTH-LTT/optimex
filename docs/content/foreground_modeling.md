---
icon: lucide/pencil
---

# Foreground Modeling

The foreground database contains the candidate processes that `optimex` will optimize — these are the processes whose capacity installation and operational dispatch are the decision variables. This guide explains the `optimex`-specific attributes you need to add to standard Brightway processes. For the underlying concepts (temporal distribution, process time vs system time, convolution), see the [Theory](theory.md) page.

---

## Standard vs optimex Processes

A standard Brightway process becomes an `optimex` process by adding:

1. **`operation_time_limits`** on the process
2. **`temporal_distribution`** on exchanges
3. **`operation`** flag on operation-phase exchanges

```python
# Standard Brightway process
{
    ("foreground", "my_process"): {
        "name": "My Process",
        "exchanges": [
            {"amount": 1, "type": "production", "input": ("foreground", "my_process")},
            {"amount": 10, "type": "technosphere", "input": ("db_2020", "electricity")},
        ],
    }
}

# optimex process (with temporal information)
{
    ("foreground", "my_process"): {
        "name": "My Process",
        "operation_time_limits": (1, 2),  # NEW: operation phase timing
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "input": ("foreground", "my_process"),
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
    }
}
```

---

## Process Lifecycle Phases

`optimex` models processes with distinct lifecycle phases using **process time** ($\delta$, also referred to as τ in code):

```
τ=0          τ=1          τ=2          τ=3
 |            |            |            |
 v            v            v            v
[Construction][  Operation  ][Decommission]
```

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

!!! warning "Amounts must sum correctly"
    For production exchanges, the amounts in the temporal distribution should sum to the total production per unit. For a process producing 1 unit total over its lifetime with `amount=1`, use fractions that sum to 1.

---

## The Operation Flag

Exchanges marked with `"operation": True` **scale with the operational level** $\mathbf{o}_{t,v}$. This enables flexible operation where a process can run below its installed capacity $\mathbf{s}_v$ (see [Theory: Flexible Operation](theory.md#step-5-flexible-operation)).

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

---

## Complete Process Example

A solar PV plant with 1-year construction and 2-year operation:

```python
from bw_temporalis import TemporalDistribution
import numpy as np

solar_pv = {
    ("foreground", "solar_pv"): {
        "name": "Solar PV Installation",
        "location": "DE",
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
        "operation_time_limits": (1, 10),
        "exchanges": [
            {"type": "production", "input": ("foreground", "hydrogen"), ...},
            # Low emissions, high cost
        ],
    },
    # Process 2: Grey hydrogen (SMR)
    ("foreground", "smr"): {
        "name": "Steam Methane Reforming",
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

Real-world technologies improve over time. An EV manufactured in 2025 will have different characteristics than one manufactured in 2040. `optimex` supports **vintage-dependent parameters** to model how foreground exchanges change based on when a process is installed (see [Theory: Foreground Evolution](theory.md#foreground-evolution-vintage-dependent-parameters)).

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
- Values are linearly interpolated for years between specified vintages
- Works for production, technosphere, and biosphere exchanges
- Extracted automatically by `LCADataProcessor`
- `vintage_improvements`: More compact when all exchanges scale uniformly
- `vintage_amounts`: More flexible when different process times need different values
- If both are specified, `vintage_amounts` takes precedence

### Optimizer Behavior

With vintage-dependent parameters, the optimizer considers:

1. **Installation timing trade-offs**: Later installations are more efficient, but may have capacity constraints
2. **Mixed vintages**: Different installation cohorts operating simultaneously with different efficiencies
3. **Background evolution**: Combined with time-varying background databases for full temporal LCA

!!! tip "Sparse Implementation"
    Vintage parameters only affect processes/flows where they're specified. Processes without vintage overrides use the standard base tensors efficiently.

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

- [Optimization Setup](optimization_setup.md): Configure demand and run the optimization
- [Constraints](constraints.md): Add deployment limits and other constraints
