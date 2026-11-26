---
icon: lucide/cpu
tags:
  - lca processing
  - setup
---

# LCA Processing

With the [background data](background_data.md) and [foreground](foreground_modeling.md) in place, the second step of the [workflow](workflow.md) turns them into optimization inputs: define the time-specific demand $\mathbf{f}_t$, configure characterization methods (static or [dynamic](theory_scope_and_limits.md#dynamic-characterization)), process the LCA data into time-explicit tensors, and convert them into `OptimizationModelInputs`.

---

## Overview

```
Demand + LCAConfig → LCADataProcessor → ModelInputManager → OptimizationModelInputs
```

This is the slow step of the workflow, so the resulting inputs can be [saved](#saving-and-loading-model-inputs) and reused for any number of optimizations.

---

## Defining Demand

Demand specifies **how much** of each product is needed **when**. Use `TemporalDistribution` from `bw_temporalis`:

```python
from datetime import datetime
import numpy as np
from bw_temporalis import TemporalDistribution
import bw2data as bd

# Get the product node from your foreground database
product = bd.get_node(database="foreground", name="hydrogen")

# Define demand over time
years = range(2020, 2040)
demand_values = [0, 0, 100, 100, 150, 150, 200, 200, 250, 250,
                 300, 300, 350, 350, 400, 400, 450, 450, 500, 500]

td_demand = TemporalDistribution(
    date=np.array([datetime(y, 1, 1).isoformat() for y in years], dtype='datetime64[s]'),
    amount=np.array(demand_values, dtype=float),
)

demand = {product: td_demand}
```

### Demand Patterns

**Constant demand:**
```python
amount=np.array([100] * 20)  # 100 units every year
```

**Growing demand:**
```python
amount=np.array([100 + i*10 for i in range(20)])  # Linear growth
```

**Step change:**
```python
amount=np.array([100]*10 + [200]*10)  # Double after 10 years
```

**Delayed start:**
```python
amount=np.array([0, 0, 0, 100, 100, 100, ...])  # Start in year 4
```

---

## LCA Configuration

The `LCAConfig` object defines temporal settings and characterization methods:

```python
from optimex import lca_processor

config = lca_processor.LCAConfig(
    demand=demand,
    temporal={
        "start_date": datetime(2020, 1, 1),
        "temporal_resolution": "year",
        "time_horizon": 100,
    },
    characterization_methods=[
        {
            "category_name": "climate_change",
            "brightway_method": ("IPCC 2021", "climate change", "GWP 100a"),
        },
    ],
)
```

Further options:

| Parameter | Description |
|-----------|-------------|
| `background_inventory` | How background inventories are calculated, see [Background Inventory](#background-inventory) |
| `foreground_db_name` | Name of the foreground database (default `"foreground"`) |
| `price_attribute` | Background node attribute holding market prices, needed for the [cost objective](minimizing_costs.md); see [Market Prices](#market-prices) (default `None`: no prices are read) |

### Temporal Settings

| Parameter | Description | Example |
|-----------|-------------|---------|
| `start_date` | Beginning of the optimization horizon | `datetime(2020, 1, 1)` |
| `temporal_resolution` | Time step granularity | `"year"`, `"month"`, `"day"` |
| `time_horizon` | Years for impact accumulation | `100` (for GWP100) |

### Characterization Methods

Each method requires:

| Field | Required | Description |
|-------|----------|-------------|
| `category_name` | Yes | Your name for this impact category |
| `brightway_method` | Yes | Tuple identifying the Brightway method |
| `metric` | No | Dynamic characterization: `"CRF"` or `"GWP"` |

**Static characterization** (default):
```python
{
    "category_name": "land_use",
    "brightway_method": ("ReCiPe", "land use"),
}
```

**Dynamic characterization** (for climate change):
```python
{
    "category_name": "climate_change",
    "brightway_method": ("IPCC 2021", "GWP 100a"),
    "metric": "CRF",  # Cumulative Radiative Forcing
}
```

!!! info "Dynamic Metrics"
    - **CRF** (Cumulative Radiative Forcing): Integrates radiative forcing over time
    - **GWP** (Global Warming Potential): Time-dependent GWP factors

    Dynamic metrics account for when emissions occur, not just how much.

### Multiple Impact Categories

```python
characterization_methods=[
    {
        "category_name": "climate_change",
        "brightway_method": ("IPCC 2021", "GWP 100a"),
        "metric": "CRF",
    },
    {
        "category_name": "water_use",
        "brightway_method": ("ReCiPe", "water consumption"),
    },
    {
        "category_name": "land_use",
        "brightway_method": ("ReCiPe", "land use"),
    },
]
```

### Background Inventory

`background_inventory` controls how the inventories of the background databases are
calculated:

```python
config = lca_processor.LCAConfig(
    ...,
    background_inventory={
        "calculation_method": "parallel",  # default; one process per background database
        "retain_flows": [iridium["code"]],  # keep flows that carry no impact
    },
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `calculation_method` | `"parallel"` | `"parallel"` calculates each background database in its own process, `"sequential"` one after another |
| `n_jobs` | `None` | Worker processes for `"parallel"`; defaults to one per background database, capped by the CPU count |
| `restrict_to_characterized_flows` | `True` | Drop elementary flows that have no characterization factor in any configured category |
| `retain_flows` | `[]` | Elementary flow codes to keep even when they carry no characterization factor |
| `cutoff` | `None` | Optional number of largest elementary flows to keep per intermediate flow. `None` keeps all |
| `use_disk_cache` | `True` | Keep calculated inventories in a cache inside the Brightway project, so a new session does not rebuild them |
| `disk_cache_dir` | `None` | Where that cache lives; defaults to `optimex-inventory-cache` in the current project |
| `path_to_save` / `path_to_load` | `None` | Save or load the whole inventory tensor as a pickle |

!!! warning "Uncharacterized flows are dropped"
    By default, an elementary flow that has **no characterization factor in any of your
    configured categories** is removed from the model. Such a flow cannot affect any
    impact, and keeping it would only inflate the model (the inventory is expressed per
    process, elementary flow and year).

    This matters when you constrain a flow directly, e.g. a resource limit through
    [`cumulative_flow_limits_max`](constraints.md). Iridium has no climate change or
    water use factor, so it is dropped unless you ask for it:

    ```python
    background_inventory={"retain_flows": [iridium["code"]]}
    ```

    Set `restrict_to_characterized_flows=False` to keep every flow instead. The
    calculation log reports how many flows were dropped.

!!! note "Parallel calculation from a script"
    Worker processes are spawned, so a **script** using the default
    `calculation_method="parallel"` must guard its entry point:

    ```python
    if __name__ == "__main__":
        main()
    ```

    Notebooks need no guard. Use `"sequential"` if a guard is not an option.

### Market Prices

For the [cost objective](minimizing_costs.md), set `price_attribute` to the node attribute holding the prices written while [setting up the background data](background_data.md#market-prices):

```python
config = lca_processor.LCAConfig(
    ...,
    price_attribute="market_price",  # attribute written by set_market_prices
)
```

The processor then resolves each directly purchased background product in every background database, the same way as for the inventories, and interpolates its prices to every system year with the same weights: linearly between database years, with the nearest database used outside that range. The result is available as `lca_data.intermediate_flow_prices`, keyed by `(intermediate flow, year)`, and is passed on to the optimization inputs.

Missing prices are handled conservatively:

- A product without a price in **any** database is left out and treated as free. All such products are listed in a single warning: check it, since a free purchase is rarely intended.
- A product priced in **some** databases but not in others raises a `ValueError`, because interpolating with the missing price would silently pull the price towards zero.

---

## Processing LCA Data

The `LCADataProcessor` extracts all necessary data from your Brightway databases:

```python
lca_data = lca_processor.LCADataProcessor(config)
```

This step:

- Identifies foreground processes and their temporal distributions
- Calculates background inventories for each database
- Constructs characterization factors (static or dynamic)
- Builds interpolation weights between background databases

!!! note "Computation Time"
    Most of this step is building and factorizing the technosphere matrix of each
    background database. Results are cached per (project, database, activity), in
    memory for the running session and on disk for the next one, so only the first
    run pays for it. On the methanol & iron case study that is ~37 s for the first
    run and ~0.3 s afterwards.

    Cache entries carry the database's `modified` token, so editing a background
    database invalidates them automatically. To clear them by hand:

    ```python
    from optimex import lca_processor

    lca_processor.clear_lca_caches()                     # this session
    lca_processor.clear_lca_caches(include_disk=True)    # and the files
    ```

---

## Converting to Optimization Inputs

The `ModelInputManager` converts LCA data to optimization-ready format:

```python
from optimex import converter

manager = converter.ModelInputManager()
model_inputs = manager.parse_from_lca_processor(lca_data)
```

To change inputs afterwards, for example to add constraints, discounting, or prices set by hand, use `override()`. It validates the changed inputs and returns them:

```python
model_inputs = manager.override(
    discount_rate=0.03,
    cumulative_category_impact_limits={"climate_change": 1.0e6},
)
```

### Saving and Loading Model Inputs

Avoid reprocessing by saving:

```python
# Save to file
manager.save_inputs("model_inputs.json")  # Human-readable
manager.save_inputs("model_inputs.pkl")   # Faster, binary

# Load later
manager = converter.ModelInputManager()
manager.load_inputs("model_inputs.json")
model_inputs = manager.model_inputs
```

The model inputs are the hand-over to the optimization: [constraints](constraints.md) are added to them, and the [objective](choosing_objective.md) is chosen when the model is built from them, see [Solving](solving.md).

---

## Next Steps

- [Choosing an Objective](choosing_objective.md): Minimize impacts or costs
- [Constraints](constraints.md): Add deployment limits, impact budgets, and more
- [Solving](solving.md): Build and solve the model
