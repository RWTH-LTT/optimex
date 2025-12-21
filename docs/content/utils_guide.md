# Optimex Utils Guide

The `optimex.utils` module provides helper functions for common optimex-specific setup tasks. These utilities simplify the process of configuring processes, exchanges, and databases for optimex optimization.

## Overview

The utils module focuses on optimex-specific functionality:
- Setting operation time limits
- Adding temporal distributions to exchanges
- Marking exchanges as operational
- Setting up temporal metadata for databases
- Creating temporal demand patterns

## Quick Start

```python
import bw2data as bd
from optimex import utils

# Assuming you have a process already created
process = bd.get_node(database="foreground", code="my_process")

# Quick setup with all optimex settings
utils.setup_optimex_process(
    process,
    operation_time_limits=(0, 10),
)
```

## Detailed Usage

### 1. Setting Operation Time Limits

Operation time limits define when a process operates during the optimization period.

```python
from optimex import utils
import bw2data as bd

process = bd.get_node(database="foreground", code="h2_production")

# Set operation from year 0 to year 10
utils.set_operation_time_limits(process, start=0, end=10)
```

### 2. Adding Temporal Distributions to Exchanges

Temporal distributions spread the impacts of exchanges over time.

```python
from optimex import utils

# Apply to all exchanges of a process
utils.add_temporal_distribution_to_exchanges(
    process,
    start=0,
    end=10,
    steps=11,  # Optional: defaults to end - start + 1
    kind="uniform",
    resolution="Y"
)

# Or apply to specific exchanges only
tech_exchanges = list(process.technosphere())
utils.add_temporal_distribution_to_exchanges(
    tech_exchanges,
    start=0,
    end=8
)
```

### 3. Marking Exchanges as Operational

Mark exchanges to indicate they occur during the operation phase.

```python
from optimex import utils

# Mark all exchanges of a process
utils.mark_exchanges_as_operation(process)

# Or mark specific exchanges
biosphere_exchanges = list(process.biosphere())
utils.mark_exchanges_as_operation(biosphere_exchanges)
```

### 4. Complete Process Setup (Recommended)

The `setup_optimex_process` function combines all the above in one call:

```python
from optimex import utils

# Basic setup - applies operation time limits, temporal distributions, 
# and marks exchanges as operational
utils.setup_optimex_process(
    process,
    operation_time_limits=(0, 10)
)

# Advanced setup with custom temporal distribution
utils.setup_optimex_process(
    process,
    operation_time_limits=(0, 8),
    temporal_distribution_params={
        'start': 0,
        'end': 8,
        'steps': 9,
        'kind': 'uniform',
        'resolution': 'Y'
    }
)

# Setup without marking as operation
utils.setup_optimex_process(
    process,
    operation_time_limits=(0, 10),
    mark_as_operation=False
)
```

### 5. Setting Database Temporal Metadata

Set representative times for background databases.

```python
from optimex import utils
import bw2data as bd

dbs = {
    2020: bd.Database("ei311_2020"),
    2030: bd.Database("ei311_2030"),
    2040: bd.Database("ei311_2040"),
    2050: bd.Database("ei311_2050"),
}

utils.setup_database_temporal_metadata(dbs)

# With custom date (e.g., mid-year)
utils.setup_database_temporal_metadata(dbs, month=6, day=15)
```

### 6. Creating Temporal Demand

Create temporal demand patterns for functional units.

```python
from optimex import utils
import bw2data as bd

product = bd.get_node(database="foreground", code="methanol")

# Simple linear demand with trend
demand = utils.create_temporal_demand(
    product,
    years=range(2025, 2075),
    trend_start=10.0,
    trend_end=20.0
)

# Demand with noise for variability
demand = utils.create_temporal_demand(
    product,
    years=range(2025, 2075),
    trend_start=10.0,
    trend_end=20.0,
    noise_std=4.0,
    random_seed=42  # For reproducibility
)

# Custom demand amounts
import numpy as np
custom_amounts = np.array([10, 15, 20, 25, 30])
demand = utils.create_temporal_demand(
    product,
    years=range(2025, 2030),
    amounts=custom_amounts
)
```

## Complete Example

Here's a complete example showing how to use these utilities in a typical optimex workflow:

```python
import bw2data as bd
from optimex import utils

# Assuming you have background and foreground databases set up
bg_h2 = bd.get_node(
    database="ei311_2020",
    name="hydrogen production, gaseous, 30 bar, from PEM electrolysis",
    location="EUR"
)

# Create a product node
fg = bd.Database("foreground")
hydrogen = fg.new_node(
    name="hydrogen",
    code="h2",
    unit="kg",
    type=bd.labels.product_node_default,
)
hydrogen.save()

# Create a process (simplified version of productify)
h2_process = fg.new_node(
    name="hydrogen from PEM electrolysis",
    code="h2_pem",
    location="EUR",
    type=bd.labels.process_node_default,
)
h2_process.save()

# Add production edge
h2_process.new_edge(
    input=hydrogen,
    amount=1.0,
    type=bd.labels.production_edge_default,
).save()

# Copy exchanges from background
for exc in bg_h2.technosphere():
    if exc["type"] == "production":
        continue
    h2_process.new_edge(
        input=exc.input,
        amount=exc.amount,
        type=bd.labels.consumption_edge_default,
    ).save()

# Now use utils to configure the process for optimex
utils.setup_optimex_process(
    h2_process,
    operation_time_limits=(0, 8),
    temporal_distribution_params={'start': 0, 'end': 8, 'steps': 9}
)

# Set up temporal metadata for databases
dbs = {
    2020: bd.Database("ei311_2020"),
    2050: bd.Database("ei311_2050"),
}
utils.setup_database_temporal_metadata(dbs)

# Create temporal demand
demand = utils.create_temporal_demand(
    hydrogen,
    years=range(2025, 2075),
    trend_start=10.0,
    trend_end=20.0,
    noise_std=4.0,
    random_seed=25
)

# Now proceed with LCA processing and optimization as usual
```

## Function Reference

### `set_operation_time_limits(process, start, end, save=True)`
Set operation time limits for a process.

**Parameters:**
- `process`: The process node
- `start`: Start year of operation
- `end`: End year of operation
- `save`: Whether to save the process

**Returns:** The modified process node

---

### `add_temporal_distribution_to_exchanges(process_or_exchanges, start=0, end=10, steps=None, kind="uniform", resolution="Y", save=True)`
Add temporal distributions to exchanges.

**Parameters:**
- `process_or_exchanges`: Process node or list of exchanges
- `start`: Start time in years
- `end`: End time in years
- `steps`: Number of time steps (default: end - start + 1)
- `kind`: Distribution type (default: "uniform")
- `resolution`: Temporal resolution (default: "Y")
- `save`: Whether to save

**Returns:** The modified process or exchange list

---

### `mark_exchanges_as_operation(process_or_exchanges, save=True)`
Mark exchanges as operational.

**Parameters:**
- `process_or_exchanges`: Process node or list of exchanges
- `save`: Whether to save

**Returns:** The modified process or exchange list

---

### `setup_optimex_process(process, operation_time_limits, temporal_distribution_params=None, mark_as_operation=True, save=True)`
Configure a process with all optimex settings.

**Parameters:**
- `process`: The process node
- `operation_time_limits`: Tuple of (start, end) years
- `temporal_distribution_params`: Optional dict with temporal distribution parameters
- `mark_as_operation`: Whether to mark exchanges as operational
- `save`: Whether to save

**Returns:** The configured process node

---

### `setup_database_temporal_metadata(databases, month=1, day=1)`
Set representative_time metadata for databases.

**Parameters:**
- `databases`: Dict mapping years to Database objects
- `month`: Month for representative date
- `day`: Day for representative date

---

### `create_temporal_demand(product, years, amounts=None, trend_start=10.0, trend_end=20.0, noise_std=0.0, random_seed=None)`
Create temporal demand patterns.

**Parameters:**
- `product`: Product node
- `years`: Range of years
- `amounts`: Custom amounts array (optional)
- `trend_start`: Starting value for trend
- `trend_end`: Ending value for trend
- `noise_std`: Standard deviation of noise
- `random_seed`: Random seed for reproducibility

**Returns:** Dict with product as key and TemporalDistribution as value
