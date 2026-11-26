---
icon: lucide/database
tags:
  - brightway setup
  - database
---

# Background Data

`optimex` builds on the [Brightway](https://docs.brightway.dev/) LCA framework. The first step of the [workflow](workflow.md) is a Brightway project with the required data: a biosphere database for elementary flows, one or more background databases representing supply chains at different points in time (see [Theory: Background Evolution](theory_how_it_works.md#step-4-modeling-temporal-evolution)), LCIA methods, and, for the [cost objective](minimizing_costs.md), market prices of the background products. The foreground database with the candidate processes has its own page, [Foreground Modeling](foreground_modeling.md).

---

## Project Structure

An `optimex` project requires three types of databases:

```
Brightway Project
├── biosphere3          # Elementary flows (emissions, resources)
├── Background DBs      # e.g., db_2020, db_2030 (ecoinvent variants)
└── foreground          # Your processes to optimize (must be named "foreground")
```

---

## Creating a Project

```python
import bw2data as bd

# Create or switch to a project
bd.projects.set_current("my_optimex_project")
```

---

## Biosphere Database

The biosphere database contains elementary flows (emissions to air/water/soil, resource extractions). If you're using ecoinvent, this is imported automatically. For custom setups:

```python
biosphere_data = {
    ("biosphere3", "CO2"): {
        "type": "emission",
        "name": "carbon dioxide",
        "categories": ("air",),
        "CAS number": "000124-38-9",  # Helps with dynamic characterization
    },
    ("biosphere3", "CH4"): {
        "type": "emission",
        "name": "methane, fossil",
        "categories": ("air",),
        "CAS number": "000074-82-8",
    },
}

bd.Database("biosphere3").write(biosphere_data)
```

!!! tip "Using ecoinvent"
    If you import ecoinvent using `bw2io`, the biosphere database is created automatically with all necessary flows and CAS numbers.

---

## Background Databases

Background databases represent the supply chain processes that support your foreground system. In `optimex`, you can have **multiple background databases at different time points** to model technological evolution.

### Single Background Database

```python
from datetime import datetime

# Create background database
bg_2020 = bd.Database("db_2020")
bg_2020.write({
    ("db_2020", "electricity"): {
        "name": "electricity production",
        "location": "GLO",
        "reference product": "electricity",
        "exchanges": [
            {"amount": 1, "type": "production", "input": ("db_2020", "electricity")},
            {"amount": 0.5, "type": "biosphere", "input": ("biosphere3", "CO2")},
        ],
    },
})

# Set representative time (required for optimex)
bg_2020.metadata["representative_time"] = datetime(2020, 1, 1).isoformat()
bg_2020.register()
```

### Multiple Time-Specific Databases

To model technology improvement over time, create multiple versions:

```python
# Database for 2020
bg_2020 = bd.Database("db_2020")
bg_2020.write({...})  # Current technology
bg_2020.metadata["representative_time"] = datetime(2020, 1, 1).isoformat()
bg_2020.register()

# Database for 2030 (cleaner technology)
bg_2030 = bd.Database("db_2030")
bg_2030.write({...})  # Improved technology with lower emissions
bg_2030.metadata["representative_time"] = datetime(2030, 1, 1).isoformat()
bg_2030.register()

# Database for 2040
bg_2040 = bd.Database("db_2040")
bg_2040.write({...})  # Even cleaner
bg_2040.metadata["representative_time"] = datetime(2040, 1, 1).isoformat()
bg_2040.register()
```

`optimex` automatically interpolates between these databases based on when exchanges occur.

!!! info "Using premise"
    The [premise](https://premise.readthedocs.io/) package can automatically generate future versions of ecoinvent based on IAM scenarios. This is the recommended approach for realistic prospective LCA.

---

## Foreground Database

The foreground database contains the processes you want to optimize. **It must be named `"foreground"`**.

See [Foreground Modeling](foreground_modeling.md) for detailed guidance on setting up foreground processes with temporal distributions.

```python
fg = bd.Database("foreground")
fg.write({
    ("foreground", "process_A"): {...},
    ("foreground", "process_B"): {...},
})
fg.register()
```

---

## Characterization Methods

Define impact assessment methods for your elementary flows:

```python
# Simple GWP method
bd.Method(("GWP", "100 years")).write([
    (("biosphere3", "CO2"), 1),
    (("biosphere3", "CH4"), 28),
])

# You can define multiple methods
bd.Method(("Water use", "m3")).write([
    (("biosphere3", "water"), 1),
])
```

!!! tip "Using existing methods"
    If you import ecoinvent with `bw2io`, standard methods like IPCC GWP and ReCiPe are available automatically.

---

## Market Prices

Only needed for the [cost objective](minimizing_costs.md). Prices are stored on the background nodes, one value per time-specific database, so they can change over time like the inventories. Price the background products that the foreground buys **directly**; their upstream supply chains need no prices.

`set_market_prices()` writes the prices from a table, with one row per node and database year:

```python
from optimex.economics import set_market_prices

background_databases = {
    2020: "ei312_REMIND-EU_SSP2_NDC_2020",
    2030: "ei312_REMIND-EU_SSP2_NDC_2030",
}
price_data = [
    {
        "name": "market group for electricity, medium voltage",
        "location": "RER",
        "product": "electricity, medium voltage",
        "unit": "kilowatt hour",
        "year": year,
        "price": price,  # constant 2020 EUR/kWh
    }
    for year, price in [(2020, 0.09), (2030, 0.07)]
]

set_market_prices(
    price_data=price_data,
    background_databases=background_databases,
    product_col="product",
    unit_col="unit",
)
```

- **Units:** each price refers to one unit of the node's reference product. Use one currency and price basis throughout: 90 EUR/MWh becomes 0.09 EUR/kWh for a product measured in kilowatt hours. No unit, currency, or inflation conversion is applied.
- **Every database:** set a price in each background database. Prospective databases, for example from `premise`, carry no prices themselves, and a product priced in some databases but not in others is rejected during [LCA processing](lca_processing.md#market-prices).
- **Matching:** rows identify nodes by name and location, optionally narrowed by reference product and unit. Codes are not used, because premise gives the same activity a different code in each scenario database.
- **Options:** `price_data` can also be a pandas DataFrame; `name_col`, `location_col`, `product_col`, `unit_col`, `year_col`, and `price_col` map custom column names. `overwrite=False` keeps existing prices, `strict=False` turns unresolvable rows into warnings instead of errors, and `price_attribute` changes the node attribute the price is written to (default `"market_price"`).

To find the products that need a price, list the background inputs of the foreground processes:

```python
purchased = {
    (exc.input["name"], exc.input["location"], exc.input["reference product"])
    for process in bd.Database("foreground")
    for exc in process.technosphere()
    if exc.input["database"] != "foreground"
}
```

Prices are only read when `LCAConfig.price_attribute` is set, so they do not affect impact-only studies.

---

## Complete Setup Example

```python
import bw2data as bd
from datetime import datetime

# 1. Create project
bd.projects.set_current("my_optimex_project")

# 2. Biosphere (if not using ecoinvent)
bd.Database("biosphere3").write({
    ("biosphere3", "CO2"): {"type": "emission", "name": "CO2"},
})

# 3. Background databases with time metadata
for year in [2020, 2030]:
    db = bd.Database(f"db_{year}")
    db.write({
        (f"db_{year}", "electricity"): {
            "name": "electricity",
            "exchanges": [
                {"amount": 1, "type": "production", "input": (f"db_{year}", "electricity")},
                {"amount": 0.5 * (1 - (year-2020)*0.01), "type": "biosphere", "input": ("biosphere3", "CO2")},
            ],
        },
    })
    db.metadata["representative_time"] = datetime(year, 1, 1).isoformat()
    db.register()

# 4. Foreground (see Foreground Modeling guide)
fg = bd.Database("foreground")
fg.write({...})  # Your processes
fg.register()

# 5. Characterization method
bd.Method(("GWP", "example")).write([
    (("biosphere3", "CO2"), 1),
])
```

---

## Verifying Your Setup

Check that all databases are correctly configured:

```python
# List all databases
print(bd.databases)

# Check background database metadata
for db_name in bd.databases:
    db = bd.Database(db_name)
    if "representative_time" in db.metadata:
        print(f"{db_name}: {db.metadata['representative_time']}")

# Verify foreground exists
assert "foreground" in bd.databases, "Foreground database required!"
```

---

## Next Steps

- [Foreground Modeling](foreground_modeling.md): Set up foreground processes with temporal distributions, operation flags, and vintage parameters
- [LCA Processing](lca_processing.md): Define the demand, configure the time-explicit LCA, and build the optimization inputs
