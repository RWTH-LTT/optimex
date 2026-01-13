# Setting Up Brightway

`optimex` builds on the [Brightway](https://docs.brightway.dev/) LCA framework. Before running an optimization, you need to set up a Brightway project with the required databases.

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

- [Foreground Modeling](foreground_modeling.md): Learn how to set up foreground processes with temporal distributions
- [Optimization Setup](optimization_setup.md): Configure demand and characterization for optimization
