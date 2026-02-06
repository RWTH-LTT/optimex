# General Usage Example for the `optimex` Package

`optimex` is a Python package for transition pathway optimization based on time-explicit Life Cycle Assessment (LCA). It helps identify optimal process portfolios and deployment timing in systems with multiple processes producing the same product, aiming to minimize dynamically accumulating environmental impacts over time.

`optimex` builds on top of the optimization framework [pyomo](https://github.com/Pyomo/pyomo) and the LCA framework [Brightway](https://docs.brightway.dev/en/latest).

Before using `optimex`, you need to set up a Brightway project. Brightway manages LCA databases, calculates impacts, and allows detailed environmental analysis.


```python
from datetime import datetime
import numpy as np
import bw2data as bd
from bw_temporalis import TemporalDistribution

bd.projects.set_current("standalone_optimex_example")
```

### Defining Demand of Functional Unit in `optimex`

Before we can process our LCA data in `optimex`, we need to define a temporally distiributed demand of our functional unit.  

The demand describes how much of the functional product (e.g., `"R1"`) is needed each year across the time horizon of interest. This is defined using a `TemporalDistribution`, which allows you to assign specific demand values to individual dates.

By explicitly modeling how demand changes over time, `optimex` can optimize not only which processes to use but also when to install and operate them.


```python
# Define temporally distributed demand from 2020 to 2030
years = range(2020, 2030)
td_demand = TemporalDistribution(
    date=np.array([datetime(year, 1, 1).isoformat() for year in years], dtype='datetime64[s]'),
    amount=np.asarray([0, 0, 10, 5, 10, 5, 10, 5, 10, 5]),
)
functional_demand = {"R1": td_demand}
```

### Required Brightway Databases: Biosphere

`optimex` requires a biosphere database listing elementary flows, typically imported from databases like ecoinvent. Here, we create a custom biosphere database named **biosphere3**, which is the standard name expected by `optimex`.

We also explicitly specify Chemical Abstracts Service (CAS) numbers for corresponding flows (in ecoinvent these are already attached) , which will assist in later characterization steps.



```python

# BIOSPHERE
biosphere_data = {
    ("biosphere3", "CO2"): {
        "type": "emission",
        "name": "carbon dioxide",
        "CAS number": "000124-38-9"
    },
    ("biosphere3", "CH4"): {
        "type": "emission",
        "name": "methane, fossil",
        "CAS number": "000074-82-8"
    },
}
bd.Database("biosphere3").write(biosphere_data)
```

    100%|██████████| 2/2 [00:00<?, ?it/s]


    14:44:30+0200 [info     ] Vacuuming database            



When conducting a Life Cycle Assessment (LCA) using `optimex`, the system is divided into two conceptual parts:

- **Foreground system**: The set of processes that are directly controlled, designed, or influenced by the decision-maker. These processes are typically the focus of modeling and change, and they represent the core of the decision or innovation being studied. In `optimex`, the foreground system consists of candidate processes whose installation and operation timings are optimized to meet a functional demand while minimizing environmental impact.

- **Background system**: The system of processes that supply inputs to the foreground system but are not affected by decisions within the scope of the study. These are typically taken from external LCA databases (e.g., ecoinvent) and represent average or generic data for energy, materials, transport, etc. In `optimex`, these background processes provide the upstream data and remain unaffected by the optimization.


### Required Databases: Background

Altough the background system is not affected by decision making in the foreground system, it will gradually **change over time** due to technological progress or policy shifts.

To reflect this temporal evolution, we define **multiple background databases** at different time points (e.g., 2020, 2030, 2040). For extensive modelling the python package `premise` offfers the ability to modify ecoinvent databases according to projected scenario pathways. 

During optimization, `optimex` dynamically links each process to the appropriate background database according to its temporal profile, ensuring that background data reflects the timing of each exchange within the process lifecycle.

For `optimex` to recognize the representative time of each background version, we store the corresponding date in the database metadata:



```python

# BACKGROUND 2020
db_2020_data = {
    ("db_2020", "I1"): {
        "name": "node I1",
        "location": "somewhere",
        "reference product": "I1",
        "exchanges": [
            {"amount": 1, "type": "production", "input": ("db_2020", "I1")},
            {"amount": 1, "type": "biosphere", "input": ("biosphere3", "CO2")},
        ],
    },
    ("db_2020", "I2"): {
        "name": "node I2",
        "location": "somewhere",
        "reference product": "I2",
        "exchanges": [
            {"amount": 1, "type": "production", "input": ("db_2020", "I2")},
            {"amount": 1, "type": "biosphere", "input": ("biosphere3", "CH4")},
        ],
    },
}
bg_2020 = bd.Database("db_2020")
bg_2020.write(db_2020_data)
bg_2020.metadata["representative_time"] = datetime(2020, 1, 1).isoformat() # optimex-specific: representative time of the background database
bg_2020.register()

# BACKGROUND 2030
db_2030_data = {
    ("db_2030", "I1"): {
        "name": "node I1",
        "location": "somewhere",
        "reference product": "I1",
        "exchanges": [
            {"amount": 1, "type": "production", "input": ("db_2030", "I1")},
            {"amount": 0.9, "type": "biosphere", "input": ("biosphere3", "CO2")},
        ],
    },
    ("db_2030", "I2"): {
        "name": "node I2",
        "location": "somewhere",
        "reference product": "I2",
        "exchanges": [
            {"amount": 1, "type": "production", "input": ("db_2030", "I2")},
            {"amount": 0.9, "type": "biosphere", "input": ("biosphere3", "CH4")},
        ],
    },
}
bg_2030 = bd.Database("db_2030")
bg_2030.write(db_2030_data)
bg_2030.metadata["representative_time"] = datetime(2030, 1, 1).isoformat() # optimex-specific: representative time of the background database
bg_2030.register()

```

    14:44:30+0200 [warning  ] Not able to determine geocollections for all datasets. This database is not ready for regionalization.


    100%|██████████| 2/2 [00:00<?, ?it/s]


    14:44:30+0200 [info     ] Vacuuming database            
    14:44:30+0200 [warning  ] Not able to determine geocollections for all datasets. This database is not ready for regionalization.


    100%|██████████| 2/2 [00:00<?, ?it/s]


    14:44:30+0200 [info     ] Vacuuming database            



### Required Databases: Foreground
Time-Explicit Modeling in the Foreground

`optimex` extends traditional LCA by explicitly modeling time, which enables modelling of flows at the actual occurence in the process life cycle. To process this information we need to define temporal distributions of the flows correspodning the foreground processes using `TemporalDistribution` again.

For example, consider a process `P1` that produces a product `R1` over a 4-year lifecycle:

- **Year 0 (Pre-operation)**: The process is installed (constructed). It does not produce any output but consumes an input `I1`.  
- **Years 1 and 2 (Operation phase)**: The process produces 0.5 units of `R1` each year and emits CO₂ accordingly.  
- **Year 3 (Post-operation)**: The process is decommissioned and does not exchange anything.

The time-explicit modelling, also allows installation of a process and then operation at different capacity levels at the following time steps. To enable flexible operation we explictly need to tell `optimex` the time limits of the operation phase (start and end of operation phase) for each process. Intermediate and elementary flows that are exchanged during operation, which are scaling relative to the operation level (e.g., production output, emissions during operation) need to be marked with `"operation": True`. Flexible operation is particularly useful in time-explicit LCA, as it allows the model to install a capacity once and operate it differently across its lifespan — offering realistic modelling of transition pathways.

In the following definition of a foreground database, all additional information `optimex` requires in comparison to a default brightway databse is commented.


```python

# FOREGROUND - temporally distributed
foreground_data = {
    ("foreground", "P1"): {
        "name": "process P1",
        "location": "somewhere",
        "reference product": "R1",
        "operation_time_limits": (1,2), # Optimex-specific: start and end year of operation phase
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "input": ("foreground", "P1"),
                "temporal_distribution": TemporalDistribution( # Optimex-specific: temporal distribution of the exchange)
                    date=np.array(range(4), dtype="timedelta64[Y]"),
                    amount=np.array([0, 0.5, 0.5, 0]),
                ),
                "operation": True, # Optimex-specific: indicates that this exchange is part of the operation phase
            },
            {
                "amount": 27.5,
                "type": "technosphere",
                "input": ("db_2020", "I1"),
                "temporal_distribution": TemporalDistribution( # Optimex-specific: temporal distribution of the exchange)
                    date=np.array(range(4), dtype="timedelta64[Y]"),
                    amount=np.array([1, 0, 0, 0]),
                ),
            },
            {
                "amount": 20,
                "type": "biosphere",
                "input": ("biosphere3", "CO2"),
                "temporal_distribution": TemporalDistribution( # Optimex-specific: temporal distribution of the exchange)
                    date=np.array(range(4), dtype="timedelta64[Y]"),
                    amount=np.array([0, 0.5, 0.5, 0]),
                ),
                "operation": True,
            },
        ],
    },
    ("foreground", "P2"): {
        "name": "process P2",
        "location": "somewhere",
        "reference product": "R1",
        "operation_time_limits": (1,2),
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "input": ("foreground", "P2"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array(range(4), dtype="timedelta64[Y]"),
                    amount=np.array([0, 0.5, 0.5, 0]),
                ),
                "operation": True,
            },
            {
                "amount": 1,
                "type": "technosphere",
                "input": ("db_2020", "I2"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array(range(4), dtype="timedelta64[Y]"),
                    amount=np.array([1, 0, 0, 0]),
                ),
            },
            {
                "amount": 20,
                "type": "biosphere",
                "input": ("biosphere3", "CO2"),
                "temporal_distribution": TemporalDistribution(
                    date=np.array(range(4), dtype="timedelta64[Y]"),
                    amount=np.array([0, 0.5, 0.5, 0]),
                ),
                "operation": True,
            },
        ],
    },
}
fg = bd.Database("foreground")
fg.write(foreground_data)
fg.register()
```

    14:44:30+0200 [warning  ] Not able to determine geocollections for all datasets. This database is not ready for regionalization.


    100%|██████████| 2/2 [00:00<?, ?it/s]


    14:44:30+0200 [info     ] Vacuuming database            


### Characterization Methods

To quantify environmental impact, we need to define **characterization factors** (CFs) for each elementary flow—such as CO₂ or CH₄ emissions. These CFs convert emissions into impact scores (e.g., in kg CO₂-eq).

> 🔎 Note: If you're using a full database like **ecoinvent**, characterization methods (e.g. IPCC, ReCiPe) are usually already included.  
> But since we're creating a custom biosphere here, we need to define our own CFs manually.



```python
bd.Method(("GWP", "example")).write([
    (("biosphere3", "CO2"), 1),
    (("biosphere3", "CH4"), 27),
])

bd.Method(("land use", "example")).write([
    (("biosphere3", "CO2"), 2),
    (("biosphere3", "CH4"), 1),
])
```

### Dynamic Characterization in `optimex`

One of the key advantages of `optimex` is its ability to perform **dynamic characterization**—assessing environmental impacts in a way that accounts for **when emissions occur**, not just how much is emitted. This is particularly relevant for impact categories like **climate change**, where environmental impact is influenced by timing and amount of emissions.

*Why Use Dynamic Characterization?*

In conventional (static) LCA, the timing of emissions is ignored—1 kg of CO₂ emitted today is treated the same as 1 kg emitted 50 years from now.  
Dynamic characterization allows `optimex` to differentiate these cases by using **time-resolved impact factors**, which in the context of climate change take atmospheric acummulation and decay of green house gases into account, providing a more accurate modelling approach of long-term environmental effects.

Currently, `optimex` supports dynamic modeling for climate change through the following metrics:

- **CRF** – Cumulative Radiative Forcing  
- **GWP** – Global Warming Potential (time-sensitive)

These metrics are implemented via the [`dynamic_characterization`](https://github.com/brightway-lca/dynamic_characterization) package.

> 📌 Note: Only climate change metrics are currently supported for dynamic impact assessment.

### Configuring LCA Processing in `optimex`

To process all the data defined before with `optimex`, we need to configure both **temporal parameters** and **characterization methods**.

**Minimum requirements:**

1. Pass the defined demand

2. Define a valid `temporal` parameters, including:
   - A `start_date` (e.g., start of system transiition and impact assessment).
   - A `temporal_resolution` (e.g., `"year"`).
   - A `time_horizon` over which impacts are accumulated.

3. In the `characterization_methods`, specify:
   - A `metric` that supports dynamic characterization in the choosen impact category (e.g., `"CRF"` as cumulative radiative forcing for climate change).
   - The associated `brightway_method` for conventional impact factors (used for preprocessing or for static categories).

If no dynamic metric is specified for a category, `optimex` will default to **static characterization** using the standard Brightway impact method.

The example below demonstrates how to configure dynamic climate change modeling with `"CRF"` and a static method for land use:




```python
from optimex import lca_processor

lca_config = lca_processor.LCAConfig(
    demand = functional_demand ,
    temporal= {
        "start_date": datetime(2020, 1, 1),
        "temporal_resolution": "year",
        "time_horizon": 100,
    },
    characterization_methods=[
        {
            "category_name": "climate_change",
            "brightway_method": ("GWP", "example"),
            "metric": "CRF",
        },
        {
            "category_name": "land_use",
            "brightway_method": ("land use", "example"),
        },
    ],
)
```

With all the necessary components defined, we are now ready to gather the LCI and prepare for optimization.

To summarize, we have:

- Defined a **custom biosphere** with elementary flows (e.g., CO₂, CH₄).
- Created **foreground** and **background** systems, including temporal distributions.
- Registered **characterization methods** (static or dynamic).
- Specified **demand over time** using `TemporalDistribution`.
- Marked processes as **functional flows** to indicate eligible supply routes.
- (Optionally) Enabled **dynamic characterization** for more time-sensitive impact assessment.


```python
lca_data_processor = lca_processor.LCADataProcessor(lca_config)
```

    2025-06-03 14:44:30.894 | INFO     | optimex.lca_processor:_parse_demand:323 - Identified demand in system time range of %s for functional flows %s
    2025-06-03 14:44:30.904 | INFO     | optimex.lca_processor:_construct_foreground_tensors:448 - Constructed foreground tensors.
    2025-06-03 14:44:30.904 | INFO     | optimex.lca_processor:log_tensor_dimensions:443 - Technosphere shape: (2 processes, 2 flows, 4 years) with 8 total entries.
    2025-06-03 14:44:30.909 | INFO     | optimex.lca_processor:log_tensor_dimensions:443 - Biosphere shape: (2 processes, 1 flows, 4 years) with 8 total entries.
    2025-06-03 14:44:30.909 | INFO     | optimex.lca_processor:log_tensor_dimensions:443 - Production shape: (2 processes, 1 flows, 4 years) with 8 total entries.
    2025-06-03 14:44:30.913 | INFO     | optimex.lca_processor:_calculate_inventory_of_db:486 - Calculating inventory for database: db_2020
    2025-06-03 14:44:30.960 | INFO     | optimex.lca_processor:_calculate_inventory_of_db:502 - Factorized LCI for database: db_2020
    100%|██████████| 2/2 [00:00<00:00, 36.60it/s]
    2025-06-03 14:44:31.027 | INFO     | optimex.lca_processor:_calculate_inventory_of_db:542 - Finished calculating inventory for database: db_2020
    2025-06-03 14:44:31.030 | INFO     | optimex.lca_processor:_calculate_inventory_of_db:486 - Calculating inventory for database: db_2030
    2025-06-03 14:44:31.053 | INFO     | optimex.lca_processor:_calculate_inventory_of_db:502 - Factorized LCI for database: db_2030
    100%|██████████| 2/2 [00:00<00:00, 71.83it/s]
    2025-06-03 14:44:31.096 | INFO     | optimex.lca_processor:_calculate_inventory_of_db:542 - Finished calculating inventory for database: db_2030
    2025-06-03 14:44:31.098 | INFO     | optimex.lca_processor:_prepare_background_inventory:649 - Computed background inventory using method: sequential
    2025-06-03 14:44:31.102 | INFO     | dynamic_characterization.dynamic_characterization:characterize:82 - No custom dynamic characterization functions provided. Using default dynamic             characterization functions. The flows that are characterized are based on the selection                of the initially chosen impact category.
    2025-06-03 14:44:31.442 | INFO     | dynamic_characterization.dynamic_characterization:characterize:82 - No custom dynamic characterization functions provided. Using default dynamic             characterization functions. The flows that are characterized are based on the selection                of the initially chosen impact category.
    2025-06-03 14:44:31.749 | INFO     | optimex.lca_processor:_construct_characterization_tensor:813 - Dynamic CRF characterization for climate_change completed.
    2025-06-03 14:44:31.765 | INFO     | optimex.lca_processor:_construct_characterization_tensor:760 - Static characterization for method land_use completed.
    2025-06-03 14:44:31.767 | INFO     | optimex.lca_processor:_construct_mapping_matrix:703 - Constructed mapping matrix for background databases based on linear interpolation.


### Saving and Reusing Model Inputs

Gathering all LCA-relevant data—including biosphere flows, foreground and background systems, temporal distributions, and characterization methods—can be **computationally expensive**.  
If you're working with similar scenarios or want to explore different optimization settings, it's often unnecessary to rebuild the entire setup from scratch.

 `OptimizationModelInputs`: Interface for Efficient Reuse

To address this, `optimex` provides a dedicated interface called **`OptimizationModelInputs`**. This object:

- Collects all the structured LCA data needed for optimization.
- Can be **saved** to disk for future use.
- Can be **loaded** later to resume or modify the scenario.
- Allows you to tweak configurations (e.g., demand, technologies, time horizon) without re-running the full data setup.

In the next steps, we’ll create and populate a `OptimizationModelInputs` object, and use it to prepare for running the optimization in `optimex`.


```python
from optimex import converter
manager = converter.ModelInputManager()
optimization_model_inputs = manager.parse_from_lca_processor(lca_data_processor) 
# manager.save("model_inputs.json") # if you want to save the model inputs to a file
# manager.load("model_inputs.json") # if you want to load the model inputs from a file
optimization_model_inputs.model_dump()

```




    {'PROCESS': ['P2', 'P1'],
     'REFERENCE_PRODUCT': ['R1'],
     'INTERMEDIATE_FLOW': ['I2', 'I1'],
     'ELEMENTARY_FLOW': ['CO2', 'CH4'],
     'BACKGROUND_ID': ['db_2020', 'db_2030'],
     'PROCESS_TIME': [0, 1, 2, 3],
     'SYSTEM_TIME': [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029],
     'CATEGORY': ['land_use', 'climate_change'],
     'demand': {('R1', 2020): 0.0,
      ('R1', 2021): 0.0,
      ('R1', 2022): 10.0,
      ('R1', 2023): 5.0,
      ('R1', 2024): 10.0,
      ('R1', 2025): 5.0,
      ('R1', 2026): 10.0,
      ('R1', 2027): 5.0,
      ('R1', 2028): 10.0,
      ('R1', 2029): 5.0},
     'operation_flow': {('P2', 'R1'): True,
      ('P2', 'CO2'): True,
      ('P1', 'R1'): True,
      ('P1', 'CO2'): True},
     'foreground_technosphere': {('P2', 'I2', 0): 1.0,
      ('P2', 'I2', 1): 0.0,
      ('P2', 'I2', 2): 0.0,
      ('P2', 'I2', 3): 0.0,
      ('P1', 'I1', 0): 27.5,
      ('P1', 'I1', 1): 0.0,
      ('P1', 'I1', 2): 0.0,
      ('P1', 'I1', 3): 0.0},
     'foreground_biosphere': {('P2', 'CO2', 0): 0.0,
      ('P2', 'CO2', 1): 10.0,
      ('P2', 'CO2', 2): 10.0,
      ('P2', 'CO2', 3): 0.0,
      ('P1', 'CO2', 0): 0.0,
      ('P1', 'CO2', 1): 10.0,
      ('P1', 'CO2', 2): 10.0,
      ('P1', 'CO2', 3): 0.0},
     'foreground_production': {('P2', 'R1', 0): 0.0,
      ('P2', 'R1', 1): 0.5,
      ('P2', 'R1', 2): 0.5,
      ('P2', 'R1', 3): 0.0,
      ('P1', 'R1', 0): 0.0,
      ('P1', 'R1', 1): 0.5,
      ('P1', 'R1', 2): 0.5,
      ('P1', 'R1', 3): 0.0},
     'background_inventory': {('db_2020', 'I2', 'CH4'): 1.0,
      ('db_2020', 'I1', 'CO2'): 1.0,
      ('db_2030', 'I2', 'CH4'): 0.8999999761581421,
      ('db_2030', 'I1', 'CO2'): 0.8999999761581421},
     'mapping': {('db_2020', 2020): 1.0,
      ('db_2020', 2021): 0.9,
      ('db_2030', 2021): 0.1,
      ('db_2020', 2022): 0.8,
      ('db_2030', 2022): 0.2,
      ('db_2020', 2023): 0.7,
      ('db_2030', 2023): 0.3,
      ('db_2020', 2024): 0.6,
      ('db_2030', 2024): 0.4,
      ('db_2020', 2025): 0.5,
      ('db_2030', 2025): 0.5,
      ('db_2020', 2026): 0.4,
      ('db_2030', 2026): 0.6,
      ('db_2020', 2027): 0.30000000000000004,
      ('db_2030', 2027): 0.7,
      ('db_2020', 2028): 0.19999999999999996,
      ('db_2030', 2028): 0.8,
      ('db_2020', 2029): 0.09999999999999998,
      ('db_2030', 2029): 0.9},
     'characterization': {('climate_change', 'CO2', 2020): 8.856378067710995e-14,
      ('climate_change', 'CO2', 2021): 8.78632948322376e-14,
      ('climate_change', 'CO2', 2022): 8.716115201983699e-14,
      ('climate_change', 'CO2', 2023): 8.645732530491629e-14,
      ('climate_change', 'CO2', 2024): 8.575178705349817e-14,
      ('climate_change', 'CO2', 2025): 8.50445089133486e-14,
      ('climate_change', 'CO2', 2026): 8.433546179417147e-14,
      ('climate_change', 'CO2', 2027): 8.362461584725384e-14,
      ('climate_change', 'CO2', 2028): 8.291194044454685e-14,
      ('climate_change', 'CO2', 2029): 8.219740415716608e-14,
      ('climate_change', 'CH4', 2020): 2.3651673669270527e-12,
      ('climate_change', 'CH4', 2021): 2.3651198384711042e-12,
      ('climate_change', 'CH4', 2022): 2.3650681065838066e-12,
      ('climate_change', 'CH4', 2023): 2.36501179951239e-12,
      ('climate_change', 'CH4', 2024): 2.3649505126261558e-12,
      ('climate_change', 'CH4', 2025): 2.3648838055087402e-12,
      ('climate_change', 'CH4', 2026): 2.364811198793221e-12,
      ('climate_change', 'CH4', 2027): 2.3647321707173162e-12,
      ('climate_change', 'CH4', 2028): 2.3646461533739266e-12,
      ('climate_change', 'CH4', 2029): 2.364552528630073e-12,
      ('land_use', 'CO2', 2020): 2.0,
      ('land_use', 'CO2', 2021): 2.0,
      ('land_use', 'CO2', 2022): 2.0,
      ('land_use', 'CO2', 2023): 2.0,
      ('land_use', 'CO2', 2024): 2.0,
      ('land_use', 'CO2', 2025): 2.0,
      ('land_use', 'CO2', 2026): 2.0,
      ('land_use', 'CO2', 2027): 2.0,
      ('land_use', 'CO2', 2028): 2.0,
      ('land_use', 'CO2', 2029): 2.0,
      ('land_use', 'CH4', 2020): 1.0,
      ('land_use', 'CH4', 2021): 1.0,
      ('land_use', 'CH4', 2022): 1.0,
      ('land_use', 'CH4', 2023): 1.0,
      ('land_use', 'CH4', 2024): 1.0,
      ('land_use', 'CH4', 2025): 1.0,
      ('land_use', 'CH4', 2026): 1.0,
      ('land_use', 'CH4', 2027): 1.0,
      ('land_use', 'CH4', 2028): 1.0,
      ('land_use', 'CH4', 2029): 1.0},
     'operation_time_limits': {'P2': (1, 2), 'P1': (1, 2)},
     'category_impact_limit': None,
     'process_limits_max': None,
     'process_limits_min': None,
     'cumulative_process_limits_max': None,
     'cumulative_process_limits_min': None,
     'process_coupling': None,
     'process_names': {'P2': 'process P2', 'P1': 'process P1'},
     'process_limits_max_default': inf,
     'process_limits_min_default': 0.0,
     'cumulative_process_limits_max_default': inf,
     'cumulative_process_limits_min_default': 0.0}



### Creating the Optimization Model

With the `OptimizationModelInputs` object prepared, we can now **populate the optimization model** in `optimex`.

This step involves specifying:

- A **name** for the model,
- The **impact category** to be minimized (e.g., `"climate_change"`).


```python
from optimex import optimizer

model = optimizer.create_model(
    optimization_model_inputs,
    name = "demo_simple_example",
    objective_category = "climate_change",
)
```

## Solving the Optimization Model

Once the model is created, you can solve it using available solvers such as **GLPK** or **Gurobi**.

Use the `solve_model` function from `optimex` to run the optimization. It offers several parameters to control solver options, convergence criteria, and output verbosity.

For detailed information on all parameters and solver configurations, please refer to the documentation.


```python
m, obj, results = optimizer.solve_model(model, solver_name="gurobi", tee=False) # choose solver here, e.g. "gurobi", "cplex", "glpk", etc.
```

    2025-06-03 14:44:32.828 | INFO     | optimex.optimizer:solve_model:601 - Solver [gurobi] termination: optimal
    2025-06-03 14:44:32.837 | INFO     | optimex.optimizer:solve_model:626 - Objective (scaled): 4.33081
    2025-06-03 14:44:32.841 | INFO     | optimex.optimizer:solve_model:627 - Objective (real):   2.81685e-10


## Postprocessing the Optimization Results

`optimex` offers powerful tools to analyze and visualize the optimal solution in detail. Key aspects you can explore include:

- **Impact over time:** View time-resolved environmental impacts according to the chosen impact category.
- **Demand fulfillment:** Track how the demand for the functional product evolves and by which processes it is met during the modeled period.
- **Installation timeline:** Identify when each process was installed during the transition pathway.
- **Operation profiles:** Examine the operational levels of processes year-by-year, especially important for flexible operation scenarios.
- **Production of functional flows:** Analyze how much of the functional product each process produces over time.

Together, these outputs provide a comprehensive understanding of the system’s dynamics, helping you evaluate trade-offs and the timing of interventions. Below is a demosntartion of some postprocessing tools.



```python
from optimex import postprocessing
pp = postprocessing.PostProcessor(m)

df_impact = pp.get_impacts()
df_impact
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Category</th>
      <th colspan="2" halign="left">land_use</th>
      <th colspan="2" halign="left">climate_change</th>
    </tr>
    <tr>
      <th>Process</th>
      <th>P2</th>
      <th>P1</th>
      <th>P2</th>
      <th>P1</th>
    </tr>
    <tr>
      <th>Time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>19.8</td>
      <td>0.000000</td>
      <td>4.682937e-11</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>400.0</td>
      <td>0.000000</td>
      <td>1.743223e-11</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>219.4</td>
      <td>0.000000</td>
      <td>5.452696e-11</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>400.0</td>
      <td>0.000000</td>
      <td>1.715036e-11</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>200.0</td>
      <td>1044.999987</td>
      <td>8.504451e-12</td>
      <td>4.443576e-11</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>0.0</td>
      <td>400.000000</td>
      <td>0.000000e+00</td>
      <td>1.686709e-11</td>
    </tr>
    <tr>
      <th>2027</th>
      <td>0.0</td>
      <td>1222.999982</td>
      <td>0.000000e+00</td>
      <td>5.113645e-11</td>
    </tr>
    <tr>
      <th>2028</th>
      <td>0.0</td>
      <td>400.000000</td>
      <td>0.000000e+00</td>
      <td>1.658239e-11</td>
    </tr>
    <tr>
      <th>2029</th>
      <td>0.0</td>
      <td>200.000000</td>
      <td>0.000000e+00</td>
      <td>8.219740e-12</td>
    </tr>
  </tbody>
</table>
</div>




```python
pp.plot_impacts()
```


    
![png](basic_optimex_example_files/basic_optimex_example_26_0.png)
    





    (<Figure size 1000x600 with 2 Axes>,
     array([<Axes: title={'center': 'land_use'}, xlabel='Time', ylabel='Value'>,
            <Axes: title={'center': 'climate_change'}, xlabel='Time', ylabel='Value'>],
           dtype=object))




```python
pp.plot_installation()

```


    
![png](basic_optimex_example_files/basic_optimex_example_27_0.png)
    





    (<Figure size 1000x600 with 1 Axes>,
     <Axes: title={'center': 'Installed Capacity'}, xlabel='Time', ylabel='Installation'>)




```python
pp.plot_production_and_demand()
```


    
![png](basic_optimex_example_files/basic_optimex_example_28_0.png)
    





    (<Figure size 1000x600 with 1 Axes>,
     <Axes: title={'center': 'Production and Demand'}, xlabel='Time', ylabel='Quantity'>)


