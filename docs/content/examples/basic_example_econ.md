---
icon: lucide/badge-euro
tags:
  - learning
  - tutorial
  - costs
---


<div hidden data-source-edit-path="notebooks/basic_example_econ.ipynb" data-source-view-path="notebooks/basic_example_econ.ipynb"></div>
# Getting started with cost minimization in `optimex`

This notebook demonstrates a compact end-to-end workflow for cost-minimizing time-explicit Life Cycle Optimization (LCO) in `optimex`. One synthetic product can be supplied by two alternative foreground processes. Both require an installation-related background purchase and a route-specific operating input. Time-dependent market prices are assigned to these directly purchased background products.

Using the same product system, the notebook determines:

1. the cost minimum without an environmental constraint,
2. the climate-impact minimum as a reference, and
3. the cost minimum subject to a cumulative climate budget.

Together, these scenarios show how prices, discounting, and a cumulative climate budget affect installation and operation decisions. All quantities, prices, lifetimes, and environmental impacts are synthetic assumptions for demonstration purposes. The example demonstrates the software workflow.

## 1. Setup

Use a notebook kernel with this repository's `optimex`, Brightway,
`bw_temporalis`, Matplotlib, and the GLPK solver installed; see the
[installation guide](../installation.md).
No ecoinvent data are needed.

The following cells write synthetic data to the dedicated Brightway project
`optimex_basic_example_econ_simple`. Rerunning replaces only this example's named
databases and method; do not store other work in that project.


```python
from datetime import datetime
import sys
import warnings

warnings.filterwarnings("ignore", message="No fast sparse solver found")
warnings.filterwarnings(
    "ignore", message=r"urllib3 .* doesn't match a supported version!"
)

import bw2data as bd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bw_temporalis import TemporalDistribution
from IPython.display import display
from loguru import logger

from optimex import converter, lca_processor, optimizer, postprocessing
from optimex.economics import set_market_prices

# Keep the tutorial output focused on the inputs and results below.
logger.remove()
logger.add(sys.stderr, level="WARNING")

bd.projects.set_current("optimex_basic_example_econ_simple")
```

## 2. Product system and assumptions

![Two alternative foreground processes supply one product. Dashed arrows denote direct background purchases.](data/basic_example_econ/system.svg)

Rectangles are **processes**; the oval is a **product**. Dashed arrows denote
fixed background supplies, not additional technology choices. Both processes
also emit CO2 directly. Construction and operating supplies carry upstream CO2.

The conventional process needs less construction input; the advanced process
needs twice as much, but emits less during operation. Their route-specific
operating inputs are named `conventional_input` and `advanced_input`.

- Demand: 10 product units each year, 2025-2034; no initial capacity.
- One installed process unit can supply 1 product unit per year for five years,
  including its installation year. Capacity is continuous and can be idle.
- Costs: illustrative **real EUR with a 2025 price basis**, discounted at 3% to
  2025. No inflation, revenues, CO2 price, end-of-life costs, or residual value.
- Climate: kg CO2-eq, including foreground and background emissions, with a
  static factor of 1 kg CO2-eq/kg CO2. Impacts are not discounted.

Only the price of the advanced operating input changes. Technical coefficients
and background emission factors stay constant, isolating the price effect. The
falling price is an external assumption.


```python
years = np.arange(2025, 2035)
lifetime = 5
annual_demand = 10.0
discount_rate = 0.03
background_databases = {2025: "background_2025", 2034: "background_2034"}

background_parameters = pd.DataFrame([
    {"input": "construction", "role": "installation, both routes",
     "kg CO2/unit": 1.0,
     "price in 2025 [EUR_2025/unit]": 4.0,
     "price in 2034 [EUR_2025/unit]": 4.0},
    {"input": "conventional_input", "role": "operation, conventional",
     "kg CO2/unit": 2.0,
     "price in 2025 [EUR_2025/unit]": 5.0,
     "price in 2034 [EUR_2025/unit]": 5.0},
    {"input": "advanced_input", "role": "operation, advanced",
     "kg CO2/unit": 1.0,
     "price in 2025 [EUR_2025/unit]": 6.0,
     "price in 2034 [EUR_2025/unit]": 3.0},
]).set_index("input")
background_parameters
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>role</th>
      <th>kg CO2/unit</th>
      <th>price in 2025 [EUR_2025/unit]</th>
      <th>price in 2034 [EUR_2025/unit]</th>
    </tr>
    <tr>
      <th>input</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>construction</th>
      <td>installation, both routes</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>conventional_input</th>
      <td>operation, conventional</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>advanced_input</th>
      <td>operation, advanced</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Brightway data and market prices

The two background snapshots have identical inventories but different prices.
`optimex` interpolates the snapshot prices for the intervening calendar years.
All prices are expressed in real EUR with a 2025 price basis. Prices are attached
to **directly purchased background nodes**, not to foreground processes or
recursively to upstream exchanges. Foreground exchange amounts determine the
quantities purchased.


```python
%%capture
biosphere = bd.Database("biosphere3")
biosphere.write({
    ("biosphere3", "CO2"): {
        "name": "carbon dioxide", "unit": "kg",
        "type": bd.labels.biosphere_node_default,
    },
})
method = ("illustrative", "climate_change")
bd.Method(method).register(unit="kg CO2-eq")
bd.Method(method).write([(("biosphere3", "CO2"), 1.0)])

for year, db_name in background_databases.items():
    data = {}
    for name, row in background_parameters.iterrows():
        data[(db_name, name)] = {
            "name": name, "reference product": name,
            "location": "GLO", "unit": "unit",
            "type": bd.labels.chimaera_node_default,
            "exchanges": [
                {"input": (db_name, name), "amount": 1.0,
                 "type": bd.labels.production_edge_default},
                {"input": ("biosphere3", "CO2"), "amount": row["kg CO2/unit"],
                 "type": bd.labels.biosphere_edge_default},
            ],
        }
    db = bd.Database(db_name)
    db.write(data)
    db.metadata["representative_time"] = datetime(year, 1, 1).isoformat()
    db.register()
```


```python
price_data = pd.DataFrame([
    {"name": name, "location": "GLO", "year": year,
     "price": row[f"price in {year} [EUR_2025/unit]"]}
    for name, row in background_parameters.iterrows()
    for year in background_databases
])
set_market_prices(price_data, background_databases)
```

## 4. Installation, operation, and demand

Relative `TemporalDistribution` dates describe years since installation.
`operation=False` links construction to installed units, even if they are idle.
`operation=True` links production, operating purchases, and direct emissions to
the units actually running in each year.

The operation distribution has five weights of 1/5. **Exchange amounts below
are full-lifetime quantities**: production of 5 units becomes 1 unit/year;
5 units of operating input become 1 unit/year. Conventional direct CO2 is
4 kg/year, advanced direct CO2 is 1 kg/year, per running process unit.


```python
installation_td = TemporalDistribution(
    date=np.array([0], dtype="timedelta64[Y]"),
    amount=np.array([1.0]),
)
operation_td = TemporalDistribution(
    date=np.arange(lifetime).astype("timedelta64[Y]"),
    amount=np.full(lifetime, 1.0 / lifetime),
)

foreground_data = {
    ("foreground", "product"): {
        "name": "Product", "unit": "unit",
        "type": bd.labels.product_node_default,
    },
}
```


```python
foreground_data[("foreground", "conventional")] = {
    "name": "Conventional", "location": "GLO",
    "type": bd.labels.process_node_default,
    "operation_time_limits": (0, lifetime - 1),
    "exchanges": [
        {"input": ("foreground", "product"), "amount": float(lifetime),
         "type": bd.labels.production_edge_default,
         "temporal_distribution": operation_td, "operation": True},
        {"input": ("background_2025", "construction"), "amount": 1.0,
         "type": bd.labels.consumption_edge_default,
         "temporal_distribution": installation_td, "operation": False},
        {"input": ("background_2025", "conventional_input"), "amount": float(lifetime),
         "type": bd.labels.consumption_edge_default,
         "temporal_distribution": operation_td, "operation": True},
        {"input": ("biosphere3", "CO2"), "amount": 4.0 * lifetime,
         "type": bd.labels.biosphere_edge_default,
         "temporal_distribution": operation_td, "operation": True},
    ],
}
```


```python
%%capture
foreground_data[("foreground", "advanced")] = {
    "name": "Advanced", "location": "GLO",
    "type": bd.labels.process_node_default,
    "operation_time_limits": (0, lifetime - 1),
    "exchanges": [
        {"input": ("foreground", "product"), "amount": float(lifetime),
         "type": bd.labels.production_edge_default,
         "temporal_distribution": operation_td, "operation": True},
        {"input": ("background_2025", "construction"), "amount": 2.0,
         "type": bd.labels.consumption_edge_default,
         "temporal_distribution": installation_td, "operation": False},
        {"input": ("background_2025", "advanced_input"), "amount": float(lifetime),
         "type": bd.labels.consumption_edge_default,
         "temporal_distribution": operation_td, "operation": True},
        {"input": ("biosphere3", "CO2"), "amount": 1.0 * lifetime,
         "type": bd.labels.biosphere_edge_default,
         "temporal_distribution": operation_td, "operation": True},
    ],
}
bd.Database("foreground").write(foreground_data)
```


```python
demand_td = TemporalDistribution(
    date=np.array([f"{year}-01-01" for year in years], dtype="datetime64[s]"),
    amount=np.full(len(years), annual_demand),
)
functional_demand = {bd.get_node(database="foreground", code="product"): demand_td}
```

## 5. Prepare optimization inputs

`LCADataProcessor` reads the Brightway system and constructs the time-explicit
inventory and impact data. Setting `price_attribute` makes it also read the
market prices written above and interpolate them to every year. `ModelInputManager` transfers these data, including
the interpolated annual prices, and adds the discount rate and its reference
year. Demand dates define the modeled years, 2025-2034. This finite-horizon
example does not account for operation after 2034 or remaining asset value, so
late investments must be interpreted with that boundary in mind.


```python
%%capture
config = lca_processor.LCAConfig(
    demand=functional_demand,
    temporal={
        "start_date": datetime(2025, 1, 1),
        "temporal_resolution": "year",
        "database_dates": {
            name: datetime(year, 1, 1) for year, name in background_databases.items()
        },
    },
    characterization_methods=[{
        "category_name": "climate_change", "brightway_method": method,
    }],
    # Two tiny background databases: no need for parallel worker processes.
    background_inventory={"calculation_method": "sequential"},
    price_attribute="market_price",
)
processor = lca_processor.LCADataProcessor(config)
manager = converter.ModelInputManager()
manager.parse_from_lca_processor(processor)
inputs = manager.override(
    discount_rate=discount_rate, discount_reference_year=int(years[0]),
)
```


```python
# Inspect how the snapshot prices were interpolated to every year.
processed_prices = pd.DataFrame({
    flow: [inputs.intermediate_flow_prices[flow, int(year)] for year in years]
    for flow in ("construction", "conventional_input", "advanced_input")
}, index=years)
processed_prices.index.name = "Year"
processed_prices.columns.name = "Price [EUR_2025/unit]"
processed_prices
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Price [EUR_2025/unit]</th>
      <th>construction</th>
      <th>conventional_input</th>
      <th>advanced_input</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2025</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.666667</td>
    </tr>
    <tr>
      <th>2027</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.333333</td>
    </tr>
    <tr>
      <th>2028</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>2029</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.666667</td>
    </tr>
    <tr>
      <th>2030</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.333333</td>
    </tr>
    <tr>
      <th>2031</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>2032</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.666667</td>
    </tr>
    <tr>
      <th>2033</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.333333</td>
    </tr>
    <tr>
      <th>2034</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 6. Cost minimum and climate reference

The first model minimizes discounted background-purchase costs
(`objective="cost"`). A second model uses the environmental objective to
establish the minimum climate impact attainable with the same demand,
inventories, and technologies.


```python
%%capture
cost_model = optimizer.create_model(inputs, name="cost_minimum", objective="cost")
cost_model, minimum_cost, _ = optimizer.solve_model(
    cost_model, solver_name="glpk", tee=False,
)
```


```python
%%capture
climate_model = optimizer.create_model(
    inputs, name="climate_reference", objective_category="climate_change",
)
climate_model, minimum_climate, _ = optimizer.solve_model(
    climate_model, solver_name="glpk", tee=False,
)
```

## 7. Cost minimum with a cumulative climate budget

The illustrative budget is **330 kg CO2-eq over all ten years**, not an annual
limit or a carbon price. It lies between the climate impacts of the two reference
solutions and therefore creates a trade-off between the two objectives. The
value is a synthetic assumption, not a policy target derived from external
evidence.

Only one input changes: `cumulative_category_impact_limits`.


```python
%%capture
def climate_total(model):
    impacts = postprocessing.PostProcessor(model).get_impacts()
    return impacts["climate_change"].to_numpy().sum()


climate_budget = 330.0
assert minimum_climate < climate_budget < climate_total(cost_model)
constrained_inputs = manager.override(
    cumulative_category_impact_limits={"climate_change": climate_budget},
)
constrained_model = optimizer.create_model(
    constrained_inputs, name="cost_with_climate_budget", objective="cost",
)
constrained_model, constrained_cost, _ = optimizer.solve_model(
    constrained_model, solver_name="glpk", tee=False,
)
```

## 8. Results

The table compares total discounted cost and cumulative climate impact.
`PostProcessor.get_costs()` splits the costs into installation-driven and
operation-driven purchases, which sum to the reported total. They follow the
corresponding background purchases and should not be interpreted as complete
CAPEX and OPEX accounts; for example, fixed operating costs are omitted.


```python
models = {
    "Cost minimum": cost_model,
    "Climate reference": climate_model,
    "Cost + climate budget": constrained_model,
}
rows = []
for scenario, model in models.items():
    costs = postprocessing.PostProcessor(model).get_costs(discounted=True)
    rows.append({
        "Scenario": scenario,
        "Total cost [EUR_2025 PV]": costs.to_numpy().sum(),
        "Climate [kg CO2-eq]": climate_total(model),
        "Installation-driven cost [EUR_2025 PV]": costs["installation"].to_numpy().sum(),
        "Operation-driven cost [EUR_2025 PV]": costs["operation"].to_numpy().sum(),
    })
comparison = pd.DataFrame(rows).set_index("Scenario")
comparison.round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total cost [EUR_2025 PV]</th>
      <th>Climate [kg CO2-eq]</th>
      <th>Installation-driven cost [EUR_2025 PV]</th>
      <th>Operation-driven cost [EUR_2025 PV]</th>
    </tr>
    <tr>
      <th>Scenario</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cost minimum</th>
      <td>494.86</td>
      <td>430.0</td>
      <td>109.01</td>
      <td>385.85</td>
    </tr>
    <tr>
      <th>Climate reference</th>
      <td>551.52</td>
      <td>240.0</td>
      <td>149.01</td>
      <td>402.51</td>
    </tr>
    <tr>
      <th>Cost + climate budget</th>
      <td>524.68</td>
      <td>330.0</td>
      <td>130.06</td>
      <td>394.62</td>
    </tr>
  </tbody>
</table>
</div>



### Installation and production

Use the existing `PostProcessor` to extract results. The table shows **newly
installed process units**, not annual production. The plots show product
supplied in each year; since output is 1 unit per running process unit and year,
they also reveal which technology operates. A mix is allowed; these are not
integer, all-or-nothing plant choices.


```python
postprocessors = {
    name: postprocessing.PostProcessor(model) for name, model in models.items()
}
routes = ["conventional", "advanced"]
production = {
    name: pp.get_production().xs("product", axis=1, level="Product")[routes]
    for name, pp in postprocessors.items()
}
installations = pd.concat({
    name: pp.get_installation()[routes] for name, pp in postprocessors.items()
}, names=["Scenario", "Installation year"])
installations.loc[installations.abs().sum(axis=1) > 1e-7].round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Process</th>
      <th>conventional</th>
      <th>advanced</th>
    </tr>
    <tr>
      <th>Scenario</th>
      <th>Installation year</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Cost minimum</th>
      <th>2025</th>
      <td>10.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2030</th>
      <td>0.00</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Climate reference</th>
      <th>2025</th>
      <td>0.00</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>2030</th>
      <td>0.00</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Cost + climate budget</th>
      <th>2025</th>
      <td>4.74</td>
      <td>5.26</td>
    </tr>
    <tr>
      <th>2030</th>
      <td>0.00</td>
      <td>10.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, layout="constrained")
for ax, (name, data) in zip(axes, production.items()):
    ax.bar(data.index, data["conventional"], color="#657783", label="Conventional")
    ax.bar(data.index, data["advanced"], bottom=data["conventional"],
           color="#27836c", label="Advanced")
    ax.plot(years, demand_td.amount, color="#242424", linestyle="--",
            linewidth=1.2, label="Demand")
    ax.set(title=name, xlabel="Year", xlim=(years[0] - 0.5, years[-1] + 0.5),
           ylim=(0, annual_demand * 1.15), xticks=[2025, 2030, 2034])
    ax.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("Product units/year")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
plt.show()
```


    
![png](basic_example_econ_files/output_25_0.png)
    


### Sanity checks and interpretation

The following checks confirm the annual-unit convention, price assignment,
demand fulfillment, cost identity, and cumulative climate budget for this
example. They do not constitute a test of every `optimex` feature. A
climate-minimizing solution is not necessarily the cheapest among multiple
solutions with the same minimum climate impact.


```python
for route in ("conventional", "advanced"):
    assert all(
        np.isclose(inputs.foreground_production[route, "product", tau], 1.0)
        for tau in range(lifetime)
    )
np.testing.assert_allclose(
    [inputs.intermediate_flow_prices["advanced_input", int(year)] for year in years],
    np.linspace(6.0, 3.0, len(years)),
)
for data in production.values():
    np.testing.assert_allclose(data.sum(axis=1), demand_td.amount, atol=1e-6)
np.testing.assert_allclose(
    comparison["Total cost [EUR_2025 PV]"],
    comparison["Installation-driven cost [EUR_2025 PV]"]
    + comparison["Operation-driven cost [EUR_2025 PV]"],
    atol=1e-6,
)
assert minimum_cost <= constrained_cost + 1e-6
assert minimum_climate - 1e-6 <= climate_total(constrained_model) <= climate_budget + 1e-6
np.testing.assert_allclose(comparison.loc["Cost minimum", "Total cost [EUR_2025 PV]"], minimum_cost)
assert constrained_cost <= comparison.loc["Climate reference", "Total cost [EUR_2025 PV]"] + 1e-6

print(f"Cumulative climate impact: {climate_total(constrained_model):.2f} / "
      f"{climate_budget:.2f} kg CO2-eq")
print(f"Additional discounted cost versus cost minimum: "
      f"{constrained_cost - minimum_cost:.2f} EUR_2025")
```

    Cumulative climate impact: 330.00 / 330.00 kg CO2-eq
    Additional discounted cost versus cost minimum: 29.82 EUR_2025


The cost minimum initially uses the conventional process and switches to the
advanced process when capacity is replaced in 2030. Although the two operating
input prices are equal in 2028, the installation-related cost difference and
the five-year lifetimes affect the investment decision. The climate reference
uses the lower-emission advanced process, while the cumulative climate budget
brings part of this capacity forward relative to the unconstrained cost minimum.

These results illustrate the interaction of installation timing, operation,
prices, discounting, and a cumulative climate budget. 

## 9. Further reading

The essential chain is **Brightway exchanges + background prices -> processed
inputs -> objective and optional constraint -> results**.

- [Minimizing costs](../objective_cost.md): pricing,
  discounting, and cost interpretation.
- [Foreground modeling](../foreground_modeling.md) and
  [constraints](../constraints.md): lifecycle timing, existing
  capacity, technology changes, and deployment limits.
- [Postprocessing](../postprocessing_guide.md): further result views.
- [Basic example](basic_example.md) and [methanol and iron](methanol_and_iron.md):
  multi-product and connected foreground systems.

Those extensions are intentionally outside this introductory example.
