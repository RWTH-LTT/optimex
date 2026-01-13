# Postprocessing Results

After solving an optimization model, `optimex` provides tools to extract, analyze, and visualize the results through the `PostProcessor` class.

---

## Creating a PostProcessor

```python
from optimex import postprocessing

# After solving
solved_model, objective, results = optimizer.solve_model(model, solver_name="glpk")

# Create postprocessor
pp = postprocessing.PostProcessor(solved_model)
```

---

## Extracting Data

### Environmental Impacts

Get impacts by process, category, and time:

```python
df_impacts = pp.get_impacts()
```

Returns a DataFrame with:
- **Index**: System time (years)
- **Columns**: MultiIndex of (Category, Process)
- **Values**: Impact in that category from that process at that time

```
Category    climate_change           land_use
Process     ProcessA    ProcessB     ProcessA    ProcessB
Time
2020        0.0         0.0          0.0         0.0
2021        150.5       0.0          25.0        0.0
2022        145.2       50.3         24.0        10.5
...
```

**Analyze specific aspects:**
```python
# Total impact per year
yearly_total = df_impacts.sum(axis=1)

# Total impact per process
process_total = df_impacts.sum(axis=0)

# Total impact in one category
climate_impacts = df_impacts["climate_change"].sum().sum()
```

---

### Installation (Capacity Deployment)

Get capacity installed by process and time:

```python
df_installation = pp.get_installation()
```

Returns a DataFrame with:
- **Index**: System time (years)
- **Columns**: Process IDs
- **Values**: Capacity installed at that time

```
Process     ProcessA    ProcessB
Time
2020        10.0        0.0
2021        5.0         0.0
2022        0.0         15.0
...
```

**Common analyses:**
```python
# Total capacity installed per process
total_capacity = df_installation.sum()

# Cumulative capacity over time
cumulative = df_installation.cumsum()

# When each process was first deployed
first_deployment = df_installation[df_installation > 0].idxmax()
```

---

### Operation Levels

Get how much each process operates per time period:

```python
df_operation = pp.get_operation()
```

Returns a DataFrame with:
- **Index**: System time (years)
- **Columns**: Process IDs
- **Values**: Operation level (units operating)

```
Process     ProcessA    ProcessB
Time
2020        0.0         0.0
2021        10.0        0.0
2022        15.0        5.0
...
```

**Capacity utilization:**
```python
# Calculate utilization rate
cumulative_capacity = df_installation.cumsum()
utilization = df_operation / cumulative_capacity
```

---

### Production

Get production by process and product:

```python
df_production = pp.get_production()
```

Returns a DataFrame with:
- **Index**: System time (years)
- **Columns**: MultiIndex of (Product, Process)
- **Values**: Production amount

---

### Demand Fulfillment

Get demand values over time:

```python
df_demand = pp.get_demand()
```

Returns a DataFrame with:
- **Index**: System time (years)
- **Columns**: Products
- **Values**: Demand amount

**Verify demand is met:**
```python
total_production = df_production.groupby(level=0, axis=1).sum()
demand_met = (total_production >= df_demand).all().all()
```

---

## Visualization

### Impact Plot

Stacked area chart of environmental impacts over time:

```python
fig = pp.plot_impacts()
```

![Impact plot example](../_static/placeholder_impacts.png)

**Customization:**
```python
fig = pp.plot_impacts(
    category="climate_change",  # Specific category (default: all)
    figsize=(12, 6),
)
```

---

### Installation Plot

Bar chart showing capacity deployment timeline:

```python
fig = pp.plot_installation()
```

![Installation plot example](../_static/placeholder_installation.png)

---

### Operation Plot

Shows operation levels over time:

```python
fig = pp.plot_operation()
```

---

### Production and Demand Plot

Compares production with demand to verify fulfillment:

```python
fig = pp.plot_production_and_demand()
```

![Production and demand plot](../_static/placeholder_production.png)

---

## Advanced Analysis

### Accessing Raw Model Data

For custom analysis, access the solved model directly:

```python
import pyomo.environ as pyo

# Get decision variable values
for p in solved_model.PROCESS:
    for t in solved_model.SYSTEM_TIME:
        installation = pyo.value(solved_model.var_installation[p, t])
        operation = pyo.value(solved_model.var_operation[p, t])
        print(f"{p}, {t}: install={installation:.2f}, operate={operation:.2f}")

# Access expressions
for c in solved_model.CATEGORY:
    total = pyo.value(solved_model.total_impact[c])
    print(f"{c}: {total}")
```

### Denormalization

`optimex` internally scales values for numerical stability. The `PostProcessor` automatically denormalizes, but for manual access:

```python
# Get scaling factors
fg_scale = solved_model.scales["foreground"]
cat_scales = solved_model.scales["characterization"]

# Denormalize a scaled value
scaled_value = pyo.value(solved_model.some_expression)
real_value = scaled_value * fg_scale * cat_scales["climate_change"]
```

---

## Exporting Results

### To CSV

```python
# Export all results
df_impacts.to_csv("impacts.csv")
df_installation.to_csv("installation.csv")
df_operation.to_csv("operation.csv")
```

### To Excel

```python
with pd.ExcelWriter("results.xlsx") as writer:
    df_impacts.to_excel(writer, sheet_name="Impacts")
    df_installation.to_excel(writer, sheet_name="Installation")
    df_operation.to_excel(writer, sheet_name="Operation")
```

### Saving Figures

```python
fig = pp.plot_impacts()
fig.savefig("impacts.png", dpi=300, bbox_inches="tight")
fig.savefig("impacts.pdf", bbox_inches="tight")
```

---

## Complete Postprocessing Example

```python
from optimex import postprocessing
import pandas as pd

# Create postprocessor
pp = postprocessing.PostProcessor(solved_model)

# Extract all data
df_impacts = pp.get_impacts()
df_installation = pp.get_installation()
df_operation = pp.get_operation()
df_production = pp.get_production()
df_demand = pp.get_demand()

# Summary statistics
print("=== Optimization Results ===")
print(f"Objective value: {objective:.2f}")
print(f"\nTotal impact by category:")
for cat in df_impacts.columns.get_level_values(0).unique():
    total = df_impacts[cat].sum().sum()
    print(f"  {cat}: {total:.2f}")

print(f"\nTotal capacity installed by process:")
for proc in df_installation.columns:
    total = df_installation[proc].sum()
    print(f"  {proc}: {total:.2f}")

# Generate all plots
fig_impacts = pp.plot_impacts()
fig_install = pp.plot_installation()
fig_prod = pp.plot_production_and_demand()

# Save results
with pd.ExcelWriter("optimization_results.xlsx") as writer:
    df_impacts.to_excel(writer, sheet_name="Impacts")
    df_installation.to_excel(writer, sheet_name="Installation")
    df_operation.to_excel(writer, sheet_name="Operation")

fig_impacts.savefig("impacts.png", dpi=300)
fig_install.savefig("installation.png", dpi=300)
```

---

## Comparing Scenarios

Run multiple scenarios and compare:

```python
results = {}

for scenario_name, constraints in scenarios.items():
    # Apply scenario-specific constraints
    model_inputs_copy = model_inputs.model_copy(deep=True)
    for key, value in constraints.items():
        setattr(model_inputs_copy, key, value)

    # Solve
    model = optimizer.create_model(model_inputs_copy, name=scenario_name, ...)
    solved, obj, _ = optimizer.solve_model(model, solver_name="glpk")

    # Store results
    pp = postprocessing.PostProcessor(solved)
    results[scenario_name] = {
        "objective": obj,
        "impacts": pp.get_impacts(),
        "installation": pp.get_installation(),
    }

# Compare objectives
for name, data in results.items():
    print(f"{name}: {data['objective']:.2f}")
```

---

## Next Steps

- [Examples](../examples/index.md): See complete worked examples
- [API Reference](../api/postprocessing.md): Full PostProcessor API documentation
