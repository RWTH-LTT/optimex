# Theory

## Why Time Matters in Life Cycle Optimization

Life Cycle Optimization (LCO) couples optimization models with Life Cycle Assessment (LCA) to design transition pathways that minimize environmental impacts while meeting demand over time. However, traditional LCO approaches assume all life cycle stages and emissions occur simultaneously — effectively collapsing the temporal dimension. This creates two blind spots:

1. **Time-specific limits cannot be verified.** If the model cannot distinguish between an emission occurring today or a decade from now, it cannot check whether a pathway respects annual carbon budgets or yearly resource extraction limits.

2. **Cumulative impacts are miscounted.** Ignoring that decommissioning happens decades after construction — in a fundamentally different background system — means total life cycle impacts are calculated against the wrong supply chain.

These blind spots arise from two interacting temporal dimensions:

- **Temporal distribution**: Life cycle stages span years or decades. Construction precedes operation, which precedes end-of-life. Each stage has distinct environmental exchanges occurring at different times.
- **Temporal evolution**: Production systems improve over time. Electricity grids decarbonize, process efficiencies increase, and supply chains shift. The same exchange has a different environmental footprint depending on *when* it occurs.

Distribution determines *when* an exchange occurs; evolution determines its *magnitude* at that moment. Both must be considered jointly. `optimex` addresses this by extending the standard LCA matrices with an explicit time dimension.

---

## Time-Explicit Extension

`optimex` extends static LCO to a time-explicit formulation through five conceptual steps.

### Step 1: Adding a Time Dimension

In static LCO, all matrices and vectors exist without any notion of time. The first step is to add an explicit time index, so that production systems, emissions, and characterization factors can differ at different points in time. However, simply indexing by time still evaluates each time-slice independently — a process at one point in time can only interact with processes at the same point.

### Step 2: Connecting Time Slices through Convolution

Real-world life cycles span multiple time periods: a facility built in 2030 may operate until 2050 and be decommissioned in 2051. To capture this, `optimex` distinguishes between two time concepts:

- **System time**: Absolute calendar time (e.g., the year 2030)
- **Process time**: Relative time within a process lifecycle (e.g., 5 years after installation)

Process exchanges are defined in process time — specifying *when* within the lifecycle they occur (construction, each year of operation, end-of-life). Convolution then translates these relative timings into absolute system times based on when a process is actually installed. This is what enables exchanges from a single process to span multiple years in the optimization.

**Example:** A process installed in 2030 with construction at process time -2 generates construction exchanges in 2028. Operational inputs spread from process time 0 to 10 translate to 2030–2040. End-of-life at process time 11 maps to 2041.

### Step 3: Separating Foreground and Background

Following standard LCA practice, `optimex` separates the system into:

- **Foreground**: The processes under study, containing the decision variables. This is where the optimization happens.
- **Background**: The broader economy and supply chains (e.g., from ecoinvent), with fixed production routes.

The foreground system uses convolution to distribute its exchanges across time. When foreground processes require inputs from the background (e.g., electricity, steel), these demands are resolved at the absolute system time when they occur — meaning a process built in 2030 sources its electricity from the 2030 background, while its end-of-life treatment in 2050 draws from the 2050 background.

### Step 4: Modeling Temporal Evolution

Both the foreground and background systems can evolve over time:

**Foreground evolution** is captured through vintage-dependent parameters. Each process exchange can have scaling factors that depend on the installation year (vintage). For example, an electrolyzer installed in 2035 might consume 25% less electricity per unit of hydrogen than one installed in 2025, reflecting technology learning. This means multiple cohorts of the same technology can coexist with different performance characteristics.

**Background evolution** is captured through time-specific databases. By providing multiple versions of the background database at different points in time (e.g., ecoinvent projected to 2020, 2030, 2040 using [premise](https://premise.readthedocs.io/)), `optimex` automatically matches each background demand to the appropriate database based on when the demand occurs. When database timestamps don't align exactly with demand times, `optimex` interpolates between the nearest available databases.

### Step 5: Flexible Operation

Traditional LCO assumes processes always run at full capacity. `optimex` separates the decision into two components:

- **Capacity installation**: How much capacity is built at a given time (vintage)
- **Operational level**: How much of that capacity is actually used at each point in time

This separation is important because it enables **vintage-specific dispatch**: when multiple cohorts of the same technology coexist, the optimizer can preferentially utilize cleaner vintages — creating an emissions-aware merit order. It also allows the model to identify strategic overcapacities, where early investment in clean technologies offsets the stranded cost of idled fossil infrastructure.

To make this work, each exchange in a process is classified as either installation-dependent (construction materials, end-of-life treatment) or operation-dependent (production output, fuel consumption, operational emissions). Installation-dependent exchanges are fixed to the installed capacity, while operation-dependent exchanges scale with how much the process is actually operated.

---

## Scope and Limitations

### Single-Tier Convolution

Unlike dynamic LCA approaches that propagate temporal distributions through entire multi-tier supply chains, `optimex` applies convolution only at the foreground tier. Upstream supply chains in the background are treated as temporally aggregated at the time of the direct exchange. This simplification is necessary because multi-tier convolution creates a circular dependency: upstream timings depend on installation timing, which is the decision variable being optimized.

**Workarounds** where finer upstream resolution is needed:

- Move critical upstream processes into the foreground, allowing explicit temporal distributions on their exchanges.
- Pre-compute time-resolved emission profiles using dynamic LCA and map them as temporal distributions on elementary flows.

### Dynamic Characterization

`optimex` supports both static and dynamic Life Cycle Impact Assessment (LCIA). For impacts where timing matters — particularly climate change — dynamic metrics like Cumulative Radiative Forcing (CRF) or dynamic Global Warming Potential (GWP) capture how different greenhouse gases accumulate and decay in the atmosphere over time. Static characterization (e.g., GWP100) can still be used for impact categories where timing is less critical.

### Data Requirements

Time-explicit LCO requires data on temporal distributions of processes, temporal evolution of production systems, and (optionally) time-specific characterization factors. When temporal information is unavailable, `optimex` defaults to static values, allowing analyses to proceed with partial temporal coverage.
