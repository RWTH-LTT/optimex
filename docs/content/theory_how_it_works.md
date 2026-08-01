---
icon: lucide/workflow
tags:
  - background
  - methodology
---

# How it Works

`optimex` extends static LCO to a time-explicit formulation through five conceptual steps.

## Step 1: Adding a Time Dimension

In static LCO, all matrices and vectors exist without any notion of time. The first step is to add an explicit time index, so that production systems, emissions, and characterization factors can differ at different points in time. However, simply indexing by time still evaluates each time-slice independently — a process at one point in time can only interact with processes at the same point.

## Step 2: Connecting Time Slices through Convolution

Real-world life cycles span multiple time periods: a facility built in 2030 may operate until 2050 and be decommissioned in 2051. To capture this, `optimex` distinguishes between two time concepts:

- **System time**: Absolute calendar time (e.g., the year 2030)
- **Process time**: Relative time within a process lifecycle (e.g., 5 years after installation)

Process exchanges are defined in process time — specifying *when* within the lifecycle they occur (construction, each year of operation, end-of-life). Convolution then translates these relative timings into absolute system times based on when a process is actually installed. This is what enables exchanges from a single process to span multiple years in the optimization.

**Example:** A process installed in 2030 with construction at process time -2 generates construction exchanges in 2028. Operational inputs spread from process time 0 to 10 translate to 2030–2040. End-of-life at process time 11 maps to 2041.

## Step 3: Separating Foreground and Background

Following standard LCA practice, `optimex` separates the system into:

- **Foreground**: The processes under study, containing the decision variables. This is where the optimization happens.
- **Background**: The broader economy and supply chains (e.g., from ecoinvent), with fixed production routes.

The foreground system uses convolution to distribute its exchanges across time. When foreground processes require inputs from the background (e.g., electricity, steel), these demands are resolved at the absolute system time when they occur — meaning a process built in 2030 sources its electricity from the 2030 background, while its end-of-life treatment in 2050 draws from the 2050 background.

## Step 4: Modeling Temporal Evolution

Both the foreground and background systems can evolve over time:

**Foreground evolution** is captured through vintage-dependent parameters. Each process exchange can have scaling factors that depend on the installation year (vintage). For example, an electrolyzer installed in 2035 might consume 25% less electricity per unit of hydrogen than one installed in 2025, reflecting technology learning. This means multiple cohorts of the same technology can coexist with different performance characteristics.

**Background evolution** is captured through time-specific databases. By providing multiple versions of the background database at different points in time (e.g., ecoinvent projected to 2020, 2030, 2040 using [premise](https://premise.readthedocs.io/)), `optimex` automatically matches each background demand to the appropriate database based on when the demand occurs. When database timestamps don't align exactly with demand times, `optimex` interpolates between the nearest available databases.

## Step 5: Flexible Operation

Traditional LCO assumes processes always run at full capacity. `optimex` separates the decision into two components:

- **Capacity installation**: How many process units are built at a given time (vintage)
- **Operational level**: How many of those units are actually running at each point in time

One installed unit corresponds to one reference-flow unit of the underlying LCA process: it delivers the production stated by its production exchange over its **whole lifetime**, so the per-time-step entry of that exchange is what one running unit yields **per year**. Installation-dependent exchanges are charged once per unit, which means the construction burden of a unit is amortized over its lifetime output — no more and no less. A demand profile that leaves part of a unit's lifetime unused therefore shows up as idle capacity and a correspondingly higher impact.

This separation is important because it enables **vintage-specific dispatch**: when multiple cohorts of the same technology coexist, the optimizer can preferentially utilize cleaner vintages — creating an emissions-aware merit order. It also allows the model to identify strategic overcapacities, where early investment in clean technologies offsets the stranded cost of idled fossil infrastructure.

To make this work, each exchange in a process is classified as either installation-dependent (construction materials, end-of-life treatment) or operation-dependent (production output, fuel consumption, operational emissions). Installation-dependent exchanges are fixed to the installed capacity, while operation-dependent exchanges scale with how much the process is actually operated.