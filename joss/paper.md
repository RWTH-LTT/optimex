---
title: "optimex: A Python Package for Time-Explicit Life Cycle Optimization"
tags:
  - Life Cycle Optimization
  - Life Cycle Assessment
  - temporal distribution
  - temporal evolution
  - dynamic LCA
  - prospective LCA
  - transition pathways
  - Brightway
  - Pyomo

authors:
  - name: Timo Diepers
    orcid: 0009-0002-8566-8618
    affiliation: 1
  - name: Jan Tautorus
    affiliation: 1

affiliations:
  - name: Institute of Technical Thermodynamics (LTT), RWTH Aachen University, Germany
    index: 1

date: 29 April 2026
bibliography: paper.bib
---

# Summary

When assessing the environmental impact of a product or technology — such as an electric vehicle or a hydrogen production facility — Life Cycle Assessment (LCA) is the standard scientific method. LCA traces all material and energy flows from raw material extraction through manufacturing, use, and final disposal, and converts them into environmental indicators such as greenhouse gas emissions or water consumption. A logical extension of LCA is Life Cycle Optimization (LCO): rather than assessing a single predefined system, LCO asks which combination of technologies and deployment schedules minimizes environmental impact while satisfying a specified demand over time. Both LCA and LCO, however, are traditionally *static*: all life cycle stages are collapsed to a single point in time, even though construction, operation, and decommissioning of real systems span years or decades and the background supply chains (power grids, steel production, transport) continuously change.

`optimex` is an open-source Python package that closes this gap. It performs *time-explicit* Life Cycle Optimization: it finds optimal technology transition pathways while accounting for *when* emissions occur and *how* product systems evolve over time. Practically, a user specifies a set of candidate technologies (e.g., steam methane reforming and PEM electrolysis for hydrogen production), a time-varying demand profile, and an environmental objective or constraint. `optimex` then automatically constructs and solves an optimization model that respects the full temporal life cycle of each technology — its construction phase, operational years with potentially improving efficiency, and end-of-life treatment — and evaluates all flows against time-appropriate supply chains. The package is built on the widely used [Brightway](https://brightway.dev) LCA framework [@Mutel:2017] and uses [Pyomo](https://www.pyomo.org) [@Bynum:2021] for mathematical optimization. Full documentation, tutorials, and interactive example notebooks are available at [optimex.readthedocs.io](https://optimex.readthedocs.io/).

# Statement of Need

Designing environmentally optimal technology transitions is a central challenge in sustainability science and engineering. Static LCO addresses this in principle, but produces misleading results whenever the temporal dimension matters — which it almost always does in rapidly decarbonizing energy and industrial systems. Two phenomena are of particular importance [@MuellerDiepers:2025; @Beloin:2020]:

**Temporal distribution.** Life cycle stages are spread across time. A wind turbine requires construction materials in the years before it operates; it generates electricity over a 20- to 30-year operational period; and it requires decommissioning and recycling afterward. Evaluating all of these against the same background supply chain ignores that the electricity grid will be very different in 2030 than in 2055.

**Temporal evolution.** Technologies improve over time and supply chains decarbonize. An electrolyzer installed in 2035 may consume 25% less electricity per kilogram of hydrogen than one installed today [@Arvidsson:2024]. Treating all vintages of a technology as identical leads to systematic overestimation of future impacts.

These two phenomena interact: a process installed in 2030 must be evaluated against the 2030 supply chain during construction, against progressively cleaner supply chains during its operational years, and against a 2060-era supply chain at end-of-life.

`optimex` addresses both dimensions within a single optimization framework. The target audience is researchers and practitioners working on energy system transitions, industrial decarbonization, circular economy planning, and prospective environmental assessment who use or are familiar with the Brightway LCA ecosystem. Users who already have Brightway-based LCA models — including temporalized models built with `bw_timex` [@MuellerDiepers:2025] — can reuse them directly in `optimex` without rebuilding anything from scratch.

# State of the Field

Several tools handle related but more limited problems. Within the Brightway ecosystem, `bw_timex` [@MuellerDiepers:2025] performs time-explicit *assessment* of a predefined product system, but does not optimize technology selection or deployment schedules. The `Temporalis` package [@Cardellini:2018] handles temporal distribution in LCA but assumes a static background system. Prospective LCA tools such as `premise` [@Sacchi:2022], `Futura` [@Joyce:2022], and `pathways` [@Sacchi:2024] model evolving supply chains but are designed for assessment rather than optimization. `TRAILS` [@Sacchi:2026] and `ProsperDyn` [@LangQuantzendorff:2025] each combine temporal distribution and evolution in assessment; neither provides optimization functionality.

On the optimization side, energy system models such as PyPSA optimize technology deployment over time, but they use simplified energy-balance representations rather than full LCA inventories and cannot trace temporal life cycle impacts through the supply chain.

`optimex` occupies a position not filled by any existing open-source tool: it combines full life cycle inventory accounting (via Brightway) with multi-period optimization (via Pyomo), while treating both temporal distribution and temporal evolution. Contributing this functionality to existing tools would require fundamental architectural changes — assessment-only tools such as `bw_timex` are not built around decision variables, and energy system models lack LCA inventory structures. `optimex` is therefore a genuine scholarly contribution rather than a duplication of existing efforts.

# Software Design

`optimex` extends static LCO to a time-explicit formulation through five design steps, each representing a deliberate architectural trade-off.

**Time indexing.** Rather than running independent single-period analyses and aggregating their results, `optimex` embeds the time dimension directly into the technosphere and biosphere tensors. This unified representation enables a single optimization solve over the full planning horizon, avoids repeated matrix inversions, and allows the optimizer to make trade-offs across time periods — for instance, accepting higher near-term impacts to enable lower cumulative impacts.

**Convolution.** Process exchanges are defined in *process time* (relative to installation, e.g., "construction materials arrive one year before first operation") and mapped to absolute calendar years through discrete convolution when a specific installation year is considered. This approach — adapted from `bw-temporalis` [@Cardellini:2018] for the optimization context — captures that construction of a 2030-installed plant generates flows in 2028–2029, while its end-of-life in 2060 draws from a 2060-era supply chain.

**Foreground–background separation.** Following standard LCA practice [@Heijungs:2002], `optimex` distinguishes the foreground (the candidate processes subject to optimization) from the background (fixed supply chains from inventory databases such as ecoinvent, optionally projected with `premise` [@Sacchi:2022]). Background databases are provided at discrete years; `optimex` linearly interpolates between them when foreground demands fall at intermediate dates. This avoids the combinatorial explosion that would result from linking every foreground process to every background year individually, while still preserving time-appropriate background conditions.

**Vintage-dependent parameters.** Foreground processes can carry scaling factors or explicit exchange amounts that depend on the installation year (vintage). A 2035-vintage electrolyzer can be given 25% lower electricity consumption than a 2025-vintage unit, reflecting technology learning. These vintage parameters are defined directly in the Brightway database using two flexible formats: uniform scaling factors (`vintage_improvements`) or explicit per-vintage, per-process-time values (`vintage_amounts`). Missing vintages are linearly interpolated.

**Flexible operation and vintage dispatch.** `optimex` separates the decision into installed capacity (how much to build, in which year) and operational level (how much to run at each time step). This enables the optimizer to identify vintage-specific dispatch strategies, where cleaner, later-vintage units are preferentially utilized when multiple cohorts coexist — forming an emissions-aware merit order. It also allows the model to identify strategic overcapacity: early installation of low-carbon capacity whose embodied impacts are repaid by emission savings over its lifetime, even if that capacity is initially underutilized.

The resulting model is formulated as a Mixed-Integer Linear Program (MILP) in Pyomo [@Bynum:2021] and is compatible with any Pyomo-supported solver (GLPK, HiGHS, Gurobi, CPLEX). Both static characterization (e.g., GWP100) and dynamic characterization — including Cumulative Radiative Forcing and time-dependent GWP via the `dynamic_characterization` package [@Diepers:2025] — are supported, retaining the timing of emissions through to the impact assessment step.

The user-facing API is organized into four modules: `lca_processor` (Brightway data extraction and time-explicit tensor construction), `converter` (translation to Pyomo-compatible input format), `optimizer` (model creation, solving, and serialization), and `postprocessing` (result extraction, tabulation, and visualization). Model inputs and solved models can be serialized to JSON or pickle format, allowing the computationally expensive LCA processing step to be cached and reused across solver runs and sensitivity analyses.

# Research Impact Statement

`optimex` was first released on PyPI in February 2025 (v0.1.0) and is also distributed via conda. Active development has continued, with version 0.4.2 released in April 2026, introducing vintage-dependent foreground parameters, separation of capacity installation from operational dispatch, and a restructured user-facing API. The package is directly integrated with `bw_timex` [@MuellerDiepers:2025], whose underlying time-explicit LCA methodology is described in a peer-reviewed article in the *International Journal of Life Cycle Assessment* [@MuellerDiepers:2025]. Users with existing temporalized `bw_timex` product system models can pass them directly to `optimex` without modification. Interactive example notebooks are provided via Binder, enabling reproducible exploration of the full workflow without local installation. `optimex` is currently being used in ongoing research on industrial transition pathway optimization at RWTH Aachen University's Institute of Technical Thermodynamics.

# AI Usage Disclosure

No generative AI tools were used in the development of the `optimex` software or its technical documentation. The initial draft of this paper was produced with AI assistance; all content was subsequently reviewed, fact-checked, and revised by the authors.

# Acknowledgements

We thank Amelie Müller and Arthur Jakobs for their foundational work on `bw_timex` and the underlying methodology of time-explicit LCA, which forms the assessment core of `optimex`.

# References
