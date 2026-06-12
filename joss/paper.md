---
title: "optimex: A Python Package for Time-Explicit Life Cycle Optimization"
tags:
  - life cycle optimization
  - life cycle assessment
  - temporal distribution
  - temporal evolution
  - prospective LCA
  - dynamic LCA
  - transition pathways
  - mathematical optimization

authors:
  - name: Timo Diepers
    orcid: 0009-0002-8566-8618
    affiliation: 1
  - name: Jan Tautorus
    orcid: 0009-0002-9216-870X
    affiliation: "2"

affiliations:
  - name: Institute of Technical Thermodynamics (LTT), RWTH Aachen University, Germany
    index: 1
  - name: Interdisciplinary Transformation University Austria (IT:U), Austria
    index: 2

date: 05 June 2026
bibliography: paper.bib
---

# Summary

`optimex` is an open-source Python package for time-explicit Life Cycle Optimization (LCO). 
LCO extends Life Cycle Assessment (LCA) by treating process selection, capacity installation, and operation as decision variables, determining process portfolios and deployment schedules that minimize an environmental objective subject to system constraints. 
`optimex` makes this optimization *time-explicit*: 
instead of collapsing each process's life cycle into a single instant, it tracks *when* every flow across its life cycle and supply chain occurs and lets each flow's magnitude reflect the state of technology and the surrounding economy *at that time*. 
Decisions made today are therefore evaluated against the conditions they will actually encounter in the future, which is what determines whether a transition pathway is genuinely feasible and environmentally sustainable once deployed.

Concretely, `optimex` simultaneously accounts for:

- the **temporal distribution** of flows across a process life cycle (construction, multi-year operation, maintenance, and end-of-life occur at different times);
- the **temporal evolution** of foreground technologies through vintage-dependent parameters (a process installed in 2035 is more efficient that one installed in 2025);
- **time-specific background supply chains**, so that upstream electricity, hydrogen, or steel inputs are assessed against future rather than present conditions; and
- **dynamic impact assessment**, where characterization factors vary over time.

The package embeds these temporal dimensions directly into the optimization problem, 
then determines installation timing, capacities, and operational levels under time-resolved environmental objectives and constraints. 
`optimex` builds on the `Brightway` LCA ecosystem [@Mutel:2017], 
sources dynamic characterization factors from `dynamic_characterization` [@DynamicCharacterization:2025], 
and uses `Pyomo` [@Bynum:2021] as its mathematical programming backend.

# Statement of need

Designing sustainable transitions requires decisions about *which* processes to deploy, *when* to deploy them, and *how* to operate them over time while accounting for impacts across their entire life cycle. 
LCO is well suited to this task because it links process and supply-chain decisions to environmental objectives and constraints [@Heijungs:2002]. 
However, existing LCO approaches treat life cycles as static, instantaneous events, collapsing the flows from construction, operation, maintenance, and end-of-life into a single point in time. 
This simplification neglects two temporal dimensions that are essential for transition pathways: 
flows are *distributed* over time, and product systems *evolve* over time. 
Crucially, the two dimensions are interlinked — when a flow occurs dictates the system conditions it encounters — so both must be considered jointly. 
Resolving when each flow actually occurs is decisive for whether a designed pathway is genuinely feasible and environmentally sustainable once deployed: 
the relevant limits are themselves time-dependent, spanning cumulative stock limits such as carbon budgets and finite mineral reserves as well as flow-related limits on the rate at which resources can be extracted or emissions absorbed. 
A pathway that appears compliant under static, time-aggregated accounting can breach these limits in practice, for example by concentrating demand for a scarce resource into a transient bottleneck that static models never reveal.

`optimex` addresses this gap by providing a general, open-source implementation of time-explicit LCO. 
Its target audience comprises LCA practitioners, energy- and industrial-systems modelers, and sustainability researchers who design technology transition pathways and need to optimize them against time-resolved environmental objectives and constraints. 
Because `optimex` is built on `Brightway`, practitioners can temporalize foreground models they have already built for standard LCA and turn them into optimization problems without rebuilding anything from scratch; temporalized product systems prepared with the time-explicit *assessment* framework `bw_timex` [@MuellerDiepers:2025] can be reused directly.

# State of the field

Holistically evaluating the environmental sustainability of a system requires accounting for entire life cycles across a broad range of environmental impact categories, which is the purpose of LCA [@Heijungs:2002]. 
LCA is traditionally static, but several methods have been proposed to add temporal considerations. 
Dynamic LCA tools such as `Temporalis` [@Cardellini:2018] capture *temporal distribution*, the timing of flows as they cascade through the many tiers of the supply chain, while assuming a fixed product system [@Beloin:2020]. 
Prospective LCA tools such as `premise` [@Sacchi:2022], `Futura` [@Joyce:2022], and `pathways` [@Sacchi:2024] instead capture *temporal evolution* but assess product systems at individual points in time [@Arvidsson:2024]. 
Recent tools combine both dimensions, including `bw_timex` [@MuellerDiepers:2025], `ProsperDyn` [@LangQuantzendorff:2025], and `TRAILS` [@Sacchi:2026], but all *assess* a predefined product system rather than determining an optimal one from a set of alternatives.

Resolving time, however, multiplies the number of viable alternatives: which technology to deploy, when, and how to operate it all interact through distributed flows and evolving conditions. 
This combinatorial complexity is precisely what optimization is meant to handle, yet conventional optimization tools cannot. 
Integrated Assessment Models and energy system models operate at too coarse a technological resolution to capture environmental impacts across full supply chains and multiple impact categories [@Weyant:2017; @Reinert:2022]. 
Life Cycle Optimization fills this role by embedding full LCA inventories into mathematical programs [@Azapagic:1998; @Katelhon:2016], but resolves time only partially. 
Multi-period LCO distributes installation decisions across a transition horizon, introducing temporal distribution at the level of installation, though it typically still relies on static inventory data. 
Such formulations can additionally draw on prospective inventory data to reflect technology evolution. 
The recent open-source tool `PULPO` [@Lechtenberg:2024], built on the same `Brightway` and `Pyomo` libraries as `optimex`, supports exactly this combination of multi-period optimization and prospective data. 
Even then, however, each technology's flows are placed at the moment of its installation, resolving only the first tier of the supply chain (*single-tier* distribution). 
Resolving the timing of flows throughout all tiers of the product system (*multi-tier* distribution) remains an open gap [@LangQuantzendorff:2025review; @Turner:2025].

In contrast, `optimex` explicitly resolves the timing of all flows throughout the product system, combining this multi-tier temporal distribution with temporal evolution. 
We achieve this by extending the traditional matrix-based LCA formulation with explicit time dimensions, exposing capacity installation and operation as time-indexed decision variables; we refer to @Diepers:2026 for the full formulation. 
Because every flow is tracked at the time it actually occurs, the resulting pathways can be checked against time-dependent stock- and flow-related limits, surfacing transient bottlenecks and cumulative-budget violations that static accounting hides. 
To our knowledge, `optimex` is the first tool to deliver time-explicit LCO in a consolidated, documented, and tested package.

# Software design

A time-explicit LCO with `optimex` follows a three-stage processing pipeline and optional visualization utilities for user inspection.

First, **LCA processing** temporalizes the foreground system and gathers all data for the optimization. 
As in conventional LCA, the user models products, processes, and their intermediate and elementary flows using the traditional `Brightway` workflows. 
Three additional steps make this representation time-explicit: 
process-relative temporal distributions (rTDs) are attached to exchanges to describe how each flow is distributed over time relative to the consuming or emitting process [@Cardellini:2018]; 
operation-dependent flows are flagged so they can be scaled separately from capacity installation; 
and vintage-dependent scaling factors are specified to capture technology evolution. 
In addition, a set of time-specific background databases (e.g., generated with `premise` [@Sacchi:2022]) can be specified. 
For dynamic impact assessment, users can draw from the `Brightway` library `dynamic_characterization`[@DynamicCharacterization:2025]. 
From these inputs, `optimex` constructs time-explicit tensors that describe the temporalized product system.

Second, **model conversion** validates the temporalized product system and converts it to `Pyomo`-compatible sets. 
A `Pydantic` model checks that processes, flows, and time indices are consistent across all tensors. 
Further, additional constraints can be added, e.g., deployment and operation limits, category impact limits, process coupling, and existing (brownfield) capacities.

Third, **optimization** builds a `Pyomo` model with time-indexed decision variables for installed capacity and operational level. 
Constraints enforce demand fulfilment, deployment and operation limits, the coupling of operation to installed capacity, and process linkages; the objective minimizes the total impact in a chosen category and metric. 
The backend is solver-agnostic, defaulting to Gurobi but compatible with open-source solvers.

Finally, **post-processing** extracts results as DataFrames and provides visualizations of impacts, installations, operation, and product balances.

Several design choices were central to the research application. 
Embedding the time dimension directly into the traditional LCA matrices keeps the formulation within the familiar matrix-based LCA structure [@Heijungs:2002] while allowing flexible temporal resolution. 
Separating capacity-dependent from operation-dependent flows is what enables vintage-specific, flexible dispatch. 
Inputs are scaled for numerical stability before being handed to the solver and denormalized during post-processing, improving computational stability. 
Moreover, the deep `Brightway` integration lets `optimex` work with any `Brightway`-compatible inventory and reuse product systems already built for standard LCA, avoiding data lock-in and facilitating adoption. 
Finally, `optimex` can export both the processed LCA data and the initialized or solved `Pyomo` model, supporting reproducibility, model sharing, and fast re-optimization without recomputing the LCA inputs.

# Research impact statement

`optimex` is the reference implementation of the time-explicit LCO framework introduced in the accompanying methodological work [@Diepers:2026], where it is applied to an integrated methanol and pig iron production system. 
There, time-explicit LCO uncovers transition strategies that remain invisible to conventional, static LCO: 
prioritizing early installation of high-efficiency technologies so that older processes can be idled once operational savings outweigh the embodied impacts of stranded assets, and diversifying technology portfolios to avoid transient resource bottlenecks.

Beyond this methodological work, `optimex` has been presented to the scientific community at two scientific conferences [@Diepers:2025presentation; @Diepers:2026poster], and has been applied in two student theses [@Tautorus:2025; @Lange:2026].

The package is built for reuse, community uptake and collaboration. 
It is distributed via PyPI and conda, developed openly on GitHub with a continuous-integration test suite and code-coverage reporting, and supported by comprehensive documentation, runnable example notebooks, and a Binder environment for zero-install exploration [@Diepers:2025optimex; @Diepers:2025docs]. 
By integrating with the widely used `Brightway` ecosystem and accepting temporalized models from `bw_timex`, `optimex` connects to an existing community of LCA practitioners rather than requiring them to adopt a wholly new workflow.

# AI usage disclosure

During development, the authors used AI coding assistants, namely Claude Code and GitHub Copilot, to aid implementation. 
The core test suite was written by the authors, and all AI-assisted output was reviewed and verified by the authors, who take full responsibility for the correctness of the code. 
Generative AI tools were also used to assist with editing portions of the documentation and this manuscript, with all text checked and approved by the authors.

# References
