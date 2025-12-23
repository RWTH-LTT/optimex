# API Reference

This section contains the API documentation automatically generated from the `optimex` source code docstrings.

`optimex` follows a modular pipeline architecture for time-explicit LCO of transition pathways:

<div class="grid cards" markdown>

-   :material-database-outline:{ .lg .middle style="color: #4dabf7" } **1. LCA Processor**

    ---

    Time-explicit LCA data processing using Brightway. Extracts temporal distributions, constructs inventory tensors, and prepares characterization factors.

    [:octicons-arrow-right-24: `lca_processor`](lca_processor.md)

-   :material-swap-horizontal:{ .lg .middle style="color: #69db7c" } **2. Converter**

    ---

    Bridges LCA processing and optimization. Validates inputs, manages scaling for numerical stability, and handles constraint configuration.

    [:octicons-arrow-right-24: `converter`](converter.md)

-   :material-cog-sync-outline:{ .lg .middle style="color: #ffa94d" } **3. Optimizer**

    ---

    Constructs and solves Pyomo optimization models. Minimizes environmental impacts while respecting demand and process constraints.

    [:octicons-arrow-right-24: `optimizer`](optimizer.md)

-   :material-chart-box-outline:{ .lg .middle style="color: #da77f2" } **4. Postprocessing**

    ---

    Extracts results from solved models. Provides DataFrames and publication-quality visualizations for impacts, installations, and production.

    [:octicons-arrow-right-24: `postprocessing`](postprocessing.md)

</div>
