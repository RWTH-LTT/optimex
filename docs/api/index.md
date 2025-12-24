# API Reference

This section contains the API documentation automatically generated from the `optimex` source code docstrings.

`optimex` follows a modular pipeline architecture for time-explicit LCO of transition pathways:

<div class="grid cards" markdown>

-   :lucide-database:{ style="color: #4dabf7" } **1. LCA Processor**

    ---

    Time-explicit LCA data processing using Brightway. Extracts temporal distributions, constructs inventory tensors, and prepares characterization factors.

    [:lucide-arrow-right: `lca_processor`](lca_processor.md)

-   :lucide-arrow-right-left:{ style="color: #69db7c" } **2. Converter**

    ---

    Bridges LCA processing and optimization. Validates inputs, manages scaling for numerical stability, and handles constraint configuration.

    [:lucide-arrow-right: `converter`](converter.md)

-   :lucide-refresh-cw:{ style="color: #ffa94d" } **3. Optimizer**

    ---

    Constructs and solves Pyomo optimization models. Minimizes environmental impacts while respecting demand and process constraints.

    [:lucide-arrow-right: `optimizer`](optimizer.md)

-   :lucide-bar-chart-3:{ style="color: #da77f2" } **4. Postprocessing**

    ---

    Extracts results from solved models. Provides DataFrames and publication-quality visualizations for impacts, installations, and production.

    [:lucide-arrow-right: `postprocessing`](postprocessing.md)

</div>
