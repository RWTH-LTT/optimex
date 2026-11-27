---
icon: lucide/layers
tags:
  - optimization
  - analysis
---

# Analyses

A single optimization gives one optimal pathway for one set of assumptions.
Most studies need more: how the optimum changes with a limit or a price, what
one objective costs in terms of another, and how robust the result is to
uncertain data. All of these solve the same product system many times, and
`optimex` solves such series together: with HiGHS or Gurobi, the model is built
once and every variant is assembled from it.

<div class="grid cards" markdown>

-   :lucide-list-ordered:{ style="margin-right: 0.25rem;" } **[Sweeps and Scenarios](../series.md)**

    ---

    Solve a range of limits, prices, or other assumptions with
    `optimizer.solve_series` and compare the results in one table.

-   :lucide-scale:{ style="margin-right: 0.25rem;" } **[Trade-offs](../trade_offs.md)**

    ---

    Trace the Pareto frontier between two objectives, such as climate impact
    and water use, or costs and a carbon budget.

-   :lucide-dices:{ style="margin-right: 0.25rem;" } **[Uncertainty](../uncertainty.md)**

    ---

    Carry the uncertainty in your Brightway data into the optimization:
    how often a technology is optimal, and how robust a pathway is.

</div>

The [uncertainty example](../examples/uncertainty_example.md) and the
[basic cost example](../examples/basic_example_econ.md#9-pareto-frontier-between-cost-and-climate-impact)
show these analyses on small systems.
