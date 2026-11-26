---
icon: lucide/network
tags:
  - product system
  - brightway setup
---

# Product System Modeling

The product system is modeled in [Brightway](https://docs.brightway.dev/), with
a few `optimex`-specific additions that bring time into it. It has two parts:

<div class="grid cards" markdown>

-   :lucide-database:{ style="margin-right: 0.25rem;" } **[Background Data](../background_data.md)**

    ---

    The Brightway project: biosphere, time-specific background databases that
    represent supply chains at different points in time, and LCIA methods.

-   :lucide-pencil:{ style="margin-right: 0.25rem;" } **[Foreground Modeling](../foreground_modeling.md)**

    ---

    The candidate processes to optimize: when their exchanges occur, which ones
    scale with operation, and how they improve with the installation year.

</div>

The `optimex` additions at a glance:

| Where | Field | Meaning |
|---|---|---|
| Background database | `representative_time` (metadata) | The year the database represents |
| Foreground process | `operation_time_limits` | Process-time window in which the process operates |
| Foreground exchange | `temporal_distribution` | When the exchange occurs, relative to installation |
| Foreground exchange | `operation` | `True` if the exchange scales with the units running |

If you already have a temporalized product system, e.g. from a time-explicit LCA
with [`bw_timex`](https://docs.brightway.dev/projects/bw-timex/en/latest/), you
can reuse it.

??? cost "Cost objective: market prices"
    To [minimize costs](../minimizing_costs.md), the background products bought
    directly by the foreground also need market prices on their nodes. See
    [Setting Market Prices](../minimizing_costs.md#setting-market-prices).
