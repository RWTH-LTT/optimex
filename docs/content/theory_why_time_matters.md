---
icon: lucide/lightbulb
tags:
  - background
  - methodology
---

# Why Time Matters in Life Cycle Optimization

Life Cycle Optimization (LCO) couples optimization models with Life Cycle Assessment (LCA) to design transition pathways that minimize environmental impacts while meeting demand over time. However, traditional LCO approaches assume all life cycle stages and emissions occur simultaneously — effectively collapsing the temporal dimension. This creates two blind spots:

1. **Time-specific limits cannot be verified.** If the model cannot distinguish between an emission occurring today or a decade from now, it cannot check whether a pathway respects annual carbon budgets or yearly resource extraction limits.

2. **Cumulative impacts are miscounted.** Ignoring that decommissioning happens decades after construction — in a fundamentally different background system — means total life cycle impacts are calculated against the wrong supply chain.

These blind spots arise from two interacting temporal dimensions:

- **Temporal distribution**: Life cycle stages span years or decades. Construction precedes operation, which precedes end-of-life. Each stage has distinct environmental exchanges occurring at different times.
- **Temporal evolution**: Production systems improve over time. Electricity grids decarbonize, process efficiencies increase, and supply chains shift. The same exchange has a different environmental footprint depending on *when* it occurs.

Distribution determines *when* an exchange occurs; evolution determines its *magnitude* at that moment. Both must be considered jointly. `optimex` addresses this by extending the standard LCA matrices with an explicit time dimension.