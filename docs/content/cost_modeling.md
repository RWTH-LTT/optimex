---
icon: lucide/coins
tags:
  - product system
  - economics
  - costs
---

# Cost Modeling

!!! cost "Only for the cost objective"
    This page is only needed if you want to [minimize costs](minimizing_costs.md)
    or report the costs of a pathway. Environmental studies can skip it.

Costs build on the same product system as the impacts: no extra processes or
exchanges are needed. What the cost objective adds is a **market price** for
each background product the foreground buys directly. This page covers what
gets priced, how the foreground structure decides when a purchase is costed, and
how prices get into the model.

The [basic cost example](examples/basic_example_econ.md) walks through all of it
with a synthetic one-product system that needs no ecoinvent data.

---

## Cost Boundary

The foreground comprises the explicitly modeled processes whose installation
and operation levels are decision variables. Background supply chains contain
no decision variables. Costs are assigned to the products purchased directly
across this boundary: the intermediate flows of the model. The boundary therefore
depends on which processes are modeled explicitly.

Prices are not propagated through the background supply chains. Their
inventories are used for the environmental assessment, while the market price of
a directly purchased product represents its full cost. Products exchanged
between foreground processes carry no price of their own: their production cost
enters through the background purchases of the processes producing them.

---

## Installation- and Operation-Driven Costs

A purchase is attributed to the decision driving it, following the same
installation/operation semantics as the environmental flows:

| Foreground exchange | Purchase scales with |
|---|---|
| `operation=True`, within the operation phase | Units of a vintage running in a given year (`var_operation`) |
| `operation=False`, or outside the operation phase | Installed units of a vintage (`var_installation`) |

Installation-driven purchases can include construction, replacements, and
end-of-life activities, depending on the temporal distributions. They need not
occur in the installation year, and the two groups are not equivalent to
financial CAPEX and OPEX.

A purchase is priced and discounted in the system year in which it occurs. For a
unit installed in `v`, an exchange at process time `tau` happens at `t = v + tau`:
a replacement ten years after a 2030 installation is priced and discounted in
2040, not in 2030.

A product has one price per year, whichever decision drives its purchase.

---

## Setting Market Prices

Prices are data on the background nodes, one price per node and background database. Price the background products that the foreground buys **directly**; their upstream supply chains need no prices.

`set_market_prices()` writes the prices from a table, with one row per node and database year:

```python
from optimex.economics import set_market_prices

background_databases = {
    2020: "ei312_REMIND-EU_SSP2_NDC_2020",
    2030: "ei312_REMIND-EU_SSP2_NDC_2030",
}
price_data = [
    {
        "name": "market group for electricity, medium voltage",
        "location": "RER",
        "product": "electricity, medium voltage",
        "unit": "kilowatt hour",
        "year": year,
        "price": price,  # constant 2020 EUR/kWh
    }
    for year, price in [(2020, 0.09), (2030, 0.07)]
]

set_market_prices(
    price_data=price_data,
    background_databases=background_databases,
    product_col="product",
    unit_col="unit",
)
```

- **Units:** each price refers to one unit of the node's reference product. Use one currency and price basis throughout: 90 EUR/MWh becomes 0.09 EUR/kWh for a product measured in kilowatt hours. No unit, currency, or inflation conversion is applied.
- **Every database:** set a price in each background database. Prospective databases, for example from `premise`, carry no prices themselves, and a product priced in some databases but not in others is rejected when the prices are [read](#reading-prices).
- **Matching:** rows identify nodes by name and location, optionally narrowed by reference product and unit. Codes are not used, because premise gives the same activity a different code in each scenario database.
- **Options:** `price_data` can also be a pandas DataFrame; `name_col`, `location_col`, `product_col`, `unit_col`, `year_col`, and `price_col` map custom column names. `overwrite=False` keeps existing prices, `strict=False` turns unresolvable rows into warnings instead of errors, and `price_attribute` changes the node attribute the price is written to (default `"market_price"`).
- **Uncertainty:** the Brightway uncertainty columns `uncertainty type`, `loc`, `scale`, `shape`, `minimum`, and `maximum` add a price distribution, e.g. `"uncertainty type": 2, "scale": 0.2` for a lognormal spread around the price. A missing `loc` is derived from the price. The optimization uses the prices themselves; a [Monte Carlo analysis](uncertainty.md#prices) samples the distributions.

To find the products that need a price, list the background inputs of the foreground processes:

```python
purchased = {
    (exc.input["name"], exc.input["location"], exc.input["reference product"])
    for process in bd.Database("foreground")
    for exc in process.technosphere()
    if exc.input["database"] != "foreground"
}
```

---

## Reading Prices

Prices are only read when `LCAConfig.price_attribute` names the node attribute holding them, so impact-only studies are unaffected:

```python
config = lca_processor.LCAConfig(
    ...,
    price_attribute="market_price",  # attribute written by set_market_prices
)
```

The processor then resolves each directly purchased background product in every background database, the same way as for the inventories, and interpolates its prices to every system year with the same weights: linearly between database years, with the nearest database used outside that range. The result is available as `lca_data.intermediate_flow_prices`, keyed by `(intermediate flow, year)`, and is passed on to the optimization inputs.

Missing prices are handled conservatively:

- A product without a price in **any** database is left out and treated as free. All such products are listed in a single warning: check it, since a free purchase is rarely intended.
- A product priced in **some** databases but not in others raises a `ValueError`, because interpolating with the missing price would silently pull the price towards zero.

### Prices Set by Hand

Prices can also be passed to the optimization inputs directly, without the
background nodes, for example for a quick test or a price scenario:

```python
model_inputs = manager.override(
    intermediate_flow_prices={("electricity_code", 2030): 0.08, ...}
)
```

Keys are `(intermediate flow, system year)`; missing entries are treated as free.

---

## Modeling Assumptions and Limits

- **Allocation:** Costs follow the LCI exchange amounts and therefore inherit the
  allocation of the inventory. The extension adds no separate allocation setting.
- **Linear costs:** At fixed prices, costs scale proportionally with continuous
  installation and operation levels. Economies of scale, indivisible plant sizes,
  and minimum plant capacities are not represented.
- **Time horizon:** The objective sums costs over `SYSTEM_TIME`. Purchases
  outside it do not enter the total; choose the horizon and temporal
  distributions to cover the costs that matter.
- **Coverage:** The objective covers the priced purchases of the modeled
  processes. It is not a complete financial account: fixed operating costs, for
  example, are only included if modeled as purchases.
