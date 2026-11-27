"""
Fast rebuilds of a linear program for inputs that differ only in their data.

Building the formulation (`optimex.formulation.build_formulation`) walks every
process, flow, vintage, and year in Python. Monte Carlo samples, however, share
everything except a few data fields, and the formulation depends on those fields
in a simple way:

- A foreground exchange sampled ``f`` times its deterministic amount scales all
  of its flow coefficients by ``f``: the rows of the flow term matrices are
  scaled.
- Background inventories, characterization factors, and prices only change the
  weights with which the flow terms enter impacts and costs.
- Demand, impact limits, and deployment and operation limits change right-hand
  sides, bounds, or which limit rows exist.

`IncrementalFormulation` builds the formulation of the deterministic inputs once,
keeps its flow terms as sparse matrices, and assembles the linear program of each
sample from them with NumPy and SciPy (`LPArrays`, solved by
`optimex.persistent.PersistentModel`). A sample that changes anything else, or
whose foreground exchanges do not change by one factor per exchange, is reported
as unsupported, and the caller builds its full formulation instead.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from optimex.converter import OptimizationModelInputs
from optimex.formulation import INFINITY, build_formulation
from optimex.persistent import SENSE_CODES, LPArrays

# Foreground tensors whose entries may change by one factor per (process, flow)
_TENSORS = {
    "tech": ("foreground_technosphere", "foreground_technosphere_vintages"),
    "bio": ("foreground_biosphere", "foreground_biosphere_vintages"),
    "intd": ("internal_demand_technosphere", None),
    "prod": ("foreground_production", "foreground_production_vintages"),
}
# Other fields a sample may change
_DATA_FIELDS = {
    "background_inventory",
    "characterization",
    "intermediate_flow_prices",
    "discount_rate",
    "discount_reference_year",
    "demand",
    "process_deployment_limits_max",
    "process_deployment_limits_min",
    "process_deployment_limits_max_default",
    "process_deployment_limits_min_default",
    "process_operation_limits_max",
    "process_operation_limits_min",
    "process_operation_limits_max_default",
    "process_operation_limits_min_default",
    "category_impact_limits",
    "cumulative_category_impact_limits",
}
# Derived from the tensors, vintages, and improvements; recomputed by scaling
_DERIVED_FIELDS = {
    "foreground_technosphere_vintage_overrides",
    "foreground_biosphere_vintage_overrides",
    "foreground_production_vintage_overrides",
}
# Blocks that depend on sample data; all others are copied from the base
_DYNAMIC_BLOCKS = {
    "ProductDemandFulfillment",
    "CategoryImpactLimits",
    "CumulativeCategoryImpactLimits",
    "ProcessOperationLimitMax",
    "ProcessOperationLimitMin",
}
_FLOW_LIMITS = (
    "flow_limits_max",
    "flow_limits_min",
    "cumulative_flow_limits_max",
    "cumulative_flow_limits_min",
)


class _Terms:
    """Flow terms ``{(process, flow, time): {variable: coefficient}}`` as a CSR matrix."""

    def __init__(self, terms: dict, n: int):
        self.keys = list(terms)
        starts, indices, values = [0], [], []
        for key in self.keys:
            row = terms[key]
            indices.extend(row.keys())
            values.extend(row.values())
            starts.append(len(indices))
        self.matrix = sp.csr_matrix(
            (np.asarray(values, float), np.asarray(indices, np.int64), np.asarray(starts, np.int64)),
            shape=(len(self.keys), n),
        )
        self.pairs = [(p, x) for p, x, _ in self.keys]
        self.times = np.array([t for _, _, t in self.keys], dtype=np.int64)

    def row_factors(self, factors: Dict[tuple, float]) -> np.ndarray:
        if not factors:
            return np.ones(len(self.keys))
        return np.array([factors.get(pair, 1.0) for pair in self.pairs])


def _pair_factors(base: dict, sample: dict) -> Optional[Dict[tuple, float]]:
    """
    One factor per (process, flow) such that sample = factor * base, or None.

    Entries that are 0 in both are ignored; a changed key set or a pair whose
    entries change by different factors returns None.
    """
    if sample is base:
        return {}
    if base is None or sample is None or len(base) != len(sample):
        return None
    factors = {}
    for key, old in base.items():
        new = sample.get(key)
        if new is None:
            return None
        if new == old:
            if old == 0:
                continue
            ratio = 1.0
        elif old == 0:
            return None
        else:
            ratio = new / old
        pair = key[:2]
        previous = factors.get(pair)
        if previous is None:
            factors[pair] = ratio
        elif not math.isclose(previous, ratio, rel_tol=1e-9, abs_tol=1e-15):
            return None
    return {pair: f for pair, f in factors.items() if f != 1.0}


def _same(a, b) -> bool:
    if a is b:
        return True
    try:
        return a == b
    except Exception:  # pragma: no cover - unusual types
        return False


@dataclass
class SampleLP:
    """The linear program of a sample, plus what reporting needs."""

    lp: LPArrays
    impact_vectors: Dict[str, np.ndarray]  # scaled total impact per category
    cost_vector: np.ndarray  # scaled, discounted total cost


class IncrementalFormulation:
    """
    The formulation of a base set of inputs, rebuilt quickly for samples of it.

    Parameters
    ----------
    base : OptimizationModelInputs
        The deterministic inputs. All samples share their scaling.
    objective, objective_category, aggregate_vintages :
        As in `optimizer.create_model`.
    """

    def __init__(
        self,
        base: OptimizationModelInputs,
        objective: str = "environmental",
        objective_category: Optional[str] = None,
        aggregate_vintages: bool = False,
    ):
        f = build_formulation(
            base,
            objective_category=objective_category,
            objective=objective,
            aggregate_vintages=aggregate_vintages,
        )
        self.base = base
        self.formulation = f
        self.scales = f.scales
        self.objective = objective
        self.objective_category = objective_category
        self.n = n = f.n_variables

        expressions = f.expressions
        self.terms = {
            "tech": _Terms(expressions["scaled_technosphere_flow"], n),
            "bio": _Terms(expressions["scaled_biosphere_flow"], n),
            "intd": _Terms(expressions["scaled_internal_demand_flow"], n),
            "prod": _Terms(expressions["scaled_production_flow"], n),
        }

        # Index maps of the data arrays
        self.times = list(base.SYSTEM_TIME)
        self.time_index = {t: i for i, t in enumerate(self.times)}
        self.categories = list(base.CATEGORY)
        self.flow_index = {x: i for i, x in enumerate(base.INTERMEDIATE_FLOW)}
        self.emission_index = {x: i for i, x in enumerate(base.ELEMENTARY_FLOW)}
        self.db_index = {x: i for i, x in enumerate(base.BACKGROUND_ID)}
        self.product_index = {x: i for i, x in enumerate(base.PRODUCT)}

        tech, bio = self.terms["tech"], self.terms["bio"]
        self._tech_t = np.array([self.time_index[t] for t in tech.times], dtype=np.int64)
        self._tech_i = np.array([self.flow_index[x] for _, x in tech.pairs], dtype=np.int64)
        self._bio_t = np.array([self.time_index[t] for t in bio.times], dtype=np.int64)
        self._bio_e = np.array([self.emission_index[x] for _, x in bio.pairs], dtype=np.int64)

        # Demand rows: (product, time) -> row; production adds, internal demand subtracts
        n_rt = len(self.product_index) * len(self.times)
        self._demand_keys = [(r, t) for r in base.PRODUCT for t in self.times]
        self._prod_rows = np.array(
            [self.product_index[r] * len(self.times) + self.time_index[t]
             for _, r, t in self.terms["prod"].keys], dtype=np.int64,
        )
        self._intd_rows = np.array(
            [self.product_index[r] * len(self.times) + self.time_index[t]
             for _, r, t in self.terms["intd"].keys], dtype=np.int64,
        )
        self._n_rt = n_rt

        # Total operation of each active (process, time), for operation limits
        operation_vars = f.variable_index["var_operation"]
        total_vars = f.variable_index["var_operation_total"]
        self._operation_keys = list(f.active_vintages_by_process_time)
        starts, indices = [0], []
        for p, t in self._operation_keys:
            if p in f.aggregated_processes:
                indices.append(total_vars[p, t])
            else:
                indices.extend(
                    operation_vars[p, v, t] for v in f.active_vintages_by_process_time[p, t]
                )
            starts.append(len(indices))
        self._operation_totals = sp.csr_matrix(
            (np.ones(len(indices)), np.asarray(indices, np.int64), np.asarray(starts, np.int64)),
            shape=(len(self._operation_keys), n),
        )

        # Static blocks, copied as they are
        starts, indices, values, senses, rhs = [0], [], [], [], []
        self._static_infeasible = False
        self._static_signature = []
        for name, block in f.constraints.items():
            if name in _DYNAMIC_BLOCKS:
                continue
            count = 0
            for terms, value in block.rows.values():
                if not terms:
                    self._static_infeasible = True
                    continue
                indices.extend(terms.keys())
                values.extend(terms.values())
                starts.append(len(indices))
                senses.append(SENSE_CODES[block.sense])
                rhs.append(value)
                count += 1
            self._static_signature.append((name, count))
        self._static = sp.csr_matrix(
            (np.asarray(values, float), np.asarray(indices, np.int64), np.asarray(starts, np.int64)),
            shape=(len(senses), n),
        )
        self._static_senses = np.asarray(senses, np.int8)
        self._static_rhs = np.asarray(rhs, float)

        installation = f.variable_index["var_installation"]
        self._installation_keys = list(installation)
        self._installation_index = np.array(list(installation.values()), dtype=np.int64)
        self._lb = np.array([0.0 if v is None else v for v in f.lower], float)
        self._ub = np.array([math.inf if v is None else v for v in f.upper], float)

        self._impact_cache = None  # (background, characterization, mapping) -> F, C

    # -- support -----------------------------------------------------------------

    def unsupported(self, inputs: OptimizationModelInputs) -> Optional[str]:
        """Why `inputs` cannot be rebuilt incrementally, or None."""
        base = self.base
        for name in OptimizationModelInputs.model_fields:
            if name in _DATA_FIELDS or name in _DERIVED_FIELDS:
                continue
            if any(name in fields for fields in _TENSORS.values()):
                continue
            if not _same(getattr(inputs, name), getattr(base, name)):
                return f"'{name}' differs from the base inputs"
        if any(getattr(inputs, name) or getattr(base, name) for name in _FLOW_LIMITS):
            return "flow limits are not supported incrementally"
        return None

    def _factors(self, inputs) -> Optional[Dict[str, Dict[tuple, float]]]:
        factors = {}
        for family, (tensor, vintages) in _TENSORS.items():
            pair_factors = _pair_factors(getattr(self.base, tensor), getattr(inputs, tensor))
            if pair_factors is None:
                return None
            if vintages is not None:
                base_v, sample_v = getattr(self.base, vintages), getattr(inputs, vintages)
                if base_v or sample_v:
                    vintage_factors = _pair_factors(base_v or {}, sample_v or {})
                    if vintage_factors is None:
                        return None
                    for pair, factor in vintage_factors.items():
                        if not math.isclose(pair_factors.get(pair, factor), factor, rel_tol=1e-9):
                            return None
                        pair_factors[pair] = factor
                    # A pair with vintage amounts must change by the same factor in both
                    for pair in {k[:2] for k in base_v or {}}:
                        if pair_factors.get(pair, 1.0) != vintage_factors.get(pair, 1.0):
                            return None
            factors[family] = pair_factors
        return factors

    # -- assembly ----------------------------------------------------------------

    def _impact_arrays(self, inputs):
        key = (id(inputs.background_inventory), id(inputs.characterization), id(inputs.mapping))
        if self._impact_cache is not None and self._impact_cache[0] == key:
            return self._impact_cache[1]
        scales = self.scales
        n_db, n_i = len(self.db_index), len(self.flow_index)
        n_e, n_t, n_c = len(self.emission_index), len(self.times), len(self.categories)
        inventory = np.zeros((n_db, n_i, n_e))
        for (db, flow, emission), amount in inputs.background_inventory.items():
            if emission in self.emission_index and flow in self.flow_index:
                inventory[self.db_index[db], self.flow_index[flow], self.emission_index[emission]] += amount
        mapping = np.zeros((n_db, n_t))
        for (db, t), weight in inputs.mapping.items():
            if t in self.time_index and db in self.db_index:
                mapping[self.db_index[db], self.time_index[t]] = weight
        characterization = np.zeros((n_c, n_e, n_t))
        category_index = {c: i for i, c in enumerate(self.categories)}
        for (c, emission, t), factor in inputs.characterization.items():
            if emission in self.emission_index and t in self.time_index:
                characterization[category_index[c], self.emission_index[emission], self.time_index[t]] = (
                    factor / scales["characterization"].get(c, 1.0)
                )
        mapped = np.einsum("dt,die->iet", mapping, inventory)
        flow_factors = np.einsum("cet,iet->cti", characterization, mapped)
        result = (flow_factors, characterization)
        self._impact_cache = (key, result)
        return result

    def build(self, inputs: OptimizationModelInputs) -> Optional[SampleLP]:
        """The linear program of `inputs`, or None if it is not supported."""
        if self.unsupported(inputs) is not None:
            return None
        factors = self._factors(inputs)
        if factors is None:
            return None
        scales = self.scales
        fg = scales["foreground"]
        n = self.n
        tech, bio = self.terms["tech"], self.terms["bio"]
        s_tech = tech.row_factors(factors["tech"])
        s_bio = bio.row_factors(factors["bio"])

        # Impacts: weights of the flow term rows per category
        flow_factors, characterization = self._impact_arrays(inputs)
        tech_t = tech.matrix.T.tocsr() if not hasattr(self, "_tech_T") else self._tech_T
        self._tech_T = tech_t
        bio_t = bio.matrix.T.tocsr() if not hasattr(self, "_bio_T") else self._bio_T
        self._bio_T = bio_t
        impact_weights, impact_vectors = {}, {}
        for ci, c in enumerate(self.categories):
            w_tech = flow_factors[ci, self._tech_t, self._tech_i] * s_tech
            w_bio = characterization[ci, self._bio_e, self._bio_t] * s_bio
            impact_weights[c] = (w_tech, w_bio)
            impact_vectors[c] = tech_t @ w_tech + bio_t @ w_bio

        # Costs
        prices = np.zeros((len(self.flow_index), len(self.times)))
        cost_scale = scales.get("cost", 1.0)
        for (flow, t), price in (inputs.intermediate_flow_prices or {}).items():
            if flow in self.flow_index and t in self.time_index:
                prices[self.flow_index[flow], self.time_index[t]] = price / cost_scale
        rate = inputs.discount_rate
        reference = inputs.discount_reference_year
        if reference is None:
            reference = min(self.times)
        discount = np.array([(1 + rate) ** -(t - reference) for t in self.times])
        w_cost = prices[self._tech_i, self._tech_t] * discount[self._tech_t] * s_tech
        cost_vector = tech_t @ w_cost

        objective_scale = scales["objective"]
        if self.objective == "cost":
            c_vec = cost_vector / objective_scale
        else:
            c_vec = impact_vectors[self.objective_category] / objective_scale

        blocks = [(self._static, self._static_senses, self._static_rhs)]
        signature = list(self._static_signature)
        infeasible = self._static_infeasible

        # Demand: production minus internal demand per (product, time)
        prod, intd = self.terms["prod"], self.terms["intd"]
        aggregate_prod = sp.csr_matrix(
            (prod.row_factors(factors["prod"]), (self._prod_rows, np.arange(len(prod.keys)))),
            shape=(self._n_rt, len(prod.keys)),
        )
        aggregate_intd = sp.csr_matrix(
            (-intd.row_factors(factors["intd"]), (self._intd_rows, np.arange(len(intd.keys)))),
            shape=(self._n_rt, len(intd.keys)),
        )
        demand_matrix = (aggregate_prod @ prod.matrix + aggregate_intd @ intd.matrix).tocsr()
        demand = inputs.demand or {}
        demand_rhs = np.array([demand.get(key, 0) / fg for key in self._demand_keys])
        empty = np.diff(demand_matrix.indptr) == 0
        if np.any(empty & (demand_rhs != 0)):
            infeasible = True
        keep = ~empty
        blocks.append((demand_matrix[keep], np.full(keep.sum(), 2, np.int8), demand_rhs[keep]))
        signature.append(("ProductDemandFulfillment", int(keep.sum())))

        # Impact limits
        limit_rows, limit_rhs = [], []
        fc_scale = scales["characterization"]
        for (c, t), limit in (inputs.category_impact_limits or {}).items():
            if c not in impact_weights or t not in self.time_index:
                continue
            w_tech, w_bio = impact_weights[c]
            ti = self.time_index[t]
            row = tech_t @ np.where(self._tech_t == ti, w_tech, 0.0) + bio_t @ np.where(
                self._bio_t == ti, w_bio, 0.0
            )
            limit_rows.append(row)
            limit_rhs.append(limit / (fg * fc_scale.get(c, 1.0)))
        if limit_rows:
            blocks.append((sp.csr_matrix(np.vstack(limit_rows)), np.zeros(len(limit_rows), np.int8),
                           np.asarray(limit_rhs)))
        signature.append(("CategoryImpactLimits", len(limit_rows)))
        cumulative_rows, cumulative_rhs = [], []
        for c, limit in (inputs.cumulative_category_impact_limits or {}).items():
            if c in impact_vectors and limit < INFINITY:
                cumulative_rows.append(impact_vectors[c])
                cumulative_rhs.append(limit / (fg * fc_scale.get(c, 1.0)))
        if cumulative_rows:
            blocks.append((sp.csr_matrix(np.vstack(cumulative_rows)),
                           np.zeros(len(cumulative_rows), np.int8), np.asarray(cumulative_rhs)))
        signature.append(("CumulativeCategoryImpactLimits", len(cumulative_rows)))

        # Operation limits on the total over vintages, only where they can bind
        for name, limits, default, sense, binding in (
            ("ProcessOperationLimitMax", inputs.process_operation_limits_max or {},
             inputs.process_operation_limits_max_default, 0, lambda v: v < INFINITY),
            ("ProcessOperationLimitMin", inputs.process_operation_limits_min or {},
             inputs.process_operation_limits_min_default, 1, lambda v: v > 0),
        ):
            values = np.array([limits.get(key, default) for key in self._operation_keys], float)
            mask = np.array([binding(v) for v in values], dtype=bool) if len(values) else np.zeros(0, bool)
            if mask.any():
                blocks.append((self._operation_totals[mask], np.full(mask.sum(), sense, np.int8), values[mask]))
            signature.append((name, int(mask.sum())))

        # Bounds: deployment limits on the installation variables
        lb, ub = self._lb, self._ub
        if not (
            inputs.process_deployment_limits_max is self.base.process_deployment_limits_max
            and inputs.process_deployment_limits_min is self.base.process_deployment_limits_min
            and inputs.process_deployment_limits_max_default == self.base.process_deployment_limits_max_default
            and inputs.process_deployment_limits_min_default == self.base.process_deployment_limits_min_default
        ):
            lb, ub = lb.copy(), ub.copy()
            maximum = inputs.process_deployment_limits_max or {}
            minimum = inputs.process_deployment_limits_min or {}
            max_default = inputs.process_deployment_limits_max_default
            min_default = inputs.process_deployment_limits_min_default
            upper = np.array([maximum.get(k, max_default) for k in self._installation_keys], float)
            lower = np.array([minimum.get(k, min_default) for k in self._installation_keys], float)
            ub[self._installation_index] = np.where(np.isfinite(upper), upper, math.inf)
            lb[self._installation_index] = np.maximum(lower, 0.0)

        A = sp.vstack([block[0] for block in blocks], format="csr")
        lp = LPArrays(
            c=c_vec,
            A=A,
            senses=np.concatenate([block[1] for block in blocks]),
            rhs=np.concatenate([block[2] for block in blocks]).astype(float),
            lb=lb,
            ub=ub,
            row_keys=signature,
            infeasible=infeasible,
        )
        return SampleLP(lp, impact_vectors, cost_vector)

    # -- results -----------------------------------------------------------------

    def objective_value(self, sample: SampleLP, x: np.ndarray) -> float:
        """The objective in real units."""
        scales = self.scales
        if self.objective == "cost":
            return float(sample.cost_vector @ x) * scales["foreground"] * scales.get("cost", 1.0)
        c = self.objective_category
        return float(sample.impact_vectors[c] @ x) * scales["foreground"] * scales[
            "characterization"
        ].get(c, 1.0)

    def totals(self, sample: SampleLP, x: np.ndarray, priced: bool):
        """Cumulative impact per category and discounted total cost, in real units."""
        scales = self.scales
        fg = scales["foreground"]
        impacts = {
            c: float(vector @ x) * fg * scales["characterization"].get(c, 1.0)
            for c, vector in sample.impact_vectors.items()
        }
        cost = float(sample.cost_vector @ x) * fg * scales.get("cost", 1.0) if priced else None
        return impacts, cost

    def installation(self, x: np.ndarray) -> pd.DataFrame:
        """Installed units, as from `PostProcessor.get_installation()`."""
        series = pd.Series(
            x[self._installation_index],
            index=pd.MultiIndex.from_tuples(self._installation_keys, names=["Process", "Time"]),
        )
        return series.unstack("Process")

    def model(self, x: np.ndarray, name: str = "model"):
        """
        A solved `LinopyModel` for the variables of `x`. Only variable-based
        results are valid (installation, operation): its expressions are those
        of the base inputs.
        """
        from optimex.linopy_backend import LinopyModel

        model = LinopyModel(name, None, self.formulation, [])
        model.load_solution(x)
        return model
