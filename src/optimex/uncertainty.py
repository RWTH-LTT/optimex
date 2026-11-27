"""
Uncertainty propagation by Monte Carlo sampling of the optimization inputs.

The uncertainty described in the Brightway databases is carried over to the
optimization by drawing complete sets of optimization inputs, one per Monte
Carlo sample, and solving or evaluating each of them:

1. `UncertaintySampler` collects the uncertain quantities of a product system
   and draws `OptimizationModelInputs` from them (`SampleSet`). Four sources are
   supported:

   - ``"exchanges"``: uncertainty distributions of the foreground exchanges,
   - ``"parameters"``: uncertain Brightway parameters (project, database, and
     activity parameters) and the foreground exchange formulas that use them,
   - ``"background"``: uncertainty of the background databases (a Brightway
     Monte Carlo run per database),
   - ``"prices"``: price distributions written with
     `optimex.economics.set_market_prices`.

2. `solve_samples` finds the optimum of every sample: how much the optimal
   pathway and its objective vary with the inputs.

3. `evaluate_design` fixes a design, e.g. the deterministic optimum, and
   evaluates it in every sample: how robust a given pathway is.

Sampling uses Brightway's own machinery wherever the data live in matrices:
foreground exchanges are drawn by `matrix_utils` from the processed datapackage
of the foreground database (as `bw2calc` does with ``use_distributions=True``),
parameters with `bw2parameters`' ``evaluate_monte_carlo`` and passed on as a
`bw_processing` array datapackage, and background databases with a
`bw2calc` Monte Carlo LCA. Extra datapackages, e.g. pre-sampled correlated
arrays, can be layered on top of the foreground and the background.

Sampled quantities are correlated where they describe the same thing. An
exchange or parameter is drawn once per sample and applies to every process
time, vintage, and system year. Prices are drawn as one quantile per product and
sample, shared by all background databases, so a price that is high in one
database is high in all of them. Background databases are sampled with common
random numbers, which correlates databases of the same structure.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import bw2data as bd
import numpy as np
import pandas as pd
from loguru import logger
from stats_arrays import UncertaintyBase, uncertainty_choices
from tqdm import tqdm

from optimex import lca_processor
from optimex.converter import OptimizationModelInputs
from optimex.postprocessing import PostProcessor
from optimex.series import OUTPUTS, SeriesResults, _solve_all  # noqa: F401

SOURCES = ("exchanges", "parameters", "background", "prices")

# Brightway's uncertainty fields and their `stats_arrays` column names
_UNCERTAINTY_FIELDS = {
    "uncertainty type": "uncertainty_type",
    "loc": "loc",
    "scale": "scale",
    "shape": "shape",
    "minimum": "minimum",
    "maximum": "maximum",
    "negative": "negative",
}
_NO_UNCERTAINTY = {0, 1}  # undefined, no uncertainty
_LOGNORMAL = 2
_EPS = 1e-12  # keeps quantiles away from 0 and 1, where many ppfs are infinite


# ----------------------------------------------------------------------------
# Distributions
# ----------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _uncertainty_type(data: Mapping) -> int:
    value = data.get("uncertainty type", data.get("uncertainty_type"))
    return 0 if _is_missing(value) else int(value)


def is_uncertain(data: Mapping) -> bool:
    """Whether a Brightway exchange, parameter, or price has a distribution."""
    return _uncertainty_type(data) not in _NO_UNCERTAINTY


def _params_array(data: Mapping, amount: float) -> np.ndarray:
    """One-row `stats_arrays` parameter array for a Brightway uncertainty dict."""
    row = {"uncertainty_type": _uncertainty_type(data)}
    for bw_name, sa_name in _UNCERTAINTY_FIELDS.items():
        if sa_name == "uncertainty_type":
            continue
        value = data.get(bw_name, data.get(sa_name))
        if not _is_missing(value):
            row[sa_name] = value
    if "loc" not in row:
        if row["uncertainty_type"] == _LOGNORMAL:
            row["loc"] = math.log(abs(amount))
        else:
            row["loc"] = amount
    if row["uncertainty_type"] == _LOGNORMAL and "negative" not in row:
        row["negative"] = amount < 0
    params = UncertaintyBase.from_dicts(row)
    uncertainty_choices[row["uncertainty_type"]].validate(params)
    return params


def sample_distribution(data: Mapping, amount: float, quantiles) -> np.ndarray:
    """
    Values of a Brightway uncertainty distribution at the given quantiles.

    Parameters
    ----------
    data : mapping
        Brightway uncertainty fields (``uncertainty type``, ``loc``, ``scale``,
        ``shape``, ``minimum``, ``maximum``, ``negative``). A missing ``loc`` is
        derived from `amount`, as in `optimex.economics.set_market_prices`.
    amount : float
        The deterministic value.
    quantiles : array-like
        Quantiles in (0, 1).

    Returns
    -------
    numpy.ndarray
        One value per quantile. ``minimum`` and ``maximum`` truncate the
        distribution, as in Brightway's Monte Carlo.
    """
    quantiles = np.clip(np.asarray(quantiles, dtype=float), _EPS, 1 - _EPS)
    if not is_uncertain(data):
        return np.full(quantiles.shape, float(amount))

    params = _params_array(data, amount)
    distribution = uncertainty_choices[int(params["uncertainty_type"][0])]
    negative = bool(params["negative"][0]) and distribution.id == _LOGNORMAL
    if negative:
        # Lognormal distributions describe the magnitude; their bounds are signed.
        params = params.copy()
        params["negative"] = False
        minimum, maximum = params["minimum"][0], params["maximum"][0]
        params["minimum"], params["maximum"] = -maximum, -minimum

    low = high = None
    try:
        if np.isfinite(params["minimum"][0]):
            low = float(distribution.cdf(params, params["minimum"].reshape(1, 1).copy())[0, 0])
        if np.isfinite(params["maximum"][0]):
            high = float(distribution.cdf(params, params["maximum"].reshape(1, 1).copy())[0, 0])
    except NotImplementedError:
        pass
    low = 0.0 if low is None else low
    high = 1.0 if high is None else high
    quantiles = low + quantiles * (high - low)

    try:
        values = distribution.ppf(params, quantiles.reshape(1, -1)).reshape(-1)
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"Sampling {distribution.description} distributions is not supported."
        ) from exc
    minimum, maximum = params["minimum"][0], params["maximum"][0]
    values = np.clip(
        values,
        minimum if np.isfinite(minimum) else -np.inf,
        maximum if np.isfinite(maximum) else np.inf,
    )
    return -values if negative else values


def uncertainty_from_record(record: Mapping, amount: float) -> Optional[dict]:
    """
    The Brightway uncertainty dict described by a record, or None.

    Reads the columns ``uncertainty type``, ``loc``, ``scale``, ``shape``,
    ``minimum``, ``maximum``, and ``negative``. Without an uncertainty type, or
    with type 0 or 1, the record has no distribution.

    Raises
    ------
    ValueError
        If the distribution is invalid, e.g. a normal distribution without scale.
    """
    if not is_uncertain(record):
        return None
    data = {
        bw_name: record[bw_name]
        for bw_name in _UNCERTAINTY_FIELDS
        if bw_name in record and not _is_missing(record[bw_name])
    }
    data["uncertainty type"] = _uncertainty_type(record)
    try:
        params = _params_array(data, amount)
    except Exception as exc:
        raise ValueError(f"invalid uncertainty distribution {data}: {exc}") from exc
    result = {"uncertainty type": int(params["uncertainty_type"][0])}
    for bw_name, sa_name in _UNCERTAINTY_FIELDS.items():
        if sa_name in ("uncertainty_type", "negative"):
            continue
        value = float(params[sa_name][0])
        if not math.isnan(value):
            result[bw_name] = value
    if bool(params["negative"][0]):
        result["negative"] = True
    return result


# ----------------------------------------------------------------------------
# Uncertain quantities
# ----------------------------------------------------------------------------

# Tensors an exchange enters, by Brightway edge type and whether its input is a
# foreground product. Each tensor also has vintage-specific variants.
_TENSOR_BY_EDGE = {
    (bd.labels.production_edge_default, True): "foreground_production",
    (bd.labels.consumption_edge_default, True): "internal_demand_technosphere",
    (bd.labels.consumption_edge_default, False): "foreground_technosphere",
    (bd.labels.biosphere_edge_default, False): "foreground_biosphere",
}


def _tensor_names(tensor: str) -> Tuple[str, ...]:
    if tensor == "internal_demand_technosphere":
        return (tensor,)
    return (tensor, f"{tensor}_vintages", f"{tensor}_vintage_overrides")


@dataclass(frozen=True)
class _ExchangeTarget:
    """The tensor entries of one foreground exchange: ``tensor[process, flow, ...]``."""

    tensor: str
    process: str
    flow: str


@dataclass(frozen=True)
class _MatrixEntry:
    """Where a foreground exchange sits in the Brightway matrices."""

    matrix: str
    row: int  # input node id
    col: int  # output node id
    flip: bool


def _matrix_entry(edge_type: str, input_id: int, output_id: int) -> _MatrixEntry:
    if edge_type == bd.labels.biosphere_edge_default:
        return _MatrixEntry("biosphere_matrix", input_id, output_id, False)
    flip = edge_type != bd.labels.production_edge_default
    return _MatrixEntry("technosphere_matrix", input_id, output_id, flip)


@dataclass
class _Quantity:
    source: str
    label: str
    distribution: dict
    amount: float


class _ParameterModel:
    """
    Brightway parameters and the foreground exchange formulas that use them.

    Formulas are evaluated in memory, with the same scoping as
    `bw2data.parameters`: activity parameters of the exchange's group, then the
    groups in its ``Group.order``, then database and project parameters. The
    databases are never modified.
    """

    def __init__(self, foreground_db: str, processes: Iterable[str]):
        from bw2data.backends import ExchangeDataset
        from bw2data.parameters import (
            ActivityParameter,
            DatabaseParameter,
            Group,
            ParameterizedExchange,
            ProjectParameter,
        )

        processes = set(processes)
        formulas = {pe.exchange: (pe.group, pe.formula) for pe in ParameterizedExchange.select()}
        self.exchanges = []  # (exchange id, group, formula, target, stored amount)
        self.entries: Dict[int, _MatrixEntry] = {}
        if formulas:
            rows = (
                ExchangeDataset.select()
                .where(ExchangeDataset.id << list(formulas))
                .order_by(ExchangeDataset.id)
            )
            for exc in rows:
                if exc.output_database != foreground_db or exc.output_code not in processes:
                    continue
                tensor = _TENSOR_BY_EDGE.get(
                    (exc.type, exc.input_database == foreground_db)
                )
                if tensor is None:
                    continue
                group, formula = formulas[exc.id]
                target = _ExchangeTarget(tensor, exc.output_code, exc.input_code)
                self.exchanges.append(
                    (exc.id, group, formula, target, float(exc.data.get("amount", 0.0)))
                )
                self.entries[exc.id] = _matrix_entry(
                    exc.type,
                    bd.get_node(database=exc.input_database, code=exc.input_code).id,
                    bd.get_node(database=exc.output_database, code=exc.output_code).id,
                )

        # The groups the formulas need, with their dependencies
        self.order, self.group_db = {}, {}
        pending = [group for _, group, _, _, _ in self.exchanges]
        while pending:
            group = pending.pop()
            if group in self.order:
                continue
            try:
                self.order[group] = list(Group.get(name=group).order)
            except Group.DoesNotExist:
                self.order[group] = []
            first = ActivityParameter.get_or_none(ActivityParameter.group == group)
            self.group_db[group] = first.database if first else foreground_db
            pending.extend(self.order[group])

        self.project = ProjectParameter.load() if self.exchanges else {}
        self.database = {
            db: DatabaseParameter.load(db) for db in set(self.group_db.values())
        }
        self.groups = {group: ActivityParameter.load(group) for group in self.order}

        # Independent uncertain parameters: no formula, with a distribution
        self.quantities = {}
        scopes = [(("project",), self.project)]
        scopes += [(("database", db), data) for db, data in self.database.items()]
        scopes += [(("activity", group), data) for group, data in self.groups.items()]
        for scope, data in scopes:
            for name, values in data.items():
                if values.get("formula") or not is_uncertain(values):
                    continue
                self.quantities[scope + (name,)] = _Quantity(
                    "parameters",
                    f"{':'.join(scope)}:{name}",
                    values,
                    float(values["amount"]),
                )

        deterministic = self.evaluate()
        for exchange_id, _, formula, target, stored in self.exchanges:
            if not math.isclose(deterministic[exchange_id], stored, rel_tol=1e-9, abs_tol=1e-12):
                logger.warning(
                    f"The exchange {target.process} <- {target.flow} has the amount "
                    f"{stored}, but its formula '{formula}' gives "
                    f"{deterministic[exchange_id]}. Recalculate the parameters "
                    "(`bd.parameters.recalculate()`) before building the inputs."
                )

    def evaluate(self) -> Dict[int, float]:
        """Deterministic exchange amounts, by exchange id."""
        return self._evaluate(None)[1]

    def monte_carlo(
        self, n_samples: int, seed: Optional[int] = None
    ) -> Tuple[Dict[tuple, np.ndarray], Dict[int, np.ndarray]]:
        """
        Monte Carlo samples of the parameters and the exchange amounts.

        Uses `bw2parameters.ParameterSet.evaluate_monte_carlo`, scope by scope,
        so a parameter is drawn once and its samples propagate through every
        formula that uses it.

        Returns
        -------
        parameters : dict
            ``{(scope..., name): samples}`` of the independent uncertain parameters.
        amounts : dict
            ``{exchange id: samples}`` of the parameterized exchanges.
        """
        from stats_arrays import MCRandomNumberGenerator

        keys = sorted(self.quantities)
        drawn = {}
        if keys:
            # The same generator `matrix_utils` uses for exchanges. Drawing here
            # rather than in `evaluate_monte_carlo` makes the samples seedable.
            params = np.hstack(
                [_params_array(self.quantities[k].distribution, self.quantities[k].amount) for k in keys]
            )
            samples = MCRandomNumberGenerator(params, seed=seed).generate(n_samples)
            samples = samples.reshape(len(keys), n_samples)
            drawn = {key: samples[row].copy() for row, key in enumerate(keys)}
        _, amounts = self._evaluate(n_samples, drawn)
        return drawn, amounts

    def _evaluate(self, n_samples: Optional[int], drawn: Optional[Mapping] = None):
        import copy

        from asteval import Interpreter
        from bw2parameters import ParameterSet

        drawn = drawn or {}

        def evaluate_set(data, global_params, scope):
            if not data:
                return {}
            # Drawn parameters enter as given samples, the rest is propagated
            given = {n: drawn[scope + (n,)] for n in data if scope + (n,) in drawn}
            params = {n: v for n, v in copy.deepcopy(data).items() if n not in given}
            global_params = {k: v for k, v in global_params.items() if k not in data}
            global_params.update(given)
            if not params:
                return given
            parameter_set = ParameterSet(params, global_params)
            if n_samples is None:
                result = parameter_set.evaluate()
            else:
                result = parameter_set.evaluate_monte_carlo(iterations=n_samples)
            return {name: result[name] for name in data}

        project = evaluate_set(self.project, {}, ("project",))
        database = {
            db: evaluate_set(data, project, ("database", db))
            for db, data in self.database.items()
        }
        values = {("project", name): value for name, value in project.items()}
        for db, data in database.items():
            values.update({("database", db, name): v for name, v in data.items()})

        local, namespace = {}, {}

        def resolve(group, visiting=()):
            if group in namespace:
                return namespace[group]
            symbols = {**project, **database.get(self.group_db[group], {})}
            for dependency in reversed(self.order[group]):
                if dependency in visiting:
                    raise ValueError(f"Circular parameter groups: {visiting + (group,)}")
                resolve(dependency, visiting + (group,))
                symbols.update(local[dependency])
            local[group] = evaluate_set(self.groups[group], symbols, ("activity", group))
            values.update({("activity", group, n): v for n, v in local[group].items()})
            symbols.update(local[group])
            namespace[group] = symbols
            return symbols

        amounts = {}
        for exchange_id, group, formula, _, _ in self.exchanges:
            interpreter = Interpreter()
            interpreter.symtable.update(resolve(group))
            result = interpreter(formula)
            if interpreter.error:
                raise ValueError(
                    f"Could not evaluate the formula '{formula}': "
                    f"{interpreter.error[0].get_error()}"
                )
            if n_samples is None:
                amounts[exchange_id] = float(result)
            else:
                amounts[exchange_id] = np.broadcast_to(
                    np.asarray(result, float), (n_samples,)
                ).copy()
        return values, amounts

    def datapackage(self, amounts: Mapping[int, np.ndarray], matrix: str):
        """
        A `bw_processing` datapackage with the sampled amounts as arrays.

        Its columns are walked in order (``sequential=True``), so column ``i`` is
        sample ``i``. Layered over the foreground datapackage, it replaces the
        stored amounts of the parameterized exchanges.
        """
        import bw_processing as bwp

        ids = [i for i in amounts if self.entries[i].matrix == matrix]
        if not ids:
            return None
        package = bwp.create_datapackage(sequential=True)
        package.add_persistent_array(
            matrix=matrix,
            name=f"optimex-parameterized-exchanges-{matrix}",
            indices_array=np.array(
                [(self.entries[i].row, self.entries[i].col) for i in ids],
                dtype=bwp.INDICES_DTYPE,
            ),
            data_array=np.vstack([amounts[i] for i in ids]),
            flip_array=np.array([self.entries[i].flip for i in ids]),
        )
        return package


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------


@dataclass
class SampleSet:
    """
    Monte Carlo samples of the optimization inputs.

    Attributes
    ----------
    inputs : list of OptimizationModelInputs
        One complete set of optimization inputs per sample.
    values : pandas.DataFrame
        The sampled values of the explicitly uncertain quantities (foreground
        exchanges, parameters, parameterized exchanges, and prices per background
        database), one row per sample, with columns (source, quantity).
        Background inventories are not listed.
    seed : int or None
        The seed the samples were drawn with.
    base : OptimizationModelInputs or None
        The deterministic inputs the samples were drawn from. Their scaling is
        shared by all samples when they are solved with a persistent solver.
    """

    inputs: List[OptimizationModelInputs]
    values: pd.DataFrame
    seed: Optional[int] = None
    base: Optional[OptimizationModelInputs] = None

    def __len__(self) -> int:
        return len(self.inputs)

    def __iter__(self):
        return iter(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index]


class UncertaintySampler:
    """
    Draws Monte Carlo samples of `OptimizationModelInputs` from the uncertainty
    of a Brightway product system.

    Parameters
    ----------
    processor : LCADataProcessor
        The processor the deterministic inputs were built from. It identifies
        the foreground and background databases and the purchased products.
    inputs : OptimizationModelInputs
        The deterministic inputs, including any limits, discounting, or other
        overrides. Each sample is a copy in which only the uncertain values
        change.
    sources : sequence of str
        Which uncertainty to sample, any of ``"exchanges"``, ``"parameters"``,
        ``"background"``, and ``"prices"``. Default: all of them.
    foreground_datapackages, background_datapackages : sequence of Datapackage
        Extra `bw_processing` datapackages layered over the foreground database,
        or over every background database, e.g. pre-sampled arrays of
        correlated exchanges. Later packages override earlier ones at the
        coordinates they define, as in `bw2calc`. Their arrays are used
        whenever their source (``"exchanges"`` or ``"background"``) is sampled.

    Notes
    -----
    Uncertainty is applied as a relative change of the deterministic value. An
    exchange sampled 20% above its amount is 20% higher at every process time
    and in every vintage, including vintage-specific amounts; vintage
    improvement factors stay as they are. Exchanges whose amount is given by a
    formula are sampled through their parameters, not through their own
    distribution, as in Brightway. Parameters are evaluated only for foreground
    exchanges; parameterized background exchanges keep their amounts.

    Characterization factors, temporal distributions, and demand stay
    deterministic.
    """

    def __init__(
        self,
        processor: lca_processor.LCADataProcessor,
        inputs: OptimizationModelInputs,
        sources: Sequence[str] = SOURCES,
        foreground_datapackages: Sequence[Any] = (),
        background_datapackages: Sequence[Any] = (),
    ):
        unknown = set(sources) - set(SOURCES)
        if unknown:
            raise ValueError(f"Unknown uncertainty sources {sorted(unknown)}; use {SOURCES}.")
        self.processor = processor
        self.inputs = inputs
        self.sources = tuple(source for source in SOURCES if source in sources)
        self.foreground_datapackages = list(foreground_datapackages)
        self.background_datapackages = list(background_datapackages)
        # Sampling reads the processed datapackages: process edited databases
        bd.databases.clean()

        self._exchanges: List[Tuple[_Quantity, _ExchangeTarget, _MatrixEntry]] = []
        # All foreground exchanges, so extra datapackages can reach any of them
        self._entries: Dict[_ExchangeTarget, Tuple[_MatrixEntry, Any]] = {}
        self._parameters: Optional[_ParameterModel] = None
        self._prices: List[Tuple[str, Dict[str, Tuple[float, Optional[dict]]]]] = []
        self._background_dbs: List[str] = []

        self._collect_exchanges()
        if "parameters" in self.sources:
            self._parameters = _ParameterModel(
                processor.foreground_db.name, inputs.PROCESS
            )
        if "background" in self.sources:
            self._background_dbs = sorted(
                {db for (db, _), weight in inputs.mapping.items() if weight}
            )
        if "prices" in self.sources:
            self._collect_prices()

    # -- collection ------------------------------------------------------------

    def _collect_exchanges(self) -> None:
        foreground = self.processor.foreground_db.name
        processes = set(self.inputs.PROCESS)
        seen = {}
        for act in self.processor.foreground_db:
            if act.get("type") != bd.labels.process_node_default or act["code"] not in processes:
                continue
            for exc in act.exchanges():
                tensor = _TENSOR_BY_EDGE.get(
                    (exc["type"], exc.input["database"] == foreground)
                )
                if tensor is None:
                    continue
                target = _ExchangeTarget(tensor, act["code"], exc.input["code"])
                if target in seen:
                    logger.warning(
                        f"{act['code']} has several exchanges with {exc.input['code']}; "
                        "only the last one enters the optimization."
                    )
                seen[target] = exc
        for target, exc in sorted(
            seen.items(), key=lambda item: (item[0].process, item[0].flow, item[0].tensor)
        ):
            entry = _matrix_entry(exc["type"], exc.input.id, exc.output.id)
            self._entries[target] = (entry, exc)
            if "exchanges" not in self.sources:
                continue
            if exc.get("formula") or not is_uncertain(exc):
                continue
            amount = float(exc["amount"])
            if amount == 0:
                logger.warning(
                    f"Skipping the uncertainty of {target.process} <- {target.flow}: "
                    "an amount of 0 cannot be varied relatively."
                )
                continue
            label = f"{target.process} <- {target.flow}"
            self._exchanges.append(
                (_Quantity("exchanges", label, dict(exc.as_dict()), amount), target, entry)
            )

    def _collect_prices(self) -> None:
        attribute = self.processor.config.price_attribute
        prices = self.inputs.intermediate_flow_prices or {}
        if attribute is None or not prices:
            return
        databases = sorted({db for (db, _), weight in self.inputs.mapping.items() if weight})
        for flow in sorted({flow for flow, _ in prices}):
            meta = self.processor.intermediate_flows.get(flow)
            per_db = {}
            for db in databases:
                node = lca_processor._resolve_node(db, meta) if isinstance(meta, dict) else bd.Database(db).get(code=flow)
                price = node.get(attribute)
                if price is None:
                    break
                per_db[db] = (float(price), node.get(f"{attribute}_uncertainty"))
            else:
                if any(dist is not None and is_uncertain(dist) for _, dist in per_db.values()):
                    self._prices.append((flow, per_db))

    # -- reporting -------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """The uncertain quantities found, with their distributions."""
        rows = []

        def describe(source, label, distribution, amount):
            rows.append(
                {
                    "source": source,
                    "quantity": label,
                    "amount": amount,
                    **{
                        name: distribution.get(name)
                        for name in ("uncertainty type", "loc", "scale", "minimum", "maximum")
                    },
                }
            )

        for quantity, _, _ in self._exchanges:
            describe("exchanges", quantity.label, quantity.distribution, quantity.amount)
        if self._parameters is not None:
            for quantity in self._parameters.quantities.values():
                describe("parameters", quantity.label, quantity.distribution, quantity.amount)
            for _, _, formula, target, stored in self._parameters.exchanges:
                rows.append(
                    {
                        "source": "parameterized exchanges",
                        "quantity": f"{target.process} <- {target.flow}",
                        "amount": stored,
                        "formula": formula,
                    }
                )
        for flow, per_db in self._prices:
            for db, (price, distribution) in per_db.items():
                describe("prices", f"{flow} @ {db}", distribution or {}, price)
        for db in self._background_dbs:
            rows.append({"source": "background", "quantity": db})
        return pd.DataFrame(rows)

    # -- sampling --------------------------------------------------------------

    def sample(self, n_samples: int, seed: Optional[int] = None) -> SampleSet:
        """
        Draw `n_samples` sets of optimization inputs.

        Every source draws from its own random stream, derived from `seed`, so
        switching a source on or off leaves the draws of the others unchanged.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        seed : int, optional
            Seed for reproducible samples.

        Returns
        -------
        SampleSet
        """
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1.")
        streams = dict(zip(SOURCES, np.random.SeedSequence(seed).spawn(len(SOURCES))))
        columns: Dict[Tuple[str, str], np.ndarray] = {}

        def seed_of(source):
            return int(streams[source].generate_state(1)[0] % (2**31 - 1))

        parameter_amounts = {}
        if self._parameters is not None and self._parameters.exchanges:
            parameters = self._parameters
            drawn, parameter_amounts = parameters.monte_carlo(n_samples, seed_of("parameters"))
            for key in sorted(drawn):
                columns["parameters", parameters.quantities[key].label] = drawn[key]
            for exchange_id, _, _, target, _ in parameters.exchanges:
                columns["parameterized exchanges", f"{target.process} <- {target.flow}"] = (
                    parameter_amounts[exchange_id]
                )

        factors = self._sample_foreground(
            n_samples, seed_of("exchanges"), parameter_amounts, columns
        )

        prices = None
        if self._prices:
            prices = self._sample_prices(n_samples, streams["prices"], columns)

        background = None
        if self._background_dbs:
            background = self._sample_background(n_samples, seed_of("background"))

        samples = []
        for index in range(n_samples):
            update = self._scaled_tensors({t: f[index] for t, f in factors.items()})
            if prices is not None:
                update["intermediate_flow_prices"] = prices[index]
            if background is not None:
                update["background_inventory"] = background[index]
            samples.append(self.inputs.model_copy(update=update))

        values = pd.DataFrame(columns)
        if len(values.columns):
            values.columns = pd.MultiIndex.from_tuples(
                values.columns, names=["source", "quantity"]
            )
        values.index.name = "Sample"
        return SampleSet(samples, values, seed, self.inputs)

    def _sample_foreground(
        self, n_samples, seed, parameter_amounts, columns
    ) -> Dict[_ExchangeTarget, np.ndarray]:
        """
        Relative factors of the foreground exchanges, per exchange and sample.

        The foreground matrices are built by `matrix_utils` from the processed
        foreground datapackage, like `bw2calc` builds them: with
        ``use_distributions=True`` when exchanges are sampled, overlaid with the
        parameterized amounts and any extra datapackages, and redrawn with
        ``next()`` for every sample. Each exchange's sampled matrix value divided
        by its deterministic value scales all of its tensor entries.
        """
        from matrix_utils import MappedMatrix

        sample_exchanges = "exchanges" in self.sources
        targets = {target: entry for _, target, entry in self._exchanges}
        if self._parameters is not None:
            by_id = {i: t for i, _, _, t, _ in self._parameters.exchanges}
            for exchange_id in parameter_amounts:
                targets[by_id[exchange_id]] = self._parameters.entries[exchange_id]
        if sample_exchanges and self.foreground_datapackages:
            # Extra datapackages can reach any foreground exchange
            targets.update({t: entry for t, (entry, _) in self._entries.items()})
        if not targets:
            return {}

        factors = {}
        for matrix in ("technosphere_matrix", "biosphere_matrix"):
            selected = [(t, e) for t, e in targets.items() if e.matrix == matrix]
            if not selected:
                continue
            # Fresh package objects per matrix: indexers are attached to packages
            base = self.processor.foreground_db.datapackage()
            static = MappedMatrix(packages=[base], matrix=matrix)
            packages = [self.processor.foreground_db.datapackage()]
            packages[0].metadata["seed"] = seed  # seeds the distributions
            parameter_package = (
                self._parameters.datapackage(parameter_amounts, matrix)
                if parameter_amounts else None
            )
            if parameter_package is not None:
                packages.append(parameter_package)
            if sample_exchanges:
                packages.extend(self.foreground_datapackages)
            sampled = MappedMatrix(
                packages=packages,
                matrix=matrix,
                use_distributions=sample_exchanges,
                use_arrays=True,
                row_mapper=static.row_mapper,
                col_mapper=static.col_mapper,
            )
            rows = static.row_mapper.map_array(np.array([e.row for _, e in selected]))
            cols = static.col_mapper.map_array(np.array([e.col for _, e in selected]))
            deterministic = np.asarray(static.matrix[rows, cols]).ravel()
            values = np.empty((n_samples, len(selected)))
            for index in range(n_samples):
                if index:
                    next(sampled)
                values[index] = np.asarray(sampled.matrix[rows, cols]).ravel()

            for column, (target, entry) in enumerate(selected):
                if deterministic[column] == 0:
                    if np.any(values[:, column] != 0):
                        logger.warning(
                            f"Skipping the uncertainty of {target.process} <- "
                            f"{target.flow}: an amount of 0 cannot be varied relatively."
                        )
                    continue
                ratio = values[:, column] / deterministic[column]
                if np.allclose(ratio, 1.0, rtol=0, atol=1e-15):
                    continue
                factors[target] = ratio

        for quantity, target, _ in self._exchanges:
            columns["exchanges", quantity.label] = quantity.amount * factors.get(
                target, np.ones(n_samples)
            )
        return factors

    def _scaled_tensors(self, factors: Mapping[_ExchangeTarget, float]) -> dict:
        by_tensor: Dict[str, Dict[Tuple[str, str], float]] = {}
        for target, factor in factors.items():
            by_tensor.setdefault(target.tensor, {})[target.process, target.flow] = factor
        update = {}
        for tensor, pair_factors in by_tensor.items():
            for name in _tensor_names(tensor):
                data = getattr(self.inputs, name, None)
                if not data:
                    continue
                update[name] = {
                    key: value * pair_factors.get((key[0], key[1]), 1.0)
                    for key, value in data.items()
                }
        return update

    def _sample_prices(self, n_samples, stream, columns) -> List[dict]:
        quantiles = np.random.default_rng(stream).random((n_samples, len(self._prices)))
        base = self.inputs.intermediate_flow_prices
        weights: Dict[int, Dict[str, float]] = {}
        for (db, year), weight in self.inputs.mapping.items():
            if weight:
                weights.setdefault(year, {})[db] = weight

        samples = [dict(base) for _ in range(n_samples)]
        for column, (flow, per_db) in enumerate(self._prices):
            drawn = {}
            for db, (price, distribution) in per_db.items():
                drawn[db] = sample_distribution(distribution or {}, price, quantiles[:, column])
                columns["prices", f"{flow} @ {db}"] = drawn[db]
            years = [year for (f, year) in base if f == flow]
            for index, sample in enumerate(samples):
                for year in years:
                    sample[flow, year] = sum(
                        weight * drawn[db][index] for db, weight in weights.get(year, {}).items()
                    )
        return samples

    def _sample_background(self, n_samples, seed) -> List[dict]:
        config = self.processor.config.background_inventory
        flows = self.processor.intermediate_flows
        kept = set(self.inputs.ELEMENTARY_FLOW)
        samples = [dict() for _ in range(n_samples)]
        for db in tqdm(self._background_dbs, desc="Sampling background databases"):
            # The same seed for every database: common random numbers
            entries = lca_processor.sample_db_inventory_entries(
                db,
                flows,
                n_samples,
                seed=seed,
                cutoff=config.cutoff,
                biosphere_db_name=self.processor.biosphere_db.name,
                datapackages=self.background_datapackages,
            )
            for sample, entry in zip(samples, entries):
                for flow, meta in flows.items():
                    inventory = entry.get(lca_processor._flow_identity(flow, meta), {})
                    for code, (_, amount) in inventory.items():
                        if code in kept:
                            sample[db, flow, code] = amount
        return samples


def sample_inputs(
    processor: lca_processor.LCADataProcessor,
    inputs: OptimizationModelInputs,
    n_samples: int,
    seed: Optional[int] = None,
    sources: Sequence[str] = SOURCES,
    **sampler_kwargs,
) -> SampleSet:
    """Shortcut for ``UncertaintySampler(processor, inputs, sources, ...).sample(n_samples, seed)``."""
    return UncertaintySampler(processor, inputs, sources, **sampler_kwargs).sample(
        n_samples, seed
    )


# ----------------------------------------------------------------------------
# Solving and evaluating samples (engine in `optimex.series`)
# ----------------------------------------------------------------------------

MonteCarloResults = SeriesResults


def solve_samples(
    samples: Iterable[OptimizationModelInputs],
    solver_name: str = "glpk",
    solver_options: Optional[dict] = None,
    keep_models: bool = False,
    progress: bool = True,
    persistent="auto",
    outputs: Sequence[str] = ("installation",),
    extract=None,
    n_jobs: int = 1,
    **model_kwargs,
) -> MonteCarloResults:
    """
    Find the optimum of every sample.

    Each sample is optimized on its own, with full knowledge of its values. The
    spread of the optima shows how much the optimal pathway, and what it
    achieves, depends on the uncertain inputs; how often a technology is chosen
    shows how robust that choice is.

    Parameters
    ----------
    samples : SampleSet or iterable of OptimizationModelInputs
        The samples to solve.
    solver_name, solver_options :
        Passed to `optimizer.solve_model`.
    keep_models : bool
        Keep the solved models in the results. Default False saves memory.
    progress : bool
        Show a progress bar.
    persistent : {"auto", True, False}
        Solve all samples with one Gurobi or HiGHS model that is updated in
        place (HiGHS also warm-starts from the previous sample), instead of
        building and solving a new model per sample (see `optimex.persistent`). ``"auto"``
        (default) does so when `solver_name` is Gurobi or HiGHS and warns
        otherwise; True raises instead of falling back; False always builds new
        models. All samples then share the scaling of the deterministic inputs.
        When several optima exist, a warm start can return a different one with
        the same objective; False reproduces the from-scratch results exactly.
    outputs : sequence of str
        Which PostProcessor tables to keep per sample: any of
        ``"installation"`` (default), ``"operation"``, and ``"production"``.
        Objective, cumulative impacts, and total cost are always kept. For large
        models, extracting tables can take as long as solving, so keep only what
        is needed.
    extract : callable, optional
        ``extract(model)`` is called with every solved model, and its return
        values are collected in `MonteCarloResults.extracted`, e.g. to keep a
        single number instead of a table. With ``n_jobs > 1`` it must be a
        module-level function, so it can be sent to the worker processes.
    n_jobs : int
        Number of worker processes; -1 uses all cores. Each worker holds its
        own model, so memory use grows with `n_jobs`. Gurobi's threads are
        divided between the workers unless ``Threads`` is set.
    **model_kwargs :
        Passed to `optimizer.create_model`, e.g. ``objective_category``,
        ``objective``, ``backend``, or ``aggregate_vintages``.

    Returns
    -------
    MonteCarloResults
    """
    return _solve_all(
        list(samples), solver_name, solver_options, keep_models, progress,
        "Solving samples", model_kwargs, persistent, getattr(samples, "base", None),
        outputs, extract, n_jobs, index_name="Sample",
    )


def evaluate_design(
    design,
    samples: Iterable[OptimizationModelInputs],
    fix: str = "installation",
    solver_name: str = "glpk",
    solver_options: Optional[dict] = None,
    keep_models: bool = False,
    progress: bool = True,
    persistent="auto",
    outputs: Sequence[str] = ("installation",),
    extract=None,
    n_jobs: int = 1,
    **model_kwargs,
) -> MonteCarloResults:
    """
    Evaluate a fixed design in every sample.

    The installations of the design, and with ``fix="pathway"`` also its
    operation, are fixed in every sample through deployment and operation
    limits; the rest is optimized as usual.

    - ``fix="installation"``: the capacity is fixed, its operation is
      re-optimized for each sample. Use it for uncertainty the operator
      observes before deciding how to run the plants, such as prices.
    - ``fix="pathway"``: installation and operation are fixed, so each sample
      only re-evaluates the same pathway. Use it for uncertainty that is never
      observed, such as most LCA data. Demand must still be met exactly, so the
      production per unit must not be uncertain.

    A sample in which the design cannot meet demand or a limit is reported as
    infeasible.

    Parameters
    ----------
    design : solved model, or dict with "installation" (and "operation")
        A solved model, e.g. the deterministic optimum, or DataFrames as
        returned by `PostProcessor.get_installation()` and
        `PostProcessor.get_operation()`.
    samples : SampleSet or iterable of OptimizationModelInputs
        The samples to evaluate the design in.
    fix : {"installation", "pathway"}
        What to fix.
    solver_name, solver_options, keep_models, progress, persistent, outputs, extract, n_jobs, **model_kwargs :
        As in `solve_samples`.

    Returns
    -------
    MonteCarloResults
    """
    if fix not in ("installation", "pathway"):
        raise ValueError('fix must be "installation" or "pathway".')
    if isinstance(design, Mapping):
        installation = design["installation"]
        operation = design.get("operation")
    else:
        pp = PostProcessor(design)
        installation = pp.get_installation()
        operation = pp.get_operation()
    if fix == "pathway" and operation is None:
        raise ValueError('fix="pathway" needs the operation of the design.')

    def as_limits(frame):
        return {
            (process, int(year)): max(float(value), 0.0)
            for year, row in frame.iterrows()
            for process, value in row.items()
        }

    limits = {
        "process_deployment_limits_max": as_limits(installation),
        "process_deployment_limits_min": as_limits(installation),
    }
    if fix == "pathway":
        limits["process_operation_limits_max"] = as_limits(operation)
        limits["process_operation_limits_min"] = as_limits(operation)

    fixed = [inputs.model_copy(update=limits) for inputs in samples]
    return _solve_all(
        fixed, solver_name, solver_options, keep_models, progress,
        "Evaluating design", model_kwargs, persistent, getattr(samples, "base", None),
        outputs, extract, n_jobs, index_name="Sample",
    )
