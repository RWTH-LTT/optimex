"""
Solving many similar optimization problems quickly, straight from arrays.

Monte Carlo samples, or a series of models that differ only in their data, share
the structure of their linear program. `PersistentModel` takes each problem as
arrays (`LPArrays`: objective, sparse constraint matrix, senses, right-hand sides,
bounds) and loads it into Gurobi (``gurobipy``) or HiGHS (``highspy``) in one
call, which bypasses Pyomo and linopy. The arrays come either from a
`Formulation` (`formulation_arrays`) or, much faster, from
`optimex.incremental.IncrementalFormulation`, which recomputes only what a
sample changes.

Whether to start from the previous solution (a warm start) depends on the
solver (``warm_start="auto"``). HiGHS solves LPs with the simplex method either
way, and a warm start made it 2 to 3 times faster in tests, also when matrix
coefficients changed. Gurobi solves an LP from scratch with concurrent barrier
and simplex, which was up to 15 times faster on large models than its
warm-started simplex, so Gurobi starts from scratch.

When several optimal solutions exist, a warm start can end on a different one
than a solve from scratch: the same objective, but e.g. a different split of
installations between equally good processes.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from optimex.formulation import Formulation

# Solver names, including Pyomo's, that map onto a persistent implementation
_ALIASES = {
    "gurobi": "gurobi",
    "gurobi_direct": "gurobi",
    "gurobi_direct_v2": "gurobi",
    "gurobi_persistent": "gurobi",
    "gurobi_persistent_v2": "gurobi",
    "appsi_gurobi": "gurobi",
    "highs": "highs",
    "appsi_highs": "highs",
}
SENSE_CODES = {"<=": 0, ">=": 1, "==": 2}


def persistent_solver(solver_name: str) -> Tuple[Optional[str], str]:
    """
    The persistent implementation for a solver name, or why there is none.

    Returns
    -------
    tuple
        ``("gurobi" | "highs", "")`` when available, else ``(None, reason)``.
    """
    solver = _ALIASES.get(solver_name)
    if solver is None:
        return None, (
            f"solver '{solver_name}' has no persistent interface in optimex; "
            "use 'gurobi' or 'highs' (pip install highspy, free)"
        )
    if solver == "gurobi":
        try:
            import gurobipy as gp

            with gp.Env(params={"OutputFlag": 0}):
                pass
        except Exception as error:  # missing package or license
            return None, f"gurobipy is not usable ({error})"
    else:
        try:
            import highspy  # noqa: F401
        except ImportError:
            return None, "highspy is not installed (pip install highspy)"
    return solver, ""


@dataclass
class LPArrays:
    """
    A linear program ``min c x  s.t.  A x (<=, >=, ==) rhs,  lb <= x <= ub``.

    Attributes
    ----------
    c : numpy.ndarray
        Objective coefficients, one per variable.
    A : scipy.sparse.csr_matrix
        Constraint matrix, one row per constraint.
    senses : numpy.ndarray
        Per row 0 for ``<=``, 1 for ``>=``, 2 for ``==`` (`SENSE_CODES`).
    rhs, lb, ub : numpy.ndarray
        Right-hand sides and variable bounds (``inf`` for none).
    row_keys : list or None
        ``(constraint name, key)`` per row, for diagnostics and warm starts.
    infeasible : bool
        True when a constant row cannot hold: no solve is needed.
    """

    c: np.ndarray
    A: sp.csr_matrix
    senses: np.ndarray
    rhs: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    row_keys: Optional[list] = None
    infeasible: bool = False


def formulation_arrays(formulation: Formulation) -> LPArrays:
    """The linear program of a `Formulation` as arrays."""
    n = formulation.n_variables
    starts, indices, values, senses, rhs, keys = [0], [], [], [], [], []
    infeasible = False
    for name, block in formulation.constraints.items():
        code = SENSE_CODES[block.sense]
        for key, (terms, value) in block.rows.items():
            if not terms:
                infeasible = True  # the formulation keeps only violated constant rows
                continue
            indices.extend(terms.keys())
            values.extend(terms.values())
            starts.append(len(indices))
            senses.append(code)
            rhs.append(value)
            keys.append((name, key))
    A = sp.csr_matrix(
        (np.asarray(values, float), np.asarray(indices, np.int64), np.asarray(starts, np.int64)),
        shape=(len(senses), n),
    )
    c = np.zeros(n)
    for index, coefficient in formulation.objective_terms.items():
        c[index] += coefficient
    return LPArrays(
        c=c,
        A=A,
        senses=np.asarray(senses, np.int8),
        rhs=np.asarray(rhs, float),
        lb=np.array([0.0 if v is None else v for v in formulation.lower], float),
        ub=np.array([math.inf if v is None else v for v in formulation.upper], float),
        row_keys=keys,
        infeasible=infeasible,
    )


class PersistentModel:
    """
    Solves a series of linear programs in one solver session.

    Parameters
    ----------
    solver_name : str
        ``"gurobi"`` or ``"highs"``, or a Pyomo name of either (e.g.
        ``"gurobi_direct_v2"``, ``"appsi_highs"``).
    solver_options : dict, optional
        Solver parameters, e.g. ``{"Method": 1}`` for Gurobi.
    warm_start : {"auto", True, False}
        Start each solve from the previous solution when the rows are the same.
        ``"auto"`` (default) does so for HiGHS but not for Gurobi, see the module
        description.

    Examples
    --------
    >>> persistent = PersistentModel("highs")
    >>> for formulation in formulations:
    ...     model, termination = persistent.solve(formulation, name="sample")
    """

    def __init__(self, solver_name: str, solver_options: Optional[dict] = None, warm_start="auto"):
        if warm_start not in ("auto", True, False):
            raise ValueError('warm_start must be "auto", True, or False.')
        solver, reason = persistent_solver(solver_name)
        if solver is None:
            raise RuntimeError(f"No persistent solver: {reason}.")
        self.solver = solver
        self.solver_options = dict(solver_options or {})
        self.warm_start = warm_start is True or (warm_start == "auto" and solver == "highs")
        self._backend = (_Gurobi if solver == "gurobi" else _Highs)(self.solver_options)
        self._previous_rows = None
        self.warm_started = False  # whether the last solve was warm-started

    def solve(self, formulation: Formulation, name: str = "model"):
        """
        Solve a `Formulation`.

        Returns
        -------
        tuple
            ``(model, termination)``. `model` is a solved `LinopyModel` when the
            termination is ``"optimal"``, else None. `termination` is a string such
            as ``"optimal"``, ``"infeasible"``, or ``"unbounded"``.
        """
        from optimex.linopy_backend import LinopyModel

        termination, values = self.solve_arrays(formulation_arrays(formulation))
        if termination != "optimal":
            return None, termination
        model = LinopyModel(name, None, formulation, [])
        model.load_solution(values)
        return model, termination

    def solve_arrays(self, lp: LPArrays) -> Tuple[str, Optional[np.ndarray]]:
        """Solve a linear program given as arrays; returns (termination, x)."""
        if lp.infeasible:
            return "infeasible", None
        rows = (lp.A.shape, lp.row_keys)
        warm = (
            self.warm_start
            and self._previous_rows is not None
            and rows[0] == self._previous_rows[0]
            and (lp.row_keys is None or lp.row_keys == self._previous_rows[1])
        )
        self.warm_started = bool(warm)
        self._previous_rows = rows
        return self._backend.solve(lp, warm)


# ----------------------------------------------------------------------------
# Solver implementations
# ----------------------------------------------------------------------------


class _Gurobi:
    _SENSES = np.array(["<", ">", "="])

    def __init__(self, options):
        import gurobipy as gp

        self.gp = gp
        self.env = gp.Env(params={"OutputFlag": 0})
        self.options = options
        self.basis = None

    def solve(self, lp, warm):
        gp = self.gp
        model = gp.Model(env=self.env)
        try:
            for key, value in self.options.items():
                model.setParam(key, value)
            x = model.addMVar(
                len(lp.c),
                lb=np.where(np.isinf(lp.lb), -gp.GRB.INFINITY, lp.lb),
                ub=np.where(np.isinf(lp.ub), gp.GRB.INFINITY, lp.ub),
                obj=lp.c,
            )
            constraints = None
            if lp.A.shape[0]:
                constraints = model.addMConstr(lp.A, x, self._SENSES[lp.senses], lp.rhs)
            model.ModelSense = gp.GRB.MINIMIZE
            if warm and self.basis is not None and constraints is not None:
                x.VBasis, constraints.CBasis = self.basis
            model.optimize()
            status = model.Status
            if status != gp.GRB.OPTIMAL:
                self.basis = None
                return {
                    gp.GRB.INFEASIBLE: "infeasible",
                    gp.GRB.UNBOUNDED: "unbounded",
                    gp.GRB.INF_OR_UNBD: "infeasible_or_unbounded",
                    gp.GRB.TIME_LIMIT: "time_limit",
                }.get(status, f"status {status}"), None
            values = np.asarray(x.X, dtype=float)
            self.basis = None
            if constraints is not None:
                try:
                    self.basis = (np.asarray(x.VBasis), np.asarray(constraints.CBasis))
                except gp.GurobiError:  # no basis, e.g. barrier without crossover
                    pass
            return "optimal", values
        finally:
            model.dispose()


class _Highs:
    def __init__(self, options):
        import highspy

        self.highspy = highspy
        self.inf = highspy.kHighsInf
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        for key, value in options.items():
            self.h.setOptionValue(key, value)
        self.basis = None

    def solve(self, lp, warm):
        highspy, inf = self.highspy, self.inf
        model = highspy.HighsLp()
        n, m = len(lp.c), lp.A.shape[0]
        model.num_col_, model.num_row_ = n, m
        model.col_cost_ = lp.c
        model.col_lower_ = np.where(np.isinf(lp.lb), -inf, lp.lb)
        model.col_upper_ = np.where(np.isinf(lp.ub), inf, lp.ub)
        model.row_lower_ = np.where(lp.senses == 0, -inf, lp.rhs)
        model.row_upper_ = np.where(lp.senses == 1, inf, lp.rhs)
        model.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        model.a_matrix_.num_col_, model.a_matrix_.num_row_ = n, m
        model.a_matrix_.start_ = lp.A.indptr.astype(np.int32)
        model.a_matrix_.index_ = lp.A.indices.astype(np.int32)
        model.a_matrix_.value_ = lp.A.data.astype(float)
        self.h.passModel(model)
        if warm and self.basis is not None:
            self.h.setBasis(self.basis)
        self.h.run()
        status = self.h.getModelStatus()
        kStatus = highspy.HighsModelStatus
        if status == kStatus.kOptimal:
            self.basis = self.h.getBasis()
            return "optimal", np.asarray(self.h.getSolution().col_value, dtype=float)
        self.basis = None
        return {
            kStatus.kInfeasible: "infeasible",
            kStatus.kUnbounded: "unbounded",
            kStatus.kUnboundedOrInfeasible: "infeasible_or_unbounded",
            kStatus.kTimeLimit: "time_limit",
        }.get(status, self.h.modelStatusToString(status)), None
