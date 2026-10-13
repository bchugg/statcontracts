"""Optimization over the safe allocation regions P and Q.

P is the full region cut out by the exponentially many subset constraints;
Q is the compact inner region with one constraint per agent. Linear
objectives are solved as LPs (over P by constraint generation, with a
minimum-cut separation oracle) and the logarithmic objective
sum_i v_i log(1 + A_i) by SLSQP with a first-order optimality certificate.

All solvers take the potential allocations F = c * E, the participation
costs c, the risk level Delta, and a financial budget B (np.inf for none).
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
from scipy.optimize import brentq, linprog, minimize


def _subset_row(
    subset: np.ndarray,
    F: np.ndarray,
    costs: np.ndarray,
    Delta: float,
) -> np.ndarray:
    """Return r such that r @ a <= 0 is the subset constraint."""
    F_S = F[subset].sum()
    row = np.full(len(F), -Delta * F_S)
    row[subset] += costs[subset].sum()
    return row


def most_violated_subset(
    a: np.ndarray,
    F: np.ndarray,
    costs: np.ndarray,
    Delta: float,
) -> tuple[np.ndarray, float]:
    """Separate the exponential subset constraints by one minimum cut."""
    m = len(a)
    W = a.sum()
    if W <= 0:
        return np.zeros(m, dtype=bool), 0.0

    q = Delta * W
    pair_weights = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            weight = costs[i] * a[j] + costs[j] * a[i]
            pair_weights[i, j] = pair_weights[j, i] = weight

    d = q * F - costs * a
    unary = d - 0.5 * pair_weights.sum(axis=1)

    source, sink = "source", "sink"
    graph = nx.DiGraph()
    graph.add_nodes_from(range(m))
    graph.add_nodes_from((source, sink))
    for i, coefficient in enumerate(unary):
        if coefficient >= 0:
            graph.add_edge(i, sink, capacity=float(coefficient))
        else:
            graph.add_edge(source, i, capacity=float(-coefficient))
    for i in range(m):
        for j in range(i + 1, m):
            capacity = 0.5 * pair_weights[i, j]
            if capacity > 0:
                graph.add_edge(i, j, capacity=float(capacity))
                graph.add_edge(j, i, capacity=float(capacity))

    _, partition = nx.minimum_cut(graph, source, sink, capacity="capacity")
    source_side, _ = partition
    subset = np.array([i in source_side for i in range(m)], dtype=bool)
    violation = (
        costs[subset].sum() * a[subset].sum()
        - Delta * F[subset].sum() * W
    )
    return subset, float(violation)


def _base_constraints(m: int, B: float) -> tuple[list[np.ndarray], list[float]]:
    if np.isfinite(B):
        return [np.ones(m)], [float(B)]
    return [], []


def solve_linear_full(
    F: np.ndarray,
    costs: np.ndarray,
    Delta: float,
    B: float,
    values: np.ndarray,
    tolerance: float = 1e-7,
    max_iterations: int = 500,
) -> tuple[np.ndarray, int, float]:
    """Maximize a linear objective over P by constraint generation."""
    start = time.perf_counter()
    m = len(F)
    if F.sum() + tolerance < costs.sum() / Delta:
        return np.zeros(m), 0, time.perf_counter() - start

    rows, rhs = _base_constraints(m, B)
    seen: set[tuple[int, ...]] = set()
    for _ in range(max_iterations):
        result = linprog(
            -values,
            A_ub=np.asarray(rows) if rows else None,
            b_ub=np.asarray(rhs) if rhs else None,
            bounds=list(zip(np.zeros(m), F)),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Full-polytope LP failed: {result.message}")
        allocation = np.maximum(result.x, 0.0)
        subset, violation = most_violated_subset(allocation, F, costs, Delta)
        scale = max(1.0, costs.sum() * allocation.sum())
        if violation <= tolerance * scale:
            return allocation, len(seen), time.perf_counter() - start

        key = tuple(np.flatnonzero(subset))
        if key in seen:
            if violation <= 100 * tolerance * scale:
                return allocation, len(seen), time.perf_counter() - start
            raise RuntimeError("Constraint generation stalled on a repeated cut")
        seen.add(key)
        rows.append(_subset_row(subset, F, costs, Delta))
        rhs.append(0.0)

    raise RuntimeError("Constraint generation exceeded its iteration limit")


def _compact_rows(F: np.ndarray, costs: np.ndarray, Delta: float) -> np.ndarray:
    m = len(F)
    C = costs.sum()
    rows = np.empty((m, m))
    for i in range(m):
        rows[i] = -Delta * F[i]
        rows[i, i] += C
    return rows


def solve_linear_compact(
    F: np.ndarray,
    costs: np.ndarray,
    Delta: float,
    B: float,
    values: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Maximize a linear objective over Q."""
    start = time.perf_counter()
    m = len(F)
    rows, rhs = _base_constraints(m, B)
    rows.extend(_compact_rows(F, costs, Delta))
    rhs.extend(np.zeros(m))
    result = linprog(
        -values,
        A_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        bounds=list(zip(np.zeros(m), F)),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Compact-polytope LP failed: {result.message}")
    return np.maximum(result.x, 0.0), time.perf_counter() - start


def evaluate_log_utility(allocation: np.ndarray, values: np.ndarray) -> float:
    """Evaluate the logarithmic objective without a piecewise approximation."""
    return float(values @ np.log1p(allocation))


def _solve_log_program(
    F: np.ndarray,
    B: float,
    values: np.ndarray,
    rows: list[np.ndarray],
    rhs: list[float],
    initial: np.ndarray,
) -> np.ndarray:
    """Solve a log-utility subproblem and certify its first-order optimality gap."""
    caps = np.minimum(F, B)
    # Normalize rows because likelihood-ratio e-values can span many orders of
    # magnitude. A nonpositive row with a nonnegative RHS is redundant for A >= 0.
    matrix = np.asarray(rows, dtype=float).reshape(-1, len(F))
    bounds_rhs = np.asarray(rhs, dtype=float)
    keep = np.any(matrix > 0, axis=1) | (bounds_rhs < 0)
    matrix, bounds_rhs = matrix[keep], bounds_rhs[keep]
    if len(matrix):
        scales = np.maximum(np.max(np.abs(matrix), axis=1), np.abs(bounds_rhs))
        matrix = matrix / scales[:, None]
        bounds_rhs = bounds_rhs / scales
    weights = values / max(1.0, values.mean())
    constraints = []
    if len(matrix):
        constraints.append({
            "type": "ineq",
            "fun": lambda a: bounds_rhs - matrix @ a,
            "jac": lambda a: -matrix,
        })
    for attempt in range(10):
        result = minimize(
            lambda a: -float(weights @ np.log1p(a)),
            np.clip(initial, 0.0, caps),
            jac=lambda a: -weights / (1.0 + a),
            bounds=list(zip(np.zeros(len(F)), caps)),
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-13 / 10**attempt, "maxiter": 1000},
        )
        # Use explicit feasibility and optimality checks: SLSQP can report a
        # line-search stopping condition even when it has reached the optimum.
        if not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Log-utility optimization failed: {result.message}")
        allocation = np.clip(result.x, 0.0, caps)
        if len(matrix) and np.max(matrix @ allocation - bounds_rhs) > 1e-7:
            raise RuntimeError("Log-utility optimizer returned an infeasible allocation")

        # Concavity gives V(y) - V(a) <= grad V(a) @ (y - a). Maximizing this
        # linear bound over the same feasible set certifies the numerical solution.
        gradient = values / (1.0 + allocation)
        certificate = linprog(
            -gradient,
            A_ub=matrix if len(matrix) else None,
            b_ub=bounds_rhs if len(matrix) else None,
            bounds=list(zip(np.zeros(len(F)), caps)),
            method="highs",
        )
        if not certificate.success:
            # Retry the same certificate without presolve if HiGHS cannot
            # solve the reduced LP.
            certificate = linprog(
                -gradient,
                A_ub=matrix if len(matrix) else None,
                b_ub=bounds_rhs if len(matrix) else None,
                bounds=list(zip(np.zeros(len(F)), caps)),
                method="highs-ds",
                options={"presolve": False},
            )
        if not certificate.success:
            raise RuntimeError(f"Log-utility optimality check failed: {certificate.message}")
        gap_bound = float(gradient @ (certificate.x - allocation))
        if gap_bound <= 1e-6 * max(1.0, evaluate_log_utility(allocation, values)):
            return allocation
        # A feasible ascent step avoids restarting at the identical point when
        # SLSQP's objective-change test stops before the gap bound is small enough.
        direction = certificate.x - allocation
        derivative = lambda step: float(values @ (direction / (1.0 + allocation + step * direction)))
        step = 1.0 if derivative(1.0) >= 0 else brentq(derivative, 0.0, 1.0, xtol=1e-15)
        initial = allocation + step * direction
    raise RuntimeError(
        f"Log-utility optimality gap is too large: {gap_bound}; {result.message}"
    )


def solve_log(
    region: str,
    F: np.ndarray,
    costs: np.ndarray,
    Delta: float,
    B: float,
    values: np.ndarray,
    tolerance: float = 1e-7,
    max_iterations: int = 500,
) -> tuple[np.ndarray, int, float]:
    """Maximize sum v_i log(1+A_i) over P or Q, to numerical tolerance."""
    if region not in ("full", "compact"):
        raise ValueError("region must be 'full' or 'compact'")
    start = time.perf_counter()
    m = len(F)
    if F.sum() < costs.sum() / Delta or B == 0:
        return np.zeros(m), 0, time.perf_counter() - start

    rows, rhs = _base_constraints(m, B)
    allocation = F * min(1.0, B / F.sum())
    if region == "compact":
        rows.extend(_compact_rows(F, costs, Delta))
        rhs.extend(np.zeros(m))
        allocation = _solve_log_program(F, B, values, rows, rhs, allocation)
        return allocation, m, time.perf_counter() - start

    seen: set[tuple[int, ...]] = set()
    for _ in range(max_iterations):
        allocation = _solve_log_program(F, B, values, rows, rhs, allocation)
        subset, violation = most_violated_subset(allocation, F, costs, Delta)
        scale = max(1.0, costs.sum() * allocation.sum())
        if violation <= tolerance * scale:
            return allocation, len(seen), time.perf_counter() - start
        key = tuple(np.flatnonzero(subset))
        if key in seen:
            raise RuntimeError("Log-utility constraint generation stalled on a repeated cut")
        seen.add(key)
        rows.append(_subset_row(subset, F, costs, Delta))
        rhs.append(0.0)

    raise RuntimeError("Log-utility constraint generation exceeded its iteration limit")
