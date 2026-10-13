"""Explicit allocation mechanisms, budget rescaling, and losses.

All mechanisms return raw allocations, before applying a financial-budget
rescaling map.  ``c`` may be either a scalar (the legacy equal-cost interface)
or one positive participation cost per agent.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------

def _cost_vector(c, m: int) -> np.ndarray:
    costs = np.asarray(c, dtype=float)
    if costs.ndim == 0:
        costs = np.full(m, float(costs))
    if costs.shape != (m,) or np.any(costs <= 0):
        raise ValueError("c must be a positive scalar or a positive length-m vector")
    return costs


def mechanism_aon(E: np.ndarray, Delta: float, c=1.0) -> np.ndarray:
    """
    All-or-nothing mechanism M^AoN_Delta.

    Allocates F_i=c_i E_i to every agent if sum(F) >= sum(c)/Delta,
    and otherwise allocates nothing.
    O(m) complexity.
    """
    E = np.asarray(E, dtype=float)
    costs = _cost_vector(c, len(E))
    F = costs * E
    if F.sum() >= costs.sum() / Delta:
        return F
    return np.zeros_like(F)


def mechanism_ebh(E: np.ndarray, Delta: float, c=1.0) -> np.ndarray:
    """
    e-BH mechanism M^{e-BH}_Delta.

    Sorts F_i=c_iE_i and finds the largest k satisfying
    k F_(k) >= sum(c)/Delta, then fully funds those k agents.
    O(m log m) complexity.
    """
    E = np.asarray(E, dtype=float)
    costs = _cost_vector(c, len(E))
    F = costs * E
    order = np.argsort(F)[::-1]
    F_sorted = F[order]
    threshold = costs.sum() / Delta

    k_star = 0
    for k in range(1, len(E) + 1):
        if k * F_sorted[k - 1] >= threshold:
            k_star = k

    A = np.zeros_like(F)
    A[order[:k_star]] = F[order[:k_star]]
    return A


def mechanism_cumulative(E: np.ndarray, Delta: float, c=1.0) -> np.ndarray:
    """Minimum-support cumulative-evidence allocation.

    The largest potential allocations are filled until their cumulative value
    reaches sum(c)/Delta; at most the final funded agent is partially funded.
    """
    E = np.asarray(E, dtype=float)
    costs = _cost_vector(c, len(E))
    F = costs * E
    threshold = costs.sum() / Delta
    if F.sum() < threshold:
        return np.zeros_like(F)

    order = np.argsort(F)[::-1]
    cumulative = np.cumsum(F[order])
    k = int(np.searchsorted(cumulative, threshold, side="left"))
    A = np.zeros_like(F)
    if k > 0:
        A[order[:k]] = F[order[:k]]
    A[order[k]] = threshold - (cumulative[k - 1] if k > 0 else 0.0)
    return A


# ---------------------------------------------------------------------------
# Budget rescaling
# ---------------------------------------------------------------------------

def rescale_m1(A: np.ndarray, B: float) -> np.ndarray:
    """M1: proportional rescaling so that sum(A) = B when over budget."""
    W = A.sum()
    if W > B:
        return A * (B / W)
    return A.copy()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def null_allocation_proportion(A: np.ndarray, is_null: np.ndarray) -> float:
    """Realized NAP: the fraction of the total allocation paid to null agents."""
    total = A.sum()
    return float(A[is_null].sum() / total) if total > 1e-12 else 0.0
