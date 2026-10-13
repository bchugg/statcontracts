"""Checks of the multi-agent solvers against enumeration and analytic solutions."""

import numpy as np
import pytest
from scipy.optimize import linprog

from statcontracts.multiagent import (
    evaluate_log_utility,
    mechanism_aon,
    mechanism_cumulative,
    mechanism_ebh,
    most_violated_subset,
    null_allocation_proportion,
    rescale_m1,
    solve_linear_compact,
    solve_linear_full,
    solve_log,
)
from statcontracts.multiagent.optimize import (
    _base_constraints,
    _solve_log_program,
    _subset_row,
)


def subsets(m):
    for mask in range(1, 1 << m):
        yield np.array([(mask >> i) & 1 for i in range(m)], dtype=bool)


def test_min_cut_separation_matches_enumeration():
    rng = np.random.default_rng(90210)
    for m in range(2, 8):
        for _ in range(20):
            costs = rng.uniform(0.2, 2.0, size=m)
            F = rng.uniform(0.0, 4.0, size=m)
            a = rng.uniform(0.0, F)
            Delta = rng.uniform(0.05, 1.0)
            _, cut_value = most_violated_subset(a, F, costs, Delta)
            brute_value = max(
                0.0,
                max(costs[S].sum() * a[S].sum() - Delta * F[S].sum() * a.sum()
                    for S in subsets(m)),
            )
            np.testing.assert_allclose(cut_value, brute_value, rtol=1e-8, atol=1e-8)


def test_constraint_generation_lp_matches_enumeration():
    rng = np.random.default_rng(1)
    for _ in range(10):
        m = 7
        costs = rng.uniform(0.5, 1.5, size=m)
        F = costs * rng.uniform(3.0, 8.0, size=m)
        values = rng.uniform(0.1, 2.0, size=m)
        Delta, B = 0.5, 4.0
        allocation, _, _ = solve_linear_full(F, costs, Delta, B, values)

        rows = [np.ones(m)] + [_subset_row(S, F, costs, Delta) for S in subsets(m)]
        rhs = [B] + [0.0] * ((1 << m) - 1)
        brute = linprog(-values, A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
                        bounds=list(zip(np.zeros(m), F)), method="highs")
        assert brute.success
        np.testing.assert_allclose(values @ allocation, values @ brute.x,
                                   rtol=1e-7, atol=1e-7)


def test_log_program_matches_enumeration():
    rng = np.random.default_rng(2)
    for m in range(2, 8):
        for _ in range(3):
            costs = rng.uniform(0.5, 1.5, size=m)
            F = costs * rng.uniform(0.0, 8.0, size=m)
            values = rng.uniform(0.1, 2.0, size=m)
            Delta = costs.sum() / F.sum() * rng.uniform(1.01, 1.5)
            B = 0.6 * F.sum()
            rows, rhs = _base_constraints(m, B)
            for S in subsets(m):
                rows.append(_subset_row(S, F, costs, Delta))
                rhs.append(0.0)
            full, _, _ = solve_log("full", F, costs, Delta, B, values)
            enumerated = _solve_log_program(F, B, values, rows, rhs, 0.6 * F)
            np.testing.assert_allclose(
                evaluate_log_utility(full, values),
                evaluate_log_utility(enumerated, values),
                rtol=1e-6, atol=1e-7,
            )
            assert np.max(np.asarray(rows) @ full - rhs) <= 1e-7 * max(1.0, costs.sum() * B)
            compact, _, _ = solve_log("compact", F, costs, Delta, B, values)
            assert (evaluate_log_utility(compact, values)
                    <= evaluate_log_utility(full, values) + 1e-6)


@pytest.mark.parametrize("region", ["full", "compact"])
def test_log_analytic_solutions(region):
    # With no binding safety constraints the KKT conditions give A_i = v_i/lambda - 1.
    allocation, _, _ = solve_log(
        region, np.full(3, 100.0), np.ones(3), 1.0, 4.0, np.array([2.0, 1.0, 1.0]))
    np.testing.assert_allclose(allocation, [2.5, 0.75, 0.75], atol=1e-6)
    # Inactive contract: total evidence below the threshold.
    allocation, _, _ = solve_log(region, np.ones(3), np.ones(3), 0.1, 4.0, np.ones(3))
    np.testing.assert_array_equal(allocation, np.zeros(3))
    # No budget: every agent is paid in full.
    allocation, _, _ = solve_log(
        region, np.array([0.0, 2.0, 5.0]), np.ones(3), 1.0, np.inf, np.ones(3))
    np.testing.assert_allclose(allocation, [0.0, 2.0, 5.0], atol=1e-6)


def test_log_compact_binding():
    allocation, _, _ = solve_log(
        "compact", np.array([0.0, 2.0, 5.0]), np.ones(3), 3.0 / 7.0, 3.5, np.ones(3))
    np.testing.assert_allclose(allocation, [0.0, 1.0, 2.5], atol=1e-6)


def test_compact_optimum_is_feasible_for_full():
    rng = np.random.default_rng(3)
    m = 6
    costs = rng.uniform(0.5, 1.5, size=m)
    F = costs * rng.uniform(1.0, 10.0, size=m)
    values = rng.uniform(0.1, 2.0, size=m)
    Delta, B = 0.3, 10.0
    compact, _ = solve_linear_compact(F, costs, Delta, B, values)
    full, _, _ = solve_linear_full(F, costs, Delta, B, values)
    _, violation = most_violated_subset(compact, F, costs, Delta)
    assert violation <= 1e-7 * max(1.0, costs.sum() * compact.sum())
    assert values @ compact <= values @ full + 1e-7


@pytest.mark.parametrize("mechanism", [mechanism_aon, mechanism_ebh, mechanism_cumulative])
def test_mechanisms_lie_in_compact_region(mechanism):
    """After budget rescaling, every mechanism's allocation lies in Q (hence P)."""
    rng = np.random.default_rng(4)
    for _ in range(50):
        m = rng.integers(2, 12)
        costs = rng.uniform(0.2, 2.0, size=m)
        E = np.exp(rng.normal(1.0, 2.0, size=m))
        Delta = rng.uniform(0.05, 0.5)
        B = rng.uniform(1.0, 50.0)
        A = rescale_m1(mechanism(E, Delta, costs), B)
        F = costs * E
        assert np.all(A >= 0) and np.all(A <= F + 1e-12)
        assert A.sum() <= B * (1 + 1e-12)
        assert np.all(costs.sum() * A <= Delta * F * A.sum() * (1 + 1e-9) + 1e-12)
        _, violation = most_violated_subset(A, F, costs, Delta)
        assert violation <= 1e-9 * max(1.0, costs.sum() * A.sum())


def test_mechanisms_on_a_small_instance():
    E = np.array([8.0, 20.0, 18.0])
    Delta = 0.1  # threshold sum(c) / Delta = 30
    np.testing.assert_allclose(mechanism_aon(E, Delta), E)
    np.testing.assert_allclose(mechanism_ebh(E, Delta), [0.0, 20.0, 18.0])
    np.testing.assert_allclose(mechanism_cumulative(E, Delta), [0.0, 20.0, 10.0])
    np.testing.assert_allclose(mechanism_aon(E, 0.05), np.zeros(3))


def test_null_allocation_proportion():
    A = np.array([1.0, 3.0, 0.0])
    assert null_allocation_proportion(A, np.array([True, False, True])) == 0.25
    assert null_allocation_proportion(np.zeros(3), np.ones(3, dtype=bool)) == 0.0


def all_subset_sums(x):
    """Sums of x over all nonempty subsets."""
    s = np.zeros(1)
    for xi in x:
        s = np.concatenate([s, s + xi])
    return s[1:]


@pytest.mark.slow
def test_full_region_optima_are_feasible():
    """Optima over P satisfy every subset constraint, checked by enumeration.

    Uses instances like those of the optimality-gap experiments, whose
    likelihood-ratio e-values span many orders of magnitude. A separation
    oracle that misses violated subsets (as the minimum cut of NetworkX 2.x
    does on some of these) stops constraint generation early and overstates
    the optimum.
    """
    m, Delta, B = 20, 0.1, 30.0
    rng = np.random.default_rng(np.random.SeedSequence([20260915, m]))
    n_checked = 0
    while n_checked < 45:
        is_null = rng.random(m) < 0.5
        data = rng.normal(np.where(is_null, 0.0, 1.0)[:, None], 2.0, size=(m, 30))
        E = np.exp(np.clip((data.sum(axis=1) - 15.0) / 4.0, -30.0, 30.0))
        costs = np.exp(rng.normal(-0.5 * 0.55**2, 0.55, size=m))
        values = np.exp(rng.normal(-0.5 * 0.8**2, 0.8, size=m))
        F = costs * E
        if F.sum() < costs.sum() / Delta:
            continue
        n_checked += 1
        cS, FS = all_subset_sums(costs), all_subset_sums(F)
        for A in (solve_linear_full(F, costs, Delta, B, values)[0],
                  solve_log("full", F, costs, Delta, B, values)[0]):
            violation = np.max(cS * all_subset_sums(A) - Delta * FS * A.sum())
            assert violation <= 1e-6 * max(1.0, costs.sum() * A.sum())
