"""Checks of the incremental milestone payments (statcontracts.sequential.milestones)."""

import math

import numpy as np
import pytest

from statcontracts.sequential import BettingReward
from statcontracts.sequential.milestones import (
    crossing_values,
    fixed_candidates,
    mean_plugin_bet,
    plugin_receipts,
    simulate_milestone_policy,
    simulate_plugin_crossings,
    solve_milestone_bellman,
)

LEVELS = (2.0, 4.0, 8.0)


def crossing_value(p, up, down, delta, target, tolerance=1e-12):
    """Independent scalar first-passage recursion for one level.

    At round n, live[k] is delta^n times the probability of k successes and
    no previous crossing. Returns (discounted hitting mass, remaining-value
    bound, last round); the remaining value is at most delta * sum(live).
    """
    if target <= 0:
        return 1.0, 0.0, 0
    if max(up, down) <= 0:
        return 0.0, 0.0, 0
    live = [1.0]
    hit = 0.0
    for n in range(1, 20001):
        next_live = [0.0] * (n + 1)
        new_hits = []
        for k, mass in enumerate(live):
            if mass == 0:
                continue
            for next_k, probability in ((k, 1 - p), (k + 1, p)):
                contribution = delta * probability * mass
                w = next_k * up + (n - next_k) * down
                if w >= target:
                    new_hits.append(contribution)
                else:
                    next_live[next_k] += contribution
        hit += math.fsum(new_hits)
        live = next_live
        remainder = delta * math.fsum(live)
        if remainder < tolerance:
            return hit, remainder, n
    raise RuntimeError("First-passage remainder did not reach tolerance")


# ===================================================================
# Fixed bets
# ===================================================================

def test_scalar_reference_hitting_values():
    """Deterministic and nearest-neighbor walks, for positive, zero, and negative drift."""
    value, remainder, _ = crossing_value(1.0, 0.4, -0.7, 0.9, 1.0)
    assert abs(value - 0.9 ** 3) < 1e-14 and remainder == 0
    assert crossing_value(0.7, 0.0, 0.0, 0.9, 1.0)[0] == 0
    for p in (0.25, 0.5, 0.75):
        for delta in (0.9, 0.98):
            # A nearest-neighbor walk hits integer targets without overshoot.
            root = 1 + math.sqrt(1 - 4 * delta ** 2 * p * (1 - p))
            rho = math.log(root / (2 * delta * p))
            for target in (1.0, 2.3, 5.0):
                exact = math.exp(-rho * math.ceil(target))
                value, remainder, _ = crossing_value(p, 1.0, -1.0, delta, target)
                assert value - 1e-13 <= exact <= value + remainder + 1e-13


@pytest.mark.parametrize("bet", [.8, 1.363585661028494, 1.5748467494864395])
def test_crossing_values_match_scalar_reference(bet):
    values, bounds = crossing_values(BettingReward(.7, .5), bet, .95, LEVELS)
    for level, value, bound in zip(LEVELS, values, bounds):
        scalar, remainder, _ = crossing_value(.7, math.log1p(.5 * bet),
                                              math.log1p(-.5 * bet), .95,
                                              math.log(level))
        assert abs(scalar - value) <= remainder + bound + 1e-12


def test_cgf_bet_is_within_its_overshoot_factor_of_the_upper_bound():
    """U_inc = sum r_j a_j^{-rho*} bounds every policy; the CGF bet attains a fraction of it."""
    reward, delta = BettingReward(.7, .5), .9
    rho, lam = reward.interior_cgf_optimal(delta)
    factor = math.exp(-rho * math.log1p(lam * .5))
    payouts = np.array(LEVELS) * np.array([.1, .1, .8])
    upper = payouts @ np.array(LEVELS) ** (-rho)
    for bet in fixed_candidates(reward, LEVELS, extra=(lam,))[::10]:
        assert crossing_values(reward, bet, delta, LEVELS, 1e-8)[0] @ payouts <= upper
    values, error = crossing_values(reward, lam, delta, LEVELS)
    assert payouts @ (values + error) >= factor * upper


def test_fixed_candidates_bracket_exact_hits():
    """Candidates include bets just below and above one that hits a milestone exactly."""
    m = .5
    candidates = np.array(fixed_candidates(BettingReward(.7, m), LEVELS))
    exact = 2 * (math.sqrt(2) - 1)  # two successes reach 2 exactly
    below, above = candidates[candidates < exact].max(), candidates[candidates > exact].min()
    assert exact - below < 1e-10 and above - exact < 1e-10
    assert np.all((candidates > 0) & (candidates < 1 / m))


# ===================================================================
# Adaptive bets
# ===================================================================

@pytest.fixture(scope="module")
def setting():
    reward, delta = BettingReward(.7, .5), .9
    payouts = np.array(LEVELS) * np.array([.1, .1, .8])
    return reward, delta, payouts


def test_pinned_bellman_matches_first_passage(setting):
    """Pinning one action checks the state discretization against the exact recursion."""
    reward, delta, payouts = setting
    exact = crossing_values(reward, 1.2, delta, LEVELS)[0] @ payouts
    pinned = solve_milestone_bellman(reward, delta, LEVELS, payouts, 2400, 1,
                                     fixed=1.2)
    assert abs(pinned["value"] - exact) < .005 * exact


def test_bellman_policy_improves_on_fixed_bets(setting):
    reward, delta, payouts = setting
    rho, _ = reward.interior_cgf_optimal(delta)
    upper = payouts @ np.array(LEVELS) ** (-rho)
    control = solve_milestone_bellman(reward, delta, LEVELS, payouts, 1200, 101,
                                      extra_actions=(1.2,))
    pinned = solve_milestone_bellman(reward, delta, LEVELS, payouts, 1200, 1,
                                     fixed=1.2)
    assert control["residual"] < 1e-10
    assert control["value"] >= pinned["value"] - 1e-12
    sim = simulate_milestone_policy(control, n_sim=20000)
    assert sim["tail_bound"] < 1e-8
    assert abs(sim["value"] - control["value"]) < 4 * sim["se"] + .01 * upper
    assert sim["value"] < upper + 4 * sim["se"]


# ===================================================================
# Plug-in bets
# ===================================================================

def test_plugin_crossings():
    """Multiple crossings in one round, sampling after a zero bet, and p = 0."""
    hits = simulate_plugin_crossings(BettingReward(1, .5), (1.01, 1.1, 2, 4, 8), 3,
                                     n_sim=8)
    assert np.all(hits == np.array([1, 2, 2, 3, 0]))
    hits = simulate_plugin_crossings(BettingReward(1, .5), (2, 4), 3, n_sim=8,
                                     initial=0)
    assert np.all(hits == np.array([3, 0]))  # Sampling continues after a zero bet.
    assert not simulate_plugin_crossings(BettingReward(0, .5), (2, 4), 5,
                                         n_sim=8).any()


def test_mean_plugin_bet():
    assert np.all(mean_plugin_bet(np.array([0, 1]), 1, .5) == [0, 1.96])
    assert np.all(mean_plugin_bet(np.array([0, 1]), 1, .5, pseudocount=.5) == [0, 1])
    with pytest.raises(ValueError):
        mean_plugin_bet(np.zeros(1), 0, .5, margin=.6)


def test_plugin_receipts_keep_covariance():
    correlated_hits = np.array([[1, 1], [2, 2], [3, 3], [0, 0]])
    totals = plugin_receipts(correlated_hits, [1, 2], .9)
    exact = 3 * np.array([.9, .9 ** 2, .9 ** 3, 0])
    assert np.allclose(totals, exact)
