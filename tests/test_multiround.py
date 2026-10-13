"""Checks of the multi-round contract and the agent's dynamic program."""

import numpy as np
import pytest

from statcontracts.sequential.multiround import (
    _increasing_concave_hull,
    play,
    solve_agent,
)
from statcontracts.singleshot import GaussianMean, optimal_license

MODEL = GaussianMean()
COSTS = [0.1] * 5


class AlwaysBet:
    """Run every stage with a fixed update function, withdrawing a fraction of the license."""

    def __init__(self, f, withdraw=0.0):
        self.f, self.withdraw = f, withdraw

    def decide(self, t, L, withdrawn, invested):
        return self.withdraw * L, np.ones_like(L, dtype=bool)

    def update(self, t, L, Z):
        return self.f(Z)


def mean_and_se(x):
    return x.mean(), x.std() / np.sqrt(len(x))


def test_one_round_reduces_to_the_single_shot_license():
    for theta1, cost, cap in [(1.0, 0.1, 5.0), (0.5, 0.2, 1.0)]:
        policy = solve_agent(MODEL, theta1, [cost], cap)
        f = optimal_license(MODEL, theta1, 0.0, cost, cap)
        np.testing.assert_allclose(policy.entry_value,
                                   MODEL.expectation(f, theta1) - cost, rtol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("theta1, cap", [(1.645, 5.0), (0.5, 1.0)])
def test_dynamic_program_matches_simulation(theta1, cap):
    policy = solve_agent(MODEL, theta1, COSTS, cap)
    mean, se = mean_and_se(play(MODEL, theta1, COSTS, policy, n_sim=50000).profit)
    assert abs(mean - policy.entry_value) < 4 * se


@pytest.mark.slow
def test_null_agents_do_not_profit_from_the_optimal_policy():
    policy = solve_agent(MODEL, 1.0, COSTS, 5.0)
    for theta in [0.0, -0.5]:
        mean, se = mean_and_se(play(MODEL, theta, COSTS, policy, n_sim=50000).profit)
        assert mean < 4 * se


@pytest.mark.slow
def test_more_rounds_never_hurt():
    values = [solve_agent(MODEL, 1.0, [0.1] * T, 5.0).entry_value for T in (1, 3, 5)]
    assert values[0] <= values[1] + 1e-9 <= values[2] + 2e-9


def test_licenses_stay_on_the_grid_and_below_the_cap():
    policy = solve_agent(MODEL, 1.0, COSTS, 5.0, n_grid=200)
    out = play(MODEL, 1.0, COSTS, policy, n_sim=5000)
    assert out.license.max() <= 5.0 + 1e-9
    np.testing.assert_allclose(out.license / policy.grid[1],
                               np.rint(out.license / policy.grid[1]), atol=1e-6)


@pytest.mark.slow
def test_evalue_updates_are_incentive_aligned():
    """Prop. 5: with e-value updates no null strategy profits, even with withdrawals."""
    def lr(Z):
        return np.exp(1.0 * Z - 0.5)

    for strategy in [AlwaysBet(lambda Z: 2.0 * (Z > 0)), AlwaysBet(lr, withdraw=0.5)]:
        mean, se = mean_and_se(play(MODEL, 0.0, COSTS, strategy, n_sim=100000).profit)
        assert mean < 4 * se


def test_non_evalue_updates_are_exploitable():
    """Converse of Prop. 5: an update with null mean 1.1 lets null agents profit."""
    strategy = AlwaysBet(lambda Z: 2.2 * (Z > 0))
    mean, se = mean_and_se(play(MODEL, 0.0, COSTS, strategy, n_sim=100000).profit)
    assert mean > 4 * se


def test_increasing_concave_hull():
    x = np.arange(6.0)
    y = np.array([0.0, 2.0, 2.5, 4.0, 3.0, 4.0])
    # Concave majorant vertices: (0,0), (1,2), (3,4); stop at the leftmost maximum.
    np.testing.assert_array_equal(_increasing_concave_hull(x, y), [0, 1, 3])
