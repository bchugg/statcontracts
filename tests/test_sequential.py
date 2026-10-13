"""Checks of the reward processes, stopping solvers, and simulators."""

import numpy as np
import pytest
from scipy.integrate import quad

from statcontracts.sequential import (
    BettingReward,
    GaussianReward,
    HoeffdingReward,
    RewardProcess,
    beta_threshold,
    kl_bernoulli,
    numeric_cgf_optimal,
    sim_cgf_aware,
    sim_drift_aware,
    sim_plug_in,
    sim_threshold,
    solve_bellman_pi,
    solve_bellman_vi,
    threshold_value_grid,
)
from statcontracts.sequential import adjusters, incremental

REWARDS = [GaussianReward(0.3, 0.5), BettingReward(0.7, 0.5), HoeffdingReward(0.7, 0.5)]


# ===================================================================
# Reward processes
# ===================================================================

@pytest.mark.parametrize("reward", REWARDS, ids=repr)
def test_growth_optimal_lambda_maximizes_drift(reward):
    lam_star = reward.growth_optimal_lambda()
    np.testing.assert_allclose(reward.increment_moments(lam_star)[0], reward.drift(),
                               rtol=1e-12)
    for lam in np.linspace(0.05, 0.95 * reward.max_positive_drift_lambda(), 50):
        assert reward.increment_moments(lam)[0] <= reward.drift() + 1e-12


@pytest.mark.parametrize("reward", REWARDS, ids=repr)
def test_increments_match_sampled_moments(reward):
    lam = 0.5 * reward.growth_optimal_lambda()
    incr, probs = reward.increments(np.array([lam]))
    X = reward.sample(np.random.default_rng(0), 400000)
    D = reward.log_increment(lam, X)
    np.testing.assert_allclose(incr[0] @ probs, D.mean(), atol=4 * D.std() / 600)
    mean, neg = reward.increment_moments(lam)
    np.testing.assert_allclose(mean, incr[0] @ probs, rtol=1e-10)
    np.testing.assert_allclose(neg, np.maximum(-D, 0).mean(), atol=4 * D.std() / 600)


@pytest.mark.parametrize("reward", REWARDS, ids=repr)
def test_cgf_root_solves_cgf_equation(reward):
    delta = 0.99
    rho = reward.cgf_root(delta)
    np.testing.assert_allclose(reward.cgf(reward.growth_optimal_lambda(), rho),
                               np.log(1 / delta), rtol=1e-10)
    # The closed forms agree with the generic bisection.
    np.testing.assert_allclose(rho, RewardProcess.cgf_root(reward, delta), rtol=1e-10)


def test_gaussian_cgf_optimal_matches_numeric_search():
    g = GaussianReward(0.3, 0.5)
    for delta in [0.9, 0.99, 0.999]:
        np.testing.assert_allclose(g.cgf_optimal(delta),
                                   numeric_cgf_optimal(g, delta, (0.0, 50.0)),
                                   rtol=1e-6)


def test_betting_drift_is_kl_and_dominates_hoeffding():
    for p in [0.55, 0.7, 0.9]:
        b, h = BettingReward(p, 0.5), HoeffdingReward(p, 0.5)
        np.testing.assert_allclose(b.drift(), kl_bernoulli(p, 0.5), rtol=1e-12)
        assert h.drift() <= b.drift()  # Pinsker


# ===================================================================
# Stopping
# ===================================================================

@pytest.mark.parametrize("reward", REWARDS[:2], ids=repr)
def test_beta_threshold_matches_monte_carlo(reward):
    """beta = E[max_{0<=k<=N} S_k] with P(N = n) = (1 - delta) delta^n."""
    lam, delta, n_sim = 0.5 * reward.growth_optimal_lambda(), 0.9, 200000
    rng = np.random.default_rng(0)
    N = rng.geometric(1 - delta, size=n_sim) - 1
    S = np.zeros(n_sim)
    best = np.zeros(n_sim)
    for k in range(1, N.max() + 1):
        live = N >= k
        S[live] += reward.log_increment(lam, reward.sample(rng, live.sum()))
        best[live] = np.maximum(best[live], S[live])
    np.testing.assert_allclose(beta_threshold(reward, lam, delta), best.mean(),
                               atol=4 * best.std() / np.sqrt(n_sim))


@pytest.mark.parametrize("reward", REWARDS[:2], ids=repr)
def test_bellman_solvers_agree_and_stop_at_beta(reward):
    delta = 0.98
    lam = reward.growth_optimal_lambda()
    beta = beta_threshold(reward, lam, delta)
    w_grid = np.linspace(-20, 3 * beta + 10, 1500)
    lam_grid = np.array([lam])

    G_pi, _, residual, _ = solve_bellman_pi(reward, delta, 0.0, w_grid, lam_grid)
    G_vi, _ = solve_bellman_vi(reward, delta, 0.0, w_grid, lam_grid, tol=1e-10,
                               max_iter=20000)
    assert residual < 1e-8
    np.testing.assert_allclose(G_pi, G_vi, atol=1e-6)

    # Proposition 3: with a fixed parameter, stop exactly on {w >= beta}.
    boundary = w_grid[(G_pi <= w_grid + 1e-9) & (w_grid > 0)].min()
    assert abs(boundary - beta) < 0.05 * beta

    # The exact threshold beats nearby thresholds.
    v_beta = threshold_value_grid(reward, delta, w_grid, lam, beta)
    for kappa in [0.7 * beta, 1.3 * beta]:
        assert threshold_value_grid(reward, delta, w_grid, lam, kappa) <= v_beta
    np.testing.assert_allclose(v_beta, np.interp(0.0, w_grid, G_pi), rtol=1e-3)


# ===================================================================
# Simulation
# ===================================================================

@pytest.mark.parametrize("reward", REWARDS, ids=repr)
def test_drift_aware_is_a_threshold_rule(reward):
    delta, C = 0.99, 2.0
    a = sim_drift_aware(reward, delta, C, n_sim=500)
    b = sim_threshold(reward, reward.growth_optimal_lambda(), delta, C,
                      reward.drift() / abs(np.log(delta)), n_sim=500)
    np.testing.assert_array_equal(a.utilities, b.utilities)
    assert a.truncated == 0.0 and np.all(np.isfinite(a.utilities))


def test_common_random_numbers():
    g = GaussianReward(0.3, 0.5)
    a = sim_cgf_aware(g, 0.99, 1.0, n_sim=500, seed=7)
    b = sim_cgf_aware(g, 0.99, 1.0, n_sim=500, seed=7)
    c = sim_cgf_aware(g, 0.99, 1.0, n_sim=500, seed=8)
    np.testing.assert_array_equal(a.utilities, b.utilities)
    assert not np.array_equal(a.utilities, c.utilities)


@pytest.mark.parametrize("reward", REWARDS[:2], ids=repr)
def test_plug_in_never_stops_at_a_loss(reward):
    result = sim_plug_in(reward, 0.99, 1.0, n_sim=500)
    assert result.truncated == 0.0
    assert np.all(result.utilities >= 0)


def test_incremental_snell_envelope_dominates_stopping():
    g = GaussianReward(0.2, 0.5)
    phi, m_grid, g_grid = incremental.solve_snell_2d(g, 0.99, 0.0, 0.5, n_m=60, n_g=20)
    stop_val = np.log(0.5) + 0.5 * m_grid[:, None]
    assert np.all(phi >= stop_val - 1e-12)
    u = incremental.sim_snell(g, 0.99, 1.0, 0.5, phi, m_grid, g_grid, n_sim=500)
    assert np.all(np.isfinite(u))


# ===================================================================
# Adjusters
# ===================================================================

W_MAX = 200.0


@pytest.mark.parametrize("h, tail", [
    (lambda w: adjusters.h_shafer(w, 0.5), 0.0),
    (adjusters.h_choe, 0.0),
    # h_mixture(w, a) ~ a Gamma(a + 1) / w^{a + 1}, so the tail is Gamma(a + 1) / W^a.
    (lambda w: adjusters.h_mixture(w, 1.0, n_kappa=4000), 1 / W_MAX),
    (lambda w: adjusters.h_mixture(w, 2.0, n_kappa=4000), 2 / W_MAX ** 2),
], ids=["shafer", "choe", "mixture-1", "mixture-2"])
def test_adjusters_are_admissible(h, tail):
    """Admissible adjusters induce densities h with int_0^infty h = 1."""
    total, _ = quad(lambda w: float(h(np.asarray(w))), 1e-9, W_MAX, limit=500)
    np.testing.assert_allclose(total + tail, 1.0, atol=1e-3)


def test_koolen_adjuster_budget():
    # int_0^W h = 1 - ln 2 / ln(1 + e^W): admissible, with a slowly vanishing tail.
    W = 30.0
    total, _ = quad(adjusters.h_koolen, 0, W, limit=500)
    np.testing.assert_allclose(total, 1 - np.log(2) / np.log1p(np.exp(W)), rtol=1e-8)
