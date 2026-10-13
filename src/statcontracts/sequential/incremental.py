"""Incremental reward schedule with the Shafer adjuster.

Reward: R_t = C * kappa * (E_t^*)^{1-kappa}, using the adjuster
A_kappa(y) = kappa * y^{1-kappa} on the running maximum E_t^* = max_{k<=t} E_k
of a sub-Gaussian e-process E_t (GaussianReward). Utility: delta^t log(R_t).

State: (M_t, G_t) where M_t = log(E_t^*), G_t = M_t - S_t (gap), S_t = log(E_t).

Strategies:
  * sim_snell              - oracle lambda, exact 2D dynamic programming
  * sim_threshold_oracle   - oracle lambda, Markov approximation threshold
  * sim_myopic             - adaptive lambda, one-step lookahead estimated from data
  * sim_threshold_adaptive - adaptive lambda, Markov approximation with estimated drift
"""

import numpy as np
from scipy.special import roots_hermite
from scipy.stats import norm

from .simulate import DEFAULT_SEED


def interp2d(m_q, g_q, m_grid, g_grid, values):
    """Bilinear interpolation on a uniform (m, g) grid."""
    dm = m_grid[1] - m_grid[0]
    dg = g_grid[1] - g_grid[0]
    n_m, n_g = values.shape
    mi = (m_q - m_grid[0]) / dm
    gi = (g_q - g_grid[0]) / dg
    mi0 = np.clip(np.floor(mi).astype(int), 0, n_m - 2)
    gi0 = np.clip(np.floor(gi).astype(int), 0, n_g - 2)
    mf = np.clip(mi - mi0, 0.0, 1.0)
    gf = np.clip(gi - gi0, 0.0, 1.0)
    return ((1 - mf) * (1 - gf) * values[mi0, gi0] +
            mf * (1 - gf) * values[mi0 + 1, gi0] +
            (1 - mf) * gf * values[mi0, gi0 + 1] +
            mf * gf * values[mi0 + 1, gi0 + 1])


def solve_snell_2d(reward, delta, log_C, kappa,
                   n_m=250, n_g=80, n_quad=20, tol=1e-5, max_iter=5000):
    """Snell envelope on the (M, G) grid by value iteration, at the GRO parameter.

    Returns (phi, m_grid, g_grid).
    """
    mu, sigma = reward.theta, reward.sigma
    lam = reward.growth_optimal_lambda()
    mu_Y = lam * mu - lam ** 2 / 8
    sigma_Y = lam * sigma
    alpha = log_C + np.log(kappa)

    m_max = max(2.5 * mu_Y / np.abs(np.log(delta)), 12.0)
    m_grid = np.linspace(0, m_max, n_m)
    g_max = max(6 * sigma_Y, 4.0)
    g_grid = np.linspace(0, g_max, n_g)

    nodes, weights = roots_hermite(n_quad)
    y_vals = mu_Y + sigma_Y * np.sqrt(2) * nodes
    gh_w = weights / np.sqrt(np.pi)

    M, G = np.meshgrid(m_grid, g_grid, indexing='ij')
    stop_val = alpha + (1 - kappa) * M
    phi = stop_val.copy()

    for it in range(max_iter):
        phi_old = phi.copy()
        cont = np.zeros((n_m, n_g))
        for y, w in zip(y_vals, gh_w):
            dm = np.maximum(0, y - G)
            m_new = M + dm
            g_new = np.maximum(0, G - y)

            in_grid = m_new <= m_grid[-1]
            mc = np.clip(m_new, m_grid[0], m_grid[-1])
            gc = np.clip(g_new, g_grid[0], g_grid[-1])

            v = interp2d(mc, gc, m_grid, g_grid, phi_old)
            v[~in_grid] = alpha + (1 - kappa) * m_new[~in_grid]
            cont += w * v

        phi = np.maximum(stop_val, delta * cont)
        if np.max(np.abs(phi - phi_old)) < tol:
            break

    return phi, m_grid, g_grid


def simulate_incremental(reward, delta, C, kappa, bet, stop, n_sim=20000,
                         max_T=2000, seed=DEFAULT_SEED):
    """Run a strategy under the incremental schedule; returns the utilities.

    bet(t, X_sum) gives the parameter for round t from the state before it;
    stop(t, S, M, X_sum, X_sq) gives the stop mask from the state after it.
    """
    alpha = np.log(C) + np.log(kappa)
    rng = np.random.default_rng(seed)
    utilities = np.full(n_sim, np.nan)
    S = np.zeros(n_sim)
    M = np.zeros(n_sim)
    X_sum = np.zeros(n_sim)
    X_sq = np.zeros(n_sim)
    alive = np.ones(n_sim, dtype=bool)

    for t in range(1, max_T + 1):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        lam = bet(t, X_sum[idx])
        X = reward.sample(rng, len(idx))
        S[idx] += reward.log_increment(lam, X)
        M[idx] = np.maximum(M[idx], S[idx])
        X_sum[idx] += X
        X_sq[idx] += X ** 2
        stop_idx = idx[stop(t, S[idx], M[idx], X_sum[idx], X_sq[idx])]
        utilities[stop_idx] = delta ** t * (alpha + (1 - kappa) * M[stop_idx])
        alive[stop_idx] = False

    rem = np.flatnonzero(alive)
    if len(rem) > 0:
        utilities[rem] = delta ** max_T * (alpha + (1 - kappa) * M[rem])
    return utilities


def _adaptive_bet(t, X_sum):
    if t == 1:
        return np.ones(len(X_sum))
    Xbar = X_sum / (t - 1)
    return np.clip(4 * Xbar, 0.01, 20.0)


def sim_snell(reward, delta, C, kappa, phi, m_grid, g_grid, **kwargs):
    """Oracle lambda; stop when the reward reaches the Snell envelope phi."""
    lam = reward.growth_optimal_lambda()
    alpha = np.log(C) + np.log(kappa)

    def stop(t, S, M, X_sum, X_sq):
        log_R = alpha + (1 - kappa) * M
        mq = np.clip(M, m_grid[0], m_grid[-1])
        gq = np.clip(M - S, g_grid[0], g_grid[-1])
        return log_R >= interp2d(mq, gq, m_grid, g_grid, phi) - 1e-9

    return simulate_incremental(reward, delta, C, kappa,
                                lambda t, X_sum: lam, stop, **kwargs)


def sim_threshold_oracle(reward, delta, C, kappa, **kwargs):
    """Oracle lambda; Markov-approximation threshold with the true drift."""
    lam = reward.growth_optimal_lambda()
    w_star = (1 - kappa) * reward.drift() / np.abs(np.log(delta)) - np.log(C)

    def stop(t, S, M, X_sum, X_sq):
        return np.log(kappa) + (1 - kappa) * M >= w_star

    return simulate_incremental(reward, delta, C, kappa,
                                lambda t, X_sum: lam, stop, **kwargs)


def sim_myopic(reward, delta, C, kappa, t0=10, **kwargs):
    """Adaptive lambda; stop when the estimated one-step gain is too small."""
    alpha = np.log(C) + np.log(kappa)

    def stop(t, S, M, X_sum, X_sq):
        if t < t0:
            return np.zeros(len(S), dtype=bool)
        G = M - S
        log_R = alpha + (1 - kappa) * M
        Xbar = X_sum / t
        Xvar = np.maximum(X_sq / t - Xbar ** 2, 0.01)
        lam_cur = np.clip(4 * Xbar, 0.01, 20.0)
        mu_hat = lam_cur * Xbar - lam_cur ** 2 / 8
        sig_hat = np.maximum(lam_cur * np.sqrt(Xvar), 0.01)
        z = (mu_hat - G) / sig_hat
        h = (mu_hat - G) * norm.cdf(z) + sig_hat * norm.pdf(z)
        return delta * (1 - kappa) * h <= (1 - delta) * log_R

    return simulate_incremental(reward, delta, C, kappa, _adaptive_bet, stop,
                                **kwargs)


def sim_threshold_adaptive(reward, delta, C, kappa, t0=10, **kwargs):
    """Adaptive lambda; Markov-approximation threshold with the estimated drift."""
    abs_ln_d = np.abs(np.log(delta))
    log_C = np.log(C)

    def stop(t, S, M, X_sum, X_sq):
        if t < t0:
            return np.zeros(len(S), dtype=bool)
        mu_hat = 2 * (X_sum / t) ** 2
        w_star = (1 - kappa) * mu_hat / abs_ln_d - log_C
        return np.log(kappa) + (1 - kappa) * M >= w_star

    return simulate_incremental(reward, delta, C, kappa, _adaptive_bet, stop,
                                **kwargs)
