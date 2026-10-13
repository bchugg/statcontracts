"""Optimal stopping for the agent: exact thresholds and Bellman solvers.

  * beta_threshold        - exact optimal threshold for a fixed parameter
                            (Proposition 3 of Chugg, Ghosh, and Ramdas): log R_t >= beta, where
                            beta = E[max_{0<=k<=N} S_k],  N ~ Geometric(1 - delta),
                            evaluated through Spitzer's identity as
                            beta = mu * delta / (1 - delta) + sum_n delta^n / n * E[S_n^-].
  * solve_bellman_pi      - the discretized Bellman equation solved exactly by
                            policy iteration with sparse linear solves.
  * solve_bellman_vi      - the same discretization solved by value iteration
                            (the solver of the submitted version of the paper).
  * bellman_benchmark     - the Bellman optimum used as the benchmark
                            (policy iteration on a fine grid, valid bets only).
  * threshold_value_grid  - value of a fixed-parameter threshold rule on the
                            same discretization (policy evaluation).

The state is the log-reward W = log(R / C), discretized on `w_grid` with
linear interpolation and constant extrapolation; Gaussian increments use
Gauss-Hermite quadrature (see RewardProcess.increments).
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


N_GRID = 4000


# ===================================================================
# Exact threshold for a fixed parameter
# ===================================================================

def beta_threshold(reward, lam, delta, tol=1e-12, n_max=200000):
    """Exact optimal log-reward threshold for a fixed parameter (Prop. 3).

    Uses E[S_n^+] = n mu + E[S_n^-], so that
        beta = mu delta / (1 - delta) + sum_{n>=1} delta^n / n E[S_n^-].
    The neglected tail of the second sum is at most
    E[Delta^-] delta^{K+1} / (1 - delta); summation stops once this bound (or
    the current term) falls below `tol`. Returns +inf if the drift is <= 0.
    """
    mu, neg1 = reward.increment_moments(lam)
    if mu <= 0:
        return np.inf
    beta = mu * delta / (1 - delta)
    for n in range(1, n_max + 1):
        term = delta ** n / n * reward.expected_neg_part(lam, n)
        beta += term
        tail = neg1 * delta ** (n + 1) / (1 - delta)
        if tail < tol or (n > 50 and term < tol * (1 - delta)):
            break
    return beta


# ===================================================================
# Bellman equation
# ===================================================================

def _interp_weights(x, w_grid):
    """Linear-interpolation indices/weights with constant extrapolation,
    matching np.interp(x, w_grid, G) (and left=G[0], right=G[-1])."""
    n = len(w_grid)
    xc = np.clip(x, w_grid[0], w_grid[-1])
    j = np.clip(np.searchsorted(w_grid, xc, side="right") - 1, 0, n - 2)
    frac = (xc - w_grid[j]) / (w_grid[j + 1] - w_grid[j])
    return j, 1.0 - frac, frac


def solve_bellman_pi(reward, delta, log_C, w_grid, lam_grid, n_quad=20,
                     max_policy_iter=200):
    """Exact solution of the discretized Bellman equation (policy iteration).

    Linear interpolation on w_grid with constant extrapolation, quadrature
    for the increment, and a finite parameter grid. Returns
    (G, best_lam, residual, n_policy_iterations), where residual is the
    sup-norm Bellman residual of the returned G.
    """
    n = len(w_grid)
    incr, probs = reward.increments(lam_grid, n_quad)
    payoff = w_grid + log_C
    rows = np.arange(n)

    def q_values(G):
        Q = np.empty((len(lam_grid), n))
        for l in range(len(lam_grid)):
            j, a, b = _interp_weights(w_grid[:, None] + incr[l][None, :],
                                      w_grid)
            Q[l] = delta * ((a * G[j] + b * G[j + 1]) @ probs)
        return Q

    G = payoff.copy()
    prev = None
    for it in range(1, max_policy_iter + 1):
        Q = q_values(G)
        pol = np.argmax(Q, axis=0)
        stop = payoff >= Q[pol, rows]
        key = (pol.tobytes(), stop.tobytes())
        if key == prev:
            break
        prev = key
        # Policy evaluation: G = payoff on stop rows, G = delta P G otherwise.
        A_rows, A_cols, A_vals = [rows], [rows], [np.ones(n)]
        for l in np.unique(pol[~stop]):
            r = rows[(~stop) & (pol == l)]
            j, a, b = _interp_weights(w_grid[r, None] + incr[l][None, :],
                                      w_grid)
            rr = np.repeat(r, 2 * len(probs))
            A_rows.append(rr)
            A_cols.append(np.concatenate([j, j + 1], axis=1).ravel())
            A_vals.append(-delta * np.concatenate(
                [a * probs, b * probs], axis=1).ravel())
        A = csr_matrix((np.concatenate(A_vals),
                        (np.concatenate(A_rows), np.concatenate(A_cols))),
                       shape=(n, n))
        G = spsolve(A, np.where(stop, payoff, 0.0))
    Q = q_values(G)
    residual = np.max(np.abs(np.maximum(payoff, Q.max(axis=0)) - G))
    return G, lam_grid[np.argmax(Q, axis=0)], residual, it


def solve_bellman_vi(reward, delta, log_C, w_grid, lam_grid, n_quad=20,
                     tol=1e-7, max_iter=2000):
    """The discretized Bellman equation solved by value iteration.

    Same discretization as solve_bellman_pi, stopped once successive iterates
    differ by less than `tol` (or after `max_iter` sweeps), so it is only
    approximately converged. Returns (G, best_lam).
    """
    incr, probs = reward.increments(lam_grid, n_quad)
    n_w = len(w_grid)

    def expected(G, l):
        args = w_grid[:, None] + incr[l][None, :]
        G_interp = np.interp(args.ravel(), w_grid, G).reshape(n_w, len(probs))
        return (G_interp * probs[None, :]).sum(axis=1)

    G = w_grid + log_C
    for _ in range(max_iter):
        G_old = G.copy()
        best_E_G = np.full(n_w, -np.inf)
        for l in range(len(lam_grid)):
            E_G = expected(G_old, l)
            improve = E_G > best_E_G
            best_E_G[improve] = E_G[improve]
        G = np.maximum(w_grid + log_C, delta * best_E_G)
        if np.max(np.abs(G - G_old)) < tol:
            break

    # recover the maximizing lambda at the converged value function
    best_lam = np.full(n_w, reward.growth_optimal_lambda())
    best_E_G = np.full(n_w, -np.inf)
    for l, lam in enumerate(lam_grid):
        E_G = expected(G, l)
        improve = E_G > best_E_G
        best_E_G[improve] = E_G[improve]
        best_lam[improve] = lam
    return G, best_lam


def bellman_benchmark(reward, delta, log_C=0.0, n_grid=N_GRID, lam_grid=None):
    """Bellman optimum on a fine grid, solved exactly by policy iteration.

    The log-reward grid W = log(R/C) spans [-max(20, beta), 3 beta + 10 - log C]
    with n_grid points, where beta is the exact threshold at the growth-optimal
    parameter (the optimal stopping boundary lies near beta, so this keeps the
    grid fine relative to the increments). Parameters default to
    reward.bellman_lambdas(). Returns (G, w_grid, best_lam, residual).
    """
    beta = beta_threshold(reward, reward.growth_optimal_lambda(), delta)
    w_grid = np.linspace(-max(20.0, beta), 3 * beta + 10 - log_C, n_grid)
    if lam_grid is None:
        lam_grid = reward.bellman_lambdas()
    G, best_lam, res, _ = solve_bellman_pi(reward, delta, log_C, w_grid,
                                           lam_grid)
    return G, w_grid, best_lam, res


def threshold_value_grid(reward, delta, w_grid, lam, kappa, n_quad=20):
    """Value at W_0 = 0 (C = 1) of committing to lam and stopping at W >= kappa.

    Policy evaluation of the discretized problem: V = w on {w >= kappa} and
    V = delta E[V(w + Delta)] elsewhere, solved as a sparse linear system.
    """
    n = len(w_grid)
    incr, probs = reward.increments(np.array([lam]), n_quad)
    rows = np.arange(n)
    cont = w_grid < kappa
    r = rows[cont]
    j, wa, wb = _interp_weights(w_grid[r, None] + incr[0][None, :], w_grid)
    A = csr_matrix((
        np.concatenate([np.ones(n),
                        -delta * np.concatenate([wa * probs, wb * probs],
                                                axis=1).ravel()]),
        (np.concatenate([rows, np.repeat(r, 2 * len(probs))]),
         np.concatenate([rows, np.concatenate([j, j + 1], axis=1).ravel()]))),
        shape=(n, n))
    V = spsolve(A, np.where(cont, 0.0, w_grid))
    return float(np.interp(0.0, w_grid, V))
