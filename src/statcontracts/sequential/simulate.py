"""Monte Carlo evaluation of the agent's stopping strategies.

Every strategy runs through `simulate`, which advances n_sim paths of the
log-reward W_t = log(R_t / C) and records the discounted log-utility
delta^tau (W_tau + log C) at the stopping time tau. Every path runs at least
one trial, and paths still running at max_T are stopped there and receive
their discounted log reward at the cap (not zero), so each strategy is
a.s. finite. A strategy is a pair of callbacks:

    bet(t, W, X_sum)   -> parameter for round t, from the state before round t
    stop(t, W, X_sum)  -> boolean stop mask, from the state after round t

where W and X_sum (running sum of the observations) cover the live paths.

Strategies:
  * sim_bellman       - full law; joint optimization over (lambda, stop),
                        following a policy from stopping.bellman_benchmark.
  * sim_cgf_optimal   - CGF-optimal lambda (maximizes psi_lambda(rho*)) and
                        threshold R_t >= exp(1/rho*).
  * sim_cgf_aware     - CGF threshold at the growth-optimal lambda:
                        stop at R_t >= exp(1/rho) where psi(rho) = log(1/delta).
  * sim_drift_aware   - drift threshold at the growth-optimal lambda,
                        R_t >= exp(mu / |log delta|); the CGF threshold with
                        psi linearized.
  * sim_plug_in       - estimate the drift of the next bet on the fly; stop at
                        R_t >= max(C, exp(mu_hat / |log delta|)).

Each simulation reseeds its generator, so strategies evaluated with the same
seed share common random numbers.
"""

from typing import NamedTuple

import numpy as np


DEFAULT_SEED = 42


class SimulationResult(NamedTuple):
    utilities: np.ndarray       # discounted log-utility of each path
    truncated: float            # fraction of paths stopped at max_T


def simulate(reward, delta, C, bet, stop, n_sim=20000, max_T=2500,
             seed=DEFAULT_SEED):
    """Run a stopping strategy given by the callbacks `bet` and `stop`."""
    log_C = np.log(C)
    rng = np.random.default_rng(seed)
    utilities = np.full(n_sim, np.nan)
    W = np.zeros(n_sim)
    X_sum = np.zeros(n_sim)
    alive = np.ones(n_sim, dtype=bool)

    for t in range(1, max_T + 1):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        lam = bet(t, W[idx], X_sum[idx])
        X = reward.sample(rng, len(idx))
        W[idx] += reward.log_increment(lam, X)
        X_sum[idx] += X
        stopping = idx[stop(t, W[idx], X_sum[idx])]
        utilities[stopping] = delta ** t * (W[stopping] + log_C)
        alive[stopping] = False

    rem = np.flatnonzero(alive)
    if len(rem):
        utilities[rem] = delta ** max_T * (W[rem] + log_C)
    return SimulationResult(utilities, len(rem) / n_sim)


def sim_threshold(reward, lam, delta, C, log_threshold, **kwargs):
    """Commit to `lam` and, after at least one trial, stop at the threshold or cap.

    Paths that have not hit log R_t >= log_threshold by max_T receive their
    discounted log reward at the cap. This implements an a.s.-finite strategy
    even when the uncapped threshold rule has a positive probability of never
    stopping.
    """
    w_star = log_threshold - np.log(C)
    return simulate(reward, delta, C,
                    bet=lambda t, W, X_sum: lam,
                    stop=lambda t, W, X_sum: W >= w_star, **kwargs)


def sim_drift_aware(reward, delta, C, **kwargs):
    """Drift threshold at the growth-optimal lambda (the agent knows the drift)."""
    return sim_threshold(reward, reward.growth_optimal_lambda(), delta, C,
                         reward.drift() / np.abs(np.log(delta)), **kwargs)


def sim_cgf_aware(reward, delta, C, **kwargs):
    """CGF threshold at the growth-optimal lambda (the agent knows psi)."""
    rho = reward.cgf_root(delta)
    return sim_threshold(reward, reward.growth_optimal_lambda(), delta, C,
                         1.0 / rho, **kwargs)


def sim_cgf_optimal(reward, delta, C, **kwargs):
    """CGF-optimal parameter and threshold R_t >= exp(1/rho*) (knows all psi_lambda)."""
    rho, lam = reward.cgf_optimal(delta)
    return sim_threshold(reward, lam, delta, C, 1.0 / rho, **kwargs)


def sim_plug_in(reward, delta, C, **kwargs):
    """Re-estimate the mean every round and plug it into the GRO parameter.

    Round t bets reward.plug_in_lambda of the mean of the first t - 1
    observations. After round t, mu_hat_t = reward.plug_in_drift is the
    estimated drift of the next bet under the fitted law, and the agent stops
    at the first t with R_t >= max(C, exp(mu_hat_t / |log delta|)). The
    R_t >= C floor (equivalently W_t >= 0) prevents stopping at a loss on a
    noisy early estimate, removing the need for a burn-in period.
    """
    log_C = np.log(C)
    abs_ln_delta = np.abs(np.log(delta))

    def bet(t, W, X_sum):
        if t == 1:
            return np.full(len(W), reward.initial_lambda)
        return reward.plug_in_lambda(X_sum / (t - 1))

    def stop(t, W, X_sum):
        drift_hat = reward.plug_in_drift(X_sum / t)
        return W >= np.maximum(0.0, drift_hat / abs_ln_delta - log_C)

    return simulate(reward, delta, C, bet, stop, **kwargs)


def sim_bellman(reward, delta, C, G, w_grid, lam_policy, **kwargs):
    """Follow a Bellman policy: bet lam_policy(W) and stop once W + log C >= G(W)."""
    log_C = np.log(C)
    return simulate(
        reward, delta, C,
        bet=lambda t, W, X_sum: np.interp(W, w_grid, lam_policy),
        stop=lambda t, W, X_sum: (W + log_C) >= np.interp(W, w_grid, G) - 1e-10,
        **kwargs)
