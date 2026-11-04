"""Incremental milestone payments under Bernoulli betting.

The principal pays r_j the first time the betting e-process
E_t = prod_{i<=t} (1 + lam_i (X_i - m)), X_i ~ Bernoulli(p), reaches the
milestone a_j (the BettingReward). A policy's value is its discounted
receipts, J(pi) = sum_j r_j E[delta^{tau_j}], where tau_j is the first t with
E_t >= a_j and a milestone that is never reached contributes zero.

  * crossing_values           - discounted first-passage values E[delta^{tau_j}]
                                of a fixed bet, by recursion over the number
                                of successes (no grid or Monte Carlo).
  * fixed_candidates          - candidate fixed bets: a dense grid, plus bets
                                whose short paths hit a milestone exactly.
  * solve_milestone_bellman   - numerical Bellman policy for adaptive bets, with
                                state (log-evidence, first unpaid milestone).
  * policy_bets               - the bets of such a policy at given states.
  * simulate_milestone_policy - Monte Carlo value of such a policy on
                                continuous (unrounded) evidence states.
  * mean_plugin_bet           - the clipped sample-mean plug-in bet.
  * simulate_plugin_crossings - first crossing rounds of the plug-in policy.
  * plugin_receipts           - pathwise discounted receipts from crossing rounds.
"""

import math

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .stopping import _interp_weights


# ===================================================================
# Fixed bets
# ===================================================================

def crossing_values(reward, bet, delta, levels, tolerance=1e-12, n_max=20000):
    """Discounted first-passage values E[delta^{tau_j}] of a fixed bet.

    At round n, live[j, k] is delta^n times the probability of k successes
    without having crossed level j. Each path is absorbed at its first
    crossing. Returns (values, remainder), where remainder bounds the omitted
    value of each level (at most delta * sum(live), even if crossing can fail
    forever); iteration stops once every remainder is below `tolerance`.
    """
    p, m = reward.p, reward.m
    if bet == 0:
        return np.zeros(len(levels)), np.zeros(len(levels))
    up, down = np.log1p(bet * (1 - m)), np.log1p(-bet * m)
    targets = np.log(np.asarray(levels))
    live = np.ones((len(levels), 1))
    hit = np.zeros(len(levels))
    for n in range(1, n_max + 1):
        nxt = np.zeros((len(levels), n + 1))
        nxt[:, :-1] += delta * (1 - p) * live
        nxt[:, 1:] += delta * p * live
        w = np.arange(n + 1) * up + (n - np.arange(n + 1)) * down
        crossed = w[None, :] >= targets[:, None]
        hit += np.sum(np.where(crossed, nxt, 0), axis=1)
        nxt[crossed] = 0
        live = nxt
        remainder = delta * live.sum(axis=1)
        if np.max(remainder) < tolerance:
            return hit, remainder
    raise RuntimeError("First-passage tail tolerance not reached")


def fixed_candidates(reward, levels, extra=(), n_grid=161, max_rounds=12):
    """Candidate fixed bets: a dense grid plus bets aligning short paths with targets.

    Crossing values can jump at an exactly attainable milestone, so smooth
    one-dimensional maximization alone is unsuitable. For each path of at
    most `max_rounds` rounds and each milestone, add the bets just below and
    above the one that hits the milestone exactly. Searching these candidates
    gives a lower bound on the best fixed value; it does not certify global
    optimality. `extra` bets are added as given.
    """
    p, m = reward.p, reward.m
    candidates = set(np.linspace(.01, 1 / m - 1e-6, n_grid))
    candidates.update(extra)
    for n in range(1, max_rounds + 1):
        for k in range(int(n * m) + 1, n + 1):
            peak = (k / n - m) / (m * (1 - m))
            peak = min(peak, 1 / m - 1e-12)
            for a in levels:
                def excess(bet):
                    return (k * math.log1p(bet * (1 - m))
                            + (n - k) * math.log1p(-bet * m) - math.log(a))
                if excess(peak) < 0:
                    continue
                roots = [brentq(excess, 0, peak, xtol=1e-14)]
                if k < n and excess(1 / m - 1e-12) < 0:
                    roots.append(brentq(excess, peak, 1 / m - 1e-12, xtol=1e-14))
                for root in roots:
                    candidates.update(root * (1 + eps) for eps in (-1e-11, 1e-11)
                                      if 0 < root * (1 + eps) < 1 / m)
    return sorted(candidates)


# ===================================================================
# Adaptive bets: numerical Bellman policy
# ===================================================================

def _after_crossing(x, stage, levels, payouts, solutions):
    """Value after one transition, paying every newly crossed milestone."""
    x = np.asarray(x)
    result = np.zeros_like(x)
    next_stage = np.searchsorted(levels, x, side="right")
    for k in range(stage, len(levels)):
        result += payouts[k] * (x >= levels[k])
    for k in range(stage + 1, len(levels)):
        mask = next_stage == k
        if np.any(mask):
            sol = solutions[k]
            result[mask] += np.interp(x[mask], sol["grid"], sol["value"], left=0)
    return result


def solve_milestone_bellman(reward, delta, levels, payouts, n_grid=2400,
                            n_actions=201, lower=-12.0, cap=1.99,
                            extra_actions=(), fixed=None):
    """Numerical Bellman policy for milestone payments, and its discretized value.

    The state is the current log-evidence and the first unpaid milestone.
    Solve the stages backwards, each by policy iteration. Values below a
    milestone and payments at its crossing are kept separate: interpolating
    across that jump would pay too early. The action search uses a uniform
    grid on [0, cap] plus `extra_actions` and the bets that hit each remaining
    milestone exactly; `fixed` pins a single action instead. cap < 1/m
    respects strict positivity of evidence. The lower state boundary is
    absorbing with zero future value. This is a numerical benchmark, not a
    certified exact optimum.

    Returns a dict with the value at log-evidence 0 and, per stage, the grid,
    values, and policy (an index into the action list), as used by
    policy_bets and simulate_milestone_policy.
    """
    p, m = reward.p, reward.m
    levels = np.log(np.asarray(levels, dtype=float))
    payouts = np.asarray(payouts, dtype=float)
    if fixed is None:
        base = np.unique(np.r_[np.linspace(0, cap, n_actions), extra_actions])
        base = base[(base >= 0) & (base <= cap)]
    else:
        base = np.array([fixed])
    solutions = [None] * len(levels)
    max_residual = 0.0
    for stage in reversed(range(len(levels))):
        boundary = levels[stage]
        # Include zero exactly and both sides of the maximum-bet crossing
        # locations, as well as a separate left limit at the payment boundary.
        critical = boundary - np.arange(1, 30) * np.log1p(cap * (1 - m))
        critical = critical[(critical > lower) & (critical < boundary)]
        grid = np.unique(np.r_[np.linspace(lower, boundary - 1e-10, n_grid),
                               0.0, critical, critical - 1e-10])
        grid = grid[(grid >= lower) & (grid < boundary)]
        n = len(grid)
        actions = np.broadcast_to(base, (n, len(base))).copy()
        if fixed is None:
            target_bets = np.expm1(levels[None, stage:] - grid[:, None]) / (1 - m)
            # A tiny upward rounding makes a target bet actually cross in
            # floating point. Values above cap are unavailable.
            target_bets *= 1 + 1e-12
            target_bets = np.where(target_bets <= cap, target_bets, np.nan)
            actions = np.c_[actions, target_bets]
        valid = np.isfinite(actions)
        safe = np.nan_to_num(actions)
        up = grid[:, None] + np.log1p(safe * (1 - m))
        down = grid[:, None] + np.log1p(-safe * m)
        iu, _, wu = _interp_weights(up, grid)
        idn, _, wd = _interp_weights(down, grid)
        continue_up = (up < boundary) & (up >= lower)
        continue_down = down >= lower
        reward_up = delta * p * _after_crossing(up, stage, levels, payouts,
                                                solutions)
        reward_up[up < boundary] = 0
        au = delta * p * continue_up * (1 - wu)
        bu = delta * p * continue_up * wu
        ad = delta * (1 - p) * continue_down * (1 - wd)
        bd = delta * (1 - p) * continue_down * wd
        rows = np.arange(n)
        value = np.zeros(n)
        previous = None
        for iteration in range(300):
            q = (reward_up + au * value[iu] + bu * value[iu + 1]
                 + ad * value[idn] + bd * value[idn + 1])
            q[~valid] = -np.inf
            policy = np.argmax(q, axis=1)
            residual = float(np.max(np.abs(q[rows, policy] - value)))
            if residual < 1e-11:
                break
            if previous is not None and np.array_equal(policy, previous):
                raise RuntimeError("Policy stable but Bellman residual too large")
            previous = policy.copy()
            select = (rows, policy)
            columns = np.c_[rows, iu[select], iu[select] + 1,
                            idn[select], idn[select] + 1]
            coefficients = np.c_[np.ones(n), -au[select], -bu[select],
                                 -ad[select], -bd[select]]
            matrix = csr_matrix((coefficients.ravel(),
                                 (np.repeat(rows, 5), columns.ravel())),
                                shape=(n, n))
            value = spsolve(matrix, reward_up[select])
        else:
            raise RuntimeError("Milestone policy iteration did not converge")
        max_residual = max(max_residual, residual)
        solutions[stage] = {"grid": grid, "value": value, "policy": policy,
                            "base": base, "iterations": iteration + 1}
    return {"value": float(np.interp(0, solutions[0]["grid"],
                                     solutions[0]["value"])),
            "solutions": solutions, "log_levels": levels, "payouts": payouts,
            "p": p, "m": m, "delta": delta, "cap": cap, "lower": lower,
            "residual": max_residual}


def policy_bets(control, stage, w):
    """Bets of a solve_milestone_bellman policy at log-evidence w and stage.

    The nearest grid state selects an action. For a milestone-targeting
    action, the bet is recomputed from the actual evidence w.
    """
    m, levels = control["m"], control["log_levels"]
    bets = np.empty(len(w))
    for k, sol in enumerate(control["solutions"]):
        mask = stage == k
        states = w[mask]
        index, _, weight = _interp_weights(states, sol["grid"])
        chosen = sol["policy"][index + (weight > .5)]
        base = sol["base"]
        fixed = chosen < len(base)
        selected = np.empty(len(chosen))
        selected[fixed] = base[chosen[fixed]]
        target = k + chosen[~fixed] - len(base)
        selected[~fixed] = np.minimum(
            control["cap"],
            np.expm1(levels[target] - states[~fixed]) / (1 - m) * (1 + 1e-12))
        bets[mask] = selected
    return bets


def simulate_milestone_policy(control, n_sim=200000, seed=73819,
                              tail_tolerance=1e-8):
    """Evaluate a solve_milestone_bellman policy without rounding evidence to its grid.

    No payment is ever made on the basis of an interpolated state. Paths
    run until the omitted receipts are below `tail_tolerance`. Returns the
    value, its standard error, and a uniform bound on receipts omitted at the
    deterministic cap.
    """
    delta, p = control["delta"], control["p"]
    levels, payouts = control["log_levels"], control["payouts"]
    horizon = int(np.ceil(np.log(tail_tolerance / payouts.sum()) / np.log(delta)))
    rng = np.random.default_rng(seed)
    w = np.zeros(n_sim)
    stage = np.zeros(n_sim, dtype=int)
    values = np.zeros(n_sim)
    for t in range(1, horizon + 1):
        active = np.flatnonzero(stage < len(levels))
        if len(active) == 0:
            break
        bets = policy_bets(control, stage[active], w[active])
        observations = rng.random(len(active)) < p
        w[active] += np.log1p(bets * (observations - control["m"]))
        for k in range(len(levels)):
            hit = (stage[active] <= k) & (w[active] >= levels[k])
            values[active[hit]] += delta ** t * payouts[k]
        stage[active] = np.maximum(stage[active],
                                   np.searchsorted(levels, w[active], side="right"))
        # The numerical policy has an absorbing lower boundary. Apply that
        # same policy to the true state, rather than extrapolating it.
        stage[active[w[active] < control["lower"]]] = len(levels)
    return {"value": float(values.mean()),
            "se": float(values.std(ddof=1) / np.sqrt(n_sim)),
            "n_sim": n_sim, "cap": horizon,
            "tail_bound": float(delta ** (horizon + 1) * payouts.sum())}


# ===================================================================
# Plug-in bets
# ===================================================================

def mean_plugin_bet(successes, observations, m, initial=.1, margin=.01,
                    pseudocount=0.0):
    """The clipped sample-mean plug-in bet, optionally with symmetric smoothing.

    Bets `initial` before any data, then the growth-optimal bet
    (p_hat - m)^+ / (m (1 - m)) with the (smoothed) mean clipped at 1 - margin.
    """
    if not 0 < margin < 1 - m:
        raise ValueError("The mean-clipping margin must lie in (0, 1-m).")
    cap = (1 - margin - m) / (m * (1 - m))
    if not 0 <= initial <= cap or pseudocount < 0:
        raise ValueError("Invalid initial bet or pseudocount.")
    if observations == 0:
        return np.full_like(successes, initial, dtype=float)
    mean = (successes + pseudocount) / (observations + 2 * pseudocount)
    return np.maximum(np.minimum(mean, 1 - margin) - m, 0) / (m * (1 - m))


def simulate_plugin_crossings(reward, levels, horizon, n_sim=500000,
                              seed=20261021, initial=.1, margin=.01,
                              pseudocount=0.0):
    """First crossing round of each level by the plug-in policy; 0 if never.

    Returns an (n_sim, len(levels)) integer array; zero means the level was
    not reached by the numerical cap `horizon`. Sample every round, including
    at a zero bet. Only paths that have collected all payments are removed.
    Draw a full vector each round so the same seed couples raw and smoothed
    estimates even when paths finish at different times.
    """
    p, m = reward.p, reward.m
    levels = np.asarray(levels, dtype=float)
    if (not 0 <= p <= 1 or not 0 < m < 1 or horizon < 1 or n_sim < 2
            or np.any(levels <= 1) or np.any(np.diff(levels) <= 0)):
        raise ValueError("Invalid model, simulation size, horizon, or levels.")
    mean_plugin_bet(np.zeros(1), 0, m, initial, margin, pseudocount)
    rng = np.random.default_rng(seed)
    targets = np.log(levels)
    hits = np.zeros((n_sim, len(levels)), dtype=np.int32)
    evidence = np.zeros(n_sim)
    successes = np.zeros(n_sim, dtype=np.int32)
    active = np.arange(n_sim)
    for t in range(1, horizon + 1):
        if not len(active):
            break
        draws = rng.random(n_sim)
        observations = (draws[active] < p).astype(float)
        bets = mean_plugin_bet(successes[active], t - 1, m, initial,
                               margin, pseudocount)
        evidence[active] += np.log1p(bets * (observations - m))
        successes[active] += observations.astype(np.int32)
        previous = hits[active]
        crossed = (previous == 0) & (evidence[active, None] >= targets)
        hits[active] = np.where(crossed, t, previous)
        active = active[hits[active, -1] == 0]
    return hits


def plugin_receipts(hits, payouts, delta):
    """Pathwise discounted receipts; totals retain covariance between levels."""
    discounted = np.where(hits > 0, delta ** hits, 0.0)
    return discounted @ np.asarray(payouts)
