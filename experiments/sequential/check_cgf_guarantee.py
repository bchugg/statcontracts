"""
Numerical checks of the CGF theorem's finite-cap formulation.

Propagate first-passage probabilities directly, without Monte Carlo or a
discretized state grid. Check the change-of-measure identity, tilted hitting
bound, and finite-cap error for positive, zero, and negative original drift.
These checks complement the proof; they do not replace it.

Run: python check_cgf_guarantee.py
"""

import numpy as np
from scipy.optimize import brentq

from statcontracts.sequential import BettingReward


def cgf_root(up, down, p, delta):
    def equation(rho):
        return delta * (p * np.exp(rho * up)
                        + (1 - p) * np.exp(rho * down)) - 1

    hi = 1.0
    while equation(hi) < 0:
        hi *= 2
    return brentq(equation, 0, hi, xtol=1e-14)


def check_threshold(up, down, p, delta, rho, horizon, caps,
                    analytic_value=None):
    """Track surviving paths by their number of up-steps under P and P_rho."""
    kappa = 1 / rho
    upper = 1 / (np.e * rho)
    fraction = (1 + rho * up) * np.exp(-rho * up)
    mean_abs = p * abs(up) + (1 - p) * abs(down)
    tilted_up = delta * p * np.exp(rho * up)
    tilted_down = delta * (1 - p) * np.exp(rho * down)
    assert abs(tilted_up + tilted_down - 1) < 1e-12
    live = np.array([1.0])
    live_tilted = np.array([1.0])
    hit_value = tilted_hit_value = 0.0
    snapshots = {}
    tolerance = 2e-10 * max(1, upper)

    for n in range(1, horizon + 1):
        next_live = np.zeros(n + 1)
        next_live[:-1] += (1 - p) * live
        next_live[1:] += p * live
        next_tilted = np.zeros(n + 1)
        next_tilted[:-1] += tilted_down * live_tilted
        next_tilted[1:] += tilted_up * live_tilted
        wealth = n * down + np.arange(n + 1) * (up - down)
        hit = wealth >= kappa

        hit_value += delta ** n * (next_live[hit] @ wealth[hit])
        tilted_hit_value += (next_tilted[hit]
                            @ (wealth[hit] * np.exp(-rho * wealth[hit])))
        assert abs(hit_value - tilted_hit_value) <= tolerance

        next_live[hit] = next_tilted[hit] = 0
        live, live_tilted = next_live, next_tilted
        assert live_tilted.sum() <= np.exp(rho * kappa) * delta ** n + 1e-11
        if n in caps:
            capped_value = hit_value + delta ** n * (live @ wealth)
            error = delta ** n * (n * mean_abs + delta * (kappa + up))
            snapshots[n] = (capped_value, live.sum(), error)

    # The uncounted tilted payoff is nonnegative and at most upper times
    # the tilted survival probability. This brackets the infinite-horizon J.
    lower_j = tilted_hit_value
    upper_j = lower_j + upper * live_tilted.sum()
    assert lower_j >= fraction * upper - tolerance
    assert upper_j <= upper + tolerance
    if analytic_value is not None:
        assert lower_j - tolerance <= analytic_value <= upper_j + tolerance
    for capped_value, _, error in snapshots.values():
        assert max(abs(capped_value - lower_j),
                   abs(capped_value - upper_j)) <= error + tolerance
        assert capped_value >= fraction * upper - error - tolerance
    return snapshots


def adaptive_value(actions, p, delta, horizon):
    """Exact backward induction on reachable integer states; stop by horizon."""
    radius = max(abs(jump) for action in actions for jump in action)
    value = np.arange(-radius * horizon, radius * horizon + 1, dtype=float)
    for t in range(horizon - 1, -1, -1):
        wealth = np.arange(-radius * t, radius * t + 1)
        continuation = []
        for up, down in actions:
            v_up = value[wealth + up + radius * (t + 1)]
            v_down = value[wealth + down + radius * (t + 1)]
            continuation.append(delta * (p * v_up + (1 - p) * v_down))
        best_continuation = np.max(continuation, axis=0)
        value = (best_continuation if t == 0 else
                 np.maximum(wealth, best_continuation))
    return float(value[0])


def main():
    # Even when one compulsory trial has negative expected log payoff,
    # stopping at time zero is not an option.
    assert abs(adaptive_value(((1, -2),), .25, .8, 1) + 1.0) < 1e-12
    # For +/-1 steps the crossing position is exactly ceil(kappa), and the
    # tilted walk hits a.s., giving an independent analytic value for J.
    caps = set(range(1, 101)) | {250, 800}
    for p in (0.25, 0.5, 0.75):
        for delta in (0.5, 0.8, 0.95):
            rho = cgf_root(1, -1, p, delta)
            boundary = np.ceil(1 / rho)
            exact_j = boundary * np.exp(-rho * boundary)
            check_threshold(1, -1, p, delta, rho, 800, caps, exact_j)
    print("PASS: 9 lattice cases, including positive, zero, and negative drift;")
    print("      tilt identity, tilted hitting bound, and 102 caps per case.")

    actions = ((1, -2), (2, -1))
    for p in (0.3, 0.5, 0.8):
        for delta in (0.7, 0.95):
            roots = [cgf_root(up, down, p, delta) for up, down in actions]
            rho = min(roots)
            upper = 1 / (np.e * rho)
            up, down = actions[int(np.argmin(roots))]
            snapshots = check_threshold(up, down, p, delta, rho, 800,
                                        {1, 5, 20, 100, 200})
            fraction = (1 + rho * up) * np.exp(-rho * up)
            for horizon, (fixed, _, error) in snapshots.items():
                optimal = adaptive_value(actions, p, delta, horizon)
                assert fixed <= optimal + 1e-10
                assert optimal <= upper + 1e-10
                assert fixed >= fraction * optimal - error - 1e-10
    print("PASS: adaptive Bellman comparisons for 6 action families and 5 horizons.")

    # Paper settings: two problematic CGF bets and one positive-drift control.
    for p, delta in ((0.7, 0.9), (0.55, 0.995), (0.7, 0.95)):
        rho, lam = BettingReward(p, 0.5).cgf_optimal(delta)
        up, down = np.log1p(lam / 2), np.log1p(-lam / 2)
        snapshots = check_threshold(up, down, p, delta, rho, 8000,
                                    {2500, 8000})
        value, survival, error = snapshots[2500]
        print(f"p={p:g}, delta={delta:g}: lambda={lam:.8f}, "
              f"drift={p * up + (1-p) * down:.8f}, "
              f"P(T>2500)={survival:.6f}, J_2500={value:.8f}, "
              f"J_8000={snapshots[8000][0]:.8f}, "
              f"cap-error bound={error:.3g}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
