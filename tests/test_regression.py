"""Pinned outputs of the sequential simulators and solvers.

Small-scale versions of the sequential experiments, with their values computed
under NumPy 2.5 / SciPy 1.18. The simulators are deterministic given the
seed, so a change to the order of random draws or to the solvers'
arithmetic shows up here. The large-jump values are those reported in the
paper.
"""

import numpy as np
import pytest

from statcontracts.sequential import (
    BettingReward,
    GaussianReward,
    HoeffdingReward,
    bellman_benchmark,
    beta_threshold,
    capped_threshold_value,
    sim_bellman,
    sim_cgf_aware,
    sim_cgf_optimal,
    sim_drift_aware,
    sim_plug_in,
)
from statcontracts.sequential import incremental, milestones

REFERENCE = {
    "gaussian": {
        "Bellman": 13.568435758965613,
        "CGF-aware threshold": 13.561943803209102,
        "drift-aware threshold": 13.567529910804188,
        "CGF-optimal": 13.56441834500637,
        "plug-in threshold": 12.313108114698572,
    },
    "betting": {
        "Bellman": 6.376364546370933,
        "CGF-aware threshold": 6.363433212266626,
        "drift-aware threshold": 6.347829716652681,
        "Hoeffding (mean only)": 6.1679236395569506,
        "CGF-optimal": 6.388083717472993,
        "plug-in threshold": 4.949941183982613,
    },
    "beta_gaussian": 36.48702097702342,
    "beta_betting": 17.123172507630816,
    "incremental": [1.5255416896257115, 1.382193842179648,
                    0.6443341143826837, 1.1785208445657593],
    # Large-jump example: (lambda, threshold) -> capped value.
    "large_jump": {"cgf": 0.18428477076732694, "cgf_exact": 0.19123690068344507,
                   "feasible_exact": 0.26111366298268796},
    "milestones": {"fixed_1.2": 1.6859496693767557, "bellman_grid": 2.0741591270276576,
                   "bellman_mc": 2.037921867670558},
}

RTOL = 1e-12


def strategy_values(reward, delta=0.995, C=1.0, n_sim=2000):
    G, w_grid, lam_opt, _ = bellman_benchmark(reward, delta, np.log(C))
    kw = dict(n_sim=n_sim)
    return {
        "Bellman": sim_bellman(reward, delta, C, G, w_grid, lam_opt, **kw),
        "CGF-aware threshold": sim_cgf_aware(reward, delta, C, **kw),
        "drift-aware threshold": sim_drift_aware(reward, delta, C, **kw),
        "CGF-optimal": sim_cgf_optimal(reward, delta, C, **kw),
        "plug-in threshold": sim_plug_in(reward, delta, C, **kw),
    }


@pytest.mark.slow
@pytest.mark.parametrize("family", ["gaussian", "betting"])
def test_agent_stopping_values(family):
    if family == "gaussian":
        results = strategy_values(GaussianReward(0.3, 0.5))
    else:
        results = strategy_values(BettingReward(0.7, 0.5))
        results["Hoeffding (mean only)"] = sim_drift_aware(
            HoeffdingReward(0.7, 0.5), 0.995, 1.0, n_sim=2000)
    for name, expected in REFERENCE[family].items():
        np.testing.assert_allclose(results[name].utilities.mean(), expected,
                                   rtol=RTOL, err_msg=name)


def test_beta_threshold_values():
    np.testing.assert_allclose(beta_threshold(GaussianReward(0.3, 0.5), 1.2, 0.995),
                               REFERENCE["beta_gaussian"], rtol=RTOL)
    np.testing.assert_allclose(beta_threshold(BettingReward(0.7, 0.5), 0.8, 0.995),
                               REFERENCE["beta_betting"], rtol=RTOL)


def test_incremental_values():
    g, delta, C, kappa = GaussianReward(0.2, 0.5), 0.99, 1.0, 0.5
    phi, mg, gg = incremental.solve_snell_2d(g, delta, 0.0, kappa, n_m=60, n_g=20)
    kw = dict(n_sim=2000)
    values = [
        incremental.sim_snell(g, delta, C, kappa, phi, mg, gg, **kw),
        incremental.sim_threshold_oracle(g, delta, C, kappa, **kw),
        incremental.sim_myopic(g, delta, C, kappa, **kw),
        incremental.sim_threshold_adaptive(g, delta, C, kappa, **kw),
    ]
    np.testing.assert_allclose([np.mean(v) for v in values], REFERENCE["incremental"],
                               rtol=RTOL)


def test_large_jump_values():
    reward = BettingReward(0.02, 0.01)
    rho, lam = reward.interior_cgf_optimal(0.9)
    for name, bet, threshold in [
            ("cgf", lam, 1 / rho),
            ("cgf_exact", lam, beta_threshold(reward, lam, 0.9)),
            ("feasible_exact", 15.5, beta_threshold(reward, 15.5, 0.9))]:
        value = capped_threshold_value(reward, bet, 0.9, threshold, horizon=600)
        np.testing.assert_allclose(value["value"], REFERENCE["large_jump"][name],
                                   rtol=RTOL, err_msg=name)


def test_milestone_values():
    reward, delta, levels = BettingReward(0.7, 0.5), 0.9, (2.0, 4.0, 8.0)
    payouts = np.array(levels) * np.array([0.1, 0.1, 0.8])
    fixed = milestones.crossing_values(reward, 1.2, delta, levels)[0] @ payouts
    control = milestones.solve_milestone_bellman(reward, delta, levels, payouts,
                                                 600, 51)
    mc = milestones.simulate_milestone_policy(control, n_sim=2000)
    ref = REFERENCE["milestones"]
    np.testing.assert_allclose(fixed, ref["fixed_1.2"], rtol=RTOL)
    np.testing.assert_allclose(control["value"], ref["bellman_grid"], rtol=RTOL)
    np.testing.assert_allclose(mc["value"], ref["bellman_mc"], rtol=RTOL)
