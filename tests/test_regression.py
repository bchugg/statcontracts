"""Pinned outputs of the sequential simulators and solvers.

Small-scale versions of the sequential experiments, with their values computed
under NumPy 2.5 / SciPy 1.18. The simulators are deterministic given the
seed, so a change to the order of random draws or to the solvers'
arithmetic shows up here.
"""

import numpy as np
import pytest

from statcontracts.sequential import (
    BettingReward,
    GaussianReward,
    HoeffdingReward,
    bellman_benchmark,
    beta_threshold,
    sim_bellman,
    sim_cgf_aware,
    sim_cgf_optimal,
    sim_drift_aware,
    sim_plug_in,
)
from statcontracts.sequential import incremental

REFERENCE = {
    "gaussian": {
        "Bellman": 13.568435758965613,
        "CGF-aware threshold": 13.561943803209102,
        "drift-aware threshold": 13.567529910804188,
        "CGF-optimal": 13.56441834500637,
        "plug-in threshold": 12.309683397173398,
    },
    "betting": {
        "Bellman": 6.376364546370933,
        "CGF-aware threshold": 6.363433212266626,
        "drift-aware threshold": 6.347829716652681,
        "Hoeffding (mean only)": 6.1679236395569506,
        "CGF-optimal": 6.388083717472993,
        "plug-in threshold": 4.9730552332567095,
    },
    "beta_gaussian": 36.48702097702342,
    "beta_betting": 17.123172507630816,
    "incremental": [1.5255416896257115, 1.382193842179648,
                    0.6443341143826837, 1.1785208445657593],
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
