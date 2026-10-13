"""
Incremental reward schedule: compare stopping strategies.

Reward R_t = C * kappa * (E_t^*)^{1-kappa} on a sub-Gaussian e-process with
Gaussian data; see statcontracts.sequential.incremental for the strategies.
Produces figures/incr_eff_vs_*.pdf.
"""

import numpy as np
import matplotlib.pyplot as plt

from statcontracts.sequential import GaussianReward
from statcontracts.sequential.incremental import (
    sim_myopic,
    sim_snell,
    sim_threshold_adaptive,
    sim_threshold_oracle,
    solve_snell_2d,
)

from common import save_figure, use_paper_style

KAPPA = 0.5
COLORS = ["#2166ac", "#e7298a", "#d95f02", "#1b9e77"]
NAMES = ["Snell (2D DP)", "Threshold (oracle)", "Myopic", "Threshold (adaptive)"]


def run_one(theta, sigma, delta, C, kappa, n_sim):
    reward = GaussianReward(theta, sigma)
    print("    Solving 2D DP ...", end=" ", flush=True)
    phi, mg, gg = solve_snell_2d(reward, delta, np.log(C), kappa)
    print("done", flush=True)
    kw = dict(n_sim=n_sim)
    return (
        np.mean(sim_snell(reward, delta, C, kappa, phi, mg, gg, **kw)),
        np.mean(sim_threshold_oracle(reward, delta, C, kappa, **kw)),
        np.mean(sim_myopic(reward, delta, C, kappa, **kw)),
        np.mean(sim_threshold_adaptive(reward, delta, C, kappa, **kw)),
    )


def sweep(param_name, param_vals, fixed_kwargs, n_sim=20000):
    results = {n: [] for n in NAMES}
    for val in param_vals:
        kwargs = dict(fixed_kwargs)
        kwargs[param_name] = val
        print(f"  {param_name} = {val}", flush=True)
        vals = run_one(n_sim=n_sim, **kwargs)
        for name, v in zip(NAMES, vals):
            results[name].append(v)
    return results


def plot_efficiency(xs, results, xlabel, filename, names=None, colors=None):
    if names is None:
        names = NAMES
    if colors is None:
        colors = COLORS
    ref = np.array(results[NAMES[0]])
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color in zip(names, colors):
        vals = np.array(results[name])
        ax.plot(xs, vals / ref, marker='o', color=color, label=name,
                markersize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative efficiency")
    ax.set_ylim(None, 1.02)
    ax.axhline(1.0, color='gray', lw=0.8, ls='--')
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    save_figure(fig, filename)


if __name__ == "__main__":
    use_paper_style(**{"legend.fontsize": 14})
    N_SIM = 20000

    print("=== Incremental reward (sub-Gaussian) ===", flush=True)

    MAIN_NAMES = ["Snell (2D DP)", "Threshold (oracle)", "Threshold (adaptive)"]
    MAIN_COLORS = ["#2166ac", "#e7298a", "#1b9e77"]

    print("Efficiency vs theta", flush=True)
    thetas = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    res = sweep("theta", thetas,
                dict(sigma=0.5, delta=0.995, C=1.0, kappa=KAPPA),
                n_sim=N_SIM)
    plot_efficiency(thetas, res, r"$\mu$", "incr_eff_vs_mu.pdf",
                    names=MAIN_NAMES, colors=MAIN_COLORS)
    plot_efficiency(thetas, res, r"$\mu$", "incr_eff_vs_mu_all.pdf")

    print("Efficiency vs delta", flush=True)
    deltas = np.array([0.98, 0.99, 0.993, 0.995, 0.997, 0.999])
    res = sweep("delta", deltas,
                dict(theta=0.2, sigma=0.5, C=1.0, kappa=KAPPA),
                n_sim=N_SIM)
    plot_efficiency(deltas, res, r"$\delta$", "incr_eff_vs_delta.pdf",
                    names=MAIN_NAMES, colors=MAIN_COLORS)
    plot_efficiency(deltas, res, r"$\delta$", "incr_eff_vs_delta_all.pdf")

    print("Done.", flush=True)
