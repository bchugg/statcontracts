"""
Agent stopping strategies as distributional knowledge degrades (Figure 2).

Two reward families (see statcontracts.sequential.rewards):

(A) Gaussian:  R_t = C * exp(sum [lambda_i X_i - lambda_i^2/8]),
    X_i ~ N(theta, sigma^2), utility v = delta^t log R_t.
(B) Betting:   R_t = C * prod (1 + lambda_i (X_i - m)),
    X_i ~ Bernoulli(p), testing H_0: E[X] <= m. As a simple baseline we also
    include the mean-only Hoeffding reward (HoeffdingReward).

Strategies are described in statcontracts.sequential.simulate. The Bellman
optimum is the denominator: efficiency = E[discounted log-utility of
strategy] / E[discounted log-utility of Bellman]. Common random numbers
(fixed seed) couple strategies to reduce ratio variance.

Results are cached in cache/; pass --recompute to rerun the sweeps.
"""

import numpy as np
import matplotlib.pyplot as plt

from statcontracts.sequential import (
    BettingReward,
    GaussianReward,
    HoeffdingReward,
    bellman_benchmark,
    sim_bellman,
    sim_cgf_aware,
    sim_cgf_optimal,
    sim_drift_aware,
    sim_plug_in,
)

from common import cached, save_figure, use_paper_style


# Per-strategy display style: color, legend label, marker.
STRAT_STYLE = {
    "CGF-optimal":           ("#7570b3", r"CGF-optimal (bet and threshold)", "v"),
    "CGF-aware threshold":   ("#1b9e77", r"CGF-aware threshold", "D"),
    "drift-aware threshold": ("#2166ac", r"Drift-aware threshold", "o"),
    "Hoeffding (mean only)": ("#d95f02", r"Hoeffding (mean only)", "s"),
    "plug-in threshold":     ("#e7298a", r"Plug-in threshold", "^"),
}

# Legend order for the combined grid (Bellman handled separately, drawn last).
LEGEND_ORDER = ["CGF-optimal", "CGF-aware threshold", "drift-aware threshold",
                "Hoeffding (mean only)", "plug-in threshold"]


def evaluate(reward, delta, C, n_sim, hoeffding=None):
    """Mean discounted log-utility of the Bellman optimum and each strategy."""
    G, w_grid, lam_opt, _ = bellman_benchmark(reward, delta, np.log(C))

    def mean(result):
        return float(np.mean(result.utilities))

    out = {
        "Bellman": mean(sim_bellman(reward, delta, C, G, w_grid, lam_opt,
                                    n_sim=n_sim)),
        "CGF-aware threshold": mean(sim_cgf_aware(reward, delta, C, n_sim=n_sim)),
        "drift-aware threshold": mean(sim_drift_aware(reward, delta, C,
                                                      n_sim=n_sim)),
        "CGF-optimal": mean(sim_cgf_optimal(reward, delta, C, n_sim=n_sim)),
        "plug-in threshold": mean(sim_plug_in(reward, delta, C, n_sim=n_sim)),
    }
    if hoeffding is not None:
        # Mean-only agent, compared against the betting Bellman optimum.
        out["Hoeffding (mean only)"] = mean(sim_drift_aware(hoeffding, delta, C,
                                                            n_sim=n_sim))
    return out


def eval_gaussian(theta, sigma, delta, C, n_sim):
    return evaluate(GaussianReward(theta, sigma), delta, C, n_sim)


def eval_betting(p, m, delta, C, n_sim):
    return evaluate(BettingReward(p, m), delta, C, n_sim,
                    hoeffding=HoeffdingReward(p, m))


def sweep(eval_fn, param, vals, fixed, strat_names, n_sim):
    results = {n: [] for n in (["Bellman"] + strat_names)}
    for v in vals:
        kw = dict(fixed)
        kw[param] = v
        print(f"  {param} = {v}", flush=True)
        d = eval_fn(n_sim=n_sim, **kw)
        for k in results:
            results[k].append(d[k])
    return results


def cached_sweep(tag, eval_fn, param, vals, fixed, strat_names, n_sim):
    """sweep() with a JSON result cache keyed on the full configuration."""
    key = {
        "param": param, "vals": [float(v) for v in vals],
        "fixed": {k: float(v) for k, v in fixed.items()},
        "strats": list(strat_names), "n_sim": int(n_sim),
    }
    return cached(tag, key, lambda: sweep(eval_fn, param, vals, fixed,
                                          strat_names, n_sim))


def plot_eff(xs, results, strat_order, xlabel, filename, ylim_lo=None,
             show_legend=True):
    ub = np.array(results["Bellman"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in strat_order:
        c, lab, mk = STRAT_STYLE[name]
        ratio = np.array(results[name]) / ub
        ax.plot(xs, ratio, marker=mk, color=c, label=lab, markersize=7)
    ax.axhline(1.0, color="gray", lw=1.2, ls="--", label="Bellman optimum")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative efficiency")
    ax.set_ylim(ylim_lo, 1.03)
    if show_legend:
        ax.legend(frameon=False, loc="lower right", fontsize=18)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_grid(panels, filename, nrows=2, ncols=2, figsize=(11, 8.5),
              legend_ncol=3, legend_y=-0.02, rect_bottom=0.1, title_fs=20,
              label_fs=18, legend_fs=16):
    """Collapse the four efficiency sweeps into one figure with a single
    shared legend below. `panels` is a list of dicts with keys
    xs, results, strat_order, xlabel, title (row-major).

    Layout is configurable: the default is the 2x2 grid (arXiv version); pass
    nrows=1, ncols=4 with a wider figsize for the one-row AAAI version. Only the
    leftmost panel in each row carries the y-axis label (the rest share the same
    "relative efficiency" meaning). The legend is wrapped so it fits within the
    canvas width -- a legend that overflows inflates the saved bbox and shrinks
    the plots when the figure is scaled to \\textwidth in the paper.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    handles = {}
    for i, (ax, panel) in enumerate(zip(axes, panels)):
        ub = np.array(panel["results"]["Bellman"])
        for name in panel["strat_order"]:
            c, lab, mk = STRAT_STYLE[name]
            ratio = np.array(panel["results"][name]) / ub
            line, = ax.plot(panel["xs"], ratio, marker=mk, color=c,
                            label=lab, markersize=7)
            handles[name] = line
        hb = ax.axhline(1.0, color="gray", lw=1.2, ls="--")
        handles["Bellman optimum"] = hb
        ax.set_xlabel(panel["xlabel"])
        if i % ncols == 0:
            ax.set_ylabel("Relative efficiency", fontsize=label_fs)
        ax.set_ylim(None, 1.03)
        ax.set_title(panel["title"], fontsize=title_fs)

    order = [n for n in LEGEND_ORDER if n in handles] + ["Bellman optimum"]
    labels = [STRAT_STYLE[n][1] if n in STRAT_STYLE else n for n in order]
    fig.legend([handles[n] for n in order], labels, frameon=False,
               loc="lower center", ncol=legend_ncol, fontsize=legend_fs,
               bbox_to_anchor=(0.5, legend_y))
    fig.tight_layout(rect=[0, rect_bottom, 1, 1])
    save_figure(fig, filename)


if __name__ == "__main__":
    use_paper_style()
    N_SIM = 20000
    deltas = np.array([0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999])

    SG = ["CGF-optimal", "CGF-aware threshold", "drift-aware threshold",
          "plug-in threshold"]
    BT = ["CGF-optimal", "CGF-aware threshold", "drift-aware threshold",
          "Hoeffding (mean only)", "plug-in threshold"]

    # ---- Gaussian ----
    print("=== Gaussian reward process ===", flush=True)

    # theta is the Gaussian mean; the drift at the GRO parameter is 2 theta^2.
    print("vs theta (mean)", flush=True)
    thetas = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    res_sg_theta = cached_sweep("subgauss_theta", eval_gaussian, "theta", thetas,
                                dict(sigma=0.5, delta=0.995, C=1.0), SG, N_SIM)
    plot_eff(thetas, res_sg_theta, SG, r"$\theta$", "subgauss_eff_vs_theta.pdf")

    print("vs delta", flush=True)
    res_sg_delta = cached_sweep("subgauss_delta", eval_gaussian, "delta", deltas,
                                dict(theta=0.3, sigma=0.5, C=1.0), SG, N_SIM)
    plot_eff(deltas, res_sg_delta, SG, r"$\delta$", "subgauss_eff_vs_delta.pdf",
             show_legend=False)

    # ---- Betting / bounded ----
    print("=== Betting reward process (Bernoulli) ===", flush=True)

    print("vs p", flush=True)
    ps = np.array([0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    res_bt_p = cached_sweep("betting_p", eval_betting, "p", ps,
                            dict(m=0.5, delta=0.995, C=1.0), BT, N_SIM)
    plot_eff(ps, res_bt_p, BT, r"$p$", "betting_eff_vs_p.pdf")

    print("vs delta", flush=True)
    res_bt_delta = cached_sweep("betting_delta", eval_betting, "delta", deltas,
                                dict(p=0.7, m=0.5, C=1.0), BT, N_SIM)
    plot_eff(deltas, res_bt_delta, BT, r"$\delta$", "betting_eff_vs_delta.pdf",
             show_legend=False)

    # ---- Combined figures for the paper ----
    print("=== Combined efficiency figures ===", flush=True)
    panels = [
        dict(xs=thetas, results=res_sg_theta, strat_order=SG, xlabel=r"$\theta$",
             title=r"Gaussian reward, varying $\theta$"),
        dict(xs=deltas, results=res_sg_delta, strat_order=SG, xlabel=r"$\delta$",
             title=r"Gaussian reward, varying $\delta$"),
        dict(xs=ps, results=res_bt_p, strat_order=BT, xlabel=r"$p$",
             title=r"Betting reward, varying $p$"),
        dict(xs=deltas, results=res_bt_delta, strat_order=BT, xlabel=r"$\delta$",
             title=r"Betting reward, varying $\delta$"),
    ]
    # 2x2 grid (arXiv / single-column version).
    plot_grid(panels, "efficiency_grid.pdf",
              nrows=2, ncols=2, figsize=(11, 8.5),
              legend_ncol=3, legend_y=-0.02, rect_bottom=0.1)
    # 1x4 row (AAAI two-column figure* version).
    plot_grid(panels, "efficiency_row.pdf",
              nrows=1, ncols=4, figsize=(22, 4.2),
              legend_ncol=6, legend_y=-0.04, rect_bottom=0.14, title_fs=19,
              legend_fs=22)

    print("Done.", flush=True)
