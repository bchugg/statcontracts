"""
Decomposing the agent's efficiency loss, and auditing the Bellman benchmark.

For each setting of the four sweeps in agent_stopping.py this script computes,
on a fine grid and without Monte Carlo noise, the values (C = 1, W_0 = 0) of

  * drift threshold at lambda_GRO            (heuristic threshold)
  * CGF threshold at lambda_GRO              (heuristic threshold)
  * exact threshold beta at lambda_GRO       (optimal stopping, growth-optimal bet)
  * exact threshold at the best fixed lambda found by the numerical search
  * Bellman optimum                          (adaptive parameter and stopping)

so that the gap to the Bellman optimum splits into stopping-rule error,
fixed-parameter selection error, and the value of adapting the parameter.
All values come from solving the discretized dynamic program exactly (policy
iteration / policy evaluation with sparse linear solves). The fixed-parameter
search covers the whole Bellman parameter grid, including parameters with
zero or negative drift, plus the growth-optimal and CGF-optimal parameters;
it reports the best parameter found, without certifying global optimality.

It also audits the Bellman benchmark: it compares the policy-iteration
benchmark used in the efficiency figure (bellman_benchmark) against the value-iteration
solver on an 800-point grid used in the submitted version, and checks (for
betting) whether negative bets, which are outside the paper's menu
lambda in [0, 1/m), are ever used.

Produces figures/decomposition_grid.pdf and figures/decomposition_row.pdf, and
prints the audit tables. Results are cached in cache/ like agent_stopping.py.
"""

import numpy as np
import matplotlib.pyplot as plt

from statcontracts.sequential import (
    BettingReward,
    GaussianReward,
    bellman_benchmark,
    beta_threshold,
    sim_bellman,
    sim_threshold,
    solve_bellman_pi,
    solve_bellman_vi,
    threshold_value_grid,
)

from common import cached, save_figure, use_paper_style

N_LAM_FIXED = 35       # coarse search over the fixed parameter ...
N_LAM_REFINE = 11      # ... then a local refinement around the best one
N_SIM = 20000          # Monte Carlo cross-checks (same as the efficiency figure)

STYLE = {
    "drift":     ("#2166ac", r"Drift threshold at $\lambda_{\mathrm{GRO}}$", "o"),
    "cgf":       ("#1b9e77", r"CGF threshold at $\lambda_{\mathrm{GRO}}$", "D"),
    "exact_gro": ("#7570b3", r"Exact threshold at $\lambda_{\mathrm{GRO}}$", "s"),
    "exact_opt": ("#d95f02", r"Best fixed $\lambda$ found (exact threshold)", "^"),
}
ORDER = ["drift", "cgf", "exact_gro", "exact_opt"]


# ===================================================================
# Per-setting evaluation
# ===================================================================

def submitted_vi_benchmark(reward, delta):
    """The value-iteration benchmark of the submitted version (800-point grid)."""
    abs_ln_delta = abs(np.log(delta))
    if isinstance(reward, GaussianReward):
        th = reward.theta
        wr = max(20, 6 * th / abs_ln_delta)
        lam_grid = np.linspace(0.01, max(12 * th, 2.0), 80)
    else:
        m = reward.m
        wr = max(15, 4 * reward.drift() / abs_ln_delta)
        lam_grid = np.linspace(-1.0 / (1 - m) + 0.01, 1.0 / m - 0.01, 200)
    w_vi = np.linspace(-wr, wr, 800)
    G_vi, lam_vi = solve_bellman_vi(reward, delta, 0.0, w_vi, lam_grid)
    return G_vi, w_vi, lam_vi


def evaluate(reward, delta):
    abs_ln_delta = abs(np.log(delta))
    lam_gro = reward.growth_optimal_lambda()
    beta_gro = beta_threshold(reward, lam_gro, delta)

    # Bellman benchmark of the efficiency figure; its grid is reused for
    # everything else.
    G, w_grid, best_lam, res = bellman_benchmark(reward, delta)
    hi = w_grid[-1]

    out = {"lam_gro": lam_gro, "beta_gro": beta_gro,
           "kappa_drift": reward.drift() / abs_ln_delta,
           "kappa_cgf": 1 / reward.cgf_root(delta)}

    # Fixed-parameter threshold rules at lambda_GRO.
    for name, kappa in [("drift", out["kappa_drift"]), ("cgf", out["kappa_cgf"]),
                        ("exact_gro", beta_gro)]:
        out[name] = threshold_value_grid(reward, delta, w_grid, lam_gro, kappa)

    # Check the exact threshold on the grid: optimal stopping at lambda_GRO should
    # stop exactly on {w >= beta}.
    G1, _, _, _ = solve_bellman_pi(reward, delta, 0.0, w_grid,
                                   np.array([lam_gro]))
    stop1 = G1 <= w_grid + 1e-9
    out["grid_boundary_gro"] = float(w_grid[stop1 & (w_grid > 0)].min())
    out["grid_value_gro"] = float(np.interp(0.0, w_grid, G1))

    # Best fixed parameter: coarse search, then local refinement.
    def fixed_value(lam):
        b = beta_threshold(reward, lam, delta)
        if not np.isfinite(b) or b > hi - 10:
            return -np.inf, b
        return threshold_value_grid(reward, delta, w_grid, lam, b), b

    lam_hi = reward.bellman_lambdas()[-1]
    _, lam_cgf = reward.cgf_optimal(delta)
    lams = np.unique(np.r_[np.linspace(0.01, lam_hi, N_LAM_FIXED),
                           lam_gro, lam_cgf])
    vals = [fixed_value(l)[0] for l in lams]
    k = int(np.argmax(vals))
    step = lam_hi / (N_LAM_FIXED - 1)
    fine = np.linspace(max(lams[k] - step, 1e-3),
                       min(lams[k] + step, lam_hi), N_LAM_REFINE)
    fine = np.unique(np.r_[fine, lams[k], lam_cgf])
    fine_vals = [fixed_value(l) for l in fine]
    kk = int(np.argmax([v for v, _ in fine_vals]))
    out["lam_stop"] = float(fine[kk])
    out["exact_opt"], out["beta_stop"] = fine_vals[kk]

    # Bellman optimum (valid bets only) on the fine grid.
    out["bellman"] = float(np.interp(0.0, w_grid, G))
    out["bellman_residual"] = float(res)
    cont = (G > w_grid + 1e-9) & (w_grid > -10)
    out["bellman_lam_range"] = [float(best_lam[cont].min()),
                                float(best_lam[cont].max())]
    out["bellman_lam_at_grid_top"] = bool(
        best_lam[cont].max() >= reward.bellman_lambdas()[-1] - 1e-12)

    # Audit: does allowing negative bets (as in the paper's solver) matter?
    if isinstance(reward, BettingReward):
        m = reward.m
        neg_lams = np.linspace(-1 / (1 - m) + 0.01, 1 / m - 0.01,
                               reward.bellman_n_lambda)
        Gn, lam_n, _, _ = solve_bellman_pi(reward, delta, 0.0, w_grid, neg_lams)
        cont_n = (Gn > w_grid + 1e-9) & (w_grid > -10)
        out["bellman_with_neg_bets"] = float(np.interp(0.0, w_grid, Gn))
        out["neg_bets_used"] = bool((lam_n[cont_n] < 0).any())

    # Monte Carlo cross-checks (common seed).
    mc_bell = sim_bellman(reward, delta, 1.0, G, w_grid, best_lam, n_sim=N_SIM)
    out["mc_bellman"] = float(mc_bell.utilities.mean())

    # Audit: the value-iteration benchmark of the submitted version (800-point
    # grid, at most 2000 sweeps), evaluated with the same random numbers.
    G_vi, w_vi, lam_vi = submitted_vi_benchmark(reward, delta)
    mc_vi = sim_bellman(reward, delta, 1.0, G_vi, w_vi, lam_vi, n_sim=N_SIM)
    out["mc_bellman_vi"] = float(mc_vi.utilities.mean())
    u, trunc = sim_threshold(reward, lam_gro, delta, 1.0, beta_gro, n_sim=N_SIM)
    out["mc_exact_gro"] = float(u.mean())
    out["mc_exact_gro_se"] = float(u.std() / np.sqrt(N_SIM))
    out["mc_truncated_frac"] = trunc
    return out


def cached_eval(tag, make_reward, param, vals, fixed):
    """Evaluate every setting of a sweep, cached on the full configuration."""
    def compute():
        results = []
        for v in vals:
            prm = dict(fixed)
            delta = prm.pop("delta", None)
            if param == "delta":
                delta = float(v)
            else:
                prm[param] = float(v)
            print(f"  {param} = {v}", flush=True)
            results.append(evaluate(make_reward(**prm), delta))
        return results

    key = {"reward": make_reward.__name__, "param": param,
           "vals": [float(v) for v in vals], "fixed": fixed,
           "n_lam_fixed": [N_LAM_FIXED, N_LAM_REFINE], "n_sim": N_SIM}
    return cached(tag, key, compute)


# ===================================================================
# Output
# ===================================================================

def print_tables(name, param, vals, res):
    print(f"\n--- {name}: decomposition (efficiency relative to Bellman) ---")
    print(f"{param:>7} {'drift':>7} {'cgf':>7} {'exact@GRO':>10} "
          f"{'best fixed':>10} {'lam_GRO':>8} {'lam_stop':>8}")
    for v, r in zip(vals, res):
        b = r["bellman"]
        print(f"{v:>7} {r['drift']/b:7.3f} {r['cgf']/b:7.3f} "
              f"{r['exact_gro']/b:10.3f} {r['exact_opt']/b:10.3f} "
              f"{r['lam_gro']:8.3f} {r['lam_stop']:8.3f}")

    print(f"\n--- {name}: Bellman audit ---")
    print(f"{param:>7} {'VI MC':>10} {'PI MC':>10} {'PI grid':>10} "
          f"{'VI/PI MC':>10} {'residual':>9} {'beta':>8} {'grid bdry':>9} "
          f"{'neg bets':>8} {'lam@top':>7} {'trunc':>6}")
    for v, r in zip(vals, res):
        pb = r["mc_bellman_vi"]
        neg = r.get("neg_bets_used")
        print(f"{v:>7} {pb:10.4f} {r['mc_bellman']:10.4f} {r['bellman']:10.4f} "
              f"{pb / r['mc_bellman']:10.4f} {r['bellman_residual']:9.1e} "
              f"{r['beta_gro']:8.3f} {r['grid_boundary_gro']:9.3f} "
              f"{'-' if neg is None else str(neg):>8} "
              f"{str(r['bellman_lam_at_grid_top']):>7} "
              f"{r['mc_truncated_frac']:6.3f}")
    print(f"  exact@GRO: grid {[round(r['exact_gro'], 4) for r in res]}")
    print(f"  exact@GRO: MC   {[round(r['mc_exact_gro'], 4) for r in res]}"
          f" (max SE {max(r['mc_exact_gro_se'] for r in res):.4f})")


def plot_decomposition(panels, filename, nrows, ncols, figsize, legend_ncol,
                       rect_bottom, legend_y, title_fs=20, label_fs=18,
                       legend_fs=15):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    handles = {}
    for i, (ax, panel) in enumerate(zip(axes, panels)):
        bell = np.array([r["bellman"] for r in panel["results"]])
        for key in ORDER:
            c, lab, mk = STYLE[key]
            ratio = np.array([r[key] for r in panel["results"]]) / bell
            handles[key], = ax.plot(panel["xs"], ratio, marker=mk, color=c,
                                    label=lab, markersize=7)
        handles["bellman"] = ax.axhline(1.0, color="gray", lw=1.2, ls="--")
        ax.set_xlabel(panel["xlabel"])
        if i % ncols == 0:
            ax.set_ylabel("Relative efficiency", fontsize=label_fs)
        ax.set_ylim(None, 1.01)
        ax.set_title(panel["title"], fontsize=title_fs)
    labels = [STYLE[k][1] for k in ORDER] + ["Bellman benchmark"]
    fig.legend([handles[k] for k in ORDER + ["bellman"]], labels,
               frameon=False, loc="lower center", ncol=legend_ncol,
               fontsize=legend_fs, bbox_to_anchor=(0.5, legend_y))
    fig.tight_layout(rect=[0, rect_bottom, 1, 1])
    save_figure(fig, filename)


if __name__ == "__main__":
    use_paper_style()
    thetas = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    ps = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    deltas = [0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]

    sweeps = [
        ("subgauss_theta", GaussianReward, "theta", thetas,
         dict(sigma=0.5, delta=0.995), r"$\theta$",
         r"Gaussian reward, varying $\theta$"),
        ("subgauss_delta", GaussianReward, "delta", deltas,
         dict(theta=0.3, sigma=0.5), r"$\delta$",
         r"Gaussian reward, varying $\delta$"),
        ("betting_p", BettingReward, "p", ps,
         dict(m=0.5, delta=0.995), r"$p$", r"Betting reward, varying $p$"),
        ("betting_delta", BettingReward, "delta", deltas,
         dict(p=0.7, m=0.5), r"$\delta$", r"Betting reward, varying $\delta$"),
    ]

    panels = []
    for tag, make_reward, param, vals, fixed, xlabel, title in sweeps:
        print(f"=== {tag} ===", flush=True)
        res = cached_eval("decomp_" + tag, make_reward, param, vals, fixed)
        print_tables(tag, param, vals, res)
        panels.append(dict(xs=vals, results=res, xlabel=xlabel, title=title))

    print("\n=== Decomposition figures ===", flush=True)
    plot_decomposition(panels, "decomposition_grid.pdf", 2, 2, (11, 8.8),
                       legend_ncol=2, rect_bottom=0.13, legend_y=-0.02)
    plot_decomposition(panels, "decomposition_row.pdf", 1, 4, (20, 4.6),
                       legend_ncol=5, rect_bottom=0.14, legend_y=-0.03,
                       title_fs=16, label_fs=15, legend_fs=14)
    print("Done.")
