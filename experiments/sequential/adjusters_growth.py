"""
Growth rates of admissible adjusters and their null-budget allocations.

See statcontracts.sequential.adjusters for the adjusters and their
cumulative milestone budgets B(u) = A(e^u)/e^u + int_0^u A(e^v)/e^v dv.

Left panel:  A(y) vs y (log-log) -- the adjuster as a function of its input.
Right panel: cumulative milestone budget, including its atom A(1) at zero.
"""

import numpy as np
import matplotlib.pyplot as plt

from statcontracts.sequential.adjusters import (
    LN2,
    budget_choe,
    budget_koolen,
    budget_mixture,
    budget_shafer,
    choe,
    koolen,
    mixture,
    shafer,
)

from common import save_figure, use_paper_style


def _plot_adjusters(ax, adjusters, y, colors, legend_ncol=1):
    """Left panel: the adjusters A(y) themselves (log-log)."""
    for (label, ay, _budget, ls), c in zip(adjusters, colors):
        ax.plot(y, np.clip(ay, 1e-3, None), color=c, linestyle=ls, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-1)
    ax.set_xlabel(r"running maximum $y = E^*$")
    ax.set_ylabel(r"adjuster $A(y)$")
    ax.legend(frameon=False, loc="upper left", ncol=legend_ncol,
              columnspacing=1.0, handlelength=1.6)


def _plot_budgets(ax, adjusters, w, colors):
    """Right panel: cumulative budget; the vertical segment and dot show A(1)."""
    for (label, _ay, budget, ls), c in zip(adjusters, colors):
        ax.plot(w, budget, color=c, linestyle=ls, label=label)
        ax.vlines(0, 0, budget[0], color=c, lw=1.2, alpha=.65)
        ax.plot(0, budget[0], "o", color=c, markersize=5, clip_on=False)
    ax.set_xlabel(r"log-evidence $u = \log a$")
    ax.set_ylabel("cumulative budget")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-.1, w[-1])


def main():
    colors = ["#2166ac", "#e7298a", "#d95f02", "#7570b3", "#1b9e77", "#e6ab02"]
    y = np.logspace(0.0, 5.0, 2000)   # running maximum E* from 1 to 1e5
    w = np.linspace(0.0, 10.0, 4001)  # log milestone level

    # (label, A(y), cumulative budget through w, linestyle)
    adjusters = [
        (r"$A_\kappa,\ \kappa=0.5$", shafer(y, 0.5), budget_shafer(w, 0.5), "--"),
        (r"$\sqrt{y}-1$",            choe(y),        budget_choe(w),        "-"),
        (r"KV",                      koolen(y),      budget_koolen(w),      "-"),
        (r"$A^{\mathrm{mix}}_a,\ a=0.5$", mixture(y, 0.5), budget_mixture(w, 0.5), "-"),
        (r"$A^{\mathrm{mix}}_a,\ a=1$",   mixture(y, 1.0), budget_mixture(w, 1.0), "-"),
        (r"$A^{\mathrm{mix}}_a,\ a=2$",   mixture(y, 2.0), budget_mixture(w, 2.0), "-"),
    ]

    for _, _, budget, _ in adjusters:
        assert np.all(np.diff(budget) >= -1e-10)
        assert 0 <= budget[0] <= budget[-1] <= 1 + 1e-6
    assert abs(adjusters[2][2][0] - 1 / (2 * LN2)) < 1e-12
    assert abs(adjusters[4][2][0] - .5) < 1e-10

    # Two-panel version (arXiv / single-column): adjusters + budget allocation.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
    _plot_adjusters(axL, adjusters, y, colors)
    _plot_budgets(axR, adjusters, w, colors)
    fig.tight_layout()
    save_figure(fig, "adjuster_growth.pdf")

    # Single-panel version (AAAI, space-limited): only the adjusters A(y).
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    _plot_adjusters(ax, adjusters, y, colors)
    fig.tight_layout()
    save_figure(fig, "adjuster_growth_single.pdf")

    # Stacked single-column version (AAAI): adjusters on top, budgets below,
    # so both panels fit the footprint of one column. The panels are short, so
    # shrink the fonts (tuned larger for the wide figure) to avoid overlap.
    with plt.rc_context({
        "axes.labelsize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }):
        # Drop the a=2 mixture here (last entry) to keep the compact legend
        # uncrowded; the other figures still show the full set.
        stacked = adjusters[:-1]
        fig, (axT, axB) = plt.subplots(2, 1, figsize=(6.5, 4.6))
        _plot_adjusters(axT, stacked, y, colors, legend_ncol=2)
        _plot_budgets(axB, stacked, w, colors)
        fig.tight_layout()
        save_figure(fig, "adjuster_growth_stacked.pdf")


if __name__ == "__main__":
    use_paper_style(**{"legend.fontsize": 16})
    main()
