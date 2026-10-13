"""
Growth rates of several admissible adjusters (and their h-densities).

See statcontracts.sequential.adjusters for the adjusters and the induced
densities h(w) = A(e^w)/e^w.

Left panel:  A(y) vs y (log-log) -- the adjuster as a function of its input.
Right panel: h(w) = A(e^w)/e^w vs w -- the induced density on the log scale.
"""

import numpy as np
import matplotlib.pyplot as plt

from statcontracts.sequential.adjusters import (
    choe,
    h_choe,
    h_koolen,
    h_mixture,
    h_shafer,
    koolen,
    mixture,
    shafer,
)

from common import save_figure, use_paper_style


def _plot_adjusters(ax, adjusters, y, colors, legend_ncol=1):
    """Left panel: the adjusters A(y) themselves (log-log)."""
    for (label, ay, _hw, ls), c in zip(adjusters, colors):
        ax.plot(y, np.clip(ay, 1e-3, None), color=c, linestyle=ls, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-1)
    ax.set_xlabel(r"running maximum $y = E^*$")
    ax.set_ylabel(r"adjuster $A(y)$")
    ax.legend(frameon=False, loc="upper left", ncol=legend_ncol,
              columnspacing=1.0, handlelength=1.6)


def _plot_densities(ax, adjusters, w, colors):
    """Right panel: the induced densities h(w) = A(e^w)/e^w."""
    for (label, _ay, hw, ls), c in zip(adjusters, colors):
        ax.plot(w, hw, color=c, linestyle=ls, label=label)
    ax.set_xlabel(r"$w = \log y$")
    ax.set_ylabel(r"$h(w) = A(e^w)/e^w$")


def main():
    colors = ["#2166ac", "#e7298a", "#d95f02", "#7570b3", "#1b9e77", "#e6ab02"]
    y = np.logspace(0.0, 5.0, 2000)   # running maximum E* from 1 to 1e5
    w = np.linspace(1e-3, 10.0, 2000)  # w = log y

    # (label, A(y), h(w), linestyle)
    adjusters = [
        (r"$A_\kappa,\ \kappa=0.5$", shafer(y, 0.5), h_shafer(w, 0.5), "--"),
        (r"$\sqrt{y}-1$",            choe(y),        h_choe(w),        "-"),
        (r"KV",                      koolen(y),      h_koolen(w),      "-"),
        (r"$A^{\mathrm{mix}}_a,\ a=0.5$", mixture(y, 0.5), h_mixture(w, 0.5), "-"),
        (r"$A^{\mathrm{mix}}_a,\ a=1$",   mixture(y, 1.0), h_mixture(w, 1.0), "-"),
        (r"$A^{\mathrm{mix}}_a,\ a=2$",   mixture(y, 2.0), h_mixture(w, 2.0), "-"),
    ]

    # Two-panel version (arXiv / single-column): adjusters + induced densities.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
    _plot_adjusters(axL, adjusters, y, colors)
    _plot_densities(axR, adjusters, w, colors)
    fig.tight_layout()
    save_figure(fig, "adjuster_growth.pdf")

    # Single-panel version (AAAI, space-limited): only the adjusters A(y).
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    _plot_adjusters(ax, adjusters, y, colors)
    fig.tight_layout()
    save_figure(fig, "adjuster_growth_single.pdf")

    # Stacked single-column version (AAAI): adjusters on top, densities below,
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
        _plot_densities(axB, stacked, w, colors)
        fig.tight_layout()
        save_figure(fig, "adjuster_growth_stacked.pdf")


if __name__ == "__main__":
    use_paper_style(**{"legend.fontsize": 16})
    main()
