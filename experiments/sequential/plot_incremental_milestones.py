"""
Plot the incremental milestone results in results/incremental_milestones.json.

Two panels (the low- and high-milestone adjusters) show each policy's discounted
receipts relative to the numerical Bellman policy against the discount factor,
placed on a log scale in the effective horizon 1/(1 - delta). The CGF-optimal
and plug-in policies use the colors and markers of the efficiency figure in
agent_stopping.py. Produces figures/incremental_milestones.pdf.
"""

import json
import math
import os

import matplotlib.pyplot as plt

from common import RESULTDIR, save_figure, use_paper_style

# (JSON key, legend label, color, marker)
POLICIES = [
    ("gro", r"Growth-optimal parameter $\lambda_{\mathrm{gro}}$", "#2166ac", "o"),
    ("cgf", r"CGF-optimal parameter $\lambda_{\mathrm{cgf}}$", "#7570b3", "v"),
    ("fixed", "Best fixed parameter", "#a6761d", "s"),
    ("plugin", "Plug-in", "#e7298a", "^"),
]
PANELS = [("early", "Low-milestone adjuster"), ("late", "High-milestone adjuster")]


def main():
    with open(os.path.join(RESULTDIR, "incremental_milestones.json")) as f:
        result = json.load(f)
    deltas = [row["delta"] for row in result["rows"]]
    xs = [math.log(1 / (1 - d)) for d in deltas]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    handles = {}
    for ax, (allocation, title) in zip(axes, PANELS):
        for key, label, color, marker in POLICIES:
            ratios = [row["allocations"][allocation][key]["value"]
                      / row["allocations"][allocation]["adaptive"]["value"]
                      for row in result["rows"]]
            handles[key], = ax.plot(xs, ratios, marker=marker, color=color,
                                    label=label, markersize=7)
        handles["bellman"] = ax.axhline(1.0, color="gray", lw=1.2, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{d:g}" for d in deltas])
        ax.set_xlabel(r"$\delta$")
        ax.set_title(title)
        ax.set_ylim(0.58, 1.03)
    axes[0].set_ylabel("Relative efficiency")

    order = [key for key, *_ in POLICIES] + ["bellman"]
    labels = [label for _, label, *_ in POLICIES] + ["Bellman benchmark"]
    fig.legend([handles[k] for k in order], labels, frameon=False,
               loc="lower center", ncol=3, fontsize=16,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    save_figure(fig, "incremental_milestones.pdf")


if __name__ == "__main__":
    # Matplotlib's default (outward) ticks, as in the paper's version of this figure.
    use_paper_style(**{"xtick.direction": "out", "ytick.direction": "out",
                       "xtick.top": False, "ytick.right": False})
    main()
