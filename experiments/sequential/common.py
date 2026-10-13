"""Shared figure style, output paths, and result caching for the sequential experiments."""

import hashlib
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statcontracts

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
CACHEDIR = os.path.join(HERE, "cache")

# Set RECOMPUTE=1 (env) or pass --recompute to force the sweeps to rerun;
# otherwise cached results are loaded so plotting can be re-iterated
# instantly without redoing the (slow) computations.
RECOMPUTE = "--recompute" in sys.argv or os.environ.get("RECOMPUTE") == "1"

PAPER_STYLE = {
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 19,
    "axes.labelsize": 23,
    "axes.titlesize": 23,
    "legend.fontsize": 13,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "lines.linewidth": 2.5,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
}


def use_paper_style(**overrides):
    plt.rcParams.update({**PAPER_STYLE, **overrides})


def save_figure(fig, filename):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


def cached(tag, key, compute):
    """Return compute(), cached as JSON under a hash of `key` and the package version.

    Any change to the configuration in `key` (or to the package version)
    yields a new cache file; identical configurations load from disk.
    """
    key = {**key, "tag": tag, "statcontracts": statcontracts.__version__}
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    path = os.path.join(CACHEDIR, f"{tag}_{h}.json")
    if os.path.exists(path) and not RECOMPUTE:
        print(f"  [cache hit] {os.path.basename(path)}", flush=True)
        with open(path) as f:
            return json.load(f)["results"]
    results = compute()
    os.makedirs(CACHEDIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"key": key, "results": results}, f)
    print(f"  [cache saved] {os.path.basename(path)}", flush=True)
    return results
