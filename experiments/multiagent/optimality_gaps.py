"""Experiments for the two optimality-gap figures in the multi-agent paper.

Compare optimization over the full polytope P, optimization over the compact
inner polytope Q, and the cumulative-evidence, AoN, and e-BH policies.
Figures are written to ``figures/`` and trial-level results, summaries, and
parameters to ``results/`` (both next to this script).

Run both paper comparisons:

    python experiments/multiagent/optimality_gaps.py

Use ``--weight-sweep`` to run only the comparison across weight dispersions
on common instances, or ``--gap-scaling`` to run only the comparison across
agent counts with a fixed budget. Use ``--quick`` for a small smoke test.
Solver checks against enumeration live in ``tests/test_multiagent.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from statcontracts.multiagent import (
    evaluate_log_utility,
    mechanism_aon,
    mechanism_cumulative,
    mechanism_ebh,
    null_allocation_proportion,
    rescale_m1,
    solve_linear_compact,
    solve_linear_full,
    solve_log,
)


HERE = Path(__file__).resolve().parent
GAP_SCALING_PATH = HERE / "figures" / "optimality_gap_scaling.png"
WEIGHT_SWEEP_PATH = HERE / "figures" / "optimality_gap_weight_sweep.png"
RESULTS_DIR = HERE / "results"

METHODS = ("Compact optimization", "Cumulative", "AoN", "e-BH")
COLORS = {
    "Compact optimization": "#4C78A8",
    "Cumulative": "#F58518",
    "AoN": "#54A24B",
    "e-BH": "#E45756",
}
LABELS = {
    "Compact optimization": r"Optimizing over $\mathsf{Q}$",
}


def sample_instance(
    rng: np.random.Generator,
    m: int,
    null_fraction: float,
    sample_size: int,
    sigma2: float,
    cost_sigma: float,
    value_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate heterogeneous costs, observable values, and normal LR e-values."""
    is_null = rng.random(m) < null_fraction
    means = np.where(is_null, 0.0, 1.0)
    data = rng.normal(means[:, None], np.sqrt(sigma2), size=(m, sample_size))
    log_e = (data.sum(axis=1) - sample_size / 2.0) / sigma2
    E = np.exp(np.clip(log_e, -30.0, 30.0))
    costs = np.exp(rng.normal(-0.5 * cost_sigma**2, cost_sigma, size=m))
    values = np.exp(rng.normal(-0.5 * value_sigma**2, value_sigma, size=m))
    return E, costs, values, is_null


def policy_allocations(E, costs, Delta, B):
    return {
        "Cumulative": rescale_m1(mechanism_cumulative(E, Delta, costs), B),
        "AoN": rescale_m1(mechanism_aon(E, Delta, costs), B),
        "e-BH": rescale_m1(mechanism_ebh(E, Delta, costs), B),
    }


def relative_gap(optimum: float, achieved: float) -> float:
    if optimum <= 1e-10:
        return 0.0
    return max(0.0, (optimum - achieved) / optimum)


def run_gap_experiment(args, rng):
    rows = []
    completed = 0
    attempts = 0
    while completed < args.trials:
        attempts += 1
        if attempts > 20 * args.trials:
            raise RuntimeError("Too few nonzero safe instances; adjust the simulation parameters")
        E, costs, values, is_null = sample_instance(
            rng,
            args.m,
            args.null_fraction,
            args.sample_size,
            args.sigma2,
            args.cost_sigma,
            args.value_sigma,
        )
        F = costs * E
        if F.sum() < costs.sum() / args.Delta:
            continue

        full_linear, cuts_linear, _ = solve_linear_full(
            F, costs, args.Delta, args.B, values
        )
        compact_linear, _ = solve_linear_compact(
            F, costs, args.Delta, args.B, values
        )
        full_concave, cuts_concave, _ = solve_log(
            "full", F, costs, args.Delta, args.B, values
        )
        compact_concave, _, _ = solve_log(
            "compact", F, costs, args.Delta, args.B, values
        )
        allocations = policy_allocations(E, costs, args.Delta, args.B)
        allocations["Compact optimization"] = compact_linear

        optimum_linear = float(values @ full_linear)
        optimum_concave = evaluate_log_utility(full_concave, values)
        for method in METHODS:
            linear_allocation = allocations[method]
            if method == "Compact optimization":
                concave_allocation = compact_concave
            else:
                concave_allocation = linear_allocation
            metrics = (
                (
                    "Linear",
                    linear_allocation,
                    float(values @ linear_allocation),
                    full_linear,
                    optimum_linear,
                ),
                (
                    "Concave",
                    concave_allocation,
                    evaluate_log_utility(concave_allocation, values),
                    full_concave,
                    optimum_concave,
                ),
            )
            for objective, allocation, achieved, full_allocation, optimum in metrics:
                rows.append(
                    {
                        "trial": completed,
                        "objective": objective,
                        "method": method,
                        "optimum": optimum,
                        "achieved": achieved,
                        "relative_gap": relative_gap(optimum, achieved),
                        "support": int(np.count_nonzero(allocation)),
                        "nap": null_allocation_proportion(allocation, is_null),
                        "full_nap": null_allocation_proportion(full_allocation, is_null),
                        "null_fraction": float(is_null.mean()),
                        "cost_cv": float(costs.std() / costs.mean()),
                        "linear_cuts": cuts_linear,
                        "concave_cuts": cuts_concave,
                    }
                )
        completed += 1
    return rows


def run_gap_scaling_experiment(args) -> None:
    """Compare quality across sizes, using an independent stream at each size."""
    rows = []
    started = time.perf_counter()
    results_path = RESULTS_DIR / "optimality_gap_scaling.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "parameters": vars(args),
        "rng": "numpy.default_rng(numpy.random.SeedSequence([seed, m]))",
        "conditioning": "sum(costs * E) >= sum(costs) / Delta",
        "budget": "fixed across agent counts",
        "concave_objective": "sum_i v_i * log(1 + A_i)",
        "concave_solver": "scipy.optimize.minimize(method=\"SLSQP\")",
        "plotted_summary": "median and interquartile range of relative gaps",
    }
    (RESULTS_DIR / "optimality_gap_scaling.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    for m in args.gap_sizes:
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, m]))
        completed = 0
        while completed < args.trials:
            batch_args = argparse.Namespace(**vars(args))
            batch_args.m = m
            batch_args.trials = min(10, args.trials - completed)
            batch = run_gap_experiment(batch_args, rng)
            for row in batch:
                row["trial"] += completed
                rows.append({"m": m, **row})
            completed += batch_args.trials
            write_csv(results_path, rows)
            elapsed = time.perf_counter() - started
            print(f"m={m}: {completed}/{args.trials} quality trials ({elapsed:.1f}s)", flush=True)

    summary = []
    for m in args.gap_sizes:
        for objective in ("Linear", "Concave"):
            for method in METHODS:
                values = np.array([
                    row["relative_gap"] for row in rows
                    if row["m"] == m and row["objective"] == objective
                    and row["method"] == method
                ])
                summary.append({
                    "m": m,
                    "objective": objective,
                    "method": method,
                    "trials": len(values),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q90": float(np.quantile(values, 0.9)),
                })
    write_csv(RESULTS_DIR / "optimality_gap_scaling_summary.csv", summary)
    plot_gap_scaling(rows, GAP_SCALING_PATH)
    print(f"Wrote {results_path}")
    print(f"Wrote {GAP_SCALING_PATH}")


def run_weight_sweep_experiment(args) -> None:
    """Compare quality across weight dispersions, reusing the same instances."""
    rows = []
    started = time.perf_counter()
    results_path = RESULTS_DIR / "optimality_gap_weight_sweep.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "parameters": vars(args),
        "rng": "numpy.default_rng(seed), reset for each value_sigma",
        "pairing": "Same evidence, costs, activation events and standard normal "
                   "weight draws Z_i at every value_sigma, with "
                   "v_i = exp(value_sigma * Z_i - value_sigma**2 / 2)",
        "conditioning": "sum(costs * E) >= sum(costs) / Delta",
        "concave_objective": "sum_i v_i * log(1 + A_i)",
        "concave_solver": "scipy.optimize.minimize(method=\"SLSQP\")",
        "plotted_summary": "median and interquartile range of relative gaps",
    }
    (RESULTS_DIR / "optimality_gap_weight_sweep.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    for value_sigma in args.value_sigmas:
        rng = np.random.default_rng(args.seed)
        completed = 0
        while completed < args.trials:
            batch_args = argparse.Namespace(**vars(args))
            batch_args.value_sigma = value_sigma
            batch_args.trials = min(10, args.trials - completed)
            batch = run_gap_experiment(batch_args, rng)
            for row in batch:
                row["trial"] += completed
                rows.append({"value_sigma": value_sigma, **row})
            completed += batch_args.trials
            write_csv(results_path, rows)
            elapsed = time.perf_counter() - started
            print(f"value_sigma={value_sigma:g}: {completed}/{args.trials} quality trials "
                  f"({elapsed:.1f}s)", flush=True)

    summary = []
    for value_sigma in args.value_sigmas:
        for objective in ("Linear", "Concave"):
            for method in METHODS:
                subset = [
                    row for row in rows
                    if row["value_sigma"] == value_sigma and row["objective"] == objective
                    and row["method"] == method
                ]
                gaps = np.array([row["relative_gap"] for row in subset])
                summary.append({
                    "value_sigma": value_sigma,
                    "objective": objective,
                    "method": method,
                    "trials": len(gaps),
                    "median": float(np.median(gaps)),
                    "q25": float(np.quantile(gaps, 0.25)),
                    "q75": float(np.quantile(gaps, 0.75)),
                    "q90": float(np.quantile(gaps, 0.9)),
                    "mean_nap": float(np.mean([row["nap"] for row in subset])),
                    "mean_full_nap": float(np.mean([row["full_nap"] for row in subset])),
                })
    write_csv(RESULTS_DIR / "optimality_gap_weight_sweep_summary.csv", summary)
    plot_weight_sweep(rows, WEIGHT_SWEEP_PATH)
    print(f"Wrote {results_path}")
    print(f"Wrote {WEIGHT_SWEEP_PATH}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 10.5,
            "figure.dpi": 140,
        }
    )


def save_figure(figure, legend_axis, output_path: Path) -> None:
    handles, legend_labels = legend_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(legend_labels),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 1), w_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_gap_curves(gap_rows, x_key: str, xlabel: str, output_path: Path) -> None:
    """Plot median relative gaps against ``x_key``, with interquartile bands."""
    configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharey=True)
    xs = sorted({row[x_key] for row in gap_rows})
    markers = ("o", "s", "^", "D")
    linestyles = ("-", "--", "-.", ":")
    for axis, objective in zip(axes, ("Linear", "Concave")):
        for method, marker, linestyle in zip(METHODS, markers, linestyles):
            medians, lower, upper = [], [], []
            for x in xs:
                values = np.array([
                    row["relative_gap"] for row in gap_rows
                    if row[x_key] == x and row["objective"] == objective
                    and row["method"] == method
                ])
                medians.append(np.median(values))
                lower.append(np.quantile(values, 0.25))
                upper.append(np.quantile(values, 0.75))
            axis.plot(
                xs, medians, marker=marker, linestyle=linestyle,
                markersize=3.5, linewidth=1.25,
                color=COLORS[method], label=LABELS.get(method, method),
            )
            axis.fill_between(xs, lower, upper, color=COLORS[method], alpha=0.10)
        axis.set_xticks(xs)
        axis.set_xlabel(xlabel)
        axis.set_ylim(0, 1)
        axis.set_title("Linear utility" if objective == "Linear" else "Logarithmic utility")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Relative optimality gap")
    save_figure(figure, axes[1], output_path)


def plot_gap_scaling(gap_rows, output_path: Path) -> None:
    plot_gap_curves(gap_rows, "m", "Number of agents", output_path)


def plot_weight_sweep(gap_rows, output_path: Path) -> None:
    plot_gap_curves(gap_rows, "value_sigma", r"Weight dispersion $\sigma_v$", output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--gap-scaling", action="store_true",
                        help="Run only the optimality-gap comparison across agent counts")
    parser.add_argument("--gap-sizes", type=int, nargs="+", default=[10, 20, 40, 80, 100])
    parser.add_argument("--weight-sweep", action="store_true",
                        help="Run only the optimality-gap comparison across weight dispersions")
    parser.add_argument("--value-sigmas", type=float, nargs="+",
                        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--null-fraction", type=float, default=0.5)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--sigma2", type=float, default=4.0)
    parser.add_argument("--cost-sigma", type=float, default=0.55)
    parser.add_argument("--value-sigma", type=float, default=0.8)
    parser.add_argument("--Delta", type=float, default=0.1)
    parser.add_argument("--B", type=float, default=30.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.trials = min(args.trials, 8)
        args.gap_sizes = [8, 16]
        args.value_sigmas = [0.0, 0.8]
        args.m = min(args.m, 12)
    return args


def main():
    args = parse_args()
    if args.weight_sweep or not args.gap_scaling:
        run_weight_sweep_experiment(args)
    if args.gap_scaling or not args.weight_sweep:
        run_gap_scaling_experiment(args)


if __name__ == "__main__":
    main()
