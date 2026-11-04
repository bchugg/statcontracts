"""
Discounted milestone receipts: fixed, plug-in, and Bellman policies.

Bernoulli betting (p = 0.7, m = 0.5) with milestones 2, 4, 8 and two
allocations of the same null-payout budget: the low-milestone adjuster puts
80% of it on the first milestone, the high-milestone adjuster 80% on the
last. Fixed policies use first-passage recursions; an adaptive policy comes
from a discretized Bellman equation and is evaluated by Monte Carlo on
continuous evidence states (see statcontracts.sequential.milestones). The
JSON record keeps every value relative to the theoretical upper bound
U_inc = sum_j r_j a_j^{-rho*}; the table and the figure
(plot_incremental_milestones.py) report fractions of the Bellman value.

Writes results/incremental_milestones.json and
results/incremental_milestones_table.tex. Pass --plugin-only to reuse the
recorded fixed/Bellman results and recompute only the plug-in policies.
"""

import argparse
import json
import math
import os
import time

import numpy as np

from statcontracts.sequential import BettingReward
from statcontracts.sequential.milestones import (
    crossing_values,
    fixed_candidates,
    plugin_receipts,
    simulate_milestone_policy,
    simulate_plugin_crossings,
    solve_milestone_bellman,
)

from common import RESULTDIR, save_text

# Table labels for the weight allocations in experiment().
ADJUSTERS = {"early": "Low-milestone", "late": "High-milestone"}


def experiment():
    p, m = 0.7, 0.5
    reward = BettingReward(p, m)
    levels = (2.0, 4.0, 8.0)
    weights = {"early": (0.8, 0.1, 0.1), "late": (0.1, 0.1, 0.8)}
    growth_bet = reward.growth_optimal_lambda()
    rows = []
    for delta in (0.9, 0.95, 0.98, 0.995):
        started = time.time()
        rho, cgf_bet = reward.interior_cgf_optimal(delta)
        factor = math.exp(-rho * math.log1p(cgf_bet * (1 - m)))
        row = {"delta": delta, "rho": rho, "lambda_cgf": cgf_bet,
               "overshoot_factor": factor, "allocations": {}}
        candidates = fixed_candidates(reward, levels, extra=(cgf_bet, growth_bet))
        hits = {}
        print(f"delta={delta}: searching {len(candidates)} fixed bets", flush=True)
        for bet in candidates:
            hits[bet] = crossing_values(reward, bet, delta, levels, 1e-8)[0]
        # Refine around several promising points for each allocation.
        for budget in weights.values():
            payouts = np.asarray(levels) * budget
            top = sorted(hits, key=lambda b: float(hits[b] @ payouts),
                         reverse=True)[:3]
            for bet in top:
                for refined in np.linspace(max(.001, bet - .015),
                                           min(1 / m - 1e-6, bet + .015), 31):
                    if refined not in hits:
                        hits[refined] = crossing_values(reward, refined, delta,
                                                        levels, 1e-8)[0]
        for name, budget in weights.items():
            payouts = np.asarray(levels) * budget
            upper = float(np.sum(payouts * np.asarray(levels) ** (-rho)))
            best_bet = max(hits, key=lambda b: float(hits[b] @ payouts))
            result = {"payouts": payouts.tolist(), "upper": upper,
                      "lambda_fixed": float(best_bet), "fixed_candidates": len(hits)}
            for policy, bet in (("gro", growth_bet), ("cgf", cgf_bet),
                                ("fixed", best_bet)):
                h, error = crossing_values(reward, bet, delta, levels)
                value = float(payouts @ h)
                result[policy] = {"value": value, "value_over_upper": value / upper,
                                  "truncation_error_bound": float(payouts @ error)}
                assert value <= upper + 1e-10
                if policy == "cgf":
                    assert value + float(payouts @ error) + 1e-10 >= factor * upper
            # Sensitivity of the Bellman benchmark to the state/action grids,
            # the lower state boundary, and the largest allowed bet.
            audits = []
            finest = None
            for n_grid, n_actions, lower, cap in ((1200, 101, -12, 1.99),
                                                  (2400, 201, -12, 1.99),
                                                  (4800, 401, -12, 1.99),
                                                  (4800, 201, -20, 1.99),
                                                  (2400, 201, -12, 1.999)):
                control = solve_milestone_bellman(
                    reward, delta, levels, payouts, n_grid, n_actions, lower, cap,
                    extra_actions=(growth_bet, cgf_bet, best_bet))
                audit = {"n_grid": n_grid, "n_actions": n_actions, "lower": lower,
                         "lambda_cap": cap, "grid_value": control["value"],
                         "residual": control["residual"]}
                if n_grid == 4800 and n_actions == 401:
                    finest = control
                    audit["simulation"] = simulate_milestone_policy(control,
                                                                    n_sim=500000)
                audits.append(audit)
                print(f"  {name}: {audit}", flush=True)
            # Pinning a fixed action checks the state discretization against
            # the exact first-passage calculation.
            pinned = solve_milestone_bellman(reward, delta, levels, payouts, 4800, 1,
                                             fixed=cgf_bet)
            result["adaptive"] = dict(audits[2]["simulation"])
            result["adaptive"]["value_over_upper"] = result["adaptive"]["value"] / upper
            result["adaptive"]["grid_value"] = finest["value"]
            result["adaptive_audit"] = audits
            result["pinned_cgf_grid_value"] = pinned["value"]
            simulation = result["adaptive"]
            assert (abs(simulation["value"] - finest["value"])
                    < 4 * simulation["se"] + .001 * upper)
            assert simulation["value"] < upper + 4 * simulation["se"]
            assert abs(pinned["value"] - result["cgf"]["value"]) < .004 * upper
            row["allocations"][name] = result
        rows.append(row)
        print(f"Completed delta={delta} in {time.time() - started:.1f}s", flush=True)
    return {"p": p, "m": m, "C": 1, "levels": levels,
            "budget_weights": weights, "lambda_gro": growth_bet,
            "crossing_tolerance": 1e-12, "rows": rows}


def add_plugin_results(result, n_sim=500000, seed=20261021, tail_tolerance=1e-8):
    """Reuse each plug-in's hitting times across discounts and payout allocations."""
    p, m, levels = result["p"], result["m"], result["levels"]
    reward = BettingReward(p, m)
    largest_delta = max(row["delta"] for row in result["rows"])
    largest_payout = max(sum(a["payouts"]) for row in result["rows"]
                         for a in row["allocations"].values())
    horizon = max(1, math.ceil(math.log(tail_tolerance / largest_payout)
                               / math.log(largest_delta)))
    audit = {"n_sim": n_sim, "seed": seed, "horizon": horizon,
             "initial_bet": .1, "mean_clip_upper": .99,
             "lambda_cap": (.99 - m) / (m * (1 - m)),
             "tail_tolerance": tail_tolerance, "policies": {}}
    crossings = {}
    for name, prior in (("plugin", 0.0), ("plugin_smoothed", .5)):
        crossings[name] = simulate_plugin_crossings(
            reward, levels, horizon, n_sim, seed, pseudocount=prior)
        audit["policies"][name] = {
            "symmetric_pseudocount": prior,
            "all_paid_fraction": float(np.mean(crossings[name][:, -1] > 0))}
        print(f"{name}: simulated {n_sim} paths through cap {horizon}", flush=True)
    for row in result["rows"]:
        delta = row["delta"]
        rho, _ = reward.interior_cgf_optimal(delta)
        assert math.isclose(row["rho"], rho, rel_tol=1e-12)
        for name, allocation in row["allocations"].items():
            payouts, upper = allocation["payouts"], allocation["upper"]
            assert math.isclose(upper, float(np.asarray(payouts)
                                             @ np.asarray(levels) ** (-rho)),
                                rel_tol=1e-12)
            totals = {}
            for policy, hits in crossings.items():
                values = plugin_receipts(hits, payouts, delta)
                totals[policy] = values
                value = float(values.mean())
                se = float(values.std(ddof=1) / math.sqrt(n_sim))
                tail = float(delta ** (horizon + 1) * sum(payouts))
                assert tail <= tail_tolerance
                assert value <= upper + 5 * se + tail
                allocation[policy] = {
                    "value": value, "se": se, "value_over_upper": value / upper,
                    "se_over_upper": se / upper, "tail_bound": tail,
                    "tail_bound_over_upper": tail / upper,
                    "n_sim": n_sim, "cap": horizon}
            difference = totals["plugin_smoothed"] - totals["plugin"]
            allocation["plugin_sensitivity"] = {
                "smoothed_minus_raw": float(difference.mean()),
                "paired_se": float(difference.std(ddof=1) / math.sqrt(n_sim)),
                "difference_over_upper": float(difference.mean() / upper),
                "paired_se_over_upper":
                    float(difference.std(ddof=1) / math.sqrt(n_sim) / upper)}
            print(f"delta={delta:g} {name}: plug-in "
                  f"{allocation['plugin']['value_over_upper']:.4f} "
                  f"(SE {allocation['plugin']['se_over_upper']:.4f}), smoothed "
                  f"{allocation['plugin_smoothed']['value_over_upper']:.4f}",
                  flush=True)
    result["plugin_audit"] = audit


def write_results(result):
    save_text("incremental_milestones.json", json.dumps(result, indent=2) + "\n")
    lines = ["% Generated by experiments/sequential/incremental_milestones.py.",
             r"\begin{tabular}{clcccccc}", r"\hline",
             r" & & & & \multicolumn{4}{c}{Fraction of Bellman value} \\",
             r"\cline{5-8}",
             r"$\delta$ & Adjuster & Upper bound & Bellman & Growth-optimal & "
             r"CGF-optimal & Best fixed & Plug-in \\",
             r"\hline"]
    for row in result["rows"]:
        for i, (name, allocation) in enumerate(row["allocations"].items()):
            bellman = allocation["adaptive"]["value"]
            fields = [f"{row['delta']:g}" if i == 0 else "", ADJUSTERS[name],
                      f"{allocation['upper']:.3f}", f"{bellman:.3f}"]
            fields.extend(f"{allocation[policy]['value'] / bellman:.3f}"
                          for policy in ("gro", "cgf", "fixed", "plugin"))
            lines.append(" & ".join(fields) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    save_text("incremental_milestones_table.tex", "\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-only", action="store_true",
                        help="Reuse the recorded fixed/Bellman results and "
                             "update plug-ins.")
    parser.add_argument("--n-sim", type=int, default=500000,
                        help="Number of paths for each incremental plug-in.")
    args = parser.parse_args()
    if args.plugin_only:
        with open(os.path.join(RESULTDIR, "incremental_milestones.json")) as f:
            result = json.load(f)
    else:
        result = experiment()
    add_plugin_results(result, n_sim=args.n_sim)
    write_results(result)
    print("Plug-in crossing and upper-bound checks passed.")


if __name__ == "__main__":
    main()
