# Sequential experiments

Code to reproduce the figures, tables, and numerical checks in *Principal-agent
hypothesis testing under adaptive experimentation*. Run the scripts from this
directory, after installing the package (`pip install -e ".[dev]"` from the
repository root). Figures are written to `figures/`, and JSON records and LaTeX
tables to `results/`.

Both reward schedules have `R_0=0`, with evidence initialized at `E_0=1`.
Deferred collection begins after the first trial. All reported experiments use
`C=1`.

```bash
python adjusters_growth.py              # adjuster figure
python incremental_milestones.py        # incremental receipts: fixed, plug-in, Bellman
python plot_incremental_milestones.py   # incremental figure from the recorded results
python agent_stopping.py                # deferred efficiency figure
python deferred_comparison.py           # large-jump example + analytical-upper-bound comparison
python gaussian_overshoot.py            # tilted Gaussian ratios + simultaneous 99% lower bounds
python certify_incremental.py --adaptive  # all-fixed upper bounds + adaptive validation
python decomposition.py                 # loss decomposition + Bellman audit
python check_exact_threshold.py         # exact threshold, all signs of the drift
python check_cgf_guarantee.py           # CGF theorem with finite stopping caps
python incremental_stopping.py          # historical; not in the paper
```

`common.py` holds the shared figure style, output paths, and result cache.

## Figures in the paper

### `adjusters_growth.py`
Plots several admissible adjusters and their cumulative milestone-budget
allocations, including the atom at zero for unconditional first-round
payments (`statcontracts.sequential.adjusters`). Produces:
- `adjuster_growth.pdf`: two panels (adjusters and cumulative budgets).
- `adjuster_growth_stacked.pdf`: single column, with the adjusters above
  the budgets.
- `adjuster_growth_single.pdf`: adjusters only (single panel).

### `incremental_milestones.py`
Computes discounted first-passage values for three milestone payments under
Bernoulli betting (`p=.7`, `m=.5`, milestones 2, 4, 8). Compares the
growth-optimal and CGF-optimal fixed parameters, the best fixed parameter found
by a search, a numerical adaptive (Bellman) policy, and the clipped sample-mean
plug-in. Two allocations spend the same null-payout budget differently across
the milestones. The JSON records absolute values and ratios to the theoretical
upper bound; the figure reports ratios to the simulated Bellman policy's value.
Writes `results/incremental_milestones.json` and
`results/incremental_milestones_table.tex`.

The routines live in `statcontracts.sequential.milestones`. The first-passage
recursion absorbs each path at its first crossing and stops when the remaining
discounted mass bounds the omitted crossing value by less than 1e-12. The
fixed-parameter search adds bets that align short paths with milestones, where
crossing values can jump, and refines around promising candidates.

The Bellman benchmark (`solve_milestone_bellman`) uses policy iteration with
current log-evidence and the next unpaid milestone as its state, keeping
crossing payments separate from interpolation below each milestone. The
experiment checks state/action grid refinement, the lower state boundary, the
largest allowed bet, and agreement with the fixed-parameter recursion. It then
evaluates the computed policy on 500,000 simulated paths using actual,
unrounded evidence. Reported adaptive values are Monte Carlo estimates of a
feasible policy, not certified optimal values. The JSON records standard
errors, simulation tail bounds, and every numerical sensitivity check.

The plug-in starts at bet 0.1, clips the sample mean at 0.99 (bet cap 1.96 at
`m=.5`), and keeps sampling at zero bets. Its 500,000 paths supply hitting
times reused across all discounts and payment allocations. Standard errors use
each path's total receipts, preserving covariance among milestones; the
numerical tail is bounded by 1e-8. A smoothed-mean check uses the same
observations and reports paired differences in the JSON.

The full run takes about ten minutes on one CPU core. `--plugin-only`
recomputes just the plug-in policies from the recorded fixed/Bellman results,
and `--n-sim` sets the plug-in path count (default 500000).

### `plot_incremental_milestones.py`
Reads `results/incremental_milestones.json` and plots the fixed and plug-in
policies relative to the numerical Bellman benchmark for both milestone
allocations. Produces `incremental_milestones.pdf`.

### `agent_stopping.py`
Compares the agent's stopping strategies under deferred rewards against a
numerical Bellman benchmark (policy iteration on a fine grid;
`statcontracts.sequential.bellman_benchmark`), for the Gaussian and betting
reward families. The plug-in rule estimates the drift of the next bet under the
fitted mean (Gaussian) or Bernoulli law, with consistent clipping in the bet
and the threshold. Every path runs at least one trial and is capped at 2500
rounds, with its discounted log reward paid at the cap. Produces:
- `efficiency_grid.pdf`: combined 2x2 grid.
- `efficiency_row.pdf`: combined 1x4 row.
- individual panels (`subgauss_eff_vs_*.pdf`, `betting_eff_vs_*.pdf`).

The Monte Carlo sweeps are the slow part (a few minutes on one CPU core), so
results are cached to `cache/`, keyed by the full configuration and the
package version. Re-running only replots. To force a full recompute:

```bash
python agent_stopping.py --recompute   # or: RECOMPUTE=1 python agent_stopping.py
```

The cache key does not cover the library code itself, so pass `--recompute`
after changing anything in `statcontracts.sequential` without bumping the
version.

## Numbers and tables

### `deferred_comparison.py`
Computes capped deferred Bernoulli threshold values by first-passage recursion
(`statcontracts.sequential.capped_threshold_value`), including signed log
payoffs on paths that reach the cap. Reproduces the large-jump example
(`p=.02`, `m=.01`, `delta=.9`), where the CGF rule loses at least 29.4%
relative to the optimum, and compares three feasible strategies directly with
the analytical adaptive upper bound at `p=.7`, `m=.5` and four discounts.
These calculations use neither state interpolation nor Monte Carlo sampling.
Writes `results/deferred_comparison.json`, `results/large_jump_table.tex`, and
`results/deferred_upper_table.tex`.

### `gaussian_overshoot.py`
Computes the exact deferred ratio as an expectation under the tilted Gaussian
law, at the discount factors `.9`, `.95`, `.98`, and `.995`, with `theta=.3`
and `sigma=.5`. Uses 400,000 paths per setting and seed 20261017. Every path
runs until it crosses the threshold; there is no time cap or state grid. The
sampled function `(1+rho*O)*exp(-rho*O)` lies in `[0,1]`, so one-sided
Hoeffding bounds with error probability `.01/4` per row give simultaneous 99%
lower bounds, rounded down to three decimals. Writes
`results/gaussian_overshoot.json` and `results/gaussian_overshoot_table.tex`.

### `certify_incremental.py`
Bounds **every** fixed bet in `[0,2)` for the impatient incremental example
(`p=.7`, `m=.5`, `delta=.9`). It partitions `b=lambda/2` into 256 intervals
and pays a milestone optimistically whenever any bet in the interval could
cross it. Rational evidence comparisons and integer first-passage recursion
through round 50 give an upper bound for each interval. Adding the uniform
remaining-payment bound gives infinite-horizon upper bounds of 1.238940
(low-milestone allocation) and 1.950746 (high-milestone), rounded upward.

With `--adaptive`, the script fixes two numerical policies and evaluates each
on 500,000 fresh paths through round 200. Downward-rounded evidence and
payments make the recorded receipts conservative. Hoeffding lower bounds, with
a union bound for simultaneous 99% coverage, exceed the all-fixed upper bounds.
This certifies the fixed-policy upper bounds deterministically and establishes
the adaptive separation with the stated statistical confidence; it does not
certify adaptive optimality. Writes `results/certify_incremental.json` and,
with `--adaptive`, `results/incremental_certificate_table.tex`.

### `decomposition.py`
Splits the efficiency loss of each deferred strategy into stopping-rule error,
fixed-parameter selection error, and the value of adapting the parameter, and
audits the Bellman benchmark (comparing it with the coarser value-iteration
solver of the submitted version, and checking for negative bets). All values
come from solving the discretized dynamic programs exactly (policy iteration
with sparse linear solves), with Monte Carlo cross-checks. The fixed-parameter
search covers the full Bellman parameter range, including parameters with zero
or negative drift, and includes the growth-optimal and CGF-optimal parameters
as candidates; it reports the best parameter found, without certifying global
optimality. Takes about 15 minutes and caches its results in `cache/`.
Produces `decomposition_grid.pdf` and `decomposition_row.pdf`, and prints a
decomposition table and a Bellman audit table for each sweep.

## Numerical checks

### `check_exact_threshold.py`
Checks the exact threshold with positive, zero, and negative drift. For
Bernoulli rewards with log-increments +1 and -1, it compares the threshold
series with the known geometric-horizon maximum and compares capped threshold
payoffs with finite-horizon optimal stopping. Uses neither Monte Carlo nor
state interpolation.

### `check_cgf_guarantee.py`
Deterministic numerical checks of the CGF theorem with finite stopping caps.
Propagates first-passage probabilities under the original and tilted laws,
checks positive, zero, and negative drift, and compares with finite-horizon
adaptive Bellman values on integer states. It also reports cap probabilities
and the theorem's error bounds for three betting settings from the paper.
These checks complement the mathematical proof.

## Historical

### `incremental_stopping.py`
Studies the terminal discounted log utility of an adjuster reward on a
sub-Gaussian e-process (`statcontracts.sequential.incremental`). This is a
different objective from the discounted incremental receipts in the current
manuscript, and its `incr_eff_vs_*.pdf` figures are not used there.

## Notes

- The number of Monte Carlo runs and the parameter grids are set near the top of
  each script's `__main__` block (or `experiment()` function).
- Simulated strategies share a seed (`statcontracts.sequential.DEFAULT_SEED = 42`
  for the deferred strategies), so they use common random numbers.
- `cache/` is regenerated automatically and can be deleted safely.
