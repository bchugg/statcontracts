# Sequential experiments

Code to reproduce the figures in *Principal-agent hypothesis testing under
adaptive experimentation*. Each script writes PDFs to `figures/`. Run them
from this directory, after installing the package (`pip install -e ".[dev]"`
from the repository root):

```bash
python adjusters_growth.py     # adjuster figure
python agent_stopping.py       # agent efficiency figures (Figure 2)
python decomposition.py        # loss decomposition + Bellman audit
python incremental_stopping.py # incremental-schedule figures (not in the paper)
```

`common.py` holds the shared figure style, output paths, and result cache.

### `adjusters_growth.py`
Plots several admissible adjusters and the densities they induce. Produces:
- `adjuster_growth.pdf`: two panels (adjusters and induced densities).
- `adjuster_growth_stacked.pdf`: single column, with the adjusters above
  the densities.
- `adjuster_growth_single.pdf`: adjusters only (single panel).

### `agent_stopping.py`
Compares the agent's stopping strategies against the full-knowledge Bellman
optimum as distributional knowledge shrinks, for the Gaussian and betting
reward families. The optimum is computed by policy iteration on a fine grid
(`statcontracts.sequential.bellman_benchmark`). Produces:
- `efficiency_grid.pdf`: combined 2x2 grid.
- `efficiency_row.pdf`: combined 1x4 row.
- individual panels (`subgauss_eff_vs_*.pdf`, `betting_eff_vs_*.pdf`).

The Monte Carlo sweeps are the slow part (about 2.5 minutes on one CPU core),
so results are cached to `cache/`, keyed by the full configuration and the
package version. Re-running only replots. To force a full recompute:

```bash
python agent_stopping.py --recompute   # or: RECOMPUTE=1 python agent_stopping.py
```

The cache key does not cover the library code itself, so pass `--recompute`
after changing anything in `statcontracts.sequential` without bumping the
version.

### `decomposition.py`
Splits the efficiency loss of each strategy into stopping-rule error,
fixed-parameter selection error, and the value of adapting the parameter. It
also audits the Bellman benchmark: it compares it with the coarser
value-iteration solver of the submitted version and checks for negative
bets. All values come from solving the discretized dynamic programs exactly
(policy iteration with sparse linear solves), with Monte Carlo cross-checks.
Takes about 15 minutes and caches its results in `cache/`. Produces:
- `decomposition_grid.pdf`: 2x2 version.
- `decomposition_row.pdf`: 1x4 version.

It prints a decomposition table and a Bellman audit table for each sweep.

### `incremental_stopping.py`
Compares stopping strategies under the incremental reward schedule (adjuster
reward on a sub-Gaussian e-process). Produces `incr_eff_vs_*.pdf`.

## Notes

- The number of Monte Carlo runs and the parameter grids are set near the top of
  each script's `__main__` block.
- All strategies are simulated with the same seed
  (`statcontracts.sequential.DEFAULT_SEED = 42`), so they share common random
  numbers.
- `cache/` is regenerated automatically and can be deleted safely.
