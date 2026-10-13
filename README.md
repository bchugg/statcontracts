# Statistical Contract Theory

Code implementing various statistical contracts found in the literature. 

The package covers the code in these papers: 

| Subpackage | Setting | Paper |
|---|---|---|
| `statcontracts.multiagent` | Many agents compete for a budget; the principal allocates using all of their e-values at once. | *Strategic Multi-Agent Resource Allocation via Statistical Contracts* (Chugg, Jordan, Ramdas) — "SMAA" |
| `statcontracts.sequential` | One agent experiments adaptively, choosing a betting parameter each round and when to stop. | *Principal-agent hypothesis testing under adaptive experimentation* (Chugg, Ghosh, Ramdas) — "SPAT" |

## Install

Requires Python >= 3.10.

```bash
conda create -n statcontracts python=3.12
conda activate statcontracts
pip install -e ".[dev]"      # package + matplotlib + pytest
```

## Layout

```
src/statcontracts/
  multiagent/
    mechanisms.py     AoN, e-BH and cumulative-evidence mechanisms; budget rescaling; NAP loss
    optimize.py       linear and log-utility optimization over P (constraint generation
                      with a min-cut separation oracle) and over the compact region Q
  sequential/
    rewards.py        reward processes: GaussianReward, BettingReward, HoeffdingReward
    stopping.py       exact fixed-parameter threshold (Spitzer), Bellman solvers
                      (policy and value iteration), Bellman benchmark
    simulate.py       Monte Carlo of the agent's stopping strategies
    incremental.py    incremental reward schedule (adjuster on the running maximum)
    adjusters.py      admissible adjusters and their induced densities
experiments/
  smaa/               reproduces the SMAA optimality-gap figures
  spat/               reproduces the SPAT figures
tests/
```

## Usage

Multi-agent: allocate a budget across agents from their e-values.

```python
import numpy as np
from statcontracts.multiagent import mechanism_ebh, rescale_m1, solve_linear_full

E = np.array([8.0, 20.0, 18.0])       # agents' e-values
c = np.ones(3)                        # participation costs
Delta, B = 0.1, 40.0                  # risk level, budget

A = rescale_m1(mechanism_ebh(E, Delta, c), B)                  # -> [0, 20, 18]
A_opt, n_cuts, _ = solve_linear_full(c * E, c, Delta, B, values=np.ones(3))
```

Sequential: compare an agent's stopping strategies against the Bellman optimum.

```python
from statcontracts.sequential import (
    BettingReward, bellman_benchmark, sim_bellman, sim_cgf_aware)

reward = BettingReward(p=0.7, m=0.5)  # Bernoulli(p) data, testing E[X] <= m
delta, C = 0.995, 1.0                 # discount factor, entry cost

G, w_grid, lam_policy, _ = bellman_benchmark(reward, delta)
best = sim_bellman(reward, delta, C, G, w_grid, lam_policy).utilities.mean()
cgf = sim_cgf_aware(reward, delta, C).utilities.mean()
print(f"CGF-aware threshold efficiency: {cgf / best:.3f}")
```

## Reproducing the papers

See [`experiments/smaa/README.md`](experiments/smaa/README.md) and
[`experiments/spat/README.md`](experiments/spat/README.md).

## Tests

```bash
pytest                # everything (~10 s)
pytest -m "not slow"  # skip the regression runs against the original SPAT code
```

`tests/test_regression.py` pins outputs of the original SPAT scripts; the
package reproduces them bit for bit. `tests/test_multiagent.py` contains the
solver checks against brute-force enumeration that were previously
`validate_implementation()` in the SMAA experiments script.

## Provenance

The code was moved here from the paper repositories:

| Source | Commit | Now |
|---|---|---|
| `SMAA/experiments/mechanisms.py` | `f9962e4` | `multiagent/mechanisms.py` (unchanged, plus the NAP loss) |
| `SMAA/experiments/optimization_experiments.py` | `f9962e4` | solvers → `multiagent/optimize.py` (unchanged); harness → `experiments/smaa/optimality_gaps.py`; validation → `tests/test_multiagent.py` |
| `spat/experiments/exact_stopping.py` | `e2dd06f` | `sequential/stopping.py`, `sequential/rewards.py` |
| `spat/experiments/agent_stopping.py` | `e2dd06f` | strategies → `sequential/simulate.py`; solvers → `sequential/stopping.py`; driver → `experiments/spat/agent_stopping.py` |
| `spat/experiments/decomposition.py` | `e2dd06f` | `threshold_value_grid` → `sequential/stopping.py`; driver → `experiments/spat/decomposition.py` |
| `spat/experiments/incremental_stopping.py` | `e2dd06f` | `sequential/incremental.py`; driver → `experiments/spat/incremental_stopping.py` |
| `spat/experiments/adjusters_growth.py` | `e2dd06f` | `sequential/adjusters.py`; plots → `experiments/spat/adjusters_growth.py` |

The per-family functions of the SPAT code (`sim_grow_subgauss`,
`sim_grow_betting`, ..., and the `family`/`prm` dictionaries) are replaced
by reward-process classes with one strategy function each.

These stay in the paper repositories, because they generate TikZ that
depends on each paper's LaTeX macros: `SMAA/paper/figures/make_mech_polytope.py`,
`SMAA/slides/figures/*.py`, and the INFORMS supplement packager
`SMAA/paper/informs/package_code.py`. Each paper repository also keeps its
original `experiments/` folder, frozen as the submitted supplement.
