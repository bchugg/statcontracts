# Statistical Contract Theory

Code implementing various statistical contracts found in the literature. 

The package covers the code in these papers: 

| Subpackage | Setting | Paper |
|---|---|---|
| `statcontracts.multiagent` | Many agents compete for a budget; the principal allocates using all of their e-values at once. | *Strategic Multi-Agent Resource Allocation via Statistical Contracts* (Chugg, Jordan, Ramdas) |
| `statcontracts.sequential` | One agent experiments adaptively, choosing a betting parameter each round and when to stop. | *Principal-agent hypothesis testing under adaptive experimentation* (Chugg, Ghosh, Ramdas) |

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
  multiagent/         reproduces the figures of the multi-agent paper
  sequential/         reproduces the figures of the sequential paper
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

See [`experiments/multiagent/README.md`](experiments/multiagent/README.md) and
[`experiments/sequential/README.md`](experiments/sequential/README.md).

## Tests

```bash
pytest                # everything (~10 s)
pytest -m "not slow"  # skip the slower regression runs
```

`tests/test_regression.py` pins outputs of small-scale versions of the
sequential experiments. `tests/test_multiagent.py` checks the multi-agent solvers
against brute-force enumeration.
