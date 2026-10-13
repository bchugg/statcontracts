# Statistical Contract Theory

Code implementing various statistical contracts found in the literature. 

The package covers the code in these papers: 

| Subpackage | Setting | Paper |
|---|---|---|
| `statcontracts.singleshot` | One agent, one dataset; the principal offers a cost and a menu of license functions. | *Principal-agent hypothesis testing* (Bates, Jordan, Sklar, Soloff) |
| `statcontracts.multiagent` | Many agents compete for a budget; the principal allocates using all of their e-values at once. | *Strategic Multi-Agent Resource Allocation via Statistical Contracts* (Chugg, Jordan, Ramdas) |
| `statcontracts.sequential` | One agent experiments adaptively, choosing a betting parameter each round and when to stop. | *Principal-agent hypothesis testing under adaptive experimentation* (Chugg, Ghosh, Ramdas); `multiround.py` implements the multi-round contract of Bates et al. |

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
  evalues.py          e-values: Gaussian and Bernoulli likelihood ratios, betting, Hoeffding
  singleshot/
    contracts.py      incentive alignment, the agent's best response, the optimal
                      (Neyman-Pearson) license, the principal's expected utility
    models.py         evidence models with exact expectations: GaussianMean, Binomial
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
    multiround.py     the finite-horizon multi-round contract with per-round costs, and
                      the agent's best response by dynamic programming
    adjusters.py      admissible adjusters and their induced densities
experiments/
  multiagent/         reproduces the figures of the multi-agent paper
  sequential/         reproduces the figures of the sequential paper
tests/
```

## Usage

Single-shot: check whether a protocol is incentive-aligned, and find the
agent's best response to the menu of all e-values.

```python
from scipy.stats import norm
from statcontracts.singleshot import (
    GaussianMean, ThresholdLicense, is_incentive_aligned, optimal_license)

model = GaussianMean(n=1)             # evidence Z ~ N(theta, 1); null theta <= 0
C, R = 1.0, 50.0                      # trial cost, market size

status_quo = ThresholdLicense(norm.isf(0.05), payout=R)    # approve if p < 0.05
is_incentive_aligned([status_quo], C, model, null_types=[0.0])    # False: 0.05 R > C
f = optimal_license(model, theta1=1.0, theta0=0.0, cost=C, cap=R) # approve if p < C/R
```

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
pytest                # everything (~25 s)
pytest -m "not slow"  # skip the slower simulation checks (~6 s)
```

`tests/test_regression.py` pins outputs of small-scale versions of the
sequential experiments. `tests/test_multiagent.py` checks the multi-agent solvers
against brute-force enumeration.
