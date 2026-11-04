# Statistical Contract Theory

Code implementing various statistical contracts found in the literature. 

The package covers the code in these papers: 

| Subpackage | Setting | Paper |
|---|---|---|
| `statcontracts.singleshot` | One agent, one dataset; the principal offers a cost and a menu of license functions. | [*Principal-agent hypothesis testing*](https://arxiv.org/abs/2205.06812) (Bates, Jordan, Sklar, Soloff) |
| `statcontracts.multiagent` | Many agents compete for a budget; the principal allocates using all of their e-values at once. | [*Strategic Multi-Agent Resource Allocation via Statistical Contracts*](https://arxiv.org/abs/2610.18336) (Chugg, Jordan, Ramdas) |
| `statcontracts.sequential` | One agent experiments adaptively, choosing a betting parameter each round and when to stop. | [*Principal-agent hypothesis testing under adaptive experimentation*](https://arxiv.org/abs/2610.38872) (Chugg, Ghosh, Ramdas); `multiround.py` implements the multi-round contract of [Bates et al.](https://arxiv.org/abs/2205.06812) (their Section 2.4 and Appendix B) |

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
    stopping.py       exact fixed-parameter threshold (Spitzer, any sign of the drift),
                      exact capped threshold values (first passage), Bellman solvers
                      (policy and value iteration), Bellman benchmark
    simulate.py       Monte Carlo of the agent's stopping strategies (deferred rewards)
    milestones.py     incremental milestone payments: first-passage values of fixed
                      bets, a numerical Bellman policy, and the plug-in policy
    incremental.py    historical incremental objective (terminal log utility of an
                      adjusted running maximum)
    multiround.py     the finite-horizon multi-round contract with per-round costs, and
                      the agent's best response by dynamic programming
    adjusters.py      admissible adjusters, their induced densities, and their
                      cumulative milestone budgets
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

Sequential, deferred rewards: compare an agent's stopping strategies against the
Bellman benchmark.

```python
from statcontracts.sequential import (
    BettingReward, bellman_benchmark, sim_bellman, sim_cgf_aware)

reward = BettingReward(p=0.7, m=0.5)  # Bernoulli(p) data, testing E[X] <= m
delta, C = 0.995, 1.0                 # discount factor, entry cost

G, w_grid, lam_policy, _ = bellman_benchmark(reward, delta)
best = sim_bellman(reward, delta, C, G, w_grid, lam_policy).utilities.mean()
cgf = sim_cgf_aware(reward, delta, C).utilities.mean()
print(f"CGF threshold efficiency: {cgf / best:.3f}")
```

Sequential, incremental rewards: discounted milestone receipts of a fixed bet,
relative to a numerical Bellman policy.

```python
from statcontracts.sequential.milestones import crossing_values, solve_milestone_bellman

reward, delta = BettingReward(p=0.7, m=0.5), 0.9
levels, payouts = (2.0, 4.0, 8.0), [0.2, 0.4, 6.4]   # 80% of the null budget on the last milestone
gro = crossing_values(reward, reward.growth_optimal_lambda(), delta, levels)[0] @ payouts
bellman = solve_milestone_bellman(reward, delta, levels, payouts)["value"]
print(f"growth-optimal bet: {gro / bellman:.3f} of the Bellman value")
```

## Reproducing the papers

See [`experiments/multiagent/README.md`](experiments/multiagent/README.md) and
[`experiments/sequential/README.md`](experiments/sequential/README.md).

## Tests

```bash
pytest                # everything (~50 s)
pytest -m "not slow"  # skip the slower simulation checks (~15 s)
```

`tests/test_regression.py` pins outputs of small-scale versions of the
sequential experiments, including the large-jump values reported in the paper.
`tests/test_milestones.py` checks the milestone computations against an
independent scalar first-passage recursion and analytic hitting values. `tests/test_multiagent.py` checks the multi-agent solvers
against brute-force enumeration.
