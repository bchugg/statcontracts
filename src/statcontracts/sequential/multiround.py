"""The finite-horizon multi-round contract of Bates, Jordan, Sklar, and Soloff.

The initial license is L(0) = 0. In each round t = 1, ..., T the agent may
withdraw P(t) from its license, and may pay C_t to run stage t of the trial.
If it runs the stage, it picks an update function f_t, observes
Z_t ~ P_theta, and its license becomes

    L(t) = (L(t-1) + C_t - P(t)) f_t(Z_t);

otherwise L(t) = L(t-1) - P(t). The agent ends with the license L(T), its
withdrawals P = P(1) + ... + P(T), and its investment C (the costs of the
stages it ran), for a profit L(T) + P - C. If every update function is an
e-value, the net profit is a supermartingale for null agents, so no
strategy profits in expectation (their Proposition 5).

  play         runs the contract for an agent strategy
  solve_agent  the agent's best response by dynamic programming (their
               Appendix B.4), for Gaussian evidence and a point alternative
"""

from typing import NamedTuple

import numpy as np

from .simulate import DEFAULT_SEED


class MultiRoundOutcome(NamedTuple):
    license: np.ndarray     # final license L(T) of each path
    withdrawn: np.ndarray   # total withdrawals P
    invested: np.ndarray    # total cost C of the stages run

    @property
    def profit(self):
        return self.license + self.withdrawn - self.invested


def play(model, theta, costs, strategy, n_sim=10000, seed=DEFAULT_SEED):
    """Run the contract for n_sim agents of type theta following `strategy`.

    The strategy is an object with two methods, vectorized over agents:

        decide(t, L, withdrawn, invested) -> (P, run)
            the withdrawal P(t) and whether to run stage t, given the license
            L(t-1) and the totals withdrawn and invested so far;
        update(t, L, Z) -> f_t(Z)
            the chosen update function evaluated at the evidence, given the
            license L = L(t-1) - P(t) after the withdrawal.

    Withdrawals are clipped to [0, L(t-1)]. `model` supplies the evidence,
    e.g. statcontracts.singleshot.GaussianMean.
    """
    rng = np.random.default_rng(seed)
    L = np.zeros(n_sim)
    withdrawn = np.zeros(n_sim)
    invested = np.zeros(n_sim)
    for t, cost in enumerate(np.asarray(costs, dtype=float), start=1):
        P, run = strategy.decide(t, L, withdrawn, invested)
        P = np.clip(P, 0.0, L)
        run = np.asarray(run, dtype=bool)
        Z = model.sample(rng, theta, n_sim)
        L = L - P
        L = np.where(run, (L + cost) * strategy.update(t, L, Z), L)
        withdrawn += P
        invested += np.where(run, cost, 0.0)
    return MultiRoundOutcome(L, withdrawn, invested)


# ===================================================================
# The agent's best response
# ===================================================================

def _increasing_concave_hull(x, y):
    """Vertices of the least concave majorant of (x, y), up to its leftmost maximum."""
    hull = []
    for i in range(len(x)):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            cross = (x[a] - x[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (x[i] - x[o])
            if cross < 0:
                break
            hull.pop()
        hull.append(i)
    hull = np.array(hull)
    return hull[:int(np.argmax(y[hull])) + 1]


class AgentPolicy:
    """The agent's optimal strategy, as computed by solve_agent.

    After t rounds with license grid[k], the agent stops if stop[t, k];
    otherwise it pays C_{t+1} and moves its license to a vertex of the
    concave majorant of values[t + 1] (see solve_agent). It never withdraws.
    `entry_value` is the agent's optimal expected profit, values[0, 0]; the
    agent enters the contract if it is positive. Pass the policy to `play`
    to simulate it.
    """

    def __init__(self, model, theta1, theta0, costs, cap, grid, values, stop,
                 log_lam, hulls, lr_coef):
        self.model, self.theta1, self.theta0 = model, theta1, theta0
        self.costs, self.cap, self.grid = costs, cap, grid
        self.values, self.stop, self.log_lam = values, stop, log_lam
        self.hulls, self.lr_coef = hulls, lr_coef
        self.cumulative_cost = np.concatenate([[0.0], np.cumsum(costs)])

    @property
    def entry_value(self):
        return self.values[0, 0]

    def _index(self, L):
        return np.clip(np.rint(L / self.grid[1]).astype(int), 0, len(self.grid) - 1)

    def next_license(self, t, k, Z):
        """License after running stage t + 1 from grid[k], given the evidence Z."""
        xs, log_slopes = self.hulls[t]
        a, b = self.lr_coef
        tau = (self.log_lam[t, k][..., None] - log_slopes - b) / a
        J = (np.asarray(Z)[..., None] >= tau).sum(axis=-1)
        return xs[J]

    def decide(self, t, L, withdrawn, invested):
        k = self._index(L)
        # Stopping is final: agents that skipped an earlier stage stay out.
        active = np.isclose(invested, self.cumulative_cost[t - 1])
        return np.zeros_like(L), active & ~self.stop[t - 1, k]

    def update(self, t, L, Z):
        return self.next_license(t - 1, self._index(L), Z) / (L + self.costs[t - 1])


def _best_update(model, theta0, theta1, lr_coef, xs, vs, stake, n_iter=200):
    """Single-round optimal update from each stake (their Appendix B.4.3).

    Maximizes E_theta1[v(g(Z))] over licenses g with E_theta0[g(Z)] <= stake,
    where v is concave and increasing with vertices (xs, vs). The optimum
    is g(z) = max{xs[j] : slope_j >= lam / LR(z)}, with lam set by bisection
    so that E_theta0[g(Z)] = stake; stakes of at least xs[-1] take g = xs[-1].
    Returns the optimal values and log lam (-inf for those stakes).
    """
    a, b = lr_coef
    saturated = stake >= xs[-1]
    log_lam = np.full(len(stake), -np.inf)
    value = np.full(len(stake), vs[-1])
    if len(xs) == 1 or saturated.all():
        return value, log_lam
    dx, dv = np.diff(xs), np.diff(vs)
    log_slopes = np.log(dv / dx)

    def tau(ll):
        return (ll[:, None] - log_slopes[None, :] - b) / a

    def expected(ll, theta, d):
        return xs[0] + (model.sf(tau(ll), theta) * d).sum(axis=1)

    target = stake[~saturated]
    spread = 40 * model.scale
    lo = np.full(len(target), a * (theta0 - spread) + b + log_slopes.min())
    hi = np.full(len(target), a * (theta0 + spread) + b + log_slopes.max())
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        above = expected(mid, theta0, dx) > target
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    log_lam[~saturated] = hi
    value[~saturated] = vs[0] + (model.sf(tau(hi), theta1) * dv).sum(axis=1)
    return value, log_lam


def solve_agent(model, theta1, costs, cap, theta0=0.0, n_grid=500):
    """The agent's best response in the multi-round contract, by dynamic programming.

    The principal offers the menu of all e-values in every round; the agent
    has type theta1 > theta0, values its license at min(L, cap), and pays
    costs[t-1] to run stage t. Evidence follows `model`, a
    statcontracts.singleshot.GaussianMean, under which the likelihood ratio
    of theta1 to the null theta0 is increasing in Z.

    The license is restricted to the grid {0, cap/n_grid, ..., cap} (their
    Appendix B.4.4; larger licenses are worth no more). With values[t, k]
    the optimal expected profit after t rounds with license grid[k],
    backward induction gives

        values[T, k] = grid[k] - C(T)
        values[t, k] = max(grid[k] - C(t),
                           max E_theta1[values[t+1](g(Z))] over g on the grid
                               with E_theta0[g(Z)] <= grid[k] + C_{t+1}),

    where C(t) is the cost of the first t stages. The inner maximum is the
    single-round problem under the concave majorant of values[t+1], whose
    solution only takes values at its vertices (their Appendix B.4.2).
    """
    if theta1 <= theta0:
        raise ValueError("theta1 must exceed theta0")
    costs = np.asarray(costs, dtype=float)
    T = len(costs)
    grid = np.linspace(0.0, cap, n_grid + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(costs)])
    # log LR(z) = a z + b
    a = model.n * (theta1 - theta0) / model.sigma ** 2
    b = -model.n * (theta1 ** 2 - theta0 ** 2) / (2 * model.sigma ** 2)

    values = np.empty((T + 1, len(grid)))
    values[T] = grid - cumulative[T]
    stop = np.ones((T, len(grid)), dtype=bool)
    log_lam = np.full((T, len(grid)), -np.inf)
    hulls = [None] * T
    for t in range(T - 1, -1, -1):
        h = _increasing_concave_hull(grid, values[t + 1])
        xs, vs = grid[h], values[t + 1][h]
        cont, log_lam[t] = _best_update(model, theta0, theta1, (a, b), xs, vs,
                                        grid + costs[t])
        stop_value = grid - cumulative[t]
        stop[t] = stop_value >= cont
        values[t] = np.maximum(stop_value, cont)
        hulls[t] = (xs, np.log(np.diff(vs) / np.diff(xs)))
    return AgentPolicy(model, theta1, theta0, costs, cap, grid, values, stop,
                       log_lam, hulls, (a, b))
