"""Reward processes for a single agent experimenting adaptively.

A reward process pairs a data-generating law with a family of e-process
increments indexed by a parameter lambda. Writing W_t = log(R_t / C) for the
agent's log-reward relative to the entry cost C, each round adds the
log-increment Delta(lambda) = log_increment(lambda, X).

Reward processes in the paper:

  GaussianReward(theta, sigma)  Delta = lam X - lam^2 / 8,        X ~ N(theta, sigma^2)
  BettingReward(p, m)           Delta = log(1 + lam (X - m)),     X ~ Bernoulli(p)
  HoeffdingReward(p, m)         Delta = lam (X - m) - lam^2 / 8,  X ~ Bernoulli(p)

The Hoeffding reward is the mean-only baseline for bounded data: it uses
only the mean of X, so its drift 2(p - m)^2 is at most KL(p || m) (Pinsker).
"""

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import roots_hermite
from scipy.stats import binom, norm


class RewardProcess:
    """Interface shared by all reward processes.

    Subclasses implement sampling, the log-increment, its moments and CGF,
    and the growth-optimal parameter. The remaining methods have generic
    numerical defaults that subclasses override with closed forms.
    """

    name = None
    # Bounds on lambda for the numerical CGF-optimal search.
    lambda_bounds = None
    # Size of the default parameter menu for the Bellman benchmark.
    bellman_n_lambda = None
    # Plug-in strategy: parameter used before any data is seen.
    initial_lambda = None

    def sample(self, rng, size):
        """Draw `size` observations X from the data-generating law."""
        raise NotImplementedError

    def log_increment(self, lam, X):
        """Log-increment of the reward process at parameter `lam`."""
        raise NotImplementedError

    def increment_moments(self, lam):
        """Mean of the log-increment and E[Delta^-] for a fixed parameter."""
        raise NotImplementedError

    def expected_neg_part(self, lam, n):
        """E[S_n^-] for the random walk S_n of log-increments; vectorized over n."""
        raise NotImplementedError

    def expected_pos_part(self, lam, n):
        """E[S_n^+] for the random walk S_n of log-increments; vectorized over n."""
        raise NotImplementedError

    def cgf(self, lam, s):
        """CGF psi_lambda(s) of the log-increment for a fixed parameter."""
        raise NotImplementedError

    def increments(self, lam_grid, n_quad=20):
        """Quadrature for the log-increment at each parameter in `lam_grid`.

        Returns (incr, probs), with incr of shape (len(lam_grid), Q) and
        probs of shape (Q,), so that E[f(Delta(lam_l))] ~ f(incr[l]) @ probs.
        """
        raise NotImplementedError

    def growth_optimal_lambda(self):
        """Growth-optimal (GRO) parameter, maximizing the drift."""
        raise NotImplementedError

    def drift(self):
        """Drift E[Delta] at the growth-optimal parameter."""
        return self.increment_moments(self.growth_optimal_lambda())[0]

    def bellman_lambdas(self, n=None):
        """Parameter menu for the Bellman benchmark."""
        raise NotImplementedError

    def max_positive_drift_lambda(self):
        """Largest parameter with positive drift."""
        raise NotImplementedError

    def cgf_root(self, delta, lam=None):
        """The rho > 0 with psi_lambda(rho) = log(1/delta), by bisection.

        `lam` defaults to the growth-optimal parameter. psi is convex with
        psi(0) = 0 and psi'(0) = drift > 0, so a unique positive root exists
        whenever the drift is positive.
        """
        if lam is None:
            lam = self.growth_optimal_lambda()
        target = np.abs(np.log(delta))
        lo, hi = 0.0, 1.0
        while self.cgf(lam, hi) < target:
            hi *= 2.0
            if hi > 1e6:
                break
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.cgf(lam, mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def cgf_optimal(self, delta):
        """rho* and the CGF-optimal parameter; see `numeric_cgf_optimal`."""
        if self.lambda_bounds is None:
            raise NotImplementedError(f"{type(self).__name__} has no lambda_bounds")
        return numeric_cgf_optimal(self, delta, self.lambda_bounds)

    def plug_in_lambda(self, mean_hat):
        """Plug-in strategy: parameter given an estimate of the mean of X."""
        raise NotImplementedError

    def plug_in_drift(self, mean_hat):
        """Plug-in strategy: drift given an estimate of the mean of X."""
        raise NotImplementedError

    def _bisect_positive_drift(self, cap):
        """Largest parameter (up to cap) with positive drift, by bisection."""
        lo, hi = 1e-9, cap
        if self.increment_moments(hi)[0] > 0:
            return hi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.increment_moments(mid)[0] > 0:
                lo = mid
            else:
                hi = mid
        return lo


def numeric_cgf_optimal(reward, delta, lam_bounds):
    """rho* and the CGF-optimal parameter, by a nested one-dimensional search.

    rho* is the largest rho > 0 with Psi(rho) := sup_lambda psi_lambda(rho)
    <= log(1/delta), and lambda_CGF attains the supremum at rho*, with lambda
    ranging over `lam_bounds`. Returns (rho_star, lam_cgf).
    """
    L = abs(np.log(delta))

    def best(rho):
        r = minimize_scalar(lambda l: -reward.cgf(l, rho), bounds=lam_bounds,
                            method="bounded", options={"xatol": 1e-10})
        return -r.fun, r.x

    lo, hi = 0.0, 1.0
    while best(hi)[0] < L:
        hi *= 2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if best(mid)[0] < L:
            lo = mid
        else:
            hi = mid
    rho = 0.5 * (lo + hi)
    return rho, best(rho)[1]


# ===================================================================
# Gaussian data, sub-Gaussian e-process
# ===================================================================

class GaussianReward(RewardProcess):
    """Delta = lam X - lam^2 / 8 with X ~ N(theta, sigma^2).

    The drift lam theta - lam^2 / 8 depends on the law only through its
    mean, so knowing the drift and knowing the mean coincide here.
    """

    name = "subgauss"
    bellman_n_lambda = 160
    initial_lambda = 1.0

    def __init__(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma

    def __repr__(self):
        return f"GaussianReward(theta={self.theta}, sigma={self.sigma})"

    def sample(self, rng, size):
        return rng.normal(self.theta, self.sigma, size=size)

    def log_increment(self, lam, X):
        return lam * X - lam ** 2 / 8

    def increment_moments(self, lam):
        mean = lam * self.theta - lam ** 2 / 8
        return mean, _normal_pos_part(-mean, abs(lam) * self.sigma)

    def expected_neg_part(self, lam, n):
        n = np.asarray(n, dtype=float)
        mean = n * (lam * self.theta - lam ** 2 / 8)
        return _normal_pos_part(-mean, np.sqrt(n) * abs(lam) * self.sigma)

    def expected_pos_part(self, lam, n):
        n = np.asarray(n, dtype=float)
        mean = n * (lam * self.theta - lam ** 2 / 8)
        return _normal_pos_part(mean, np.sqrt(n) * abs(lam) * self.sigma)

    def cgf(self, lam, s):
        th, sig = self.theta, self.sigma
        return s * (lam * th - lam ** 2 / 8) + s ** 2 * lam ** 2 * sig ** 2 / 2

    def increments(self, lam_grid, n_quad=20):
        nodes, weights = roots_hermite(n_quad)
        x_vals = self.theta + self.sigma * np.sqrt(2) * nodes
        probs = weights / np.sqrt(np.pi)
        incr = lam_grid[:, None] * x_vals[None, :] - lam_grid[:, None] ** 2 / 8
        return incr, probs

    def growth_optimal_lambda(self):
        return 4 * self.theta

    def drift(self):
        return 2 * self.theta ** 2

    def bellman_lambdas(self, n=None):
        n = self.bellman_n_lambda if n is None else n
        return np.linspace(0.01, max(12 * self.theta, 2.0), n)

    def max_positive_drift_lambda(self):
        return 8 * self.theta

    def cgf_root(self, delta, lam=None):
        """Closed form: the CGF is quadratic, psi(s) = mean s + var s^2 / 2."""
        if lam is None:
            lam = self.growth_optimal_lambda()
            mean = self.drift()
        else:
            mean = lam * self.theta - lam ** 2 / 8
        a = lam ** 2 * self.sigma ** 2 / 2.0
        c = -np.abs(np.log(delta))
        return (-mean + np.sqrt(mean ** 2 - 4 * a * c)) / (2 * a)

    def cgf_optimal(self, delta):
        """Closed form for the Gaussian reward."""
        L = abs(np.log(delta))
        th, sig = self.theta, self.sigma
        rho = L / (2 * (th ** 2 + 2 * L * sig ** 2))
        return rho, th / (0.25 - rho * sig ** 2)

    def plug_in_lambda(self, mean_hat):
        return np.clip(4 * np.asarray(mean_hat), 0.0, 20.0)

    def plug_in_drift(self, mean_hat):
        """Drift of the next (clipped) plug-in bet under the fitted mean."""
        lam = self.plug_in_lambda(mean_hat)
        return lam * mean_hat - lam ** 2 / 8


# ===================================================================
# Bernoulli data
# ===================================================================

class _BernoulliReward(RewardProcess):
    """Bernoulli(p) data tested against mean m; the increment takes two values."""

    def __init__(self, p, m):
        self.p = p
        self.m = m

    def __repr__(self):
        return f"{type(self).__name__}(p={self.p}, m={self.m})"

    def _up(self, lam):
        """Log-increment when X = 1."""
        raise NotImplementedError

    def _dn(self, lam):
        """Log-increment when X = 0."""
        raise NotImplementedError

    def sample(self, rng, size):
        return rng.binomial(1, self.p, size=size).astype(float)

    def increment_moments(self, lam):
        p, up, dn = self.p, self._up(lam), self._dn(lam)
        neg = (1 - p) * max(-dn, 0.0) + p * max(-up, 0.0)
        return p * up + (1 - p) * dn, neg

    def _walk(self, lam):
        """(up, dn, p) relabeled, if needed, so that up >= dn.

        Then S_n = n dn + K (up - dn) with K ~ Binomial(n, p), and
        E[K; K in A] = n p P(K' + 1 in A) with K' ~ Binomial(n - 1, p).
        """
        up, dn, p = self._up(lam), self._dn(lam), self.p
        if up < dn:
            return dn, up, 1 - p
        return up, dn, p

    def expected_neg_part(self, lam, n):
        up, dn, p = self._walk(lam)
        n = np.asarray(n, dtype=float)
        jump = up - dn
        if jump == 0:
            return n * max(-dn, 0.0)
        k = np.ceil(-n * dn / jump) - 1           # largest count with S_n < 0
        return (-n * dn * binom.cdf(k, n, p)
                - jump * n * p * binom.cdf(k - 1, n - 1, p))

    def expected_pos_part(self, lam, n):
        up, dn, p = self._walk(lam)
        n = np.asarray(n, dtype=float)
        jump = up - dn
        if jump == 0:
            return n * max(dn, 0.0)
        k = np.floor(-n * dn / jump) + 1          # smallest count with S_n > 0
        return (n * dn * binom.sf(k - 1, n, p)
                + jump * n * p * binom.sf(k - 2, n - 1, p))

    def cgf(self, lam, s):
        p = self.p
        return np.log(p * np.exp(s * self._up(lam))
                      + (1 - p) * np.exp(s * self._dn(lam)))

    def increments(self, lam_grid, n_quad=20):
        incr = np.stack([self._up(lam_grid), self._dn(lam_grid)], axis=1)
        return incr, np.array([self.p, 1 - self.p])


class BettingReward(_BernoulliReward):
    """Delta = log(1 + lam (X - m)) with X ~ Bernoulli(p), testing E[X] <= m.

    Valid bets are lam in [0, 1/m). The growth-optimal bet is
    (p - m) / (m (1 - m)), with drift KL(p || m).
    """

    name = "betting"
    bellman_n_lambda = 400
    initial_lambda = 0.1

    def __init__(self, p, m):
        super().__init__(p, m)
        self.lambda_bounds = (0.0, 1 / m - 1e-6)

    def _up(self, lam):
        return np.log(1 + lam * (1 - self.m))

    def _dn(self, lam):
        return np.log(1 - lam * self.m)

    def log_increment(self, lam, X):
        return np.log(1 + lam * (X - self.m))

    def cgf(self, lam, s):
        p, m = self.p, self.m
        return np.log(p * (1 + lam * (1 - m)) ** s + (1 - p) * (1 - lam * m) ** s)

    def growth_optimal_lambda(self):
        return (self.p - self.m) / (self.m * (1 - self.m))

    def drift(self):
        return kl_bernoulli(self.p, self.m)

    def bellman_lambdas(self, n=None):
        n = self.bellman_n_lambda if n is None else n
        return np.linspace(0.0, 1 / self.m - 0.01, n)

    def max_positive_drift_lambda(self):
        return self._bisect_positive_drift(1 / self.m - 0.01)

    def plug_in_lambda(self, mean_hat):
        """Growth-optimal bet for the fitted law, with the mean clipped at 0.99."""
        m = self.m
        p_hat = np.clip(mean_hat, 0.0, 0.99)
        return np.maximum(p_hat - m, 0.0) / (m * (1 - m))

    def plug_in_drift(self, mean_hat):
        """Drift of the next plug-in bet under the fitted (clipped) law.

        This is KL(p_hat || m) when p_hat > m, and zero otherwise.
        """
        m = self.m
        p_hat = np.clip(mean_hat, 0.0, 0.99)
        lam = self.plug_in_lambda(mean_hat)
        return p_hat * np.log1p(lam * (1 - m)) + (1 - p_hat) * np.log1p(-lam * m)

    def interior_cgf_optimal(self, delta):
        """rho* and the CGF-optimal bet, when the root is interior.

        Requires m < p and log(1/delta) < log(p/m), so that rho* lies in
        (0, 1). There the rho-th moment is strictly concave in lambda, and its
        first-order condition gives the maximizing bet explicitly; rho* is
        then found by bisecting the log moment. More precise than the generic
        `cgf_optimal`. Returns (rho_star, lam_cgf).
        """
        p, m = self.p, self.m
        target = -math.log(delta)
        if not (0 < m < p < 1 and 0 < target < math.log(p / m)):
            raise ValueError("needs m < p and log(1/delta) < log(p/m)")
        log_odds = math.log(p * (1 - m) / ((1 - p) * m))

        def optimum(rho):
            log_ratio = -log_odds / (1 - rho)
            ratio = math.exp(log_ratio)
            denominator = m + (1 - m) * ratio
            bet = (1 - ratio) / denominator
            log_moment = math.log(p + (1 - p) * math.exp(rho * log_ratio))
            return bet, log_moment - rho * math.log(denominator)

        lo, hi = 0.0, 1.0
        for _ in range(70):
            rho = (lo + hi) / 2
            if optimum(rho)[1] < target:
                lo = rho
            else:
                hi = rho
        rho = (lo + hi) / 2
        bet, log_moment = optimum(rho)
        assert abs(log_moment - target) < 1e-13
        return rho, bet


class HoeffdingReward(_BernoulliReward):
    """Delta = lam (X - m) - lam^2 / 8 with X ~ Bernoulli(p).

    The mean-only baseline: valid for any law on [0, 1], with growth-optimal
    parameter 4 (p - m) and drift 2 (p - m)^2.
    """

    name = "hoeffding"
    lambda_bounds = (0.0, 8.0)

    def _up(self, lam):
        return lam * (1 - self.m) - lam ** 2 / 8

    def _dn(self, lam):
        return -lam * self.m - lam ** 2 / 8

    def log_increment(self, lam, X):
        return lam * (X - self.m) - lam ** 2 / 8

    def growth_optimal_lambda(self):
        return 4 * (self.p - self.m)

    def drift(self):
        return 2 * (self.p - self.m) ** 2

    def max_positive_drift_lambda(self):
        return 8 * (self.p - self.m)


def _normal_pos_part(mean, sd):
    """E[Y^+] for Y ~ N(mean, sd^2), allowing sd = 0."""
    if np.all(sd == 0):
        return np.maximum(mean, 0.0)
    return sd * norm.pdf(mean / sd) + mean * norm.cdf(mean / sd)


def kl_bernoulli(p, m):
    """KL(Bernoulli(p) || Bernoulli(m))."""
    if p <= 0 or p >= 1:
        return 0.0
    return p * np.log(p / m) + (1 - p) * np.log((1 - p) / (1 - m))
