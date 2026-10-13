"""Evidence models for single-shot contracts.

A model gives the law of the agent's evidence Z under each type theta,
through a sufficient statistic, so that expected licenses are computed
exactly rather than by simulation:

  GaussianMean(n, sigma)  Z = mean of n draws from N(theta, sigma^2)
  Binomial(n)             Z = number of successes in n draws from Bernoulli(theta)

Both have monotone likelihood ratios in Z, so the likelihood-ratio test of
theta0 against any theta1 > theta0 rejects for large Z. Licenses are
functions of Z that accept numpy arrays.
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import binom, norm


class ThresholdLicense:
    """f(z) = payout * (1{z > t} + gamma * 1{z = t}).

    An approval rule paying `payout` when the evidence exceeds the threshold
    t. With discrete evidence, gamma pays a fraction of the payout at the
    boundary, so that the license can have any size exactly.
    """

    def __init__(self, threshold, payout=1.0, gamma=0.0):
        self.threshold = threshold
        self.payout = payout
        self.gamma = gamma

    def __repr__(self):
        return (f"ThresholdLicense(threshold={self.threshold}, "
                f"payout={self.payout}, gamma={self.gamma})")

    def __call__(self, z):
        z = np.asarray(z)
        return self.payout * ((z > self.threshold)
                              + self.gamma * (z == self.threshold))

    @property
    def breakpoints(self):
        return (self.threshold,)

    def scaled(self, payout):
        """The same approval rule paying `payout` instead."""
        return ThresholdLicense(self.threshold, payout, self.gamma)


class GaussianMean:
    """Z is the mean of n independent draws from N(theta, sigma^2)."""

    def __init__(self, n=1, sigma=1.0):
        self.n = n
        self.sigma = sigma
        self.scale = sigma / np.sqrt(n)

    def __repr__(self):
        return f"GaussianMean(n={self.n}, sigma={self.sigma})"

    def sample(self, rng, theta, size=None):
        return rng.normal(theta, self.scale, size=size)

    def sf(self, t, theta):
        """P_theta(Z > t)."""
        return norm.sf(t, loc=theta, scale=self.scale)

    def expectation(self, f, theta):
        """E_theta[f(Z)], exactly for threshold licenses and by quadrature otherwise.

        Quadrature splits the range at f.breakpoints, if f has them, so that
        licenses with jumps are integrated accurately.
        """
        if isinstance(f, ThresholdLicense):
            return f.payout * self.sf(f.threshold, theta)
        lo, hi = theta - 12 * self.scale, theta + 12 * self.scale
        points = [p for p in getattr(f, "breakpoints", ()) if lo < p < hi]
        value, _ = quad(lambda z: float(f(np.asarray(z)))
                        * norm.pdf(z, loc=theta, scale=self.scale),
                        lo, hi, points=points or None, limit=200)
        return value

    def log_likelihood_ratio(self, z, theta1, theta0):
        """log dP_theta1 / dP_theta0 as a function of the sample mean z."""
        return self.n * ((theta1 - theta0) * np.asarray(z)
                         - (theta1 ** 2 - theta0 ** 2) / 2) / self.sigma ** 2

    def likelihood_ratio_test(self, theta1, theta0, level):
        """Critical function of the level-`level` test of theta0 against theta1 > theta0."""
        if theta1 <= theta0:
            raise ValueError("theta1 must exceed theta0")
        return ThresholdLicense(theta0 + self.scale * norm.isf(level))


class Binomial:
    """Z is the number of successes in n independent draws from Bernoulli(theta)."""

    def __init__(self, n=1):
        self.n = n

    def __repr__(self):
        return f"Binomial(n={self.n})"

    def sample(self, rng, theta, size=None):
        return rng.binomial(self.n, theta, size=size)

    def sf(self, t, theta):
        """P_theta(Z > t)."""
        return binom.sf(t, self.n, theta)

    def expectation(self, f, theta):
        """E_theta[f(Z)], summed over the support."""
        z = np.arange(self.n + 1)
        return float(np.asarray(f(z), dtype=float) @ binom.pmf(z, self.n, theta))

    def log_likelihood_ratio(self, z, theta1, theta0):
        """log dP_theta1 / dP_theta0 as a function of the success count z."""
        z = np.asarray(z)
        return (z * np.log(theta1 / theta0)
                + (self.n - z) * np.log((1 - theta1) / (1 - theta0)))

    def likelihood_ratio_test(self, theta1, theta0, level):
        """Critical function of the level-`level` test of theta0 against theta1 > theta0.

        Rejects above the smallest k with P_theta0(Z > k) <= level, and
        rejects with weight gamma at k so that the size is exactly `level`.
        """
        if theta1 <= theta0:
            raise ValueError("theta1 must exceed theta0")
        tail = self.sf(np.arange(-1, self.n + 1), theta0)  # P(Z > k), k = -1..n
        k = int(np.argmax(tail <= level)) - 1
        gamma = (level - tail[k + 1]) / binom.pmf(k, self.n, theta0) if k >= 0 else 0.0
        return ThresholdLicense(k, 1.0, min(gamma, 1.0))
