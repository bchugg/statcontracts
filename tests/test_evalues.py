"""Checks that the e-values have expectation at most one under their nulls."""

import itertools

import numpy as np
import pytest

from statcontracts.evalues import (
    bernoulli_lr_log_evalue,
    betting_log_evalue,
    gaussian_lr_log_evalue,
    hoeffding_log_evalue,
)


def bernoulli_expectation(log_evalue, p, n=6):
    """E[exp(log_evalue(X))] for X of n iid Bernoulli(p) draws, by enumeration."""
    xs = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
    probs = p ** xs.sum(axis=1) * (1 - p) ** (n - xs.sum(axis=1))
    return float(probs @ np.exp(log_evalue(xs)))


@pytest.mark.parametrize("p", [0.1, 0.3, 0.4])
def test_bernoulli_evalues(p):
    m = 0.4
    cases = [
        lambda x: bernoulli_lr_log_evalue(x, 0.7, m),
        lambda x: betting_log_evalue(x, 1.5, m),
        lambda x: betting_log_evalue(x, 1 / m, m),
        lambda x: hoeffding_log_evalue(x, 2.0, m),
    ]
    for log_evalue in cases:
        value = bernoulli_expectation(log_evalue, p)
        assert value <= 1 + 1e-12
        if p == m and log_evalue is not cases[3]:
            np.testing.assert_allclose(value, 1.0, rtol=1e-12)


def test_gaussian_lr_evalue():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 2.0, size=(400000, 5))
    e = np.exp(gaussian_lr_log_evalue(x, theta1=0.5, theta0=0.0, sigma=2.0))
    np.testing.assert_allclose(e.mean(), 1.0, atol=4 * e.std() / np.sqrt(len(e)))
    # Matches the likelihood ratio computed from the densities.
    from scipy.stats import norm
    direct = (norm.logpdf(x, 0.5, 2.0) - norm.logpdf(x, 0.0, 2.0)).sum(axis=1)
    np.testing.assert_allclose(gaussian_lr_log_evalue(x, 0.5, 0.0, 2.0), direct,
                               rtol=1e-10, atol=1e-10)
