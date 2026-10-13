"""Checks of the single-shot contract against Bates, Jordan, Sklar, and Soloff."""

import numpy as np
import pytest
from scipy.stats import norm

from statcontracts.singleshot import (
    Binomial,
    GaussianMean,
    ThresholdLicense,
    best_response,
    is_incentive_aligned,
    max_aligned_level,
    null_profit,
    optimal_license,
    principal_utility,
)

MODELS = [(GaussianMean(n=10, sigma=1.0), 0.0, 0.5), (Binomial(n=30), 0.5, 0.7)]
IDS = ["gaussian", "binomial"]


class EvalueLicense:
    """f(z) = min(C * LR(z), cap): a capped likelihood-ratio e-value license."""

    def __init__(self, model, theta1, theta0, cost, cap):
        self.args = model, theta1, theta0
        self.cost, self.cap = cost, cap

    def __call__(self, z):
        model, theta1, theta0 = self.args
        lr = np.exp(model.log_likelihood_ratio(z, theta1, theta0))
        return np.minimum(self.cost * lr, self.cap)


@pytest.mark.parametrize("model, theta0, theta1", MODELS, ids=IDS)
def test_likelihood_ratio_evalue_licenses_are_aligned(model, theta0, theta1):
    """Prop. 1: C times an e-value is an incentive-aligned license."""
    C = 2.0
    f = EvalueLicense(model, theta1, theta0, C, np.inf)
    np.testing.assert_allclose(model.expectation(f, theta0), C, rtol=1e-7)
    nulls = [theta0, theta0 - 0.1, theta0 - 0.3]
    assert is_incentive_aligned([f], C, model, nulls)
    assert not is_incentive_aligned([f.__class__(model, theta1, theta0, 1.1 * C, np.inf)],
                                    C, model, nulls)


@pytest.mark.parametrize("model, theta0, theta1", MODELS, ids=IDS)
def test_optimal_license_has_exact_size(model, theta0, theta1):
    C, R = 1.0, 20.0
    f = optimal_license(model, theta1, theta0, C, R)
    np.testing.assert_allclose(model.expectation(f, theta0), C, rtol=1e-10)
    assert is_incentive_aligned([f], C, model, [theta0, theta0 - 0.2])


@pytest.mark.parametrize("model, theta0, theta1", MODELS, ids=IDS)
def test_optimal_license_beats_other_aligned_licenses(model, theta0, theta1):
    """Prop. 4: the Neyman-Pearson license maximizes E_theta1[min(f, R)]."""
    C, R = 1.0, 20.0
    best = model.expectation(optimal_license(model, theta1, theta0, C, R), theta1)
    others = [EvalueLicense(model, alt, theta0, C, R)
              for alt in [theta1, theta0 + 0.5 * (theta1 - theta0), 2 * theta1 - theta0]]
    others += [model.likelihood_ratio_test(theta1, theta0, level).scaled(C / level)
               for level in [0.01, 0.2, 0.5]]
    for f in others:
        assert is_incentive_aligned([f], C, model, [theta0])
        assert model.expectation(lambda z: np.minimum(f(z), R), theta1) <= best + 1e-9


def test_gaussian_example():
    """Section 3.1: Z ~ N(theta, 1), null theta = 0, alternative theta = 1."""
    model, C = GaussianMean(), 1.0
    status_quo_threshold = norm.isf(0.05)
    for R in [5.0, 50.0]:
        f_ia = optimal_license(model, 1.0, 0.0, C, R)
        np.testing.assert_allclose(f_ia.threshold, norm.isf(C / R))  # p-value < C/R
        f_sq = ThresholdLicense(status_quo_threshold, R)
        assert is_incentive_aligned([f_ia], C, model, [0.0])
        # Status quo is aligned only when C/R >= 0.05, and then more conservative.
        assert is_incentive_aligned([f_sq], C, model, [0.0]) == (C / R >= 0.05)
        if C / R >= 0.05:
            assert model.expectation(f_ia, 1.0) >= model.expectation(f_sq, 1.0)


def test_status_quo_utility_is_negative_with_many_nulls():
    """Figure 1(b): with R/C = 50, nulls enter the status quo trial."""
    model, C, R = GaussianMean(), 1.0, 50.0
    menu_sq = [ThresholdLicense(norm.isf(0.05), R)]

    def utility(theta, L):  # null approval costs 1, nonnull approval gains 10
        return np.where(L > 0, 10.0 if theta > 0 else -1.0, 0.0)

    def status_quo(theta):
        return best_response(menu_sq, C, model, theta)

    def aligned(theta):
        return optimal_license(model, theta, 0.0, C, R)

    types = [0.0, 1.0]
    for pi0 in [0.5, 0.99, 1.0]:
        weights = [pi0, 1 - pi0]
        u_sq = principal_utility(status_quo, model, types, weights, utility)
        u_ia = principal_utility(aligned, model, types, weights, utility)
        assert u_ia >= 0
        if pi0 == 1.0:
            np.testing.assert_allclose(u_sq, -0.05)  # every null enters
            assert u_ia == 0.0                       # every null opts out


def test_best_response():
    model, C = GaussianMean(), 1.0
    menu = [ThresholdLicense(1.0, 4.0), ThresholdLicense(2.0, 20.0)]
    assert best_response(menu, C, model, 0.0) is None
    assert best_response(menu, C, model, 3.0) is menu[1]


def test_binomial_test_randomizes_at_the_boundary():
    model = Binomial(n=10)
    f = model.likelihood_ratio_test(0.7, 0.5, 0.05)
    assert 0 < f.gamma < 1
    np.testing.assert_allclose(model.expectation(f, 0.5), 0.05, rtol=1e-12)


@pytest.mark.parametrize("alpha, payout, expected_millions, aligned", [
    (0.000625, 1e9, -49, True),
    (0.000625, 1e10, -44, True),
    (0.000625, 1e11, 13, False),
    (0.005, 1e9, -45, True),
    (0.005, 1e10, 0, None),
    (0.005, 1e11, 450, False),
    (0.0494, 1e9, -1, None),
    (0.0494, 1e10, 444, False),
    (0.0494, 1e11, 4890, False),
])
def test_fda_protocols(alpha, payout, expected_millions, aligned):
    """Table 1: expected value of a placebo, for a $50M Phase III trial (None = borderline)."""
    cost = 5e7
    profit = null_profit(alpha, payout, cost)
    np.testing.assert_allclose(profit / 1e6, expected_millions, atol=0.7)
    if aligned is not None:
        assert (alpha <= max_aligned_level(cost, payout)) == aligned
