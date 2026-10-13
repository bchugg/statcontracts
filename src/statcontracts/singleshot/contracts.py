"""The single-shot statistical contract (Bates, Jordan, Sklar, and Soloff).

The principal offers a cost C and a menu F of license functions. An agent of
private type theta may pay C, choose f in F, and generate evidence
Z ~ P_theta; it then receives the license L = f(Z). The contract is
incentive-aligned if E_theta[f(Z)] <= C for every null type theta and every
f in F, so that null agents cannot profit in expectation. Equivalently,
F / C consists of e-values (their Proposition 1), and incentive-aligned
menus are exactly the maximin optimal ones (their Theorem 1).

Expectations are computed by an evidence model (see models.py).
"""

import numpy as np


def expected_profit(f, cost, model, theta):
    """E_theta[f(Z)] - C: an agent's expected profit from license f."""
    return model.expectation(f, theta) - cost


def is_incentive_aligned(menu, cost, model, null_types, rtol=1e-9):
    """Whether E_theta[f(Z)] <= C for every license f and null type theta."""
    return all(model.expectation(f, theta) <= cost * (1 + rtol)
               for f in menu for theta in null_types)


def best_response(menu, cost, model, theta):
    """The license an agent of type theta picks from a finite menu (eq. (2)).

    The agent maximizes its expected license E_theta[f(Z)], and opts out
    (returns None) if no license has positive expected profit.
    """
    values = [model.expectation(f, theta) for f in menu]
    k = int(np.argmax(values))
    return menu[k] if values[k] > cost else None


def optimal_license(model, theta1, theta0, cost, cap):
    """Best response to the menu of all incentive-aligned licenses (Prop. 4).

    For an agent of type theta1 whose license is worth at most `cap` (e.g.
    the market size R), and the null theta <= theta0, the incentive-aligned
    license maximizing E_theta1[min(f(Z), cap)] is the likelihood-ratio test
    of theta0 against theta1 at level cost / cap, paying cap on rejection.
    Returns None if the agent opts out: when theta1 <= theta0 (a null
    agent), or when the license's expected value does not exceed the cost.
    """
    if theta1 <= theta0 or cap <= cost:
        return None
    f = model.likelihood_ratio_test(theta1, theta0, cost / cap).scaled(cap)
    return f if model.expectation(f, theta1) > cost else None


class _Composed:
    """z -> g(f(z)), keeping the breakpoints of f for quadrature."""

    def __init__(self, g, f):
        self.g, self.f = g, f
        self.breakpoints = getattr(f, "breakpoints", ())

    def __call__(self, z):
        return self.g(self.f(z))


def principal_utility(strategy, model, types, weights, utility):
    """The principal's expected utility E_{theta ~ Q} E[u(theta, L) I] (eq. (1)).

    strategy(theta) is the license an agent of type theta plays, or None if
    it opts out (utility zero); Q puts mass weights[i] on types[i]; and
    utility(theta, L) is the principal's utility from granting license L to
    an agent of type theta, vectorized over L.
    """
    total = 0.0
    for theta, weight in zip(types, weights):
        f = strategy(theta)
        if f is not None:
            u = _Composed(lambda L, theta=theta: utility(theta, L), f)
            total += weight * model.expectation(u, theta)
    return total


# ===================================================================
# Approval protocols
# ===================================================================

def null_profit(alpha, payout, cost):
    """A null agent's expected profit under an approval protocol.

    Approval, which a null agent obtains with probability alpha (the type I
    error), is worth `payout`; entering costs `cost`.
    """
    return alpha * payout - cost


def max_aligned_level(cost, payout):
    """Largest type I error at which an approval protocol is incentive-aligned: C / R."""
    return cost / payout
