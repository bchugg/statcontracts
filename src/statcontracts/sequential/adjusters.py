"""Adjusters for the incremental reward schedule.

An adjuster A: [1, infty) -> [0, infty) is nondecreasing with
int_1^infty A(y)/y^2 dy <= 1; applied to the running maximum E* of an
e-process it locks in a guaranteed (increasing) reward C * A(E*). Writing
h(w) = A(e^w)/e^w, the adjuster condition is equivalent to
int_0^infty h(u) du <= 1, so h acts like a density on the log-evidence
scale; admissible adjusters have int h = 1 exactly. The actual milestone
budget is A(1) at zero plus dA(a)/a at log-level log(a). Its cumulative mass
through u is B(u) = A(e^u)/e^u + int_0^u h(v) dv, which rises from the
atom A(1) at zero to int h = 1.

  shafer:   A_kappa(y) = kappa y^{1-kappa}                 (slope 1-kappa)
  choe:     A(y) = sqrt(y) - 1                             (slope -> 1/2)
  koolen:   A(y) = y^2 ln2 / ((1+y) ln^2(1+y))             (slope -> 1, log-corrected)
  mixture:  A_a(y) = a y int_0^1 kappa^a y^{-kappa} dkappa (Beta(a,1) mix of Shafer)

Each h_<name> is the induced density h(w) of the adjuster <name>, and each
budget_<name> its cumulative milestone budget B(u).
"""

import numpy as np

LN2 = np.log(2.0)


# ---- adjusters A(y) -------------------------------------------------
def shafer(y, kappa):
    return kappa * y ** (1.0 - kappa)


def choe(y):
    return np.sqrt(y) - 1.0


def koolen(y):
    return y ** 2 * LN2 / ((1.0 + y) * np.log(1.0 + y) ** 2)


def mixture(y, a, n_kappa=500):
    """A_a(y) = a y int_0^1 kappa^a y^{-kappa} dkappa, a Beta(a,1) mix of Shafer.

    The integral is evaluated by the trapezoidal rule on n_kappa points.
    """
    y = np.asarray(y, dtype=float)
    kappas = np.linspace(1e-8, 1.0, n_kappa)
    integrand = kappas ** a * y[..., None] ** (-kappas)
    return a * y * np.trapezoid(integrand, kappas, axis=-1)


# ---- induced densities h(w) = A(e^w)/e^w ----------------------------
def h_shafer(w, kappa):
    return kappa * np.exp(-kappa * w)


def h_choe(w):
    return (np.exp(w / 2.0) - 1.0) / np.exp(w)


def h_koolen(w):
    return koolen(np.exp(w)) / np.exp(w)


def h_mixture(w, a, n_kappa=500):
    return mixture(np.exp(w), a, n_kappa) / np.exp(w)


# ---- cumulative milestone budgets B(u) = A(e^u)/e^u + int_0^u h -----
def budget_shafer(u, kappa):
    return 1.0 - (1.0 - kappa) * np.exp(-kappa * u)


def budget_choe(u):
    return 1.0 - np.exp(-u / 2.0)


def budget_koolen(u):
    # int_0^u h = 1 - ln 2 / ln(1 + e^u).
    return h_koolen(u) + 1.0 - LN2 / np.log1p(np.exp(u))


def budget_mixture(u, a, n_t=4001):
    """Beta(a, 1) mixture of the Shafer budgets, by the trapezoidal rule.

    Integrates over t = kappa^a, which is uniform, avoiding the Beta density's
    singularity at zero.
    """
    t = np.linspace(0.0, 1.0, n_t)
    kappa = t ** (1.0 / a)
    u = np.asarray(u, dtype=float)
    values = [np.trapezoid(budget_shafer(x, kappa), t) for x in u.ravel()]
    return np.reshape(values, u.shape)
