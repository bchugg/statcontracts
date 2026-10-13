"""Adjusters for the incremental reward schedule.

An adjuster A: [1, infty) -> [0, infty) is nondecreasing with
int_1^infty A(y)/y^2 dy <= 1; applied to the running maximum E* of an
e-process it locks in a guaranteed (increasing) reward C * A(E*). Writing
h(w) = A(e^w)/e^w, the adjuster condition is equivalent to
int_0^infty h(u) du <= 1, so h acts like a density on the log-evidence
scale; admissible adjusters have int h = 1 exactly.

  shafer:   A_kappa(y) = kappa y^{1-kappa}                 (slope 1-kappa)
  choe:     A(y) = sqrt(y) - 1                             (slope -> 1/2)
  koolen:   A(y) = y^2 ln2 / ((1+y) ln^2(1+y))             (slope -> 1, log-corrected)
  mixture:  A_a(y) = a y int_0^1 kappa^a y^{-kappa} dkappa (Beta(a,1) mix of Shafer)

Each h_<name> is the induced density h(w) of the adjuster <name>.
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
