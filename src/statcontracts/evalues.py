"""E-values for common nulls.

An e-value for a null H0 is a nonnegative statistic E with E[E] <= 1 under
every distribution in H0. Each function returns log E, computed from the
observations x along `axis`:

  gaussian_lr_log_evalue   N(theta1, sigma^2) against N(theta0, sigma^2);
                           an e-value for theta <= theta0 when theta1 > theta0
  bernoulli_lr_log_evalue  Bernoulli(p1) against Bernoulli(p0);
                           an e-value for p <= p0 when p1 > p0
  betting_log_evalue       prod (1 + lam (x - m)) for x in [0, 1];
                           an e-value for E[X] <= m when 0 <= lam <= 1/m
  hoeffding_log_evalue     prod exp(lam (x - m) - lam^2 / 8) for x in [0, 1];
                           an e-value for E[X] <= m when lam >= 0
"""

import numpy as np


def gaussian_lr_log_evalue(x, theta1, theta0=0.0, sigma=1.0, axis=-1):
    """Log likelihood ratio of N(theta1, sigma^2) to N(theta0, sigma^2)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    return ((theta1 - theta0) * np.sum(x, axis=axis)
            - n * (theta1 ** 2 - theta0 ** 2) / 2) / sigma ** 2


def bernoulli_lr_log_evalue(x, p1, p0, axis=-1):
    """Log likelihood ratio of Bernoulli(p1) to Bernoulli(p0)."""
    x = np.asarray(x, dtype=float)
    k = np.sum(x, axis=axis)
    n = x.shape[axis]
    return k * np.log(p1 / p0) + (n - k) * np.log((1 - p1) / (1 - p0))


def betting_log_evalue(x, lam, m, axis=-1):
    """Log wealth of betting lam on each observation exceeding its null mean m.

    Betting lam = 1/m loses everything when x = 0, giving log wealth -inf.
    """
    with np.errstate(divide="ignore"):
        return np.sum(np.log1p(lam * (np.asarray(x, dtype=float) - m)), axis=axis)


def hoeffding_log_evalue(x, lam, m, axis=-1):
    """Log of the product of Hoeffding e-values exp(lam (x - m) - lam^2 / 8)."""
    x = np.asarray(x, dtype=float)
    return np.sum(lam * (x - m) - lam ** 2 / 8, axis=axis)
