"""Empirical CVaR and t-based confidence intervals."""

from __future__ import annotations
import numpy as np
import scipy.stats


def cvar(x, alpha: float) -> float:
    """
    Empirical CVaR_alpha (mean of the worst (1-alpha) tail).

    x     : 1-D array-like of samples
    alpha : tail quantile (e.g. 0.90 -> mean of the top 10%)
    """
    x = np.asarray(x, dtype=float)
    q = np.quantile(x, alpha)
    tail = x[x >= q]
    return float(tail.mean()) if len(tail) else float(q)


def mean_cvar(x, alpha: float) -> tuple[float, float]:
    """
    Empirical (mean, CVaR_alpha) of the same samples in one pass.

    Reporting both sides by side de-confounds a CVaR-only comparison: a
    placement that lowers CVaR while raising the mean is trading mean for tail
    (a genuine risk dimension); one that raises both is simply dominated.
    """
    x = np.asarray(x, dtype=float)
    return float(x.mean()), cvar(x, alpha)


def ci95_t(x) -> tuple[float, float, float, float]:
    """
    Two-sided 95% t-interval for the mean.

    Returns (mean, std_ddof1, lo, hi).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    t = scipy.stats.t.ppf(0.975, df=n - 1)
    half = t * s / np.sqrt(n)
    return m, s, m - half, m + half
