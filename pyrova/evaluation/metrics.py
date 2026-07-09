"""Empirical CVaR (optionally scenario-weighted) and t-based confidence intervals."""

from __future__ import annotations
import numpy as np
import scipy.stats


def _weighted_tail(x: np.ndarray, w: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Samples sorted by descending loss with their included tail mass.

    Returns (xs, phi). phi_i: fractional tail mass of sample i (the boundary
    atom enters fractionally), sum(phi) = 1-alpha (up to the total available
    mass).
    """
    order = np.argsort(-x)
    xs, ws = x[order], w[order]
    m = 1.0 - alpha
    cum = np.concatenate(([0.0], np.cumsum(ws)))
    phi = np.clip(m - cum[:-1], 0.0, ws)
    return xs, phi


def cvar(x, alpha: float, weights=None) -> float:
    """
    Empirical CVaR_alpha (mean of the worst (1-alpha) tail).

    x       : 1-D array-like of samples
    alpha   : tail level (e.g. 0.90 -> the worst 10%)
    weights : optional per-sample probability weights (normalised internally).
              With weights, CVaR is the weighted average of the top (1-alpha)
              of probability MASS, the boundary sample included fractionally.
              With uniform weights this equals the unweighted estimator whenever
              (1-alpha)*N is an integer (no ties) — e.g. N=40, alpha=0.9.

    WARNING: the unweighted (quantile-mask) branch averages the ceil(N(1-alpha))
    worst samples, so its effective tail fraction is nominal only when (1-alpha)N
    is integral; at small N it exceeds alpha (e.g. 12.5% at N=16, alpha=0.9). Pass
    weights for the exact fractional-boundary tail.
    """
    x = np.asarray(x, dtype=float)
    if weights is None:
        q = np.quantile(x, alpha)
        tail = x[x >= q]
        return float(tail.mean()) if len(tail) else float(q)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    xs, phi = _weighted_tail(x, w, alpha)
    mass = phi.sum()
    return float((phi * xs).sum() / mass) if mass > 0 else float(xs[0])


def mean_cvar(x, alpha: float, weights=None) -> tuple[float, float]:
    """
    Empirical (mean, CVaR_alpha) of the same samples in one pass.

    Reporting both sides by side de-confounds a CVaR-only comparison: a
    placement that lowers CVaR while raising the mean is trading mean for tail
    (a genuine risk dimension); one that raises both is simply dominated.

    weights : optional per-sample probability weights (see `cvar`); the mean
              becomes the weighted mean.
    """
    x = np.asarray(x, dtype=float)
    if weights is None:
        return float(x.mean()), cvar(x, alpha)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float((w * x).sum()), cvar(x, alpha, weights=w)


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


def ci95_nadeau_bengio(x, test_over_train: float) -> tuple[float, float, float, float]:
    """95% t-interval for the mean of J repeated random-subsampling estimates,
    with the Nadeau-Bengio (2003) variance correction.

    Repeated train/test splits of ONE fixed dataset are positively correlated
    (every pair of splits shares data), so the naive s/sqrt(J) standard error
    understates the variance and the plain t-CI is anti-conservative. The
    correction inflates it: NB SE = s*sqrt(1/J + n_te/n_tr), with
    test_over_train = n_te/n_tr.

    Returns (mean, std_ddof1, lo, hi).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    t = scipy.stats.t.ppf(0.975, df=n - 1)
    half = t * s * np.sqrt(1.0 / n + test_over_train)
    return m, s, m - half, m + half


def paired_t_p(x) -> float:
    """Two-sided p-value of the one-sample t-test for mean(x) == 0 (the paired
    test when x is a vector of per-seed differences)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = x.std(ddof=1)
    if s == 0.0:
        return 0.0 if x.mean() != 0.0 else 1.0
    t = abs(x.mean()) / (s / np.sqrt(n))
    return float(2.0 * scipy.stats.t.sf(t, df=n - 1))


def holm(pvals, alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down: which of the family of p-values stay
    significant at familywise level alpha. Use whenever a sweep reports
    per-cell significance stars — a '>=1 of many cells' verdict without this
    has familywise error ~ 1-(1-alpha)^m, not alpha."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    keep = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            keep[i] = True
        else:
            break
    return keep.tolist()
