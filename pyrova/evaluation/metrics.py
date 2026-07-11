"""Empirical CVaR (optionally scenario-weighted) and t-based confidence intervals."""

from __future__ import annotations
import numpy as np
import scipy.stats


def _weighted_tail(x: np.ndarray, w: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Samples sorted by descending loss with each one's fractional tail mass; returns (xs, phi)."""
    order = np.argsort(-x)
    xs, ws = x[order], w[order]
    m = 1.0 - alpha
    cum = np.concatenate(([0.0], np.cumsum(ws)))
    phi = np.clip(m - cum[:-1], 0.0, ws)
    return xs, phi


def cvar(x, alpha: float, weights=None) -> float:
    """Empirical CVaR_alpha (mean of the worst 1-alpha tail); optional weights give the exact fractional tail."""
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
    """Empirical (mean, CVaR_alpha) of the same samples in one pass; optional per-sample weights."""
    x = np.asarray(x, dtype=float)
    if weights is None:
        return float(x.mean()), cvar(x, alpha)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float((w * x).sum()), cvar(x, alpha, weights=w)


def ci95_t(x) -> tuple[float, float, float, float]:
    """Two-sided 95% t-interval for the mean. Returns (mean, std_ddof1, lo, hi)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    t = scipy.stats.t.ppf(0.975, df=n - 1)
    half = t * s / np.sqrt(n)
    return m, s, m - half, m + half


def ci95_nadeau_bengio(x, test_over_train: float) -> tuple[float, float, float, float]:
    """95% t-interval for J correlated repeated-split estimates via the Nadeau-Bengio variance correction (test_over_train = n_te/n_tr); returns (mean, std_ddof1, lo, hi)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    t = scipy.stats.t.ppf(0.975, df=n - 1)
    half = t * s * np.sqrt(1.0 / n + test_over_train)
    return m, s, m - half, m + half


def paired_t_p(x) -> float:
    """Two-sided p-value of the one-sample t-test for mean(x) == 0 (paired test over per-seed differences)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = x.std(ddof=1)
    if s == 0.0:
        return 0.0 if x.mean() != 0.0 else 1.0
    t = abs(x.mean()) / (s / np.sqrt(n))
    return float(2.0 * scipy.stats.t.sf(t, df=n - 1))


def holm(pvals, alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down: which family p-values stay significant at familywise level alpha."""
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
