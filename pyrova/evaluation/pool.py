"""Pool per-pair D*_k values scattered across result files into one t-CI."""

from __future__ import annotations
import re
import sys

import numpy as np
import scipy.stats


def pool(paths: list[str]) -> tuple[int, float, float, float, float]:
    """Pool every 'D*_k = <val>' line in `paths` into (n, mean, lo, hi, p); files hold disjoint pairs, so n over-counts if a cell was submitted twice."""
    vals: list[float] = []
    pat = re.compile(r"D\*_k\s*=\s*([+-]?\d+\.?\d*)")
    for p in paths:
        with open(p) as fh:
            vals.extend(float(m.group(1)) for m in pat.finditer(fh.read()))
    x = np.asarray(vals, dtype=float)
    n = len(x)
    if n < 2:
        raise SystemExit(f"only {n} D*_k value(s) found across {len(paths)} file(s)")
    m = float(x.mean())
    s = float(x.std(ddof=1))
    t = scipy.stats.t.ppf(0.975, df=n - 1)
    half = t * s / np.sqrt(n)
    tstat = abs(m) / (s / np.sqrt(n))
    pval = float(2.0 * scipy.stats.t.sf(tstat, df=n - 1))
    return n, m, m - half, m + half, pval


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    n, m, lo, hi, p = pool(sys.argv[1:])
    print(f"{n} pairs  D* = {m:+.4f} K  CI95[{lo:+.4f},{hi:+.4f}]  p={p:.4f}")
