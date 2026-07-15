"""Pool per-pair values from sharded result files into one t-CI.

    python -m pyrova.evaluation.pool [--key NAME] FILE...

NAME is the per-pair label to pool (default D*_k), e.g. dCVaR_strong.
"""

from __future__ import annotations
import re
import sys

import numpy as np
import scipy.stats


def pool(paths: list[str], key: str = "D*_k") -> tuple[int, float, float, float, float]:
    """Pool every '<key> = <val>' line in `paths` into (n, mean, lo, hi, p).
    Files must hold disjoint pairs; a duplicated shard inflates n."""
    vals: list[float] = []
    pat = re.compile(re.escape(key) + r"\s*=\s*([+-]?\d+\.?\d*)")
    for p in paths:
        with open(p) as fh:
            vals.extend(float(m.group(1)) for m in pat.finditer(fh.read()))
    x = np.asarray(vals, dtype=float)
    n = len(x)
    if n < 2:
        raise SystemExit(f"only {n} {key} value(s) found across {len(paths)} file(s)")
    m = float(x.mean())
    s = float(x.std(ddof=1))
    t = scipy.stats.t.ppf(0.975, df=n - 1)
    half = t * s / np.sqrt(n)
    tstat = abs(m) / (s / np.sqrt(n))
    pval = float(2.0 * scipy.stats.t.sf(tstat, df=n - 1))
    return n, m, m - half, m + half, pval


if __name__ == "__main__":
    args = sys.argv[1:]
    key = "D*_k"
    if args and args[0] == "--key":
        key = args[1]
        args = args[2:]
    if not args:
        raise SystemExit(__doc__)
    n, m, lo, hi, p = pool(args, key)
    print(f"{n} pairs  {key} = {m:+.4f} K  CI95[{lo:+.4f},{hi:+.4f}]  p={p:.4f}")
