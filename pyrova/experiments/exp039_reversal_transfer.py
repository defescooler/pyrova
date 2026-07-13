"""Grid-transfer check of the large-N i.i.d. oracle gap: D* = trueCVaR(mean-
oracle) - trueCVaR(cvar-oracle) on ev6 i.i.d. power at N_ORACLE=6000 and a
240-iteration budget, single-start arms trained at 24^2 WITH raster jitter and
evaluated on an independent 64^2 grid against an 8000-scenario holdout; common
random numbers — each pair's jitter seed is shared across the mean/cvar arms.

One array task = 3 pairs (offset via PYROVA_PAIR_OFFSET); run 3-4 tasks for
9-12 pairs. PYROVA_NORACLE, PYROVA_BUDGET, PYROVA_PAIRS, and PYROVA_FLP
override the defaults.

Set PYROVA_SMOKE=1 for a tiny execution check.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_flp, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p
from pyrova.experiments.exp003_mean_cvar_correlation import scen_set, chip_box

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
FLP = Path(os.environ.get("PYROVA_FLP", PKG / "inputs/floorplans/ev6.flp"))
ALPHA = 0.9
TRAIN_GRID = 24               # jitter + independent 64^2 eval are the transfer controls
EVAL_GRID = 32 if SMOKE else 64
JITTER = 1.0
N_ITER = int(os.environ.get("PYROVA_BUDGET", 15 if SMOKE else 240))
N_ORACLE = int(os.environ.get("PYROVA_NORACLE", 96 if SMOKE else 6000))
N_PAIRS = int(os.environ.get("PYROVA_PAIRS", 2 if SMOKE else 3))
PAIR_OFFSET = int(os.environ.get("PYROVA_PAIR_OFFSET", "0"))
N_TEST = 200 if SMOKE else 8000


def main():
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    units = parse_flp(str(FLP))
    n = len(units)
    cw, ch = chip_box(units)
    solver = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()
    tot = 2.0 * n
    test = scen_set(units, tot, np.random.default_rng(99), N_TEST)

    tag = f"{N_ORACLE}" if PAIR_OFFSET == 0 else f"{N_ORACLE}_off{PAIR_OFFSET}"
    out = PKG / f"results/exp039_transfer_{FLP.stem}_{tag}.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"reversal transfer check: {FLP.stem} i.i.d., N_ORACLE={N_ORACLE}, "
         f"budget={N_ITER}, train@{TRAIN_GRID}+jitter{JITTER} -> eval@{EVAL_GRID}, "
         f"{N_PAIRS} pairs (offset {PAIR_OFFSET}), N_TEST={N_TEST}. "
         + ("[SMOKE]" if SMOKE else "[full]"))

    def fit(tr, mode, jseed):
        pl = DiffPlacer(solver, units, cw, ch, TRAIN_GRID, TRAIN_GRID, alpha=ALPHA)
        pl.optimize(tr, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=jseed)
        return pl.get_units()

    def eval_cvar(up):
        s = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
        s.build(); s.factorize()
        pk = np.array([float(s.silicon_layer(s.solve(s.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - ambient
            for pw in test])
        return cvar(pk, ALPHA)

    t0 = time.perf_counter()
    Dk = []
    for k in range(PAIR_OFFSET, PAIR_OFFSET + N_PAIRS):
        tr = scen_set(units, tot, np.random.default_rng(10_000 + k), N_ORACLE)
        js = 390_000 + 1000 * k                     # CRN: shared across the two arms
        c_m = eval_cvar(fit(tr, "mean", js))
        c_c = eval_cvar(fit(tr, "cvar", js))
        Dk.append(c_m - c_c)
        emit(f"  pair {k}: D*_k = {Dk[-1]:+.4f}")

    dt = time.perf_counter() - t0
    m, _, lo, hi = ci95_t(Dk)
    emit(f"D*(transfer, N={N_ORACLE}) = {m:+.4f} K CI[{lo:+.4f},{hi:+.4f}] "
         f"p={paired_t_p(Dk):.4f}")
    emit(f"runtime: {dt:.1f}s total, {dt/max(N_PAIRS,1):.1f}s/pair "
         f"(2 fits + 2 evals@{EVAL_GRID}^2 per pair)")
    emit("(CI>0 -> the reversal survives trap controls; <=0 -> matched-grid ghost. "
         "Pool pairs across offsets for the combined CI.)")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
