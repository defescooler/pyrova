"""BOOM resolution check over 10 fresh 60/20 splits (seed base 95_000),
alpha=0.9. Arms trained@18: mean-std (120 it), mean-strong (best-of-3 x 240
it, training-mean selection), blend gamma=0.75 (120 it); arms trained@36:
mean-std, blend. ALL placements evaluated at both 18^2 and 64^2 with our
solver; Nadeau-Bengio CIs. When the reference binary is present, also
compares our-solver@32 vs reference@32 on one split's placements (20
programs).
"""

from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import mean_cvar, cvar, ci95_nadeau_bengio, ci95_t
from pyrova.workloads.boom_traces import BoomWorkload, resolve_paths

CONFIG = PKG / "inputs/configs/thermal.config"
HOTSPOT = ROOT / "Tools/HotSpot/hotspot"
ALPHA = 0.90
N_SPLITS = 10
N_TRAIN = 60
SEED_BASE = 95_000
TARGET_PEAK = 40.0
EVAL_GRIDS = (18, 64)


def fit(solver, wl, tr, mode, gamma, n_iter, nr, restarts=1, rrng=None):
    best, best_obj = None, np.inf
    for r in range(restarts):
        pl = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, nr, nr,
                        alpha=ALPHA, blend_gamma=gamma)
        if r > 0:
            pl.raw_x += rrng.standard_normal(pl.n) * 0.5
            pl.raw_y += rrng.standard_normal(pl.n) * 0.5
        pl.optimize(tr, mode=mode, n_iter=n_iter, lr=2e-2, verbose=False)
        obj = pl.objective_and_grad(tr, mode=mode)[0]
        if obj < best_obj:
            best, best_obj = pl, obj
    return best


def eval_peaks(cfg, wl, units_placed, scen, nr):
    s = GridFDSolver(cfg, units_placed, wl.chip_w, wl.chip_h, nr, nr)
    s.build(); s.factorize()
    amb = cfg["ambient"]
    out = np.zeros(len(scen))
    for i, pw in enumerate(scen):
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(units_placed)}
        T = s.solve(s.build_rhs(bp))
        out[i] = float(s.silicon_layer(T).max()) - amb
    return out


def main():
    csvp, rptp = resolve_paths()
    if not csvp:
        print("BOOM_DATA not found.")
        return
    cfg = parse_config(str(CONFIG))
    wl = BoomWorkload(csvp, rptp, config_id="0")
    solvers = {}
    for nr in (18, 36):
        s = GridFDSolver(cfg, wl.units, wl.chip_w, wl.chip_h, nr, nr)
        s.build(); s.factorize()
        solvers[nr] = s

    def peaks_fn(scen):
        p = DiffPlacer(solvers[18], wl.units, wl.chip_w, wl.chip_h, 18, 18, alpha=ALPHA)
        cx, cy = p.get_positions()
        return p._scenario_peaks(cx, cy, scen)
    wl.scale_to_peak(peaks_fn, TARGET_PEAK)
    scen = wl.scenarios()
    n = len(scen)
    ratio = (n - N_TRAIN) / N_TRAIN

    out = PKG / "results/exp021_boom_resolution.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"exp021: BOOM resolution check. {N_SPLITS} fresh splits (base {SEED_BASE}), "
         f"{N_TRAIN}/{n - N_TRAIN}, alpha={ALPHA}; arms trained@18 (mean-std/"
         f"mean-strong bo3x240/blend) and @36 (mean-std/blend); eval@{EVAL_GRIDS}.")

    ARMS = [("ms18", 18, "mean", 0.0, 120, 1), ("mS18", 18, "mean", 0.0, 240, 3),
            ("bl18", 18, "blend", 0.75, 120, 1), ("ms36", 36, "mean", 0.0, 120, 1),
            ("bl36", 36, "blend", 0.75, 120, 1)]
    C = {a[0]: {g: [] for g in EVAL_GRIDS} for a in ARMS}
    keep_units = None
    for seed in range(N_SPLITS):
        perm = np.random.default_rng(SEED_BASE + seed).permutation(n)
        tr = [scen[i] for i in perm[:N_TRAIN]]
        te = [scen[i] for i in perm[N_TRAIN:]]
        rrng = np.random.default_rng(SEED_BASE + 500 + seed)
        units_by = {}
        for tag, nr, mode, g, it, rs in ARMS:
            pl = fit(solvers[nr], wl, tr, mode, g, it, nr, restarts=rs, rrng=rrng)
            units_by[tag] = pl.get_units()
            for ge in EVAL_GRIDS:
                pk = eval_peaks(cfg, wl, units_by[tag], te, ge)
                C[tag][ge].append(cvar(pk, ALPHA))
        if seed == 0:
            keep_units = {k: (units_by[k], te) for k in ("ms18", "bl18")}
        print(f"  split {seed + 1}/{N_SPLITS} done", flush=True)

    for tag in C:
        for ge in EVAL_GRIDS:
            C[tag][ge] = np.asarray(C[tag][ge])

    emit(f"\n  OOS CVaR by arm and evaluation grid (mean over {N_SPLITS} splits):")
    emit(f"  {'arm':>6} " + " ".join(f"eval@{g:<4}" for g in EVAL_GRIDS))
    for tag in C:
        emit(f"  {tag:>6} " + " ".join(f"{C[tag][g].mean():8.3f}" for g in EVAL_GRIDS))

    def delta(a, b, ge):
        d = C[a][ge] - C[b][ge]
        g, _, lo, hi = ci95_nadeau_bengio(d, ratio)
        return g, lo, hi

    emit("\n  R1/R2 (blend vs mean-strong, trained@18):")
    for ge in EVAL_GRIDS:
        g, lo, hi = delta("mS18", "bl18", ge)
        emit(f"    eval@{ge}: dCVaR(mean-strong - blend) = {g:+.3f} [{lo:+.2f},{hi:+.2f}]")
    g18, _, _ = delta("mS18", "bl18", 18)
    g64, lo64, hi64 = delta("mS18", "bl18", 64)
    r1 = (lo64 <= 0 <= hi64) and abs(g64 - g18) <= 1.0
    r2 = g64 >= -1.0
    emit(f"    R1 (equalised null resolution-stable): {'SURVIVES' if r1 else 'CONTINGENT'}")
    emit(f"    R2 (efficiency observation): {'SURVIVES' if r2 else 'DEMOTED'}")

    emit("\n  R3 (training resolution, eval@64):")
    for a36, a18 in (("ms36", "ms18"), ("bl36", "bl18")):
        d = C[a18][64] - C[a36][64]      # >0: training @36 is better
        g, _, lo, hi = ci95_nadeau_bengio(d, ratio)
        emit(f"    {a18}->({a36}): gain from 36^2 training = {g:+.3f} [{lo:+.2f},{hi:+.2f}]"
             + ("  UNDER-RESOLVED*" if lo > 0 else ""))

    if HOTSPOT.exists():
        from pyrova.experiments.exp018_hotspot_crosscheck import hotspot_peak
        emit("\n  P5 (local): our@32 vs reference@32, split-0 placements, 20 programs:")
        for tag, (up, te) in keep_units.items():
            ours = eval_peaks(cfg, wl, up, te, 32)
            with tempfile.TemporaryDirectory() as td:
                ref = np.array([hotspot_peak(up, pw, Path(td), f"p5{tag}{i}",
                                             cfg["ambient"])
                                for i, pw in enumerate(te)])
            mae = float(np.mean(np.abs(ours - ref)))
            emit(f"    {tag}: MAE={mae * 1e3:.1f} mK  r={np.corrcoef(ours, ref)[0, 1]:.6f}"
                 f"  -> {'solvers AGREE (exp018-B MAE was resolution gap)' if mae < 0.05 else 'DISAGREEMENT beyond resolution — investigate'}")
    else:
        emit("\n  P5 SKIPPED (reference binary not present on this host).")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
