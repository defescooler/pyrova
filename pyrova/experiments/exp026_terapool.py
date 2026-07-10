"""exp026: the spatial-tile regime — a PREDICTED NULL on real geometry.

TeraPool (128 tiles / 16 SubGroups, published post-layout geometry) is the
tiled-manycore case. The extraction (workloads/terapool.py) found NO published
spatially structured per-tile activity: all kernels are SPMD, so hotspot
mobility can only come from RANDOM load imbalance (CV anchored to MemPool's
3-17% imbalance losses). Unstructured mobility is this project's i.i.d.
regime, where the established verdict (exp015-B) is "no separable tail
dimension". This experiment therefore tests a FALSIFIABLE PREDICTION rather
than hunting a positive:

  PREDICTION (pre-registered): D* ~ 0 on real tiled geometry, because the
  3-condition screen's condition 2 (mode-structured hotspot mobility) fails.

Every evaluation trap is controlled (exp023 template): 120-it budgets,
raster_jitter=1.0 training, eval@64x64, mean baseline best-of-3.

Design:
  GATE-A (mode structure): hotspot across the 5 pure uniform kernel vectors —
      characterizes whether any MODE-structured mobility exists (expected:
      1 block, i.e. none, as published).
  GATE-B (scenario mobility): hotspot histogram over 200 imbalance-noised
      scenarios; must span >= 3 SubGroups for the D* question to be
      non-degenerate (expected: passes — mobility exists but is random).
  E1: oracle D*, N_ORACLE=1000, 5 pairs, paired CI, eval@64.
  E2: mean-strong vs cvar vs blend(0.75), N in {32, 128}, 5 seeds.
PRE-REGISTERED READINGS:
  - NULL CONFIRMED if E1 CI contains 0: the i.i.d. verdict transfers to real
    tiled geometry and the screen correctly says "don't bother" — the useful
    output is the validated screen, not a placement win.
  - SURPRISE if E1 CI > 0: unstructured mobility alone suffices — the
    mechanism account (mode-structured mobility required) needs revision;
    report as an open question, do not spin.
  - RED FLAG if E1 CI < 0: optimizer attribution required before any reading
    (exp013 protocol).
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import cvar, ci95_t
from pyrova.workloads.terapool import TeraPoolWorkloadModel, terapool_units, N_SG

CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA = 0.90
TRAIN_GRID = 18
EVAL_GRID = 64
N_ITER = 120
N_ORACLE = 1000
N_OR_PAIRS = 5
N_TRAINS = [32, 128]
N_SEEDS = 5
N_TEST = 1000
JITTER = 1.0
TARGET_PEAK = 40.0   # exp021 BOOM protocol


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def fit(solver, units, chip_w, chip_h, train, mode, gamma=0.75,
        restarts=1, rrng=None, seed=0):
    best, best_obj = None, np.inf
    for r in range(restarts):
        pl = DiffPlacer(solver, units, chip_w, chip_h, TRAIN_GRID, TRAIN_GRID,
                        alpha=ALPHA, blend_gamma=gamma)
        if r > 0:
            pl.raw_x += rrng.standard_normal(pl.n) * 0.5
            pl.raw_y += rrng.standard_normal(pl.n) * 0.5
        pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=940_000 + seed + 37 * r)
        obj = pl.objective_and_grad(train, mode=mode)[0]
        if obj < best_obj:
            best, best_obj = pl, obj
    return best


def eval_mc(cfg, units_placed, scen, chip_w, chip_h, ambient):
    s = GridFDSolver(cfg, units_placed, chip_w, chip_h, EVAL_GRID, EVAL_GRID)
    s.build(); s.factorize()
    pk = np.zeros(len(scen))
    for i, pw in enumerate(scen):
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(units_placed)}
        T = s.solve(s.build_rhs(bp))
        pk[i] = float(s.silicon_layer(T).max()) - ambient
    return float(pk.mean()), cvar(pk, ALPHA)


def nearest_block(units, flat, chip_w, chip_h):
    cell = (flat // EVAL_GRID, flat % EVAL_GRID)
    return min(units, key=lambda u: (u["leftx"] + u["width"] / 2 - (cell[1] + .5) * chip_w / EVAL_GRID) ** 2
               + (u["bottomy"] + u["height"] / 2 - (EVAL_GRID - cell[0] - .5) * chip_h / EVAL_GRID) ** 2)["name"]


def main():
    units = terapool_units()
    cfg = parse_config(str(CONFIG))
    chip_w, chip_h = chip_box(units)
    ambient = cfg["ambient"]
    solver = GridFDSolver(cfg, units, chip_w, chip_h, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()

    out = PKG / "results/exp026_terapool.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    model0 = TeraPoolWorkloadModel(units, seed=0)

    def peaks_fn(scen):
        p = DiffPlacer(solver, units, chip_w, chip_h, TRAIN_GRID, TRAIN_GRID,
                       alpha=ALPHA)
        cx, cy = p.get_positions()
        return p._scenario_peaks(cx, cy, scen)

    k = model0.calibrate_scale(peaks_fn, TARGET_PEAK)
    st = model0.regime_stats()
    emit(f"exp026: TeraPool tiled-manycore ({N_SG} SubGroups, bbox "
         f"{chip_w*1e3:.1f}x{chip_h*1e3:.1f} mm; geometry per arXiv:2603.01629). "
         f"alpha={ALPHA}; train@{TRAIN_GRID}+jitter={JITTER}; eval@{EVAL_GRID}; "
         f"{N_ITER} it; mean best-of-3; scale k={k:.2f} to weighted-mean peak "
         f"{TARGET_PEAK} K.")
    emit(f"regime stats: total-CV={st['total_cv']:.3f}  "
         f"hot-SG share={100*st['hot_sg_share_mean']:.1f}%  "
         f"argmax entropy={st['argmax_entropy_bits']:.2f} bits (max {np.log2(N_SG):.2f})")
    emit("PREDICTION (pre-registered): D* ~ 0 — published activity is SPMD-"
         "uniform; imbalance mobility is unstructured (i.i.d. regime).")

    s64 = GridFDSolver(cfg, units, chip_w, chip_h, EVAL_GRID, EVAL_GRID)
    s64.build(); s64.factorize()

    # GATE-A: mode-structured mobility over pure uniform kernel vectors.
    hot_a = set()
    for r, kn in zip(model0.rel, model0.kernel_names):
        pw = np.full(N_SG, r / N_SG) * model0.scale
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(units)}
        T = s64.solve(s64.build_rhs(bp))
        blk = nearest_block(units, int(np.argmax(s64.silicon_layer(T))), chip_w, chip_h)
        hot_a.add(blk)
        emit(f"  GATE-A kernel {kn:7s}: hotspot near {blk}")
    emit(f"GATE-A: mode-structured hotspot spans {len(hot_a)} block(s) — "
         + ("mode structure EXISTS (unexpected vs published SPMD uniformity)"
            if len(hot_a) >= 3 else
            "no mode-structured mobility (as published; screen condition 2 fails)"))

    # GATE-B: scenario mobility with imbalance (D* question must be non-degenerate).
    gb = TeraPoolWorkloadModel(units, seed=42)
    gb.scale = model0.scale
    hot_b = {}
    for pw in gb.sample(200):
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(units)}
        T = s64.solve(s64.build_rhs(bp))
        blk = nearest_block(units, int(np.argmax(s64.silicon_layer(T))), chip_w, chip_h)
        hot_b[blk] = hot_b.get(blk, 0) + 1
    emit(f"GATE-B: scenario hotspot spans {len(hot_b)} SubGroups over 200 draws "
         f"(requires >= 3) -> {'PASS' if len(hot_b) >= 3 else 'FAIL'}  "
         f"histogram={dict(sorted(hot_b.items(), key=lambda x: -x[1]))}")
    if len(hot_b) < 3:
        emit("GATE-B FAILED — hotspot immobile even with imbalance; D* is "
             "degenerate here and no verdict prints (screen: don't bother).")
        fh.close()
        return

    def make_model(seed):
        m = TeraPoolWorkloadModel(units, seed=seed)
        m.scale = model0.scale
        return m

    # E1: oracle D* with all traps controlled.
    emit(f"\nE1: oracle D*, N_ORACLE={N_ORACLE}, {N_OR_PAIRS} pairs, "
         f"common {N_TEST}-scenario holdout at {EVAL_GRID}^2:")
    test = make_model(777).sample(N_TEST)
    Dk = []
    for k_ in range(N_OR_PAIRS):
        train = make_model(10_000 + k_).sample(N_ORACLE)
        rrng = np.random.default_rng(20_000 + k_)
        p_m = fit(solver, units, chip_w, chip_h, train, "mean", restarts=3,
                  rrng=rrng, seed=k_)
        p_c = fit(solver, units, chip_w, chip_h, train, "cvar", seed=k_)
        _, c_m = eval_mc(cfg, p_m.get_units(), test, chip_w, chip_h, ambient)
        _, c_c = eval_mc(cfg, p_c.get_units(), test, chip_w, chip_h, ambient)
        Dk.append(c_m - c_c)
        emit(f"  pair {k_}: D*_k = {Dk[-1]:+.3f} K")
    Dm, _, D_lo, D_hi = ci95_t(Dk)
    emit(f"  D* = {Dm:+.3f} K CI[{D_lo:+.3f},{D_hi:+.3f}]")

    # E2: small-N learnability vs the STRONG mean baseline.
    emit(f"\nE2: learnability vs mean-strong (bo3), {N_SEEDS} seeds:")
    for n in N_TRAINS:
        dC, dM = {"cvar": [], "blend": []}, {"cvar": [], "blend": []}
        for seed in range(N_SEEDS):
            train = make_model(30_000 + 100 * seed + n).sample(n)
            rrng = np.random.default_rng(40_000 + seed)
            pm = fit(solver, units, chip_w, chip_h, train, "mean", restarts=3,
                     rrng=rrng, seed=seed)
            m_m, c_m = eval_mc(cfg, pm.get_units(), test, chip_w, chip_h, ambient)
            for arm, mode in (("cvar", "cvar"), ("blend", "blend")):
                pl = fit(solver, units, chip_w, chip_h, train, mode, seed=seed)
                m_a, c_a = eval_mc(cfg, pl.get_units(), test, chip_w, chip_h, ambient)
                dC[arm].append(c_m - c_a); dM[arm].append(m_m - m_a)
            print(f"  N={n} seed {seed + 1}/{N_SEEDS}", flush=True)
        for arm in ("cvar", "blend"):
            gc, _, lo, hi = ci95_t(dC[arm])
            gm, _, mlo, mhi = ci95_t(dM[arm])
            fc = "*" if lo > 0 else ("x" if hi < 0 else " ")
            emit(f"  N={n} {arm:5s}: dCVaR={gc:+.3f}{fc} [{lo:+.3f},{hi:+.3f}]  "
                 f"dMean={gm:+.3f} [{mlo:+.3f},{mhi:+.3f}]")

    # Pre-registered verdict against the prediction.
    if D_lo <= 0 <= D_hi:
        v = (f"NULL CONFIRMED as predicted: D*={Dm:+.3f} [{D_lo:+.3f},{D_hi:+.3f}] "
             f"— the i.i.d. verdict transfers to real tiled geometry; the "
             f"3-condition screen (condition 2: mode-structured mobility) "
             f"correctly said don't bother.")
    elif D_lo > 0:
        v = (f"SURPRISE: D*={Dm:+.3f} [{D_lo:+.3f},{D_hi:+.3f}] > 0 under "
             f"unstructured mobility — contradicts the mechanism account; "
             f"open question, report without spin.")
    else:
        v = (f"RED FLAG: D*={Dm:+.3f} [{D_lo:+.3f},{D_hi:+.3f}] < 0 — apply the "
             f"exp013 attribution protocol (budget response) before any reading.")
    emit(f"\nPRE-REGISTERED VERDICT: {v}")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
