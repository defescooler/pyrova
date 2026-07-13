"""Controlled die-size sweep of the oracle gap: the hetero-SoC workload model
(6 modes, distinct heavy hotspots) is held fixed while the geometry is scaled
by linear factors [0.25, 0.40, 0.63, 1.00] over the 11.5x13 mm baseline
(~9 / 24 / 57 / 143 mm^2 dies), with total power recalibrated per scale to a
40 K mean peak dT. Estimand per scale: oracle D* = trueCVaR(mean-arm) -
trueCVaR(cvar-arm), symmetric single-start arms at matched budget, train@18
with raster jitter, guaranteed-legal evaluation on an independent 64^2 grid;
the same scenario streams and test set at every scale, so cross-scale
differences are paired per seed. Fail-closed gates: >=3 distinct hotspot
blocks over the 6 pure modes at each scale, guaranteed legality.

Set PYROVA_SMOKE=1 for a tiny execution check.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.optimizer.legalize import legalize_units_exact, LegalizationInfeasible
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p
from pyrova.workloads.hetero_soc import (soc_units, HeteroSoCWorkloadModel,
                                         _BLOCKS, _MODES)

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
EVAL_ALPHA = 0.90
TARGET_PEAK = 40.0            # every scale calibrated to the same operating point
TRAIN_GRID = 18
JITTER = 1.0
DENSITY_W = 5.0e2
DENSITY_LAM0 = 3.0e1
DENSITY_GRID = (48, 48)

SCALES = [0.25, 1.0] if SMOKE else [0.25, 0.40, 0.63, 1.00]
EVAL_GRID = 32 if SMOKE else 64
N_ITER = 12 if SMOKE else 100
N_ORACLE = 100 if SMOKE else 2000
N_PAIRS = 2 if SMOKE else 6
N_TEST = 200 if SMOKE else 2000
N_CALIB = 50 if SMOKE else 300


def scaled_units(s: float) -> list[dict]:
    """The tiled hetero-SoC layout with every length multiplied by s."""
    return [dict(name=u["name"], width=u["width"] * s, height=u["height"] * s,
                 leftx=u["leftx"] * s, bottomy=u["bottomy"] * s)
            for u in soc_units()]


def block_at_peak(units, cw, ch, nx, ny, T):
    """Name of the block containing the hottest cell (nearest centre if the peak
    falls in whitespace)."""
    j, i = np.unravel_index(int(np.argmax(T)), T.shape)
    px, py = (i + 0.5) * cw / nx, (j + 0.5) * ch / ny
    for u in units:
        if (u["leftx"] <= px <= u["leftx"] + u["width"]
                and u["bottomy"] <= py <= u["bottomy"] + u["height"]):
            return u["name"]
    cx = np.array([u["leftx"] + u["width"] / 2 for u in units])
    cy = np.array([u["bottomy"] + u["height"] / 2 for u in units])
    return units[int(np.argmin((cx - px) ** 2 + (cy - py) ** 2))]["name"]


def main():
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    base_units = soc_units()
    pmax = np.array([b[3] for b in _BLOCKS])

    out = PKG / "results/exp042_die_size.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"die-size sweep [hetero_soc workload, geometry-only]: scales={SCALES}, "
         f"target mean peak dT={TARGET_PEAK:.0f}K, N_ORACLE={N_ORACLE}, "
         f"{N_PAIRS} pairs, {N_ITER} it, train@{TRAIN_GRID}+jitter -> "
         f"guaranteed-legal eval@{EVAL_GRID} (alpha={EVAL_ALPHA}). "
         + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    # Scenario streams are power-only, hence identical at every scale: the
    # cross-scale contrast is paired per seed by construction.
    test_raw = HeteroSoCWorkloadModel(base_units, seed=99).sample(N_TEST)
    trains_raw = [HeteroSoCWorkloadModel(base_units, seed=10_000 + k).sample(N_ORACLE)
                  for k in range(N_PAIRS)]

    Dstar = {}          # scale -> list of per-pair D*
    dMean = {}
    g1_pass = {}
    ovl_max = 0.0
    infeasible = 0

    for s in SCALES:
        units = scaled_units(s)
        cw = max(u["leftx"] + u["width"] for u in units)
        ch = max(u["bottomy"] + u["height"] for u in units)

        # Calibrate the operating point: dT is linear in power, so one mean-peak
        # measurement on the tiled layout fixes the rescale exactly.
        se = GridFDSolver(cfg, units, cw, ch, EVAL_GRID, EVAL_GRID)
        se.build(); se.factorize()

        def peak(solver, up, pw):
            return float(solver.silicon_layer(solver.solve(solver.build_rhs(
                {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - ambient

        m0 = float(np.mean([peak(se, units, pw) for pw in test_raw[:N_CALIB]]))
        c = TARGET_PEAK / m0

        # G1: the 6 pure modes must still resolve distinct hotspots at this scale.
        hot = {block_at_peak(units, cw, ch, EVAL_GRID, EVAL_GRID,
                             se.silicon_layer(se.solve(se.build_rhs(
                                 {u["name"]: float(v)
                                  for u, v in zip(units, np.array(mv) * pmax * c)}))))
               for mv in _MODES.values()}
        g1_pass[s] = len(hot) >= 3
        emit(f"\nscale={s:.2f}: die {cw*1e3:.1f}x{ch*1e3:.1f}mm "
             f"({cw*ch*1e6:.0f}mm^2), power x{c:.3f} (tiled mean peak {m0:.1f}K) | "
             f"G1 hotspot blocks over modes: {len(hot)} {sorted(hot)} -> "
             f"{'PASS' if g1_pass[s] else 'FAIL (mechanism filtered out; excluded)'}")
        if not g1_pass[s]:
            continue

        test = [pw * c for pw in test_raw]
        st = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
        st.build(); st.factorize()

        def eval_cvar_mean(up):
            sv = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
            sv.build(); sv.factorize()
            pk = np.array([peak(sv, up, pw) for pw in test])
            return cvar(pk, EVAL_ALPHA), float(pk.mean())

        def fit(tr, mode, jseed):
            pl = DiffPlacer(st, units, cw, ch, TRAIN_GRID, TRAIN_GRID,
                            alpha=EVAL_ALPHA, nonoverlap_w=1e4,
                            density_w=DENSITY_W, density_lam0=DENSITY_LAM0,
                            density_grid=DENSITY_GRID)
            pl.optimize(tr, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                        raster_jitter=JITTER, jitter_seed=jseed)
            up, of = legalize_units_exact(pl.get_units(), cw, ch)
            return up, of

        Dstar[s], dMean[s] = [], []
        for k in range(N_PAIRS):
            tr = [pw * c for pw in trains_raw[k]]
            js = 500_000 + 1000 * k                      # CRN across arms and scales
            try:
                um, fm = fit(tr, "mean", js)
                uc, fc = fit(tr, "cvar", js)
            except LegalizationInfeasible:
                infeasible += 1
                emit(f"  scale={s:.2f} pair {k+1}: INFEASIBLE, skipped")
                continue
            ovl_max = max(ovl_max, fm, fc)
            c_m, m_m = eval_cvar_mean(um)
            c_c, m_c = eval_cvar_mean(uc)
            Dstar[s].append(c_m - c_c)
            dMean[s].append(m_m - m_c)
            print(f"  scale={s:.2f} pair {k+1}/{N_PAIRS}", flush=True)
        d = np.array(Dstar[s])
        if len(d) >= 2:
            m, _, lo, hi = ci95_t(d)
            emit(f"  scale={s:.2f}: D*={m:+.4f} [{lo:+.4f},{hi:+.4f}] "
                 f"dMean={np.mean(dMean[s]):+.4f}  (n={len(d)} pairs)")

    g2 = ovl_max < 1e-3 and infeasible == 0
    emit(f"\nG2 (legality): max overlap {100*ovl_max:.3f}%, {infeasible} infeasible "
         f"-> {'PASS' if g2 else 'FAIL'}")

    emit("\n===== PRE-REGISTERED VERDICT =====")
    passing = [s for s in SCALES if g1_pass.get(s) and len(Dstar.get(s, [])) >= 2]
    if not g2 or len(passing) < 2:
        emit("NO VERDICT — legality gate failed or fewer than two scales passed G1.")
        fh.close(); return
    s_lo, s_hi = min(passing), max(passing)
    n = min(len(Dstar[s_lo]), len(Dstar[s_hi]))
    delta = np.array(Dstar[s_lo][:n]) - np.array(Dstar[s_hi][:n])   # paired by seed
    md, _, dlo, dhi = ci95_t(delta)
    ms, _, slo, shi = ci95_t(np.array(Dstar[s_lo]))
    emit(f"paired D*({s_lo:.2f}) - D*({s_hi:.2f}) = {md:+.4f} [{dlo:+.4f},{dhi:+.4f}] "
         f"(p={paired_t_p(delta):.4f}, n={n}); D*({s_lo:.2f}) = {ms:+.4f} [{slo:+.4f},{shi:+.4f}]")
    if dlo > 0 and slo > 0:
        emit("CONFIRMED-DIE-SIZE: shrinking the die at a fixed workload and operating "
             "point grows the payoff — geometry (spreading-length filtering) is a "
             "real driver of D*, not just workload structure.")
    else:
        emit("NULL: the paired cross-scale difference does not certify a die-size "
             "effect; the observed magnitude spread must be attributed to workload "
             "structure until a larger sweep says otherwise.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
