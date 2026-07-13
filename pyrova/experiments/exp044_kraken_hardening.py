"""Oracle gap on the measurement-anchored Kraken small-SoC model: D* =
trueCVaR(mean-arm) - trueCVaR(cvar-arm) at alpha=0.9 over 20 independent
oracle pairs (N_ORACLE=2000), symmetric single-start arms at matched budget
with common random numbers within a pair, train@18 with raster jitter,
density-spread fitting, guaranteed-legal evaluation on an independent 64^2
grid, total power calibrated to a 40 K mean-peak operating point; the (mean,
CVaR) pair is reported side by side. Fail-closed gates: >=3 distinct hotspot
blocks over the pure modes on the tiled layout, guaranteed legality.

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
from pyrova.workloads.kraken_soc import kraken_units, KrakenWorkloadModel

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
EVAL_ALPHA = 0.90
TARGET_PEAK = 40.0
TRAIN_GRID = 18
JITTER = 1.0
DENSITY_W = 5.0e2
DENSITY_LAM0 = 3.0e1
DENSITY_GRID = (48, 48)

EVAL_GRID = 32 if SMOKE else 64
N_ITER = 12 if SMOKE else 100
N_ORACLE = 100 if SMOKE else 2000
N_PAIRS = 2 if SMOKE else 20
N_TEST = 200 if SMOKE else 2000
N_CALIB = 50 if SMOKE else 300


def block_at_peak(units, cw, ch, nx, ny, T):
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
    units = kraken_units()
    cw = max(u["leftx"] + u["width"] for u in units)
    ch = max(u["bottomy"] + u["height"] for u in units)

    out = PKG / "results/exp044_kraken_hardening.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"kraken headline hardening: {len(units)} blocks, die {cw*1e3:.1f}x{ch*1e3:.1f}mm. "
         f"{N_PAIRS} independent oracle pairs (was 5), N_ORACLE={N_ORACLE}, {N_ITER} it, "
         f"symmetric single-start + CRN, train@{TRAIN_GRID}+jitter -> guaranteed-legal "
         f"eval@{EVAL_GRID} (alpha={EVAL_ALPHA}). "
         + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    # Calibrate once on the tiled layout (dT linear in power); all pair models
    # share the calibrated scale so every pair sees the same distribution.
    se = GridFDSolver(cfg, units, cw, ch, EVAL_GRID, EVAL_GRID)
    se.build(); se.factorize()

    def peak(solver, up, pw):
        return float(solver.silicon_layer(solver.solve(solver.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - ambient

    calib = KrakenWorkloadModel(units, seed=1)
    raw = calib.sample(N_CALIB)
    m0 = float(np.mean([peak(se, units, pw) for pw in raw]))
    c = TARGET_PEAK / m0
    emit(f"calibration: tiled mean peak {m0:.1f}K -> power x{c:.3f} for a "
         f"{TARGET_PEAK:.0f}K operating point")

    def make_scaled(seed):
        m = KrakenWorkloadModel(units, seed=seed)
        m.scale = m.scale * c
        return m

    hot = {block_at_peak(units, cw, ch, EVAL_GRID, EVAL_GRID,
                         se.silicon_layer(se.solve(se.build_rhs(
                             {u["name"]: float(v) for u, v in zip(units, mv * c)}))))
           for mv in calib.modes}
    g0 = len(hot) >= 3
    emit(f"G0 (mechanism): hotspot blocks over modes: {len(hot)} {sorted(hot)} -> "
         f"{'PASS' if g0 else 'FAIL'}")
    if not g0:
        emit("\n===== PRE-REGISTERED VERDICT =====")
        emit("NO VERDICT — mechanism gate failed."); fh.close(); return

    test = make_scaled(99).sample(N_TEST)
    st = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    st.build(); st.factorize()

    def eval_cvar_mean(up):
        sv = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
        sv.build(); sv.factorize()
        pk = np.array([peak(sv, up, pw) for pw in test])
        return cvar(pk, EVAL_ALPHA), float(pk.mean())

    def fit(tr, mode, jseed):
        pl = DiffPlacer(st, units, cw, ch, TRAIN_GRID, TRAIN_GRID, alpha=EVAL_ALPHA,
                        nonoverlap_w=1e4, density_w=DENSITY_W,
                        density_lam0=DENSITY_LAM0, density_grid=DENSITY_GRID)
        pl.optimize(tr, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=jseed)
        up, of = legalize_units_exact(pl.get_units(), cw, ch)
        return up, of

    dC, dM = [], []
    ovl_max = 0.0
    infeasible = 0
    for k in range(N_PAIRS):
        tr = make_scaled(10_000 + k).sample(N_ORACLE)
        js = 500_000 + 1000 * k                          # CRN across arms
        try:
            um, fm = fit(tr, "mean", js)
            uc, fc = fit(tr, "cvar", js)
        except LegalizationInfeasible:
            infeasible += 1
            emit(f"  pair {k+1}: INFEASIBLE, skipped")
            continue
        ovl_max = max(ovl_max, fm, fc)
        c_m, m_m = eval_cvar_mean(um)
        c_c, m_c = eval_cvar_mean(uc)
        dC.append(c_m - c_c)
        dM.append(m_m - m_c)
        print(f"  pair {k+1}/{N_PAIRS}: dCVaR={dC[-1]:+.3f} dMean={dM[-1]:+.3f}", flush=True)

    g2 = ovl_max < 1e-3 and infeasible == 0
    emit(f"\nG2 (legality): max overlap {100*ovl_max:.3f}%, {infeasible} infeasible -> "
         f"{'PASS' if g2 else 'FAIL'}")

    emit("\n===== PRE-REGISTERED VERDICT =====")
    if not g2 or len(dC) < 2:
        emit("NO VERDICT — legality gate failed or too few feasible pairs.")
        fh.close(); return
    d = np.array(dC)
    m, _, lo, hi = ci95_t(d)
    mm, _, mlo, mhi = ci95_t(np.array(dM))
    shape = "trade (mean-for-tail)" if mm <= 0 else "domination (check optimizer)"
    emit(f"D* = {m:+.4f} K CI95[{lo:+.4f},{hi:+.4f}] (p={paired_t_p(d):.5f}, "
         f"n={len(d)} pairs; was 5-pair [+0.94,+2.66])")
    emit(f"dMean = {mm:+.4f} [{mlo:+.4f},{mhi:+.4f}] -> {shape}")
    if lo > 0:
        emit("HARDENED: the headline survives at 4x the pair count; quote THIS CI.")
    else:
        emit("NOT HARDENED: the 20-pair CI includes 0 — the 5-pair CI was "
             "optimistic; the headline must be requalified accordingly.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
