"""exp025: does risk-aware placement pay on a real-silicon-anchored SoC?

exp023 confirmed the mechanism exists (+0.068 K) on the INVENTED hetero-SoC.
Kraken is the same regime exhibited by real silicon: a multi-sensor nano-UAV
SoC whose subsystems are duty-cycled by the application (event-driven SNE vs
frame-driven CUTIE vs cluster DSP), with per-subsystem powers MEASURED on
chip (workloads/kraken_soc.py carries the provenance and the ASSUMED items:
non-CUTIE areas, sub-block power splits, duty cycles). Scale is normalized to
a target peak dT (exp021 BOOM protocol) — only cross-mode/cross-block
structure is claimed.

Every evaluation trap is controlled (the exp023 template):
  budget (exp013):   all arms 120 iterations;
  grid (exp018/020): raster_jitter=1.0 training [exp022]; ALL eval at 64x64;
  baseline (exp019): mean arm is best-of-3 restarts, training-objective pick.

Design:
  GATE (validity, enforced): across the 6 measured modes the hotspot at the
      initial tiling must span >= 3 distinct blocks; else no verdict.
  E1 (existence): oracle D* = CVaR(mean-oracle) - CVaR(cvar-oracle),
      N_ORACLE=1000, 5 independent pairs, paired CI.
  E2 (learnability): mean-strong vs cvar vs blend(0.75) at N in {32, 128},
      5 seeds, dCVaR/dMean vs the STRONG mean baseline.
PRE-REGISTERED READINGS (mirror exp023):
  - PAYS if E1 D* CI > 0 AND some E2 cell has dCVaR CI > 0 with dMean <= 0:
    the exp023 existence result extends to a measurement-anchored model of
    real silicon (still NOT a real-traces result — scope stays one rung down).
  - DOES NOT PAY if E1 CI <= 0 — the favorable-regime result does not extend
    to this real mode structure; report which regime stat differs from
    hetero-SoC (candidate: the 373 mW fusion mode co-activates everything,
    compressing hotspot mobility).
  - MIXED: report both quantities, no slogan.
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
from pyrova.workloads.kraken_soc import KrakenWorkloadModel, kraken_units

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
TARGET_PEAK = 40.0   # exp021 BOOM protocol: probability-weighted mean peak dT


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
                    raster_jitter=JITTER, jitter_seed=930_000 + seed + 37 * r)
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


def main():
    units = kraken_units()
    cfg = parse_config(str(CONFIG))
    chip_w, chip_h = chip_box(units)
    ambient = cfg["ambient"]
    solver = GridFDSolver(cfg, units, chip_w, chip_h, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()

    out = PKG / "results/exp025_kraken_soc.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    # Scale calibration (exp021 protocol): weighted mean pure-mode peak = target.
    model0 = KrakenWorkloadModel(units, seed=0)

    def peaks_fn(scen):
        p = DiffPlacer(solver, units, chip_w, chip_h, TRAIN_GRID, TRAIN_GRID,
                       alpha=ALPHA)
        cx, cy = p.get_positions()
        return p._scenario_peaks(cx, cy, scen)

    k = model0.calibrate_scale(peaks_fn, TARGET_PEAK)
    st = model0.regime_stats()
    emit(f"exp025: Kraken measurement-anchored SoC ({len(units)} blocks, die "
         f"{chip_w*1e3:.1f}x{chip_h*1e3:.1f} mm). alpha={ALPHA}; "
         f"train@{TRAIN_GRID}+jitter={JITTER}; eval@{EVAL_GRID}; {N_ITER} it; "
         f"mean arm best-of-3; scale k={k:.1f} to weighted-mean peak "
         f"{TARGET_PEAK} K (BOOM protocol — relative structure unchanged).")
    emit(f"regime stats: E[total]={st['e_total_W']:.2f} (scaled)  "
         f"total-CV={st['total_cv']:.3f}  "
         f"heaviest-engine share={100*st['heaviest_engine_share_mean']:.0f}%")
    emit("assumed (printed per SCOPE CONTRACT): non-CUTIE areas, sub-block power "
         "splits, mode duty cycles " + str(dict(
             event=0.30, frame=0.10, dronet_p=0.10, dronet_e=0.15,
             fusion=0.15, idle=0.20)))

    # Validity gate: measured modes must move the hotspot at the initial tiling.
    s64 = GridFDSolver(cfg, units, chip_w, chip_h, EVAL_GRID, EVAL_GRID)
    s64.build(); s64.factorize()
    argmax_by_mode = {}
    for mi, mname in enumerate(model0.mode_names):
        pw = model0.modes[mi] * model0.scale
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(units)}
        T = s64.solve(s64.build_rhs(bp))
        flat = int(np.argmax(s64.silicon_layer(T)))
        cell = (flat // EVAL_GRID, flat % EVAL_GRID)
        blk = min(units, key=lambda u: (u["leftx"] + u["width"] / 2 - (cell[1] + .5) * chip_w / EVAL_GRID) ** 2
                  + (u["bottomy"] + u["height"] / 2 - (EVAL_GRID - cell[0] - .5) * chip_h / EVAL_GRID) ** 2)
        argmax_by_mode[mname] = blk["name"]
        emit(f"  mode {mname:9s}: hotspot near {blk['name']}")
    n_distinct = len(set(argmax_by_mode.values()))
    emit(f"GATE: hotspot spans {n_distinct} distinct blocks across modes "
         f"(requires >= 3) -> {'PASS' if n_distinct >= 3 else 'FAIL'}")
    if n_distinct < 3:
        emit("GATE FAILED — the measured mode structure does not move the "
             "hotspot on this model; no verdict (this is itself the screen's "
             "answer for Kraken: condition 2 unmet).")
        fh.close()
        return

    def make_model(seed):
        m = KrakenWorkloadModel(units, seed=seed)
        m.scale = model0.scale
        return m

    # E1: oracle existence with all traps controlled.
    emit(f"\nE1: oracle D*, N_ORACLE={N_ORACLE}, {N_OR_PAIRS} pairs, "
         f"eval on common {N_TEST}-scenario holdout at {EVAL_GRID}^2:")
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
    e2 = {}
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
        e2[n] = {}
        for arm in ("cvar", "blend"):
            gc, _, lo, hi = ci95_t(dC[arm])
            gm, _, mlo, mhi = ci95_t(dM[arm])
            fc = "*" if lo > 0 else ("x" if hi < 0 else " ")
            emit(f"  N={n} {arm:5s}: dCVaR={gc:+.3f}{fc} [{lo:+.3f},{hi:+.3f}]  "
                 f"dMean={gm:+.3f} [{mlo:+.3f},{mhi:+.3f}]")
            e2[n][arm] = (gc, lo, gm, mhi)

    # Pre-registered verdict.
    trade = [(n, a) for n in N_TRAINS for a in ("cvar", "blend")
             if e2[n][a][1] > 0 and e2[n][a][2] <= 0]
    if D_lo > 0 and trade:
        v = (f"PAYS: on the measurement-anchored Kraken model, risk-aware "
             f"placement buys {Dm:+.3f} K CI[{D_lo:+.2f},{D_hi:+.2f}] of true "
             f"tail (oracle) and the trade is learnable at {trade} — extends "
             f"exp023's existence result to real-silicon-anchored structure "
             f"(scope: model, not traces).")
    elif D_hi <= 0:
        v = (f"DOES NOT PAY: D*={Dm:+.3f} [{D_lo:+.2f},{D_hi:+.2f}] — the "
             f"exp023 result does not extend to Kraken's measured mode "
             f"structure; report the regime-stat comparison before theorizing.")
    else:
        v = (f"MIXED: D*={Dm:+.3f} [{D_lo:+.2f},{D_hi:+.2f}]; learnable trade "
             f"cells: {trade or 'none'} — report both, no slogan.")
    emit(f"\nPRE-REGISTERED VERDICT: {v}")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
