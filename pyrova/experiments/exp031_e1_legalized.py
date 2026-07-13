"""Raw-vs-legalized oracle re-evaluation on two testbeds (hetero-SoC, Kraken):
reproduces recorded oracle placements exactly from pinned seeds (same fit
code, jitter and restart streams), then scores every placement at 64^2 both
raw and after overlap-only legalization (optimizer/legalize.py, target <0.1%
residual), reporting D* = trueCVaR(mean-arm) - trueCVaR(cvar-arm) under both;
the raw reproduction doubles as a fidelity check, and any placement that
fails to legalize below 0.1% excludes its pair.

Set PYROVA_SMOKE=1 for a tiny execution check.
Set PYROVA_TESTBED=hetero|kraken|both (default both) to split jobs.
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
from pyrova.optimizer.legalize import legalize_units, overlap_frac
from pyrova.evaluation.metrics import cvar, ci95_t
from pyrova.workloads.hetero_soc import HeteroSoCWorkloadModel, soc_units
from pyrova.workloads.kraken_soc import KrakenWorkloadModel, kraken_units

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA = 0.90
TRAIN_GRID = 18
EVAL_GRID = 32 if SMOKE else 64
N_ITER = 10 if SMOKE else 120
N_ORACLE = 40 if SMOKE else 1000
N_OR_PAIRS = 2 if SMOKE else 5
N_TEST = 100 if SMOKE else 1000
JITTER = 1.0
TARGET_PEAK = 40.0
LEGAL_TOL = 1e-3   # 0.1% of block area


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def fit(solver, units, cw, ch, train, mode, jitter_base, restarts=1,
        rrng=None, seed=0):
    best, best_obj = None, np.inf
    for r in range(restarts):
        pl = DiffPlacer(solver, units, cw, ch, TRAIN_GRID, TRAIN_GRID,
                        alpha=ALPHA, blend_gamma=0.75)
        if r > 0:
            pl.raw_x += rrng.standard_normal(pl.n) * 0.5
            pl.raw_y += rrng.standard_normal(pl.n) * 0.5
        pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=jitter_base + seed + 37 * r)
        obj = pl.objective_and_grad(train, mode=mode)[0]
        if obj < best_obj:
            best, best_obj = pl, obj
    return best


def eval_cvar(cfg, up, scen, cw, ch, ambient):
    s = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
    s.build(); s.factorize()
    pk = np.array([float(s.silicon_layer(s.solve(s.build_rhs(
        {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - ambient
        for pw in scen])
    return cvar(pk, ALPHA)


def run_testbed(name, units, make_model, jitter_base, cfg, emit):
    cw, ch = chip_box(units)
    ambient = cfg["ambient"]
    solver = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()
    test = make_model(777).sample(N_TEST)
    emit(f"\n===== {name}: {len(units)} blocks, E1 x {N_OR_PAIRS} pairs, "
         f"N_ORACLE={N_ORACLE}, eval@{EVAL_GRID} raw+legalized =====")
    D_raw, D_leg, excl = [], [], 0
    for k in range(N_OR_PAIRS):
        train = make_model(10_000 + k).sample(N_ORACLE)
        rrng = np.random.default_rng(20_000 + k)
        p_m = fit(solver, units, cw, ch, train, "mean", jitter_base,
                  restarts=3, rrng=rrng, seed=k)
        p_c = fit(solver, units, cw, ch, train, "cvar", jitter_base, seed=k)
        row = {}
        ok = True
        for arm, pl in (("mean", p_m), ("cvar", p_c)):
            up = pl.get_units()
            f_raw = overlap_frac(up)
            lu, f_leg = legalize_units(up, cw, ch, tol_frac=LEGAL_TOL)
            if f_leg > LEGAL_TOL:
                emit(f"  pair {k} {arm}: LEGALIZATION FAILED "
                     f"({100*f_raw:.3f}% -> {100*f_leg:.3f}%) — pair EXCLUDED")
                ok = False
            row[arm] = (eval_cvar(cfg, up, test, cw, ch, ambient),
                        eval_cvar(cfg, lu, test, cw, ch, ambient), f_raw, f_leg)
        if not ok:
            excl += 1
            continue
        D_raw.append(row["mean"][0] - row["cvar"][0])
        D_leg.append(row["mean"][1] - row["cvar"][1])
        emit(f"  pair {k}: D*_raw={D_raw[-1]:+.3f}  D*_legal={D_leg[-1]:+.3f}  "
             f"(overlap mean {100*row['mean'][2]:.2f}->{100*row['mean'][3]:.2f}%, "
             f"cvar {100*row['cvar'][2]:.2f}->{100*row['cvar'][3]:.2f}%)")
    if len(D_leg) < 3:
        emit(f"  {name}: too few legal pairs ({len(D_leg)}) — NO VERDICT.")
        return
    gr, _, rlo, rhi = ci95_t(D_raw)
    gl, _, llo, lhi = ci95_t(D_leg)
    emit(f"  D* raw   = {gr:+.3f} [{rlo:+.3f},{rhi:+.3f}]  (fidelity check vs recorded)")
    emit(f"  D* legal = {gl:+.3f} [{llo:+.3f},{lhi:+.3f}]  excluded pairs: {excl}")
    if llo > 0:
        emit(f"  READING: SURVIVES — {name} E1 is not overlap leakage; "
             f"drop the legality qualifier.")
    elif rlo > 0 >= llo:
        emit(f"  READING: OVERLAP ARTIFACT — raw positive, legal CI<=0; "
             f"trap 4 claims the {name} headline; re-scope now.")
    else:
        emit(f"  READING: INCONCLUSIVE at this power — report both CIs.")


def main():
    testbed = os.environ.get("PYROVA_TESTBED", "both")
    cfg = parse_config(str(CONFIG))
    suffix = "" if testbed == "both" else f"_{testbed}"
    out = PKG / f"results/exp031_e1_legalized{suffix}.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"exp031 (P11): E1 headlines under legalized evaluation. alpha={ALPHA}, "
         f"train@{TRAIN_GRID}+jitter, {N_ITER} it, legal tol {100*LEGAL_TOL:.1f}%, "
         f"testbed={testbed}. "
         + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    # hetero-SoC (pinned jitter base 910_000)
    if testbed in ("hetero", "both"):
        h_units = soc_units()
        run_testbed("hetero-SoC (exp023 E1)", h_units,
                    lambda s: HeteroSoCWorkloadModel(h_units, seed=s),
                    910_000, cfg, emit)
    if testbed == "hetero":
        fh.close()
        print(f"\nWrote {out.relative_to(ROOT)}")
        return

    # Kraken (pinned jitter base 930_000; scale calibrated as in the recorded run)
    k_units = kraken_units()
    kcw, kch = chip_box(k_units)
    ksolver = GridFDSolver(cfg, k_units, kcw, kch, TRAIN_GRID, TRAIN_GRID)
    ksolver.build(); ksolver.factorize()
    model0 = KrakenWorkloadModel(k_units, seed=0)

    def peaks_fn(scen):
        p = DiffPlacer(ksolver, k_units, kcw, kch, TRAIN_GRID, TRAIN_GRID, alpha=ALPHA)
        cx, cy = p.get_positions()
        return p._scenario_peaks(cx, cy, scen)

    model0.calibrate_scale(peaks_fn, TARGET_PEAK)

    def make_kraken(seed):
        m = KrakenWorkloadModel(k_units, seed=seed)
        m.scale = model0.scale
        return m

    run_testbed("Kraken (exp025 E1)", k_units, make_kraken, 930_000, cfg, emit)
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
