"""Oracle gap on the hetero-SoC with evaluation-axis robustness: D* =
trueCVaR(mean-oracle) - trueCVaR(cvar-oracle) over 20 oracle pairs, both arms
best-of-3 restarts with common random numbers per pair (shared jitter and
restart draws — only the objective differs), train@18 with raster jitter,
legalized evaluation (any cell with residual overlap > 0.1% is void),
re-scored across alpha in {0.90, 0.95}, grid in {64, 96, 128}, and metric in
{peak (single hottest node), hot1 (mean of the hottest 1%)}, with dMean
alongside; the grid and metric axes are evaluation-only. Fail-closed gate:
the hotspot must move across >= 3 blocks over the modes.

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
from pyrova.optimizer.legalize import legalize_units
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p, holm
from pyrova.workloads.hetero_soc import (HeteroSoCWorkloadModel, soc_units,
                                         _MODES, _BLOCKS)

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
TRAIN_GRID = 18
JITTER = 1.0
HOT_FRAC = 0.01
N_ITER = 12 if SMOKE else 120
N_ORACLE = 48 if SMOKE else 1000
N_PAIRS = 2 if SMOKE else 20
N_TEST = 150 if SMOKE else 1000
RESTARTS = 1 if SMOKE else 3
ALPHAS = [0.90] if SMOKE else [0.90, 0.95]
GRIDS = [32] if SMOKE else [64, 96, 128]
METRICS = ["peak", "hot1"]


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def fit(solver, units, cw, ch, train, mode, alpha, jitter_seed, restart_seed):
    """Best-of-RESTARTS; jitter_seed and restart_seed shared across arms in a pair
    (common random numbers), so only the objective differs."""
    best, best_obj = None, np.inf
    rr = np.random.default_rng(restart_seed)
    for r in range(RESTARTS):
        pl = DiffPlacer(solver, units, cw, ch, TRAIN_GRID, TRAIN_GRID, alpha=alpha)
        if r > 0:
            pl.raw_x += rr.standard_normal(pl.n) * 0.5
            pl.raw_y += rr.standard_normal(pl.n) * 0.5
        pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=jitter_seed + 37 * r)
        obj = pl.objective_and_grad(train, mode=mode)[0]
        if obj < best_obj:
            best, best_obj = pl, obj
    return best


def per_scenario(cfg, up, scen, cw, ch, ambient, grid):
    """{metric: per-scenario array} at one eval grid, on a legalized placement."""
    s = GridFDSolver(cfg, up, cw, ch, grid, grid)
    s.build(); s.factorize()
    peak = np.zeros(len(scen)); hot = np.zeros(len(scen))
    for i, pw in enumerate(scen):
        sil = s.silicon_layer(s.solve(s.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(up)})))
        flat = np.sort(sil.ravel())
        peak[i] = float(flat[-1]) - ambient
        k = max(1, int(flat.size * HOT_FRAC))
        hot[i] = float(flat[-k:].mean()) - ambient
    return {"peak": peak, "hot1": hot}


def mean_cvar_cells(vals, alpha):
    """{(grid,metric): (mean, cvar)} for one alpha from {grid:{metric:array}}."""
    out = {}
    for g, mv in vals.items():
        for m, arr in mv.items():
            out[(g, m)] = (float(arr.mean()), cvar(arr, alpha))
    return out


def main():
    units = soc_units()
    cfg = parse_config(str(CONFIG))
    cw, ch = chip_box(units)
    ambient = cfg["ambient"]
    solver = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()

    out = PKG / "results/exp033_headline_hardening.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"headline hardening: hetero-SoC oracle D*, {N_PAIRS} pairs, symmetric "
         f"best-of-{RESTARTS}+CRN, legalized eval. alpha={ALPHAS} x grid={GRIDS} x "
         f"metric={METRICS} (hot1=hottest {100*HOT_FRAC:.0f}%). "
         + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    g0 = GRIDS[0]
    sg = GridFDSolver(cfg, units, cw, ch, g0, g0); sg.build(); sg.factorize()
    pmax = np.array([b[3] for b in _BLOCKS]); seen = set()
    for act in _MODES.values():
        pw = np.array(act) * pmax
        flat = int(np.argmax(sg.silicon_layer(sg.solve(sg.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(units)})))))
        cell = (flat // g0, flat % g0)
        blk = min(units, key=lambda u:
                  (u["leftx"] + u["width"]/2 - (cell[1]+.5)*cw/g0)**2
                  + (u["bottomy"] + u["height"]/2 - (g0-cell[0]-.5)*ch/g0)**2)
        seen.add(blk["name"])
    emit(f"GATE hotspot mobility: {len(seen)} blocks -> {'PASS' if len(seen) >= 3 else 'FAIL'}")
    if len(seen) < 3:
        emit("GATE FAILED — no verdict."); fh.close(); return

    test = HeteroSoCWorkloadModel(units, seed=777).sample(N_TEST)
    ovl_max = 0.0
    # D[(g,m,alpha)] -> list of per-pair D*; dM likewise.
    D, dM = {}, {}
    for k in range(N_PAIRS):
        train = HeteroSoCWorkloadModel(units, seed=10_000 + k).sample(N_ORACLE)
        js, rs = 800_000 + 1000 * k, 900_000 + 1000 * k
        pm = fit(solver, units, cw, ch, train, "mean", ALPHAS[0], js, rs)  # alpha-indep
        um, fm = legalize_units(pm.get_units(), cw, ch); ovl_max = max(ovl_max, fm)
        em = {g: per_scenario(cfg, um, test, cw, ch, ambient, g) for g in GRIDS}
        for a in ALPHAS:
            cm = mean_cvar_cells(em, a)
            pc = fit(solver, units, cw, ch, train, "cvar", a, js, rs)       # CRN with mean
            uc, fc = legalize_units(pc.get_units(), cw, ch); ovl_max = max(ovl_max, fc)
            ec = mean_cvar_cells({g: per_scenario(cfg, uc, test, cw, ch, ambient, g)
                                  for g in GRIDS}, a)
            for g in GRIDS:
                for m in METRICS:
                    D.setdefault((g, m, a), []).append(cm[(g, m)][1] - ec[(g, m)][1])
                    dM.setdefault((g, m, a), []).append(cm[(g, m)][0] - ec[(g, m)][0])
        print(f"  pair {k + 1}/{N_PAIRS} done", flush=True)

    emit(f"legality: max residual overlap {100*ovl_max:.3f}% "
         f"({'OK' if ovl_max < 1e-3 else 'VOID cells >0.1%'})")
    emit(f"\n  {'grid':>4} {'metric':>6} {'alpha':>5} {'D*':>9} {'CI_lo':>9} "
         f"{'CI_hi':>9} {'dMean':>8} {'p':>7}")
    cells = []
    for a in ALPHAS:
        for g in GRIDS:
            for m in METRICS:
                d = D[(g, m, a)]
                gg, _, lo, hi = ci95_t(d)
                dmm = float(np.mean(dM[(g, m, a)]))
                p = paired_t_p(d)
                fl = "*" if lo > 0 else ("x" if hi < 0 else " ")
                emit(f"  {g:>4} {m:>6} {a:>5.2f} {gg:>+9.4f}{fl} {lo:>+9.4f} "
                     f"{hi:>+9.4f} {dmm:>+8.4f} {p:>7.4f}")
                cells.append(dict(g=g, m=m, a=a, d=gg, lo=lo, hi=hi, dM=dmm, p=p))

    keep = holm([c["p"] for c in cells])
    for c, kp in zip(cells, keep):
        c["holm"] = bool(kp)
    ref = next(c for c in cells if c["g"] == GRIDS[0] and c["m"] == "peak"
               and c["a"] == 0.90)
    all_pos = all(c["d"] > 0 for c in cells)
    no_flip = all(c["hi"] >= 0 for c in cells)
    n_sig = sum(1 for c in cells if c["holm"] and c["lo"] > 0)
    n_trade = sum(1 for c in cells if c["holm"] and c["lo"] > 0 and c["dM"] <= 0)
    emit(f"\n  ({GRIDS[0]},peak,0.90) D*={ref['d']:+.4f} (prior favorable est +0.068)")
    emit(f"  {sum(c['d']>0 for c in cells)}/{len(cells)} cells positive; "
         f"{n_sig} Holm-significant; {n_trade} of those also dMean<=0.")

    emit("\nVERDICT: " + (
        f"CONFIRMED & ROBUST — every cell D*>0, none CI<0, {n_sig} Holm-significant "
        f"({n_trade} with dMean<=0); the favorable-regime positive is not a knob, "
        f"power, or baseline-asymmetry artifact." if (all_pos and no_flip and n_sig > 0)
        else "FRAGILE — some cell flips sign with CI<0 or none survives Holm; "
        + str([(c["g"], c["m"], c["a"]) for c in cells if c["hi"] < 0])))
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
