"""Three sign-stability sweeps. G: structured-family mean-opt vs cvar-opt
(N_TRAIN=128, alpha=0.95, 30 iterations, 5 seeds, 1500-scenario holdout) at
grids {18, 24, 30}, paired t-CIs. L: BOOM blend gamma=0.75 vs mean (120
iterations, 20 repeated 60/20 splits, Nadeau-Bengio CIs) at lr {5e-3, 2e-2}.
Y: the same BOOM comparison at synthesised layouts ncol {4, 8} (ncol=6
covered by sweep L at lr=2e-2).
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_flp, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import mean_cvar, ci95_t, ci95_nadeau_bengio
from pyrova.workloads.structured import StructuredWorkloadModel
from pyrova.workloads.boom_traces import BoomWorkload, resolve_paths

FLP = PKG / "inputs/floorplans/ev6.flp"
CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA_G = 0.95
N_TRAIN_G = 128
GRIDS = [18, 24, 30]
N_SEEDS_G = 5
N_TEST = 1500
N_ITER_G = 30

ALPHA_B = 0.90
N_ITER_B = 120
LRS = [5e-3, 2e-2]
NCOLS = [4, 8]           # ncol=6 covered by sweep L at lr=2e-2
N_SPLITS = 20
N_TRAIN_B = 60
TARGET_PEAK = 40.0


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def oos(pl, scen, alpha):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), alpha)


def boom_arm(cfg, ncol: int, lr: float, emit, label: str):
    """Blend gamma=0.75 vs mean-opt on BOOM over repeated 60/20 splits, with
    configurable synthesised geometry (ncol) and learning rate."""
    csvp, rptp = resolve_paths()
    if not csvp:
        emit(f"  [{label}] SKIPPED: BOOM_DATA not found.")
        return None
    wl = BoomWorkload(csvp, rptp, config_id="0", ncol=ncol)
    solver = GridFDSolver(cfg, wl.units, wl.chip_w, wl.chip_h, 18, 18)
    solver.build(); solver.factorize()

    def peaks_fn(scen):
        p = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, 18, 18, alpha=ALPHA_B)
        cx, cy = p.get_positions()
        return p._scenario_peaks(cx, cy, scen)
    wl.scale_to_peak(peaks_fn, TARGET_PEAK)
    scen = wl.scenarios()
    n = len(scen); ratio = (n - N_TRAIN_B) / N_TRAIN_B

    dC, dM = [], []
    for seed in range(N_SPLITS):
        perm = np.random.default_rng(40_000 + seed).permutation(n)
        tr = [scen[i] for i in perm[:N_TRAIN_B]]
        te = [scen[i] for i in perm[N_TRAIN_B:]]
        res = {}
        for g, mode in ((0.0, "mean"), (0.75, "blend")):
            pl = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, 18, 18,
                            alpha=ALPHA_B, blend_gamma=g)
            pl.optimize(tr, mode=mode, n_iter=N_ITER_B, lr=lr, verbose=False)
            res[g] = oos(pl, te, ALPHA_B)
        dC.append(res[0.0][1] - res[0.75][1])
        dM.append(res[0.0][0] - res[0.75][0])
        print(f"  [{label}] split {seed+1}/{N_SPLITS}", flush=True)
    gc, _, lo, hi = ci95_nadeau_bengio(dC, ratio)
    gm, _, mlo, mhi = ci95_nadeau_bengio(dM, ratio)
    fc = "*" if lo > 0 else ("x" if hi < 0 else " ")
    fm = "*" if mlo > 0 else ("x" if mhi < 0 else " ")
    emit(f"  [{label}] dCVaR={gc:+.3f}{fc} [{lo:+.2f},{hi:+.2f}]  "
         f"dMean={gm:+.3f}{fm} [{mlo:+.2f},{mhi:+.2f}]")
    return dict(gc=gc, lo=lo, hi=hi, gm=gm, mlo=mlo)


def main():
    cfg = parse_config(str(CONFIG))
    out = PKG / "results/exp016_robustness_sweeps.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    # G: grid-resolution stability of the structured comparison
    units = parse_flp(str(FLP))
    chip_w, chip_h = chip_box(units)
    emit(f"exp016-G: exp005 survivor (structured, N={N_TRAIN_G}, alpha={ALPHA_G}) "
         f"at grids {GRIDS}, {N_SEEDS_G} seeds, {N_ITER_G} iter, N_TEST={N_TEST}.")
    g_rows = []
    for nr in GRIDS:
        solver = GridFDSolver(cfg, units, chip_w, chip_h, nr, nr)
        solver.build(); solver.factorize()
        dC, dM = [], []
        for seed in range(N_SEEDS_G):
            model = StructuredWorkloadModel(
                units, seed=100_000 * seed + 100 * N_TRAIN_G + int(round(ALPHA_G * 100)))
            train = model.sample(N_TRAIN_G)
            test = model.sample(N_TEST)
            res = {}
            for mode in ("mean", "cvar"):
                pl = DiffPlacer(solver, units, chip_w, chip_h, nr, nr, alpha=ALPHA_G)
                pl.optimize(train, mode=mode, n_iter=N_ITER_G, lr=2e-2, verbose=False)
                res[mode] = oos(pl, test, ALPHA_G)
            dC.append(res["mean"][1] - res["cvar"][1])
            dM.append(res["mean"][0] - res["cvar"][0])
            print(f"  [G grid={nr}] seed {seed+1}/{N_SEEDS_G}", flush=True)
        gc, _, lo, hi = ci95_t(dC)
        gm, _, _, _ = ci95_t(dM)
        fc = "*" if lo > 0 else ("x" if hi < 0 else " ")
        emit(f"  grid {nr}x{nr}: dCVaR={gc:+.3f}{fc} [{lo:+.3f},{hi:+.3f}]  dMean={gm:+.3f}")
        g_rows.append(gc)
    g_stable = all(v > 0 for v in g_rows)
    emit(f"  G VERDICT: {'STABLE — dCVaR>0 at every grid; the structured positive is grid-robust.' if g_stable else 'SIGN FLIP across grids — the structured positive is discretisation-contingent; requalify the exp005 claim.'}")

    # L: learning-rate dependence of the blend domination (BOOM)
    emit(f"\nexp016-L: BOOM blend gamma=0.75 vs mean, {N_ITER_B} iter, lr sweep {LRS} "
         f"(ncol=6, exp015 splits).")
    l_rows = {}
    for lr in LRS:
        l_rows[lr] = boom_arm(cfg, ncol=6, lr=lr, emit=emit, label=f"L lr={lr:g}")
    if all(l_rows.values()):
        dom = {lr: r["mlo"] > 0 for lr, r in l_rows.items()}
        if all(dom.values()):
            emit("  L VERDICT: domination at BOTH lr — the blend objective's advantage is "
                 "real optimisation behaviour, not an lr artifact; mechanism still open.")
        elif dom[2e-2] and not dom[5e-3]:
            emit("  L VERDICT: domination only at lr=2e-2 — mean-opt's loss on its own "
                 "metric is an Adam step-size artifact; requalify exp015-A: at the "
                 "smaller lr read the fresh dCVaR/dMean above as the honest comparison.")
        else:
            emit("  L VERDICT: mixed pattern — report both rows; no single-sentence claim.")

    # Y: geometry dependence of the blend win (BOOM)
    emit(f"\nexp016-Y: BOOM blend vs mean at synthesised layouts ncol {NCOLS} "
         f"(plus ncol=6 from sweep L at lr=2e-2), {N_ITER_B} iter, lr=2e-2.")
    y_rows = [l_rows.get(2e-2)] if l_rows.get(2e-2) else []
    for ncol in NCOLS:
        y_rows.append(boom_arm(cfg, ncol=ncol, lr=2e-2, emit=emit, label=f"Y ncol={ncol}"))
    y_rows = [r for r in y_rows if r]
    pos = sum(1 for r in y_rows if r["gc"] > 0)
    sig = sum(1 for r in y_rows if r["lo"] > 0)
    if pos >= 2 and sig >= 1:
        emit(f"  Y VERDICT: geometry-robust — dCVaR>0 in {pos}/{len(y_rows)} layouts, "
             f"CI>0 in {sig}.")
    else:
        emit(f"  Y VERDICT: geometry-contingent — dCVaR>0 in only {pos}/{len(y_rows)} "
             f"layouts (CI>0 in {sig}); scope the blend claim to the tested geometry.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
