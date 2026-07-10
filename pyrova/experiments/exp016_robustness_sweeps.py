"""exp016: sign-stability of the two confirmed positives across the untested knobs.

Every verdict in the suite was produced at one grid resolution (18x18), one
Adam learning rate (2e-2), and (for BOOM) one synthesised geometry (ncol=6
squares). Three sweeps check that the surviving positive results are not
artifacts of those choices:

  G. GRID: exp005's strongest Holm survivor (structured, N_TRAIN=128,
     alpha=0.95) re-run at grids {18, 24, 30}. Discretisation is the cheapest
     remaining way the structured positive could be an artifact (cell size ~
     small-block size at 18x18).
  L. LEARNING RATE: exp015-A's blend-vs-mean comparison (BOOM, 120 iters,
     gamma in {0, 0.75}) at lr {5e-3, 2e-2}. Diagnoses the open "domination"
     mechanism: if mean-opt stops losing on its own metric at the smaller lr,
     the blend win at lr=2e-2 is an Adam-oscillation artifact of the smooth
     objective; if domination persists across lr, the blend objective's
     advantage is real optimisation behaviour.
  Y. GEOMETRY: the same BOOM comparison at synthesised layouts ncol {4, 8}
     (6 is covered by sweep L at lr=2e-2). The blend win must not be a
     property of one gridded-squares layout.

PRE-REGISTERED READINGS:
  G: STABLE if dCVaR (mean-opt minus cvar-opt) > 0 at every grid; the claim
     is grid-robust. Any sign flip -> the structured positive is
     discretisation-contingent and must be requalified.
  L: domination (dMean CI>0 for gamma=0.75) at BOTH lr -> real objective
     behaviour, mechanism still open; domination only at lr=2e-2 ->
     lr-artifact, requalify exp015-A accordingly.
  Y: blend dCVaR>0 (point estimate) in at least 2 of 3 geometries and CI>0 in
     at least 1 -> geometry-robust; all-ns or sign flips -> geometry-
     contingent, say so in CLAUDE.md.
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
ALPHA_G = 0.95           # exp005 survivor cell
N_TRAIN_G = 128
GRIDS = [18, 24, 30]
N_SEEDS_G = 5
N_TEST = 1500
N_ITER_G = 30            # exp005's budget, for comparability with the original cell

ALPHA_B = 0.90
N_ITER_B = 120           # exp015's matched budget
LRS = [5e-3, 2e-2]
NCOLS = [4, 8]           # 6 == exp015 baseline, covered by sweep L at lr=2e-2
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
    """Blend gamma=0.75 vs mean-opt on BOOM at the exp015 protocol, with
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
        perm = np.random.default_rng(40_000 + seed).permutation(n)   # exp012/exp015 splits
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

    # G: grid-resolution stability of the exp005 survivor cell
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
    emit(f"  G VERDICT: {'STABLE — dCVaR>0 at every grid; the structured positive is grid-robust.' if g_stable else 'SIGN FLIP across grids — the structured positive is discretisation-contingent; requalify exp005 in CLAUDE.md.'}")

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
             f"layouts (CI>0 in {sig}); scope the blend claim to the tested geometry "
             f"in CLAUDE.md.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
