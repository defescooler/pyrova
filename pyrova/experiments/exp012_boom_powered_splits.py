"""Powered BOOM split design: paired arms gamma in {0 (mean-opt), 0.75 (blend),
1 (pure CVaR)} over 20 repeated disjoint 60/20 splits of the 80-program pool
(needs BOOM_DATA), 30 iterations at grid 18x18, alpha=0.9; per gamma, OOS
mean/CVaR and dCVaR/dMean vs mean-opt with Nadeau-Bengio-corrected CIs.
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
from pyrova.evaluation.metrics import mean_cvar, ci95_nadeau_bengio
from pyrova.workloads.boom_traces import BoomWorkload, resolve_paths

CONFIG = PKG / "inputs/configs/thermal.config"
CONFIG_ID = "0"
ALPHA = 0.90
NR = NC = 18
N_ITER = 30
N_SPLITS = 20
N_TRAIN = 60            # 60/20 split
GAMMAS = [0.0, 0.75, 1.0]
TARGET_PEAK = 40.0


def fit_gamma(solver, wl, train, gamma):
    mode = "mean" if gamma == 0.0 else ("cvar" if gamma == 1.0 else "blend")
    pl = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, NR, NC,
                    alpha=ALPHA, blend_gamma=gamma)
    pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False)
    return pl


def main():
    csvp, rptp = resolve_paths()
    if not csvp:
        print("BOOM_DATA not found; see workloads/boom_traces.py.")
        return

    out = PKG / "results/exp012_boom_powered_splits.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s); fh.write(s + "\n")

    cfg = parse_config(str(CONFIG))
    wl = BoomWorkload(csvp, rptp, config_id=CONFIG_ID)
    solver = GridFDSolver(cfg, wl.units, wl.chip_w, wl.chip_h, NR, NC)
    solver.build(); solver.factorize()

    def peaks_fn(scen):
        p = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, NR, NC, alpha=ALPHA)
        cx, cy = p.get_positions()
        return p._scenario_peaks(cx, cy, scen)
    wl.scale_to_peak(peaks_fn, TARGET_PEAK)
    scen = wl.scenarios()
    n = len(scen)
    n_test = n - N_TRAIN
    ratio = n_test / N_TRAIN

    emit(f"exp012: powered BOOM split design. {n} programs, {N_TRAIN}/{n_test} x "
         f"{N_SPLITS} splits, alpha={ALPHA}, grid {NR}x{NC}, {N_ITER} iter, "
         f"gammas={GAMMAS}.")
    emit(f"NB half-width factor sqrt(1/J + n_te/n_tr) = "
         f"{np.sqrt(1.0/N_SPLITS + ratio):.2f} sd (exp009's 40/40x10 was "
         f"{np.sqrt(0.1 + 1.0):.2f} sd -> ~{np.sqrt(1.1)/np.sqrt(1.0/N_SPLITS + ratio):.1f}x tighter).")
    emit(f"CAVEAT: test tail at alpha={ALPHA} is ~{max(1, int(n_test * (1 - ALPHA)))} "
         f"programs per split — noisier per-split deltas, unbiased, averaged over J={N_SPLITS}.")

    means = {g: [] for g in GAMMAS}
    cvars = {g: [] for g in GAMMAS}
    for seed in range(N_SPLITS):
        # Fresh split-RNG stream family (seed offset 40_000), disjoint from
        # the other BOOM split designs.
        perm = np.random.default_rng(40_000 + seed).permutation(n)
        tr = [scen[i] for i in perm[:N_TRAIN]]
        te = [scen[i] for i in perm[N_TRAIN:]]
        for g in GAMMAS:
            pl = fit_gamma(solver, wl, tr, g)
            cx, cy = pl.get_positions()
            m, c = mean_cvar(pl._scenario_peaks(cx, cy, te), ALPHA)
            means[g].append(m); cvars[g].append(c)
    means = {g: np.asarray(v) for g, v in means.items()}
    cvars = {g: np.asarray(v) for g, v in cvars.items()}

    emit(f"\n  {'gamma':>6}{'OOSmean':>9}{'OOSCVaR':>9}  {'dCVaR vs mean-opt':>24}  {'dMean vs mean-opt':>24}")
    verdict_075 = None
    for g in GAMMAS:
        dc = cvars[0.0] - cvars[g]
        dm = means[0.0] - means[g]
        gc, _, lo, hi = ci95_nadeau_bengio(dc, ratio)
        gm, _, mlo, mhi = ci95_nadeau_bengio(dm, ratio)
        fc = "*" if lo > 0 else ("x" if hi < 0 else " ")
        s_c = "        -       " if g == 0.0 else f"{gc:+.3f}{fc} [{lo:+.2f},{hi:+.2f}]"
        s_m = "        -       " if g == 0.0 else f"{gm:+.3f}  [{mlo:+.2f},{mhi:+.2f}]"
        emit(f"  {g:>6.2f}{means[g].mean():>9.3f}{cvars[g].mean():>9.3f}  {s_c:>24}  {s_m:>24}")
        if g == 0.75:
            verdict_075 = (gc, lo, hi)

    gc, lo, hi = verdict_075
    if lo > 0:
        v = (f"CONFIRMED: blend gamma=0.75 beats mean-opt on OOS CVaR with the NB CI "
             f"strictly positive ({gc:+.3f} [{lo:+.2f},{hi:+.2f}]) — first real-workload win.")
    elif gc > 0:
        v = (f"WEAK SUPPORT: point estimate positive ({gc:+.3f}) but CI [{lo:+.2f},{hi:+.2f}] "
             f"spans 0 even at the tighter design — more real PROGRAMS (not splits) is the "
             f"only remaining lever on this dataset.")
    else:
        v = (f"KILLED: the exp010 point estimate did not replicate ({gc:+.3f} "
             f"[{lo:+.2f},{hi:+.2f}]); treat the BOOM blend positive as split noise.")
    emit(f"\nPRE-REGISTERED VERDICT: {v}")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
