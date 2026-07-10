"""exp014: the ONE well-posed DRO retrial (run only after exp013's verdict).

Implements every correction from DRO_DERIVATION.md "Answers" (self-review,
review delegated):
  1. CERTIFIED penalty: global Lambda over all Si nodes (upper-bounds the
     worst-case CVaR), analytic gradient — placer mode='dro_exact'.
  2. MAHALANOBIS ground metric: D = diag(per-block power std of the TRAINING
     scenarios), so eps is in units of transport-sigmas and the adversary's
     budget reflects how blocks actually vary (the only change with a
     mechanism to break the penalty's collinearity with CVaR).
  3. FRACTIONAL-TAIL estimator in training (uniform weights -> exact nominal
     alpha), removing the effective-alpha inconsistency.
  4. eps CALIBRATED per seed by 2-fold cross-validation on the training set
     (Esfahani-Kuhn Sec 7.2 practice), grid in sigma-units.

PRE-REGISTERED READING:
  - DRO earns a stay iff structured vs_cvar0 (CVaR(pure-CVaR) - CVaR(dro_exact))
    has paired CI > 0 at N=32 or N=64 while the CV procedure picks eps > 0.
  - If CV collapses to the smallest eps and vs_cvar0 ~ 0: the corrected penalty
    is inert -> close DRO permanently (the approximate-penalty caveat no longer
    shields it).
  - vs_cvar0 CI < 0 anywhere: close DRO permanently.
BOOM arm (exp012's powered 60/20 x 20 design) is reported for completeness with
the same reading; prior is null (mechanism: FP thermally light).

NOTE on the exp013 optimizer confound: the PRIMARY criterion here (dro_exact
vs pure CVaR) compares two tail-driven arms at the same budget, so the
convergence-speed confound largely cancels within the comparison. The vs_mean
column IS exposed to it and is reported as context only.
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
ALPHA = 0.90
NR = NC = 18
N_ITER = 30
N_SEEDS = 5
N_TEST = 1500
N_SMALLS = [32, 64]
EPS_GRID = [0.1, 0.25, 0.5, 1.0]     # sigma-units (Mahalanobis metric)
BOOM_SPLITS = 20
BOOM_TRAIN = 60
TARGET_PEAK = 40.0


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def fit(solver, units, chip_w, chip_h, train, mode, eps=0.0, sigma=None):
    pl = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=ALPHA,
                    eps_dro=eps, dro_sigma=sigma)
    w = np.full(len(train), 1.0 / len(train))      # fractional-tail (exact alpha)
    pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                weights=None if mode == "mean" else w)
    return pl


def oos(pl, scen):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), ALPHA)


def cv_pick_eps(solver, units, chip_w, chip_h, train, sigma, rng):
    """2-fold CV on the training set: pick eps minimising held-out CVaR."""
    idx = rng.permutation(len(train))
    folds = [[train[i] for i in idx[::2]], [train[i] for i in idx[1::2]]]
    best_eps, best = 0.0, np.inf
    for eps in EPS_GRID:
        score = 0.0
        for a, b in ((0, 1), (1, 0)):
            pl = fit(solver, units, chip_w, chip_h, folds[a], "dro_exact",
                     eps=eps, sigma=sigma)
            score += oos(pl, folds[b])[1]
        if score < best:
            best, best_eps = score, eps
    return best_eps


def run_structured(solver, units, chip_w, chip_h, emit):
    test = StructuredWorkloadModel(units, seed=777).sample(N_TEST)
    for n in N_SMALLS:
        rows = {"mean": [], "cvar": [], "dro_exact": []}
        eps_picked = []
        for seed in range(N_SEEDS):
            print(f"  [structured N={n}] seed {seed+1}/{N_SEEDS} ...", flush=True)
            model = StructuredWorkloadModel(units, seed=200_000 * (seed + 1) + n)
            train = model.sample(n)
            sigma = np.asarray(train).std(axis=0, ddof=1) + 1e-12
            eps = cv_pick_eps(solver, units, chip_w, chip_h, train, sigma,
                              np.random.default_rng(300_000 + seed))
            eps_picked.append(eps)
            rows["mean"].append(oos(fit(solver, units, chip_w, chip_h, train, "mean"), test))
            rows["cvar"].append(oos(fit(solver, units, chip_w, chip_h, train, "cvar"), test))
            rows["dro_exact"].append(
                oos(fit(solver, units, chip_w, chip_h, train, "dro_exact",
                        eps=eps, sigma=sigma), test))
        c = {k: np.array([r[1] for r in v]) for k, v in rows.items()}
        m = {k: np.array([r[0] for r in v]) for k, v in rows.items()}
        g0, _, lo0, hi0 = ci95_t(c["cvar"] - c["dro_exact"])
        gm, _, lom, him = ci95_t(c["mean"] - c["dro_exact"])
        f0 = "*" if lo0 > 0 else ("x" if hi0 < 0 else "ns")
        emit(f"  [structured N={n}] eps(CV-picked)={eps_picked}")
        emit(f"    mean=({m['mean'].mean():.2f},{c['mean'].mean():.2f})  "
             f"cvar=({m['cvar'].mean():.2f},{c['cvar'].mean():.2f})  "
             f"dro_exact=({m['dro_exact'].mean():.2f},{c['dro_exact'].mean():.2f})")
        emit(f"    vs_cvar0={g0:+.3f} CI[{lo0:+.3f},{hi0:+.3f}] {f0}   "
             f"vs_mean={gm:+.3f} CI[{lom:+.3f},{him:+.3f}]")
        yield n, g0, lo0, hi0, eps_picked


def main():
    units = parse_flp(str(FLP))
    cfg = parse_config(str(CONFIG))
    chip_w, chip_h = chip_box(units)
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()

    out = PKG / "results/exp014_dro_exact_retrial.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s); fh.write(s + "\n")

    emit(f"exp014: corrected-DRO retrial (certified global Lambda, Mahalanobis sigma, "
         f"fractional tail, eps by 2-fold CV in sigma-units {EPS_GRID}).")
    emit(f"ev6 structured arm: N in {N_SMALLS}, {N_SEEDS} seeds, N_TEST={N_TEST}, "
         f"alpha={ALPHA}, grid {NR}x{NC}, {N_ITER} iter.")
    emit("vs_cvar0 = CVaR(pure-CVaR) - CVaR(dro_exact): the pre-registered criterion.")

    verdicts = list(run_structured(solver, units, chip_w, chip_h, emit))

    # BOOM arm (powered design), for completeness.
    csvp, rptp = resolve_paths()
    if csvp:
        wl = BoomWorkload(csvp, rptp, config_id="0")
        bs = GridFDSolver(cfg, wl.units, wl.chip_w, wl.chip_h, NR, NC)
        bs.build(); bs.factorize()

        def peaks_fn(scen):
            p = DiffPlacer(bs, wl.units, wl.chip_w, wl.chip_h, NR, NC, alpha=ALPHA)
            cx, cy = p.get_positions()
            return p._scenario_peaks(cx, cy, scen)
        wl.scale_to_peak(peaks_fn, TARGET_PEAK)
        scen = wl.scenarios()
        n_all = len(scen)
        ratio = (n_all - BOOM_TRAIN) / BOOM_TRAIN
        dc0, dcm = [], []
        eps_picked = []
        for seed in range(BOOM_SPLITS):
            print(f"  [BOOM] split {seed+1}/{BOOM_SPLITS} ...", flush=True)
            perm = np.random.default_rng(40_000 + seed).permutation(n_all)
            tr = [scen[i] for i in perm[:BOOM_TRAIN]]
            te = [scen[i] for i in perm[BOOM_TRAIN:]]
            sigma = np.asarray(tr).std(axis=0, ddof=1) + 1e-12
            eps = cv_pick_eps(bs, wl.units, wl.chip_w, wl.chip_h, tr, sigma,
                              np.random.default_rng(310_000 + seed))
            eps_picked.append(eps)
            c_m = oos(fit(bs, wl.units, wl.chip_w, wl.chip_h, tr, "mean"), te)[1]
            c_c = oos(fit(bs, wl.units, wl.chip_w, wl.chip_h, tr, "cvar"), te)[1]
            c_d = oos(fit(bs, wl.units, wl.chip_w, wl.chip_h, tr, "dro_exact",
                          eps=eps, sigma=sigma), te)[1]
            dc0.append(c_c - c_d); dcm.append(c_m - c_d)
        g0, _, lo0, hi0 = ci95_nadeau_bengio(dc0, ratio)
        gm, _, lom, him = ci95_nadeau_bengio(dcm, ratio)
        emit(f"\n  [BOOM {BOOM_TRAIN}/{n_all-BOOM_TRAIN} x {BOOM_SPLITS}, NB CI] "
             f"eps(CV)={eps_picked}")
        emit(f"    vs_cvar0={g0:+.3f} CI[{lo0:+.3f},{hi0:+.3f}]   "
             f"vs_mean={gm:+.3f} CI[{lom:+.3f},{him:+.3f}]")

    # Pre-registered verdict on the structured arm.
    stay = [n for n, g0, lo0, hi0, _ in verdicts if lo0 > 0]
    kill = [n for n, g0, lo0, hi0, _ in verdicts if hi0 < 0]
    inert = all(all(e == EPS_GRID[0] for e in eps) and lo0 <= 0 <= hi0
                for _, g0, lo0, hi0, eps in verdicts)
    if stay:
        v = (f"DRO EARNS A STAY: corrected penalty beats pure CVaR at N={stay} "
             f"(CI>0) with CV-chosen eps>0 — scale up before any final word.")
    elif kill:
        v = f"CLOSE DRO PERMANENTLY: corrected penalty significantly worse at N={kill}."
    elif inert:
        v = ("CLOSE DRO PERMANENTLY: CV collapses to the smallest eps and the "
             "corrected penalty is inert — the approximate-penalty caveat no "
             "longer shields the negative.")
    else:
        v = "INCONCLUSIVE at this power: point estimates and CV picks reported above."
    emit(f"\nPRE-REGISTERED VERDICT: {v}")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
