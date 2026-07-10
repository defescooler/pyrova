"""exp015: re-test the two headline claims under CONVERGENCE-MATCHED optimisation.

exp013 measured that the CVaR objective under-converges at the suite's standard
30-iteration budget (own-metric deficit +0.25 K, 61% closed by 4x iterations,
restarts irrelevant). Two results must therefore be re-tested with all arms at
the HIGH budget (120 iterations), because both are exposed to the confound:

  A. exp012's blend win on BOOM (gamma=0.75 dCVaR +0.90* [+0.07,+1.73]) ALSO
     showed dMean +0.89* — mean-opt losing on its own metric, the signature of
     under-convergence. If the blend's win was "mean-opt needed more
     iterations", it disappears here; if it survives, it is a real,
     budget-robust placement-quality win (and the pre-registered claim
     upgrades from borderline to confirmed-at-matched-budgets).

  B. exp003's D* < 0 (i.i.d., ev6) — the mathematically-impossible-for-exact-
     optimisation negative. Re-measured with cvar at 4x iterations: prediction
     (from exp013) is D* -> ~0 (no separable tail dimension, no longer
     significantly negative). If D* stays significantly negative even at 4x,
     the convergence story is incomplete.

PRE-REGISTERED READINGS:
  A: CONFIRMED-ROBUST if gamma=0.75 dCVaR NB-CI > 0 at 120 iters;
     OPTIMIZER-ARTIFACT if the point estimate collapses toward 0;
     either way report dMean alongside (domination vs trade).
  B: RESOLVED if D*(120it) CI includes 0; RESIDUAL-ANOMALY if CI < 0 persists.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_flp, parse_config, random_power_map
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import mean_cvar, ci95_t, ci95_nadeau_bengio
from pyrova.workloads.boom_traces import BoomWorkload, resolve_paths

FLP = PKG / "inputs/floorplans/ev6.flp"
CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA = 0.90
NR = NC = 18
N_ITER_HI = 120          # exp013's 4x budget, applied to ALL arms
GAMMAS = [0.0, 0.75, 1.0]
N_SPLITS = 20
N_TRAIN = 60
TARGET_PEAK = 40.0
N_OR_PAIRS = 3
N_ORACLE = 1000
N_TEST_OR = 4000


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def iid_set(units, tot, rng, k):
    return [np.array([random_power_map(units, tot, rng)[u["name"]] for u in units])
            for _ in range(k)]


def fit(solver, units, chip_w, chip_h, train, gamma):
    mode = "mean" if gamma == 0.0 else ("cvar" if gamma == 1.0 else "blend")
    pl = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=ALPHA,
                    blend_gamma=gamma)
    pl.optimize(train, mode=mode, n_iter=N_ITER_HI, lr=2e-2, verbose=False)
    return pl


def oos(pl, scen):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), ALPHA)


def main():
    out = PKG / "results/exp015_equalized_optimizer.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    cfg = parse_config(str(CONFIG))

    # A: BOOM blend at matched high budget
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
        n = len(scen); ratio = (n - N_TRAIN) / N_TRAIN
        emit(f"exp015-A: BOOM blend, ALL arms at {N_ITER_HI} iterations "
             f"({N_TRAIN}/{n-N_TRAIN} x {N_SPLITS} splits, NB CI, alpha={ALPHA}). "
             f"Same splits as exp012 (seed base 40_000).")
        means = {g: [] for g in GAMMAS}
        cvars = {g: [] for g in GAMMAS}
        for seed in range(N_SPLITS):
            perm = np.random.default_rng(40_000 + seed).permutation(n)
            tr = [scen[i] for i in perm[:N_TRAIN]]
            te = [scen[i] for i in perm[N_TRAIN:]]
            for g in GAMMAS:
                m, c = oos(fit(bs, wl.units, wl.chip_w, wl.chip_h, tr, g), te)
                means[g].append(m); cvars[g].append(c)
            emit(f"  split {seed+1}/{N_SPLITS} done")
        means = {g: np.asarray(v) for g, v in means.items()}
        cvars = {g: np.asarray(v) for g, v in cvars.items()}
        emit(f"  {'gamma':>6}{'OOSmean':>9}{'OOSCVaR':>9}  {'dCVaR vs mean':>22}  {'dMean vs mean':>22}")
        a075 = None
        for g in GAMMAS:
            dc = cvars[0.0] - cvars[g]; dm = means[0.0] - means[g]
            gc, _, lo, hi = ci95_nadeau_bengio(dc, ratio)
            gm, _, mlo, mhi = ci95_nadeau_bengio(dm, ratio)
            fc = "*" if lo > 0 else ("x" if hi < 0 else " ")
            sc = "        -     " if g == 0.0 else f"{gc:+.3f}{fc} [{lo:+.2f},{hi:+.2f}]"
            sm = "        -     " if g == 0.0 else f"{gm:+.3f}  [{mlo:+.2f},{mhi:+.2f}]"
            emit(f"  {g:>6.2f}{means[g].mean():>9.3f}{cvars[g].mean():>9.3f}  {sc:>22}  {sm:>22}")
            if g == 0.75:
                a075 = (gc, lo, hi, gm, mlo)
        gc, lo, hi, gm, mlo = a075
        if lo > 0 and mlo <= 0:
            va = (f"CONFIRMED-ROBUST as a mean-for-tail trade: dCVaR {gc:+.3f} CI>0 at "
                  f"matched {N_ITER_HI}-iteration budgets with dMean no longer dominated.")
        elif lo > 0:
            va = (f"CONFIRMED-ROBUST as a placement-quality win: dCVaR {gc:+.3f} CI>0 at "
                  f"matched budgets, but blend still beats mean-opt on the mean too "
                  f"(dMean {gm:+.3f}) — a domination pattern, not the theory's trade.")
        elif gc > 0:
            va = (f"WEAKENED: point estimate {gc:+.3f} positive but CI spans 0 at matched "
                  f"budgets — exp012's borderline star does not confirm.")
        else:
            va = (f"OPTIMIZER-ARTIFACT: the exp012 win vanishes at matched budgets "
                  f"({gc:+.3f}); attribute it to mean-opt under-convergence.")
        emit(f"  A VERDICT: {va}")
    else:
        emit("exp015-A SKIPPED: BOOM_DATA not found.")

    # B: D* under the equalized optimizer
    units = parse_flp(str(FLP))
    chip_w, chip_h = chip_box(units)
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()
    tot = 2.0 * len(units)
    emit(f"\nexp015-B: i.i.d. D* with both oracles at {N_ITER_HI} iterations "
         f"(ev6, N_ORACLE={N_ORACLE}, {N_OR_PAIRS} pairs, N_TEST={N_TEST_OR}).")
    test = iid_set(units, tot, np.random.default_rng(99), N_TEST_OR)
    Dk = []
    for k in range(N_OR_PAIRS):
        train = iid_set(units, tot, np.random.default_rng(50_000 + k), N_ORACLE)
        c_m = oos(fit(solver, units, chip_w, chip_h, train, 0.0), test)[1]
        c_c = oos(fit(solver, units, chip_w, chip_h, train, 1.0), test)[1]
        Dk.append(c_m - c_c)
        emit(f"  pair {k}: D*_k = {Dk[-1]:+.3f}")
    Dm, _, lo, hi = ci95_t(Dk)
    emit(f"  D*({N_ITER_HI}it) = {Dm:+.3f} K CI[{lo:+.3f},{hi:+.3f}]")
    emit("  B VERDICT: " + (
        "RESOLVED — D* is no longer significantly negative under the equalized "
        "optimizer; exp003's negative was convergence-speed, and the i.i.d. claim "
        "should read 'no separable tail dimension' (not 'harmful')."
        if lo <= 0 <= hi else
        ("RESIDUAL-ANOMALY — D* CI still excludes 0 below; the convergence story is "
         "incomplete (investigate smoothed-CVaR gradients before further claims)."
         if hi < 0 else
         "SURPRISE — D* now significantly POSITIVE; a true i.i.d. tail dimension "
         "was masked by under-convergence. Re-examine exp004's negatives.")))
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
