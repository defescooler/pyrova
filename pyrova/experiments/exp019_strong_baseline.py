"""BOOM strong-baseline comparison over 20 fresh 60/20 splits (seed base
90_000), alpha=0.9, all arms lr=2e-2. Arms per split: mean-std (Adam, 120
it, single start), mean-strong (best of 5 jittered restarts x 240 it,
selected on TRAINING mean — selection never sees the holdout), blend
gamma=0.75 (Adam, 120 it, single start). Reports OOS dCVaR/dMean with
Nadeau-Bengio CIs for blend vs each mean baseline and mean-strong vs
mean-std.
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
ALPHA = 0.90
NR = NC = 18
N_ITER = 120
N_ITER_STRONG = 240
K_RESTART = 5
JITTER = 0.5
N_SPLITS = 20
N_TRAIN = 60
SEED_BASE = 90_000       # fresh split family
TARGET_PEAK = 40.0


def fit(solver, wl, train, mode, gamma, n_iter, restarts=1, rrng=None):
    best, best_obj = None, np.inf
    for r in range(restarts):
        pl = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, NR, NC,
                        alpha=ALPHA, blend_gamma=gamma)
        if r > 0:
            pl.raw_x += rrng.standard_normal(pl.n) * JITTER
            pl.raw_y += rrng.standard_normal(pl.n) * JITTER
        pl.optimize(train, mode=mode, n_iter=n_iter, lr=2e-2, verbose=False)
        obj = pl.objective_and_grad(train, mode=mode)[0]
        if obj < best_obj:
            best, best_obj = pl, obj
    return best


def oos(pl, scen):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), ALPHA)


def main():
    csvp, rptp = resolve_paths()
    if not csvp:
        print("BOOM_DATA not found.")
        return
    cfg = parse_config(str(CONFIG))
    wl = BoomWorkload(csvp, rptp, config_id="0")
    solver = GridFDSolver(cfg, wl.units, wl.chip_w, wl.chip_h, NR, NC)
    solver.build(); solver.factorize()

    def peaks_fn(scen):
        p = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, NR, NC, alpha=ALPHA)
        cx, cy = p.get_positions()
        return p._scenario_peaks(cx, cy, scen)
    wl.scale_to_peak(peaks_fn, TARGET_PEAK)
    scen = wl.scenarios()
    n = len(scen)
    ratio = (n - N_TRAIN) / N_TRAIN

    out = PKG / "results/exp019_strong_baseline.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"exp019: strong mean baseline + fresh-split replication. {n} programs, "
         f"{N_TRAIN}/{n - N_TRAIN} x {N_SPLITS} FRESH splits (seed base {SEED_BASE}), "
         f"alpha={ALPHA}, grid {NR}x{NC}, lr=2e-2, NB CIs.")
    emit(f"arms: mean-std ({N_ITER} it, 1 start) / mean-strong (best-of-{K_RESTART} "
         f"x {N_ITER_STRONG} it, selection on TRAINING mean) / blend g=0.75 ({N_ITER} it).")

    res = {a: {"m": [], "c": []} for a in ("mean-std", "mean-strong", "blend")}
    for seed in range(N_SPLITS):
        perm = np.random.default_rng(SEED_BASE + seed).permutation(n)
        tr = [scen[i] for i in perm[:N_TRAIN]]
        te = [scen[i] for i in perm[N_TRAIN:]]
        rrng = np.random.default_rng(SEED_BASE + 500 + seed)
        arms = {
            "mean-std": fit(solver, wl, tr, "mean", 0.0, N_ITER),
            "mean-strong": fit(solver, wl, tr, "mean", 0.0, N_ITER_STRONG,
                               restarts=K_RESTART, rrng=rrng),
            "blend": fit(solver, wl, tr, "blend", 0.75, N_ITER),
        }
        for a, pl in arms.items():
            m, c = oos(pl, te)
            res[a]["m"].append(m); res[a]["c"].append(c)
        print(f"  split {seed + 1}/{N_SPLITS} done", flush=True)

    for a in res:
        res[a]["m"] = np.asarray(res[a]["m"]); res[a]["c"] = np.asarray(res[a]["c"])

    def row(label, base, arm):
        dc = res[base]["c"] - res[arm]["c"]
        dm = res[base]["m"] - res[arm]["m"]
        gc, _, lo, hi = ci95_nadeau_bengio(dc, ratio)
        gm, _, mlo, mhi = ci95_nadeau_bengio(dm, ratio)
        fc = "*" if lo > 0 else ("x" if hi < 0 else " ")
        fm = "*" if mlo > 0 else ("x" if mhi < 0 else " ")
        emit(f"  {label:34s} dCVaR={gc:+.3f}{fc} [{lo:+.2f},{hi:+.2f}]  "
             f"dMean={gm:+.3f}{fm} [{mlo:+.2f},{mhi:+.2f}]")
        return gc, lo, hi, gm, mlo, mhi

    emit(f"\n  {'arm':14s}{'OOSmean':>9}{'OOSCVaR':>9}")
    for a in ("mean-std", "mean-strong", "blend"):
        emit(f"  {a:14s}{res[a]['m'].mean():>9.3f}{res[a]['c'].mean():>9.3f}")
    emit("")
    row("mean-strong vs mean-std (baseline gain)", "mean-std", "mean-strong")
    sec = row("blend vs mean-std (exp015-A replication)", "mean-std", "blend")
    pri = row("blend vs mean-STRONG (primary)", "mean-strong", "blend")

    gc, lo, hi, gm, mlo, mhi = pri
    if lo > 0:
        v = ("SURVIVES: blend beats the 10x-effort mean baseline on OOS CVaR with "
             "NB-CI > 0 on fresh splits — claim 11 upgrades to replicated-vs-strong-baseline"
             + (" (and still dominates on the mean)" if mlo > 0 else
                " (domination reduced: dMean CI spans 0 — the trade reading applies)"))
    elif gc > 0:
        v = ("WEAKENED: positive point estimate vs the strong baseline but CI spans 0 — "
             "report both comparisons; the historical claim was partly baseline-limited.")
    else:
        v = ("BASELINE ARTIFACT: the strong mean baseline closes the gap — re-scope "
             "claim 11 to equal-effort single-start training and demote the domination finding.")
    emit(f"\nPRE-REGISTERED VERDICT: {v}")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
