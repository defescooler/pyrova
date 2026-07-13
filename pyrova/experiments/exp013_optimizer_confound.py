"""Optimizer-budget attribution, i.i.d. on ev6. Part A: own-metric OOS-CVaR
deficit vs mean(std) across arms mean(std) / mean(best-of-3) / cvar(std) /
cvar(4x iterations) / cvar(best-of-3) at N_ORACLE=1000 (3 pairs, N_TEST=4000),
restart selection on the training objective only; the budget response
attributes a deficit to optimisation vs estimation (estimator noise is
budget-invariant). Part B: paired vs_mean for blend gamma=0.5 at N_train=16
over 20 fresh seeds with a 95% t-CI.
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
from pyrova.evaluation.metrics import mean_cvar, ci95_t, paired_t_p

FLP = PKG / "inputs/floorplans/ev6.flp"
CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA = 0.90
NR = NC = 18
N_ITER = 30
N_ORACLE = 1000
N_TEST_A = 4000
N_PAIRS = 3
K_RESTART = 3
JITTER = 0.5            # raw-parameter init jitter for restarts

N_TRAIN_B = 16
N_TEST_B = 1500
N_SEEDS_B = 20


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def iid_set(units, tot, rng, k):
    return [np.array([random_power_map(units, tot, rng)[u["name"]] for u in units])
            for _ in range(k)]


def train_obj(pl, scen, mode):
    """Final TRAINING objective value (for restart selection)."""
    return pl.objective_and_grad(scen, mode=mode)[0]


def fit(solver, units, chip_w, chip_h, train, mode, n_iter, gamma=0.5,
        restarts=1, restart_rng=None):
    """Train `restarts` placers (jittered inits after the first), return the one
    with the best final TRAINING objective — selection never sees the holdout."""
    best, best_obj = None, np.inf
    for r in range(restarts):
        pl = DiffPlacer(solver, units, chip_w, chip_h, NR, NC,
                        alpha=ALPHA, blend_gamma=gamma)
        if r > 0:
            pl.raw_x += restart_rng.standard_normal(pl.n) * JITTER
            pl.raw_y += restart_rng.standard_normal(pl.n) * JITTER
        pl.optimize(train, mode=mode, n_iter=n_iter, lr=2e-2, verbose=False)
        obj = train_obj(pl, train, mode)
        if obj < best_obj:
            best, best_obj = pl, obj
    return best


def oos(pl, scen):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), ALPHA)


def main():
    units = parse_flp(str(FLP))
    cfg = parse_config(str(CONFIG))
    chip_w, chip_h = chip_box(units)
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()
    tot = 2.0 * len(units)

    out = PKG / "results/exp013_optimizer_confound.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s); fh.write(s + "\n")

    emit(f"exp013 PART A: optimizer-confound test on ev6, i.i.d., alpha={ALPHA}, "
         f"N_ORACLE={N_ORACLE}, N_TEST={N_TEST_A}, {N_PAIRS} pairs, grid {NR}x{NC}.")
    emit(f"arms: mean(std {N_ITER}it) / mean(bo{K_RESTART}) / cvar(std) / "
         f"cvar(4x={4*N_ITER}it) / cvar(bo{K_RESTART}); restart selection on the "
         f"TRAINING objective only.")
    emit("deficit = OOS CVaR(arm) - OOS CVaR(mean-std of the same pair); "
         ">0 is a red flag (population D*>=0); budget response attributes it "
         "to optimisation vs estimation (see docstring).")

    test = iid_set(units, tot, np.random.default_rng(99), N_TEST_A)
    arms = ["mean-std", f"mean-bo{K_RESTART}", "cvar-std", "cvar-4x", f"cvar-bo{K_RESTART}"]
    deficits = {a: [] for a in arms}
    for k in range(N_PAIRS):
        train = iid_set(units, tot, np.random.default_rng(50_000 + k), N_ORACLE)
        rrng = np.random.default_rng(60_000 + k)
        pls = {
            "mean-std": fit(solver, units, chip_w, chip_h, train, "mean", N_ITER),
            f"mean-bo{K_RESTART}": fit(solver, units, chip_w, chip_h, train, "mean",
                                       N_ITER, restarts=K_RESTART, restart_rng=rrng),
            "cvar-std": fit(solver, units, chip_w, chip_h, train, "cvar", N_ITER),
            "cvar-4x": fit(solver, units, chip_w, chip_h, train, "cvar", 4 * N_ITER),
            f"cvar-bo{K_RESTART}": fit(solver, units, chip_w, chip_h, train, "cvar",
                                       N_ITER, restarts=K_RESTART, restart_rng=rrng),
        }
        res = {a: oos(pl, test) for a, pl in pls.items()}
        base_c = res["mean-std"][1]
        emit(f"  pair {k}: " + "  ".join(
            f"{a}=({res[a][0]:.2f},{res[a][1]:.2f})" for a in arms))
        for a in arms:
            deficits[a].append(res[a][1] - base_c)

    emit("\n  own-metric deficit vs mean-std [K] (mean over pairs, +sd):")
    d_std = float(np.mean(deficits["cvar-std"]))
    closed = {}
    for a in arms[1:]:
        d = np.asarray(deficits[a])
        emit(f"    {a:<10} {d.mean():+.3f} (sd {d.std(ddof=1):.3f})")
        closed[a] = d.mean()
    if d_std > 0:
        for a in ("cvar-4x", f"cvar-bo{K_RESTART}"):
            frac = 1.0 - closed[a] / d_std
            emit(f"  {a} closes {100*frac:.0f}% of cvar-std's deficit ({d_std:+.3f} -> {closed[a]:+.3f})")
        big = any(1.0 - closed[a] / d_std >= 0.5 for a in ("cvar-4x", f"cvar-bo{K_RESTART}"))
        emit("  PART A READING: " + (
            "deficit substantially OPTIMIZER-INDUCED (>=50% closed by more optimisation "
            "quality) — the i.i.d. negatives are partly about the optimiser, not the "
            "objective; smoothed/variance-reduced CVaR optimisation is warranted."
            if big else
            "deficit persists under 4x iterations and restarts — the CVaR landscape is "
            "intrinsically harder for this pipeline; phrase all negatives as "
            "'for this optimiser', and method work needs more than budget/restarts."))
    else:
        emit("  PART A READING: cvar-std shows no deficit at this N/grid — the exp003 "
             "D*<0 did not reproduce under this protocol; reconcile before further claims.")

    # Part B
    emit(f"\nexp013 PART B: replicate exp010 iid anomaly (N={N_TRAIN_B}, gamma=0.5), "
         f"{N_SEEDS_B} fresh seeds, N_TEST={N_TEST_B}.")
    dC = []
    for s in range(N_SEEDS_B):
        rng = np.random.default_rng(70_000 + s)
        train = iid_set(units, tot, rng, N_TRAIN_B)
        test_b = iid_set(units, tot, rng, N_TEST_B)
        pm = fit(solver, units, chip_w, chip_h, train, "mean", N_ITER)
        pb = fit(solver, units, chip_w, chip_h, train, "blend", N_ITER, gamma=0.5)
        _, cm = oos(pm, test_b)
        _, cb = oos(pb, test_b)
        dC.append(cm - cb)
    g, _, lo, hi = ci95_t(dC)
    p = paired_t_p(dC)
    emit(f"  vs_mean(gamma=0.5) = {g:+.3f} K CI[{lo:+.3f},{hi:+.3f}] p={p:.4f} "
         f"({N_SEEDS_B} seeds)")
    emit("  PRE-REGISTERED: " + (
        f"REPLICATED (CI>0) — mild tail-blending helps i.i.d. at N={N_TRAIN_B}; "
        "this contradicts D*<=0 and needs a mechanism (likely the same optimizer-"
        "variance story as PART A: the blend's averaged gradient optimises better)."
        if lo > 0 else
        "NOT replicated — exp010's cell was multiplicity noise, as presumed."))
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
