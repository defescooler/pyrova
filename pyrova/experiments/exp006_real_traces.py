"""Real-trace probe on ev6: mean-opt vs CVaR-opt over an N_train x alpha grid,
scenarios resampled from the .ptrace programs in PROGRAMS with disjoint
train/test row pools per split; per cell, paired dCVaR/dMean (= mean-opt minus
CVaR-opt) with 95% t-CIs over 5 seeds, plus measured cross-scenario FP/MEM and
FP/INT cluster correlations. With a single program in PROGRAMS the script runs
as a sanity check only; add a second architecturally-different trace to
PROGRAMS for a real across-program test — no code change.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # pyrova/experiments
PKG = HERE.parent                               # pyrova
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_flp, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import mean_cvar, ci95_t
from pyrova.workloads.real_traces import RealTraceWorkloadModel
from pyrova.workloads.structured import _family

FLP = PKG / "inputs/floorplans/ev6.flp"
CONFIG = PKG / "inputs/configs/thermal.config"

# Real ev6-compatible programs. Workload uncertainty here means ACROSS programs.
# Drop a second architecturally-different trace (e.g. a memory-bound SPEC benchmark on the
# Alpha 21264) here to turn this from a sanity check into a real test — no code change.
PROGRAMS = [("gcc", str(ROOT / "Tools/HotSpot/examples/example1/gcc.ptrace"))]

ALPHAS = [0.80, 0.90, 0.95]
N_TRAINS = [16, 32, 64, 128, 256]
TRAIN_FRAC = 0.5
N_TEST = 300
N_SEEDS = 5
NR = NC = 18
N_ITER = 30


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def oos_mean_cvar(pl, scen, alpha):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), alpha)


def family_corr(units, corr):
    """nanmean: a constant-power block has undefined self-correlation entries and
    must not NaN the cluster average."""
    fams = [_family(u["name"]) for u in units]
    idx = {f: [i for i, fa in enumerate(fams) if fa == f] for f in ("FP", "INT", "MEM")}
    return (float(np.nanmean(corr[np.ix_(idx["FP"], idx["MEM"])])),
            float(np.nanmean(corr[np.ix_(idx["FP"], idx["INT"])])))


def split_pools(progs, rng):
    """Disjoint train/test row pools, partitioned within each program (no row shared)."""
    train, test = [], []
    for data in progs:
        perm = rng.permutation(len(data))
        cut = max(1, int(len(data) * TRAIN_FRAC))
        train.append(data[perm[:cut]]); test.append(data[perm[cut:]])
    return np.vstack(train), np.vstack(test)


def deltas_at(solver, units, chip_w, chip_h, progs, n_train, alpha):
    """(dCVaR, dMean) with 95% CIs across seeds. dX = mean-opt minus CVaR-opt."""
    dC, dM = [], []
    for seed in range(N_SEEDS):
        # alpha in the seed: the alpha columns of a row are independent splits.
        rng = np.random.default_rng(100_000 * seed + 100 * n_train + int(round(alpha * 100)))
        tr_pool, te_pool = split_pools(progs, rng)
        train = [tr_pool[i] for i in rng.choice(len(tr_pool), n_train, replace=False)]
        nte = min(N_TEST, len(te_pool))
        test = [te_pool[i] for i in rng.choice(len(te_pool), nte, replace=False)]
        p_mean = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=alpha)
        p_mean.optimize(train, mode="mean", n_iter=N_ITER, lr=2e-2, verbose=False)
        p_cvar = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=alpha)
        p_cvar.optimize(train, mode="cvar", n_iter=N_ITER, lr=2e-2, verbose=False)
        mm, cm = oos_mean_cvar(p_mean, test, alpha)
        mc, cc = oos_mean_cvar(p_cvar, test, alpha)
        dC.append(cm - cc); dM.append(mm - mc)
    gc, _, c_lo, c_hi = ci95_t(dC)
    gm, _, m_lo, m_hi = ci95_t(dM)
    return gc, c_lo, c_hi, gm, m_lo, m_hi


def main():
    units = parse_flp(str(FLP))
    cfg = parse_config(str(CONFIG))
    chip_w, chip_h = chip_box(units)
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()

    progs = [RealTraceWorkloadModel([p], units, seed=0).data for _, p in PROGRAMS]
    stacked = np.vstack(progs)
    fm, fi = family_corr(units, np.corrcoef(stacked.T))
    train_pool = sum(max(1, int(len(d) * TRAIN_FRAC)) for d in progs)
    valid = [nt for nt in N_TRAINS if nt <= train_pool]

    out = PKG / "results/exp006_real_traces.txt"
    fh = open(out, "w")

    def emit(s):
        print(s); fh.write(s + "\n")

    emit(f"Real-trace probe on ev6 ({len(units)} blocks). {len(PROGRAMS)} program(s): "
         f"{[lbl for lbl, _ in PROGRAMS]}, {len(stacked)} total timesteps.")
    emit(f"Disjoint train/test (no shared rows). Train pool {train_pool} rows -> "
         f"valid N_TRAIN {valid}; dropped (exceed pool) {[n for n in N_TRAINS if n not in valid]}.")
    te_pool_n = len(stacked) - train_pool
    emit(f"NOTE: test pool is only ~{te_pool_n} rows, so OOS CVaR at a=0.95 averages ~"
         f"{max(1, int(te_pool_n * 0.05))} rows — far below the project's N_TEST>=1500 rule. "
         f"Tolerated ONLY because this is a sanity check, not a hypothesis test.")
    anti = [lab for lab, v in (("FP/MEM", fm), ("FP/INT", fi)) if v < -0.1]
    emit(f"MEASURED across-scenario structure: corr(FP,MEM)={fm:+.3f}  corr(FP,INT)={fi:+.3f} "
         f"-> {'anti-correlated (' + ', '.join(anti) + ')' if anti else 'NOT anti-correlated'}")
    if len(PROGRAMS) == 1:
        emit("WARNING: ONE program = a single steady-state operating point, NOT a workload "
             "distribution. Across-timestep variation is not the across-PROGRAM uncertainty "
             "the hypothesis concerns. This is a SANITY CHECK, not a valid test. Add a 2nd "
             "architecturally-different program to PROGRAMS for a real test.")
    emit("cell = dCVaR[flag]/dMean [K], both = mean-opt minus CVaR-opt. '*' dCVaR CI>0, 'x' CI<0.")
    emit(f"  {'N_TRAIN':>8} | " + " ".join(f"a={a:<11.2f}" for a in ALPHAS))
    emit("  " + "-" * (10 + 14 * len(ALPHAS)))
    for nt in valid:
        cells = []
        for a in ALPHAS:
            gc, c_lo, c_hi, gm, m_lo, m_hi = deltas_at(solver, units, chip_w, chip_h, progs, nt, a)
            flag = "*" if c_lo > 0 else ("x" if c_hi < 0 else " ")
            cells.append(f"{gc:+.2f}{flag}/{gm:+.2f}".rjust(13))
            fh.write(f"    # N_TRAIN={nt} a={a}: dCVaR={gc:+.3f} CI[{c_lo:+.3f},{c_hi:+.3f}]  "
                     f"dMean={gm:+.3f} CI[{m_lo:+.3f},{m_hi:+.3f}]\n")
        emit(f"  {nt:>8} | " + " ".join(cells))

    if len(PROGRAMS) == 1:
        emit("\nVerdict: NOT a valid test (single program). gcc is one program in steady state; "
             "with no workload diversity there is nothing to be robust against, so a ~0 gap is "
             "expected and uninformative about real across-program uncertainty.")
    else:
        emit(f"\nMulti-program test over {len(PROGRAMS)} programs; read the dCVaR/dMean cells and "
             f"the measured corr(FP,MEM)={fm:+.2f} together.")
    fh.close()
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
