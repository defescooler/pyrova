"""Attribute the persistent i.i.d. D*<0: is it empirical-CVaR estimator bias or
incomplete optimization?

Population theory gives true D* >= 0, yet the powered budget ladder found D*
significantly negative even at 240 iterations, with budget closing only ~37%
(non-significant). Two artifacts remain: (i) the cvar arm minimizes EMPIRICAL
CVaR on N_ORACLE draws, an optimistically-biased objective whose bias shrinks
with N_ORACLE and is ~budget-invariant; (ii) optimization not fully converged
even at 240 iters. The budget ladder fixed N and varied iters. This fixes iters
(high) and varies N_ORACLE: if D* rises toward 0 as N_ORACLE grows, the residual
negative is estimator bias; if it stays put, it is not.

Runs ONE N_ORACLE per invocation (set PYROVA_NORACLE, or use the SLURM array
index to pick from the grid), so the expensive large-N cells parallelize across
array tasks. Each task appends its cell to its own result file; aggregate after.

i.i.d. workload on ev6, grid 24^2, budget fixed at 240, matched to the ladder.

Set PYROVA_SMOKE=1 for a tiny local execution check.
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

from pyrova.thermal.fd_solver import GridFDSolver, parse_flp, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import ci95_t, paired_t_p
from pyrova.experiments.exp003_mean_cvar_correlation import (scen_set, chip_box,
                                                             oos_mean_cvar)

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
# PYROVA_FLP switches the design (e.g. the second bench floorplan2) so the
# N-ladder can test whether the large-N tail dimension is design-generic.
FLP = Path(os.environ.get("PYROVA_FLP", PKG / "inputs/floorplans/ev6.flp"))
ALPHA = 0.9
NR = NC = 24
N_ITER = int(os.environ.get("PYROVA_BUDGET", 15 if SMOKE else 240))
N_PAIRS = int(os.environ.get("PYROVA_PAIRS", 2 if SMOKE else 5))
# Fresh independent pairs across array tasks: offset shifts the oracle seed so a
# task computes pairs [offset .. offset+N_PAIRS); aggregate cells by hand after.
PAIR_OFFSET = int(os.environ.get("PYROVA_PAIR_OFFSET", "0"))
N_TEST = 200 if SMOKE else 8000
N_GRID = [96, 192] if SMOKE else [1500, 3000, 6000]


def pick_noracle() -> int:
    if os.environ.get("PYROVA_NORACLE"):
        return int(os.environ["PYROVA_NORACLE"])
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    return N_GRID[min(idx, len(N_GRID) - 1)]


def trained(solver, units, cw, ch, train, mode):
    pl = DiffPlacer(solver, units, cw, ch, NR, NC, alpha=ALPHA)
    pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False)
    return pl


def main():
    n_oracle = pick_noracle()
    cfg = parse_config(str(CONFIG))
    units = parse_flp(str(FLP))
    n = len(units)
    cw, ch = chip_box(units)
    solver = GridFDSolver(cfg, units, cw, ch, NR, NC)
    solver.build(); solver.factorize()
    tot = 2.0 * n
    test = scen_set(units, tot, np.random.default_rng(99), N_TEST)

    tag = f"{n_oracle}" if PAIR_OFFSET == 0 else f"{n_oracle}_off{PAIR_OFFSET}"
    if FLP.stem != "ev6":                      # keep historical ev6 filenames stable
        tag = f"{FLP.stem}_{tag}"
    out = PKG / f"results/exp034_noracle_{tag}.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"N_ORACLE ladder cell: {FLP.stem} i.i.d., N_ORACLE={n_oracle}, budget={N_ITER} "
         f"(fixed), {N_PAIRS} pairs (offset {PAIR_OFFSET}), grid {NR}^2, "
         f"N_TEST={N_TEST}. " + ("[SMOKE]" if SMOKE else "[full]"))

    Dk = []
    for k in range(PAIR_OFFSET, PAIR_OFFSET + N_PAIRS):
        or_train = scen_set(units, tot, np.random.default_rng(10_000 + k), n_oracle)
        mo = trained(solver, units, cw, ch, or_train, "mean")
        co = trained(solver, units, cw, ch, or_train, "cvar")
        _, c_mo = oos_mean_cvar(mo, test)
        _, c_co = oos_mean_cvar(co, test)
        Dk.append(c_mo - c_co)
        print(f"  N={n_oracle} pair {k+1}/{N_PAIRS} D*_k={Dk[-1]:+.4f}", flush=True)

    Dm, _, lo, hi = ci95_t(Dk)
    emit(f"D*(N_ORACLE={n_oracle}) = {Dm:+.4f} K CI[{lo:+.4f},{hi:+.4f}] "
         f"p={paired_t_p(Dk):.4f}")
    emit("(rising toward 0 as N_ORACLE grows => estimator bias; flat negative => "
         "not bias. Compare cells across the array.)")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
