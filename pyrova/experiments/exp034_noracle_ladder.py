"""Oracle-gap ladder in training-sample size: D* = trueCVaR(mean-oracle) -
trueCVaR(cvar-oracle) per N_ORACLE in {1500, 3000, 6000} under i.i.d. power,
budget fixed at 240 iterations, matched train/eval on a 24^2 grid (no raster
jitter), 5 pairs per cell against an 8000-scenario holdout.

Runs ONE N_ORACLE per invocation: set PYROVA_NORACLE or use the SLURM array
index; PYROVA_PAIRS / PYROVA_PAIR_OFFSET shard independent pairs across array
tasks; PYROVA_FLP switches the floorplan (default ev6); PYROVA_BUDGET
overrides the iteration budget. Each task writes its own result file;
aggregate the cells afterwards.

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
    if FLP.stem != "ev6":                      # default ev6 filenames stay unprefixed
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
