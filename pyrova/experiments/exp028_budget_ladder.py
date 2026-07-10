"""exp028: is the i.i.d. D*<0 a budget artifact? The pivot, properly powered.

The project's overall verdict 1 -- "the i.i.d. negative was convergence speed,
not structure" -- currently rests on two thin runs: exp013-A (3 oracle pairs, no
CI on the '61% closed' fraction) and exp015-B (3 pairs, and it changed budget,
N_ORACLE AND pair-count simultaneously, so the sign change 'resolved' nothing
cleanly). This experiment isolates the ONE variable.

It reproduces exp003's existence block EXACTLY -- same i.i.d. random_power_map,
same grid 24^2, N_ORACLE=1500, N_TEST=4000, tot=2*n, alpha=0.9, same benches --
and changes only: (i) a BUDGET LADDER n_iter in {40,120,240}; (ii) >=10 oracle
pairs; (iii) the ladder is PAIRED (each pair's three budgets share one oracle
draw + one holdout), so the improvement per pair has a paired CI.

Population theory: min_p CVaR(p) <= CVaR(mean-opt) => true D* >= 0. So a measured
D*<0 must be optimizer under-convergence or finite-oracle estimator variance.
Estimator variance is budget-invariant; optimizer under-convergence shrinks with
budget. Hence:

PRE-REGISTERED READING (per bench):
  - BUDGET ARTIFACT CONFIRMED if D*(40) CI<0 (reproduces exp003) AND the paired
    improvement dD*(240)=D*(240)-D*(40) CI>0 AND D*(240) CI includes or exceeds 0
    -- the negative closes with budget, as the convergence-speed story requires.
  - STRUCTURAL / OPEN if D*(240) CI stays < 0 despite the ladder -- more budget
    does not close it; the negative is not (only) convergence speed and needs a
    deeper look (estimator variance, or a real pathology).
  - NO EFFECT TO EXPLAIN if D*(40) CI already includes 0 on this bench.

Set PYROVA_SMOKE=1 for a tiny local execution check (not a result).
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
from pyrova.evaluation.metrics import ci95_t
from pyrova.experiments.exp003_mean_cvar_correlation import (scen_set, chip_box,
                                                             oos_mean_cvar)

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
BENCHES = [PKG / "inputs/floorplans/ev6.flp",
           ROOT / "Tools/HotSpot/examples/example3/floorplan2.flp"]
ALPHA = 0.9
NR = NC = 24
N_ORACLE = 64 if SMOKE else 1500
N_TEST = 200 if SMOKE else 4000
N_OR_PAIRS = 2 if SMOKE else 12
BUDGETS = [40, 120] if SMOKE else [40, 120, 240]


def trained_b(solver, units, chip_w, chip_h, train, mode, n_iter):
    pl = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=ALPHA)
    pl.optimize(train, mode=mode, n_iter=n_iter, lr=2e-2, verbose=False)
    return pl


def run(path, cfg, emit):
    units = parse_flp(str(path))
    n = len(units)
    chip_w, chip_h = chip_box(units)
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()
    tot = 2.0 * n
    test = scen_set(units, tot, np.random.default_rng(99), N_TEST)

    emit(f"\n=== {path.stem} ({n} blocks), alpha={ALPHA}, grid {NR}^2, "
         f"N_ORACLE={N_ORACLE}, {N_OR_PAIRS} paired pairs ===")

    # D_k[b] : per-pair D* at each budget; paired across budgets (shared or_train + holdout).
    D = {b: [] for b in BUDGETS}
    for k in range(N_OR_PAIRS):
        or_train = scen_set(units, tot, np.random.default_rng(10_000 + k), N_ORACLE)
        for b in BUDGETS:
            mo = trained_b(solver, units, chip_w, chip_h, or_train, "mean", b)
            co = trained_b(solver, units, chip_w, chip_h, or_train, "cvar", b)
            _, c_mo = oos_mean_cvar(mo, test)
            _, c_co = oos_mean_cvar(co, test)
            D[b].append(c_mo - c_co)
        print(f"  {path.stem} pair {k + 1}/{N_OR_PAIRS} done", flush=True)

    stats = {}
    for b in BUDGETS:
        Dm, _, lo, hi = ci95_t(D[b])
        stats[b] = (Dm, lo, hi)
        emit(f"  D*({b:3d} it) = {Dm:+.3f} K CI[{lo:+.3f},{hi:+.3f}]")
    # Paired improvement vs the 40-iter rung.
    base = np.asarray(D[BUDGETS[0]])
    imp = {}
    for b in BUDGETS[1:]:
        d = np.asarray(D[b]) - base
        dm, _, lo, hi = ci95_t(d)
        imp[b] = (dm, lo, hi)
        emit(f"  paired dD*({b} vs {BUDGETS[0]}) = {dm:+.3f} K CI[{lo:+.3f},{hi:+.3f}]"
             f"  (>0 => the deficit closes with budget)")
    # Closure fraction (aggregate; only meaningful if the base rung is negative).
    if stats[BUDGETS[0]][0] < 0:
        top = BUDGETS[-1]
        closed = 100.0 * (stats[top][0] - stats[BUDGETS[0]][0]) / (-stats[BUDGETS[0]][0])
        emit(f"  closure at {top} it: {closed:.0f}% of the 40-it deficit "
             f"(paired CI on dD* above is the inferential statement)")

    # Pre-registered verdict.
    b0, bT = BUDGETS[0], BUDGETS[-1]
    D0_lo, D0_hi = stats[b0][1], stats[b0][2]
    DT_lo, DT_hi = stats[bT][1], stats[bT][2]
    imp_lo = imp[bT][1] if bT in imp else -1.0
    if D0_hi >= 0:
        v = (f"NO EFFECT TO EXPLAIN on {path.stem}: D*(40) CI already includes 0 "
             f"(no reproduced negative here).")
    elif imp_lo > 0 and DT_hi >= 0:
        v = (f"BUDGET ARTIFACT CONFIRMED on {path.stem}: D*(40) CI<0 reproduces "
             f"exp003, the paired improvement to {bT} it is CI>0, and D*({bT}) CI "
             f"reaches 0 -- the i.i.d. negative is convergence speed, now on "
             f"{N_OR_PAIRS} paired pairs with a CI on the closure.")
    elif DT_hi < 0:
        v = (f"STRUCTURAL / OPEN on {path.stem}: D*({bT}) CI still < 0 after "
             f"{bT} iters -- budget does not close it; not (only) convergence "
             f"speed. Investigate estimator variance / pathology.")
    else:
        v = (f"MIXED on {path.stem}: D*(40)={stats[b0][0]:+.3f}, "
             f"D*({bT})={stats[bT][0]:+.3f}, dD*={imp.get(bT, ('na',))[0]} -- "
             f"report the ladder, no slogan.")
    emit(f"  VERDICT: {v}")
    return stats


def main():
    cfg = parse_config(str(CONFIG))
    out = PKG / "results/exp028_budget_ladder.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit("exp028: i.i.d. oracle D* vs optimizer BUDGET (the exp013/exp015 pivot, "
         "properly powered). Isolates budget: exp003's setup exactly, only n_iter "
         f"and pair-count change. {'[SMOKE - not a result]' if SMOKE else '[full run]'}")
    emit(f"budgets={BUDGETS}, N_OR_PAIRS={N_OR_PAIRS}, N_ORACLE={N_ORACLE}, "
         f"N_TEST={N_TEST}, grid={NR}^2, alpha={ALPHA}.")
    for p in BENCHES:
        if p.exists():
            run(p, cfg, emit)
        else:
            emit(f"\n(skip {p} -- not found)")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
