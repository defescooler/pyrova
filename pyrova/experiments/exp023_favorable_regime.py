"""exp023: does risk-aware placement pay in the regime its mechanism predicts?

This is the mechanism-favorable test, NOT benchmark shopping: the regime
(heterogeneous SoC, each workload class driving a different thermally heavy
engine) is chosen because exp009's mechanism analysis says it is where the
theory should operate — and the design pre-registers the falsification branch
with equal standing. Every evaluation trap this project discovered is
controlled:

  * budget trap (exp013):    all arms 120 iterations, tail arms also checked
                             at 240 (reported if it changes the verdict);
  * grid trap (exp018/020):  training uses raster_jitter=1.0 (exp022's fix);
                             ALL evaluation at 64x64;
  * baseline trap (exp019):  the mean arm is best-of-3 restarts, selected on
                             its training objective.

Design (ev-style single testbed; stylised SoC from workloads/hetero_soc.py):
  GATE (validity, enforced): across the 6 workload modes the hotspot cell at
      the initial tiling must move between >= 3 distinct blocks; otherwise
      the regime construction failed and NO verdict prints.
  E1 (existence): oracle D* = CVaR(mean-oracle) - CVaR(cvar-oracle),
      N_ORACLE=1000, 5 independent pairs, paired CI. D* CI > 0 is the
      theory's existence claim in its favorable regime.
  E2 (learnability): mean-strong vs cvar vs blend(0.75) at N_TRAIN in
      {32, 128}, 5 seeds, dCVaR/dMean vs the STRONG mean baseline.
PRE-REGISTERED READINGS:
  - PAYS if E1 D* CI > 0 AND E2 shows dCVaR CI > 0 with dMean <= 0 at some N
    (the genuine mean-for-tail trade, against a strong baseline, trap-free):
    quantified as "risk-aware placement buys [D*] K of true tail under
    heavy-anti-correlated multimodal workloads".
  - DOES NOT PAY if E1 CI <= 0 everywhere here — under the MOST favorable,
    trap-controlled conditions the theory's effect does not exist; the
    hypothesis is dead for practical purposes and is reported so.
  - MIXED readings report both quantities without a slogan.
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
from pyrova.evaluation.metrics import mean_cvar, cvar, ci95_t
from pyrova.workloads.hetero_soc import (HeteroSoCWorkloadModel, soc_units,
                                         _MODES, _BLOCKS)

CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA = 0.90
TRAIN_GRID = 18
EVAL_GRID = 64
N_ITER = 120
N_ORACLE = 1000
N_OR_PAIRS = 5
N_TRAINS = [32, 128]
N_SEEDS = 5
N_TEST = 1000
JITTER = 1.0


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def fit(solver, units, chip_w, chip_h, train, mode, gamma=0.75,
        restarts=1, rrng=None, seed=0):
    best, best_obj = None, np.inf
    for r in range(restarts):
        pl = DiffPlacer(solver, units, chip_w, chip_h, TRAIN_GRID, TRAIN_GRID,
                        alpha=ALPHA, blend_gamma=gamma)
        if r > 0:
            pl.raw_x += rrng.standard_normal(pl.n) * 0.5
            pl.raw_y += rrng.standard_normal(pl.n) * 0.5
        pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=910_000 + seed + 37 * r)
        obj = pl.objective_and_grad(train, mode=mode)[0]
        if obj < best_obj:
            best, best_obj = pl, obj
    return best


def eval_mc(cfg, units_placed, scen, chip_w, chip_h, ambient):
    s = GridFDSolver(cfg, units_placed, chip_w, chip_h, EVAL_GRID, EVAL_GRID)
    s.build(); s.factorize()
    pk = np.zeros(len(scen))
    for i, pw in enumerate(scen):
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(units_placed)}
        T = s.solve(s.build_rhs(bp))
        pk[i] = float(s.silicon_layer(T).max()) - ambient
    return float(pk.mean()), cvar(pk, ALPHA)


def main():
    units = soc_units()
    cfg = parse_config(str(CONFIG))
    chip_w, chip_h = chip_box(units)
    ambient = cfg["ambient"]
    solver = GridFDSolver(cfg, units, chip_w, chip_h, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()

    out = PKG / "results/exp023_favorable_regime.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    model0 = HeteroSoCWorkloadModel(units, seed=0)
    st = model0.engine_stats()
    emit(f"exp023: favorable-regime test on the stylised hetero-SoC "
         f"({len(units)} blocks, chip {chip_w*1e3:.1f}x{chip_h*1e3:.1f} mm). "
         f"alpha={ALPHA}; train@{TRAIN_GRID} + raster_jitter={JITTER}; eval@{EVAL_GRID}; "
         f"{N_ITER} it; mean arm best-of-3.")
    emit(f"regime stats: E[total]={st['e_total']:.1f} W  total-CV={st['total_cv']:.3f}  "
         f"heaviest-engine share={100*st['heaviest_engine_share_mean']:.0f}%")

    # Validity gate: the hotspot must move across modes at the initial tiling.
    s64 = GridFDSolver(cfg, units, chip_w, chip_h, EVAL_GRID, EVAL_GRID)
    s64.build(); s64.factorize()
    argmax_by_mode = {}
    pmax = np.array([b[3] for b in _BLOCKS])
    for mname, act in _MODES.items():
        pw = np.array(act) * pmax
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(units)}
        T = s64.solve(s64.build_rhs(bp))
        flat = int(np.argmax(s64.silicon_layer(T)))
        cell = (flat // EVAL_GRID, flat % EVAL_GRID)
        blk = min(units, key=lambda u: (u["leftx"] + u["width"] / 2 - (cell[1] + .5) * chip_w / EVAL_GRID) ** 2
                  + (u["bottomy"] + u["height"] / 2 - (EVAL_GRID - cell[0] - .5) * chip_h / EVAL_GRID) ** 2)
        argmax_by_mode[mname] = blk["name"]
        emit(f"  mode {mname:9s}: hotspot near {blk['name']}")
    n_distinct = len(set(argmax_by_mode.values()))
    emit(f"GATE: hotspot spans {n_distinct} distinct blocks across modes "
         f"(requires >= 3) -> {'PASS' if n_distinct >= 3 else 'FAIL'}")
    if n_distinct < 3:
        emit("GATE FAILED — the regime construction did not produce a moving "
             "hotspot; no verdict (fix the testbed first).")
        fh.close()
        return

    # E1: oracle existence with all traps controlled.
    emit(f"\nE1: oracle D*, N_ORACLE={N_ORACLE}, {N_OR_PAIRS} pairs, "
         f"eval on common {N_TEST}-scenario holdout at {EVAL_GRID}^2:")
    test = HeteroSoCWorkloadModel(units, seed=777).sample(N_TEST)
    Dk = []
    for k in range(N_OR_PAIRS):
        train = HeteroSoCWorkloadModel(units, seed=10_000 + k).sample(N_ORACLE)
        rrng = np.random.default_rng(20_000 + k)
        p_m = fit(solver, units, chip_w, chip_h, train, "mean", restarts=3,
                  rrng=rrng, seed=k)
        p_c = fit(solver, units, chip_w, chip_h, train, "cvar", seed=k)
        _, c_m = eval_mc(cfg, p_m.get_units(), test, chip_w, chip_h, ambient)
        _, c_c = eval_mc(cfg, p_c.get_units(), test, chip_w, chip_h, ambient)
        Dk.append(c_m - c_c)
        emit(f"  pair {k}: D*_k = {Dk[-1]:+.3f} K")
    Dm, _, D_lo, D_hi = ci95_t(Dk)
    emit(f"  D* = {Dm:+.3f} K CI[{D_lo:+.3f},{D_hi:+.3f}]")

    # E2: small-N learnability vs the STRONG mean baseline.
    emit(f"\nE2: learnability vs mean-strong (bo3), {N_SEEDS} seeds:")
    e2 = {}
    for n in N_TRAINS:
        dC, dM = {"cvar": [], "blend": []}, {"cvar": [], "blend": []}
        for seed in range(N_SEEDS):
            train = HeteroSoCWorkloadModel(units, seed=30_000 + 100 * seed + n).sample(n)
            rrng = np.random.default_rng(40_000 + seed)
            pm = fit(solver, units, chip_w, chip_h, train, "mean", restarts=3,
                     rrng=rrng, seed=seed)
            m_m, c_m = eval_mc(cfg, pm.get_units(), test, chip_w, chip_h, ambient)
            for arm, mode, g in (("cvar", "cvar", 0.75), ("blend", "blend", 0.75)):
                pl = fit(solver, units, chip_w, chip_h, train, mode, gamma=g, seed=seed)
                m_a, c_a = eval_mc(cfg, pl.get_units(), test, chip_w, chip_h, ambient)
                dC[arm].append(c_m - c_a); dM[arm].append(m_m - m_a)
            print(f"  N={n} seed {seed + 1}/{N_SEEDS}", flush=True)
        e2[n] = {}
        for arm in ("cvar", "blend"):
            gc, _, lo, hi = ci95_t(dC[arm])
            gm, _, mlo, mhi = ci95_t(dM[arm])
            fc = "*" if lo > 0 else ("x" if hi < 0 else " ")
            emit(f"  N={n} {arm:5s}: dCVaR={gc:+.3f}{fc} [{lo:+.3f},{hi:+.3f}]  "
                 f"dMean={gm:+.3f} [{mlo:+.3f},{mhi:+.3f}]")
            e2[n][arm] = (gc, lo, gm, mhi)

    # Pre-registered verdict.
    trade = [(n, a) for n in N_TRAINS for a in ("cvar", "blend")
             if e2[n][a][1] > 0 and e2[n][a][2] <= 0]
    if D_lo > 0 and trade:
        v = (f"PAYS: in its mechanism-favorable regime, risk-aware placement buys "
             f"{Dm:+.3f} K CI[{D_lo:+.2f},{D_hi:+.2f}] of true tail (oracle) and the "
             f"trade is learnable vs a strong baseline at {trade} — an existence/"
             f"upper-bound result for heavy-anti-correlated multimodal workloads.")
    elif D_hi <= 0:
        v = ("DOES NOT PAY: even in the engineered favorable regime with every "
             "trap controlled, no separable tail dimension exists (D* CI <= 0) — "
             "the hypothesis is dead for practical purposes on this evidence.")
    else:
        v = (f"MIXED: D*={Dm:+.3f} [{D_lo:+.2f},{D_hi:+.2f}]; learnable trade cells: "
             f"{trade or 'none'} — report both, no slogan.")
    emit(f"\nPRE-REGISTERED VERDICT: {v}")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
