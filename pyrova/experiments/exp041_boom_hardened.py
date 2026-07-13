"""Paired placement comparison on the 80-program BOOM pool: cvar, cvar_wide
(train alpha=0.6, score 0.9), and blend (gamma=0.75) arms against a mean
baseline over repeated 60/20 splits — best-of-3 restarts with common random
numbers per arm, train@18 with raster jitter, guaranteed-legal evaluation on
an independent 64^2 grid, Nadeau-Bengio CIs with dMean alongside.

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

from pyrova.thermal.fd_solver import GridFDSolver, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.optimizer.legalize import legalize_units_exact, LegalizationInfeasible
from pyrova.evaluation.metrics import cvar, ci95_nadeau_bengio
from pyrova.workloads.boom_traces import BoomWorkload, resolve_paths

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
CONFIG_ID = "0"
ALPHA = 0.90
ALPHA_WIDE = 0.60
NR = NC = 18
TARGET_PEAK = 40.0
JITTER = 1.0

EVAL_GRID = 32 if SMOKE else 64
N_ITER = 8 if SMOKE else 120
N_SPLITS = 2 if SMOKE else 40
N_TRAIN = 12 if SMOKE else 60
RESTARTS = 1 if SMOKE else 3

# name -> (mode, train_alpha, gamma)
ARMS = {"cvar": ("cvar", ALPHA, 0.5),
        "cvar_wide": ("cvar", ALPHA_WIDE, 0.5),
        "blend": ("blend", ALPHA, 0.75)}


def main():
    csvp, rptp = resolve_paths()
    if not csvp:
        print("BOOM_DATA not found; see workloads/boom_traces.py.")
        return
    cfg = parse_config(str(CONFIG))
    wl = BoomWorkload(csvp, rptp, config_id=CONFIG_ID)
    cw, ch = wl.chip_w, wl.chip_h
    solver = GridFDSolver(cfg, wl.units, cw, ch, NR, NC)
    solver.build(); solver.factorize()
    amb = cfg["ambient"]

    def train_peaks(scen):
        p = DiffPlacer(solver, wl.units, cw, ch, NR, NC, alpha=ALPHA)
        cx, cy = p.get_positions()
        return p._scenario_peaks(cx, cy, scen)
    wl.scale_to_peak(train_peaks, TARGET_PEAK)
    scen = wl.scenarios()
    n = len(scen)
    n_test = n - N_TRAIN

    out = PKG / "results/exp041_boom_hardened.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"BOOM hardened: {n} programs, {N_TRAIN}/{n_test} x {N_SPLITS} splits, "
         f"alpha={ALPHA}, best-of-{RESTARTS} restarts (CRN), train@{NR}+jitter -> "
         f"guaranteed-legal eval@{EVAL_GRID}, {N_ITER} it. arms={list(ARMS)}. "
         + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    def eval_cvar_mean(up, test):
        s = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
        s.build(); s.factorize()
        pk = np.array([float(s.silicon_layer(s.solve(s.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - amb
            for pw in test])
        return cvar(pk, ALPHA), float(pk.mean())

    def fit(train, mode, alpha, gamma, seed0):
        """Best-of-RESTARTS by final training objective, then guaranteed-legal."""
        best_obj, best_pl = None, None
        for r in range(RESTARTS):
            pl = DiffPlacer(solver, wl.units, cw, ch, NR, NC,
                            alpha=alpha, blend_gamma=gamma)
            hist = pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                               raster_jitter=JITTER, jitter_seed=seed0 + r)
            if best_obj is None or hist[-1] < best_obj:
                best_obj, best_pl = hist[-1], pl
        up, _ = legalize_units_exact(best_pl.get_units(), cw, ch)
        return up

    dC = {a: [] for a in ARMS}
    dM = {a: [] for a in ARMS}
    for sp in range(N_SPLITS):
        rng = np.random.default_rng(700_000 + sp)
        perm = rng.permutation(n)
        tr = [scen[i] for i in perm[:N_TRAIN]]
        te = [scen[i] for i in perm[N_TRAIN:]]
        seed0 = 800_000 + 100 * sp                     # CRN: shared restart seeds
        try:
            um = fit(tr, "mean", ALPHA, 0.5, seed0)
            arms_up = {a: fit(tr, m, al, g, seed0) for a, (m, al, g) in ARMS.items()}
        except LegalizationInfeasible:
            print(f"  split {sp+1}: INFEASIBLE, skipped", flush=True)
            continue
        c_m, m_m = eval_cvar_mean(um, te)
        for a in ARMS:
            c_a, m_a = eval_cvar_mean(arms_up[a], te)
            dC[a].append(c_m - c_a)
            dM[a].append(m_m - m_a)
        print(f"  split {sp+1}/{N_SPLITS} done", flush=True)

    ratio = n_test / N_TRAIN
    emit(f"\nNB half-width factor sqrt(1/J + n_te/n_tr) = "
         f"{np.sqrt(1.0/max(len(dC['cvar']),1) + ratio):.2f} sd")
    verdicts = []
    for a in ARMS:
        d = np.array(dC[a])
        if len(d) < 2:
            emit(f"  {a:10s}: too few feasible splits"); continue
        m, _, lo, hi = ci95_nadeau_bengio(d, ratio)
        dmean = float(np.mean(dM[a]))
        sig = lo > 0
        verdicts.append((a, sig, dmean <= 0))
        emit(f"  {a:10s}: dCVaR={m:+.3f} NB-CI[{lo:+.3f},{hi:+.3f}]{'*' if sig else ' '} "
             f"dMean={dmean:+.3f}  ({'trade' if dmean <= 0 else 'domination'})")

    emit("\n===== PRE-REGISTERED VERDICT =====")
    wins = [a for a, sig, trade in verdicts if sig and trade]
    if wins:
        emit(f"CONFIRMED: risk-aware placement helps on real BOOM at {wins} "
             f"(NB-CI>0, dMean<=0) — first trap-controlled real-workload win.")
    elif any(sig for _, sig, _ in verdicts):
        emit("DOMINATION-ONLY: some arm is NB-CI>0 but with dMean>0 — an "
             "under-converged mean baseline, not a mean-for-tail trade.")
    else:
        emit("UNDERPOWERED/NULL: no arm's dCVaR NB-CI clears 0. The 80-program "
             "pool cannot resolve the placement question at this tail; the lever "
             "is more programs, not more method.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
