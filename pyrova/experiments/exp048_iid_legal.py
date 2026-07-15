"""i.i.d.-power oracle gap on MCNC ami33 (33 blocks, utilization 0.55):
per-pair D* and dMean at alpha=0.9, fresh evaluation set per pair, CRN
within a pair, N_ORACLE=6000, 240 iterations, train@24+jitter with density
spreading, guaranteed-legal eval@64^2.

Sharded via PYROVA_PAIRS / PYROVA_PAIR_OFFSET; pool per-pair lines with
pyrova.evaluation.pool. PYROVA_SMOKE=1 for an execution check.
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
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p
from pyrova.experiments.exp003_mean_cvar_correlation import scen_set
from pyrova.workloads.mcnc import load_yal

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
EVAL_ALPHA = 0.90
TRAIN_GRID = 24
EVAL_GRID = 32 if SMOKE else 64
JITTER = 1.0
DENSITY_W = 5.0e2
DENSITY_LAM0 = 3.0e1
DENSITY_GRID = (48, 48)
UTILIZATION = 0.55

N_ITER = int(os.environ.get("PYROVA_BUDGET", 15 if SMOKE else 240))
N_ORACLE = int(os.environ.get("PYROVA_NORACLE", 96 if SMOKE else 6000))
N_PAIRS = int(os.environ.get("PYROVA_PAIRS", 2 if SMOKE else 3))
PAIR_OFFSET = int(os.environ.get("PYROVA_PAIR_OFFSET", "0"))
N_TEST = 200 if SMOKE else 8000

TRAIN_SEED_BASE = 10_000
JITTER_SEED_BASE = 390_000     # same jitter stream as the EV6 i.i.d. runs
FRESH_TEST_BASE = 900_000


def main():
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    units, _nets, cw, ch = load_yal(ROOT / "ami33.yal.txt", utilization=UTILIZATION)
    tot = 2.0 * len(units)

    tag = f"{N_ORACLE}" if PAIR_OFFSET == 0 else f"{N_ORACLE}_off{PAIR_OFFSET}"
    out = PKG / f"results/exp048_iid_legal_{tag}.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"i.i.d. oracle gap, legal [ami33 @ {UTILIZATION:.2f} util]: {N_PAIRS} pairs "
         f"(offset {PAIR_OFFSET}), N_ORACLE={N_ORACLE}, {N_ITER} it, fresh eval set per "
         f"pair (N_TEST={N_TEST}), train@{TRAIN_GRID}+jitter -> guaranteed-legal "
         f"eval@{EVAL_GRID} (alpha={EVAL_ALPHA}). "
         + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    st = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    st.build(); st.factorize()

    def fit(tr, mode, jseed):
        pl = DiffPlacer(st, units, cw, ch, TRAIN_GRID, TRAIN_GRID, alpha=EVAL_ALPHA,
                        nonoverlap_w=1e4, density_w=DENSITY_W,
                        density_lam0=DENSITY_LAM0, density_grid=DENSITY_GRID)
        pl.optimize(tr, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=jseed)
        return legalize_units_exact(pl.get_units(), cw, ch)

    def peaks_on(up, scen):
        """Per-scenario peak dT; the RHS is keyed by name since scen_set yields
        arrays in unit order."""
        sv = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
        sv.build(); sv.factorize()
        return np.array([float(sv.silicon_layer(sv.solve(sv.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - ambient
            for pw in scen])

    d_cvar, d_mean = [], []
    ovl_max = 0.0
    infeasible = 0

    for k in range(PAIR_OFFSET, PAIR_OFFSET + N_PAIRS):
        tr = scen_set(units, tot, np.random.default_rng(TRAIN_SEED_BASE + k), N_ORACLE)
        js = JITTER_SEED_BASE + 1000 * k                 # CRN across arms
        try:
            um, fm = fit(tr, "mean", js)
            uc, fc = fit(tr, "cvar", js)
        except LegalizationInfeasible:
            infeasible += 1
            emit(f"  pair {k}: INFEASIBLE, skipped")
            continue
        ovl_max = max(ovl_max, fm, fc)

        fresh = scen_set(units, tot, np.random.default_rng(FRESH_TEST_BASE + k), N_TEST)
        pm, pc = peaks_on(um, fresh), peaks_on(uc, fresh)
        d_cvar.append(cvar(pm, EVAL_ALPHA) - cvar(pc, EVAL_ALPHA))
        d_mean.append(float(pm.mean()) - float(pc.mean()))

        # Pool-readable per-pair lines (one estimate per contrast).
        emit(f"  pair {k}: D*_k = {d_cvar[-1]:+.4f}   dMean_k = {d_mean[-1]:+.4f}")

    g2 = ovl_max < 1e-3 and infeasible == 0
    emit(f"\nG2 (legality): max overlap {100*ovl_max:.3f}%, {infeasible} infeasible -> "
         f"{'PASS' if g2 else 'FAIL'}")

    emit("\n===== VERDICT (this shard; pool across shards) =====")
    if SMOKE:
        emit("NO VERDICT — smoke run (plumbing check only).")
        fh.close(); return
    if not g2 or len(d_cvar) < 2:
        emit("NO VERDICT — legality gate failed or too few feasible pairs.")
        fh.close(); return

    dc, dm = np.array(d_cvar), np.array(d_mean)
    mc, _, lc, hc = ci95_t(dc)
    mm, _, lm, hm = ci95_t(dm)
    emit(f"D*    = {mc:+.4f} [{lc:+.4f},{hc:+.4f}] p={paired_t_p(dc):.4f} sd={dc.std(ddof=1):.4f}")
    emit(f"dMean = {mm:+.4f} [{lm:+.4f},{hm:+.4f}]")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
