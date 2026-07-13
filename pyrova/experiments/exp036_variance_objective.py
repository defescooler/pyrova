"""Objective x sample-size curve for the i.i.d. tail dimension: per (objective,
N_ORACLE), D* = trueCVaR(mean-oracle) - trueCVaR(arm-oracle), scored at
alpha=0.9 on a 4000-scenario holdout. Arms: cvar (empirical CVaR at
alpha=0.9), blend ((1-g)*mean + g*CVaR, g=0.5), and cvar_wide (CVaR trained at
alpha=0.6, scored at 0.9). Matched budgets, train@18 with raster jitter,
independent 64^2 evaluation, common random numbers across arms within a pair,
Holm across the three objectives within an N cell.

Runs ONE N_ORACLE per invocation (PYROVA_NORACLE or the SLURM array index;
PYROVA_PAIRS overrides the pair count); aggregate the per-N files into the
objective x N curve afterwards.

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
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p, holm
from pyrova.experiments.exp003_mean_cvar_correlation import scen_set, chip_box

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
FLP = PKG / "inputs/floorplans/ev6.flp"
EVAL_ALPHA = 0.90
TRAIN_GRID = 18
EVAL_GRID = 32 if SMOKE else 64
JITTER = 1.0
N_ITER = 12 if SMOKE else 100
N_TEST = 300 if SMOKE else 4000
N_GRID = [200, 400] if SMOKE else [300, 800, 2000, 4000]

# objective -> (mode, train_alpha, blend_gamma)
ARMS = {"cvar": ("cvar", 0.90, 0.5),
        "blend": ("blend", 0.90, 0.5),
        "cvar_wide": ("cvar", 0.60, 0.5)}


def pick(env, default):
    v = os.environ.get(env)
    return int(v) if v else default


def n_oracle_and_pairs():
    if os.environ.get("PYROVA_NORACLE"):
        n = int(os.environ["PYROVA_NORACLE"])
    else:
        idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        n = N_GRID[min(idx, len(N_GRID) - 1)]
    # fewer pairs at large N to fit the walltime
    default_pairs = 2 if SMOKE else (10 if n <= 800 else 8 if n <= 2000 else 6)
    return n, pick("PYROVA_PAIRS", default_pairs)


def fit(solver, units, cw, ch, train, mode, alpha, gamma, jseed, rseed):
    pl = DiffPlacer(solver, units, cw, ch, TRAIN_GRID, TRAIN_GRID,
                    alpha=alpha, blend_gamma=gamma)
    # single start both arms (matched budget); CRN via shared jseed. rseed
    # reserved for restart parity if enabled later.
    pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                raster_jitter=JITTER, jitter_seed=jseed)
    return pl


def eval_true_cvar(cfg, up, scen, cw, ch, ambient):
    s = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
    s.build(); s.factorize()
    pk = np.array([float(s.silicon_layer(s.solve(s.build_rhs(
        {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - ambient
        for pw in scen])
    return cvar(pk, EVAL_ALPHA)


def main():
    n_oracle, n_pairs = n_oracle_and_pairs()
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    units = parse_flp(str(FLP))
    n = len(units)
    cw, ch = chip_box(units)
    solver = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()
    tot = 2.0 * n
    test = scen_set(units, tot, np.random.default_rng(99), N_TEST)

    out = PKG / f"results/exp036_variance_{n_oracle}.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"objective x N cell: ev6 i.i.d., N_ORACLE={n_oracle}, {n_pairs} pairs, "
         f"budget={N_ITER}, train@{TRAIN_GRID}+jitter, eval@{EVAL_GRID} (alpha_eval="
         f"{EVAL_ALPHA}). D* = trueCVaR(mean) - trueCVaR(arm). "
         + ("[SMOKE]" if SMOKE else "[full]"))

    D = {a: [] for a in ARMS}
    for k in range(n_pairs):
        tr = scen_set(units, tot, np.random.default_rng(10_000 + k), n_oracle)
        js, rs = 700_000 + 1000 * k, 800_000 + 1000 * k        # CRN: shared across arms
        c_mean = eval_true_cvar(cfg, fit(solver, units, cw, ch, tr, "mean", 0.9,
                                         0.5, js, rs).get_units(),
                                test, cw, ch, ambient)
        for a, (mode, al, g) in ARMS.items():
            c_a = eval_true_cvar(cfg, fit(solver, units, cw, ch, tr, mode, al, g,
                                          js, rs).get_units(), test, cw, ch, ambient)
            D[a].append(c_mean - c_a)
        print(f"  N={n_oracle} pair {k + 1}/{n_pairs} done", flush=True)

    cells = []
    for a in ARMS:
        arr = np.array(D[a])
        m, _, lo, hi = ci95_t(arr)
        cells.append(dict(arm=a, D=m, lo=lo, hi=hi, p=paired_t_p(arr)))
    keep = holm([c["p"] for c in cells])
    for c, kp in zip(cells, keep):
        c["holm"] = bool(kp)

    emit(f"\n  {'objective':10} {'D*':>9} {'CI_lo':>9} {'CI_hi':>9} {'p':>7}  Holm")
    for c in cells:
        fl = "*" if c["lo"] > 0 else ("x" if c["hi"] < 0 else " ")
        emit(f"  {c['arm']:10} {c['D']:>+9.4f}{fl} {c['lo']:>+9.4f} {c['hi']:>+9.4f} "
             f"{c['p']:>7.4f}  {'sig' if c['holm'] and c['lo'] > 0 else ''}")
    win = [c for c in cells if c["holm"] and c["lo"] > 0]
    if win:
        b = max(win, key=lambda c: c["D"])
        emit(f"\n  N={n_oracle}: '{b['arm']}' beats mean on the true tail "
             f"(D*={b['D']:+.4f} Holm-sig). Plain cvar D*={next(c['D'] for c in cells if c['arm']=='cvar'):+.4f}.")
    else:
        emit(f"\n  N={n_oracle}: no objective clears CI>0 (all overfit at this N).")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
