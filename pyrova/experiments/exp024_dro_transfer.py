"""exp024 (ledger P6): does the corrected-DRO advantage survive trap-free
measurement?

exp017's positive (dro_exact beats pure CVaR on the hand-built family:
+0.289* at N=64, +0.370* at N=128, Holm) was measured with training AND
evaluation at 18x18 — the protocol exp020 showed to be artifact-prone, and
exp022's fix now makes a clean measurement cheap. Both arms here train at
18x18 WITH rasterization jitter (exp022), at the matched 120-iteration budget
(exp013/exp015), with eps selected per seed by 2-fold CV in Mahalanobis
sigma-units and fractional-tail training (exp014's corrections); ALL
evaluation at 64x64.

PRE-REGISTERED READING (Holm over the two N cells on vs_cvar0):
  - SURVIVES if vs_cvar0 = CVaR(pure-CVaR) - CVaR(dro_exact) has CI > 0 in a
    Holm-surviving cell: claim 7 upgrades to trap-free (still family-scoped
    per exp017).
  - ARTIFACT if no cell survives (CI <= 0 or Holm-killed): claim 7's
    hand-built positive joins claims 6/11 as a matched-grid artifact and DRO
    closes with no surviving positive.
"""

from __future__ import annotations
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
from pyrova.workloads.structured import StructuredWorkloadModel

FLP = PKG / "inputs/floorplans/ev6.flp"
CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA = 0.90
TRAIN_GRID = 18
EVAL_GRID = 64
N_ITER = 120
N_SEEDS = 8
N_TRAINS = [64, 128]
N_TEST = 500
EPS_GRID = [0.1, 0.25, 0.5]
JITTER = 1.0


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def fit(solver, units, chip_w, chip_h, train, mode, eps=0.0, sigma=None, seed=0):
    pl = DiffPlacer(solver, units, chip_w, chip_h, TRAIN_GRID, TRAIN_GRID,
                    alpha=ALPHA, eps_dro=eps, dro_sigma=sigma)
    w = np.full(len(train), 1.0 / len(train))      # fractional tail: exact alpha
    pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                weights=w, raster_jitter=JITTER, jitter_seed=920_000 + seed)
    return pl


def eval_cvar(cfg, units_placed, scen, chip_w, chip_h, ambient) -> float:
    s = GridFDSolver(cfg, units_placed, chip_w, chip_h, EVAL_GRID, EVAL_GRID)
    s.build(); s.factorize()
    pk = np.zeros(len(scen))
    for i, pw in enumerate(scen):
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(units_placed)}
        T = s.solve(s.build_rhs(bp))
        pk[i] = float(s.silicon_layer(T).max()) - ambient
    return cvar(pk, ALPHA)


def cv_pick_eps(solver, units, chip_w, chip_h, train, sigma, rng,
                cfg, ambient) -> float:
    """2-fold CV on the training set (fine-grid scoring of held-out CVaR)."""
    idx = rng.permutation(len(train))
    folds = [[train[i] for i in idx[::2]], [train[i] for i in idx[1::2]]]
    best_eps, best = 0.0, np.inf
    for eps in EPS_GRID:
        score = 0.0
        for a, b in ((0, 1), (1, 0)):
            pl = fit(solver, units, chip_w, chip_h, folds[a], "dro_exact",
                     eps=eps, sigma=sigma)
            score += eval_cvar(cfg, pl.get_units(), folds[b], chip_w, chip_h, ambient)
        if score < best:
            best, best_eps = score, eps
    return best_eps


def main():
    units = parse_flp(str(FLP))
    cfg = parse_config(str(CONFIG))
    chip_w, chip_h = chip_box(units)
    ambient = cfg["ambient"]
    solver = GridFDSolver(cfg, units, chip_w, chip_h, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()

    out = PKG / "results/exp024_dro_transfer.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"exp024 (P6): trap-free re-measurement of the corrected-DRO advantage. "
         f"Hand-built family, N in {N_TRAINS}, {N_SEEDS} seeds, alpha={ALPHA}; "
         f"train@{TRAIN_GRID}+jitter, {N_ITER} it, eps by 2-fold CV {EPS_GRID} "
         f"(sigma-units, fine-grid scored); eval@{EVAL_GRID}, N_TEST={N_TEST}.")
    emit("vs_cvar0 = CVaR(pure-CVaR) - CVaR(dro_exact) at eval@64; Holm over the 2 cells.")

    cells = []
    for n in N_TRAINS:
        d0, eps_picked = [], []
        for seed in range(N_SEEDS):
            model = StructuredWorkloadModel(
                units, seed=500_000 + 10_000 * seed + n)
            train = model.sample(n)
            test = model.sample(1500)[:N_TEST]
            sigma = np.asarray(train).std(axis=0, ddof=1) + 1e-12
            eps = cv_pick_eps(solver, units, chip_w, chip_h, train, sigma,
                              np.random.default_rng(510_000 + seed), cfg, ambient)
            eps_picked.append(eps)
            c_c = eval_cvar(cfg, fit(solver, units, chip_w, chip_h, train, "cvar",
                                     seed=seed).get_units(),
                            test, chip_w, chip_h, ambient)
            c_d = eval_cvar(cfg, fit(solver, units, chip_w, chip_h, train, "dro_exact",
                                     eps=eps, sigma=sigma, seed=seed).get_units(),
                            test, chip_w, chip_h, ambient)
            d0.append(c_c - c_d)
            print(f"  [N={n}] seed {seed + 1}/{N_SEEDS} eps={eps:g} "
                  f"d={d0[-1]:+.3f}", flush=True)
        g, _, lo, hi = ci95_t(d0)
        p = paired_t_p(d0)
        flag = "*" if lo > 0 else ("x" if hi < 0 else " ")
        emit(f"  N={n}: vs_cvar0={g:+.3f}{flag} [{lo:+.3f},{hi:+.3f}] p={p:.4f}  "
             f"eps(CV)={eps_picked}")
        cells.append(dict(n=n, g=g, lo=lo, p=p))

    keep = holm([c["p"] for c in cells])
    surv = [c for c, k in zip(cells, keep) if k and c["g"] > 0]
    emit("\nPRE-REGISTERED VERDICT: " + (
        f"SURVIVES trap-free measurement at N={[c['n'] for c in surv]} — claim 7 "
        f"upgrades (still family-scoped per exp017)."
        if surv else
        "ARTIFACT — no Holm-surviving positive under jittered training and "
        "fine-grid evaluation; claim 7's hand-built positive joins claims 6/11 "
        "as a matched-grid artifact, and DRO closes with no surviving positive."))
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
