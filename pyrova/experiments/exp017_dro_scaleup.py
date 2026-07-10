"""exp017: scale-up of exp014's DRO positive, per the pre-registered stay.

exp014 found the corrected penalty (certified global Lambda, Mahalanobis sigma,
fractional tail, eps by CV) beats pure CVaR at structured N=64: vs_cvar0
+0.122* [+0.045,+0.198], 5 seeds, 30 iterations. The stay's conditions before
"DRO helps under structure" becomes a claim:

  (i)   more seeds (8 here vs 5),
  (ii)  a second N (128 — the exp005 learnable regime),
  (iii) convergence-matched budgets (120 iterations, exp013/exp015 standard),
  (iv)  a SECOND structured family with no designed cluster semantics
        (RandomModesWorkloadModel, two family seeds) — the exp005 model is
        hand-built, so a positive confined to it would be circular.

PRE-REGISTERED READING (Holm over the 6 (family, N) cells on vs_cvar0):
  - CLAIM CONFIRMED if >=1 hand-built cell AND >=1 random-family cell survive
    Holm with vs_cvar0 > 0: "under multimodal structure, the corrected
    Wasserstein-CVaR penalty improves out-of-sample tail risk over pure CVaR."
  - PARTIAL if only hand-built cells survive: the positive is real but may be
    a property of the designed family; say so.
  - WITHDRAWN if no cell survives Holm: exp014's cell was a small-sample
    positive; DRO closes.
Confound rule: family_stats printed per family (exp008 standard).
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
from pyrova.evaluation.metrics import mean_cvar, ci95_t, paired_t_p, holm
from pyrova.workloads.structured import StructuredWorkloadModel, RandomModesWorkloadModel

FLP = PKG / "inputs/floorplans/ev6.flp"
CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA = 0.90
NR = NC = 18
N_ITER = 120                     # matched high budget (exp013/exp015)
N_SEEDS = 8
N_TRAINS = [64, 128]
N_TEST = 1500
EPS_GRID = [0.1, 0.25, 0.5]      # sigma-units; exp014's CV chose 0.25 at N=64
FAMILY_SEEDS = [1, 2]            # two random-modes families


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def make_model(family: str, units, seed: int):
    if family == "hand-built":
        return StructuredWorkloadModel(units, seed=seed)
    fam_seed = int(family.split(":")[1])
    return RandomModesWorkloadModel(units, family_seed=fam_seed, seed=seed)


def fit(solver, units, chip_w, chip_h, train, mode, eps=0.0, sigma=None):
    pl = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=ALPHA,
                    eps_dro=eps, dro_sigma=sigma)
    w = np.full(len(train), 1.0 / len(train))    # fractional tail: exact alpha
    pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                weights=None if mode == "mean" else w)
    return pl


def oos(pl, scen):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), ALPHA)


def cv_pick_eps(solver, units, chip_w, chip_h, train, sigma, rng) -> float:
    """2-fold CV on the training set: eps minimising held-out CVaR."""
    idx = rng.permutation(len(train))
    folds = [[train[i] for i in idx[::2]], [train[i] for i in idx[1::2]]]
    best_eps, best = 0.0, np.inf
    for eps in EPS_GRID:
        score = 0.0
        for a, b in ((0, 1), (1, 0)):
            pl = fit(solver, units, chip_w, chip_h, folds[a], "dro_exact",
                     eps=eps, sigma=sigma)
            score += oos(pl, folds[b])[1]
        if score < best:
            best, best_eps = score, eps
    return best_eps


def main():
    units = parse_flp(str(FLP))
    cfg = parse_config(str(CONFIG))
    chip_w, chip_h = chip_box(units)
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()

    out = PKG / "results/exp017_dro_scaleup.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    families = ["hand-built"] + [f"random:{fs}" for fs in FAMILY_SEEDS]
    emit(f"exp017: DRO scale-up on ev6 — {N_SEEDS} seeds, N in {N_TRAINS}, "
         f"{N_ITER} iter (matched budget), eps by 2-fold CV in {EPS_GRID} sigma-units, "
         f"alpha={ALPHA}, N_TEST={N_TEST}, grid {NR}x{NC}.")
    emit(f"families: {families}; vs_cvar0 = CVaR(pure-CVaR) - CVaR(dro_exact); "
         f"Holm over the {len(families) * len(N_TRAINS)} (family,N) cells.")
    for fam in families:
        st = make_model(fam, units, seed=0)
        stats = (st.family_stats() if hasattr(st, "family_stats")
                 else {"total_cv": None})
        emit(f"  {fam} stats: " + "  ".join(f"{k}={v:.3f}" for k, v in stats.items()
                                            if v is not None))

    cells = []
    for fam in families:
        for n in N_TRAINS:
            d0 = []
            for seed in range(N_SEEDS):
                base = 400_000 + 10_000 * (families.index(fam) + 1)
                model = make_model(fam, units, seed=base + 100 * seed + n)
                train = model.sample(n)
                test = make_model(fam, units, seed=base + 777).sample(N_TEST)
                sigma = np.asarray(train).std(axis=0, ddof=1) + 1e-12
                eps = cv_pick_eps(solver, units, chip_w, chip_h, train, sigma,
                                  np.random.default_rng(base + 900 + seed))
                c_c = oos(fit(solver, units, chip_w, chip_h, train, "cvar"), test)[1]
                c_d = oos(fit(solver, units, chip_w, chip_h, train, "dro_exact",
                              eps=eps, sigma=sigma), test)[1]
                d0.append(c_c - c_d)
                print(f"  [{fam} N={n}] seed {seed+1}/{N_SEEDS} eps={eps:g}", flush=True)
            g, _, lo, hi = ci95_t(d0)
            p = paired_t_p(d0)
            flag = "*" if lo > 0 else ("x" if hi < 0 else " ")
            emit(f"  [{fam} N={n}] vs_cvar0={g:+.3f}{flag} [{lo:+.3f},{hi:+.3f}] p={p:.4f}")
            cells.append(dict(fam=fam, n=n, g=g, lo=lo, p=p))

    keep = holm([c["p"] for c in cells])
    surv = [c for c, k in zip(cells, keep) if k and c["g"] > 0]
    hand = [c for c in surv if c["fam"] == "hand-built"]
    rand = [c for c in surv if c["fam"] != "hand-built"]
    emit(f"\nHolm survivors (vs_cvar0>0): "
         + (", ".join(f"{c['fam']} N={c['n']} ({c['g']:+.3f})" for c in surv) or "NONE"))
    if hand and rand:
        v = ("CLAIM CONFIRMED: the corrected Wasserstein-CVaR penalty improves OOS tail "
             "risk over pure CVaR under multimodal structure, in both the hand-built and "
             "a random mode family.")
    elif surv:
        v = ("PARTIAL: survivors only in "
             + ("the hand-built family" if hand else "random families")
             + " — the positive may be family-specific; scope the claim accordingly.")
    else:
        v = ("WITHDRAWN: no cell survives Holm at the scaled design — treat exp014's "
             "N=64 cell as a small-sample positive; DRO closes.")
    emit(f"PRE-REGISTERED VERDICT: {v}")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
