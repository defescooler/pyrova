"""Learnability sweep (N_train x alpha): does targeting the tail help TRUE tail risk?

Sweeps the small-N -> large-N curve for mean-opt vs CVaR-opt on the i.i.d. synthetic
workload. The scientific story is the SHAPE of the curve in N_train (does a CVaR
benefit appear, persist, or collapse), not any single cell.

De-confounded metric (review): the old single
    gap = OOS CVaR(mean-opt) - OOS CVaR(CVaR-opt)
conflates (A) scoring CVaR-opt on the very functional it minimised with (B) CVaR-opt
overfitting the noisy empirical tail. Each cell instead reports TWO deltas on a
large holdout (OOS ~= true):

    dCVaR = OOS CVaR(mean-opt) - OOS CVaR(CVaR-opt)   (>0 => CVaR-opt has lower tail)
    dMean = OOS mean(mean-opt) - OOS mean(CVaR-opt)   (<0 => CVaR-opt pays mean)

Read together: dCVaR>0 with dMean<0 is a genuine mean-for-tail trade; dCVaR<=0 with
dMean<=0 is CVaR-opt dominated (overfitting on both); dCVaR<0 means CVaR-opt is
worse even on its own metric. Expectation on this i.i.d. workload: overfitting —
CVaR-opt reaches parity only as N grows, never a benefit. Structured analogue:
exp005; overfitting-free oracle existence test: exp003.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # pyrova/experiments
PKG = HERE.parent                               # pyrova
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_flp, parse_config, random_power_map
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import mean_cvar, ci95_t, paired_t_p, holm

CONFIG = PKG / "inputs/configs/thermal.config"
BENCHES = [PKG / "inputs/floorplans/ev6.flp",
           ROOT / "Tools/HotSpot/examples/example3/floorplan2.flp"]
ALPHAS = [0.80, 0.90, 0.95]
N_TRAINS = [16, 32, 64, 128, 256]
N_TEST = 1500          # large holdout so OOS ~= true (larger -> closer)
N_SEEDS = 5
NR = NC = 18
N_ITER = 30


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def scen_set(units, tot, rng, k):
    return [np.array([random_power_map(units, tot, rng)[u["name"]] for u in units])
            for _ in range(k)]


def oos_mean_cvar(pl, scen, alpha):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), alpha)


def deltas_at(solver, units, chip_w, chip_h, tot, n_train, alpha):
    """(dCVaR, dMean) with 95% CIs across seeds + paired-t p for dCVaR.
    dX = mean-opt minus CVaR-opt. The cell seed includes alpha so the three
    alpha columns of a row are independent draws, not one draw shown thrice."""
    dC, dM = [], []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(100_000 * seed + 100 * n_train + int(round(alpha * 100)))
        train = scen_set(units, tot, rng, n_train)
        test = scen_set(units, tot, rng, N_TEST)
        p_mean = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=alpha)
        p_mean.optimize(train, mode="mean", n_iter=N_ITER, lr=2e-2, verbose=False)
        p_cvar = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=alpha)
        p_cvar.optimize(train, mode="cvar", n_iter=N_ITER, lr=2e-2, verbose=False)
        mm, cm = oos_mean_cvar(p_mean, test, alpha)
        mc, cc = oos_mean_cvar(p_cvar, test, alpha)
        dC.append(cm - cc); dM.append(mm - mc)
    gc, _, c_lo, c_hi = ci95_t(dC)
    gm, _, m_lo, m_hi = ci95_t(dM)
    return gc, c_lo, c_hi, gm, m_lo, m_hi, paired_t_p(dC)


def cell(gc, c_lo, c_hi, gm) -> str:
    flag = "*" if c_lo > 0 else ("x" if c_hi < 0 else " ")   # CVaR significance
    return f"{gc:+.2f}{flag}/{gm:+.2f}"                       # dCVaR[flag]/dMean


def run(path: Path, cfg, fh) -> None:
    units = parse_flp(str(path))
    n = len(units)
    chip_w, chip_h = chip_box(units)
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()
    tot = 2.0 * n

    def emit(s):
        print(s); fh.write(s + "\n")

    emit(f"\n=== {path.stem} ({n} blocks) ===")
    emit("cell = dCVaR[flag]/dMean [K], both = mean-opt minus CVaR-opt. "
         "'*' dCVaR CI>0 (CVaR-opt lower tail), 'x' dCVaR CI<0 (worse on its own metric).")
    emit("read together: dCVaR>0 & dMean<0 = genuine trade; both<=0 = overfit/dominated.")
    emit(f"  {'N_TRAIN':>8} | " + " ".join(f"a={a:<11.2f}" for a in ALPHAS))
    emit("  " + "-" * (10 + 14 * len(ALPHAS)))
    rows = []
    for nt in N_TRAINS:
        cells = []
        for a in ALPHAS:
            gc, c_lo, c_hi, gm, m_lo, m_hi, p_c = deltas_at(solver, units, chip_w, chip_h, tot, nt, a)
            cells.append(f"{cell(gc, c_lo, c_hi, gm):>13}")
            rows.append(dict(nt=nt, a=a, gc=gc, lo=c_lo, hi=c_hi, p=p_c))
            fh.write(f"    # {path.stem} N_TRAIN={nt} a={a}: "
                     f"dCVaR={gc:+.3f} CI[{c_lo:+.3f},{c_hi:+.3f}] p={p_c:.4f}  "
                     f"dMean={gm:+.3f} CI[{m_lo:+.3f},{m_hi:+.3f}]\n")
        emit(f"  {nt:>8} | " + " ".join(cells))

    # Familywise-corrected reading over the len(rows)-cell family: per-cell '*'
    # at 95% has ~54% chance of >=1 false positive across 15 cells under the
    # global null; only Holm-surviving cells count as significant claims.
    keep = holm([r["p"] for r in rows])
    sig = [f"N={r['nt']},a={r['a']}({r['gc']:+.2f})" for r, k in zip(rows, keep)
           if k and r["gc"] > 0]
    neg = [f"N={r['nt']},a={r['a']}({r['gc']:+.2f})" for r, k in zip(rows, keep)
           if k and r["gc"] < 0]
    emit(f"  Holm ({len(rows)} cells): dCVaR>0 surviving: {', '.join(sig) if sig else 'NONE'}; "
         f"dCVaR<0 surviving: {', '.join(neg) if neg else 'none'}")


def main():
    cfg = parse_config(str(CONFIG))
    out = PKG / "results/exp004_tail_sample_sweep.txt"
    print(f"Tail sample / alpha learnability sweep (N_TEST={N_TEST}, {N_SEEDS} seeds, "
          f"grid {NR}x{NC}, {N_ITER} iter, i.i.d. synthetic workload). De-confounded: "
          f"each cell reports dCVaR and dMean side by side.")
    with open(out, "w") as fh:
        fh.write(f"Tail sample / alpha learnability sweep. N_TEST={N_TEST}, {N_SEEDS} seeds, "
                 f"grid {NR}x{NC}, {N_ITER} iter, i.i.d. synthetic workload.\n")
        fh.write("Metric: dCVaR/dMean side by side (= mean-opt minus CVaR-opt) on a large "
                 "holdout. Replaces the old one-sided OOS-CVaR gap, which conflated scoring "
                 "CVaR-opt on its own metric with tail overfitting.\n")
        for p in BENCHES:
            if p.exists():
                run(p, cfg, fh)
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
