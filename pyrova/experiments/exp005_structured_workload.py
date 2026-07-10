"""Structured-workload learnability sweep: does the tail dimension become real & learnable?

exp004 on the STRUCTURED workload (5-mode CPU mixture with anti-correlated FP/INT/MEM
clusters; see workloads/structured.py). The anti-correlation creates a genuine,
separable tail dimension that minimising the mean does not capture, so here CVaR-opt
can beat mean-opt once the tail is learnable: the N_train curve REVERSES positive
rather than merely closing to parity (the i.i.d. case, exp004).

IMPORTANT — this experiment tests mean-opt vs PURE CVaR-opt (eps=0). A positive is
evidence that *a tail dimension exists and pure CVaR can learn it under structure*,
NOT that the Wasserstein-DRO penalty helps; the DRO penalty is tested separately in
exp007. Do not re-label this as a DRO result.

Caveat: the structured workload is hand-designed to contain exactly the
anti-correlation the theory needs, so this proves "structure suffices," not that
real workloads have it. exp006 is the real-trace probe.

Metric: de-confounded dCVaR/dMean cells (= mean-opt minus CVaR-opt) on a large
holdout, identical to exp004 (definitions and reading rule there); dCVaR>0 with
dMean<0 is the mean-for-tail signature.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # pyrova/experiments
PKG = HERE.parent                               # pyrova
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_flp, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import mean_cvar, ci95_t, paired_t_p, holm
from pyrova.workloads.structured import StructuredWorkloadModel

FLP = PKG / "inputs/floorplans/ev6.flp"
CONFIG = PKG / "inputs/configs/thermal.config"
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


def oos_mean_cvar(pl, scen, alpha):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), alpha)


def deltas_at(solver, units, chip_w, chip_h, n_train, alpha):
    """(dCVaR, dMean) with 95% CIs across seeds. dX = mean-opt minus CVaR-opt."""
    dC, dM = [], []
    for seed in range(N_SEEDS):
        # alpha enters the seed so the alpha columns of a row are independent
        # draws rather than one draw presented as three corroborating cells.
        model = StructuredWorkloadModel(
            units, seed=100_000 * seed + 100 * n_train + int(round(alpha * 100)))
        train = model.sample(n_train)
        test = model.sample(N_TEST)
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
    flag = "*" if c_lo > 0 else ("x" if c_hi < 0 else " ")
    return f"{gc:+.2f}{flag}/{gm:+.2f}"


def main():
    units = parse_flp(str(FLP))
    cfg = parse_config(str(CONFIG))
    chip_w, chip_h = chip_box(units)
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()

    out = PKG / "results/exp005_structured_workload.txt"
    fh = open(out, "w")

    def emit(s):
        print(s); fh.write(s + "\n")

    emit(f"Structured-workload learnability sweep on ev6 ({len(units)} blocks). "
         f"N_TEST={N_TEST}, {N_SEEDS} seeds, grid {NR}x{NC}, {N_ITER} iter.")
    emit("Tests mean-opt vs PURE CVaR-opt (eps=0). NOT a DRO result; the Wasserstein "
         "penalty is tested in exp007.")
    emit("cell = dCVaR[flag]/dMean [K], both = mean-opt minus CVaR-opt. "
         "'*' dCVaR CI>0 (CVaR-opt lower tail), 'x' dCVaR CI<0.")
    emit("read together: dCVaR>0 & dMean<0 = genuine mean-for-tail trade (real tail dimension).")
    emit(f"  {'N_TRAIN':>8} | " + " ".join(f"a={a:<11.2f}" for a in ALPHAS))
    emit("  " + "-" * (10 + 14 * len(ALPHAS)))
    rows = []
    for nt in N_TRAINS:
        cells = []
        for a in ALPHAS:
            gc, c_lo, c_hi, gm, m_lo, m_hi, p_c = deltas_at(solver, units, chip_w, chip_h, nt, a)
            cells.append(f"{cell(gc, c_lo, c_hi, gm):>13}")
            rows.append(dict(nt=nt, a=a, gc=gc, lo=c_lo, gm=gm, p=p_c))
            fh.write(f"    # N_TRAIN={nt} a={a}: dCVaR={gc:+.3f} CI[{c_lo:+.3f},{c_hi:+.3f}] "
                     f"p={p_c:.4f}  dMean={gm:+.3f} CI[{m_lo:+.3f},{m_hi:+.3f}]\n")
        emit(f"  {nt:>8} | " + " ".join(cells))

    # Verdict with familywise control: a '>=1 of 15 cells' criterion has ~54%
    # false-positive probability under the global null and is not evidence.
    keep = holm([r["p"] for r in rows])
    sig_pos = [r for r, k in zip(rows, keep) if k and r["gc"] > 0]
    trade = [r for r in sig_pos if r["gm"] < 0]
    if trade:
        lab = ", ".join(f"N={r['nt']},a={r['a']}({r['gc']:+.2f})" for r in trade)
        emit(f"\nVerdict (pure CVaR, not DRO): mean-for-tail trade SUPPORTED after "
             f"Holm correction ({len(rows)} cells) in: {lab}")
    elif sig_pos:
        lab = ", ".join(f"N={r['nt']},a={r['a']}" for r in sig_pos)
        emit(f"\nVerdict (pure CVaR, not DRO): dCVaR>0 survives Holm in {lab}, "
             f"but without dMean<0 — not the mean-for-tail signature.")
    else:
        emit(f"\nVerdict (pure CVaR, not DRO): NO cell survives Holm correction "
             f"({len(rows)} cells) — tail dimension not supported at this power.")
    fh.close()
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
