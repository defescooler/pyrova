"""How much workload structure does CVaR placement need? — a correlation-threshold sweep.

exp005 showed pure-CVaR placement beats the mean under ONE hand-designed
anti-correlated workload (measured corr(FP,MEM)=-0.35); exp006 found the only real
trace (gcc) positively correlated (+0.64) with a null. This sweeps the cross-cluster
correlation to find the threshold corr* where CVaR stops helping, then locates gcc's
+0.64 relative to it.

Workload: `CorrelatedWorkloadModel` (workloads/structured.py) — a DISCRETE-mode model (so the
CVaR tail is a sharp, learnable hotspot, unlike a continuous model whose tail is noise) with
a knob `mix` in [0,1] interpolating from CONTRAST modes (one cluster hot at a time -> clusters
anti-correlate, mix=0) to COMMON modes (clusters co-activate -> positive correlation, mix=1).
Realized corr(FP,MEM)/corr(FP,INT) are MEASURED and form the x-axis; `mix` is only the knob.

Metric: de-confounded dCVaR/dMean per mix (= mean-opt minus CVaR-opt) on a large
holdout with 95% CIs, at a learnable N, as exp004/exp005 (definitions there).

VALIDITY GATE (ENFORCED IN CODE, not just documented). The sweep is only trustworthy if it
reproduces the known endpoints: the anti-correlated anchor (mix=0) must give dCVaR
significantly > 0 (the exp005 result), and the co-activated anchor (mix=1) must give
dCVaR <= 0 (no separable tail dimension). If either fails, the script prints GATE FAILED
and refuses to emit a threshold interpretation.

KNOWN CONFOUND (measured and printed per mix): `mix` also moves the total-power CV
(anti-correlated modes nearly cancel in the total) and the mean hot-block power, partly
intrinsically. The x-axis is really "mode-family composition", not correlation in
isolation — any finding is about the model's mode families, not correlation per se.
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
from pyrova.evaluation.metrics import mean_cvar, ci95_t
from pyrova.workloads.structured import CorrelatedWorkloadModel, _family

FLP = PKG / "inputs/floorplans/ev6.flp"
CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA = 0.90
N_TRAIN = 128          # learnable regime (exp005: dCVaR significant at N>=64)
N_TEST = 1500          # large holdout so OOS ~= true
N_SEEDS = 8
NR = NC = 18
N_ITER = 30
# mix=0 -> CONTRAST only (anti-correlated); mix=1 -> COMMON only (co-activated).
MIXES = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def oos_mean_cvar(pl, scen):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), ALPHA)


def measured_corr(units, mix):
    """Realized corr(FP,MEM), corr(FP,INT) from a large sample at this mix."""
    batch = np.array(CorrelatedWorkloadModel(units, mix, seed=12345).sample(2000))
    C = np.corrcoef(batch.T)
    fams = [_family(u["name"]) for u in units]
    idx = {f: [i for i, fa in enumerate(fams) if fa == f] for f in ("FP", "INT", "MEM")}
    fm = float(np.nanmean(C[np.ix_(idx["FP"], idx["MEM"])]))
    fi = float(np.nanmean(C[np.ix_(idx["FP"], idx["INT"])]))
    return fm, fi


def deltas_at(solver, units, chip_w, chip_h, mix):
    """(dCVaR, dMean) with 95% CIs across seeds at this mix. dX = mean-opt minus CVaR-opt."""
    dC, dM = [], []
    off = int(round(mix * 100))                  # non-negative model seed offset
    for seed in range(N_SEEDS):
        model = CorrelatedWorkloadModel(units, mix, seed=1000 * seed + off)
        train = model.sample(N_TRAIN)
        test = model.sample(N_TEST)
        p_mean = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=ALPHA)
        p_mean.optimize(train, mode="mean", n_iter=N_ITER, lr=2e-2, verbose=False)
        p_cvar = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=ALPHA)
        p_cvar.optimize(train, mode="cvar", n_iter=N_ITER, lr=2e-2, verbose=False)
        mm, cm = oos_mean_cvar(p_mean, test)
        mc, cc = oos_mean_cvar(p_cvar, test)
        dC.append(cm - cc); dM.append(mm - mc)
    gc, _, c_lo, c_hi = ci95_t(dC)
    gm, _, m_lo, m_hi = ci95_t(dM)
    return gc, c_lo, c_hi, gm, m_lo, m_hi


def main():
    units = parse_flp(str(FLP))
    cfg = parse_config(str(CONFIG))
    chip_w, chip_h = chip_box(units)
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()

    out = PKG / "results/exp008_correlation_threshold.txt"
    fh = open(out, "w")

    def emit(s):
        print(s); fh.write(s + "\n")

    emit(f"Correlation-threshold sweep on ev6 ({len(units)} blocks). N_TRAIN={N_TRAIN}, "
         f"alpha={ALPHA}, N_TEST={N_TEST}, {N_SEEDS} seeds, grid {NR}x{NC}, {N_ITER} iter.")
    emit("mix = CONTRAST(anti-corr,0) -> COMMON(co-activated,1); corr(FP,MEM)/corr(FP,INT) MEASURED.")
    emit("dCVaR/dMean = mean-opt minus CVaR-opt. '*' dCVaR CI>0 (CVaR helps), 'x' dCVaR CI<0.")
    emit("totCV / hotW = measured total-power CV and mean hottest-block power per mix — the")
    emit("quantities the mix knob CO-VARIES with correlation (confound; see module docstring).")
    emit(f"  {'mix':>4} | {'corrFM':>7} {'corrFI':>7} | {'totCV':>6} {'hotW':>6} | "
         f"{'dCVaR':>8} {'  95% CI':>16} | {'dMean':>7}")
    emit("  " + "-" * 76)

    rows = []
    for mix in MIXES:
        fm, fi = measured_corr(units, mix)
        st = CorrelatedWorkloadModel(units, mix, seed=12345).mix_stats()
        gc, c_lo, c_hi, gm, m_lo, m_hi = deltas_at(solver, units, chip_w, chip_h, mix)
        flag = "*" if c_lo > 0 else ("x" if c_hi < 0 else " ")
        rows.append(dict(mix=mix, fm=fm, fi=fi, gc=gc, lo=c_lo, hi=c_hi, gm=gm,
                         cv=st["total_cv"], hot=st["mean_hot_block_w"]))
        emit(f"  {mix:>4.2f} | {fm:>7.3f} {fi:>7.3f} | {st['total_cv']:>6.3f} "
             f"{st['mean_hot_block_w']:>6.1f} | {gc:>+7.2f}{flag} "
             f"[{c_lo:>+6.2f},{c_hi:>+6.2f}] | {gm:>+7.2f}")

    # Pre-registered validity gates (BOTH enforced)
    anti, common = rows[0], rows[-1]
    gate_anti = anti["lo"] > 0                    # must reproduce exp005's positive
    gate_common = not (common["lo"] > 0)          # must NOT be significantly positive
    emit("")
    emit(f"GATE 1 anti-corr anchor (mix=0, corr={anti['fm']:+.2f}): dCVaR={anti['gc']:+.2f} "
         f"[{anti['lo']:+.2f},{anti['hi']:+.2f}] -> {'PASS' if gate_anti else 'FAIL'} "
         f"(requires CI>0, the exp005 positive)")
    emit(f"GATE 2 co-activated anchor (mix=1, corr={common['fm']:+.2f}): dCVaR={common['gc']:+.2f} "
         f"[{common['lo']:+.2f},{common['hi']:+.2f}] -> {'PASS' if gate_common else 'FAIL'} "
         f"(requires dCVaR NOT significantly >0: no tail dimension when clusters co-activate)")

    if not (gate_anti and gate_common):
        emit("GATE FAILED -> the threshold reading is NOT to be trusted (pre-registered rule).")
        if not gate_common:
            emit("Gate 2 failing means the model has a learnable tail dimension even with fully "
                 "co-activated clusters — the 'benefit at every correlation' pattern is then a "
                 "property OF THE MODEL (its discrete-mode structure and/or the total-power-CV "
                 "confound above), not evidence about correlation. Redesign the COMMON family "
                 "(or explain its residual structure) before interpreting the sweep.")
        fh.close()
        print(f"Wrote {out.relative_to(ROOT)}")
        return

    # Interpretation (only reachable with both gates passed)
    sig_help = [r for r in rows if r["lo"] > 0]
    clean = [r for r in rows if r["lo"] > 0 and r["gm"] >= 0]   # better on BOTH tail and mean
    peak = max(rows, key=lambda r: r["gc"])
    seq = [r["gc"] for r in rows]
    decays_monotone = all(a >= b for a, b in zip(seq, seq[1:]))
    if len(sig_help) == len(rows):
        shape = ("decays monotonically as correlation rises" if decays_monotone else
                 f"is NON-monotone (peak +{peak['gc']:.2f} K at corr={peak['fm']:+.2f}, "
                 f"not at the most anti-correlated point)")
        emit(f"FINDING: no cell with dCVaR CI<=0 in this sweep; the benefit {shape} and is still "
             f"+{common['gc']:.2f} at corr={common['fm']:+.2f}.")
    else:
        edge = max((r for r in sig_help), key=lambda r: r["fm"], default=None)
        emit(f"CVaR significantly helps up to corr(FP,MEM)={edge['fm']:+.3f}." if edge else
             "CVaR does not significantly help anywhere in the sweep.")
    if clean:
        emit(f"De-confounded reading: the cleanest win (lower tail AND mean) sits at corr in "
             f"[{min(r['fm'] for r in clean):+.2f},{max(r['fm'] for r in clean):+.2f}]; "
             f"strong anti-correlation buys tail at a mean cost (mix=0 dMean={anti['gm']:+.2f}).")
    emit("SCOPE: one chip, a self-designed discrete-mode model built from exp005's own structure, "
         "and an x-axis that co-varies total-power CV and hot-block power with correlation "
         "(columns above). Defensible claim: 'within this model, CVaR helps across the swept "
         "compositions.' NOT defensible: any general law about correlation gating the benefit.")
    fh.close()
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
