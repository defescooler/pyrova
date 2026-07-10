"""Multi-program real-workload test: does CVaR placement help on real RISC-V workloads?

Uses the mcpat-calib BOOM dataset: 80 real benchmarks with per-functional-unit McPAT
power on a RISC-V BOOM core — the genuine ACROSS-program workload distribution at
functional-block granularity that exp006 lacked (its only bundled trace was gcc, a
single program). Data is GPL-3.0 and NOT bundled; clone it and set BOOM_DATA (see
pyrova/workloads/boom_traces.py).

Two questions, two confidence levels:

  (1) STRUCTURE (data-only, robust): do real workloads have the anti-correlated functional
      clusters the theory needs? Measured cross-program corr(FP,INT)/(FP,MEM)/(INT,MEM) and
      total-power CV. This needs no floorplan and no placement, so it is the solid result.

  (2) PLACEMENT (geometry-contingent): does CVaR-opt beat mean-opt out-of-sample on these
      real programs? De-confounded dCVaR/dMean (= mean-opt minus CVaR-opt) over disjoint
      train/test splits of the 80 programs. Caveat: the floorplan layout is synthesised from
      McPAT component areas (gridded), so this result depends on a geometry choice and is
      reported with that caveat — do not over-read a point estimate that is not significant.

Priors going in: real corr(FP,INT) is strongly negative (good for the theory) but FP
is thermally light (~1.5% of power) and the hot INT/MEM clusters are positively correlated,
so a placement benefit is not guaranteed even with the right correlation sign.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # pyrova/experiments
PKG = HERE.parent                               # pyrova
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_config
from pyrova.optimizer.placer import DiffPlacer, _build_rhs_at
from pyrova.evaluation.metrics import mean_cvar, ci95_nadeau_bengio
from pyrova.workloads.boom_traces import BoomWorkload, resolve_paths

CONFIG = PKG / "inputs/configs/thermal.config"
CONFIG_ID = "0"
ALPHA = 0.90
NR = NC = 18
N_ITER = 30
N_SPLITS = 10
TRAIN_FRAC = 0.5
TARGET_PEAK = 40.0       # rescale power so mean per-program peak dT ~= this [K] (physical range)


def oos_mean_cvar(pl, scen):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), ALPHA)


def main():
    csvp, rptp = resolve_paths()
    if not csvp:
        print("BOOM_DATA not found. Clone the (GPL-3.0) dataset and retry:\n"
              "  git clone --depth 1 https://github.com/zhaijw18/mcpat-calib-public.git\n"
              "  BOOM_DATA=$(pwd)/mcpat-calib-public python -m pyrova.experiments.exp009_boom_real_traces")
        return                                  # no data -> do not clobber an existing result file

    out = PKG / "results/exp009_boom_real_traces.txt"
    fh = open(out, "w")

    def emit(s):
        print(s); fh.write(s + "\n")

    cfg = parse_config(str(CONFIG))
    wl = BoomWorkload(csvp, rptp, config_id=CONFIG_ID)
    solver = GridFDSolver(cfg, wl.units, wl.chip_w, wl.chip_h, NR, NC)
    solver.build(); solver.factorize()

    def peaks(scen):
        p = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, NR, NC, alpha=ALPHA)
        cx, cy = p.get_positions()
        return p._scenario_peaks(cx, cy, scen)
    wl.scale_to_peak(peaks, TARGET_PEAK)

    emit(f"BOOM real-trace test: {wl.n_programs} programs, {len(wl.leaves)} blocks, config {CONFIG_ID}, "
         f"grid {NR}x{NC}, alpha={ALPHA}.")
    cov_ok = 0.95 <= wl.coverage <= 1.02
    emit(f"power leaf-coverage of Core = {wl.coverage:.3f} "
         f"({'no double-count' if cov_ok else 'SUSPECT — check parent/child components'}); "
         f"chip {wl.chip_w*1e3:.2f}x{wl.chip_h*1e3:.2f} mm (areas from McPAT, layout synthesised).")

    # (1) Structure: measured cross-program correlation (robust, data-only)
    c = wl.family_corr()
    emit("\n(1) REAL cross-program structure (data-only, robust):")
    emit(f"    corr(FP,INT)={c['FP_INT']:+.3f}  corr(FP,MEM)={c['FP_MEM']:+.3f}  "
         f"corr(INT,MEM)={c['INT_MEM']:+.3f}   total-power CV={wl.total_power_cv():.3f}")
    # Cross-config robustness computed HERE (previously an unrecorded ad-hoc claim).
    for cid in ("1", "2", "3"):
        try:
            wc = BoomWorkload(csvp, rptp, config_id=cid)
        except ValueError:
            continue
        cc2 = wc.family_corr()
        emit(f"    config {cid}: corr(FP,INT)={cc2['FP_INT']:+.3f}  "
             f"corr(FP,MEM)={cc2['FP_MEM']:+.3f}  corr(INT,MEM)={cc2['INT_MEM']:+.3f}  "
             f"CV={wc.total_power_cv():.3f}  (same smallboom areas — power only)")
    emit(f"    vs gcc single-program (exp006): corr(FP,MEM)=+0.64 corr(FP,INT)=+0.71 CV=0.057")
    emit(f"    vs exp005 hand-designed: corr(FP,MEM)=-0.35 corr(FP,INT)=-0.28")
    anti_pairs = [lab for lab, k in (("FP/INT", "FP_INT"), ("FP/MEM", "FP_MEM"))
                  if c[k] < -0.1]
    if anti_pairs:
        emit(f"    -> real workloads DO show anti-correlated functional clusters "
             f"({', '.join(anti_pairs)}) — the structure the theory needs "
             f"(gcc was the wrong, single-program test).")
    else:
        emit("    -> real workloads do NOT show anti-correlated functional clusters "
             "(neither FP/INT nor FP/MEM below -0.1).")

    # Mechanism quantities (measured in-script, not asserted from memory)
    fam = np.array(wl.families)
    fp_share = wl.power[:, fam == "FP"].sum() / wl.power.sum()
    fp_share_max = float((wl.power[:, fam == "FP"].sum(1) / wl.power.sum(1)).max())
    base = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, NR, NC, alpha=ALPHA)
    cx0, cy0 = base.get_positions()
    argmaxes = []
    for pw in wl.scenarios():
        T = solver.solve(_build_rhs_at(solver, wl.units, cx0, cy0, pw))
        argmaxes.append(int(np.argmax(solver.silicon_layer(T))))
    _, counts = np.unique(argmaxes, return_counts=True)
    n_moves = int(len(argmaxes) - counts.max())
    emit(f"    mechanism: FP cluster carries {100*fp_share:.1f}% of total power "
         f"(max {100*fp_share_max:.1f}% in any program); hotspot cell is identical in "
         f"{counts.max()}/{len(argmaxes)} programs at the original layout "
         f"({n_moves} programs move it).")

    # (2) Placement: de-confounded mean vs CVaR, disjoint train/test
    scen = wl.scenarios()
    n = len(scen); cut = int(n * TRAIN_FRAC)
    dC, dM = [], []
    for seed in range(N_SPLITS):
        perm = np.random.default_rng(seed).permutation(n)
        tr = [scen[i] for i in perm[:cut]]
        te = [scen[i] for i in perm[cut:]]
        pm = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, NR, NC, alpha=ALPHA)
        pm.optimize(tr, mode="mean", n_iter=N_ITER, lr=2e-2, verbose=False)
        pc = DiffPlacer(solver, wl.units, wl.chip_w, wl.chip_h, NR, NC, alpha=ALPHA)
        pc.optimize(tr, mode="cvar", n_iter=N_ITER, lr=2e-2, verbose=False)
        mm, cm = oos_mean_cvar(pm, te)
        mc, cc = oos_mean_cvar(pc, te)
        dC.append(cm - cc); dM.append(mm - mc)
    ratio = (n - cut) / cut
    g, _, lo, hi = ci95_nadeau_bengio(dC, ratio)
    gm, _, mlo, mhi = ci95_nadeau_bengio(dM, ratio)
    flag = "*" if lo > 0 else ("x" if hi < 0 else "ns")
    emit(f"\n(2) PLACEMENT (geometry-contingent), disjoint {cut}/{n-cut}, {N_SPLITS} repeated "
         f"random splits of ONE 80-program pool:")
    emit(f"    CI = Nadeau-Bengio-corrected t (repeated splits share data; a naive across-split "
         f"t-CI would be ~{np.sqrt(1.0/N_SPLITS + ratio)/np.sqrt(1.0/N_SPLITS):.1f}x too narrow).")
    emit(f"    dCVaR = OOS CVaR(mean-opt) - CVaR(cvar-opt) = {g:+.3f} K  CI[{lo:+.3f},{hi:+.3f}]  {flag}")
    emit(f"    dMean = {gm:+.3f} K  CI[{mlo:+.3f},{mhi:+.3f}]")
    if flag == "*":
        verdict = "CVaR-opt significantly lowers OOS tail risk on real workloads."
    elif flag == "x":
        verdict = "CVaR-opt is significantly WORSE OOS (overfits the real tail)."
    else:
        verdict = (f"no significant benefit (CI spans 0). Consistent with the measured mechanism "
                   f"above: FP carries ~{100*fp_share:.1f}% of power and the hotspot moves in only "
                   f"{n_moves}/{n} programs, so the anti-correlation has little thermal leverage.")
    emit(f"    -> {verdict}")
    emit(f"    POWER CAVEAT: OOS CVaR at alpha={ALPHA} on a {n-cut}-program test set averages the "
         f"top ~{max(1, int((n-cut)*(1-ALPHA)))} programs — far below the N_TEST>=1500 the "
         f"synthetic experiments use; with 80 real programs this is the best available, so treat "
         f"(2) as low-powered, not as a precise null.")
    emit("    NOTE: layout synthesised from McPAT areas; treat (2) as geometry-contingent, (1) as robust.")
    fh.close()
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
