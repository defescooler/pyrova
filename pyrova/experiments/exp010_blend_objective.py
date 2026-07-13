"""Blend-objective comparison: J_gamma = (1-gamma)*mean + gamma*CVaR_alpha,
gamma in {0, 0.25, 0.5, 0.75, 1}, paired within seed/split. Arms: structured
and i.i.d. synthetic on ev6 (N_train in {16, 32, 128}, 5 seeds, 1500-scenario
holdout, 95% t-CIs) and BOOM real workloads (10 repeated 40/40 splits,
Nadeau-Bengio-corrected CIs; needs BOOM_DATA). Per (arm, N, gamma): OOS
mean/CVaR plus paired vs_mean = CVaR(gamma=0) - CVaR(gamma) and
vs_cvar1 = CVaR(gamma=1) - CVaR(gamma), same estimator training and scoring.
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
from pyrova.evaluation.metrics import mean_cvar, ci95_t, ci95_nadeau_bengio
from pyrova.workloads.structured import StructuredWorkloadModel
from pyrova.workloads.boom_traces import BoomWorkload, resolve_paths

FLP = PKG / "inputs/floorplans/ev6.flp"
CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA = 0.90
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]     # 0 == mean-opt, 1 == pure CVaR-opt
N_TRAINS = [16, 32, 128]
N_TEST = 1500
N_SEEDS = 5
NR = NC = 18
N_ITER = 30
N_SPLITS_BOOM = 10
TRAIN_FRAC_BOOM = 0.5
TARGET_PEAK = 40.0


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def iid_set(units, tot, rng, k):
    return [np.array([random_power_map(units, tot, rng)[u["name"]] for u in units])
            for _ in range(k)]


def fit_gamma(solver, units, chip_w, chip_h, train, gamma):
    mode = "mean" if gamma == 0.0 else ("cvar" if gamma == 1.0 else "blend")
    pl = DiffPlacer(solver, units, chip_w, chip_h, NR, NC,
                    alpha=ALPHA, blend_gamma=gamma)
    pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False)
    return pl


def oos(pl, scen):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), ALPHA)


def report_rows(emit, means, cvars, ci_fn):
    """Table + paired deltas vs both endpoints. means/cvars: {gamma: np.array}."""
    verdicts = {}
    emit(f"    {'gamma':>6}{'OOSmean':>9}{'OOSCVaR':>9}  {'vs_mean':>13}  {'vs_cvar1':>13}")
    for g in GAMMAS:
        vs_mean = cvars[0.0] - cvars[g]
        vs_cv1 = cvars[1.0] - cvars[g]
        gm, _, gm_lo, gm_hi = ci_fn(vs_mean)
        g1, _, g1_lo, g1_hi = ci_fn(vs_cv1)
        fm = "*" if gm_lo > 0 else ("x" if gm_hi < 0 else " ")
        f1 = "*" if g1_lo > 0 else ("x" if g1_hi < 0 else " ")
        s_m = "     -   " if g == 0.0 else f"{gm:+.2f}{fm}"
        s_1 = "     -   " if g == 1.0 else f"{g1:+.2f}{f1}"
        emit(f"    {g:>6.2f}{means[g].mean():>9.3f}{cvars[g].mean():>9.3f}  "
             f"{s_m:>13}  {s_1:>13}")
        verdicts[g] = dict(vs_mean_lo=gm_lo, vs_cv1_lo=g1_lo, vs_cv1_hi=g1_hi)
    return verdicts


def run_synth(arm_name, sampler, solver, units, chip_w, chip_h, test, emit):
    """sampler(seed, n) -> train list. Paired across gammas within a seed."""
    emit(f"\n=== arm: {arm_name} ===")
    conclusions = {}
    for n in N_TRAINS:
        means = {g: [] for g in GAMMAS}
        cvars = {g: [] for g in GAMMAS}
        for seed in range(N_SEEDS):
            train = sampler(seed, n)
            for g in GAMMAS:
                pl = fit_gamma(solver, units, chip_w, chip_h, train, g)
                m, c = oos(pl, test)
                means[g].append(m); cvars[g].append(c)
        means = {g: np.asarray(v) for g, v in means.items()}
        cvars = {g: np.asarray(v) for g, v in cvars.items()}
        emit(f"  N_TRAIN={n}  ({N_SEEDS} seeds, N_TEST={N_TEST}, alpha={ALPHA})")
        conclusions[n] = report_rows(emit, means, cvars, ci_fn=ci95_t)
    return conclusions


def main():
    units = parse_flp(str(FLP))
    cfg = parse_config(str(CONFIG))
    chip_w, chip_h = chip_box(units)
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()
    tot = 2.0 * len(units)

    out = PKG / "results/exp010_blend_objective.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s); fh.write(s + "\n")

    emit(f"exp010: mean-anchored blend objective (1-g)*mean + g*CVaR on ev6 "
         f"({len(units)} blocks), grid {NR}x{NC}, {N_ITER} iter, gammas={GAMMAS}.")
    emit("vs_mean = CVaR(g=0) - CVaR(g); vs_cvar1 = CVaR(g=1) - CVaR(g); both >0 = g wins. "
         "'*' paired CI>0, 'x' CI<0.")
    emit("Pre-registered: (a) structured N<=32: some 0<g<1 with vs_cvar1 CI>0; "
         "(b) structured N=128: none; (c) iid: no g>0 with vs_mean CI>0; "
         "(d) BOOM: prior null (mechanism), any vs_mean CI>0 would be the first real-workload win.")

    test_st = StructuredWorkloadModel(units, seed=777).sample(N_TEST)
    conc_st = run_synth(
        "structured",
        lambda seed, n: StructuredWorkloadModel(
            units, seed=100_000 * seed + n).sample(n),
        solver, units, chip_w, chip_h, test_st, emit)

    test_iid = iid_set(units, tot, np.random.default_rng(777), N_TEST)
    conc_iid = run_synth(
        "iid (control)",
        lambda seed, n: iid_set(units, tot, np.random.default_rng(100_000 * seed + n), n),
        solver, units, chip_w, chip_h, test_iid, emit)

    # BOOM real-workload arm (only with the external GPL dataset)
    conc_boom = None
    csvp, rptp = resolve_paths()
    if csvp:
        wl = BoomWorkload(csvp, rptp, config_id="0")
        bsolver = GridFDSolver(cfg, wl.units, wl.chip_w, wl.chip_h, NR, NC)
        bsolver.build(); bsolver.factorize()

        def peaks_fn(scen):
            p = DiffPlacer(bsolver, wl.units, wl.chip_w, wl.chip_h, NR, NC, alpha=ALPHA)
            cx, cy = p.get_positions()
            return p._scenario_peaks(cx, cy, scen)
        wl.scale_to_peak(peaks_fn, TARGET_PEAK)
        scen = wl.scenarios()
        n_all = len(scen); cut = int(n_all * TRAIN_FRAC_BOOM)
        ratio = (n_all - cut) / cut
        means = {g: [] for g in GAMMAS}
        cvars = {g: [] for g in GAMMAS}
        for seed in range(N_SPLITS_BOOM):
            perm = np.random.default_rng(seed).permutation(n_all)
            tr = [scen[i] for i in perm[:cut]]
            te = [scen[i] for i in perm[cut:]]
            for g in GAMMAS:
                pl = fit_gamma(bsolver, wl.units, wl.chip_w, wl.chip_h, tr, g)
                m, c = oos(pl, te)
                means[g].append(m); cvars[g].append(c)
        means = {g: np.asarray(v) for g, v in means.items()}
        cvars = {g: np.asarray(v) for g, v in cvars.items()}
        emit(f"\n=== arm: BOOM real workloads (80 programs, exp009 setup) ===")
        emit(f"  {N_SPLITS_BOOM} repeated {cut}/{n_all-cut} splits, Nadeau-Bengio-corrected CI "
             f"(splits share data); test-tail = top ~{max(1, int((n_all-cut)*(1-ALPHA)))} programs "
             f"(low power — see exp009 caveat).")
        conc_boom = report_rows(emit, means, cvars,
                                ci_fn=lambda x: ci95_nadeau_bengio(x, ratio))
    else:
        emit("\n=== arm: BOOM — SKIPPED (BOOM_DATA not found) ===")

    # Verdict block
    emit("\nVERDICT vs pre-registration:")
    a_hits = [(n, g) for n in (16, 32) for g in GAMMAS[1:-1]
              if conc_st[n][g]["vs_cv1_lo"] > 0]
    emit(f"  (a) structured small-N blend beats pure CVaR: "
         f"{'CONFIRMED at ' + str(a_hits) if a_hits else 'FALSIFIED (no gamma, CI>0)'}")
    b_bad = [g for g in GAMMAS[:-1] if conc_st[128][g]["vs_cv1_lo"] > 0]
    emit(f"  (b) structured N=128 pure CVaR unbeaten: "
         f"{'CONFIRMED' if not b_bad else 'VIOLATED by gamma=' + str(b_bad)}")
    c_bad = [(n, g) for n in N_TRAINS for g in GAMMAS[1:]
             if conc_iid[n][g]["vs_mean_lo"] > 0]
    emit(f"  (c) iid: no tail objective beats mean: "
         f"{'CONFIRMED' if not c_bad else 'VIOLATED at ' + str(c_bad)}")
    if conc_boom is not None:
        d_hits = [g for g in GAMMAS[1:] if conc_boom[g]["vs_mean_lo"] > 0]
        d_msg = (f"POSITIVE — gamma={d_hits} beats mean-opt on real workloads (NB-corrected)"
                 if d_hits else
                 "null, as the mechanism predicted (FP light, hotspot stable)")
        emit(f"  (d) BOOM: {d_msg}")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
