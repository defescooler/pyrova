"""Mechanism test: does a TRUE tail dimension exist, and is it learnable at small N?

De-confounded successor (review) to the old one-sided
    gap = OOS CVaR(mean-opt) - OOS CVaR(CVaR-opt),
which sums two effects: (A) CVaR-opt is scored on the very functional it minimised,
and (B) at small N it overfits the noisy empirical tail — the sign of one number
cannot separate a true, exploited tail dimension from overfitting. Three reported
quantities remove both confounds:

  1. EXISTENCE (overfitting-free). Train N_OR_SEEDS independent mean-oracle /
     CVaR-oracle PAIRS at a LARGE N_ORACLE so neither overfits, evaluate on a
     HUGE holdout (OOS ~= true). Per pair
       D*_k = trueCVaR(mean-oracle_k) - trueCVaR(cvar-oracle_k)
     and D* is reported with a paired 95% t-CI across oracle seeds — the
     nonconvex placer's run-to-run noise a single oracle pair cannot provide.
     D* CI>0 means a true, separable tail dimension exists; CI spanning 0 means
     minimising the mean already minimises the tail.

  2. SIDE-BY-SIDE (mean, CVaR) at small N. For mean-opt and CVaR-opt trained at
     N_SMALL, report BOTH out-of-sample mean and CVaR. Lower CVaR with higher
     mean is a genuine mean-for-tail trade; higher on both is dominated (pure
     overfitting).

  3. REGRET to the CVaR-oracle. regret(p) = OOS CVaR(p) - OOS CVaR(cvar-oracle),
     the distance to the best achievable tail. Learnability = whether small-N
     CVaR-opt's regret is below small-N mean-opt's; if not, the tail dimension
     exists (D*>0) but is not learnable at this N.

Part B (descriptive): correlation of mean-dT vs CVaR-dT across placements near the
optimum (Pearson < 1 => a separate, if weak, risk dimension exists).

All holdouts are large enough that OOS ~= true, so the across-seed CI reflects
training (small-N estimator) variance, not scoring noise. i.i.d. synthetic workload
on ev6 + floorplan2; structured-workload analogue: exp005; DRO penalty vs pure
CVaR: exp007.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import scipy.stats

HERE = Path(__file__).resolve().parent          # pyrova/experiments
PKG = HERE.parent                               # pyrova
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_flp, parse_config, random_power_map
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import mean_cvar, ci95_t

CONFIG = PKG / "inputs/configs/thermal.config"
BENCHES = [PKG / "inputs/floorplans/ev6.flp",
           ROOT / "Tools/HotSpot/examples/example3/floorplan2.flp"]
ALPHA = 0.9
N_SMALL = 32           # small-N regime where the empirical tail is noisy
N_ORACLE = 1500        # large-N "oracle" training: overfitting-free
N_OR_SEEDS = 5         # independent oracle pairs: D* gets a CI, not a point estimate
N_TEST = 4000          # huge holdout so OOS CVaR ~= true CVaR
NR = NC = 24
N_ITER = 40
N_SEEDS = 8
K_PERTURB = 200
N_PERTURB_TEST = 1000  # Part B is descriptive; a holdout subsample keeps it cheap


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def scen_set(units, tot, rng, k):
    return [np.array([random_power_map(units, tot, rng)[u["name"]] for u in units])
            for _ in range(k)]


def trained(solver, units, chip_w, chip_h, train, mode):
    pl = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=ALPHA)
    pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False)
    return pl


def oos_mean_cvar(pl, scen):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), ALPHA)


def center_distance(plA, plB, diag) -> float:
    ax, ay = plA.get_positions()
    bx, by = plB.get_positions()
    return float(np.sqrt(((ax - bx) ** 2 + (ay - by) ** 2).mean())) / diag


def run(path: Path, cfg, emit) -> dict:
    units = parse_flp(str(path))
    n = len(units)
    chip_w, chip_h = chip_box(units)
    diag = float(np.hypot(chip_w, chip_h))
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()
    tot = 2.0 * n

    # One huge holdout per bench: every placement (oracle and small-N) is scored on
    # it, so OOS ~= true and the only across-seed variance is training variance.
    test = scen_set(units, tot, np.random.default_rng(99), N_TEST)

    emit(f"\n=== {path.stem} ({n} blocks), alpha={ALPHA} ===")

    # (1) Existence: independent oracle pairs at large N (overfitting-free).
    # Oracle RNG streams (10_000+k) are disjoint from the small-N seeds and the
    # holdout stream; each pair trains on its own fresh N_ORACLE draw.
    Dk, c_cos, dists = [], [], []
    or_rows = []
    for k in range(N_OR_SEEDS):
        or_train = scen_set(units, tot, np.random.default_rng(10_000 + k), N_ORACLE)
        mean_or = trained(solver, units, chip_w, chip_h, or_train, "mean")
        cvar_or = trained(solver, units, chip_w, chip_h, or_train, "cvar")
        m_mo, c_mo = oos_mean_cvar(mean_or, test)
        m_co, c_co = oos_mean_cvar(cvar_or, test)
        Dk.append(c_mo - c_co)
        c_cos.append(c_co)
        dists.append(100.0 * center_distance(mean_or, cvar_or, diag))
        or_rows.append((m_mo, c_mo, m_co, c_co))
    Dm, _, D_lo, D_hi = ci95_t(Dk)
    c_ref = float(min(c_cos))            # best achieved tail placement = regret reference
    verdict = ("tail dimension EXISTS (CI>0)" if D_lo > 0 else
               "tail dimension NEGATIVE (CI<0)" if D_hi < 0 else
               "no separable tail dimension (CI spans 0)")
    emit(f"  ORACLE (N={N_ORACLE}, {N_OR_SEEDS} independent pairs):")
    for k, (m_mo, c_mo, m_co, c_co) in enumerate(or_rows):
        emit(f"    pair {k}: mean-oracle (OOS mean,CVaR)=({m_mo:.3f},{c_mo:.3f})  "
             f"cvar-oracle=({m_co:.3f},{c_co:.3f})  D*_k={Dk[k]:+.3f}")
    emit(f"    D* = {Dm:+.3f} K CI[{D_lo:+.3f},{D_hi:+.3f}]  ({verdict})")
    emit(f"    oracle CVaR run-to-run std = {np.std(c_cos, ddof=1):.3f} K "
         f"(optimizer noise a single-pair D* would hide)")
    emit(f"    oracle mean<->CVaR center distance = {np.mean(dists):.2f}% diag")

    # (2,3) Small-N placements: side-by-side (mean,CVaR) and regret-to-oracle.
    Mm, Cm, Mc, Cc = [], [], [], []
    last = None
    for seed in range(N_SEEDS):
        train = scen_set(units, tot, np.random.default_rng(seed), N_SMALL)
        p_mean = trained(solver, units, chip_w, chip_h, train, "mean")
        p_cvar = trained(solver, units, chip_w, chip_h, train, "cvar")
        mm, cm = oos_mean_cvar(p_mean, test)
        mc, cc = oos_mean_cvar(p_cvar, test)
        Mm.append(mm); Cm.append(cm); Mc.append(mc); Cc.append(cc)
        last = p_mean

    Mm, Cm, Mc, Cc = map(np.asarray, (Mm, Cm, Mc, Cc))
    dmean, _, dm_lo, dm_hi = ci95_t(Mm - Mc)      # >0 => CVaR-opt has higher mean
    dcvar, _, dc_lo, dc_hi = ci95_t(Cm - Cc)      # >0 => CVaR-opt has lower CVaR
    # Regret reference: the best cvar-oracle placement across pairs (a concrete
    # placement scored on the same holdout). The CI covers small-N training
    # variance only; the reference's own optimizer noise is the oracle std above.
    reg_m, _, rm_lo, rm_hi = ci95_t(Cm - c_ref)   # mean-opt regret to best oracle
    reg_c, _, rc_lo, rc_hi = ci95_t(Cc - c_ref)   # cvar-opt regret to best oracle

    emit(f"  SMALL-N (N={N_SMALL}, {N_SEEDS} seeds), OOS averaged on N_TEST={N_TEST}:")
    emit(f"    mean-opt: OOS mean={Mm.mean():.3f}  OOS CVaR={Cm.mean():.3f}")
    emit(f"    cvar-opt: OOS mean={Mc.mean():.3f}  OOS CVaR={Cc.mean():.3f}")
    emit(f"    dMean (mean-opt - cvar-opt) = {dmean:+.3f} K CI[{dm_lo:+.3f},{dm_hi:+.3f}]  "
         f"(<0 => cvar-opt pays mean)")
    emit(f"    dCVaR (mean-opt - cvar-opt) = {dcvar:+.3f} K CI[{dc_lo:+.3f},{dc_hi:+.3f}]  "
         f"(>0 => cvar-opt buys tail)")
    trade = "genuine mean-for-tail trade" if (dmean < 0 and dcvar > 0) else (
            "dominated (overfit): cvar-opt worse on both" if (dmean <= 0 and dcvar <= 0)
            else "mixed")
    emit(f"    -> small-N cvar-opt is: {trade}")
    emit(f"    regret to best cvar-oracle:  mean-opt={reg_m:+.3f} CI[{rm_lo:+.3f},{rm_hi:+.3f}]  "
         f"cvar-opt={reg_c:+.3f} CI[{rc_lo:+.3f},{rc_hi:+.3f}]  "
         f"({'cvar-opt closer' if reg_c < reg_m else 'mean-opt closer'} to the best achieved tail)")

    # Part B: correlation of mean-dT vs CVaR-dT across nearby placements.
    # Descriptive only (no null is tested: sampling noise alone keeps an
    # empirical Pearson below 1). Scored on a holdout subsample for speed.
    part_b_test = test[:N_PERTURB_TEST]
    base_x, base_y = last.raw_x.copy(), last.raw_y.copy()
    rng = np.random.default_rng(123)
    ms, cs = [], []
    for _ in range(K_PERTURB):
        sigma = rng.uniform(0.1, 1.5)
        last.raw_x = base_x + rng.standard_normal(n) * sigma
        last.raw_y = base_y + rng.standard_normal(n) * sigma
        m, c = oos_mean_cvar(last, part_b_test)
        ms.append(m); cs.append(c)
    last.raw_x, last.raw_y = base_x, base_y
    pear = float(np.corrcoef(ms, cs)[0, 1])
    spear = float(scipy.stats.spearmanr(ms, cs).statistic)
    emit(f"  across {K_PERTURB} placements (descriptive, N_TEST={N_PERTURB_TEST}): "
         f"corr(mean-dT, CVaR-dT) Pearson={pear:.4f} Spearman={spear:.4f}")

    return dict(name=path.stem, Dstar=Dm, D_lo=D_lo, D_hi=D_hi,
                dmean=dmean, dcvar=dcvar, reg_m=reg_m, reg_c=reg_c,
                pearson=pear, spearman=spear)


def main():
    cfg = parse_config(str(CONFIG))
    out = PKG / "results/exp003_mean_cvar_correlation.txt"
    fh = open(out, "w")

    def emit(s):
        print(s); fh.write(s + "\n")

    emit("Mechanism: TRUE tail dimension (oracle D*) + de-confounded small-N "
         "(mean,CVaR) side-by-side + regret. i.i.d. synthetic workload.")
    emit(f"alpha={ALPHA}, N_SMALL={N_SMALL}, N_ORACLE={N_ORACLE} x {N_OR_SEEDS} oracle pairs, "
         f"N_TEST={N_TEST}, {N_SEEDS} small-N seeds, grid {NR}x{NC}, {N_ITER} iter.")
    emit("Replaces the old one-sided gap = OOS CVaR(mean-opt) - OOS CVaR(CVaR-opt), "
         "which conflated (A) scoring CVaR-opt on its own metric and (B) tail "
         "overfitting; the sign of one number cannot separate them.")
    for p in BENCHES:
        if p.exists():
            run(p, cfg, emit)
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
