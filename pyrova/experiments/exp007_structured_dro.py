"""Does the Wasserstein-DRO penalty earn its keep over pure CVaR at small N?

The missing experiment. Every prior eps-sweep (the retired exp002, eval_dro_benchmarks)
ran on the i.i.d. workload — exactly the regime where exp004 shows there is no tail
dimension, so the penalty has nothing to regularise toward. exp005's positive result
is PURE CVaR (eps=0), not DRO. The Wasserstein penalty has therefore never been
tested where it could plausibly help: a STRUCTURED workload at SMALL N, where a real
tail dimension exists but the empirical tail (~N(1-alpha) scenarios) is too noisy for
pure CVaR to learn without overfitting.

Hypothesis: on the structured workload at small N, the DRO penalty regularises the
noisy empirical tail and lets CVaR-opt recover the structured tail signal at a smaller
N than pure CVaR can. Mechanically, DRO should reduce CVaR-opt's tail overfitting, so
DRO-opt's OOS CVaR < pure-CVaR-opt's OOS CVaR at small N, with the advantage shrinking
as N grows and the empirical tail becomes well-estimated.

Design (de-confounded, review):
  * Two workload arms: STRUCTURED (primary) and i.i.d. (matched NEGATIVE CONTROL —
    DRO should NOT beat pure CVaR where no tail dimension exists).
  * Placements: mean-opt, pure-CVaR-opt (eps=0), DRO-opt over an eps-sweep.
  * For every placement, side-by-side (OOS mean, OOS CVaR) on a LARGE holdout
    (OOS ~= true), 5 seeds, 95% CI on paired differences. Three reported deltas:
      vs_mean  = OOS CVaR(mean-opt)    - OOS CVaR(placement)   (>0: beats mean on tail)
      vs_cvar0 = OOS CVaR(pure-CVaR)   - OOS CVaR(placement)   (>0: DRO beats pure CVaR -- its job)
      regret   = OOS CVaR(placement)   - OOS CVaR(cvar-oracle) (distance to the achievable tail)
  * cvar-oracle = pure CVaR trained at large N_ORACLE (overfitting-free reference).

Outcomes:
  - structured vs_cvar0 CI>0 at small N AND i.i.d. vs_cvar0 ~ 0  -> DRO earns its keep
    precisely where there is a noisy tail to regularise (clean positive).
  - both ~ 0  -> penalty inert / eps miscalibrated for this problem scale.
  - structured vs_cvar0 CI<0  -> DRO over-regularises even the real tail.

NOT a single-config verdict: coarse grid, 5 seeds, eps uncalibrated to a physical
Wasserstein radius. Reports CIs so each cell is interpretable.
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
from pyrova.evaluation.metrics import mean_cvar, ci95_t
from pyrova.workloads.structured import StructuredWorkloadModel

FLP = PKG / "inputs/floorplans/ev6.flp"
CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA = 0.90
N_SMALLS = [16, 32, 64]         # small-N regime where the empirical tail is noisy
EPS_SWEEP = [0.05, 0.1, 0.2]    # DRO radii (the dual is ~10x the old surrogate; keep modest)
N_ORACLE = 1500                 # overfitting-free pure-CVaR reference
N_TEST = 1500                   # large holdout so OOS ~= true
N_SEEDS = 5
NR = NC = 18
N_ITER = 30
TOTAL_P_IID = None              # set per-bench to 2.0 * n_blocks


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def iid_set(units, tot, rng, k):
    return [np.array([random_power_map(units, tot, rng)[u["name"]] for u in units])
            for _ in range(k)]


class Arm:
    """A workload arm: a fixed large test set plus a per-seed/per-n train sampler."""

    def __init__(self, name, units, tot):
        self.name = name
        self.units = units
        self.tot = tot

    def test(self):
        raise NotImplementedError

    def train(self, seed, n):
        raise NotImplementedError


class IIDArm(Arm):
    def test(self):
        return iid_set(self.units, self.tot, np.random.default_rng(777), N_TEST)

    def train(self, seed, n):
        return iid_set(self.units, self.tot, np.random.default_rng(seed), n)


class StructuredArm(Arm):
    def test(self):
        return StructuredWorkloadModel(self.units, seed=777).sample(N_TEST)

    def train(self, seed, n):
        return StructuredWorkloadModel(self.units, seed=1000 * seed + n).sample(n)


def fit(solver, units, chip_w, chip_h, train, mode, eps):
    pl = DiffPlacer(solver, units, chip_w, chip_h, NR, NC,
                    alpha=ALPHA, eps_dro=eps)
    pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False)
    return pl


def oos_cvar(pl, scen):
    cx, cy = pl.get_positions()
    return mean_cvar(pl._scenario_peaks(cx, cy, scen), ALPHA)   # (mean, cvar)


def run_arm(arm: Arm, solver, chip_w, chip_h, emit):
    units = arm.units
    test = arm.test()

    # Overfitting-free pure-CVaR reference (oracle), once per arm.
    oracle = fit(solver, units, chip_w, chip_h,
                 arm.train(0, N_ORACLE) if isinstance(arm, IIDArm)
                 else StructuredWorkloadModel(units, seed=0).sample(N_ORACLE),
                 "cvar", 0.0)
    _, c_oracle = oos_cvar(oracle, test)

    labels = ["mean", "cvar(e0)"] + [f"dro(e{e:g})" for e in EPS_SWEEP]
    modes = [("mean", 0.0), ("cvar", 0.0)] + [("dro", e) for e in EPS_SWEEP]

    emit(f"\n=== arm: {arm.name} ===   cvar-oracle(N={N_ORACLE}) OOS CVaR = {c_oracle:.3f} K")
    for n in N_SMALLS:
        # per-seed (mean, cvar) for each placement
        means = {lab: [] for lab in labels}
        cvars = {lab: [] for lab in labels}
        for seed in range(N_SEEDS):
            train = arm.train(seed, n)
            for lab, (mode, eps) in zip(labels, modes):
                pl = fit(solver, units, chip_w, chip_h, train, mode, eps)
                m, c = oos_cvar(pl, test)
                means[lab].append(m); cvars[lab].append(c)
        means = {lab: np.asarray(v) for lab, v in means.items()}
        cvars = {lab: np.asarray(v) for lab, v in cvars.items()}

        emit(f"  N_TRAIN={n}  (alpha={ALPHA}, {N_SEEDS} seeds, N_TEST={N_TEST})")
        hdr = (f"    {'placement':<11}{'OOSmean':>9}{'OOSCVaR':>9}  "
               f"{'vs_mean':>14}  {'vs_cvar0':>14}  {'regret':>8}")
        emit(hdr)
        for lab in labels:
            vs_mean = cvars["mean"] - cvars[lab]      # >0: placement beats mean on tail
            vs_cv0 = cvars["cvar(e0)"] - cvars[lab]   # >0: DRO beats pure CVaR
            regret = float(cvars[lab].mean() - c_oracle)
            gm, _, gm_lo, gm_hi = ci95_t(vs_mean)
            g0, _, g0_lo, g0_hi = ci95_t(vs_cv0)
            fm = "*" if gm_lo > 0 else ("x" if gm_hi < 0 else " ")
            f0 = "*" if g0_lo > 0 else ("x" if g0_hi < 0 else " ")
            vs_mean_s = "    -    " if lab == "mean" else f"{gm:+.2f}{fm}"
            vs_cv0_s = "    -    " if lab in ("mean", "cvar(e0)") else f"{g0:+.2f}{f0}"
            emit(f"    {lab:<11}{means[lab].mean():>9.3f}{cvars[lab].mean():>9.3f}  "
                 f"{vs_mean_s:>14}  {vs_cv0_s:>14}  {regret:>+8.3f}")


def main():
    units = parse_flp(str(FLP))
    cfg = parse_config(str(CONFIG))
    chip_w, chip_h = chip_box(units)
    solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
    solver.build(); solver.factorize()
    tot = 2.0 * len(units)

    out = PKG / "results/exp007_structured_dro.txt"
    fh = open(out, "w")

    def emit(s):
        print(s); fh.write(s + "\n")

    emit(f"DRO vs pure CVaR at small N on ev6 ({len(units)} blocks). "
         f"grid {NR}x{NC}, {N_ITER} iter, eps={EPS_SWEEP}.")
    emit("vs_mean = CVaR(mean-opt) - CVaR(placement) (>0 beats mean on tail).")
    emit("vs_cvar0 = CVaR(pure-CVaR) - CVaR(placement) (>0 = DRO beats pure CVaR, its job).")
    emit("regret = CVaR(placement) - CVaR(cvar-oracle) (distance to achievable tail).")
    emit("'*' CI>0, 'x' CI<0 on the paired per-seed difference. STRUCTURED is the test; "
         "IID is the negative control (DRO should not beat pure CVaR there).")

    # Primary arm: structured. Negative control: i.i.d.
    run_arm(StructuredArm("structured", units, tot), solver, chip_w, chip_h, emit)
    run_arm(IIDArm("iid", units, tot), solver, chip_w, chip_h, emit)

    emit("\nReading: DRO earns its keep iff structured vs_cvar0 CI>0 at small N while "
         "the iid control stays ~0. Both ~0 => penalty inert / eps miscalibrated.")
    fh.close()
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
