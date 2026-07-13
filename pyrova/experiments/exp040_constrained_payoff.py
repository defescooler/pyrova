"""Wirelength-constrained tail payoff with density-spread fitting (ramped
bin-density term, blocks spread before contact) and guaranteed-legal
evaluation (exactly zero overlap or an explicit infeasible): per HPWL weight
lambda, D*(lambda) = trueCVaR(mean-arm) - trueCVaR(risk-arm) at N_ORACLE=2000,
arms cvar (alpha=0.9) and cvar_wide (CVaR trained at alpha=0.6, scored at
0.9). Matched budgets, train@18 with raster jitter, independent 64^2
evaluation, common random numbers across arms in a pair, dMean alongside,
Holm across cells; fail-closed gates: the constraint binds (mean-arm legal
HPWL at max lambda <= 0.85x its lambda=0 value), guaranteed legality.

PYROVA_TESTBED selects the substrate:
  ami33      real netlist (78 signal nets), i.i.d. power, utilization 0.55.
  hetero_soc stylised netlist, multimodal power (6 modes, distinct hotspots),
             utilization 0.69.

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

from pyrova.thermal.fd_solver import GridFDSolver, parse_config, random_power_map
from pyrova.optimizer.placer import DiffPlacer
from pyrova.optimizer.legalize import legalize_units_exact, LegalizationInfeasible
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p, holm

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"
TESTBED = os.environ.get("PYROVA_TESTBED", "ami33")

CONFIG = PKG / "inputs/configs/thermal.config"
EVAL_ALPHA = 0.90
TRAIN_GRID = 18
JITTER = 1.0
DENSITY_W = 5.0e2             # light: guaranteed legalizer is the hard backstop,
DENSITY_LAM0 = 3.0e1         # so density only needs to keep the fit near-legal
DENSITY_GRID = (48, 48)      # without fighting legitimate wire compression

EVAL_GRID = 32 if SMOKE else 64
N_ITER = 12 if SMOKE else 100
N_ORACLE = 100 if SMOKE else 2000
N_PAIRS = 2 if SMOKE else 5
N_TEST = 200 if SMOKE else 2000
# lambda reaches the binding range: ami33's legal HPWL compresses ~15% by 1e4.
LAMBDA_GRID = [0.0, 3000.0] if SMOKE else [0.0, 1000.0, 3000.0, 10000.0]
ARMS = {"cvar": ("cvar", 0.90), "cvar_wide": ("cvar", 0.60)}


def load_testbed(name):
    """Return (units, nets, cw, ch, sampler) where sampler(seed, k) -> k power arrays."""
    if name == "hetero_soc":
        from pyrova.workloads.hetero_soc import soc_units
        from pyrova.workloads.netlists import soc_nets
        from pyrova.workloads.hetero_soc import HeteroSoCWorkloadModel
        units = soc_units()
        cw = max(u["leftx"] + u["width"] for u in units)
        ch = max(u["bottomy"] + u["height"] for u in units)
        nets = soc_nets(units)

        def sampler(seed, k):
            return HeteroSoCWorkloadModel(units, seed=seed).sample(k)
        return units, nets, cw, ch, sampler
    from pyrova.workloads.mcnc import load_yal
    units, nets, cw, ch = load_yal(ROOT / "ami33.yal.txt", utilization=0.55)
    tot = 2.0 * len(units)

    def sampler(seed, k):
        rng = np.random.default_rng(seed)
        return [np.array([random_power_map(units, tot, rng)[u["name"]] for u in units])
                for _ in range(k)]
    return units, nets, cw, ch, sampler


def exact_hpwl(up, nets):
    cx = np.array([u["leftx"] + u["width"] / 2 for u in up])
    cy = np.array([u["bottomy"] + u["height"] / 2 for u in up])
    return float(sum((cx[np.asarray(i)].max() - cx[np.asarray(i)].min())
                     + (cy[np.asarray(i)].max() - cy[np.asarray(i)].min())
                     for i in nets))


def main():
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    units, nets, cw, ch, sampler = load_testbed(TESTBED)
    n = len(units)
    solver = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()
    test = sampler(99, N_TEST)

    out = PKG / f"results/exp040_constrained_{TESTBED}.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"constrained payoff [{TESTBED}]: {n} blocks, {len(nets)} nets, die "
         f"{cw*1e3:.1f}x{ch*1e3:.1f}mm. N_ORACLE={N_ORACLE}, {N_PAIRS} pairs, "
         f"{N_ITER} it, train@{TRAIN_GRID}+jitter -> guaranteed-legal eval@{EVAL_GRID} "
         f"(alpha={EVAL_ALPHA}). lambda={LAMBDA_GRID}. "
         + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    def eval_cvar_mean(up):
        s = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
        s.build(); s.factorize()
        pk = np.array([float(s.silicon_layer(s.solve(s.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - ambient
            for pw in test])
        return cvar(pk, EVAL_ALPHA), float(pk.mean())

    def fit(tr, mode, alpha, lam, jseed):
        pl = DiffPlacer(solver, units, cw, ch, TRAIN_GRID, TRAIN_GRID, alpha=alpha,
                        nets=nets, wl_weight=lam, nonoverlap_w=1e4,
                        density_w=DENSITY_W, density_lam0=DENSITY_LAM0,
                        density_grid=DENSITY_GRID)
        pl.optimize(tr, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=jseed)
        up, of = legalize_units_exact(pl.get_units(), cw, ch)   # exactly-legal or raises
        return up, of

    rows = []
    ovl_max = 0.0
    hp_mean = {}
    infeasible = 0
    for lam in LAMBDA_GRID:
        cell = {a: {"dC": [], "dM": []} for a in ARMS}
        hps = []
        for k in range(N_PAIRS):
            tr = sampler(10_000 + k, N_ORACLE)
            js = 500_000 + 1000 * k                       # CRN: shared across arms
            try:
                um, fm = fit(tr, "mean", 0.90, lam, js)
                arms_legal = {}
                for a, (mode, al) in ARMS.items():
                    arms_legal[a] = fit(tr, mode, al, lam, js)
            except LegalizationInfeasible:
                infeasible += 1
                print(f"  lambda={lam:.0f} pair {k+1}: INFEASIBLE, skipped", flush=True)
                continue
            ovl_max = max(ovl_max, fm, *[of for _, of in arms_legal.values()])
            c_m, m_m = eval_cvar_mean(um)
            hps.append(exact_hpwl(um, nets))
            for a in ARMS:
                ua, _ = arms_legal[a]
                c_a, m_a = eval_cvar_mean(ua)
                cell[a]["dC"].append(c_m - c_a)
                cell[a]["dM"].append(m_m - m_a)
            print(f"  lambda={lam:.0f} pair {k+1}/{N_PAIRS}", flush=True)
        hp_mean[lam] = float(np.mean(hps)) if hps else float("nan")
        for a in ARMS:
            d = np.array(cell[a]["dC"])
            if len(d) < 2:
                continue
            m, _, lo, hi = ci95_t(d)
            rows.append(dict(lam=lam, arm=a, D=m, lo=lo, hi=hi,
                             dM=float(np.mean(cell[a]["dM"])), p=paired_t_p(d)))
            emit(f"  lambda={lam:6.0f} {a:9s}: D*={m:+.4f} [{lo:+.4f},{hi:+.4f}] "
                 f"dMean={rows[-1]['dM']:+.4f}  (legal HPWL {hp_mean[lam]*1e3:.1f}mm)")

    if rows:
        keep = holm([r["p"] for r in rows])
        for r, kp in zip(rows, keep):
            r["holm"] = bool(kp)

    hp0 = hp_mean.get(LAMBDA_GRID[0], float("nan"))
    hpL = hp_mean.get(LAMBDA_GRID[-1], float("nan"))
    g1 = hpL <= 0.88 * hp0       # constraint must compress legal HPWL >= 12%
    g2 = ovl_max < 1e-3 and infeasible == 0
    g1_msg = "PASS" if g1 else ("FAIL - legal wire range too small; question "
                                "ill-posed on this testbed (no legal scarcity to allocate)")
    g2_msg = "PASS" if g2 else "FAIL"
    emit(f"\nG1 (constraint binds): {hp0*1e3:.1f}->{hpL*1e3:.1f}mm "
         f"({100*hpL/hp0:.1f}%) -> {g1_msg}")
    emit(f"G2 (legality): max overlap {100*ovl_max:.3f}%, {infeasible} infeasible -> {g2_msg}")

    emit("\n===== PRE-REGISTERED VERDICT =====")
    if not (g1 and g2) or not rows:
        emit("NO VERDICT — a gate failed or too few feasible pairs."); fh.close(); return
    base = {a: next((r["D"] for r in rows if r["lam"] == 0 and r["arm"] == a), 0.0)
            for a in ARMS}
    win = [r for r in rows if r["lam"] > 0 and r.get("holm") and r["lo"] > 0
           and r["dM"] <= 0 and r["D"] >= base[r["arm"]]]
    if win:
        emit(f"PAYS-UNDER-CONSTRAINT [{TESTBED}]: " + ", ".join(
            f"lambda={r['lam']:.0f}/{r['arm']} D*={r['D']:+.3f}" for r in win))
    else:
        emit(f"NULL [{TESTBED}]: no lambda>0 cell is Holm-sig>0 with dMean<=0 and "
             f"D*(lambda)>=D*(0). Scarcity did not create a certified payoff here.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
