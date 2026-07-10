"""The wirelength-constrained payoff on MCNC ami33 — the first testbed where the
question is well-posed.

Prior constrained attempts were void or ill-posed: soft-overlap stacking faked
wire savings, and the bundled dies had no legal wire range to allocate (EV6 is a
100%-utilized tiling; the SoC's legal range was ~5-13% with identical arms).
ami33 fixes both: REAL netlist (78 signal nets from the YAL NETWORK section),
utilization a legitimate free parameter (0.55 here), and a measured 28% legal
wire compression range with distinct arms (gate run).

Estimand per lambda: D*(lambda) = trueCVaR(mean-arm) - trueCVaR(risk-arm), all
arms trained at wl_weight=lambda, LEGALIZED (overlap removed), evaluated on an
independent fine grid. Arms: plain cvar and cvar_wide (CVaR trained at a wider
alpha=0.6 tail, scored at 0.9 — the variance-reduced objective, since plain
empirical CVaR needs N_ORACLE >~ 6000 to beat mean-training under i.i.d. and
N=2000 is what a lambda-sweep affords).

Controls: matched budget; train@18 + raster jitter -> eval@64; common random
numbers across arms in a pair; legalize-then-eval with legality gate; Holm
across (arm, lambda) cells.

GATES (fail-closed): G1 legal wire range (mean-arm legal HPWL at max lambda
<= 0.85x its lambda=0 value); G2 legality (max residual overlap < 0.1%).
The lambda=0 column doubles as the unconstrained i.i.d. tail-dimension check on
this design at this N — if it is <= 0 for both risk arms, lambda>0 positives
are interpreted as scarcity-created (the interesting outcome) but flagged.

READING: PAYS-UNDER-CONSTRAINT if some lambda>0 cell has D* Holm-sig > 0 with
dMean <= 0 and D*(lambda) >= D*(0) (scarcity creates/preserves the trade).
NULL if all lambda>0 CIs include or fall below 0.

Set PYROVA_SMOKE=1 for a tiny local execution check.
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
from pyrova.optimizer.legalize import legalize_units
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p, holm
from pyrova.workloads.mcnc import load_yal

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
YAL = ROOT / "ami33.yal.txt"
EVAL_ALPHA = 0.90
UTIL = 0.55
TRAIN_GRID = 18
JITTER = 1.0
NONOVERLAP_W = 1e6            # moderate: mobile placements; legalizer removes residue

EVAL_GRID = 32 if SMOKE else 64
N_ITER = 12 if SMOKE else 100
N_ORACLE = 100 if SMOKE else 2000
N_PAIRS = 2 if SMOKE else 5
N_TEST = 200 if SMOKE else 2000
LAMBDA_GRID = [0.0, 100.0] if SMOKE else [0.0, 30.0, 100.0, 300.0]
ARMS = {"cvar": ("cvar", 0.90), "cvar_wide": ("cvar", 0.60)}


def sset(units, tot, rng, k):
    return [np.array([random_power_map(units, tot, rng)[u["name"]] for u in units])
            for _ in range(k)]


def exact_hpwl(up, nets):
    cx = np.array([u["leftx"] + u["width"] / 2 for u in up])
    cy = np.array([u["bottomy"] + u["height"] / 2 for u in up])
    return float(sum((cx[np.asarray(i)].max() - cx[np.asarray(i)].min())
                     + (cy[np.asarray(i)].max() - cy[np.asarray(i)].min())
                     for i in nets))


def main():
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    units, nets, cw, ch = load_yal(YAL, utilization=UTIL)
    n = len(units)
    tot = 2.0 * n
    solver = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()
    test = sset(units, tot, np.random.default_rng(99), N_TEST)

    out = PKG / "results/exp038_ami33_constrained.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"ami33 constrained payoff: {n} blocks, {len(nets)} real nets, die "
         f"{cw*1e3:.1f}mm sq (util={UTIL}). i.i.d. power, N_ORACLE={N_ORACLE}, "
         f"{N_PAIRS} pairs, {N_ITER} it, train@{TRAIN_GRID}+jitter, "
         f"legalized eval@{EVAL_GRID} (alpha={EVAL_ALPHA}). lambda={LAMBDA_GRID}. "
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
                        nets=nets, wl_weight=lam, nonoverlap_w=NONOVERLAP_W)
        pl.optimize(tr, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=jseed)
        return legalize_units(pl.get_units(), cw, ch)

    rows = []
    ovl_max = 0.0
    hp_mean = {}
    for lam in LAMBDA_GRID:
        cell = {a: {"dC": [], "dM": []} for a in ARMS}
        hps = []
        for k in range(N_PAIRS):
            tr = sset(units, tot, np.random.default_rng(10_000 + k), N_ORACLE)
            js = 500_000 + 1000 * k                       # CRN: shared across arms
            um, fm = fit(tr, "mean", 0.90, lam, js)
            ovl_max = max(ovl_max, fm)
            c_m, m_m = eval_cvar_mean(um)
            hps.append(exact_hpwl(um, nets))
            for a, (mode, al) in ARMS.items():
                ua, fa = fit(tr, mode, al, lam, js)
                ovl_max = max(ovl_max, fa)
                c_a, m_a = eval_cvar_mean(ua)
                cell[a]["dC"].append(c_m - c_a)
                cell[a]["dM"].append(m_m - m_a)
            print(f"  lambda={lam:.0f} pair {k + 1}/{N_PAIRS}", flush=True)
        hp_mean[lam] = float(np.mean(hps))
        for a in ARMS:
            d = np.array(cell[a]["dC"])
            m, _, lo, hi = ci95_t(d)
            rows.append(dict(lam=lam, arm=a, D=m, lo=lo, hi=hi,
                             dM=float(np.mean(cell[a]["dM"])), p=paired_t_p(d)))
            emit(f"  lambda={lam:6.0f} {a:9s}: D*={m:+.4f} [{lo:+.4f},{hi:+.4f}] "
                 f"dMean={rows[-1]['dM']:+.4f}  (legal HPWL {hp_mean[lam]*1e3:.1f}mm)")

    keep = holm([r["p"] for r in rows])
    for r, kp in zip(rows, keep):
        r["holm"] = bool(kp)

    hp0, hpL = hp_mean[LAMBDA_GRID[0]], hp_mean[LAMBDA_GRID[-1]]
    g1 = hpL <= 0.85 * hp0
    g2 = ovl_max < 1e-3
    emit(f"\nG1 (wire range): {hp0*1e3:.1f}->{hpL*1e3:.1f}mm "
         f"({100*hpL/hp0:.1f}%) -> {'PASS' if g1 else 'FAIL'}")
    emit(f"G2 (legality): max overlap {100*ovl_max:.3f}% -> {'PASS' if g2 else 'FAIL'}")

    emit("\n===== PRE-REGISTERED VERDICT =====")
    if not (g1 and g2):
        emit("NO VERDICT — a gate failed."); fh.close(); return
    base = {a: next(r["D"] for r in rows if r["lam"] == 0 and r["arm"] == a)
            for a in ARMS}
    win = [r for r in rows if r["lam"] > 0 and r["holm"] and r["lo"] > 0
           and r["dM"] <= 0 and r["D"] >= base[r["arm"]]]
    if win:
        b = max(win, key=lambda r: r["D"])
        emit(f"PAYS-UNDER-CONSTRAINT — {b['arm']} at lambda={b['lam']:.0f}: "
             f"D*={b['D']:+.4f} [{b['lo']:+.3f},{b['hi']:+.3f}] Holm-sig, "
             f"dMean={b['dM']:+.4f}<=0, >= its lambda=0 value {base[b['arm']]:+.4f}. "
             f"First well-posed constrained positive (real netlist, legal, trap-free).")
    elif all(r["hi"] <= 0 for r in rows if r["lam"] > 0):
        emit("NULL — every lambda>0 CI <= 0: under wire scarcity, mean-training "
             "matches or beats the risk objectives on this design/workload/N.")
    else:
        emit("INCONCLUSIVE — no lambda>0 cell clears Holm CI>0 with a clean trade; "
             "report the table, no slogan. (lambda=0 column = the unconstrained "
             "check at this N.)")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
