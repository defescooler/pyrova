"""exp030: the matched-HPWL (Pareto-frontier) constrained payoff — the DEFENSIBLE
form of exp027, with two caveats fixed.

exp027 compares arms at a SHARED lambda, but the arms then land at DIFFERENT
achieved wirelengths, so D*(lambda) confounds 'better tail' with 'spent more
wire' (a reviewer kills it in one line). And its oracle D* is
optimizer-noise-dominated (5 pairs, sigma ~ 0.6 K), so it can only detect a
>~0.8 K effect. This experiment fixes both:

  1. MATCHED HPWL. For each arm (mean / cvar / blend) sweep lambda and record its
     (achieved exact HPWL, OOS mean, OOS CVaR) FRONTIER. Compare arms at matched
     HPWL by interpolating each arm's CVaR onto a common wire-budget axis: at a
     given wire budget L, which objective gives lower OOS CVaR? This is the
     honest `min CVaR s.t. HPWL <= L`.
  2. COMMON RANDOM NUMBERS. Every arm in a pair shares the SAME jitter path and
     restart perturbations (only the objective differs), so the paired frontier
     difference cancels shared optimizer noise — a large variance reduction over
     exp027's independent-seed arms.

Traps controlled as in exp027 (matched budget, jitter->64 eval, symmetric
best-of-R). EV6-canonical (ev6.desc netlist) primary; the domination check
(dMean<=0) is GATED into the verdict.

GATES: G1 mechanism (corr(FP,INT)<0); G2 the frontier must actually move
(unconstrained->tight HPWL falls and mean peak rises).

PRE-REGISTERED READING: at some target wire budget L, dCVaR(L) = CVaR_mean(L) -
CVaR_cvar(L) (or blend) has CI>0 with dMean(L)<=0 -> risk-aware placement buys
tail AT MATCHED WIRE (the defensible constrained payoff). Holm across the L
cells. If no L clears CI>0 -> null/open.

Set PYROVA_SMOKE=1 for a tiny local execution check (not a result).
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

from pyrova.core.design import Design
from pyrova.thermal.fd_solver import GridFDSolver, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.objectives.overlap import nonoverlap_penalty
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p, holm
from pyrova.workloads.structured import StructuredWorkloadModel, _family
from pyrova.workloads.netlists import ev6_nets

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
FLP = PKG / "inputs/floorplans/ev6.flp"
ALPHA = 0.90
TRAIN_GRID = 18
JITTER = 1.0
ARMS = (("mean", "mean", 0.0), ("cvar", "cvar", 0.0), ("blend", "blend", 0.75))

EVAL_GRID = 32 if SMOKE else 64
N_ITER = 15 if SMOKE else 120
N_ORACLE = 64 if SMOKE else 600
N_OR_PAIRS = 2 if SMOKE else 8
N_TEST = 200 if SMOKE else 1000
RESTARTS = 1 if SMOKE else 3
LAMBDA_GRID = ([0.0, 900.0] if SMOKE else [0.0, 100.0, 300.0, 900.0, 2500.0])
# Target wire budgets as fractions of the unconstrained (lambda=0) mean HPWL.
HPWL_FRACS = [0.85, 0.75, 0.65]


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def exact_hpwl(up, nets):
    cx = np.array([u["leftx"] + u["width"] / 2 for u in up])
    cy = np.array([u["bottomy"] + u["height"] / 2 for u in up])
    return float(sum((cx[np.asarray(i)].max() - cx[np.asarray(i)].min())
                     + (cy[np.asarray(i)].max() - cy[np.asarray(i)].min()) for i in nets))


def fit(solver, units, cw, ch, train, mode, gamma, wl_weight, nets,
        restarts, jitter_seed, restart_seed):
    """Best-of-`restarts`. CRN: jitter_seed and restart_seed are passed in and
    SHARED across arms within a pair, so only the objective differs."""
    best, best_obj = None, np.inf
    rr = np.random.default_rng(restart_seed)
    now = max(1e8, 1e5 * wl_weight)   # legality: default 1e4 lets blocks stack
    for r in range(restarts):
        pl = DiffPlacer(solver, units, cw, ch, TRAIN_GRID, TRAIN_GRID, alpha=ALPHA,
                        blend_gamma=gamma, nets=nets, wl_weight=wl_weight, nonoverlap_w=now)
        if r > 0:
            pl.raw_x += rr.standard_normal(pl.n) * 0.5
            pl.raw_y += rr.standard_normal(pl.n) * 0.5
        pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=jitter_seed + 37 * r)
        obj = pl.objective_and_grad(train, mode=mode)[0]
        if obj < best_obj:
            best, best_obj = pl, obj
    return best


def eval_mc(cfg, up, scen, cw, ch, ambient):
    s = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
    s.build(); s.factorize()
    pk = np.array([float(s.silicon_layer(s.solve(s.build_rhs(
        {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - ambient
        for pw in scen])
    return float(pk.mean()), cvar(pk, ALPHA), pk


def interp_on_hpwl(points, L):
    """points: list of (hpwl, value) for one arm/pair over lambda. Returns value
    interpolated at wire budget L, or None if L is outside the achieved range."""
    pts = sorted(points)
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    if L < xs.min() or L > xs.max():
        return None
    return float(np.interp(L, xs, ys))


def main():
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    d = Design.from_flp(str(FLP))
    units = d.macro_flp_dicts()
    cw, ch = d.chip_width, d.chip_height
    nets = ev6_nets(units)
    solver = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()
    tot_area = sum(u["width"] * u["height"] for u in units)

    out = PKG / "results/exp030_matched_hpwl.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"exp030: MATCHED-HPWL constrained payoff + CRN variance reduction. "
         f"EV6 + {len(nets)} canonical ev6.desc nets. alpha={ALPHA}, train@{TRAIN_GRID}"
         f"+jitter, eval@{EVAL_GRID}, {N_ITER} it, best-of-{RESTARTS}, {N_OR_PAIRS} pairs. "
         + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    # G1 mechanism gate.
    P = np.array(StructuredWorkloadModel(units, seed=999).sample(2000))
    fam = np.array([_family(u["name"]) for u in units])
    r_fi = float(np.corrcoef(P[:, fam == "FP"].sum(1), P[:, fam == "INT"].sum(1))[0, 1])
    emit(f"G1 (corr(FP,INT)): {r_fi:+.3f} -> {'PASS' if r_fi < 0 else 'FAIL'}")
    if r_fi >= 0:
        emit("G1 FAILED — no anti-correlation; no run."); fh.close(); return

    test = StructuredWorkloadModel(units, seed=777).sample(N_TEST)

    # Frontier: front[arm][k] = list of (hpwl, mean, cvar) over lambda.
    front = {a[0]: [[] for _ in range(N_OR_PAIRS)] for a in ARMS}
    overlap_max = 0.0
    peak0, peakL = [], []
    for k in range(N_OR_PAIRS):
        tr = StructuredWorkloadModel(units, seed=10_000 + k).sample(N_ORACLE)
        jseed, rseed = 600_000 + 1000 * k, 700_000 + 1000 * k   # CRN: shared across arms
        for lam in LAMBDA_GRID:
            for name, mode, gamma in ARMS:
                pl = fit(solver, units, cw, ch, tr, mode, gamma, lam, nets,
                         RESTARTS, jseed, rseed)
                up = pl.get_units()
                m, c, _ = eval_mc(cfg, up, test, cw, ch, ambient)
                hp = exact_hpwl(up, nets)
                front[name][k].append((hp, m, c))
                cx, cy = pl.get_positions()
                pen, _, _ = nonoverlap_penalty(cx, cy, pl.widths, pl.heights)
                overlap_max = max(overlap_max, pen / tot_area)
                if name == "mean" and lam == LAMBDA_GRID[0]:
                    peak0.append(m)
                if name == "mean" and lam == LAMBDA_GRID[-1]:
                    peakL.append(m)
        print(f"  pair {k + 1}/{N_OR_PAIRS} done", flush=True)

    # Legality + binding report.
    emit(f"legality: max residual overlap = {100*overlap_max:.3f}% of block area "
         f"({'OK (<1%)' if overlap_max < 0.01 else 'CHECK'})")
    hp0 = np.mean([front['mean'][k][0][0] for k in range(N_OR_PAIRS)])
    hpL = np.mean([front['mean'][k][-1][0] for k in range(N_OR_PAIRS)])
    binds = (hpL <= 0.9 * hp0) and (np.mean(peakL) > np.mean(peak0))
    emit(f"G2 (frontier moves): mean HPWL {hp0*1e3:.1f}->{hpL*1e3:.1f} mm, "
         f"peak {np.mean(peak0):.2f}->{np.mean(peakL):.2f} K -> {'PASS' if binds else 'FAIL'}")

    # Raw frontiers (for plotting), averaged over pairs.
    emit("\nfrontiers (mean over pairs), per lambda:")
    for name, _, _ in ARMS:
        row = []
        for li, lam in enumerate(LAMBDA_GRID):
            hp = np.mean([front[name][k][li][0] for k in range(N_OR_PAIRS)])
            cc = np.mean([front[name][k][li][2] for k in range(N_OR_PAIRS)])
            row.append(f"L{lam:.0f}:(HPWL={hp*1e3:.1f},CVaR={cc:.2f})")
        emit(f"  {name:5s}: " + "  ".join(row))

    # Matched-HPWL comparison at target wire budgets.
    emit("\nMATCHED-HPWL (interpolated onto common wire budget; CRN-paired CIs):")
    cells = []
    for f in HPWL_FRACS:
        L = f * hp0
        for arm in ("cvar", "blend"):
            dC, dM = [], []
            for k in range(N_OR_PAIRS):
                cm = interp_on_hpwl([(p[0], p[2]) for p in front["mean"][k]], L)
                ca = interp_on_hpwl([(p[0], p[2]) for p in front[arm][k]], L)
                mm = interp_on_hpwl([(p[0], p[1]) for p in front["mean"][k]], L)
                ma = interp_on_hpwl([(p[0], p[1]) for p in front[arm][k]], L)
                if None in (cm, ca, mm, ma):
                    continue
                dC.append(cm - ca); dM.append(mm - ma)
            if len(dC) < 3:
                emit(f"  L={f:.2f}*uncon {arm:5s}: too few valid pairs ({len(dC)})")
                continue
            gc, _, lo, hi = ci95_t(dC); gm, _, mlo, mhi = ci95_t(dM)
            cell = dict(f=f, arm=arm, dC=gc, lo=lo, hi=hi, dM=gm, mlo=mlo, mhi=mhi,
                        p=paired_t_p(dC), n=len(dC))
            cells.append(cell)
            fl = "*" if lo > 0 else ("x" if hi < 0 else " ")
            emit(f"  L={f:.2f}*uncon ({L*1e3:.1f}mm) {arm:5s}: dCVaR={gc:+.3f}{fl} "
                 f"[{lo:+.3f},{hi:+.3f}] dMean={gm:+.3f} [{mlo:+.3f},{mhi:+.3f}] (n={len(dC)})")

    keep = holm([c["p"] for c in cells]) if cells else []
    for c, kp in zip(cells, keep):
        c["holm"] = bool(kp)

    emit("\n===== PRE-REGISTERED VERDICT =====")
    if not binds:
        emit("NO VERDICT — G2 failed (frontier did not move).")
    trade = [c for c in cells if c.get("holm") and c["lo"] > 0 and c["dM"] <= 0]
    dom = [c for c in cells if c.get("holm") and c["lo"] > 0 and c["dM"] > 0]
    if binds and trade:
        b = max(trade, key=lambda c: c["dC"])
        emit(f"PAYS AT MATCHED WIRE — at L={b['f']:.2f}*unconstrained the {b['arm']} "
             f"objective buys dCVaR={b['dC']:+.3f} K [{b['lo']:+.2f},{b['hi']:+.2f}] "
             f"(Holm) with dMean={b['dM']:+.3f}<=0: a real mean-for-tail trade at "
             f"equal wirelength. The defensible Problem-A positive.")
    elif binds and dom:
        b = max(dom, key=lambda c: c["dC"])
        emit(f"DOMINATION at L={b['f']:.2f} ({b['arm']}): dCVaR>0 but dMean="
             f"{b['dM']:+.3f}>0 (under-converged baseline), gated OUT.")
    elif binds:
        emit("NULL / OPEN — no wire budget shows a matched-HPWL tail gain "
             "(CI>0). At equal wirelength, minimising the mean is tail-competitive.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
