"""exp027: the wirelength-constrained payoff — Problem A, the decisive Phase-3 question.

Does risk-aware (CVaR) placement's advantage over mean placement GROW as the
HPWL budget tightens? Every prior experiment optimised the UNCONSTRAINED thermal
problem, whose optimum is trivial maximal spreading that both arms reach — so
D* ~= 0 by construction (2026-07 audit, verdict A) and the favorable-regime
payoff was only +0.068 K. Here separation COSTS wire:

    obj = (thermal term) + wl_weight * smoothHPWL(nets)

As wl_weight rises the arms can no longer freely spread the heavy anti-correlated
engines. The theory predicts risk-aware placement should spend its now-scarce
separation on the tail-driving clusters, so D*(lambda) should rise above D*(0).
This is the HPWL-constrained payoff the audit named "the decisive open question."

Benchmarking follows the field's canonical unit (one real design + one power
source + a synthetic parameter sweep; HotFloorplan, Ziabari 2014, etc.). Here
the HPWL budget is that synthetic parameter:
  EV6 : Alpha EV6 (canonical HotSpot floorplan) + the CANONICAL HotFloorplan
        connectivity (inputs/floorplans/ev6.desc) + StructuredWorkloadModel
        mechanism sweep.                                          [PRIMARY]
  SoC : hetero-SoC favorable regime (exp023) + soc_nets -- HAND-BUILT/stylised,
        the lower "synthetic-configurations" rung; a secondary sensitivity
        check only, never a prevalence claim.                     [secondary]
The netlist for EV6 is NOT synthetic: ev6.desc is HotSpot's published EV6
connectivity, cited across thermal-floorplanning papers since 2006.

ALL THREE EVALUATION TRAPS controlled, plus the exp023 restart asymmetry FIXED:
  * budget (exp013):   all arms N_ITER, matched;
  * grid (exp018/020): train@18 + raster_jitter=1.0; eval@64 (independent);
  * baseline (exp019): BOTH arms best-of-R restarts (SYMMETRIC — exp023 gave
                       restarts only to the mean arm; corrected here).

GATES (enforced, fail-closed):
  G1 regime : SoC hotspot must move across >= 3 blocks (exp023 gate); EV6
              corr(FP,INT) < 0 (the anti-correlation the mechanism needs).
  G2 binds  : across the lambda grid the mean-arm HPWL must fall (to <= 0.9x)
              and its mean peak must rise (the constraint is actually active);
              else the sweep is vacuous and NO payoff verdict prints.

ESTIMAND per lambda: oracle D*(lambda) = trueCVaR(mean-oracle) -
  trueCVaR(cvar-oracle), both arms trained at wl_weight=lambda with symmetric
  restarts, each evaluated on an independent 64^2 holdout. N_OR_PAIRS pairs,
  paired t-CI, Holm across the lambda cells.

PRE-REGISTERED READINGS:
  - PAYS-UNDER-CONSTRAINT if some lambda>0 has D*(lambda) Holm-significant > 0
    AND D*(lambda) > D*(0) with dMean <= 0 there (payoff created/amplified by
    scarcity; a genuine mean-for-tail trade, not domination).
  - NULL if every lambda has D* CI <= 0 (even with scarce separation the tail
    is not separable): the unconstrained null extends to the constrained problem.
  - FAIRNESS CAVEAT (reported): at a shared lambda the two arms may achieve
    different HPWL, so D* mixes 'better tail' with 'spent more wire'; a
    matched-HPWL follow-up is the natural refinement (limitation printed).

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
from pyrova.workloads.hetero_soc import (HeteroSoCWorkloadModel, soc_units,
                                         _MODES, _BLOCKS)
from pyrova.workloads.netlists import ev6_nets, soc_nets

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
FLP = PKG / "inputs/floorplans/ev6.flp"
ALPHA = 0.90
TRAIN_GRID = 18
JITTER = 1.0

# Full (PACE) defaults; PYROVA_SMOKE shrinks everything for a local exec check.
EVAL_GRID = 32 if SMOKE else 64
N_ITER = 15 if SMOKE else 120
N_ORACLE = 64 if SMOKE else 600
N_OR_PAIRS = 2 if SMOKE else 5
N_TEST = 200 if SMOKE else 1000
RESTARTS = 1 if SMOKE else 3
LAMBDA_GRID = ([0.0, 900.0] if SMOKE
               else [0.0, 100.0, 300.0, 900.0, 2500.0])


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def fit(solver, units, chip_w, chip_h, train, mode, wl_weight, nets,
        restarts, rrng, seed):
    """Train a placement; best-of-`restarts` on the training objective (symmetric
    across arms). Wirelength term active when wl_weight>0."""
    best, best_obj = None, np.inf
    for r in range(restarts):
        # nonoverlap_w scaled with wl_weight: overlap areas are ~1e-7 m^2, so the
        # default 1e4 is ~4000x too weak under wire pressure and blocks STACK to
        # cheat HPWL (measured 8-19% overlap). This keeps residual overlap <0.2%.
        now = max(1e8, 1e5 * wl_weight)
        pl = DiffPlacer(solver, units, chip_w, chip_h, TRAIN_GRID, TRAIN_GRID,
                        alpha=ALPHA, nets=nets, wl_weight=wl_weight, nonoverlap_w=now)
        if r > 0:
            pl.raw_x += rrng.standard_normal(pl.n) * 0.5
            pl.raw_y += rrng.standard_normal(pl.n) * 0.5
        pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=270_000 + seed + 37 * r)
        obj = pl.objective_and_grad(train, mode=mode)[0]
        if obj < best_obj:
            best, best_obj = pl, obj
    return best


def eval_mc(cfg, units_placed, scen, chip_w, chip_h, ambient):
    """OOS (mean peak, CVaR) on an independent EVAL_GRID solver."""
    s = GridFDSolver(cfg, units_placed, chip_w, chip_h, EVAL_GRID, EVAL_GRID)
    s.build(); s.factorize()
    pk = np.zeros(len(scen))
    for i, pw in enumerate(scen):
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(units_placed)}
        T = s.solve(s.build_rhs(bp))
        pk[i] = float(s.silicon_layer(T).max()) - ambient
    return float(pk.mean()), cvar(pk, ALPHA)


def exact_hpwl(units_placed, nets):
    cx = np.array([u["leftx"] + u["width"] / 2 for u in units_placed])
    cy = np.array([u["bottomy"] + u["height"] / 2 for u in units_placed])
    tot = 0.0
    for idx in nets:
        idx = np.asarray(idx)
        tot += (cx[idx].max() - cx[idx].min()) + (cy[idx].max() - cy[idx].min())
    return float(tot)


def run_testbed(tag, units, chip_w, chip_h, nets, model_factory, cfg, emit):
    """lambda-sweep of oracle D* on one testbed. Returns a dict for the verdict."""
    ambient = cfg["ambient"]
    solver = GridFDSolver(cfg, units, chip_w, chip_h, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()
    test = model_factory(777, N_TEST)

    emit(f"\n===== testbed {tag}: {len(units)} blocks, chip "
         f"{chip_w*1e3:.1f}x{chip_h*1e3:.1f} mm, {len(nets)} nets =====")
    emit(f"  train@{TRAIN_GRID}+jitter{JITTER}, eval@{EVAL_GRID}, {N_ITER} it, "
         f"both arms best-of-{RESTARTS}; N_ORACLE={N_ORACLE}, {N_OR_PAIRS} pairs.")

    tot_area = sum(u["width"] * u["height"] for u in units)
    ovl_max = 0.0
    rows = []
    for lam in LAMBDA_GRID:
        Dk, dMk, hpm, hpc, pkm = [], [], [], [], []
        for k in range(N_OR_PAIRS):
            train = model_factory(10_000 + k, N_ORACLE)
            rng_m = np.random.default_rng(20_000 + 7 * k)
            rng_c = np.random.default_rng(50_000 + 7 * k)
            pm = fit(solver, units, chip_w, chip_h, train, "mean", lam, nets,
                     RESTARTS, rng_m, k)
            pc = fit(solver, units, chip_w, chip_h, train, "cvar", lam, nets,
                     RESTARTS, rng_c, k)
            for pl in (pm, pc):
                pen, _, _ = nonoverlap_penalty(*pl.get_positions(), pl.widths, pl.heights)
                ovl_max = max(ovl_max, pen / tot_area)
            um, uc = pm.get_units(), pc.get_units()
            m_m, c_m = eval_mc(cfg, um, test, chip_w, chip_h, ambient)
            m_c, c_c = eval_mc(cfg, uc, test, chip_w, chip_h, ambient)
            Dk.append(c_m - c_c); dMk.append(m_m - m_c)
            hpm.append(exact_hpwl(um, nets)); hpc.append(exact_hpwl(uc, nets))
            pkm.append(m_m)
        Dm, _, lo, hi = ci95_t(Dk)
        dMm = float(np.mean(dMk))
        p = paired_t_p(Dk)
        row = dict(lam=lam, D=Dm, lo=lo, hi=hi, p=p, dMean=dMm,
                   hpm=float(np.mean(hpm)), hpc=float(np.mean(hpc)),
                   peak=float(np.mean(pkm)))
        rows.append(row)
        div = abs(row["hpc"] - row["hpm"]) / (row["hpm"] + 1e-12)
        emit(f"  lambda={lam:7.0f}: D*={Dm:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}  "
             f"dMean={dMm:+.3f}  HPWL(mean/cvar)={row['hpm']*1e3:.1f}/{row['hpc']*1e3:.1f} mm"
             f"{'  [arms diverge >10% wire]' if div > 0.10 else ''}")

    # Holm across the lambda cells.
    keep = holm([r["p"] for r in rows])
    for r, k in zip(rows, keep):
        r["holm"] = bool(k)

    # G2: the constraint must bind — mean-arm HPWL falls and peak rises.
    hp0, hpL = rows[0]["hpm"], rows[-1]["hpm"]
    pk0, pkL = rows[0]["peak"], rows[-1]["peak"]
    binds = (hpL <= 0.9 * hp0) and (pkL > pk0)
    emit(f"  G2 (constraint binds): HPWL {hp0*1e3:.1f}->{hpL*1e3:.1f} mm, "
         f"peak {pk0:.2f}->{pkL:.2f} K -> {'PASS' if binds else 'FAIL'}")
    legal = ovl_max < 0.01
    emit(f"  LEGALITY: max residual overlap {100*ovl_max:.3f}% of block area -> "
         f"{'OK' if legal else 'ILLEGAL (results void — raise nonoverlap_w)'}")
    return dict(tag=tag, rows=rows, binds=binds and legal)


def gate_soc(units, chip_w, chip_h, cfg, emit):
    """G1 for the SoC: hotspot moves across >= 3 blocks (exp023 gate)."""
    s = GridFDSolver(cfg, units, chip_w, chip_h, EVAL_GRID, EVAL_GRID)
    s.build(); s.factorize()
    pmax = np.array([b[3] for b in _BLOCKS])
    seen = set()
    for act in _MODES.values():
        pw = np.array(act) * pmax
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(units)}
        flat = int(np.argmax(s.silicon_layer(s.solve(s.build_rhs(bp)))))
        cell = (flat // EVAL_GRID, flat % EVAL_GRID)
        blk = min(units, key=lambda u:
                  (u["leftx"] + u["width"]/2 - (cell[1]+.5)*chip_w/EVAL_GRID)**2
                  + (u["bottomy"] + u["height"]/2 - (EVAL_GRID-cell[0]-.5)*chip_h/EVAL_GRID)**2)
        seen.add(blk["name"])
    ok = len(seen) >= 3
    emit(f"  G1 (SoC hotspot mobility): {len(seen)} distinct blocks -> "
         f"{'PASS' if ok else 'FAIL'}")
    return ok


def gate_ev6(units, emit):
    """G1 for EV6: corr(FP-total, INT-total) < 0 over sampled scenarios."""
    m = StructuredWorkloadModel(units, seed=999)
    P = np.array(m.sample(2000))
    fams = np.array([_family(u["name"]) for u in units])
    fp = P[:, fams == "FP"].sum(1)
    it = P[:, fams == "INT"].sum(1)
    r = float(np.corrcoef(fp, it)[0, 1])
    ok = r < 0
    emit(f"  G1 (EV6 corr(FP,INT)): {r:+.3f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    cfg = parse_config(str(CONFIG))
    out = PKG / "results/exp027_constrained_payoff.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit("exp027: wirelength-CONSTRAINED payoff (Problem A). "
         f"obj = thermal + lambda*smoothHPWL. alpha={ALPHA}. "
         + ("[SMOKE — not a result]" if SMOKE else "[full run]"))
    emit(f"lambda grid: {LAMBDA_GRID}")

    results = []
    emit("\n--- G1 regime gates ---")

    # --- EV6 (PRIMARY): canonical design + canonical HotFloorplan netlist ---
    d = Design.from_flp(str(FLP))
    eu = d.macro_flp_dicts()
    enets = ev6_nets(eu)          # canonical ev6.desc connectivity (14 nets)
    emit(f"  EV6: {len(enets)} canonical nets from ev6.desc (HotFloorplan).")
    ev6_ok = gate_ev6(eu, emit)
    if ev6_ok:
        results.append(run_testbed(
            "EV6", eu, d.chip_width, d.chip_height, enets,
            lambda seed, n: StructuredWorkloadModel(eu, seed=seed).sample(n),
            cfg, emit))
    else:
        emit("  EV6 G1 FAILED — no anti-correlation; no EV6 sweep.")

    # --- SoC (secondary, stylised sensitivity check) ---
    su = soc_units()
    scw, sch = chip_box(su)
    snets = soc_nets(su)
    soc_ok = gate_soc(su, scw, sch, cfg, emit)
    if soc_ok:
        results.append(run_testbed(
            "SoC", su, scw, sch, snets,
            lambda seed, n: HeteroSoCWorkloadModel(su, seed=seed).sample(n),
            cfg, emit))
    else:
        emit("  SoC G1 FAILED — regime construction broken; no SoC sweep.")

    # --- Pre-registered verdicts, per testbed ---
    emit("\n===== PRE-REGISTERED VERDICTS =====")
    for res in results:
        tag, rows, binds = res["tag"], res["rows"], res["binds"]
        if not binds:
            emit(f"{tag}: NO VERDICT — G2 failed (constraint did not bind); the "
                 f"lambda grid is vacuous on this testbed (widen it and rerun).")
            continue
        D0 = rows[0]["D"]
        win = [r for r in rows if r["lam"] > 0 and r["holm"] and r["lo"] > 0
               and r["D"] > D0 and r["dMean"] <= 0]
        anywin = [r for r in rows if r["lam"] > 0 and r["holm"] and r["lo"] > 0]
        if win:
            b = max(win, key=lambda r: r["D"])
            emit(f"{tag}: PAYS-UNDER-CONSTRAINT — at lambda={b['lam']:.0f} "
                 f"(HPWL {100*b['hpm']/rows[0]['hpm']:.0f}% of unconstrained) "
                 f"D*={b['D']:+.3f} K [{b['lo']:+.2f},{b['hi']:+.2f}] Holm-sig, "
                 f"up from D*(0)={D0:+.3f}, dMean={b['dMean']:+.3f}<=0: scarcity "
                 f"creates a genuine mean-for-tail trade. Existence under the "
                 f"HPWL constraint, stylised netlist (scope: existence not prevalence).")
        elif anywin:
            b = max(anywin, key=lambda r: r["D"])
            emit(f"{tag}: PARTIAL — D* Holm-significant at lambda={b['lam']:.0f} "
                 f"(D*={b['D']:+.3f}) but NOT a clean trade (dMean={b['dMean']:+.3f}>0 "
                 f"= domination, or not above D*(0)={D0:+.3f}). Report, no slogan.")
        elif all(r["hi"] <= 0 for r in rows):
            emit(f"{tag}: NULL — every lambda has D* CI <= 0; the unconstrained "
                 f"null extends to the wirelength-constrained problem here.")
        else:
            emit(f"{tag}: INCONCLUSIVE — no lambda reaches Holm significance; "
                 f"underpowered ({N_OR_PAIRS} pairs). OPEN, not negative.")
        div = [r for r in rows if abs(r["hpc"]-r["hpm"])/(r["hpm"]+1e-12) > 0.10]
        if div:
            emit(f"  FAIRNESS CAVEAT: arms diverge >10% in wire at lambda="
                 f"{[int(r['lam']) for r in div]}; D* there mixes tail-quality with "
                 f"wire spent. Matched-HPWL follow-up is the refinement.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
