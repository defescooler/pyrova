"""exp029: the wirelength-constrained payoff on REAL power (BOOM) — Problem A, real arm.

Companion to exp027 (EV6-canonical, synthetic mechanism sweep). This is the
real-power instance of the field's canonical unit: ONE real design (BOOM, from
mcpat-calib) + REAL per-block power (80 RISC-V benchmarks) + a HIERARCHY-DERIVED
netlist (workloads/netlists.boom_nets, from the McPAT module tree — not
hand-drawn) + the HPWL budget as the synthetic parameter.

Question: as the wirelength budget tightens, does risk-aware (CVaR) or blend
placement beat mean placement OOS on real workloads? Design mirrors exp012's
powered BOOM split protocol, with ALL THREE audit traps controlled and the
exp012 verdict sin fixed:

  * budget (exp013):   all arms N_ITER, matched;
  * grid (exp018/020): train@18 + raster_jitter=1.0; eval@64 (independent);
  * baseline (exp019): ALL arms best-of-R restarts (symmetric strong baseline);
  * repeated overlapping splits -> Nadeau-Bengio-corrected CIs;
  * the dMean domination diagnostic is GATED INTO THE VERDICT (exp012 printed
    it but did not gate on it, declaring a "win" under full domination).

Per lambda: 20 splits of the 80 programs (60 train / 20 test), dCVaR and dMean of
{cvar, blend} vs mean, NB-CI, Holm across the lambda cells.

GATES (fail-closed):
  G2 binds : mean-arm HPWL falls (<= 0.9x) and mean peak rises across lambda.

PRE-REGISTERED READINGS (per arm):
  - PAYS-UNDER-CONSTRAINT if some lambda>0 has dCVaR NB-CI>0 AND dMean<=0 (real
    mean-for-tail trade, not domination) AND dCVaR(lambda)>dCVaR(0).
  - DOMINATION (not the theory) if dCVaR NB-CI>0 but dMean>0 there.
  - NULL / UNINFORMATIVE if no lambda clears NB-CI>0 (BOOM's ~2-3 program test
    tail at alpha=0.9 is the known power blocker, exp009).

Requires the BOOM dataset (set BOOM_DATA or clone into ./mcpat-calib-public).
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

from pyrova.thermal.fd_solver import GridFDSolver, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import cvar, ci95_nadeau_bengio, paired_t_p, holm
from pyrova.workloads.boom_traces import BoomWorkload, resolve_paths
from pyrova.workloads.netlists import boom_nets

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
ALPHA = 0.90
GAMMA = 0.75
TRAIN_GRID = 18
JITTER = 1.0
CONFIG_ID = "0"

EVAL_GRID = 32 if SMOKE else 64
N_ITER = 15 if SMOKE else 120
N_TRAIN = 60
N_SPLITS = 2 if SMOKE else 20
RESTARTS = 1 if SMOKE else 3
TARGET_PEAK = 40.0
LAMBDA_GRID = ([0.0, 20000.0] if SMOKE
               else [0.0, 1000.0, 5000.0, 20000.0, 80000.0])


def fit(solver, units, cw, ch, train, mode, wl_weight, nets, restarts, rrng, seed):
    best, best_obj = None, np.inf
    for r in range(restarts):
        # nonoverlap_w scaled with wl_weight (overlap areas ~1e-7 m^2; default 1e4
        # lets blocks stack to cheat HPWL). Keeps residual overlap <0.2%.
        now = max(1e8, 1e5 * wl_weight)
        pl = DiffPlacer(solver, units, cw, ch, TRAIN_GRID, TRAIN_GRID, alpha=ALPHA,
                        blend_gamma=GAMMA, nets=nets, wl_weight=wl_weight, nonoverlap_w=now)
        if r > 0:
            pl.raw_x += rrng.standard_normal(pl.n) * 0.5
            pl.raw_y += rrng.standard_normal(pl.n) * 0.5
        pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=290_000 + seed + 37 * r)
        obj = pl.objective_and_grad(train, mode=mode)[0]
        if obj < best_obj:
            best, best_obj = pl, obj
    return best


def eval_mc(cfg, units_placed, scen, cw, ch, ambient):
    s = GridFDSolver(cfg, units_placed, cw, ch, EVAL_GRID, EVAL_GRID)
    s.build(); s.factorize()
    pk = np.zeros(len(scen))
    for i, pw in enumerate(scen):
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(units_placed)}
        pk[i] = float(s.silicon_layer(s.solve(s.build_rhs(bp))).max()) - ambient
    return float(pk.mean()), cvar(pk, ALPHA)


def exact_hpwl(units_placed, nets):
    cx = np.array([u["leftx"] + u["width"] / 2 for u in units_placed])
    cy = np.array([u["bottomy"] + u["height"] / 2 for u in units_placed])
    return float(sum((cx[np.asarray(idx)].max() - cx[np.asarray(idx)].min())
                     + (cy[np.asarray(idx)].max() - cy[np.asarray(idx)].min())
                     for idx in nets))


def main():
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    out = PKG / "results/exp029_boom_constrained.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    csvp, rptp = resolve_paths()
    if not csvp:
        emit("BOOM dataset not found (set BOOM_DATA or clone mcpat-calib-public). "
             "No run.")
        fh.close(); return
    wl = BoomWorkload(csvp, rptp, config_id=CONFIG_ID)
    nets = boom_nets(wl.units, rptp)
    cw, ch = wl.chip_w, wl.chip_h
    solver = GridFDSolver(cfg, wl.units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()

    def train_peaks(scen):
        p = DiffPlacer(solver, wl.units, cw, ch, TRAIN_GRID, TRAIN_GRID, alpha=ALPHA)
        return p._scenario_peaks(*p.get_positions(), scen)
    wl.scale_to_peak(train_peaks, TARGET_PEAK)
    scen = wl.scenarios()
    n_prog = len(scen)

    emit(f"exp029: wirelength-CONSTRAINED payoff on REAL BOOM power. "
         f"{n_prog} programs, {len(wl.units)} leaves, {len(nets)} hierarchy nets, "
         f"corr(FP,INT)={wl.family_corr()['FP_INT']:+.3f}, totCV={wl.total_power_cv():.3f}. "
         + ("[SMOKE - not a result]" if SMOKE else "[full run]"))
    emit(f"alpha={ALPHA}, train@{TRAIN_GRID}+jitter{JITTER}, eval@{EVAL_GRID}, "
         f"{N_ITER} it, ALL arms best-of-{RESTARTS} (symmetric), {N_SPLITS} splits "
         f"60/20, NB-CI. lambda grid: {LAMBDA_GRID}")

    ratio = (n_prog - N_TRAIN) / N_TRAIN
    rows = []
    for lam in LAMBDA_GRID:
        dC = {"cvar": [], "blend": []}
        dM = {"cvar": [], "blend": []}
        hpm, pkm = [], []
        for sp in range(N_SPLITS):
            perm = np.random.default_rng(900_000 + sp).permutation(n_prog)
            tr = [scen[i] for i in perm[:N_TRAIN]]
            te = [scen[i] for i in perm[N_TRAIN:]]
            rng_m = np.random.default_rng(910_000 + sp)
            pm = fit(solver, wl.units, cw, ch, tr, "mean", lam, nets, RESTARTS, rng_m, sp)
            um = pm.get_units()
            m_m, c_m = eval_mc(cfg, um, te, cw, ch, ambient)
            hpm.append(exact_hpwl(um, nets)); pkm.append(m_m)
            for arm, mode in (("cvar", "cvar"), ("blend", "blend")):
                rng_a = np.random.default_rng(920_000 + 100 * sp + (0 if arm == "cvar" else 1))
                pa = fit(solver, wl.units, cw, ch, tr, mode, lam, nets, RESTARTS, rng_a, sp)
                m_a, c_a = eval_mc(cfg, pa.get_units(), te, cw, ch, ambient)
                dC[arm].append(c_m - c_a); dM[arm].append(m_m - m_a)
            print(f"  lambda={lam:.0f} split {sp + 1}/{N_SPLITS}", flush=True)
        row = {"lam": lam, "hpm": float(np.mean(hpm)), "peak": float(np.mean(pkm))}
        for arm in ("cvar", "blend"):
            gc, _, clo, chi = ci95_nadeau_bengio(dC[arm], ratio)
            gm, _, mlo, mhi = ci95_nadeau_bengio(dM[arm], ratio)
            row[arm] = dict(dC=gc, clo=clo, chi=chi, dM=gm, mlo=mlo, mhi=mhi,
                            p=paired_t_p(dC[arm]))
            emit(f"  lambda={lam:7.0f} {arm:5s}: dCVaR={gc:+.3f} [{clo:+.3f},{chi:+.3f}] "
                 f"dMean={gm:+.3f} [{mlo:+.3f},{mhi:+.3f}]  (HPWL={row['hpm']*1e3:.2f}mm, "
                 f"peak={row['peak']:.2f}K)")
        rows.append(row)

    # Holm across the lambda cells, per arm.
    for arm in ("cvar", "blend"):
        keep = holm([r[arm]["p"] for r in rows])
        for r, k in zip(rows, keep):
            r[arm]["holm"] = bool(k)

    hp0, hpL = rows[0]["hpm"], rows[-1]["hpm"]
    binds = (hpL <= 0.9 * hp0) and (rows[-1]["peak"] > rows[0]["peak"])
    emit(f"\n  G2 (constraint binds): HPWL {hp0*1e3:.2f}->{hpL*1e3:.2f} mm, "
         f"peak {rows[0]['peak']:.2f}->{rows[-1]['peak']:.2f} K -> "
         f"{'PASS' if binds else 'FAIL'}")

    emit("\n===== PRE-REGISTERED VERDICTS =====")
    if not binds:
        emit("NO VERDICT — G2 failed (constraint did not bind); widen the lambda grid.")
    for arm in ("cvar", "blend"):
        d0 = rows[0][arm]["dC"]
        trade = [r for r in rows if r["lam"] > 0 and r[arm]["holm"]
                 and r[arm]["clo"] > 0 and r[arm]["dM"] <= 0 and r[arm]["dC"] > d0]
        dom = [r for r in rows if r["lam"] > 0 and r[arm]["holm"]
               and r[arm]["clo"] > 0 and r[arm]["dM"] > 0]
        if binds and trade:
            b = max(trade, key=lambda r: r[arm]["dC"])
            emit(f"{arm}: PAYS-UNDER-CONSTRAINT — at lambda={b['lam']:.0f} "
                 f"(HPWL {100*b['hpm']/rows[0]['hpm']:.0f}%) dCVaR={b[arm]['dC']:+.3f} "
                 f"[{b[arm]['clo']:+.2f},{b[arm]['chi']:+.2f}] Holm-sig, dMean="
                 f"{b[arm]['dM']:+.3f}<=0: a real mean-for-tail trade on REAL power "
                 f"under the HPWL constraint.")
        elif binds and dom:
            b = max(dom, key=lambda r: r[arm]["dC"])
            emit(f"{arm}: DOMINATION (not the theory) — dCVaR>0 at lambda={b['lam']:.0f} "
                 f"but dMean={b[arm]['dM']:+.3f}>0 (beats mean on ITS metric too = "
                 f"under-converged baseline, exp013 pattern), gated OUT.")
        else:
            emit(f"{arm}: NULL / UNINFORMATIVE — no lambda clears NB-CI>0 (BOOM's "
                 f"~2-3 program test tail at alpha={ALPHA} is the power blocker, exp009). "
                 f"OPEN, not negative.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
