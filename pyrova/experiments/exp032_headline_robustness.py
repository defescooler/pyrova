"""exp032: is the claim-16 headline robust to the reviewer's three cheap
knobs — tail level alpha, evaluation grid, and hotspot metric?

The +0.068 K favorable-regime positive (exp023, legality-verified exp031) was
measured at alpha=0.90, eval@64, single-hottest-node peak. Reviewers will ask
whether it is an artifact of those three choices. This experiment reproduces
exp023's E1 EXACTLY (pinned seeds, 5 oracle pairs, hetero-SoC) under legalized
evaluation (gate-0 protocol) and re-scores every placement across:

  alpha  in {0.90, 0.95}
  grid   in {64, 96, 128}          (is 64^2 converged?)
  metric in {peak, hotspot1pct}    (peak = single hottest silicon node;
                                     hotspot1pct = mean of hottest 1% of
                                     nodes — a less spiky integrated hotspot)

Training is UNCHANGED (mean arm best-of-3, alpha-independent; cvar arm trained
per alpha on the single-node peak — we vary the EVALUATION, not the objective:
the honest test is "does a placement optimized for spiky max keep its tail
advantage under smoother metrics / finer grids / a stricter tail?").

NETLIST sensitivity (the 4th caveat) is NOT here — it applies only to the
CONSTRAINED problem (exp030), which has nets; the unconstrained headline has
none. Flagged for exp030's design, not smuggled in.

PRE-REGISTERED READING (a robustness confirmation, not a fresh significance
hunt): claim 16 is ROBUST if all 12 cells (2 alpha x 3 grid x 2 metric) have
D* point estimate > 0 AND none has CI strictly < 0, with the (0.90, 64, peak)
cell reproducing exp031's +0.072. FRAGILE if any cell flips sign with CI<0
(a knob, not the mechanism, was carrying the result).

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
from pyrova.optimizer.legalize import legalize_units, overlap_frac
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p, holm
from pyrova.workloads.hetero_soc import HeteroSoCWorkloadModel, soc_units

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
TRAIN_GRID = 18
N_ITER = 10 if SMOKE else 120
N_ORACLE = 40 if SMOKE else 1000
N_OR_PAIRS = 2 if SMOKE else 5
N_TEST = 100 if SMOKE else 1000
JITTER = 1.0
ALPHAS = [0.90, 0.95]
GRIDS = [64] if SMOKE else [64, 96, 128]
METRICS = ["peak", "hot1"]
HOT_FRAC = 0.01
JITTER_BASE = 910_000   # exp023's E1 jitter base — EXACT reproduction


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def fit(solver, units, cw, ch, train, mode, alpha, restarts=1, rrng=None, seed=0):
    best, best_obj = None, np.inf
    for r in range(restarts):
        pl = DiffPlacer(solver, units, cw, ch, TRAIN_GRID, TRAIN_GRID,
                        alpha=alpha, blend_gamma=0.75)
        if r > 0:
            pl.raw_x += rrng.standard_normal(pl.n) * 0.5
            pl.raw_y += rrng.standard_normal(pl.n) * 0.5
        pl.optimize(train, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=JITTER_BASE + seed + 37 * r)
        obj = pl.objective_and_grad(train, mode=mode)[0]
        if obj < best_obj:
            best, best_obj = pl, obj
    return best


def per_scenario(cfg, up, scen, cw, ch, ambient, grid):
    """Return {metric: array of per-scenario values} at one eval grid."""
    s = GridFDSolver(cfg, up, cw, ch, grid, grid)
    s.build(); s.factorize()
    peak = np.zeros(len(scen)); hot = np.zeros(len(scen))
    for i, pw in enumerate(scen):
        sil = s.silicon_layer(s.solve(s.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(up)})))
        peak[i] = float(sil.max()) - ambient
        k = max(1, int(len(sil) * HOT_FRAC))
        hot[i] = float(np.sort(sil)[-k:].mean()) - ambient
    return {"peak": peak, "hot1": hot}


def cvar_cells(vals):
    """{ (grid,metric,alpha): cvar } from a {grid:{metric:array}} dict."""
    out = {}
    for g, mv in vals.items():
        for m, arr in mv.items():
            for a in ALPHAS:
                out[(g, m, a)] = cvar(arr, a)
    return out


def main():
    units = soc_units()
    cfg = parse_config(str(CONFIG))
    cw, ch = chip_box(units)
    ambient = cfg["ambient"]
    solver = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    solver.build(); solver.factorize()

    out = PKG / "results/exp032_headline_robustness.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"exp032 (claim-16 robustness): hetero-SoC E1, {N_OR_PAIRS} pairs, "
         f"N_ORACLE={N_ORACLE}, legalized eval. alpha={ALPHAS} x grid={GRIDS} x "
         f"metric={METRICS} (hot1=mean of hottest {100*HOT_FRAC:.0f}% nodes). "
         + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    test = HeteroSoCWorkloadModel(units, seed=777).sample(N_TEST)
    D = {}
    for k in range(N_OR_PAIRS):
        train = HeteroSoCWorkloadModel(units, seed=10_000 + k).sample(N_ORACLE)
        rrng = np.random.default_rng(20_000 + k)
        um = legalize_units(fit(solver, units, cw, ch, train, "mean", ALPHAS[0],
                                restarts=3, rrng=rrng, seed=k).get_units(), cw, ch)[0]
        em = {g: per_scenario(cfg, um, test, cw, ch, ambient, g) for g in GRIDS}
        cm = cvar_cells(em)
        for a in ALPHAS:
            uc = legalize_units(fit(solver, units, cw, ch, train, "cvar", a,
                                    seed=k).get_units(), cw, ch)[0]
            ec = {g: per_scenario(cfg, uc, test, cw, ch, ambient, g) for g in GRIDS}
            cc = cvar_cells(ec)
            for g in GRIDS:
                for m in METRICS:
                    D.setdefault((g, m, a), []).append(cm[(g, m, a)] - cc[(g, m, a)])
        print(f"  pair {k + 1}/{N_OR_PAIRS} done", flush=True)

    emit(f"\n  {'grid':>4} {'metric':>6} {'alpha':>5} {'D*':>8} "
         f"{'CI_lo':>8} {'CI_hi':>8} {'p':>7}")
    cells = []
    for a in ALPHAS:
        for g in GRIDS:
            for m in METRICS:
                d = D[(g, m, a)]
                gg, _, lo, hi = ci95_t(d)
                p = paired_t_p(d)
                fl = "*" if lo > 0 else ("x" if hi < 0 else " ")
                emit(f"  {g:>4} {m:>6} {a:>5.2f} {gg:>+8.3f}{fl} "
                     f"{lo:>+8.3f} {hi:>+8.3f} {p:>7.4f}")
                cells.append(dict(g=g, m=m, a=a, d=gg, lo=lo, hi=hi, p=p))

    ref = next(c for c in cells if c["g"] == 64 and c["m"] == "peak" and c["a"] == 0.90)
    all_pos = all(c["d"] > 0 for c in cells)
    no_flip = all(c["hi"] >= 0 for c in cells)
    keep = holm([c["p"] for c in cells])
    n_sig = sum(1 for c, k in zip(cells, keep) if k and c["lo"] > 0)
    emit(f"\n  reproduction: (0.90,64,peak) D*={ref['d']:+.3f} "
         f"(exp031 legal was +0.072)")
    emit(f"  {sum(c['d'] > 0 for c in cells)}/{len(cells)} cells positive; "
         f"{n_sig} Holm-significant (CI>0).")
    emit("\nPRE-REGISTERED VERDICT: " + (
        "ROBUST — claim 16 holds across alpha, eval grid, and hotspot metric; "
        "the +0.07 K positive is not a knob artifact." if (all_pos and no_flip) else
        "FRAGILE — at least one cell flips sign with CI<0; the headline depends "
        "on an evaluation choice. Cells: "
        + str([(c["g"], c["m"], c["a"]) for c in cells if c["hi"] < 0])))
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
