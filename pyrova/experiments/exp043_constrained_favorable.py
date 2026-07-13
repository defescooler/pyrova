"""Wirelength-constrained payoff on real nets with multimodal power: MCNC
ami33's netlist (78 signal nets) and geometry with an engineered mode-mixture
workload over its 33 blocks (the 6 largest blocks become engines, each mode
driving a different engine near max), total power calibrated to a 40 K
mean-peak operating point. Per HPWL weight lambda, D*(lambda) =
trueCVaR(mean-arm) - trueCVaR(risk-arm) at N_ORACLE=2000, arms cvar
(alpha=0.9) and cvar_wide (CVaR trained at alpha=0.6, scored at 0.9). Matched
budgets, train@18 with raster jitter, density-spread fitting, guaranteed-legal
evaluation on an independent 64^2 grid, common random numbers across arms,
shared scenario streams across lambda cells (cross-lambda contrasts paired per
seed), dMean alongside, Holm across cells; fail-closed gates: >=3 distinct
hotspot blocks across the modes, the constraint binds (mean-arm legal HPWL at
max lambda <= 0.88x its lambda=0 value), guaranteed legality.

PYROVA_SCALE (default 1.0) shrinks all lengths for a small-die re-run.
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

from pyrova.thermal.fd_solver import GridFDSolver, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.optimizer.legalize import legalize_units_exact, LegalizationInfeasible
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p, holm
from pyrova.workloads.mcnc import load_yal

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"
SCALE = float(os.environ.get("PYROVA_SCALE", "1.0"))

CONFIG = PKG / "inputs/configs/thermal.config"
EVAL_ALPHA = 0.90
TARGET_PEAK = 40.0
TRAIN_GRID = 18
JITTER = 1.0
DENSITY_W = 5.0e2
DENSITY_LAM0 = 3.0e1
DENSITY_GRID = (48, 48)

N_ENGINES = 6                 # the 6 largest blocks become mode-driven engines
ENGINE_DENSITY = 3.0          # engine max power density, W/mm^2 (pre-calibration)
BACKGROUND_DENSITY = 0.8      # non-engine density at full activity
NOISE = 0.15
IDLE_P = 0.25                 # idle-mode probability; active modes share the rest

EVAL_GRID = 32 if SMOKE else 64
N_ITER = 12 if SMOKE else 100
N_ORACLE = 100 if SMOKE else 2000
N_PAIRS = 2 if SMOKE else 5
N_TEST = 200 if SMOKE else 2000
N_CALIB = 50 if SMOKE else 300
LAMBDA_GRID = [0.0, 3000.0] if SMOKE else [0.0, 1000.0, 3000.0, 10000.0]
ARMS = {"cvar": ("cvar", 0.90), "cvar_wide": ("cvar", 0.60)}


class ModalAmi33Model:
    """Mode-mixture power over the ami33 blocks: each active mode drives one
    engine near max while the others idle — heavy anti-correlation by
    construction, the hotspot moving between the engine blocks."""

    def __init__(self, units, seed=0):
        self.rng = np.random.default_rng(seed)
        area = np.array([u["width"] * u["height"] * 1e6 for u in units])  # mm^2
        eng = np.argsort(area)[::-1][:N_ENGINES]
        self.engines = sorted(int(i) for i in eng)
        self.pmax = area * BACKGROUND_DENSITY
        self.pmax[self.engines] = area[self.engines] * ENGINE_DENSITY
        # activity per mode: own engine 1.0, other engines 0.10, background 0.30
        self.modes = []
        for e in self.engines:
            act = np.full(len(units), 0.30)
            act[self.engines] = 0.10
            act[e] = 1.00
            self.modes.append(act)
        self.modes.append(np.full(len(units), 0.08))                      # idle
        self.mode_p = np.array([(1.0 - IDLE_P) / N_ENGINES] * N_ENGINES + [IDLE_P])

    def sample(self, n):
        out = []
        for _ in range(n):
            m = self.rng.choice(len(self.modes), p=self.mode_p)
            p = self.modes[m] * self.pmax
            p = p * (1.0 + self.rng.uniform(-NOISE, NOISE, size=len(p)))
            out.append(np.maximum(p, 1e-4))
        return out


def block_at_peak(units, cw, ch, nx, ny, T):
    j, i = np.unravel_index(int(np.argmax(T)), T.shape)
    px, py = (i + 0.5) * cw / nx, (j + 0.5) * ch / ny
    for u in units:
        if (u["leftx"] <= px <= u["leftx"] + u["width"]
                and u["bottomy"] <= py <= u["bottomy"] + u["height"]):
            return u["name"]
    cx = np.array([u["leftx"] + u["width"] / 2 for u in units])
    cy = np.array([u["bottomy"] + u["height"] / 2 for u in units])
    return units[int(np.argmin((cx - px) ** 2 + (cy - py) ** 2))]["name"]


def exact_hpwl(up, nets):
    cx = np.array([u["leftx"] + u["width"] / 2 for u in up])
    cy = np.array([u["bottomy"] + u["height"] / 2 for u in up])
    return float(sum((cx[np.asarray(i)].max() - cx[np.asarray(i)].min())
                     + (cy[np.asarray(i)].max() - cy[np.asarray(i)].min())
                     for i in nets))


def main():
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    units, nets, cw, ch = load_yal(ROOT / "ami33.yal.txt", utilization=0.55)
    if SCALE != 1.0:
        for u in units:
            for k in ("width", "height", "leftx", "bottomy"):
                u[k] *= SCALE
        cw *= SCALE; ch *= SCALE
    n = len(units)

    out = PKG / f"results/exp043_constrained_favorable{'' if SCALE == 1.0 else f'_s{SCALE:g}'}.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    model = ModalAmi33Model(units, seed=0)
    emit(f"constrained+favorable [ami33 nets x modal power]: {n} blocks, {len(nets)} nets, "
         f"die {cw*1e3:.1f}x{ch*1e3:.1f}mm (scale {SCALE:g}), engines="
         f"{[units[i]['name'] for i in model.engines]}. N_ORACLE={N_ORACLE}, "
         f"{N_PAIRS} pairs, {N_ITER} it, train@{TRAIN_GRID}+jitter -> "
         f"guaranteed-legal eval@{EVAL_GRID} (alpha={EVAL_ALPHA}). "
         f"lambda={LAMBDA_GRID}. " + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    # Calibrate the operating point on the benchmark layout (dT linear in power).
    se = GridFDSolver(cfg, units, cw, ch, EVAL_GRID, EVAL_GRID)
    se.build(); se.factorize()

    def peak(solver, up, pw):
        return float(solver.silicon_layer(solver.solve(solver.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - ambient

    raw = ModalAmi33Model(units, seed=99).sample(max(N_CALIB, 1))
    m0 = float(np.mean([peak(se, units, pw) for pw in raw]))
    c = TARGET_PEAK / m0
    emit(f"calibration: benchmark-layout mean peak {m0:.1f}K -> power x{c:.3f} "
         f"for a {TARGET_PEAK:.0f}K operating point")

    # G0: the modes must resolve distinct hotspots on this geometry.
    hot = {block_at_peak(units, cw, ch, EVAL_GRID, EVAL_GRID,
                         se.silicon_layer(se.solve(se.build_rhs(
                             {u["name"]: float(v) for u, v in
                              zip(units, mv * model.pmax * c)}))))
           for mv in model.modes[:N_ENGINES]}
    g0 = len(hot) >= 3
    emit(f"G0 (mechanism): hotspot blocks over modes: {len(hot)} {sorted(hot)} -> "
         f"{'PASS' if g0 else 'FAIL'}")
    if not g0:
        emit("\n===== PRE-REGISTERED VERDICT =====")
        emit("NO VERDICT — the engineered modes do not create mobile hotspots here.")
        fh.close(); return

    test = [pw * c for pw in ModalAmi33Model(units, seed=98).sample(N_TEST)]
    st = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    st.build(); st.factorize()

    def eval_cvar_mean(up):
        sv = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
        sv.build(); sv.factorize()
        pk = np.array([peak(sv, up, pw) for pw in test])
        return cvar(pk, EVAL_ALPHA), float(pk.mean())

    def fit(tr, mode, alpha, lam, jseed):
        pl = DiffPlacer(st, units, cw, ch, TRAIN_GRID, TRAIN_GRID, alpha=alpha,
                        nets=nets, wl_weight=lam, nonoverlap_w=1e4,
                        density_w=DENSITY_W, density_lam0=DENSITY_LAM0,
                        density_grid=DENSITY_GRID)
        pl.optimize(tr, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=jseed)
        up, of = legalize_units_exact(pl.get_units(), cw, ch)
        return up, of

    rows = []
    ovl_max = 0.0
    hp_mean = {}
    infeasible = 0
    for lam in LAMBDA_GRID:
        cell = {a: {"dC": [], "dM": []} for a in ARMS}
        hps = []
        for k in range(N_PAIRS):
            tr = [pw * c for pw in ModalAmi33Model(units, seed=10_000 + k).sample(N_ORACLE)]
            js = 500_000 + 1000 * k
            try:
                um, fm = fit(tr, "mean", 0.90, lam, js)
                arms_legal = {a: fit(tr, mode, al, lam, js)
                              for a, (mode, al) in ARMS.items()}
            except LegalizationInfeasible:
                infeasible += 1
                emit(f"  lambda={lam:.0f} pair {k+1}: INFEASIBLE, skipped")
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
    g1 = hpL <= 0.88 * hp0
    g2 = ovl_max < 1e-3 and infeasible == 0
    emit(f"\nG1 (constraint binds): {hp0*1e3:.1f}->{hpL*1e3:.1f}mm "
         f"({100*hpL/hp0:.1f}%) -> {'PASS' if g1 else 'FAIL'}")
    emit(f"G2 (legality): max overlap {100*ovl_max:.3f}%, {infeasible} infeasible -> "
         f"{'PASS' if g2 else 'FAIL'}")

    emit("\n===== PRE-REGISTERED VERDICT =====")
    if not (g1 and g2) or not rows:
        emit("NO VERDICT — a gate failed or too few feasible pairs."); fh.close(); return
    base = {a: next((r for r in rows if r["lam"] == 0 and r["arm"] == a), None)
            for a in ARMS}
    for a, r0 in base.items():
        if r0 is not None:
            sig = " (CI>0)" if r0["lo"] > 0 else " (ns)"
            emit(f"unconstrained existence, {a}: D*(0)={r0['D']:+.4f} "
                 f"[{r0['lo']:+.4f},{r0['hi']:+.4f}]{sig}")
    win = [r for r in rows if r["lam"] > 0 and r.get("holm") and r["lo"] > 0
           and r["dM"] <= 0 and (base[r["arm"]] is None or r["D"] >= base[r["arm"]]["D"])]
    if win:
        emit("PAYS-UNDER-CONSTRAINT: " + ", ".join(
            f"lambda={r['lam']:.0f}/{r['arm']} D*={r['D']:+.3f}" for r in win))
    else:
        emit("NULL: no lambda>0 cell is Holm-sig>0 with dMean<=0 and D*(lambda)>=D*(0) "
             "— with both ingredients present, scarcity did not certify a payoff.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
