"""Kraken oracle gap against single-start and best-of-3-restart mean
baselines: per-pair dCVaR and dMean at alpha=0.9, fresh evaluation set per
pair, CRN within a pair, train@18+jitter with density spreading,
guaranteed-legal eval@64^2, power calibrated to a 40 K operating point.

Sharded via PYROVA_PAIRS / PYROVA_PAIR_OFFSET; pool per-pair lines with
pyrova.evaluation.pool. PYROVA_SMOKE=1 for an execution check.
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
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p
from pyrova.workloads.kraken_soc import kraken_units, KrakenWorkloadModel

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
EVAL_ALPHA = 0.90
TARGET_PEAK = 40.0
TRAIN_GRID = 18
JITTER = 1.0
RESTART_SIGMA = 1.0            # sigmoid-space init jitter for the restarted arm
DENSITY_W = 5.0e2
DENSITY_LAM0 = 3.0e1
DENSITY_GRID = (48, 48)

EVAL_GRID = 32 if SMOKE else 64
N_ITER = 12 if SMOKE else 100
N_ORACLE = 100 if SMOKE else 2000
N_PAIRS = int(os.environ.get("PYROVA_PAIRS", 2 if SMOKE else 10))
PAIR_OFFSET = int(os.environ.get("PYROVA_PAIR_OFFSET", "0"))
N_TEST = 200 if SMOKE else 2000
N_CALIB = 50 if SMOKE else 300
K_RESTART = 3

TRAIN_SEED_BASE = 10_000       # same streams as the prior small-SoC runs: single-start arms reproduce those pairs
JITTER_SEED_BASE = 500_000
FRESH_TEST_BASE = 900_000
RESTART_SEED_BASE = 800_000    # disjoint from every stream above


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


def main():
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    units = kraken_units()
    cw = max(u["leftx"] + u["width"] for u in units)
    ch = max(u["bottomy"] + u["height"] for u in units)

    tag = "" if PAIR_OFFSET == 0 else f"_off{PAIR_OFFSET}"
    out = PKG / f"results/exp047_kraken_baselines{tag}.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"kraken oracle gap vs mean baselines: {len(units)} blocks, die {cw*1e3:.1f}x{ch*1e3:.1f}mm. "
         f"{N_PAIRS} pairs (offset {PAIR_OFFSET}), N_ORACLE={N_ORACLE}, {N_ITER} it, "
         f"arms: mean-std (1 start) / mean-strong (best-of-{K_RESTART}) / cvar (1 start), "
         f"fresh eval set per pair (N_TEST={N_TEST}), train@{TRAIN_GRID}+jitter -> "
         f"guaranteed-legal eval@{EVAL_GRID} (alpha={EVAL_ALPHA}). "
         + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    # Calibrate once on the tiled layout (dT linear in power); all pair models
    # share the calibrated scale so every pair sees the same distribution.
    se = GridFDSolver(cfg, units, cw, ch, EVAL_GRID, EVAL_GRID)
    se.build(); se.factorize()

    def peak(solver, up, pw):
        return float(solver.silicon_layer(solver.solve(solver.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - ambient

    calib = KrakenWorkloadModel(units, seed=1)
    m0 = float(np.mean([peak(se, units, pw) for pw in calib.sample(N_CALIB)]))
    c = TARGET_PEAK / m0
    emit(f"calibration: tiled mean peak {m0:.1f}K -> power x{c:.3f} for a "
         f"{TARGET_PEAK:.0f}K operating point")

    def make_scaled(seed):
        m = KrakenWorkloadModel(units, seed=seed)
        m.scale = m.scale * c
        return m

    hot = {block_at_peak(units, cw, ch, EVAL_GRID, EVAL_GRID,
                         se.silicon_layer(se.solve(se.build_rhs(
                             {u["name"]: float(v) for u, v in zip(units, mv * c)}))))
           for mv in calib.modes}
    g0 = len(hot) >= 3
    emit(f"G0 (mechanism): hotspot blocks over modes: {len(hot)} {sorted(hot)} -> "
         f"{'PASS' if g0 else 'FAIL'}")
    if not g0:
        emit("\n===== PRE-REGISTERED VERDICT =====")
        emit("NO VERDICT — mechanism gate failed."); fh.close(); return

    st = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    st.build(); st.factorize()

    def eval_cvar_mean(up, test):
        sv = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
        sv.build(); sv.factorize()
        pk = np.array([peak(sv, up, pw) for pw in test])
        return cvar(pk, EVAL_ALPHA), float(pk.mean())

    def fit(tr, mode, jseed, restarts=1, rrng=None):
        """The best restart is picked on the arm's own training objective; the
        strong baseline is never tuned on evaluation data."""
        best, best_obj, best_of = None, np.inf, None
        for r in range(restarts):
            pl = DiffPlacer(st, units, cw, ch, TRAIN_GRID, TRAIN_GRID,
                            alpha=EVAL_ALPHA, nonoverlap_w=1e4, density_w=DENSITY_W,
                            density_lam0=DENSITY_LAM0, density_grid=DENSITY_GRID)
            if r > 0:
                pl.raw_x += rrng.standard_normal(pl.n) * RESTART_SIGMA
                pl.raw_y += rrng.standard_normal(pl.n) * RESTART_SIGMA
            pl.optimize(tr, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                        raster_jitter=JITTER, jitter_seed=jseed)
            obj = pl.objective_and_grad(tr, mode=mode)[0]
            if obj < best_obj:
                up, of = legalize_units_exact(pl.get_units(), cw, ch)
                best, best_obj, best_of = up, obj, of
        return best, best_of

    d_std, d_strong, m_std, m_strong = [], [], [], []
    ovl_max = 0.0
    infeasible = 0

    for k in range(PAIR_OFFSET, PAIR_OFFSET + N_PAIRS):
        tr = make_scaled(TRAIN_SEED_BASE + k).sample(N_ORACLE)
        js = JITTER_SEED_BASE + 1000 * k                 # CRN across all three arms
        rrng = np.random.default_rng(RESTART_SEED_BASE + k)
        try:
            um, fm = fit(tr, "mean", js)
            us, fs = fit(tr, "mean", js, restarts=K_RESTART, rrng=rrng)
            uc, fc = fit(tr, "cvar", js)
        except LegalizationInfeasible:
            infeasible += 1
            emit(f"  pair {k}: INFEASIBLE, skipped")
            continue
        ovl_max = max(ovl_max, fm, fs, fc)

        test = make_scaled(FRESH_TEST_BASE + k).sample(N_TEST)
        c_m, mu_m = eval_cvar_mean(um, test)
        c_s, mu_s = eval_cvar_mean(us, test)
        c_c, mu_c = eval_cvar_mean(uc, test)
        d_std.append(c_m - c_c); m_std.append(mu_m - mu_c)
        d_strong.append(c_s - c_c); m_strong.append(mu_s - mu_c)

        # Pool-readable per-pair lines (one estimate per arm contrast).
        emit(f"  pair {k}: dCVaR_std = {d_std[-1]:+.4f}   dCVaR_strong = {d_strong[-1]:+.4f}   "
             f"dMean_std = {m_std[-1]:+.4f}   dMean_strong = {m_strong[-1]:+.4f}")

    g2 = ovl_max < 1e-3 and infeasible == 0
    emit(f"\nG2 (legality): max overlap {100*ovl_max:.3f}%, {infeasible} infeasible -> "
         f"{'PASS' if g2 else 'FAIL'}")

    emit("\n===== PRE-REGISTERED VERDICT (this shard; pool across shards) =====")
    if SMOKE:
        emit("NO VERDICT — smoke run (plumbing check only).")
        fh.close(); return
    if not g2 or len(d_std) < 3:
        emit("NO VERDICT — legality gate failed or too few feasible pairs.")
        fh.close(); return

    for name, dc, dm in (("vs mean-std   ", d_std, m_std),
                         ("vs mean-strong", d_strong, m_strong)):
        dc, dm = np.array(dc), np.array(dm)
        mc, _, lc, hc = ci95_t(dc)
        mm, _, lm, hm = ci95_t(dm)
        shape = "trade (mean-for-tail)" if mm <= 0 else "domination (under-optimized baseline)"
        emit(f"{name}: dCVaR = {mc:+.4f} [{lc:+.4f},{hc:+.4f}] p={paired_t_p(dc):.4f}   "
             f"dMean = {mm:+.4f} [{lm:+.4f},{hm:+.4f}] -> {shape}")

    dcs = np.array(d_strong); dms = np.array(m_strong)
    _, _, lo_s, _ = ci95_t(dcs)
    mm_s = float(np.mean(dms))
    if lo_s > 0 and mm_s <= 0:
        emit("READING: placement quality — the gap survives the restarted baseline "
             "with the trade shape.")
    elif lo_s > 0:
        emit("READING: gap survives the restarted baseline but wins on both metrics — "
             "residual under-optimization; do not quote as a mean-for-tail trade.")
    else:
        emit("READING: baseline-strength artifact — the restarted mean arm closes the gap.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
