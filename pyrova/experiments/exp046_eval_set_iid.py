"""Fresh-vs-shared evaluation-set contrast on the EV6 i.i.d. oracle gap:
each pair is scored on its own evaluation scenarios and on one shared
reference set, plus a within-pair bootstrap. Raw (unlegalized) placements
are evaluated and residual overlap is reported: the fully utilized EV6 die
admits no legal projection.

Per pair: symmetric single-start arms, CRN, N_ORACLE=6000, 240 iterations,
train@24+jitter -> eval@64^2. Sharded via PYROVA_PAIRS / PYROVA_PAIR_OFFSET;
pool per-pair lines with pyrova.evaluation.pool. PYROVA_SMOKE=1 for an
execution check.
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

from pyrova.thermal.fd_solver import GridFDSolver, parse_config, parse_flp
from pyrova.optimizer.placer import DiffPlacer
from pyrova.objectives.overlap import nonoverlap_penalty
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p
from pyrova.experiments.exp003_mean_cvar_correlation import scen_set, chip_box

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"

CONFIG = PKG / "inputs/configs/thermal.config"
FLP = Path(os.environ.get("PYROVA_FLP", PKG / "inputs/floorplans/ev6.flp"))
EVAL_ALPHA = 0.90
TRAIN_GRID = 24
EVAL_GRID = 32 if SMOKE else 64
JITTER = 1.0
DENSITY_W = 5.0e2
DENSITY_LAM0 = 3.0e1
DENSITY_GRID = (48, 48)

N_ITER = int(os.environ.get("PYROVA_BUDGET", 15 if SMOKE else 240))
N_ORACLE = int(os.environ.get("PYROVA_NORACLE", 96 if SMOKE else 6000))
N_PAIRS = int(os.environ.get("PYROVA_PAIRS", 2 if SMOKE else 3))
PAIR_OFFSET = int(os.environ.get("PYROVA_PAIR_OFFSET", "0"))
N_TEST = 200 if SMOKE else 8000
N_BOOT = 50 if SMOKE else 400

SHARED_TEST_SEED = 99           # shared reference set
FRESH_TEST_BASE = 900_000       # per-pair evaluation sets


def main():
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    units = parse_flp(str(FLP))
    cw, ch = chip_box(units)
    tot = 2.0 * len(units)

    tag = f"{N_ORACLE}" if PAIR_OFFSET == 0 else f"{N_ORACLE}_off{PAIR_OFFSET}"
    out = PKG / f"results/exp046_eval_set_iid_{tag}.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"evaluation-set conditioning [i.i.d., {FLP.stem}]: {N_PAIRS} pairs "
         f"(offset {PAIR_OFFSET}), N_ORACLE={N_ORACLE}, {N_ITER} it, N_TEST={N_TEST} "
         f"(fresh per pair vs one shared set), {N_BOOT} bootstrap resamples, "
         f"train@{TRAIN_GRID}+jitter -> eval@{EVAL_GRID} (raw placements, as exp039; "
         f"alpha={EVAL_ALPHA}). " + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    shared_test = scen_set(units, tot, np.random.default_rng(SHARED_TEST_SEED), N_TEST)

    st = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    st.build(); st.factorize()

    def fit(tr, mode, jseed):
        """No density term and no legalization pass: EV6 tiles its die at 100%
        utilization, so raw placements are the object under study."""
        pl = DiffPlacer(st, units, cw, ch, TRAIN_GRID, TRAIN_GRID, alpha=EVAL_ALPHA)
        pl.optimize(tr, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=jseed)
        return pl.get_units()

    block_area = sum(u["width"] * u["height"] for u in units)

    def overlap_frac(up):
        """Residual pairwise overlap as a fraction of block area; reported, not
        gated, since raw placements are what this protocol evaluates."""
        cx = np.array([u["leftx"] + u["width"] / 2 for u in up])
        cy = np.array([u["bottomy"] + u["height"] / 2 for u in up])
        w = np.array([u["width"] for u in up])
        h = np.array([u["height"] for u in up])
        pen, _, _ = nonoverlap_penalty(cx, cy, w, h)
        return pen / block_area

    def peaks_on(up, scen):
        """Per-scenario peak dT; the RHS is keyed by name since scen_set yields
        arrays in unit order."""
        sv = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
        sv.build(); sv.factorize()
        return np.array([float(sv.silicon_layer(sv.solve(sv.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - ambient
            for pw in scen])

    d_fresh, d_shared, boot_sd = [], [], []
    ovl_max = 0.0
    rng_boot = np.random.default_rng(12345 + PAIR_OFFSET)

    for k in range(PAIR_OFFSET, PAIR_OFFSET + N_PAIRS):
        tr = scen_set(units, tot, np.random.default_rng(10_000 + k), N_ORACLE)
        js = 390_000 + 1000 * k                       # CRN across arms
        um = fit(tr, "mean", js)
        uc = fit(tr, "cvar", js)
        ovl_max = max(ovl_max, overlap_frac(um), overlap_frac(uc))

        fresh = scen_set(units, tot, np.random.default_rng(FRESH_TEST_BASE + k), N_TEST)
        pm_f, pc_f = peaks_on(um, fresh), peaks_on(uc, fresh)
        pm_s, pc_s = peaks_on(um, shared_test), peaks_on(uc, shared_test)

        df_k = cvar(pm_f, EVAL_ALPHA) - cvar(pc_f, EVAL_ALPHA)
        ds_k = cvar(pm_s, EVAL_ALPHA) - cvar(pc_s, EVAL_ALPHA)
        d_fresh.append(df_k)
        d_shared.append(ds_k)

        bs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng_boot.integers(0, N_TEST, N_TEST)
            bs[b] = cvar(pm_f[idx], EVAL_ALPHA) - cvar(pc_f[idx], EVAL_ALPHA)
        boot_sd.append(float(bs.std(ddof=1)))

        # Pool-readable per-pair lines (one estimate per design).
        emit(f"  pair {k}: D*_k = {df_k:+.4f}   D*_shared_k = {ds_k:+.4f}   "
             f"eval_sd_k = {boot_sd[-1]:.4f}")

    emit(f"\nresidual overlap on evaluated placements: max {100*ovl_max:.3f}% of block area "
         f"(diagnostic; exp039 evaluated raw placements and this run reproduces that)")

    if len(d_fresh) >= 2:
        df = np.array(d_fresh); ds = np.array(d_shared); bsd = np.array(boot_sd)
        mf, _, lof, hif = ci95_t(df)
        ms, _, los, his = ci95_t(ds)
        emit(f"\nthis shard only (pool across shards for the verdict):")
        emit(f"  fresh  D* = {mf:+.4f} [{lof:+.4f},{hif:+.4f}] p={paired_t_p(df):.4f} "
             f"sd={df.std(ddof=1):.4f}")
        emit(f"  shared D* = {ms:+.4f} [{los:+.4f},{his:+.4f}] p={paired_t_p(ds):.4f} "
             f"sd={ds.std(ddof=1):.4f}")
        emit(f"  mean within-pair evaluation sd = {np.sqrt((bsd**2).mean()):.4f}")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
