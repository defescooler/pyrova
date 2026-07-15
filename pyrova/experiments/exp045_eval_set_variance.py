"""Fresh-vs-shared evaluation-set contrast on the oracle gap: each pair's
placements are scored on their own evaluation scenarios and on one shared
reference set, plus a within-pair bootstrap of the evaluation scenarios, so
the two interval designs are compared on identical fits.

Per pair: symmetric single-start arms, CRN, train@18+jitter with density
spreading, guaranteed-legal eval@64^2, power calibrated to a 40 K operating
point. PYROVA_TESTBED selects hetero_soc or kraken; PYROVA_SMOKE=1 for an
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

from pyrova.thermal.fd_solver import GridFDSolver, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.optimizer.legalize import legalize_units_exact, LegalizationInfeasible
from pyrova.evaluation.metrics import cvar, ci95_t, paired_t_p
from pyrova.workloads.hetero_soc import soc_units, HeteroSoCWorkloadModel, _BLOCKS, _MODES
from pyrova.workloads.kraken_soc import kraken_units, KrakenWorkloadModel

SMOKE = os.environ.get("PYROVA_SMOKE") == "1"
TESTBED = os.environ.get("PYROVA_TESTBED", "hetero_soc")   # hetero_soc | kraken

CONFIG = PKG / "inputs/configs/thermal.config"
EVAL_ALPHA = 0.90
TARGET_PEAK = 40.0
TRAIN_GRID = 18
JITTER = 1.0
DENSITY_W = 5.0e2
DENSITY_LAM0 = 3.0e1
DENSITY_GRID = (48, 48)

EVAL_GRID = 32 if SMOKE else 64
N_ITER = 12 if SMOKE else 100
N_ORACLE = 100 if SMOKE else 2000
N_PAIRS = 3 if SMOKE else int(os.environ.get("PYROVA_PAIRS", "20"))
N_TEST = 200 if SMOKE else 2000
N_BOOT = 50 if SMOKE else 400

SHARED_TEST_SEED = 777          # shared reference set
FRESH_TEST_BASE = 900_000       # per-pair evaluation sets


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


def load_testbed():
    """(units, sampler(seed)->model with .sample(n), pure-mode power vectors) for
    the selected testbed; Kraken is rescaled to the common 40 K operating point."""
    if TESTBED == "kraken":
        units = kraken_units()
        base = KrakenWorkloadModel(units, seed=1)
        return units, base, None
    units = soc_units()
    return units, HeteroSoCWorkloadModel(units, seed=1), np.array([b[3] for b in _BLOCKS])


def main():
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    units, base_model, pmax = load_testbed()
    cw = max(u["leftx"] + u["width"] for u in units)
    ch = max(u["bottomy"] + u["height"] for u in units)

    out = PKG / f"results/exp045_eval_set_variance_{TESTBED}.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"evaluation-set conditioning [{TESTBED}]: {N_PAIRS} pairs, N_ORACLE={N_ORACLE}, "
         f"{N_ITER} it, N_TEST={N_TEST} (fresh per pair vs one shared set), "
         f"{N_BOOT} bootstrap resamples, train@{TRAIN_GRID}+jitter -> "
         f"guaranteed-legal eval@{EVAL_GRID} (alpha={EVAL_ALPHA}). "
         + ("[SMOKE - not a result]" if SMOKE else "[full run]"))

    se = GridFDSolver(cfg, units, cw, ch, EVAL_GRID, EVAL_GRID)
    se.build(); se.factorize()

    def peak_of(pw):
        return float(se.silicon_layer(se.solve(se.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(units)}))).max()) - ambient

    # Common operating point (dT linear in power, so one measurement fixes it).
    m0 = float(np.mean([peak_of(pw) for pw in base_model.sample(300 if not SMOKE else 50)]))
    calib = TARGET_PEAK / m0
    emit(f"calibration [{TESTBED}]: tiled mean peak {m0:.1f}K -> power x{calib:.3f} "
         f"for a {TARGET_PEAK:.0f}K operating point")

    def model(seed):
        """Workload sampler for this testbed at the common operating point."""
        if TESTBED == "kraken":
            m = KrakenWorkloadModel(units, seed=seed)
            m.scale = m.scale * calib
            return m
        m = HeteroSoCWorkloadModel(units, seed=seed)
        raw = m.sample
        m.sample = lambda n: [pw * calib for pw in raw(n)]
        return m

    if TESTBED == "kraken":
        modes = [mv * calib for mv in base_model.modes]
    else:
        modes = [np.array(mv) * pmax * calib for mv in _MODES.values()]
    hot = {block_at_peak(units, cw, ch, EVAL_GRID, EVAL_GRID,
                         se.silicon_layer(se.solve(se.build_rhs(
                             {u["name"]: float(v) for u, v in zip(units, mv)}))))
           for mv in modes}
    g0 = len(hot) >= 3
    emit(f"G0 (mechanism): hotspot blocks over modes: {len(hot)} {sorted(hot)} -> "
         f"{'PASS' if g0 else 'FAIL'}")
    if not g0:
        emit("\n===== VERDICT =====\nNO VERDICT — mechanism gate failed."); fh.close(); return

    shared_test = model(SHARED_TEST_SEED).sample(N_TEST)

    st = GridFDSolver(cfg, units, cw, ch, TRAIN_GRID, TRAIN_GRID)
    st.build(); st.factorize()

    def fit(tr, mode, jseed):
        pl = DiffPlacer(st, units, cw, ch, TRAIN_GRID, TRAIN_GRID, alpha=EVAL_ALPHA,
                        nonoverlap_w=1e4, density_w=DENSITY_W,
                        density_lam0=DENSITY_LAM0, density_grid=DENSITY_GRID)
        pl.optimize(tr, mode=mode, n_iter=N_ITER, lr=2e-2, verbose=False,
                    raster_jitter=JITTER, jitter_seed=jseed)
        return legalize_units_exact(pl.get_units(), cw, ch)

    def peaks_on(up, scen):
        """Per-scenario peak dT of a placement over a scenario list."""
        sv = GridFDSolver(cfg, up, cw, ch, EVAL_GRID, EVAL_GRID)
        sv.build(); sv.factorize()
        return np.array([float(sv.silicon_layer(sv.solve(sv.build_rhs(
            {u["name"]: float(pw[b]) for b, u in enumerate(up)}))).max()) - ambient
            for pw in scen])

    d_fresh, d_shared, boot_sd = [], [], []
    ovl_max = 0.0
    infeasible = 0
    rng_boot = np.random.default_rng(12345)

    for k in range(N_PAIRS):
        tr = model(10_000 + k).sample(N_ORACLE)
        js = 500_000 + 1000 * k                       # CRN across arms
        try:
            um, fm = fit(tr, "mean", js)
            uc, fc = fit(tr, "cvar", js)
        except LegalizationInfeasible:
            infeasible += 1
            emit(f"  pair {k+1}: INFEASIBLE, skipped")
            continue
        ovl_max = max(ovl_max, fm, fc)

        fresh = model(FRESH_TEST_BASE + k).sample(N_TEST)
        pm_f, pc_f = peaks_on(um, fresh), peaks_on(uc, fresh)
        pm_s, pc_s = peaks_on(um, shared_test), peaks_on(uc, shared_test)

        df = cvar(pm_f, EVAL_ALPHA) - cvar(pc_f, EVAL_ALPHA)
        ds = cvar(pm_s, EVAL_ALPHA) - cvar(pc_s, EVAL_ALPHA)
        d_fresh.append(df)
        d_shared.append(ds)

        # Within-pair evaluation variance: resample the evaluation scenarios,
        # placements held fixed. No new solves - the peaks are already computed.
        bs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng_boot.integers(0, N_TEST, N_TEST)
            bs[b] = cvar(pm_f[idx], EVAL_ALPHA) - cvar(pc_f[idx], EVAL_ALPHA)
        boot_sd.append(float(bs.std(ddof=1)))

        print(f"  pair {k+1}/{N_PAIRS}: D*_fresh={df:+.4f} D*_shared={ds:+.4f} "
              f"eval-sd={boot_sd[-1]:.4f}", flush=True)

    g2 = ovl_max < 1e-3 and infeasible == 0
    emit(f"\nG2 (legality): max overlap {100*ovl_max:.3f}%, {infeasible} infeasible -> "
         f"{'PASS' if g2 else 'FAIL'}")
    if not g2 or len(d_fresh) < 3:
        emit("\n===== VERDICT =====\nNO VERDICT — legality gate failed or too few pairs.")
        fh.close(); return
    if SMOKE:
        # A smoke exercises the plumbing at sample sizes that cannot support an
        # interval; rendering one invites it to be read as a measurement.
        emit("\n===== VERDICT =====\nNO VERDICT — smoke run (plumbing check only).")
        fh.close(); return

    df = np.array(d_fresh); ds = np.array(d_shared); bsd = np.array(boot_sd)

    mf, _, lof, hif = ci95_t(df)
    ms, _, los, his = ci95_t(ds)
    emit(f"\ncorrected design (fresh evaluation set per pair):")
    emit(f"  D* = {mf:+.4f} K CI95[{lof:+.4f},{hif:+.4f}]  p={paired_t_p(df):.4f}  "
         f"n={len(df)}  sd={df.std(ddof=1):.4f}")
    emit(f"historical design (one shared evaluation set):")
    emit(f"  D* = {ms:+.4f} K CI95[{los:+.4f},{his:+.4f}]  p={paired_t_p(ds):.4f}  "
         f"n={len(ds)}  sd={ds.std(ddof=1):.4f}")

    # Magnitudes, not a subtraction: v_total - v_eval is an unbiased but
    # high-variance estimate of the training component and can go negative on a
    # small sample, so we report the two measured spreads and their ratio and
    # let the comparison of the two intervals carry the verdict.
    sd_total = float(df.std(ddof=1))            # fresh design: train + eval variance
    sd_eval = float(np.sqrt((bsd ** 2).mean()))  # within-pair evaluation-set sd
    sd_shared = float(ds.std(ddof=1))            # shared design: training variance only
    w_fresh = hif - mf
    w_shared = his - ms
    ratio = w_fresh / max(w_shared, 1e-12)

    emit(f"\nvariance components [K]")
    emit(f"  sd across pairs, fresh eval set   = {sd_total:.4f}   (training + evaluation)")
    emit(f"  sd across pairs, shared eval set  = {sd_shared:.4f}   (training only)")
    emit(f"  sd within pair, bootstrap of eval = {sd_eval:.4f}   (evaluation only)")
    if sd_total > sd_eval:
        share = 100.0 * (sd_eval ** 2) / (sd_total ** 2)
        emit(f"  evaluation share of total variance = {share:.1f}%")
    else:
        share = float("nan")
        emit(f"  evaluation share not resolvable at n={len(df)} "
             f"(within-pair sd exceeds across-pair sd: sample too small)")
    emit(f"  CI half-width: fresh {w_fresh:.4f} vs shared {w_shared:.4f}  "
         f"(fresh/shared = {ratio:.2f}x)")

    emit("\n===== VERDICT =====")
    if lof > 0:
        emit(f"HOLDS UNDER THE CORRECTED DESIGN: D* = {mf:+.4f} K "
             f"[{lof:+.4f},{hif:+.4f}], lower bound still above zero.")
    else:
        emit(f"DOES NOT HOLD UNDER THE CORRECTED DESIGN: D* = {mf:+.4f} K "
             f"[{lof:+.4f},{hif:+.4f}] spans zero; the historical claim rested on the "
             f"conditional interval and must be requalified.")
    if ratio > 1.15:
        emit(f"The shared-evaluation-set design UNDERSTATED the interval by {ratio:.2f}x: "
             f"evaluation-set uncertainty is a material component and must be replicated.")
    elif ratio < 0.87:
        emit(f"The fresh-evaluation-set interval is {1/ratio:.2f}x NARROWER, i.e. the shared "
             f"set happened to be an unlucky draw; the correction changes the estimate, not "
             f"only its width. Both designs are reported.")
    else:
        emit(f"The two designs agree to within {abs(1-ratio)*100:.0f}% in interval width: "
             f"evaluation-set conditioning was a real but immaterial omission. The corrected "
             f"design is nonetheless the one to use, since it costs nothing.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
