"""Reference-model transfer check: final placements from both arms of two
paired comparisons are re-evaluated in the reference solver (HotSpot grid
model, external C binary, 32x32 — an independent discretisation from the
18x18 training grid), reporting the paired dCVaR sign per solver and the
per-scenario peak correlation r. Arm A: structured hand-built family,
N_TRAIN=128, alpha=0.95, mean-opt vs cvar-opt, 2 seeds, reference evaluation
on a 200-scenario subsample of the holdout. Arm B: BOOM 60/20 split (seed
base 40_000), mean vs blend gamma=0.75 at 120 iterations, reference
evaluation on all 20 held-out programs.
"""

from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import (GridFDSolver, parse_flp, parse_config,
                                      read_reference_grid_steady)
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import cvar
from pyrova.workloads.structured import StructuredWorkloadModel
from pyrova.workloads.boom_traces import BoomWorkload, resolve_paths

FLP = PKG / "inputs/floorplans/ev6.flp"
CONFIG = PKG / "inputs/configs/thermal.config"
HOTSPOT = ROOT / "Tools/HotSpot/hotspot"
NR = NC = 18             # our solver's grid (as in every experiment)
REF_NR = REF_NC = 32     # reference binary requires powers of two; a DIFFERENT
                         # resolution makes this an independent-discretisation
                         # check, not a re-run of our own stencil
N_HS = 200               # reference-solver scenario budget per arm-A placement


def chip_box(units):
    w = max(u["leftx"] + u["width"] for u in units) - min(u["leftx"] for u in units)
    h = max(u["bottomy"] + u["height"] for u in units) - min(u["bottomy"] for u in units)
    return w, h


def write_flp(units, path: Path) -> None:
    with open(path, "w") as f:
        for u in units:
            f.write(f"{u['name']}\t{u['width']:.9e}\t{u['height']:.9e}\t"
                    f"{u['leftx']:.9e}\t{u['bottomy']:.9e}\n")


def hotspot_peak(units, pw: np.ndarray, workdir: Path, tag: str,
                 ambient: float, chip_w: float | None = None,
                 chip_h: float | None = None) -> float:
    """Peak silicon dT from the reference binary for one placement + power map.

    The reference infers the die size from the floorplan bounding box; an
    optimised placement whose bbox is smaller than the nominal die would be
    simulated on a SMALLER chip (measured: ~8 K error on the BOOM die). When
    chip_w/chip_h are given, two zero-power corner markers pin the inferred
    die to the nominal dimensions. Always pass them for optimised placements.
    """
    flp = workdir / f"{tag}.flp"
    ptr = workdir / f"{tag}.ptrace"
    grid = workdir / f"{tag}.grid"
    steady = workdir / f"{tag}.steady"
    units_out = list(units)
    pw_out = np.asarray(pw, dtype=float)
    if chip_w is not None and chip_h is not None:
        eps = 1e-6
        units_out = units_out + [
            dict(name="_CNR0", width=eps, height=eps, leftx=0.0, bottomy=0.0),
            dict(name="_CNR1", width=eps, height=eps,
                 leftx=chip_w - eps, bottomy=chip_h - eps)]
        pw_out = np.concatenate([pw_out, [0.0, 0.0]])
    write_flp(units_out, flp)
    names = [u["name"] for u in units_out]
    with open(ptr, "w") as f:
        f.write("\t".join(names) + "\n")
        f.write("\t".join(f"{p:.9e}" for p in pw_out) + "\n")
    cmd = [str(HOTSPOT), "-c", str(CONFIG), "-f", str(flp), "-p", str(ptr),
           "-model_type", "grid", "-grid_rows", str(REF_NR), "-grid_cols", str(REF_NC),
           "-grid_steady_file", str(grid), "-steady_file", str(steady)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"reference solver failed:\n{r.stderr[:500]}")
    T = read_reference_grid_steady(str(grid), 4, REF_NR, REF_NC)
    return float(T[0].max()) - ambient          # layer 0 = silicon


def our_peaks(solver, units_orig, pl, scen) -> np.ndarray:
    cx, cy = pl.get_positions()
    return pl._scenario_peaks(cx, cy, scen)


def compare(units_placed_by_arm, scen, workdir, tag, ambient, ours_by_arm,
            alpha, emit, chip_w=None, chip_h=None):
    """Reference vs our peaks for the same placements/scenarios; sign check.
    chip_w/chip_h pin the reference's die inference (see hotspot_peak)."""
    ref = {}
    for arm, units in units_placed_by_arm.items():
        vals = []
        for s, pw in enumerate(scen):
            vals.append(hotspot_peak(units, pw, workdir, f"{tag}_{arm}_{s}", ambient,
                                     chip_w=chip_w, chip_h=chip_h))
            if (s + 1) % 50 == 0:
                print(f"  [{tag}/{arm}] {s + 1}/{len(scen)} reference solves", flush=True)
        ref[arm] = np.array(vals)
    arms = list(units_placed_by_arm)
    a0, a1 = arms
    d_ref = cvar(ref[a0], alpha) - cvar(ref[a1], alpha)
    d_our = cvar(ours_by_arm[a0], alpha) - cvar(ours_by_arm[a1], alpha)
    r = np.corrcoef(np.concatenate([ref[a0], ref[a1]]),
                    np.concatenate([ours_by_arm[a0], ours_by_arm[a1]]))[0, 1]
    mae = float(np.mean(np.abs(np.concatenate([ref[a0], ref[a1]])
                               - np.concatenate([ours_by_arm[a0], ours_by_arm[a1]]))))
    emit(f"  [{tag}] dCVaR ours={d_our:+.3f} K  reference={d_ref:+.3f} K  "
         f"sign {'AGREES' if np.sign(d_ref) == np.sign(d_our) else 'FLIPS'}; "
         f"per-scenario peak agreement r={r:.6f}, MAE={mae * 1e3:.1f} mK")
    return np.sign(d_ref) == np.sign(d_our)


def main():
    if not HOTSPOT.exists():
        raise SystemExit(f"reference binary not found at {HOTSPOT}")
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    out = PKG / "results/exp018_hotspot_crosscheck.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"exp018: reference-model transfer check. Placements trained on our "
         f"{NR}x{NC} solver; evaluated in the reference binary at its own "
         f"{REF_NR}x{REF_NC} grid (independent discretisation). "
         f"Arm A: structured N=128 a=0.95 mean vs cvar, 2 seeds, {N_HS} reference "
         f"scenarios. Arm B: BOOM split 40000, mean vs blend g=0.75, 120 it, 20 programs.")

    ok = []
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)

        # Arm A
        units = parse_flp(str(FLP))
        chip_w, chip_h = chip_box(units)
        solver = GridFDSolver(cfg, units, chip_w, chip_h, NR, NC)
        solver.build(); solver.factorize()
        for seed in range(2):
            model = StructuredWorkloadModel(
                units, seed=100_000 * seed + 100 * 128 + 95)
            train = model.sample(128)
            test = model.sample(1500)[:N_HS]
            placed, ours = {}, {}
            for mode in ("mean", "cvar"):
                pl = DiffPlacer(solver, units, chip_w, chip_h, NR, NC, alpha=0.95)
                pl.optimize(train, mode=mode, n_iter=30, lr=2e-2, verbose=False)
                placed[mode] = pl.get_units()
                ours[mode] = our_peaks(solver, units, pl, test)
            ok.append(compare(placed, test, wd, f"A{seed}", ambient, ours, 0.95, emit,
                              chip_w=chip_w, chip_h=chip_h))

        # Arm B
        csvp, rptp = resolve_paths()
        if csvp:
            wl = BoomWorkload(csvp, rptp, config_id="0")
            bs = GridFDSolver(cfg, wl.units, wl.chip_w, wl.chip_h, NR, NC)
            bs.build(); bs.factorize()

            def peaks_fn(scen):
                p = DiffPlacer(bs, wl.units, wl.chip_w, wl.chip_h, NR, NC, alpha=0.9)
                cx, cy = p.get_positions()
                return p._scenario_peaks(cx, cy, scen)
            wl.scale_to_peak(peaks_fn, 40.0)
            scen = wl.scenarios()
            perm = np.random.default_rng(40_000).permutation(len(scen))
            tr = [scen[i] for i in perm[:60]]
            te = [scen[i] for i in perm[60:]]
            placed, ours = {}, {}
            for g, mode in ((0.0, "mean"), (0.75, "blend")):
                pl = DiffPlacer(bs, wl.units, wl.chip_w, wl.chip_h, NR, NC,
                                alpha=0.9, blend_gamma=g)
                pl.optimize(tr, mode=mode, n_iter=120, lr=2e-2, verbose=False)
                placed[f"g{g:g}"] = pl.get_units()
                ours[f"g{g:g}"] = our_peaks(bs, wl.units, pl, te)
            ok.append(compare(placed, te, wd, "B", ambient, ours, 0.9, emit,
                              chip_w=wl.chip_w, chip_h=wl.chip_h))
        else:
            emit("  Arm B SKIPPED: BOOM_DATA not found.")

    emit(f"\nPRE-REGISTERED VERDICT: "
         + ("TRANSFERS — every paired dCVaR sign agrees between our solver and "
            "the reference model; claims 6 and 11 are not artifacts of our "
            "discretisation." if all(ok) else
            "SIGN FLIP in at least one arm — flag the corresponding claim as "
            "model-contingent before publication."))
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
