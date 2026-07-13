"""Matched-grid solver-vs-reference diagnostic: mean-opt and cvar-opt
placements (structured family, N_TRAIN=128, alpha=0.95, trained at 18^2, 2
seeds) are evaluated in BOTH our solver and the reference binary at the SAME
32x32 grid, so the only difference is the model, not the resolution. Reports
per-scenario peak-dT MAE and correlation r, plus dCVaR per solver with a
sign-agreement check; 80 test scenarios per (seed, arm). Reference die
inference is pinned with corner markers via the imported
exp018_hotspot_crosscheck.hotspot_peak.
"""

from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import GridFDSolver, parse_flp, parse_config
from pyrova.optimizer.placer import DiffPlacer
from pyrova.evaluation.metrics import cvar
from pyrova.workloads.structured import StructuredWorkloadModel
from pyrova.experiments.exp018_hotspot_crosscheck import (hotspot_peak, chip_box,
                                                          HOTSPOT)

CONFIG = PKG / "inputs/configs/thermal.config"
FLP = PKG / "inputs/floorplans/ev6.flp"
MATCH_GRID = 32          # BOTH solvers evaluated here
TRAIN_GRID = 18          # placements trained here
ALPHA = 0.95
N_TEST = 80              # reference solves per (seed, arm) -- keep runtime sane
N_SEEDS = 2


def our_peaks_at(cfg, units, scen, cw, ch, grid, ambient):
    s = GridFDSolver(cfg, units, cw, ch, grid, grid)
    s.build(); s.factorize()
    out = np.zeros(len(scen))
    for i, pw in enumerate(scen):
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(units)}
        out[i] = float(s.silicon_layer(s.solve(s.build_rhs(bp))).max()) - ambient
    return out


def main():
    if not HOTSPOT.exists():
        raise SystemExit(f"reference binary not found at {HOTSPOT}")
    cfg = parse_config(str(CONFIG))
    ambient = cfg["ambient"]
    units0 = parse_flp(str(FLP))
    cw, ch = chip_box(units0)
    train_solver = GridFDSolver(cfg, units0, cw, ch, TRAIN_GRID, TRAIN_GRID)
    train_solver.build(); train_solver.factorize()

    out = PKG / "results/exp018b_matched_grid.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"exp018b: matched-grid solver-vs-reference diagnostic. Placements "
         f"trained@{TRAIN_GRID}^2 (structured N=128, alpha={ALPHA}), BOTH solvers "
         f"evaluated@{MATCH_GRID}^2 on {N_TEST} scenarios, {N_SEEDS} seeds.")

    all_our, all_ref = [], []
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        for seed in range(N_SEEDS):
            model = StructuredWorkloadModel(units0, seed=100_000 * seed + 100 * 128 + 95)
            train = model.sample(128)
            test = model.sample(1500)[:N_TEST]
            our_c, ref_c = {}, {}
            for mode in ("mean", "cvar"):
                pl = DiffPlacer(train_solver, units0, cw, ch, TRAIN_GRID, TRAIN_GRID,
                                alpha=ALPHA)
                pl.optimize(train, mode=mode, n_iter=30, lr=2e-2, verbose=False)
                units_p = pl.get_units()
                our = our_peaks_at(cfg, units_p, test, cw, ch, MATCH_GRID, ambient)
                ref = np.array([hotspot_peak(units_p, pw, wd, f"m{seed}_{mode}_{i}",
                                             ambient, chip_w=cw, chip_h=ch)
                                for i, pw in enumerate(test)])
                our_c[mode], ref_c[mode] = our, ref
                all_our.append(our); all_ref.append(ref)
                mae = float(np.mean(np.abs(our - ref)))
                emit(f"  seed {seed} {mode:4s}: matched@{MATCH_GRID} peak MAE="
                     f"{mae*1e3:.1f} mK  our_meanpeak={our.mean():.3f}  ref={ref.mean():.3f} K")
            dour = cvar(our_c["mean"], ALPHA) - cvar(our_c["cvar"], ALPHA)
            dref = cvar(ref_c["mean"], ALPHA) - cvar(ref_c["cvar"], ALPHA)
            agree = np.sign(dour) == np.sign(dref)
            emit(f"  seed {seed}: matched-{MATCH_GRID} dCVaR our={dour:+.3f} K  "
                 f"ref={dref:+.3f} K  sign {'AGREES' if agree else 'FLIPS'}")

    our = np.concatenate(all_our); ref = np.concatenate(all_ref)
    mae = float(np.mean(np.abs(our - ref)))
    r = float(np.corrcoef(our, ref)[0, 1])
    emit(f"\nMATCHED-GRID AGREEMENT ({MATCH_GRID}^2, {len(our)} points): "
         f"MAE={mae*1e3:.1f} mK, r={r:.6f}")
    emit("Interpretation: this is the honest solver-vs-reference number at matched "
         "resolution (contrast exp018's 1.5-3.0 K, which was our@18 vs ref@32 and "
         "confounded by discretisation). Use THIS number for solver-fidelity claims.")
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
