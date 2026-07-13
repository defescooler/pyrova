"""Floorplan2 arm of the i.i.d. oracle-gap budget ladder: runs the imported
exp028_budget_ladder.run (estimand, budgets, pair count, and oracle sizes
come from that module, so the arms cannot drift) on the floorplan2 bench
only.
"""

from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from pyrova.thermal.fd_solver import parse_config
from pyrova.experiments.exp028_budget_ladder import run, BUDGETS, N_OR_PAIRS, N_ORACLE

FLP2 = ROOT / "Tools/HotSpot/examples/example3/floorplan2.flp"


def main():
    cfg = parse_config(str(PKG / "inputs/configs/thermal.config"))
    out = PKG / "results/exp028b_floorplan2.txt"
    fh = open(out, "w")

    def emit(s=""):
        print(s, flush=True); fh.write(s + "\n"); fh.flush()

    emit(f"budget ladder, floorplan2 only. budgets={BUDGETS}, "
         f"N_OR_PAIRS={N_OR_PAIRS}, N_ORACLE={N_ORACLE}.")
    if not FLP2.exists():
        emit(f"floorplan2 not found at {FLP2}"); fh.close(); return
    run(FLP2, cfg, emit)
    fh.close()
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
