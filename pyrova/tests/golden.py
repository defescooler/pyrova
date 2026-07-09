"""Golden-output regression harness for the thermal solver and placer.

Pins the field itself, which a smooth assembly bug could shift while still
passing the derivative (FD) checks in ``test_gradients.py``.

    python -m pyrova.tests.golden --write     # regenerate the reference (rare)
    python -m pyrova.tests.golden             # check current code against it

WARNING: snapshots are per-platform (BLAS-dependent); cross-platform
bit-equality will fail. Regenerate on a new machine and rely on the gradient
checks for cross-platform truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pyrova.core.design import Design
from pyrova.thermal.fd_solver import GridFDSolver, random_power_map
from pyrova.optimizer.placer import DiffPlacer

FLP = "pyrova/inputs/floorplans/ev6.flp"
REF = Path(__file__).with_name("_golden_ref.npz")

# Solver is an exact LU solve, so a faithful rewrite matches to round-off; 1e-6 K
# slack catches real regressions without flagging noise.
FIELD_ATOL = 1e-6
OBJ_RTOL = 1e-9


def _solver(nr: int, nc: int):
    d = Design.from_flp(FLP)
    units = d.macro_flp_dicts()
    s = GridFDSolver(d.thermal_config.as_dict(), units, d.chip_width, d.chip_height, nr, nc)
    s.build()
    s.factorize()
    return d, units, s


def _scenarios(units, n, seed):
    rng = np.random.default_rng(seed)
    return [np.array([random_power_map(units, 50.0, rng)[u["name"]] for u in units])
            for _ in range(n)]


def compute() -> dict:
    """Deterministic snapshot of the protected numerics on fixed inputs."""
    out: dict[str, np.ndarray] = {}

    for nr in (18, 24):
        d, units, s = _solver(nr, nr)
        rng = np.random.default_rng(0)
        powers = random_power_map(units, 50.0, rng)
        T = s.solve_from_powers(powers)
        peak, grad = s.peak_T_gradient(powers)
        out[f"field_{nr}"] = T
        out[f"peak_{nr}"] = np.array([peak])
        out[f"grad_{nr}"] = np.array([grad[u["name"]] for u in units])

    d, units, s = _solver(16, 16)
    scen = _scenarios(units, 6, seed=3)
    for mode in ("mean", "cvar", "dro"):
        pl = DiffPlacer(s, units, d.chip_width, d.chip_height, 16, 16,
                        alpha=0.9, eps_dro=0.5, nonoverlap_w=0.0)
        obj, g_rx, g_ry = pl.objective_and_grad(scen, mode=mode)
        out[f"obj_{mode}"] = np.array([obj])
        out[f"grx_{mode}"] = g_rx
        out[f"gry_{mode}"] = g_ry
    return out


def write() -> None:
    np.savez(REF, **compute())
    print(f"wrote golden reference -> {REF}")


def check() -> bool:
    if not REF.exists():
        raise SystemExit(f"no golden reference at {REF}; run with --write first")
    ref = np.load(REF)
    cur = compute()
    ok = True
    # A key added to compute() without --write, or dropped from it, is itself a
    # failure rather than silently skipped.
    missing = sorted(set(cur) ^ set(ref.files))
    if missing:
        print(f"  key mismatch between snapshot and compute(): {missing}")
        ok = False
    for key in sorted(set(cur) & set(ref.files)):
        a, b = cur[key], ref[key]
        atol = FIELD_ATOL if key.startswith(("field", "peak", "grad")) else 0.0
        rtol = 0.0 if atol else OBJ_RTOL
        max_abs = float(np.abs(a - b).max())
        passed = np.allclose(a, b, atol=atol, rtol=rtol)
        ok &= passed
        print(f"  {key:12s} max|Δ|={max_abs:.3e}  {'ok' if passed else 'MISMATCH'}")
    print("GOLDEN OK" if ok else "GOLDEN MISMATCH")
    return ok


def test_golden() -> None:
    """pytest entry point: the current numerics reproduce the reference snapshot."""
    assert check(), "solver/placer output drifted from the golden reference"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="regenerate the reference")
    args = ap.parse_args()
    if args.write:
        write()
    elif not check():
        raise SystemExit(1)
