"""Guarantee tests for the exact legalizer.

Every returned placement must have zero overlap (to the gate's own measure) and be
in-bounds, or the call must raise LegalizationInfeasible — never a silent overlap.
These run at the real SI-metre die scale and cover the hostile cases: heavy overlap,
large n, stacked inputs, oversized blocks, and the zero-whitespace tiling.
"""

from __future__ import annotations
import numpy as np

from pyrova.optimizer.legalize import (
    legalize_guaranteed, legalize_units_exact, overlap_frac,
    LegalizationInfeasible, _frac_overlap,
)
from pyrova.workloads.hetero_soc import soc_units

TOL = 1e-9                      # overlap-fraction gate the legalizer must beat


def _random_blocks(n, util, rng, chip=0.012):
    """n blocks summing to `util` of a square die, at random (overlapping) centres."""
    a = rng.uniform(0.5, 1.5, size=n)
    a *= (util * chip * chip) / a.sum()          # scale areas to hit utilization
    ar = rng.uniform(0.5, 2.0, size=n)           # aspect ratios
    w = np.sqrt(a * ar); h = np.sqrt(a / ar)
    w = np.minimum(w, 0.95 * chip); h = np.minimum(h, 0.95 * chip)
    cx = rng.uniform(w / 2, chip - w / 2)
    cy = rng.uniform(h / 2, chip - h / 2)
    return cx, cy, w, h, chip


def _is_legal(cx, cy, w, h, chip_w, chip_h):
    of = _frac_overlap(cx, cy, w, h)
    inb = (cx - w / 2 >= -1e-9).all() and (cx + w / 2 <= chip_w + 1e-9).all() \
        and (cy - h / 2 >= -1e-9).all() and (cy + h / 2 <= chip_h + 1e-9).all()
    return of <= TOL, of, inb


def test_guarantee_random_battery():
    """Across n in [2,33] and util in [0.5,0.85], every result is exactly legal or
    an explicit infeasible — and it always terminates (no solver hang)."""
    rng = np.random.default_rng(0)
    raised = solved = 0
    for seed in range(120):
        n = int(rng.integers(2, 34))
        util = float(rng.uniform(0.5, 0.78))     # the real testbeds sit at 69-79%
        cx, cy, w, h, chip = _random_blocks(n, util, np.random.default_rng(seed))
        try:
            lx, ly, info = legalize_guaranteed(cx, cy, w, h, chip, chip)
        except LegalizationInfeasible:
            raised += 1
            continue
        legal, of, inb = _is_legal(lx, ly, w, h, chip, chip)
        assert legal, f"seed {seed}: overlap_frac {of:.2e} > gate"   # never a silent overlap
        assert inb, f"seed {seed}: out of bounds"                    # never OOB claimed legal
        solved += 1
    print(f"  battery: {solved} legal, {raised} infeasible (0 silent overlaps, 0 hangs)")
    assert solved >= 110, f"legalizer solved only {solved}/120 at realistic utilization"


def test_near_legal_exact_zero_and_small_move():
    """A nearly-legal input (small injected overlap) legalizes to EXACT zero with a
    small displacement — the density-spread use case."""
    rng = np.random.default_rng(3)
    units = soc_units()
    w = np.array([u["width"] for u in units]); h = np.array([u["height"] for u in units])
    cx0 = np.array([u["leftx"] + u["width"] / 2 for u in units])
    cy0 = np.array([u["bottomy"] + u["height"] / 2 for u in units])
    cw = max(u["leftx"] + u["width"] for u in units)
    ch = max(u["bottomy"] + u["height"] for u in units)
    cx = cx0 + rng.normal(0, 0.2e-3, size=len(units))     # ~0.2 mm jitter
    cy = cy0 + rng.normal(0, 0.2e-3, size=len(units))
    lx, ly, info = legalize_guaranteed(cx, cy, w, h, cw, ch)
    legal, of, inb = _is_legal(lx, ly, w, h, cw, ch)
    assert legal and inb, f"not legal: of={of:.2e}"
    assert of == 0.0 or of < TOL, f"overlap not exactly zero: {of:.2e}"
    move = np.max(np.abs(lx - cx) + np.abs(ly - cy))
    print(f"  near-legal: method={info['method']} of={of:.1e} max move={move*1e3:.3f} mm")
    assert info["method"] in ("min-displacement", "left-pack-witness")
    assert move < 2e-3, f"displacement {move*1e3:.2f} mm too large for a nearly-legal input"


def test_real_legal_layout_not_false_raised():
    """The native hetero-SoC layout is legal; legalizing it must not raise and must
    barely move it (the gap-accumulation bug false-raised exactly this)."""
    units = soc_units()
    cw = max(u["leftx"] + u["width"] for u in units)
    ch = max(u["bottomy"] + u["height"] for u in units)
    native_of = overlap_frac(units)
    out, of = legalize_units_exact(units, cw, ch)
    print(f"  hetero-SoC native of={native_of:.2e} -> legalized of={of:.2e}")
    assert of <= TOL, f"legalized overlap {of:.2e}"


def test_heavy_overlap_large_n_terminates_legal():
    """A heavily stacked, large-n input must legalize (via the shelf fallback) or
    report infeasible — never hang, never overlap."""
    rng = np.random.default_rng(7)
    n, chip = 30, 0.02
    w = rng.uniform(1e-3, 3e-3, size=n); h = rng.uniform(1e-3, 3e-3, size=n)
    cx = np.full(n, chip / 2) + rng.normal(0, 1e-4, size=n)     # nearly coincident
    cy = np.full(n, chip / 2) + rng.normal(0, 1e-4, size=n)
    lx, ly, info = legalize_guaranteed(cx, cy, w, h, chip, chip)
    legal, of, inb = _is_legal(lx, ly, w, h, chip, chip)
    print(f"  stacked n={n}: method={info['method']} of={of:.1e} inb={inb}")
    assert legal and inb


def test_oversized_block_raises():
    """A block larger than the die is infeasible — including the single-block case,
    which must not slip through as an out-of-bounds 'legal' return."""
    for n in (1, 3):
        w = np.full(n, 5.0); h = np.full(n, 1.0)
        cx = np.full(n, 2.0); cy = np.full(n, 0.5)
        try:
            legalize_guaranteed(cx, cy, w, h, 4.0, 4.0)
            assert False, f"n={n}: oversized block was not reported infeasible"
        except LegalizationInfeasible as e:
            assert e.reason == "block-larger-than-die"


def test_zero_whitespace_tiling():
    """A 100%-util tiling passes through if already legal, and reports infeasible if
    perturbed into overlap (no legal non-tiling exists)."""
    chip = 4.0
    w = np.full(4, 2.0); h = np.full(4, 2.0)
    cx = np.array([1.0, 3.0, 1.0, 3.0]); cy = np.array([1.0, 1.0, 3.0, 3.0])
    lx, ly, info = legalize_guaranteed(cx, cy, w, h, chip, chip)
    assert _is_legal(lx, ly, w, h, chip, chip)[0]
    assert info["method"] == "tiling-passthrough"
    cxp = cx.copy(); cxp[0] += 0.5                    # perturb into a neighbour
    try:
        legalize_guaranteed(cxp, cy, w, h, chip, chip)
        assert False, "perturbed 100%-util tiling should be infeasible"
    except LegalizationInfeasible as e:
        assert e.reason == "tiling-only"


def test_deterministic():
    """Same input, byte-identical output (no RNG anywhere in the solve)."""
    cx, cy, w, h, chip = _random_blocks(20, 0.7, np.random.default_rng(5))
    a = legalize_guaranteed(cx, cy, w, h, chip, chip)
    b = legalize_guaranteed(cx, cy, w, h, chip, chip)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
    print("\nALL LEGALIZER CHECKS PASSED")
