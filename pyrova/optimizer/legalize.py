"""Legalization passes that remove macro overlap after placement."""

from __future__ import annotations
import numpy as np
from scipy.optimize import linprog

from pyrova.objectives.overlap import nonoverlap_penalty

# Feasibility slack in normalised (die-relative) units, and the in-bounds slack
# as a die-relative fraction. Both absorb solver round-off without admitting
# overlap the 0.1%-of-area legality threshold would flag.
_FEAS_TOL = 1e-9
_BOUND_TOL = 1e-7
_OVERLAP_TOL = 1e-9        # dimensionless (overlap area / block area)


class LegalizationInfeasible(Exception):
    """Raised when no legal arrangement is produced, instead of returning overlap."""

    def __init__(self, reason: str, diagnostics: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics or {}


def legalize_positions(cx: np.ndarray, cy: np.ndarray,
                       widths: np.ndarray, heights: np.ndarray,
                       chip_w: float, chip_h: float,
                       tol_frac: float = 1e-3, max_iter: int = 4000,
                       ) -> tuple[np.ndarray, np.ndarray, float]:
    """Push block centres apart until overlap < tol_frac of total block area; returns (cx, cy, achieved_overlap_frac)."""
    cx, cy = cx.copy(), cy.copy()
    tot = float(np.sum(widths * heights))
    step0 = 0.02 * float(min(widths.min(), heights.min()))
    lo_x, hi_x = widths / 2.0, chip_w - widths / 2.0
    lo_y, hi_y = heights / 2.0, chip_h - heights / 2.0
    pen = np.inf
    for it in range(max_iter):
        pen, gx, gy = nonoverlap_penalty(cx, cy, widths, heights)
        if pen / tot < tol_frac:
            break
        norm = max(np.abs(gx).max(), np.abs(gy).max(), 1e-30)
        # anneal the step slightly so late iterations settle
        step = step0 * (1.0 if it < max_iter // 2 else 0.5)
        cx = np.clip(cx - step * gx / norm, lo_x, hi_x)
        cy = np.clip(cy - step * gy / norm, lo_y, hi_y)
    return cx, cy, float(pen / tot)


def legalize_units(units_placed: list[dict], chip_w: float, chip_h: float,
                   **kw) -> tuple[list[dict], float]:
    """`legalize_positions` on solver unit dicts; returns (new units, overlap_frac)."""
    w = np.array([u["width"] for u in units_placed])
    h = np.array([u["height"] for u in units_placed])
    cx = np.array([u["leftx"] + u["width"] / 2 for u in units_placed])
    cy = np.array([u["bottomy"] + u["height"] / 2 for u in units_placed])
    cx, cy, frac = legalize_positions(cx, cy, w, h, chip_w, chip_h, **kw)
    out = [{**u, "leftx": float(cx[i] - w[i] / 2), "bottomy": float(cy[i] - h[i] / 2)}
           for i, u in enumerate(units_placed)]
    return out, frac


def overlap_frac(units_placed: list[dict]) -> float:
    """Residual pairwise overlap as a fraction of total block area."""
    w = np.array([u["width"] for u in units_placed])
    h = np.array([u["height"] for u in units_placed])
    cx = np.array([u["leftx"] + u["width"] / 2 for u in units_placed])
    cy = np.array([u["bottomy"] + u["height"] / 2 for u in units_placed])
    pen, _, _ = nonoverlap_penalty(cx, cy, w, h)
    return float(pen / np.sum(w * h))


# Guaranteed legalization.
#
# Two rectangles are disjoint once their centres are separated by half the summed
# extent in either axis. Fixing one such axis per pair makes non-overlap a set of
# linear constraints, so a placement meeting them has exactly zero overlap.
# Separation is at equality (touching is legal) with no added margin — a margin
# would accumulate along a touching row and overflow the die.


def _frac_overlap(cx, cy, w, h) -> float:
    pen, _, _ = nonoverlap_penalty(np.asarray(cx), np.asarray(cy),
                                   np.asarray(w), np.asarray(h))
    return float(pen / np.sum(w * h))


def _assign_axes(cx0, cy0, w, h):
    """Freeze one separation axis per pair as acyclic forward edges; returns (edges_x, edges_y), each of (a, b, min_separation)."""
    n = len(cx0)
    ex, ey = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ox = (w[i] + w[j]) / 2.0 - abs(cx0[i] - cx0[j])
            oy = (h[i] + h[j]) / 2.0 - abs(cy0[i] - cy0[j])
            if ox <= 0.0 and oy <= 0.0:
                use_x = (-ox) >= (-oy)           # keep the roomier axis
            elif ox <= 0.0:
                use_x = True
            elif oy <= 0.0:
                use_x = False
            else:
                use_x = ox <= oy                 # push apart along the cheaper axis
            if use_x:
                a, b = (i, j) if (cx0[i], i) < (cx0[j], j) else (j, i)
                ex.append((a, b, (w[a] + w[b]) / 2.0))
            else:
                a, b = (i, j) if (cy0[i], i) < (cy0[j], j) else (j, i)
                ey.append((a, b, (h[a] + h[b]) / 2.0))
    return ex, ey


def _left_pack(edges, coord0, lo, hi):
    """Longest-path left-pack placing each block at its minimum feasible coordinate; returns (coords, feasible)."""
    n = len(lo)
    order = sorted(range(n), key=lambda i: (coord0[i], i))
    preds = [[] for _ in range(n)]
    for a, b, d in edges:
        preds[b].append((a, d))
    v = np.array(lo, dtype=float)
    for i in order:
        for a, d in preds[i]:
            if v[a] + d > v[i]:
                v[i] = v[a] + d
    return v, bool(np.all(v <= hi + _FEAS_TOL))


def _min_displacement_1d(coord0, edges, lo, hi):
    """Min total L1 displacement from `coord0` subject to the separation edges and box bounds (exact LP); returns coords or None."""
    n = len(coord0)
    # variables z = [coords (n), slacks t (n)]; minimize sum t with t >= |x - x0|.
    c = np.concatenate([np.zeros(n), np.ones(n)])
    rows, b = [], []
    for i in range(n):
        r = np.zeros(2 * n); r[i] = 1.0; r[n + i] = -1.0; rows.append(r); b.append(coord0[i])
        r = np.zeros(2 * n); r[i] = -1.0; r[n + i] = -1.0; rows.append(r); b.append(-coord0[i])
    for a, bb, d in edges:
        r = np.zeros(2 * n); r[a] = 1.0; r[bb] = -1.0; rows.append(r); b.append(-d)
    bounds = [(lo[i], hi[i]) for i in range(n)] + [(0.0, None)] * n
    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(b), bounds=bounds, method="highs")
    return res.x[:n] if res.success else None


def _bfdh(order, w, h, chip_w, chip_h, tol):
    """Best-fit decreasing-height shelf packing for one block order; returns (cx, cy) or None if the stack exceeds the die."""
    n = len(w)
    cx = np.zeros(n); cy = np.zeros(n)
    shelves = []            # each: [y_bottom, height, x_used]
    top = 0.0
    for i in order:
        best, best_rem = -1, None
        for si, sh in enumerate(shelves):
            rem = chip_w - sh[2] - w[i]
            if rem >= -tol and (best_rem is None or rem < best_rem):
                best, best_rem = si, rem
        if best < 0:                                   # open a new shelf
            if top + h[i] > chip_h + tol:
                return None
            shelves.append([top, h[i], 0.0]); best = len(shelves) - 1
            top += h[i]
        sh = shelves[best]
        cx[i] = sh[2] + w[i] / 2.0
        cy[i] = sh[0] + h[i] / 2.0
        sh[2] += w[i]
    return cx, cy


def _shelf_pack(w, h, chip_w, chip_h):
    """Constructive legality backstop (repositions blocks, not minimal-move) when the frozen order is unsatisfiable; returns the first fitting arrangement or None."""
    tol = _BOUND_TOL * max(chip_w, chip_h)
    idx = np.arange(len(w))
    keys = (
        sorted(idx, key=lambda i: (-h[i], -w[i], i)),      # tallest first
        sorted(idx, key=lambda i: (-w[i], -h[i], i)),      # widest first
        sorted(idx, key=lambda i: (-w[i] * h[i], i)),      # largest area first
        sorted(idx, key=lambda i: (-max(w[i], h[i]), i)),  # largest extent first
    )
    for order in keys:
        placed = _bfdh(order, w, h, chip_w, chip_h, tol)
        if placed is not None:
            return placed
    return None


def legalize_guaranteed(cx0, cy0, widths, heights, chip_w, chip_h,
                        objective: str = "l1"):
    """Centres with provably zero overlap and all blocks in-bounds, moved minimally, or raise LegalizationInfeasible; returns (cx, cy, info)."""
    cx0 = np.asarray(cx0, float); cy0 = np.asarray(cy0, float)
    w = np.asarray(widths, float); h = np.asarray(heights, float)
    n = len(cx0)
    tot = float(np.sum(w * h))
    btol = _BOUND_TOL * max(chip_w, chip_h)

    # Pre-gates (before any single-block shortcut, so an oversized block is caught).
    big = np.where((w > chip_w + btol) | (h > chip_h + btol))[0]
    if big.size:
        raise LegalizationInfeasible("block-larger-than-die",
                                     {"blocks": big.tolist()})
    U = tot / (chip_w * chip_h)
    if U > 1.0 + 1e-6:
        raise LegalizationInfeasible("overpacked", {"utilization": U})

    def _clamp(cx, cy):
        return (np.clip(cx, w / 2.0, chip_w - w / 2.0),
                np.clip(cy, h / 2.0, chip_h - h / 2.0))

    def _accept(cx, cy, method):
        cx, cy = _clamp(cx, cy)
        of = _frac_overlap(cx, cy, w, h)
        inb = (cx - w / 2.0 >= -btol).all() and (cx + w / 2.0 <= chip_w + btol).all() \
            and (cy - h / 2.0 >= -btol).all() and (cy + h / 2.0 <= chip_h + btol).all()
        return (cx, cy, of, inb)

    if n <= 1:
        cx, cy = _clamp(cx0, cy0)
        return cx, cy, {"utilization": U, "method": "single"}

    if U >= 1.0 - 1e-6:                       # zero whitespace: only a tiling is legal
        if _frac_overlap(cx0, cy0, w, h) <= _OVERLAP_TOL:
            cx, cy = _clamp(cx0, cy0)
            return cx, cy, {"utilization": U, "method": "tiling-passthrough"}
        raise LegalizationInfeasible("tiling-only", {"utilization": U})

    # Solve in die-relative units so tolerances are scale-independent.
    s = max(chip_w, chip_h)
    ex, ey = _assign_axes(cx0, cy0, w, h)
    lox, hix = (w / 2.0) / s, (chip_w - w / 2.0) / s
    loy, hiy = (h / 2.0) / s, (chip_h - h / 2.0) / s
    exn = [(a, b, d / s) for a, b, d in ex]
    eyn = [(a, b, d / s) for a, b, d in ey]

    wx, feas_x = _left_pack(exn, cx0 / s, lox, hix)
    wy, feas_y = _left_pack(eyn, cy0 / s, loy, hiy)

    if feas_x and feas_y:
        if objective == "l1":
            rx = _min_displacement_1d(cx0 / s, exn, lox, hix)
            ry = _min_displacement_1d(cy0 / s, eyn, loy, hiy)
        else:
            rx = ry = None
        for cand, method in (((rx, ry), "min-displacement"), ((wx, wy), "left-pack-witness")):
            sx, sy = cand
            if sx is None or sy is None:
                continue
            cx, cy, of, inb = _accept(sx * s, sy * s, method)
            if of <= _OVERLAP_TOL and inb:
                return cx, cy, {"utilization": U, "method": method,
                                "overlap_frac": of, "n_edges_x": len(ex),
                                "n_edges_y": len(ey)}

    # Frozen order unsatisfiable (e.g. a heavily stacked input): construct a legal
    # arrangement directly. Legal-or-infeasible, never overlapping.
    shelf = _shelf_pack(w, h, chip_w, chip_h)
    if shelf is not None:
        cx, cy, of, inb = _accept(shelf[0], shelf[1], "shelf")
        if of <= _OVERLAP_TOL and inb:
            return cx, cy, {"utilization": U, "method": "shelf", "overlap_frac": of}

    raise LegalizationInfeasible("no-legal-arrangement-constructed",
                                 {"utilization": U, "feas_x": feas_x, "feas_y": feas_y})


def legalize_units_exact(units_placed: list[dict], chip_w: float, chip_h: float,
                         **kw) -> tuple[list[dict], float]:
    """`legalize_guaranteed` on solver unit dicts; returns (new units, overlap_frac), raises LegalizationInfeasible if none produced."""
    w = np.array([u["width"] for u in units_placed])
    h = np.array([u["height"] for u in units_placed])
    cx = np.array([u["leftx"] + u["width"] / 2 for u in units_placed])
    cy = np.array([u["bottomy"] + u["height"] / 2 for u in units_placed])
    cx, cy, info = legalize_guaranteed(cx, cy, w, h, chip_w, chip_h, **kw)
    out = [{**u, "leftx": float(cx[i] - w[i] / 2), "bottomy": float(cy[i] - h[i] / 2)}
           for i, u in enumerate(units_placed)]
    return out, float(info.get("overlap_frac", _frac_overlap(cx, cy, w, h)))
