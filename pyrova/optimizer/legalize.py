"""Minimal-displacement overlap-removal projection (legalization pass).

Gradient descent on the pairwise overlap area alone, in centre coordinates,
clamped to the die. NOT a placement-quality pass: it makes the minimal move
that removes overlap, so evaluating a legalized placement measures the
placement's own quality rather than the soft penalty's residual overlap.
Soft overlap penalties leave 0.5-2% residual overlap on optimized placements
(much worse under wirelength pressure), which flatters the placement unless
removed here first.

Converges on dies with whitespace (hetero-SoC 69%, Kraken 77%, BOOM ~76%
utilization). On a 100%-utilized tiling (EV6) no legal non-tiling exists and
this (or any) legalizer cannot help — check utilization first.

WARNING: report `overlap_frac` for every evaluated placement; any comparison
cell with post-legalization overlap > 0.1% is void.
"""

from __future__ import annotations
import numpy as np

from pyrova.objectives.overlap import nonoverlap_penalty


def legalize_positions(cx: np.ndarray, cy: np.ndarray,
                       widths: np.ndarray, heights: np.ndarray,
                       chip_w: float, chip_h: float,
                       tol_frac: float = 1e-3, max_iter: int = 4000,
                       ) -> tuple[np.ndarray, np.ndarray, float]:
    """Push block centres apart until overlap < tol_frac of total block area.

    Deterministic. Returns (cx, cy, achieved_overlap_frac).
    """
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
