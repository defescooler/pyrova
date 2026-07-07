"""Soft non-overlap penalty and its gradient w.r.t. macro centres."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.design import Design


def nonoverlap_penalty(cx: np.ndarray, cy: np.ndarray,
                       widths: np.ndarray, heights: np.ndarray,
                       gap: float = 0.0) -> tuple[float, np.ndarray, np.ndarray]:
    """Sum of pairwise rectangle-overlap areas (plus `gap` margin) and its exact
    gradient. Returns (penalty, grad_cx, grad_cy) w.r.t. macro centres."""
    n = len(cx)
    pen = 0.0
    gcx = np.zeros(n)
    gcy = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            ox = (widths[i] + widths[j]) / 2.0 + gap - dx
            oy = (heights[i] + heights[j]) / 2.0 + gap - dy
            if ox <= 0 or oy <= 0:
                continue
            pen += ox * oy
            sgx = 1.0 if cx[i] < cx[j] else -1.0
            sgy = 1.0 if cy[i] < cy[j] else -1.0
            gcx[i] += oy * sgx;  gcx[j] -= oy * sgx
            gcy[i] += ox * sgy;  gcy[j] -= ox * sgy
    return pen, gcx, gcy


def overlap_penalty(design: "Design",
                    gap: float = 0.0) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Soft non-overlap penalty for a Design.

    Returns (penalty, grad_cx, grad_cy) where the gradients are w.r.t. macro
    centre coordinates, in macro index order (same ordering as design.macros).
    """
    cx = np.array([m.centre_x for m in design.macros])
    cy = np.array([m.centre_y for m in design.macros])
    widths  = np.array([m.width  for m in design.macros])
    heights = np.array([m.height for m in design.macros])
    return nonoverlap_penalty(cx, cy, widths, heights, gap)
