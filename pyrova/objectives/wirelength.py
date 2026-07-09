"""Half-perimeter wirelength over net bounding boxes, plus a differentiable
log-sum-exp surrogate used by the constrained placer (Phase 3).

`hpwl` is the exact (non-differentiable) HPWL for reporting and budget checks.
`smooth_hpwl_grad` is the log-sum-exp surrogate the gradient placer optimises,
smooth everywhere so it has no subgradient kinks.

WARNING: the surrogate's `gamma` trades accuracy for stiffness — it converges
to true HPWL from above as gamma shrinks, but a smaller gamma stiffens the
gradient.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.design import Design


def hpwl(design: "Design") -> float:
    """Total half-perimeter wirelength over `design.nets` (per-net bounding box of
    the connected macro centres), in metres. Returns 0.0 when the design has no nets."""
    nets = getattr(design, "nets", None)
    if not nets:
        return 0.0
    centres = {m.name: (m.centre_x, m.centre_y) for m in design.macros}
    total = 0.0
    for net in nets:
        pts = [centres[n] for n in net if n in centres]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


def _lse_span(v: np.ndarray, inv_gamma: float, gamma: float):
    """Smooth `max(v) - min(v)` via log-sum-exp and its gradient w.r.t. v."""
    a = v * inv_gamma
    amax = a.max()                    # shift for numerical stability
    ep = np.exp(a - amax)
    sp = ep.sum()
    b = -v * inv_gamma
    bmax = b.max()
    em = np.exp(b - bmax)
    sm = em.sum()
    span = gamma * (np.log(sp) + amax) + gamma * (np.log(sm) + bmax)
    grad = ep / sp - em / sm
    return span, grad


def smooth_hpwl_grad(cx, cy, nets, gamma: float):
    """Log-sum-exp HPWL surrogate and its exact gradient w.r.t. macro centres.

    cx, cy : (n_macros,) centre coordinates [m], macro-index order.
    nets   : list of nets, each a sequence of macro INDICES into cx/cy.
    gamma  : smoothing length [m]; smaller -> closer to exact HPWL, stiffer grad.

    Returns (value, grad_cx, grad_cy). The surrogate is smooth everywhere, so
    the gradient is exact (no subgradient kinks) and the FD check is tight.
    """
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    n = len(cx)
    inv = 1.0 / gamma
    val = 0.0
    gcx = np.zeros(n)
    gcy = np.zeros(n)
    for net in nets:
        idx = np.asarray(net, dtype=int)
        if idx.size < 2:
            continue
        sx, gx = _lse_span(cx[idx], inv, gamma)
        sy, gy = _lse_span(cy[idx], inv, gamma)
        val += sx + sy
        np.add.at(gcx, idx, gx)
        np.add.at(gcy, idx, gy)
    return val, gcx, gcy


def smooth_hpwl(design: "Design", gamma: float | None = None) -> float:
    """Differentiable log-sum-exp HPWL value for a Design (metres).

    Returns 0.0 when the design has no nets. `gamma` defaults to 1% of the mean
    chip span. Names not present among the macros are dropped from their net.
    """
    nets = getattr(design, "nets", None)
    if not nets:
        return 0.0
    idx_of = {m.name: i for i, m in enumerate(design.macros)}
    cx = np.array([m.centre_x for m in design.macros])
    cy = np.array([m.centre_y for m in design.macros])
    net_idx = [[idx_of[nm] for nm in net if nm in idx_of] for net in nets]
    if gamma is None:
        gamma = 0.01 * 0.5 * (design.chip_width + design.chip_height)
    return smooth_hpwl_grad(cx, cy, net_idx, gamma)[0]
