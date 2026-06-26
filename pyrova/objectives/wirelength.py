"""Half-perimeter wirelength over net bounding boxes."""

from __future__ import annotations
from typing import TYPE_CHECKING

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


def smooth_hpwl(design: "Design", gamma: float = 1.0) -> float:
    """TODO: Differentiable log-sum-exp HPWL surrogate. """
    raise NotImplementedError("smooth_hpwl() not yet implemented")
