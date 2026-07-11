"""Bin-density overflow penalty and its exact gradient w.r.t. macro centres."""

from __future__ import annotations

import numpy as np


def _ramp_step(z, r):
    """Smoothed unit step of half-width r."""
    mid = (z > -r) & (z < r)
    return np.where(mid, (z + r) / (2.0 * r), np.where(z >= r, 1.0, 0.0))


def _ramp_step_integral(z, r):
    """Antiderivative of _ramp_step."""
    mid = (z > -r) & (z < r)
    return np.where(mid, (z + r) ** 2 / (4.0 * r), np.where(z >= r, z, 0.0))


def _cover(c, size, edges, r):
    """Covered length of each block in each bin and its derivative w.r.t. the centre."""
    d = edges[None, :] - c[:, None]
    half = size[:, None] / 2.0
    G = _ramp_step_integral(d + half, r) - _ramp_step_integral(d - half, r)
    k = _ramp_step(d + half, r) - _ramp_step(d - half, r)
    dG = -k
    G[:, 0] = 0.0; G[:, -1] = size          # clamp edge bins so a wall block conserves mass
    dG[:, 0] = 0.0; dG[:, -1] = 0.0
    L = G[:, 1:] - G[:, :-1]
    dL = dG[:, 1:] - dG[:, :-1]
    return L, dL


def density_penalty(cx, cy, widths, heights, chip_w, chip_h, nr, nc, t=1.0):
    """Bin over-fullness penalty and its gradient; returns (D, grad_cx, grad_cy)."""
    cx = np.asarray(cx, float); cy = np.asarray(cy, float)
    w = np.asarray(widths, float); h = np.asarray(heights, float)
    cw = chip_w / nc; ch = chip_h / nr; A_bin = cw * ch
    xe = np.arange(nc + 1) * cw
    ye = np.arange(nr + 1) * ch
    Lx, dLx = _cover(cx, w, xe, cw / 2.0)
    Ly, dLy = _cover(cy, h, ye, ch / 2.0)
    rho = (Ly.T @ Lx) / A_bin
    over = np.maximum(0.0, rho - t)
    D = 0.5 * float(np.sum(over ** 2))
    gx = np.einsum('bi,ib->b', Ly, over @ dLx.T) / A_bin
    gy = np.einsum('bi,ib->b', dLy, over @ Lx.T) / A_bin
    return D, gx, gy


def density_weight_ramp(it, n_iter, lam0, lam_max):
    """Geometric ramp from lam0 to lam_max over n_iter steps; lam0<=0 stays at lam_max."""
    if lam_max <= 0.0:
        return 0.0
    lo = lam0 if lam0 > 0.0 else lam_max
    frac = (it - 1) / max(1, n_iter - 1)
    return float(lo * (lam_max / lo) ** frac)
