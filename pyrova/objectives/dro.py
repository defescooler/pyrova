"""Worst-case CVaR penalty over a type-1 Wasserstein ambiguity ball."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..thermal.fd_solver import GridFDSolver


def wasserstein_cvar_penalty(peaks, grad_norms, epsilon: float,
                             alpha: float = 0.9) -> float:
    """Additive DRO term that turns empirical CVaR into worst-case CVaR.

    `peaks`      : per-scenario peak temperatures (any constant offset cancels)
    `grad_norms` : per-scenario L2 norm of d(peak)/d(power)
    Returns epsilon, scaled by the inverse tail probability, times the largest
    power-sensitivity among scenarios in the CVaR tail. Zero when epsilon <= 0.
    """
    peaks = np.asarray(peaks, dtype=float)
    grad_norms = np.asarray(grad_norms, dtype=float)
    if epsilon <= 0.0 or peaks.size == 0:
        return 0.0
    q = np.quantile(peaks, alpha)
    tail = peaks >= q
    lam = float(grad_norms[tail].max()) if tail.any() else float(grad_norms.max())
    return epsilon / (1.0 - alpha) * lam


def dro_penalty(solver: "GridFDSolver", power_maps, epsilon: float,
                alpha: float = 0.9) -> float:
    """Worst-case-CVaR DRO penalty for the current placement over observed scenarios.

    `power_maps` : list of {block_name: power_W} scenarios
    Solves each scenario, takes peak dT and the adjoint power-sensitivity, and
    returns the Wasserstein penalty added to empirical CVaR.
    """
    ambient = solver.cfg["ambient"]
    peaks, grad_norms = [], []
    for pw in power_maps:
        peak, grad = solver.peak_T_gradient(pw)
        peaks.append(peak - ambient)
        g = np.array([grad.get(u["name"], 0.0) for u in solver.units])
        grad_norms.append(float(np.linalg.norm(g)))
    return wasserstein_cvar_penalty(peaks, grad_norms, epsilon, alpha)
