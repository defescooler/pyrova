"""Worst-case CVaR penalty terms over a type-1 Wasserstein ambiguity ball.

Two penalty families: the tail-data approximation used by placer mode 'dro',
and the certified global-Lambda terms used by mode 'dro_exact'.

WARNING: the tail-data penalty (`wasserstein_cvar_penalty` / `dro_penalty`)
estimates the dual's Lipschitz constant from observed tail scenarios only, a
lower bound measured at ~50% of the exact constant, so it under-penalizes; it
matches the dual's shape but not its closed form. `exact_lipschitz` /
`exact_penalty_terms` compute the certified data-independent constant instead.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..thermal.fd_solver import GridFDSolver


def wasserstein_cvar_penalty(peaks: np.ndarray, grad_norms: np.ndarray,
                             epsilon: float, alpha: float = 0.9) -> float:
    """Additive DRO term in the shape of the type-1 Wasserstein dual.

    `peaks`      : per-scenario peak temperatures (any constant offset cancels;
                   used only to select the CVaR tail)
    `grad_norms` : per-scenario L2 norm of d(peak)/d(power)
    Returns epsilon, scaled by the inverse tail probability, times the largest
    power-sensitivity among scenarios in the CVaR tail. Zero when epsilon <= 0.

    WARNING: the max over tail data points is a lower bound on the dual's true
    Lipschitz constant; use `exact_lipschitz` for a certified penalty.
    """
    peaks = np.asarray(peaks, dtype=float)
    grad_norms = np.asarray(grad_norms, dtype=float)
    if epsilon <= 0.0 or peaks.size == 0:
        return 0.0
    q = np.quantile(peaks, alpha)
    tail = peaks >= q
    lam = float(grad_norms[tail].max()) if tail.any() else float(grad_norms.max())
    return epsilon / (1.0 - alpha) * lam


def dro_penalty(solver: "GridFDSolver", power_maps: list[dict[str, float]],
                epsilon: float, alpha: float = 0.9) -> float:
    """Tail-scenario DRO penalty for the current placement over observed scenarios.

    `power_maps` : list of {block_name: power_W} scenarios
    Solves each scenario, takes peak dT and the adjoint power-sensitivity, and
    returns the Wasserstein-dual-shaped penalty added to empirical CVaR.
    """
    peaks, grad_norms = [], []
    for pw in power_maps:
        peak_dt, grad = solver.peak_T_gradient(pw)
        peaks.append(peak_dt)
        g = np.array([grad.get(u["name"], 0.0) for u in solver.units])
        grad_norms.append(float(np.linalg.norm(g)))
    return wasserstein_cvar_penalty(peaks, grad_norms, epsilon, alpha)


def si_adjoint_rows(solver: "GridFDSolver") -> np.ndarray:
    """(n_si, N) matrix of adjoint vectors lam_i = G^{-T} e_i for every Si node.

    G is placement-independent (positions enter only the RHS), so this is
    computed ONCE per solver and reused across placer iterations."""
    nr, nc = solver.nr, solver.nc
    L = np.zeros((nr * nc, solver.N))
    for i in range(nr):
        for j in range(nc):
            e = np.zeros(solver.N)
            e[solver._nidx(solver.SI, i, j)] = 1.0
            L[i * nc + j] = solver.adjoint_dT_dQ(e)
    return L


def exact_penalty_terms(solver: "GridFDSolver", lam_rows: np.ndarray,
                        sigma: np.ndarray | None = None
                        ) -> tuple[float, int, np.ndarray]:
    """Certified global dual coefficient at the CURRENT placement.

    Returns (Lambda, i_star, a_star):
      Lambda  = max_i ||D a_i||_2 with a_i = A(p)^T lam_i and D = diag(sigma)
                (sigma=None -> unscaled L2 ground metric),
      i_star  = argmax Si node (flat index),
      a_star  = its sensitivity row (n_units,).
    `CVaR_hat + eps/(1-alpha)*Lambda` upper-bounds the worst-case CVaR. Passing
    sigma = per-block power std makes the ground metric Mahalanobis rather than
    plain L2."""
    A = solver.power_injection_matrix()          # (N, n_units)
    rows = lam_rows @ A                          # (n_si, n_units): all a_i
    scaled = rows if sigma is None else rows * np.asarray(sigma)[None, :]
    norms = np.linalg.norm(scaled, axis=1)
    i_star = int(np.argmax(norms))
    return float(norms[i_star]), i_star, rows[i_star]


def exact_lipschitz(solver: "GridFDSolver") -> float:
    """Global Lipschitz constant Lambda(p) = max_i ||row_i(G^{-1} A)||_2 of peak
    dT w.r.t. the block power vector, over ALL silicon nodes i (independent of
    the observed scenarios). One adjoint solve per Si node — evaluation/reporting
    only; too slow for the optimisation inner loop.

    This is the constant the exact Wasserstein dual uses; comparing it with the
    tail-data max quantifies how much the tail-data penalty underestimates.
    """
    nr, nc = solver.nr, solver.nc
    n_units = len(solver.units)
    # Accumulate a_i = A(p)^T lam_i on the fly to avoid materialising A.
    best = 0.0
    for i in range(nr):
        for j in range(nc):
            dL_dT = np.zeros(solver.N)
            dL_dT[solver._nidx(solver.SI, i, j)] = 1.0
            lam = solver.adjoint_dT_dQ(dL_dT)
            g = np.zeros(n_units)
            for b, u in enumerate(solver.units):
                lx, by = u["leftx"], u["bottomy"]
                rx_b, ty_b = lx + u["width"], by + u["height"]
                block_area = u["width"] * u["height"]
                acc = 0.0
                for ii, jj, clx, crx, cbot, ctop in solver._touched_cells(u):
                    ow = max(0.0, min(rx_b, crx) - max(lx, clx))
                    oh = max(0.0, min(ty_b, ctop) - max(by, cbot))
                    if ow * oh > 0:
                        acc += lam[solver._nidx(solver.SI, ii, jj)] * (ow * oh) / block_area
                g[b] = acc
            best = max(best, float(np.linalg.norm(g)))
    return best
