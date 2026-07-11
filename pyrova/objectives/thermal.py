"""Peak-temperature and CVaR thermal objective terms."""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..evaluation.metrics import cvar, mean_cvar

if TYPE_CHECKING:                       # avoid hard import; solver is passed in
    from ..core.design import Design
    from ..thermal.fd_solver import GridFDSolver


def _as_power_dict(design: "Design", power_map) -> dict[str, float]:
    """Coerce a power_map (dict | PowerScenario | array in macro order) -> dict."""
    if isinstance(power_map, dict):
        return power_map
    if hasattr(power_map, "powers"):                     # PowerScenario
        return power_map.powers
    import numpy as np                                   # array in macro order
    arr = np.asarray(power_map, dtype=float)
    return dict(zip(design.macro_names, arr))


def peak_temperature(design: "Design", solver: "GridFDSolver", power_map) -> float:
    """Peak silicon temperature rise for one workload; returns peak_dT = max(T_silicon) - ambient [K]."""
    powers = _as_power_dict(design, power_map)
    solver.units = design.macro_flp_dicts()
    Q = solver.build_rhs(powers)
    T = solver.solve(Q)
    T_si = solver.silicon_layer(T)
    return float(T_si.max()) - solver.cfg["ambient"]


def cvar_temperature(design: "Design", solver: "GridFDSolver",
                     power_maps, alpha: float = 0.90) -> float:
    """CVaR_alpha of peak dT across a set of workloads [K]."""
    peaks = [peak_temperature(design, solver, pm) for pm in power_maps]
    return cvar(peaks, alpha)


def mean_cvar_temperature(design: "Design", solver: "GridFDSolver",
                          power_maps, alpha: float = 0.90) -> tuple[float, float]:
    """(mean, CVaR_alpha) of peak dT across a set of workloads in one solve pass [K]."""
    peaks = [peak_temperature(design, solver, pm) for pm in power_maps]
    return mean_cvar(peaks, alpha)
