"""Finite-difference gradient checks for the solver adjoint and the placer."""

from __future__ import annotations
import numpy as np

from pyrova.core.design import Design
from pyrova.thermal.fd_solver import GridFDSolver, random_power_map
from pyrova.optimizer.placer import DiffPlacer

FLP = "pyrova/inputs/floorplans/ev6.flp"


def _solver(nr=24, nc=24):
    d = Design.from_flp(FLP)
    units = d.macro_flp_dicts()
    s = GridFDSolver(d.thermal_config.as_dict(), units, d.chip_width, d.chip_height, nr, nc)
    s.build(); s.factorize()
    return d, units, s


def _peak_dT(solver, powers):
    T = solver.solve(solver.build_rhs(powers))
    return float(solver.silicon_layer(T).max()) - solver.cfg["ambient"]


def test_adjoint_power_gradient(h=1e-3, tol=2e-2):
    """peak_T_gradient (adjoint) vs central FD of peak T w.r.t. block power."""
    _, units, s = _solver()
    rng = np.random.default_rng(0)
    powers = random_power_map(units, 50.0, rng)
    _, grad = s.peak_T_gradient(powers)

    # Check the blocks with the largest analytic sensitivity (most informative).
    names = sorted(grad, key=lambda n: -abs(grad[n]))[:6]
    errs = []
    for nm in names:
        pp = dict(powers); pp[nm] += h
        pm = dict(powers); pm[nm] -= h
        fd = (_peak_dT(s, pp) - _peak_dT(s, pm)) / (2 * h)
        rel = abs(fd - grad[nm]) / (abs(fd) + 1e-12)
        errs.append(rel)
        print(f"  dPeak/dP[{nm:8s}]  adjoint={grad[nm]:+.4f}  fd={fd:+.4f}  rel={rel:.2e}")
    med = float(np.median(errs))
    print(f"  median rel error = {med:.2e}")
    assert med < tol, f"adjoint power gradient off (median rel {med:.2e})"


def test_position_gradient(h=2e-3, tol=1e-3):
    """Placer position gradient (adjoint x dQ/dp x sigmoid) vs holistic FD of
    the objective w.r.t. the raw (sigmoid) parameters. Evaluated off the grid:
    overlap area is kinked at cell boundaries, where the analytic derivative is
    a (valid) subgradient and central FD returns the averaged limit."""
    d, units, s = _solver()
    rng = np.random.default_rng(1)
    scen = [random_power_map(units, 50.0, rng) for _ in range(4)]
    scen = [np.array([sc[u["name"]] for u in units]) for sc in scen]

    # Isolate the thermal term: drop the non-overlap penalty for the check.
    pl = DiffPlacer(s, units, d.chip_width, d.chip_height, 24, 24,
                    alpha=0.9, nonoverlap_w=0.0)
    pl.raw_x += rng.standard_normal(pl.n) * 0.3   # jitter macros off cell edges
    pl.raw_y += rng.standard_normal(pl.n) * 0.3
    _, g_rx, _ = pl.objective_and_grad(scen, mode="mean")

    def obj_at(rx, ry):
        pl.raw_x, pl.raw_y = rx, ry
        return pl.objective_and_grad(scen, mode="mean")[0]

    rx0, ry0 = pl.raw_x.copy(), pl.raw_y.copy()
    idx = np.argsort(-np.abs(g_rx))[:6]      # most sensitive coordinates
    errs = []
    for i in idx:
        rp = rx0.copy(); rp[i] += h
        rm = rx0.copy(); rm[i] -= h
        fd = (obj_at(rp, ry0) - obj_at(rm, ry0)) / (2 * h)
        pl.raw_x, pl.raw_y = rx0.copy(), ry0.copy()
        rel = abs(fd - g_rx[i]) / (abs(fd) + 1e-12)
        errs.append(rel)
        print(f"  dObj/draw_x[{i:2d}]  analytic={g_rx[i]:+.4e}  fd={fd:+.4e}  rel={rel:.2e}")
    med = float(np.median(errs))
    print(f"  median rel error = {med:.2e}")
    assert med < tol, f"position gradient off (median rel {med:.2e})"


def test_dro_penalty_scaling():
    """Worst-case-CVaR penalty is zero at eps=0, positive otherwise, and scales
    linearly in eps (it is eps/(1-alpha) times a fixed tail sensitivity)."""
    d, units, s = _solver(nr=16, nc=16)
    rng = np.random.default_rng(7)
    scen = [np.array([random_power_map(units, 50.0, rng)[u["name"]] for u in units])
            for _ in range(8)]

    def pen(eps):
        pl = DiffPlacer(s, units, d.chip_width, d.chip_height, 16, 16,
                        alpha=0.9, eps_dro=eps, nonoverlap_w=0.0)
        return pl.dro_term(scen)

    p0, p1, p2 = pen(0.0), pen(0.5), pen(1.0)
    print(f"  penalty  eps0={p0:.4f}  eps0.5={p1:.4f}  eps1={p2:.4f}")
    assert p0 == 0.0, "penalty should vanish at eps=0"
    assert p1 > 0.0, "penalty should be positive for eps>0"
    rel = abs(p2 - 2.0 * p1) / (abs(p2) + 1e-12)
    print(f"  linearity rel error = {rel:.2e}")
    assert rel < 1e-6, "penalty should be linear in eps"


def test_dro_has_teeth():
    """DRO gradient is non-zero and distinct from CVaR, and DRO optimisation
    descends the worst-case-CVaR penalty it targets."""
    d, units, s = _solver(nr=16, nc=16)
    rng = np.random.default_rng(3)
    scen = [np.array([random_power_map(units, 50.0, rng)[u["name"]] for u in units])
            for _ in range(6)]

    pl = DiffPlacer(s, units, d.chip_width, d.chip_height, 16, 16,
                    alpha=0.9, eps_dro=0.5, nonoverlap_w=0.0)
    _, gx_cvar, _ = pl.objective_and_grad(scen, mode="cvar")
    _, gx_dro, _ = pl.objective_and_grad(scen, mode="dro")
    diff = float(np.abs(gx_dro - gx_cvar).max())
    print(f"  max |g_dro - g_cvar| = {diff:.3e}")
    assert diff > 1e-6, "DRO gradient is a no-op (regressed to CVaR-only)"

    P0 = pl.dro_term(scen)
    pl.optimize(scen, mode="dro", n_iter=12, lr=2e-2, verbose=False)
    P1 = pl.dro_term(scen)
    print(f"  DRO penalty  start={P0:.4f}  after DRO={P1:.4f}")
    assert P1 < P0, "DRO did not reduce its worst-case-CVaR penalty"


if __name__ == "__main__":
    print("test_adjoint_power_gradient:")
    test_adjoint_power_gradient()
    print("test_position_gradient:")
    test_position_gradient()
    print("test_dro_penalty_scaling:")
    test_dro_penalty_scaling()
    print("test_dro_has_teeth:")
    test_dro_has_teeth()
    print("\nALL CHECKS PASSED")
