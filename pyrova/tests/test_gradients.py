"""Finite-difference gradient checks for the solver adjoint and the placer.

WARNING: assertions are on the MAX error over probed coordinates, not the
median (a median hides outliers). Coordinates that straddle a subgradient kink
(peak-cell argmax switch or CVaR tail-mask change within the FD stencil) are
detected by comparing central FD at two step sizes and exempted from the strict
tolerance; most probed coordinates must still be clean.
"""

from __future__ import annotations
import numpy as np

from pyrova.core.design import Design, ThermalConfig
from pyrova.core.io import parse_config
from pyrova.thermal.fd_solver import GridFDSolver, random_power_map
from pyrova.optimizer.placer import DiffPlacer, cvar_and_grad

FLP = "pyrova/inputs/floorplans/ev6.flp"
CONFIG = "pyrova/inputs/configs/thermal.config"


def _solver(nr=24, nc=24):
    d = Design.from_flp(FLP)
    units = d.macro_flp_dicts()
    s = GridFDSolver(d.thermal_config.as_dict(), units, d.chip_width, d.chip_height, nr, nc)
    s.build(); s.factorize()
    return d, units, s


def _peak_dT(solver, powers):
    T = solver.solve(solver.build_rhs(powers))
    return float(solver.silicon_layer(T).max()) - solver.cfg["ambient"]


def test_config_defaults_match_bundled():
    """ThermalConfig defaults == the bundled thermal.config for every field the
    solver reads. Tests run on the defaults and experiments on the file; this
    pins both to the same thermal stack."""
    cfg_file = parse_config(CONFIG)
    cfg_def = ThermalConfig().as_dict()
    bad = [k for k in cfg_def
           if k in cfg_file and not np.isclose(cfg_def[k], cfg_file[k], rtol=1e-12)]
    for k in sorted(cfg_def):
        mark = "MISMATCH" if k in bad else "ok"
        print(f"  {k:12s} default={cfg_def[k]:<12g} file={cfg_file.get(k, 'absent')!s:<12s} {mark}")
    assert not bad, f"ThermalConfig defaults diverge from {CONFIG}: {bad}"


def test_adjoint_power_gradient(h=1e-3, tol=1e-6):
    """peak_T_gradient (adjoint) vs central FD of peak dT w.r.t. block power.

    T is exactly linear in Q (one LU solve), so away from an argmax tie the FD
    is near-exact and the tolerance can be tight: an assembly/adjoint bug shows
    up as an O(1) discrepancy, not a 1% one."""
    _, units, s = _solver()
    rng = np.random.default_rng(0)
    powers = random_power_map(units, 50.0, rng)
    _, grad = s.peak_T_gradient(powers)

    # Probe the blocks with the largest analytic sensitivity (most informative).
    names = sorted(grad, key=lambda n: -abs(grad[n]))[:6]
    worst = 0.0
    for nm in names:
        pp = dict(powers); pp[nm] += h
        pm = dict(powers); pm[nm] -= h
        fd = (_peak_dT(s, pp) - _peak_dT(s, pm)) / (2 * h)
        rel = abs(fd - grad[nm]) / (abs(fd) + 1e-12)
        worst = max(worst, rel)
        print(f"  dPeak/dP[{nm:8s}]  adjoint={grad[nm]:+.4f}  fd={fd:+.4f}  rel={rel:.2e}")
    print(f"  max rel error = {worst:.2e}")
    assert worst < tol, f"adjoint power gradient off (max rel {worst:.2e})"


def _fd_two_scale(f, x0, i, h, kink_rtol=1e-2):
    """Central FD of f at coordinate i with steps h and h/4.

    Returns (fd_fine, kinked): disagreement beyond kink_rtol means the stencil
    straddles a non-smoothness (argmax switch / tail-mask change / cell
    boundary), where the analytic subgradient need not match FD."""
    def central(step):
        xp = x0.copy(); xp[i] += step
        xm = x0.copy(); xm[i] -= step
        return (f(xp) - f(xm)) / (2 * step)
    fd_h, fd_h4 = central(h), central(h / 4)
    kinked = abs(fd_h - fd_h4) > kink_rtol * (abs(fd_h4) + 1e-12)
    return fd_h4, kinked


def _check_position_gradient(mode: str, h=2e-3, tol=1e-4, min_clean=3):
    """Placer position gradient vs holistic FD of the objective in raw params."""
    d, units, s = _solver()
    rng = np.random.default_rng(1)
    scen = [random_power_map(units, 50.0, rng) for _ in range(4)]
    scen = [np.array([sc[u["name"]] for u in units]) for sc in scen]

    # Isolate the thermal term: drop the non-overlap penalty for the check.
    pl = DiffPlacer(s, units, d.chip_width, d.chip_height, 24, 24,
                    alpha=0.75, nonoverlap_w=0.0)
    pl.raw_x += rng.standard_normal(pl.n) * 0.3   # jitter macros off cell edges
    pl.raw_y += rng.standard_normal(pl.n) * 0.3
    rx0, ry0 = pl.raw_x.copy(), pl.raw_y.copy()
    _, g_rx, _ = pl.objective_and_grad(scen, mode=mode)
    pl.raw_x, pl.raw_y = rx0.copy(), ry0.copy()

    def obj_value(rx):
        """Objective VALUE only (cheap: no gradient loops)."""
        cx = pl.cx_min + (pl.cx_max - pl.cx_min) / (1.0 + np.exp(-np.clip(rx, -50, 50)))
        cy = pl.cy_min + (pl.cy_max - pl.cy_min) / (1.0 + np.exp(-np.clip(ry0, -50, 50)))
        peaks = pl._scenario_peaks(cx, cy, scen)
        if mode == "mean":
            return float(peaks.mean())
        return cvar_and_grad(peaks, pl.alpha)[0]

    idx = np.argsort(-np.abs(g_rx))[:6]      # most sensitive coordinates
    worst_clean, n_clean = 0.0, 0
    for i in idx:
        fd, kinked = _fd_two_scale(obj_value, rx0, int(i), h)
        rel = abs(fd - g_rx[i]) / (abs(fd) + 1e-12)
        tag = "KINK (subgradient, exempt)" if kinked else f"rel={rel:.2e}"
        print(f"  [{mode}] dObj/draw_x[{int(i):2d}]  analytic={g_rx[i]:+.4e}  fd={fd:+.4e}  {tag}")
        if not kinked:
            n_clean += 1
            worst_clean = max(worst_clean, rel)
    print(f"  [{mode}] clean coords {n_clean}/6, max rel error = {worst_clean:.2e}")
    assert n_clean >= min_clean, f"{mode}: too few kink-free coordinates ({n_clean}/6)"
    assert worst_clean < tol, f"{mode} position gradient off (max clean rel {worst_clean:.2e})"


def test_position_gradient_mean():
    _check_position_gradient("mean")


def test_position_gradient_cvar():
    _check_position_gradient("cvar")


def test_smooth_hpwl_gradient(h=1e-7, tol=1e-6):
    """Smooth-HPWL surrogate gradient vs central FD. The log-sum-exp span is
    C-infinity, so the FD is near-exact and the tolerance tight; also checks the
    surrogate brackets exact HPWL from above."""
    from pyrova.objectives.wirelength import smooth_hpwl_grad
    rng = np.random.default_rng(5)
    n = 12
    cx = rng.uniform(0.0, 1.0, n)
    cy = rng.uniform(0.0, 1.0, n)
    nets = [list(rng.choice(n, size=int(rng.integers(2, 6)), replace=False))
            for _ in range(6)]
    gamma = 0.05
    val, gcx, gcy = smooth_hpwl_grad(cx, cy, nets, gamma)

    def value(cx_, cy_):
        return smooth_hpwl_grad(cx_, cy_, nets, gamma)[0]

    worst = 0.0
    for k in range(n):
        for arr, g in ((cx, gcx), (cy, gcy)):
            a = arr.copy(); a[k] += h
            b = arr.copy(); b[k] -= h
            if arr is cx:
                fd = (value(a, cy) - value(b, cy)) / (2 * h)
            else:
                fd = (value(cx, a) - value(cx, b)) / (2 * h)
            worst = max(worst, abs(fd - g[k]) / (abs(fd) + 1e-9))
    print(f"  smooth_hpwl max rel grad error = {worst:.2e}")
    assert worst < tol, f"smooth_hpwl gradient off (max rel {worst:.2e})"

    # Surrogate overshoots exact HPWL by <= gamma*ln(k), never undershoots.
    exact = 0.0
    for net in nets:
        idx = np.asarray(net)
        exact += (cx[idx].max() - cx[idx].min()) + (cy[idx].max() - cy[idx].min())
    assert val >= exact - 1e-12, "surrogate should upper-bound exact HPWL"
    print(f"  smooth={val:.4f} exact={exact:.4f} overshoot={val - exact:.4f}")


def test_position_gradient_hpwl(h=2e-3, tol=1e-4):
    """Wirelength-only baseline mode (obj = wl_weight*smoothHPWL, no thermal): the
    gradient is the pure smooth-HPWL gradient chained through the sigmoid, smooth
    everywhere, so every probed coordinate is clean."""
    from pyrova.objectives.wirelength import smooth_hpwl_grad
    d, units, s = _solver()
    rng = np.random.default_rng(4)
    scen = [random_power_map(units, 50.0, rng) for _ in range(3)]
    scen = [np.array([sc[u["name"]] for u in units]) for sc in scen]
    n = len(units)
    nets = [list(rng.choice(n, size=int(rng.integers(2, 6)), replace=False))
            for _ in range(8)]
    wl_w = 3.0e3
    pl = DiffPlacer(s, units, d.chip_width, d.chip_height, 24, 24, alpha=0.75,
                    nonoverlap_w=0.0, nets=nets, wl_weight=wl_w)
    pl.raw_x += rng.standard_normal(pl.n) * 0.3
    pl.raw_y += rng.standard_normal(pl.n) * 0.3
    rx0, ry0 = pl.raw_x.copy(), pl.raw_y.copy()
    obj0, g_rx, _ = pl.objective_and_grad(scen, mode="hpwl")
    assert obj0 > 0, "hpwl-mode objective should be the (positive) wirelength"
    pl.raw_x, pl.raw_y = rx0.copy(), ry0.copy()

    def obj_value(rx):
        cx = pl.cx_min + (pl.cx_max - pl.cx_min) / (1.0 + np.exp(-np.clip(rx, -50, 50)))
        cy = pl.cy_min + (pl.cy_max - pl.cy_min) / (1.0 + np.exp(-np.clip(ry0, -50, 50)))
        return wl_w * smooth_hpwl_grad(cx, cy, [np.asarray(nt) for nt in nets],
                                       pl.wl_gamma)[0]

    worst = 0.0
    for i in np.argsort(-np.abs(g_rx))[:6]:
        fd, _ = _fd_two_scale(obj_value, rx0, int(i), h)
        worst = max(worst, abs(fd - g_rx[i]) / (abs(fd) + 1e-12))
    print(f"  [hpwl] max rel error = {worst:.2e}")
    assert worst < tol, f"hpwl-mode position gradient off (max rel {worst:.2e})"


def test_position_gradient_wirelength(h=2e-3, tol=1e-4, min_clean=3):
    """Placer position gradient with the wirelength term active vs FD of the
    assembled objective (thermal mean + wl_weight*smoothHPWL)."""
    from pyrova.objectives.wirelength import smooth_hpwl_grad
    d, units, s = _solver()
    rng = np.random.default_rng(2)
    scen = [random_power_map(units, 50.0, rng) for _ in range(4)]
    scen = [np.array([sc[u["name"]] for u in units]) for sc in scen]
    n = len(units)
    nets = [list(rng.choice(n, size=int(rng.integers(2, 6)), replace=False))
            for _ in range(8)]
    wl_w = 3.0e3

    pl = DiffPlacer(s, units, d.chip_width, d.chip_height, 24, 24, alpha=0.75,
                    nonoverlap_w=0.0, nets=nets, wl_weight=wl_w)
    pl.raw_x += rng.standard_normal(pl.n) * 0.3
    pl.raw_y += rng.standard_normal(pl.n) * 0.3
    rx0, ry0 = pl.raw_x.copy(), pl.raw_y.copy()
    _, g_rx, _ = pl.objective_and_grad(scen, mode="mean")
    pl.raw_x, pl.raw_y = rx0.copy(), ry0.copy()

    def obj_value(rx):
        cx = pl.cx_min + (pl.cx_max - pl.cx_min) / (1.0 + np.exp(-np.clip(rx, -50, 50)))
        cy = pl.cy_min + (pl.cy_max - pl.cy_min) / (1.0 + np.exp(-np.clip(ry0, -50, 50)))
        peaks = pl._scenario_peaks(cx, cy, scen)
        wl = smooth_hpwl_grad(cx, cy, [np.asarray(nt) for nt in nets], pl.wl_gamma)[0]
        return float(peaks.mean()) + wl_w * wl

    idx = np.argsort(-np.abs(g_rx))[:6]
    worst_clean, n_clean = 0.0, 0
    for i in idx:
        fd, kinked = _fd_two_scale(obj_value, rx0, int(i), h)
        rel = abs(fd - g_rx[i]) / (abs(fd) + 1e-12)
        tag = "KINK (subgradient, exempt)" if kinked else f"rel={rel:.2e}"
        print(f"  [wl] dObj/draw_x[{int(i):2d}]  analytic={g_rx[i]:+.4e}  fd={fd:+.4e}  {tag}")
        if not kinked:
            n_clean += 1
            worst_clean = max(worst_clean, rel)
    print(f"  [wl] clean coords {n_clean}/6, max rel error = {worst_clean:.2e}")
    assert n_clean >= min_clean, f"wirelength: too few kink-free coordinates ({n_clean}/6)"
    assert worst_clean < tol, f"wirelength position gradient off (max clean rel {worst_clean:.2e})"


def test_density_gradient_direct(h=1e-8, tol=1e-6):
    """Density overflow penalty gradient vs central FD, in metre space (isolates
    the penalty math from the placer chain). A coarse grid is used so overlapping
    blocks clearly drive some bin over the legality target; away from the grid
    breakpoints the penalty is piecewise-quadratic in position, so central FD is
    near-exact."""
    from pyrova.objectives.density import density_penalty
    chip, nr, nc = 0.016, 8, 8
    cx = np.array([0.0080, 0.0086, 0.0120])          # first two overlap; third breaks symmetry
    cy = np.array([0.0080, 0.0083, 0.0120])
    w = np.array([0.0030, 0.0030, 0.0030])
    hgt = np.array([0.0030, 0.0030, 0.0030])
    D, gx, gy = density_penalty(cx, cy, w, hgt, chip, chip, nr, nc, t=1.0)
    assert D > 0.0, "test config must have an over-full bin (rho>t) or the check is vacuous"

    def val_x(cc):
        return density_penalty(cc, cy, w, hgt, chip, chip, nr, nc, t=1.0)[0]

    def val_y(cc):
        return density_penalty(cx, cc, w, hgt, chip, chip, nr, nc, t=1.0)[0]

    worst_clean, n_clean = 0.0, 0
    for b in range(len(cx)):
        fd, kinked = _fd_two_scale(val_x, cx, b, h)
        if not kinked:
            n_clean += 1
            worst_clean = max(worst_clean, abs(fd - gx[b]) / (abs(fd) + 1e-12))
        fd, kinked = _fd_two_scale(val_y, cy, b, h)
        if not kinked:
            n_clean += 1
            worst_clean = max(worst_clean, abs(fd - gy[b]) / (abs(fd) + 1e-12))
    print(f"  [density] clean coords {n_clean}/6, max rel error = {worst_clean:.2e}")
    assert n_clean >= 5, f"too few kink-free coords ({n_clean}/6) — reposition off lattice alignment"
    assert worst_clean < tol, f"density gradient off (max clean rel {worst_clean:.2e})"


def test_position_gradient_density(h=2e-3, tol=1e-4, min_clean=3):
    """Density term chained through the sigmoid to the raw parameters. Isolated via
    mode='hpwl' with zero wirelength and zero pairwise weight, so the objective is
    exactly lambda_D * (bin-overflow penalty); the macros are squeezed together so
    the term is active."""
    from pyrova.objectives.density import density_penalty
    d, units, s = _solver()
    rng = np.random.default_rng(3)
    scen = [np.array([random_power_map(units, 50.0, rng)[u["name"]] for u in units])
            for _ in range(2)]
    ndr = ndc = 16
    pl = DiffPlacer(s, units, d.chip_width, d.chip_height, 24, 24,
                    nonoverlap_w=0.0, density_w=1.0, density_grid=(ndr, ndc))
    pl.raw_x = rng.standard_normal(pl.n) * 0.1          # cluster the macros so rho>t somewhere
    pl.raw_y = rng.standard_normal(pl.n) * 0.1
    rx0, ry0 = pl.raw_x.copy(), pl.raw_y.copy()
    _, g_rx, _ = pl.objective_and_grad(scen, mode="hpwl", density_lambda=1.0)
    pl.raw_x, pl.raw_y = rx0.copy(), ry0.copy()

    def obj_value(rx):
        cx = pl.cx_min + (pl.cx_max - pl.cx_min) / (1.0 + np.exp(-np.clip(rx, -50, 50)))
        cy = pl.cy_min + (pl.cy_max - pl.cy_min) / (1.0 + np.exp(-np.clip(ry0, -50, 50)))
        return density_penalty(cx, cy, pl.widths, pl.heights,
                               pl.chip_w, pl.chip_h, ndr, ndc)[0]

    idx = np.argsort(-np.abs(g_rx))[:6]
    worst_clean, n_clean = 0.0, 0
    for i in idx:
        fd, kinked = _fd_two_scale(obj_value, rx0, int(i), h)
        if not kinked:
            n_clean += 1
            worst_clean = max(worst_clean, abs(fd - g_rx[i]) / (abs(fd) + 1e-12))
    print(f"  [density-chain] clean coords {n_clean}/6, max rel error = {worst_clean:.2e}")
    assert n_clean >= min_clean, f"density chain: too few kink-free coords ({n_clean}/6)"
    assert worst_clean < tol, f"density-mode position gradient off (max clean rel {worst_clean:.2e})"


def test_weighted_cvar_uniform_equivalence():
    """Uniform weights reproduce the unweighted CVaR exactly (value and gradient,
    metrics and placer paths) whenever (1-alpha)*N is an integer, e.g. N=40,
    alpha=0.9 and neighbours."""
    from pyrova.evaluation.metrics import cvar as m_cvar, mean_cvar
    rng = np.random.default_rng(11)
    for n in (20, 40, 80):
        x = rng.normal(50.0, 8.0, size=n)          # continuous -> no ties
        w = np.full(n, 1.0 / n)
        alpha = 0.9
        v_u = m_cvar(x, alpha)
        v_w = m_cvar(x, alpha, weights=w)
        assert abs(v_u - v_w) < 1e-12, f"metrics.cvar mismatch at N={n}: {v_u} vs {v_w}"
        mc_u = mean_cvar(x, alpha)
        mc_w = mean_cvar(x, alpha, weights=w)
        assert max(abs(a - b) for a, b in zip(mc_u, mc_w)) < 1e-12
        val_u, g_u = cvar_and_grad(x, alpha)
        val_w, g_w = cvar_and_grad(x, alpha, weights=w)
        assert abs(val_u - val_w) < 1e-12, f"cvar_and_grad value mismatch at N={n}"
        assert np.abs(g_u - g_w).max() < 1e-12, f"cvar_and_grad grad mismatch at N={n}"
        print(f"  N={n}: unweighted == uniform-weighted (value and gradient)")

    # Placer objective path: weights=uniform must equal weights=None exactly.
    d, units, s = _solver(nr=16, nc=16)
    rng = np.random.default_rng(3)
    scen = [np.array([random_power_map(units, 50.0, rng)[u["name"]] for u in units])
            for _ in range(10)]
    pl = DiffPlacer(s, units, d.chip_width, d.chip_height, 16, 16,
                    alpha=0.9, nonoverlap_w=0.0)
    for mode in ("mean", "cvar"):
        o_u, gx_u, gy_u = pl.objective_and_grad(scen, mode=mode)
        o_w, gx_w, gy_w = pl.objective_and_grad(scen, mode=mode,
                                                weights=np.full(len(scen), 0.1))
        assert abs(o_u - o_w) < 1e-12 and np.abs(gx_u - gx_w).max() < 1e-12 \
            and np.abs(gy_u - gy_w).max() < 1e-12, f"placer {mode} mismatch"
        print(f"  placer mode={mode}: uniform weights == unweighted")


if __name__ == "__main__":
    print("test_config_defaults_match_bundled:")
    test_config_defaults_match_bundled()
    print("test_adjoint_power_gradient:")
    test_adjoint_power_gradient()
    print("test_position_gradient (mean):")
    test_position_gradient_mean()
    print("test_position_gradient (cvar):")
    test_position_gradient_cvar()
    print("test_smooth_hpwl_gradient:")
    test_smooth_hpwl_gradient()
    print("test_position_gradient (hpwl-only baseline):")
    test_position_gradient_hpwl()
    print("test_position_gradient (wirelength):")
    test_position_gradient_wirelength()
    print("test_density_gradient_direct:")
    test_density_gradient_direct()
    print("test_position_gradient (density):")
    test_position_gradient_density()
    print("test_weighted_cvar_uniform_equivalence:")
    test_weighted_cvar_uniform_equivalence()
    print("\nALL CHECKS PASSED")
