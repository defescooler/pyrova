"""Differentiable floorplanner: Adam over sigmoid-bounded macro centres, with
thermal gradients from the adjoint solve.

Modes: 'mean' (expected peak dT), 'cvar' (empirical CVaR of peak dT), 'blend'
((1-gamma)*mean + gamma*CVaR — mean-anchored shrinkage of the noisy empirical
tail; gamma=blend_gamma), 'dro' (CVaR plus the tail-data approximation of the
type-1 Wasserstein penalty; historical — see objectives/dro.py caveats), and
'dro_exact' (CVaR plus the CERTIFIED global-Lambda dual penalty, optionally
Mahalanobis-scaled via dro_sigma; analytic gradient, no finite differences —
DRO_DERIVATION.md "Answers" 1-3).

Every mode optionally carries a wirelength term: with `nets` and `wl_weight>0`
the objective becomes (thermal term) + wl_weight*smoothHPWL, i.e. the
wirelength-penalised placement problem (Phase 3). With wl_weight=0 the placer
reproduces the historical unconstrained thermal optimisation exactly.
"""

from __future__ import annotations

import numpy as np

from pyrova.thermal.fd_solver import GridFDSolver
from pyrova.objectives.dro import (wasserstein_cvar_penalty, si_adjoint_rows,
                                   exact_penalty_terms)
from pyrova.objectives.overlap import nonoverlap_penalty
from pyrova.objectives.wirelength import smooth_hpwl_grad


# Helpers

def _make_units(units_orig: list[dict], cx: np.ndarray, cy: np.ndarray) -> list[dict]:
    """Unit dicts with the same sizes but centres moved to (cx, cy)."""
    return [{
        "name":    u["name"],
        "width":   u["width"],
        "height":  u["height"],
        "leftx":   float(cx[b]) - u["width"] / 2.0,
        "bottomy": float(cy[b]) - u["height"] / 2.0,
    } for b, u in enumerate(units_orig)]


def _build_rhs_at(solver: GridFDSolver, units_orig: list[dict],
                  cx: np.ndarray, cy: np.ndarray,
                  block_powers: np.ndarray) -> np.ndarray:
    """Set solver.units to the (cx, cy) placement and build the RHS for `block_powers`."""
    solver.units = _make_units(units_orig, cx, cy)
    bp = {u["name"]: float(block_powers[b]) for b, u in enumerate(units_orig)}
    return solver.build_rhs(bp)


def sigmoid_vec(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def cvar_and_grad(peaks: np.ndarray, alpha: float,
                  weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """CVaR_alpha and its (frozen-tail-mask) gradient w.r.t. each peak.

    CVaR_alpha = inf_tau {tau + E[(M-tau)_+]/(1-alpha)}, estimated empirically.
    The differentiable sibling of ``evaluation.metrics.cvar``. With `weights`
    (per-scenario probability weights, normalised internally) the tail is the
    top (1-alpha) of probability MASS (boundary scenario fractional):
    val = sum(phi_i * peak_i) / (1-alpha), grad_i = phi_i / (1-alpha), with
    phi_i the included mass. Uniform weights reproduce the unweighted estimator
    whenever (1-alpha)*N is an integer (no ties).
    """
    if weights is None:
        q = np.quantile(peaks, alpha)
        mask = peaks >= q
        count = max(1, int(mask.sum()))
        val = float(peaks[mask].mean())
        grad = np.where(mask, 1.0 / count, 0.0)
        return val, grad
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    order = np.argsort(-peaks)
    ws = w[order]
    m = 1.0 - alpha
    cum = np.concatenate(([0.0], np.cumsum(ws)))
    phi_sorted = np.clip(m - cum[:-1], 0.0, ws)
    phi = np.zeros_like(w)
    phi[order] = phi_sorted
    mass = phi.sum()
    if mass <= 0:                    # degenerate: all mass below the tail level
        i = int(order[0])
        grad = np.zeros_like(w); grad[i] = 1.0
        return float(peaks[i]), grad
    val = float((phi * peaks).sum() / mass)
    grad = phi / mass
    return val, grad


# Placer

class DiffPlacer:
    """Adam floorplanner over sigmoid-bounded macro centres.

    Parameters
    ----------
    solver       : GridFDSolver (built + factorized)
    units_orig   : original block list (unit dicts)
    chip_w/h     : chip dimensions [m]
    nr, nc       : thermal grid resolution
    alpha        : CVaR tail probability (default 0.9)
    eps_dro      : Wasserstein ball radius for the worst-case-CVaR penalty (default 0)
    nonoverlap_w : soft non-overlap penalty weight
    fd_delta_dro : finite-difference step for the DRO sensitivity term [m]

    WARNING: the non-overlap penalty is measured in area units (m^2), so
    `nonoverlap_w` must be re-scaled if the chip/block dimensions change —
    otherwise it no longer balances the thermal (K) objective term.
    """

    def __init__(self, solver: GridFDSolver, units_orig: list[dict],
                 chip_w: float, chip_h: float, nr: int, nc: int,
                 alpha: float = 0.9, eps_dro: float = 0.0,
                 blend_gamma: float = 0.5, dro_sigma: np.ndarray | None = None,
                 nonoverlap_w: float = 1e4, fd_delta_dro: float = 1e-6,
                 nets: list | None = None, wl_weight: float = 0.0,
                 wl_gamma: float | None = None):
        self.solver = solver
        self.units_orig = units_orig
        self.chip_w = chip_w
        self.chip_h = chip_h
        self.nr = nr
        self.nc = nc
        self.alpha = alpha
        self.eps_dro = eps_dro
        self.blend_gamma = blend_gamma
        self.dro_sigma = None if dro_sigma is None else np.asarray(dro_sigma, dtype=float)
        self.nonoverlap_w = nonoverlap_w
        self.fd_delta_dro = fd_delta_dro
        self._lam_rows = None      # cached per-Si-node adjoints (dro_exact)

        # Wirelength constraint (Phase 3). `nets` is a list of nets, each a list
        # of macro indices into units_orig; `wl_weight` (>0) adds the smooth-HPWL
        # term wl_weight*smoothHPWL to every mode's objective, turning the
        # unconstrained thermal problem into the wirelength-penalised one. The
        # smoothing length wl_gamma defaults to 1% of the mean chip span.
        self.nets = [np.asarray(net, dtype=int) for net in nets] if nets else []
        self.wl_weight = wl_weight
        self.wl_gamma = (wl_gamma if wl_gamma is not None
                         else 0.01 * 0.5 * (chip_w + chip_h))

        self.n = len(units_orig)
        self.widths = np.array([u["width"] for u in units_orig])
        self.heights = np.array([u["height"] for u in units_orig])

        # Centre bounds: each block must fit inside the chip.
        self.cx_min = self.widths / 2.0
        self.cx_max = chip_w - self.widths / 2.0
        self.cy_min = self.heights / 2.0
        self.cy_max = chip_h - self.heights / 2.0

        self.raw_x, self.raw_y = self._encode_original()

    def _encode_original(self) -> tuple[np.ndarray, np.ndarray]:
        cx0 = np.array([u["leftx"] + u["width"] / 2.0 for u in self.units_orig])
        cy0 = np.array([u["bottomy"] + u["height"] / 2.0 for u in self.units_orig])
        cx0 = np.clip(cx0, self.cx_min + 1e-9, self.cx_max - 1e-9)
        cy0 = np.clip(cy0, self.cy_min + 1e-9, self.cy_max - 1e-9)
        fx = np.clip((cx0 - self.cx_min) / (self.cx_max - self.cx_min + 1e-30), 1e-6, 1.0 - 1e-6)
        fy = np.clip((cy0 - self.cy_min) / (self.cy_max - self.cy_min + 1e-30), 1e-6, 1.0 - 1e-6)
        return np.log(fx / (1.0 - fx)), np.log(fy / (1.0 - fy))

    def get_positions(self) -> tuple[np.ndarray, np.ndarray]:
        cx = self.cx_min + (self.cx_max - self.cx_min) * sigmoid_vec(self.raw_x)
        cy = self.cy_min + (self.cy_max - self.cy_min) * sigmoid_vec(self.raw_y)
        return cx, cy

    def get_units(self) -> list[dict]:
        cx, cy = self.get_positions()
        return _make_units(self.units_orig, cx, cy)

    def wirelength(self, exact: bool = True) -> float:
        """Current HPWL over self.nets [m]: exact bounding-box (default) or the
        smooth log-sum-exp surrogate. Returns 0.0 when no nets are set."""
        if not self.nets:
            return 0.0
        cx, cy = self.get_positions()
        if not exact:
            return float(smooth_hpwl_grad(cx, cy, self.nets, self.wl_gamma)[0])
        total = 0.0
        for idx in self.nets:
            if len(idx) < 2:
                continue
            total += float((cx[idx].max() - cx[idx].min())
                           + (cy[idx].max() - cy[idx].min()))
        return total

    def reset(self) -> None:
        self.raw_x, self.raw_y = self._encode_original()

    # DRO sensitivity terms

    def _dro_lipschitz(self, cx: np.ndarray, cy: np.ndarray, pw: np.ndarray) -> float:
        """Power-sensitivity norm ||d(peak)/d(power)|| at a given scenario power."""
        self.solver.units = _make_units(self.units_orig, cx, cy)
        bp = {u["name"]: float(pw[b]) for b, u in enumerate(self.units_orig)}
        _, g_dict = self.solver.peak_T_gradient(bp)
        g = np.array([g_dict.get(u["name"], 0.0) for u in self.units_orig])
        return float(np.linalg.norm(g))

    def _scenario_peaks(self, cx, cy, scenario_powers) -> np.ndarray:
        """Per-scenario peak dT at the given placement."""
        T_amb = self.solver.cfg["ambient"]
        peaks = np.zeros(len(scenario_powers))
        for s, pw in enumerate(scenario_powers):
            T = self.solver.solve(_build_rhs_at(self.solver, self.units_orig, cx, cy, pw))
            peaks[s] = float(self.solver.silicon_layer(T).max()) - T_amb
        return peaks

    def _dro_penalty_at(self, cx, cy, peaks, scenario_powers):
        """DRO penalty value and the worst-case tail scenario index.

        Power-sensitivity is evaluated only on the CVaR tail; the tail scenario
        with the largest norm sets the Wasserstein penalty and the gradient.
        """
        q = np.quantile(peaks, self.alpha)
        tail = np.where(peaks >= q)[0]
        gnorms = np.zeros(len(scenario_powers))
        worst_i, worst = int(tail[0]), -1.0
        for s in tail:
            gnorms[s] = self._dro_lipschitz(cx, cy, scenario_powers[s])
            if gnorms[s] > worst:
                worst, worst_i = gnorms[s], int(s)
        pen = wasserstein_cvar_penalty(peaks, gnorms, self.eps_dro, self.alpha)
        return pen, worst_i

    def dro_term(self, scenario_powers: list[np.ndarray]) -> float:
        """Worst-case-CVaR DRO penalty at the current placement (0 if eps_dro<=0)."""
        if self.eps_dro <= 0.0:
            return 0.0
        cx, cy = self.get_positions()
        peaks = self._scenario_peaks(cx, cy, scenario_powers)
        return self._dro_penalty_at(cx, cy, peaks, scenario_powers)[0]

    def dro_exact_term(self) -> float:
        """Certified global-Lambda penalty eps/(1-alpha)*max_i ||D a_i(p)|| at the
        current placement (scenario-independent; 0 if eps_dro<=0)."""
        if self.eps_dro <= 0.0:
            return 0.0
        if self._lam_rows is None:
            self._lam_rows = si_adjoint_rows(self.solver)
        cx, cy = self.get_positions()
        self.solver.units = _make_units(self.units_orig, cx, cy)
        Lam, _, _ = exact_penalty_terms(self.solver, self._lam_rows, self.dro_sigma)
        return self.eps_dro / (1.0 - self.alpha) * Lam

    def tail_sensitivity(self, scenario_powers: list[np.ndarray]) -> float:
        """Worst-case ||d(peak)/d(power)|| over the CVaR tail (K/W), independent of
        eps_dro. This is the Lambda the DRO penalty scales."""
        cx, cy = self.get_positions()
        peaks = self._scenario_peaks(cx, cy, scenario_powers)
        q = np.quantile(peaks, self.alpha)
        tail = np.where(peaks >= q)[0]
        return max(self._dro_lipschitz(cx, cy, scenario_powers[s]) for s in tail)

    # Objective + gradient

    def objective_and_grad(self, scenario_powers: list[np.ndarray],
                           mode: str = "dro",
                           weights: np.ndarray | None = None,
                           offset: tuple[float, float] = (0.0, 0.0)
                           ) -> tuple[float, np.ndarray, np.ndarray]:
        """Objective and (grad_raw_x, grad_raw_y) for mode in
        {'mean','cvar','blend','dro','dro_exact'}.

        scenario_powers : list of arrays shape (n_blocks,) [W].
        weights         : optional per-scenario probability weights (all modes
                          except 'dro'). The adjoint per scenario is unchanged;
                          only the outer aggregation over scenarios is weighted.
        """
        if weights is not None and mode == "dro":
            raise ValueError("weighted scenarios are not supported for mode='dro'")
        solver = self.solver
        cx, cy = self.get_positions()
        # Rasterization jitter regularizes against training-grid overfitting: a
        # rigid sub-cell shift of the whole floorplan relative to the grid.
        # Gradients are w.r.t. the shifted positions, which equal d/d(cx) since
        # the offset is constant. Edge blocks may lose a sliver of power
        # off-grid; acceptable as training noise only.
        ox, oy = offset
        if ox != 0.0 or oy != 0.0:
            cx = cx + ox
            cy = cy + oy
        N_scen = len(scenario_powers)
        T_amb = solver.cfg["ambient"]
        N = solver.N

        # d(centre)/d(raw): sigmoid derivative times the bound span.
        sx = sigmoid_vec(self.raw_x)
        sy = sigmoid_vec(self.raw_y)
        dcx_draw = sx * (1.0 - sx) * (self.cx_max - self.cx_min)
        dcy_draw = sy * (1.0 - sy) * (self.cy_max - self.cy_min)

        # Solve each scenario; record peak dT and the peak Si cell. (The full
        # field is not needed later: the adjoint of a linear solve depends only
        # on dL/dT, not on T.)
        peaks = np.zeros(N_scen)
        peak_si_flat = np.zeros(N_scen, dtype=int)
        for s, pw in enumerate(scenario_powers):
            T = solver.solve(_build_rhs_at(solver, self.units_orig, cx, cy, pw))
            T_si = solver.silicon_layer(T)
            idx = int(np.argmax(T_si))
            peaks[s] = float(T_si.flat[idx]) - T_amb
            peak_si_flat[s] = idx

        # Objective term and d(obj)/d(peaks).
        if weights is None:
            mean_val, d_mean = float(peaks.mean()), np.full(N_scen, 1.0 / N_scen)
        else:
            w = np.asarray(weights, dtype=float)
            w = w / w.sum()
            mean_val, d_mean = float((w * peaks).sum()), w
        if mode == "mean":
            obj_t, d_peaks = mean_val, d_mean
        elif mode == "blend":
            g = self.blend_gamma
            c_val, d_c = cvar_and_grad(peaks, self.alpha, weights=weights)
            obj_t = (1.0 - g) * mean_val + g * c_val
            d_peaks = (1.0 - g) * d_mean + g * d_c
        else:  # cvar, dro, dro_exact
            obj_t, d_peaks = cvar_and_grad(peaks, self.alpha, weights=weights)

        L_pen = 0.0
        do_dro = mode == "dro" and self.eps_dro > 0.0
        if do_dro:
            L_pen, worst_i = self._dro_penalty_at(cx, cy, peaks, scenario_powers)
            dro_pw = scenario_powers[worst_i]
            dro_w = self.eps_dro / (1.0 - self.alpha)   # penalty = eps/(1-alpha) * Lambda
        do_dxx = mode == "dro_exact" and self.eps_dro > 0.0
        if do_dxx:
            if self._lam_rows is None:
                self._lam_rows = si_adjoint_rows(solver)
            solver.units = _make_units(self.units_orig, cx, cy)
            Lam, i_star, a_star = exact_penalty_terms(solver, self._lam_rows,
                                                      self.dro_sigma)
            dxx_w = self.eps_dro / (1.0 - self.alpha)   # penalty = eps/(1-alpha) * Lambda
            L_pen = dxx_w * Lam

        # Thermal gradient via adjoint: d(obj)/d(c_b) = sum_s d_peaks[s] * lam_s^T dQ_s/d(c_b).
        # WARNING: the per-scenario peak cell (peak_si_flat) and the CVaR tail
        # mask are both frozen here (envelope theorem). The gradient is exact
        # only away from argmax/tail-boundary kinks; near a kink it is a valid
        # subgradient of one branch, not the true derivative.
        g_cx = np.zeros(self.n)
        g_cy = np.zeros(self.n)
        solver.units = _make_units(self.units_orig, cx, cy)
        for s in range(N_scen):
            if abs(d_peaks[s]) < 1e-14:
                continue
            pi, pj = divmod(int(peak_si_flat[s]), self.nc)
            # adjoint: G^T lam = e_{i*}
            dL_dT = np.zeros(N)
            dL_dT[solver._nidx(solver.SI, pi, pj)] = 1.0
            lam = solver.adjoint_dT_dQ(dL_dT)

            bp = {u["name"]: float(scenario_powers[s][b]) for b, u in enumerate(self.units_orig)}
            dcx, dcy = solver.rhs_position_grad(lam, bp)
            scale = d_peaks[s]
            for b, u in enumerate(self.units_orig):
                g_cx[b] += scale * dcx[u["name"]]
                g_cy[b] += scale * dcy[u["name"]]

        # Exact-dual penalty gradient (analytic): penalty = eps/(1-alpha) *
        # ||D a_{i*}(p)||_2 with i* frozen (envelope), a_i = A(p)^T lam_i.
        # Component b of a_{i*} depends on positions only through block b's own
        # overlap column, and its derivative is rhs_position_grad at unit power.
        if do_dxx and Lam > 0.0:
            s2 = np.ones(self.n) if self.dro_sigma is None else self.dro_sigma ** 2
            coef = dxx_w * (s2 * a_star) / Lam
            ones = {u["name"]: 1.0 for u in self.units_orig}
            dcx_a, dcy_a = solver.rhs_position_grad(self._lam_rows[i_star], ones)
            for b, u in enumerate(self.units_orig):
                g_cx[b] += coef[b] * dcx_a[u["name"]]
                g_cy[b] += coef[b] * dcy_a[u["name"]]

        # DRO term gradient: central FD of the worst-tail sensitivity scalar. The
        # tail set / worst scenario are frozen (envelope theorem), as cvar_and_grad
        # freezes the tail mask.
        if do_dro:
            dL = self.fd_delta_dro
            for b in range(self.n):
                cxp = cx.copy(); cxp[b] += dL
                cxm = cx.copy(); cxm[b] -= dL
                g_cx[b] += dro_w * (self._dro_lipschitz(cxp, cy, dro_pw)
                                    - self._dro_lipschitz(cxm, cy, dro_pw)) / (2.0 * dL)
                cyp = cy.copy(); cyp[b] += dL
                cym = cy.copy(); cym[b] -= dL
                g_cy[b] += dro_w * (self._dro_lipschitz(cx, cyp, dro_pw)
                                    - self._dro_lipschitz(cx, cym, dro_pw)) / (2.0 * dL)

        # Wirelength term (Phase 3): wl_weight * smooth-HPWL over the netlist.
        # HPWL is translation-invariant, so the rasterization offset (a rigid
        # shift) does not affect it; its gradient adds directly to the centres.
        L_wl = 0.0
        if self.wl_weight > 0.0 and self.nets:
            wl_val, g_cx_wl, g_cy_wl = smooth_hpwl_grad(cx, cy, self.nets, self.wl_gamma)
            L_wl = self.wl_weight * wl_val
            g_cx += self.wl_weight * g_cx_wl
            g_cy += self.wl_weight * g_cy_wl

        # Soft non-overlap penalty (shared with objectives.overlap).
        pen, gcx_no, gcy_no = nonoverlap_penalty(cx, cy, self.widths, self.heights)
        obj = obj_t + L_pen + L_wl + self.nonoverlap_w * pen

        # Chain rule to the raw (pre-sigmoid) parameters.
        g_rx = (g_cx + self.nonoverlap_w * gcx_no) * dcx_draw
        g_ry = (g_cy + self.nonoverlap_w * gcy_no) * dcy_draw

        solver.units = _make_units(self.units_orig, cx, cy)   # leave solver consistent
        return obj, g_rx, g_ry

    # Optimiser

    def optimize(self, scenario_powers: list[np.ndarray], mode: str = "dro",
                 n_iter: int = 150, lr: float = 1e-2, verbose: bool = True,
                 weights: np.ndarray | None = None,
                 raster_jitter: float = 0.0, jitter_seed: int = 0) -> list[float]:
        """Adam optimiser. Returns the per-iteration objective history.

        raster_jitter > 0 draws a fresh rigid sub-cell offset each iteration
        (uniform in +-raster_jitter/2 cells) so placements cannot lock onto
        the rasterization grid, regularizing against training-grid
        overfitting. Default 0.0 reproduces the historical optimiser exactly.
        """
        beta1, beta2, eps_a = 0.9, 0.999, 1e-8
        n = self.n
        m_rx = np.zeros(n); v_rx = np.zeros(n)
        m_ry = np.zeros(n); v_ry = np.zeros(n)
        history = []
        jrng = np.random.default_rng(jitter_seed) if raster_jitter > 0 else None
        cw = self.chip_w / self.nc
        ch = self.chip_h / self.nr

        for it in range(1, n_iter + 1):
            off = ((float(jrng.uniform(-0.5, 0.5)) * cw * raster_jitter,
                    float(jrng.uniform(-0.5, 0.5)) * ch * raster_jitter)
                   if jrng is not None else (0.0, 0.0))
            obj, g_rx, g_ry = self.objective_and_grad(scenario_powers, mode,
                                                      weights=weights, offset=off)
            history.append(obj)

            m_rx = beta1 * m_rx + (1 - beta1) * g_rx
            v_rx = beta2 * v_rx + (1 - beta2) * g_rx ** 2
            m_ry = beta1 * m_ry + (1 - beta1) * g_ry
            v_ry = beta2 * v_ry + (1 - beta2) * g_ry ** 2

            bc1 = 1 - beta1 ** it
            bc2 = 1 - beta2 ** it
            self.raw_x -= lr * (m_rx / bc1) / (np.sqrt(v_rx / bc2) + eps_a)
            self.raw_y -= lr * (m_ry / bc1) / (np.sqrt(v_ry / bc2) + eps_a)

            if verbose and (it == 1 or it % 25 == 0 or it == n_iter):
                cx, cy = self.get_positions()
                pen, _, _ = nonoverlap_penalty(cx, cy, self.widths, self.heights)
                print(f"  iter {it:4d}  obj={obj:.4f}  overlap_pen={pen:.4e}")

        return history
