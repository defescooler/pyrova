"""Differentiable floorplanner: Adam over sigmoid-bounded macro centres,
with thermal gradients from the adjoint solve."""

from __future__ import annotations
import copy
import math
import numpy as np
import sys, os

from pyrova.thermal.fd_solver import GridFDSolver, parse_flp, parse_config, chip_dimensions
from pyrova.objectives.dro import wasserstein_cvar_penalty


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_units(units_orig: list[dict],
                cx: np.ndarray, cy: np.ndarray) -> list[dict]:
    """Build units list with updated centre positions."""
    out = []
    for b, u in enumerate(units_orig):
        out.append({
            'name':    u['name'],
            'width':   u['width'],
            'height':  u['height'],
            'leftx':   float(cx[b]) - u['width']  / 2.0,
            'bottomy': float(cy[b]) - u['height'] / 2.0,
        })
    return out


def _build_rhs_at(solver: GridFDSolver,
                  units_orig: list[dict],
                  cx: np.ndarray, cy: np.ndarray,
                  block_powers: np.ndarray) -> np.ndarray:
    """Build RHS with solver.units set to (cx, cy) positions."""
    solver.units = _make_units(units_orig, cx, cy)
    bp = {u['name']: float(block_powers[b])
          for b, u in enumerate(units_orig)}
    return solver.build_rhs(bp)


def sigmoid_vec(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def nonoverlap_penalty(cx: np.ndarray, cy: np.ndarray,
                       widths: np.ndarray, heights: np.ndarray,
                       gap: float = 0.0) -> tuple[float, np.ndarray, np.ndarray]:
    """Soft non-overlap penalty and gradient."""
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


def cvar_and_grad(peaks: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """CVaR_alpha and gradient w.r.t. each element."""
    q = np.quantile(peaks, alpha)
    mask = peaks >= q
    count = max(1, int(mask.sum()))
    val = float(peaks[mask].mean())
    grad = np.where(mask, 1.0 / count, 0.0)
    return val, grad


# ---------------------------------------------------------------------------
# Placer
# ---------------------------------------------------------------------------

class DiffPlacer:
    """
    Parameters
    ----------
    solver       : GridFDSolver (built + factorized)
    units_orig   : original block list
    chip_w/h     : chip dimensions [m]
    nr, nc       : thermal grid resolution
    alpha        : CVaR tail probability (default 0.9)
    eps_dro      : Wasserstein ball radius for the worst-case-CVaR penalty (default 0)
    nonoverlap_w : soft non-overlap penalty weight
    fd_delta     : finite-difference step for dQ/d(cx/cy) [m]
    """

    def __init__(self, solver: GridFDSolver,
                 units_orig: list[dict],
                 chip_w: float, chip_h: float,
                 nr: int, nc: int,
                 alpha: float = 0.9,
                 eps_dro: float = 0.0,
                 nonoverlap_w: float = 1e4,
                 fd_delta: float = 1e-7,
                 fd_delta_dro: float = 1e-6):
        self.solver = solver
        self.units_orig = units_orig
        self.chip_w = chip_w
        self.chip_h = chip_h
        self.nr = nr
        self.nc = nc
        self.alpha = alpha
        self.eps_dro = eps_dro
        self.nonoverlap_w = nonoverlap_w
        self.fd_delta = fd_delta
        self.fd_delta_dro = fd_delta_dro

        n = len(units_orig)
        self.n = n
        self.widths  = np.array([u['width']  for u in units_orig])
        self.heights = np.array([u['height'] for u in units_orig])

        # Centre bounds: block must fit inside chip
        self.cx_min = self.widths  / 2.0
        self.cx_max = chip_w - self.widths  / 2.0
        self.cy_min = self.heights / 2.0
        self.cy_max = chip_h - self.heights / 2.0

        self.raw_x, self.raw_y = self._encode_original()

    def _encode_original(self) -> tuple[np.ndarray, np.ndarray]:
        cx0 = np.array([u['leftx']   + u['width']  / 2.0 for u in self.units_orig])
        cy0 = np.array([u['bottomy'] + u['height'] / 2.0 for u in self.units_orig])
        cx0 = np.clip(cx0, self.cx_min + 1e-9, self.cx_max - 1e-9)
        cy0 = np.clip(cy0, self.cy_min + 1e-9, self.cy_max - 1e-9)
        fx = np.clip((cx0 - self.cx_min) / (self.cx_max - self.cx_min + 1e-30),
                     1e-6, 1.0 - 1e-6)
        fy = np.clip((cy0 - self.cy_min) / (self.cy_max - self.cy_min + 1e-30),
                     1e-6, 1.0 - 1e-6)
        return np.log(fx / (1.0 - fx)), np.log(fy / (1.0 - fy))

    def get_positions(self) -> tuple[np.ndarray, np.ndarray]:
        sx = sigmoid_vec(self.raw_x)
        sy = sigmoid_vec(self.raw_y)
        cx = self.cx_min + (self.cx_max - self.cx_min) * sx
        cy = self.cy_min + (self.cy_max - self.cy_min) * sy
        return cx, cy

    def get_units(self) -> list[dict]:
        cx, cy = self.get_positions()
        return _make_units(self.units_orig, cx, cy)

    def reset(self):
        self.raw_x, self.raw_y = self._encode_original()

    # ------------------------------------------------------------------
    def _dro_lipschitz(self, cx: np.ndarray, cy: np.ndarray,
                       pw: np.ndarray) -> float:
        """Power-sensitivity norm ||d(peak)/d(power)|| at a given scenario power."""
        solver = self.solver
        solver.units = _make_units(self.units_orig, cx, cy)
        bp = {u['name']: float(pw[b]) for b, u in enumerate(self.units_orig)}
        _, g_dict = solver.peak_T_gradient(bp)
        g = np.array([g_dict.get(u['name'], 0.0) for u in self.units_orig])
        return float(np.linalg.norm(g))

    def dro_term(self, scenario_powers: list[np.ndarray]) -> float:
        """Worst-case-CVaR DRO penalty at the current placement (0 if eps_dro<=0)."""
        if self.eps_dro <= 0.0:
            return 0.0
        cx, cy = self.get_positions()
        peaks = self._scenario_peaks(cx, cy, scenario_powers)
        return self._dro_penalty_at(cx, cy, peaks, scenario_powers)[0]

    def tail_sensitivity(self, scenario_powers: list[np.ndarray]) -> float:
        """Worst-case power-sensitivity ||d(peak)/d(power)|| over the CVaR tail (K/W),
        independent of eps_dro. This is the Lambda the DRO penalty scales."""
        cx, cy = self.get_positions()
        peaks = self._scenario_peaks(cx, cy, scenario_powers)
        q = np.quantile(peaks, self.alpha)
        tail = np.where(peaks >= q)[0]
        return max(self._dro_lipschitz(cx, cy, scenario_powers[s]) for s in tail)

    def _scenario_peaks(self, cx, cy, scenario_powers) -> np.ndarray:
        """Per-scenario peak dT at the given placement."""
        solver = self.solver
        T_amb = solver.cfg['ambient']
        peaks = np.zeros(len(scenario_powers))
        for s, pw in enumerate(scenario_powers):
            T = solver.solve(_build_rhs_at(solver, self.units_orig, cx, cy, pw))
            peaks[s] = float(solver.silicon_layer(T).max()) - T_amb
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

    # ------------------------------------------------------------------
    def objective_and_grad(self, scenario_powers: list[np.ndarray],
                           mode: str = 'dro') -> tuple[float, np.ndarray, np.ndarray]:
        """
        Compute objective and (g_raw_x, g_raw_y).

        scenario_powers : list of N_scen arrays of shape (n_blocks,) [W]
        mode            : 'mean' | 'cvar' | 'dro'
        Returns (obj, grad_raw_x, grad_raw_y).
        """
        solver = self.solver
        cx, cy = self.get_positions()
        N_scen = len(scenario_powers)
        T_amb = solver.cfg['ambient']
        N = solver.N

        # Sigmoid derivatives: d(cx_b)/d(raw_x_b)
        sx = sigmoid_vec(self.raw_x)
        sy = sigmoid_vec(self.raw_y)
        dcx_draw = sx * (1.0 - sx) * (self.cx_max - self.cx_min)  # shape (n,)
        dcy_draw = sy * (1.0 - sy) * (self.cy_max - self.cy_min)

        # Solve thermal for each scenario
        peaks = np.zeros(N_scen)
        peak_si_flat = np.zeros(N_scen, dtype=int)
        Ts = []

        for s, pw in enumerate(scenario_powers):
            Q = _build_rhs_at(solver, self.units_orig, cx, cy, pw)
            T = solver.solve(Q)
            T_si = solver.silicon_layer(T)
            idx = int(np.argmax(T_si))
            peaks[s] = float(T_si.flat[idx]) - T_amb
            peak_si_flat[s] = idx
            Ts.append(T)

        # Objective and d(obj)/d(peaks)
        if mode == 'mean':
            obj_t = float(peaks.mean())
            d_peaks = np.full(N_scen, 1.0 / N_scen)
        else:  # cvar or dro
            obj_t, d_peaks = cvar_and_grad(peaks, self.alpha)

        # Worst-case-CVaR DRO penalty: eps/(1-alpha) times the largest
        # power-sensitivity over the CVaR tail (the exact type-1 Wasserstein
        # dual for this piecewise-linear loss). Gradient added below, after the
        # per-scenario thermal gradient is assembled.
        L_pen = 0.0
        do_dro = mode == 'dro' and self.eps_dro > 0.0
        if do_dro:
            L_pen, worst_i = self._dro_penalty_at(cx, cy, peaks, scenario_powers)
            dro_pw = scenario_powers[worst_i]
            dro_w = self.eps_dro / (1.0 - self.alpha)

        # Thermal gradient via adjoint: dobj/d(cx_b) = sum_s d_peaks[s] x lambda_s^T x (dQ_s/d(cx_b)),
        # with dQ_s/d(pos) computed analytically (solver.rhs_position_grad).
        g_cx = np.zeros(self.n)
        g_cy = np.zeros(self.n)
        solver.units = _make_units(self.units_orig, cx, cy)

        for s in range(N_scen):
            if abs(d_peaks[s]) < 1e-14:
                continue
            T = Ts[s]
            pw = scenario_powers[s]
            pi, pj = divmod(int(peak_si_flat[s]), self.nc)

            dL_dT = np.zeros(N)
            dL_dT[solver._nidx(solver.SI, pi, pj)] = 1.0
            lam = solver.adjoint_dT_dQ(T, dL_dT)  # shape (N,)

            bp = {u['name']: float(pw[b]) for b, u in enumerate(self.units_orig)}
            dcx, dcy = solver.rhs_position_grad(lam, bp)
            scale = d_peaks[s]
            for b, u in enumerate(self.units_orig):
                g_cx[b] += scale * dcx[u['name']]
                g_cy[b] += scale * dcy[u['name']]

        # DRO term gradient: d/dp [eps/(1-alpha) * ||dT_peak/dP||] for the
        # worst-case tail scenario, via central FD on the sensitivity scalar.
        # The tail set and worst scenario are held fixed (envelope theorem),
        # matching how cvar_and_grad freezes the tail mask.
        if do_dro:
            dL = self.fd_delta_dro
            for b in range(self.n):
                cxp = cx.copy(); cxp[b] += dL
                cxm = cx.copy(); cxm[b] -= dL
                g_cx[b] += dro_w * (
                    self._dro_lipschitz(cxp, cy, dro_pw)
                    - self._dro_lipschitz(cxm, cy, dro_pw)) / (2.0 * dL)
                cyp = cy.copy(); cyp[b] += dL
                cym = cy.copy(); cym[b] -= dL
                g_cy[b] += dro_w * (
                    self._dro_lipschitz(cx, cyp, dro_pw)
                    - self._dro_lipschitz(cx, cym, dro_pw)) / (2.0 * dL)

        # Non-overlap penalty
        pen, gcx_no, gcy_no = nonoverlap_penalty(cx, cy, self.widths, self.heights)
        obj = obj_t + L_pen + self.nonoverlap_w * pen

        # Chain rule: d(raw_x) = d(cx) x d(cx)/d(raw_x)
        g_rx = (g_cx + self.nonoverlap_w * gcx_no) * dcx_draw
        g_ry = (g_cy + self.nonoverlap_w * gcy_no) * dcy_draw

        # Restore solver.units
        solver.units = _make_units(self.units_orig, cx, cy)
        return obj, g_rx, g_ry

    # ------------------------------------------------------------------
    def optimize(self, scenario_powers: list[np.ndarray],
                 mode: str = 'dro',
                 n_iter: int = 150,
                 lr: float = 1e-2,
                 verbose: bool = True) -> list[float]:
        """Adam optimiser. Returns per-iteration objective history."""
        beta1, beta2, eps_a = 0.9, 0.999, 1e-8
        n = self.n
        m_rx = np.zeros(n); v_rx = np.zeros(n)
        m_ry = np.zeros(n); v_ry = np.zeros(n)
        history = []

        for it in range(1, n_iter + 1):
            obj, g_rx, g_ry = self.objective_and_grad(scenario_powers, mode)
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
                print(f'  iter {it:4d}  obj={obj:.4f}  overlap_pen={pen:.4e}')

        return history
