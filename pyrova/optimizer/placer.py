"""Differentiable floorplanner: Adam over sigmoid-bounded macro centres, driven by adjoint thermal gradients."""

from __future__ import annotations

import numpy as np

from pyrova.thermal.fd_solver import GridFDSolver
from pyrova.objectives.overlap import nonoverlap_penalty
from pyrova.objectives.wirelength import smooth_hpwl_grad
from pyrova.objectives.density import density_penalty, density_weight_ramp


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
    """CVaR_alpha and its frozen-tail-mask gradient w.r.t. each peak; returns (CVaR, grad)."""
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
    """Adam floorplanner over sigmoid-bounded macro centres; nonoverlap_w is in area units (m^2) and must rescale with chip/block dimensions."""

    def __init__(self, solver: GridFDSolver, units_orig: list[dict],
                 chip_w: float, chip_h: float, nr: int, nc: int,
                 alpha: float = 0.9, blend_gamma: float = 0.5,
                 nonoverlap_w: float = 1e4,
                 nets: list | None = None, wl_weight: float = 0.0,
                 wl_gamma: float | None = None,
                 density_w: float = 0.0, density_lam0: float = 0.0,
                 density_grid: tuple[int, int] | None = None,
                 density_t: float = 1.0):
        self.solver = solver
        self.units_orig = units_orig
        self.chip_w = chip_w
        self.chip_h = chip_h
        self.nr = nr
        self.nc = nc
        self.alpha = alpha
        self.blend_gamma = blend_gamma
        self.nonoverlap_w = nonoverlap_w

        # `nets`: list of nets, each a list of macro indices into units_orig.
        # wl_weight>0 adds wl_weight*smoothHPWL to every mode's objective, turning
        # the unconstrained thermal problem into the wirelength-penalised one.
        # wl_gamma (smoothing length) defaults to 1% of the mean chip span.
        self.nets = [np.asarray(net, dtype=int) for net in nets] if nets else []
        self.wl_weight = wl_weight
        self.wl_gamma = (wl_gamma if wl_gamma is not None
                         else 0.01 * 0.5 * (chip_w + chip_h))

        # Bin-overflow spreading term for legality. density_w>0 is the peak weight
        # of the penalty, ramped from density_lam0 over the optimiser iterations.
        # It supplements — never replaces — the pairwise non-overlap contact term,
        # which remains the sub-bin symmetry-breaker. density_grid defaults to the
        # thermal grid; a coarse grid has a sub-bin blind spot, so under wirelength
        # pressure prefer a grid whose bins are no larger than the smallest macro.
        # density_w is in K-equivalent units and, like nonoverlap_w, must be
        # re-scaled if the chip/block dimensions or the density grid change.
        self.density_w = density_w
        self.density_lam0 = density_lam0
        self.density_grid = density_grid
        self.density_t = density_t

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
        """Current HPWL over self.nets [m]: exact bounding-box (default) or smooth surrogate; 0.0 with no nets."""
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

    def _scenario_peaks(self, cx, cy, scenario_powers) -> np.ndarray:
        """Per-scenario peak dT at the given placement."""
        T_amb = self.solver.cfg["ambient"]
        peaks = np.zeros(len(scenario_powers))
        for s, pw in enumerate(scenario_powers):
            T = self.solver.solve(_build_rhs_at(self.solver, self.units_orig, cx, cy, pw))
            peaks[s] = float(self.solver.silicon_layer(T).max()) - T_amb
        return peaks

    # Objective + gradient

    def objective_and_grad(self, scenario_powers: list[np.ndarray],
                           mode: str = "cvar",
                           weights: np.ndarray | None = None,
                           offset: tuple[float, float] = (0.0, 0.0),
                           density_lambda: float | None = None
                           ) -> tuple[float, np.ndarray, np.ndarray]:
        """Objective and gradients w.r.t. the raw pre-sigmoid parameters; returns (obj, grad_raw_x, grad_raw_y)."""
        solver = self.solver
        cx, cy = self.get_positions()
        # Rasterisation jitter regularises against training-grid overfitting: a
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
        elif mode == "hpwl":                 # wirelength-only baseline (no thermal term)
            obj_t, d_peaks = 0.0, np.zeros(N_scen)
        elif mode == "blend":
            g = self.blend_gamma
            c_val, d_c = cvar_and_grad(peaks, self.alpha, weights=weights)
            obj_t = (1.0 - g) * mean_val + g * c_val
            d_peaks = (1.0 - g) * d_mean + g * d_c
        else:  # cvar
            obj_t, d_peaks = cvar_and_grad(peaks, self.alpha, weights=weights)

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
            lam = solver.adjoint_solve(dL_dT)

            bp = {u["name"]: float(scenario_powers[s][b]) for b, u in enumerate(self.units_orig)}
            dcx, dcy = solver.rhs_position_grad(lam, bp)
            scale = d_peaks[s]
            for b, u in enumerate(self.units_orig):
                g_cx[b] += scale * dcx[u["name"]]
                g_cy[b] += scale * dcy[u["name"]]

        # Wirelength term: wl_weight * smooth-HPWL over the netlist. HPWL is
        # translation-invariant, so the rasterisation offset (a rigid shift) does
        # not affect it; its gradient adds directly to the centres.
        L_wl = 0.0
        if self.wl_weight > 0.0 and self.nets:
            wl_val, g_cx_wl, g_cy_wl = smooth_hpwl_grad(cx, cy, self.nets, self.wl_gamma)
            L_wl = self.wl_weight * wl_val
            g_cx += self.wl_weight * g_cx_wl
            g_cy += self.wl_weight * g_cy_wl

        # Density spreading term: a global bin-overflow field on the same jittered
        # centres as the thermal solve, weighted by the ramped weight supplied per
        # iteration. Graded (weak early / strong late) so it never freezes the
        # placer the way a stiff pairwise weight would.
        L_dens = 0.0
        lamD = self.density_w if density_lambda is None else density_lambda
        if lamD > 0.0:
            ndr, ndc = self.density_grid or (self.nr, self.nc)
            D_val, g_cx_d, g_cy_d = density_penalty(
                cx, cy, self.widths, self.heights, self.chip_w, self.chip_h,
                ndr, ndc, t=self.density_t)
            L_dens = lamD * D_val
            g_cx += lamD * g_cx_d
            g_cy += lamD * g_cy_d

        # Soft non-overlap penalty (shared with objectives.overlap).
        pen, gcx_no, gcy_no = nonoverlap_penalty(cx, cy, self.widths, self.heights)
        obj = obj_t + L_wl + L_dens + self.nonoverlap_w * pen

        # Chain rule to the raw (pre-sigmoid) parameters.
        g_rx = (g_cx + self.nonoverlap_w * gcx_no) * dcx_draw
        g_ry = (g_cy + self.nonoverlap_w * gcy_no) * dcy_draw

        solver.units = _make_units(self.units_orig, cx, cy)   # leave solver consistent
        return obj, g_rx, g_ry

    # Optimiser

    def optimize(self, scenario_powers: list[np.ndarray], mode: str = "cvar",
                 n_iter: int = 150, lr: float = 1e-2, verbose: bool = True,
                 weights: np.ndarray | None = None,
                 raster_jitter: float = 0.0, jitter_seed: int = 0,
                 callback=None) -> list[float]:
        """Adam optimiser; returns the per-iteration objective history."""
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
            lamD_it = density_weight_ramp(it, n_iter, self.density_lam0, self.density_w)
            obj, g_rx, g_ry = self.objective_and_grad(scenario_powers, mode,
                                                      weights=weights, offset=off,
                                                      density_lambda=lamD_it)
            history.append(obj)

            m_rx = beta1 * m_rx + (1 - beta1) * g_rx
            v_rx = beta2 * v_rx + (1 - beta2) * g_rx ** 2
            m_ry = beta1 * m_ry + (1 - beta1) * g_ry
            v_ry = beta2 * v_ry + (1 - beta2) * g_ry ** 2

            bc1 = 1 - beta1 ** it
            bc2 = 1 - beta2 ** it
            self.raw_x -= lr * (m_rx / bc1) / (np.sqrt(v_rx / bc2) + eps_a)
            self.raw_y -= lr * (m_ry / bc1) / (np.sqrt(v_ry / bc2) + eps_a)

            if callback is not None:                 # per-iteration state for viz/logging
                callback(it, self, obj)

            if verbose and (it == 1 or it % 25 == 0 or it == n_iter):
                cx, cy = self.get_positions()
                pen, _, _ = nonoverlap_penalty(cx, cy, self.widths, self.heights)
                extra = ""
                if self.density_w > 0.0:
                    ndr, ndc = self.density_grid or (self.nr, self.nc)
                    d_over = density_penalty(cx, cy, self.widths, self.heights,
                                             self.chip_w, self.chip_h, ndr, ndc,
                                             t=self.density_t)[0]
                    extra = f"  density_over={d_over:.4e}  lamD={lamD_it:.3e}"
                print(f"  iter {it:4d}  obj={obj:.4f}  overlap_pen={pen:.4e}{extra}")

        return history
