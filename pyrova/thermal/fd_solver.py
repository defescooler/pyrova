"""Steady-state finite-difference thermal solver for a 4-layer grid
model (Si/TIM/Spreader/Heatsink) with adjoint gradients."""

from __future__ import annotations
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized, spsolve


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def parse_flp(path: str) -> list[dict]:
    """Parse a .flp floorplan file. Returns list of unit dicts."""
    units = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            units.append({
                'name': parts[0],
                'width': float(parts[1]),
                'height': float(parts[2]),
                'leftx': float(parts[3]),
                'bottomy': float(parts[4]),
            })
    return units


def parse_ptrace(path: str) -> tuple[list[str], list[list[float]]]:
    """Parse a .ptrace power-trace file. Returns (block_names, list_of_power_rows)."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    block_names = lines[0].split('\t')
    rows = [[float(v) for v in ln.split('\t')] for ln in lines[1:]]
    return block_names, rows


def parse_config(path: str) -> dict:
    """Parse a .config file into a dict of {key: value}."""
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Tokens: -key value  (comments after #)
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith('-'):
                key = parts[0][1:]
                try:
                    val = float(parts[1])
                except ValueError:
                    val = parts[1]
                cfg[key] = val
    return cfg


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def getr(k: float, length: float, area: float) -> float:
    """Thermal resistance R = length / (k * area)."""
    return length / (k * area)


def chip_dimensions(units: list[dict]) -> tuple[float, float]:
    """Infer chip W and H from the flp bounding box."""
    max_x = max(u['leftx'] + u['width'] for u in units)
    max_y = max(u['bottomy'] + u['height'] for u in units)
    min_x = min(u['leftx'] for u in units)
    min_y = min(u['bottomy'] for u in units)
    return max_x - min_x, max_y - min_y


# ---------------------------------------------------------------------------
# Package-node index constants
# ---------------------------------------------------------------------------
SP_W, SP_E, SP_N, SP_S = 0, 1, 2, 3
SINK_C_W, SINK_C_E, SINK_C_N, SINK_C_S = 4, 5, 6, 7
SINK_W, SINK_E, SINK_N, SINK_S = 8, 9, 10, 11
EXTRA = 12


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

class GridFDSolver:
    """
    4-layer grid model (Si/TIM/Spreader/Heatsink) with a 5-point
    conductance stencil and 12 package peripheral nodes.

    Layers (low->high):
      0 = Silicon    (power-dissipating)
      1 = TIM        (interface material)
      2 = Spreader
      3 = Heatsink   (convective BC to ambient)
    """

    SI, TIM, SP, HS = 0, 1, 2, 3

    def __init__(self, cfg: dict, units: list[dict], chip_w: float,
                 chip_h: float, nr: int, nc: int):
        self.cfg = cfg
        self.units = units
        self.chip_w = chip_w
        self.chip_h = chip_h
        self.nr = nr
        self.nc = nc
        self.nl = 4
        self.N_grid = self.nl * nr * nc
        self.N = self.N_grid + EXTRA

        cw = chip_w / nc
        ch = chip_h / nr
        self.cw = cw
        self.ch = ch

        self._init_layer_params()
        self._init_package_params()

        self.G = None
        self._factor = None

    # ------------------------------------------------------------------
    def _init_layer_params(self):
        cfg = self.cfg
        cw, ch = self.cw, self.ch
        k  = [cfg['k_chip'], cfg['k_interface'], cfg['k_spreader'], cfg['k_sink']]
        t  = [cfg['t_chip'], cfg['t_interface'], cfg['t_spreader'], cfg['t_sink']]
        self.k_layer = k
        self.t_layer = t

        self.rx = [getr(k[l], cw,  ch  * t[l]) for l in range(4)]
        self.ry = [getr(k[l], ch,  cw  * t[l]) for l in range(4)]
        self.rz = [getr(k[l], t[l], cw * ch)   for l in range(4)]

        # Heatsink: add convective resistance per cell
        self.rz[self.HS] += (cfg['r_convec'] * cfg['s_sink'] ** 2 / (cw * ch))

    # ------------------------------------------------------------------
    def _init_package_params(self):
        cfg = self.cfg
        W, H = self.chip_w, self.chip_h
        k_sp = cfg['k_spreader'];  t_sp = cfg['t_spreader'];  s_sp = cfg['s_spreader']
        k_hs = cfg['k_sink'];      t_hs = cfg['t_sink'];      s_hs = cfg['s_sink']
        r_cv = cfg['r_convec']

        pk = {}
        pk['r_sp1_x']   = getr(k_sp, (s_sp - W) / 4, (s_sp + 3*H) / 4 * t_sp)
        pk['r_sp1_y']   = getr(k_sp, (s_sp - H) / 4, (s_sp + 3*W) / 4 * t_sp)
        pk['r_hs1_x']   = getr(k_hs, (s_sp - W) / 4, (s_sp + 3*H) / 4 * t_hs)
        pk['r_hs1_y']   = getr(k_hs, (s_sp - H) / 4, (s_sp + 3*W) / 4 * t_hs)
        pk['r_hs2_x']   = getr(k_hs, (s_sp - W) / 4, (3*s_sp + H) / 4 * t_hs)
        pk['r_hs2_y']   = getr(k_hs, (s_sp - H) / 4, (3*s_sp + W) / 4 * t_hs)
        pk['r_hs']      = getr(k_hs, (s_hs - s_sp) / 4, (s_hs + 3*s_sp) / 4 * t_hs)
        pk['r_sp_per_x']    = getr(k_sp, t_sp, (s_sp + H) * (s_sp - W) / 4)
        pk['r_sp_per_y']    = getr(k_sp, t_sp, (s_sp + W) * (s_sp - H) / 4)
        pk['r_hs_c_per_x']  = getr(k_hs, t_hs, (s_sp + H) * (s_sp - W) / 4)
        pk['r_hs_c_per_y']  = getr(k_hs, t_hs, (s_sp + W) * (s_sp - H) / 4)
        pk['r_hs_per']      = getr(k_hs, t_hs, (s_hs**2 - s_sp**2) / 4)
        pk['r_amb_c_per_x'] = r_cv * s_hs**2 / ((s_sp + H) * (s_sp - W) / 4)
        pk['r_amb_c_per_y'] = r_cv * s_hs**2 / ((s_sp + W) * (s_sp - H) / 4)
        pk['r_amb_per']     = r_cv * s_hs**2 / ((s_hs**2 - s_sp**2) / 4)
        self.pk = pk

    # ------------------------------------------------------------------
    def _nidx(self, l, i, j) -> int:
        return l * self.nr * self.nc + i * self.nc + j

    def _pidx(self, pkg) -> int:
        return self.N_grid + pkg

    # ------------------------------------------------------------------
    def build(self):
        """Assemble the conductance matrix G. Returns G (scipy csr_matrix)."""
        nr, nc, nl = self.nr, self.nc, self.nl
        rx, ry, rz = self.rx, self.ry, self.rz
        pk = self.pk
        HS, SP = self.HS, self.SP

        data, row_idx, col_idx = [], [], []

        def add(r, c, v):
            row_idx.append(r)
            col_idx.append(c)
            data.append(v)

        # ---- Grid cells ------------------------------------------------
        for l in range(nl):
            for i in range(nr):
                for j in range(nc):
                    n   = self._nidx(l, i, j)
                    dia = 0.0

                    # Lateral x (columns, j-direction)
                    if j > 0:
                        add(n, self._nidx(l, i, j-1), -1/rx[l]); dia += 1/rx[l]
                    if j < nc-1:
                        add(n, self._nidx(l, i, j+1), -1/rx[l]); dia += 1/rx[l]

                    # Lateral y (rows, i-direction)
                    if i > 0:
                        add(n, self._nidx(l, i-1, j), -1/ry[l]); dia += 1/ry[l]
                    if i < nr-1:
                        add(n, self._nidx(l, i+1, j), -1/ry[l]); dia += 1/ry[l]

                    # Vertical: down to l-1, uses lower layer's rz (= rz[l-1])
                    if l > 0:
                        Ra = rz[l-1]
                        add(n, self._nidx(l-1, i, j), -1/Ra); dia += 1/Ra

                    # Vertical: up to l+1, uses CURRENT (lower) layer's rz (= rz[l]).
                    # Between layers n1 and n2 the resistance is rz[min(n1,n2)],
                    # the lower layer's resistance.
                    if l < nl-1:
                        Rb = rz[l]
                        add(n, self._nidx(l+1, i, j), -1/Rb); dia += 1/Rb

                    # Package peripheral connections
                    if l == SP:
                        if i == 0:    # north edge
                            R = ry[l]/2 + nc * pk['r_sp1_y']
                            add(n, self._pidx(SP_N), -1/R); dia += 1/R
                        if i == nr-1: # south edge
                            R = ry[l]/2 + nc * pk['r_sp1_y']
                            add(n, self._pidx(SP_S), -1/R); dia += 1/R
                        if j == 0:    # west edge
                            R = rx[l]/2 + nr * pk['r_sp1_x']
                            add(n, self._pidx(SP_W), -1/R); dia += 1/R
                        if j == nc-1: # east edge
                            R = rx[l]/2 + nr * pk['r_sp1_x']
                            add(n, self._pidx(SP_E), -1/R); dia += 1/R

                    elif l == HS:
                        # All HS cells: ambient via rz[HS] (includes convective R)
                        dia += 1/rz[HS]

                        if i == 0:    # north edge -> SINK_C_N
                            R = ry[l]/2 + nc * pk['r_hs1_y']
                            add(n, self._pidx(SINK_C_N), -1/R); dia += 1/R
                        if i == nr-1: # south edge -> SINK_C_S
                            R = ry[l]/2 + nc * pk['r_hs1_y']
                            add(n, self._pidx(SINK_C_S), -1/R); dia += 1/R
                        if j == 0:    # west edge -> SINK_C_W
                            R = rx[l]/2 + nr * pk['r_hs1_x']
                            add(n, self._pidx(SINK_C_W), -1/R); dia += 1/R
                        if j == nc-1: # east edge -> SINK_C_E
                            R = rx[l]/2 + nr * pk['r_hs1_x']
                            add(n, self._pidx(SINK_C_E), -1/R); dia += 1/R

                    add(n, n, dia)

        # ---- Package nodes ---------------------------------------------
        # SINK_N: <-> SINK_C_N, <-> ambient
        n = self._pidx(SINK_N); dia = 0
        R = pk['r_hs2_y'] + pk['r_hs']
        add(n, self._pidx(SINK_C_N), -1/R); dia += 1/R
        dia += 1/(pk['r_hs_per'] + pk['r_amb_per'])
        add(n, n, dia)

        # SINK_S
        n = self._pidx(SINK_S); dia = 0
        R = pk['r_hs2_y'] + pk['r_hs']
        add(n, self._pidx(SINK_C_S), -1/R); dia += 1/R
        dia += 1/(pk['r_hs_per'] + pk['r_amb_per'])
        add(n, n, dia)

        # SINK_W
        n = self._pidx(SINK_W); dia = 0
        R = pk['r_hs2_x'] + pk['r_hs']
        add(n, self._pidx(SINK_C_W), -1/R); dia += 1/R
        dia += 1/(pk['r_hs_per'] + pk['r_amb_per'])
        add(n, n, dia)

        # SINK_E
        n = self._pidx(SINK_E); dia = 0
        R = pk['r_hs2_x'] + pk['r_hs']
        add(n, self._pidx(SINK_C_E), -1/R); dia += 1/R
        dia += 1/(pk['r_hs_per'] + pk['r_amb_per'])
        add(n, n, dia)

        # SINK_C_N: <-> all HS north-edge cells, <-> SP_N, <-> SINK_N, <-> ambient
        n = self._pidx(SINK_C_N); dia = 0
        for j in range(nc):
            R = ry[HS]/2 + nc * pk['r_hs1_y']
            add(n, self._nidx(HS, 0, j), -1/R); dia += 1/R
        R = pk['r_sp_per_y']
        add(n, self._pidx(SP_N), -1/R); dia += 1/R
        R = pk['r_hs2_y'] + pk['r_hs']
        add(n, self._pidx(SINK_N), -1/R); dia += 1/R
        dia += 1/(pk['r_hs_c_per_y'] + pk['r_amb_c_per_y'])
        add(n, n, dia)

        # SINK_C_S
        n = self._pidx(SINK_C_S); dia = 0
        for j in range(nc):
            R = ry[HS]/2 + nc * pk['r_hs1_y']
            add(n, self._nidx(HS, nr-1, j), -1/R); dia += 1/R
        R = pk['r_sp_per_y']
        add(n, self._pidx(SP_S), -1/R); dia += 1/R
        R = pk['r_hs2_y'] + pk['r_hs']
        add(n, self._pidx(SINK_S), -1/R); dia += 1/R
        dia += 1/(pk['r_hs_c_per_y'] + pk['r_amb_c_per_y'])
        add(n, n, dia)

        # SINK_C_W
        n = self._pidx(SINK_C_W); dia = 0
        for i in range(nr):
            R = rx[HS]/2 + nr * pk['r_hs1_x']
            add(n, self._nidx(HS, i, 0), -1/R); dia += 1/R
        R = pk['r_sp_per_x']
        add(n, self._pidx(SP_W), -1/R); dia += 1/R
        R = pk['r_hs2_x'] + pk['r_hs']
        add(n, self._pidx(SINK_W), -1/R); dia += 1/R
        dia += 1/(pk['r_hs_c_per_x'] + pk['r_amb_c_per_x'])
        add(n, n, dia)

        # SINK_C_E
        n = self._pidx(SINK_C_E); dia = 0
        for i in range(nr):
            R = rx[HS]/2 + nr * pk['r_hs1_x']
            add(n, self._nidx(HS, i, nc-1), -1/R); dia += 1/R
        R = pk['r_sp_per_x']
        add(n, self._pidx(SP_E), -1/R); dia += 1/R
        R = pk['r_hs2_x'] + pk['r_hs']
        add(n, self._pidx(SINK_E), -1/R); dia += 1/R
        dia += 1/(pk['r_hs_c_per_x'] + pk['r_amb_c_per_x'])
        add(n, n, dia)

        # SP_N: <-> all SP north-edge cells, <-> SINK_C_N
        n = self._pidx(SP_N); dia = 0
        for j in range(nc):
            R = ry[SP]/2 + nc * pk['r_sp1_y']
            add(n, self._nidx(SP, 0, j), -1/R); dia += 1/R
        R = pk['r_sp_per_y']
        add(n, self._pidx(SINK_C_N), -1/R); dia += 1/R
        add(n, n, dia)

        # SP_S
        n = self._pidx(SP_S); dia = 0
        for j in range(nc):
            R = ry[SP]/2 + nc * pk['r_sp1_y']
            add(n, self._nidx(SP, nr-1, j), -1/R); dia += 1/R
        R = pk['r_sp_per_y']
        add(n, self._pidx(SINK_C_S), -1/R); dia += 1/R
        add(n, n, dia)

        # SP_W
        n = self._pidx(SP_W); dia = 0
        for i in range(nr):
            R = rx[SP]/2 + nr * pk['r_sp1_x']
            add(n, self._nidx(SP, i, 0), -1/R); dia += 1/R
        R = pk['r_sp_per_x']
        add(n, self._pidx(SINK_C_W), -1/R); dia += 1/R
        add(n, n, dia)

        # SP_E
        n = self._pidx(SP_E); dia = 0
        for i in range(nr):
            R = rx[SP]/2 + nr * pk['r_sp1_x']
            add(n, self._nidx(SP, i, nc-1), -1/R); dia += 1/R
        R = pk['r_sp_per_x']
        add(n, self._pidx(SINK_C_E), -1/R); dia += 1/R
        add(n, n, dia)

        self.G = sparse.csr_matrix(
            (data, (row_idx, col_idx)), shape=(self.N, self.N), dtype=np.float64)
        return self.G

    # ------------------------------------------------------------------
    def build_rhs(self, block_powers: dict[str, float]) -> np.ndarray:
        """Assemble the RHS vector Q (shape (N,)) from per-block powers {name: W}."""
        Q = np.zeros(self.N)
        cfg = self.cfg
        ambient = cfg['ambient']
        nr, nc = self.nr, self.nc
        cw, ch = self.cw, self.ch
        HS = self.HS

        # Map block powers to Si-layer grid cells by area-weighted overlap:
        # Q_cell += P_block * overlap_area / block_area  [W]
        chip_h = self.chip_h

        for u in self.units:
            bname = u['name']
            if bname not in block_powers:
                continue
            P = block_powers[bname]
            if P == 0:
                continue
            lx, by = u['leftx'], u['bottomy']
            rx_b, ty_b = lx + u['width'], by + u['height']
            block_area = u['width'] * u['height']

            # Grid rows: i=0 is top (high y), i=nr-1 is bottom (low y)
            i1 = nr - math.ceil((ty_b) / ch)
            i2 = nr - math.floor(by / ch)
            j1 = math.floor(lx / cw)
            j2 = math.ceil(rx_b / cw)
            i1 = max(0, i1); i2 = min(nr, i2)
            j1 = max(0, j1); j2 = min(nc, j2)

            for i in range(i1, i2):
                # Cell y bounds (bottom, top) in physical coords
                cell_bot = chip_h - (i + 1) * ch
                cell_top = chip_h - i * ch
                for j in range(j1, j2):
                    # Cell x bounds
                    cell_lx = j * cw
                    cell_rx = (j + 1) * cw
                    # Overlap
                    ow = min(rx_b, cell_rx) - max(lx, cell_lx)
                    oh = min(ty_b, cell_top) - max(by, cell_bot)
                    ow = max(0.0, ow); oh = max(0.0, oh)
                    overlap_area = ow * oh
                    if overlap_area <= 0:
                        continue
                    n = self._nidx(self.SI, i, j)
                    # Power contribution: P_block * overlap_area / block_area [W]
                    Q[n] += P * overlap_area / block_area

        # Heatsink cells: add T_amb / rz_hs
        for i in range(nr):
            for j in range(nc):
                n = self._nidx(HS, i, j)
                Q[n] += ambient / self.rz[HS]

        # Package node RHS
        pk = self.pk
        Q[self._pidx(SP_W)] = 0
        Q[self._pidx(SP_E)] = 0
        Q[self._pidx(SP_N)] = 0
        Q[self._pidx(SP_S)] = 0
        Q[self._pidx(SINK_C_W)] = ambient / (pk['r_hs_c_per_x'] + pk['r_amb_c_per_x'])
        Q[self._pidx(SINK_C_E)] = ambient / (pk['r_hs_c_per_x'] + pk['r_amb_c_per_x'])
        Q[self._pidx(SINK_C_N)] = ambient / (pk['r_hs_c_per_y'] + pk['r_amb_c_per_y'])
        Q[self._pidx(SINK_C_S)] = ambient / (pk['r_hs_c_per_y'] + pk['r_amb_c_per_y'])
        Q[self._pidx(SINK_W)]   = ambient / (pk['r_hs_per'] + pk['r_amb_per'])
        Q[self._pidx(SINK_E)]   = ambient / (pk['r_hs_per'] + pk['r_amb_per'])
        Q[self._pidx(SINK_N)]   = ambient / (pk['r_hs_per'] + pk['r_amb_per'])
        Q[self._pidx(SINK_S)]   = ambient / (pk['r_hs_per'] + pk['r_amb_per'])

        return Q

    # ------------------------------------------------------------------
    def factorize(self):
        """LU-factorize G; call once, then reuse for multiple RHS."""
        if self.G is None:
            self.build()
        self._factor = factorized(self.G.tocsc())

    def solve(self, Q: np.ndarray) -> np.ndarray:
        """Solve G*T = Q. Returns T of shape (N,)."""
        if self._factor is not None:
            return self._factor(Q)
        if self.G is None:
            self.build()
        return spsolve(self.G, Q)

    # ------------------------------------------------------------------
    def solve_from_powers(self, block_powers: dict[str, float]) -> np.ndarray:
        """Build the RHS and solve for the full temperature vector."""
        Q = self.build_rhs(block_powers)
        return self.solve(Q)

    def silicon_layer(self, T: np.ndarray) -> np.ndarray:
        """Extract Si-layer temperatures as (nr, nc) array."""
        nr, nc = self.nr, self.nc
        return T[self.SI * nr * nc : (self.SI+1) * nr * nc].reshape(nr, nc)

    def all_layers(self, T: np.ndarray) -> np.ndarray:
        """Extract grid temperatures as (nl, nr, nc) array."""
        return T[:self.N_grid].reshape(self.nl, self.nr, self.nc)

    # ------------------------------------------------------------------
    # Adjoint gradient for DRO / CVaR optimization
    # ------------------------------------------------------------------
    def adjoint_dT_dQ(self, T: np.ndarray, dL_dT: np.ndarray) -> np.ndarray:
        """
        Adjoint solve dL/dQ = G^{-T} @ dL/dT. G is symmetric, so G^T = G and
        the forward LU factor is reused (one back-substitution, no re-factorize).
        """
        if self._factor is not None:
            return self._factor(dL_dT)
        return spsolve(self.G.T.tocsc(), dL_dT)

    def peak_T_gradient(self, block_powers: dict[str, float]) -> tuple[float, dict[str, float]]:
        """
        Compute peak Si temperature and its gradient w.r.t. block powers.
        Returns (peak_T, {block_name: dPeakT/dP_block}).
        """
        Q = self.build_rhs(block_powers)
        T = self.solve(Q)
        nr, nc = self.nr, self.nc
        T_si = self.silicon_layer(T)
        idx_flat = np.argmax(T_si)
        peak_i, peak_j = divmod(int(idx_flat), nc)
        peak_val = float(T_si[peak_i, peak_j])

        # dL/dT: 1 at peak Si cell, 0 elsewhere
        dL_dT = np.zeros(self.N)
        dL_dT[self._nidx(self.SI, peak_i, peak_j)] = 1.0
        lam = self.adjoint_dT_dQ(T, dL_dT)

        # dQ/dP_block = overlap_area / block_area at Si cells
        chip_h = self.chip_h
        cw, ch = self.cw, self.ch
        grad = {}
        for u in self.units:
            bname = u['name']
            lx, by = u['leftx'], u['bottomy']
            rx_b, ty_b = lx + u['width'], by + u['height']
            block_area = u['width'] * u['height']
            i1 = max(0, nr - math.ceil(ty_b / ch))
            i2 = min(nr, nr - math.floor(by / ch))
            j1 = max(0, math.floor(lx / cw))
            j2 = min(nc, math.ceil(rx_b / cw))
            g = 0.0
            for i in range(i1, i2):
                cell_bot = chip_h - (i + 1) * ch
                cell_top = chip_h - i * ch
                for j in range(j1, j2):
                    cell_lx = j * cw; cell_rx = (j + 1) * cw
                    ow = max(0.0, min(rx_b, cell_rx) - max(lx, cell_lx))
                    oh = max(0.0, min(ty_b, cell_top) - max(by, cell_bot))
                    if ow * oh <= 0:
                        continue
                    n = self._nidx(self.SI, i, j)
                    g += lam[n] * (ow * oh) / block_area
            grad[bname] = g
        return peak_val, grad

    def rhs_position_grad(self, lam: np.ndarray,
                          block_powers: dict[str, float]
                          ) -> tuple[dict[str, float], dict[str, float]]:
        """Exact position gradient of lam^T Q per block, as (dcx, dcy) dicts.

        Block-to-cell power overlap is piecewise-linear in position, so the
        analytic derivative is exact (a one-sided subgradient on cell boundaries).
        """
        nr, nc = self.nr, self.nc
        cw, ch = self.cw, self.ch
        chip_h = self.chip_h
        dcx: dict[str, float] = {}
        dcy: dict[str, float] = {}
        for u in self.units:
            name = u['name']
            P = block_powers.get(name, 0.0)
            if P == 0.0:
                dcx[name] = 0.0; dcy[name] = 0.0
                continue
            lx, by = u['leftx'], u['bottomy']
            rx_b, ty_b = lx + u['width'], by + u['height']
            block_area = u['width'] * u['height']
            i1 = max(0, nr - math.ceil(ty_b / ch))
            i2 = min(nr, nr - math.floor(by / ch))
            j1 = max(0, math.floor(lx / cw))
            j2 = min(nc, math.ceil(rx_b / cw))
            gx = 0.0
            gy = 0.0
            for i in range(i1, i2):
                cell_bot = chip_h - (i + 1) * ch
                cell_top = chip_h - i * ch
                oh = min(ty_b, cell_top) - max(by, cell_bot)
                if oh <= 0:
                    continue
                doh = (1.0 if ty_b < cell_top else 0.0) - (1.0 if by > cell_bot else 0.0)
                for j in range(j1, j2):
                    cell_lx = j * cw; cell_rx = (j + 1) * cw
                    ow = min(rx_b, cell_rx) - max(lx, cell_lx)
                    if ow <= 0:
                        continue
                    dow = (1.0 if rx_b < cell_rx else 0.0) - (1.0 if lx > cell_lx else 0.0)
                    l = lam[self._nidx(self.SI, i, j)]
                    gx += l * oh * dow
                    gy += l * ow * doh
            dcx[name] = P / block_area * gx
            dcy[name] = P / block_area * gy
        return dcx, dcy


# ---------------------------------------------------------------------------
# External reference-solver runner
# ---------------------------------------------------------------------------

def run_reference_grid(flp_path: str, ptrace_path: str, config_path: str,
                     reference_bin: str, nr: int, nc: int,
                     out_grid: str, out_steady: str) -> None:
    """Run the external reference solver in grid mode; writes grid_steady + block_steady files."""
    import subprocess
    cmd = [
        reference_bin,
        '-c', config_path,
        '-f', flp_path,
        '-p', ptrace_path,
        '-model_type', 'grid',
        '-grid_rows', str(nr),
        '-grid_cols', str(nc),
        '-grid_steady_file', out_grid,
        '-steady_file', out_steady,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'reference solver failed:\n{r.stderr}')


def read_reference_grid_steady(path: str, nl: int, nr: int, nc: int) -> np.ndarray:
    """
    Parse a grid_steady_file into array of shape (nl, nr, nc).
    File format: 'Layer n:\\n' then lines '  cell_idx\\t temp\\n'.
    """
    T = np.zeros((nl, nr, nc))
    with open(path) as f:
        layer = -1
        for line in f:
            line = line.strip()
            if line.startswith('Layer'):
                layer = int(line.split()[1].rstrip(':'))
                continue
            if line and layer >= 0:
                parts = line.split()
                cidx = int(parts[0])
                temp = float(parts[1])
                i, j = divmod(cidx, nc)
                T[layer, i, j] = temp
    return T


def read_reference_block_steady(path: str) -> dict[str, float]:
    """Parse block steady-state output."""
    temps = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                temps[parts[0]] = float(parts[1])
    return temps


# ---------------------------------------------------------------------------
# Randomised power map generator
# ---------------------------------------------------------------------------

def random_power_map(units: list[dict], total_power: float,
                     rng: np.random.Generator,
                     hot_fraction: float = 0.4) -> dict[str, float]:
    """
    Generate a randomised but physically plausible block power map.
    total_power: target total chip power in W.
    hot_fraction: fraction of units that are 'active' (the rest get idle power ~5%).
    """
    n = len(units)
    # Choose which units are hot
    n_hot = max(1, int(n * hot_fraction))
    hot_idx = rng.choice(n, size=n_hot, replace=False)
    hot_set = set(hot_idx)

    areas = np.array([u['width'] * u['height'] for u in units])
    total_area = areas.sum()

    raw = np.ones(n) * 0.05  # idle baseline
    for idx in hot_set:
        raw[idx] = rng.uniform(0.5, 3.0)

    # Normalise to target total power (weighted by area to keep density variation)
    power_density = raw / total_area
    powers = power_density * areas * (total_power / (power_density * areas).sum())

    return {u['name']: float(powers[i]) for i, u in enumerate(units)}
