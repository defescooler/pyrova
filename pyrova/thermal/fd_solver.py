"""Steady-state finite-difference thermal solver for a 4-layer grid model
(Si / TIM / Spreader / Heatsink) with adjoint gradients.

The solver assembles the same linear conductance system the reference iterative
grid solver converges to and solves it exactly by LU. ``G`` is symmetric, so the
adjoint reuses the forward factorisation. The field, peak, and adjoint gradient
of this module are pinned by ``pyrova.tests.golden``; run ``make verify``
before and after any change to the numerics.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized, spsolve

# Parsing lives in core.io; re-exported here for the historical import path.
from pyrova.core.io import chip_dimensions, parse_config, parse_flp, parse_ptrace

__all__ = [
    "GridFDSolver", "getr", "random_power_map",
    "parse_flp", "parse_config", "parse_ptrace", "chip_dimensions",
    "run_reference_grid", "read_reference_grid_steady", "read_reference_block_steady",
]


def getr(k: float, length: float, area: float) -> float:
    """Thermal resistance R = length / (k * area)."""
    return length / (k * area)


# Package peripheral node indices (offsets past the grid nodes).
SP_W, SP_E, SP_N, SP_S = 0, 1, 2, 3
SINK_C_W, SINK_C_E, SINK_C_N, SINK_C_S = 4, 5, 6, 7
SINK_W, SINK_E, SINK_N, SINK_S = 8, 9, 10, 11
EXTRA = 12

# Per-direction bookkeeping: which package nodes, which axis, whether the shared
# edge runs along rows (y) or columns (x). Drives the package-node assembly.
_DIRS = ("N", "S", "W", "E")
_AXIS = {"N": "y", "S": "y", "W": "x", "E": "x"}
_SP = {"W": SP_W, "E": SP_E, "N": SP_N, "S": SP_S}
_SINK_C = {"W": SINK_C_W, "E": SINK_C_E, "N": SINK_C_N, "S": SINK_C_S}
_SINK = {"W": SINK_W, "E": SINK_E, "N": SINK_N, "S": SINK_S}


class GridFDSolver:
    """4-layer grid model with a 5-point conductance stencil and 12 package nodes.

    Layers (low -> high): 0 = Silicon (power-dissipating), 1 = TIM, 2 = Spreader,
    3 = Heatsink (convective BC to ambient).
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

        self.cw = chip_w / nc
        self.ch = chip_h / nr

        self._init_layer_params()
        self._init_package_params()

        self.G = None
        self._factor = None

    # Parameter tables

    def _init_layer_params(self):
        cfg = self.cfg
        cw, ch = self.cw, self.ch
        k = [cfg["k_chip"], cfg["k_interface"], cfg["k_spreader"], cfg["k_sink"]]
        t = [cfg["t_chip"], cfg["t_interface"], cfg["t_spreader"], cfg["t_sink"]]
        self.k_layer = k
        self.t_layer = t

        self.rx = [getr(k[l], cw, ch * t[l]) for l in range(4)]
        self.ry = [getr(k[l], ch, cw * t[l]) for l in range(4)]
        self.rz = [getr(k[l], t[l], cw * ch) for l in range(4)]
        # Heatsink: add convective resistance per cell.
        self.rz[self.HS] += cfg["r_convec"] * cfg["s_sink"] ** 2 / (cw * ch)

    def _init_package_params(self):
        cfg = self.cfg
        W, H = self.chip_w, self.chip_h
        k_sp, t_sp, s_sp = cfg["k_spreader"], cfg["t_spreader"], cfg["s_spreader"]
        k_hs, t_hs, s_hs = cfg["k_sink"], cfg["t_sink"], cfg["s_sink"]
        r_cv = cfg["r_convec"]

        self.pk = {
            "r_sp1_x":   getr(k_sp, (s_sp - W) / 4, (s_sp + 3 * H) / 4 * t_sp),
            "r_sp1_y":   getr(k_sp, (s_sp - H) / 4, (s_sp + 3 * W) / 4 * t_sp),
            "r_hs1_x":   getr(k_hs, (s_sp - W) / 4, (s_sp + 3 * H) / 4 * t_hs),
            "r_hs1_y":   getr(k_hs, (s_sp - H) / 4, (s_sp + 3 * W) / 4 * t_hs),
            "r_hs2_x":   getr(k_hs, (s_sp - W) / 4, (3 * s_sp + H) / 4 * t_hs),
            "r_hs2_y":   getr(k_hs, (s_sp - H) / 4, (3 * s_sp + W) / 4 * t_hs),
            "r_hs":      getr(k_hs, (s_hs - s_sp) / 4, (s_hs + 3 * s_sp) / 4 * t_hs),
            "r_sp_per_x":    getr(k_sp, t_sp, (s_sp + H) * (s_sp - W) / 4),
            "r_sp_per_y":    getr(k_sp, t_sp, (s_sp + W) * (s_sp - H) / 4),
            "r_hs_c_per_x":  getr(k_hs, t_hs, (s_sp + H) * (s_sp - W) / 4),
            "r_hs_c_per_y":  getr(k_hs, t_hs, (s_sp + W) * (s_sp - H) / 4),
            "r_hs_per":      getr(k_hs, t_hs, (s_hs ** 2 - s_sp ** 2) / 4),
            "r_amb_c_per_x": r_cv * s_hs ** 2 / ((s_sp + H) * (s_sp - W) / 4),
            "r_amb_c_per_y": r_cv * s_hs ** 2 / ((s_sp + W) * (s_sp - H) / 4),
            "r_amb_per":     r_cv * s_hs ** 2 / ((s_hs ** 2 - s_sp ** 2) / 4),
        }

    # Node indexing

    def _nidx(self, l, i, j) -> int:
        return l * self.nr * self.nc + i * self.nc + j

    def _pidx(self, pkg) -> int:
        return self.N_grid + pkg

    def _edge_cells(self, direction: str):
        """(i, j) of the grid cells along one chip edge."""
        nr, nc = self.nr, self.nc
        if direction == "N":
            return [(0, j) for j in range(nc)]
        if direction == "S":
            return [(nr - 1, j) for j in range(nc)]
        if direction == "W":
            return [(i, 0) for i in range(nr)]
        return [(i, nc - 1) for i in range(nr)]  # E

    def _on_edge(self, i: int, j: int, direction: str) -> bool:
        """Whether grid cell (i, j) lies on the given chip edge."""
        if direction == "N":
            return i == 0
        if direction == "S":
            return i == self.nr - 1
        if direction == "W":
            return j == 0
        return j == self.nc - 1  # E

    def _edge_R(self, layer: int, direction: str, kind: str) -> float:
        """Resistance from an edge grid cell to its peripheral spreader/sink node."""
        if _AXIS[direction] == "y":
            return self.ry[layer] / 2 + self.nc * self.pk[f"r_{kind}1_y"]
        return self.rx[layer] / 2 + self.nr * self.pk[f"r_{kind}1_x"]

    # Assembly

    def build(self) -> sparse.csr_matrix:
        """Assemble and cache the conductance matrix G (scipy csr_matrix)."""
        nr, nc, nl = self.nr, self.nc, self.nl
        rx, ry, rz = self.rx, self.ry, self.rz
        pk = self.pk
        HS, SP = self.HS, self.SP

        data, rows, cols = [], [], []

        def add(r, c, v):
            rows.append(r); cols.append(c); data.append(v)

        # Grid cells: 5-point stencil + vertical layer coupling + edge coupling
        # to package nodes. Between layers the resistance is the lower layer's rz.
        for l in range(nl):
            for i in range(nr):
                for j in range(nc):
                    n = self._nidx(l, i, j)
                    dia = 0.0
                    if j > 0:
                        add(n, self._nidx(l, i, j - 1), -1 / rx[l]); dia += 1 / rx[l]
                    if j < nc - 1:
                        add(n, self._nidx(l, i, j + 1), -1 / rx[l]); dia += 1 / rx[l]
                    if i > 0:
                        add(n, self._nidx(l, i - 1, j), -1 / ry[l]); dia += 1 / ry[l]
                    if i < nr - 1:
                        add(n, self._nidx(l, i + 1, j), -1 / ry[l]); dia += 1 / ry[l]
                    if l > 0:
                        add(n, self._nidx(l - 1, i, j), -1 / rz[l - 1]); dia += 1 / rz[l - 1]
                    if l < nl - 1:
                        add(n, self._nidx(l + 1, i, j), -1 / rz[l]); dia += 1 / rz[l]

                    if l == SP:
                        for d in _DIRS:
                            if self._on_edge(i, j, d):
                                R = self._edge_R(SP, d, "sp")
                                add(n, self._pidx(_SP[d]), -1 / R); dia += 1 / R
                    elif l == HS:
                        dia += 1 / rz[HS]                 # ambient via convective rz
                        for d in _DIRS:
                            if self._on_edge(i, j, d):
                                R = self._edge_R(HS, d, "hs")
                                add(n, self._pidx(_SINK_C[d]), -1 / R); dia += 1 / R

                    add(n, n, dia)

        # Package nodes. Each block below is the symmetric partner of the edge
        # coupling added above, plus the peripheral/ambient resistances.
        for d in _DIRS:
            ax = _AXIS[d]

            # Spreader peripheral node: to its edge SP cells and to SINK_C.
            n = self._pidx(_SP[d]); dia = 0.0
            for i, j in self._edge_cells(d):
                R = self._edge_R(SP, d, "sp")
                add(n, self._nidx(SP, i, j), -1 / R); dia += 1 / R
            R = pk[f"r_sp_per_{ax}"]
            add(n, self._pidx(_SINK_C[d]), -1 / R); dia += 1 / R
            add(n, n, dia)

            # Sink centre-peripheral node: to its edge HS cells, SP, outer sink, ambient.
            n = self._pidx(_SINK_C[d]); dia = 0.0
            for i, j in self._edge_cells(d):
                R = self._edge_R(HS, d, "hs")
                add(n, self._nidx(HS, i, j), -1 / R); dia += 1 / R
            R = pk[f"r_sp_per_{ax}"]
            add(n, self._pidx(_SP[d]), -1 / R); dia += 1 / R
            R = pk[f"r_hs2_{ax}"] + pk["r_hs"]
            add(n, self._pidx(_SINK[d]), -1 / R); dia += 1 / R
            dia += 1 / (pk[f"r_hs_c_per_{ax}"] + pk[f"r_amb_c_per_{ax}"])
            add(n, n, dia)

            # Outer sink node: to its SINK_C node and to ambient.
            n = self._pidx(_SINK[d]); dia = 0.0
            R = pk[f"r_hs2_{ax}"] + pk["r_hs"]
            add(n, self._pidx(_SINK_C[d]), -1 / R); dia += 1 / R
            dia += 1 / (pk["r_hs_per"] + pk["r_amb_per"])
            add(n, n, dia)

        self.G = sparse.csr_matrix(
            (data, (rows, cols)), shape=(self.N, self.N), dtype=np.float64)
        return self.G

    # Block <-> grid geometry

    def _touched_cells(self, u: dict):
        """Yield (i, j, cell_lx, cell_rx, cell_bot, cell_top) for every Si cell a
        block's bounding box reaches. Row i=0 is the top (high y) of the chip."""
        nr, nc, cw, ch = self.nr, self.nc, self.cw, self.ch
        chip_h = self.chip_h
        by, ty_b = u["bottomy"], u["bottomy"] + u["height"]
        lx, rx_b = u["leftx"], u["leftx"] + u["width"]
        i1 = max(0, nr - math.ceil(ty_b / ch))
        i2 = min(nr, nr - math.floor(by / ch))
        j1 = max(0, math.floor(lx / cw))
        j2 = min(nc, math.ceil(rx_b / cw))
        for i in range(i1, i2):
            cell_bot = chip_h - (i + 1) * ch
            cell_top = chip_h - i * ch
            for j in range(j1, j2):
                yield i, j, j * cw, (j + 1) * cw, cell_bot, cell_top

    def build_rhs(self, block_powers: dict[str, float]) -> np.ndarray:
        """Assemble the RHS Q (shape (N,)) from per-block powers {name: W}.

        Block power is spread onto Si cells by area-weighted overlap; heatsink
        cells and package nodes carry the ambient boundary term.
        """
        Q = np.zeros(self.N)
        ambient = self.cfg["ambient"]

        for u in self.units:
            P = block_powers.get(u["name"], 0.0)
            if not P:
                continue
            lx, by = u["leftx"], u["bottomy"]
            rx_b, ty_b = lx + u["width"], by + u["height"]
            block_area = u["width"] * u["height"]
            for i, j, clx, crx, cbot, ctop in self._touched_cells(u):
                ow = max(0.0, min(rx_b, crx) - max(lx, clx))
                oh = max(0.0, min(ty_b, ctop) - max(by, cbot))
                overlap_area = ow * oh
                if overlap_area <= 0:
                    continue
                Q[self._nidx(self.SI, i, j)] += P * overlap_area / block_area

        HS = self.HS
        for i in range(self.nr):
            for j in range(self.nc):
                Q[self._nidx(HS, i, j)] += ambient / self.rz[HS]

        pk = self.pk
        for d in _DIRS:
            ax = _AXIS[d]
            Q[self._pidx(_SP[d])] = 0.0
            Q[self._pidx(_SINK_C[d])] = ambient / (pk[f"r_hs_c_per_{ax}"] + pk[f"r_amb_c_per_{ax}"])
            Q[self._pidx(_SINK[d])] = ambient / (pk["r_hs_per"] + pk["r_amb_per"])
        return Q

    # Solves

    def factorize(self) -> None:
        """LU-factorize G once, then reuse for multiple RHS."""
        if self.G is None:
            self.build()
        self._factor = factorized(self.G.tocsc())

    def solve(self, Q: np.ndarray) -> np.ndarray:
        """Solve G*T = Q for the full temperature vector (shape (N,))."""
        if self._factor is not None:
            return self._factor(Q)
        if self.G is None:
            self.build()
        return spsolve(self.G, Q)

    def solve_from_powers(self, block_powers: dict[str, float]) -> np.ndarray:
        return self.solve(self.build_rhs(block_powers))

    def silicon_layer(self, T: np.ndarray) -> np.ndarray:
        """Si-layer temperatures as an (nr, nc) array."""
        nr, nc = self.nr, self.nc
        return T[self.SI * nr * nc:(self.SI + 1) * nr * nc].reshape(nr, nc)

    def all_layers(self, T: np.ndarray) -> np.ndarray:
        """Grid temperatures as an (nl, nr, nc) array."""
        return T[:self.N_grid].reshape(self.nl, self.nr, self.nc)

    # Adjoint gradients

    def adjoint_dT_dQ(self, dL_dT: np.ndarray) -> np.ndarray:
        """Adjoint solve: returns lam with G^T lam = dL/dT. G is symmetric, so the
        forward LU factor is reused; only the un-factorised path solves G^T
        explicitly."""
        if self._factor is not None:
            return self._factor(dL_dT)
        return spsolve(self.G.T.tocsc(), dL_dT)

    def peak_T_gradient(self, block_powers: dict[str, float]) -> tuple[float, dict[str, float]]:
        """Peak Si temperature rise dT = T_peak - T_ambient [K] and its gradient
        w.r.t. each block's power [K/W]. Returns dT, never absolute T (project
        invariant: all thermal metrics are ambient-relative).

        WARNING: the gradient holds the argmax cell fixed. It is a subgradient at
        placements where the hot cell changes; callers probing near such kinks
        must detect them separately.
        """
        Q = self.build_rhs(block_powers)
        T = self.solve(Q)
        nc = self.nc
        T_si = self.silicon_layer(T)
        peak_i, peak_j = divmod(int(np.argmax(T_si)), nc)
        peak_dt = float(T_si[peak_i, peak_j]) - self.cfg["ambient"]

        # adjoint: G^T lam = e_{i*}
        dL_dT = np.zeros(self.N)
        dL_dT[self._nidx(self.SI, peak_i, peak_j)] = 1.0
        lam = self.adjoint_dT_dQ(dL_dT)

        # grad = A(p)^T lam; dQ/dP_b = overlap_area / block_area on the Si cells
        # block b touches (column b of A(p)).
        grad = {}
        for u in self.units:
            lx, by = u["leftx"], u["bottomy"]
            rx_b, ty_b = lx + u["width"], by + u["height"]
            block_area = u["width"] * u["height"]
            g = 0.0
            for i, j, clx, crx, cbot, ctop in self._touched_cells(u):
                ow = max(0.0, min(rx_b, crx) - max(lx, clx))
                oh = max(0.0, min(ty_b, ctop) - max(by, cbot))
                if ow * oh <= 0:
                    continue
                g += lam[self._nidx(self.SI, i, j)] * (ow * oh) / block_area
            grad[u["name"]] = g
        return peak_dt, grad

    def power_injection_matrix(self) -> np.ndarray:
        """A(p): dQ/dP as a dense (N, n_units) matrix — column b holds block b's
        area-overlap fractions on the Si cells (no ambient terms). T = G^{-1}
        (A P + b_amb), so row_i(G^{-1}A) = a_i = A(p)^T lam_i for the node-i
        adjoint lam_i (G^T lam_i = e_i)."""
        A = np.zeros((self.N, len(self.units)))
        for b, u in enumerate(self.units):
            lx, by = u["leftx"], u["bottomy"]
            rx_b, ty_b = lx + u["width"], by + u["height"]
            block_area = u["width"] * u["height"]
            for i, j, clx, crx, cbot, ctop in self._touched_cells(u):
                ow = max(0.0, min(rx_b, crx) - max(lx, clx))
                oh = max(0.0, min(ty_b, ctop) - max(by, cbot))
                if ow * oh > 0:
                    A[self._nidx(self.SI, i, j), b] += (ow * oh) / block_area
        return A

    def rhs_position_grad(self, lam: np.ndarray,
                          block_powers: dict[str, float]
                          ) -> tuple[dict[str, float], dict[str, float]]:
        """Exact position gradient of lam^T Q per block, as (dcx, dcy) dicts.

        Block-to-cell overlap is piecewise-linear in position, so the analytic
        derivative is exact (a one-sided subgradient on cell boundaries).
        """
        dcx: dict[str, float] = {}
        dcy: dict[str, float] = {}
        for u in self.units:
            name = u["name"]
            P = block_powers.get(name, 0.0)
            if P == 0.0:
                dcx[name] = dcy[name] = 0.0
                continue
            lx, by = u["leftx"], u["bottomy"]
            rx_b, ty_b = lx + u["width"], by + u["height"]
            block_area = u["width"] * u["height"]
            gx = gy = 0.0
            for i, j, clx, crx, cbot, ctop in self._touched_cells(u):
                oh = min(ty_b, ctop) - max(by, cbot)
                if oh <= 0:
                    continue
                ow = min(rx_b, crx) - max(lx, clx)
                if ow <= 0:
                    continue
                dow = (1.0 if rx_b < crx else 0.0) - (1.0 if lx > clx else 0.0)
                doh = (1.0 if ty_b < ctop else 0.0) - (1.0 if by > cbot else 0.0)
                l = lam[self._nidx(self.SI, i, j)]
                gx += l * oh * dow
                gy += l * ow * doh
            dcx[name] = P / block_area * gx
            dcy[name] = P / block_area * gy
        return dcx, dcy


# External reference-solver runner

def run_reference_grid(flp_path: str, ptrace_path: str, config_path: str,
                       reference_bin: str, nr: int, nc: int,
                       out_grid: str, out_steady: str) -> None:
    """Run the external reference solver in grid mode (writes grid + block steady files)."""
    import subprocess
    cmd = [
        reference_bin, "-c", config_path, "-f", flp_path, "-p", ptrace_path,
        "-model_type", "grid", "-grid_rows", str(nr), "-grid_cols", str(nc),
        "-grid_steady_file", out_grid, "-steady_file", out_steady,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"reference solver failed:\n{r.stderr}")


def read_reference_grid_steady(path: str, nl: int, nr: int, nc: int) -> np.ndarray:
    """Parse a grid_steady_file ('Layer n:' then 'idx temp' lines) into (nl, nr, nc)."""
    T = np.zeros((nl, nr, nc))
    layer = -1
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Layer"):
                layer = int(line.split()[1].rstrip(":"))
            elif line and layer >= 0:
                cidx, temp = line.split()[:2]
                i, j = divmod(int(cidx), nc)
                T[layer, i, j] = float(temp)
    return T


def read_reference_block_steady(path: str) -> dict[str, float]:
    """Parse a block steady-state output into {block_name: temperature}."""
    temps = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                temps[parts[0]] = float(parts[1])
    return temps


# Randomised power-map generator

def random_power_map(units: list[dict], total_power: float,
                     rng: np.random.Generator,
                     hot_fraction: float = 0.4) -> dict[str, float]:
    """A randomised but plausible block power map summing to `total_power` [W].

    A `hot_fraction` of units draw active power (0.5-3.0 W raw); the rest idle at
    a 5% baseline. Raw values are area-weighted, then normalised to the target.
    """
    n = len(units)
    n_hot = max(1, int(n * hot_fraction))
    hot_idx = rng.choice(n, size=n_hot, replace=False)

    areas = np.array([u["width"] * u["height"] for u in units])
    total_area = areas.sum()

    raw = np.full(n, 0.05)
    # Draw in ascending index order (explicit sort, not set-iteration order,
    # which is a CPython hash-table detail): the uniform draws are
    # order-dependent and this makes the stream floorplan-size-independent.
    for idx in np.sort(hot_idx):
        raw[idx] = rng.uniform(0.5, 3.0)

    weighted = raw * areas
    powers = weighted * (total_power / weighted.sum())
    return {u["name"]: float(powers[i]) for i, u in enumerate(units)}
