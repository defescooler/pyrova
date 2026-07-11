"""Kraken nano-UAV SoC: a measurement-anchored heterogeneous testbed (silicon per-subsystem powers; areas, splits, and duty cycles assumed)."""

from __future__ import annotations
import numpy as np

# Blocks: architectural sub-units that exist in the design [K-HC/K-SH §II].
# Areas mm^2: die is 9 mm^2 [K-HC Fig.5]; CUTIE total is 2.96 mm^2 with memory
# ~60% of it [TCN-C — the only published per-block area]. All OTHER areas are
# ASSUMED from architectural composition (SRAM capacity, core counts) and sum
# to 6.7 of 9 mm^2 (rest: pads/PLL/IO, not placed).
#   name, width [mm], height [mm]
_BLOCKS = [
    ("FC",         0.75, 0.60),  # fabric ctrl, 1x RV32 + peri     [area ASSUMED]
    ("L2_0",       1.00, 0.65),  # 512 KiB half of 1 MiB L2 [K-HC] [area ASSUMED]
    ("L2_1",       1.00, 0.65),  #                                 [area ASSUMED]
    ("CL_CORES",   0.95, 0.74),  # 8x RV32 cluster cores [K-HC]    [area ASSUMED]
    ("CL_TCDM",    0.80, 0.50),  # 128 KiB TCDM [K-HC]             [area ASSUMED]
    ("SNE_0",      0.75, 0.60),  # 4 of 8 SNE slices [K-SH]        [area ASSUMED]
    ("SNE_1",      0.75, 0.60),  #                                 [area ASSUMED]
    ("CUTIE_PE",   1.25, 0.95),  # 96 OCUs; CUTIE=2.96mm^2, mem~60% [TCN-C MEASURED total; split ASSUMED]
    ("CUTIE_FMEM", 1.20, 0.85),  # 158 kB feature memory [K-HC]
    ("CUTIE_WMEM", 0.95, 0.80),  # 117 kB weight memory [K-HC]
]
_DIE_W_MM = 3.0   # die published as 9 mm^2 only [K-HC]; W x H ASSUMED square
_DIE_H_MM = 3.0

# Subsystem -> sub-block power split fractions (ASSUMED; compute-heavy blocks
# take the larger share). The SUBSYSTEM totals they multiply are MEASURED.
_SPLIT = {
    "SNE":    {"SNE_0": 0.5, "SNE_1": 0.5},
    "CUTIE":  {"CUTIE_PE": 0.60, "CUTIE_FMEM": 0.25, "CUTIE_WMEM": 0.15},
    "CL":     {"CL_CORES": 0.75, "CL_TCDM": 0.25},
    "L2":     {"L2_0": 0.5, "L2_1": 0.5},
}

# Modes: application phases with MEASURED subsystem powers [mW].
#   event   : DVS -> SNE depth estimation, 98 mW @220 MHz 0.8 V [K-SH §IV]
#   frame   : CUTIE CIFAR ternary classification, 110 mW @330 MHz 0.8 V [K-SH §IV]
#   dronet_p: Tiny-PULP-Dronet on FC+cluster, 165 mW @280/300 MHz 0.8 V [K-SH §IV]
#   dronet_e: same, efficiency point, 23 mW @110 MHz 0.55 V [K-SH §IV]
#   fusion  : all three concurrent, 373 mW total [K-SH §V] (numerically the sum
#             of the three single-task measurements — flagged in the extraction)
#   idle    : SoC power floor ~2 mW [K-HC Fig.5]; distribution ASSUMED ~by area
# FC baseline in non-cluster modes: ~3.5 mW acquisition [C-ES]. L2 activity
# power is not separately published; small mode-dependent values ASSUMED.
_IDLE = {"FC": 0.4, "L2_0": 0.3, "L2_1": 0.3, "CL_CORES": 0.25, "CL_TCDM": 0.1,
         "SNE_0": 0.15, "SNE_1": 0.15, "CUTIE_PE": 0.15, "CUTIE_FMEM": 0.1,
         "CUTIE_WMEM": 0.1}   # sums ~2 mW [K-HC floor; split ASSUMED]


def _mode_power(sne=0.0, cutie=0.0, cl=0.0, fc=0.0, l2=0.0) -> dict:
    p = dict(_IDLE)
    for k, v in _SPLIT["SNE"].items():
        p[k] += sne * v
    for k, v in _SPLIT["CUTIE"].items():
        p[k] += cutie * v
    for k, v in _SPLIT["CL"].items():
        p[k] += cl * v
    for k, v in _SPLIT["L2"].items():
        p[k] += l2 * v
    p["FC"] += fc
    return p


_MODES = {
    # subsystem totals in mW           SNE   CUTIE  CL     FC    L2
    "event":    _mode_power(sne=98.0,               fc=3.5, l2=2.0),
    "frame":    _mode_power(cutie=110.0,            fc=3.5, l2=2.0),
    "dronet_p": _mode_power(cl=140.0,               fc=25.0, l2=6.0),  # 165 total incl. FC [K-SH]
    "dronet_e": _mode_power(cl=18.0,                fc=5.0,  l2=2.0),  # 23 total incl. FC [K-SH]
    "fusion":   _mode_power(sne=98.0, cutie=110.0, cl=140.0, fc=25.0, l2=8.0),
    "idle":     _mode_power(),
}
# Duty cycles are NOT published — mission-profile mix ASSUMED (always-on event
# watching dominant on a nano-UAV).
_MODE_PROBS = {"event": 0.30, "frame": 0.10, "dronet_p": 0.10, "dronet_e": 0.15,
               "fusion": 0.15, "idle": 0.20}


def kraken_units() -> list[dict]:
    """Block list as solver unit dicts (metres), tiled left-to-right; only sizes and the 3x3 mm die box matter (the placer moves blocks)."""
    units, x, y, row_h = [], 0.0, 0.0, 0.0
    chip_w = _DIE_W_MM * 1e-3
    for name, w_mm, h_mm in _BLOCKS:
        w, h = w_mm * 1e-3, h_mm * 1e-3
        if x + w > chip_w:
            x, y = 0.0, y + row_h
            row_h = 0.0
        units.append(dict(name=name, width=w, height=h, leftx=x, bottomy=y))
        x += w
        row_h = max(row_h, h)
    return units


class KrakenWorkloadModel:
    """Mode-mixture sampler over the Kraken model; sample(n) -> power arrays in WATTS after `scale` (raw table is mW)."""

    def __init__(self, units: list[dict], seed: int = 0, noise: float = 0.15):
        block_names = [b[0] for b in _BLOCKS]
        assert [u["name"] for u in units] == block_names, \
            "units must be kraken_units() output (order defines mode vectors)"
        self.units = units
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.mode_names = list(_MODES)
        self.modes = np.array([[_MODES[m][b] for b in block_names]
                               for m in self.mode_names]) * 1e-3   # mW -> W
        self.mode_p = np.array([_MODE_PROBS[m] for m in self.mode_names])
        self.scale = 1.0

    def calibrate_scale(self, scenario_peaks_fn, target: float) -> float:
        """Set `scale` so the probability-weighted mean peak dT of the pure mode vectors equals `target` [K]."""
        base = scenario_peaks_fn([m.copy() for m in self.modes])
        self.scale = float(target / np.dot(self.mode_p, base))
        return self.scale

    def sample(self, n: int) -> list[np.ndarray]:
        out = []
        for _ in range(n):
            m = self.rng.choice(len(self.mode_names), p=self.mode_p)
            p = self.modes[m] * self.scale
            p = p * (1.0 + self.rng.uniform(-self.noise, self.noise, size=len(p)))
            out.append(np.maximum(p, 1e-6))
        return out

    def regime_stats(self, n: int = 4000, seed: int = 12345) -> dict:
        """Confound/regime statistics to report with any placement comparison on this model."""
        rng = np.random.default_rng(seed)
        saved, self.rng = self.rng, rng
        try:
            P = np.array(self.sample(n))
        finally:
            self.rng = saved
        tot = P.sum(1)
        names = [b[0] for b in _BLOCKS]
        heavy = [names.index(k) for k in
                 ("SNE_0", "SNE_1", "CUTIE_PE", "CL_CORES")]
        shares = P[:, heavy].max(1) / tot
        return {"e_total_W": float(tot.mean()),
                "total_cv": float(tot.std() / tot.mean()),
                "heaviest_engine_share_mean": float(shares.mean())}
