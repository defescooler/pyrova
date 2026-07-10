"""Stylised heterogeneous-SoC testbed: the regime the theory's mechanism
predicts risk-aware placement should pay off in.

Motivation (mechanism-driven, not outcome-driven): exp009 showed the theory's
required anti-correlation exists in real workloads but was thermally
weightless on a small core (FP ~1.5% of power). The mechanism needs
anti-correlated clusters that are each HEAVY enough to own the hotspot. That
is the everyday situation on heterogeneous SoCs — different workload classes
light different high-power engines (game -> GPU, ML inference -> NPU,
compile -> CPU, video -> codec). This module encodes that regime in a
deliberately stylised form:

  * Blocks: a laptop-class SoC block list with areas/aspect ratios and
    per-engine max powers in plausible TDP-class ranges (stylised — no
    specific product is modelled).
  * Modes: workload classes as activity vectors over engines, each mode
    driving a DIFFERENT heavy engine near its max.

SCOPE CONTRACT: this is an ENGINEERED FAVORABLE REGIME. Results on it are
existence/upper-bound statements ("risk-aware placement can pay this much
under these conditions"), never prevalence statements about real chips. The
testbed has a built-in validity gate (exp023): the hotspot must actually move
across modes, otherwise the regime construction failed and no verdict prints.
"""

from __future__ import annotations
import numpy as np

# name, width [mm], height [mm], max dynamic power [W]
# Stylised laptop-class SoC (~120 mm^2, ~45 W all-engines-max, realistic
# aspect ratios; no specific die is modelled).
_BLOCKS = [
    ("CPU_P0",   3.2, 2.4, 7.0),   # performance cores (2 clusters)
    ("CPU_P1",   3.2, 2.4, 7.0),
    ("CPU_E",    2.8, 2.0, 3.5),   # efficiency cluster
    ("GPU_0",    4.5, 3.0, 9.0),   # GPU halves
    ("GPU_1",    4.5, 3.0, 9.0),
    ("NPU",      3.6, 2.8, 8.0),
    ("MediaEng", 2.4, 2.0, 4.0),   # video codec
    ("ISP",      2.2, 1.8, 3.0),
    ("SLC",      5.0, 2.6, 2.0),   # system-level cache: big, cool
    ("DDR_PHY",  6.0, 1.2, 2.5),
    ("Modem_IO", 3.0, 1.6, 2.0),
    ("Uncore",   3.4, 2.2, 2.5),
]

# Workload classes: activity in [0,1] per block. Each mode drives a DIFFERENT
# heavy engine near max — heavy anti-correlation by construction.
_MODES = {
    #             P0    P1    E     G0    G1    NPU   Med   ISP   SLC   DDR   Mdm   Unc
    "game":     (0.45, 0.45, 0.20, 1.00, 1.00, 0.10, 0.30, 0.10, 0.60, 0.70, 0.10, 0.50),
    "ml_infer": (0.35, 0.35, 0.25, 0.15, 0.15, 1.00, 0.05, 0.10, 0.70, 0.80, 0.05, 0.50),
    "compile":  (1.00, 1.00, 0.60, 0.05, 0.05, 0.05, 0.05, 0.05, 0.55, 0.60, 0.05, 0.45),
    "video":    (0.20, 0.15, 0.30, 0.15, 0.15, 0.05, 1.00, 0.20, 0.40, 0.45, 0.15, 0.35),
    "camera":   (0.30, 0.25, 0.35, 0.20, 0.20, 0.40, 0.35, 1.00, 0.45, 0.50, 0.10, 0.40),
    "idle":     (0.05, 0.04, 0.10, 0.03, 0.03, 0.02, 0.03, 0.03, 0.15, 0.15, 0.05, 0.12),
}
_MODE_PROBS = {"game": 0.15, "ml_infer": 0.15, "compile": 0.15, "video": 0.20,
               "camera": 0.10, "idle": 0.25}


def soc_units() -> list[dict]:
    """Block list as solver unit dicts (metres), tiled left-to-right rows.

    The initial tiling is arbitrary (the placer moves blocks); only sizes and
    the chip bounding box matter.
    """
    units, x, y, row_h, chip_w = [], 0.0, 0.0, 0.0, 11.5e-3
    for name, w_mm, h_mm, _ in _BLOCKS:
        w, h = w_mm * 1e-3, h_mm * 1e-3
        if x + w > chip_w:
            x, y = 0.0, y + row_h
            row_h = 0.0
        units.append(dict(name=name, width=w, height=h, leftx=x, bottomy=y))
        x += w
        row_h = max(row_h, h)
    return units


class HeteroSoCWorkloadModel:
    """Mode-mixture sampler over the stylised SoC (same API as the other
    workload models: ``sample(n) -> list of power arrays in units order``)."""

    def __init__(self, units: list[dict], seed: int = 0, noise: float = 0.15):
        self.units = units
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        # units come from soc_units() in _BLOCKS order; verify, don't assume.
        block_names = [b[0] for b in _BLOCKS]
        assert [u["name"] for u in units] == block_names, \
            "units must be soc_units() output (order defines the mode vectors)"
        self.pmax = np.array([b[3] for b in _BLOCKS])
        self.mode_names = list(_MODES)
        self.modes = np.array([_MODES[m] for m in self.mode_names])
        self.mode_p = np.array([_MODE_PROBS[m] for m in self.mode_names])

    def sample(self, n: int) -> list[np.ndarray]:
        out = []
        for _ in range(n):
            m = self.rng.choice(len(self.mode_names), p=self.mode_p)
            p = self.modes[m] * self.pmax
            p = p * (1.0 + self.rng.uniform(-self.noise, self.noise, size=len(p)))
            out.append(np.maximum(p, 1e-4))
        return out

    def engine_stats(self, n: int = 4000, seed: int = 12345) -> dict:
        """Confound/regime statistics to print with any exp023-style run."""
        rng = np.random.default_rng(seed)
        saved, self.rng = self.rng, rng
        try:
            P = np.array(self.sample(n))
        finally:
            self.rng = saved
        tot = P.sum(1)
        heavy = [i for i, b in enumerate(_BLOCKS) if b[3] >= 7.0]
        shares = P[:, heavy].max(1) / tot
        return {"e_total": float(tot.mean()), "total_cv": float(tot.std() / tot.mean()),
                "heaviest_engine_share_mean": float(shares.mean())}
