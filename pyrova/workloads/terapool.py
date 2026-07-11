"""TeraPool 1024-core cluster: a real-geometry tiled-manycore testbed.

TeraPool (PULP, GF 12 nm, 1024 RV32 cores) is the spatial-tile regime: identical
heavy blocks whose thermal risk comes only from where activity lands. Geometry and
6G-SDR kernel-power ratios are published post-layout — NO SILICON (arXiv:2603.01629
/ 2408.08882); per-tile imbalance (CV anchored to MemPool, arXiv:2303.17742),
traffic, and duty cycles are ASSUMED. Published workloads are SPMD with no
structured per-tile variation, so this is a PREDICTED NULL: only random imbalance
moves the hotspot, which carries no separable tail dimension. Absolute watts are
rescaled to a target peak dT via `calibrate_scale`; only relative structure is claimed.
"""

from __future__ import annotations
import numpy as np

# Geometry [TP]: SubGroup = 3.03 mm^2 at 58% pre-top-routing utilization ->
# 1.74 x 1.74 mm placeable macro; 4 Groups of 4 SubGroups, point-symmetric
# grid with 0.68 mm channels between Groups. Initial tiling: 4x4 SubGroups,
# intra-Group pitch 1.84 mm (0.1 mm ASSUMED intra spacing), 0.68 mm channel
# between Group halves. Cluster 81.8 mm^2 total (~9.05 mm square with ~40%
# routing) — the placement box is the initial-tiling bbox.
_SG_MM = 1.74
_GAP_IN = 0.10   # ASSUMED intra-group spacing
_GAP_GR = 0.68   # inter-group channel [TP]
N_SG = 16

# 6G-SDR kernel set [SDR]. Relative cluster power per kernel DERIVED as
# 1/(GOPS/W) at equal delivered throughput (ASSUMPTION: kernels run at
# comparable OP rates; IPC > 0.6 for all [SDR]) — normalised to beamforming:
#   FFT 125/93=1.344, chest 125/96=1.302, beamf 1.000, matinv 125/61=2.049.
# idle: SPM clock gating cuts idle bank energy 98% [MP]; residual ASSUMED 5%.
_KERNELS = {
    "fft":    1.344,
    "chest":  1.302,
    "beamf":  1.000,
    "matinv": 2.049,
    "idle":   0.050,
}
# Duty cycles not published — PUSCH-pipeline-ish mix ASSUMED.
_KERNEL_PROBS = {"fft": 0.25, "chest": 0.20, "beamf": 0.20, "matinv": 0.15,
                 "idle": 0.20}


def terapool_units() -> list[dict]:
    """16 SubGroup macros (metres), 4x4 with the inter-Group channel gaps.

    Blocks are identical by construction [TP]; the placer may rearrange them
    inside the initial-tiling bbox.
    """
    units = []
    sg = _SG_MM * 1e-3
    for r in range(4):
        for c in range(4):
            x = c * (sg + _GAP_IN * 1e-3) + (_GAP_GR - _GAP_IN) * 1e-3 * (c >= 2)
            y = r * (sg + _GAP_IN * 1e-3) + (_GAP_GR - _GAP_IN) * 1e-3 * (r >= 2)
            units.append(dict(name=f"SG_{r}{c}", width=sg, height=sg,
                              leftx=x, bottomy=y))
    return units


class TeraPoolWorkloadModel:
    """Kernel-mixture sampler with per-SubGroup random load imbalance (same
    API as the other workload models: ``sample(n) -> list of power arrays``).

    Per scenario: kernel k (mixture), traffic level t ~ U(0.4, 1.0) [ASSUMED
    base-station load variation], per-SubGroup imbalance multipliers m_i
    (lognormal, CV=`imbalance_cv`, renormalised to mean 1 so imbalance moves
    power around without changing the total [MP: imbalance redistributes
    work]). Power_i = rel[k] * t * m_i / N_SG, times `scale`.
    """

    def __init__(self, units: list[dict], seed: int = 0,
                 imbalance_cv: float = 0.15, noise: float = 0.05):
        assert [u["name"] for u in units] == [f"SG_{r}{c}" for r in range(4)
                                              for c in range(4)], \
            "units must be terapool_units() output"
        self.units = units
        # CV anchored to [MP]: 3% (ray tracing) to 17% (BFS) speedup lost to
        # imbalance; 0.15 default ASSUMED within that range.
        self.imbalance_cv = imbalance_cv
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.kernel_names = list(_KERNELS)
        self.rel = np.array([_KERNELS[k] for k in self.kernel_names])
        self.kernel_p = np.array([_KERNEL_PROBS[k] for k in self.kernel_names])
        self.scale = 1.0

    def calibrate_scale(self, scenario_peaks_fn, target: float) -> float:
        """Set `scale` so the probability-weighted mean peak dT of the pure
        uniform kernel vectors equals `target` [K] (linear solver => exact)."""
        pure = [np.full(N_SG, r / N_SG) for r in self.rel]
        base = scenario_peaks_fn(pure)
        self.scale = float(target / np.dot(self.kernel_p, base))
        return self.scale

    def sample(self, n: int) -> list[np.ndarray]:
        sig = np.sqrt(np.log(1.0 + self.imbalance_cv ** 2))
        out = []
        for _ in range(n):
            k = self.rng.choice(len(self.kernel_names), p=self.kernel_p)
            t = self.rng.uniform(0.4, 1.0)
            m = self.rng.lognormal(-0.5 * sig * sig, sig, size=N_SG)
            m = m / m.mean()
            p = self.rel[k] * t * m / N_SG * self.scale
            p = p * (1.0 + self.rng.uniform(-self.noise, self.noise, size=N_SG))
            out.append(np.maximum(p, 1e-9))
        return out

    def regime_stats(self, n: int = 4000, seed: int = 12345) -> dict:
        """Regime statistics to report with any placement comparison on this
        model."""
        rng = np.random.default_rng(seed)
        saved, self.rng = self.rng, rng
        try:
            P = np.array(self.sample(n))
        finally:
            self.rng = saved
        tot = P.sum(1)
        return {"e_total": float(tot.mean()),
                "total_cv": float(tot.std() / tot.mean()),
                "hot_sg_share_mean": float((P.max(1) / tot).mean()),
                "argmax_entropy_bits": float(
                    -(np.bincount(P.argmax(1), minlength=N_SG) / n
                      * np.log2(np.maximum(np.bincount(P.argmax(1),
                                                       minlength=N_SG) / n,
                                           1e-12))).sum())}
