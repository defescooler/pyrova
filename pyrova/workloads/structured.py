"""Mode-mixture workload model with anti-correlated functional-unit activity."""

from __future__ import annotations
import numpy as np


def _family(name: str) -> str:
    """Map an ev6 block name to a functional family."""
    if name.startswith("FP"):
        return "FP"
    if name.startswith("Int"):
        return "INT"
    if name.startswith("L2") or name in ("Icache", "Dcache"):
        return "MEM"
    return "CTRL"          # Bpred, DTB, ITB, LdStQ


# Power density per family (W/m^2): logic is small and hot, cache is large and cool.
DENSITY = {"FP": 6.0e6, "INT": 6.0e6, "CTRL": 1.5e6, "MEM": 0.18e6}

# Per-mode relative activity per family. FP is anti-correlated with INT and MEM:
# compute_fp lights FP while INT/MEM are cold, and vice versa.
MODES = {
    "idle":        {"FP": 0.03, "INT": 0.03, "MEM": 0.05, "CTRL": 0.05},
    "compute_fp":  {"FP": 1.00, "INT": 0.25, "MEM": 0.10, "CTRL": 0.30},
    "compute_int": {"FP": 0.05, "INT": 1.00, "MEM": 0.15, "CTRL": 0.35},
    "memory":      {"FP": 0.05, "INT": 0.15, "MEM": 1.00, "CTRL": 0.35},
    "mixed":       {"FP": 0.50, "INT": 0.50, "MEM": 0.50, "CTRL": 0.50},
}
MODE_PROBS = {"idle": 0.15, "compute_fp": 0.25, "compute_int": 0.25,
              "memory": 0.20, "mixed": 0.15}


class StructuredWorkloadModel:
    """Sample power scenarios from a mixture of CPU operating modes."""

    def __init__(self, units: list[dict], total_power: float = 110.0,
                 seed: int = 0, noise: float = 0.12):
        self.units = units
        self.total_power = total_power
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.families = [_family(u["name"]) for u in units]
        areas = np.array([u["width"] * u["height"] for u in units])
        peak = np.array([DENSITY[f] for f in self.families]) * areas
        self.scale = peak / peak.sum()                  # full-activity power fractions
        self.mode_names = list(MODES)
        self.mode_p = np.array([MODE_PROBS[m] for m in self.mode_names])

    def mode_power(self, mode: str) -> np.ndarray:
        """Noise-free mean power vector for a named mode (units order)."""
        w = np.array([MODES[mode][f] for f in self.families])
        return w * self.scale * self.total_power

    def sample(self, n: int) -> list[np.ndarray]:
        """Return n power arrays (units order), same format as random_power_map."""
        out = []
        for _ in range(n):
            mode = self.mode_names[self.rng.choice(len(self.mode_names), p=self.mode_p)]
            p = self.mode_power(mode)
            p = p * (1.0 + self.rng.uniform(-self.noise, self.noise, size=len(p)))
            out.append(p)
        return out
