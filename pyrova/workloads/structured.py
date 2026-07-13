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
    """Sample power scenarios from a mixture of CPU operating modes; ``total_power`` is the full-activity total, not the mean chip power."""

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
        """Noise-free mean power vector for a named mode, (n_units,) [W] in units order."""
        w = np.array([MODES[mode][f] for f in self.families])
        return w * self.scale * self.total_power

    def sample(self, n: int) -> list[np.ndarray]:
        """Return n power vectors, each (n_units,) [W] in units order; same format as random_power_map."""
        out = []
        for _ in range(n):
            mode = self.mode_names[self.rng.choice(len(self.mode_names), p=self.mode_p)]
            p = self.mode_power(mode)
            p = p * (1.0 + self.rng.uniform(-self.noise, self.noise, size=len(p)))
            out.append(p)
        return out


class CorrelatedWorkloadModel:
    """Discrete-mode sampler with a cross-cluster correlation knob `mix` in [0,1]; a `mix` sweep confounds correlation with total-power CV (see `mix_stats`)."""

    # one cluster hot at a time -> functional clusters anti-correlate
    CONTRAST = (
        {"FP": 1.00, "INT": 0.20, "MEM": 0.10, "CTRL": 0.30},
        {"FP": 0.20, "INT": 1.00, "MEM": 0.10, "CTRL": 0.30},
        {"FP": 0.10, "INT": 0.20, "MEM": 1.00, "CTRL": 0.30},
    )
    # clusters rise/fall together -> positive correlation
    COMMON = (
        {"FP": 1.00, "INT": 1.00, "MEM": 1.00, "CTRL": 0.50},
        {"FP": 0.12, "INT": 0.12, "MEM": 0.12, "CTRL": 0.10},
    )
    FAMS = ("FP", "INT", "MEM", "CTRL")

    def __init__(self, units: list[dict], mix: float, total_power: float = 110.0,
                 seed: int = 0, noise: float = 0.12):
        self.units = units
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.families = [_family(u["name"]) for u in units]
        areas = np.array([u["width"] * u["height"] for u in units])
        self.weight = np.array([DENSITY[f] for f in self.families]) * areas
        # Fixed scale so E[total] ~= total_power at this mix (mean activity over the mode set
        # that mix selects), isolating the correlation knob from the overall power level.
        mc = {f: float(np.mean([m[f] for m in self.COMMON])) for f in self.FAMS}
        mk = {f: float(np.mean([m[f] for m in self.CONTRAST])) for f in self.FAMS}
        e_act = np.array([self.mix * mc[f] + (1.0 - self.mix) * mk[f] for f in self.families])
        self.scale = total_power / float((e_act * self.weight).sum())

    def _mode_power(self, mode: dict) -> np.ndarray:
        a = np.array([mode[f] for f in self.families])
        return a * self.weight * self.scale

    def sample(self, n: int) -> list[np.ndarray]:
        """Return n power vectors, each (n_units,) [W] in units order; same format as random_power_map."""
        out = []
        for _ in range(n):
            if self.rng.random() < self.mix:
                mode = self.COMMON[self.rng.integers(len(self.COMMON))]
            else:
                mode = self.CONTRAST[self.rng.integers(len(self.CONTRAST))]
            p = self._mode_power(mode)
            p = p * (1.0 + self.rng.uniform(-self.noise, self.noise, size=len(p)))
            out.append(p)
        return out

    def mix_stats(self, n: int = 4000, seed: int = 12345) -> dict[str, float]:
        """Confound quantities the `mix` knob co-varies: E[total], total-power CV, mean hottest-block power."""
        rng = np.random.default_rng(seed)
        saved = self.rng
        self.rng = rng
        try:
            P = np.array(self.sample(n))
        finally:
            self.rng = saved
        tot = P.sum(axis=1)
        return {
            "e_total": float(tot.mean()),
            "total_cv": float(tot.std() / tot.mean()),
            "mean_hot_block_w": float(P.max(axis=1).mean()),
        }
