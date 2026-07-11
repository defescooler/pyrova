"""Workload model that resamples real .ptrace timesteps."""

from __future__ import annotations
from pathlib import Path

import numpy as np

from pyrova.core.io import parse_ptrace


def _read_ptrace(path: str):
    """(block_names, data[T, B]) [W] from a .ptrace file."""
    names, rows = parse_ptrace(path)
    return names, np.asarray(rows, dtype=float)


class RealTraceWorkloadModel:
    """Sample power scenarios by drawing timestep rows from validated .ptrace files."""

    def __init__(self, ptrace_paths: list[str], flp_units: list[dict], seed: int = 0):
        """Each ptrace's block-name set must equal the floorplan's; columns realign to `flp_units` order."""
        unit_names = [u["name"] for u in flp_units]
        uset = set(unit_names)
        mats = []
        for p in ptrace_paths:
            names, data = _read_ptrace(p)
            if set(names) != uset or len(names) != len(unit_names):
                raise ValueError(f"{Path(p).name}: block names do not match the floorplan "
                                 f"({len(names)} cols vs {len(unit_names)} units)")
            perm = [names.index(nm) for nm in unit_names]
            mats.append(data[:, perm])
        self.data = np.vstack(mats)
        self.rng = np.random.default_rng(seed)

    @property
    def n_scenarios(self) -> int:
        return int(self.data.shape[0])

    @property
    def block_correlation(self) -> np.ndarray:
        return np.corrcoef(self.data.T)

    def sample(self, n: int) -> list[np.ndarray]:
        """n per-block power arrays [W] in `flp_units` order; resamples with replacement when n > n_scenarios."""
        idx = self.rng.choice(self.n_scenarios, size=n, replace=(n > self.n_scenarios))
        return [self.data[i].copy() for i in idx]
