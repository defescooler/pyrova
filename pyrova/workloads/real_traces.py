"""Workload model that resamples real Wattch-generated .ptrace timesteps."""

from __future__ import annotations
from pathlib import Path

import numpy as np

from pyrova.core.io import parse_ptrace


def _read_ptrace(path: str):
    """Return (block_names, data[T,B]) via the shared ``core.io`` parser."""
    names, rows = parse_ptrace(path)
    return names, np.asarray(rows, dtype=float)


class RealTraceWorkloadModel:
    """Sample power scenarios by drawing timestep rows from validated .ptrace files."""

    def __init__(self, ptrace_paths: list[str], flp_units: list[dict], seed: int = 0):
        unit_names = [u["name"] for u in flp_units]
        uset = set(unit_names)
        mats = []
        for p in ptrace_paths:
            names, data = _read_ptrace(p)
            if set(names) != uset or len(names) != len(unit_names):
                raise ValueError(f"{Path(p).name}: block names do not match the floorplan "
                                 f"({len(names)} cols vs {len(unit_names)} units)")
            perm = [names.index(nm) for nm in unit_names]      # align columns to units order
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
        """Return n power arrays (units order); resamples with replacement if n > n_scenarios."""
        idx = self.rng.choice(self.n_scenarios, size=n, replace=(n > self.n_scenarios))
        return [self.data[i].copy() for i in idx]
