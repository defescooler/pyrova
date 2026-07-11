"""Edinburgh/Tomusk gem5+McPAT per-component power across real programs."""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np

# Longest matching prefix wins. Parents (IFU/EXU/Renaming/LSU/MMU/Register Files/Scheduler/Branch Predictor) omitted: they would double-count leaf power.
PFX_FAM = {
    "Instruction Cache": "MEM", "Branch Target Buffer": "CTRL",
    "Global Predictor": "CTRL", "L1_Local Predictor": "CTRL",
    "L2_Local Predictor": "CTRL", "Local Predictor": "CTRL",
    "Chooser": "CTRL", "RAS": "CTRL",
    "Instruction Buffer": "CTRL", "Instruction Decoder": "CTRL",
    "Int Front End RAT": "INT", "FP Front End RAT": "FP",
    "FP Free List": "FP", "Free List": "CTRL",
    "Data Cache": "MEM", "LoadQ": "MEM", "StoreQ": "MEM",
    "Itlb": "MEM", "Dtlb": "MEM",
    "Integer RF": "INT", "Floating Point RF": "FP",
    "FP Instruction Window": "FP", "Instruction Window": "INT",
    "ROB": "CTRL", "Integer ALUs": "INT", "Floating Point Units": "FP",
    "Complex ALUs": "INT", "Results Broadcast Bus": "CTRL",
}
_KEYS = sorted(PFX_FAM, key=len, reverse=True)
_RUN_RE = re.compile(r"Runtime Dynamic = ([\d.eE+-]+) W")
_AREA_RE = re.compile(r"Area = ([\d.eE+-]+) mm\^2")
_CORE_RE = re.compile(r"^\s*Core:")


def parse_mcpat_report(path: str | Path) -> tuple[dict[str, float], dict[str, float], float]:
    """(leaf_runtime_W, leaf_area_m2, core_runtime_W) from a print-level-5 report."""
    power: dict[str, float] = {}
    area: dict[str, float] = {}
    core_w = float("nan")
    current: str | None = None
    want_core = False
    for ln in Path(path).read_text(errors="replace").splitlines():
        stripped = ln.strip().rstrip(":").strip()
        matched = None
        for k in _KEYS:
            if stripped.startswith(k):
                matched = k
                break
        if matched is not None:
            current = matched
        elif _CORE_RE.match(ln):
            want_core = True
        m = _AREA_RE.search(ln)
        if m and current is not None and current not in area:
            area[current] = float(m.group(1)) * 1e-6
        m = _RUN_RE.search(ln)
        if m:
            if current is not None:
                power.setdefault(current, float(m.group(1)))
                current = None
            elif want_core and np.isnan(core_w):
                core_w = float(m.group(1))
                want_core = False
    return power, area, core_w


class TomuskWorkload:
    """N real programs (SPEC INT / FPMark / DENBench) as power scenarios over the McPAT leaf components."""

    def __init__(self, extract_root: str | Path, config_id: str,
                 suites: tuple[str, ...] = ("spi", "fp", "de"), ncol: int = 6,
                 coverage_band: tuple[float, float] = (0.85, 1.05)):
        root = Path(extract_root)
        self.config_id = str(config_id)
        rows: list[tuple[str, dict[str, float]]] = []
        area_ref: dict[str, float] | None = None
        coverages = []
        for rep in sorted(root.glob(f"exp2_*_0/{self.config_id}/mcpat_report")):
            run = rep.parent.parent.name           # exp2_spi1_xalan_0
            m = re.match(r"exp2_(\w+?)1_(.+)_0$", run)
            if not m or m.group(1) not in suites:
                continue
            power, area, core_w = parse_mcpat_report(rep)
            rows.append((f"{m.group(1)}:{m.group(2)}", power))
            if area_ref is None and area:
                area_ref = area
            if np.isfinite(core_w) and core_w > 0:
                coverages.append(sum(power.values()) / core_w)
        if not rows:
            raise ValueError(f"no mcpat_report under {root} for config {self.config_id}")
        self.names = [n for n, _ in rows]
        self.leaves = sorted({k for _, p in rows for k in p})
        self.families = [PFX_FAM[l] for l in self.leaves]
        self.power = np.array([[p.get(l, 0.0) for l in self.leaves] for _, p in rows])

        self.coverage = float(np.mean(coverages)) if coverages else float("nan")
        if coverages and not coverage_band[0] <= self.coverage <= coverage_band[1]:
            raise ValueError(
                f"leaf power covers {self.coverage:.3f} of Core runtime dynamic — "
                "component taxonomy is missing or double-counting something")

        if area_ref is None:
            raise ValueError("no Area lines parsed — is the report print level 5?")
        self.area = np.array([area_ref.get(l, 1e-9) for l in self.leaves])
        side = np.sqrt(self.area)
        cw = float(side.max()) * 1.15
        self.units = [dict(name=self.leaves[i], width=float(side[i]), height=float(side[i]),
                           leftx=float((i % ncol) * cw), bottomy=float((i // ncol) * cw))
                      for i in range(len(self.leaves))]
        self.chip_w = ncol * cw
        self.chip_h = ((len(self.leaves) + ncol - 1) // ncol) * cw

    @property
    def n_programs(self) -> int:
        return len(self.power)

    def scenarios(self) -> list[np.ndarray]:
        """Per-program power vectors (units order), one array per benchmark."""
        return [self.power[i].copy() for i in range(len(self.power))]

    def scale_to_peak(self, scenario_peaks_fn, target: float) -> None:
        """Rescale all power so the mean per-program peak dT equals `target` [K]."""
        base = scenario_peaks_fn(self.scenarios()).mean()
        self.power = self.power * (target / base)

    def family_corr(self) -> dict[str, float]:
        """Cross-PROGRAM correlation between functional-cluster total power."""
        f = np.array(self.families)
        fp = {k: self.power[:, f == k].sum(1) for k in ("FP", "INT", "MEM")}
        c = lambda a, b: float(np.corrcoef(fp[a], fp[b])[0, 1])
        return {"FP_INT": c("FP", "INT"), "FP_MEM": c("FP", "MEM"),
                "INT_MEM": c("INT", "MEM")}

    def total_power_cv(self) -> float:
        t = self.power.sum(1)
        return float(t.std() / t.mean())

    def fp_share(self) -> tuple[float, float]:
        """(mean, max) share of total power carried by the FP cluster."""
        f = np.array(self.families)
        share = self.power[:, f == "FP"].sum(1) / self.power.sum(1)
        return float(share.mean()), float(share.max())

    def fpu_floor_ratio(self) -> float:
        """FP-cluster power on SPEC-INT / on FPMark programs; ~1.0 means the idle-FPU floor dominates (artifact, not signal)."""
        f = np.array(self.families)
        fp = self.power[:, f == "FP"].sum(1)
        spec = [i for i, n in enumerate(self.names) if n.startswith("spi:")]
        fpm = [i for i, n in enumerate(self.names) if n.startswith("fp:")]
        if not spec or not fpm:
            return float("nan")
        return float(fp[spec].mean() / max(fp[fpm].mean(), 1e-12))
