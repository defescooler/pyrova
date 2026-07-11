"""Real BOOM (RISC-V) per-functional-unit power across 80 benchmarks."""

from __future__ import annotations
import csv
import re
from pathlib import Path

import numpy as np

# BP omitted: it is the PARENT of GP/L1_LP/L2_LP/Chooser/RAS — including it would double-count.
LEAF_FAM = {
    "FP_RRAT": "FP", "FP_List": "FP", "FP_RF": "FP", "FP_Win": "FP", "FPU": "FP",
    "Int_RRAT": "INT", "Int_RF": "INT", "Int_ALU": "INT", "Com_ALU": "INT", "Inst_Win": "INT",
    "IC": "MEM", "DC": "MEM", "L1_LP": "MEM", "L2_LP": "MEM", "LDQ": "MEM", "STQ": "MEM",
    "Itlb": "MEM", "Dtlb": "MEM",
    "BTB": "CTRL", "GP": "CTRL", "Chooser": "CTRL", "RAS": "CTRL", "Inst_Buff": "CTRL",
    "Inst_Dec": "CTRL", "Free_List": "CTRL", "ROB": "CTRL", "Rslt_BB": "CTRL",
}

# Report names carry suffixes like "(FPUs) (Count: 1 )", so match leaves by prefix.
_PFX = {
    "Instruction Cache": "IC", "Branch Target Buffer": "BTB", "Global Predictor": "GP",
    "L1_Local Predictor": "L1_LP", "L2_Local Predictor": "L2_LP", "Chooser": "Chooser",
    "RAS": "RAS", "Instruction Buffer": "Inst_Buff", "Instruction Decoder": "Inst_Dec",
    "Int Front End RAT": "Int_RRAT", "FP Front End RAT": "FP_RRAT", "FP Free List": "FP_List",
    "Free List": "Free_List", "Data Cache": "DC", "LoadQ": "LDQ", "StoreQ": "STQ",
    "Itlb": "Itlb", "Dtlb": "Dtlb", "Integer RF": "Int_RF", "Floating Point RF": "FP_RF",
    "FP Instruction Window": "FP_Win", "Instruction Window": "Inst_Win", "ROB": "ROB",
    "Integer ALUs": "Int_ALU", "Floating Point Units": "FPU", "Complex ALUs": "Com_ALU",
    "Results Broadcast Bus": "Rslt_BB",
}


def _parse_areas(rpt_path: str) -> dict[str, float]:
    """Per-leaf area [m^2] from a McPAT .rpt (matches report names by prefix)."""
    keys = sorted(_PFX, key=len, reverse=True)
    lines = Path(rpt_path).read_text().splitlines()
    area: dict[str, float] = {}
    for i, ln in enumerate(lines):
        nm = ln.strip().rstrip(":").strip()
        for k in keys:
            if nm.startswith(k):
                for j in range(i + 1, min(i + 5, len(lines))):
                    m = re.search(r"Area = ([\d.eE+-]+) mm\^2", lines[j])
                    if m:
                        area[_PFX[k]] = float(m.group(1)) * 1e-6
                        break
                break
    area.setdefault("Rslt_BB", 1e-9)          # broadcast bus area negligible / sometimes absent
    return area


def resolve_paths(boom_dir: str | None = None) -> tuple[str, str] | tuple[None, None]:
    """Find feature-demo.csv + mcpat.rpt under a cloned mcpat-calib repo, or (None, None)."""
    import os
    root = Path(__file__).resolve().parents[2]
    roots = [boom_dir, os.environ.get("BOOM_DATA"),
             root / "Tools/mcpat-calib-public", root / "mcpat-calib-public",
             "Tools/mcpat-calib-public", "mcpat-calib-public"]
    for r in roots:
        if not r:
            continue
        base = Path(r)
        csvp = base / "boom-data/train_data/feature-demo.csv"
        rptp = base / "boom-data/smallboom/dhrystone/mcpat.rpt"
        if csvp.exists() and rptp.exists():
            return str(csvp), str(rptp)
    return None, None


class BoomWorkload:
    """80 real BOOM benchmarks as power scenarios over ~27 functional blocks."""

    def __init__(self, csv_path: str, rpt_path: str, config_id: str = "0", ncol: int = 6):
        self.leaves = list(LEAF_FAM)
        self.families = [LEAF_FAM[c] for c in self.leaves]
        with open(csv_path) as fh:
            rows = [r for r in csv.DictReader(fh) if r["Config_ID"] == str(config_id)]
        if not rows:
            raise ValueError(f"no rows for config {config_id}")
        self.power = np.array([[float(r[c + ".Dynamic"]) for c in self.leaves] for r in rows])
        self.core = np.array([float(r["Core.Dynamic"]) for r in rows])
        cov = float((self.power.sum(1) / self.core).mean())
        if not 0.90 <= cov <= 1.05:
            raise ValueError(
                f"leaf power covers {cov:.3f} of Core.Dynamic — expected ~1.0; "
                "a parent/child component is missing or double-counted")
        area = _parse_areas(rpt_path)
        self.area = np.array([area[c] for c in self.leaves])
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

    @property
    def coverage(self) -> float:
        """sum(leaf power)/Core power — ~1.0 confirms no parent/child double-count."""
        return float((self.power.sum(1) / self.core).mean())

    def scale_to_peak(self, scenario_peaks_fn, target: float) -> None:
        """Rescale all power (and `core`, to keep `coverage` valid) so mean per-program peak dT equals `target` [K]."""
        base = scenario_peaks_fn(self.scenarios()).mean()
        k = target / base
        self.power = self.power * k
        self.core = self.core * k

    def scenarios(self) -> list[np.ndarray]:
        """Per-program power vectors (units order), one array per benchmark."""
        return [self.power[i].copy() for i in range(len(self.power))]

    def family_corr(self) -> dict[str, float]:
        """Cross-PROGRAM correlation between functional-cluster total power."""
        f = np.array(self.families)
        fp = {k: self.power[:, f == k].sum(1) for k in ("FP", "INT", "MEM")}
        c = lambda a, b: float(np.corrcoef(fp[a], fp[b])[0, 1])
        return {"FP_INT": c("FP", "INT"), "FP_MEM": c("FP", "MEM"), "INT_MEM": c("INT", "MEM")}

    def total_power_cv(self) -> float:
        t = self.power.sum(1)
        return float(t.std() / t.mean())
