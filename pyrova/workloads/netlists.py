"""Netlists (connectivity for HPWL) for the constrained-placement problem."""

from __future__ import annotations
from pathlib import Path

from pyrova.core.io import parse_desc_connectivity

EV6_DESC = Path(__file__).resolve().parent.parent / "inputs/floorplans/ev6.desc"


def nets_by_name(units: list[dict], groups: list[list[str]]) -> list[list[int]]:
    """Map name-groups to index-nets, dropping absent names and <2-pin nets."""
    idx = {u["name"]: i for i, u in enumerate(units)}
    out = []
    for g in groups:
        net = [idx[n] for n in g if n in idx]
        if len(net) >= 2:
            out.append(net)
    return out


def _subblocks(units: list[dict], agg_name: str) -> list[int]:
    """Indices of the split `.flp` blocks an aggregated `.desc` unit maps to: exact name match or `agg_name` + '_<suffix>'."""
    out = []
    for i, u in enumerate(units):
        nm = u["name"]
        if nm == agg_name or nm.startswith(agg_name + "_"):
            out.append(i)
    return out


def ev6_nets(units: list[dict], desc_path: str | Path = EV6_DESC) -> list[list[int]]:
    """Canonical Alpha-EV6 netlist from `ev6.desc`: each connectivity edge becomes one net over the two units' split sub-blocks."""
    edges = parse_desc_connectivity(str(desc_path))
    nets = []
    for a, b, _w in edges:
        net = sorted(set(_subblocks(units, a)) | set(_subblocks(units, b)))
        if len(net) >= 2:
            nets.append(net)
    return nets


def _boom_leaf_units(rpt_path: str) -> dict[str, str]:
    """Map each BOOM leaf component to its Core architectural sub-unit, read from the McPAT report hierarchy (a pre-order traversal)."""
    from pyrova.workloads.boom_traces import _PFX
    from pyrova.core.io import _data_lines
    core_units = ("Instruction Fetch Unit", "Renaming Unit", "Load Store Unit",
                  "Memory Management Unit", "Execution Unit")
    keys = sorted(_PFX, key=len, reverse=True)
    current = None
    out: dict[str, str] = {}
    for line in _data_lines(rpt_path):
        nm = line.strip().rstrip(":").strip()
        if nm in core_units:
            current = nm
            continue
        for k in keys:
            if nm.startswith(k):
                if current is not None:
                    out[_PFX[k]] = current
                break
    return out


def boom_nets(units: list[dict], rpt_path: str) -> list[list[int]]:
    """BOOM netlist derived from the McPAT module hierarchy: one net per Core sub-unit plus a sibling-backbone net."""
    unit_of = _boom_leaf_units(rpt_path)
    idx = {u["name"]: i for i, u in enumerate(units)}
    groups: dict[str, list[int]] = {}
    for leaf, unit in unit_of.items():
        if leaf in idx:
            groups.setdefault(unit, []).append(idx[leaf])
    nets = [sorted(v) for v in groups.values() if len(v) >= 2]
    backbone = sorted(v[0] for v in groups.values() if v)
    if len(backbone) >= 2:
        nets.append(backbone)
    return nets


def soc_nets(units: list[dict]) -> list[list[int]]:
    """Stylised hub-topology netlist for the hetero-SoC where wirelength wants the two heaviest engines adjacent and thermal wants them apart."""
    groups = [
        # SLC cache hub — everyone caches through it
        ["SLC", "CPU_P0", "CPU_P1", "CPU_E", "GPU_0", "GPU_1", "NPU",
         "MediaEng", "ISP"],
        # DDR memory bandwidth users
        ["DDR_PHY", "SLC", "CPU_P0", "GPU_0", "NPU"],
        # coherent CPU cluster
        ["CPU_P0", "CPU_P1", "CPU_E"],
        # GPU halves (tightly bound)
        ["GPU_0", "GPU_1"],
        # uncore / IO fabric
        ["Uncore", "CPU_P0", "Modem_IO", "ISP"],
        # media/camera path
        ["MediaEng", "ISP", "SLC"],
    ]
    return nets_by_name(units, groups)
