"""Netlists (connectivity for HPWL) for the constrained-placement experiment
(exp027 / Problem A).

PROVENANCE — two tiers, kept explicit:
  * ev6_nets: the CANONICAL Alpha-EV6 connectivity from HotSpot's HotFloorplan
    (`inputs/floorplans/ev6.desc`, copied verbatim from the HotSpot
    distribution's example6). This is the netlist the thermal-floorplanning
    literature has cited since 2006 — NOT synthetic. Its 16 aggregated blocks
    are mapped onto the split `ev6.flp` blocks (L2 -> L2_left/L2/L2_right,
    Bpred -> Bpred_0/1/2, ...).
  * soc_nets: HAND-BUILT, STYLISED (the lower "synthetic-configurations" rung).
    Kept only as a secondary sensitivity testbed for the mechanism; every
    result on it is an existence statement, never prevalence.

A net is a list of MACRO INDICES into the given `units` list (solver-unit
dicts). `nets_by_name` maps names to indices and drops names not present.
"""

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
    """Indices of the split `.flp` blocks an aggregated `.desc` unit maps to:
    an exact name match or `agg_name` + '_<suffix>' (e.g. L2 -> L2, L2_left,
    L2_right; Bpred -> Bpred_0/1/2). IntReg matches IntReg_* but not IntMap."""
    out = []
    for i, u in enumerate(units):
        nm = u["name"]
        if nm == agg_name or nm.startswith(agg_name + "_"):
            out.append(i)
    return out


def ev6_nets(units: list[dict], desc_path: str | Path = EV6_DESC) -> list[list[int]]:
    """Canonical Alpha-EV6 netlist from HotFloorplan's `ev6.desc`.

    Each connectivity edge (unitA, unitB) becomes one net over the union of the
    two aggregated units' split sub-blocks in `units`, so two connected
    functional units are pulled together (and their sub-blocks kept compact).
    14 canonical edges. Falls back to nothing for names absent from `units`.
    """
    edges = parse_desc_connectivity(str(desc_path))
    nets = []
    for a, b, _w in edges:
        net = sorted(set(_subblocks(units, a)) | set(_subblocks(units, b)))
        if len(net) >= 2:
            nets.append(net)
    return nets


def _boom_leaf_units(rpt_path: str) -> dict[str, str]:
    """Map each BOOM leaf component -> its Core architectural sub-unit, read from
    the McPAT report hierarchy (a pre-order traversal, so a unit header precedes
    its leaves). Purely structural: no hand-drawn edges."""
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
    """BOOM netlist DERIVED FROM the McPAT module hierarchy (not hand-drawn).

    One net per Core architectural sub-unit (its leaf components communicate
    within the unit), plus one sibling-backbone net connecting a representative
    leaf of each unit (they share the Core parent). `units` are BoomWorkload
    unit dicts (names = McPAT leaves)."""
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
    """Stylised netlist for the hetero-SoC (workloads/hetero_soc.py).

    Hub topology: every compute engine caches through the SLC and streams from
    DDR; the CPU performance cores form a coherent cluster and the two GPU
    halves are tightly bound. The two heaviest engines (GPU halves at 9 W, CPU
    P-cores at 7 W) are exactly the ones wirelength wants adjacent and thermal
    wants apart.
    """
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
