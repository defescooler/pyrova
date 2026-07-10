"""MCNC floorplanning benchmark loader (YAL): real block geometry + the real
signal netlist, with the die box (utilization) as a free parameter.

Why this testbed: the constrained-payoff question needs (i) a REAL netlist and
(ii) whitespace to allocate — the bundled designs have one or the other, never
both. MCNC ami33 (33 blocks, 1988 benchmark suite) has published geometry and
connectivity, and classic floorplanning leaves the die outline to the tool, so
utilization is legitimately a swept parameter rather than an assumption.

Nets are recovered from the NETWORK section: a signal shared by >= 2 blocks is
one net over those blocks. Power/rail signals (GND/POW/VDD/VSS and the P?G/P?F
rails, which touch nearly every block) are excluded, as HPWL conventionally
covers signal nets only; pad-only signals drop out via the >=2-block rule.

Block dimensions are abstract benchmark units; `load_yal` rescales all blocks
so total block area matches `total_block_area_m2`, then sizes a square die for
the requested utilization. Initial placement is a row tiling (the placer moves
blocks; only sizes, die and nets matter).
"""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np

# rails/globals excluded from HPWL nets (touch ~all blocks; not signal routing)
_POWER_SIGNALS = {"GND", "POW", "VDD", "VSS", "P1G", "P1F", "P2G", "P2F"}


def parse_yal(path: str | Path) -> tuple[dict[str, tuple[float, float]],
                                         list[list[str]]]:
    """(blocks {name: (w, h)} in file units, nets as block-name lists)."""
    text = Path(path).read_text()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)

    blocks: dict[str, tuple[float, float]] = {}
    for m in re.finditer(r"MODULE\s+(\w+)\s*;\s*TYPE\s+(\w+)\s*;.*?DIMENSIONS([^;]+);",
                         text, re.S):
        name = m.group(1)
        if m.group(2).upper() == "PARENT":     # chip bound, not a placeable block
            continue
        xy = [float(v) for v in m.group(3).split()]
        xs, ys = xy[0::2], xy[1::2]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w > 0 and h > 0:
            blocks[name] = (w, h)

    nets: list[list[str]] = []
    net_m = re.search(r"NETWORK\s*;(.*?)ENDNETWORK", text, re.S)
    if net_m:
        # signal -> set of blocks touching it; entries are one statement per ';'
        sig_blocks: dict[str, set[str]] = {}
        for stmt in net_m.group(1).split(";"):
            toks = stmt.split()
            if len(toks) < 3 or not toks[0].startswith("C_"):
                continue
            blk = toks[1]
            if blk not in blocks:
                continue
            for sig in toks[2:]:
                if sig in _POWER_SIGNALS:
                    continue
                sig_blocks.setdefault(sig, set()).add(blk)
        n_blk = len(blocks)
        for sig, bs in sig_blocks.items():
            # >=2 blocks = a routable net; >50% of blocks = a global (clock/rail
            # variant), excluded like the named rails
            if 2 <= len(bs) <= n_blk // 2:
                nets.append(sorted(bs))
    return blocks, nets


def load_yal(path: str | Path, utilization: float = 0.55,
             total_block_area_m2: float = 60e-6):
    """(units, nets_idx, chip_w, chip_h) ready for the solver/placer.

    utilization        : total block area / die area (the whitespace knob)
    total_block_area_m2: silicon scale for the abstract benchmark units;
                         default ~60 mm^2 of blocks (laptop-SoC-class die)
    """
    blocks, nets_names = parse_yal(path)
    names = sorted(blocks)
    raw_area = sum(w * h for w, h in blocks.values())
    s = float(np.sqrt(total_block_area_m2 / raw_area))    # uniform isotropic scale

    die_area = total_block_area_m2 / utilization
    chip = float(np.sqrt(die_area))                       # square die

    # row tiling for the initial placement (placer re-positions everything)
    units, x, y, row_h = [], 0.0, 0.0, 0.0
    for nm in names:
        w, h = blocks[nm][0] * s, blocks[nm][1] * s
        if x + w > chip:
            x, y = 0.0, y + row_h
            row_h = 0.0
        units.append(dict(name=nm, width=w, height=h,
                          leftx=x, bottomy=min(y, chip - h)))
        x += w
        row_h = max(row_h, h)

    idx = {u["name"]: i for i, u in enumerate(units)}
    nets_idx = [[idx[n] for n in net] for net in nets_names]
    return units, nets_idx, chip, chip
