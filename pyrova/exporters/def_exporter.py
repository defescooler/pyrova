"""Write macro placements to a minimal floorplan DEF."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.design import Design

# metres -> DBU scale: 1e6 um/m * 1000 DBU/um (1 DBU = 1 nm, standard floorplan-DEF resolution).
DBU_PER_MICRON = 1000
_M_TO_DBU = 1e6 * DBU_PER_MICRON


def export_def(design: "Design", path: str, design_name: str | None = None) -> None:
    """Write `design`'s macro placements to a minimal floorplan DEF at `path`."""
    name = design_name or design.name

    def to_dbu(metres: float) -> int:
        return int(round(metres * _M_TO_DBU))

    chip_w = to_dbu(design.chip_width)
    chip_h = to_dbu(design.chip_height)

    with open(path, "w") as f:
        f.write("# Macro-placement DEF - COMPONENTS only.\n")
        f.write("# Not a complete DEF: no NETS / PINS / ROWS / TRACKS.\n")
        f.write("# Fixed-macro handoff for a downstream place-and-route flow.\n")
        f.write("VERSION 5.8 ;\n")
        f.write(f"DESIGN {name} ;\n")
        f.write(f"UNITS DISTANCE MICRONS {DBU_PER_MICRON} ;\n\n")
        f.write(f"DIEAREA ( 0 0 ) ( {chip_w} {chip_h} ) ;\n\n")
        f.write(f"COMPONENTS {len(design.macros)} ;\n")
        for m in design.macros:
            f.write(f"  - {m.name} {m.name.upper()}\n")
            f.write(f"    + FIXED ( {to_dbu(m.left_x)} {to_dbu(m.bottom_y)} ) N ;\n")
        f.write("END COMPONENTS\n\n")
        f.write("END DESIGN\n")
