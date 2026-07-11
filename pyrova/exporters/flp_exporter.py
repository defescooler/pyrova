"""Write a Design to .flp floorplan format."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.design import Design


def export_flp(design: "Design", path: str) -> None:
    """Write `design`'s macros to `path` as whitespace-separated `name width height left_x bottom_y` (metres)."""
    with open(path, "w") as f:
        f.write("# generated floorplan\n")
        f.write("# name\twidth\theight\tleft_x\tbottom_y   (metres)\n")
        for m in design.macros:
            f.write(f"{m.name}\t{m.width:.9f}\t{m.height:.9f}"
                    f"\t{m.left_x:.9f}\t{m.bottom_y:.9f}\n")
