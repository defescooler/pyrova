"""Parsers for the Pyrova input formats (.flp, .config, .ptrace) and floorplan
geometry helpers. Single home for the ``unit dict`` — ``{name, width, height,
leftx, bottomy}`` in metres — the block representation shared by the solver,
placer, workloads, and plots.
"""

from __future__ import annotations


def _data_lines(path: str):
    """Yield stripped, non-blank, non-comment lines of a text file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def parse_flp(path: str) -> list[dict]:
    """Parse a HotSpot ``.flp`` floorplan into unit dicts (extra columns ignored)."""
    units = []
    for line in _data_lines(path):
        parts = line.split()
        units.append({
            "name":    parts[0],
            "width":   float(parts[1]),
            "height":  float(parts[2]),
            "leftx":   float(parts[3]),
            "bottomy": float(parts[4]),
        })
    return units


def parse_desc_connectivity(path: str) -> list[tuple[str, str, float]]:
    """Parse the connectivity section of a HotFloorplan ``.desc`` file.

    The ``.desc`` format has two sections: block area/aspect lines (5 fields:
    ``name area min_aspect max_aspect rotable``) and connectivity lines (3
    fields: ``unit1 unit2 wire_density``). Returns the connectivity edges as
    ``(unit1, unit2, wire_density)``; block lines are ignored (geometry comes
    from the ``.flp``). This is the canonical netlist source (e.g. the bundled
    ``ev6.desc`` is HotSpot's HotFloorplan Alpha-EV6 connectivity)."""
    edges = []
    for line in _data_lines(path):
        parts = line.split()
        if len(parts) == 3:
            edges.append((parts[0], parts[1], float(parts[2])))
    return edges


def parse_config(path: str) -> dict:
    """Parse a ``.config`` file of ``-key value`` lines into ``{key: value}``.

    Values that look numeric are floats; everything else stays a string.
    """
    cfg: dict[str, float | str] = {}
    for line in _data_lines(path):
        line = line.split("#")[0].strip()
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("-"):
            key = parts[0][1:]
            try:
                cfg[key] = float(parts[1])
            except ValueError:
                cfg[key] = parts[1]
    return cfg


def parse_ptrace(path: str) -> tuple[list[str], list[list[float]]]:
    """Parse a whitespace-separated ``.ptrace`` into (block_names, power-rows).

    The first row is treated as a header of block names when it is non-numeric;
    a headerless file gets synthetic ``col<i>`` names. This is the single
    ptrace parser (``workloads.real_traces`` builds on it).
    """
    lines = list(_data_lines(path))
    first = lines[0].split()
    try:
        float(first[0])
        names = [f"col{i}" for i in range(len(first))]
        body = lines
    except ValueError:
        names = first
        body = lines[1:]
    rows = [[float(v) for v in ln.split()] for ln in body]
    return names, rows


def bounding_box(units: list[dict]) -> tuple[float, float, float, float]:
    """Axis-aligned (min_x, min_y, max_x, max_y) of a set of unit dicts."""
    min_x = min(u["leftx"] for u in units)
    min_y = min(u["bottomy"] for u in units)
    max_x = max(u["leftx"] + u["width"] for u in units)
    max_y = max(u["bottomy"] + u["height"] for u in units)
    return min_x, min_y, max_x, max_y


def chip_dimensions(units: list[dict]) -> tuple[float, float]:
    """Chip (width, height) as the floorplan bounding-box extent."""
    min_x, min_y, max_x, max_y = bounding_box(units)
    return max_x - min_x, max_y - min_y
