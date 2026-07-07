#!/usr/bin/env python3
"""Visualize block temperatures on a .flp floorplan."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

from pyrova.core.io import parse_flp


def read_flp(path: Path) -> list[dict]:
    return parse_flp(str(path))


def read_steady(path: Path) -> dict[str, float]:
    temps = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, temp, *_ = line.split()
            temps[name] = float(temp)
    return temps


def read_ttrace(path: Path, frame: int) -> dict[str, float]:
    with path.open() as f:
        header = f.readline().strip().split()
        rows = [line.strip().split() for line in f if line.strip()]

    if not rows:
        raise ValueError(f"{path} has no temperature frames")

    if frame < 0:
        frame = len(rows) + frame
    if frame < 0 or frame >= len(rows):
        raise ValueError(f"frame {frame} out of range; file has {len(rows)} frames")

    values = [float(v) for v in rows[frame]]
    return dict(zip(header, values))


def plot(units: list[dict], temps: dict[str, float], out: Path, title: str) -> None:
    block_temps = [temps[u["name"]] for u in units if u["name"] in temps]
    if not block_temps:
        raise ValueError("No floorplan unit names matched the temperature file")

    t_min = min(block_temps)
    t_max = max(block_temps)
    norm = colors.Normalize(vmin=t_min, vmax=t_max)
    cmap = plt.get_cmap("inferno")

    width = max(u["leftx"] + u["width"] for u in units)
    height = max(u["bottomy"] + u["height"] for u in units)

    fig, ax = plt.subplots(figsize=(9, 8))
    for u in units:
        temp = temps.get(u["name"])
        face = cmap(norm(temp)) if temp is not None else "lightgray"
        rect = Rectangle(
            (u["leftx"], u["bottomy"]),
            u["width"],
            u["height"],
            facecolor=face,
            edgecolor="black",
            linewidth=0.7,
        )
        ax.add_patch(rect)

        area = u["width"] * u["height"]
        if area > 5e-7:
            ax.text(
                u["leftx"] + u["width"] / 2,
                u["bottomy"] + u["height"] / 2,
                u["name"],
                ha="center",
                va="center",
                fontsize=6,
                color="white" if temp is not None and norm(temp) > 0.55 else "black",
            )

    hottest = max((u for u in units if u["name"] in temps), key=lambda u: temps[u["name"]])
    hot_temp = temps[hottest["name"]]
    ax.set_title(f"{title}\nHottest block: {hottest['name']} = {hot_temp:.2f} K")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature [K]")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flp", required=True, type=Path)
    parser.add_argument("--steady", type=Path)
    parser.add_argument("--ttrace", type=Path)
    parser.add_argument("--frame", type=int, default=-1, help="ttrace frame index; -1 means last")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if bool(args.steady) == bool(args.ttrace):
        raise SystemExit("Pass exactly one of --steady or --ttrace")

    units = read_flp(args.flp)
    if args.steady:
        temps = read_steady(args.steady)
        title = f"Steady-state block temperatures ({args.steady.name})"
    else:
        temps = read_ttrace(args.ttrace, args.frame)
        title = f"Transient block temperatures ({args.ttrace.name}, frame {args.frame})"

    plot(units, temps, args.out, title)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
