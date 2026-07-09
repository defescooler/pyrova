"""Macro, PowerScenario, ThermalConfig, and the Design container they compose."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .io import bounding_box, parse_flp


@dataclass
class Macro:
    """A placed macro / functional block on the chip (all lengths in metres)."""
    name: str
    width: float
    height: float
    left_x: float          # from chip origin
    bottom_y: float        # from chip origin

    @property
    def centre_x(self) -> float:
        return self.left_x + self.width / 2.0

    @property
    def centre_y(self) -> float:
        return self.bottom_y + self.height / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    def moved(self, left_x: float, bottom_y: float) -> "Macro":
        return Macro(self.name, self.width, self.height, left_x, bottom_y)

    def as_flp_dict(self) -> dict:
        """This macro as a solver unit dict."""
        return {
            "name":    self.name,
            "width":   self.width,
            "height":  self.height,
            "leftx":   self.left_x,
            "bottomy": self.bottom_y,
        }

    @staticmethod
    def from_flp_dict(u: dict) -> "Macro":
        return Macro(u["name"], u["width"], u["height"], u["leftx"], u["bottomy"])


@dataclass
class PowerScenario:
    """One workload scenario: per-macro power in Watts."""
    label: str
    powers: dict[str, float]

    def total(self) -> float:
        return sum(self.powers.values())

    def as_array(self, macro_order: list[str]) -> np.ndarray:
        return np.array([self.powers.get(n, 0.0) for n in macro_order], dtype=np.float64)


@dataclass
class ThermalConfig:
    """Thermal-stack parameters mirroring the ``.config`` fields the solver reads.

    Defaults match the bundled ``inputs/configs/thermal.config`` (and the
    reference solver's built-in defaults) exactly — the experiments and the test
    suite must run on the same stack. If you change one, change the other and
    regenerate the golden snapshot.
    """
    ambient: float = 318.15       # K
    r_convec: float = 0.1         # K/W

    t_chip: float      = 0.00015
    k_chip: float      = 130.0
    t_interface: float = 2.0e-05
    k_interface: float = 4.0
    t_spreader: float  = 0.001
    k_spreader: float  = 400.0
    s_spreader: float  = 0.03
    t_sink: float      = 0.0069
    k_sink: float      = 400.0
    s_sink: float      = 0.06

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ThermalConfig":
        fields = {f.name for f in dataclasses.fields(ThermalConfig)}
        return ThermalConfig(**{k: v for k, v in d.items() if k in fields})


@dataclass
class Design:
    """Macros, power scenarios, thermal config, and constraints for one design."""
    name: str
    chip_width: float          # metres
    chip_height: float         # metres
    macros: list[Macro]        = field(default_factory=list)
    nets: list[list[str]]      = field(default_factory=list)  # connectivity for HPWL
    power_scenarios: list[PowerScenario] = field(default_factory=list)
    thermal_config: ThermalConfig = field(default_factory=ThermalConfig)
    constraints: dict = field(default_factory=lambda: {
        "T_max": None,          # K, None = unconstrained
        "fixed_macros": set(),  # macro names that cannot be moved
    })

    # Constructors

    @classmethod
    def from_flp(cls, path: str, name: str | None = None,
                 thermal_config: ThermalConfig | None = None) -> "Design":
        """Build a Design from a ``.flp`` file; chip size is its bounding box.

        Macros are re-origined to (0, 0): the solver grid spans [0, chip_w] x
        [0, chip_h], and a floorplan whose bounding box starts above the origin
        would otherwise place macros off-grid (their power silently dropped).
        """
        units = parse_flp(path)
        if not units:
            raise ValueError(f"No macros parsed from {path}")
        min_x, min_y, max_x, max_y = bounding_box(units)
        macros = [Macro(u["name"], u["width"], u["height"],
                        u["leftx"] - min_x, u["bottomy"] - min_y)
                  for u in units]
        return cls(
            name=name or os.path.splitext(os.path.basename(path))[0],
            chip_width=max_x - min_x,
            chip_height=max_y - min_y,
            macros=macros,
            thermal_config=thermal_config or ThermalConfig(),
        )

    @classmethod
    def from_def(cls, path: str, lef_path: str | None = None) -> "Design":
        """Build a Design from DEF (+ LEF for macro sizes/pins). Not yet implemented."""
        raise NotImplementedError("Design.from_def() pending DEF/LEF parser")

    @classmethod
    def from_netlist(cls, path: str) -> "Design":
        """Build a Design from a gate-level/RTL netlist. Not yet implemented."""
        raise NotImplementedError("Design.from_netlist() pending netlist front-end")

    # Accessors

    @property
    def macro_names(self) -> list[str]:
        return [m.name for m in self.macros]

    def macro_by_name(self, name: str) -> Optional[Macro]:
        return next((m for m in self.macros if m.name == name), None)

    def with_macros(self, macros: list[Macro]) -> "Design":
        """Shallow copy with updated macro positions."""
        return dataclasses.replace(self, macros=list(macros))

    def macro_flp_dicts(self) -> list[dict]:
        """The macros as solver unit dicts."""
        return [m.as_flp_dict() for m in self.macros]
