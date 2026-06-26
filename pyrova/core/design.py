"""Macro, PowerScenario, ThermalConfig, and the Design container they compose."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class Macro:
    """A placed macro / functional block on the chip."""
    name: str
    width: float          # metres
    height: float         # metres
    left_x: float         # metres, from chip origin
    bottom_y: float       # metres, from chip origin

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
        """Return this macro as a solver unit dict."""
        return {
            "name":    self.name,
            "width":   self.width,
            "height":  self.height,
            "leftx":   self.left_x,
            "bottomy": self.bottom_y,
        }


@dataclass
class PowerScenario:
    """One workload scenario: per-macro power in Watts."""
    label: str
    powers: dict[str, float]   # macro_name -> W

    def total(self) -> float:
        return sum(self.powers.values())

    def as_array(self, macro_order: list[str]) -> "np.ndarray":
        import numpy as np
        return np.array([self.powers.get(n, 0.0) for n in macro_order], dtype=np.float64)


@dataclass
class ThermalConfig:
    """
    Thermal stack parameters, mirroring the .config fields the solver uses.
    Defaults match the bundled example config.
    """
    ambient: float = 318.15       # K
    r_convec: float = 0.1         # K/W

    # Layer stack (Si -> TIM -> Spreader -> Heatsink)
    t_chip: float      = 0.00015
    k_chip: float      = 130.0
    t_interface: float = 0.0001
    k_interface: float = 4.0
    t_spreader: float  = 0.001
    k_spreader: float  = 400.0
    s_spreader: float  = 0.02
    t_sink: float      = 0.0069
    k_sink: float      = 400.0
    s_sink: float      = 0.06

    def as_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ThermalConfig":
        import dataclasses
        fields = {f.name for f in dataclasses.fields(ThermalConfig)}
        return ThermalConfig(**{k: v for k, v in d.items() if k in fields})


@dataclass
class Design:
    """Macros, power scenarios, thermal config, and constraints for one design."""
    name: str
    chip_width: float          # metres
    chip_height: float         # metres
    macros: list[Macro]        = field(default_factory=list)
    power_scenarios: list[PowerScenario] = field(default_factory=list)
    thermal_config: ThermalConfig = field(default_factory=ThermalConfig)
    constraints: dict = field(default_factory=lambda: {
        "T_max": None,          # K, None = unconstrained
        "fixed_macros": set(),  # macro names that cannot be moved
    })

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_flp(cls, path: str, name: str | None = None,
                 thermal_config: ThermalConfig | None = None) -> "Design":
        """
        Build a Design from a .flp floorplan file.

        chip_width/height are the floorplan bounding box (max - min per axis).
        If `name` is None it is derived from the filename stem.
        """
        macros: list[Macro] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                macros.append(Macro(
                    name=parts[0],
                    width=float(parts[1]),
                    height=float(parts[2]),
                    left_x=float(parts[3]),
                    bottom_y=float(parts[4]),
                ))
        if not macros:
            raise ValueError(f"No macros parsed from {path}")

        max_x = max(m.left_x + m.width for m in macros)
        max_y = max(m.bottom_y + m.height for m in macros)
        min_x = min(m.left_x for m in macros)
        min_y = min(m.bottom_y for m in macros)
        if name is None:
            name = os.path.splitext(os.path.basename(path))[0]
        return cls(
            name=name,
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

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                #
    # ------------------------------------------------------------------ #

    @property
    def macro_names(self) -> list[str]:
        return [m.name for m in self.macros]

    def macro_by_name(self, name: str) -> Optional[Macro]:
        for m in self.macros:
            if m.name == name:
                return m
        return None

    def with_macros(self, macros: list[Macro]) -> "Design":
        """Return a shallow copy with updated macro positions."""
        import dataclasses
        return dataclasses.replace(self, macros=list(macros))

    def macro_flp_dicts(self) -> list[dict]:
        """Return the macros as solver unit dicts."""
        return [m.as_flp_dict() for m in self.macros]
