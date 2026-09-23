"""Small scenarios on a cleared patch of a stock map, for fast RL iteration.

A scenario removes every pre-placed unit (bases, creeps, mines), clears trees
around its center, spawns the configured units and resets in-game in
milliseconds. Unit positions are relative to the scenario center, which by
default is the most open walkable area of the map.

    Scenario.skirmish(["hfoo"] * 4, ["ogru"] * 3)      # micro: kill the other side
    Scenario.move_to_target("hfoo", distance=1200)       # navigation: reach a point fast
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

from .data.mapbuild import SpawnSpec, stock_map_path
from .data.terrain import load_terrain, open_area_center


@dataclass(frozen=True)
class Scenario:
    units: tuple[SpawnSpec, ...]  # positions relative to the center
    map: str = "(2)EchoIsles"
    center: tuple[float, float] | None = None  # None: the map's most open walkable spot
    clear_radius: float = 1400.0
    victory: str = "elimination"  # elimination | none
    max_game_seconds: float = 90.0
    resources: tuple[tuple[int, int, int], ...] = ()
    target: tuple[float, float, float] | None = None  # (dx, dy, radius) relative to the center, for navigation
    name: str = "scenario"

    def resolved_center(self) -> tuple[float, float]:
        if self.center is not None:
            return self.center
        return _auto_center(self.map)

    def absolute_units(self) -> tuple[SpawnSpec, ...]:
        cx, cy = self.resolved_center()
        return tuple(SpawnSpec(u.player, u.unit, cx + u.x, cy + u.y, u.facing) for u in self.units)

    def absolute_target(self) -> tuple[float, float, float] | None:
        if self.target is None:
            return None
        cx, cy = self.resolved_center()
        return cx + self.target[0], cy + self.target[1], self.target[2]

    # ---- ready-made scenarios -------------------------------------------------------------

    @classmethod
    def skirmish(cls, player0: Sequence[str], player1: Sequence[str], separation: float = 700.0,
                 spacing: float = 90.0, max_game_seconds: float = 90.0, **kw) -> "Scenario":
        """Two groups facing each other `separation` apart; last side standing wins."""
        units = []
        for player, codes, side in ((0, player0, -1), (1, player1, 1)):
            cols = max(1, math.ceil(math.sqrt(len(codes))))
            for i, code in enumerate(codes):
                row, col = divmod(i, cols)
                x = side * (separation / 2 + row * spacing)
                y = (col - (cols - 1) / 2) * spacing
                units.append(SpawnSpec(player, code, x, y, 180.0 if side > 0 else 0.0))
        return cls(tuple(units), victory="elimination", max_game_seconds=max_game_seconds,
                   name=f"skirmish_{len(player0)}v{len(player1)}", **kw)

    @classmethod
    def move_to_target(cls, unit: str = "hfoo", distance: float = 1000.0, angle_deg: float = 0.0,
                       radius: float = 100.0, max_game_seconds: float = 30.0, **kw) -> "Scenario":
        """One unit must reach a point `distance` away; success is judged in Python."""
        a = math.radians(angle_deg)
        start = SpawnSpec(0, unit, -distance / 2 * math.cos(a), -distance / 2 * math.sin(a), angle_deg)
        target = (distance / 2 * math.cos(a), distance / 2 * math.sin(a), radius)
        return cls((start,), victory="none", max_game_seconds=max_game_seconds, target=target,
                   name=f"move_{unit}_{int(distance)}", **kw)


@lru_cache(maxsize=None)
def _auto_center(map_name: str) -> tuple[float, float]:
    terrain = load_terrain(stock_map_path(map_name))
    margin = 1024.0
    min_x, min_y = terrain.offset_x + margin, terrain.offset_y + margin
    max_x = terrain.offset_x + (terrain.width - 1) * 128 - margin
    max_y = terrain.offset_y + (terrain.height - 1) * 128 - margin
    x, y, _ = open_area_center(terrain, (min_x, min_y, max_x, max_y))
    return round(x / 32) * 32.0, round(y / 32) * 32.0
