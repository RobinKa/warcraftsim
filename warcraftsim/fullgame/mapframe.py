"""Other maps in one process: each game's coordinates turned into the frame of the run's map, at the game's door.

The features' frame (features.set_geometry: STARTS, the point bins) is process-wide, read in Python and in C (featnative),
and several games share a process (their threads). So a game on another map is not given a frame of its own: FramedGame
sits between GameInstance and play_one and shows everything that holds a position (the units' rows, the start locations,
the trees, the items on the ground) as if the map's bases stood where the run's map's do, and turns the positions in the
commands back. All the code between (the features, the masks, the teachers, the macros, the trackers, late starts) reads
home coordinates; the learner's stored steps carry the features as they were made, and nothing recomputes them.

The transform (MapFrame), the bases' line turned level, as bc.level_frame does for people's games:
* each map's left start (the lower x, as features.set_geometry) lands exactly on the home map's left start, its right start
  on the home map's right one;
* within RIGID of a base, along the line, positions keep their distances (a turn and a shift): the base, its natural, its
  camp. Between those stretches the x along the line is scaled so the bases end up as far apart as at home, so the points'
  bins still read "this far towards the enemy base". Across the line nothing is scaled.

bc.game_frame sets a game's own squeeze instead (the middle's scale differs by map), which needs set_geometry per game: fine
for one encoder reading one game, not for threads. Here the middle's frame scale is home's times `(home D - 2 RIGID) / (D - 2 RIGID)`:
Last Refuge 0.85, Amazonia 1.12, Terenas Stand LV 0.66.

A game on the home map gets no FramedGame at all, and without --map-pool no game does.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache

import numpy as np

RIGID = 3000.0  # (world units from a base along the line where a map keeps its own distances; naturals lie within it)
# (native.py's unit_array columns)
UA_X, UA_Y, UA_FACING = 3, 4, 5


class MapFrame:
    """World positions of a map with starts `starts` <-> the home map's frame (starts `home`: level, left and right)."""

    def __init__(self, starts, home, rigid: float = RIGID):
        (ax, ay), (bx, by) = sorted((float(x), float(y)) for x, y in starts)
        (hx0, hy0), (hx1, hy1) = sorted((float(x), float(y)) for x, y in home)
        if abs(hy0 - hy1) > 1e-6:
            raise ValueError(f"home starts {home}: the frame needs them level")
        self.ang = math.atan2(by - ay, bx - ax)
        self.c, self.s = math.cos(self.ang), math.sin(self.ang)
        self.mx, self.my = (ax + bx) / 2, (ay + by) / 2
        self.hx, self.hy = (hx0 + hx1) / 2, hy0
        self.half, self.home_half = math.hypot(bx - ax, by - ay) / 2, (hx1 - hx0) / 2
        self.a, self.b = self.half - rigid, self.home_half - rigid
        if self.a <= 0 or self.b <= 0:
            raise ValueError(f"starts {starts} (home {home}): closer than {2 * rigid:g}, no middle to scale")
        self.deg = math.degrees(self.ang)

    def to_home(self, x, y):
        """World -> home (arrays or numbers; floats)."""
        dx, dy = np.asarray(x, np.float64) - self.mx, np.asarray(y, np.float64) - self.my
        u, v = self.c * dx + self.s * dy, -self.s * dx + self.c * dy
        au = np.abs(u)
        u = np.where(au >= self.a, np.sign(u) * (au - self.a + self.b), u * self.b / self.a)
        return self.hx + u, self.hy + v

    def to_world1(self, x: float, y: float) -> tuple[float, float]:
        """Home -> world, one point."""
        u, v = float(x) - self.hx, float(y) - self.hy
        au = abs(u)
        u = math.copysign(au - self.b + self.a, u) if au >= self.b else u * self.a / self.b
        return self.mx + self.c * u - self.s * v, self.my + self.s * u + self.c * v

    def to_home1(self, x: float, y: float) -> tuple[float, float]:
        """World -> home, one point (rounded: the protocol's positions are integers)."""
        hx, hy = self.to_home(x, y)
        return int(round(float(hx))), int(round(float(hy)))

    def rows(self, ua: np.ndarray) -> None:
        """A native observation's unit table (int32, native.py's columns), in place."""
        if not len(ua):
            return
        x, y = self.to_home(ua[:, UA_X], ua[:, UA_Y])
        ua[:, UA_X], ua[:, UA_Y] = np.rint(x), np.rint(y)
        ua[:, UA_FACING] = np.mod(np.rint(ua[:, UA_FACING] - self.deg), 360)

    def observation(self, obs):
        """`obs` in home coordinates, in place (a new observation's: GameInstance keeps nothing of it but itself)."""
        ua = getattr(obs, "unit_array", None)
        if ua is not None:
            self.rows(ua)  # (obs.units: lazy over this table, not made yet)
        else:  # (the parser's table keeps its Unit objects between observations: copies)
            obs.units = [replace(u, x=hx, y=hy, facing=int(round(u.facing - self.deg)) % 360)
                         for u in obs.units for hx, hy in (self.to_home1(u.x, u.y),)]
        for p in obs.players.values():
            p.start_x, p.start_y = self.to_home1(p.start_x, p.start_y)
        for d in obs.destructables or ():
            d.x, d.y = self.to_home1(d.x, d.y)
        if getattr(obs, "ground", None):
            obs.ground = [(t, *self.to_home1(x, y), vis, up) for t, x, y, vis, up in obs.ground]
        if getattr(obs, "issued", None):  # (the advisor's orders: kinds 1 and 2 hold a place, bc.level_frame)
            obs.issued = [replace(o, **dict(zip("xy", self.to_home1(o.x, o.y)))) if o.kind in (1, 2) else o for o in obs.issued]
        return obs

    def commands(self, commands) -> list:
        """A step's commands with their positions in the world (a new list: the caller's keep home's, its views read them)."""
        out = []
        for c in commands:
            x = getattr(c, "x", None)
            if x is None:
                out.append(c)
            else:
                wx, wy = self.to_world1(x, c.y)
                out.append(replace(c, x=wx, y=wy))
        return out


class FramedGame:
    """A GameInstance whose observations and commands are in a MapFrame's coordinates; all else is the instance's."""

    def __init__(self, game, frame: MapFrame):
        self._game, self.frame = game, frame

    def __getattr__(self, name: str):
        return getattr(self._game, name)

    def start(self):
        return self.frame.observation(self._game.start())

    def step(self, commands=()):
        return self.frame.observation(self._game.step(self.frame.commands(commands)))

    def restart(self, *args, **kwargs):
        return self.frame.observation(self._game.restart(*args, **kwargs))


def parse_pool(text: str, rules_of: str | None = None) -> list[tuple[str, float]]:
    """--map-pool's form: "echofast2,lastrefugefast2:0.25,amazoniafast2:0.25" -> [(map, weight)] (a map without a weight: 1).
    `rules_of`: every map must play by that map's rules (the run's costs, research steps, mine scale and teacher pace are
    one table each)."""
    pool = []
    for part in text.split(","):
        if part.strip():
            name, _, w = part.strip().partition(":")
            pool.append((name, float(w) if w else 1.0))
    if not pool or any(w < 0 for _, w in pool) or not sum(w for _, w in pool) > 0:
        raise ValueError(f"map pool {text!r}: maps with weights, at least one above 0")
    if rules_of is not None:
        from ..data.duelmap import map_rules
        for name, _ in pool:
            if map_rules(name) != map_rules(rules_of):
                raise ValueError(f"map pool: {name} plays by other rules than {rules_of}")
    return pool


def draw(rng, pool) -> str:
    """A launch's map."""
    return rng.choices([p[0] for p in pool], [p[1] for p in pool])[0]


@lru_cache(None)
def map_starts(map_name: str) -> tuple[tuple[float, float], ...]:
    """A map's start locations (its w3i's players)."""
    from ..data.mapbuild import stock_map_path
    from ..data.mpq import MpqArchive
    from ..data.w3i import parse_w3i
    with MpqArchive(stock_map_path(map_name)) as m:
        info = parse_w3i(m.read("war3map.w3i"))
    return tuple((float(p.start_x), float(p.start_y)) for p in info.players)


@lru_cache(None)
def _frame(map_name: str, home: tuple) -> MapFrame | None:
    starts = map_starts(map_name)
    if len(starts) != 2:
        raise ValueError(f"{map_name}: {len(starts)} start locations; a framed game needs a two-player map")
    if tuple(sorted(starts)) == tuple(sorted(home)):
        return None
    return MapFrame(starts, home)


def frame_for(map_name: str) -> MapFrame | None:
    """The frame a game on `map_name` needs in this process (None: its starts are the process's already)."""
    from . import features as fx
    return _frame(map_name, tuple(tuple(s) for s in fx.STARTS))


def framed(game, map_name: str):
    """`game` as play_one should see it: a FramedGame on a map whose starts are not the process's frame's."""
    frame = frame_for(map_name)
    return game if frame is None else FramedGame(game, frame)
