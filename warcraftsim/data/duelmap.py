"""A small melee map for learning the whole game: "duel" (48 x 48 tiles) or "duelN".

Two bases on a flat field, as close as a real map allows (4600 apart; the smallest stock
two-player maps are 80 x 80 tiles with bases 8400-9500 apart), laid out like Echo Isles' main
bases so the built-in AI plays as it does there:

* a start location per player at (-2300, 0) and (2300, 0), mirrored; a 12500-gold mine ~720 from
  each, where Echo Isles has its main mines (781);
* a wall of trees three deep along the map's edge, ~450 behind each start location (Echo Isles' nearest
  trees: 550-700): lumber, and the map's border;
* creep camps for the heroes' experience, all farther than 1500 from the start locations (melee
  initialization removes creeps within 1500 of them): kobolds near each base, ogres and trolls to
  the north and gnolls to the south of the middle. No items, shops or expansions.

"duelfast" (FAST rules): the same map with hit points halved, build, train and research times
cut to a third and costs halved for every unit type and upgrade (Rules, through the map's object
data); with a 50% handicap, units have a quarter of their hit points.

The terrain is flat open land (data/flatmap.py). Mines, creeps and trees are made by the map
script (a function called where the stock script made its pre-placed units), so the game applies
their pathing as it does for pre-placed ones.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import struct
import threading
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from .flatmap import _resized, empty_doodads
from .mpq import GameArchives, MpqArchive
from .objects import parse_slk

DEFAULT_SIZE = 48


@dataclass(frozen=True)
class Rules:
    """Faster games, through the map's object data (war3map.w3u / w3q override the game's unit and
    upgrade tables): every unit type's (buildings' and creeps' too) hit points, build / train
    times and costs, and every upgrade's research times and costs, scaled. Heroes' hit points
    come mostly from strength: the hit points per strength point are scaled too (the map's
    gameplay constants, war3mapMisc.txt)."""
    hp: float = 1.0
    time: float = 1.0
    cost: float = 1.0

    @property
    def tag(self) -> str:
        return "" if self == Rules() else f"_hp{self.hp:g}_t{self.time:.2g}_c{self.cost:g}"


FAST = Rules(hp=0.5, time=1 / 3, cost=0.5)
BASE_X = 2300.0  # start locations at (-BASE_X, 0) and (BASE_X, 0)
MINE_OFFSET = (-150.0, 700.0)  # from the start location (x mirrored for the right base)
MINE_GOLD = 12500
TREE = "LTlt"  # Lordaeron Summer Tree Wall, as around Echo Isles' bases
TREE_ROWS = 3
# (x, y, unit types): neutral hostile camps
CAMPS = [
    (-1400.0, -1700.0, ("nkob", "nkob", "nkog")),  # kobolds, near the left base
    (1400.0, -1700.0, ("nkob", "nkob", "nkog")),   # and the right one
    (0.0, 1700.0, ("nogr", "nftt", "nftt")),       # an ogre and two forest trolls, north
    (0.0, -2150.0, ("ngnb", "ngna", "ngna")),      # a gnoll brute and two poachers, south
]
_lock = threading.Lock()


def duel_layout(size: int = DEFAULT_SIZE, seed: int = 0) -> dict:
    """Where everything goes: starts, mines, trees (x, y, facing, variation) and camps."""
    half = size * 64.0
    starts = [(-BASE_X, 0.0), (BASE_X, 0.0)]
    mines = [(sx + (MINE_OFFSET[0] if sx < 0 else -MINE_OFFSET[0]), sy + MINE_OFFSET[1]) for sx, sy in starts]
    rng = random.Random(seed)
    trees = []
    edge = TREE_ROWS * 128.0
    n = int(half // 128)
    for i in range(-n, n):
        for j in range(-n, n):
            x, y = (i + 0.5) * 128.0, (j + 0.5) * 128.0
            if min(half - abs(x), half - abs(y)) < edge:
                trees.append((x, y, rng.uniform(0, 360), rng.randrange(10)))
    return {"size": size, "starts": starts, "mines": mines, "trees": trees, "camps": CAMPS}


def _script_units(layout: dict) -> str:
    lines = ["function W3S_DuelUnits takes nothing returns nothing",
             "    local unit u",
             "    local player p = Player(PLAYER_NEUTRAL_PASSIVE)",
             "    local player h = Player(PLAYER_NEUTRAL_AGGRESSIVE)"]
    for x, y in layout["mines"]:
        lines += [f"    set u = CreateUnit(p, 'ngol', {x:.1f}, {y:.1f}, 270.0)",
                  f"    call SetResourceAmount(u, {MINE_GOLD})"]
    for cx, cy, types in layout["camps"]:
        for k, t in enumerate(types):
            dx, dy = ((0.0, 0.0), (-90.0, -70.0), (90.0, -70.0))[k % 3]
            lines.append(f"    set u = CreateUnit(h, '{t}', {cx + dx:.1f}, {cy + dy:.1f}, 270.0)")
    for x, y, facing, var in layout["trees"]:
        lines.append(f"    call CreateDestructable('{TREE}', {x:.1f}, {y:.1f}, {facing:.1f}, 1.0, {var})")
    lines += ["    set u = null", "    set p = null", "    set h = null", "endfunction", "", ""]
    return "\n".join(lines)


def _object_mods(objects: list[tuple[str, list[tuple[str, int]]]], levels: bool) -> bytes:
    """A war3map.w3u / w3q file: modified originals (id, [(field, integer value)]), no new objects.
    `levels`: the format of leveled tables (w3a / w3d / w3q), with a level and data column per
    modification."""
    out = struct.pack("<ii", 2, len(objects))
    for oid, mods in objects:
        out += oid.encode("latin-1") + b"\0\0\0\0" + struct.pack("<i", len(mods))
        for field, value in mods:
            out += field.encode("latin-1") + struct.pack("<i", 0)  # type: integer
            if levels:
                out += struct.pack("<ii", 0, 0)
            out += struct.pack("<ii", int(value), 0)
    return out + struct.pack("<i", 0)


def rules_files(rules: Rules) -> dict[str, bytes]:
    """war3map.w3u and war3map.w3q for `rules` (none for the game's own rules)."""
    if rules == Rules():
        return {}
    with GameArchives() as g:
        units = parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))
        upgrades = parse_slk(g.read("Units\\UpgradeData.slk").decode("latin-1"))

    def num(row: dict, key: str) -> int | None:
        try:
            return int(float(row[key]))
        except (KeyError, ValueError):
            return None

    def scaled(v: int | None, f: float, least: int) -> int | None:
        return None if v is None or v <= 0 or f == 1.0 else max(least, round(v * f))

    w3u = []
    for row in units:
        oid = row.get("unitBalanceID", "")
        if len(oid) != 4:
            continue
        mods = [(field, v) for field, v in (
            ("uhpm", scaled(num(row, "HP"), rules.hp, 1)), ("ubld", scaled(num(row, "bldtm"), rules.time, 1)),
            ("ugol", scaled(num(row, "goldcost"), rules.cost, 1)),
            ("ulum", scaled(num(row, "lumbercost"), rules.cost, 1))) if v is not None]
        if mods:
            w3u.append((oid, mods))
    w3q = []
    for row in upgrades:
        oid = row.get("upgradeid", "")
        if len(oid) != 4:
            continue
        mods = [(field, v) for field, v in (
            ("gtib", scaled(num(row, "timebase"), rules.time, 1)), ("gtim", scaled(num(row, "timemod"), rules.time, 1)),
            ("gglb", scaled(num(row, "goldbase"), rules.cost, 1)), ("gglm", scaled(num(row, "goldmod"), rules.cost, 1)),
            ("glmb", scaled(num(row, "lumberbase"), rules.cost, 1)),
            ("glmm", scaled(num(row, "lumbermod"), rules.cost, 1))) if v is not None]
        if mods:
            w3q.append((oid, mods))
    files = {"war3map.w3u": _object_mods(w3u, False), "war3map.w3q": _object_mods(w3q, True)}
    if rules.hp != 1.0:
        # heroes' hit points come from strength (25 per point): the map's gameplay constants,
        # a copy of the game's with that one changed
        with GameArchives() as g:
            misc = g.read("Units\\MiscGame.txt").decode("latin-1")
        bonus = 25 * rules.hp
        files["war3mapMisc.txt"] = re.sub(r"(?m)^StrHitPointBonus=.*$", f"StrHitPointBonus={bonus:g}",
                                          misc).encode("latin-1")
    return files


def make_duel_map(source: str | Path, dest: str | Path, size: int = DEFAULT_SIZE, rules: Rules = Rules()) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    layout = duel_layout(size)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    shutil.copyfile(source, tmp)
    try:
        with MpqArchive(tmp, writable=True) as m:
            files = _resized(m, size, layout["starts"])
            files["war3map.doo"] = empty_doodads(m.read("war3map.doo"))
            script = files["war3map.j"].decode("latin-1")
            main = script.index("function main takes nothing returns nothing")
            script = script[:main] + _script_units(layout) + script[main:]
            # the units come before InitBlizzard, where the stock script made its pre-placed ones
            script = re.sub(r"(\n\s*call InitBlizzard\(\s*\))", r"\n    call W3S_DuelUnits()\1", script, count=1)
            files["war3map.j"] = script.replace("\r\n", "\n").replace("\n", "\r\n").encode("latin-1")
            files.update(rules_files(rules))
            for name, data in files.items():
                m.write(name, data)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def parse_duel_name(name: str) -> tuple[int, Rules] | None:
    """"duel" -> (48, the game's rules), "duelN" -> N tiles, "duelfast[N]" -> with FAST rules;
    else None."""
    m = re.fullmatch(r"duel(fast)?(\d*)", name)
    if not m:
        return None
    size = int(m.group(2)) if m.group(2) else DEFAULT_SIZE
    if size < 32 or size > 128 or size % 2:
        raise ValueError("duel map size must be an even number of tiles between 32 and 128")
    return size, FAST if m.group(1) else Rules()


def duel_map_path(name: str) -> Path:
    """The cached duel map for a "duel..." name (built on first use)."""
    from .mapbuild import stock_map_path

    size, rules = parse_duel_name(name)
    src = stock_map_path("(2)EchoIsles")
    out = paths.CACHE_DIR / "maps" / f"duel{size}{rules.tag}_v2.w3x"
    with _lock:
        if not out.exists():
            make_duel_map(src, out, size, rules)
    return out
