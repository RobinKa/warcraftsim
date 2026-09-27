"""A small melee map for learning the whole game: "duel" (48 x 48 tiles) or "duelN".

Two bases on a flat field, as close as a real map allows (4600 apart; the smallest stock
two-player maps are 80 x 80 tiles with bases 8400-9500 apart), laid out like Echo Isles' main
bases so the built-in AI plays as it does there:

* a start location per player at (-2300, 0) and (2300, 0), mirrored; a 12500-gold mine ~720 from
  each, where Echo Isles has its main mines (781);
* a wall of trees three deep along the map's edge, ~450 behind each start location (Echo Isles' nearest
  trees: 550-700): lumber, and the map's border;
* creep camps for the heroes' experience, against the tree wall and more than 2000 from the start
  locations (melee initialization removes creeps within 1500 of them; closer ones walked into a
  base on the small map): kobolds on each base's side, ogres and trolls north and gnolls south of
  the middle. No items, shops or expansions.

"duelfast" (FAST rules): the same map with hit points halved, build, train and research times
cut to a third and costs halved for every unit type and upgrade (Rules, through the map's object
data); with a 50% handicap, units have a quarter of their hit points.

"duelrush" (RUSH rules, ~2-minute games): a faster version of the whole game. Everything that
takes time runs 7 times faster (attacks, casts, cooldowns, production, regeneration, day and
night, the gameplay constants that are times, and the built-in AI's waits, through overriding
copies of its scripts), except movement: the engine stops units at 522, so every unit moves 1.3
times faster (the fastest, 400, at the limit), and the map is smaller (40 tiles, bases 3000
apart). Hit points and costs are halved, production takes half the time on top, players start
with twice the gold and lumber, and the built-in AI attacks main bases from force level 20 (not
40) and by night too (it otherwise waited for day, or went creeping).

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
from dataclasses import dataclass, replace
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
    harvest: float = 1.0  # gold and lumber per trip and lumber per chop, every race's workers
    day: float = 1.0  # how much faster the day/night cycle runs (the built-in AI attacks by day)
    start: float = 1.0  # starting gold and lumber
    # everything that takes time, k times faster, as if the game ran at k times its speed: attack
    # and cast timings, turning, ability cooldowns and durations (harvesting intervals too),
    # regeneration, production (on top of `time`), day and night, and the built-in AI's own waits
    # (its scripts, overridden by the map's copies). Movement can't follow: the engine stops every
    # unit at 522 (measured: MaxUnitSpeed above it changes nothing), so all units move `move` times
    # faster, the most that keeps the fastest (400) under it and every unit's speed in proportion;
    # the map is smaller instead (duel_layout's base distance).
    speed: float = 1.0
    max_speed: int = 522  # the gameplay constants' movement speed limit (MaxUnitSpeed / MaxBldgSpeed)
    # the built-in AI's scripts: it attacks the enemy's main base only once its force reaches this
    # level (the game's: 40) or it has siege units, and (the game's rule) only by day; else it goes
    # creeping, and on a small map there is soon nothing left to creep
    ai_siege_level: int = 40
    ai_by_night: bool = False

    @property
    def move(self) -> float:
        return min(self.speed, self.max_speed / FASTEST_UNIT)

    @property
    def tag(self) -> str:
        if self == Rules():
            return ""
        tag = f"_hp{self.hp:g}_t{self.time:.2g}_c{self.cost:g}"
        if (self.harvest, self.day, self.start) != (1.0, 1.0, 1.0):
            tag += f"_h{self.harvest:g}_d{self.day:g}_s{self.start:g}"
        if self.speed != 1.0:
            tag += f"_x{self.speed:g}"
        if self.max_speed != 522:
            tag += f"_m{self.max_speed}"
        if (self.ai_siege_level, self.ai_by_night) != (40, False):
            tag += f"_ai{self.ai_siege_level}{'n' if self.ai_by_night else ''}"
        return tag


FAST = Rules(hp=0.5, time=1 / 3, cost=0.5)
# the game three times as fast (Rules.speed), and cheaper and faster production on top
RUSH = Rules(hp=0.5, time=0.5, cost=0.5, start=2.0, speed=7.0, ai_siege_level=20, ai_by_night=True)
FASTEST_UNIT = 400  # the base movement speed of the fastest melee units (gyrocopter, hippogryph)
# harvest abilities' integer fields: gold per trip (Har3, Bgm1, Egm1), lumber per trip (Har2),
# lumber per chop (Har1; wisps: Wha2), with the data column they are in
HARVEST_FIELDS = {"Ahar": (("Har1", 1), ("Har2", 2), ("Har3", 3)), "Ahrl": (("Har1", 1), ("Har2", 2)),
                  "Abgm": (("Bgm1", 1),), "Aegm": (("Egm1", 1),), "Awha": (("Wha2", 2),)}
BASE_X = 2300.0  # start locations at (-BASE_X, 0) and (BASE_X, 0)
MINE_OFFSET = (-150.0, 700.0)  # from the start location (x mirrored for the right base)
MINE_GOLD = 12500
TREE = "LTlt"  # Lordaeron Summer Tree Wall, as around Echo Isles' bases
TREE_ROWS = 3
# neutral hostile camps, against the tree wall: kobolds on each base's side (point-symmetric),
# an ogre and two forest trolls north, a gnoll brute and two poachers south
KOBOLDS, OGRES, GNOLLS = ("nkob", "nkob", "nkog"), ("nogr", "nftt", "nftt"), ("ngnb", "ngna", "ngna")
_lock = threading.Lock()


def duel_layout(size: int = DEFAULT_SIZE, seed: int = 0, base_x: float = BASE_X) -> dict:
    """Where everything goes: starts, mines, trees (x, y, facing, variation) and camps."""
    half = size * 64.0
    starts = [(-base_x, 0.0), (base_x, 0.0)]
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
    # camps: just inside the tree wall, >2000 from the start locations (melee initialization
    # clears 1500 around them; closer camps walked into a base on the small map)
    inner = half - edge - 150.0
    camps = [(-(base_x - 600.0), -inner, KOBOLDS), (base_x - 600.0, inner, KOBOLDS),
             (0.0, inner, OGRES), (0.0, -inner, GNOLLS)]
    return {"size": size, "starts": starts, "mines": mines, "trees": trees, "camps": camps}


def _script_units(layout: dict) -> str:
    """The map's units, in three functions: mines and trees are made once; a melee restart in the
    game (harness W3S_MeleeReset) keeps the mines (refilled: the built-in AI's towns hold on to
    them) and re-runs W3S_DuelCreeps."""
    lines = ["function W3S_DuelMines takes nothing returns nothing",
             "    local unit u",
             "    local player p = Player(PLAYER_NEUTRAL_PASSIVE)"]
    for x, y in layout["mines"]:
        lines += [f"    set u = CreateUnit(p, 'ngol', {x:.1f}, {y:.1f}, 270.0)",
                  f"    call SetResourceAmount(u, {MINE_GOLD})"]
    lines += ["    set u = null", "    set p = null", "endfunction", "",
              "function W3S_DuelCreeps takes nothing returns nothing",
              "    local player h = Player(PLAYER_NEUTRAL_AGGRESSIVE)"]
    for cx, cy, types in layout["camps"]:
        for k, t in enumerate(types):
            dx, dy = ((0.0, 0.0), (-90.0, -70.0), (90.0, -70.0))[k % 3]
            lines.append(f"    call CreateUnit(h, '{t}', {cx + dx:.1f}, {cy + dy:.1f}, 270.0)")
    lines += ["    set h = null", "endfunction", "",
              "function W3S_DuelTrees takes nothing returns nothing"]
    for x, y, facing, var in layout["trees"]:
        lines.append(f"    call CreateDestructable('{TREE}', {x:.1f}, {y:.1f}, {facing:.1f}, 1.0, {var})")
    lines += ["endfunction", "",
              "function W3S_DuelUnits takes nothing returns nothing",
              "    call W3S_DuelMines()", "    call W3S_DuelCreeps()", "endfunction", "", ""]
    return "\n".join(lines)


def _object_mods(objects: list[tuple[str, list[tuple]]], levels: bool) -> bytes:
    """A war3map.w3u / w3q / w3a file: modified originals (id, [(field, value[, level, data
    column])]; int values are integers, floats "unreal" reals), no new objects. `levels`: the format of leveled tables (w3a / w3d / w3q),
    with a level and data column per modification."""
    out = struct.pack("<ii", 2, len(objects))
    for oid, mods in objects:
        out += oid.encode("latin-1") + b"\0\0\0\0" + struct.pack("<i", len(mods))
        for field, value, *where in mods:
            real = isinstance(value, float)
            out += field.encode("latin-1") + struct.pack("<i", 2 if real else 0)  # unreal or integer
            if levels:
                out += struct.pack("<ii", *(where or (0, 0)))
            out += struct.pack("<fi" if real else "<ii", value, 0)
    return out + struct.pack("<i", 0)


def rules_files(rules: Rules) -> dict[str, bytes]:
    """war3map.w3u and war3map.w3q for `rules` (none for the game's own rules)."""
    if rules == Rules():
        return {}
    with GameArchives() as g:
        units = parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))
        upgrades = parse_slk(g.read("Units\\UpgradeData.slk").decode("latin-1"))
        unit_abilities = parse_slk(g.read("Units\\UnitAbilities.slk").decode("latin-1"))
        weapons = {r.get("serpent"): r for r in  # (the ID column's header in 1.29)
                    parse_slk(g.read("Units\\UnitWeapons.slk").decode("latin-1"))}
        unitdata = {r.get("unitID"): r for r in parse_slk(g.read("Units\\UnitData.slk").decode("latin-1"))}
        abilities = {r.get("alias"): r for r in parse_slk(g.read("Units\\AbilityData.slk").decode("latin-1"))}
        misc = g.read("Units\\MiscGame.txt").decode("latin-1")
    k = rules.speed
    # only what these games can have: the four races' units (and "other": summons, some buildings)
    # and this map's creeps, and their abilities (every unit and ability of the game made the map
    # load slower)
    races = {"human", "orc", "undead", "nightelf", "other"}
    creeps = {t for camp in (KOBOLDS, OGRES, GNOLLS) for t in camp}
    used = {r["unitID"] for r in unitdata.values() if r.get("race") in races or r.get("unitID") in creeps}
    used |= creeps | {"ngol"}
    used_abilities = set()
    for r in unit_abilities:
        if r.get("unitAbilID") in used:
            for col in ("abilList", "heroAbilList", "auto"):
                used_abilities.update(a.strip() for a in (r.get(col) or "").split(",") if len(a.strip()) == 4)

    def num(row: dict, key: str) -> int | None:
        try:
            return int(float(row[key]))
        except (KeyError, ValueError):
            return None

    def scaled(v: int | None, f: float, least: int) -> int | None:
        return None if v is None or v <= 0 or f == 1.0 else max(least, round(v * f))

    def real(row: dict, key: str) -> float | None:
        try:
            return float(row[key])
        except (KeyError, ValueError, TypeError):
            return None

    def rscaled(v: float | None, f: float, cap: float | None = None) -> float | None:
        if v is None or v <= 0 or f == 1.0:
            return None
        return min(v * f, cap) if cap is not None else v * f

    w3u = []
    for row in units:
        oid = row.get("unitBalanceID", "")
        if len(oid) != 4 or oid not in used:
            continue
        mods = [(field, v) for field, v in (
            ("uhpm", scaled(num(row, "HP"), rules.hp, 1)), ("ubld", scaled(num(row, "bldtm"), rules.time / k, 1)),
            ("ugol", scaled(num(row, "goldcost"), rules.cost, 1)),
            ("ulum", scaled(num(row, "lumbercost"), rules.cost, 1)),
            ("urtm", scaled(num(row, "reptm"), 1 / k, 1))) if v is not None]
        if k != 1.0:
            spd = num(row, "spd")
            if spd and spd > 0:
                mods.append(("umvs", min(rules.max_speed, round(spd * rules.move))))
            mx = num(row, "maxSpd")
            if mx and mx > 0:
                mods.append(("umas", min(rules.max_speed, round(mx * rules.move))))
            reals = [("uhpr", rscaled(real(row, "regenHP"), k)), ("umpr", rscaled(real(row, "regenMana"), k))]
            w, d = weapons.get(oid, {}), unitdata.get(oid, {})
            reals += [(f, rscaled(real(w, c), 1 / k)) for f, c in (
                ("ua1c", "cool1"), ("ua2c", "cool2"), ("udp1", "dmgpt1"), ("udp2", "dmgpt2"), ("ubs1", "backSw1"),
                ("ubs2", "backSw2"), ("ucpt", "castpt"), ("ucbs", "castbsw"))]
            reals.append(("umvr", rscaled(real(d, "turnRate"), k, 3.0)))
            mods += [(f, float(v)) for f, v in reals if v is not None]
        if mods:
            w3u.append((oid, mods))
    w3q = []
    for row in upgrades:
        oid = row.get("upgradeid", "")
        if len(oid) != 4:
            continue
        mods = [(field, v) for field, v in (
            ("gtib", scaled(num(row, "timebase"), rules.time / k, 1)),
            ("gtim", scaled(num(row, "timemod"), rules.time / k, 1)),
            ("gglb", scaled(num(row, "goldbase"), rules.cost, 1)), ("gglm", scaled(num(row, "goldmod"), rules.cost, 1)),
            ("glmb", scaled(num(row, "lumberbase"), rules.cost, 1)),
            ("glmm", scaled(num(row, "lumbermod"), rules.cost, 1))) if v is not None]
        if mods:
            w3q.append((oid, mods))
    files = {"war3map.w3u": _object_mods(w3u, False), "war3map.w3q": _object_mods(w3q, True)}
    w3a: dict[str, list] = {}
    if rules.harvest != 1.0:
        for aid, fields in HARVEST_FIELDS.items():
            row = abilities.get(aid, {})
            w3a.setdefault(aid, []).extend(
                (field, v, 1, col) for field, col in fields
                if (v := scaled(num(row, f"Data{'ABCD'[col - 1]}1"), rules.harvest, 1)) is not None)
    if k != 1.0:
        for aid, row in abilities.items():
            if not aid or len(aid) != 4 or aid not in used_abilities:
                continue
            levels = max(1, num(row, "levels") or 1)
            for level in range(1, levels + 1):
                for field, col in (("acdn", "Cool"), ("adur", "Dur"), ("ahdu", "HeroDur"), ("acas", "Cast")):
                    v = rscaled(real(row, f"{col}{level}"), 1 / k)
                    if v is not None:
                        w3a.setdefault(aid, []).append((field, float(v), level, 0))
            # gold mining time (a worker inside a gold mine / haunted mine)
            for aid_, field, col in (("Agld", "Gld2", 2), ("Abgm", "Bgm2", 2), ("Aegm", "Egm2", 2)):
                if aid == aid_ and (v := rscaled(real(row, f"Data{'ABCD'[col - 1]}1"), 1 / k)) is not None:
                    w3a.setdefault(aid, []).append((field, float(v), 1, col))
    if w3a:
        files["war3map.w3a"] = _object_mods([(a, m) for a, m in w3a.items() if m], True)
    # the map's gameplay constants: a copy of the game's with some changed
    constants = {}
    if rules.hp != 1.0:  # heroes' hit points come from strength (25 per point)
        constants["StrHitPointBonus"] = 25 * rules.hp
    if k != 1.0:
        # the constants that are times or rates: how long a creep chases before it goes home (unscaled,
        # creeps followed fleeing units into the bases), the cap on hero revival times, how fast a
        # halted construction loses hit points; the slowest movement (slows) with the rest
        constants.update(MaxUnitSpeed=rules.max_speed, MaxBldgSpeed=rules.max_speed, StrRegenBonus=0.05 * k,
                         IntRegenBonus=0.05 * k, GuardReturnTime=5.0 / k, HeroMaxReviveTime=150 * rules.time / k,
                         ConstructionLifeDrainRate=10.0 * k * rules.hp, MinUnitSpeed=150 * rules.move,
                         MinBldgSpeed=25 * rules.move)
    if constants:
        for key, value in constants.items():
            misc, n = re.subn(rf"(?m)^{key}=[^\t\r\n/]*", f"{key}={value:g}", misc)
            if n != 1:
                raise ValueError(f"gameplay constant {key} not found")
        files["war3mapMisc.txt"] = misc.encode("latin-1")
    if k != 1.0 or (rules.ai_siege_level, rules.ai_by_night) != (40, False):
        files.update(ai_scripts(1 / k, rules.ai_siege_level, rules.ai_by_night))
    return files


def ai_scripts(factor: float, siege_level: int = 40, by_night: bool = False) -> dict[str, bytes]:
    """The built-in AI's scripts with every wait scaled by `factor` (the map's copies override the
    game's): common.ai gets W3S_Sleep, which every Sleep call goes through. `siege_level`: the
    force level at which it attacks main bases; `by_night`: it does so at night too."""
    out = {}
    with GameArchives() as g:
        for name in ("common.ai", "human.ai", "orc.ai", "undead.ai", "elf.ai"):
            text = g.read(f"Scripts\\{name}").decode("latin-1")
            text, n = re.subn(r"\bcall Sleep\(", "call W3S_Sleep(", text)
            if name != "common.ai":
                text, n = re.subn(r"(set has_siege\s*=\s*level >= )40\b", rf"\g<1>{siege_level}", text)
                if n != 1:
                    raise ValueError(f"{name}: the siege level rule was not found")
            elif by_night:
                text, n = re.subn(r"set can_siege = has_siege and \(air_units or \(daytime>=4 and daytime<=12\)\)",
                                  "set can_siege = has_siege", text)
                if n != 1:
                    raise ValueError("common.ai: the daytime rule was not found")
            if name == "common.ai":
                first = re.search(r"(?m)^function ", text)
                text = (text[:first.start()] + "function W3S_Sleep takes real seconds returns nothing\r\n"
                        f"    call Sleep(seconds * {factor:.4f})\r\nendfunction\r\n\r\n" + text[first.start():])
            out[f"Scripts\\{name}"] = text.encode("latin-1")
    return out


def _script_rules(rules: Rules) -> str:
    return "\n".join([
        "function W3S_DuelRules takes nothing returns nothing",
        "    local integer i = 0",
        "    local player p",
        f"    call SetTimeOfDayScale({rules.day * rules.speed:.3f})",
        "    loop",
        "        exitwhen i >= bj_MAX_PLAYERS",
        "        set p = Player(i)",
        "        if GetPlayerSlotState(p) == PLAYER_SLOT_STATE_PLAYING then",
        f"            call SetPlayerState(p, PLAYER_STATE_RESOURCE_GOLD, R2I(GetPlayerState(p, PLAYER_STATE_RESOURCE_GOLD) * {rules.start:.3f}))",
        f"            call SetPlayerState(p, PLAYER_STATE_RESOURCE_LUMBER, R2I(GetPlayerState(p, PLAYER_STATE_RESOURCE_LUMBER) * {rules.start:.3f}))",
        "        endif",
        "        set i = i + 1",
        "    endloop",
        "    set p = null",
        "endfunction", "", ""])


def make_duel_map(source: str | Path, dest: str | Path, size: int = DEFAULT_SIZE, rules: Rules = Rules(),
                  base_x: float = BASE_X) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    layout = duel_layout(size, base_x=base_x)
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
            script = re.sub(r"(\n\s*call InitBlizzard\(\s*\))", r"\n    call W3S_DuelTrees()\n    call W3S_DuelUnits()\1",
                            script, count=1)
            if (rules.day * rules.speed, rules.start) != (1.0, 1.0):  # after melee initialization (which sets both)
                script, n = re.subn(r"(\n\s*call RunInitializationTriggers\(\s*\))",
                                    r"\1\n    call W3S_DuelRules()", script, count=1)
                if n != 1:
                    raise ValueError("the map script has no RunInitializationTriggers call")
                main = script.index("function main takes nothing returns nothing")
                script = script[:main] + _script_rules(rules) + script[main:]
            files["war3map.j"] = script.replace("\r\n", "\n").replace("\n", "\r\n").encode("latin-1")
            files.update(rules_files(rules))
            for name, data in files.items():
                m.write(name, data)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


RUSH_SIZE, RUSH_BASE_X = 40, 1500.0  # bases 3000 apart. A walk between them, in the game's own
# time: 3000 / 351 * speed = 60 s at x7 (43 s at x5; Echo Isles: 9856 / 270 = 37 s): walking
# takes a larger share of the game than on a real map, the price of the short games
# Built-in AI against built-in AI, 12 games each (all races, 50% handicap): x5 1.3-4.1 minutes
# (median 2.5), x7 1.1-3.3 (median 2.2) and one stalemate.


def parse_duel_name(name: str) -> tuple[int, Rules, float] | None:
    """"duel" -> (48 tiles, the game's rules, base x), "duelN" -> N tiles, "duelfast[N]" -> FAST
    rules, "duelrush[N]" -> RUSH rules on a smaller map (40 tiles, bases 3000 apart),
    "duelrushxK[N]" -> RUSH at speed K (e.g. duelrushx5); else None."""
    m = re.fullmatch(r"duel(fast|rush(?:x(\d+))?)?(\d*)", name)
    if not m:
        return None
    rush = (m.group(1) or "").startswith("rush")
    size = int(m.group(3)) if m.group(3) else (RUSH_SIZE if rush else DEFAULT_SIZE)
    if size < 32 or size > 128 or size % 4:  # (38 tiles: the game crashed within seconds)
        raise ValueError("duel map size must be a multiple of 4 tiles between 32 and 128")
    rules = FAST if m.group(1) == "fast" else RUSH if rush else Rules()
    if m.group(2):
        rules = replace(rules, speed=float(m.group(2)))
    return size, rules, RUSH_BASE_X if rush else BASE_X


def duel_map_path(name: str) -> Path:
    """The cached duel map for a "duel..." name (built on first use)."""
    from .mapbuild import stock_map_path

    size, rules, base_x = parse_duel_name(name)
    src = stock_map_path("(2)EchoIsles")
    out = paths.CACHE_DIR / "maps" / f"duel{size}_b{base_x:g}{rules.tag}_v9.w3x"
    with _lock:
        if not out.exists():
            make_duel_map(src, out, size, rules, base_x)
    return out
