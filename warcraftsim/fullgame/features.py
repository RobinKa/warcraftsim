"""Demonstration games (fullgame/collect.py) -> what one player saw and ordered, per step.

Pure numpy (no warcraftsim imports), so the torch trainer can load it by path like rl/.

For player p at step t:
* entities: p's own living units (buildings too) first, then the enemy's and neutral units
  (creeps, gold mines) that p could see (the units' visibility bits), at most MAX_ENT. Positions
  are mirrored so that p's own base is on the left (the duel maps are mirror-symmetric).
* entity features (F floats), each entity's unit type and current order (vocabulary indices).
* global features: resources, supply, game time, both races, p's upgrade levels.
* labels for each of the first MAX_OWN own entities: the order it got during the step (the last
  one if several; class 0: none), as a class of the order vocabulary, i.e. (order id, kind),
  and its target: another entity (a pointer), or a point (x / y bins over the map). Orders on
  trees become kind TREE with the tree's position (at play time the nearest tree there is used).
"""

from __future__ import annotations

from functools import lru_cache

import bisect
import json
import os
import math
from collections import Counter
from pathlib import Path

import numpy as np

_LAZY: dict = {}  # (the sibling modules the actors' per-step code uses, imported on first use: an import statement in a
# function took ~5,000 instructions a call; this module imports none at its top, so it loads by path too)


def _mod(name: str):
    m = _LAZY.get(name)
    if m is None:
        import importlib
        import sys
        m = _LAZY[name] = sys.modules[__name__] if name == "" else importlib.import_module(f".{name}", __package__)
    return m

# The frame a player sees positions in: relative to its own start location, at the duel maps' scale (SCALE
# world units a unit of position), x mirrored for the player on the right, its start where the duel maps'
# left base is (ANCHOR_X, 0). On the duel maps (starts at (-2300, 0) and (2300, 0)) that is their frame of
# before: x / 3072, y / 3072. On a larger map the base area looks the same (Echo Isles' was first scaled to
# its size, and the duelfast policy's buildings landed 2.3 times as far out, in the trees: 5 of 67 barracks
# orders taken). Point targets are bins of 1/64 of a unit (48 world units) over [-BIN_RANGE, BIN_RANGE]:
# BINS = 128 on the duel maps; a larger map's need more (the network's config["bins"], scripts/grow_bins.py).
# set_geometry for another map (map_geometry: from its start locations and bounds); from the environment
# too (processes started anew).
SCALE = 3072.0
ANCHOR_X = -2300.0 / SCALE
DUEL_D = 4600.0  # the duel maps' distance between the bases
NEAR = 1500.0  # around either base the duel maps' scale; between them x is squeezed to DUEL_D (a larger map's
# enemy base where the duel maps' was: the duelfast policy's attacks went to its old place, mid-map, and 41 of
# its first 89 games on Echo Isles tied)
BINS_PER_UNIT = 64
STARTS: tuple[tuple[float, float], tuple[float, float]] = ((-2300.0, 0.0), (2300.0, 0.0))  # left, right
MAP_CX = 0.0  # the line between the bases (Encoder.side)
BIN_RANGE = 1.0
BINS = 128


def set_geometry(starts, bin_range: float) -> None:
    """The map's two start locations ((x, y), (x, y); bases left and right: the view mirrors x only) and
    the reach of its point bins."""
    global STARTS, MAP_CX, BIN_RANGE, BINS
    (ax, ay), (bx, by) = sorted((float(x), float(y)) for x, y in starts)
    STARTS, MAP_CX = ((ax, ay), (bx, by)), (ax + bx) / 2
    BIN_RANGE = float(bin_range)
    BINS = int(round(2 * BINS_PER_UNIT * BIN_RANGE))
    os.environ["WARCRAFTSIM_MAP_GEOMETRY"] = f"{ax},{ay},{bx},{by},{BIN_RANGE}"


if os.environ.get("WARCRAFTSIM_MAP_GEOMETRY"):
    _g = [float(v) for v in os.environ["WARCRAFTSIM_MAP_GEOMETRY"].split(",")]
    set_geometry(((_g[0], _g[1]), (_g[2], _g[3])), _g[4])


def own_start(sign: float) -> tuple[float, float]:
    """The start location a player's view is anchored at (sign 1: the left base, -1: the right one)."""
    return STARTS[0] if sign > 0 else STARTS[1]


def _squeeze(d):
    """A distance in front of the own base -> the duel maps' (piecewise linear: as is within NEAR of either
    base, the stretch between them squeezed to the duel maps' length)."""
    d = np.asarray(d, dtype=np.float64)
    D = STARTS[1][0] - STARTS[0][0]
    if abs(D - DUEL_D) < 1e-6:
        return d
    k = (DUEL_D - 2 * NEAR) / (D - 2 * NEAR)
    return np.where(d <= NEAR, d, np.where(d >= D - NEAR, d - (D - DUEL_D), NEAR + (d - NEAR) * k))


def _unsqueeze(e):
    e = np.asarray(e, dtype=np.float64)
    D = STARTS[1][0] - STARTS[0][0]
    if abs(D - DUEL_D) < 1e-6:
        return e
    k = (DUEL_D - 2 * NEAR) / (D - 2 * NEAR)
    return np.where(e <= NEAR, e, np.where(e >= DUEL_D - NEAR, e + (D - DUEL_D), NEAR + (e - NEAR) / k))


def _frame_xy1(x: float, y: float, sign: float) -> tuple[float, float]:
    """frame_xy for one point, in plain floats (the same float64 arithmetic: numpy's calls on numbers had made the
    privileged block's two points ~a fifth of View.step)."""
    ox, oy = own_start(sign)
    d = sign * (x - ox)
    D = STARTS[1][0] - STARTS[0][0]
    if abs(D - DUEL_D) >= 1e-6:
        k = (DUEL_D - 2 * NEAR) / (D - 2 * NEAR)
        d = d if d <= NEAR else d - (D - DUEL_D) if d >= D - NEAR else NEAR + (d - NEAR) * k
    return d / SCALE + ANCHOR_X, (y - oy) / SCALE


def frame_xy(x, y, sign: float):
    """World positions -> a player's frame (arrays or numbers)."""
    ox, oy = own_start(sign)
    return _squeeze(sign * (np.asarray(x) - ox)) / SCALE + ANCHOR_X, (np.asarray(y) - oy) / SCALE


def map_geometry(path) -> tuple[tuple, float]:
    """A map's geometry for set_geometry: its first two start locations, and the bins' reach: the
    farthest its camera bounds lie from either player's frame's origin, rounded up to a quarter."""
    from ..data.mpq import MpqArchive
    from ..data.w3i import parse_w3i

    with MpqArchive(path) as m:
        info = parse_w3i(m.read("war3map.w3i"))
    starts = tuple(sorted((p.start_x, p.start_y) for p in info.players[:2]))
    xs, ys = info.camera_bounds[0::2], info.camera_bounds[1::2]
    saved = STARTS
    globals()["STARTS"] = starts  # (frame_xy with this map's bases)
    try:
        reach = 0.0
        for sign in (1.0, -1.0):
            for x in xs:
                for y in ys:
                    fx_, fy_ = frame_xy(x, y, sign)
                    reach = max(reach, abs(float(fx_)), abs(float(fy_)))
    finally:
        globals()["STARTS"] = saved
    return starts, max(1.0, math.ceil(reach * 4) / 4)


MAX_ENT, MAX_OWN = 160, 96
MOVE, AIMOVE = 851986, 851988  # the AI's own move order is played back as a move
# orders that are not decisions: internal ones the AI gives to most units (851974, 852660), and
# the engine's own: resumeharvesting (a worker going back after dropping off: 7.7% of recorded
# orders; a policy that learned it interrupted working harvesters, the game refused two thirds),
# returnresources, and autocasts (recharge: moon wells; ambush: archers at night)
DROPPED_ORDERS = {851974, 852660, 852017, 852020, 852157, 852131}
HARVEST, SMART = 852018, 851971
GOLD_MINES = (1852272492, 1969713004, 1701277548)  # ngol, ugol (haunted), egol (entangled): harvest targets
HARVESTING = {852018, 852017, 852020}  # current orders of a worker in its harvest cycle


def harvest_resource(order: int, kind: int, target_type: int | None) -> str | None:
    """What a harvest-like order sends a worker to: "lumber" (harvest or smart on a tree), "gold"
    (harvest on a unit, smart on a gold mine), None: no harvest (smart on anything else)."""
    if order not in (HARVEST, SMART):
        return None
    if kind == TREE:
        return "lumber"
    if kind == UNIT and (order == HARVEST or target_type in GOLD_MINES):
        return "gold"
    return None


def redundant(order: int, kind: int, current: int, resource: str | None = None, assigned: str | None = None,
              same_mine: bool = True) -> bool:
    """A harvest order for a worker already harvesting that resource: the AI re-issues these without
    effect; learned as decisions, a policy re-ordered its workers every few seconds to other trees
    and reset their work (240 gold from 27 peasants in 80 s). `resource`: what the order harvests
    (harvest_resource), `assigned`: what the worker harvests (View.assign; None: no order of the
    player's sent it, so the map's melee start or a rally point: gold). A miner sent to the trees is
    a decision: dropping those too (both show the same current order) left the clone with almost no
    lumber as human and orc, hoarding gold it could not spend on farms and barracks."""
    if current not in HARVESTING or order not in (HARVEST, SMART) or kind not in (UNIT, TREE):
        return False
    if resource == "gold" and not same_mine:  # (another mine: a transfer, an expansion's workers; the audit of 2026-10-22)
        return False
    return resource is None or resource == (assigned or "gold")


WORKER = 4  # UnitFlags.WORKER
BUILD_COMMIT_STEPS = 400  # (a worker's building order that lasts longer: let it be re-ordered. 60 until 2026-10-22:
# the walk to an expansion took ~30 s of it and the hall was left unbuilt; a real-game hall takes 150-180 s)
PASSIVE_IDS = np.array([int.from_bytes(c.encode(), "big") for c in (
    "hhou", "otrb", "uzig", "emow", "hwtw", "hgtw", "hctw", "hatw", "owtw", "uzg1", "uzg2", "etrp", "ugol", "egol")], np.int64)


_OWN_ALIVE: dict[int, tuple] = {}  # player -> (a step's rows, their living units of the player's)


def own_alive(rows: np.ndarray, player: int, mask: np.ndarray | None = None) -> np.ndarray:
    """rows[(owner == player) & living] (`mask`: that mask, when the caller has it), copied once per rows object and
    player (play_one's rows: the trackers' at a step's end, the teacher's at the next step's start). Read only."""
    hit = _OWN_ALIVE.get(player)
    if hit is not None and hit[0] is rows:
        return hit[1]
    out = rows[(rows[:, C_OWNER] == player) & (rows[:, C_FLAGS] & DEAD == 0) if mask is None else mask]
    _OWN_ALIVE[player] = (rows, out)
    return out


def codes(values) -> np.ndarray:
    """A few int codes (unit types, orders) as isin's sorted array."""
    return np.unique(np.fromiter((int(v) for v in values), np.int64))


def isin(a: np.ndarray, sorted_codes: np.ndarray) -> np.ndarray:
    """np.isin(a, sorted_codes) for a 1-d int array and a few codes sorted (codes()): a binary search (np.isin's
    general path took ~60-100 us a call on a step's rows, ~30,000 calls a game in an actor)."""
    if not len(sorted_codes):
        return np.zeros(len(a), bool)
    return sorted_codes[np.minimum(np.searchsorted(sorted_codes, a), len(sorted_codes) - 1)] == a


def building(flags: int, current: int) -> bool:
    """A worker whose current order is a building: on its way to the site or constructing. The
    policy picks every unit's order every step; another order cancels the build, and the clone did
    that to 47% of its builds (the AI to almost none), so it built too few farms and hoarded."""
    return bool(flags & WORKER) and current >= TYPE_CODE
IMMEDIATE, POINT, UNIT, SKILL, TREE = range(5)
KIND_NAMES = ("immediate", "point", "unit", "skill", "tree")
RACES = ("human", "orc", "undead", "nightelf")
RESEARCH_FINISH = 9

# unit columns (collect.UNIT_COLS)
C_STEP, C_ID, C_TYPE, C_OWNER, C_X, C_Y, C_FACING, C_HP, C_MAXHP, C_MANA, C_MAXMANA, C_ORDER, C_FLAGS, \
    C_VIS, C_RESOURCE, C_HLEVEL, C_HXP, C_SKILLPTS = range(18)
# then (a live game's rows, play.unit_rows; a demonstration's from its "heroes", hero_rows) a hero's 6 items' types
# and per ability slot its level and cooldown left (0.1 s); rows with 18 columns have none
C_ITEM0, C_ABIL0 = 18, 24
ROW_COLS = 32
DEAD, STRUCTURE, CONSTRUCTING = 1024, 2, 8
FOOD_MAX = 100  # (the game's food ceiling: a player's food cap counts farms past it)
SUMMONED = 256  # (UnitFlags.SUMMONED)
REVIVE_AFTER_STEPS = 16  # (a dead hero's revival is taken ~7 s after its death)
REVIVE_STEPS = 240  # (a revival the altar took: not offered again for 120 s, or until the hero stands; 60 s had reopened
# it while a real game's revival still ran, up to 110 s: ~17 refused a human real game; audit 9)
NATURAL_EVERY = 4  # (steps between two looks at the natural expansion: free, taken, its camp standing)
FLAG_BITS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048)  # UnitFlags but DEAD
F0 = 3 + 2 + 4 + len(FLAG_BITS) + 3 + 2 + 1 + 2 + 1  # 29: ... production (queued, busy), and a worker on lumber
# then (2026-10-12; a checkpoint from before gets zero weights for them, model.load): the hit points' change since
# the view's last step (a unit being hit), and the position as sines and cosines at 5 frequencies (attention can
# then compare two units' positions: cos(k(x_i - x_j)) is a product of the two units' features)
A_HPD = F0
POS_FREQS = (1, 2, 4, 8, 16)  # x and y in [-1, 1]: wavelengths from the map's width down to ~380 world units
A_POS = A_HPD + 1
# then (features v3): an own unit's last order the player gave it (View.note_orders): its target relative to
# the unit (a point, or the target unit where it is now; / 1536, mirrored), whether that was a point or a unit,
# and how long ago (steps / 40, at most 1; 1 when it never got one). Where a unit is already sent: without it a
# group's units could not see that they head for one point (their point orders scattered, fgself-13)
A_LAST = A_POS + 4 * len(POS_FREQS)
LAST_SCALE, LAST_AGE = 1536.0, 40.0
F_BASE = A_LAST + 5  # 55
# inputs v2 (selfplay --inputs 2, 2026-10-20; a checkpoint from before gets zero weights for them, model.load): a
# hero's state, as the harness reports it (protocol.Unit.abilities, items): per ability slot its level / 3, whether
# it can be cast now (learned, cooled down, the mana: 0 or 1), its cooldown left / 60 s and whether a skill point
# may go there now (0 or 1; an enemy hero's levels and items only: its cooldowns the game doesn't show); per
# inventory slot whether it holds an item, a usable one, one that heals, restores mana, a town portal
# (herodata.item_table). Then what a building makes now (one of: a worker, an army unit, a hero, research, its
# next tier): before, it saw only that it made something
A_HERO = F_BASE
HERO_ABIL_F, HERO_ITEM_F = 4, 5
HERO_F = 4 * HERO_ABIL_F + 6 * HERO_ITEM_F  # 46
A_PROD = A_HERO + HERO_F
PROD_F = 5
WORKER_TYPES = (1752196449, 1869636975, 1969316719, 1702327152)  # hpea, opeo, uaco, ewsp
COOLDOWN_SCALE = 60.0
# inputs v3 (--inputs 3): the enemy's units and buildings it saw and no longer sees, where it saw them last (View.ghosts:
# after the entities in sight; a building until its place is in sight again, a unit for MEMORY_SECONDS): whether an
# entity is such a memory, and how long ago it was seen / 60 s (at most 1). Their place counts as in sight within
# SIGHT of one of the player's units (most units see 800 at night and more by day)
A_MEM = A_PROD + PROD_F
MEM_F = 2
MAX_GHOSTS, MEMORY_SECONDS, SIGHT = 40, 60.0, 800.0
# and the global vector (between the strategy block and the privileged one: a network without them reads what it
# knows and its strategy block as before): the time of day (sin, cos of its angle; Units\MiscData.txt: a day of
# 480 s, dawn at 6, dusk at 18; melee games start at 8) and whether it is night (creeps sleep, night elves hide)
DAY_SECONDS, DAWN, DUSK, START_HOUR = 480.0, 6.0, 18.0, 8.0
INPUTS = 1
F = F_BASE
GX = 0


# privileged information (selfplay --opsd-coef, --opsd-share: self-distillation from the same network given what the
# player can't see, the user's idea): one more entity channel, an enemy out of sight (the privileged view has every
# enemy unit), and one more global number, whether the view is privileged (the policy then also reads the privileged
# block: model.FullGameNet "privileged"). A view that is not privileged has zeros there.
PRIVILEGED = os.environ.get("WARCRAFTSIM_PRIVILEGED", "0") == "1"
A_HID = -1  # (set_inputs: the last entity channel when PRIVILEGED)


def set_inputs(v: int, privileged: bool | None = None) -> None:
    """The run's inputs (selfplay --inputs): 1 as before, 2 the hero and production blocks and the clock, 3 and the
    enemies it saw before (View.ghosts); `privileged` (None: as it was) the privileged channels after them."""
    global INPUTS, F, GX, PRIVILEGED, A_HID
    INPUTS = int(v)
    if privileged is not None:
        PRIVILEGED = bool(privileged)
    F = F_BASE if INPUTS < 2 else A_PROD + PROD_F if INPUTS < 3 else A_MEM + MEM_F
    GX = 0 if INPUTS < 2 else 3
    if PRIVILEGED:
        A_HID, F, GX = F, F + 1, GX + 1
    os.environ["WARCRAFTSIM_INPUTS"] = str(INPUTS)  # (processes started anew)
    os.environ["WARCRAFTSIM_PRIVILEGED"] = str(int(PRIVILEGED))


set_inputs(int(os.environ.get("WARCRAFTSIM_INPUTS", 1)))
# the unit features in use (selfplay --features): channels from here up to F_BASE are zeros, so their weights get no
# gradient and the network is as one without them (29: as before 2026-10-12, 50: without the last order)
ACTIVE_F = int(os.environ.get("WARCRAFTSIM_FEATURES", F))  # (also from the environment: processes started anew)
# selfplay --tech-mask: orders the player lacks the tech for (techtree: a building, a tier, an altar; a research at
# its last level or under way; a hero past the limit or one it has) are masked with those it can't pay for
TECH_MASK = os.environ.get("WARCRAFTSIM_TECH_MASK", "0") == "1"
# selfplay --valid-mask: the network masks each hero's orders it can't carry out now (model.FullGameNet "valid":
# an ability not learned or cooling down, no skill point for it, an empty inventory slot); needs inputs 2
VALID_MASK = os.environ.get("WARCRAFTSIM_VALID_MASK", "0") == "1"


# selfplay --neutral: the neutral buildings' wares and the tavern heroes' orders in every network's vocabulary
# (fullgame/neutral.py; model.load extends a checkpoint's and grows its network); 2: no mercenaries (neutral.who_buys)
NEUTRAL = int(os.environ.get("WARCRAFTSIM_NEUTRAL", "0"))
SHOP_IDS = np.array([int.from_bytes(c.encode(), "big") for c in ("ntav", "ngme", "nmer", "nmrk")], np.int64)


CREEP_ARMY, CREEP_FROM = 6, 150.0  # --macros: creep the natural's camp with 6 army units or more, from second 150
EXPAND_PENDING_STEPS = 120  # --macros: an expand order under way this long (the walk, ~30 s): no other (refused: none)
# selfplay --macros: the agent's own orders for what the game spreads over many (fullgame/macros.py: expand)
MACROS = int(os.environ.get("WARCRAFTSIM_MACROS", "0") or 0)  # (its level: 2 adds pick up, macros.LEVELS)


def set_macros(level: int) -> None:
    global MACROS
    MACROS = int(level)
    os.environ["WARCRAFTSIM_MACROS"] = str(MACROS)


# --macros-off: macro orders the policy may not give (masked), so it learns their low-level orders itself from the lessons
# and rewards (the user, 2026-10-25: diverse play without the army-mix, creep and expand orders' help)
MACROS_OFF = frozenset(x for x in os.environ.get("WARCRAFTSIM_MACROS_OFF", "").split(",") if x)


def set_macros_off(names) -> None:
    global MACROS_OFF
    MACROS_OFF = frozenset(n for n in names if n)
    os.environ["WARCRAFTSIM_MACROS_OFF"] = ",".join(sorted(MACROS_OFF))


def set_neutral(v: int) -> None:
    global NEUTRAL
    NEUTRAL = int(v)
    os.environ["WARCRAFTSIM_NEUTRAL"] = str(NEUTRAL)


def set_masks(tech: bool, valid: bool) -> None:
    global TECH_MASK, VALID_MASK
    TECH_MASK, VALID_MASK = bool(tech), bool(valid)
    os.environ["WARCRAFTSIM_TECH_MASK"], os.environ["WARCRAFTSIM_VALID_MASK"] = str(int(TECH_MASK)), str(int(VALID_MASK))
# the global vector's strategy block (strategy conditioning, fullgame/strategy.py: the policy's target strategy,
# zeros for none), between what the player knows and the privileged block; 0: none (a run's networks all
# have it or none: set_z from the run's network)
ZDIM = int(os.environ.get("WARCRAFTSIM_Z", 0))


def set_z(n: int) -> None:
    global ZDIM
    ZDIM = int(n)
    os.environ["WARCRAFTSIM_Z"] = str(ZDIM)  # (processes started anew: videos)
# the global vector's privileged block (the value head alone reads it, model.FullGameNet.value; AlphaStar's
# value with the opponent's information was its largest RL ablation: 82% against 22%): the opponent's gold,
# lumber, food used and cap, upkeep; then for each side (own, enemy), all its units seen or not: workers,
# army units, buildings, heroes, the heroes' levels, the army's hit points and maximum, its centre (x mirrored, y)
PRIV = 5 + 2 * 9
HERO = 1
TYPE_CODE = 0x1000000  # order ids at or above this are unit / building / upgrade codes
# events: a building's production (a = the building)
TRAIN_START, TRAIN_FINISH, TRAIN_CANCEL = 5, 6, 7
PRODUCTION_START = {5, 8, 11}  # train, research, upgrade started
PRODUCTION_END = {6, 7, 9, 10, 12, 13}  # finished or cancelled


def load_game(path: str | Path) -> dict:
    z = np.load(path)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(str(z["meta"]))
    return out


def _step_slices(col: np.ndarray, steps: int) -> np.ndarray:
    """[steps + 1] boundaries of each step's rows (rows are in step order)."""
    return np.searchsorted(col, np.arange(steps + 1), side="left")


def is_hero_code(code: str) -> bool:
    """A hero's unit code (an altar's or a tavern's: "Hpal", "Npbm"; not research, "Rhde")."""
    return len(code) == 4 and code[0].isupper() and code[0] != "R" and code[1:].islower()


def hero_classes(vocab: dict) -> np.ndarray:
    """The order classes that make a hero: trained at an altar (immediate) or hired at a tavern (unit target)."""
    return np.asarray([c for c, (o, k) in enumerate(vocab["orders"], start=1)
                       if o >= TYPE_CODE and k in (IMMEDIATE, UNIT) and is_hero_code(rawcode(o))], np.int64)


def first_hero_free(costs: np.ndarray, vocab: dict) -> np.ndarray:
    """`costs` with the heroes' gold and lumber 0: a player's first hero costs only its food (1.29: the hero token,
    bj_MELEE_STARTING_HERO_TOKENS; tested at the altar and the tavern, 2026-10-21. Before, the mask held a first
    hero back until 425 gold and 100 lumber)."""
    c = np.array(costs, copy=True)
    c[hero_classes(vocab), :2] = 0
    return c


REVIVE = (0.4, 0.1, 4.0, 700)  # MiscGame: ReviveBaseFactor, ReviveLevelFactor, ReviveMaxFactor, HeroMaxReviveCostGold


def revive_costs(enc: "Encoder", rows: np.ndarray, player: int, dead: np.ndarray | None = None) -> np.ndarray:
    """The encoder's costs with each dead hero's order (an altar's, the one that trained it; a tavern hero's hire) at
    its revival's: its gold times 0.4 + 0.1 per level past the first (at most 4 times it, and 700), no lumber; the costs
    as they are when no hero of the player is dead (the mask had held a dead hero's revival back for good, the audit
    of 2026-10-22). `dead`: the player's dead heroes' rows, when the caller has them."""
    mine = (rows[(rows[:, C_OWNER] == player) & (rows[:, C_FLAGS] & HERO > 0) & (rows[:, C_FLAGS] & DEAD > 0)]
            if dead is None else dead)
    return _revive(enc, mine[:, [C_TYPE, C_HLEVEL]].tolist() if len(mine) else [])


def _revive(enc: "Encoder", dead: list) -> np.ndarray:
    """revive_costs from the dead heroes' [type, level]s (View.step's: featnative.side's, or their rows')."""
    if not dead:
        return enc.costs
    c = np.array(enc.costs, copy=True)
    base, per, most, cap = REVIVE
    for t, lvl in dead:
        k = enc.order_index.get((int(t), IMMEDIATE)) or enc.order_index.get((int(t), UNIT))
        if k is not None:
            g = int(enc.costs[k][0])
            c[k, 0] = min(int(g * (base + per * max(int(lvl) - 1, 0))), int(g * most), cap)
            c[k, 1] = 0
    return c


def affordable(costs: np.ndarray, gold: int, lumber: int, fu: int, fc: int, reserve=None, also=None) -> np.ndarray:
    """[n_orders + 1] bool: the order classes the player can pay for (row 0, no order: always). `reserve`
    (order class, gold, lumber), --z-build: what the target's next item needs is kept from the others
    (the item itself as before; free orders stay). `also`: another one kept (a tavern hero's hire on its way)."""
    c = costs
    fc = min(fc, FOOD_MAX)  # (the game's ceiling: farms past it give nothing; at 100 every unit was refused; audit 4)
    avail = (c[:, 0] <= gold) & (c[:, 1] <= lumber) & ((c[:, 2] == 0) | (fu + c[:, 2] <= fc))
    kept = [r for r in (reserve, also) if r is not None]
    if kept:
        rg, rl = sum(r[1] for r in kept), sum(r[2] for r in kept)
        keep = {r[0]: avail[r[0]] for r in kept}
        avail &= ((c[:, 0] <= max(0, gold - rg)) & (c[:, 1] <= max(0, lumber - rl))) | (c[:, :2].sum(1) == 0)
        for k, v in keep.items():
            avail[k] = v
    avail[0] = True
    return avail


def relabel(order: int, kind: int, target_known: bool, target_is_tree: bool) -> tuple[int, int] | None:
    """A recorded order as (order id, kind) of the vocabulary, None: not a decision to learn."""
    if order in DROPPED_ORDERS:
        return None
    if order == AIMOVE:
        order = MOVE
    if kind == 2:
        if target_known:
            return order, UNIT
        if target_is_tree:
            return order, TREE
        return None  # an item or a unit this player didn't see
    return order, {0: IMMEDIATE, 1: POINT, 3: SKILL}[kind]


def build_vocab(paths: list[str | Path], min_count: int = 5) -> dict:
    """Unit types, current orders, order classes and upgrades seen in the games."""
    types, cur, orders, upgrades = Counter(), Counter(), Counter(), Counter()
    names: dict[int, str] = {}
    for p in paths:
        g = load_game(p)
        if not names:  # the game's order ids by name (the same in every game)
            names = {int(v): k for k, v in (g["meta"].get("order_names") or {}).items()}
        u = g["units"]
        types.update(u[:, C_TYPE].tolist())
        cur.update(u[u[:, C_ORDER] != 0, C_ORDER].tolist())
        trees = set(g["trees"][:, 0].tolist())
        ids = set(u[:, C_ID].tolist())
        for r in g["orders"]:
            lab = relabel(int(r[2]), int(r[3]), int(r[6]) in ids, int(r[6]) in trees)
            if lab is not None:
                orders[lab] += 1
        ev = g["events"]
        upgrades.update(ev[ev[:, 1] == RESEARCH_FINISH, 3].tolist())
    keep = lambda c: [k for k, n in c.most_common() if n >= min_count]  # noqa: E731
    return {"types": keep(types), "current_orders": keep(cur)[:255],
            "orders": [list(k) for k in keep(orders)], "upgrades": keep(upgrades),
            "counts": {"orders": [orders[tuple(k)] for k in keep(orders)]},
            "order_names": {str(k): v for k, v in names.items()}}


@lru_cache(maxsize=8192)
def _rawcode(v: int) -> str:
    return v.to_bytes(4, "big").decode("latin-1")


def rawcode(v: int) -> str:  # (memoised: ~30 a step in the tech mask's key alone)
    return _rawcode(int(v))


def order_label(order: int, kind: int, names: dict[int, str]) -> str:
    """An order class for people: "attack (unit)", "train hpea", "build hbar (point)", ..."""
    if order >= TYPE_CODE:  # a unit, building or upgrade code: train / build / research / learn
        code = rawcode(order)
        verb = {IMMEDIATE: "research" if code.startswith("R") else "train", POINT: "build",
                SKILL: "learn"}.get(kind, "")  # (upgrade codes start with R)
        return f"{verb} {code}".strip()
    name = names.get(order, str(order))
    return name if kind == IMMEDIATE else f"{name} ({KIND_NAMES[kind]})"


def demo_order_names(runs_dir: str | Path) -> dict[int, str]:
    """The game's order names by id, from a recorded game's meta (runs/fullgame/*/game*.npz)."""
    for p in sorted(Path(runs_dir).glob("fullgame/*/game*.npz"))[:1]:
        try:
            with np.load(p) as z:
                return {int(v): k for k, v in (json.loads(str(z["meta"])).get("order_names") or {}).items()}
        except (OSError, ValueError, KeyError):
            pass
    return {}


def describe_spaces(vocab: dict, order_names: dict[int, str] | None = None, agents: str = "") -> dict:
    """The observation and action spaces for people, in the format the dashboard shows
    (puffer.tasks.describe_spaces): run.json "spaces" of a self-play run."""
    names = order_names or {int(k): v for k, v in (vocab.get("order_names") or {}).items()}
    enc = Encoder(vocab)
    ent = (["own", "enemy", "neutral", "x (mirrored: own base left)", "y", "hp / max hp", "max hp / 1000",
            "mana / max mana", "max mana / 1000"] + [f"flag {b}" for b in FLAG_BITS]
           + ["hero level / 10", "resource / 12500 (mines)", "skill points / 3", "sin facing", "cos facing (mirrored)",
              "has an order", "queued production / 5", "producing", "a worker on lumber",
              "hit points' change since the last step / max hp"]
           + [f"{f}({k}π{c})" for k in POS_FREQS for c in ("x", "y") for f in ("sin", "cos")]
           + ["last order's target: dx / 1536 (mirrored)", "last order's target: dy / 1536", "last order: a point",
              "last order: a unit", "last order: steps ago / 40"])
    if F > F_BASE:
        ent += ([f"hero ability {k + 1}: {w}" for k in range(4) for w in ("level / 3", "can be cast now",
                                                                           "cooldown left / 60 s", "can learn now")]
                + [f"hero item {k + 1}: {w}" for k in range(6) for w in ("held", "usable", "heals", "restores mana",
                                                                          "town portal")]
                + [f"making: {w}" for w in ("a worker", "an army unit", "a hero", "research", "its next tier")])
    if INPUTS >= 3:
        ent += ["an enemy it remembers (out of sight now)", "seen seconds ago / 60"]
    glob = (["gold / 1000", "lumber / 1000", "food used / 100", "food cap / 100", "upkeep / 2", "time (steps / 1800)"]
            + [f"own race: {r}" for r in RACES] + [f"enemy race: {r}" for r in RACES]
            + [f"upgrade {rawcode(u)} (level / 3)" for u in vocab["upgrades"]]
            + [f"strategy: slot {k} {w}" for w in ("made", "time / horizon") for k in range(ZDIM // 2)]
            + (["time of day: sin", "time of day: cos", "night"] if GX else [])
            + [f"privileged (value only): opponent {k}" for k in ("gold / 1000", "lumber / 1000", "food used / 100",
                                                                   "food cap / 100", "upkeep / 2")]
            + [f"privileged (value only): {side} {k}" for side in ("own", "enemy")
               for k in ("workers / 20", "army units / 30", "buildings / 30", "heroes / 3", "hero levels / 20",
                         "army hp / 3000", "army max hp / 3000", "army centre x", "army centre y")])
    orders = ["none"] + [order_label(int(o), int(k), names) for o, k in vocab["orders"]]
    return {"observation": {"size": f"{enc.G} + up to {MAX_ENT} × {F}", "blocks": [
                {"name": "global token", "rows": 1, "features": glob},
                {"name": f"entities: own units first (up to {MAX_OWN}), then the enemy's and neutral ones it sees",
                 "rows": MAX_ENT, "features": ent},
                {"name": f"each entity's unit type (one of {enc.n_types}, embedded)", "rows": MAX_ENT,
                 "features": ["type"]},
                {"name": f"each entity's current order (one of {enc.n_cur}, embedded)", "rows": MAX_ENT,
                 "features": ["current order"]}]},
            "actions": {"units": f"up to {MAX_OWN}", "sizes": [enc.n_orders, MAX_ENT, BINS, BINS],
                        "heads": [
                            {"name": "order", "size": enc.n_orders, "options": orders, "used_by": []},
                            {"name": "target unit (pointer)", "size": f"≤ {MAX_ENT}", "options": ["any entity it sees"],
                             "used_by": ["an order on a unit"]},
                            {"name": "point x", "size": BINS, "options": [f"{BINS} bins of {SCALE / BINS_PER_UNIT:.0f}"],
                             "used_by": ["an order at a point or on a tree (the nearest)"]},
                            {"name": "point y (given x)", "size": BINS, "options": [f"{BINS} bins"],
                             "used_by": ["an order at a point or on a tree"]}],
                        "masks": "Orders: those the unit's type got in the demonstrations (none always); "
                                 "the pointer: the entities in view. Harvest orders to workers already harvesting "
                                 "are not sent (the demonstrations' AI re-issued them without effect)."
                                 + (" Orders the player lacks the tech for (--tech-mask)." if TECH_MASK else "")
                                 + (" A hero's orders it can't carry out now (--valid-mask)." if VALID_MASK else "")},
            "agents": agents, "reward": "+1 for a win, -1 for a loss, 0 for a tie (the time limit), at the end.",
            "reward_scale": 1}


_FLAG_BITS = np.array(FLAG_BITS, np.int64)


def _sorted_map(d: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = np.fromiter(d.keys(), np.int64, len(d))
    vals = np.fromiter(d.values(), np.int64, len(d))
    order = np.argsort(keys)
    return keys[order], vals[order]


def _lookup(x: np.ndarray, keys: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """d.get(v, 0) for every v in x, with d as sorted (keys, vals)."""
    if len(keys) == 0:
        return np.zeros(len(x), np.int64)
    i = np.searchsorted(keys, x)
    i = np.minimum(i, len(keys) - 1)
    return np.where(keys[i] == x, vals[i], 0)


def _bin(v: np.ndarray) -> np.ndarray:
    """A position in a player's frame -> its bin."""
    return np.clip(np.floor((np.asarray(v) + BIN_RANGE) * BINS_PER_UNIT).astype(np.int64), 0, BINS - 1)


def bin_center(b: np.ndarray | int) -> np.ndarray | float:
    """A bin -> the position in a player's frame at its middle."""
    return (np.asarray(b) + 0.5) / BINS_PER_UNIT - BIN_RANGE


def world_bins(x, y, sign: float) -> tuple[np.ndarray, np.ndarray]:
    """A point in the world -> its x and y bins in a player's (mirrored) view."""
    fx_, fy_ = frame_xy(x, y, sign)
    return _bin(fx_), _bin(fy_)


def bins_world(bx: int, by: int, sign: float) -> tuple[float, float]:
    """x and y bins in a player's view -> the point in the world (bin_center and _unsqueeze in Python's floats: the same
    float64 arithmetic, without numpy's calls on numbers, ~80k instructions an order)."""
    ox, oy = own_start(sign)
    e = (((int(bx) + 0.5) / BINS_PER_UNIT - BIN_RANGE) - ANCHOR_X) * SCALE
    D = STARTS[1][0] - STARTS[0][0]
    if not abs(D - DUEL_D) < 1e-6:
        k = (DUEL_D - 2 * NEAR) / (D - 2 * NEAR)
        e = e if e <= NEAR else e + (D - DUEL_D) if e >= DUEL_D - NEAR else NEAR + (e - NEAR) / k
    return float(ox + sign * e), float(oy + ((int(by) + 0.5) / BINS_PER_UNIT - BIN_RANGE) * SCALE)


class Encoder:
    def __init__(self, vocab: dict, costs: np.ndarray | None = None):
        """`costs` (costs.order_costs: gold, lumber, food per order class): the views also give
        each step's availability mask ("avail": what the player can pay for)."""
        self.vocab = vocab
        self.costs = costs
        # the first hero is free (the game's hero token: only its food; at the altar or the tavern): the costs then
        self.costs_first = first_hero_free(costs, vocab) if costs is not None else None
        # --macros: the expand order's class, and per race the class whose cost and mask it takes (its hall's build;
        # undead: haunting a mine)
        ix = {tuple(k): i + 1 for i, k in enumerate(vocab["orders"])}
        self.expand_class = ix.get((990001, IMMEDIATE))
        self.creep_class = ix.get((990002, IMMEDIATE))
        self.mix_class = ix.get((990003, IMMEDIATE))
        self.pickup_class = ix.get((990004, IMMEDIATE))
        self.expand_via = {r: ix.get((int.from_bytes(h.encode(), "big"), UNIT if r == "undead" else POINT))
                           for r, h in (("human", "htow"), ("orc", "ogre"), ("undead", "ugol"), ("nightelf", "etol"))}
        self.type_index = {t: i + 1 for i, t in enumerate(vocab["types"])}  # 0: unknown
        self.cur_index = {o: i + 1 for i, o in enumerate(vocab["current_orders"])}
        self.order_index = {tuple(k): i + 1 for i, k in enumerate(vocab["orders"])}  # 0: no order
        self.upgrade_index = {u: i for i, u in enumerate(vocab["upgrades"])}
        self.n_types, self.n_cur = len(self.type_index) + 1, len(self.cur_index) + 1
        self.n_orders = len(self.order_index) + 1
        self.order_kind = np.array([IMMEDIATE] + [k[1] for k in vocab["orders"]], np.int64)
        self.G0 = 6 + 2 * len(RACES) + len(self.upgrade_index)  # what the player knows
        self.ZD = ZDIM  # the target strategy (strategy.vector; none: zeros)
        self.GX = GX  # the clock (inputs 2)
        self.G = self.G0 + self.ZD + self.GX + PRIV  # and what only the value reads (privileged: the whole game, View.priv)
        self.tech = TECH_MASK
        # the lookups as sorted arrays (vectorized; per-entity dict lookups were most of a step's cost)
        self._type_keys, self._type_vals = _sorted_map(self.type_index)
        self._cur_keys, self._cur_vals = _sorted_map(self.cur_index)

    @staticmethod
    def side(units_step0: np.ndarray, player: int) -> float:
        """-1 if the player's base is on the right (its view is mirrored), else 1."""
        own = units_step0[(units_step0[:, C_OWNER] == player) & (units_step0[:, C_FLAGS] & STRUCTURE > 0)]
        return -1.0 if len(own) and own[:, C_X].mean() > MAP_CX else 1.0

    def entities(self, rows: np.ndarray, player: int, sign: float, production=None,
                 lumber: set[int] | None = None, ghosts: tuple[np.ndarray, np.ndarray] | None = None,
                 omniscient: bool = False) -> tuple[np.ndarray, ...]:
        """One step's units -> (rows in entity order, n own, float features, types, current orders). `ghosts`
        (inputs 3, View.remembered): the enemies seen before and out of sight now, (their rows as last seen, seconds
        since), after those in sight (self.ghost0: the first). `omniscient` (the privileged view): every enemy unit,
        those out of sight after those in sight and marked (A_HID)."""
        if ghosts is None and not omniscient and A_HID < 0:  # (in C, featnative.py: the same features, ~10x faster)
            featnative = _mod("featnative")
            if featnative.lib() is not None:
                sel, n_own, f, types, cur = featnative.entities(self, _mod(""), rows, player, sign, production, lumber)
                if F > F_BASE:
                    self._hero_block(sel, n_own, f)
                    making = production[2] if production is not None and len(production) > 2 else None
                    if making:  # (inputs 2: what each own building makes, View.making)
                        k = _lookup(sel[:n_own, C_ID], *_sorted_map({b: m[0] + 1 for b, m in making.items()}))
                        hit = np.nonzero(k > 0)[0]
                        f[hit, A_PROD + k[hit] - 1] = 1.0
                return sel, n_own, f, types, cur
        alive = rows[:, C_FLAGS] & DEAD == 0
        vis = (rows[:, C_VIS] >> player) & 1 == 1
        own = rows[alive & (rows[:, C_OWNER] == player)]
        if len(own) > MAX_OWN:  # (past MAX_OWN: heroes, the army, buildings that make something, workers, then farms and
            fl = own[:, C_FLAGS]  # towers; the rows are in first-seen order, so the newest were cut: 12-17% of the games,
            rank = np.where(fl & HERO > 0, 0, np.where(fl & (STRUCTURE | WORKER) == 0, 1,  # won 2-5%; audit 2)
                            np.where(fl & STRUCTURE > 0, np.where(np.isin(own[:, C_TYPE], PASSIVE_IDS), 4, 2), 3)))
            own = own[np.argsort(rank, kind="stable")]
        own = own[:MAX_OWN]
        enemy = rows[alive & (rows[:, C_OWNER] == 1 - player) & vis]
        hidden = rows[alive & (rows[:, C_OWNER] == 1 - player) & ~vis] if omniscient else None
        if hidden is not None:
            enemy = np.concatenate([enemy, hidden])
        known = vis
        if NEUTRAL:  # (--neutral: the shops always, as the game's minimap shows the neutral buildings from the start)
            known = vis | np.isin(rows[:, C_TYPE], SHOP_IDS)
        neutral = rows[alive & (rows[:, C_OWNER] != player) & (rows[:, C_OWNER] != 1 - player) & known]
        parts = [own, enemy, neutral]
        if ghosts is not None and len(ghosts[0]):
            parts.append(ghosts[0].astype(rows.dtype, copy=False)[:, :rows.shape[1]])
        sel = np.concatenate(parts)[:MAX_ENT]
        n, n_own = len(sel), len(own)
        self.ghost0 = g0 = min(n, n_own + len(enemy) + len(neutral))
        f = np.zeros((n, F), np.float32)
        f[:n_own, 0] = 1
        f[n_own:n_own + len(enemy), 1] = 1
        f[n_own + len(enemy):g0, 2] = 1
        if hidden is not None and len(hidden):  # (the privileged view: the enemies out of sight)
            k0 = n_own + len(enemy) - len(hidden)
            f[k0:min(n, n_own + len(enemy)), A_HID] = 1.0
        if g0 < n:  # (inputs 3: the enemies it remembers; with --neutral the neutral buildings too)
            enemy_ghost = sel[g0:, C_OWNER] == 1 - player
            f[g0:, 1], f[g0:, 2] = enemy_ghost, ~enemy_ghost
            f[g0:, A_MEM] = 1
            f[g0:, A_MEM + 1] = np.minimum(ghosts[1][:n - g0] / 60.0, 1.0)
        f[:, 3], f[:, 4] = frame_xy(sel[:, C_X], sel[:, C_Y], sign)
        f[:, 5] = sel[:, C_HP] / np.maximum(sel[:, C_MAXHP], 1)
        f[:, 6] = sel[:, C_MAXHP] / 1000.0
        f[:, 7] = np.where(sel[:, C_MAXMANA] > 0, sel[:, C_MANA] / np.maximum(sel[:, C_MAXMANA], 1), 0)
        f[:, 8] = sel[:, C_MAXMANA] / 1000.0
        f[:, 9:9 + len(FLAG_BITS)] = (sel[:, C_FLAGS][:, None] & _FLAG_BITS) > 0
        f[n_own:, 9 + FLAG_BITS.index(512)] = 0.0  # (an enemy's illusion looks real: the game tells its owner only;
        # the flag had told a Blademaster's images apart; audit 4)
        a = 9 + len(FLAG_BITS)
        f[:, a] = sel[:, C_HLEVEL] / 10.0
        f[:, a + 1] = sel[:, C_RESOURCE] / 12500.0
        f[:n_own, a + 2] = sel[:n_own, C_SKILLPTS] / 3.0  # (an enemy hero's unspent points: the game shows none)
        rad = np.radians(sel[:, C_FACING].astype(np.float32))
        f[:, a + 3] = np.sin(rad)
        f[:, a + 4] = sign * np.cos(rad)
        f[:, a + 5] = sel[:, C_ORDER] != 0
        for i, k in enumerate(POS_FREQS):
            px, py = math.pi * k * f[:, 3], math.pi * k * f[:, 4]
            f[:, A_POS + 4 * i:A_POS + 4 * i + 4] = np.stack([np.sin(px), np.cos(px), np.sin(py), np.cos(py)], 1)
        # production (buildings' current order stays 0 while they train): what the player queued, and
        # whether something is being made (View keeps both from its orders and the events)
        if production is not None:
            queued, busy = production[:2]
            ids = sel[:, C_ID]
            if queued:
                f[:, a + 6] = np.minimum(_lookup(ids, *_sorted_map(queued)), 5) / 5.0
        if lumber:  # own workers the player sent to the trees (their current order is the same as gold miners')
            f[:n_own, a + 8] = np.isin(sel[:n_own, C_ID], np.fromiter(lumber, np.int64, len(lumber)))
        if production is not None and busy:  # (it sat under `lumber`: a building's "making" read 0 while no worker
            f[:, a + 7] = np.isin(ids, np.fromiter(busy, np.int64, len(busy)))  # chopped; the audit of 2026-10-22)
        if F > F_BASE:
            self._hero_block(sel, n_own, f)
            making = production[2] if production is not None and len(production) > 2 else None
            if making:  # (inputs 2: what each own building makes, View.making)
                k = _lookup(sel[:n_own, C_ID], *_sorted_map({b: m[0] + 1 for b, m in making.items()}))
                hit = np.nonzero(k > 0)[0]
                f[hit, A_PROD + k[hit] - 1] = 1.0
        self.counts = (int((alive & (rows[:, C_OWNER] == player)).sum()), n_own + len(enemy) + len(neutral))
        types = _lookup(sel[:, C_TYPE], self._type_keys, self._type_vals)
        cur = _lookup(sel[:, C_ORDER], self._cur_keys, self._cur_vals)
        cur[n_own:][sel[n_own:, C_ORDER] >= TYPE_CODE] = 0  # (an enemy worker walking to build: what it will build
        return sel, n_own, f, types, cur                      # is not shown before the frame stands; audit 4)

    @staticmethod
    def _hero_block(sel: np.ndarray, n_own: int, f: np.ndarray) -> None:
        """The heroes' A_HERO features (inputs 2), from rows with their items and abilities (ROW_COLS)."""
        if sel.shape[1] < ROW_COLS:
            return
        heroes = np.nonzero(sel[:, C_FLAGS] & HERO)[0]
        if not len(heroes):
            return
        herodata = _mod("herodata")
        table, items = herodata.hero_table(), herodata.item_table()
        for i in heroes.tolist():
            r = sel[i].tolist()
            own = i < n_own  # (an enemy's: its levels and items, as the game shows them)
            slots = table.get(r[C_TYPE], ())
            for k in range(4):
                lvl, cd = r[C_ABIL0 + 2 * k], r[C_ABIL0 + 2 * k + 1]
                a = A_HERO + HERO_ABIL_F * k
                f[i, a] = lvl / 3.0
                ab = slots[k] if k < len(slots) else None
                if not own or ab is None:
                    continue
                f[i, a + 2] = min(cd / 10.0 / COOLDOWN_SCALE, 1.0)
                cost = ab.mana[min(lvl, len(ab.mana)) - 1] if ab.mana and lvl > 0 else 0.0
                f[i, a + 1] = float(bool(ab.order) and lvl > 0 and cd == 0 and r[C_MANA] >= cost)
                f[i, a + 3] = float(r[C_SKILLPTS] > 0 and lvl < ab.levels and r[C_HLEVEL] >= ab.hero_level_for(lvl + 1))
            for k in range(6):
                it = r[C_ITEM0 + k]
                if it:
                    b = A_HERO + 4 * HERO_ABIL_F + HERO_ITEM_F * k
                    f[i, b] = 1.0
                    f[i, b + 1:b + 5] = items.get(it, (0, 0, 0, 0))

    def view(self, player: int, sign: float, races: list[int]) -> "View":
        return View(self, player, sign, races)

    def encode(self, game: dict, player: int, z: np.ndarray | None = None) -> dict:
        """The whole game from `player`'s side: arrays over its steps (padded to MAX_ENT). With costs:
        orders the player couldn't pay for at the step are no labels and queue nothing (the AI retries
        its train orders until it can afford them: 61% of its train and research orders were attempts
        the game refused, and they left stale queues), nor are train / research orders that started
        nothing within 2 steps at a building that wasn't busy (refused for tech or the hero limit);
        and "avail" is each step's availability mask. The side a policy played with the built-in AI
        advising it (collect.py --shadow): the AI's orders are the labels, the policy's own orders
        ("policy_orders") what the features count (queues, workers on lumber). `z`: the strategy block's input
        (strategy.vector; none: zeros)."""
        meta = game["meta"]
        T = meta["steps"] + 1
        u, pl, ev, od = game["units"], game["players"], game["events"], game["orders"]
        if F > F_BASE and "heroes" in game:  # (inputs 2: the heroes' items and abilities)
            u = hero_rows(u, game["heroes"])
        us, ps, es, os_ = (_step_slices(a[:, 0], T) for a in (u, pl, ev, od))
        sh = meta.get("shadow")
        po = game.get("policy_orders") if sh and sh["player"] == player else None
        pos = _step_slices(po[:, 0], T) if po is not None else None
        sign = self.side(u[us[0]:us[1]], player)
        trees = {int(r[0]): (int(r[2]), int(r[3])) for r in game["trees"]}
        races = [RACES.index(r) if r in RACES else 0 for r in meta["races"]]
        out = {"ent": np.zeros((T, MAX_ENT, F), np.float16), "type": np.zeros((T, MAX_ENT), np.int16),
               "cur": np.zeros((T, MAX_ENT), np.int16), "mask": np.zeros((T, MAX_ENT), bool),
               "n_own": np.zeros(T, np.int16), "glob": np.zeros((T, self.G), np.float32),
               "y_order": np.zeros((T, MAX_OWN), np.int16), "y_ptr": np.full((T, MAX_OWN), -1, np.int16),
               "y_x": np.full((T, MAX_OWN), -1, np.int16), "y_y": np.full((T, MAX_OWN), -1, np.int16),
               "avail": np.ones((T, self.n_orders), bool)}
        view = self.view(player, sign, races)
        view.z = z  # (a strategy as the input: bc's hindsight targets)
        view.step_seconds = float(meta.get("step_seconds", 0.5))
        for t in range(T):
            prow = pl[ps[t]:ps[t + 1]]
            me = prow[prow[:, 1] == player]
            opp = prow[prow[:, 1] == 1 - player]
            st = view.step(u[us[t]:us[t + 1]], me[0] if len(me) else None, ev[es[t]:es[t + 1]], t,
                           opp[0] if len(opp) else None)
            n = st["n"]
            out["ent"][t, :n], out["type"][t, :n], out["cur"][t, :n] = st["ent"], st["type"], st["cur"]
            out["mask"][t, :n] = True
            out["n_own"][t] = st["n_own"]
            out["glob"][t] = st["glob"]
            if st["avail"] is not None:
                out["avail"][t] = st["avail"]
            # labels: each own unit's last order in the step
            rows_t = od[os_[t]:os_[t + 1]]
            y, paid = self.step_labels(view, st, rows_t, me[0] if len(me) else None, ev[es[t]:es[min(t + 3, T)]], trees)
            for k in ("y_order", "y_ptr", "y_x", "y_y"):
                out[k][t] = y[k]
            if po is None:
                view.record_orders((int(r[1]), int(r[2]), int(r[3])) for r, ok in zip(rows_t, paid) if ok)
                view.note_orders((tuple(r[1:7]) for r, ok in zip(rows_t.tolist(), paid) if ok), t, trees)
            else:  # what the policy did (the game accepted it)
                view.record_orders((int(r[1]), int(r[2]), int(r[3])) for r in po[pos[t]:pos[t + 1]])
                view.note_orders((tuple(r[1:7]) for r in po[pos[t]:pos[t + 1]].tolist()), t, trees)
            view.track_harvest(rows_t if po is None else po[pos[t]:pos[t + 1]], st, trees)
        return out

    def step_labels(self, view: "View", st: dict, rows: np.ndarray, me, later: np.ndarray, trees: dict,
                    taught: bool = False) -> tuple[dict, list]:
        """One step's labels (y_order, y_ptr, y_x, y_y [MAX_OWN]) from the orders the AI gave in it
        (rows: step, unit, order, kind, x, y, target), in the state `st` (View.step; the view as it
        was then), and which of the orders count (paid for, and train / research orders that started
        something within `later`'s events: the step's and up to two after; a building busy already
        queues them). Self-play's distillation from the built-in AI advising (protocol.ShadowAI)
        labels its live steps with it, the demonstrations' encoder its recorded ones. `taught`: orders never carried
        out (the target builder's, the teacher's: they started nothing because they were not given; the filter
        dropped every train, research and hero label at an idle building, the audit of 2026-10-22)."""
        y = {"y_order": np.zeros(MAX_OWN, np.int16), "y_ptr": np.full(MAX_OWN, -1, np.int16),
             "y_x": np.full(MAX_OWN, -1, np.int16), "y_y": np.full(MAX_OWN, -1, np.int16)}
        index = st["index"]
        paid = self._paid(rows, st, me, trees)
        if self.costs is not None and not taught:  # train / research orders that started nothing: refused (tech, hero limit)
            started = {int(r[2]) for r in later.tolist() if r[1] in PRODUCTION_START}  # (a few events: np.isin cost more)
            paid = [ok and not (int(r[3]) == 0 and int(r[2]) >= TYPE_CODE and int(r[1]) in view.own_buildings
                                and int(r[1]) not in started and int(r[1]) not in view.busy)
                    for r, ok in zip(rows, paid)]
        for r, ok in zip(rows, paid):
            k = index.get(int(r[1]))
            if not ok or k is None or k >= min(st["n_own"], MAX_OWN):
                continue
            target = int(r[6])
            lab = relabel(int(r[2]), int(r[3]), target in index, target in trees)
            c = self.order_index.get(lab) if lab is not None else None
            ttype = int(st["sel"][index[target], C_TYPE]) if target in index else None
            res = harvest_resource(lab[0], lab[1], ttype) if lab is not None else None
            same = True  # (gold from another mine than the worker's: a transfer, kept)
            if res == "gold" and target in index:
                known = view.mine_of.get(int(r[1]))
                tx, ty = st["sel"][index[target], C_X], st["sel"][index[target], C_Y]
                same = (known == target if known is not None
                        else math.hypot(float(st["sel"][k, C_X] - tx), float(st["sel"][k, C_Y] - ty)) < 1500.0)
            if c is None or redundant(lab[0], lab[1], int(st["sel"][k, C_ORDER]), res, view.assign.get(int(r[1])), same):
                continue
            y["y_order"][k] = c
            y["y_ptr"][k] = -1
            y["y_x"][k] = y["y_y"][k] = -1
            if lab[1] == UNIT:
                y["y_ptr"][k] = index[target]
            elif lab[1] in (POINT, TREE):
                x, yy = (trees[target] if lab[1] == TREE else (int(r[4]), int(r[5])))
                y["y_x"][k], y["y_y"][k] = world_bins(x, yy, view.sign)
        return y, paid


def _paid_orders(enc: "Encoder", rows: np.ndarray, st: dict, me, trees: dict) -> list[bool]:
    """Which of a step's recorded orders the player could pay for (costs deducted in order); orders
    of other players' units and orders that cost nothing: True. All True without costs."""
    ok = [True] * len(rows)
    if enc.costs is None or me is None:
        return ok
    gold, lumber, fu, fc = int(me[2]), int(me[3]), int(me[4]), int(me[5])
    index, n_own = st["index"], st["n_own"]
    for j, r in enumerate(rows):
        k = index.get(int(r[1]))
        if k is None or k >= n_own:
            continue
        lab = relabel(int(r[2]), int(r[3]), int(r[6]) in index, int(r[6]) in trees)
        c = enc.order_index.get(lab) if lab is not None else None
        if c is None:
            continue
        cg, cl, cf = (int(v) for v in (enc.costs_first if st.get("first_free") else enc.costs)[c])  # (a first hero: free)
        if cg == cl == cf == 0:
            continue
        if cg <= gold and cl <= lumber and (cf == 0 or fu + cf <= min(fc, FOOD_MAX)):
            gold, lumber, fu = gold - cg, lumber - cl, fu + cf
        else:
            ok[j] = False
    return ok


Encoder._paid = lambda self, rows, st, me, trees: _paid_orders(self, rows, st, me, trees)


# what the tech tree's tables miss (audit 8: refused in the game): a unit an upgrade enables (troll berserkers: the
# berserker upgrade turns the headhunters, ~59 refused a game as orc), and abilities a research teaches (footmen's
# defend ~23 a game, crypt fiends' burrow ~31 and web ~9)
UNIT_RESEARCH = {"otbk": "Robk"}
ABILITY_RESEARCH = {852055: "Rhde", 852056: "Rhde", 852533: "Rubu", 852534: "Rubu", 852211: "Ruwb", 852212: "Ruwb",
                    852213: "Ruwb"}  # (defend, undefend, burrow, unburrow, web, webon, weboff)


class TechMask:
    """--tech-mask: which order classes a player has the tech for (fullgame/techtree.py, the game's own
    tables): a unit or building whose requirements it has finished (a hall counts as each lower tier), a
    research's next level (not past its last, not one under way), a hero while it has fewer than 3 and none
    of that type, with what its place among them needs (an altar, then each tier). Orders without a unit,
    building or research code: always, but for those a research teaches (ABILITY_RESEARCH). Cached while what it
    has stays the same (most steps)."""

    def __init__(self, vocab: dict, race: str):
        from . import techtree as tt
        self.tt, self.race, self.tree, self.max_level = tt, race, tt.tree(), tt.research_levels()
        self.items = [(c, rawcode(o)) for c, (o, k) in enumerate(vocab["orders"], start=1)
                      if o >= TYPE_CODE and k in (IMMEDIATE, POINT, UNIT)]  # (UNIT: a neutral building's ware)
        self.wares = {c for c, (o, k) in enumerate(vocab["orders"], start=1) if o >= TYPE_CODE and k == UNIT}
        names = vocab.get("order_names") or {}  # (the game's ability tables, tt.ability_research; the hand list besides)
        table = tt.ability_research(race)
        self.abilities = [(c, table[names[str(o)]] if names.get(str(o)) in table else (ABILITY_RESEARCH[o], 1))
                          for c, (o, k) in enumerate(vocab["orders"], start=1)
                          if o < TYPE_CODE and (names.get(str(o)) in table or o in ABILITY_RESEARCH)]
        if self.wares:  # (in stock from this game second on: neutral.stock_times)
            from .neutral import stock_times
            self.stock = stock_times()
        self.n = len(vocab["orders"]) + 1
        self._key: tuple | None = None
        self._ok = np.ones(self.n, bool)
        self._stock_times = sorted(getattr(self, "stock", {}).values())

    def stocked(self, now: float) -> int:
        """How many wares are in stock by `now` (the mask changes with it)."""
        return bisect.bisect_right(self._stock_times, now)

    def allowed(self, done: frozenset, levels: dict[str, int], heroes: frozenset, researching: frozenset,
                now: float = 1e9, dead: frozenset = frozenset()) -> np.ndarray:
        """`done`: the player's finished buildings' types, `levels`: research -> level, `heroes`: its heroes'
        types (with one in training), `researching`: research under way, `now`: the game second (a neutral
        building's wares in stock)."""
        key = (done, tuple(sorted(levels.items())), heroes, researching, dead,
               tuple(sorted(c for c, v in getattr(self, "stock", {}).items() if v <= now)))
        if key == self._key:
            return self._ok
        tt = self.tt
        have = tt.counts_as(done, self.race) | {c for c, v in levels.items() if v > 0}
        ok = np.ones(self.n, bool)
        for c, code in self.items:
            if c in self.wares and self.stock.get(code, 0.0) > now:  # (not in stock yet)
                ok[c] = False
            elif code[0] == "R":  # research: its next level
                lvl = levels.get(code, 0)
                ok[c] = (lvl < self.max_level.get(code, 1) and code not in researching
                         and not tt.missing(code, self.race, have, lvl, self.tree)
                         and not tt.short_levels(code, lvl, levels, self.tree))
            elif tt.is_hero(code) and code in dead:  # revived at an altar, with the order that trained it (`dead`: its
                ok[c] = not tt.missing(code, self.race, have, 0, self.tree)  # dead heroes' types; a tavern's hero too, with
                # an altar: its hire's order is the altar's revival, play.BCAgent. A dead tavern hero was lost for good)
            elif tt.is_hero(code):
                ok[c] = len(heroes) < 3 and code not in heroes and not tt.missing(code, self.race, have, len(heroes), self.tree)
            else:
                ok[c] = (not tt.missing(code, self.race, have, 0, self.tree) and code not in researching
                         and (code not in UNIT_RESEARCH or levels.get(UNIT_RESEARCH[code], 0) > 0))
        for c, (r, a) in self.abilities:  # (at its level: inner fire, polymorph, bloodlust need level 2; audit 11)
            ok[c] = levels.get(r, 0) >= a
        self._key, self._ok = key, ok
        return ok


def hero_rows(units: np.ndarray, heroes: np.ndarray) -> np.ndarray:
    """A demonstration's unit rows (UNIT_COLS) with its heroes' items and abilities (collect.HERO_COLS: step, id,
    6 items, 4 ability levels, 4 cooldowns in 0.1 s) in the columns past them (ROW_COLS, as a live game's)."""
    out = np.zeros((len(units), ROW_COLS), np.int64)
    out[:, :units.shape[1]] = units
    if len(heroes) and len(units):
        ku = units[:, C_STEP].astype(np.int64) * (1 << 32) + units[:, C_ID]
        kh = heroes[:, 0].astype(np.int64) * (1 << 32) + heroes[:, 1]
        order = np.argsort(kh)
        kh = kh[order]
        j = np.minimum(np.searchsorted(kh, ku), len(kh) - 1)
        hit = np.nonzero(kh[j] == ku)[0]
        h = heroes[order[j[hit]]]
        out[hit, C_ITEM0:C_ITEM0 + 6] = h[:, 2:8]
        out[hit, C_ABIL0:C_ABIL0 + 8:2] = h[:, 8:12]
        out[hit, C_ABIL0 + 1:C_ABIL0 + 8:2] = h[:, 12:16]
    return out


class View:
    """What one player sees, step by step (demonstrations, or a live game): the entities, the
    global features, and the upgrades it has researched so far."""

    def __init__(self, enc: Encoder, player: int, sign: float, races: list[int]):
        self.enc, self.player, self.sign, self.races = enc, player, sign, races
        self.active_f = ACTIVE_F  # (an evaluation game sets F: every checkpoint has zero weights past its own)
        self.z: np.ndarray | None = None  # the target strategy (strategy.vector; the encoder's ZD numbers)
        self.zmask: np.ndarray | None = None  # --z-enforce: the orders the target allows (strategy.allowed)
        self.zmask_until = 0  # ... until this step (the strategies' horizon)
        # --z-build: the gold and lumber saved for the target's next item (its order class, its cost) that
        # the other orders may not spend (the policy spent all on units, and a next tier never came)
        self.reserve: tuple[int, int, int] | None = None
        self.upgrades = np.zeros(len(enc.upgrade_index), np.float32)
        self.own_buildings: set[int] = set()
        self.queued: dict[int, int] = {}  # building -> units / research ordered and not yet done
        self.busy: set[int] = set()  # buildings making something
        self.assign: dict[int, str] = {}  # own worker -> "gold" / "lumber": what its last orders sent it to
        self.mine_of: dict[int, int] = {}  # own worker -> the gold mine its last accepted gold order sent it to
        self.hp_last = (np.zeros(0, np.int64), np.zeros(0, np.int64))  # the units in view the last step (sorted ids)
        # and their hit points
        self.last_order: dict[int, tuple] = {}  # own unit -> (kind 0 none/1 point/2 unit, x, y, target, step)
        # inputs 2 and --tech-mask: what each own building makes now (its kind, A_PROD's, and code), research levels
        self.making: dict[int, tuple[int, int]] = {}
        self.dead_heroes: dict[int, int] = {}  # own dead heroes: type -> unit id (BCAgent revives them)
        self.levels: dict[int, int] = {}  # upgrade id -> its level (all of the player's research, not only the vocabulary's)
        self.step_seconds = 0.5  # (the clock: game seconds of a step)
        self.t_now = 0  # (the step: the tech mask's wares in stock)
        # privileged information (PRIVILEGED): act on the privileged view (a teacher game), or give it beside the
        # player's (dual: the self-distillation's teacher sees it)
        self.privileged_act = self.dual = False
        self.techmask: TechMask | None = None
        self.ghosts: dict[int, tuple[np.ndarray, int]] = {}  # (inputs 3) enemy unit -> its row when last seen, the step
        self.max_own = self.max_ent = 0  # the most own units and entities in view at a step (MAX_OWN, MAX_ENT: cut)

    def track_harvest(self, orders, st: dict, trees) -> None:
        """What each own worker harvests, from the step's recorded orders (step, unit, order, kind,
        x, y, target): a harvest order sets it, any other order (but the engine's own) clears it."""
        index, n_own = st["index"], st["n_own"]
        for r in orders:
            unit, order, kind, target = int(r[1]), int(r[2]), int(r[3]), int(r[6])
            k = index.get(unit)
            if k is None or k >= n_own or order in DROPPED_ORDERS:
                continue
            lab = relabel(order, kind, target in index, target in trees)
            ttype = int(st["sel"][index[target], C_TYPE]) if target in index else None
            res = harvest_resource(lab[0], lab[1], ttype) if lab is not None else None
            if res is None:
                self.assign.pop(unit, None)
            else:
                self.assign[unit] = res
                self.added(unit)

    def added(self, unit: int) -> None:
        """A unit's entry put in assign, mine_of or last_order: unless it is among the living units of the view's last
        step, the next step forgets those dicts' gone units by a full pass (step: by the units gone since otherwise)."""
        prev = self.__dict__.get("_units_prev")
        if prev is None or unit not in prev:
            self._keys_safe = False

    def note_orders(self, orders, t: int, trees: dict | None = None) -> None:
        """The player's accepted orders of step t, (unit, order id, recorded kind 0 immediate / 1 point /
        2 unit, x, y, target id): each unit's last one (the A_LAST features from step t + 1)."""
        for unit, order, kind, x, y, target in orders:
            if int(order) in DROPPED_ORDERS:
                continue
            if kind == 2 and trees and int(target) in trees:  # (a tree: the point it stands on)
                kind, (x, y) = 1, trees[int(target)]
            self.last_order[int(unit)] = ((1, float(x), float(y), 0, t) if kind == 1 else
                                          (2, 0.0, 0.0, int(target), t) if kind == 2 else (0, 0.0, 0.0, 0, t))
            self.added(int(unit))

    def record_orders(self, orders) -> None:
        """The player's orders of the last step (unit, order id, kind): train / research orders
        queue at their building (called after step(): they count from the next step on)."""
        for unit, order, kind in orders:
            if kind == 0 and order >= TYPE_CODE and unit in self.own_buildings:
                self.queued[unit] = self.queued.get(unit, 0) + 1
                if order >> 24 == 0x52:  # (a research queued behind units: under way until it starts or is cancelled, the
                    self.__dict__.setdefault("pending_research", {})[order] = (unit, getattr(self, "t_now", 0))
                    # mask had offered it again, the game refusing it; audit 11)

    def priv(self, rows: np.ndarray, opp: np.ndarray | None) -> np.ndarray:
        """The privileged block (PRIV): the opponent's player row (PLAYER_COLS) and every living unit."""
        out = np.zeros(PRIV, np.float32)
        if opp is not None:
            _, _, gold, lumber, fu, fc, upkeep = opp[:7].tolist()  # (Python's numbers: numpy's scalars' arithmetic
            # had cost more than the rest; the same quotients)
            out[:5] = gold / 1000.0, lumber / 1000.0, fu / 100.0, fc / 100.0, upkeep / 2.0
        featnative = _mod("featnative")
        if featnative.lib() is not None:  # (in C: the same numbers)
            featnative.priv_sides(_mod(""), rows, self.player, self.sign, out)
            return out
        live = rows[rows[:, C_FLAGS] & DEAD == 0]  # (the masks made once for both sides: ~5% of an actor's time before)
        fl, owner = live[:, C_FLAGS], live[:, C_OWNER]
        worker, struct, hero = fl & WORKER > 0, fl & STRUCTURE > 0, fl & HERO > 0
        army = ~worker & ~struct
        for i, who in enumerate((self.player, 1 - self.player)):
            m = owner == who
            am, hm = army & m, hero & m
            a = 5 + 9 * i
            n_army = np.count_nonzero(am)
            out[a:a + 7] = (np.count_nonzero(worker & m) / 20.0, n_army / 30.0, np.count_nonzero(struct & m) / 30.0,
                            np.count_nonzero(hm) / 3.0, live[hm, C_HLEVEL].sum() / 20.0, live[am, C_HP].sum() / 3000.0,
                            live[am, C_MAXHP].sum() / 3000.0)
            if n_army:
                out[a + 7], out[a + 8] = _frame_xy1(float(live[am, C_X].mean()), float(live[am, C_Y].mean()), self.sign)
        return out

    def remembered(self, rows: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        """(inputs 3) The enemy units and buildings the player saw and doesn't see now: (their rows as last seen,
        seconds since), buildings first, then the most recently seen, at most MAX_GHOSTS. Each is forgotten when it is
        in sight again (then an entity as any), when its place is (within SIGHT of one of the player's units: it
        moved, or died there), a unit after MEMORY_SECONDS; one seen dying, at once."""
        p = self.player
        enemy = rows[(rows[:, C_OWNER] == 1 - p) | ((rows[:, C_OWNER] != p) & (rows[:, C_FLAGS] & STRUCTURE > 0)
                                                    if NEUTRAL else False)]  # (--neutral: the shops too)
        seen = enemy[(enemy[:, C_VIS] >> p) & 1 == 1]
        dead = seen[:, C_FLAGS] & DEAD > 0
        for i in seen[dead, C_ID].tolist():
            self.ghosts.pop(i, None)
        live = seen[~dead]
        for r in live:
            self.ghosts[int(r[C_ID])] = (r.copy(), t)
        if not self.ghosts:
            return np.zeros((0, rows.shape[1]), rows.dtype), np.zeros(0, np.float32)
        now = set(live[:, C_ID].tolist())
        mine = rows[(rows[:, C_OWNER] == p) & (rows[:, C_FLAGS] & DEAD == 0)][:, [C_X, C_Y]].astype(np.float64)
        keep, gone = [], []
        for i, (r, t0) in self.ghosts.items():
            if i in now:
                continue
            age = (t - t0) * self.step_seconds
            near = len(mine) and float(((mine - r[[C_X, C_Y]]) ** 2).sum(1).min()) < SIGHT * SIGHT
            if near or (not r[C_FLAGS] & STRUCTURE and age > MEMORY_SECONDS):
                gone.append(i)
            else:
                keep.append((not r[C_FLAGS] & STRUCTURE, age, r))
        for i in gone:
            del self.ghosts[i]
        keep.sort(key=lambda x: (x[0], x[1]))
        keep = keep[:MAX_GHOSTS]
        if not keep:
            return np.zeros((0, rows.shape[1]), rows.dtype), np.zeros(0, np.float32)
        return np.stack([r for _, _, r in keep]), np.asarray([a for _, a, _ in keep], np.float32)

    def has_hero(self, rows: np.ndarray, own_hero: np.ndarray | None = None, hero_types: list | None = None) -> bool:
        """Whether the player has had a hero (dead ones stay among the rows) or makes one now: its first is free.
        `own_hero`: the rows' mask of the player's heroes, when the caller has it (or `hero_types`: their types)."""
        if hero_types is not None:
            had = bool(hero_types)
        else:
            if own_hero is None:
                own_hero = (rows[:, C_OWNER] == self.player) & (rows[:, C_FLAGS] & HERO > 0)
            had = bool(own_hero.any())
        return (had or any(m[0] == 2 for m in self.making.values())
                or bool(getattr(self, "hiring_heroes", ())))  # (a tavern hire on its way: the first hero; audit 10)

    def tech_allowed(self, rows: np.ndarray, built: np.ndarray | None = None, own_hero: np.ndarray | None = None,
                     dead_rows: np.ndarray | None = None, done_types: list | None = None, hero_types: list | None = None,
                     dead_pairs: list | None = None) -> np.ndarray:
        """--tech-mask: the order classes the player has the tech for (TechMask), from its finished buildings,
        research, heroes (dead ones too: they count) and what it makes now. `built`, `own_hero`, `dead_rows`: the
        rows' masks of the player's living buildings and of its heroes, its dead heroes' rows, when the caller has
        them (View.step: made once a step for this and the other masks; or `done_types`, `hero_types` and `dead_pairs`,
        its finished buildings' and heroes' types and its dead heroes' [id, type]s: featnative.side's)."""
        if self.techmask is None:
            self.techmask = TechMask(self.enc.vocab, RACES[self.races[self.player]])
        if done_types is not None:
            done, heroes = frozenset(done_types), frozenset(hero_types)
        else:
            if built is None:
                own, fl = rows[:, C_OWNER] == self.player, rows[:, C_FLAGS]
                built, own_hero = own & (fl & STRUCTURE > 0) & (fl & DEAD == 0), own & (fl & HERO > 0)
                dead_rows = rows[own_hero & (fl & DEAD > 0)]
            done = frozenset(rows[built & (rows[:, C_FLAGS] & CONSTRUCTING == 0), C_TYPE].tolist())
            heroes = frozenset(rows[own_hero, C_TYPE].tolist())
            dead_pairs = dead_rows[:, [C_ID, C_TYPE]].tolist() if len(dead_rows) else []
        making = frozenset(m[1] for m in self.making.values() if m[0] == 2) | frozenset(getattr(self, "hiring_heroes", ()))
        # (a tavern hire on its way counts as a hero: the altar's free hero had been trained during the walk, the tavern
        # hero then a second without tier 2; 0 of 1,630 targets with a tavern first hero got one first; audit 10)
        since = getattr(self, "dead_since", {})  # (a hero's revival: the game takes it ~7 s after its death; the mask
        reviving = getattr(self, "reviving", {})  # (a revival under way: an altar took its order; audit 6)
        dead = frozenset(c for i, c in dead_pairs
                         if self.t_now - since.get(i, self.t_now) >= REVIVE_AFTER_STEPS
                         and self.t_now - reviving.get(i, -REVIVE_STEPS) >= REVIVE_STEPS) - making if dead_pairs \
            else frozenset()  # had allowed it at once: ~30 refused a game; audit 5. None dead: most steps)
        pending = getattr(self, "pending_research", {})
        for code in [c for c, (b, t0) in pending.items() if b not in self.own_buildings or self.t_now - t0 > 240]:
            del pending[code]  # (its building gone, or two minutes on: given up)
        researching = frozenset(m[1] for m in self.making.values() if m[0] == 3) | frozenset(pending)
        if any(m[0] == 4 for m in self.making.values()):  # (a tier under way: its next type not again, as research;
            up = _mod("techtree").tree().upgrades  # refused ~400-580 a tier type in fgself-146; audit 10)
            researching |= frozenset(int.from_bytes(to.encode(), "big") for m in self.making.values() if m[0] == 4
                                     for to in up.get(rawcode(m[1]), ()))
        now = self.t_now * self.step_seconds
        key = (done, heroes | making, dead, researching, tuple(sorted(self.levels.items())), self.techmask.stocked(now))
        if key == getattr(self, "_tech_key", None):  # (the type codes as they are: their names only when it changed)
            return self._tech_ok
        self._tech_key = key
        self._tech_ok = self.techmask.allowed(frozenset(rawcode(c) for c in done),
                                              {rawcode(k): v for k, v in self.levels.items()},
                                              frozenset(rawcode(c) for c in heroes | making),
                                              frozenset(rawcode(c) for c in researching), now,
                                              frozenset(rawcode(c) for c in dead))
        return self._tech_ok

    def step(self, rows: np.ndarray, me: np.ndarray | None, events: np.ndarray, t: int,
             opp: np.ndarray | None = None) -> dict:
        """rows: this step's units (UNIT_COLS), me: the player's row (PLAYER_COLS) or None, events:
        this step's (step, kind, a, b, c)."""
        enc, player = self.enc, self.player
        # (in C, featnative.view_step and side: the masks below and the entities in two calls, the same numbers; the
        # numpy code for the remembered enemies and the privileged view, or without the library)
        featnative = _mod("featnative")
        fast = A_HID < 0 and not PRIVILEGED and INPUTS < 3 and featnative.lib() is not None
        if fast:
            pk = featnative.side(rows, player, fresh=True, owner=self)
            n_built, n_alive, n_dead, n_heroes, n_done, army_n = pk[:6]
            o = 9
            built_ids, o = pk[o:o + n_built], o + n_built
            alive_ids, o = pk[o:o + n_alive], o + n_alive
            dead3, o = pk[o:o + 3 * n_dead], o + 3 * n_dead
            hero_types, o = pk[o:o + n_heroes], o + n_heroes
            done_types, o = pk[o:o + n_done], o + n_done
            mix_types = pk[o:o + pk[6]]
            own = fl = own_alive = built = own_hero = dead = None
        else:
            own, fl = rows[:, C_OWNER] == player, rows[:, C_FLAGS]
            own_alive = own & (fl & DEAD == 0)
            built = own_alive & (fl & STRUCTURE > 0)
        # the player's buildings standing now; what it knew of the others' production forgotten (a building that died
        # mid-research had kept that research masked for the game, and its id, given to an enemy unit later, that
        # unit's production; audit 3)
        self.own_buildings = set(built_ids) if fast else set(rows[built, C_ID].tolist())
        for d in (self.making, self.queued):  # (the keys gone by a set difference: the dicts as a loop leaves them)
            for k in d.keys() - self.own_buildings:
                del d[k]
        self.busy &= self.own_buildings
        units = set(alive_ids) if fast else set(rows[own_alive, C_ID].tolist())
        # (a unit gone: what it harvested, its mine and its last order forgotten, not given to the next unit with its id:
        # ~72 reuses a game; audit 4. Their keys are all among the last step's living units unless added() said
        # otherwise: then the units gone since are the keys to forget, the same as a pass over each dict)
        prev = self.__dict__.get("_units_prev")
        if prev is not None and self.__dict__.get("_keys_safe"):
            gone = prev - units
            if gone:
                for d in (self.assign, self.mine_of, self.last_order):
                    for k in gone:
                        d.pop(k, None)
        else:
            for d in (self.assign, self.mine_of, self.last_order):
                for k in d.keys() - units:
                    del d[k]
        self._units_prev, self._keys_safe = units, True
        events = events.tolist() if isinstance(events, np.ndarray) else events  # (Python's numbers for the loops below:
        # the same values)
        for r in events:  # production started / ended at a building of the player's (an enemy's: unseen)
            kind, a = int(r[1]), int(r[2])
            if a not in self.own_buildings:
                continue
            if kind in PRODUCTION_START:
                self.busy.add(a)
                if a in self.own_buildings:
                    code = int(r[3])
                    self.making[a] = (3 if kind == 8 else 4 if kind == 11 else 2 if code >> 24 < 0x61 else
                                      0 if code in WORKER_TYPES else 1, code)  # (a hero's id: upper case)
            elif kind in PRODUCTION_END:
                self.busy.discard(a)
                if self.queued.get(a, 0) > 0:
                    self.queued[a] -= 1
                m = self.making.get(a)  # (a unit's end: the one made, its type c; a cancel's: b. Research, a tier: any)
                if m is not None and (m[0] >= 3 or m[1] == int(r[4] if kind == TRAIN_FINISH else r[3])):
                    del self.making[a]
            if kind == RESEARCH_FINISH and a in self.own_buildings:
                self.levels[int(r[3])] = int(r[4])
            if kind in (8, 10):  # (research started or cancelled: no longer queued)
                getattr(self, "pending_research", {}).pop(int(r[3]), None)
        lumber = self.lumber_now = {u for u, r in self.assign.items() if r == "lumber"}  # (the step's choppers: the
        if fast:  # (its dead heroes: type, id, level)                              # teacher's too, play_one)
            dead_types, dead_ids = dead3[0::3], dead3[1::3]
            self.dead_heroes = dict(zip(dead_types, dead_ids))
            ids = set(dead_ids)
        else:
            own_hero = own & (fl & HERO > 0)  # (and the dead ones' rows: for the masks below too)
            dead = rows[own_hero & (fl & DEAD > 0)]
            self.dead_heroes = {int(t): int(i) for t, i in dead[:, [C_TYPE, C_ID]].tolist()}
            ids = set(dead[:, C_ID].tolist())
        since = self.__dict__.setdefault("dead_since", {})
        for i in ids:
            since.setdefault(i, t)
        for i in [i for i in since if i not in ids]:
            del since[i]
        for i in [i for i in getattr(self, "reviving", {}) if i not in ids]:
            del self.reviving[i]  # (alive again)
        priv = None  # (the privileged view: every enemy unit; acting on it, or beside the player's for distillation)
        if fast:
            sel, n_own, f, types, cur, vk, pv = featnative.view_step(enc, _mod(""), self, rows, player, self.sign,
                                                                     (self.queued, self.busy, self.making), lumber, True)
            n, _, n_enemy, n_neutral, own_all, nbw, nrw, nfr = vk[:8]
            enc.ghost0 = min(n, n_own + n_enemy + n_neutral)
            enc.counts = (own_all, n_own + n_enemy + n_neutral)
            self._marks = (vk[8:8 + nbw], vk[8 + nbw:8 + nbw + nrw], vk[8 + nbw + nrw:8 + nbw + nrw + nfr])
            self.max_own, self.max_ent = max(self.max_own, enc.counts[0]), max(self.max_ent, enc.counts[1])
            ids = sel[:, C_ID].tolist()
        else:
            if PRIVILEGED and (self.privileged_act or self.dual):
                priv = enc.entities(rows, player, self.sign, (self.queued, self.busy, self.making), lumber, omniscient=True)
            sel, n_own, f, types, cur = enc.entities(rows, player, self.sign, (self.queued, self.busy, self.making), lumber,
                                                     self.remembered(rows, t) if INPUTS >= 3 and not self.privileged_act else None)
            if priv is not None and self.privileged_act:
                sel, n_own, f, types, cur = priv
            self.max_own, self.max_ent = max(self.max_own, enc.counts[0]), max(self.max_ent, enc.counts[1])
            ids, hp = sel[:, C_ID].tolist(), sel[:, C_HP]
            was, was_hp = self.hp_last
            j = np.minimum(np.searchsorted(was, sel[:, C_ID]), max(len(was) - 1, 0))
            last = np.where(was[j] == sel[:, C_ID], was_hp[j], hp).astype(np.float32) if len(was) else hp.astype(np.float32)
            f[:, A_HPD] = np.minimum(np.maximum((hp - last) / np.maximum(sel[:, C_MAXHP], 1), -1.0), 1.0)  # (0: not in
            # view the last step; np.clip's own overhead was most of it)
            o = np.argsort(sel[:, C_ID], kind="stable")
            self.hp_last = (sel[o, C_ID], hp[o])
            self._marks = None
        index = {u: k for k, u in enumerate(ids)} if self.active_f > A_LAST else {}
        if self.active_f > A_LAST:  # (a per-unit loop: skipped when the run does not use them, --features)
            f[:n_own, A_LAST + 4] = 1.0  # (never ordered: long ago)
        for k in range(n_own if self.active_f > A_LAST else 0):
            lo = self.last_order.get(ids[k])
            if lo is None:
                continue
            kind, x, y, target, t0 = lo
            f[k, A_LAST + 4] = min(1.0, (t - t0) / LAST_AGE)
            if kind == 2 and target in index:  # (a target out of sight: its kind only)
                x, y = float(sel[index[target], C_X]), float(sel[index[target], C_Y])
            if kind == 1 or (kind == 2 and target in index):
                f[k, A_LAST] = np.clip(self.sign * (x - sel[k, C_X]) / LAST_SCALE, -2.0, 2.0)
                f[k, A_LAST + 1] = np.clip((y - sel[k, C_Y]) / LAST_SCALE, -2.0, 2.0)
            if kind:
                f[k, A_LAST + 1 + kind] = 1.0
        for r in events:  # research done (by one of the player's buildings)
            if r[1] == RESEARCH_FINISH and int(r[2]) in self.own_buildings and int(r[3]) in enc.upgrade_index:
                self.upgrades[enc.upgrade_index[int(r[3])]] = r[4] / 3.0
        g = np.zeros(enc.G, np.float32)
        if me is not None:
            _, _, gold, lumber, fu, fc, upkeep = me[:7].tolist()  # (Python's numbers, as View.priv's)
            g[:6] = gold / 1000.0, lumber / 1000.0, fu / 100.0, fc / 100.0, upkeep / 2.0, t / 1800.0
        g[6 + self.races[player]] = 1
        g[6 + len(RACES) + self.races[1 - player]] = 1
        g[6 + 2 * len(RACES):enc.G0] = self.upgrades
        if self.z is not None and enc.ZD:
            g[enc.G0:enc.G0 + enc.ZD] = self.z
        if INPUTS >= 2:  # the clock (inputs 2)
            hour = (START_HOUR + t * self.step_seconds * 24.0 / DAY_SECONDS) % 24.0
            ang = 2.0 * math.pi * hour / 24.0
            x = enc.G0 + enc.ZD
            g[x:x + 3] = math.sin(ang), math.cos(ang), float(hour < DAWN or hour >= DUSK)
        if fast:  # (View.priv's numbers: the per-side block made by view_step)
            if opp is not None:
                _, _, o_gold, o_lumber, o_fu, o_fc, o_upkeep = opp[:7].tolist()
                pv[:5] = o_gold / 1000.0, o_lumber / 1000.0, o_fu / 100.0, o_fc / 100.0, o_upkeep / 2.0
            g[enc.G0 + enc.ZD + enc.GX:] = pv
        else:
            g[enc.G0 + enc.ZD + enc.GX:] = self.priv(rows, opp)
        avail = None
        free = enc.costs is not None and not (self.has_hero(rows, hero_types=hero_types) if fast
                                              else self.has_hero(rows, own_hero))
        self.free_hero = free  # (the agent's step budget: the same costs, play.BCAgent.commands)
        if enc.costs is not None:  # what the player can pay for now (orders beyond it are masked)
            c = (enc.costs_first if free else
                 _revive(enc, list(zip(dead_types, dead3[2::3]))) if fast else revive_costs(enc, rows, self.player, dead))
            gold, lumber, fu, fc = (int(me[2]), int(me[3]), int(me[4]), int(me[5])) if me is not None else (0, 0, 0, 0)
            # (an expand order on its way keeps no money: the game takes a building's cost at its order, and the hall's
            # cost kept again during the walk had masked every paid order for 25-60 s, a penalty on expanding; the
            # pending expand only blocks a second one, below; audit 8)
            k = getattr(self, "hire_class", None)  # (a tavern hero's hire on its way: paid at the sale, play.BCAgent; the
            hire = None if k is None or getattr(self, "hire_free", False) else (k, int(c[k][0]), int(c[k][1]))  # first free)
            kept = [r for r in (self.reserve, hire) if r is not None]
            avail = featnative.afford(c, gold, lumber, fu, fc, kept, self) if fast else None
            if avail is None:
                avail = affordable(c, gold, lumber, fu, fc, self.reserve, hire)
            if enc.tech:  # (--tech-mask)
                self.t_now = t
                avail &= (self.tech_allowed(rows, done_types=done_types, hero_types=hero_types,
                                            dead_pairs=list(zip(dead_ids, dead_types))) if fast
                          else self.tech_allowed(rows, built, own_hero, dead))
        if avail is not None and enc.expand_class is not None:  # (--macros: expand, while the natural is free and
            site = _mod("macros").site                            # its hall affordable; night elves' tree of life
            race = RACES[self.races[player]]                      # entangles its mine, play.BCAgent._entangle)
            if t - getattr(self, "_site_at", -NATURAL_EVERY) >= NATURAL_EVERY or t < getattr(self, "_site_at", 0):
                self.expand_site = site(rows, getattr(self, "start", None), True) if MACROS else None  # (every
                self._site_at = t  # NATURAL_EVERY steps: a natural is taken or freed over seconds; 4% of an actor's
                # time when every step)
            via = enc.expand_via.get(race)
            busy = getattr(self, "expanding", None) is not None and t < self.expanding[0]
            avail[enc.expand_class] = bool(self.expand_site is not None and via is not None and avail[via] and not busy
                                           and "expand" not in MACROS_OFF)
        if avail is not None and enc.creep_class is not None:  # (--macros: creep the natural, while its camp stands)
            camp = _mod("macros").camp
            army = army_n if fast else int((own_alive & (fl & (STRUCTURE | WORKER) == 0)).sum())
            strong = army >= CREEP_ARMY and t * self.step_seconds >= CREEP_FROM  # (the teacher's conditions: from minute 1,
            if not strong or t - getattr(self, "_camp_at", -NATURAL_EVERY) >= NATURAL_EVERY or t < getattr(self, "_camp_at", 0):
                self.creep_at = camp(rows, getattr(self, "start", None), True) if MACROS and strong else None  # any
                self._camp_at = t  # unit sent it:
            avail[enc.creep_class] = self.creep_at is not None and "creep" not in MACROS_OFF  # 436 attack-moves a game; audit 3)
        if avail is not None and enc.mix_class is not None:  # (--macros: train the target's mix, while it lacks a unit
            target = getattr(self, "mix_target", None) or {}  # the side can train now)
            if not fast:
                mix_types = rows[own & (fl & (DEAD | STRUCTURE | WORKER | HERO | SUMMONED) == 0), C_TYPE].tolist()
            n = max(len(mix_types), 1)  # (summons are no army it trained: they had lowered the trained units' shares)
            have: dict[str, float] = {}
            per: dict[int, int] = {}  # (each type's count, in the order first met: its share is 1 / n added that many
            for c in mix_types:       # times, as one unit at a time had added it)
                per[c] = per.get(c, 0) + 1
            inc = 1.0 / n
            for c, m in per.items():
                v = 0.0
                for _ in range(m):
                    v += inc
                have[rawcode(int(c))] = v
            self.mix_have = have
            idx = enc.order_index
            self.mix_can = lambda u: (k := idx.get((int.from_bytes(u.encode(), "big"), IMMEDIATE))) is not None and bool(avail[k])
            # (a unit no train order makes, a mercenary, a zeppelin: not one it can train; class 0, no order, had
            # read as one: the mix order available with only those lacking, and doing nothing; audit 3)
            avail[enc.mix_class] = bool(MACROS and "mix" not in MACROS_OFF
                                        and any(s > have.get(u, 0.0) and self.mix_can(u) for u, s in target.items()))
        if avail is not None and enc.pickup_class is not None:  # (--macros 2: pick up, while a hero has an item lying
            pickable = _mod("macros").pickable                    # near; the ground items: play.BCAgent.observe)
            self.pickers = pickable(rows, player, getattr(self, "ground", ())) if MACROS >= 2 else []
            avail[enc.pickup_class] = bool(self.pickers)
        if self.zmask is not None and t < self.zmask_until:  # (--z-enforce: only the target's items)
            avail = self.zmask.copy() if avail is None else avail & self.zmask
        if self.active_f < F_BASE and (not fast or self.active_f > A_LAST):  # (view_step's: zeroed there, unless the
            f[:, self.active_f:F_BASE] = 0.0                                   # last orders' loop above wrote some)
        out = {"n": len(sel), "sel": sel, "n_own": n_own, "ent": f, "type": types, "cur": cur, "glob": g,
               "first_free": free, "index": dict(zip(ids, range(len(ids)))), "avail": avail,
               "ghost0": getattr(enc, "ghost0", len(sel)) if not self.privileged_act else len(sel)}
        if priv is not None:  # (the privileged view's global flag; its entities beside the player's: "p…")
            flag = enc.G0 + enc.ZD + enc.GX - 1
            pg = g.copy()
            pg[flag] = 1.0
            if self.privileged_act:
                out["glob"] = pg
            else:
                pf = priv[2]
                if self.active_f < F_BASE:
                    pf[:, self.active_f:F_BASE] = 0.0
                out.update(pent=pf, ptype=priv[3], pcur=priv[4], pn=len(priv[0]), pglob=pg)
        return out
