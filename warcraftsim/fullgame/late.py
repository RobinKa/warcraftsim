"""Late-game starts (docs/proposals/five-changes.md, item 2): a share of the training games start from a mid-game state
captured from the agent's own games, so the policy trains what its games rarely reach from minute 0 (tier 2-3 armies,
mixed units, anti-air, upgrades, mines running dry, long games: its wins past 18 minutes were ~31%, half its wins came
before 12). Evaluation games never start from one.

Capture (selfplay --late-capture P): in this share of the training games, at a game minute drawn in --late-minutes
(8-16), the game's state as one JSON line (the learner appends it to runs/<run>/late-states.jsonl, with the game's
winner and length once it ends): the map, both races and start locations, and per side its gold, lumber, research
levels (from the game's research events), its units and complete buildings (type, position, hit points and mana per
mille, a role: "s" a building, "g" / "l" a harvester on gold / lumber, as its agent's view sent it, else guessed from its
order and where it is) and its heroes (level, experience, skills, items, dead or alive); the gold mines' gold, the creeps
still standing. A decided game gives none (a side
without a complete town hall, or with under DECIDED of the other's material).

Start (the scenario "late", --scenarios ...,late --late-states PATH[,PATH]): a training game's first step sets up a
captured state of its map and races whose start locations are the game's. Echo Isles is only roughly mirror-symmetric
about the line between its starts (43% of its trees and 92% of its pathing cells match their mirror image, 2026-10-29),
so a state is never mirrored: it fits as captured or with its sides swapped (races and start locations together), and
half the states of a matchup fit a game's draw of the start locations. The commands (step 0, `commands`):
  creeps     the camps' units the captured game had killed taken out (RemoveUnit: no bounty, no drop)
  mines      each mine's gold as captured (SetMineGold); one the captured game had mined out collapses
  halls      the start's town hall kept where the state has one of its type there, else taken out (a keep, a stronghold,
             halls of the dead, a tree of ages replace it rather than stand beside it); its workers (and undead's ghoul)
             taken out; the start's haunted / entangled mine stays (a night elf's tree replaced takes its entangled
             mine along: entangled again at step 1)
  research   each level (SetResearch: before the units, whose hit points it may raise)
  units      every other captured unit and complete building placed (Place: hit points and mana per mille; a militia as
             the peasant it was), an expansion's haunted mine haunted (its entangled mine: at step 1)
  heroes     at their levels and experience with their skills and items, a dead one killed (its altar may revive it),
             the free first hero used up
  resources  the captured gold and lumber (SetResources)
Steps 1 and 2 follow up (`again`): the gold of the mines the setup made or uncovered, the entangled mines entangled at
once and marked complete. No side acts at steps 0 and 1 and they are not training steps: the setup is the game's start
(the shaping's potential is the late state's from step 2). At step 2 each agent side's placed harvesters get their
captured jobs back (`jobs`: harvest the nearest own mine, or the nearest tree), so its economy runs as it did; its view
learns them, and the research levels (features.View.levels: only research events set them). Only games between agents
start late (selfplay.play_one: the built-in AI's script starts its build order and hero picks anew at minute 0, and it
floated gold from a placed base). Strategy targets (z) are off in a late start (their timings count from minute 0); the
game clock and the time of day start at 0.

Not reproduced: trees cut down (the new game's trees are whole), items on the ground, cooldowns, buffs, orders under way
(a building's queue, research under way, a building being built or upgraded: only complete buildings are captured),
summoned units and illusions, units inside a transport or a burrow (placed beside it).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import features as fx
from ..protocol import (Place, RemoveUnit, SetMineGold, SetResearch, SetResources, TargetDestructable, TargetOrder,
                        UnitFlags, encode_commands, fourcc, rawcode)

VERSION = 1
MINUTES = (8.0, 16.0)  # --late-minutes: the capture's game minute, drawn in this range
DECIDED = 0.25  # a side with under this share of the other's material: a decided game, no capture
NEUTRAL_HOSTILE = 12
HALLS = {"htow": 1, "hkee": 2, "hcas": 3, "ogre": 1, "ostr": 2, "ofrt": 3, "unpl": 1, "unp1": 2, "unp2": 3,
         "etol": 1, "etoa": 2, "etoe": 3}
MINES = ("ngol", "ugol", "egol")
OWN_MINE = {"undead": "ugol", "nightelf": "egol"}  # (the mine a race's workers harvest; others: the plain gold mine)
HARVESTERS = ("hpea", "opeo", "uaco", "ewsp", "ugho")  # (workers, and undead's ghouls on lumber)
GOLD_ONLY, LUMBER_ONLY = ("uaco",), ("ugho",)
MORPHS = {"hmil": "hpea"}  # (a militia, placed as one, would never turn back: the peasant it was)
SKIP = int(UnitFlags.SUMMONED | UnitFlags.ILLUSION | UnitFlags.CONSTRUCTING)
NEAR_MINE = 450.0  # a harvester this near a gold mine and harvesting: on gold (no agent's view to say)
START_SAME = 32.0  # start locations this close: the same
HALL_SAME = 64.0  # a start's hall and a captured one of its type this close: the same hall, kept
MINE_SAME = 150.0  # a mine and a captured one this close: the same mine
CREEP_SAME = 700.0  # a creep and a captured one of its type this close: still standing (camps' units move a little)
JOB_SAME = 200.0  # a placed harvester and its captured self this close: the same unit (placed where it stood)
JOB_REACH = 2500.0  # the farthest mine or tree a placed harvester is sent to
MAX_INTS = 6000  # a state's step-0 commands at most (the step's 8000, protocol.MAX_COMMAND_INTS, with room to spare)


def _r(v: float) -> int:
    return int(round(float(v)))


def _permille(v: int, of: int) -> int:
    return min(1000, max(1, _r(1000.0 * v / of))) if of > 0 else 1000


def note_research(seen: dict, events, rows: np.ndarray) -> None:
    """Each side's research levels so far ({side: {code: level}}), from a step's research events (features.RESEARCH_FINISH:
    a = the building, b = the research, c = its new level; the building's owner from the step's rows)."""
    for e in events:
        if int(e.kind) == fx.RESEARCH_FINISH:
            at = rows[rows[:, fx.C_ID] == e.a]
            if len(at) and int(at[0, fx.C_OWNER]) in seen:
                seen[int(at[0, fx.C_OWNER])][rawcode(int(e.b))] = int(e.c)


def _role(r, mines: np.ndarray, jobs: dict) -> str:
    """A unit's role: "s" a building; a harvester's job, "g" gold or "l" lumber: its agent's view's (View.assign), else
    its order (a harvest cycle's) and where it was (in a mine or near one: gold); else ""."""
    t, uid = rawcode(int(r[fx.C_TYPE])), int(r[fx.C_ID])
    if int(r[fx.C_FLAGS]) & UnitFlags.STRUCTURE:
        return "s"
    if t not in HARVESTERS:
        return ""
    if jobs.get(uid) in ("gold", "lumber"):
        return jobs[uid][0]
    order = int(r[fx.C_ORDER])  # (the built-in AI sends its workers with a right-click: "smart")
    if order not in fx.HARVESTING and order != fx.SMART and not int(r[fx.C_FLAGS]) & UnitFlags.HIDDEN:
        return ""
    if t in GOLD_ONLY or t in LUMBER_ONLY:
        return "g" if t in GOLD_ONLY else "l"
    near = len(mines) and float(np.hypot(mines[:, 0] - r[fx.C_X], mines[:, 1] - r[fx.C_Y]).min()) < NEAR_MINE
    return "g" if near or int(r[fx.C_FLAGS]) & UnitFlags.HIDDEN else "l"


def capture(rows: np.ndarray, players: dict, races: list[str], map_name: str, research: dict, game_time: float,
            jobs: dict | None = None, mat: dict | None = None) -> dict | None:
    """A game's state at a step (rows: play.unit_rows of its observation; players: Observation.players; research:
    note_research's; jobs: the agents' views' View.assign, unit -> "gold" / "lumber"; mat: trace.material), or None for a
    decided game. Positions are the game's (the start locations with them)."""
    from ..data.abilities import hero_abilities
    if any(players.get(s) is None for s in (0, 1)):
        return None
    if mat is not None and min(mat[0], mat[1]) < DECIDED * max(mat[0], mat[1]):
        return None
    jobs = jobs or {}
    fl = rows[:, fx.C_FLAGS]
    types = [rawcode(int(t)) for t in rows[:, fx.C_TYPE]]
    alive = (fl & UnitFlags.DEAD) == 0
    is_mine = np.asarray([t in MINES for t in types], bool)
    mines = rows[alive & is_mine]
    mine_xy = mines[:, [fx.C_X, fx.C_Y]].astype(float)
    abil = hero_abilities()
    sides = []
    for s in (0, 1):
        own = (rows[:, fx.C_OWNER] == s) & ((fl & SKIP) == 0)
        if not any(alive[i] and own[i] and HALLS.get(types[i]) for i in range(len(rows))):
            return None  # (no town hall left: decided)
        units, heroes = [], []
        for i in np.flatnonzero(own):
            r, t = rows[i], types[i]
            hp, mana = _permille(r[fx.C_HP], r[fx.C_MAXHP]), _permille(r[fx.C_MANA], r[fx.C_MAXMANA])
            if r[fx.C_FLAGS] & UnitFlags.HERO:
                slots = abil.get(t, ())
                levels = [int(r[fx.C_ABIL0 + 2 * k]) for k in range(len(slots))]
                skills = [c for n in range(1, max(levels, default=0) + 1) for c, lv in zip(slots, levels) if lv >= n]
                items = [rawcode(int(v)) for v in r[fx.C_ITEM0:fx.C_ITEM0 + 6] if v]
                heroes.append([t, _r(r[fx.C_X]), _r(r[fx.C_Y]), hp, mana, int(r[fx.C_HLEVEL]), int(r[fx.C_HXP]), skills,
                               items, int(not alive[i])])
            elif alive[i]:
                units.append([MORPHS.get(t, t), _r(r[fx.C_X]), _r(r[fx.C_Y]), hp, mana, _role(r, mine_xy, jobs)])
        p = players[s]
        sides.append({"gold": int(p.gold), "lumber": int(p.lumber), "research": dict(sorted(research.get(s, {}).items())),
                      "units": units, "heroes": heroes})
    creeps = rows[alive & (rows[:, fx.C_OWNER] == NEUTRAL_HOSTILE) & ((fl & UnitFlags.STRUCTURE) == 0)]
    return {"v": VERSION, "map": map_name, "races": list(races), "minute": round(game_time / 60.0, 2),
            "starts": [[int(players[s].start_x), int(players[s].start_y)] for s in (0, 1)], "sides": sides,
            "mines": [[rawcode(int(r[fx.C_TYPE])), _r(r[fx.C_X]), _r(r[fx.C_Y]), int(r[fx.C_RESOURCE])] for r in mines],
            "creeps": [[rawcode(int(r[fx.C_TYPE])), _r(r[fx.C_X]), _r(r[fx.C_Y])] for r in creeps]}


def swapped(state: dict) -> dict:
    """The state with its sides swapped (races, start locations, units: the game's other player plays each)."""
    out = dict(state, races=state["races"][::-1], starts=state["starts"][::-1], sides=state["sides"][::-1])
    for k in ("winner", "learner"):  # (the captured game's, by side)
        if state.get(k) in (0, 1):
            out[k] = 1 - state[k]
    return out


def _same(a, b, d: float = START_SAME) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= d


def fit(state: dict, map_name: str, races: list[str], starts: list) -> dict | None:
    """The state as a game of `map_name` with `races` and start locations `starts` ([(x, y)] by side) has it: as
    captured, with its sides swapped, or None (another map, other races, other start locations: never mirrored)."""
    if state.get("map") != map_name:
        return None
    for st in (state, swapped(state)):
        if list(st["races"]) == list(races) and all(_same(st["starts"][s], starts[s]) for s in (0, 1)):
            return st
    return None


def _key(map_name: str, races, starts) -> tuple:
    return (map_name, *races, *(_r(v) for xy in starts for v in xy))


class States:
    """The captured states of --late-states files, by map, races and start locations as a game draws them (each state
    under both its orientations): a line's file and offset, read when picked (a run's file grows to thousands of lines;
    12 actors held none of it). A file is read again for its new lines at most every `refresh` seconds (the run that
    captures them may be the one that uses them)."""

    def __init__(self, files: list[str], refresh: float = 600.0):
        self.files = [Path(f) for f in files]
        self.refresh = refresh
        self.index: dict[tuple, list[tuple[int, int, bool]]] = {}
        self.done = [0] * len(self.files)  # (each file's bytes indexed)
        self.checked = 0.0
        self.count = 0

    def _update(self) -> None:
        if time.time() - self.checked < self.refresh:
            return
        self.checked = time.time()
        for k, f in enumerate(self.files):
            try:
                size = f.stat().st_size
            except FileNotFoundError:
                continue
            if size <= self.done[k]:
                continue
            with open(f, "rb") as fh:
                fh.seek(self.done[k])
                at = self.done[k]
                for line in fh:
                    if not line.endswith(b"\n"):  # (a line being written: next time)
                        break
                    try:
                        st = json.loads(line)
                        for flip in (False, True):
                            o = swapped(st) if flip else st
                            self.index.setdefault(_key(o["map"], o["races"], o["starts"]), []).append((k, at, flip))
                        self.count += 1
                    except (ValueError, KeyError, TypeError):
                        pass
                    at += len(line)
                self.done[k] = at

    def pick(self, map_name: str, races: list[str], starts: list, rng) -> dict | None:
        """A state that fits the game (fit), drawn uniformly, or None."""
        self._update()
        options = self.index.get(_key(map_name, races, starts))
        if not options:
            return None
        k, at, flip = options[rng.randrange(len(options))]
        with open(self.files[k], "rb") as fh:
            fh.seek(at)
            st = json.loads(fh.readline())
        return fit(st, map_name, races, starts)


_STATES: dict[tuple, States] = {}


def states(files: list[str]) -> States:
    """This process's States of `files` (made once)."""
    key = tuple(files)
    if key not in _STATES:
        _STATES[key] = States(list(files))
    return _STATES[key]


def _order(unit: list) -> int:
    """Placement order: halls first (an expansion's tree before the mine it entangles), other buildings, the mines,
    then units."""
    t = unit[0]
    return 0 if t in HALLS else 2 if t in MINES else 1 if unit[5] == "s" else 3


def commands(state: dict, rows: np.ndarray) -> list | None:
    """The first step's commands that set up `state` (fit's: its sides are the game's) in a game whose first
    observation's units are `rows` (play.unit_rows), or None when they would not fit in a step (MAX_INTS)."""
    out: list = []
    fl = rows[:, fx.C_FLAGS]
    types = [rawcode(int(t)) for t in rows[:, fx.C_TYPE]]
    alive = (fl & UnitFlags.DEAD) == 0
    # the creeps the captured game had killed: out (each captured one keeps the nearest of its type)
    left: dict[str, list] = {}
    for t, x, y in state.get("creeps", ()):
        left.setdefault(t, []).append((x, y))
    for i in np.flatnonzero(alive & (rows[:, fx.C_OWNER] == NEUTRAL_HOSTILE) & ((fl & UnitFlags.STRUCTURE) == 0)):
        pts, x, y = left.get(types[i], []), rows[i, fx.C_X], rows[i, fx.C_Y]
        d = [math.hypot(px - x, py - y) for px, py in pts]
        if d and min(d) <= CREEP_SAME:
            pts.pop(int(np.argmin(d)))
        else:
            out.append(RemoveUnit(int(rows[i, fx.C_ID])))
    # the mines' gold: a gold mine a player's haunted / entangled one stood on in the captured game gets that one's (the
    # placing takes the gold with it; a player's haunting left the gold mine hidden under it with its gold of then);
    # else the captured mine of its type there, else any there; none there: mined out, it collapses
    for i in np.flatnonzero(alive & np.asarray([t in MINES for t in types], bool)):
        x, y = rows[i, fx.C_X], rows[i, fx.C_Y]
        here = [m for m in state.get("mines", ()) if _same(m[1:3], (x, y), MINE_SAME)]
        m = next((m for m in here if types[i] == "ngol" and m[0] != "ngol"), None)
        m = m or next((m for m in here if m[0] == types[i]), here[0] if here else None)
        gold = int(m[3]) if m is not None else 0
        if gold != int(rows[i, fx.C_RESOURCE]):
            out.append(SetMineGold(int(rows[i, fx.C_ID]), gold))
    for s in (0, 1):
        side = state["sides"][s]
        own = np.flatnonzero(alive & (rows[:, fx.C_OWNER] == s))
        units = sorted(side["units"], key=_order)
        kept: set[int] = set()  # (the captured units the start already has: its hall, its haunted mine)
        for i in own:  # (its hall kept where the state has one of its type, else out with its workers; its haunted or
            t, x, y = types[i], rows[i, fx.C_X], rows[i, fx.C_Y]  # entangled mine stays, its gold set above)
            same = next((k for k, u in enumerate(units) if k not in kept and u[0] == t and (t in HALLS or t in MINES)
                         and _same(u[1:3], (x, y), MINE_SAME if t in MINES else HALL_SAME)), None)
            if same is not None:
                kept.add(same)
            elif t not in MINES:
                out.append(RemoveUnit(int(rows[i, fx.C_ID])))
        for code, level in side.get("research", {}).items():
            out.append(SetResearch(s, code, int(level)))
        for k, (t, x, y, hp, mana, _role) in enumerate(units):
            if k not in kept and t != "egol":  # (an entangled mine: at step 1, again; its gold mine shows then)
                out.append(Place(s, t, x, y, hp, mana))
        for t, x, y, hp, mana, level, xp, skills, items, dead in side.get("heroes", ()):
            out.append(Place(s, t, x, y, hp, mana, int(level), tuple(skills), tuple(items), bool(dead), int(xp)))
        out.append(SetResources(s, int(side["gold"]), int(side["lumber"])))
    return out if len(encode_commands(out)) <= MAX_INTS else None


def again(state: dict, rows: np.ndarray) -> list:
    """The setup's follow-up, at steps 1 and 2 (rows: the step's units). The gold of the mines it made or uncovered: a
    haunted mine placed has a gold mine of the type's own gold hidden under it (12,500, the map's scaling not applied; it
    comes back if the haunted one falls); each mine whose gold is not its captured self's (of its type, there) gets it,
    one where the captured game had none collapses. The entangled mines (step 1): each side's captured one without one
    there now is entangled at once by its tree (Place's "egol": a night elf's start tree replaced took its entangled mine
    along, its gold mine back a step later; left alone, the new tree entangled it again over ~20 s without gold
    coming in); at step 2 it is marked complete (SetMineGold: the instant entangle has no construction finish, and the
    agent harvests no mine under construction)."""
    out = []
    fl = rows[:, fx.C_FLAGS]
    alive = rows[((fl & UnitFlags.DEAD) == 0) & np.isin(rows[:, fx.C_TYPE], [fourcc(t) for t in MINES])]
    for r in alive:
        t, x, y = rawcode(int(r[fx.C_TYPE])), r[fx.C_X], r[fx.C_Y]
        here = [m for m in state.get("mines", ()) if _same(m[1:3], (x, y), MINE_SAME)]
        m = next((m for m in here if m[0] == t), None)
        gold = int(m[3]) if m is not None else None if here else 0
        if gold is not None and (gold != int(r[fx.C_RESOURCE]) or int(r[fx.C_FLAGS]) & UnitFlags.CONSTRUCTING):
            out.append(SetMineGold(int(r[fx.C_ID]), gold))
    egols = alive[alive[:, fx.C_TYPE] == fourcc("egol")]
    for s in (0, 1):
        for t, x, y, *_ in state["sides"][s]["units"]:
            if t == "egol" and not (np.hypot(egols[:, fx.C_X] - x, egols[:, fx.C_Y] - y) <= MINE_SAME).any():
                out.append(Place(s, "egol", x, y))
    return out


def jobs(state: dict, rows: np.ndarray, side: int, race: str, trees: dict) -> list[tuple]:
    """Step 2's harvest orders for `side`'s placed harvesters (rows: the step's units; trees: destructable id -> (x, y)
    of the trees standing): [(command, unit, "gold" / "lumber", mine or tree)]. Each captured harvester with a job is
    matched to the nearest placed unit of its type (placed where it stood) and sent to the nearest own mine (undead's
    haunted, night elves' entangled, else a gold mine) or the nearest tree within JOB_REACH."""
    fl = rows[:, fx.C_FLAGS]
    alive = (fl & UnitFlags.DEAD) == 0
    own = rows[alive & (rows[:, fx.C_OWNER] == side) & ((fl & UnitFlags.STRUCTURE) == 0)]
    mine_type = fourcc(OWN_MINE.get(race, "ngol"))
    mines = rows[alive & (rows[:, fx.C_TYPE] == mine_type)
                 & ((rows[:, fx.C_OWNER] == side) if race in OWN_MINE else True)]
    tree_ids = np.asarray(list(trees), np.int64)
    tree_xy = np.asarray([trees[k] for k in tree_ids.tolist()], float).reshape(-1, 2)
    used: set[int] = set()
    out = []
    for t, x, y, _hp, _mana, job in state["sides"][side]["units"]:  # (role: "g" / "l" a harvester's job)
        if job not in ("g", "l"):
            continue
        cand = own[own[:, fx.C_TYPE] == fourcc(t)]
        if not len(cand):
            continue
        d = np.hypot(cand[:, fx.C_X] - x, cand[:, fx.C_Y] - y)
        d[np.isin(cand[:, fx.C_ID], list(used))] = np.inf
        k = int(np.argmin(d))
        if d[k] > JOB_SAME:
            continue
        uid = int(cand[k, fx.C_ID])
        used.add(uid)
        ux, uy = float(cand[k, fx.C_X]), float(cand[k, fx.C_Y])
        if job == "g" and len(mines):
            dm = np.hypot(mines[:, fx.C_X] - ux, mines[:, fx.C_Y] - uy)
            j = int(np.argmin(dm))
            if dm[j] <= JOB_REACH:
                target = int(mines[j, fx.C_ID])
                out.append((TargetOrder(uid, fx.HARVEST, target), uid, "gold", target))
        elif job == "l" and len(tree_xy):
            dt = np.hypot(tree_xy[:, 0] - ux, tree_xy[:, 1] - uy)
            j = int(np.argmin(dt))
            if dt[j] <= JOB_REACH:
                target = int(tree_ids[j])
                out.append((TargetDestructable(uid, fx.HARVEST, target), uid, "lumber", target))
    return out


def levels(state: dict, side: int) -> dict[int, int]:
    """A side's research levels as features.View.levels has them (upgrade id -> level)."""
    return {fourcc(c): int(v) for c, v in state["sides"][side].get("research", {}).items()}


def noted(view, given: list[tuple], results) -> int:
    """A view's workers the game took `jobs`' orders for (results: the step's answers to them, in order): its View.assign
    (and a gold one's mine, View.mine_of), as the agent's own accepted harvest orders set them. Returns how many."""
    n = 0
    for (_c, uid, job, target), ok in zip(given, results):
        if ok:
            view.assign[uid] = job
            view.added(uid)
            if job == "gold":
                view.mine_of[uid] = target
            n += 1
    return n


def append(path: str | os.PathLike, state: dict) -> None:
    """One state as a line of `path` (the learner's: one writer)."""
    with open(path, "a") as f:
        f.write(json.dumps(state, separators=(",", ":")) + "\n")
