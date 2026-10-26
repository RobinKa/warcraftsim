"""More kinds of strategy conditioning (the user's idea, 2026-10-20: "no rush", and many more, combinable with a
build target and each other): a game's modes, each a switch and a parameter in the strategy block (after the
build target's slots: strategy.vector), measured from the game each step and paid as the build target is, a
potential of minus the violations so far (selfplay --z-modes: each kind's chance to be on in a game with a target,
--z-mode-reward: the weight of one whole violation).

    no_rush    no army unit near the enemy's start before second X (violation: the share of [0, X] it was there)
    rush       N army units near the enemy's start at once by second X
    expand     a second hall (one started counts) by second X
    no_expand  one hall only, the whole horizon
    workers    N workers at second 480 (violation: the share missing)
    creep      a hero at level L by second 420 (the share missing)
    harass     N of the enemy's workers killed by second 600 (the share missing)
    scout      a unit near the enemy's start by second 120
    army       F food of army at second 600 (the share missing)
    low_upkeep at most 50 food until second 900 (the share of the time above)

The army's composition kinds (the user's idea, 2026-10-26: ask the policy to limit or forbid the unit it spams,
footmen, grunts, ghouls, archers; selfplay --z-comp: each one's chance, a block of their own after the others:
COMP_KINDS, comp_vector). Its army: its living units but buildings, workers and heroes, by food (a type of no food,
a summon, left out):

    cap        from second 300 to 900, while its army has 12 food or more, no type above C of the army's food
               (violation: the share of that time above, each step weighted by how far above: (share - C) / (1 - C))
    forbid     none of one basic unit until second 900 (parameter 0: the race's spam unit, footman, grunt, ghoul,
               archer; 1: its other basic unit, rifleman, headhunter, crypt fiend, huntress; undead's 3 ghouls
               allowed, its lumber; violation: the most alive at once beyond that, a quarter each)
    variety    N types in its army at once by second 600 (the share missing)

A composition kind changes the game's build target to agree (adapt): the forbidden unit out of its items and its
army's mix, and the mix capped at C (the target builder, the teacher's mix lesson, the mix's input and reward
follow it); --z-forbid-mask also masks the forbidden unit's train order (features.View.forbid).

"Near the enemy's start": within NEAR of its start location (its base; on Echo Isles the bases are ~9,800 apart).
A violation decided at a deadline counts from then on; the build target's items count alike (due by now)."""

from __future__ import annotations

import random

import numpy as np

from . import rulesnative
from .features import codes, isin, own_alive

KINDS = ("no_rush", "rush", "expand", "no_expand", "workers", "creep", "harass", "scout", "army", "low_upkeep")
EXCLUSIVE = (("no_rush", "rush"), ("expand", "no_expand"), ("low_upkeep", "army"))
MDIM = 2 * len(KINDS)  # the strategy block's part: per kind, on and its parameter (scaled to ~[0, 1])
# the army's composition kinds (--z-comp): a block of their own at the strategy block's end (after the mix's), so
# a network from before them grows by zeros there (model.load) and the modes' part keeps its place
COMP_KINDS = ("cap", "forbid", "variety")
CDIM = 2 * len(COMP_KINDS)
NEAR, SCOUT_NEAR = 2500.0, 1500.0
HORIZON = 900.0
# per kind: (the parameter's range to draw from, its scale in the input)
PARAMS = {"no_rush": ((240.0, 600.0), HORIZON), "rush": ((180.0, 360.0), HORIZON), "expand": ((240.0, 480.0), HORIZON),
          "no_expand": ((0.0, 0.0), 1.0), "workers": ((15.0, 30.0), 30.0), "creep": ((3.0, 5.0), 10.0),
          "harass": ((3.0, 8.0), 10.0), "scout": ((0.0, 0.0), 1.0), "army": ((20.0, 60.0), 100.0),
          "low_upkeep": ((0.0, 0.0), 1.0), "cap": ((0.4, 0.55), 1.0), "forbid": ((0.0, 1.0), 1.0),
          "variety": ((3.0, 4.0), 5.0)}
RUSH_UNITS = 4  # rush: this many army units at once
DECIDED_AT = {"workers": 480.0, "creep": 420.0, "harass": 600.0, "army": 600.0, "variety": 600.0}  # (the second
# each is decided at)
CAP_FROM, CAP_LEAST = 300.0, 12  # cap: measured from this second, while the army has this much food
# forbid: each race's basic units (parameter 0 the spam unit, drawn FORBID_FIRST of the time), and how many of one
# may live at once (undead's ghouls gather its lumber: the teacher's choppers); a violation of 1 at FORBID_FULL more
FORBID = {"human": ("hfoo", "hrif"), "orc": ("ohun", "ogru"), "undead": ("ugho", "ucry"), "nightelf": ("earc", "esen")}
# (orc's spam unit is the headhunter, ~41 a game to ~5 grunts: grunts first had forbidden what it hardly made; audit 14)
FORBID_FIRST, FORBID_FULL = 0.75, 4
ALLOW = {"ugho": 3}
# (the measures each kind reads while undecided: Tracker.step's need bits; no_rush's until its second, no_expand's
# until HORIZON)
NEEDS = {"no_rush": 64, "rush": 64, "expand": 1, "no_expand": 1, "workers": 16, "creep": 32, "harass": 2, "scout": 8,
         "army": 4, "low_upkeep": 0, "cap": 4 | 128, "forbid": 256, "variety": 128}
UNTIL = {"no_rush", "no_expand", "cap", "forbid"}  # (measured until a second: no_rush's, the others' HORIZON)
INTEGER = {"workers", "creep", "harass", "army", "variety"}
# unit flags (protocol.UnitFlags) and row columns (features)
HERO, STRUCTURE, WORKER, DEAD = 1, 2, 4, 1024
C_TYPE, C_OWNER, C_X, C_Y, C_FLAGS, C_HLEVEL = 2, 3, 4, 5, 12, 15
GOLD_MINES = {int.from_bytes(c.encode(), "big") for c in ("ngol", "ugol", "egol")}
GOLD_MINE_CODES = codes(GOLD_MINES)
EXPANSION_FROM_START, EXPANSION_AT_MINE = 1500.0, 1000.0  # (as trace.Production's expansions)
HALLS = {int.from_bytes(c.encode(), "big") for c in
         ("htow", "hkee", "hcas", "ogre", "ostr", "ofrt", "unpl", "unp1", "unp2", "etol", "etoa", "etoe")}
DEATH = 1
MH = rulesnative.MH  # (rulesnative.modes' head)
_NOTHING = [0] * MH  # (no measure asked)
_MINES_TABLE = rulesnative.table(dict.fromkeys(GOLD_MINE_CODES.tolist(), 1))
_FOOD_TABLE: tuple = (None,) * 5 + (None, -1)  # (the trackers' food table: the C's arrays, the dict, its length)


def draw(rng: random.Random, chance: float, comp: float = 0.0) -> dict[str, float]:
    """A game's modes: each kind on with `chance` (of a pair that exclude each other, one), with its parameter;
    each composition kind with `comp` (0: none, and no draw of the generator's more than before them)."""
    on = {k for k in KINDS if rng.random() < chance}
    for a, b in EXCLUSIVE:
        if a in on and b in on:
            on.discard(rng.choice((a, b)))
    out = {}
    for k in KINDS:
        if k in on:
            (lo, hi), _ = PARAMS[k]
            v = rng.uniform(lo, hi)
            out[k] = float(round(v)) if k in INTEGER else round(v, 1)
    if comp > 0:
        for k in COMP_KINDS:
            if rng.random() < comp:
                (lo, hi), _ = PARAMS[k]
                out[k] = (0.0 if rng.random() < FORBID_FIRST else 1.0) if k == "forbid" else \
                    float(rng.randint(int(lo), int(hi))) if k == "variety" else round(rng.uniform(lo, hi), 2)
    return out


def comp_vector(modes: dict[str, float] | None) -> np.ndarray:
    """The composition kinds' block (--z-comp; zeros: none)."""
    z = np.zeros(CDIM, np.float32)
    for i, k in enumerate(COMP_KINDS):
        if modes and k in modes:
            z[2 * i], z[2 * i + 1] = 1.0, modes[k] / PARAMS[k][1]
    return z


def forbidden(modes: dict[str, float] | None, race: str | None) -> str | None:
    """The unit `modes` forbid a side of `race` (None: none)."""
    if not modes or "forbid" not in modes or race not in FORBID:
        return None
    return FORBID[race][int(modes["forbid"])]


def capped(n: dict, cap: float) -> dict:
    """An army's counts with no share above `cap` (the excess spread over the others by their shares, again until
    none is above; all alike when they cannot be: fewer than 1 / cap types). `n` itself when none is above."""
    total = float(sum(n.values()))
    if not total or max(n.values()) <= cap * total + 1e-9:
        return n
    if len(n) * cap < 1.0:
        return {c: total / len(n) for c in n}
    fixed: set = set()
    share = {c: v / total for c, v in n.items()}
    while True:
        over = {c for c, v in share.items() if c not in fixed and v > cap + 1e-12}
        if not over:
            return {c: v * total for c, v in share.items()}
        fixed |= over
        rest = sum(v for c, v in n.items() if c not in fixed)
        left = 1.0 - cap * len(fixed)
        share = {c: cap if c in fixed else left * v / rest for c, v in n.items()}


def adapt(st: dict | None, modes: dict[str, float] | None, race: str | None) -> dict | None:
    """A build target `st` that agrees with the composition kinds of `modes` (a copy; `st` itself when none
    changes it): a forbidden unit out of its items and its army's counts, its army's counts capped (capped).
    Again on its own answer it changes nothing (a replayed game's targets)."""
    if not st or not modes:
        return st
    u = forbidden(modes, race)
    if u is not None and (u in (st.get("t") or {}) or u in (st.get("n") or {})):
        st = {**st, "t": {c: v for c, v in (st.get("t") or {}).items() if c != u},
              "n": {c: v for c, v in (st.get("n") or {}).items() if c != u}}
    if "cap" in modes and st.get("n"):
        from .strategy import is_army
        army = {c: v for c, v in st["n"].items() if is_army(c) and v > 0}
        new = capped(army, modes["cap"]) if army else army
        if new is not army:
            st = {**st, "n": {**st["n"], **new}}
    return st


def vector(modes: dict[str, float] | None) -> np.ndarray:
    """The strategy block's part for `modes` (zeros: none)."""
    z = np.zeros(MDIM, np.float32)
    for i, k in enumerate(KINDS):
        if modes and k in modes:
            z[2 * i], z[2 * i + 1] = 1.0, modes[k] / PARAMS[k][1]
    return z


class Tracker:
    """One side's modes in a game: their violations so far (step), each in [0, 1]."""

    def __init__(self, modes: dict[str, float], side: int, enemy_start: tuple[float, float], food: dict[int, int],
                 worker_types: set[int], own_start: tuple[float, float] | None = None, race: str | None = None):
        """`own_start`: an expansion is a hall far from it and at a gold mine (None: any second hall, as before;
        a second hall at home met "expand" and broke "no_expand", the audit of 2026-10-22). `race`: the side's
        (forbid's unit)."""
        self.modes, self.side, self.enemy, self.own = dict(modes), side, enemy_start, own_start
        u = forbidden(modes, race)
        if "forbid" in modes and u is None:
            raise ValueError(f"forbid needs the side's race (got {race!r})")
        self.forbid_code = int.from_bytes(u.encode(), "big") if u else 0
        self.allow = ALLOW.get(u, 0)
        self.food, self.worker_types = food, worker_types  # (unit type -> its food; the workers' types)
        self.worker_codes = codes(worker_types)  # (features.isin's: per step)
        self.hall_codes = codes(set(HALLS) | ({int.from_bytes(b"ugol", "big")} if own_start is not None else set()))
        self.v = {k: 0.0 for k in modes}
        self.done: set[str] = set()  # (a goal reached: no violation later)
        self.kills = 0
        self.enemy_seen: set[int] = set()
        self.mine_seen: set[int] = set()
        self.t = 0.0

    def step(self, rows: np.ndarray, events, food_used: int, now: float) -> float:
        """The game at second `now` (rows: features' unit columns; events: protocol.Event; the side's food used)
        -> the sum of the violations so far."""
        dt, self.t = max(0.0, now - self.t), now
        s, m, done = self.side, self.modes, self.done
        # (the side's measures the modes below read: in C, rulesnative.modes, numpy's calls had cost ~250k instructions
        # a step; _measure_np the numpy's; none once every mode is decided or past its time (no harass)). need: 1 the
        # halls, 2 the units seen, 4 the army's food, 8 scouted, 16 the workers, 32 the heroes' level, 64 near (the C
        # measures all but 1, 2 and 4 anyway)
        c = self.__dict__.get("_needs")
        if c is None or c[0] != len(done):  # (made again when a mode is decided: `done` only grows)
            base, until = 2 if "harass" in m else 0, []
            for k, x in m.items():
                if k not in done:
                    if k in UNTIL:
                        until.append((x if k == "no_rush" else HORIZON, NEEDS[k]))
                    else:
                        base |= NEEDS.get(k, 0)
            c = self._needs = (len(done), base, until)
        need = c[1]
        for t, bits in c[2]:
            if now <= t:
                need |= bits
        got = (rulesnative.modes(self, rows, need, self._constants) or self._measure_np(rows, need) if need
               else _NOTHING)
        near, scouted, workers, level, food, halls, ne, no, top, kinds, nforbid = got[:MH]
        if "harass" in m:  # (the enemy's workers and its own units as seen so far: a dead unit is mostly gone from the
            self.enemy_seen.update(got[MH:MH + ne])  # step that reports its death: ~half the kills missed; audit 8)
            self.mine_seen.update(got[MH + ne:MH + ne + no])
            self.kills += sum(1 for e in events if int(e.kind) == DEATH and e.a in self.enemy_seen and e.b in self.mine_seen)
        for k, x in m.items():
            if k in done:
                continue
            if k == "no_rush" and now <= x and near > 0:
                self.v[k] = min(1.0, self.v[k] + dt / x)
            elif k == "rush":
                if near >= RUSH_UNITS:
                    done.add(k)
                elif now >= x:
                    self.v[k] = 1.0
            elif k in ("expand", "no_expand"):  # (halls: the main hall and those at another gold mine; audit 4)
                if k == "expand" and halls >= 2:
                    done.add(k)
                elif k == "expand" and now >= x:
                    self.v[k] = 1.0
                elif k == "no_expand" and halls >= 2 and now <= HORIZON:
                    self.v[k] = 1.0
            elif k == "scout":
                if scouted:
                    done.add(k)
                elif now >= 120.0:
                    self.v[k] = 1.0
            elif k == "low_upkeep" and now <= HORIZON and food_used > 50:
                self.v[k] = min(1.0, self.v[k] + dt / HORIZON)
            elif k == "cap":
                if CAP_FROM < now <= HORIZON and food >= CAP_LEAST and top > x * food:
                    w = min(1.0, (top / food - x) / (1.0 - x))
                    self.v[k] = min(1.0, self.v[k] + min(dt, now - CAP_FROM) / (HORIZON - CAP_FROM) * w)
            elif k == "forbid":
                if now <= HORIZON and nforbid > self.allow:
                    self.v[k] = max(self.v[k], min(1.0, (nforbid - self.allow) / FORBID_FULL))
            elif k in ("workers", "creep", "harass", "army", "variety"):
                at = DECIDED_AT[k]
                have = workers if k == "workers" else level if k == "creep" else self.kills if k == "harass" else \
                    kinds if k == "variety" else food
                if have >= x:
                    done.add(k)
                    self.v[k] = 0.0
                elif now >= at:
                    self.v[k] = (x - have) / x
                    done.add(k)  # (decided at its time)
        return sum(self.v.values())

    def _constants(self) -> tuple:
        """rulesnative.modes' constants of this tracker (asked once): the side, the enemy's start, the squared
        distances, its own start, the tables of the halls' types, the gold mines', the workers' and the food."""
        global _FOOD_TABLE
        ft = _FOOD_TABLE
        if ft[5] is not self.food or ft[6] != len(self.food):  # (play_one's trackers share one food table)
            ft = _FOOD_TABLE = (*rulesnative.table(list(self.food), list(self.food.values())), self.food, len(self.food))
        return (self.side, self.enemy, NEAR * NEAR, SCOUT_NEAR * SCOUT_NEAR, self.own, EXPANSION_FROM_START ** 2,
                int(EXPANSION_AT_MINE ** 2), rulesnative.table(dict.fromkeys(self.hall_codes.tolist(), 1)), _MINES_TABLE,
                rulesnative.table(dict.fromkeys(self.worker_codes.tolist(), 1)), ft, self.forbid_code)

    def _measure_np(self, rows: np.ndarray, need: int) -> list:
        """rulesnative.modes' measures by numpy (the path without the library): [near, scouted, workers, the heroes'
        highest level, the army's food, halls, the enemy's workers seen, its own units, the most food of one army
        type, the army's types of food, the forbidden unit's] and those ids; each only when `need` asks for it (0
        else)."""
        s = self.side
        mine = own_alive(rows, s)
        fl = mine[:, C_FLAGS]
        army = mine[(fl & (STRUCTURE | WORKER) == 0)]
        near = scouted = workers = level = food = halls = top = kinds = nforbid = 0
        if need & 64:
            d2 = (army[:, C_X] - self.enemy[0]) ** 2 + (army[:, C_Y] - self.enemy[1]) ** 2
            near = int((d2 < NEAR * NEAR).sum())
        if need & 8:
            all_d2 = (mine[:, C_X] - self.enemy[0]) ** 2 + (mine[:, C_Y] - self.enemy[1]) ** 2
            scouted = int(bool((all_d2 < SCOUT_NEAR * SCOUT_NEAR).any()))
        if need & 16:
            workers = int(isin(mine[:, C_TYPE], self.worker_codes).sum())
        if need & 32:
            level = int(mine[fl & HERO > 0, C_HLEVEL].max(initial=0))
        if need & (4 | 128):
            per: dict[int, int] = {}
            for c in army[army[:, C_FLAGS] & HERO == 0, C_TYPE].tolist():
                f = self.food.get(c, 0)
                food += f
                if f > 0:
                    per[c] = per.get(c, 0) + f
            if need & 128:
                top, kinds = max(per.values(), default=0), len(per)
        if need & 256:
            nforbid = int((mine[:, C_TYPE] == self.forbid_code).sum())
        if need & 1:  # (undead: a haunted mine far from home; audit 4)
            hall = mine[isin(mine[:, C_TYPE], self.hall_codes) & (fl & STRUCTURE > 0)]
            if self.own is None:
                halls = len(hall)
            else:  # (the main hall and those at another gold mine)
                gm = rows[isin(rows[:, C_TYPE], GOLD_MINE_CODES)]
                far = (hall[:, C_X] - self.own[0]) ** 2 + (hall[:, C_Y] - self.own[1]) ** 2 > EXPANSION_FROM_START ** 2
                at = (((hall[:, None, C_X] - gm[None, :, C_X]) ** 2 + (hall[:, None, C_Y] - gm[None, :, C_Y]) ** 2)
                      .min(1, initial=1e18) < EXPANSION_AT_MINE ** 2) if len(gm) else np.zeros(len(hall), bool)
                halls = 1 + int((far & at).sum())
        ids_e = rows[(rows[:, C_OWNER] == 1 - s) & (rows[:, C_FLAGS] & WORKER > 0), 1].tolist() if need & 2 else []
        ids_o = mine[:, 1].tolist() if need & 2 else []
        return [near, scouted, workers, level, food, halls, len(ids_e), len(ids_o), top, kinds, nforbid, *ids_e,
                *ids_o]

    def report(self) -> dict[str, list[float]]:
        """Per mode: [its parameter, its violation]."""
        return {k: [x, round(self.v[k], 3)] for k, x in self.modes.items()}
