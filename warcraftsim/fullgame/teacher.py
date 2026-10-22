"""A heuristic teacher for what the policy rarely stumbles on (the user's idea, 2026-10-20: "buying a first hero with
a worker", on-policy distillation with our own advisor instead of the built-in AI). Its orders are never carried
out: they label the policy's step for the units it would order (as the target builder's do, --z-build 2), and the
cloning loss on those labels (--opd-coef) pulls the policy toward them while the reward decides whether they stay.
selfplay --teach 1 (needs --neutral 1: the orders it teaches are the neutral buildings' wares).

    a tavern hero: the worker nearest the tavern hires one (the game's draw of the race's usual picks), as the first hero
                   when the player has an altar and none, or the second with a tier-2 hall, from 135 s, the money there
    a town portal: a hero near the goblin merchant buys one when it has none and a free slot, from 440 s
    a potion:      a hero below 40% of its hit points drinks what heals in its inventory
    an expansion:  when its target's expansion is due (a hall, or undead's haunted gold mine, in its items): the army
                   clears the natural expansion's camp (at 6 units or more), then the worker nearest builds the hall
                   where people put theirs (undead: an acolyte haunts the mine), then three workers go to mine there.
                   (No policy had ever expanded: 0% of its games, people 81% of Echo Isles'. Night elves not yet: a
                   tree of life's mine needs the entangle order, which the agent can't give.)

Each kind at most every GAP steps a side, so a long wait for the money is not a label a step."""

from __future__ import annotations

import random

import numpy as np

from ..protocol import Build, Buy, ImmediateOrder, PointOrder, TargetOrder
from . import features as fx
from .neutral import fourcc

# the tavern heroes people pick for each race (1.29 ladder habits)
PICKS = {"human": ("Npbm", "Nfir", "Ntin"), "orc": ("Ntin", "Npbm", "Nalc"), "undead": ("Nplh", "Nbrn", "Nalc"),
         "nightelf": ("Npbm", "Nngs", "Nbst")}
WORKERS = {fourcc(c) for c in ("hpea", "opeo", "uaco", "ewsp")}
ALTARS = {fourcc(c) for c in ("halt", "oalt", "uaod", "eate")}
TIER2 = {fourcc(c) for c in ("hkee", "hcas", "ostr", "ofrt", "unp1", "unp2", "etoa", "etoe")}
TAVERN, MERCHANT, PORTAL = fourcc("ntav"), fourcc("ngme"), fourcc("stwp")
GAP = 20  # steps between two lessons of a kind
HALL = {"human": "htow", "orc": "ogre", "undead": "unpl", "nightelf": "etol"}
HAUNTED = fourcc("ugol")
MINES = {fourcc(c) for c in ("ngol", "ugol", "egol")}
# where people built their halls at Echo Isles' natural expansions (95 games of 1.29: the mine -> the hall's offset)
EXPAND_SPOTS = {(-4864, -3200): (100, 800), (4352, -3200): (-300, 700)}
NATURAL_FROM_START = 1500.0  # (the main mine is nearer its start)
CAMP_NEAR = 900.0  # creeps this near the natural mine guard it
ARMY_TO_CREEP = 6  # army units it takes to clear a camp
ATTACK, HARVEST = 851983, 852018
HIDDEN = 16  # (UnitFlags.HIDDEN: a worker inside a gold mine)
NEAR_MERCHANT = 2500.0
USESLOT = 852008


class Teacher:
    """One side's lessons in one game (commands: what it would order now)."""

    def __init__(self, side: int, race: str, rng: random.Random, target: dict | None = None):
        """`target` (the side's target strategy, if it has one): its tavern hero is the one taught, and none when it
        has none (another hero would break a narrow target)."""
        from .neutral import TAVERN_HEROES
        self.side = side
        if target is None:
            self.pick = fourcc(rng.choice(PICKS.get(race, PICKS["human"])))
        else:
            mine = sorted((t, c) for c, t in target.get("t", {}).items() if c in TAVERN_HEROES)
            self.pick = fourcc(mine[0][1]) if mine else None
        self.last: dict[str, int] = {}
        self.taught: dict[str, int] = {}  # each kind's lessons this game (the episode row)
        self.race = race
        due = [v for c, v in (target or {}).get("t", {}).items() if c in (HALL.get(race), "ugol")]
        self.expand_at = min(due) if due and race != "nightelf" else None  # (its target's expansion: when)

    def _due(self, kind: str, t: int) -> bool:
        if t - self.last.get(kind, -GAP) < GAP:
            return False
        self.last[kind] = t
        self.taught[kind] = self.taught.get(kind, 0) + 1
        return True

    def commands(self, rows: np.ndarray, player, now: float, t: int) -> list:
        """rows: the step's units (features' columns, ROW_COLS with the heroes' items), player: its PlayerState."""
        if player is None or rows.shape[1] < fx.ROW_COLS:
            return []
        out = []
        fl = rows[:, fx.C_FLAGS]
        mine = rows[(rows[:, fx.C_OWNER] == self.side)]
        alive = mine[mine[:, fx.C_FLAGS] & fx.DEAD == 0]
        heroes = mine[mine[:, fx.C_FLAGS] & fx.HERO > 0]  # (dead ones count: they can be revived)
        live_heroes = alive[alive[:, fx.C_FLAGS] & fx.HERO > 0]
        shops = rows[fl & fx.DEAD == 0]  # (the neutral buildings: on the game's minimap from the start)
        tavern, merchant = shops[shops[:, fx.C_TYPE] == TAVERN], shops[shops[:, fx.C_TYPE] == MERCHANT]
        done = alive[(alive[:, fx.C_FLAGS] & fx.STRUCTURE > 0) & (alive[:, fx.C_FLAGS] & fx.CONSTRUCTING == 0), fx.C_TYPE]
        # a tavern hero: the first with an altar, the second with tier 2
        n_heroes = len(heroes)
        if (self.pick is not None and len(tavern) and now >= 135.0 and n_heroes < 2
                and (n_heroes == 0 or player.gold >= 425 and player.lumber >= 135)  # (the first hero is free: food only)
                and player.food_used + 5 <= player.food_cap and fourcc_in(done, ALTARS)
                and (n_heroes == 0 or fourcc_in(done, TIER2)) and self.pick not in set(heroes[:, fx.C_TYPE].tolist())):
            tv = tavern[0]
            workers = alive[np.isin(alive[:, fx.C_TYPE], list(WORKERS))]
            if len(workers):
                d = np.hypot(workers[:, fx.C_X] - tv[fx.C_X], workers[:, fx.C_Y] - tv[fx.C_Y])
                k = int(np.argmin(d))
                if self._due("tavern hero", t):  # (the nearest worker: it walks there, protocol.Buy)
                    out.append(Buy(int(workers[k, fx.C_ID]), int(tv[fx.C_ID]), self.pick))
        for h in live_heroes:
            items = [int(v) for v in h[fx.C_ITEM0:fx.C_ITEM0 + 6]]
            # a potion when it is low
            if h[fx.C_HP] < 0.4 * max(int(h[fx.C_MAXHP]), 1):
                from .herodata import item_table
                heal = [k for k, it in enumerate(items) if it and item_table().get(it, (0, 0, 0, 0))[1]]
                if heal and self._due(f"potion {int(h[fx.C_ID])}", t):
                    out.append(ImmediateOrder(int(h[fx.C_ID]), USESLOT + heal[0]))
            # a town portal from the goblin merchant
            if (len(merchant) and now >= 440.0 and PORTAL not in items and 0 in items and player.gold >= 500):
                m = merchant[0]
                if np.hypot(h[fx.C_X] - m[fx.C_X], h[fx.C_Y] - m[fx.C_Y]) < NEAR_MERCHANT and self._due("town portal", t):
                    out.append(Buy(int(h[fx.C_ID]), int(m[fx.C_ID]), PORTAL))
        if self.expand_at is not None and now >= self.expand_at:
            out += self._expansion(rows, alive, player, t)
        return out

    def _expansion(self, rows: np.ndarray, alive: np.ndarray, player, t: int) -> list:
        """The next step of taking the natural expansion: clearing its camp, the hall (undead: the haunted mine), the
        workers that mine there."""
        sx, sy = getattr(player, "start_x", None), getattr(player, "start_y", None)
        mines = rows[np.isin(rows[:, fx.C_TYPE], list(MINES)) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]
        if sx is None or not len(mines):
            return []
        d = np.hypot(mines[:, fx.C_X] - sx, mines[:, fx.C_Y] - sy)
        far = np.argsort(d)[d[np.argsort(d)] > NATURAL_FROM_START]
        if not len(far):
            return []
        nat = mines[far[0]]
        mx, my = float(nat[fx.C_X]), float(nat[fx.C_Y])
        near = lambda r, rad: np.hypot(r[:, fx.C_X] - mx, r[:, fx.C_Y] - my) < rad  # noqa: E731
        hall = fourcc(HALL[self.race])
        workers = alive[np.isin(alive[:, fx.C_TYPE], list(WORKERS)) & (alive[:, fx.C_FLAGS] & HIDDEN == 0)]
        taken = alive[np.isin(alive[:, fx.C_TYPE], [hall, HAUNTED]) & near(alive, 1000.0)]
        if len(taken):  # (taken: once finished, three workers to mine there)
            if (taken[:, fx.C_FLAGS] & fx.CONSTRUCTING > 0).all() or (near(workers, 900.0).sum() >= 4) or not len(workers):
                return []
            target = taken[taken[:, fx.C_TYPE] == HAUNTED]
            mine_id = int(target[0, fx.C_ID]) if len(target) else int(nat[fx.C_ID])
            far_w = workers[~near(workers, 900.0)]
            pick = far_w[np.argsort(np.hypot(far_w[:, fx.C_X] - mx, far_w[:, fx.C_Y] - my))[:3]]
            return ([TargetOrder(int(w[fx.C_ID]), HARVEST, mine_id) for w in pick]
                    if len(pick) and self._due("expansion: workers", t) else [])
        creeps = rows[(rows[:, fx.C_OWNER] == 12) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0) & near(rows, CAMP_NEAR)]
        if len(creeps):  # (its camp first: the army)
            army = alive[(alive[:, fx.C_FLAGS] & (fx.STRUCTURE | fx.WORKER) == 0)]
            if len(army) < ARMY_TO_CREEP or not self._due("expansion: its camp", t):
                return []
            return [PointOrder(int(u[fx.C_ID]), ATTACK, int(mx), int(my)) for u in army]
        if not len(workers) or not self._due("expansion: the hall", t):
            return []
        if self.race == "undead":  # (an acolyte haunts the mine: its gold, no hall needed)
            w = workers[int(np.argmin(np.hypot(workers[:, fx.C_X] - mx, workers[:, fx.C_Y] - my)))]
            return [TargetOrder(int(w[fx.C_ID]), HAUNTED, int(nat[fx.C_ID]))]
        ox, oy = EXPAND_SPOTS.get((int(round(mx)), int(round(my))), (None, None))
        if ox is None:  # (another map's: 750 toward its start)
            n = max(float(np.hypot(sx - mx, sy - my)), 1.0)
            ox, oy = 750.0 * (sx - mx) / n, 750.0 * (sy - my) / n
        x, y = mx + ox, my + oy
        w = workers[int(np.argmin(np.hypot(workers[:, fx.C_X] - x, workers[:, fx.C_Y] - y)))]
        return [Build(int(w[fx.C_ID]), hall, x, y)]


def fourcc_in(types: np.ndarray, codes: set[int]) -> bool:
    return bool(np.isin(types, list(codes)).any())
