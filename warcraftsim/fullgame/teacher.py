"""A heuristic teacher for what the policy rarely stumbles on (the user's idea, 2026-10-20: "buying a first hero with
a worker", on-policy distillation with our own advisor instead of the built-in AI). Its orders are never carried
out: they label the policy's step for the units it would order (as the target builder's do, --z-build 2), and the
cloning loss on those labels (--opd-coef) pulls the policy toward them while the reward decides whether they stay.
selfplay --teach 1 (needs --neutral 1: the orders it teaches are the neutral buildings' wares).

    a tavern hero: a worker near the tavern hires one (the game's draw of the race's usual picks), as the first hero
                   when the player has an altar and none, or the second with a tier-2 hall, from 135 s, the money there
    a town portal: a hero near the goblin merchant buys one when it has none and a free slot, from 440 s
    a potion:      a hero below 40% of its hit points drinks what heals in its inventory

Each kind at most every GAP steps a side, so a long wait for the money is not a label a step."""

from __future__ import annotations

import random

import numpy as np

from ..protocol import Buy, ImmediateOrder
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
NEAR_TAVERN, NEAR_MERCHANT = 3000.0, 2500.0
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
        shops = rows[(fl & fx.DEAD == 0) & ((rows[:, fx.C_VIS] >> self.side) & 1 == 1)]
        tavern, merchant = shops[shops[:, fx.C_TYPE] == TAVERN], shops[shops[:, fx.C_TYPE] == MERCHANT]
        done = alive[(alive[:, fx.C_FLAGS] & fx.STRUCTURE > 0) & (alive[:, fx.C_FLAGS] & fx.CONSTRUCTING == 0), fx.C_TYPE]
        # a tavern hero: the first with an altar, the second with tier 2
        n_heroes = len(heroes)
        if (self.pick is not None and len(tavern) and now >= 135.0 and n_heroes < 2 and player.gold >= 425 and player.lumber >= 135
                and player.food_used + 5 <= player.food_cap and fourcc_in(done, ALTARS)
                and (n_heroes == 0 or fourcc_in(done, TIER2)) and self.pick not in set(heroes[:, fx.C_TYPE].tolist())):
            tv = tavern[0]
            workers = alive[np.isin(alive[:, fx.C_TYPE], list(WORKERS))]
            if len(workers):
                d = np.hypot(workers[:, fx.C_X] - tv[fx.C_X], workers[:, fx.C_Y] - tv[fx.C_Y])
                k = int(np.argmin(d))
                if d[k] < NEAR_TAVERN and self._due("tavern hero", t):
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
        return out


def fourcc_in(types: np.ndarray, codes: set[int]) -> bool:
    return bool(np.isin(types, list(codes)).any())
