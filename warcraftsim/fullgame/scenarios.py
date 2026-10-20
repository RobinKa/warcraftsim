"""Scenarios (the user's idea, 2026-10-20: the micro curriculum's way kept on): a share of the training games
start with a situation the harness sets up at the first step (selfplay --scenarios, --scenario-share), so the
policy meets what a game from scratch rarely brings it to. Both sides get the same value, so the material
shaping starts even. Evaluation games never have one.

    clash   each side an army (its race's tier-1 and tier-2 units, ARMY gold) and a hero at level 3 between the bases
    raid    the opponent's army (ARMY gold) at the learner's base, the learner's as large at home
    rich    both sides 1,500 more gold and 700 more lumber: the tiers, the tech, the heroes early

Weight averaging of specialists (forks from one checkpoint, each on one scenario, merged): scripts/average_weights.py."""

from __future__ import annotations

import math
import random

from ..protocol import SetResources, Spawn, SpawnHero

SCENARIOS = ("clash", "raid", "rich")
ARMY = 1200  # gold an army is worth
UNITS = {"human": ("hfoo", "hrif", "hkni", "hmpr", "hsor"), "orc": ("ogru", "ohun", "orai", "otau", "oshm"),
         "undead": ("ugho", "ucry", "uabo", "unec", "ugar"), "nightelf": ("earc", "esen", "edry", "edoc", "ehip")}
HEROES = {"human": ("Hpal", "Hamg", "Hmkg", "Hblm"), "orc": ("Obla", "Ofar", "Otch", "Oshd"),
          "undead": ("Udea", "Ulic", "Udre", "Ucrl"), "nightelf": ("Ekee", "Emoo", "Edem", "Ewar")}


def army(race: str, rng: random.Random, budget: int = ARMY) -> list[str]:
    """Unit types of `race` worth about `budget` gold (the game's costs)."""
    from ..data.objects import unit_table
    table = unit_table()
    out, spent = [], 0
    kinds = [c for c in UNITS.get(race, ()) if c in table and table[c].gold > 0]
    while kinds and spent < budget:
        c = rng.choice(kinds)
        out.append(c)
        spent += table[c].gold
    return out


def _ring(x: float, y: float, k: int) -> tuple[float, float]:
    """The k-th spot around (x, y): units side by side, not on top of each other."""
    a, r = k * 2.39996, 60.0 * math.sqrt(k)
    return x + r * math.cos(a), y + r * math.sin(a)


def commands(name: str, obs, side: int, races: list[str], rng: random.Random) -> list:
    """The first step's commands of scenario `name` for the learner's `side` (obs: the game's first observation)."""
    starts = {p: (s.start_x, s.start_y) for p, s in obs.players.items() if p in (0, 1)}
    if len(starts) < 2:
        return []
    out = []
    if name == "rich":
        for p in (0, 1):
            q = obs.players[p]
            out.append(SetResources(p, q.gold + 1500, q.lumber + 700))
        return out
    for p in (0, 1):
        (x0, y0), (x1, y1) = starts[p], starts[1 - p]
        if name == "clash":  # (between the bases, a third of the way from its own)
            at = (x0 + (x1 - x0) / 3.0, y0 + (y1 - y0) / 3.0)
            out.append(SpawnHero(p, rng.choice(HEROES.get(races[p], HEROES["human"])), *_ring(*at, 0), 3))
        elif name == "raid":  # (the opponent's army at the learner's base, on the way in; the learner's at home)
            (lx, ly), (ox, oy) = starts[side], starts[1 - side]
            n = math.hypot(ox - lx, oy - ly) or 1.0
            r = 600.0 if p == side else 1500.0
            at = (lx + r * (ox - lx) / n, ly + r * (oy - ly) / n)
        else:
            return []
        for k, c in enumerate(army(races[p], rng), start=1):
            out.append(Spawn(p, c, *_ring(*at, k)))
    return out
