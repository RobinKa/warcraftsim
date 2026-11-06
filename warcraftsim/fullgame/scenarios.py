"""Scenarios (the user's idea, 2026-10-20: the micro curriculum's way kept on): a share of the training games
start with a situation the harness sets up at the first step (selfplay --scenarios, --scenario-share), so the
policy meets what a game from scratch rarely brings it to. Both sides get the same value, so the material
shaping starts even (base2 against the built-in AI: only the learner, its own way, below). Evaluation games never have one.

    natural both natural expansions' camps removed (the gold mine nearest each start but its main): expanding needs
            no creeping first (the teacher's expansion lessons, --teach, then apply; fgself-123's army sent to clear
            the camp lost its games)
    clash   each side an army (its race's tier-1 and tier-2 units, ARMY gold) and a hero at level 3 between the bases
    raid    the opponent's army (ARMY gold) at the learner's base, the learner's as large at home
    rich    both sides 1,500 more gold and 700 more lumber: the tiers, the tech, the heroes early
    base2   a two-base game: each agent's side has its natural running from the start (below)
    dry     both mains' gold mines start with DRY_SHARE of their gold (one share for both sides) and both naturals' camps
            cleared: the main runs dry in a few minutes, so a side that does not expand loses its income (the long games
            are lost on income and the policy never expands; base2 handed it a running natural, this makes it build one)

The two-base game (the first scenario given to one side only, against the built-in AI): against the insane AI's double
income the policy's one mine caps it at ~500 gold a minute and it never expands; so it gets experience of what a second
mine buys and of holding it. The natural's camp is cleared, its race's hall stands where people build it
(macros.hall_spot; undead: the mine haunted, as the policy's expand order does, acolytes need no hall; a night elf's tree
nearer its mine, as the melee script's start has it, and the mine entangled), and 5 workers of the race gather there. Only the agents' sides (selfplay.play_one's bots): against the
built-in AI or the hand-written player only the learner's (the AI's script does badly from a placed base, and the second
mine evens out its double income); between agents both. Set up by the late starts' machinery (late.py: the state base2
returns, late.again and late.jobs at steps 1 and 2, no side acts before step 2), so the first reward holds no gift (the
shaping's potential is the setup's from step 2), and neither does what the side made (Production starts at step 2). A
target's expansion is done at its own time (base2_made: no reward either way, the builder builds no other), a mode about
expanding is not drawn. Evaluation games never have it; neither does a game the built-in AI advises or GRPO's.

Weight averaging of specialists (forks from one checkpoint, each on one scenario, merged): scripts/average_weights.py."""

from __future__ import annotations

import math
import random

from . import late, macros
from ..protocol import ClearCreeps, Place, SetMineGold, SetResources, Spawn, SpawnHero

SCENARIOS = ("clash", "raid", "rich", "natural", "base2", "dry")
DRY_SHARE = (0.25, 0.6)  # (dry: the mains' gold left, a share of what they start with: echofast2's main, mined out
#   at ~12-13 minutes from full, then lasts ~3-8)
MINES = ("ngol", "ugol", "egol")
BASE2_WORKERS = 5
TREE_FROM_MINE, TREE_BOX = 650.0, 448.0  # (a night elf's start: the melee script's tree from its mine, within a box)
WORKERS = {"human": "hpea", "orc": "opeo", "undead": "uaco", "nightelf": "ewsp"}
NATURAL_FROM_START, CAMP_RADIUS = 1500.0, 900
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


def base2(obs, sides, races: list[str]) -> tuple[list, dict] | None:
    """The two-base start of `sides` (the module's docstring) in a game whose first observation is `obs`: its first step's
    commands and its state in late.py's form for late.again and late.jobs (only what they read: every mine's gold as it
    is, what the setup makes at each natural, the workers' jobs). None: a side has no natural (no free mine past its main)."""
    if any(obs.players.get(s) is None for s in sides):
        return None
    mines = [[u.type, late._r(u.x), late._r(u.y), int(u.resource)] for u in obs.units if u.type in late.MINES and u.alive]
    state: dict = {"mines": mines, "sides": [{"units": []}, {"units": []}]}
    out: list = []
    for s in sides:
        sx, sy = obs.players[s].start_x, obs.players[s].start_y
        far = sorted((math.hypot(m[1] - sx, m[2] - sy), k) for k, m in enumerate(mines)
                     if m[0] == "ngol" and math.hypot(m[1] - sx, m[2] - sy) > NATURAL_FROM_START)
        if not far:
            return None
        _, mx, my, gold = mines[far[0][1]]
        hx, hy = macros.hall_spot(mx, my, sx, sy)
        race, units = races[s], state["sides"][s]["units"]
        if race == "nightelf":  # (its tree as the melee script's start: the instant entangle does nothing from ~780 or 650)
            n = math.hypot(hx - mx, hy - my) or 1.0
            hx = mx + max(-TREE_BOX, min(TREE_BOX, TREE_FROM_MINE * (hx - mx) / n))
            hy = my + max(-TREE_BOX, min(TREE_BOX, TREE_FROM_MINE * (hy - my) / n))
        out.append(ClearCreeps(mx, my, CAMP_RADIUS))
        if race != "undead":
            units.append([macros.HALLS[race], late._r(hx), late._r(hy), 1000, 1000, "s"])
        if race in late.OWN_MINE:  # (its mine's gold on the map's scale: late.again's, at steps 1 and 2)
            own = late.OWN_MINE[race]
            mines.append([own, mx, my, gold])
            units.append([own, mx, my, 1000, 1000, "s"])
        at = (mx + hx) / 2.0, (my + hy) / 2.0  # (between the mine and the hall, clear of both)
        for k in range(BASE2_WORKERS):
            x, y = _ring(*at, k)
            units.append([WORKERS[race], late._r(x), late._r(y), 1000, 1000, "g"])
        out += [Place(s, u[0], u[1], u[2]) for u in units if u[0] != "egol"]  # (an entangled mine: at step 1, late.again)
    return out, state


def base2_made(target: dict) -> dict[str, float]:
    """What a two-base start makes of a target strategy's items: an expansion it asks for, done at its own time (made and
    due alike: no reward either way at any time; the builder takes it as made)."""
    return {c: target["t"][c] for c in late.EXPANSION_ITEMS if c in target["t"]}


def commands(name: str, obs, side: int, races: list[str], rng: random.Random) -> list:
    """The first step's commands of scenario `name` for the learner's `side` (obs: the game's first observation)."""
    starts = {p: (s.start_x, s.start_y) for p, s in obs.players.items() if p in (0, 1)}
    if len(starts) < 2:
        return []
    out = []
    if name == "dry":  # (each start's main: its nearest gold mine within NATURAL_FROM_START; its natural: the nearest past)
        mines = [(u.id, u.x, u.y, u.resource) for u in obs.units if u.type in MINES]
        share = rng.uniform(*DRY_SHARE)
        for x0, y0 in starts.values():
            by = sorted((math.hypot(x - x0, y - y0), i, x, y, g) for i, x, y, g in mines)
            main = next((m for m in by if m[0] <= NATURAL_FROM_START), None)
            if main is not None and main[4] > 0:
                out.append(SetMineGold(main[1], max(1, int(main[4] * share))))
            far = next((m for m in by if m[0] > NATURAL_FROM_START), None)
            if far is not None:
                out.append(ClearCreeps(far[2], far[3], CAMP_RADIUS))
        return out
    if name == "natural":  # (each start's natural: its nearest gold mine past its main)
        mines = [(u.x, u.y) for u in obs.units if u.type in ("ngol", "ugol", "egol")]
        for x0, y0 in starts.values():
            far = sorted((math.hypot(x - x0, y - y0), x, y) for x, y in mines if math.hypot(x - x0, y - y0) > NATURAL_FROM_START)
            if far:
                out.append(ClearCreeps(far[0][1], far[0][2], CAMP_RADIUS))
        return out
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
