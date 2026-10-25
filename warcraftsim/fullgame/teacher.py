"""A heuristic teacher for what the policy rarely stumbles on (the user's idea, 2026-10-20: "buying a first hero with
a worker", on-policy distillation with our own advisor instead of the built-in AI). Its orders are never carried
out: they label the policy's step for the units it would order (as the target builder's do, --z-build 2), and the
cloning loss on those labels (--opd-coef) pulls the policy toward them while the reward decides whether they stay.
selfplay --teach 1 (needs --neutral 1: the orders it teaches are the neutral buildings' wares).

    a first hero:  (a game without a target) from FIRST_HERO_FROM, a side without a hero: its altar trains one of the
                   race's usual first heroes (the game's draw), and with no altar the worker nearest its hall builds one
                   450-800 from it, as the target builder places buildings. Without a target the human policy built an
                   altar in a fifth of its games (`fgself-134`): it rushed the built-in AI down and lost to policies.
    a tavern hero: the worker nearest the tavern hires one: with a target its tavern hero, when the side has as many
                   heroes as come before it in the target (its next hero); without, the game's draw of the race's usual
                   picks as the second hero, from 135 s. With an altar, the tier the hero count needs (2 for the second,
                   3 for the third) and the money
    a town portal: a hero near the goblin merchant buys one when it has none and a free slot, from 440 s
    a potion:      a hero below 40% of its hit points drinks what heals in its inventory
    an item:       (--macros 2) a hero with an item lying near (macros.pickable) and no enemy unit within ENEMY_NEAR
                   picks it up (the agent's pick-up order: creeps' drops lay where they fell, the policy had no order
                   that took them)
    the army mix:  from MIX_FROM, with 5 army units or more, the unit its target's army has the largest share of
                   more than its own army (by MIX_SLACK at least) trained at a building that makes it (the target's
                   mix as an input and a reward, fgself-130, left the mix gap where it was: 0.62)
    an expansion:  when its target's expansion is due (a hall, or undead's haunted gold mine, in its items) and the
                   natural expansion's camp is gone (cleared, or --scenarios natural): the worker nearest builds the
                   hall where people put theirs (undead: an acolyte haunts the mine), then three workers go to mine there.
                   (No policy had ever expanded: 0% of its games, people 81% of Echo Isles'. Night elves: the agent
                   has a tree of life entangle the mine, play.BCAgent._entangle, and the workers then mine that.)

    lumber:        (--teach 3) a target's next item short of lumber (the builder's) and fewer than CHOPPERS on lumber: the race's
                   chopper (undead: a ghoul) nearest the tree nearest its hall harvests it (the policy had all but stopped
                   chopping: human 38 lumber a minute to 872 gold)

Each kind at most every GAP steps a side, so a long wait for the money is not a label a step."""

from __future__ import annotations

import random

import numpy as np

from ..protocol import Build, Buy, ImmediateOrder, TargetOrder
from . import features as fx
from .neutral import fourcc

# the tavern heroes people pick for each race (1.29 ladder habits)
PICKS = {"human": ("Npbm", "Nfir", "Ntin"), "orc": ("Ntin", "Npbm", "Nalc"), "undead": ("Nplh", "Nbrn", "Nalc"),
         "nightelf": ("Npbm", "Nngs", "Nbst")}
WORKERS = {fourcc(c) for c in ("hpea", "opeo", "uaco", "ewsp")}
ALTARS = {fourcc(c) for c in ("halt", "oalt", "uaod", "eate")}
ALTAR = {"human": "halt", "orc": "oalt", "undead": "uaod", "nightelf": "eate"}
# the first heroes people train at the altar (1.29 ladder habits)
FIRST_HEROES = {"human": ("Hamg", "Hmkg", "Hpal"), "orc": ("Obla", "Ofar", "Otch"), "undead": ("Udea", "Udre", "Ulic"),
                "nightelf": ("Edem", "Ekee", "Emoo")}
FIRST_HERO_FROM = 45.0
TIER2 = {fourcc(c) for c in ("hkee", "hcas", "ostr", "ofrt", "unp1", "unp2", "etoa", "etoe")}
TIER3 = {fourcc(c) for c in ("hcas", "ofrt", "unp2", "etoe")}
TAVERN_FROM = 135.0  # (a game without a target: its tavern hero, the second, from then)
TAVERN, MERCHANT, PORTAL = fourcc("ntav"), fourcc("ngme"), fourcc("stwp")
GAP = 20  # steps between two lessons of a kind
HALL = {"human": "htow", "orc": "ogre", "undead": "unpl", "nightelf": "etol"}
HALLS_ALL = ("htow", "hkee", "hcas", "ogre", "ostr", "ofrt", "unpl", "unp1", "unp2", "etol", "etoa", "etoe")
HAUNTED, ENTANGLED = fourcc("ugol"), fourcc("egol")
HALL_CODES, OWN_MINE_CODES = fx.codes(fourcc(h) for h in HALLS_ALL), fx.codes((HAUNTED, ENTANGLED))
MINES = {fourcc(c) for c in ("ngol", "ugol", "egol")}
# (the sets above as features.isin's sorted arrays)
WORKER_CODES, ALTAR_CODES, MINE_CODES = fx.codes(WORKERS), fx.codes(ALTARS), fx.codes(MINES)
# where people built their halls at Echo Isles' natural expansions (95 games of 1.29: the mine -> the hall's offset)
EXPAND_SPOTS = {(-4864, -3200): (100, 800), (4352, -3200): (-300, 700)}
NATURAL_FROM_START = 1500.0  # (the main mine is nearer its start)
CAMP_NEAR = 900.0  # creeps this near the natural mine guard it
HARVEST = 852018
HIDDEN = 16  # (UnitFlags.HIDDEN: a worker inside a gold mine)
MIX_FROM, MIX_SLACK, MIX_LEAST = 120.0, 0.15, 5
# the lumber lesson's choppers (undead's: ghouls) and how many at most: people put ~5 workers on lumber (2-3 ghouls),
# the policy almost none (human 38 lumber a minute to 872 gold in fgself-145; fgself-120: 105), and a target's items
# waited for lumber more than for anything else (a keep 516k blocked steps, halls of the dead 1.1M; audit 9)
CHOPPERS = {"human": ("hpea", 5), "orc": ("opeo", 5), "undead": ("ugho", 3), "nightelf": ("ewsp", 5)}
NEAR_MERCHANT = 2500.0
ENEMY_NEAR = 800.0  # (an item lesson waits while an enemy unit is this near the hero: a fight)
USESLOT = 852008


class Teacher:
    """One side's lessons in one game (commands: what it would order now)."""

    def __init__(self, side: int, race: str, rng: random.Random, target: dict | None = None, target_hero: bool = False,
                 lumber: bool = False):
        """`target` (the side's target strategy, if it has one): its tavern hero is the one taught, and none when it
        has none (another hero would break a narrow target). `target_hero` (--teach 2): its first hero from the altar
        taught too, when that is its first hero (the builder taught it only once due: the policy had made another one
        ~10-20 s before, and the second needs a tier; 25-37% of the first heroes were the target's, chance among 3-4)."""
        from .neutral import TAVERN_HEROES
        self.side = side
        self.lumber = lumber  # (--teach 3: the lumber lesson)
        # the tavern hero taught, and the heroes the side has when it is hired: a target's, at its place among the
        # target's heroes (its next hero), as soon as the altar, the tier and the money permit. (As the first or
        # second hero only, from 135 s: the lesson had fired 68 times in 3,000 target games, ~1,300 of them with a
        # tavern hero; by 135 s a first hero was the altar's, and a third was never taught: audit 7's sequel)
        self.pick_at, self.pick_from = 1, TAVERN_FROM
        if target is None:
            self.pick = fourcc(rng.choice(PICKS.get(race, PICKS["human"])))
        else:
            from .techtree import is_hero
            heroes = [c for _, c in sorted((t, c) for c, t in target.get("t", {}).items() if is_hero(c))]
            k = next((i for i, c in enumerate(heroes) if c in TAVERN_HEROES), None)
            self.pick = fourcc(heroes[k]) if k is not None else None
            self.pick_at, self.pick_from = (k if k is not None else 0), 0.0
        self.last: dict[str, int] = {}
        self.taught: dict[str, int] = {}  # each kind's lessons this game (the episode row)
        # (a game without a target: its first hero from the altar, FIRST_HEROES)
        self.first = fourcc(rng.choice(FIRST_HEROES.get(race, FIRST_HEROES["human"]))) if target is None else None
        self.first_from = FIRST_HERO_FROM  # (the target's hero: as soon as the altar stands, the mask and food permitting;
        if target is not None and target_hero:  # 45 s had held 71% of its lessons back: the altar was up at ~33 s, audit 7)
            from .techtree import is_hero
            heroes = sorted((t, c) for c, t in target.get("t", {}).items() if is_hero(c))
            if heroes and heroes[0][1] not in TAVERN_HEROES:  # (a tavern hero first: the tavern's lesson below)
                self.first, self.first_from = fourcc(heroes[0][1]), 0.0
        self.rng = rng
        self.race = race
        due = [v for c, v in (target or {}).get("t", {}).items() if c in (HALL.get(race), "ugol")]
        self.expand_at = min(due) if due else None  # (its target's expansion: when)
        from .strategy import is_army
        army = {c: v for c, v in ((target or {}).get("n") or {}).items() if is_army(c) and v > 0}
        total = sum(army.values())
        self.mix = {c: v / total for c, v in army.items()} if total else {}  # (its target's army mix)

    def _due(self, kind: str, t: int, counted: str | None = None) -> bool:
        """Whether a lesson of `kind` may be given now (and it is counted, as `counted`: one hero's of many kinds)."""
        if t - self.last.get(kind, -GAP) < GAP:
            return False
        self.last[kind] = t
        self.taught[counted or kind] = self.taught.get(counted or kind, 0) + 1
        return True

    def commands(self, rows: np.ndarray, player, now: float, t: int, can=None, pickers=()) -> list:
        """rows: the step's units (features' columns, ROW_COLS with the heroes' items), player: its PlayerState.
        `can(code[, kind])`: whether the policy could order that unit now (its mask: the money, the tech; kind: the
        order's, immediate by default, features.POINT for a building); the army mix teaches only those (99% of its
        lessons had named a unit the policy could not order then, and taught nothing).
        `pickers`: the side's heroes with an item lying near (macros.pickable, the view's of the step)."""
        self.can = can
        if player is None or rows.shape[1] < fx.ROW_COLS:
            return []
        out = []
        fl, typ = rows[:, fx.C_FLAGS], rows[:, fx.C_TYPE]  # (masks over all the rows, copies of the few they pick:
        own, living = rows[:, fx.C_OWNER] == self.side, fl & fx.DEAD == 0  # (a copy of every living unit a step had
        alive = fx.own_alive(rows, self.side, own & living)  # been ~9% of an actor's time; the living own ones: the
        heroes = rows[own & (fl & fx.HERO > 0)]  # trackers' copy of the step before. Dead heroes count: revived)
        fa = alive[:, fx.C_FLAGS]
        live_heroes = alive[fa & fx.HERO > 0]
        # the neutral buildings: on the game's minimap from the start
        tavern, merchant = rows[living & (typ == TAVERN)], rows[living & (typ == MERCHANT)]
        done = alive[(fa & fx.STRUCTURE > 0) & (fa & fx.CONSTRUCTING == 0), fx.C_TYPE]
        n_heroes = len(heroes)
        if self.first is not None and n_heroes == 0 and now >= self.first_from:  # (the altar itself from 45 s: a target's
            out += self._first_hero(alive, player, t, build=now >= FIRST_HERO_FROM)  # builder teaches it when due)
        # a tavern hero: when the side has as many heroes as come before it (the target's, or one) or more (an altar's
        # made first: 269 of 289 human games with a tavern first hero; audit 9), with an altar, the
        # tier the hero count needs (the second hero tier 2, the third tier 3) and the money (the mask's, else 425/135;
        # the first hero is free: food only)
        if (self.pick is not None and len(tavern) and now >= self.pick_from and self.pick_at <= n_heroes < 3
                and (n_heroes == 0 or (can(fx.rawcode(self.pick), fx.UNIT) if can is not None
                                       else player.gold >= 425 and player.lumber >= 135))
                and player.food_used + 5 <= min(player.food_cap, fx.FOOD_MAX) and fourcc_in(done, ALTARS)
                and (n_heroes == 0 or fourcc_in(done, TIER2)) and (n_heroes < 2 or fourcc_in(done, TIER3))
                and self.pick not in set(heroes[:, fx.C_TYPE].tolist())):
            tv = tavern[0]
            workers = alive[fx.isin(alive[:, fx.C_TYPE], WORKER_CODES)]
            if len(workers):
                d = np.hypot(workers[:, fx.C_X] - tv[fx.C_X], workers[:, fx.C_Y] - tv[fx.C_Y])
                k = int(np.argmin(d))
                if self._due("tavern hero", t):  # (the nearest worker: it walks there, protocol.Buy)
                    out.append(Buy(int(workers[k, fx.C_ID]), int(tv[fx.C_ID]), self.pick))
        pickers = set(pickers or ())
        enemies = rows[(rows[:, fx.C_OWNER] == 1 - self.side) & (fl & (fx.DEAD | fx.STRUCTURE) == 0)] \
            if pickers and fx.MACROS >= 2 else rows[:0]  # (only for a hero with an item near)
        for h in live_heroes:
            items = [int(v) for v in h[fx.C_ITEM0:fx.C_ITEM0 + 6]]
            # an item lying near, picked up when no enemy is near
            if int(h[fx.C_ID]) in pickers and fx.MACROS >= 2 and not (
                    np.hypot(enemies[:, fx.C_X] - h[fx.C_X], enemies[:, fx.C_Y] - h[fx.C_Y]) < ENEMY_NEAR).any():
                if self._due(f"item {int(h[fx.C_ID])}", t, "item"):
                    from .macros import PICKUP
                    out.append(ImmediateOrder(int(h[fx.C_ID]), PICKUP))
            # a potion when it is low
            if h[fx.C_HP] < 0.4 * max(int(h[fx.C_MAXHP]), 1):
                from .herodata import item_table
                heal = [k for k, it in enumerate(items) if it and item_table().get(it, (0, 0, 0, 0))[1]]
                if heal and self._due(f"potion {int(h[fx.C_ID])}", t, "potion"):
                    out.append(ImmediateOrder(int(h[fx.C_ID]), USESLOT + heal[0]))
            # a town portal from the goblin merchant
            if (len(merchant) and now >= 440.0 and PORTAL not in items and 0 in items and player.gold >= 500):
                m = merchant[0]
                if np.hypot(h[fx.C_X] - m[fx.C_X], h[fx.C_Y] - m[fx.C_Y]) < NEAR_MERCHANT and self._due("town portal", t):
                    out.append(Buy(int(h[fx.C_ID]), int(m[fx.C_ID]), PORTAL))
        if self.expand_at is not None and now >= self.expand_at:
            out += self._expansion(rows, alive, player, t)
        if self.mix and now >= MIX_FROM:
            out += self._mix(alive, t)
        eco = getattr(self, "economy", None)  # (short, the choppers' ids, the trees: play_one's, from the builder)
        if self.lumber and eco is not None and eco[0] and eco[2] and t - self.last.get("lumber", -GAP) >= GAP:
            out += self._lumber(alive, eco[1], eco[2], t)
        return out

    def _lumber(self, alive: np.ndarray, choppers: set, trees: dict, t: int) -> list:
        """A target's next item short of lumber and fewer than CHOPPERS on it: the race's chopper nearest the tree
        nearest its hall (one not chopping, not hidden, not building) harvests that tree."""
        code, most = CHOPPERS.get(self.race, (None, 0))
        if code is None or len(choppers) >= most:
            return []
        fl = alive[:, fx.C_FLAGS]
        cand = alive[(alive[:, fx.C_TYPE] == fourcc(code)) & (fl & (HIDDEN | fx.STRUCTURE) == 0)
                     & (alive[:, fx.C_ORDER] < fx.TYPE_CODE) & ~np.isin(alive[:, fx.C_ID], list(choppers))]
        halls = alive[fx.isin(alive[:, fx.C_TYPE], HALL_CODES)]
        if not len(cand) or not len(halls):
            return []
        ids = np.fromiter(trees.keys(), np.int64, len(trees))
        xy = np.asarray(list(trees.values()), np.float64).reshape(-1, 2)
        h = halls[0]
        k = int(np.argmin((xy[:, 0] - h[fx.C_X]) ** 2 + (xy[:, 1] - h[fx.C_Y]) ** 2))
        w = cand[int(np.argmin((cand[:, fx.C_X] - xy[k, 0]) ** 2 + (cand[:, fx.C_Y] - xy[k, 1]) ** 2))]
        return [TargetOrder(int(w[fx.C_ID]), HARVEST, int(ids[k]))] if self._due("lumber", t) else []

    def _first_hero(self, alive: np.ndarray, player, t: int, build: bool = True) -> list:
        """A side without a hero: its altar trains its first hero, and with no altar (nor one being built) a worker
        builds one near its hall; each when the policy could order it (`can`)."""
        can = getattr(self, "can", None)
        altars = alive[fx.isin(alive[:, fx.C_TYPE], ALTAR_CODES)]
        done = altars[altars[:, fx.C_FLAGS] & fx.CONSTRUCTING == 0]
        if len(done):
            if (can is None or can(fx.rawcode(self.first))) and player.food_used + 5 <= min(player.food_cap, fx.FOOD_MAX) \
                    and self._due("first hero", t):
                return [ImmediateOrder(int(done[0, fx.C_ID]), self.first)]
            return []
        if len(altars) or not build:
            return []  # (being built)
        code = ALTAR.get(self.race)
        workers = alive[fx.isin(alive[:, fx.C_TYPE], WORKER_CODES) & (alive[:, fx.C_FLAGS] & HIDDEN == 0)
                        & (alive[:, fx.C_FLAGS] & fx.STRUCTURE == 0)]
        halls = alive[fx.isin(alive[:, fx.C_TYPE], HALL_CODES)]
        if code is None or not len(workers) or (can is not None and not can(code, fx.POINT)) or not self._due("altar", t):
            return []
        at = halls[0] if len(halls) else workers[0]
        w = workers[int(np.argmin(np.hypot(workers[:, fx.C_X] - at[fx.C_X], workers[:, fx.C_Y] - at[fx.C_Y])))]
        a, r = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(450, 800)
        return [Build(int(w[fx.C_ID]), fourcc(code), float(at[fx.C_X] + r * np.cos(a)), float(at[fx.C_Y] + r * np.sin(a)))]

    def _mix(self, alive: np.ndarray, t: int) -> list:
        """The unit its target's army has the largest share of more than its own, trained at a building that makes it."""
        from .techtree import tree
        fl = alive[:, fx.C_FLAGS]
        army = alive[fl & (fx.STRUCTURE | fx.WORKER | fx.HERO) == 0]
        if len(army) < MIX_LEAST:
            return []
        have = {}
        for c in army[:, fx.C_TYPE].tolist():
            k = int(c).to_bytes(4, "big").decode("latin-1")
            have[k] = have.get(k, 0) + 1 / len(army)
        done = alive[(fl & fx.STRUCTURE > 0) & (fl & fx.CONSTRUCTING == 0)]
        can = getattr(self, "can", None)
        for unit, gap in sorted(((u, s - have.get(u, 0.0)) for u, s in self.mix.items()), key=lambda x: -x[1]):
            if gap < MIX_SLACK:
                return []
            if can is not None and not can(unit):
                continue  # (not now: the next most lacking that it can)
            at = done[np.isin(done[:, fx.C_TYPE], [fourcc(m) for m in tree().makers.get(unit, ())])]
            if len(at):
                if fx.MACROS:  # (the building's mix order: which unit is its business, macros.MIX)
                    from .macros import MIX
                    return [ImmediateOrder(int(at[0, fx.C_ID]), MIX)] if self._due("army mix", t) else []
                return [ImmediateOrder(int(at[0, fx.C_ID]), fourcc(unit))] if self._due("army mix", t) else []
        return []

    def _expansion(self, rows: np.ndarray, alive: np.ndarray, player, t: int) -> list:
        """The next step of taking the natural expansion: clearing its camp, the hall (undead: the haunted mine), the
        workers that mine there."""
        sx, sy = getattr(player, "start_x", None), getattr(player, "start_y", None)
        mines = rows[fx.isin(rows[:, fx.C_TYPE], MINE_CODES) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]
        if sx is None or not len(mines):
            return []
        d = np.hypot(mines[:, fx.C_X] - sx, mines[:, fx.C_Y] - sy)
        o = np.argsort(d)
        far = o[d[o] > NATURAL_FROM_START]
        if not len(far):
            return []
        nat = mines[far[0]]
        mx, my = float(nat[fx.C_X]), float(nat[fx.C_Y])
        near = lambda r, rad: np.hypot(r[:, fx.C_X] - mx, r[:, fx.C_Y] - my) < rad  # noqa: E731
        hall = fourcc(HALL[self.race])
        any_worker = fx.isin(alive[:, fx.C_TYPE], WORKER_CODES)
        workers = alive[any_worker & (alive[:, fx.C_FLAGS] & HIDDEN == 0)]
        mining = alive[any_worker]  # (with those inside a mine: wisps in an entangled one are hidden, and the lesson
        from .macros import HOLD_CODES  # had never seen 4 there: 22 a long night elf game; audit 8)
        taken = alive[fx.isin(alive[:, fx.C_TYPE], HOLD_CODES)]  # (the distance only to those: per step)
        taken = taken[near(taken, 1000.0)]
        if len(taken):  # (taken: once finished, three workers to mine there)
            if (taken[:, fx.C_FLAGS] & fx.CONSTRUCTING > 0).all() or (near(mining, 900.0).sum() >= 4) or not len(workers):
                return []
            target = alive[fx.isin(alive[:, fx.C_TYPE], OWN_MINE_CODES)]
            target = target[near(target, 1000.0)]
            if self.race == "nightelf" and not len(target):
                return []  # (its tree of life entangles the mine first)
            mine_id = int(target[0, fx.C_ID]) if len(target) else int(nat[fx.C_ID])
            far_w = workers[~near(workers, 900.0)]
            pick = far_w[np.argsort(np.hypot(far_w[:, fx.C_X] - mx, far_w[:, fx.C_Y] - my))[:3]]
            return ([TargetOrder(int(w[fx.C_ID]), HARVEST, mine_id) for w in pick]
                    if len(pick) and self._due("expansion: workers", t) else [])
        creeps = rows[(rows[:, fx.C_OWNER] == 12) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]
        creeps = creeps[near(creeps, CAMP_NEAR)]
        if len(creeps):  # (guarded. Sending the whole army to clear it every 10 s (fgself-123) lost the games that had
            # an expansion due, 13% won to 32%. With --macros: its heroes, while the army is 6 units or more, are shown
            # the creep order (the policy decides who goes with them))
            army = alive[(alive[:, fx.C_FLAGS] & (fx.STRUCTURE | fx.WORKER) == 0)]
            heroes = army[army[:, fx.C_FLAGS] & fx.HERO > 0]
            if not fx.MACROS or len(army) < 6 or not len(heroes) or not self._due("expansion: its camp", t):
                return []
            from .macros import CREEP
            return [ImmediateOrder(int(h[fx.C_ID]), CREEP) for h in heroes]
        if not len(workers) or not self._due("expansion: the hall", t):
            return []
        if fx.MACROS:  # (the agent's expand order: where to build is its own business, macros.py)
            from .macros import EXPAND
            w = workers[int(np.argmin(np.hypot(workers[:, fx.C_X] - mx, workers[:, fx.C_Y] - my)))]
            return [ImmediateOrder(int(w[fx.C_ID]), EXPAND)]
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


_CODES: dict[int, tuple] = {}  # (a set of codes -> (the set, its sorted array): fourcc_in's, kept by identity)


def fourcc_in(types: np.ndarray, codes: set[int]) -> bool:
    c = _CODES.get(id(codes))
    if c is None or c[0] is not codes:
        c = _CODES[id(codes)] = (codes, fx.codes(codes))
    return bool(fx.isin(types, c[1]).any())
