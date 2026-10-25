"""A hand-written Warcraft III player (the user's idea, 2026-10-25): a scripted opponent strong enough to beat the
learner shows where the learner is weak, can join its training league (selfplay --heuristic-share) and serves as a
yardstick that does not move.

    agent = HeuristicAgent(side, "orc")
    agent.begin(obs)                    # the game's first observation
    while not obs.game_over:
        agent.observe(obs, t)
        cmds = agent.commands()
        obs = game.step(cmds)
        agent.accepted(cmds, obs.command_results[:len(cmds)])

It sees what its player sees: enemy units only while they are visible to it (buildings remembered where it saw
them), its own units, the gold mines, the trees and where the map's creep camps stand (a player knows the map).
Nothing in it learns. Every half second it:

* economy: sends idle workers to its mines (5 a mine) or the trees nearest its hall, trains workers up to the plan's
  count, builds supply ahead of its production (never supply-blocked), follows the race's build order and keeps
  every production building training the unit its army mix lacks most (waiting for that one's money: the cheaper
  ones had starved it), takes the natural expansion once its camp is cleared (or its main mine runs low), and moves
  workers off a mine that runs dry;
* heroes: trains the plan's hero at the altar (the first one is free), learns its skills in the plan's order,
  revives it when it dies (its money kept first), casts its spells in fights (nukes and stuns on enemy heroes, the
  most clumped or the weakest enemies, heals on hurt own units, summons and area spells when enough enemies are
  near) and sends it home when it is hurt;
* army: gathers in front of its base; clears the creep camps near home and the natural's for the hero's levels
  while nothing threatens; defends the base when enemies come near (workers into burrows for orc); attacks when its
  army reaches the plan's size (or the enemy's army just died at its base), attack-moving to the enemy's buildings it
  knows (else its start, then the other mines) while the leaders wait for the rest. In a fight each unit attacks
  the enemy in reach with the least hit points left after the damage the others already aim at it (focus fire
  without overkill), hurt melee units step back while they are hit, and the army retreats home when the enemies
  near it are clearly stronger (Lanchester: hit points times damage a second, both sides).

Plans (PLANS) give each race's build order, army mix, heroes and timings. `log`: a timeline (a line every 30 game
seconds and the army's decisions) for heuristic_match's rows (summary())."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..protocol import (Build, Command, EventKind, ImmediateOrder, LearnSkill, Observation, PointOrder,
                        TargetDestructable, TargetOrder, rawcode)

# orders (the harness resolves the strings; these are the game's ids, checked in a game: obs.orders)
ATTACK, MOVE, SMART, STOP, HOLD = 851983, 851986, 851971, 851972, 851993
HARVEST, RETURN, RESUME, REPAIR, REVIVE, CANCEL, SETRALLY = 852018, 852020, 852017, 852024, 852039, 851976, 851980
BATTLESTATIONS, STANDDOWN, BOARD = 852099, 852113, 852043
HARVESTING = {HARVEST, RETURN, RESUME}
# unit flags (protocol.UnitFlags)
HERO, STRUCTURE, WORKER, CONSTRUCTING, HIDDEN, LOADED, SLEEPING = 1, 2, 4, 8, 16, 32, 64
SUMMONED, ILLUSION, DEAD, FLYING = 256, 512, 1024, 2048
CREEPS = 12  # neutral hostile's player
TYPE_CODE = 0x41000000  # (an order at or above this is an object's code: train, build, research)


def fourcc(c: str) -> int:
    return int.from_bytes(c.encode("latin-1"), "big")


# ---- the races' plans ----------------------------------------------------------------------------------------------
@dataclass
class Plan:
    race: str
    worker: str
    hall: str
    supply: str
    supply_food: int
    altar: str
    heroes: tuple[str, ...]
    # the build order: (game second it is due from, food used it is due from, building, how many of it by then)
    builds: tuple[tuple[float, int, str, int], ...]
    army: tuple[tuple[str, float], ...]  # the army mix: unit, weight
    research: tuple[tuple[float, str], ...] = ()  # (due from, research): one level each, in this order
    gold_workers: int = 5  # a mine's
    lumber_workers: tuple[tuple[float, int], ...] = ((0.0, 3), (150.0, 5))  # (from, how many)
    attack_food: int = 40  # the army's food (heroes not counted) it attacks with
    attack_at: float = 720.0  # from then it attacks with two thirds of attack_food
    queue: int = 2  # units queued at a production building at most (one training, one waiting)
    towers: tuple[tuple[float, str, int], ...] = ()  # (due from, tower, how many) built in front of the base
    creep_from: float = 100.0  # creep camps near home from then, with the hero and creep_units units
    creep_until: float = 480.0
    creep_units: int = 3
    expand_at: float = 420.0  # the natural expansion from then (its camp cleared first), or its main mine low
    # against the races whose first waves come at ~4:30 (the learner's undead and orc): another build order and towers
    rush_builds: tuple[tuple[float, int, str, int], ...] | None = None
    rush_towers: tuple[tuple[float, str, int], ...] | None = None


PLANS: dict[str, Plan] = {
    "orc": Plan(
        race="orc", worker="opeo", hall="ogre", supply="otrb", supply_food=10, altar="oalt",
        heroes=("Ofar",),
        builds=((0.0, 0, "oalt", 1), (0.0, 0, "obar", 1), (55.0, 9, "obar", 2), (110.0, 14, "obar", 3),
                (240.0, 26, "ofor", 1)),
        # watch towers in front of the base (the learner's waves fight it there; the lumber had piled up unspent)
        towers=((300.0, "owtw", 2), (420.0, "owtw", 4)),
        # against undead and orc an earlier war mill and more towers sooner: every loss to the learner's undead came
        # at 6.5-9.6 minutes, its ghouls twice the grunts, and the games that held to ~13 minutes were won (its mine dry)
        rush_builds=((0.0, 0, "oalt", 1), (0.0, 0, "obar", 1), (55.0, 9, "obar", 2), (110.0, 14, "obar", 3),
                     (150.0, 16, "ofor", 1)),
        rush_towers=((200.0, "owtw", 2), (300.0, "owtw", 4), (420.0, "owtw", 6)),
        army=(("ogru", 1.0), ("ohun", 1.0)),
        research=((360.0, "Rome"), (480.0, "Rora"), (600.0, "Roar")),
        lumber_workers=((0.0, 3),),  # (5 had floated 600-1,300 lumber: peons the army lacked)
        # (patient: against the learner, games past 14 minutes were won 61 of 78, its one mine dry by then)
        attack_food=60, attack_at=840.0),
    "human": Plan(
        race="human", worker="hpea", hall="htow", supply="hhou", supply_food=6, altar="halt",
        heroes=("Hmkg",),
        builds=((0.0, 0, "halt", 1), (0.0, 0, "hbar", 1), (100.0, 12, "hbar", 2), (200.0, 22, "hbar", 3),
                (280.0, 30, "hbla", 1)),
        army=(("hfoo", 2.0), ("hrif", 1.0)),
        research=((300.0, "Rhde"), (420.0, "Rhme"), (540.0, "Rhar")),
        attack_food=45, attack_at=660.0),
    "undead": Plan(
        race="undead", worker="uaco", hall="unpl", supply="uzig", supply_food=10, altar="uaod",
        heroes=("Udea",),
        builds=((0.0, 0, "uaod", 1), (0.0, 0, "usep", 1), (100.0, 12, "usep", 2), (200.0, 22, "usep", 3),
                (280.0, 30, "ugrv", 1)),
        army=(("ugho", 1.0),),
        lumber_workers=((0.0, 0),),  # (ghouls chop: _ghoul_lumber)
        attack_food=45, attack_at=660.0),
    "nightelf": Plan(
        race="nightelf", worker="ewsp", hall="etol", supply="emow", supply_food=10, altar="eate",
        heroes=("Edem",),
        builds=((0.0, 0, "eate", 1), (0.0, 0, "eaom", 1), (100.0, 12, "eaom", 2), (200.0, 22, "eaom", 3)),
        army=(("earc", 1.0),),
        attack_food=45, attack_at=660.0),
}
# skills in the order learned (then the built-in AI's order: herodata.next_skill)
SKILLS = {
    "Ofar": ("AOsf", "AOcl", "AOsf", "AOcl", "AOsf", "AOeq", "AOcl", "AOfs", "AOfs", "AOfs"),
    "Otch": ("AOsh", "AOws", "AOsh", "AOws", "AOsh", "AOre", "AOws", "AOae", "AOae", "AOae"),
    "Obla": ("AOwk", "AOcr", "AOwk", "AOcr", "AOcr", "AOww", "AOwk", "AOmi", "AOmi", "AOmi"),
    "Oshd": ("AOhw", "AOhx", "AOhw", "AOhx", "AOhw", "AOvd", "AOhx", "AOsw", "AOsw", "AOsw"),
    "Hmkg": ("AHtb", "AHtc", "AHtb", "AHtc", "AHtb", "AHav", "AHtc", "AHbh", "AHbh", "AHbh"),
    "Hpal": ("AHhb", "AHad", "AHhb", "AHad", "AHhb", "AHre", "AHad", "AHds", "AHds", "AHds"),
    "Hamg": ("AHwe", "AHab", "AHwe", "AHab", "AHwe", "AHmt", "AHab", "AHbz", "AHbz", "AHbz"),
    "Udea": ("AUdc", "AUau", "AUdc", "AUau", "AUdc", "AUan", "AUau", "AUdp", "AUdp", "AUdp"),
    "Ulic": ("AUfn", "AUfu", "AUfn", "AUfu", "AUfn", "AUdd", "AUfu", "AUdr", "AUdr", "AUdr"),
    "Udre": ("AUcs", "AUav", "AUcs", "AUav", "AUcs", "AUin", "AUav", "AUsl", "AUsl", "AUsl"),
    "Edem": ("AEmb", "AEim", "AEmb", "AEim", "AEmb", "AEme", "AEev", "AEev", "AEev", "AEim"),
    "Ekee": ("AEer", "AEah", "AEer", "AEah", "AEer", "AEtq", "AEah", "AEfn", "AEfn", "AEfn"),
}
# what a spell's order does (herodata's cast order strings): how the hero casts it
SPELLS = {
    "chainlightning": "nuke", "thunderbolt": "nuke", "frostnova": "nuke", "hex": "disable", "sleep": "disable",
    "entanglingroots": "disable", "manaburn": "manaburn", "shadowstrike": "nuke", "forkedlightning": "nuke",
    "shockwave": "line", "carrionswarm": "line", "impale": "line",
    "thunderclap": "stomp", "stomp": "stomp",
    "spiritwolf": "summon", "waterelemental": "summon",
    "holybolt": "heal", "healingwave": "heal", "deathcoil": "deathcoil",
    "windwalk": "escape", "mirrorimage": "escape", "divineshield": "escape",
}
SPELL_RANGE = {"nuke": 650.0, "disable": 600.0, "manaburn": 300.0, "line": 650.0, "heal": 700.0, "deathcoil": 750.0}
STOMP_RADIUS = 260.0
TOWERS_ATTACK = {fourcc(c) for c in ("hgtw", "hctw", "hatw", "owtw", "otrb", "uzg1", "uzg2", "etrp", "ntt1", "ntx2")}
HALL_TYPES = {"htow", "hkee", "hcas", "ogre", "ostr", "ofrt", "unpl", "unp1", "unp2", "etol", "etoa", "etoe"}
MINE_ID = fourcc("ngol")
HAUNTED, ENTANGLED = fourcc("ugol"), fourcc("egol")
# where people built their halls at Echo Isles' natural expansions (macros.EXPAND_SPOTS: the mine -> the hall's offset)
EXPAND_SPOTS = {(-4864, -3200): (100, 800), (4352, -3200): (-300, 700)}

DEFEND_RADIUS = 1700.0  # enemies this near the hall threaten the base
NEAR_BUILDING = 800.0  # or this near another building of the player's
FIGHT_RADIUS = 1100.0  # enemies this near the army's centre are in its fight
ENGAGE = 900.0  # a unit fights the enemies within this of it
RETREAT_RATIO = 0.6  # the army's strength below this share of the enemies' near it: retreat (away from home)
RETREAT_STEPS = 16  # at least this long (8 s) once retreating
RALLY_OUT = 650.0  # the rally point: this far from the hall toward the enemy's base
HERO_LOW, HERO_OK = 0.38, 0.75  # a hero this hurt goes home until it has this share of its hit points again
PULL_LOW, PULL_STEPS = 0.0, 4  # a hurt melee unit steps back this long (off: chasers as fast caught them)
CAMP_JOIN = 550.0  # creeps this near each other: one camp
CREEP_REACH = 3600.0  # camps this near home are crept
MINE_LOW = 3000  # a mine with less gold left than this: the next one's hall built
QUEUE_EXPIRE = {"R": 200.0}  # (seconds a queued entry is kept per place in the queue, by its code's first letter)
REVIVE_WAIT = 240  # a revival the altar took: not ordered again for 120 s (it takes up to ~110 s)
FOOD_CAP = 80  # no army past this food (81 and more: 60% of the income lost to upkeep)
REGROUP_STEPS = 90  # after a retreat no attack for 45 s
TOWER_COST = 2500.0  # a known enemy tower near a target counts as this much more distance


# switches (heuristic_match.py --variant heuristic:HeuristicAgent:sticky=0,...): what each changed, for side-by-side runs
OPTIONS = {
    "rush_caution": 1,  # against undead and orc: no early creeping far from home, the natural later
    "hero_behind": 1,  # the hero follows behind the army's centre instead of walking in first
    "sticky": 1,  # a unit keeps its target while it is in reach and not dying already
    "pull": 0,  # hurt melee units step back while hit
    "focus": 2,  # each unit in a fight attacks the weakest enemy in reach (0: they attack-move, the game picks; 2:
    # ranged units only; 16 games each on the same matchups, 2026-10-25: 1 and 0 level, 17 and 16 of 32 won)
    "spells": 1,  # heroes cast their spells
    "towers": 1,  # enemy towers that shoot are fight targets
    "hero_home": 1,  # a hurt hero walks home until healed
    "towers_built": 1,  # the plan's towers in front of the base
    "hold_home": 1,  # a wave as strong as the army is met near the hall (its towers, burrows with their peons)
    "strict_budget": 0,  # 1: money kept for a due building short of lumber holds every other order back (the check
    # before 2026-10-25's fix: in effect buildings and supply first)
}


@lru_cache(1)
def unit_data() -> dict[str, tuple]:
    """Unit type -> (hit points, gold, lumber, food, range, cooldown, damage, armor, speed) from the game's tables."""
    from ..data.objects import combat_stats, unit_table
    ut, cs = unit_table(), combat_stats()
    out = {}
    for c, u in ut.items():
        s = cs.get(c)
        out[c] = (u.hp, u.gold, u.lumber, u.food, s.range if s else 0.0, s.cooldown if s else 0.0,
                  s.damage if s else 0.0, s.armor if s else 0.0, s.speed if s else 0.0)
    return out


@lru_cache(1)
def research_costs() -> dict[str, tuple[int, int, int, int]]:
    """Research -> (gold, lumber, gold per level, lumber per level) from UpgradeData.slk."""
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        rows = parse_slk(g.read("Units\\UpgradeData.slk").decode("latin-1"))

    def num(r, k):
        try:
            return int(float(r.get(k) or 0))
        except ValueError:
            return 0
    return {r["upgradeid"]: (num(r, "goldbase"), num(r, "lumberbase"), num(r, "goldmod"), num(r, "lumbermod"))
            for r in rows if r.get("upgradeid")}


@lru_cache(1)
def footprints() -> dict[str, int]:
    """Building type -> its footprint's side in pathing cells (32 world units; UnitData.slk's pathTex, "12x12...")."""
    import re
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        rows = parse_slk(g.read("Units\\UnitData.slk").decode("latin-1"))
    out = {}
    for r in rows:
        m = re.search(r"(\d+)x(\d+)", str(r.get("pathTex") or ""))
        if m and r.get("unitID"):
            out[r["unitID"]] = max(int(m.group(1)), int(m.group(2)))
    out.setdefault("unpl", 16)
    out.setdefault("etol", 12)
    return out


@lru_cache(1)
def upgraded_from() -> dict[str, str]:
    from .techtree import tree
    return {to: frm for frm, tos in tree().upgrades.items() for to in tos}


def dps(code: str, level: int = 0) -> float:
    d = unit_data().get(code)
    if d is None or d[5] <= 0:
        return 8.0 if code[:1].isupper() else 0.0
    return d[6] / d[5] * (1.0 + 0.12 * max(level - 1, 0))


def power(units, center=None, radius: float = FIGHT_RADIUS, workers: bool = True) -> float:
    """Lanchester's strength of a group: its hit points times its damage a second (heroes by level), / 1000."""
    hp = dmg = 0.0
    for u in units:
        if center is not None and math.hypot(u.x - center[0], u.y - center[1]) > radius:
            continue
        if u.is_worker and not workers:
            continue
        hp += max(u.hp, 0)
        dmg += dps(u.type, u.hero_level) * (0.5 if u.is_worker else 1.0)
    return hp * dmg / 1000.0


def camps_of(units) -> list[dict]:
    """The creep camps: creeps (neutral hostile) within CAMP_JOIN of one another -> [{"ids", "x", "y"}]."""
    creeps = [u for u in units if u.owner == CREEPS and u.alive and not u.is_structure]
    out: list[dict] = []
    left = list(creeps)
    while left:
        group = [left.pop()]
        grew = True
        while grew:
            grew = False
            for u in list(left):
                if any(math.hypot(u.x - g.x, u.y - g.y) < CAMP_JOIN for g in group):
                    group.append(u)
                    left.remove(u)
                    grew = True
        out.append({"ids": {u.id for u in group}, "x": sum(u.x for u in group) / len(group),
                    "y": sum(u.y for u in group) / len(group),
                    "power": power(group), "level": sum(unit_data().get(u.type, (0,) * 9)[0] for u in group)})
    return out


class Placer:
    """Building spots: the map's buildable cells (war3map.wpm) less the trees, the gold mines, the buildings standing
    and the spots taken; a building goes where its footprint and a cell of room around it are free, nearest a centre
    (behind the hall from its mine), outside the miners' lane."""

    def __init__(self, map_name: str, trees: dict[int, tuple[int, int]], mines: list[tuple[float, float]]):
        from ..data.mapbuild import stock_map_path
        from ..data.terrain import load_terrain
        self.terrain = load_terrain(stock_map_path(map_name))
        self.base = self.terrain.buildable.copy()
        self.trees = trees
        for x, y in mines:
            self._block(self.base, x, y, 16)
        self.taken: list[tuple[float, float, int, float]] = []  # (x, y, cells, until game second): pending spots

    def _block(self, grid: np.ndarray, x: float, y: float, cells: int) -> None:
        r, c = self.terrain.cell(x, y)
        h = cells // 2
        grid[max(r - h, 0):max(r + cells - h, 0), max(c - h, 0):max(c + cells - h, 0)] = False

    def free_grid(self, structures: list[tuple[float, float, str]], now: float) -> np.ndarray:
        g = self.base.copy()
        for x, y in self.trees.values():
            r, c = self.terrain.cell(x, y)
            g[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = False
        fp = footprints()
        for x, y, code in structures:
            self._block(g, x, y, fp.get(code, 8))
        self.taken = [s for s in self.taken if s[3] > now]
        for x, y, cells, _ in self.taken:
            self._block(g, x, y, cells)
        return g

    def find(self, code: str, center: tuple[float, float], structures: list[tuple[float, float, str]], now: float,
             avoid=(), rmax: float = 1400.0, margin: int = 1) -> tuple[float, float] | None:
        """A spot for `code` nearest `center` within rmax, its footprint plus `margin` cells free, outside `avoid`
        ((x, y, radius) circles: the mine's lane) -> its centre in world units (marked taken for 40 s), or None."""
        cells = footprints().get(code, 8)
        g = self.free_grid(structures, now)
        k = cells + 2 * margin
        blocked = (~g).astype(np.int32)
        S = np.zeros((blocked.shape[0] + 1, blocked.shape[1] + 1), np.int32)
        S[1:, 1:] = blocked.cumsum(0).cumsum(1)
        win = S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]  # blocked cells in each k x k window
        rows, cols = np.nonzero(win == 0)
        if not len(rows):
            return None
        # a window at (row, col) covers cells row..row+k-1: the building's centre is at the window's middle
        cx = self.terrain.offset_x + (cols + k / 2) * 32.0
        cy = self.terrain.offset_y + (rows + k / 2) * 32.0
        d = np.hypot(cx - center[0], cy - center[1])
        keep = d <= rmax
        for ax, ay, ar in avoid:
            keep &= np.hypot(cx - ax, cy - ay) > ar
        if not keep.any():
            return None
        i = int(np.argmin(np.where(keep, d, np.inf)))
        x, y = float(cx[i]), float(cy[i])
        self.taken.append((x, y, cells, now + 40.0))
        return x, y


@dataclass
class Memory:
    """An enemy building seen: what, where, when."""
    type: str
    x: float
    y: float
    seen: float


class HeuristicAgent:
    """The scripted player of one side (see the module's docstring)."""

    def __init__(self, side: int, race: str, plan: Plan | None = None, rng: random.Random | None = None,
                 map_name: str = "(2)EchoIsles", log: bool = True, step_seconds: float = 0.5, **options):
        self.side, self.enemy, self.race = side, 1 - side, race
        self.opt = {**OPTIONS, **options}  # (switches for side-by-side variants: heuristic_match.py --variant)
        self.plan = plan or PLANS[race]
        if self.opt.get("hero"):  # (another first hero: a variant)
            import dataclasses
            self.plan = dataclasses.replace(self.plan, heroes=(str(self.opt["hero"]),))
        self.rng = rng or random.Random()
        self.map_name = map_name
        try:  # (the map's rules: echofast's costs are halved, the training games' map)
            from ..data.duelmap import map_rules
            self.cost_scale = float(map_rules(map_name).cost)
        except Exception:  # noqa: BLE001
            self.cost_scale = 1.0
        self.step_s = step_seconds
        self.logging = log
        self.log: list[dict] = []
        self.t = 0
        self.issued = 0
        self.refused = Counter()

    # ---- the game's start -------------------------------------------------------------------------------------
    def begin(self, obs: Observation) -> None:
        me, it = obs.players[self.side], obs.players.get(self.enemy)
        self.start = (float(me.start_x), float(me.start_y))
        self.enemy_start = (float(it.start_x), float(it.start_y)) if it is not None else (-self.start[0], self.start[1])
        self.order_id = dict(obs.orders or {})
        races = {1: "human", 2: "orc", 3: "undead", 4: "nightelf"}
        self.enemy_race = races.get(int(it.race), "") if it is not None else ""
        counts = Counter(d.type for d in (obs.destructables or []))
        self.trees = {d.id: (d.x, d.y) for d in (obs.destructables or [])
                      if d.life > 0 and d.type[1:2] == "T" and counts[d.type] >= 20}
        mines = [u for u in obs.units if u.type_id == MINE_ID]
        self.mines = {u.id: (u.x, u.y) for u in mines}
        self.mine_gold = {u.id: u.resource for u in mines}
        self.main_mine = min(mines, key=lambda u: (u.x - self.start[0]) ** 2 + (u.y - self.start[1]) ** 2).id if mines else None
        try:
            self.placer = Placer(self.map_name, self.trees, [(u.x, u.y) for u in mines])
        except Exception:  # noqa: BLE001 (no terrain: the harness's own spot search around a guess)
            self.placer = None
        dx, dy = self.enemy_start[0] - self.start[0], self.enemy_start[1] - self.start[1]
        n = max(math.hypot(dx, dy), 1.0)
        self.toward = (dx / n, dy / n)
        mx, my = self.mines.get(self.main_mine, (self.start[0], self.start[1] + 600))
        away = (self.start[0] - mx, self.start[1] - my)
        na = max(math.hypot(*away), 1.0)
        self.away = (away[0] / na, away[1] / na)  # from the mine through the hall
        self.rally = (self.start[0] + RALLY_OUT * self.toward[0], self.start[1] + RALLY_OUT * self.toward[1])
        # the natural expansion: the mine nearest the start but its own, and where its hall goes
        others = sorted((m for m in self.mines if m != self.main_mine),
                        key=lambda m: math.hypot(self.mines[m][0] - self.start[0], self.mines[m][1] - self.start[1]))
        self.natural = others[0] if others else None
        if self.natural is not None:
            nx, ny = self.mines[self.natural]
            off = EXPAND_SPOTS.get((int(nx), int(ny)))
            if off is None:
                d = max(math.hypot(self.start[0] - nx, self.start[1] - ny), 1.0)
                off = (750 * (self.start[0] - nx) / d, 750 * (self.start[1] - ny) / d)
            self.natural_spot = (nx + off[0], ny + off[1])
        self.camps = camps_of(obs.units)  # (the map's creep camps: where they stand at the start)
        self.camp_done: set[int] = set()
        self.creep_target: int | None = None
        self.assign: dict[int, str] = {}  # worker -> "gold" / "lumber" / "build"
        self.mine_of: dict[int, int] = {}  # gold worker -> its mine
        self.tree_of: dict[int, int] = {}  # lumber worker -> its tree
        self.building: dict[int, tuple[str, float]] = {}  # worker -> (building it was sent to build, since)
        self.pending: list[list] = []  # [code, worker, x, y, since, gold, lumber]: build orders not yet started
        self.queue: dict[int, list[list]] = {}  # production building -> [[type, started, since], ...] (accepted orders)
        self.research_done: Counter = Counter()  # research -> levels done or under way
        self.memory: dict[int, Memory] = {}  # enemy buildings seen
        self.mode, self.mode_since = "gather", 0
        self.retreat_until = 0
        self.attack_food = self.plan.attack_food
        self.last_order: dict[int, tuple[int, int, int]] = {}  # unit -> (order, target or point key, step)
        self.pulled: dict[int, int] = {}  # unit -> step it rejoins
        self.pulled_at: dict[int, int] = {}
        self.hero_home: set[int] = set()
        self.hp_last: dict[int, int] = {}
        self.hero_ordered: dict[str, float] = {}  # hero type -> when ordered
        self.revive_tried: dict[int, int] = {}
        self.reviving: dict[int, int] = {}  # dead hero -> the step its revival was taken
        self.no_attack_until = 0  # (after a retreat: the army gathers again first)
        self._home_power = 0.0
        self.threat_at = -999  # (the last step enemies were at the base)
        self.in_burrows = False
        self.enemy_army_seen = 0.0
        self.last_log = -999.0
        self.events_log: list[str] = []
        self.counts = Counter()  # what it did (the row's summary)
        self.my_ids: set[int] = set()
        self.search: list[tuple[float, float]] = []  # places to look for the enemy's last buildings
        self.dead_at: dict[int, float] = {}

    # ---- each step --------------------------------------------------------------------------------------------
    def observe(self, obs: Observation, t: int) -> None:
        self.obs, self.t = obs, t
        self.now = obs.game_time
        me = self.side
        units = obs.units
        self.mine_units = [u for u in units if u.owner == me and u.alive]
        self.my_ids = {u.id for u in self.mine_units}
        self.dead_heroes = [u for u in units if u.owner == me and not u.alive and u.is_hero]
        vis = [u for u in units if u.alive and u.owner == self.enemy and u.visible(me)]
        self.enemy_units = [u for u in vis if not u.is_structure]
        self.enemy_buildings = [u for u in vis if u.is_structure]
        self.creeps = [u for u in units if u.alive and u.owner == CREEPS and u.visible(me)]
        present = set()
        for u in units:
            if u.type_id == MINE_ID or u.type_id in (HAUNTED, ENTANGLED):
                present.add(u.id)
                if u.type_id == MINE_ID and u.id in self.mines and u.visible(me):
                    self.mine_gold[u.id] = u.resource
        for m in [m for m in self.mines if m not in present]:  # (a mine gone: dry)
            self._note(f"mine {m} gone")
            del self.mines[m]
        for e in obs.events:
            k = int(e.kind)
            if k == int(EventKind.TREE_DEATH):
                self.trees.pop(e.a, None)
            elif k == int(EventKind.DEATH):
                self.memory.pop(e.a, None)
                self.queue.pop(e.a, None)
                for c in self.camps:
                    c["ids"].discard(e.a)
                if e.a in self.heroes_ids():
                    self.dead_at[e.a] = self.now
                    self._note(f"hero died")
            elif k == int(EventKind.CONSTRUCT_START) and e.a in self.my_ids:
                code = rawcode(e.b)
                for p in self.pending:  # (the build order it fulfils)
                    if p[0] == code:
                        self.pending.remove(p)
                        break
                self.counts[f"built {code}"] += 1
            elif k == int(EventKind.TRAIN_START) and e.a in self.queue:
                for q in self.queue[e.a]:
                    if not q[1] and q[0] == rawcode(e.b):
                        q[1] = True
                        break
            elif k in (int(EventKind.TRAIN_FINISH), int(EventKind.TRAIN_CANCEL), int(EventKind.RESEARCH_FINISH),
                       int(EventKind.RESEARCH_CANCEL), int(EventKind.UPGRADE_FINISH), int(EventKind.UPGRADE_CANCEL)):
                q = self.queue.get(e.a)
                if q:
                    code = rawcode(e.c if k == int(EventKind.TRAIN_FINISH) else e.b)
                    j = next((i for i, x in enumerate(q) if x[0] == code), 0)
                    q.pop(j)
                if k == int(EventKind.RESEARCH_CANCEL):
                    self.research_done[rawcode(e.b)] -= 1
        for u in self.enemy_buildings:  # (enemy buildings seen: remembered where they stand)
            self.memory[u.id] = Memory(u.type, u.x, u.y, self.now)
        if self.memory and self.t % 4 == 0:  # (a remembered one no longer there where the player sees: gone)
            seen = {u.id for u in self.enemy_buildings}
            eyes = np.array([(u.x, u.y) for u in self.mine_units if not u.flags & CONSTRUCTING], float).reshape(-1, 2)
            if len(eyes):
                for i, m in list(self.memory.items()):
                    if i not in seen and (np.hypot(eyes[:, 0] - m.x, eyes[:, 1] - m.y) < 450).any():
                        del self.memory[i]
        p = obs.players.get(me)
        self.gold, self.lumber = (p.gold, p.lumber) if p else (0, 0)
        self.food, self.cap = (p.food_used, min(p.food_cap, 100)) if p else (0, 0)
        self.pending = [x for x in self.pending if self.now - x[4] < 45.0 and x[1] in self.my_ids]
        for b in [b for b in self.queue if b not in self.my_ids]:
            del self.queue[b]
        for b, qs in self.queue.items():  # (an entry no event ended: gone after three times its build time; a missed
            # finish had blocked a barracks for good, 26 grunts in 11 minutes from three of them)
            keep = [q for k, q in enumerate(qs) if self.now - q[2] < QUEUE_EXPIRE.get(q[0][:1], 100.0) * (1 + k)]
            if len(keep) < len(qs):
                self.counts["queue expired"] += len(qs) - len(keep)
                qs[:] = keep

    def heroes_ids(self) -> set[int]:
        return {u.id for u in getattr(self, "heroes", ())}

    def accepted(self, cmds: list[Command], results: list[bool]) -> None:
        """The game's answers to the last step's orders: a train or research order accepted queues; a build refused
        frees its spot and worker."""
        for c, ok in zip(cmds, results):
            if not ok:
                o = getattr(c, "building", None) or getattr(c, "order", 0)
                self.refused[f"{type(c).__name__}:{rawcode(o) if o >= TYPE_CODE else o}"] += 1
            if isinstance(c, ImmediateOrder) and c.order >= TYPE_CODE:
                code = rawcode(c.order)
                if ok:
                    self.queue.setdefault(c.unit, []).append([code, False, self.now])
                elif code[:1] == "R":
                    self.research_done[code] -= 1
                elif code[:1].isupper():  # (a hero refused: again soon)
                    self.hero_ordered.pop(code, None)
            elif isinstance(c, Build) and not ok:
                self.pending = [x for x in self.pending if x[1] != c.unit]
                self.building.pop(c.unit, None)
                self.assign.pop(c.unit, None)
            elif isinstance(c, TargetOrder) and c.order == REVIVE and ok:
                self.counts["revivals"] += 1
                self.reviving[c.target] = self.t  # (under way: not again for REVIVE_WAIT)
            elif isinstance(c, TargetOrder) and c.order == HARVEST and not ok:
                self.assign.pop(c.unit, None)  # (again next step: another mine or a tree)
                self.mine_of.pop(c.unit, None)

    def act(self, obs: Observation, t: int) -> list[Command]:
        self.observe(obs, t)
        return self.commands()

    def commands(self) -> list[Command]:
        out: list[Command] = []
        self.budget = [self.gold - sum(p[5] for p in self.pending), self.lumber - sum(p[6] for p in self.pending)]
        self.cmd_units: set[int] = set()
        self._classify()
        out += self._heroes()  # (first: a revival's money before anything else's)
        out += self._economy()
        out += self._army()
        if self.logging and self.now - self.last_log >= 30.0:
            self.last_log = self.now
            self._snapshot()
        self.issued += len(out)
        return out

    # ---- who is who -------------------------------------------------------------------------------------------
    def _classify(self) -> None:
        self.workers = [u for u in self.mine_units if u.is_worker and not u.is_structure]
        self.structures = [u for u in self.mine_units if u.is_structure]
        self.done = [u for u in self.structures if not u.flags & CONSTRUCTING]
        self.army = [u for u in self.mine_units if not u.is_structure and not u.is_worker]
        self.heroes = [u for u in self.army if u.is_hero]
        halls = [u for u in self.done if u.type in HALL_TYPES]
        self.halls = halls
        self.hall = min(halls, key=lambda u: (u.x - self.start[0]) ** 2 + (u.y - self.start[1]) ** 2) if halls else None
        self.home = (self.hall.x, self.hall.y) if self.hall else self.start
        self.have = Counter(u.type for u in self.structures) + Counter(p[0] for p in self.pending)
        self.have_done = Counter(u.type for u in self.done)

    # ---- money ------------------------------------------------------------------------------------------------
    def _pay(self, gold: int, lumber: int) -> bool:
        """Pays from the step's budget if it can (only the resources the order needs are checked: money kept for a
        building short of lumber had made the lumber budget negative and refused every grunt, which costs none)."""
        if self.opt["strict_budget"]:  # (the old check: a negative budget refuses everything)
            if gold > self.budget[0] or lumber > self.budget[1]:
                return False
        elif (gold > 0 and gold > self.budget[0]) or (lumber > 0 and lumber > self.budget[1]):
            return False
        self.budget[0] -= gold
        self.budget[1] -= lumber
        return True

    def _keep(self, gold: int, lumber: int) -> None:
        """Money kept for something due that the player can't pay for yet: what comes after waits."""
        self.budget[0] -= gold
        self.budget[1] -= lumber

    def _cost(self, code: str, level: int = 0) -> tuple[int, int]:
        """What `code` costs (gold, lumber) under the map's rules (costs.order_costs' scaling)."""
        if code[:1] == "R":
            g, l, gm, lm = research_costs().get(code, (100, 50, 0, 0))
            g, l = g + gm * level, l + lm * level
        else:
            d = unit_data().get(code)
            if d is None:
                return 0, 0
            base = upgraded_from().get(code)
            if base and base in unit_data():
                b = unit_data()[base]
                return self._scaled(d[1]) - self._scaled(b[1]), self._scaled(d[2]) - self._scaled(b[2])
            g, l = d[1], d[2]
        return self._scaled(g), self._scaled(l)

    def _scaled(self, v: int) -> int:
        return v if self.cost_scale == 1.0 or v <= 0 else max(1, round(v * self.cost_scale))

    def _food(self, code: str) -> int:
        d = unit_data().get(code)
        return d[3] if d else 0

    # ---- economy -----------------------------------------------------------------------------------------------
    def _economy(self) -> list[Command]:
        out: list[Command] = []
        out += self._supply()
        out += self._expand()
        out += self._build_order()
        out += self._workers()
        return out

    def _lumber_target(self) -> int:
        n = 0
        for at, k in self.plan.lumber_workers:
            if self.now >= at:
                n = k
        return n

    def _working_mines(self) -> list[int]:
        """The gold mines its workers mine: those with a finished hall of the player's near them (undead: its haunted
        mines, night elves: its entangled ones), the main one first."""
        if self.race in ("undead", "nightelf"):
            kind = HAUNTED if self.race == "undead" else ENTANGLED
            own = [u for u in self.obs.units if u.type_id == kind and u.owner == self.side and u.alive
                   and not u.flags & CONSTRUCTING]
            return [u.id for u in own]
        out = []
        for m, (x, y) in self.mines.items():
            if any(math.hypot(h.x - x, h.y - y) < 1100 for h in self.halls):
                out.append(m)
        out.sort(key=lambda m: (m != self.main_mine, m))
        return out

    def _workers(self) -> list[Command]:
        out: list[Command] = []
        plan = self.plan
        lumber_t = self._lumber_target()
        for w in list(self.assign):  # (workers gone)
            if w not in self.my_ids:
                self.assign.pop(w, None)
                self.tree_of.pop(w, None)
                self.mine_of.pop(w, None)
        for w, (code, since) in list(self.building.items()):  # a builder whose building started or never did: free
            u = next((x for x in self.workers if x.id == w), None)
            if u is None or self.now - since > 70.0 or (u.order < TYPE_CODE and self.now - since > 3.0
                                                         and not any(p[1] == w for p in self.pending)):
                self.building.pop(w, None)
                if self.assign.get(w) == "build":
                    self.assign.pop(w, None)
        mines = self._working_mines()
        for m in list(self.mine_of.values()):  # (a mine gone or not worked any more: its workers again)
            if m not in mines:
                for w in [w for w, mm in self.mine_of.items() if mm == m]:
                    self.mine_of.pop(w, None)
                    if self.assign.get(w) == "gold":
                        self.assign.pop(w, None)
        on_mine = Counter(self.mine_of.values())
        n_lumber = sum(1 for r in self.assign.values() if r == "lumber")
        gold_cap = plan.gold_workers * len(mines)
        n_gold = sum(on_mine[m] for m in mines)
        for u in self.workers:
            if u.id in self.building or u.flags & LOADED:
                continue
            if u.id not in self.assign and u.order in HARVESTING:  # (the melee start's miners, new workers: the
                self.assign[u.id] = "gold"  # hall's rally on its mine)
                m = mines[0] if mines else None
                if m is not None:
                    self.mine_of[u.id] = m
                    on_mine[m] += 1
                    n_gold += 1
                continue
            boarding = u.order == BOARD and not self.in_burrows
            if u.order != 0 and u.id in self.assign and not boarding:
                continue
            if u.order != 0 and u.order not in (MOVE, ATTACK, STOP, HOLD, SMART) and not boarding:
                continue
            if u.flags & HIDDEN:
                continue
            role = self.assign.get(u.id)
            if role == "gold":
                n_gold -= 1
                on_mine[self.mine_of.get(u.id)] -= 1
            elif role == "lumber":
                n_lumber -= 1
            if self.race == "undead" or not self._can_chop():
                role = "gold"
            elif n_gold < gold_cap and (n_lumber >= min(lumber_t, 2) or n_gold < 3):
                role = "gold"
            elif n_lumber < lumber_t or not mines:
                role = "lumber"
            else:
                role = "gold" if mines else "lumber"
            cmd = self._send(u, role, mines, on_mine)
            if cmd is not None:
                out.append(cmd)
                self.assign[u.id] = role
                if role == "gold":
                    n_gold += 1
                    on_mine[self.mine_of[u.id]] += 1
                else:
                    n_lumber += 1
        # rebalance a step: a mine with too many (another's short), lumber short, or too many on lumber
        if self.t % 4 == 0:
            over = [m for m in mines if on_mine[m] > plan.gold_workers]
            under = [m for m in mines if on_mine[m] < plan.gold_workers]
            if over and under:
                w = next((u for u in self.workers if self.mine_of.get(u.id) == over[0] and not u.flags & HIDDEN
                          and u.id not in self.building), None)
                if w is not None:
                    self.mine_of[w.id] = under[0]
                    self.cmd_units.add(w.id)
                    out.append(TargetOrder(w.id, HARVEST, under[0]))
            elif n_gold > gold_cap and n_lumber < lumber_t and self._can_chop():
                w = next((u for u in self.workers if self.assign.get(u.id) == "gold" and not u.flags & HIDDEN
                          and u.id not in self.building), None)
                if w is not None and (cmd := self._send(w, "lumber", mines, on_mine)) is not None:
                    out.append(cmd)
                    self.assign[w.id] = "lumber"
                    self.mine_of.pop(w.id, None)
            elif n_lumber > lumber_t + 1 and n_gold < gold_cap and mines:
                w = next((u for u in self.workers if self.assign.get(u.id) == "lumber" and u.id not in self.building), None)
                if w is not None and (cmd := self._send(w, "gold", mines, on_mine)) is not None:
                    out.append(cmd)
                    self.assign[w.id] = "gold"
        # more workers: at every hall near a mine, up to the plan's count
        want = gold_cap + lumber_t + 1 + (1 if len(mines) > 1 else 0)
        if self.race == "undead":
            want = gold_cap + 1
        if len(self.workers) < want:
            for h in self.halls:
                if self._queued(h) < 1 and len(self.workers) < want:
                    out += self._train(h, plan.worker)
        if self.race == "undead":
            out += self._ghoul_lumber()
        return out

    def _can_chop(self) -> bool:
        return self.race != "undead"

    def _send(self, u, role: str, mines: list[int], on_mine: Counter) -> Command | None:
        if role == "gold":
            if not mines:
                return None
            m = min(mines, key=lambda m: (on_mine[m] >= self.plan.gold_workers, on_mine[m],
                                          math.hypot(self.mines.get(m, (u.x, u.y))[0] - u.x, self.mines.get(m, (u.x, u.y))[1] - u.y)))
            self.mine_of[u.id] = m
            self.tree_of.pop(u.id, None)
            self.cmd_units.add(u.id)
            return TargetOrder(u.id, HARVEST, m)
        tree = self._tree_for(u)
        if tree is None:
            return None
        self.tree_of[u.id] = tree
        self.mine_of.pop(u.id, None)
        self.cmd_units.add(u.id)
        return TargetDestructable(u.id, HARVEST, tree)

    def _tree_for(self, u) -> int | None:
        """A tree near the hall (the nearest few, spread over the workers), not between the hall and the mine."""
        if not self.trees:
            return None
        hx, hy = self.home
        if getattr(self, "_tree_cache_t", -99) < self.t - 20 or not getattr(self, "_tree_ids", None) \
                or any(i not in self.trees for i in self._tree_ids):
            ids = np.fromiter(self.trees.keys(), np.int64, len(self.trees))
            xy = np.array([self.trees[i] for i in ids.tolist()], float)
            d = np.hypot(xy[:, 0] - hx, xy[:, 1] - hy)
            m = self.mines.get(self.main_mine)
            if m is not None:  # (not by the mine: the miners' path)
                d = d + 400.0 * (np.hypot(xy[:, 0] - m[0], xy[:, 1] - m[1]) < 500)
            order = np.argsort(d)[:12]
            self._tree_ids, self._tree_cache_t = ids[order].tolist(), self.t
        used = Counter(self.tree_of.values())
        return min(self._tree_ids, key=lambda i: used[i])

    def _ghoul_lumber(self) -> list[Command]:
        """Undead: a few ghouls chop (its only lumber gatherers), at home while the army fights elsewhere."""
        out = []
        ghouls = [u for u in self.army if u.type == "ugho"]
        chopping = [u for u in ghouls if self.assign.get(u.id) == "lumber"]
        want = 3 if self.now < 400 else 4
        for u in chopping:
            if u.order == 0:  # (idle again: back to its tree)
                tree = self._tree_for(u)
                if tree is not None:
                    out.append(TargetDestructable(u.id, HARVEST, tree))
                    self.cmd_units.add(u.id)
        if len(chopping) < want and self.have_done.get("ugrv", 0) + self.have_done.get("unpl", 0) > 0:
            free = [u for u in ghouls if u.id not in self.assign and self.mode != "defend"]
            for u in free[:want - len(chopping)]:
                tree = self._tree_for(u)
                if tree is not None:
                    self.assign[u.id] = "lumber"
                    self.tree_of[u.id] = tree
                    out.append(TargetDestructable(u.id, HARVEST, tree))
                    self.cmd_units.add(u.id)
        return out

    def _queued(self, b) -> int:
        return len(self.queue.get(b.id, ()))

    def _train(self, b, code: str) -> list[Command]:
        g, l = self._cost(code)
        food = self._food(code)
        if food and self.food + food + self._waiting_food() > self.cap:
            return []
        if food and code != self.plan.worker and self.food + food + self._waiting_food() > FOOD_CAP:
            return []
        if not self._pay(g, l):
            return []
        self.cmd_units.add(b.id)
        return [ImmediateOrder(b.id, fourcc(code))]

    def _waiting_food(self) -> int:
        return sum(self._food(q[0]) for qs in self.queue.values() for q in qs if not q[1])

    def _supply(self) -> list[Command]:
        plan = self.plan
        if self.cap >= 100:
            return []
        building = sum(1 for u in self.structures if u.type == plan.supply and u.flags & CONSTRUCTING)
        building += sum(1 for p in self.pending if p[0] == plan.supply)
        building += sum(1 for u in self.structures if u.type in HALL_TYPES and u.flags & CONSTRUCTING)
        prod = sum(1 for u in self.done if self._makes_army(u.type)) + len(self.halls)
        # (what the production buildings use while a supply building goes up: ~3 food each per 50 s)
        margin = 3 + 3 * prod + (3 if self.now > 240 else 0)
        if not self.heroes and not self.dead_heroes and self.have.get(self.plan.altar):  # (the first hero's 5 food)
            margin += 5
        if self.cap + building * plan.supply_food - self.food > margin or self.cap + building * plan.supply_food >= 100 \
                or self.cap + building * plan.supply_food >= FOOD_CAP + 6:
            return []
        if self.now < 10 or building >= 1 + prod // 3:
            return []
        c = self._build(plan.supply)
        if not c:
            self._keep(*self._cost(plan.supply))
        return c

    def _makes_army(self, code: str) -> bool:
        from .techtree import tree
        return any(code in tree().makers.get(u, ()) for u, _ in self.plan.army)

    def _build_order(self) -> list[Command]:
        out: list[Command] = []
        rush = self.opt["rush_caution"] and self.enemy_race in ("undead", "orc")
        for at, food, code, n in (self.plan.rush_builds if rush and self.plan.rush_builds else self.plan.builds):
            if self.now < at or self.food < food:
                continue
            if self.have[code] >= n:
                continue
            if not self._tech_ok(code):
                continue
            c = self._build(code)
            out += c
            if not c:  # (not affordable yet: its money kept from the army)
                self._keep(*self._cost(code))
            break
        towers = self.plan.rush_towers if rush and self.plan.rush_towers else self.plan.towers
        for at, code, n in towers if self.opt["towers_built"] else ():
            if self.now >= at and self.have[code] < n and self._tech_ok(code):
                hx, hy = self.home  # (between the hall and the rally point: where the waves come in)
                out += self._build(code, near=((hx + self.rally[0]) / 2, (hy + self.rally[1]) / 2))
                break
        return out

    def _expand(self) -> list[Command]:
        """The natural's hall (human, orc): from Plan.expand_at with its camp cleared, or when the main mine runs low;
        the worker nearest builds it where people do."""
        if self.race not in ("human", "orc") or self.natural is None or self.natural not in self.mines:
            return []
        hall = self.plan.hall
        if any(math.hypot(h.x - self.natural_spot[0], h.y - self.natural_spot[1]) < 900
               for h in self.structures if h.type in HALL_TYPES):
            return []
        if any(p[0] == hall for p in self.pending):
            return []
        low = self.mine_gold.get(self.main_mine, 0) < MINE_LOW or self.main_mine not in self.mines
        if self.now < self._expand_at() and not low:
            return []
        camp = self._camp_at(*self.mines[self.natural])
        if camp is not None and camp["ids"]:
            return []  # (its camp first: the army's creep mode)
        if any(math.hypot(e.x - self.natural_spot[0], e.y - self.natural_spot[1]) < 1200 for e in self.enemy_units):
            return []
        c = self._build(hall, near=self.natural_spot, rmax=500.0, avoid_lane=False)
        if c:
            self._note("expanding")
        else:
            self._keep(*self._cost(hall))
        return c

    def _expand_at(self) -> float:
        """The natural from Plan.expand_at; against the races whose first waves come at ~4:30 (the learner's undead
        and orc), from 9 minutes: the natural is ~6,000 from home, and the army at its camp had met their waves."""
        return self.plan.expand_at + (120.0 if self.opt["rush_caution"] and self.enemy_race in ("undead", "orc") else 0.0)

    def _camp_at(self, x: float, y: float) -> dict | None:
        near = [c for c in self.camps if math.hypot(c["x"] - x, c["y"] - y) < 900]
        return max(near, key=lambda c: len(c["ids"])) if near else None

    def _tech_ok(self, code: str) -> bool:
        from . import techtree
        have = techtree.counts_as(set(self.have_done) | {c for c, n in self.research_done.items() if n > 0}, self.race)
        return not techtree.missing(code, self.race, have, 0)

    def _build(self, code: str, near: tuple[float, float] | None = None, rmax: float = 1400.0,
               avoid_lane: bool = True) -> list[Command]:
        g, l = self._cost(code)
        if (g > 0 and g > self.budget[0]) or (l > 0 and l > self.budget[1]):
            return []
        w = self._builder(near)
        if w is None:
            return []
        spot = self._spot(code, near, rmax, avoid_lane)
        if spot is None:
            return []
        self._pay(g, l)
        self.pending.append([code, w.id, spot[0], spot[1], self.now, g, l])
        self.building[w.id] = (code, self.now)
        self.assign[w.id] = "build"
        self.tree_of.pop(w.id, None)
        self.mine_of.pop(w.id, None)
        self.cmd_units.add(w.id)
        return [Build(w.id, fourcc(code), spot[0], spot[1])]

    def _builder(self, near=None):
        """The worker to build: an idle one, else a gold worker outside the mine (a new one replaces it: workers count
        their builders), else a lumber one, nearest the site. (Lumber workers first had left nobody chopping while the
        barracks went up: 10-30 lumber for two minutes, gold piling up.)"""
        hx, hy = near or self.home
        cands = [u for u in self.workers if u.id not in self.building and not u.flags & (HIDDEN | LOADED)
                 and u.id not in self.cmd_units]
        if not cands:
            return None
        idle = [u for u in cands if u.order == 0]
        gold = [u for u in cands if self.assign.get(u.id) == "gold"]
        pool = idle or gold or cands
        return min(pool, key=lambda u: (u.x - hx) ** 2 + (u.y - hy) ** 2)

    def _spot(self, code: str, near=None, rmax: float = 1400.0, avoid_lane: bool = True):
        hx, hy = self.home
        if near is None:  # (behind the hall from the mine's side, a little toward the enemy)
            near = (hx + 450 * self.away[0] + 150 * self.toward[0], hy + 450 * self.away[1] + 150 * self.toward[1])
        if self.placer is None:
            a = self.rng.uniform(0, 2 * math.pi)
            return near[0] + 200 * math.cos(a), near[1] + 200 * math.sin(a)
        structs = [(u.x, u.y, u.type) for u in self.obs.units if u.is_structure and u.alive]
        avoid = []
        for mid in self._working_mines() if avoid_lane else ():  # (the miners' lane: between a hall and its mine)
            m = self.mines.get(mid)
            h = min(self.halls, key=lambda h: math.hypot(h.x - m[0], h.y - m[1])) if m and self.halls else None
            if m is None or h is None:
                continue
            for f in (0.25, 0.5, 0.75):
                avoid.append((h.x + f * (m[0] - h.x), h.y + f * (m[1] - h.y), 300.0))
            avoid.append((m[0], m[1], 420.0))
        return self.placer.find(code, near, structs, self.now, avoid, rmax=rmax)

    # ---- heroes ------------------------------------------------------------------------------------------------
    def _heroes(self) -> list[Command]:
        out: list[Command] = []
        altar = next((u for u in self.done if u.type == self.plan.altar), None)
        alive = {u.type for u in self.heroes}
        dead = {u.type for u in self.dead_heroes}
        if altar is not None:
            for h in self.dead_heroes:  # a revival first, its money kept until the altar takes it
                cost = int(425 * min(0.4 + 0.1 * max(h.hero_level - 1, 0), 4.0)) + 5
                if self.t - self.reviving.get(h.id, -REVIVE_WAIT) < REVIVE_WAIT:
                    continue
                if self.now - self.dead_at.get(h.id, -99) < 8:
                    continue
                if self.budget[0] >= cost and self.t - self.revive_tried.get(h.id, -99) >= 6:
                    self.revive_tried[h.id] = self.t
                    self._pay(cost, 0)
                    self.cmd_units.add(altar.id)
                    out.append(TargetOrder(altar.id, REVIVE, h.id))
                else:
                    self._keep(cost, 0)
                break
            for code in self.plan.heroes[:1 + self._extra_heroes()]:
                if code in alive or code in dead:
                    continue
                if self.now - self.hero_ordered.get(code, -99) < 30 or self._queued(altar):
                    continue
                if self.food + self._food(code) > self.cap:  # (a hero needs its food: supply first)
                    continue
                first = not self.heroes and not self.dead_heroes  # (the first hero is free)
                if first or self._pay(*self._cost(code)):
                    self.hero_ordered[code] = self.now
                    self.cmd_units.add(altar.id)
                    out.append(ImmediateOrder(altar.id, fourcc(code)))
                    break
        from .herodata import next_skill
        for h in self.heroes:  # skill points
            if h.skill_points <= 0:
                continue
            levels = [lv for lv, _ in h.abilities] + [0] * 4
            code = self._next_skill(h, levels) or next_skill(h.type, levels[:4], h.hero_level)
            if code is not None:
                out.append(LearnSkill(h.id, fourcc(code)))
        return out

    def _extra_heroes(self) -> int:
        return 0

    def _next_skill(self, h, levels) -> str | None:
        from .herodata import hero_table
        slots = hero_table().get(h.type_id, ())
        index = {a.code: k for k, a in enumerate(slots) if a is not None}
        need: Counter = Counter()
        for code in SKILLS.get(h.type, ()):
            need[code] += 1
            k = index.get(code)
            if k is None:
                continue
            a = slots[k]
            if levels[k] < need[code] and levels[k] < a.levels and h.hero_level >= a.hero_level_for(levels[k] + 1):
                return code
        return None

    # ---- the army ----------------------------------------------------------------------------------------------
    def _army(self) -> list[Command]:
        out: list[Command] = []
        out += self._research()
        out += self._produce()
        units = [u for u in self.army if self.assign.get(u.id) != "lumber"]
        threats = self._threats()
        if threats:
            self.threat_at = self.t
        out += self._battlestations(threats)
        if not units:
            return out
        center = self._center(units)
        near_enemies = [e for e in self.enemy_units if math.hypot(e.x - center[0], e.y - center[1]) < FIGHT_RADIUS]
        near_towers = [b for b in self.enemy_buildings if b.type_id in TOWERS_ATTACK
                       and math.hypot(b.x - center[0], b.y - center[1]) < FIGHT_RADIUS]
        mine_power = power(units, center, workers=False)
        self._home_power = power([u for u in units if math.hypot(u.x - self.home[0], u.y - self.home[1]) < DEFEND_RADIUS + 800])
        their_power = power(near_enemies + near_towers)
        army_food = sum(self._food(u.type) for u in units if not u.is_hero and not u.flags & SUMMONED)
        at_home = math.hypot(center[0] - self.home[0], center[1] - self.home[1]) < DEFEND_RADIUS
        if their_power > 0:
            self.enemy_army_seen = max(self.enemy_army_seen, their_power)
        prev = self.mode
        if threats and (self.mode != "attack" or self._base_in_danger(threats, mine_power)):
            self.mode = "defend"
        elif self.mode == "defend" and not threats:
            self.mode = "gather"
            if army_food >= 30 and self.enemy_army_seen > 0 and mine_power > 1.5 * self.enemy_army_seen:
                self.mode = "attack"  # (its army died at the base: on to theirs)
                self._note("counter-attack")
        if self.mode == "creep" and (self._creep_camp() is None or not self.heroes or
                                     any(h.hp < 0.4 * h.max_hp for h in self.heroes)):
            self.mode = "gather"
        ready = self.t >= self.no_attack_until and (
            army_food >= self.attack_food or (self.now >= self.plan.attack_at and army_food >= self.attack_food * 2 / 3)
            or self.food >= FOOD_CAP - 4)
        natural = self._natural_camp_due(units, mine_power)
        if self.mode == "creep" and natural is not None and self.creep_target == self.camps.index(natural):
            pass  # (the natural's camp first: the expansion waits for it)
        elif self.mode in ("gather", "creep") and ready:
            self.mode = "attack"
            self.creep_target = None
        elif self.mode == "gather" and self.t >= self.no_attack_until and self._creep_camp(units, mine_power) is not None:
            self.mode = "creep"
        if self.mode == "attack" and army_food < max(10, self.attack_food / 4) and not near_enemies:
            self.mode = "gather"
        retreat = (their_power > 0 and mine_power < RETREAT_RATIO * their_power and not at_home
                   and self.mode in ("attack", "gather", "creep"))
        if retreat and self.t >= self.retreat_until:
            self.retreat_until = self.t + RETREAT_STEPS
            self.no_attack_until = self.t + REGROUP_STEPS
            self.attack_food = min(max(self.attack_food + 8, army_food + 10), FOOD_CAP - 6)
            self.creep_target = None
            self.mode = "gather"
            self._note(f"retreat: {mine_power:.0f} vs {their_power:.0f}")
        if self.mode != prev:
            self.mode_since = self.t
            self._note(f"mode {prev} -> {self.mode} (army food {army_food}, power {mine_power:.0f} vs {their_power:.0f})")
        retreating = self.t < self.retreat_until
        goal = self._goal(center, threats)
        out += self._move_army(units, center, goal, retreating)
        return out

    def _creep_camp(self, units=None, mine_power: float = 0.0) -> dict | None:
        """The camp to creep: the current one while it stands; else (from Plan.creep_from to creep_until, with a
        healthy hero and Plan.creep_units more) the nearest standing camp within CREEP_REACH of home that the army
        is well stronger than, the natural's first once the expansion is due."""
        if self.creep_target is not None:
            c = self.camps[self.creep_target]
            if c["ids"]:
                return c
            self.creep_target = None
            self.counts["camps cleared"] += 1
            self._note("camp cleared")
        if units is None:
            return None
        if not self.heroes or any(h.hp < 0.7 * h.max_hp for h in self.heroes):
            return None
        if len(units) - len(self.heroes) < self.plan.creep_units:
            return None
        hx, hy = self.home
        natural = self._natural_camp_due(units, mine_power)
        if natural is not None:
            self.creep_target = self.camps.index(natural)
            return natural
        if not self.plan.creep_from <= self.now <= self.plan.creep_until or self.t - self.threat_at < 60:
            return None
        reach = CREEP_REACH
        if self.opt["rush_caution"] and self.enemy_race in ("undead", "orc") and self.now < 420:  # (first waves ~4:30)
            reach = 2600.0 if self.now < 200 else 0.0
        cands = [(math.hypot(c["x"] - hx, c["y"] - hy), i) for i, c in enumerate(self.camps)
                 if c["ids"] and math.hypot(c["x"] - hx, c["y"] - hy) < reach and mine_power > 2.5 * c["power"]
                 and math.hypot(c["x"] - self.enemy_start[0], c["y"] - self.enemy_start[1]) > 3500]
        if not cands:
            return None
        self.creep_target = min(cands)[1]
        return self.camps[self.creep_target]

    def _natural_camp_due(self, units, mine_power: float) -> dict | None:
        """The natural's camp while it stands, once the expansion is near due and the army is well stronger."""
        if self.race not in ("human", "orc") or self.natural not in self.mines:
            return None
        natural = self._camp_at(*self.mines[self.natural])
        if self.opt["rush_caution"] and self.t - self.threat_at < 120:
            return None
        early = 120.0 if self.enemy_race not in ("undead", "orc") or not self.opt["rush_caution"] else 0.0
        due = self.now >= self._expand_at() - early or self.mine_gold.get(self.main_mine, 0) < MINE_LOW + 3000
        if natural is not None and natural["ids"] and due and mine_power > 2.0 * natural["power"]:
            return natural
        return None

    def _threats(self) -> list:
        """Visible enemy units near the base (a hall, or another building of the player's)."""
        bs = [(u.x, u.y) for u in self.structures]
        hs = [(u.x, u.y) for u in self.halls] or [self.home]
        out = []
        for e in self.enemy_units:
            if e.flags & FLYING and not e.is_hero and unit_data().get(e.type, (0,) * 9)[6] <= 0:
                continue
            if any(math.hypot(e.x - x, e.y - y) < DEFEND_RADIUS for x, y in hs) \
                    or any(math.hypot(e.x - x, e.y - y) < NEAR_BUILDING for x, y in bs):
                out.append(e)
        return out

    def _base_in_danger(self, threats, mine_power: float) -> bool:
        return power(threats) > 0.5 * max(mine_power, 1.0)

    def _center(self, units) -> tuple[float, float]:
        hero = next((u for u in units if u.is_hero and u.id not in self.hero_home), None)
        xs = np.array([u.x for u in units], float)
        ys = np.array([u.y for u in units], float)
        mx, my = float(np.median(xs)), float(np.median(ys))
        if hero is not None and math.hypot(hero.x - mx, hero.y - my) < 900:
            return (hero.x + mx) / 2, (hero.y + my) / 2
        return mx, my

    def _goal(self, center, threats) -> tuple[float, float]:
        if self.mode == "defend" and threats:
            tx, ty = float(np.median([e.x for e in threats])), float(np.median([e.y for e in threats]))
            if self.opt["hold_home"] and power(threats) > 0.7 * max(self._home_power, 1.0):
                # (a wave as strong as the army: met near the hall, its towers and burrows, not where it stands; 21 of
                # 29 losses before 14 minutes came at home without an attack of ours)
                hx, hy = self.home
                return hx + 0.4 * (tx - hx), hy + 0.4 * (ty - hy)
            return tx, ty
        if self.mode == "creep":
            c = self._creep_camp()
            if c is not None:
                return c["x"], c["y"]
        if self.mode == "attack":
            if self.enemy_units:  # (its army seen near: there)
                near = min(self.enemy_units, key=lambda e: (e.x - center[0]) ** 2 + (e.y - center[1]) ** 2)
                if math.hypot(near.x - center[0], near.y - center[1]) < 1500:
                    return near.x, near.y
            if self.memory:
                towers = [m for m in self.memory.values() if fourcc(m.type) in TOWERS_ATTACK]

                def cost(m):  # (distance, and the known towers that cover it)
                    n = sum(1 for tw in towers if tw is not m and math.hypot(tw.x - m.x, tw.y - m.y) < 800)
                    return math.hypot(m.x - center[0], m.y - center[1]) + TOWER_COST * n
                m = min(self.memory.values(), key=cost)
                return m.x, m.y
            if math.hypot(center[0] - self.enemy_start[0], center[1] - self.enemy_start[1]) > 700 and not self.search:
                return self.enemy_start
            # nothing left where it knew: the other mines in turn
            if not self.search:
                self.search = sorted(self.mines.values(), key=lambda p: math.hypot(p[0] - center[0], p[1] - center[1]))
                self.search = [p for p in self.search if math.hypot(p[0] - self.home[0], p[1] - self.home[1]) > 1500]
                self.search.append(self.enemy_start)
            if self.search and math.hypot(center[0] - self.search[0][0], center[1] - self.search[0][1]) < 600:
                self.search.pop(0)
            return self.search[0] if self.search else self.enemy_start
        return self.rally

    def _move_army(self, units, center, goal, retreating: bool) -> list[Command]:
        out: list[Command] = []
        fighting = self._fight_targets(units, retreating)
        gx, gy = goal
        dx, dy = gx - center[0], gy - center[1]
        d = math.hypot(dx, dy)
        for u in units:
            if u.id in fighting:  # (first: its fight's order marked the unit as ordered already; checked after that
                out += fighting[u.id]  # mark, every focus-fire order and spell had been dropped until 2026-10-25)
                continue
            if u.id in self.cmd_units:
                continue
            if u.id in self.hero_home:
                continue
            if retreating:
                out += self._order_point(u, MOVE, *self.rally, every=8)
                continue
            if self.opt["hero_behind"] and u.is_hero and self.mode in ("attack", "defend", "creep") and len(units) > 3 and d > 300:
                # (the hero, faster than the army, had walked in first and died: it follows the centre, behind it)
                bx, by = center[0] - 250 * dx / d, center[1] - 250 * dy / d
                if math.hypot(u.x - bx, u.y - by) > 300:
                    out += self._order_point(u, MOVE, bx, by, every=4)
                continue
            if self.mode in ("attack", "defend", "creep"):
                # the leaders wait for the rest: a unit well ahead of the group's centre toward the goal goes back to it
                ahead = d > 800 and ((u.x - center[0]) * dx + (u.y - center[1]) * dy) / d > 450 \
                    and math.hypot(u.x - center[0], u.y - center[1]) > 500
                if ahead or math.hypot(u.x - center[0], u.y - center[1]) > 1600 and d > 800:
                    out += self._order_point(u, ATTACK, center[0], center[1], every=6)
                else:
                    out += self._order_point(u, ATTACK, gx, gy, every=20)
            else:  # gather at the rally point
                if math.hypot(u.x - gx, u.y - gy) > 450:
                    out += self._order_point(u, ATTACK, gx, gy, every=20)
        return out

    def _order_point(self, u, order: int, x: float, y: float, every: int = 6) -> list[Command]:
        """A point order, unless the unit has it already (the same order to about the same place, given within
        `every` steps, and the unit still carrying it out)."""
        last = self.last_order.get(u.id)
        key = (int(x) // 128) * 1000 + int(y) // 128
        if last is not None and last[0] == order and last[1] == key and u.order == order and self.t - last[2] < every * 4:
            return []
        if last is not None and last[0] == order and abs(last[1] - key) <= 1002 and u.order == order and self.t - last[2] < every:
            return []
        self.last_order[u.id] = (order, key, self.t)
        self.cmd_units.add(u.id)
        return [PointOrder(u.id, order, x, y)]

    def _order_target(self, u, order: int, target: int) -> list[Command]:
        last = self.last_order.get(u.id)
        if last is not None and last[0] == order and last[1] == -target and u.order == order and self.t - last[2] < 12:
            return []
        self.last_order[u.id] = (order, -target, self.t)
        self.cmd_units.add(u.id)
        return [TargetOrder(u.id, order, target)]

    def _fight_targets(self, units, retreating: bool) -> dict[int, list[Command]]:
        """Each unit near enemies: a spell (heroes), a step back (hurt melee units, a hurt hero home), or the enemy
        in reach with the least hit points left after what the others aim at it."""
        out: dict[int, list[Command]] = {}
        creeping = self.mode == "creep"
        foes = self.enemy_units + [c for c in self.creeps if creeping or self._creep_awake(c, units)]
        if self.opt["towers"]:  # (towers that shoot: in the fight's targets; attack-moving units had ignored them for
            foes += [b for b in self.enemy_buildings if b.type_id in TOWERS_ATTACK]  # the units around them)
        if not foes:
            for u in units:  # (no enemy in sight: a hero sent home stays until healed)
                if u.id in self.hero_home:
                    if u.hp >= HERO_OK * u.max_hp:
                        self.hero_home.discard(u.id)
                    else:
                        out[u.id] = self._order_point(u, MOVE, *self.home, every=10)
            return out
        fx_ = np.array([e.x for e in foes], float)
        fy_ = np.array([e.y for e in foes], float)
        assigned: Counter = Counter()
        for u in sorted(units, key=lambda u: (not u.is_hero, u.id)):
            d = np.hypot(fx_ - u.x, fy_ - u.y)
            if u.is_hero and self.opt["hero_home"] and (u.hp < HERO_LOW * u.max_hp or u.id in self.hero_home):
                if u.hp >= HERO_OK * u.max_hp:
                    self.hero_home.discard(u.id)
                else:
                    spell = self._cast(u, foes, d, units, escape_only=True)
                    if u.id not in self.hero_home:
                        self._note(f"hero {u.type} home at {u.hp}/{u.max_hp}")
                    self.hero_home.add(u.id)
                    out[u.id] = spell or self._order_point(u, MOVE, *self.home, every=6)
                    continue
            if d.min() > ENGAGE:
                continue
            if u.is_hero and self.opt["spells"]:
                spell = self._cast(u, foes, d, units)
                if spell:
                    out[u.id] = spell
                    continue
            if retreating or not self.opt["focus"]:
                continue
            data = unit_data().get(u.type, (0,) * 9)
            rng = max(data[4], 100.0)
            melee = rng <= 150
            if melee and self.opt["focus"] == 2:  # (ranged units only: melee ones attack-move, the game picks)
                continue
            # a hurt melee unit steps back while it is hit (the attackers chase or switch: both waste their blows)
            if self.opt["pull"] and melee and not u.is_hero and not u.flags & SUMMONED and u.hp < 0.3 * u.max_hp \
                    and self.hp_last.get(u.id, u.hp) > u.hp and self.t - self.pulled_at.get(u.id, -99) > 40:
                k = int(np.argmin(d))
                vx, vy = u.x - foes[k].x, u.y - foes[k].y
                n = max(math.hypot(vx, vy), 1.0)
                self.pulled[u.id] = self.t + PULL_STEPS
                self.pulled_at[u.id] = self.t
                out[u.id] = self._order_point(u, MOVE, u.x + 450 * vx / n, u.y + 450 * vy / n, every=1)
                continue
            if self.pulled.get(u.id, -1) > self.t:
                out[u.id] = []
                continue
            reach = rng + (160.0 if melee else 100.0)
            cand = np.flatnonzero(d <= reach)
            if not len(cand):
                k = int(np.argmin(d))  # (none in reach: toward the nearest, attack-moving)
                out[u.id] = self._order_point(u, ATTACK, foes[k].x, foes[k].y, every=4)
                continue
            best, score = None, None
            hit = dps(u.type, u.hero_level) * 1.5
            last = self.last_order.get(u.id)
            keep = None
            if self.opt["sticky"] and last is not None and last[0] == ATTACK and last[1] < 0 and u.order == ATTACK:
                keep = next((k for k in cand.tolist() if foes[k].id == -last[1]), None)
            if keep is not None and foes[keep].hp - assigned[foes[keep].id] > 0:
                # (its target still in reach and not dying already: kept, a melee unit walking to each new weakest
                # enemy had struck less)
                assigned[foes[keep].id] += hit
                out[u.id] = []
                continue
            for k in cand.tolist():
                e = foes[k]
                left = e.hp - assigned[e.id]
                s = left if left > 0 else e.max_hp + 400  # (dying already: someone else's)
                if e.is_hero:
                    s *= 0.6
                elif e.is_worker:
                    s *= 2.0
                elif e.flags & SUMMONED:
                    s *= 1.3
                if e.owner == CREEPS:
                    s *= 1.5
                if e.is_structure:  # (a tower: after the units in reach, its hit points count less, it shoots on)
                    s = s * 0.5 + 300.0
                s += d[k] * (0.8 if melee else 0.2)
                if score is None or s < score:
                    best, score = e, s
            assigned[best.id] += hit
            out[u.id] = self._order_target(u, ATTACK, best.id)
        for u in units:
            self.hp_last[u.id] = u.hp
        return out

    def _creep_awake(self, c, units) -> bool:
        """A creep fights back near the army (the army right on it)."""
        return any(math.hypot(c.x - u.x, c.y - u.y) < 450 for u in units[:40])

    # ---- spells ------------------------------------------------------------------------------------------------
    def _cast(self, h, foes, d, units, escape_only: bool = False) -> list[Command]:
        from .herodata import hero_table
        slots = hero_table().get(h.type_id, ())
        prio = {"escape": 0, "stomp": 1, "nuke": 2, "line": 2, "disable": 3, "heal": 3, "deathcoil": 3, "manaburn": 4,
                "summon": 5}  # (what is cast first when several are ready)
        order = sorted(range(len(slots)), key=lambda k: prio.get(SPELLS.get(getattr(slots[k], "order", None) or ""), 9))
        for k in order:
            a = slots[k]
            if a is None or a.order is None or k >= len(h.abilities):
                continue
            level, cd = h.abilities[k]
            if level <= 0 or cd > 0.05:
                continue
            mana = a.mana[min(level, len(a.mana)) - 1] if a.mana else 0.0
            if h.mana < mana:
                continue
            kind = SPELLS.get(a.order)
            oid = self.order_id.get(a.order)
            if kind is None or oid is None or (escape_only and kind != "escape"):
                continue
            cmd = self._spell(h, kind, oid, a.order, foes, d, units)
            if cmd is not None:
                self.counts[f"cast {a.order}"] += 1
                self.cmd_units.add(h.id)
                self.last_order[h.id] = (oid, 0, self.t)
                return [cmd]
        return []

    def _spell(self, h, kind: str, oid: int, order: str, foes, d, units) -> Command | None:
        hurt = h.hp < 0.45 * h.max_hp
        live = [i for i in range(len(foes)) if foes[i].owner != CREEPS or d[i] < 600]
        if kind == "escape":
            return ImmediateOrder(h.id, oid) if hurt and d.min() < 450 else None
        if kind == "summon":  # (only while none is out: wolves every cooldown had left no mana for chain lightning)
            out_now = sum(1 for u in units if u.flags & SUMMONED and math.hypot(u.x - h.x, u.y - h.y) < 1500)
            return ImmediateOrder(h.id, oid) if d.min() < 750 and out_now == 0 else None
        if kind == "stomp":
            n = int((d < STOMP_RADIUS).sum())
            return ImmediateOrder(h.id, oid) if n >= 3 or (n >= 1 and hurt) else None
        if kind in ("nuke", "disable", "manaburn"):
            rng = SPELL_RANGE[kind]
            cand = [i for i in live if d[i] <= rng and not foes[i].is_structure]
            if kind == "manaburn":
                cand = [i for i in cand if foes[i].mana > 50]
            if not cand:
                return None
            # the enemy hero first, else the most clumped (a chain), else the one closest to dying
            heroes = [i for i in cand if foes[i].is_hero]
            if heroes:
                i = min(heroes, key=lambda i: foes[i].hp)
            elif order in ("chainlightning", "forkedlightning"):
                pts = np.array([(foes[i].x, foes[i].y) for i in cand], float)
                i = max((int((np.hypot(pts[:, 0] - foes[i].x, pts[:, 1] - foes[i].y) < 450).sum()), -foes[i].hp, i)
                        for i in cand)[2]
            else:
                i = min(cand, key=lambda i: foes[i].hp if foes[i].hp > 60 else 10 ** 6)
            if kind == "disable" and foes[i].flags & SUMMONED:
                return None
            if kind == "nuke" and not foes[i].is_hero and foes[i].hp < 120 and len(cand) > 1:
                return None
            return TargetOrder(h.id, oid, foes[i].id)
        if kind == "line":
            cand = [i for i in live if d[i] <= SPELL_RANGE["line"]]
            if len(cand) < 2:
                return None
            pts = np.array([(foes[i].x, foes[i].y) for i in cand], float)
            return PointOrder(h.id, oid, float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))
        if kind in ("heal", "deathcoil"):
            undead = self.race == "undead"
            mine = [u for u in units if u.hp < 0.55 * u.max_hp and not u.flags & SUMMONED
                    and math.hypot(u.x - h.x, u.y - h.y) < SPELL_RANGE[kind]]
            if kind == "deathcoil":
                mine = [u for u in mine if undead and u.id != h.id]
            elif undead and order == "holybolt":
                mine = []  # (holy light does not heal undead units)
            if mine:
                u = max(mine, key=lambda u: (u.is_hero, u.max_hp - u.hp))
                return TargetOrder(h.id, oid, u.id)
            if kind == "deathcoil" or order == "holybolt":  # (death coil hurts the living, holy light the undead)
                cand = [i for i in live if d[i] <= SPELL_RANGE[kind] and (
                    (kind == "deathcoil" and not foes[i].type.startswith("u")) or
                    (order == "holybolt" and foes[i].type.startswith("u")))]
                if cand:
                    i = min(cand, key=lambda i: foes[i].hp)
                    return TargetOrder(h.id, oid, foes[i].id)
        return None

    # ---- production --------------------------------------------------------------------------------------------
    def _produce(self) -> list[Command]:
        """Each production building with room in its queue trains the unit of the mix the army lacks most that it
        makes; one it can't pay for yet keeps its money (the rest wait)."""
        from .techtree import tree
        out: list[Command] = []
        makers = tree().makers
        mix = self.plan.army
        have = Counter(u.type for u in self.army) + Counter(q[0] for qs in self.queue.values() for q in qs)
        total_w = sum(w for _, w in mix)
        for b in self.done:
            if not self._makes_army(b.type) or self._queued(b) >= self.plan.queue:
                continue
            total = sum(have[c] for c, _ in mix) + 1
            options = sorted((have[c] / total - w / total_w, c) for c, w in mix
                             if b.type in makers.get(c, ()) and self._tech_ok(c))
            if not options:
                continue
            c = options[0][1]
            cmd = self._train(b, c)
            if cmd:
                out += cmd
                have[c] += 1
            else:
                g, l = self._cost(c)
                if self.food + self._food(c) + self._waiting_food() <= min(self.cap, FOOD_CAP):
                    self._keep(g, l)
                break
        return out

    def _research(self) -> list[Command]:
        from .techtree import tree
        out: list[Command] = []
        for at, code in self.plan.research:
            if self.now < at or self.research_done[code] > 0:
                continue
            makers = tree().makers.get(code, ())
            b = next((u for u in self.done if u.type in makers and not self._queued(u)), None)
            if b is None or not self._tech_ok(code):
                continue
            g, l = self._cost(code)
            if self._pay(g, l):
                self.research_done[code] += 1
                self.cmd_units.add(b.id)
                out.append(ImmediateOrder(b.id, fourcc(code)))
            break
        return out

    # ---- orc's burrows -----------------------------------------------------------------------------------------
    def _battlestations(self, threats) -> list[Command]:
        if self.race != "orc":
            return []
        burrows = [u for u in self.done if u.type == "otrb"]
        if not burrows:
            return []
        m = self.mines.get(self.main_mine, self.home)
        close = [e for e in threats if min(math.hypot(e.x - self.home[0], e.y - self.home[1]),
                                           math.hypot(e.x - m[0], e.y - m[1])) < 900]
        danger = power(close) if close else 0.0
        army_home = [u for u in self.army if math.hypot(u.x - self.home[0], u.y - self.home[1]) < DEFEND_RADIUS]
        mine = power(army_home)
        # (the workers into the burrows only when the enemy is at the workers and stronger than the army at home: in
        # burrows they gather nothing, and the learner's early raids had kept them there for minutes)
        if not self.in_burrows and close and danger > 1.2 * max(mine, 1.0) and danger > 150:
            self.in_burrows = True
            self.burrow_since = self.t
            self._note(f"battle stations: danger {danger:.0f} vs army {mine:.0f}")
            self.counts["battle stations"] += 1
            return [ImmediateOrder(b.id, BATTLESTATIONS) for b in burrows]
        if self.in_burrows and (not close or danger < 0.6 * mine) and self.t - getattr(self, "burrow_since", 0) > 10:
            self.in_burrows = False
            return [ImmediateOrder(b.id, STANDDOWN) for b in burrows]
        if not self.in_burrows and self.opt["hold_home"] and close and danger > 0.8 * max(mine, 1.0) and any(
                u.order == BOARD or (u.order == 0 and u.flags & HIDDEN) for u in self.workers):
            self.in_burrows = True  # (the game's own boarding, kept against a wave the army can't hold alone)
            self.burrow_since = self.t
            self.counts["boarding kept"] += 1
            return []
        if not self.in_burrows and self.t - getattr(self, "stood_down", -99) >= 6 and any(
                u.order == BOARD or (u.order == 0 and u.flags & HIDDEN) for u in self.workers):
            # (the game itself sends a computer slot's peons into its burrows when enemies come near the base: the
            # learner's raids had kept them there for minutes, gathering nothing; tested 2026-10-25, no order of ours)
            self.stood_down = self.t
            self.counts["peons out"] += 1
            return [ImmediateOrder(b.id, STANDDOWN) for b in burrows]
        return []

    # ---- the log -----------------------------------------------------------------------------------------------
    def _note(self, s: str) -> None:
        if self.logging:
            self.events_log.append(f"{getattr(self, 'now', 0.0):6.0f} {s}")

    def _snapshot(self) -> None:
        army = Counter(u.type for u in self.army)
        self.log.append({"t": round(self.now), "food": self.food, "cap": self.cap, "gold": self.gold, "lumber": self.lumber,
                         "workers": len(self.workers), "army": dict(army), "mode": self.mode,
                         "heroes": [(u.type, u.hero_level, round(u.hp / max(u.max_hp, 1), 2)) for u in self.heroes],
                         "enemy_food": getattr(self.obs.players.get(self.enemy), "food_used", None),
                         "gathered": getattr(self.obs.players.get(self.side), "gold_gathered", None),
                         "roles": dict(Counter(self.assign.get(u.id, "none") for u in self.workers)),
                         "in_mine": sum(1 for u in self.workers if u.flags & HIDDEN),
                         "worker_orders": dict(Counter(str(u.order) for u in self.workers).most_common(4)),
                         "enemy_seen": len(self.enemy_units), "mines": {m: self.mine_gold.get(m) for m in self._working_mines()},
                         "buildings": dict(Counter(u.type for u in self.structures))})

    def summary(self) -> dict:
        return {"issued": self.issued, "refused": dict(self.refused.most_common(10)), "counts": dict(self.counts),
                "timeline": self.log, "notes": self.events_log[-120:]}
