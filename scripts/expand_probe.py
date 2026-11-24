"""The second base's lessons (--expand-lesson-from: teacher.py's expansion lessons from a game second on, in games
without a target), checked in the game itself (2026-11-24). Real Echo Isles, ONE game at a time, no learner, no GPU.

    python scripts/expand_probe.py engine --race human [--from 360] [--army 6] [--hero-level 1] [--camp group|heroes]
        [--minutes 14] [--lumber 2] [--side 1] [--seed 1] [--ai insane]
    python scripts/expand_probe.py policy --race human --weights runs/fgself-252/averaged/swa-17-27.pt --run runs/fgself-253
        [--from 360] [--minutes 12] [--follow 0|1] [--opp orc] [--ai insane] [--seed 1] [--json out.json]

engine  No policy. A scripted side (its workers on the main's gold and lumber, its hall training workers up to the
        race's count) gets, at --from, what the trigger asks for: a hero (--hero, level --hero-level, no skills) and
        --army army units (the race's first unit, or "hfoo:4,hrif:2") beside its hall. A Teacher with --expand-lesson-from
        looks on and its lessons are CARRIED OUT as the orders they name: the camp's attack-move, the hall's build order
        (undead: the haunt), the workers' harvest. The other side gives no orders (--ai: the built-in AI plays it).
        Per game: the camp (dead when, units lost, the heroes' hit points), the hall (order taken, started, finished,
        where), the workers at the natural, the gold gathered a minute before and after. --camp heroes: the camp
        lesson's orders for the heroes alone (the target games' camp lesson). --lumber N: the script's choppers beside
        its 5 gold workers (7: a one-base economy of 12; 2: the trained policy's 7 workers). --side 1: the right start.
policy  The trained policy on the CPU plays a real game without a target (the run's settings from its run.json, read
        only; selfplay.play_one with its teachers), --expand-lesson-from on. Logged per lesson step: how many units it
        labels, and the policy's chance of the labelled order, of the place given the order, of the pointer given the
        order (against the floors: would the lesson push?). And whether the policy expands by itself. --follow 1: the
        labelled units are given the lessons' orders (the labels' classes, bins and pointers through the agent's own
        command path) and left alone until the next lesson: what the policy's game looks like when it does as taught.
        --scenario natural: the camp cleared at the first step (a share of the training games). --reserve N:
        --expand-lesson-reserve. --lessons 0: no lessons, as an evaluation game (the reserve's rule alone).

What it found (2026-11-24; the engine's own facts are in docs/reference/engine-notes.md).

engine, the final lessons carried out (a level-1 hero without skills and 6 of the race's first unit at 360 s; the script's
workers 5 on gold and 2 on lumber, undead 5 and 3 spare, as the policy's; the other side idle), each race from each start:
    race, start     camp dead (units lost)  hall begun, done  at the new mine  gold a minute before -> after
    human, left     461 s (1 of 6)          491 s, 615 s      5 (and 1 idle)   600 -> 1,060 (the main at 4: below)
    human, right    461 s (1)               488 s, 612 s      5 (and 1 idle)   600 -> 1,200
    orc, left       455 s (0)               485 s, 635 s      5 (and 1 idle)   600 -> 1,170
    orc, right      454 s (0)               481 s, 631 s      5 (and 1 idle)   600 -> 1,200
    undead, left    405 s (1)               432 s, 532 s      5                600 -> 1,200
    undead, right   398 s (0)               426 s, 526 s      5 (1 at home)    600 -> 1,200
    night elf, left 460 s (0)               481 s, 601 s      5                600 -> 1,200
    night elf, right 460 s (1)              480 s, 600 s      5                600 -> 1,200
  Every order was taken (the hall's at people's spot lands at (-4736, -2368) and (4032, -2496)). Five minutes from the
  trigger to the second mine's income. What the first versions got wrong, and the fixes in teacher.py:
  * the trigger asked for at every step: the camp cost a unit, 5 were left, no hall was ever taught (now: on for the game)
  * a hero alone dies at the camp, at level 1 and 3, all four races (the target games' camp lesson names the heroes
    alone); a hero and 2 units: human and orc lose everything; and 3: cleared with the hero at a quarter of its hit points;
    twice the camp's strength (4-6 units): 0-1 lost (now: the heroes and the army near them, twice its strength)
  * a second and third builder labelled while the first walked (now: none while one is on its way), and the builder one of
    the main's gold workers (now: a worker without work or a chopper first)
  * the workers' lesson sent the three nearest whatever they did: 4 of the main's 5 gold workers; 600 -> 720 a minute.
    And it counted whoever stood within 900: workers trained there stood idle (now: until five work the mine, the main
    keeps five, two choppers stay; a wisp in an entangled mine counts)
  * the policy has 7-11 workers: a second mine's have to be trained (now: "expansion: a worker")
  * a human gold peasant walks to the new town hall by itself and helps (the game's doing): 120 gold a minute less meanwhile

policy (fgself-252's 17-27M average on the CPU, fgself-253's settings, no target, against the insane AI, --from 360):
  * plain games, 8 (two a race): the trigger is on at 360 s in all (9-14 army units, a hero of level 2-3 at 7-68% of its
    hit points; 7-11 workers; 50-430 gold in the bank). The camp's lesson is given 3, 0 / 1, 2 / 0, 0 / 4, 13 times
    (human / orc / undead / night elf): in 57-100% of the steps a hero is dead or under 60%. At the labelled steps the
    policy's chance of an attack-move is 0.0002-0.017 a unit and of the camp's cell given one 0.0000-0.0006: under the
    place's floor every time, so the lesson would push the place; the order it never pushes. No expansion by itself (a
    tree of life begun once beside the standing camp). 5 of the 8 games are won before minute 10.
  * doing as taught (--follow 1), plain: human 5 and orc 6 lessons followed (night elf none given), the heroes never
    within 1,000 of the camp: the walk is 35-40 s and the hero is under 60% again before it ends. The camp is not
    cleared in the policy's games against the insane AI.
  * the camp cleared (--scenario natural): the hall is due from 360 s to the end and its cost is in the bank in 0 of
    720, 0 of 360, 0 of 720 and 65 of 720 steps: no hall lesson but night elf's 4. The policy banks nothing.
  * with the hall's cost kept (--reserve 360), no follow: human and night elf build the hall by themselves at 516 and 586 s,
    orc gets 11 lessons (its chance of the order 0.0008 a step, of the place given it 0.68) and does not, undead never
    has the 220 lumber. Without lessons (--lessons 0, an evaluation game's rules): human, orc and night elf build it by
    themselves at 516, 472 and 425 s; then one worker mines there, or none, and the human's gold a minute is 480 where
    it was 600 (its builder gone from the main): the hall alone brings nothing.
  * with the cost kept and doing as taught: human: hall at 505 s, done 646 s, 4 workers trained, 5 at the new mine at
    13 minutes, 870 gold in the last minute with the new workers still arriving (600 before; 360-400 while the hall
    went up), won at 13.8 minutes (the same game without: not over at 12). Night elf: the tree at 446 s, done 566 s, won at 10.7 minutes before its wisps mined. Orc: won at 8.6
    minutes with the hall half built. The labels' chances there: the hall's order 0.002-0.009 a step and its place
    0.33-0.85; a harvest 0.26 and the new mine given it 0.02-0.14 (under the pointer's floor: pushed); a worker's
    training 0.00003-0.0001 (under the order's floor: pushed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

ATTACK, MOVE, HARVEST, ENTANGLE = 851983, 851986, 852018, 852147
WORKER = {"human": "hpea", "orc": "opeo", "undead": "uaco", "nightelf": "ewsp"}
OWN_MINE = {"undead": "ugol", "nightelf": "egol"}
FIRST_UNIT = {"human": "hfoo", "orc": "ogru", "undead": "ugho", "nightelf": "earc"}
HERO = {"human": "Hamg", "orc": "Ofar", "undead": "Udea", "nightelf": "Edem"}
FARM = {"human": "hhou", "orc": "otrb", "undead": "uzig", "nightelf": "emow"}
ON_LUMBER = {"human": 7, "orc": 7, "undead": 0, "nightelf": 7}  # (the script's workers beside the mine's 5: a one-base
# economy of 12, as people's 15-18 less the builders)
SPARE = {"undead": 3}  # (undead's acolytes cannot chop: three to build, as --econ-masks allows it)
HIDDEN = 16


def fourcc(c: str) -> int:
    return int.from_bytes(c.encode(), "big")


def army_spec(spec: str, race: str) -> list[str]:
    """"6" -> six of the race's first unit; "hfoo:4,hrif:2" -> those."""
    if spec.isdigit():
        return [FIRST_UNIT[race]] * int(spec)
    out = []
    for part in spec.split(","):
        code, _, n = part.partition(":")
        out += [code] * int(n or 1)
    return out


class Economy:
    """The scripted side's economy: 5 workers on the main's gold, `lumber` on the trees (undead: its ghoul; SPARE idle
    acolytes), its hall training until it has them. A worker it did not ask for (a lesson's) is left to the lessons."""

    def __init__(self, obs, side: int, race: str, lumber: int):
        self.side, self.race, self.lumber = side, race, lumber
        p = obs.players[side]
        self.start = (p.start_x, p.start_y)
        self.worker = WORKER[race]
        self.job: dict[int, str] = {}
        self.pending = 0
        self.asked = -100
        self.trees = sorted(obs.destructables or [], key=lambda d: (d.x - p.start_x) ** 2 + (d.y - p.start_y) ** 2)
        self.next_tree = 0
        self.done = False

    def main_mine(self, obs):
        want = OWN_MINE.get(self.race, "ngol")
        ms = [u for u in obs.units if u.type == want and u.alive and (want == "ngol" or u.owner == self.side)
              and math.hypot(u.x - self.start[0], u.y - self.start[1]) < 1500]
        return ms[0] if ms else None

    def commands(self, obs, t: int) -> list:
        from warcraftsim.protocol import ImmediateOrder, TargetDestructable, TargetOrder
        side, out = self.side, []
        own = [u for u in obs.units if u.owner == side and u.alive]
        workers = [u for u in own if u.type == self.worker]
        mine = self.main_mine(obs)
        alive = {u.id for u in workers}
        self.job = {i: j for i, j in self.job.items() if i in alive}
        count = Counter(self.job.values())
        for u in workers:
            if u.id in self.job:
                continue
            if t > 1 and not self.pending:  # (not the script's: a lesson's worker)
                self.job[u.id] = "taught"
                continue
            self.pending = max(self.pending - 1, 0)
            if count["gold"] < 5 and mine is not None:
                self.job[u.id] = "gold"
                out.append(TargetOrder(u.id, HARVEST, mine.id))
            elif count["lumber"] < self.lumber and self.trees:
                self.job[u.id] = "lumber"
                out.append(TargetDestructable(u.id, HARVEST, self.trees[self.next_tree % len(self.trees)].id))
                self.next_tree += 2
            else:
                self.job[u.id] = "spare"
            count[self.job[u.id]] += 1
        if t == 1 and self.race == "undead":  # (its ghoul chops)
            out += [TargetDestructable(u.id, HARVEST, self.trees[0].id) for u in own if u.type == "ugho"]
        if t == 1:  # (food for its workers and the army made later: three supply buildings, made, not built)
            from warcraftsim.protocol import Spawn
            sx, sy = self.start
            out += [Spawn(side, FARM[self.race], sx + (700.0 if sx < 0 else -700.0), sy - 500.0 + 350.0 * k) for k in range(3)]
        want = 5 + self.lumber + SPARE.get(self.race, 0)
        p = obs.players[side]
        halls = [u for u in own if u.is_structure and u.type in ("htow", "hkee", "ogre", "ostr", "unpl", "unp1", "etol", "etoa")
                 and not int(u.flags) & 8]
        mine_ = sum(1 for j in self.job.values() if j in ("gold", "lumber", "spare", "builder", "natural"))
        if mine_ + self.pending < want and not self.done and halls and p.gold >= 90 and p.food_used < p.food_cap and t - self.asked >= 4:
            out.append(ImmediateOrder(halls[0].id, fourcc(self.worker)))
            self.pending += 1
            self.asked = t
        elif mine_ >= want:
            self.done = True  # (its count reached once: no more of the script's)
        return out


def engine(a) -> int:
    from warcraftsim.data.objects import unit_table
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame import teacher as tm
    from warcraftsim.fullgame.play import unit_rows
    from warcraftsim.protocol import Build, ImmediateOrder, PointOrder, Spawn, SpawnHero, TargetOrder
    from warcraftsim.runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup
    fx.set_neutral(2)
    fx.set_macros(3)
    fx.set_macros_off(["mix", "creep", "expand"])
    if a.margin is not None:
        tm.EARLY_MARGIN = a.margin
    if a.need is not None:
        tm.EARLY_ARMY = a.need
    race, side = a.race, a.side
    slots = [Agent(race, handicap=100, difficulty="normal"),
             BuiltinAI(a.opp, a.ai, handicap=100) if a.ai else Agent(a.opp, handicap=100, difficulty="normal")]
    slots = slots[::-1] if side else slots  # (the second slot: the other start)
    gs = GameSetup(map="(2)EchoIsles", slots=slots, step_seconds=0.5, draw=False, native_obs=False, d3d_thread=False,
                   render_threads=0, nice=19, warm_spare=False, window=(320, 240), max_game_seconds=0, victory="none",
                   cuts=a.cuts, seed=a.seed, **({"shim_dir": a.shim} if a.shim else {}))
    table = unit_table()
    hall_code = "ugol" if race == "undead" else tm.HALL[race]
    cost = (table[hall_code].gold, table[hall_code].lumber)
    out: dict = {"race": race, "seed": a.seed, "from": a.start, "camp": a.camp, "hall_cost": cost}
    with GameInstance(gs, name=f"expandprobe{a.name}", timeout=600) as g:
        obs = g.start()
        p = obs.players[side]
        sx, sy = p.start_x, p.start_y
        out["start"] = (sx, sy)
        mines = [u for u in obs.units if u.type == "ngol" and math.hypot(u.x - sx, u.y - sy) > 1500]
        nat = min(mines, key=lambda u: math.hypot(u.x - sx, u.y - sy))
        camp = {u.id: u.type for u in obs.units if u.owner == 12 and math.hypot(u.x - nat.x, u.y - nat.y) < 900}
        out["natural"], out["camp_units"] = (nat.x, nat.y), sorted(camp.values())
        eco = Economy(obs, side, race, ON_LUMBER[race] if a.lumber is None else (0 if race == "undead" else a.lumber))
        busy_until: dict[int, int] = {}  # (a hall a lesson's worker trains at: busy 15 s, as the view's `busy`)
        tc = tm.Teacher(side, race, random.Random(a.seed), None, expand_from=a.start, pace=1.0)
        tc.first = tc.pick = None  # (the expansion's lessons alone)
        army: dict[int, str] = {}
        spawned = False
        ev: dict = {"lessons": Counter(), "taken": Counter(), "refused": Counter()}
        gold_at: dict[int, int] = {}  # (game minute -> gold gathered by then)
        hall_seen = hall_done = camp_dead = first_camp = first_hall = first_workers = None
        hall_xy = None
        power = None
        mining_log: list = []
        t, entangle_t = 0, -100
        cmds_kind: list = []
        while obs.game_time < a.minutes * 60 and not obs.game_over:
            now = obs.game_time
            p = obs.players[side]
            rows = unit_rows(obs, t)
            if int(now // 60) not in gold_at:
                gold_at[int(now // 60)] = p.gold_gathered
            cmds = eco.commands(obs, t) if t >= 1 else []
            kinds = [None] * len(cmds)
            units = {u.id: u for u in obs.units}
            if not spawned and now >= a.start - 1.0:  # (what the trigger asks for, beside its hall, toward the map's middle)
                ax = sx + (500.0 if sx < 0 else -500.0)
                spec = [(HERO[race] if not a.hero else a.hero, a.hero_level)] + [(c, 0) for c in army_spec(a.army, race)]
                before = set(units)
                g_cmds = [SpawnHero(side, c, ax, sy - 300.0 - 40.0 * k, lv) if lv else Spawn(side, c, ax + 60.0 * (k % 4), sy - 400.0 - 60.0 * (k // 4))
                          for k, (c, lv) in enumerate(spec)]
                cmds += g_cmds
                kinds += ["spawn"] * len(g_cmds)
                spawned = True
                out["spawn_at"] = round(now, 1)
            elif spawned and not army:
                army = {u.id: u.type for u in obs.units if u.owner == side and u.alive and not u.is_structure
                        and not u.is_worker and u.id not in before}
                out["army"] = sorted(army.values())
            # the agent's own entangle (play.BCAgent._entangle, --macros): a finished tree of life within 1200 of a
            # free mine entangles it, asked again every 20 steps
            if race == "nightelf" and t - entangle_t >= 20:
                for u in obs.units:
                    if u.owner == side and u.alive and u.type == "etol" and not int(u.flags) & 8 \
                            and math.hypot(u.x - nat.x, u.y - nat.y) < 1200 and nat.id in units and units[nat.id].type == "ngol" \
                            and not any(m.type == "egol" and math.hypot(m.x - nat.x, m.y - nat.y) < 300 for m in obs.units):
                        cmds.append(TargetOrder(u.id, ENTANGLE, nat.id))
                        kinds.append("entangle")
                        entangle_t = t
            can = lambda code, kind=0, p=p: (p.gold >= table[code].gold and p.lumber >= table[code].lumber)  # noqa: E731
            tc.economy = (False, {i for i, j in eco.job.items() if j == "lumber"}, {})  # (the choppers, as the view's)
            tc.busy = {h for h, until in busy_until.items() if t < until}
            lessons = tc.commands(rows, p, now, t, can) if spawned and army else []
            if a.camp == "heroes":  # (the camp's attack-move for the heroes alone)
                lessons = [c for c in lessons if not (isinstance(c, PointOrder) and c.order == ATTACK
                                                      and c.unit in units and not units[c.unit].is_hero)]
            for c in lessons:
                if isinstance(c, PointOrder) and c.order == ATTACK:
                    k = "camp"
                    if first_camp is None:
                        first_camp = now
                        grp = rows[np.isin(rows[:, fx.C_ID], [x.unit for x in lessons])]
                        cr = rows[np.isin(rows[:, fx.C_ID], list(camp))]
                        power = (round(tm._power(grp), 1), round(tm._power(cr), 1))
                elif (isinstance(c, TargetOrder) and c.order == HARVEST) or (isinstance(c, PointOrder) and hall_done is not None):
                    k = "workers"  # (the harvest, or the walk to a mine out of its sight)
                    first_workers = now if first_workers is None else first_workers
                    eco.job[c.unit] = "natural"
                elif isinstance(c, ImmediateOrder):  # (a worker more, at the hall nearest the natural)
                    k = "train"
                    busy_until[c.unit] = t + 30
                else:
                    k = "hall"
                    first_hall = now if first_hall is None else first_hall
                    out.setdefault("builder_was", eco.job.get(c.unit))
                    if a.debug:
                        print(f"    {now:.1f} s hall lesson: {c} (the unit's job {eco.job.get(c.unit)}); jobs {Counter(eco.job.values())}", flush=True)
                    eco.job[c.unit] = "builder"
                    if isinstance(c, Build):
                        out["hall_asked"] = (round(c.x), round(c.y))
                kinds.append(k)
            ev["lessons"].update(Counter(k for k in kinds[len(kinds) - len(lessons):]))
            cmds += lessons
            cmds_kind = kinds
            obs = g.step(cmds)
            t += 1
            for k, ok in zip(cmds_kind, obs.command_results):
                if k in ("camp", "hall", "workers", "train", "entangle", "spawn"):
                    ev["taken" if ok else "refused"][k] += 1
            # what became of it
            units = {u.id: u for u in obs.units}
            now = obs.game_time
            if a.debug and first_hall is not None and t % 10 == 0 and now < first_hall + 200:
                print(f"    {now:.0f} s workers off their job's place:", [(u.id, eco.job.get(u.id), u.x, u.y, u.order, int(u.flags)) for u in obs.units
                      if u.owner == side and u.alive and u.type == WORKER[race] and (eco.job.get(u.id) in ("builder", "natural")
                      or math.hypot(u.x - sx, u.y - sy) > 1800)], flush=True)
            if camp_dead is None and first_camp is not None and not any(i in units and units[i].alive for i in camp):
                camp_dead = now
                out["army_lost"] = sum(1 for i in army if i not in units or not units[i].alive)
                out["heroes_hp"] = [(u.type, u.hp, u.max_hp, u.hero_level) for i in army if (u := units.get(i)) and u.is_hero]
            holds = [u for u in obs.units if u.owner == side and u.alive and u.type in (tm.HALL[race], "ugol", "egol")
                     and math.hypot(u.x - nat.x, u.y - nat.y) < 1000]
            taken = [u for u in holds if u.type == hall_code or (race != "undead" and u.type == tm.HALL[race])]
            if taken and hall_seen is None:
                hall_seen, hall_xy = now, (taken[0].x, taken[0].y)
            if taken and hall_done is None and not int(taken[0].flags) & 8:
                hall_done = now
            if t % 120 == 0:
                ws = [u for u in obs.units if u.owner == side and u.alive and u.type == WORKER[race]]
                near = [u for u in ws if math.hypot(u.x - nat.x, u.y - nat.y) < 900]
                mining = [u for u in near if u.order in (852018, 852017, 852020, 851971) or int(u.flags) & HIDDEN]
                mining_log.append((round(now), len(ws), len(near), len(mining)))
                mm = eco.main_mine(obs)
                at_main = Counter(eco.job.get(u.id, "?") for u in ws if mm is not None and math.hypot(u.x - mm.x, u.y - mm.y) < 900)
                if a.debug:
                    print("    gold workers away:", [(u.id, u.x, u.y, u.order, int(u.flags)) for u in ws if eco.job.get(u.id) == "gold"
                                                     and mm is not None and math.hypot(u.x - mm.x, u.y - mm.y) >= 900], flush=True)
                print(f"  {now / 60:4.1f} min: gold {p.gold} gathered {p.gold_gathered} lumber {p.lumber} food {p.food_used}/{p.food_cap} "
                      f"upkeep {p.upkeep} at the main mine {dict(at_main)} workers {len(ws)} at the natural "
                      f"{len(near)} mining {len(mining)}; camp {sum(1 for i in camp if i in units and units[i].alive)}/{len(camp)} "
                      f"army {sum(1 for i in army if i in units and units[i].alive)}/{len(army)} lessons {dict(tc.taught)} "
                      f"hall {hall_seen and round(hall_seen)}/{hall_done and round(hall_done)}", flush=True)
        p = obs.players[side]
        gold_at[int(obs.game_time // 60)] = gold_at.get(int(obs.game_time // 60), p.gold_gathered)
        per_min = {m: gold_at[m + 1] - gold_at[m] for m in sorted(gold_at) if m + 1 in gold_at}
        m0 = int(a.start // 60) - 1
        ws = [u for u in obs.units if u.owner == side and u.alive and u.type == WORKER[race]]
        near = [u for u in ws if math.hypot(u.x - nat.x, u.y - nat.y) < 900]
        mining = [u for u in near if u.order in (852018, 852017, 852020, 851971) or int(u.flags) & HIDDEN]
        main = eco.main_mine(obs)
        census = Counter()
        for u in ws:  # (each worker: the script's job for it, where it is, what it does)
            where = ("natural" if math.hypot(u.x - nat.x, u.y - nat.y) < 900 else
                     "main mine" if main is not None and math.hypot(u.x - main.x, u.y - main.y) < 900 else "elsewhere")
            doing = ("in a mine" if int(u.flags) & HIDDEN else
                     {852018: "harvest", 852017: "harvest", 852020: "harvest", 851971: "smart", 0: "idle"}.get(u.order, str(u.order)))
            census[f"{eco.job.get(u.id, '?')}/{where}/{doing}"] += 1
        out["census"] = dict(census)
        out["buildings_at_natural"] = [(u.type, u.x, u.y, bool(int(u.flags) & 8)) for u in obs.units
                                       if u.owner == side and u.alive and u.is_structure and math.hypot(u.x - nat.x, u.y - nat.y) < 1500]
        print(f"  workers at the end (job/where/doing): {dict(census)}; its buildings at the natural: {out['buildings_at_natural']}", flush=True)
        out.update(power=power, first_camp=first_camp, camp_dead=camp_dead, first_hall=first_hall, hall_started=hall_seen,
                   hall_done=hall_done, hall_at=hall_xy, first_workers=first_workers, lessons=dict(ev["lessons"]),
                   taken=dict(ev["taken"]), refused=dict(ev["refused"]), workers=len(ws), at_natural=len(near),
                   mining_natural=len(mining), income_before=per_min.get(m0), income_last=per_min.get(max(per_min, default=0)),
                   per_minute=per_min, taught=dict(tc.taught),
                   army_left=sum(1 for i in army if (u := {u.id: u for u in obs.units}.get(i)) and u.alive), army_n=len(army))
    r = lambda v: None if v is None else round(v)  # noqa: E731
    print(f"RESULT {race} start {out['start']} camp {out['camp_units']} power {power}: camp lesson at {r(first_camp)} s, dead at "
          f"{r(camp_dead)} s, army lost {out.get('army_lost')} of {len(army)} (left {out['army_left']}), heroes {out.get('heroes_hp')}; "
          f"hall lesson at {r(first_hall)} s (asked {out.get('hall_asked')}), started {r(hall_seen)} s at {hall_xy}, done {r(hall_done)} s; "
          f"workers lesson at {r(first_workers)} s, at the natural {len(near)} mining {len(mining)}; lessons {out['lessons']} taken "
          f"{out['taken']} refused {out['refused']}; gold a minute before {out['income_before']}, last {out['income_last']}; "
          f"per minute {per_min}", flush=True)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(out, default=str) + "\n")
    return 0


KINDS = ("expansion: its camp", "expansion: the hall", "expansion: workers", "expansion: a worker")


class PolicyProbe:
    """The policy game's stand-ins (the inference, the learner's queue), the lessons' labels against the policy's own
    chances, and with --follow the lessons carried out through the agent's own command path."""

    def __init__(self, net, vocab: dict, side: int, seed: int, follow: bool, floors: dict):
        import torch
        from warcraftsim.fullgame import features as fx
        self.net, self.vocab, self.side, self.follow, self.floors = net, vocab, side, follow, floors
        self.gen = torch.Generator().manual_seed(seed)
        self.names = {int(k): v for k, v in (vocab.get("order_names") or {}).items()}
        self.orders = [(0, 0)] + [tuple(o) for o in vocab["orders"]]
        self.labels = ["none"] + [fx.order_label(o, k, self.names) for o, k in self.orders[1:]]
        self.seen: dict = {}  # (the step's states and their tensors, by the state's id)
        self.mine = None  # (the probed side's state of the step)
        self.label_side = None  # (the side whose lessons are being labelled)
        self.now_lessons: list = []  # (the step's expansion lessons: (kind, command))
        self.records: list = []  # (a labelled unit of an expansion lesson: its kind and the policy's chances)
        self.steps: list = []  # (a lesson step: kind, t, units named, labels kept)
        self.hold: dict = {}  # (--follow: unit -> [class, target id, x bin, y bin, the step it is free again, given])
        self.bot = None
        self.t = 0
        self.own: dict = {}  # (what the policy did by itself: counts)
        self.minute: list = []
        self.state: dict = {}

    def put(self, msg: dict) -> None:
        pass

    def request(self, items: list) -> list:
        self.seen = {}
        return [self.forward(st) for _, st in items]

    def _tensors(self, st: dict):
        import torch
        from warcraftsim.fullgame import features as fx
        n = st["n"]
        ent = torch.from_numpy(np.asarray(st["ent"], np.float32)[None, :n])
        typ = torch.from_numpy(np.asarray(st["type"], np.int64)[None, :n])
        cur = torch.from_numpy(np.asarray(st["cur"], np.int64)[None, :n])
        mask = torch.ones(1, n, dtype=torch.bool)
        glob = torch.from_numpy(np.asarray(st["glob"], np.float32)[None])
        n_own = torch.tensor([min(st["n_own"], fx.MAX_OWN)])
        avail = torch.from_numpy(np.asarray(st["avail"], bool)[None]) if st.get("avail") is not None else None
        h = torch.from_numpy(np.asarray(st["h"], np.float32)[None]) if st.get("h") is not None else None
        return ent, typ, cur, mask, glob, n_own, avail, h

    def forward(self, st: dict) -> dict:
        import torch
        from warcraftsim.fullgame.model import act
        tensors = self._tensors(st)
        ent, typ, cur, mask, glob, n_own, avail, h = tensors
        shared = torch.tensor([bool(st.get("shared_noise"))])
        with torch.no_grad():
            out = act(self.net, ent, typ, cur, mask, glob, n_own, avail, h, generator=self.gen, shared=shared)
        o = int(n_own[0])
        res = {"order": out["order"][0, :o].numpy().astype(np.int64), "tgt": out["tgt"][0, :o].numpy().astype(np.int64),
               "bx": out["bx"][0, :o].numpy().astype(np.int64), "by": out["by"][0, :o].numpy().astype(np.int64),
               "logp": out["logp"][0, :o].numpy().copy(), "value": float(out["value"][0]),
               "entropy": float(out["entropy"][0]), "version": 0}
        if self.net.memory:
            res["h"] = out["h"][0].numpy().copy()
        self.seen[id(st)] = (st, tensors)  # (each side's of the step: the probed side's is `mine`, BCAgent.observe's)
        return res

    def on_labels(self, y: dict) -> None:
        """The step's lessons' labels (the encoder's, as the learner gets them) against the policy's chances in the state
        it acted in: of the labelled order, of the place given that order (the cell, and the 5 x 5 cells around it), of
        the pointer given it."""
        import torch
        from warcraftsim.fullgame import features as fx
        from warcraftsim.fullgame.model import point_parts
        last = self.seen.get(id(self.mine)) if self.mine is not None else None
        if not self.now_lessons or last is None:
            return
        st, (ent, typ, cur, mask, glob, n_own, avail, h) = last
        net = self.net
        O = min(fx.MAX_OWN, ent.shape[1])
        yo = torch.from_numpy(np.asarray(y["y_order"][:O], np.int64))[None]
        yx, yy, _ = point_parts(torch.from_numpy(np.asarray(y["y_x"][:O], np.int64))[None],
                                torch.from_numpy(np.asarray(y["y_y"][:O], np.int64))[None])
        with torch.no_grad():
            g, u = net.encode(ent, typ, cur, mask, glob)
            g, _ = net.context(g, h)
            lg = net.order_logits(g, u[:, :O], typ, n_own, avail=avail, ent=ent, emask=mask if net.omask else None)
            p_order = torch.softmax(lg[0].float(), -1)
            ptr, xl, z = net.target_logits(g, u, mask, yo, ent, typ)
            lp_x = torch.log_softmax(xl[0].float(), -1)
            lp_y = torch.log_softmax(net.y_logits(z, yx)[0].float(), -1)
            p_ptr = torch.softmax(ptr[0].float(), -1)
        index, sel = st["index"], st["sel"]
        seen = Counter()
        for kind, c in self.now_lessons:
            seen[kind] += 1
            k = index.get(int(c.unit))
            rec = {"kind": kind, "t": self.t, "unit": fx.rawcode(int(sel[k, fx.C_TYPE])) if k is not None else None, "kept": False}
            self.records.append(rec)
            if k is None or k >= O or int(y["y_order"][k]) <= 0:
                continue  # (no label: not among its units, an order the vocabulary lacks, or one the encoder dropped)
            cls = int(y["y_order"][k])
            rec.update(kept=True, order=self.labels[cls], open=bool(avail is None or avail[0, cls]),
                       p_order=float(p_order[k, cls]), p_none=float(p_order[k, 0]))
            if int(y["y_x"][k]) >= 0:
                bx, by = int(yx[0, k]), int(yy[0, k])
                rec["p_place"] = float(torch.exp(lp_x[k, bx] + lp_y[k, by]))
                near = 0.0
                with torch.no_grad():  # (the 5 x 5 cells around it: the policy's own scatter)
                    for dx in range(-2, 3):
                        if not 0 <= bx + dx < lp_x.shape[-1]:
                            continue
                        ly = torch.log_softmax(net.y_logits(z, torch.full_like(yx, bx + dx))[0, k].float(), -1)
                        lo, hi = max(by - 2, 0), min(by + 3, ly.shape[-1])
                        near += float(torch.exp(lp_x[k, bx + dx]) * torch.exp(ly[lo:hi]).sum())
                rec["p_near"] = near
                rec["place"] = [round(v) for v in fx.bins_world(int(y["y_x"][k]), int(y["y_y"][k]), self.bot.view.sign)]
            if int(y["y_ptr"][k]) >= 0:
                j = int(y["y_ptr"][k])
                rec["p_ptr"] = float(p_ptr[k, j])
                rec["target"] = fx.rawcode(int(sel[j, fx.C_TYPE]))
                rec["target_id"] = int(sel[j, fx.C_ID])
            if self.follow:  # (the label itself, given at the next step through the agent's own command path)
                self.hold[int(c.unit)] = [cls, rec.get("target_id"), int(y["y_x"][k]), int(y["y_y"][k]), self.t + 20, False]
        for kind, n in seen.items():
            self.steps.append({"kind": kind, "t": self.t, "units": n,
                               "kept": sum(1 for r in self.records[-len(self.now_lessons):] if r["kind"] == kind and r["kept"])})
        self.now_lessons = []


def policy(a) -> int:
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ghoul_probe import run_cfg
    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.data.objects import unit_table
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame import selfplay as sp
    from warcraftsim.fullgame import strategy as strat
    from warcraftsim.fullgame import teacher as tm
    from warcraftsim.fullgame.model import load
    from warcraftsim.fullgame.play import BCAgent
    from warcraftsim.protocol import Build, ImmediateOrder, PointOrder, TargetOrder
    from warcraftsim.runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup
    if a.weights is None or a.run is None:
        raise SystemExit("policy: --weights and --run")
    vocab = json.loads((a.run / "vocab.json").read_text())
    run = json.loads((a.run / "run.json").read_text())["config"]
    over = dict(kv.split("=", 1) for kv in a.set)
    over.setdefault("z_share", 0.0)  # (a game without a target)
    over["max_steps"] = int(a.minutes * 120)
    over["max_minutes"] = a.minutes + 1.0
    cfg = run_cfg(a.run, vocab, over)
    cfg["expand_lesson_from"], cfg["expand_lesson_share"] = float(a.start), float(a.share)
    cfg["expand_lesson_reserve"] = int(a.reserve)
    import tempfile
    cfg["run_dir"] = tempfile.mkdtemp(prefix="expandprobe")  # (nothing of this game goes into the run's folder)
    fx.ACTIVE_F = cfg["features"]
    fx.set_inputs(cfg["inputs"], bool(cfg["privileged"]))
    fx.set_masks(cfg["tech_mask"], cfg["valid_mask"])
    fx.set_neutral(cfg["neutral"])
    fx.set_macros(int(cfg["macros"] or 0))
    fx.set_spells(int(cfg["spells"] or 0))
    fx.set_shop_items(int(cfg["shop_items"] or 0))
    fx.set_full_vocab(int(cfg["full_vocab"] or 0))
    fx.set_obs_full(int(cfg["obs_full"] or 0))
    fx.set_fine(int(cfg["fine_points"] or 0))
    fx.set_macros_off(str(cfg["macros_off"] or "").split(","))
    fx.set_econ_masks(cfg["econ_masks"])
    fx.set_order_masks(cfg["order_masks"])
    fx.set_value_tax(cfg["value_tax"])
    if (width := strat.block_width(cfg["z_modes"], cfg["z_mix"], cfg["z_comp"], cfg["z_shop"])):
        fx.set_z(width)
    fx.set_geometry(*fx.map_geometry(stock_map_path(cfg["map"])))
    net, _ = load(a.weights, "cpu", vocab=vocab)
    fx.set_z(net.config.get("z", 0))
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    side, race = a.side, a.race

    def num(key, default):
        v = run.get(key)
        return default if v in (None, "None", "") else float(v)
    floors = {"order": num("teach_floor", 0.0), "place": num("teach_floor_place", 0.0) or num("teach_floor", 0.0)}
    floors["pointer"] = a.pointer_floor or floors["place"]
    probe = PolicyProbe(net, vocab, side, a.seed, bool(a.follow), floors)
    table = unit_table()
    hall_code = "ugol" if race == "undead" else tm.HALL[race]
    cost = (table[hall_code].gold, table[hall_code].lumber)
    worker_t = fourcc(WORKER[race])

    # the wrappers: the agent (the probed side's), the teacher's lessons, the labels
    b_begin, b_commands, b_observe = BCAgent.begin, BCAgent.commands, BCAgent.observe

    def begin(self, obs, races):
        b_begin(self, obs, races)
        if self.player == side:
            probe.bot = self

    def observe(self, obs, t, rows=None):
        st = b_observe(self, obs, t, rows)
        if self.player == side:
            probe.t, probe.mine = t, st
        return st

    def commands(self, st, order, tgt, bx, by):
        if self.player == side and probe.hold:
            order, tgt, bx, by = (np.array(v).copy() for v in (order, tgt, bx, by))
            index = st["index"]
            for uid, h in list(probe.hold.items()):
                k = index.get(uid)
                if k is None or k >= len(order) or probe.t > h[4]:
                    del probe.hold[uid]
                    continue
                if not h[5]:  # (the lesson's order, once; its pointer by the unit it named)
                    j = index.get(h[1]) if h[1] is not None else 0
                    if j is None:
                        del probe.hold[uid]
                        continue
                    order[k], tgt[k], bx[k], by[k] = h[0], j, max(h[2], 0), max(h[3], 0)
                    h[5] = True
                    probe.own["followed " + probe.labels[h[0]]] = probe.own.get("followed " + probe.labels[h[0]], 0) + 1
                else:  # (then left alone until the next lesson, or 10 s)
                    order[k] = 0
        out = b_commands(self, st, order, tgt, bx, by)
        if self.player == side:
            nat = probe.state.get("nat")
            for c in out:
                if isinstance(c, Build) and c.building in tm.HALL_CODES.tolist():
                    probe.own["hall builds sent"] = probe.own.get("hall builds sent", 0) + 1
                    probe.own.setdefault("hall builds at", []).append((probe.t, round(c.x), round(c.y)))
                elif isinstance(c, TargetOrder) and c.order == tm.HAUNTED:
                    probe.own["haunts sent"] = probe.own.get("haunts sent", 0) + 1
                elif isinstance(c, TargetOrder) and c.order == HARVEST and nat is not None and c.target in nat["mine_ids"]:
                    probe.own["harvests sent to the natural"] = probe.own.get("harvests sent to the natural", 0) + 1
                elif isinstance(c, PointOrder) and c.order == ATTACK and nat is not None \
                        and math.hypot(c.x - nat["x"], c.y - nat["y"]) < 1000:
                    probe.own["attack-moves sent to the natural"] = probe.own.get("attack-moves sent to the natural", 0) + 1
        return out
    BCAgent.begin, BCAgent.commands, BCAgent.observe = begin, commands, observe

    t_commands = tm.Teacher.commands

    def lessons(self, rows, player, now, t, can=None, pickers=()):
        before = dict(self.taught)
        if self.side == side:
            probe.state["reserved"] = self.reserved  # (--expand-lesson-reserve: the steps the hall's cost was kept)
        out = t_commands(self, rows, player, now, t, can, pickers)
        if self.side != side:
            return out
        fired = [k for k in KINDS if self.taught.get(k, 0) > before.get(k, 0)]
        ids = set(rows[:, fx.C_ID].tolist())
        probe.now_lessons = []
        for c in out:
            kind = None
            if isinstance(c, PointOrder) and c.order == ATTACK and KINDS[0] in fired:
                kind = KINDS[0]
            elif KINDS[1] in fired and (isinstance(c, Build) and c.building in tm.HALL_CODES.tolist()
                                        or isinstance(c, TargetOrder) and c.order == tm.HAUNTED
                                        or isinstance(c, PointOrder) and c.order == MOVE):
                kind = KINDS[1]
            elif isinstance(c, TargetOrder) and c.order == HARVEST and KINDS[2] in fired and c.target in ids:
                kind = KINDS[2]
            elif isinstance(c, PointOrder) and c.order == MOVE and KINDS[2] in fired:  # (to a mine out of its sight)
                kind = KINDS[2]
            elif isinstance(c, ImmediateOrder) and c.order == worker_t and KINDS[3] in fired:
                kind = KINDS[3]
            if kind is not None:
                probe.now_lessons.append((kind, c))
        watch(self, rows, player, now, t, can)
        return out

    def watch(self, rows, player, now, t, can):
        # the side's state, the natural's, and whether the hall could be paid (the lesson's own test is the mask's)
        sx, sy = player.start_x, player.start_y
        alive = fx.own_alive(rows, side)
        e = tm._expansion_facts(rows, alive, sx, sy)
        s = probe.state
        if e[0] >= 0:
            mx, my = float(e[2]), float(e[3])
            mines = rows[fx.isin(rows[:, fx.C_TYPE], tm.MINE_CODES)]
            s["nat"] = {"x": mx, "y": my, "mine_ids": set(mines[np.hypot(mines[:, fx.C_X] - mx, mines[:, fx.C_Y] - my) < 300, fx.C_ID].tolist())}
            camp, taken = bool(e[10]), bool(e[5])
            if self.expand_on and s.get("on_at") is None:
                army = tm._army_of(alive)
                heroes = alive[alive[:, fx.C_FLAGS] & fx.HERO > 0]
                s["on_at"] = round(now)
                s["at_trigger"] = {"army": Counter(fx.rawcode(int(v)) for v in army[:, fx.C_TYPE].tolist()).most_common(),
                                   "heroes": [(fx.rawcode(int(h[fx.C_TYPE])), int(h[fx.C_HLEVEL]), int(h[fx.C_HP]), int(h[fx.C_MAXHP])) for h in heroes],
                                   "workers": int((alive[:, fx.C_TYPE] == worker_t).sum()), "gold": player.gold, "lumber": player.lumber,
                                   "food": (player.food_used, player.food_cap)}
            if taken and s.get("taken_at") is None:
                s["taken_at"] = round(now)
            if not camp and s.get("camp_gone_at") is None:
                s["camp_gone_at"] = round(now)
            if self.expand_on and not camp and not taken:
                s["hall_steps"] = s.get("hall_steps", 0) + 1
                s["hall_steps_paid"] = s.get("hall_steps_paid", 0) + int(player.gold >= cost[0] and player.lumber >= cost[1])
                s["hall_steps_open"] = s.get("hall_steps_open", 0) + int(can is not None and any(can(hall_code, k) for k in (fx.POINT, fx.UNIT)))
            if self.expand_on and camp:
                s["camp_steps"] = s.get("camp_steps", 0) + 1
                why = "a lesson could be given"  # (why the camp's lesson was not: the teacher's own tests, again)
                heroes = alive[alive[:, fx.C_FLAGS] & fx.HERO > 0]
                creeps = rows[(rows[:, fx.C_OWNER] == 12) & (rows[:, fx.C_FLAGS] & (fx.DEAD | fx.STRUCTURE) == 0)]
                creeps = creeps[np.hypot(creeps[:, fx.C_X] - mx, creeps[:, fx.C_Y] - my) < tm.CAMP_NEAR]
                if not self._safe:
                    why = "an enemy army near its halls"
                elif not len(heroes):
                    why = "no living hero"
                else:
                    hx, hy = float(heroes[:, fx.C_X].mean()), float(heroes[:, fx.C_Y].mean())
                    fight = np.hypot(hx - float(creeps[:, fx.C_X].mean()), hy - float(creeps[:, fx.C_Y].mean())) < tm.EARLY_FIGHT
                    low, margin = (tm.EARLY_FIGHT_HP, 1.0) if fight else (tm.CREEP_HP, tm.EARLY_MARGIN)
                    s["camp_fight_steps"] = s.get("camp_fight_steps", 0) + int(fight)
                    if (heroes[:, fx.C_HP] <= low * np.maximum(heroes[:, fx.C_MAXHP], 1)).any():
                        why = "a hero low on hit points (under 60%, 30% in the fight)"
                    else:
                        army = tm._army_of(alive)
                        group = np.concatenate([heroes, army[np.hypot(army[:, fx.C_X] - hx, army[:, fx.C_Y] - hy) < tm.ESCORT]])
                        ratio = tm._power(group) / max(tm._power(creeps), 1e-6)
                        s.setdefault("ratios", []).append(round(ratio, 2))
                        if ratio < margin:
                            why = "too weak for the camp (under twice its strength, its strength in the fight)"
                s.setdefault("camp_why", Counter())[why] += 1
            if t % 120 == 0:
                army = tm._army_of(alive)
                ws = alive[alive[:, fx.C_TYPE] == worker_t]
                near = ws[np.hypot(ws[:, fx.C_X] - mx, ws[:, fx.C_Y] - my) < 900]
                heroes = alive[alive[:, fx.C_FLAGS] & fx.HERO > 0]
                line = {"min": round(now / 60, 1), "gold": player.gold, "lumber": player.lumber, "gathered": player.gold_gathered,
                        "food": player.food_used, "army": len(army), "heroes": [int(h[fx.C_HLEVEL]) for h in heroes], "workers": len(ws),
                        "at_natural": len(near), "camp": camp, "taken": taken, "on": bool(self.expand_on),
                        "taught": {k.split(": ")[1]: v for k, v in self.taught.items() if k in KINDS}}
                probe.minute.append(line)
                print("  " + json.dumps(line), flush=True)
    tm.Teacher.commands = lessons

    t_saving = tm.Teacher.saving

    def saving(self, rows, player, now, t):  # (--lessons 0: a side without lessons, the reserve's rule alone)
        if self.side == side:
            probe.state["reserved"] = self.reserved
        t_saving(self, rows, player, now, t)
        if self.side == side and player is not None:
            av = probe.mine.get("avail") if probe.mine is not None else None
            oix = probe.bot.enc.order_index
            watch(self, rows, player, now, t, None if av is None else
                  lambda u, k=fx.IMMEDIATE: (c := oix.get((fourcc(u), k))) is not None and bool(av[c]))
    tm.Teacher.saving = saving

    m_labels = sp.merge_labels

    def merge(step, y, teach=False):
        if teach:
            probe.on_labels(y)
        return m_labels(step, y, teach)
    sp.merge_labels = merge

    races = [a.opp, a.opp]
    races[side] = race
    slots = [None, None]
    slots[side] = Agent(race, handicap=cfg["handicap"], difficulty="normal")
    slots[1 - side] = BuiltinAI(a.opp, a.ai or "insane", handicap=cfg["handicap"])
    setup = GameSetup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"], max_game_seconds=cfg["max_minutes"] * 60,
                      victory="decisive", window=(320, 240), draw=False, wait_floor_ms=cfg["wait_floor_ms"], native_obs=True,
                      nice=19, d3d_thread=False, render_threads=0, warm_spare=False, cuts=cfg.get("game_cuts") or None,
                      obs_full=bool(cfg.get("obs_full")), seed=a.seed, **({"shim_dir": a.shim} if a.shim else {}))
    opp = {"name": f"script:ai-{a.ai or 'insane'} (real)", "kind": "ai", "advise": False}
    if a.scenario:  # (--scenario natural: the camp cleared at the first step, as a share of the training games start)
        cfg["scenarios"], cfg["scenario_share"] = [a.scenario], 1.0
        opp["name"] = f"script:ai-{a.ai or 'insane'}"  # (a "(real)" launch has no scenario)
    with GameInstance(setup, name=f"expandprobe{a.name}", timeout=600) as g:
        obs = g.start()
        if a.scenario == "natural" and not a.lessons:  # (a game without lessons has no scenario: its camps cleared here)
            from warcraftsim.fullgame import scenarios

            class First:
                """The game with the scenario's commands added to its first step's."""

                def __init__(self, game, extra):
                    self.game, self.extra = game, extra

                def step(self, cmds=None):
                    cmds, self.extra = list(cmds or []) + self.extra, []
                    return self.game.step(cmds)

                def __getattr__(self, name):
                    return getattr(self.game, name)
            g = First(g, scenarios.commands("natural", obs, side, races, random.Random(a.seed)))
        ep = sp.play_one(g, obs, cfg, vocab, races, side, opp, probe, probe, 0, record=False, train=bool(a.lessons))
    # the summary: per lesson kind, its dose and the policy's chances against the floors
    s, prod = probe.state, ep.get("prod") or {}
    if s.get("ratios"):  # (its heroes' group against the camp, when healthy and safe: the lowest, the median, the highest)
        s["ratios"] = [min(s["ratios"]), float(np.median(s["ratios"])), max(s["ratios"])]
    if "camp_why" in s:
        s["camp_why"] = dict(s["camp_why"])
    out = {"race": race, "opp": a.opp, "seed": a.seed, "follow": a.follow, "from": a.start, "floors": floors, "scenario": ep.get("scenario"),
           "state": {k: v for k, v in s.items() if k != "nat"}, "own": probe.own, "minutes": probe.minute,
           "outcome": ep.get("outcome"), "game_time": ep.get("game_time"), "taught": ep.get("taught"), "taught_kept": ep.get("taught_kept"),
           "expansions": prod.get("expansions"), "expansion_t": prod.get("expansion_t"), "kinds": {}}
    med = lambda v: float(np.median(v)) if len(v) else None  # noqa: E731
    for kind in KINDS:
        rs = [r for r in probe.records if r["kind"] == kind]
        kept = [r for r in rs if r["kept"]]
        po = [r["p_order"] for r in kept]
        pp = [r["p_place"] for r in kept if "p_place" in r]
        pn = [r["p_near"] for r in kept if "p_near" in r]
        pq = [r["p_ptr"] for r in kept if "p_ptr" in r]
        out["kinds"][kind] = {
            "lesson_steps": sum(1 for x in probe.steps if x["kind"] == kind), "unit_labels": len(rs), "kept": len(kept),
            "open": sum(1 for r in kept if r["open"]), "orders": Counter(r["order"] for r in kept).most_common(3),
            "p_order_median": med(po), "p_order_mean": float(np.mean(po)) if po else None,
            "under_order_floor": sum(1 for v in po if v < floors["order"]),
            "p_place_median": med(pp), "p_place_near_median": med(pn), "under_place_floor": sum(1 for v in pp if v < floors["place"]),
            "p_ptr_median": med(pq), "under_pointer_floor": sum(1 for v in pq if v < floors["pointer"]), "with_place": len(pp),
            "with_pointer": len(pq), "first_t": min((r["t"] for r in rs), default=None)}
    print(f"RESULT {race} vs {a.opp} ({a.ai or 'insane'}) seed {a.seed} follow {a.follow} scenario {ep.get('scenario')}: outcome {ep.get('outcome')} at "
          f"{(ep.get('game_time') or 0) / 60:.1f} min; trigger on at {s.get('on_at')} s with {s.get('at_trigger')}; camp gone at "
          f"{s.get('camp_gone_at')} s, natural taken at {s.get('taken_at')} s; steps with the camp standing {s.get('camp_steps', 0)} "
          f"({s.get('camp_why')}; its group's strength to the camp's {s.get('ratios')}), "
          f"with the hall due {s.get('hall_steps', 0)} (its cost in the bank {s.get('hall_steps_paid', 0)}, its order open "
          f"{s.get('hall_steps_open', 0)}, kept for it {s.get('reserved', 0)} steps); the policy's own: {probe.own}; expansions "
          f"{prod.get('expansions')} at {prod.get('expansion_t')}; "
          f"taught {ep.get('taught')} kept {ep.get('taught_kept')}", flush=True)
    for kind, v in out["kinds"].items():
        print(f"  {kind}: {v}", flush=True)
    got = [(m["min"], m["gathered"], m["workers"], m["at_natural"]) for m in probe.minute]
    out["gold_a_minute"] = [(b[0], b[1] - a_[1], b[2], b[3]) for a_, b in zip(got, got[1:])]
    print(f"  gold gathered in the minute up to (minute, gold, its workers, those at the natural): {out['gold_a_minute']}", flush=True)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(out, default=str) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("engine", "policy"))
    ap.add_argument("--race", default="human")
    ap.add_argument("--opp", default="orc")
    ap.add_argument("--ai", default="", help="the other side the built-in AI of this difficulty (engine: idle without)")
    ap.add_argument("--from", dest="start", type=float, default=360.0, help="--expand-lesson-from: game seconds")
    ap.add_argument("--minutes", type=float, default=14.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--side", type=int, default=0, help="the probed side's slot (engine: 0 starts left, 1 right)")
    ap.add_argument("--army", default="6", help="engine: the army made at --from (a count of the race's first unit, or hfoo:4,hrif:2)")
    ap.add_argument("--hero", default="", help="engine: the hero made (default: the race's usual first)")
    ap.add_argument("--hero-level", type=int, default=1)
    ap.add_argument("--camp", default="group", choices=("group", "heroes"), help="engine: who carries out the camp's lesson")
    ap.add_argument("--margin", type=float, default=None, help="engine: teacher.EARLY_MARGIN for this game")
    ap.add_argument("--need", type=int, default=None, help="engine: teacher.EARLY_ARMY for this game (a smaller army's test)")
    ap.add_argument("--lumber", type=int, default=None, help="engine: the script's workers on lumber beside the mine's 5 (default "
                    "7: a one-base economy of 12; 2: the trained policy's 7 workers)")
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--follow", type=int, default=0)
    ap.add_argument("--share", type=float, default=1.0)
    ap.add_argument("--pointer-floor", type=float, default=0.0, help="policy: --teach-floor-pointer (0: the run's place floor)")
    ap.add_argument("--scenario", default="", help="policy: the game starts in this scenario (natural: its camp cleared)")
    ap.add_argument("--reserve", type=int, default=0, help="policy: --expand-lesson-reserve (steps a game the hall's cost is kept)")
    ap.add_argument("--lessons", type=int, default=1, help="policy: 0: a game without lessons, as an evaluation game (the reserve's "
                    "rule alone: play_one's savers)")
    ap.add_argument("--set", nargs="*", default=[], help="policy: settings changed for this game: key=value")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--name", default="")
    ap.add_argument("--cuts", type=int, default=17592186041849, help="W3SIM_CUTS (the runs')")
    ap.add_argument("--shim", type=Path, default=None)
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    return engine(a) if a.what == "engine" else policy(a)


if __name__ == "__main__":
    raise SystemExit(main())
