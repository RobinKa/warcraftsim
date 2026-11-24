"""The second base's lessons (--expand-lesson-from: teacher.py's expansion lessons from a game second on, in games
without a target), checked in the game itself (2026-11-24). Real Echo Isles, ONE game at a time, no learner, no GPU.

    python scripts/expand_probe.py engine --race human [--from 360] [--army 6] [--hero-level 1] [--camp group|heroes]
        [--minutes 14] [--refill 1] [--seed 1] [--ai insane]
    python scripts/expand_probe.py policy --race human --weights runs/fgself-252/averaged/swa-17-27.pt --run runs/fgself-253
        [--from 360] [--minutes 12] [--follow 0|1] [--opp orc] [--ai insane] [--seed 1] [--json out.json]

engine  No policy. A scripted side (its workers on the main's gold and lumber, its hall training workers up to the
        race's count) gets, at --from, what the trigger asks for: a hero (--hero, level --hero-level, no skills) and
        --army army units (the race's first unit, or "hfoo:4,hrif:2") beside its hall. A Teacher with --expand-lesson-from
        looks on and its lessons are CARRIED OUT as the orders they name: the camp's attack-move, the hall's build order
        (undead: the haunt), the workers' harvest. The other side gives no orders (--ai: the built-in AI plays it).
        Per game: the camp (dead when, units lost, the heroes' hit points), the hall (order taken, started, finished,
        where), the workers at the natural, the gold gathered a minute before and after. --camp heroes: the camp
        lesson's orders for the heroes alone (the target games' camp lesson). --refill 1: once the natural stands the
        script also trains workers until both mines have 5 (not a lesson's: what the second base is worth filled).
policy  The trained policy on the CPU plays a real game without a target (the run's settings from its run.json, read
        only; selfplay.play_one with its teachers), --expand-lesson-from on. Logged per lesson step: how many units it
        labels, and the policy's chance of the labelled order, of the place given the order, of the pointer given the
        order (against the floors: would the lesson push?). And whether the policy expands by itself. --follow 1: the
        labelled units are given the lessons' orders (the labels' classes, bins and pointers through the agent's own
        command path) and left alone until the next lesson: what the policy's game looks like when it does as taught.
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
    """The scripted side's economy: 5 workers on the main's gold, ON_LUMBER on the trees (undead: its ghoul; SPARE idle
    acolytes), the hall training until it has them. --refill: once the natural stands, workers trained until both mines
    have 5."""

    def __init__(self, obs, side: int, race: str, refill: bool):
        self.side, self.race, self.refill = side, race, refill
        p = obs.players[side]
        self.start = (p.start_x, p.start_y)
        self.worker = WORKER[race]
        self.job: dict[int, str] = {}
        self.pending = 0
        self.asked = -100
        self.trees = sorted(obs.destructables or [], key=lambda d: (d.x - p.start_x) ** 2 + (d.y - p.start_y) ** 2)
        self.next_tree = 0
        self.natural_mine = None  # (the id of the mine its natural's workers harvest: the probe sets it)
        self.natural_hall = None

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
            if self.pending:
                self.pending -= 1
            if count["gold"] < 5 and mine is not None:
                self.job[u.id] = "gold"
                out.append(TargetOrder(u.id, HARVEST, mine.id))
            elif count["lumber"] < ON_LUMBER[self.race] and self.trees:
                self.job[u.id] = "lumber"
                out.append(TargetDestructable(u.id, HARVEST, self.trees[self.next_tree % len(self.trees)].id))
                self.next_tree += 2
            elif self.refill and self.natural_mine is not None and count["natural"] < 5:
                self.job[u.id] = "natural"
                out.append(TargetOrder(u.id, HARVEST, self.natural_mine))
            else:
                self.job[u.id] = "spare"
            count[self.job[u.id]] += 1
        if t == 1 and self.race == "undead":  # (its ghoul chops)
            out += [TargetDestructable(u.id, HARVEST, self.trees[0].id) for u in own if u.type == "ugho"]
        if t == 1:  # (food for its workers and the army made later: three supply buildings, made, not built)
            from warcraftsim.protocol import Spawn
            sx, sy = self.start
            out += [Spawn(side, FARM[self.race], sx + (700.0 if sx < 0 else -700.0), sy - 500.0 + 350.0 * k) for k in range(3)]
        want = 5 + ON_LUMBER[self.race] + SPARE.get(self.race, 0)
        if self.refill and self.natural_mine is not None:
            want = len(workers) + (5 - count["gold"]) + (5 - count["natural"])
        p = obs.players[side]
        halls = [u for u in own if u.is_structure and u.type in ("htow", "hkee", "ogre", "ostr", "unpl", "unp1", "etol", "etoa")
                 and not int(u.flags) & 8]
        if len(workers) + self.pending < want and halls and p.gold >= 90 and p.food_used < p.food_cap and t - self.asked >= 4:
            hall = halls[-1] if (self.refill and count["gold"] >= 5 and len(halls) > 1) else halls[0]
            out.append(ImmediateOrder(hall.id, fourcc(self.worker)))
            self.pending += 1
            self.asked = t
        return out


def engine(a) -> int:
    from warcraftsim.data.objects import unit_table
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame import teacher as tm
    from warcraftsim.fullgame.play import unit_rows
    from warcraftsim.protocol import Build, PointOrder, Spawn, SpawnHero, TargetOrder
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
        eco = Economy(obs, side, race, bool(a.refill))
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
                elif isinstance(c, TargetOrder) and c.order == HARVEST:
                    k = "workers"
                    first_workers = now if first_workers is None else first_workers
                    eco.job[c.unit] = "natural"
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
                if k in ("camp", "hall", "workers", "entangle", "spawn"):
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
            own_mine = next((u for u in holds if u.type in ("ugol", "egol") and not int(u.flags) & 8), None)
            if hall_done is not None and eco.natural_mine is None:
                if race in OWN_MINE:
                    eco.natural_mine = own_mine.id if own_mine is not None else None
                else:
                    eco.natural_mine = nat.id
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


def policy(a) -> int:
    raise SystemExit("policy: not yet")


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
    ap.add_argument("--refill", type=int, default=0)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--follow", type=int, default=0)
    ap.add_argument("--share", type=float, default=1.0)
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
