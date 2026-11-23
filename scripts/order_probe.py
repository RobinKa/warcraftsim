"""What the game itself does with the orders a policy gives again and again (2026-11-23, bug finder 11; the engine
alone, no policy: one game on Echo Isles between two agent slots, units made with the debug commands on open ground).

    python scripts/order_probe.py [--only attack,balanced,dose,walk,smart,edge,spell] [--steps 60]

  attack  an attacker next to an enemy building: attack-moved once; attack-moved again every step (the same point);
          every step to points around it; ordered onto the building itself every step; a move and an attack-move in
          turn every step. The building's hit points lost in --steps steps: what the repeats cost.
  balanced  the same question with the places' differences taken out: once, every step, every 2 steps and every step
          to points around, each in each of four places in turn (the numbers to quote: a place where nothing can be
          hit shows in its own column; `attack` and `dose` put each way in one place only)
  dose    more unit types, the repeats every 1, 2, 4 and 8 steps, at four enemy peasants and at a barracks
  walk    footmen walk the same long way: moved once; moved again every step; every step to points around the goal;
          attack-moved every step. How far each is from the goal over time.
  smart   a peasant's and a footman's right-click on a point (order 851971) and the order 851973 (in the
          vocabulary from the built-in AI's games): whether the game takes them, what the unit then does.
  edge    a footman attack-moved, and a peasant moved, to a point above the map (the old point bins' top edge:
          y = the start's + 3048): whether the game takes it, where the unit goes.
  spell   an archmage's blizzard (a channel) on enemy peasants: left alone; given an attack-move at the next step;
          two and four steps later; blizzard again every step. The peasants' hit points lost, the mana spent. And a
          lich's frost nova from within its range and from outside it (a walk first): left alone, and given an
          attack-move at the next step.

What it found (2026-11-23; docs/reference/engine-notes.md has the table):
  * an attack-move given again starts the unit's swing over. Again every step (0.5 s), at a barracks, 30 s, summed
    over four places: a footman 0 of 593, an archer 0 of 368, a crypt fiend 0 of 461, a death knight 224 of 749, an
    archmage 339 of 695, a grunt 587 of 806, a mountain king 657 of 810, a demon hunter 1,173 of 1,386, a far seer
    593 of 665, a rifleman 455 of 457; every 2 steps 72-100%. An attack on the unit itself again every step costs
    nothing; a move and an attack-move in turn every step deals 0-30% (--order-masks bit 8: the repeats not sent)
  * a move again every step walks ~10% slower
  * smart on a point is refused (a peasant's, a footman's), on a unit taken; 851973 is refused on a point, an enemy
    building and an own unit
  * an order to a point above the map (y 5,992 at the left base) is taken: the unit walks ~1,700 north to the map's
    edge (-3,984, 4,641) and stands
  * a cast is cut off by the next order: a lich in range, frost nova, an attack-move a step later: no nova, no mana
    spent (176 damage left alone); a blizzard followed by a move a step later 132 damage, left alone 369
  * places 2 and 5 of its open ground are no good for a building's test (2: the barracks is out of reach, every way 0;
    5: something else hits it): `balanced` uses 0, 1, 3 and 4
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SMART, STOP, ATTACK, MOVE, HARVEST = 851971, 851972, 851983, 851986, 852018
BLIZZARD, FROSTNOVA = 852089, 852226
ORDER_NAMES = {0: "none", SMART: "smart", STOP: "stop", ATTACK: "attack", MOVE: "move", 851973: "851973", 851990: "patrol",
               HARVEST: "harvest", BLIZZARD: "blizzard", FROSTNOVA: "frostnova", 851988: "AImove", 851993: "holdposition"}


class Game:
    """The running game and its helpers: units made on open ground, found again by type and place."""

    def __init__(self, g, obs):
        self.g, self.obs = g, obs
        self.results: list = []

    def step(self, cmds=()):
        self.obs = self.g.step(list(cmds))
        self.results = list(self.obs.command_results)
        return self.obs

    def unit(self, uid: int):
        return next((u for u in self.obs.units if u.id == uid), None)

    def spawn(self, specs: list) -> list:
        """specs: (player, type, x, y[, hero level]) -> their ids (None: not made), found among the new units."""
        from warcraftsim.protocol import Spawn, SpawnHero
        before = {u.id for u in self.obs.units}
        cmds = [SpawnHero(p, t, x, y, lv[0]) if lv else Spawn(p, t, x, y) for p, t, x, y, *lv in specs]
        self.step(cmds)
        new = [u for u in self.obs.units if u.id not in before]
        out, used = [], set()
        for p, t, x, y, *_ in specs:
            cand = [u for u in new if u.owner == p and u.type == t and u.id not in used]
            u = min(cand, key=lambda u: (u.x - x) ** 2 + (u.y - y) ** 2, default=None)
            out.append(u.id if u is not None else None)
            if u is not None:
                used.add(u.id)
        return out

    def remove(self, ids) -> None:
        from warcraftsim.protocol import RemoveUnit
        self.step([RemoveUnit(i) for i in ids if i is not None])


def open_ground(game: Game, lanes: int, apart: float = 1500.0) -> list[tuple[float, float]]:
    """Points with room around them: a footman made there lands on the spot, no unit of the game's within 1400, no
    tree within 450, `apart` from each other."""
    from warcraftsim.protocol import Spawn
    obs = game.obs
    others = [(u.x, u.y) for u in obs.units if u.alive]
    trees = [(d.x, d.y) for d in (obs.destructables or [])]
    cand = [(float(x), float(y)) for x in range(-6600, 6601, 600) for y in range(-5000, 4601, 600)]
    cand = [c for c in cand if all((c[0] - x) ** 2 + (c[1] - y) ** 2 > 1400 ** 2 for x, y in others)
            and all((c[0] - x) ** 2 + (c[1] - y) ** 2 > 450 ** 2 for x, y in trees)]
    before = {u.id for u in obs.units}
    game.step([Spawn(0, "hfoo", x, y) for x, y in cand])
    new = [u for u in game.obs.units if u.id not in before]
    good = [c for c in cand if any(abs(u.x - c[0]) < 40 and abs(u.y - c[1]) < 40 for u in new)]
    game.remove([u.id for u in new])
    out: list = []
    for c in sorted(good, key=lambda c: abs(c[1] - 0.0) + abs(c[0])):  # (the middle first: far from both bases)
        if all((c[0] - x) ** 2 + (c[1] - y) ** 2 >= apart ** 2 for x, y in out):
            out.append(c)
        if len(out) >= lanes:
            break
    return out


KINDS = ("once", "every step", "every step, around", "on the unit every step", "move and attack in turn")
DOSE = ("once", "every step", "every 2 steps", "every 4 steps", "every 8 steps", "every step, around")


def attack(game: Game, lanes: list, steps: int, types: list, modes: tuple = KINDS, target: str = "hbar") -> None:
    """Per attacker type and mode: the hit points its targets lost (an enemy barracks, or with target "hpea" four enemy
    peasants side by side: units, which an attack-move takes before buildings)."""
    from warcraftsim.protocol import PointOrder, TargetOrder
    n_t = 1 if target == "hbar" else 4
    print(f"\n== attack: the hit points lost by {'an enemy barracks' if n_t == 1 else 'four enemy peasants'} in {steps} "
          f"steps ({steps * 0.5:.0f} s), by how the attacker is ordered (an attack-move at them)")
    for typ, level in types:
        use = lanes[:len(modes)]
        specs = []
        for x, y in use:
            specs += [(1, target, x + 260.0 + 50.0 * (k % 2), y + 50.0 * (k // 2)) for k in range(n_t)]
            specs.append((0, typ, x - 60.0, y, *([level] if level else [])))
        ids = game.spawn(specs)
        groups = [(ids[(n_t + 1) * k:(n_t + 1) * k + n_t], ids[(n_t + 1) * k + n_t]) for k in range(len(use))]
        hp0 = {t: game.unit(t).hp for ts, _ in groups for t in ts if t is not None}
        first: dict = {}

        def lost(ts) -> int:
            return sum(hp0[t] - (u.hp if (u := game.unit(t)) is not None and u.alive else 0) for t in ts if t is not None)
        for s in range(steps):
            cmds = []
            for k, (mode, (ts, atk)) in enumerate(zip(modes, groups)):
                if atk is None or ts[0] is None:
                    continue
                tx, ty = use[k][0] + 260.0, use[k][1]
                every = {"every step": 1, "every 2 steps": 2, "every 4 steps": 4, "every 8 steps": 8}.get(mode)
                if mode == "once":
                    if s == 0:
                        cmds.append(PointOrder(atk, ATTACK, tx, ty))
                elif every is not None:
                    if s % every == 0:
                        cmds.append(PointOrder(atk, ATTACK, tx, ty))
                elif mode == "every step, around":
                    a = 2 * math.pi * (s % 8) / 8
                    cmds.append(PointOrder(atk, ATTACK, tx + 96 * math.cos(a), ty + 96 * math.sin(a)))
                elif mode == "on the unit every step":
                    live = [t for t in ts if (u := game.unit(t)) is not None and u.alive]
                    if live:
                        cmds.append(TargetOrder(atk, ATTACK, live[0]))
                else:
                    cmds.append(PointOrder(atk, MOVE if s % 2 else ATTACK, tx, ty))
            game.step(cmds)
            for k, (ts, _) in enumerate(groups):
                if k not in first and lost(ts) > 0:
                    first[k] = s
        row = [f"{mode}: {lost(ts)} (first hit at step {first.get(k)})" for k, (mode, (ts, _)) in enumerate(zip(modes, groups))]
        print(f"  {typ}{' level ' + str(level) if level else ''}: " + "; ".join(row), flush=True)
        game.remove([i for ts, atk in groups for i in (*ts, atk)])


def balanced(game: Game, lanes: list, steps: int, types: list) -> None:
    """The attack-move's repeats without the places' differences: each way of ordering in each of four places in turn
    (a round each), an enemy barracks' hit points lost summed over the rounds; and per place, so that a place where
    nothing can be hit shows."""
    from warcraftsim.protocol import PointOrder
    modes = ("once", "every step", "every 2 steps", "every step, around")
    use = [lanes[k] for k in (0, 1, 3, 4) if k < len(lanes)]
    if len(use) < len(modes):
        print("\n== balanced: not enough open points")
        return
    print(f"\n== balanced: an enemy barracks' hit points lost in {steps} steps ({steps * 0.5:.0f} s) by how its attacker "
          f"is attack-moved at it, summed over {len(modes)} rounds (each way in each of {len(use)} places once); "
          f"[per round: the place's number]")
    for typ, level in types:
        total = {m: 0 for m in modes}
        per = {m: [] for m in modes}
        for rnd in range(len(modes)):
            where = {m: use[(k + rnd) % len(use)] for k, m in enumerate(modes)}
            specs = []
            for m in modes:
                x, y = where[m]
                specs += [(1, "hbar", x + 260.0, y), (0, typ, x - 60.0, y, *([level] if level else []))]
            ids = game.spawn(specs)
            pair = {m: (ids[2 * k], ids[2 * k + 1]) for k, m in enumerate(modes)}
            hp0 = {m: game.unit(pair[m][0]).hp for m in modes if pair[m][0] is not None}
            for s in range(steps):
                cmds = []
                for m in modes:
                    tgt, atk = pair[m]
                    if tgt is None or atk is None:
                        continue
                    tx, ty = where[m][0] + 260.0, where[m][1]
                    if m == "every step, around":
                        a = 2 * math.pi * (s % 8) / 8
                        cmds.append(PointOrder(atk, ATTACK, tx + 96 * math.cos(a), ty + 96 * math.sin(a)))
                    elif s % {"once": 10 ** 9, "every step": 1, "every 2 steps": 2}[m] == 0:
                        cmds.append(PointOrder(atk, ATTACK, tx, ty))
                game.step(cmds)
            for k, m in enumerate(modes):
                u = game.unit(pair[m][0]) if pair[m][0] is not None else None
                lost = hp0[m] - u.hp if u is not None and m in hp0 else 0
                total[m] += lost
                per[m].append(f"{lost}@{use.index(where[m])}")
            game.remove([i for p in pair.values() for i in p])
        print(f"  {typ}{' level ' + str(level) if level else '':<8} " + "; ".join(f"{m}: {total[m]} {per[m]}" for m in modes),
              flush=True)


def walk(game: Game, lanes: list, steps: int) -> None:
    from warcraftsim.protocol import PointOrder
    if len(lanes) < 2:
        print("\n== walk: no two open points")
        return
    a = lanes[0]
    b = max(lanes[1:], key=lambda c: (c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2)
    modes = ("moved once", "moved every step", "moved every step, around", "attack-moved every step", "attack-moved once")
    ids = game.spawn([(0, "hfoo", a[0] + 40.0 * k, a[1]) for k in range(len(modes))])
    print(f"\n== walk: footmen from {a} to {b} ({math.hypot(b[0] - a[0], b[1] - a[1]):.0f} away): the distance left, every 10 steps")
    left = {m: [] for m in modes}
    for s in range(steps):
        cmds = []
        for mode, u in zip(modes, ids):
            if u is None:
                continue
            ang = 2 * math.pi * (s % 8) / 8
            if mode == "moved once" and s == 0:
                cmds.append(PointOrder(u, MOVE, *b))
            elif mode == "moved every step":
                cmds.append(PointOrder(u, MOVE, *b))
            elif mode == "moved every step, around":
                cmds.append(PointOrder(u, MOVE, b[0] + 96 * math.cos(ang), b[1] + 96 * math.sin(ang)))
            elif mode == "attack-moved every step":
                cmds.append(PointOrder(u, ATTACK, *b))
            elif mode == "attack-moved once" and s == 0:
                cmds.append(PointOrder(u, ATTACK, *b))
        game.step(cmds)
        if s % 10 == 9:
            for mode, u in zip(modes, ids):
                x = game.unit(u)
                left[mode].append(round(math.hypot(x.x - b[0], x.y - b[1])) if x is not None else None)
    for mode in modes:
        print(f"  {mode:<28} {left[mode]}", flush=True)
    game.remove(ids)


def smart(game: Game, lanes: list) -> None:
    from warcraftsim.protocol import PointOrder, TargetOrder
    x, y = lanes[0]
    pea, foo, foe = game.spawn([(0, "hpea", x, y), (0, "hfoo", x + 80.0, y), (1, "hbar", x + 500.0, y)])
    print("\n== smart: whether the game takes the order, and the unit's current order a step later")

    def try_(what: str, cmd, unit: int) -> None:
        game.step([cmd])
        ok = game.results[:1]
        game.step([])
        u = game.unit(unit)
        print(f"  {what:<46} taken={ok} then reads {ORDER_NAMES.get(u.order, u.order) if u else None}", flush=True)
        game.step([PointOrder(unit, MOVE, x, y)])
        for _ in range(4):
            game.step([])
    try_("peasant: smart on a point", PointOrder(pea, SMART, x + 300.0, y + 100.0), pea)
    try_("footman: smart on a point", PointOrder(foo, SMART, x + 300.0, y + 100.0), foo)
    try_("peasant: smart on the enemy building", TargetOrder(pea, SMART, foe), pea)
    try_("footman: smart on the enemy building", TargetOrder(foo, SMART, foe), foo)
    try_("peasant: 851973 on a point", PointOrder(pea, 851973, x + 300.0, y + 100.0), pea)
    try_("footman: 851973 on a point", PointOrder(foo, 851973, x + 300.0, y + 100.0), foo)
    try_("peasant: 851973 on the enemy building", TargetOrder(pea, 851973, foe), pea)
    try_("footman: 851973 on the enemy building", TargetOrder(foo, 851973, foe), foo)
    try_("footman: 851973 on the peasant", TargetOrder(foo, 851973, pea), foo)
    game.remove([pea, foo, foe])


def edge(game: Game, steps: int) -> None:
    from warcraftsim.protocol import PointOrder
    obs = game.obs
    p = obs.players[0]
    sx, sy = float(p.start_x), float(p.start_y)
    top = sy + 3048.0  # (the old bins' top edge: bin 239 of 352, features.bins_world)
    foo, pea, far = game.spawn([(0, "hfoo", sx + 500.0, sy - 300.0), (0, "hpea", sx + 560.0, sy - 300.0),
                                (0, "hfoo", sx + 548.0, sy - 5900.0)])
    print(f"\n== edge: orders to a point above the map (y {top:.0f}; the start {sx:.0f}, {sy:.0f})")
    game.step([PointOrder(foo, ATTACK, sx + 548.0, top), PointOrder(pea, MOVE, sx + 548.0, top),
               PointOrder(far, ATTACK, sx + 548.0, top)])
    print("  taken:", game.results[:3])
    for s in range(steps * 3):
        game.step([])
        if s % 20 == 19:
            line = []
            for name, u in (("footman at the base", foo), ("peasant at the base", pea), ("footman 5900 south", far)):
                x = game.unit(u)
                line.append(f"{name}: ({x.x:.0f}, {x.y:.0f}) {ORDER_NAMES.get(x.order, x.order)}" if x is not None else f"{name}: gone")
            print(f"  step {s + 1:3d}: " + "; ".join(line), flush=True)
    game.remove([foo, pea, far])


def spell(game: Game, lanes: list) -> None:
    from warcraftsim.protocol import LearnSkill, PointOrder, TargetOrder
    print("\n== spell: an archmage's blizzard on six enemy peasants 500 away (their hit points lost, the mana spent)")
    modes = ("left alone", "attack-move at the next step", "attack-move 2 steps later", "attack-move 4 steps later",
             "blizzard again every step", "a move at the next step")
    use = lanes[:len(modes)]
    specs = []
    for x, y in use:
        specs.append((0, "Hamg", x, y, 2))
        specs += [(1, "hpea", x + 500.0 + 40.0 * (k % 3), y + 40.0 * (k // 3)) for k in range(6)]
    ids = game.spawn(specs)
    heroes = [ids[7 * k] for k in range(len(use))]
    peas = [ids[7 * k + 1:7 * k + 7] for k in range(len(use))]
    game.step([LearnSkill(h, "AHbz") for h in heroes if h is not None])
    print("  learned:", game.results[:len(heroes)])
    game.step([])
    hp0 = [sum(game.unit(p).hp for p in ps if p is not None and game.unit(p) is not None) for ps in peas]
    mana0 = [game.unit(h).mana if h is not None else 0 for h in heroes]
    taken = {}
    for s in range(24):
        cmds, who = [], []
        for k, (mode, h) in enumerate(zip(modes, heroes)):
            if h is None:
                continue
            tx, ty = use[k][0] + 540.0, use[k][1] + 20.0
            hx, hy = use[k]
            if s == 0 or (mode == "blizzard again every step"):
                cmds.append(PointOrder(h, BLIZZARD, tx, ty))
                who.append((mode, s))
            elif (mode, s) in (("attack-move at the next step", 1), ("attack-move 2 steps later", 2),
                               ("attack-move 4 steps later", 4)):
                cmds.append(PointOrder(h, ATTACK, hx - 200.0, hy))
                who.append((mode, s))
            elif mode == "a move at the next step" and s == 1:
                cmds.append(PointOrder(h, MOVE, hx - 200.0, hy))
                who.append((mode, s))
        game.step(cmds)
        for (mode, s_), ok in zip(who, game.results):
            taken.setdefault(mode, []).append(int(bool(ok)))
    for k, mode in enumerate(modes):
        hp = sum(game.unit(p).hp for p in peas[k] if p is not None and game.unit(p) is not None)
        h = game.unit(heroes[k]) if heroes[k] is not None else None
        print(f"  {mode:<30} hit points lost {hp0[k] - hp:5d}, mana spent {mana0[k] - (h.mana if h else 0):4d}, "
              f"orders taken {taken.get(mode)}", flush=True)
    game.remove([i for i in ids if i is not None])
    print("== spell: a lich's frost nova on an enemy peasant (its hit points lost, the mana spent)")
    cases = (("in range (600), left alone", 600.0, None), ("in range (600), attack-move at the next step", 600.0, 1),
             ("a walk first (1100), left alone", 1100.0, None), ("a walk first (1100), attack-move at the next step", 1100.0, 1),
             ("a walk first (1100), attack-move 3 steps later", 1100.0, 3))
    use = lanes[:len(cases)]
    specs = []
    for (name, dist, _), (x, y) in zip(cases, use):
        specs += [(0, "Ulic", x, y, 2), (1, "hpea", x + dist, y)]
    ids = game.spawn(specs)
    game.step([LearnSkill(ids[2 * k], "AUfn") for k in range(len(use)) if ids[2 * k] is not None])
    game.step([])
    hp0 = [game.unit(ids[2 * k + 1]).hp for k in range(len(use))]
    mana0 = [game.unit(ids[2 * k]).mana for k in range(len(use))]
    for s in range(16):
        cmds = []
        for k, ((name, dist, when), (x, y)) in enumerate(zip(cases, use)):
            if s == 0:
                cmds.append(TargetOrder(ids[2 * k], FROSTNOVA, ids[2 * k + 1]))
            elif when is not None and s == when:
                cmds.append(PointOrder(ids[2 * k], ATTACK, x - 200.0, y))
        game.step(cmds)
        if s == 0:
            print("  taken:", game.results[:len(use)])
    for k, (name, _, _) in enumerate(cases):
        t, h = game.unit(ids[2 * k + 1]), game.unit(ids[2 * k])
        print(f"  {name:<48} hit points lost {hp0[k] - (t.hp if t else 0):4d}, mana spent {mana0[k] - (h.mana if h else 0):4d}",
              flush=True)
    game.remove([i for i in ids if i is not None])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="attack,walk,smart,edge,spell")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--cuts", type=int, default=17592186041849, help="W3SIM_CUTS (the runs')")
    ap.add_argument("--shim", type=Path, default=None)
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup
    slots = [Agent("human", handicap=100, difficulty="normal"), Agent("orc", handicap=100, difficulty="normal")]
    gs = GameSetup(map="(2)EchoIsles", slots=slots, step_seconds=0.5, draw=False, native_obs=False, d3d_thread=False,
                   render_threads=0, nice=19, warm_spare=False, window=(320, 240), max_game_seconds=0, victory="none",
                   cuts=a.cuts, **({"shim_dir": a.shim} if a.shim else {}))
    only = set(a.only.split(","))
    with GameInstance(gs, name="orderprobe", timeout=600) as g:
        game = Game(g, g.start())
        lanes = open_ground(game, 6)
        print("open ground:", lanes, flush=True)
        if not lanes:
            return 1
        if "attack" in only:
            attack(game, lanes, a.steps, [("hfoo", 0), ("Udea", 1), ("earc", 0), ("Ofar", 1), ("ugho", 0)])
        if "dose" in only:  # (how often the attack-move comes again, on units and on a building; more types)
            types = [("hfoo", 0), ("hrif", 0), ("ogru", 0), ("ohun", 0), ("ucry", 0), ("earc", 0), ("esen", 0), ("Hamg", 1),
                     ("Hmkg", 1), ("Obla", 1), ("Ofar", 1), ("Udea", 1), ("Ulic", 1), ("Edem", 1), ("Ekee", 1)]
            attack(game, lanes, a.steps, types, DOSE, "hpea")
            attack(game, lanes, a.steps, types[:3] + types[7:9], DOSE, "hbar")
        if "balanced" in only:  # (the repeats' cost with the places' differences taken out)
            balanced(game, lanes, a.steps, [("hfoo", 0), ("hrif", 0), ("ogru", 0), ("earc", 0), ("ucry", 0), ("Udea", 1),
                                            ("Hamg", 1), ("Ofar", 1), ("Hmkg", 1), ("Edem", 1)])
        if "walk" in only:
            walk(game, lanes, a.steps)
        if "smart" in only:
            smart(game, lanes)
        if "edge" in only:
            edge(game, a.steps)
        if "spell" in only:
            spell(game, lanes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
