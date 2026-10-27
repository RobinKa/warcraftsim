"""Which of undead's orders the game refuses, and why (audit 17: fgself-158's undead games against insane wasted orders
on acolytes sent to a full haunted mine, Death Coil on own units at full health, the heroes' item slots and burrowed
crypt fiends): each check sends one order and prints the game's answer and the state that decides it. Results
2026-10-27: features.FUTILE_MASK's comment (and the policy's own real games: the --econ-masks 64 commit's
evaluation worker runs with each refusal's context). The spells' answers did not settle the rule (a fresh Death
Knight's coil on its own full-health ghoul refused in one run, taken in another; Dark Ritual refused at full mana,
cannibalize refused even hurt beside a fresh corpse): the Death Coil rule follows fgself-158's refusal records.

    python3 scripts/undead_probe.py --part harvest   # acolytes at a haunted mine: how many it takes, and when
    python3 scripts/undead_probe.py --part spells    # Death Coil, Dark Ritual, cannibalize on each kind of target
    python3 scripts/undead_probe.py --part items     # each ware's immediate use from an inventory slot

Instance names start with "undeadprobe".
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("WAYLAND_DISPLAY", None)  # (games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"

from warcraftsim.protocol import (Buy, ImmediateOrder, LearnSkill, PointOrder, SetResources, Spawn, SpawnHero,  # noqa: E402
                                  TargetOrder, UseItem, fourcc)
from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup  # noqa: E402

HARVEST, MOVE, ATTACK, STOP, HOLD = 852018, 851986, 851983, 851972, 851993
DEATHCOIL, DARKRITUAL, FROSTNOVA, CANNIBALIZE, USESLOT = 852222, 852219, 852226, 852188, 852008
HARVESTING = (852018, 852017, 852020, 851971)
MERCHANT_ITEMS = ("stwp", "bspd", "dust", "tret", "prvt", "cnob", "stel", "pnvl", "shea", "spro", "pinv")
TOMB_ITEMS = ("rnec", "skul", "dust", "ocor", "stwp", "phea", "pman", "sreg", "rhth", "ritd", "shea", "pinv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="(2)EchoIsles")
    ap.add_argument("--name", default="undeadprobe0")
    ap.add_argument("--part", default="harvest", choices=("harvest", "spells", "items"))
    ap.add_argument("--shim", default="/home/tora/warcraftsim/build/shim")
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    setup = GameSetup(map=a.map, slots=[Agent("undead"), Agent("orc")], step_seconds=0.5, draw=False,
                      native_obs=False, d3d_thread=False, render_threads=0, nice=10, warm_spare=False,
                      window=(320, 240), max_game_seconds=0, shim_dir=a.shim)
    g = GameInstance(setup, name=a.name, timeout=300)
    obs = g.start()

    def step(cmds=()):
        nonlocal obs
        obs = g.step(list(cmds))
        return obs.command_results

    def wait(n):
        for _ in range(n):
            step()

    def own(kind, p=0):
        return [u for u in obs.units if u.owner == p and u.type == kind and u.alive]

    def u(uid):
        return obs.unit(uid)

    def show(uid):
        x = u(uid)
        return "gone" if x is None or not x.alive else (f"{x.type} hp {x.hp}/{x.max_hp} mp {x.mana}/{x.max_mana} "
                                                         f"order {x.order} flags {int(x.flags)}")

    def check(label, cmd, then=0):
        res = step([cmd])
        print(f"{label:58s} {'TAKEN  ' if res and res[-1] else 'REFUSED'}", flush=True)
        wait(then)
        return bool(res and res[-1])

    p0, p1 = obs.players[0], obs.players[1]
    sx, sy = p0.start_x, p0.start_y
    hall = own("unpl")[0]
    mine = own("ugol")[0]
    print(f"necropolis {hall.id} at {hall.x},{hall.y}; haunted mine {mine.id} at {mine.x},{mine.y} hp {mine.hp}/{mine.max_hp}",
          flush=True)
    if a.part == "harvest":
        step([Spawn(0, "uaco", hall.x - 300 + 60 * k, hall.y - 300) for k in range(5)]
             + [Spawn(0, "uaco", hall.x + 900, hall.y + 900 + 60 * k) for k in range(3)])
        step()
        acos = [x.id for x in own("uaco")]
        far = acos[-3:]
        near = acos[:-3]
        print(f"acolytes near {near} far {far}", flush=True)

        def at_mine():
            out = []
            for x in own("uaco"):
                d = math.hypot(x.x - mine.x, x.y - mine.y)
                out.append((x.id, x.order, int(x.flags), int(d)))
            counted = sum(1 for _, o, f, d in out if d < 350 and (o in HARVESTING or f & 16))
            return counted, out

        res = step([TargetOrder(k, HARVEST, mine.id) for k in near[:5]])
        print(f"5 harvests: {['TAKEN' if r else 'REFUSED' for r in res]}", flush=True)
        wait(24)
        n, rows = at_mine()
        print(f"after 12 s: play-counted miners {n}; acolytes (id, order, flags, distance): {rows}", flush=True)
        check("6th harvest (near, idle), 5 mining", TargetOrder(near[5], HARVEST, mine.id), 6)
        print(f"   6th: {show(near[5])}", flush=True)
        step([PointOrder(near[0], MOVE, hall.x - 900, hall.y - 900)])
        wait(4)
        n, _ = at_mine()
        print(f"one miner sent away: play-counted miners {n}", flush=True)
        check("far acolyte's harvest, 4 mining", TargetOrder(far[0], HARVEST, mine.id))
        n, _ = at_mine()
        print(f"   play-counted miners {n} (the far one walking: {int(math.hypot(u(far[0]).x - mine.x, u(far[0]).y - mine.y))} away)",
              flush=True)
        check("2nd far acolyte's harvest, 4 mining + 1 walking", TargetOrder(far[1], HARVEST, mine.id))
        check("near idle acolyte's harvest, 4 mining + 1 walking", TargetOrder(near[5], HARVEST, mine.id))
        wait(30)
        n, rows = at_mine()
        print(f"after 15 s: play-counted miners {n}; {rows}", flush=True)
        check("the miner sent away: its harvest back", TargetOrder(near[0], HARVEST, mine.id), 2)
        check("a miner's own harvest again (mining)", TargetOrder(near[1], HARVEST, mine.id), 2)
        # (a damaged mine: the extras away, taurens hit the mine, then leave)
        extras = [x.id for x in own("uaco") if x.order not in HARVESTING]
        step([PointOrder(x, MOVE, hall.x + 1200, hall.y + 1200) for x in extras])
        wait(10)
        step([Spawn(1, "otau", mine.x + 250, mine.y + 150 + 60 * k) for k in range(4)])
        step()
        tau = [x.id for x in own("otau", 1)]
        for k in range(80):
            step([TargetOrder(x, ATTACK, mine.id) for x in tau] if k % 4 == 0 else [])
            if u(mine.id).hp < 0.7 * u(mine.id).max_hp:
                break
        step([PointOrder(x, MOVE, mine.x + 3000, mine.y - 3000) for x in tau])
        wait(8)
        n, rows = at_mine()
        print(f"mine {show(mine.id)}; play-counted miners {n}; {rows}", flush=True)
        miners = [i for i, o, f, d in rows if o in HARVESTING and d < 350]
        if len(miners) >= 5:
            step([PointOrder(miners[0], MOVE, hall.x - 900, hall.y - 900)])
            wait(4)
        n, rows = at_mine()
        print(f"   play-counted miners {n}", flush=True)
        if check("A idle acolyte's harvest, mine damaged, 4 mining, no restorers",
                 TargetOrder(extras[0], HARVEST, mine.id), 2):
            step([PointOrder(extras[0], MOVE, hall.x + 1200, hall.y + 1200)])
        step([PointOrder(x, MOVE, mine.x + 150, mine.y - 200) for x in extras[1:3]])
        for _ in range(40):
            step()
            if all(u(x).order == 852202 for x in extras[1:3]):
                break
        n, rows = at_mine()
        print(f"mine {show(mine.id)}; play-counted miners {n}; {rows}", flush=True)
        check("B idle acolyte's harvest, mine damaged, 4 mining, restorers near",
              TargetOrder(extras[3], HARVEST, mine.id), 2)
        check("C a restorer's harvest, mine damaged, 4 mining", TargetOrder(extras[1], HARVEST, mine.id), 8)
        n, rows = at_mine()
        print(f"mine {show(mine.id)}; play-counted miners {n}; {rows}", flush=True)
    elif a.part == "spells":  # (the enemies in sight: an unseen unit takes no spell; a fresh caster for each check)
        px, py = hall.x + 500, hall.y + 500
        step([Spawn(0, "ugho", px, py + 150), Spawn(0, "ugho", px + 120, py + 150), Spawn(0, "uabo", px - 120, py + 150),
              SpawnHero(0, "Ulic", px - 200, py - 200, 1)])
        step()
        mine_gh = [x.id for x in own("ugho") if math.hypot(x.x - px, x.y - py) < 400]
        abo = own("uabo")[0].id
        aco = own("uaco")[0].id
        other_lich = own("Ulic")[0].id
        step([Spawn(1, "ogru", px + 350, py), Spawn(1, "ugho", px + 350, py + 120), Spawn(1, "opeo", px + 350, py - 120),
              Spawn(1, "opeo", px + 250, py - 160)])
        step()
        gr = [x.id for x in own("ogru", 1)][0]
        eg = [x.id for x in own("ugho", 1)][0]
        pes = [x.id for x in own("opeo", 1) if math.hypot(x.x - px, x.y - py) < 600]
        step([ImmediateOrder(x, HOLD) for x in [gr, eg] + pes])

        def caster(kind, skill):
            step([SpawnHero(0, kind, px, py, 3)])
            step()
            h = [x for x in own(kind) if x.id != other_lich][-1].id
            step([LearnSkill(h, fourcc(skill))])
            wait(2)
            return h

        def coil(label, target):
            h = caster("Udea", "AUdc")
            check(f"deathcoil: {label}", TargetOrder(h, DEATHCOIL, h if target == "self" else target), 8)
            print(f"   dk {show(h)} abilities {u(h).abilities}", flush=True)
            step([PointOrder(h, MOVE, hall.x - 400, hall.y - 400)])

        print(f"enemies seen by 0: {[u(x).visible(0) for x in [gr, eg] + pes]}", flush=True)
        coil("own ghoul, full health", mine_gh[0])
        coil("own lich (hero), full health", other_lich)
        coil("itself", "self")
        coil("enemy ghoul (undead)", eg)
        coil("enemy grunt (living)", gr)
        coil("own acolyte, full health", aco)
        coil("own necropolis", hall.id)
        step([TargetOrder(gr, ATTACK, mine_gh[0])])
        for _ in range(30):
            step()
            if u(mine_gh[0]).hp < 0.8 * u(mine_gh[0]).max_hp:
                break
        step([ImmediateOrder(gr, HOLD), ImmediateOrder(mine_gh[0], STOP)])
        print(f"   ghoul hurt: {show(mine_gh[0])}", flush=True)
        coil("own ghoul, damaged", mine_gh[0])
        print(f"   ghoul {show(mine_gh[0])}", flush=True)
        for label, target in (("own ghoul full health", mine_gh[1]), ("own acolyte", aco), ("enemy ghoul", eg)):
            h = caster("Ulic", "AUdr")
            check(f"darkritual: {label}, lich full mana", TargetOrder(h, DARKRITUAL, target), 6)
            print(f"   lich {show(h)} abilities {u(h).abilities}; target {show(target)}", flush=True)
        step([TargetOrder(gr, ATTACK, abo)])
        for _ in range(30):
            step()
            if u(abo).hp < 0.9 * u(abo).max_hp:
                break
        step([ImmediateOrder(gr, HOLD), ImmediateOrder(abo, STOP)])
        check(f"cannibalize: abomination damaged {show(abo)}, no corpse", ImmediateOrder(abo, CANNIBALIZE), 2)
        step([TargetOrder(x, ATTACK, pes[0]) for x in (mine_gh[1],)])
        for _ in range(40):
            step()
            if u(pes[0]) is None or not u(pes[0]).alive:
                break
        step([ImmediateOrder(mine_gh[1], STOP)])
        wait(8)
        for x in (mine_gh[1], mine_gh[2] if len(mine_gh) > 2 else mine_gh[0], abo):
            st = show(x)
            check(f"cannibalize: {st}, a corpse near", ImmediateOrder(x, CANNIBALIZE), 6)
            print(f"   {show(x)}", flush=True)
    else:  # (one hero beside a tomb of relics and a goblin merchant of the player's base, one ware at a time)
        step([SetResources(0, 20000, 20000), Spawn(0, "utom", hall.x + 600, hall.y - 600),
              Spawn(0, "hvlt", hall.x + 850, hall.y - 600), Spawn(0, "ovln", hall.x + 600, hall.y - 850),
              Spawn(15, "ngme", hall.x + 850, hall.y - 850)])
        wait(4)
        tomb = own("utom")[0]
        merch = [x for x in obs.units if x.type in ("ngme", "hvlt", "ovln") and math.hypot(x.x - tomb.x, x.y - tomb.y) < 600]
        print(f"tomb {tomb.id}; merchant beside it {[m.id for m in merch]}", flush=True)
        wait(900)  # (the shops' first stock: most wares come in minutes into a game)
        step([SpawnHero(0, "Udre", tomb.x + 125, tomb.y - 125, 3)])
        step()
        hero = own("Udre")[0].id
        step([LearnSkill(hero, fourcc("AUcs"))])
        wait(2)

        def buy(w):
            """The ware bought (from the tomb, else the merchant beside it) -> its slot, or None."""
            for shop in [tomb] + merch:
                before = list(u(hero).items)
                step([Buy(hero, shop.id, fourcc(w))])
                wait(6)
                after = list(u(hero).items)
                new = [k for k in range(len(after)) if after[k] and (k >= len(before) or after[k] != before[k])]
                if new:
                    return new[0]
            return None

        def test(w, label, forms=("immediate", "useitem")):
            s = buy(w)
            if s is None:
                print(f"{w}: not bought", flush=True)
                return
            for f in forms:
                ok = check(f"{w} slot {s + 1} {f} ({label})",
                           ImmediateOrder(hero, USESLOT + s) if f == "immediate" else UseItem(hero, s), 4)
                it = u(hero).items[s] if s < len(u(hero).items) else 0
                print(f"   slot after: {_raw(it) if it else None}; {show(hero)}", flush=True)
                if ok or not it:
                    break
        for w in ("phea", "pman", "hslv", "shea", "sreg", "stwp", "pinv", "spro", "tret", "stel", "wswd", "skul",
                  "pnvl", "ritd"):
            test(w, "full health and mana")
            if len([x for x in u(hero).items if x]) >= 5:  # (a new hero when the slots fill)
                step([SpawnHero(0, "Udre", tomb.x + 150, tomb.y + 250, 3)])
                step()
                hero = [x for x in own("Udre") if x.id != hero][-1].id
                step([LearnSkill(hero, fourcc("AUcs"))])
                wait(2)
        # hurt and drained: a heal and a mana potion again
        step([Spawn(1, "ogru", u(hero).x + 80, u(hero).y + 80), Spawn(1, "ogru", u(hero).x - 80, u(hero).y + 80)])
        step()
        gr = [x.id for x in own("ogru", 1)]
        step([TargetOrder(x, ATTACK, hero) for x in gr])
        for _ in range(40):
            step()
            if u(hero).hp < 0.6 * u(hero).max_hp:
                break
        step([PointOrder(x, MOVE, hall.x + 3000, hall.y + 3000) for x in gr]
             + [PointOrder(hero, 852218, u(hero).x + 400, u(hero).y)])  # (carrion swarm: its mana)
        wait(8)
        print(f"hero hurt and drained: {show(hero)}", flush=True)
        for w in ("phea", "pman", "hslv", "shea", "dust"):
            test(w, "hurt, mana spent")
    g.close()
    return 0


def _raw(v: int) -> str:
    from warcraftsim.protocol import rawcode
    return rawcode(int(v))


if __name__ == "__main__":
    raise SystemExit(main())
