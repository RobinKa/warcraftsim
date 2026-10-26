"""Which of human's militia orders the game takes from a script's order (audit 16: "back to work" refused ~160 times a
human game, a militia's harvest and its right-click on a tree refused): a peasant's own call to arms, the hall's town
bell on and off, a militia's back to work in each form the vocabulary or the ability data has, and what each one does
(the unit's type and order over the following steps). Results 2026-10-26: features.MILITIA_*'s comment.

    python3 scripts/militia_probe.py --part 1   # the forms near the hall (and a keep elsewhere: --second 1)
    python3 scripts/militia_probe.py --part 2   # militia far away, the hall upgrading and upgraded
    python3 scripts/militia_probe.py --part 3   # the hall training workers

Instance names start with "militiaprobe". Each line printed is one check: what was sent, the game's answer, and what
the unit did after.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("WAYLAND_DISPLAY", None)  # (games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"

from warcraftsim.protocol import ImmediateOrder, PointOrder, SetResources, Spawn, TargetDestructable, TargetOrder  # noqa: E402
from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup  # noqa: E402

HARVEST, SMART, REPAIR, MOVE = 852018, 851971, 852024, 851986
MILITIA_CONVERT, MILITIA, MILITIA_OFF, TOWNBELL_ON, TOWNBELL_OFF, MILITIA_UNCONVERT = (
    852071, 852072, 852073, 852082, 852083, 852651)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="(2)EchoIsles")
    ap.add_argument("--name", default="militiaprobe0")
    ap.add_argument("--part", type=int, default=1, help="1: the forms near the hall; 2: far away, an upgraded keep")
    ap.add_argument("--second", type=int, default=1, help="a keep and a second hall of the player's too")
    ap.add_argument("--shim", default="/home/tora/warcraftsim/build/shim")
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    setup = GameSetup(map=a.map, slots=[Agent("human"), Agent("orc")], step_seconds=0.5, draw=False,
                      native_obs=False, d3d_thread=False, render_threads=0, nice=10, warm_spare=False,
                      window=(320, 240), max_game_seconds=0, shim_dir=a.shim)
    g = GameInstance(setup, name=a.name, timeout=300)
    obs = g.start()
    trees = list(obs.destructables or [])

    def step(cmds=()):
        nonlocal obs
        obs = g.step(list(cmds))
        return obs.command_results

    def own(kind):
        return [u for u in obs.units if u.owner == 0 and u.type == kind and u.alive]

    def show(uid):
        u = obs.unit(uid)
        return f"{u.type}/{u.order}" if u is not None and u.alive else "gone"

    def watch(label, cmds, units, steps=24):
        """Send `cmds`, print the answers, then each unit's type / order every 4 steps."""
        res = step(cmds)
        print(f"{label}: {['TAKEN' if r else 'REFUSED' for r in res]}", flush=True)
        for k in range(steps):
            if k % 4 == 3:
                print(f"   +{k + 1:2d} " + " ".join(f"{u}:{show(u)}" for u in units), flush=True)
            step()

    p0 = obs.players[0]
    sx, sy = p0.start_x, p0.start_y
    step([Spawn(0, "hpea", sx - 200 + 40 * k, sy - 350) for k in range(8)])
    step()
    hall = own("htow")[0]
    mine = min((u for u in obs.units if u.type == "ngol"), key=lambda u: u.dist(hall.x, hall.y))
    tree = min(trees, key=lambda d: (d.x - hall.x) ** 2 + (d.y - hall.y) ** 2) if trees else None
    peas = [u.id for u in own("hpea")]
    print(f"hall {hall.id} mine {mine.id} tree {tree.id if tree else None} peasants {peas}", flush=True)
    step([TargetOrder(p, HARVEST, mine.id) for p in peas])
    for _ in range(6):
        step()
    watch("townbelloff, no bell", [ImmediateOrder(hall.id, TOWNBELL_OFF)], [], 1)
    watch("peasant militiaconvert (immediate)", [ImmediateOrder(peas[2], MILITIA_CONVERT)], [], 1)
    keep = None
    if a.second:  # (a keep and a second town hall of the player's, elsewhere)
        step([Spawn(0, "hkee", sx + 1400, sy + 1400), Spawn(0, "htow", sx - 1400, sy + 1400)])
        step()
        keep = [u for u in own("hkee")][0]

    def fresh(k):
        """A militia of peasant k's own call to arms (a fresh one: 45 s)."""
        step([ImmediateOrder(peas[k], MILITIA)])
        for _ in range(16):
            step()
            u = obs.unit(peas[k])
            if u is not None and u.type == "hmil":
                return peas[k]
        print(f"   peasant {peas[k]} not a militia: {show(peas[k])}", flush=True)
        return peas[k]

    k = iter(range(len(peas)))
    if a.part == 1:
        m = fresh(next(k)); watch("A militiaoff (immediate)", [ImmediateOrder(m, MILITIA_OFF)], [m], 12)
        m = fresh(next(k)); watch("B militiaunconvert (target hall)", [TargetOrder(m, MILITIA_UNCONVERT, hall.id)], [m], 12)
        m = fresh(next(k)); watch("C smart (target hall)", [TargetOrder(m, SMART, hall.id)], [m], 12)
        m = fresh(next(k)); watch("D harvest (target mine)", [TargetOrder(m, HARVEST, mine.id)], [m], 4)
        if tree is not None:
            m = fresh(next(k)); watch("D2 smart (tree)", [TargetDestructable(m, SMART, tree.id)], [m], 4)
        m = fresh(next(k))
        step([PointOrder(m, MOVE, hall.x + 2200, hall.y - 600)])
        for _ in range(16):
            step()
        u = obs.unit(m)
        watch(f"E militiaunconvert (target hall) {int(u.dist(hall.x, hall.y))} away", [TargetOrder(m, MILITIA_UNCONVERT, hall.id)], [m], 12)
        if keep is not None:
            m = fresh(next(k)); watch("F militiaunconvert (target keep)", [TargetOrder(m, MILITIA_UNCONVERT, keep.id)], [m], 12)
        m = fresh(next(k))
        step([SetResources(0, 0, 0)])
        watch("G militiaunconvert (target hall) at 0 gold", [TargetOrder(m, MILITIA_UNCONVERT, hall.id)], [m], 12)
        step([SetResources(0, 500, 500)])
        m = fresh(next(k)); watch("H militiaunconvert (target peasant)", [TargetOrder(m, MILITIA_UNCONVERT, peas[-1])], [m], 4)
        m = fresh(next(k)); watch("I townbelloff with a peasant's militia", [ImmediateOrder(hall.id, TOWNBELL_OFF)], [m], 12)
        watch("townbellon", [ImmediateOrder(hall.id, TOWNBELL_ON)], [], 12)
        mil = [u.id for u in own("hmil")]
        watch(f"J townbelloff with the bell's militia {len(mil)}", [ImmediateOrder(hall.id, TOWNBELL_OFF)], mil[:4], 12)
        watch("townbellon", [ImmediateOrder(hall.id, TOWNBELL_ON)], [], 12)
        mil = [u.id for u in own("hmil")]
        watch("K the bell's militia: militiaunconvert (target hall), militiaoff, smart hall",
              [TargetOrder(mil[0], MILITIA_UNCONVERT, hall.id), ImmediateOrder(mil[1], MILITIA_OFF),
               TargetOrder(mil[2], SMART, hall.id)], mil[:4], 12)
        if keep is not None:
            watch("L the bell's militia: militiaunconvert (target keep)", [TargetOrder(mil[3], MILITIA_UNCONVERT, keep.id)],
                  mil[3:4], 12)
    elif a.part == 3:  # (the hall training workers, as in a real game)
        step([SetResources(0, 5000, 5000)] + [Spawn(0, "hhou", sx - 900 + 160 * j, sy + 900) for j in range(5)])
        step()
        hpea = int.from_bytes(b"hpea", "big")
        print("hall trains:", step([ImmediateOrder(hall.id, hpea)] * 4), show(hall.id), flush=True)
        m = fresh(next(k)); watch("R militiaunconvert (target hall training)", [TargetOrder(m, MILITIA_UNCONVERT, hall.id)], [m], 12)
        watch("S peasant militiaconvert (target hall training)", [TargetOrder(peas[-1], MILITIA_CONVERT, hall.id)], [peas[-1]], 12)
        m = fresh(next(k)); watch("T militiaoff, hall training", [ImmediateOrder(m, MILITIA_OFF)], [m], 12)
        watch("townbellon (training)", [ImmediateOrder(hall.id, TOWNBELL_ON)], [], 12)
        mil = [u.id for u in own("hmil")]
        print("hall", show(hall.id), flush=True)
        watch("U the bell's militia, hall training: militiaunconvert (target hall)",
              [TargetOrder(mil[0], MILITIA_UNCONVERT, hall.id)], mil[:1], 12)
    else:
        p1 = obs.players[1]
        mx, my = (sx + p1.start_x) / 2, (sy + p1.start_y) / 2
        watch("townbellon", [ImmediateOrder(hall.id, TOWNBELL_ON)], [], 12)
        mil = [u.id for u in own("hmil")]
        step([PointOrder(u, MOVE, mx, my) for u in mil[:4]])
        for _ in range(36):
            step()
        far = [int(obs.unit(u).dist(hall.x, hall.y)) for u in mil[:4]]
        watch(f"M far militia {far}: militiaunconvert (target hall), militiaoff",
              [TargetOrder(mil[0], MILITIA_UNCONVERT, hall.id), ImmediateOrder(mil[1], MILITIA_OFF)], mil[:4], 40)
        for _ in range(60):  # (the rest expire)
            step()
        step([SetResources(0, 5000, 5000)])
        res = step([ImmediateOrder(hall.id, int.from_bytes(b"hkee", "big"))])
        print(f"keep upgrade: {res} hall {show(hall.id)}", flush=True)
        for _ in range(4):
            step()
        m = fresh(next(k))
        watch("N militiaunconvert (target hall upgrading), militiaoff", [TargetOrder(m, MILITIA_UNCONVERT, hall.id)], [m], 12)
        m = fresh(next(k)); watch("N2 militiaoff, hall upgrading", [ImmediateOrder(m, MILITIA_OFF)], [m], 12)
        for n in range(600):
            step()
            if show(hall.id).startswith("hkee"):
                print(f"keep done after {n} steps", flush=True)
                break
        m = fresh(next(k)); watch("O militiaunconvert (target upgraded keep)", [TargetOrder(m, MILITIA_UNCONVERT, hall.id)], [m], 12)
        m = fresh(next(k)); watch("O2 militiaoff, upgraded keep", [ImmediateOrder(m, MILITIA_OFF)], [m], 12)
        res = step([ImmediateOrder(hall.id, TOWNBELL_ON)])
        print(f"keep townbellon {res}", flush=True)
        for _ in range(12):
            step()
        mil = [u.id for u in own("hmil")]
        watch(f"P keep's militia {len(mil)}: militiaunconvert (target keep), militiaoff",
              [TargetOrder(mil[0], MILITIA_UNCONVERT, hall.id), ImmediateOrder(mil[1], MILITIA_OFF)], mil[:3], 12)
        watch("Q townbelloff at the keep", [ImmediateOrder(hall.id, TOWNBELL_OFF)], mil[2:4], 12)
    g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
