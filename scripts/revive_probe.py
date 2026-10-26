"""What the game refuses about a dead hero's revival at an altar (play.REVIVE: TargetOrder(altar, 852039, hero)), in a
real game: when after the death it is taken, what it costs (gold, lumber) at each hero level, whether the altar takes it
with too little gold or food, a second time while the first runs, and a tavern hero's (results 2026-10-26: features.revive_gold).

    python3 scripts/revive_probe.py --map echofast2 --race human
    python3 scripts/revive_probe.py --map "(2)EchoIsles" --race undead

Instance names start with "reviveprobe". Each line printed is one check: what was sent, the game's answer, the
player's gold / lumber / food before and after.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("WAYLAND_DISPLAY", None)  # (games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"

from warcraftsim.protocol import ImmediateOrder, PointOrder, SetResources, Spawn, SpawnHero, TargetOrder, fourcc  # noqa: E402
from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup  # noqa: E402

REVIVE, CANCEL, ATTACK, MOVE = 852039, 851976, 851983, 851986
RACE = {"human": ("halt", "Hpal", "Hamg", "hfoo", "hhou"), "orc": ("oalt", "Obla", "Ofar", "ogru", "otrb"),
        "undead": ("uaod", "Udea", "Ulic", "ugho", "uzig"), "nightelf": ("eate", "Edem", "Ekee", "earc", "emow")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="echofast2")
    ap.add_argument("--race", default="human")
    ap.add_argument("--levels", default="1,2,3,5")
    ap.add_argument("--name", default="reviveprobe0")
    ap.add_argument("--shim", default="/home/tora/warcraftsim/build/shim")
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    altar_t, hero_t, hero2_t, filler_t, farm_t = RACE[a.race]
    setup = GameSetup(map=a.map, slots=[Agent(a.race), Agent("orc")], step_seconds=0.5, draw=False,
                      native_obs=False, d3d_thread=False, render_threads=0, nice=10, warm_spare=False,
                      window=(320, 240), max_game_seconds=0, shim_dir=a.shim)
    g = GameInstance(setup, name=a.name, timeout=300)
    obs = g.start()
    t = [0]

    def step(cmds=()):
        nonlocal obs
        obs = g.step(list(cmds))
        t[0] += 1
        return obs.command_results

    def me():
        p = obs.players[0]
        return p.gold, p.lumber, p.food_used, p.food_cap

    def find(kind, alive=True):
        return [u for u in obs.units if u.owner == 0 and u.type == kind and (u.alive if alive else not u.alive)]

    p0 = obs.players[0]
    sx, sy = p0.start_x, p0.start_y
    step([Spawn(0, altar_t, sx + 700, sy - 500)] + [Spawn(0, farm_t, sx - 900 + 160 * k, sy + 900) for k in range(8)])
    step()
    altar = find(altar_t)[0]
    print(f"map {a.map} race {a.race} altar {altar.id} {altar.type} flags {int(altar.flags)} me {me()}", flush=True)

    p1 = obs.players[1]

    def kill(hero_id):
        """The other player's taurens next to the hero, ordered to attack it, then sent home (no food of the
        player's): returns the steps it took."""
        h = obs.unit(hero_id)
        step([Spawn(1, "otau", h.x + 60 * (k % 3) - 60, h.y + 80 + 60 * (k // 3)) for k in range(6)])
        killers = [u for u in obs.units if u.owner == 1 and u.type == "otau" and u.alive]
        for n in range(400):
            h = obs.unit(hero_id)
            if h is None or not h.alive:
                step([PointOrder(k.id, MOVE, p1.start_x, p1.start_y) for k in killers])
                return n
            step([TargetOrder(k.id, ATTACK, hero_id) for k in killers] if n % 4 == 0 else [])
        raise RuntimeError("the hero did not die")

    def try_revive(hero_id, gold, lumber=0, extra=(), cancel=True, label=""):
        """Set the resources, then the revival: the game's answer, and what it took (gold before - after)."""
        step([SetResources(0, gold, lumber)])
        before = me()
        res = step([*extra, TargetOrder(altar.id, REVIVE, hero_id)])
        after = me()
        ok = res[-1]
        h = obs.unit(hero_id)
        print(f"  {label:34s} gold {gold:5d} -> {'TAKEN  ' if ok else 'REFUSED'} me {before} -> {after} paid "
              f"{before[0] - after[0]} gold {before[1] - after[1]} lumber; hero lvl {h.hero_level if h else None} "
              f"flags {int(h.flags) if h else None} altar order {obs.unit(altar.id).order}", flush=True)
        if ok and cancel:
            step([ImmediateOrder(altar.id, CANCEL)])
            print(f"  {'(cancelled)':34s} me {me()}", flush=True)
        return ok, before[0] - after[0]

    for lvl in [int(x) for x in a.levels.split(",")]:
        step([SpawnHero(0, hero_t, sx + 300, sy + 300, lvl)])
        step()
        hero = [u for u in find(hero_t)][-1]
        print(f"hero {hero.id} {hero.type} level {hero.hero_level} me {me()}", flush=True)
        n = kill(hero.id)
        died = t[0]
        h = obs.unit(hero.id)
        print(f" dead after {n} steps: level {h.hero_level} flags {int(h.flags)} me {me()}", flush=True)
        # (1) how soon after the death the altar takes it (plenty of gold; cancelled at once)
        cost = None
        for k in range(40):
            ok, paid = try_revive(hero.id, 5000, 5000, label=f"{t[0] - died} steps after the death")
            if ok:
                cost = paid
                break
        if cost is None:
            print(" never taken in 40 steps", flush=True)
            continue
        # (2) the gold it needs: one less and exactly
        try_revive(hero.id, cost - 1, 0, label=f"cost {cost} - 1 gold, no lumber")
        try_revive(hero.id, cost, 0, label=f"cost {cost} exactly, no lumber")
        # (4) twice: the second while the first runs
        ok, _ = try_revive(hero.id, 5000, 5000, cancel=False, label="first of two")
        try_revive(hero.id, 5000, 5000, cancel=False, label="second while the first runs")
        # the revival's time
        for k in range(600):
            h = obs.unit(hero.id)
            if h is not None and h.alive:
                print(f"  revived {k} steps after the order (level {h.hero_level}) me {me()}", flush=True)
                break
            step()
    # (5) food: the player at its cap, a dead hero, the revival with gold
    step([SpawnHero(0, hero_t, sx + 300, sy + 300, 1)])
    step()
    hero = find(hero_t)[-1]
    kill(hero.id)
    for k in range(60):
        g0, l0, fu, fc = me()
        if fc - fu < 5:
            break
        step([Spawn(0, filler_t, sx - 300 + 30 * k, sy - 300)])
    for k in range(20):
        if try_revive(hero.id, 5000, 5000, label=f"food {me()[2]}/{me()[3]}")[0]:
            break
    step([Spawn(0, farm_t, sx - 600, sy + 600)])
    step()
    try_revive(hero.id, 5000, 5000, label=f"a farm more: food {me()[2]}/{me()[3]}")
    # (6) a tavern hero's revival at the altar
    step([SpawnHero(0, "Nbst", sx + 300, sy + 300, 1)])
    step()
    tav = find("Nbst")[-1]
    kill(tav.id)
    for k in range(30):
        if try_revive(tav.id, 5000, 5000, label=f"tavern hero {t[0]}")[0]:
            break
    g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
