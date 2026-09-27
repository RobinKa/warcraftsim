"""Demonstrations from the built-in AI: built-in AI against built-in AI on a duel map, every
step's state and the orders each AI gave (GameSetup.record_ai_orders), one .npz per game.

    python -m warcraftsim.fullgame.collect --games 200 --parallel 8 --out runs/fullgame/demos-1

Per game (arrays of int32 rows; `step` is the observation's index, orders given during step t
show up in observation t + 1 and are stored with step t, the state they were given in):
  units    step id type owner x y facing hp max_hp mana max_mana order flags visible_to resource
           hero_level hero_xp skill_points          (every unit alive or dead, every step)
  heroes   step id item1..6 ability_level1..4 cooldown1..4 (tenths of a second)
  players  step player gold lumber food_used food_cap upkeep gold_gathered lumber_gathered
           structures result
  events   step kind a b c
  orders   step unit order kind x y target     (kind: 0 immediate, 1 point, 2 target, 3 hero skill)
  trees    id type x y life                    (at the start)
and meta.json-like `meta` (a JSON string): races, difficulties, handicap, map, step seconds, the
order names, the result and the length.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..runtime.instance import BuiltinAI, GameInstance, GameSetup

RACES = ("human", "orc", "undead", "nightelf")
UNIT_COLS = ("step", "id", "type", "owner", "x", "y", "facing", "hp", "max_hp", "mana", "max_mana", "order",
             "flags", "visible_to", "resource", "hero_level", "hero_xp", "skill_points")
HERO_COLS = ("step", "id", *(f"item{k}" for k in range(1, 7)), *(f"ability_level{k}" for k in range(1, 5)),
             *(f"cooldown{k}" for k in range(1, 5)))
PLAYER_COLS = ("step", "player", "gold", "lumber", "food_used", "food_cap", "upkeep", "gold_gathered",
               "lumber_gathered", "structures", "result")


def play_game(setup: GameSetup, name: str, max_steps: int = 4000) -> dict:
    """One game; returns the arrays of the .npz."""
    units, heroes, players, events, orders = [], [], [], [], []
    with GameInstance(setup, name=name, timeout=180) as g:
        obs = g.start()
        trees = [(d.id, d.type_id, d.x, d.y, d.life) for d in (obs.destructables or [])]
        order_names = dict(obs.orders or {})
        t = 0
        while True:
            for u in obs.units:
                units.append((t, u.id, u.type_id, u.owner, u.x, u.y, u.facing, u.hp, u.max_hp, u.mana, u.max_mana,
                              u.order, int(u.flags), u.visible_to, u.resource, u.hero_level, u.hero_xp,
                              u.skill_points))
                if u.is_hero:
                    items = (tuple(u.items) + (0,) * 6)[:6]
                    ab = (tuple(u.abilities) + ((0, 0.0),) * 4)[:4]
                    heroes.append((t, u.id, *items, *(a[0] for a in ab), *(int(round(a[1] * 10)) for a in ab)))
            for p, s in obs.players.items():
                players.append((t, p, s.gold, s.lumber, s.food_used, s.food_cap, s.upkeep, s.gold_gathered,
                                s.lumber_gathered, s.structures, int(s.result)))
            for e in obs.events:
                events.append((t, int(e.kind), e.a, e.b, e.c))
            if obs.game_over or t >= max_steps:
                break
            obs = g.step()
            for o in obs.issued:  # given during step t, in the state of observation t
                orders.append((t, o.unit, o.order, o.kind, o.x, o.y, o.target))
            t += 1
        result = {p: s.result.name for p, s in obs.players.items()}
    as_array = lambda rows, n: np.asarray(rows, np.int32).reshape(-1, n)  # noqa: E731
    return {"units": as_array(units, len(UNIT_COLS)), "heroes": as_array(heroes, len(HERO_COLS)),
            "players": as_array(players, len(PLAYER_COLS)), "events": as_array(events, 5),
            "orders": as_array(orders, 7), "trees": as_array(trees, 5),
            "meta_extra": {"order_names": order_names, "result": result, "steps": t,
                           "game_seconds": obs.game_time}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--map", default="duelrush")
    ap.add_argument("--races", default="human,orc", help="the races to draw each side's from (or 'all')")
    ap.add_argument("--difficulty", default="normal", help="easy / normal / insane, or several: normal,insane")
    ap.add_argument("--handicap", type=int, default=50)
    ap.add_argument("--step-seconds", type=float, default=0.5)
    ap.add_argument("--max-minutes", type=float, default=4.0, help="a tie after this much game time")
    ap.add_argument("--victory", default="decisive", help="melee, or decisive (also lost with no town hall and "
                                                          "no units: no minutes of waiting for a last building)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    races = RACES if args.races == "all" else tuple(args.races.split(","))
    diffs = tuple(args.difficulty.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "collect.json").write_text(json.dumps({**vars(args), "out": str(args.out), "time": time.time()},
                                                      indent=1))
    rng = random.Random(args.seed)
    plans = [(i, rng.choice(races), rng.choice(races), rng.choice(diffs), rng.choice(diffs))
             for i in range(args.games)]
    lock = threading.Lock()
    done = {"n": 0, "t0": time.time()}
    names = queue.Queue()  # one game instance name per worker: games reuse their names' prefixes
    for k in range(args.parallel):
        names.put(f"demo{k}")

    def one(plan) -> None:
        i, r0, r1, d0, d1 = plan
        path = args.out / f"game{i:05d}.npz"
        if path.exists():
            return
        setup = GameSetup(map=args.map, slots=[BuiltinAI(r0, d0, handicap=args.handicap),
                                               BuiltinAI(r1, d1, handicap=args.handicap)],
                          step_seconds=args.step_seconds, max_game_seconds=args.max_minutes * 60,
                          record_ai_orders=True, victory=args.victory,
                          warm_spare=False)  # one game per instance: a spare would load for nothing
        t0 = time.time()
        name = names.get()
        try:
            data = play_game(setup, name)
        except Exception as e:  # noqa: BLE001 (one game fewer)
            print(f"game {i}: failed: {e}", flush=True)
            return
        finally:
            names.put(name)
        extra = data.pop("meta_extra")
        meta = {"races": [r0, r1], "difficulties": [d0, d1], "handicap": args.handicap, "map": args.map,
                "victory": args.victory,
                "step_seconds": args.step_seconds, **extra}
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, meta=json.dumps(meta), **data)
        tmp.replace(path)
        with lock:
            done["n"] += 1
            rate = done["n"] / (time.time() - done["t0"]) * 3600
            print(f"game {i}: {r0}/{d0} vs {r1}/{d1}: {extra['result']} after {extra['game_seconds'] / 60:.1f} min "
                  f"({len(data['orders'])} orders, {time.time() - t0:.0f}s; {done['n']} done, {rate:.0f}/h)",
                  flush=True)

    with ThreadPoolExecutor(args.parallel) as ex:
        list(ex.map(one, plans))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
