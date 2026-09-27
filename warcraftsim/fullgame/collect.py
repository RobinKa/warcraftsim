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
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from ..protocol import Observation
from ..runtime.instance import BuiltinAI, GameInstance, GameSetup

RACES = ("human", "orc", "undead", "nightelf")
UNIT_COLS = ("step", "id", "type", "owner", "x", "y", "facing", "hp", "max_hp", "mana", "max_mana", "order",
             "flags", "visible_to", "resource", "hero_level", "hero_xp", "skill_points")
HERO_COLS = ("step", "id", *(f"item{k}" for k in range(1, 7)), *(f"ability_level{k}" for k in range(1, 5)),
             *(f"cooldown{k}" for k in range(1, 5)))
PLAYER_COLS = ("step", "player", "gold", "lumber", "food_used", "food_cap", "upkeep", "gold_gathered",
               "lumber_gathered", "structures", "result")


def play_game(g: GameInstance, obs, max_steps: int = 4000) -> dict:
    """One game from its first observation `obs`; returns the arrays of the .npz."""
    units, heroes, players, events, orders = [], [], [], [], []
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


def series(setup: GameSetup, name: str, n: int, play: Callable[[GameInstance, Observation, int], None],
           timeout: float = 180) -> None:
    """play(game instance, first observation, k) for k < n: games of one setup in one running
    game. Each after the first reloads the map in the process (GameInstance.restart: the
    engine's RestartGame, ~6 s on duelrush instead of a ~10 s launch). A failed game is lost;
    the rest go on in a new process."""
    k = failures = 0
    while k < n:
        if failures:
            time.sleep(min(60, 5 * failures))  # e.g. the name in use: not a spin
        try:
            with GameInstance(setup, name=name, timeout=timeout) as g:
                obs = g.start()
                while True:
                    k += 1
                    play(g, obs, k - 1)
                    if k >= n:
                        break
                    obs = g.restart()
        except Exception as e:  # noqa: BLE001
            print(f"{name}: game {k - 1} failed: {type(e).__name__}: {e}", flush=True)
            failures += 1
            if failures >= 10:
                print(f"{name}: giving up on {n - k} games", flush=True)
                return


def claim_slot(runs: Path, kind: str = "fullgame") -> tuple[int, object]:
    """A machine-wide slot number for the games' names (their Wine prefixes); held while the
    returned file stays open."""
    import fcntl
    lock_dir = runs / ".slots"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for slot in range(64):
        f = open(lock_dir / f"{kind}-{slot}.lock", "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return slot, f
        except OSError:
            f.close()
    raise RuntimeError("no free slot")


def write_info(out: Path, info: dict) -> None:
    tmp = out / "collect.json.tmp"
    tmp.write_text(json.dumps(info, indent=1))
    tmp.replace(out / "collect.json")


def game_row(i: int, meta: dict, data: dict | None = None) -> dict:
    """A game's line in games.jsonl (the dashboard's view of a collection)."""
    result = {int(k): v for k, v in meta.get("result", {}).items()}
    winner = next((p for p, r in result.items() if r == "VICTORY"), None)
    return {"time": time.time(), "game": i, "races": meta.get("races"), "difficulties": meta.get("difficulties"),
            "winner": winner, "result": {str(k): v for k, v in result.items()},
            "minutes": round(meta.get("game_seconds", 0) / 60, 2), "steps": meta.get("steps"),
            "orders": int(len(data["orders"])) if data is not None else meta.get("orders")}


def index(out: Path) -> int:
    """games.jsonl rebuilt from a collection's .npz files (collections from before it)."""
    rows = []
    for path in sorted(out.glob("game*.npz")):
        if ".tmp" in path.name:
            continue
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            n_orders = int(z["orders"].shape[0])
        row = game_row(int(path.stem[4:]), meta)
        row.update(time=path.stat().st_mtime, orders=n_orders)
        rows.append(row)
    rows.sort(key=lambda r: r["time"])
    tmp = out / "games.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(out / "games.jsonl")
    info_path = out / "collect.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {"out": str(out)}
    info.setdefault("status", "finished")
    if rows:
        info.setdefault("finished", rows[-1]["time"])
    write_info(out, info)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index", action="store_true", help="only rebuild --out's games.jsonl from its games")
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
    ap.add_argument("--screen", default="320x240", help="the games' virtual screen (nothing looks at the pixels: "
                                                        "a small one saves ~15%% of the CPU)")
    ap.add_argument("--wait-floor-ms", type=int, default=5, help="GameSetup.wait_floor_ms")
    ap.add_argument("--games-per-process", type=int, default=8,
                    help="games of one matchup in one running game (restarts reload the map in it)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.index:
        print(f"{args.out}: {index(args.out)} games")
        return 0
    races = RACES if args.races == "all" else tuple(args.races.split(","))
    screen = tuple(int(v) for v in args.screen.split("x"))
    from ..data.duelmap import duel_map_path, parse_duel_name
    map_file = duel_map_path(args.map).name if parse_duel_name(args.map) else args.map  # the rules' version
    diffs = tuple(args.difficulty.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    info = {**vars(args), "out": str(args.out), "time": time.time(), "status": "collecting", "pid": os.getpid(),
            "command": "python -m warcraftsim.fullgame.collect " + " ".join(sys.argv[1:] if argv is None else argv)}
    write_info(args.out, info)
    rng = random.Random(args.seed)
    plans = [(i, rng.choice(races), rng.choice(races), rng.choice(diffs), rng.choice(diffs))
             for i in range(args.games)]
    lock = threading.Lock()
    done = {"n": 0, "t0": time.time()}
    # one game instance name per worker (games reuse their names' Wine prefixes); a machine-wide
    # slot keeps collections running at once apart
    slot, slot_lock = claim_slot(args.out.parent.parent if args.out.parent.name == "fullgame" else args.out.parent,
                                 kind="collect")
    names = queue.Queue()
    for k in range(args.parallel):
        names.put(f"demo{slot}_{k}")
    # games of one matchup run one after the other in one process (a restart reloads the map but
    # keeps the slots), in chunks so that every matchup is played from the start
    todo = [p for p in plans if not (args.out / f"game{p[0]:05d}.npz").exists()]
    by_matchup: dict[tuple, list] = {}
    for p in todo:
        by_matchup.setdefault(p[1:], []).append(p)
    chunks = [ps[s:s + args.games_per_process] for ps in by_matchup.values()
              for s in range(0, len(ps), args.games_per_process)]
    chunks.sort(key=lambda c: c[0][0])

    def save(plan, data: dict, seconds: float) -> None:
        i, r0, r1, d0, d1 = plan
        path = args.out / f"game{i:05d}.npz"
        extra = data.pop("meta_extra")
        meta = {"races": [r0, r1], "difficulties": [d0, d1], "handicap": args.handicap, "map": args.map,
                "map_file": map_file,
                "victory": args.victory,
                "step_seconds": args.step_seconds, **extra}
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, meta=json.dumps(meta), **data)
        tmp.replace(path)
        row = game_row(i, meta, data)
        row["seconds"] = round(seconds, 1)
        with lock:
            with open(args.out / "games.jsonl", "a") as f:  # the dashboard's view of the collection
                f.write(json.dumps(row) + "\n")
            done["n"] += 1
            rate = done["n"] / (time.time() - done["t0"]) * 3600
            print(f"game {i}: {r0}/{d0} vs {r1}/{d1}: {extra['result']} after {extra['game_seconds'] / 60:.1f} min "
                  f"({len(data['orders'])} orders, {seconds:.0f}s; {done['n']} done, {rate:.0f}/h)", flush=True)

    def run(chunk) -> None:
        _, r0, r1, d0, d1 = chunk[0]
        setup = GameSetup(map=args.map, slots=[BuiltinAI(r0, d0, handicap=args.handicap),
                                               BuiltinAI(r1, d1, handicap=args.handicap)],
                          step_seconds=args.step_seconds, max_game_seconds=args.max_minutes * 60,
                          record_ai_orders=True, victory=args.victory, window=screen, wait_floor_ms=args.wait_floor_ms)
        name = names.get()
        t0 = [time.time()]

        def play(g, obs, k) -> None:
            save(chunk[k], play_game(g, obs), time.time() - t0[0])
            t0[0] = time.time()

        try:
            series(setup, name, len(chunk), play)
        finally:
            names.put(name)

    try:
        with ThreadPoolExecutor(args.parallel) as ex:
            list(ex.map(run, chunks))
        info["status"] = "finished"
    except BaseException:
        info["status"] = "stopped"
        raise
    finally:
        info["finished"] = time.time()
        write_info(args.out, info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
