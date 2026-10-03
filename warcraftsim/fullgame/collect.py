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
from collections import Counter
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
from ..runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup

RACES = ("human", "orc", "undead", "nightelf")
UNIT_COLS = ("step", "id", "type", "owner", "x", "y", "facing", "hp", "max_hp", "mana", "max_mana", "order",
             "flags", "visible_to", "resource", "hero_level", "hero_xp", "skill_points")
HERO_COLS = ("step", "id", *(f"item{k}" for k in range(1, 7)), *(f"ability_level{k}" for k in range(1, 5)),
             *(f"cooldown{k}" for k in range(1, 5)))
PLAYER_COLS = ("step", "player", "gold", "lumber", "food_used", "food_cap", "upkeep", "gold_gathered",
               "lumber_gathered", "structures", "result")


class Takeover:
    """A policy (play.BCAgent) plays `player` until step `at`; then the built-in AI takes it over
    (protocol.StartAI) and its orders are recorded from there: demonstrations from states the
    policy reached (DAgger's idea, with the built-in AI as the expert). The clone fell behind from
    its first seconds (a farm late, a barracks twice) into states the AI's games never show."""

    def __init__(self, bot, player: int, at: int):
        self.bot, self.player, self.at = bot, player, at
        self.sent: list = []

    def act(self, obs, t: int) -> list:
        from ..protocol import StartAI
        self.sent = self.bot.act(obs, t) if t < self.at else [StartAI(self.player)] if t == self.at else []
        return self.sent

    def after(self, obs) -> None:
        """The game's answers to the policy's orders (its production features count the accepted)."""
        if self.sent and type(self.sent[0]).__name__ != "StartAI":
            self.bot.accepted(self.sent, obs.command_results[:len(self.sent)])


def command_row(t: int, c) -> tuple | None:
    """A command a policy gave in step t as a recorded order row (step, unit, order, kind, x, y,
    target), as the game reports the built-in AI's (Observation.issued); None: not an order."""
    from ..protocol import Build, ImmediateOrder, LearnSkill, PointOrder, TargetDestructable, TargetOrder, fourcc
    if isinstance(c, ImmediateOrder):
        return (t, c.unit, c.order, 0, 0, 0, 0)
    if isinstance(c, PointOrder):
        return (t, c.unit, c.order, 1, int(c.x), int(c.y), 0)
    if isinstance(c, Build):
        return (t, c.unit, fourcc(c.building), 1, int(c.x), int(c.y), 0)
    if isinstance(c, TargetOrder):
        return (t, c.unit, c.order, 2, 0, 0, c.target)
    if isinstance(c, TargetDestructable):
        return (t, c.unit, c.order, 2, 0, 0, c.destructable)
    if isinstance(c, LearnSkill):
        return (t, c.hero, fourcc(c.ability), 3, 0, 0, 0)
    return None


class Shadow:
    """A policy (play.BCAgent) plays `player` the whole game while the built-in AI advises it
    (protocol.ShadowAI): the AI's orders are recorded (the labels) and undone, so every state is
    the policy's own: on-policy distillation with the built-in AI as the teacher. The policy's
    accepted orders are kept too ("policy_orders"): the side's features (what it queued, which
    workers cut lumber) follow what it did, not what the AI advised."""

    def __init__(self, bot, player: int):
        self.bot, self.player = bot, player
        bot.repair_builds = True  # (a builder the advisor pulled away resumes with "repair")
        self.sent: list = []
        self.t = 0
        self.orders: list[tuple] = []

    def act(self, obs, t: int) -> list:
        from ..protocol import ShadowAI
        self.t = t
        self.sent = self.bot.act(obs, t)
        return ([ShadowAI(self.player)] if t == 0 else []) + self.sent

    def after(self, obs) -> None:
        res = list(obs.command_results)[1 if self.t == 0 else 0:][:len(self.sent)]
        if self.sent:
            self.bot.accepted(self.sent, res)
        for c, ok in zip(self.sent, res):
            row = command_row(self.t, c) if ok else None
            if row is not None:
                self.orders.append(row)


def play_game(g: GameInstance, obs, max_steps: int = 4000, values: dict | None = None,
              driver: Takeover | Shadow | None = None) -> dict:
    """One game from its first observation `obs`; returns the arrays of the .npz. `values` (unit
    type values, fullgame.trace.unit_values): also a trace for the video's panel ("trace").
    `driver`: a policy plays a side until the built-in AI takes it over (Takeover), or all game
    with the built-in AI advising it (Shadow)."""
    from . import features as fx
    from .trace import material, trace_step
    units, heroes, players, events, orders = [], [], [], [], []
    trees = [(d.id, d.type_id, d.x, d.y, d.life) for d in (obs.destructables or [])]
    order_names = dict(obs.orders or {})
    names = {v: k for k, v in order_names.items()}
    kinds = {0: fx.IMMEDIATE, 1: fx.POINT, 2: fx.UNIT, 3: fx.SKILL}  # issued-order kinds -> the labels'
    trace: list[dict] = []
    t = 0
    while True:
        if values is not None:
            trace.append(trace_step(t, obs, material(obs, values), {}))
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
        obs = g.step(driver.act(obs, t) if driver is not None else [])
        if driver is not None:
            driver.after(obs)
        chose: dict[str, Counter] = {}
        owner = {u.id: u.owner for u in obs.units}
        for o in obs.issued:  # given during step t, in the state of observation t
            orders.append((t, o.unit, o.order, o.kind, o.x, o.y, o.target))
            if values is not None and o.order not in fx.DROPPED_ORDERS and owner.get(o.unit) in (0, 1):
                label = fx.order_label(o.order, kinds.get(o.kind, fx.IMMEDIATE), names)
                chose.setdefault(str(owner[o.unit]), Counter())[label] += 1
        if values is not None and trace:
            trace[-1]["orders"] = {p: dict(c) for p, c in chose.items()}
        t += 1
    result = {p: s.result.name for p, s in obs.players.items()}
    as_array = lambda rows, n: np.asarray(rows, np.int32).reshape(-1, n)  # noqa: E731
    return {"units": as_array(units, len(UNIT_COLS)), "heroes": as_array(heroes, len(HERO_COLS)),
            "players": as_array(players, len(PLAYER_COLS)), "events": as_array(events, 5),
            "orders": as_array(orders, 7), "trees": as_array(trees, 5),
            **({"policy_orders": as_array(driver.orders, 7)} if isinstance(driver, Shadow) else {}),
            "meta_extra": {"order_names": order_names, "result": result, "steps": t,
                           "game_seconds": obs.game_time},
            **({"trace": {"steps": trace, "gamma": 0.997,
                          "outcome": {str(p): 1.0 if r == "VICTORY" else -1.0 if r == "DEFEAT" else 0.0
                                      for p, r in result.items() if p in (0, 1)}}} if values is not None else {})}


def series(setup: GameSetup, name: str, n: int, play: Callable[[GameInstance, Observation, int, bool], None],
           timeout: float = 180) -> None:
    """play(game instance, first observation, k, fresh) for k < n: games of one setup in one
    running game. Each after the first reloads the map in the process (GameInstance.restart: the
    engine's RestartGame, ~6 s on duelrush instead of a ~10 s launch). `fresh`: the first game of
    a launch (its replay holds just that game: one to film). A failed game is lost; the rest go
    on in a new process."""
    k = failures = 0
    while k < n:
        if failures:
            time.sleep(min(60, 5 * failures))  # e.g. the name in use: not a spin
        try:
            with GameInstance(setup, name=name, timeout=timeout) as g:
                obs = g.start()
                fresh = True
                while True:
                    k += 1
                    play(g, obs, k - 1, fresh)
                    if k >= n:
                        break
                    fresh = g._ended  # filmed (save_replay ended it): the restart launches anew
                    obs = g.restart()
        except Exception as e:  # noqa: BLE001
            print(f"{name}: game {k - 1} failed: {type(e).__name__}: {e}", flush=True)
            failures += 1
            if failures >= 10:
                print(f"{name}: giving up on {n - k} games", flush=True)
                return


def claim_slot(runs: Path, kind: str = "fullgame") -> tuple[int, object]:
    """A machine-wide slot number for the games' names (their Wine prefixes); held while the
    returned file stays open. The locks live in the cache, not under `runs`: runs elsewhere (a
    smoke test's) claimed the same slot and collided with the running games' names."""
    import fcntl

    from .. import paths
    lock_dir = paths.CACHE_DIR / "slots"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for slot in range(64):
        f = open(lock_dir / f"{kind}-{slot}.lock", "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return slot, f
        except OSError:
            f.close()
    raise RuntimeError("no free slot")


class Films:
    """Game videos rendered in the background: the first game of a launch (its replay holds just
    that game) at most every `every` minutes (0: whenever one is fresh) and at most `limit`
    (None: no limit), in <out>/videos, listed in <out>/media.jsonl (the dashboard's Replays)."""

    def __init__(self, out: Path, every: float, limit: int | None, name: str):
        self.out, self.every, self.limit, self.name = out, every * 60, limit, name
        self.next = time.time()  # the first one at once: something to look at early
        self.count = 0
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(1)  # one at a time: a render plays the game in real time

    def due(self) -> bool:
        """Whether to film the next fresh game (claims it)."""
        with self.lock:
            if (self.limit is not None and self.count >= self.limit) or (self.every > 0 and time.time() < self.next) \
                    or (self.every <= 0 and self.limit is None):
                return False
            self.next = time.time() + self.every
            self.count += 1
            return True

    def film(self, g: GameInstance, stem: str, row: dict, trace: dict | None = None) -> None:
        """Save the game's replay (ends it) and render it; `row`: the media row's fields; `trace`:
        the video's side panel (fullgame.overlay)."""
        try:
            replay = g.save_replay(self.out / "replays" / f"{stem}.w3g")
        except Exception as e:  # noqa: BLE001 (one video fewer)
            print(f"video {stem}: replay not saved: {e}", flush=True)
            return
        if trace is not None:
            replay.with_suffix(".trace.json").write_text(json.dumps(trace))
        self.pool.submit(self._render, g.setup, replay, stem, row, trace)

    def _render(self, setup: GameSetup, replay: Path, stem: str, row: dict, trace: dict | None = None) -> None:
        from ..video import render_replay
        try:
            overlay = None
            if trace is not None:
                from .overlay import FullGameOverlay
                overlay = FullGameOverlay(trace)
            out = render_replay(setup, replay, self.out / "videos" / f"{stem}.mp4", name=self.name, fit_all=True, crf=28,
                                overlay=overlay,
                                max_steps=int(setup.max_game_seconds / setup.step_seconds) + 40)
            with self.lock, open(self.out / "media.jsonl", "a") as f:
                f.write(json.dumps({"time": time.time(), "kind": "video", "file": str(out.relative_to(self.out)),
                                    **row}) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"video {stem}: not rendered: {e}", flush=True)

    def wait(self) -> None:
        self.pool.shutdown(wait=True)


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
            "orders": int(len(data["orders"])) if data is not None else meta.get("orders"),
            **({"takeover": {k: meta["takeover"][k] for k in ("player", "step")}} if meta.get("takeover") else {})}


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
    ap.add_argument("--video-every", type=float, default=10.0, help="minutes between game videos (0: none)")
    ap.add_argument("--games-per-process", type=int, default=8,
                    help="games of one matchup in one running game (restarts reload the map in it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=5,
                    help="games per load of the map, each with two players of its own (GameSetup.pairs; 1: every "
                         "restart reloads the map)")
    ap.add_argument("--policy", type=Path, help="takeover games (the torch Python): this policy (a fullgame/bc.py "
                                                "policy.pt) plays one side until the built-in AI takes it over")
    ap.add_argument("--takeover", default="10-180", help="with --policy: the step the AI takes over at, drawn from this range")
    ap.add_argument("--shadow", action="store_true", help="with --policy: the policy plays its side all game and "
                                                          "the built-in AI advises it (protocol.ShadowAI): labels in "
                                                          "the policy's own states")
    ap.add_argument("--device", default="cpu", help="with --policy: where the policy runs")
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
    info = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "time": time.time(),
            "status": "collecting", "pid": os.getpid(),
            "command": "python -m warcraftsim.fullgame.collect " + " ".join(sys.argv[1:] if argv is None else argv)}
    write_info(args.out, info)
    rng = random.Random(args.seed)
    lo, hi = (int(v) for v in args.takeover.split("-"))
    # (game, races, difficulties, the side the policy plays until the takeover step: -1 none)
    plans = [(i, rng.choice(races), rng.choice(races), rng.choice(diffs), rng.choice(diffs),
              i % 2 if args.policy else -1, rng.randint(lo, hi) if args.policy else 0) for i in range(args.games)]
    policy = None
    if args.policy:
        import torch

        from .costs import order_costs
        from .model import load
        torch.set_num_threads(2)  # (the games' threads call it at once; the games need the cores)
        net, ck = load(args.policy, args.device)
        policy = {"net": net, "vocab": ck["vocab"], "device": torch.device(args.device),
                  "costs": order_costs(ck["vocab"], args.map)}
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
        by_matchup.setdefault(p[1:6], []).append(p)
    chunks = [ps[s:s + args.games_per_process] for ps in by_matchup.values()
              for s in range(0, len(ps), args.games_per_process)]
    chunks.sort(key=lambda c: c[0][0])

    def save(plan, data: dict, seconds: float) -> None:
        i, r0, r1, d0, d1, side, at = plan
        path = args.out / f"game{i:05d}.npz"
        extra = data.pop("meta_extra")
        meta = {"races": [r0, r1], "difficulties": [d0, d1], "handicap": args.handicap, "map": args.map,
                "map_file": map_file,
                "victory": args.victory,
                "step_seconds": args.step_seconds, **extra}
        if side >= 0 and args.shadow:  # the policy played `side` all game, the AI's labels throughout
            meta["shadow"] = {"player": side, "policy": str(args.policy)}
        elif side >= 0:  # the policy played `side` until step `at` (bc.py: no labels there before it)
            meta["takeover"] = {"player": side, "step": at, "policy": str(args.policy)}
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
        return row["winner"], extra["game_seconds"] / 60

    def run(chunk) -> None:
        _, r0, r1, d0, d1, side, _ = chunk[0]
        slots = [BuiltinAI(r0, d0, handicap=args.handicap), BuiltinAI(r1, d1, handicap=args.handicap)]
        if side >= 0:  # the policy's side: an agent until the built-in AI (its difficulty) takes over
            slots[side] = Agent((r0, r1)[side], handicap=args.handicap, difficulty=(d0, d1)[side])
        setup = GameSetup(map=args.map, slots=slots,
                          step_seconds=args.step_seconds, max_game_seconds=args.max_minutes * 60,
                          record_ai_orders=True, victory=args.victory, window=screen, wait_floor_ms=args.wait_floor_ms,
                          d3d_thread=False, render_threads=0,  # (nobody watches: drawn in the game's thread)
                          pairs=args.pairs)
        name = names.get()
        t0 = [time.time()]

        def play(g, obs, k, fresh) -> None:
            film = fresh and films.due()
            driver = None
            if side >= 0:
                from .play import BCAgent
                bot = BCAgent(policy["net"], policy["vocab"], side, policy["device"], costs=policy["costs"])
                bot.begin(obs, [RACES.index(r) for r in (r0, r1)])
                driver = Shadow(bot, side) if args.shadow else Takeover(bot, side, chunk[k][6])
            data = play_game(g, obs, values=values if film else None, driver=driver)
            trace = data.pop("trace", None)
            winner, minutes = save(chunk[k], data, time.time() - t0[0])
            if film:
                i = chunk[k][0]
                if trace is not None:
                    sides = [{"player": 0, "name": f"built-in AI {d0} ({r0})", "kind": "ai"},
                             {"player": 1, "name": f"built-in AI {d1} ({r1})", "kind": "ai"}]
                    if side >= 0 and args.shadow:
                        sides[side] = {"player": side, "kind": "agent",
                                       "name": f"the policy, the AI ({(d0, d1)[side]}) advising ({(r0, r1)[side]})"}
                    elif side >= 0:
                        sides[side] = {"player": side, "kind": "agent", "name": f"the clone, the AI ({(d0, d1)[side]}) from "
                                                                                f"{chunk[k][6] * args.step_seconds:.0f} s ({(r0, r1)[side]})"}
                    trace.update(title=f"{args.out.name} · game {i}", sides=sides)
                films.film(g, f"game{i:05d}", trace=trace, row={
                    "episode": i, "title": f"game {i}: {r0} ({d0}) vs {r1} ({d1})",
                    "outcome": 0.0 if winner is None else 1.0 if winner == 0 else -1.0,
                    "result": "tie" if winner is None else f"{(r0, r1)[winner]} won",
                    "sub": f"after {minutes:.1f} game minutes"})
            t0[0] = time.time()

        try:
            series(setup, name, len(chunk), play)
        finally:
            names.put(name)

    films = Films(args.out, args.video_every, None, f"demo{slot}_video")
    from .trace import unit_values
    values = unit_values() if args.video_every > 0 else None
    try:
        with ThreadPoolExecutor(args.parallel) as ex:
            list(ex.map(run, chunks))
        films.wait()
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
