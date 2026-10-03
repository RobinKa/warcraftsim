"""Play a behavior-cloned whole-game policy (fullgame/bc.py) against the built-in AI.

    python3 -m warcraftsim.fullgame.play runs/bc/fullgame-1/policy.pt --games 20 --race human --ai-race orc

(the torch Python). The agent takes a computer slot without an AI: every order its units get
comes from the policy. Each step it sees what the demonstrations' players saw (fullgame/features:
its own units, the enemy's and neutral units it can see, resources, supply, time, races,
upgrades), and each own unit gets the order the policy samples (or none): an order at once
(train, research, stop, ...), at a point (move, build: the order is the building's type), on a unit
(attack, harvest, repair, ...), on a tree (harvest: the tree nearest the chosen point) or a hero
skill to learn.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
from collections import Counter
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..protocol import (Build, Command, EventKind, ImmediateOrder, LearnSkill, Observation, PointOrder,
                        TargetDestructable, TargetOrder)
from ..runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup
from . import features as fx
from .collect import Films, series
from .model import load

TYPE_CODE = fx.TYPE_CODE  # orders at or above this are unit / building / upgrade / ability codes


def rawcode_or(order: int) -> str:
    from ..protocol import rawcode
    return rawcode(order) if order and order >= TYPE_CODE else str(order)


def unit_rows(obs: Observation, t: int) -> np.ndarray:
    """The observation's units as rows of collect.UNIT_COLS."""
    ua = getattr(obs, "unit_array", None)
    if ua is not None:  # (native observations: the table's first 17 columns are these, after the step)
        out = np.empty((len(ua), 18), np.int64)
        out[:, 0] = t
        out[:, 1:] = ua[:, :17]
        return out
    return np.asarray([(t, u.id, u.type_id, u.owner, u.x, u.y, u.facing, u.hp, u.max_hp, u.mana, u.max_mana,
                        u.order, int(u.flags), u.visible_to, u.resource, u.hero_level, u.hero_xp, u.skill_points)
                       for u in obs.units], np.int64).reshape(-1, 18)


REPAIR = 852024  # the order a worker resumes a building with


class BCAgent:
    """The policy playing one player of a live game."""

    def __init__(self, net, vocab: dict, player: int, device, temperature: float = 1.0, order_temperature: float = 1.0,
                 costs=None):
        """`costs` (costs.order_costs): orders the player can't pay for yet are masked."""
        self.net, self.enc, self.player, self.device = net, fx.Encoder(vocab, costs), player, device
        self.temperature = temperature
        self.order_temperature = order_temperature  # sharpens the choice among orders only
        self.orders = [(0, 0)] + [tuple(o) for o in vocab["orders"]]  # class -> (order id, kind)
        self.view = None
        self.trees: dict[int, tuple[int, int]] = {}
        self.issued = 0
        # a worker repairing counts as building too: with the built-in AI advising (collect.Shadow) a
        # builder the AI pulled away is sent back to its building with "repair", and the policy then
        # gave it another order (human buildings started by 2 minutes, finished: 1.2 a game, 4.2 without)
        self.repair_builds = False

    def begin(self, obs: Observation, races: list[int]) -> None:
        rows = unit_rows(obs, 0)
        sign = self.enc.side(rows, self.player)
        self.view = self.enc.view(self.player, sign, races)
        self.trees = {d.id: (d.x, d.y) for d in (obs.destructables or [])}
        self.h = None  # the memory core's state (FullGameNet.memory)
        self.building_since: dict[int, int] = {}  # worker -> the step it started its building order
        self.t = 0

    def accepted(self, cmds: list[Command], results: list[bool]) -> None:
        """The game's answer to the last step's orders: accepted train / research orders queue, and
        accepted orders set what the workers harvest (View.assign)."""
        self.view.record_orders((c.unit, c.order, 0) for c, ok in zip(cmds, results)
                                if ok and isinstance(c, ImmediateOrder))
        for c, ok, res in zip(cmds, results, getattr(self, "sent_resources", [])):
            unit = getattr(c, "unit", None)
            if ok and unit is not None:
                if res is None:
                    self.view.assign.pop(unit, None)
                else:
                    self.view.assign[unit] = res

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Categorical(logits=logits / self.temperature).sample()

    def observe(self, obs: Observation, t: int, rows: np.ndarray | None = None) -> dict | None:
        """This step's view (features.View.step), or None when the player has no units. `rows`:
        unit_rows(obs, t) when already made (both sides of a self-play game share them)."""
        for e in obs.events:
            if int(e.kind) == int(EventKind.TREE_DEATH):
                self.trees.pop(e.a, None)
        rows = unit_rows(obs, t) if rows is None else rows
        self.t = t
        p = obs.players.get(self.player)
        me = (np.asarray([t, self.player, p.gold, p.lumber, p.food_used, p.food_cap, p.upkeep, p.gold_gathered,
                          p.lumber_gathered, p.structures, int(p.result)]) if p is not None else None)
        events = np.asarray([(t, int(e.kind), e.a, e.b, e.c) for e in obs.events], np.int64).reshape(-1, 5)
        st = self.view.step(rows, me, events, t)
        sel = st["sel"][:st["n_own"]]  # the workers busy with a building order, and since when
        busy = {int(u) for u, f, o in zip(sel[:, fx.C_ID], sel[:, fx.C_FLAGS], sel[:, fx.C_ORDER])
                if fx.building(int(f), int(o)) or (self.repair_builds and int(o) == REPAIR and int(f) & fx.WORKER)}
        self.building_since = {u: self.building_since.get(u, t) for u in busy}
        return st if st["n_own"] > 0 else None

    def commands(self, st: dict, order, tgt, bx, by) -> list[Command]:
        """The orders sampled for the view's own units (order class, pointer, x and y bins per unit;
        sequences over the first min(n_own, MAX_OWN) units) as game commands."""
        out: list[Command] = []
        self.last_sent: list[tuple[int, int]] = []  # (order id, kind) of what it sent: the video's panel
        self.sent_resources: list[str | None] = []  # per command: what it sends the worker to harvest (accepted())
        sel = st["sel"]
        for i in range(min(st["n_own"], fx.MAX_OWN, len(order))):
            c = int(order[i])
            if c == 0:
                continue
            oid, kind = self.orders[c]
            if kind == fx.UNIT:
                ttype = int(sel[int(tgt[i]), fx.C_TYPE])
                res = fx.harvest_resource(oid, kind, fx.GOLD_MINES[0] if oid == fx.HARVEST else ttype)
            else:
                res = fx.harvest_resource(oid, kind, None)
            unit = int(sel[i, fx.C_ID])
            if fx.redundant(oid, kind, int(sel[i, fx.C_ORDER]), res, self.view.assign.get(unit)):
                continue  # it harvests that already
            if self.committed(unit):
                continue  # on its way to build, or building: another order would cancel it
            self.last_sent.append((oid, kind))
            x = float(self.view.sign * fx.bin_center(int(bx[i])))
            y = float(fx.bin_center(int(by[i])))
            if kind == fx.IMMEDIATE:
                out.append(ImmediateOrder(unit, oid))
            elif kind == fx.SKILL:
                out.append(LearnSkill(unit, oid))
            elif kind == fx.POINT:
                out.append(Build(unit, oid, x, y) if oid >= TYPE_CODE else PointOrder(unit, oid, x, y))
            elif kind == fx.UNIT:
                target = sel[int(tgt[i])]
                if oid == fx.HARVEST and int(target[fx.C_TYPE]) not in fx.GOLD_MINES:
                    # harvest on anything but a gold mine: the nearest mine in view (77% of the clone's
                    # harvest orders pointed elsewhere and were refused)
                    mines = sel[np.isin(sel[:, fx.C_TYPE], fx.GOLD_MINES)]
                    if not len(mines):
                        continue
                    d = (mines[:, fx.C_X] - sel[i, fx.C_X]) ** 2 + (mines[:, fx.C_Y] - sel[i, fx.C_Y]) ** 2
                    target = mines[int(np.argmin(d))]
                out.append(TargetOrder(unit, oid, int(target[fx.C_ID])))
            elif kind == fx.TREE and self.trees:
                tree = min(self.trees, key=lambda k: (self.trees[k][0] - x) ** 2 + (self.trees[k][1] - y) ** 2)
                out.append(TargetDestructable(unit, oid, tree))
            if len(out) > len(self.sent_resources):  # (aligned with out: accepted() pairs them)
                self.sent_resources.append(res)
        self.issued += len(out)
        return out

    def committed(self, unit: int) -> bool:
        """A worker busy with a building order (features.building) for less than BUILD_COMMIT_STEPS."""
        since = self.building_since.get(unit)
        return since is not None and self.t - since < fx.BUILD_COMMIT_STEPS

    def act(self, obs: Observation, t: int) -> list[Command]:
        st = self.observe(obs, t)
        if st is None:
            return []
        n, n_own = st["n"], st["n_own"]
        dev = self.device
        with torch.no_grad():
            ent = torch.as_tensor(st["ent"], device=dev).unsqueeze(0).float()
            typ = torch.as_tensor(st["type"], device=dev).unsqueeze(0).long()
            cur = torch.as_tensor(st["cur"], device=dev).unsqueeze(0).long()
            mask = torch.ones(1, n, dtype=torch.bool, device=dev)
            glob = torch.as_tensor(st["glob"], device=dev).unsqueeze(0)
            g, u = self.net.encode(ent, typ, cur, mask, glob)
            g, self.h = self.net.context(g, self.h)
            O = min(n_own, fx.MAX_OWN)
            avail = torch.as_tensor(st["avail"], device=dev)[None] if st.get("avail") is not None else None
            logits = self.net.order_logits(g, u[:, :O], typ, torch.tensor([O], device=dev), avail=avail)
            order = self._sample(logits)  # [1, O]
            if self.order_temperature != 1.0:  # whether a unit gets an order stays as learned; which one sharpens
                again = self._sample(logits[..., 1:] / self.order_temperature) + 1
                order = torch.where(order > 0, again, order)
            ptr, xl, z = self.net.target_logits(g, u, mask, order)
            tgt, bx_ = self._sample(ptr)[0], self._sample(xl)
            by = self._sample(self.net.y_logits(z, bx_))[0]
        return self.commands(st, order[0].tolist(), tgt.tolist(), bx_[0].tolist(), by.tolist())


def matchup_setup(map_name: str, race: str, ai_race: str, difficulty: str, agent_side: int, handicap: int,
                  max_minutes: float, step_seconds: float, screen: tuple[int, int] = (320, 240)) -> GameSetup:
    agent = Agent(race, handicap=handicap)
    ai = BuiltinAI(ai_race, difficulty, handicap=handicap)
    slots = [agent, ai] if agent_side == 0 else [ai, agent]
    return GameSetup(map=map_name, slots=slots, step_seconds=step_seconds, max_game_seconds=max_minutes * 60,
                     victory="decisive", window=screen,  # a small screen: nothing looks at the pixels
                     d3d_thread=False, render_threads=0,  # (and drawn in the game's own thread)
                     record_ai_orders=True)  # (the AI's orders: shown in the videos' panel)


def play_game(g: GameInstance, obs: Observation, net, vocab: dict, device, agent_side: int,
              temperature: float, order_temperature: float = 1.0, values: dict | None = None, costs=None) -> dict:
    """One game from its first observation `obs` (the game's setup: matchup_setup). `values`
    (unit type values, fullgame.trace.unit_values): also a trace for the video's panel ("trace")."""
    from .trace import material, trace_step
    slots = g.setup.slots
    races = [fx.RACES.index(s.race) if s.race in fx.RACES else 0 for s in slots]
    bot = BCAgent(net, vocab, agent_side, device, temperature, order_temperature, costs=costs)
    t0 = time.time()
    order_names = {v: k for k, v in (obs.orders or {}).items()}
    bot.begin(obs, races)
    t = 0
    sent = failed = 0
    by_kind: dict[str, list[int]] = {}  # command kind -> [sent, failed]
    names = {v: k for k, v in (obs.orders or {}).items()}
    kinds = {0: fx.IMMEDIATE, 1: fx.POINT, 2: fx.UNIT, 3: fx.SKILL}
    trace: list[dict] = []
    held = {0: [], 1: []}  # gold + lumber on hand each step (hoarding: gathered and not spent)
    food_1min: dict[int, int] = {}
    while not obs.game_over:
        for p in (0, 1):
            s_ = obs.players.get(p)
            if s_ is not None:
                held[p].append(s_.gold + s_.lumber)
                if obs.game_time >= 60 and p not in food_1min:
                    food_1min[p] = s_.food_used
        cmds = bot.act(obs, t)
        if values is not None:
            mine = Counter(fx.order_label(o, k, names) for o, k in getattr(bot, "last_sent", []))
            trace.append(trace_step(t, obs, material(obs, values), {agent_side: mine}))
        obs = g.step(cmds)
        if values is not None and trace:
            owner = {u.id: u.owner for u in obs.units}
            ai = Counter(fx.order_label(o.order, kinds.get(o.kind, fx.IMMEDIATE), names) for o in obs.issued
                         if o.order not in fx.DROPPED_ORDERS and owner.get(o.unit) == 1 - agent_side)
            trace[-1]["orders"][str(1 - agent_side)] = dict(ai)
        bot.accepted(cmds, obs.command_results)
        for c, ok in zip(cmds, obs.command_results):  # the game refused the order
            order = getattr(c, "order", None) or getattr(c, "building", None) or 0  # (Build: the building's type)
            name = order_names.get(order, None) or rawcode_or(order)
            k = by_kind.setdefault(f"{type(c).__name__}:{name}", [0, 0])
            k[0] += 1
            k[1] += not ok
        sent += len(cmds)
        failed += sum(not ok for ok in obs.command_results[:len(cmds)])
        t += 1
    results = {p: s.result.name for p, s in obs.players.items()}
    sides = {("agent" if p == agent_side else "ai"): {"gold": s.gold_gathered, "lumber": s.lumber_gathered,
                                                        "food": f"{s.food_used}/{s.food_cap}",
                                                        "structures": s.structures,
                                                        "held": round(sum(held[p]) / max(len(held[p]), 1)),
                                                        "food_1min": food_1min.get(p)}
             for p, s in obs.players.items() if p in (0, 1)}
    outcome = results.get(agent_side, "?")
    ai = slots[1 - agent_side]
    out = {"outcome": outcome, "minutes": round(obs.game_time / 60, 2), "orders": bot.issued, "steps": t,
            "failed": failed, "by_kind": by_kind, "sides": sides,
            "race": slots[agent_side].race, "ai_race": ai.race, "difficulty": ai.difficulty, "side": agent_side,
            "seconds": round(time.time() - t0, 1)}
    if values is not None:
        res = {s: 1.0 if r == "VICTORY" else -1.0 if r == "DEFEAT" else 0.0 for s, r in results.items() if s in (0, 1)}
        out["trace"] = {"steps": trace, "gamma": 0.997, "outcome": {str(s): v for s, v in res.items()},
                        "sides": [{"player": agent_side, "name": f"the clone ({slots[agent_side].race})", "kind": "agent"},
                                  {"player": 1 - agent_side, "name": f"built-in AI {ai.difficulty} ({ai.race})",
                                   "kind": "ai"}]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--map", default="duelrush")
    ap.add_argument("--race", default="human", help="the agent's race, or 'all' (random each game)")
    ap.add_argument("--ai-race", default="orc", help="the built-in AI's race, or 'all'")
    ap.add_argument("--difficulty", default="normal")
    ap.add_argument("--handicap", type=int, default=50)
    ap.add_argument("--max-minutes", type=float, default=4.0)
    ap.add_argument("--step-seconds", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--order-temperature", type=float, default=1.0,
                    help="sharpens which order a unit gets, not whether it gets one")
    ap.add_argument("--games-per-process", type=int, default=4,
                    help="games of one matchup in one running game (restarts reload the map in it)")
    ap.add_argument("--label", help="the evaluation's name on the dashboard")
    ap.add_argument("--videos", type=int, default=2, help="games to film (the first of a launch each)")
    ap.add_argument("--avail-mask", type=int, default=1, help="mask the orders the player can't pay for yet")
    ap.add_argument("--mirror", action="store_true", help="the AI plays the agent's race")
    ap.add_argument("--device", help="default: cuda if available")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="results (.jsonl; default: next to the checkpoint, play.jsonl)")
    args = ap.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net, ck = load(args.checkpoint, device)
    vocab = ck["vocab"]
    rng = random.Random(args.seed)
    plans = [(i, rng.choice(fx.RACES) if args.race == "all" else args.race,
              rng.choice(fx.RACES) if args.ai_race == "all" else args.ai_race, i % 2) for i in range(args.games)]
    if args.mirror:
        plans = [(i, race, race, side) for i, race, _, side in plans]
    out = args.out or args.checkpoint.with_name("play.jsonl")
    eval_id = f"{args.checkpoint.stem}@{int(time.time())}"  # this launch's games (the dashboard groups by it)
    label = args.label or (f"{args.checkpoint.name} (epoch {ck.get('epoch')}) vs {args.difficulty} AI"
                           + (f", temperature {args.temperature:g}" if args.temperature != 1.0 else "")
                           + (f", order temperature {args.order_temperature:g}" if args.order_temperature != 1.0 else "")
                           + (", mirror matchups" if args.mirror else "") + ("" if args.avail_mask else ", no availability mask"))
    names = queue.Queue()
    for k in range(args.parallel):
        names.put(f"bcplay{k}")
    # games of one matchup run one after the other in one process (restarts reload the map)
    by_matchup: dict[tuple, list] = {}
    for p in plans:
        by_matchup.setdefault(p[1:], []).append(p)
    chunks = [ps[s:s + args.games_per_process] for ps in by_matchup.values()
              for s in range(0, len(ps), args.games_per_process)]
    chunks.sort(key=lambda c: c[0][0])
    results: list[dict] = []
    lock = threading.Lock()

    def run(chunk) -> None:
        _, race, ai_race, side = chunk[0]
        setup = matchup_setup(args.map, race, ai_race, args.difficulty, side, args.handicap, args.max_minutes,
                              args.step_seconds)

        def play(g, obs, k, fresh) -> None:
            i = chunk[k][0]
            film = fresh and films.due()
            r = play_game(g, obs, net, vocab, device, side, args.temperature, args.order_temperature,
                          values=values if film else None, costs=costs)
            trace = r.pop("trace", None)
            r.update({"game": i, "checkpoint": str(args.checkpoint), "time": time.time(), "map": args.map,
                      "eval": eval_id, "label": label, "epoch": ck.get("epoch"), "temperature": args.temperature,
                      "order_temperature": args.order_temperature})
            with lock:
                results.append(r)
                with open(out, "a") as f:
                    f.write(json.dumps(r) + "\n")
            if film:
                if trace is not None:
                    trace["title"] = f"{args.checkpoint.parent.name} · {label}"[:70]
                films.film(g, f"{eval_id.replace('@', '-')}-game{i:03d}", trace=trace, row={
                    "episode": i, "eval": eval_id, "label": label,
                    "title": f"game {i}: the clone ({race}) vs the built-in AI ({ai_race}, {args.difficulty})",
                    "outcome": {"VICTORY": 1.0, "DEFEAT": -1.0}.get(r["outcome"], 0.0),
                    "sub": f"{r['outcome'].lower()} after {r['minutes']:.1f} game minutes · {r['orders']} orders, "
                           f"{r['failed']} refused"})
            print(f"game {i}: {race} (side {side}) vs {ai_race} {args.difficulty}: {r['outcome']} after "
                  f"{r['minutes']} min ({r['orders']} orders, {r['failed']} refused {r['by_kind']}; {r['sides']})",
                  flush=True)

        name = names.get()
        try:
            series(setup, name, len(chunk), play)
        finally:
            names.put(name)

    from .costs import order_costs
    costs = order_costs(vocab, args.map) if args.avail_mask else None  # (unaffordable orders masked)
    films = Films(out.parent, 0.0, args.videos, "bcplay_video")  # videos: the fit's Replays on the dashboard
    from .trace import unit_values
    values = unit_values() if args.videos > 0 else None
    with ThreadPoolExecutor(args.parallel) as ex:
        list(ex.map(run, chunks))
    if films.count:
        print(f"rendering {films.count} videos", flush=True)
    films.wait()
    wins = sum(r["outcome"] == "VICTORY" for r in results)
    ties = sum(r["outcome"] == "TIE" for r in results)
    print(f"{args.checkpoint}: {wins} wins, {ties} ties, {len(results) - wins - ties} losses in {len(results)} games "
          f"against the built-in AI ({args.difficulty})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
