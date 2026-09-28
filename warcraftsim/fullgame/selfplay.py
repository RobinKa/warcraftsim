"""Self-play reinforcement learning on whole games, from a behavior-cloned policy (after AlphaStar).

    python3 -m warcraftsim.fullgame.selfplay --name fgself-1 --init runs/bc/fullgame-1/policy.pt

(the torch Python). Actor processes play games on the duel map and send trajectories to the
learner (this process), which trains the policy and a value head with PPO:
* Actors: each runs a few games (threads) and batches its agents' network calls on the GPU. An
  agent sees and orders as the clone does in fullgame/play.py (play.BCAgent's view and commands).
  Games of one setup (races, built-in AI or not) run one after another in one process
  (GameInstance.restart reloads the map).
* The league (league.json): the learner plays itself (both sides train), past snapshots
  (prioritized fictitious self-play: the ones it beats less, more often) and the built-in AI (a
  fixed anchor: the win rate against it is the run's yardstick, "script:ai-<difficulty>").
* Reward: +1 for a win, -1 for a loss, 0 for a tie (the time limit).
* PPO per unit: each own unit's decision (its order, and the targets that order uses) has its own
  clipped ratio; they share the step's advantage (GAE, computed by the actors with the values
  they acted with).
* A KL term towards the behavior-cloned policy (AlphaStar's towards its supervised policy) keeps
  the policy near what it learned from the AI; the value head starts untrained, so the first
  updates train the value only (--value-warmup).

Writes runs/<name>/ as the dashboard reads it: run.json, train.jsonl, episodes.jsonl,
league.json, checkpoints/<agent steps>.pt (play.py and further runs can load them).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
import traceback
import weakref
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import features as fx
from .collect import claim_slot
from .costs import order_costs
from .trace import DEAD_FLAG, material, trace_step, unit_values  # noqa: F401
from .play import unit_rows
from .model import FullGameNet, act, evaluate, load
from ..rl.league import Member, pfsp_weight

RUNS = Path(__file__).resolve().parents[2] / "runs"


# ---- actors ---------------------------------------------------------------------------------------

class Nets:
    """An actor's networks: the learner's current weights (reloaded when the learner publishes
    new ones) and past snapshots (a few, most recently used)."""

    def __init__(self, cfg: dict, device):
        self.cfg, self.device = cfg, device
        self.current_path = Path(cfg["run_dir"]) / "current.pt"
        self.current: FullGameNet | None = None
        self.version = -1
        self.mtime = 0.0
        self.checked = 0.0
        self.snapshots: dict[str, FullGameNet] = {}
        self.reload()

    def reload(self) -> None:
        self.checked = time.time()
        try:
            mtime = self.current_path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime == self.mtime:
            return
        for _ in range(3):
            try:
                ck = torch.load(self.current_path, map_location=self.device, weights_only=False)
                break
            except (EOFError, RuntimeError, OSError):  # being replaced
                time.sleep(0.2)
        else:
            return
        if self.current is None:
            self.current = FullGameNet(**ck["config"]).to(self.device).eval()
        self.current.load_state_dict(ck["model"])
        self.version, self.mtime = ck["version"], mtime

    def get(self, key: str) -> FullGameNet:
        if key == "current":
            if time.time() - self.checked > 5.0:
                self.reload()
            return self.current
        net = self.snapshots.pop(key, None)
        if net is None:
            net, _ = load(key, self.device)
        self.snapshots[key] = net  # most recently used last
        # the whole league stays loaded (~8 MB each on the GPU): with a cache of 4 and 8+ snapshots
        # it reloaded one from disk on almost every call (83% of an actor's time, the games waiting)
        while len(self.snapshots) > self.cfg["max_past"] + 2:
            self.snapshots.pop(next(iter(self.snapshots)))
        return net


class Inference:
    """Batches the network calls of an actor's agents (one per game side) on the GPU."""

    def __init__(self, nets: Nets, device, max_batch: int, compile_: bool = True):
        self.nets, self.device, self.max_batch = nets, device, max_batch
        # the current policy's calls compiled with CUDA graphs: the network is small, its calls
        # latency-bound (~150 kernels): 4.6 instead of 9.2 ms. Graphs need fixed shapes: the batch
        # padded to max_batch, the entities to MAX_ENT. Past snapshots (fewer calls) run eagerly.
        # every network's calls compiled (past snapshots too: eager ones took ~10 ms each, and a batch
        # with several snapshots made several; the inference thread is what the games wait for)
        self.compile = compile_ and device.type == "cuda"
        if self.compile:  # one compiled variant per network (the same code): more than dynamo's default 8
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 256)
        self._compiled = weakref.WeakKeyDictionary()  # net -> its compiled act (dropped with the net)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def request(self, items: list[tuple[str, dict]]) -> list[dict]:
        """[(net key, a view step)] -> [{order, tgt, bx, by, logp (per own unit), value, version}]
        (and the new state "h" for a network with memory, from the step's "h": the last one or None)."""
        done = threading.Event()
        out: list = [None] * len(items)
        for k, (key, st) in enumerate(items):
            self.q.put((key, st, out, k, done, len(items)))
        done.wait()
        return out

    def _run(self) -> None:
        pending: dict[int, int] = {}
        while True:
            batch = [self.q.get()]
            deadline = time.time() + 0.003  # a few ms for the other games' agents
            while True:
                try:
                    batch.append(self.q.get(timeout=max(0.0, deadline - time.time())))
                except queue.Empty:
                    break
            groups: dict[str, list] = {}
            for item in batch:
                groups.setdefault(item[0], []).append(item)
            for key, items in groups.items():
                try:
                    results = self._forward(self.nets.get(key), [it[1] for it in items])
                except Exception:  # noqa: BLE001 (the games must not hang on it)
                    traceback.print_exc()
                    results = [None] * len(items)
                version = self.nets.version if key == "current" else -1
                for (key_, st, out, k, done, n), res in zip(items, results):
                    if res is not None:
                        res["version"] = version
                    out[k] = res
                    pending[id(out)] = pending.get(id(out), 0) + 1
                    if pending[id(out)] == n:
                        del pending[id(out)]
                        done.set()

    @torch.no_grad()
    def _forward(self, net: FullGameNet, sts: list[dict]) -> list[dict]:
        fixed = self.compile and len(sts) <= self.max_batch
        B, E = (self.max_batch, fx.MAX_ENT) if fixed else (len(sts), max(st["n"] for st in sts))
        G = len(sts[0]["glob"])
        cuda = self.device.type == "cuda"
        if fixed and cuda:  # the same shapes every call: pinned host buffers, reused
            if getattr(self, "_host", None) is None:
                self._host = [torch.zeros(B, E, fx.F).pin_memory(), torch.zeros(B, E, dtype=torch.long).pin_memory(),
                              torch.zeros(B, E, dtype=torch.long).pin_memory(), torch.zeros(B, E, dtype=torch.bool).pin_memory(),
                              torch.zeros(B, G).pin_memory(), torch.zeros(B, dtype=torch.long).pin_memory(),
                              torch.ones(B, net.config["n_orders"], dtype=torch.bool).pin_memory()]
            host = self._host
            for h in host:
                h.zero_()
        else:
            host = [torch.zeros(B, E, fx.F), torch.zeros(B, E, dtype=torch.long), torch.zeros(B, E, dtype=torch.long),
                    torch.zeros(B, E, dtype=torch.bool), torch.zeros(B, G), torch.zeros(B, dtype=torch.long),
                    torch.ones(B, net.config["n_orders"], dtype=torch.bool)]
        if net.memory:  # the states in: a buffer of their own (the league's networks may differ in size)
            d = net.config["d"]
            if fixed and cuda:
                self._host_h = getattr(self, "_host_h", {})
                if d not in self._host_h:
                    self._host_h[d] = torch.zeros(B, d).pin_memory()
                host_h = self._host_h[d]
                host_h.zero_()
            else:
                host_h = torch.zeros(B, d)
            hs = host_h.numpy()
            for i, st in enumerate(sts):
                if st.get("h") is not None:
                    hs[i] = st["h"]
            host = host + [host_h]
        ent, typ, cur, mask, glob, n_own, avail = (h.numpy() for h in host[:7])
        avail[:] = True
        mask[:, 0] = True  # (padding rows: one entity, or attention over nothing gives NaNs)
        for i, st in enumerate(sts):
            n = st["n"]
            ent[i, :n], typ[i, :n], cur[i, :n], mask[i, :n] = st["ent"], st["type"], st["cur"], True
            glob[i], n_own[i] = st["glob"], min(st["n_own"], fx.MAX_OWN)
            if st.get("avail") is not None:
                avail[i] = st["avail"]
        x = [h.to(self.device, non_blocking=True) for h in host]
        if fixed and net not in self._compiled:
            self._compiled[net] = torch.compile(lambda *a, net=net: act(net, *a), mode="reduce-overhead", dynamic=False)
        out = (self._compiled[net] if fixed else lambda *a: act(net, *a))(*x)
        O = out["order"].shape[1]
        # everything back in one copy (seven were seven waits)
        packed = torch.cat([out["order"].float(), out["tgt"].float(), out["bx"].float(), out["by"].float(), out["logp"].float(),
                            out["value"].float()[:, None], out["entropy"].float()[:, None]]
                           + ([out["h"].float()] if net.memory else []), 1)
        if cuda:  # wait sleeping: CUDA's default sync spins a core the games need
            done = torch.cuda.Event(blocking=True)
            done.record()
            done.synchronize()
        p = packed.cpu().numpy()
        order, tgt, bx, by = (p[:, k * O:(k + 1) * O].astype(np.int64) for k in range(4))
        logp, value, entropy = p[:, 4 * O:5 * O], p[:, 5 * O], p[:, 5 * O + 1]
        res = []
        for i in range(len(sts)):
            o = n_own[i]
            res.append({"order": order[i, :o], "tgt": tgt[i, :o], "bx": bx[i, :o], "by": by[i, :o],
                        "logp": logp[i, :o].copy(), "value": float(value[i]), "entropy": float(entropy[i])})
            if net.memory:
                res[-1]["h"] = p[i, 5 * O + 2:].copy()
        return res


class Trajectory:
    """One trained side's steps; complete chunks (GAE with the values it acted with) go to the learner."""

    def __init__(self, cfg: dict, out_q, info: dict):
        self.cfg, self.out_q, self.info = cfg, out_q, info
        self.steps: list[dict] = []

    def add(self, st: dict, res: dict) -> None:
        n = st["n"]
        self.steps.append({"ent": st["ent"].astype(np.float16), "type": st["type"].astype(np.int16),
                           "cur": st["cur"].astype(np.int16), "glob": st["glob"], "n": n,
                           "n_own": min(st["n_own"], fx.MAX_OWN),
                           "order": res["order"].astype(np.int16), "tgt": res["tgt"].astype(np.int16),
                           "bx": res["bx"].astype(np.int16), "by": res["by"].astype(np.int16),
                           "logp": res["logp"].astype(np.float32), "value": res["value"], "reward": 0.0,
                           "avail": st.get("avail"),
                           "h": st.get("h"),  # the memory's state going in (None: the game's start, or none)
                           "done": False, "version": res["version"]})
        if len(self.steps) > self.cfg["chunk"]:
            self.flush(bootstrap=True)

    def end(self, reward: float) -> None:
        if self.steps:
            self.steps[-1]["reward"] = reward
            self.steps[-1]["done"] = True
            self.flush(bootstrap=False)

    def flush(self, bootstrap: bool) -> None:
        """GAE over the steps (the last one only bootstraps the value when the game goes on)."""
        gamma, lam = self.cfg["gamma"], self.cfg["lam"]
        steps = self.steps[:-1] if bootstrap else self.steps
        next_v = self.steps[-1]["value"] if bootstrap else 0.0
        adv = 0.0
        for s in reversed(steps):
            nonterminal = 0.0 if s["done"] else 1.0
            delta = s["reward"] + gamma * next_v * nonterminal - s["value"]
            adv = delta + gamma * lam * nonterminal * adv
            s["adv"], s["ret"] = adv, adv + s["value"]
            next_v = s["value"]
        if steps:
            self.out_q.put({"steps": steps, "info": self.info})
        self.steps = self.steps[-1:] if bootstrap else []


def potential(obs, side: int, values: dict, scale: float) -> float:
    """Reward shaping's potential: the side's material lead (what its living units and buildings
    cost, times their hit points left, minus the enemy's) over `scale`. Full information: the
    reward is not an input. Training a unit raises it, destroying enemy ones raises it."""
    m = material(obs, values)
    return (m[side] - m[1 - side]) / scale


def _choose(rng: random.Random, options: list[dict]) -> dict:
    total = sum(o["p"] for o in options)
    x = rng.random() * total
    for o in options:
        x -= o["p"]
        if x <= 0:
            return o
    return options[-1]


def game_loop(wid: int, k: int, cfg: dict, infer: Inference, out_q, stop, render_q) -> None:
    from ..runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup
    from .play import BCAgent

    rng = random.Random(cfg["seed"] * 1000 + wid * 16 + k)
    if cfg.get("avail_mask") and "costs_array" not in cfg:
        cfg["costs_array"] = np.asarray(cfg["costs"], np.int64)
    # videos: each actor's first game slot in turn (together one every video_every minutes); the
    # filmed game runs in a fresh process, so its replay holds just that game
    films = k == 0 and cfg["video_every"] > 0
    period = cfg["video_every"] * 60 * cfg["actors"]
    next_video = time.time() + wid * cfg["video_every"] * 60  # actor 0 at once
    vocab = json.loads(Path(cfg["vocab"]).read_text())
    spec_path = Path(cfg["run_dir"]) / "league_spec.json"
    name = f"fgsp{cfg['slot']}_{wid}_{k}"
    while not stop.is_set() and os.getppid() == cfg["learner_pid"]:
        try:
            spec = json.loads(spec_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)
            continue
        races = [rng.choice(cfg["races"]), rng.choice(cfg["races"])]
        if cfg["mirror"]:  # both sides the same race (balanced by construction)
            races[1] = races[0]
        side = rng.randrange(2)  # the learner's
        ai = _choose(rng, spec["launch"])  # {"kind": "agents"} or {"kind": "ai", "difficulty"[, "delay"]}
        slots = [Agent(races[0], handicap=cfg["handicap"]), Agent(races[1], handicap=cfg["handicap"])]
        late = ai["kind"] == "ai" and "delay" in ai  # the curriculum: the AI may start late (an agent slot till then)
        real = late and rng.random() < cfg["real_share"]  # the real game instead: the yardstick, no curriculum
        late = late and not real
        if ai["kind"] == "ai":
            slots[1 - side] = (Agent(races[1 - side], handicap=cfg["handicap"], difficulty=ai["difficulty"]) if late
                               else BuiltinAI(races[1 - side], ai["difficulty"], handicap=cfg["handicap"]))
            if late:  # (the handicap is the launch's: the level when it started)
                slots[side] = Agent(races[side], handicap=int(ai.get("handicap", cfg["handicap"])))
        agents_only = ai["kind"] != "ai"
        # without the built-in AI a restart resets the game by script (0.1 s; the AI does not survive
        # that, so games against it reload the map: ~5 s)
        setup = GameSetup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"],
                          max_game_seconds=cfg["max_minutes"] * 60, victory="decisive", window=(320, 240),
                          wait_floor_ms=cfg["wait_floor_ms"], melee_reset=agents_only and cfg["scripted_reset"],
                          native_obs=cfg["native_obs"])
        per_launch = cfg["games_per_process"] * (cfg["agent_games_factor"] if agents_only and cfg["scripted_reset"] else 1)
        try:
            with GameInstance(setup, name=name, timeout=120) as g:
                obs = g.start()
                fresh = True
                for n in range(per_launch):
                    if stop.is_set() or os.getppid() != cfg["learner_pid"]:
                        return
                    try:
                        spec = json.loads(spec_path.read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
                    opp = ({"name": f"script:ai-{ai['difficulty']}" + (" (real)" if real else ""), "kind": "ai"}
                           if ai["kind"] == "ai" else _choose(rng, spec["agents"]))
                    if late:  # the curriculum's current delay for this AI
                        delay = next((x.get("delay", 0.0) for x in spec["launch"] if x.get("difficulty") == ai["difficulty"]), 0.0)
                        opp["start"] = int(round(delay / cfg["step_seconds"]))
                    # a video: the first game of a launch (its replay then holds just this game)
                    film = films and fresh and time.time() >= next_video
                    ep = play_one(g, obs, cfg, vocab, races, side, opp, infer, out_q, wid, record=film)
                    if film:
                        next_video = time.time() + period
                        film_game(g, ep, cfg, wid, render_q)
                    if n + 1 < per_launch:
                        due = films and time.time() >= next_video
                        fresh = due or g._ended  # (a filmed game ended its process: the restart relaunches)
                        obs = g.restart(relaunch=due)
        except Exception as e:  # noqa: BLE001 (a new launch)
            print(f"actor {wid}/{k}: {type(e).__name__}: {e}", flush=True)
            time.sleep(2)


def play_one(g, obs, cfg: dict, vocab: dict, races: list[str], side: int, opp: dict, infer: Inference,
             out_q, wid: int, record: bool = False) -> dict:
    """One game from its first observation; `record`: also a trace for the video's panel
    (fullgame.overlay), in the returned episode row's "trace"."""
    from .play import BCAgent

    from ..protocol import StartAI
    race_ix = [fx.RACES.index(r) if r in fx.RACES else 0 for r in races]
    bots: dict[int, BCAgent] = {}
    keys: dict[int, str] = {}
    trajs: dict[int, Trajectory] = {}
    late = 1 - side if opp.get("start") is not None else None  # the built-in AI's side, idle until opp["start"]
    for s in (0, 1):
        if g.setup.slots[s].kind != "agent" or s == late:
            continue
        bot = BCAgent(None, vocab, s, None, costs=cfg.get("costs_array"))
        bot.begin(obs, race_ix)
        bots[s] = bot
        keys[s] = "current" if s == side or opp["kind"] == "self" else opp["path"]
        if keys[s] == "current":
            trajs[s] = Trajectory(cfg, out_q, {"worker": wid, "opponent": opp["name"]})
    t, t0 = 0, time.time()
    values, scale = cfg["values"], cfg["shaping_scale"]
    mat = material(obs, values)
    phi = {s: (mat[s] - mat[1 - s]) / scale for s in trajs}  # at the last recorded state
    ret = {s: 0.0 for s in trajs}
    trace: list[dict] = []
    names = {int(k): v for k, v in (cfg.get("order_names") or {}).items()}
    while True:
        rows = unit_rows(obs, t)  # once for both sides
        sts = {s: bot.observe(obs, t, rows) for s, bot in bots.items()}
        live = [s for s in bots if sts[s] is not None]
        for s in live:
            sts[s]["h"] = bots[s].h
        results = infer.request([(keys[s], sts[s]) for s in live]) if live else []
        cmds, spans = [], {}
        for s, res in zip(live, results):
            if res is None:
                continue
            c = bots[s].commands(sts[s], res["order"], res["tgt"], res["bx"], res["by"])
            bots[s].h = res.get("h")
            spans[s] = (len(cmds), len(cmds) + len(c))
            cmds += c
            if s in trajs:
                trajs[s].add(sts[s], res)
                phi[s] = (mat[s] - mat[1 - s]) / scale
        if record:
            chose = {s: Counter(fx.order_label(*bots[s].orders[int(c)], names) for c in r["order"] if c)
                     for s, r in zip(live, results) if r is not None}
            trace.append(trace_step(t, obs, mat, chose, value={s: r["value"] for s, r in zip(live, results) if r},
                                    entropy={s: r["entropy"] for s, r in zip(live, results) if r}))
        if late is not None and t == opp["start"]:  # the built-in AI takes over its side (after the bots' orders)
            cmds.append(StartAI(late))
        obs = g.step(cmds)
        for s, (a, b) in spans.items():
            bots[s].accepted(cmds[a:b], obs.command_results[a:b])
        t += 1
        mat = material(obs, values)
        if obs.game_over or t >= cfg["max_steps"]:
            break
        for s, tr in trajs.items():  # shaping: gamma * phi(s') - phi(s), for the step just taken
            if s in spans and tr.steps:  # (it recorded a step this time)
                r = cfg["shaping"] * (cfg["gamma"] * (mat[s] - mat[1 - s]) / scale - phi[s])
                if tr.steps:
                    tr.steps[-1]["reward"] += r
                    ret[s] += r
                    if record and trace:
                        trace[-1].setdefault("reward", {})[str(s)] = round(r, 4)
    outcome = {}
    for s in (0, 1):
        r = obs.players.get(s)
        res = r.result.name if r is not None else "TIE"
        outcome[s] = 1.0 if res == "VICTORY" else -1.0 if res == "DEFEAT" else 0.0
    lead = {s: (mat[s] - mat[1 - s]) / scale for s in (0, 1)}
    for s, tr in trajs.items():
        # the end: the outcome; a tie (the time limit) goes to the side ahead in material; and the
        # potential back to 0 (so the shaping only moves credit around and sums to ~0 over a game)
        terminal = outcome[s] if outcome[s] != 0.0 else cfg["tie_break"] * math.tanh(2.0 * lead[s])
        last = terminal - cfg["shaping"] * phi.get(s, 0.0)
        tr.end(last)
        ret[s] += last
        if record and trace:
            trace[-1].setdefault("reward", {})[str(s)] = round(last, 4)
    me, other = obs.players.get(side), obs.players.get(1 - side)
    ep = {
        "time": time.time(), "worker": wid, "opponent": opp["name"], "outcome": outcome[side],
        "return": round(ret.get(side, outcome[side]), 4), "material_lead": round(lead[side], 3), "length": t, "game_time": obs.game_time, "wall_seconds": round(time.time() - t0, 1),
        "races": races, "side": side, "race": races[side], "opponent_race": races[1 - side],
        "gold": me.gold_gathered if me else 0, "opponent_gold": other.gold_gathered if other else 0,
        "orders": bots[side].issued if side in bots else 0,
        **({"ai_delay": opp["start"] * cfg["step_seconds"], "handicap": g.setup.slots[side].handicap}
           if late is not None else {})}
    out_q.put({"episode": ep})
    if record:  # the video's panel: A = the learner's side
        who = {"self": "itself", "ai": "built-in AI"}.get(opp["kind"], opp["name"])
        other = (f"built-in AI {opp['name'].split('-', 1)[1]}" if opp["kind"] == "ai"
                 else "itself" if opp["kind"] == "self" else f"snapshot {opp['name']}")
        ep["trace"] = {"steps": trace, "gamma": cfg["gamma"], "outcome": {str(s): v for s, v in outcome.items()},
                       "title": f"{Path(cfg['run_dir']).name} · policy v{infer.nets.version} · vs {who}",
                       "sides": [{"player": side, "name": f"learner ({races[side]})", "kind": "agent"},
                                 {"player": 1 - side, "name": f"{other} ({races[1 - side]})",
                                  "kind": "ai" if opp["kind"] == "ai" else "agent"}]}
    return ep


def film_game(g, ep: dict, cfg: dict, wid: int, render_q) -> None:
    """Save the game's replay (and the trace for its panel) and queue it for the renderer
    process (render_main): rendering in the actor took its games' GIL for minutes."""
    run_dir = Path(cfg["run_dir"])
    stem = f"game-{int(ep['time'])}-a{wid}"
    try:
        replay = g.save_replay(run_dir / "replays" / f"{stem}.w3g")
    except Exception as e:  # noqa: BLE001
        print(f"video: replay not saved: {e}", flush=True)
        return
    trace = ep.pop("trace", None)
    if trace is not None:
        replay.with_suffix(".trace.json").write_text(json.dumps(trace))
    row = {"episode": ep.get("episode_id", ""), "title": f"{ep['race']} vs {ep['opponent_race']} ({ep['opponent']})",
           "outcome": ep["outcome"], "return": ep["return"],
           "sub": f"{ep['game_time'] / 60:.1f} game minutes · material lead {ep['material_lead']:+.2f} · return {ep['return']:+.2f}",
           "opponent": ep["opponent"], "game_time": ep["game_time"]}
    render_q.put((g.setup, str(replay), stem, trace, row, time.time()))


def render_main(cfg: dict, render_q, stop) -> None:
    """The run's video renderer: one game at a time, real time (the game's own renderer and sound);
    listed in media.jsonl for the dashboard. When it falls behind, the older games are skipped."""
    import signal

    from ..runtime import reaper
    from ..video import render_replay
    from .overlay import FullGameOverlay
    signal.signal(signal.SIGTERM, _exit)
    run_dir = Path(cfg["run_dir"])
    try:
        while not stop.is_set() and os.getppid() == cfg["learner_pid"]:
            try:
                job = render_q.get(timeout=2)
            except queue.Empty:
                continue
            while True:  # the newest waiting game
                try:
                    job = render_q.get_nowait()
                except queue.Empty:
                    break
            setup, replay, stem, trace, row, t = job
            try:
                out = render_replay(setup, Path(replay), run_dir / "videos" / f"{stem}.mp4", name=f"fgvid{cfg['slot']}",
                                    crf=28, overlay=FullGameOverlay(trace) if trace else None, fit_all=True,
                                    max_steps=cfg["max_steps"] + 40)
                with open(run_dir / "media.jsonl", "a") as f:
                    f.write(json.dumps({"time": time.time(), "kind": "video", "file": str(out.relative_to(run_dir)),
                                        **row}) + "\n")
            except Exception as e:  # noqa: BLE001 (one video fewer)
                print(f"video: {stem} not rendered: {e}", flush=True)
    finally:
        reaper.reap()


def _exit(*_):
    raise SystemExit(0)


def _interrupt(*_):
    raise KeyboardInterrupt


def actor_main(wid: int, cfg: dict, out_q, stop, render_q) -> None:
    import signal

    from ..runtime import reaper
    signal.signal(signal.SIGTERM, _exit)
    out_q.cancel_join_thread()  # exiting must not wait to flush trajectories nobody reads any more
    torch.set_num_threads(1)
    try:
        device = torch.device(cfg["device"])
        nets = Nets(cfg, device)
        while nets.current is None and not stop.is_set() and os.getppid() == cfg["learner_pid"]:
            time.sleep(1)
            nets.reload()
        infer = Inference(nets, device, 2 * cfg["games_per_actor"], cfg["compile"])
        threads = [threading.Thread(target=game_loop, args=(wid, k, cfg, infer, out_q, stop, render_q), daemon=True)
                   for k in range(cfg["games_per_actor"])]
        for th in threads:
            th.start()
            time.sleep(2)  # launches spread out
        while not stop.is_set() and os.getppid() == cfg["learner_pid"]:  # the learner gone (killed): stop too
            time.sleep(1)
    finally:
        reaper.reap()  # the games (multiprocessing children skip atexit)


# ---- the league -----------------------------------------------------------------------------------

class League:
    def __init__(self, run_dir: Path, ai: list[str], shares: dict, max_past: int, pfsp: str,
                 curriculum: tuple[float, float, float, int] | None = None, hp: bool = True):
        """`curriculum`: (start level, step, max delay in seconds, base handicap): games against the
        built-in AI get easier or harder towards a 50% score, a level per difficulty in [0, 1] that
        a loss raises by `step`, a win lowers (a tie leaves it). From 0 to 0.5 the learner's units
        get more hit points (its handicap from the base to 100, the AI's staying at the base); from
        0.5 to 1 also the AI starts late (its units idle till then; protocol.StartAI), up to the max
        delay. At 0 the game is the real one. None: no curriculum. `hp` False: the level is the late
        start alone (0 to the max delay), the hit points stay even (twice the hit points let a small
        army win fights: the learner stopped needing to spend, which it needs in the real game)."""
        self.run_dir, self.shares, self.max_past, self.pfsp = run_dir, shares, max_past, pfsp
        self.scripts = {f"script:ai-{d}": Member(f"script:ai-{d}") for d in ai}
        self.rule, self.hp = curriculum, hp
        self.level = {n: curriculum[0] for n in self.scripts} if curriculum else {}
        # with a curriculum some launches play the real game (the yardstick: "script:ai-X (real)")
        self.real = {f"{n} (real)": Member(f"{n} (real)") for n in self.scripts} if curriculum else {}
        self.past: list[Member] = []
        self.self_member = Member("self")

    def handicap_delay(self, name: str) -> tuple[int, float]:
        """The learner's handicap (hit points in percent) and the AI's delay (seconds) at `name`'s level."""
        _, _, top, base = self.rule
        lv = self.level[name]
        if not self.hp:
            return base, round(lv * top, 1)
        hp = base + int(min(1.0, 2 * lv) * (100 - base) / 10.0 + 0.5) * 10  # (handicaps in steps of 10)
        return hp, round(max(0.0, 2 * lv - 1) * top, 1)

    def curriculum(self, name: str, outcome: float) -> None:
        """A game against the built-in AI `name` ended (+1 / 0 / -1 for the learner): its level moves."""
        if name in self.level:
            self.level[name] = min(1.0, max(0.0, self.level[name] - self.rule[1] * outcome))

    def member(self, name: str) -> Member | None:
        if name == "self":
            return self.self_member
        return self.scripts.get(name) or self.real.get(name) or next((m for m in self.past if m.name == name), None)

    def add_snapshot(self, path: Path, steps: int) -> None:
        self.past.append(Member(f"past:{steps}", path=str(path), steps=steps))
        if len(self.past) > self.max_past:  # keep the first (the clone) and the newest
            del self.past[1]

    def restore(self, summary: dict) -> None:
        """Members and records from a league.json (resuming a run)."""
        for row in summary.get("members", []):
            if row["name"] in self.scripts or row["name"] in self.real:
                m = self.scripts.get(row["name"]) or self.real[row["name"]]
            elif row.get("path") and Path(row["path"]).exists():
                m = Member(row["name"], path=row["path"], steps=row.get("steps", 0))
                self.past.append(m)
            else:
                continue
            m.wins, m.losses, m.draws = row.get("wins", 0.0), row.get("losses", 0.0), row.get("draws", 0.0)
            m.recent = list(row.get("recent") or [])
        for n, v in (summary.get("level") or {}).items():
            if n in self.level:
                self.level[n] = v
        me = summary.get("self") or {}
        self.self_member.wins, self.self_member.losses = me.get("wins", 0.0), me.get("losses", 0.0)
        self.self_member.draws, self.self_member.recent = me.get("draws", 0.0), list(me.get("recent") or [])

    def spec(self) -> dict:
        """What the actors draw from: per launch built-in AI or agents; per agent game an opponent."""
        ai_share = self.shares["ai"] if self.scripts else 0.0
        launch = [{"kind": "agents", "p": 1.0 - ai_share}]
        launch += [{"kind": "ai", "difficulty": n.split("-", 1)[1], "p": ai_share / len(self.scripts),
                    **(dict(zip(("handicap", "delay"), self.handicap_delay(n)), level=self.level[n]) if n in self.level else {})}
                   for n in self.scripts]
        agents = [{"name": "self", "kind": "self", "p": self.shares["self"] if self.past else 1.0}]
        if self.past:
            w = [pfsp_weight(m.win_rate(), self.pfsp) for m in self.past]
            total = sum(w) or 1.0
            agents += [{"name": m.name, "kind": "past", "path": m.path, "p": self.shares["past"] * x / total}
                       for m, x in zip(self.past, w)]
        return {"launch": launch, "agents": agents}

    def write(self) -> None:
        spec = self.run_dir / "league_spec.json"
        tmp = spec.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.spec()))
        tmp.replace(spec)

        def row(m: Member) -> dict:
            p = m.win_rate()
            return {"name": m.name, "steps": m.steps, "games": m.games, "wins": m.wins, "losses": m.losses,
                    "draws": m.draws, "win_rate": p, "weight": pfsp_weight(p, self.pfsp) if m.path else None,
                    "path": m.path, "recent": m.recent}
        summary = {"pfsp": self.pfsp, "members": [row(m) for m in [*self.scripts.values(), *self.real.values(), *self.past]],
                   "self": row(self.self_member), "level": dict(self.level)}
        tmp = self.run_dir / "league.json.tmp"
        tmp.write_text(json.dumps(summary))
        tmp.replace(self.run_dir / "league.json")

    def train_keys(self) -> dict:
        out = {f"league/{n}": m.win_rate() for n, m in [*self.scripts.items(), *self.real.items()] if m.win_rate() is not None}
        past = [x for m in self.past for x in m.recent[-50:]]
        if past:
            out["league/past"] = sum(past) / len(past)
        if self.self_member.win_rate() is not None:
            out["league/self"] = self.self_member.win_rate()
        out["league/members"] = len(self.past)
        for n in self.level:  # the curriculum: its level, the learner's hit points, the AI's late start
            hp, delay = self.handicap_delay(n)
            short = n.split(":", 1)[1]
            out.update({f"curriculum/{short} level": self.level[n], f"curriculum/{short} handicap": hp,
                        f"curriculum/{short} delay": delay})
        return out


# ---- the learner ----------------------------------------------------------------------------------

def collate(steps: list[dict], device) -> dict:
    B, E = len(steps), max(s["n"] for s in steps)
    O = min(fx.MAX_OWN, E)
    G = len(steps[0]["glob"])
    ent = np.zeros((B, E, fx.F), np.float32)
    typ, cur = np.zeros((B, E), np.int64), np.zeros((B, E), np.int64)
    mask = np.zeros((B, E), bool)
    glob = np.zeros((B, G), np.float32)
    n_own = np.zeros(B, np.int64)
    acts = {k: np.zeros((B, O), np.int64) for k in ("order", "tgt", "bx", "by")}
    logp = np.zeros((B, O), np.float32)
    n_orders = next((len(s["avail"]) for s in steps if s.get("avail") is not None), 0)
    avail = np.ones((B, n_orders), bool) if n_orders else None
    for i, s in enumerate(steps):
        if avail is not None and s.get("avail") is not None:
            avail[i] = s["avail"]
        n, o = s["n"], s["n_own"]
        ent[i, :n], typ[i, :n], cur[i, :n], mask[i, :n], glob[i], n_own[i] = s["ent"], s["type"], s["cur"], True, s["glob"], o
        for k in acts:
            acts[k][i, :o] = s[k]
        logp[i, :o] = s["logp"]
    t = lambda a: torch.from_numpy(a).to(device, non_blocking=True)  # noqa: E731
    return {"ent": t(ent), "type": t(typ), "cur": t(cur), "mask": t(mask), "glob": t(glob), "n_own": t(n_own),
            **{k: t(v) for k, v in acts.items()}, "logp": t(logp), "avail": t(avail) if avail is not None else None,
            "adv": torch.tensor([s["adv"] for s in steps], device=device, dtype=torch.float32),
            "ret": torch.tensor([s["ret"] for s in steps], device=device, dtype=torch.float32),
            "own": torch.arange(O, device=device)[None] < t(n_own)[:, None]}


def pieces(chunks: list[list[dict]], seq_len: int) -> list[list[dict]]:
    """The chunks (consecutive steps of one game side) cut into sequences of at most seq_len steps."""
    return [c[a:a + seq_len] for c in chunks for a in range(0, len(c), seq_len)]


def collate_seq(seqs: list[list[dict]], device) -> dict:
    """Sequences as collate() batches of B * T steps (T: the longest; shorter ones padded with steps
    that have no units, "valid" False), plus "seq" for evaluate(): (B, T, the first steps' states
    [B, d], no starts: a sequence stays in one game)."""
    B, T = len(seqs), max(len(q) for q in seqs)
    first = seqs[0][0]
    G = len(first["glob"])
    pad = {"ent": np.zeros((1, fx.F), np.float16), "type": np.zeros(1, np.int16), "cur": np.zeros(1, np.int16),
           "glob": np.zeros(G, np.float32), "n": 1, "n_own": 0, "logp": np.zeros(0, np.float32), "adv": 0.0, "ret": 0.0,
           "avail": None, **{k: np.zeros(0, np.int16) for k in ("order", "tgt", "bx", "by")}}
    flat = [q[t] if t < len(q) else pad for q in seqs for t in range(T)]
    mb = collate(flat, device)
    mb["valid"] = torch.tensor([t < len(q) for q in seqs for t in range(T)], device=device)
    d = next((len(q[0]["h"]) for q in seqs if q[0].get("h") is not None), None)
    h0 = None
    if d is not None:
        h0 = torch.from_numpy(np.stack([q[0]["h"] if q[0].get("h") is not None else np.zeros(d, np.float32)
                                        for q in seqs])).to(device)
    mb["seq"] = (B, T, h0, None)
    return mb


def ppo_update(net: FullGameNet, ref: FullGameNet | None, opt, steps: list[dict], args, warmup: bool, device,
               chunks: list[list[dict]] | None = None) -> dict:
    """PPO epochs over the steps; for a network with memory over sequences cut from `chunks` (the
    actors' pieces of trajectories, in order), from the states the actors had."""
    advs = np.array([s["adv"] for s in steps], np.float32)
    mean, std = float(advs.mean()), float(advs.std()) + 1e-8
    stats: dict[str, list[float]] = {}
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(device.type == "cuda" and args.bf16))
    net.train()
    seqs = pieces(chunks, args.seq_len) if net.memory else None
    per_mb = max(1, args.minibatch // args.seq_len)
    for _ in range(args.epochs):
        if seqs is None:
            order = np.random.permutation(len(steps))
            batches = (collate([steps[i] for i in order[a:a + args.minibatch]], device)
                       for a in range(0, len(order), args.minibatch))
        else:
            order = np.random.permutation(len(seqs))
            batches = (collate_seq([seqs[i] for i in order[a:a + per_mb]], device) for a in range(0, len(order), per_mb))
        for mb in batches:
            seq = mb.get("seq")
            with autocast:  # bf16 matmuls (the losses and log-probabilities in fp32)
                ev = evaluate(net, mb["ent"], mb["type"], mb["cur"], mb["mask"], mb["glob"], mb["n_own"],
                              mb["order"], mb["tgt"], mb["bx"], mb["by"], mb["avail"], seq=seq)
            ev["value"] = ev["value"].float()
            own = mb["own"].float()
            n_units = own.sum().clamp(min=1)
            adv = ((mb["adv"] - mean) / std)[:, None]
            log_ratio = (ev["logp"] - mb["logp"]) * own
            ratio = log_ratio.exp()
            pg = -(torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv) * own).sum() / n_units
            if "valid" in mb:  # (the padding steps: no units, no value target)
                valid = mb["valid"].float()
                v_loss = 0.5 * (((ev["value"] - mb["ret"]) ** 2) * valid).sum() / valid.sum()
            else:
                v_loss = 0.5 * ((ev["value"] - mb["ret"]) ** 2).mean()
            entropy = (ev["entropy"] * own).sum() / n_units
            loss = args.vf_coef * v_loss - args.ent_coef * entropy
            if not warmup:
                loss = loss + pg
            ref_kl = torch.zeros((), device=device)
            if ref is not None and args.ref_kl > 0:
                with torch.no_grad(), autocast:
                    ev_ref = evaluate(ref, mb["ent"], mb["type"], mb["cur"], mb["mask"], mb["glob"], mb["n_own"],
                                      mb["order"], mb["tgt"], mb["bx"], mb["by"], mb["avail"],
                                      seq=seq if ref.memory else None)
                lp, lp_ref = torch.log_softmax(ev["logits"].float(), -1), torch.log_softmax(ev_ref["logits"].float(), -1)
                kl = (lp.exp() * (lp - lp_ref)).sum(-1)
                ref_kl = (kl * own).sum() / n_units
                # also while the value warms up: it shares the network's trunk, so training it alone
                # moved the policy too (the KL to the clone doubled in the first five updates)
                loss = loss + args.ref_kl * ref_kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            opt.step()
            with torch.no_grad():
                approx_kl = (((ratio - 1) - log_ratio) * own).sum() / n_units
                clipfrac = (((ratio - 1).abs() > args.clip).float() * own).sum() / n_units
            for k, v in (("loss/policy", pg), ("loss/value", v_loss), ("loss/entropy", entropy), ("loss/kl", approx_kl),
                         ("loss/clipfrac", clipfrac), ("loss/ref_kl", ref_kl), ("grad_norm", gn)):
                stats.setdefault(k, []).append(float(v))
    net.eval()
    return {k: sum(v) / len(v) for k, v in stats.items()}


def publish(net: FullGameNet, version: int, run_dir: Path) -> None:
    tmp = run_dir / "current.pt.tmp"
    torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "config": net.config,
                "version": version}, tmp)
    tmp.replace(run_dir / "current.pt")


def git_info() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        subject = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "warcraftsim"], cwd=root, capture_output=True,
                                    text=True, timeout=10).stdout.strip())
        return {"commit": commit, "subject": subject, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", type=Path, required=True, help="a behavior-cloned policy (fullgame/bc.py) or checkpoint")
    ap.add_argument("--memory", type=int, default=-1, help="1: the policy gets a memory core (a new one does nothing "
                                                            "at first), 0: none; -1: as --init")
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--timesteps", type=float, default=50e6, help="agent steps to train on")
    ap.add_argument("--actors", type=int, default=4, help="actor processes (each a GPU context: few, with many games)")
    ap.add_argument("--games-per-actor", type=int, default=8)
    ap.add_argument("--games-per-process", type=int, default=6, help="games of one setup per launch")
    ap.add_argument("--scripted-reset", type=int, default=1, help="games without the built-in AI restart by script")
    ap.add_argument("--agent-games-factor", type=int, default=4, help="those run this many times as many games per launch")
    ap.add_argument("--map", default="duelrush")
    ap.add_argument("--races", default="all")
    ap.add_argument("--mirror", type=int, default=1, help="both sides play the same race (on duelrush the races "
                                                           "are far from balanced: night elf beat the others 96%%)")
    ap.add_argument("--handicap", type=int, default=50)
    ap.add_argument("--step-seconds", type=float, default=0.5)
    ap.add_argument("--max-minutes", type=float, default=4.0)
    ap.add_argument("--wait-floor-ms", type=int, default=5)
    ap.add_argument("--ai", default="easy,normal", help="built-in AI anchors: difficulties (comma-separated; '' for none)")
    ap.add_argument("--ai-share", type=float, default=0.25, help="share of launches against the built-in AI")
    ap.add_argument("--curriculum", type=float, default=0.75,
                    help="the starting level of the curriculum against the built-in AI (League: the learner's hit "
                         "points up to twice the AI's, then the AI starting late), moved towards a 50%% score; "
                         "-1: none, the real game")
    ap.add_argument("--curriculum-step", type=float, default=0.02, help="how much a loss raises the level (a win lowers it)")
    ap.add_argument("--curriculum-delay", type=float, default=120.0, help="the AI's late start at level 1 (seconds)")
    ap.add_argument("--real-share", type=float, default=0.1, help="with a curriculum: the share of launches against "
                                                                   "the built-in AI that play the real game (the yardstick)")
    ap.add_argument("--curriculum-hp", type=int, default=1, help="0: the level is the AI's late start alone, the hit "
                                                                 "points even (1: up to twice the learner's first)")
    ap.add_argument("--self-share", type=float, default=0.5, help="of the agent games: against itself")
    ap.add_argument("--snapshot-every", type=int, default=20, help="updates between league snapshots")
    ap.add_argument("--max-past", type=int, default=12)
    ap.add_argument("--pfsp", default="hard")
    ap.add_argument("--batch-steps", type=int, default=8192, help="agent steps per update")
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--ref-kl", type=float, default=0.05, help="the KL term towards the initial (cloned) policy")
    ap.add_argument("--value-warmup", type=int, default=-1, help="first updates: the value head only (default: 10, "
                                                                  "2 when BC trained the value head)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--bf16", type=int, default=1, help="the updates' matmuls in bfloat16")
    ap.add_argument("--compile", type=int, default=1, help="the actors' network calls compiled (CUDA graphs)")
    ap.add_argument("--native-obs", type=int, default=1, help="observations parsed in C (warcraftsim.native)")
    ap.add_argument("--avail-mask", type=int, default=1, help="mask the orders the player can't pay for yet")
    ap.add_argument("--chunk", type=int, default=64, help="steps per trajectory piece an actor sends")
    ap.add_argument("--seq-len", type=int, default=16, help="a network with memory: steps per training sequence "
                                                            "(from the state the actor had at its first)")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--video-every", type=float, default=4.0, help="minutes between game videos (0: none; the actors take turns)")
    ap.add_argument("--shaping", type=float, default=1.0, help="weight of the material-lead reward shaping (0: none)")
    ap.add_argument("--shaping-scale", type=float, default=2000.0, help="material (gold + lumber cost) worth 1 of potential")
    ap.add_argument("--tie-break", type=float, default=0.5, help="a tie's reward: this times tanh(2 x material lead)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", help="default: cuda if available (asking opens the GPU driver)")
    ap.add_argument("--resume", action="store_true", help="continue the run --name: its latest checkpoint, league and counts")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    import signal
    signal.signal(signal.SIGTERM, _interrupt)  # kill: a clean stop (status, a checkpoint, the actors)

    run_dir = args.runs / args.name
    for sub_dir in ("checkpoints", "replays", "videos"):
        (run_dir / sub_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(4)  # the games need the cores
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net, ck = load(args.init, device, None if args.memory < 0 else bool(args.memory))
    vocab = ck["vocab"]
    trained_for = ck.get("value_reward")  # the rewards BC's value head learned (fullgame/bc.py)
    ours = {"gamma": args.gamma, "shaping": args.shaping, "shaping_scale": args.shaping_scale, "tie_break": args.tie_break}
    if trained_for and trained_for != ours:
        print(f"note: the value head learned the returns of {trained_for}, this run's rewards are {ours}", flush=True)
    if args.value_warmup < 0:  # a value head BC trained on these rewards needs only a short warm-up
        args.value_warmup = 2 if trained_for == ours else 10
    (run_dir / "vocab.json").write_text(json.dumps(vocab))
    ref = copy.deepcopy(net).eval() if args.ref_kl > 0 else None
    if ref is not None:
        for p in ref.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    slot, slot_lock = claim_slot(args.runs)
    races = list(fx.RACES) if args.races == "all" else args.races.split(",")
    ai = [d for d in args.ai.split(",") if d]
    init_rel = os.path.relpath(args.init.resolve(), args.runs.resolve().parent)
    info = {"name": args.name, "kind": "run", "trainer": "fullgame", "task": f"fullgame_{args.map}",
            "status": "launching", "created": time.time(), "timesteps": int(args.timesteps),
            "envs": args.actors * args.games_per_actor, "init_from": init_rel, "init_steps": 0,
            "description": (f"Self-play on whole games ({args.map}) from a behavior-cloned policy: PPO per unit, "
                            f"a league of itself, past snapshots and the built-in AI ({', '.join(ai) or 'none'})"),
            "config": {k: str(v) for k, v in vars(args).items()},
            "spaces": fx.describe_spaces(vocab, None if vocab.get("order_names") else fx.demo_order_names(args.runs),
                                         "two per game against itself or a past snapshot, one against the built-in AI"),
            "launch": {"command": "python3 -m warcraftsim.fullgame.selfplay " + " ".join(sys.argv[1:] if argv is None else argv),
                       "git": git_info()}}

    resumed = None
    if args.resume and (run_dir / "run.json").exists():  # the run goes on: its weights, league and counts
        old = json.loads((run_dir / "run.json").read_text())
        rows = [json.loads(line) for line in open(run_dir / "train.jsonl")] if (run_dir / "train.jsonl").exists() else []
        cks = sorted((run_dir / "checkpoints").glob("[0-9]*.pt"))
        if cks:
            last = torch.load(cks[-1], map_location=device, weights_only=False)
            net.load_state_dict(last["model"])
        resumed = {"steps": rows[-1]["agent_steps"] if rows else 0, "update": rows[-1]["epoch"] if rows else 0,
                   "episodes": sum(1 for _ in open(run_dir / "episodes.jsonl")) if (run_dir / "episodes.jsonl").exists() else 0,
                   "checkpoint": cks[-1].name if cks else None}
        info.update(created=old.get("created", info["created"]), init_from=old.get("init_from", info["init_from"]),
                    resumes=old.get("resumes", []) + [{"time": time.time(), **resumed}])

    def save_info():
        tmp = run_dir / "run.json.tmp"
        tmp.write_text(json.dumps(info, indent=1))
        tmp.replace(run_dir / "run.json")
    save_info()
    if args.note:
        (run_dir / "notes.md").write_text(args.note + "\n")

    league = League(run_dir, ai, {"ai": args.ai_share, "self": args.self_share, "past": 1.0 - args.self_share},
                    args.max_past, args.pfsp,
                    curriculum=((args.curriculum, args.curriculum_step, args.curriculum_delay, args.handicap)
                                if args.curriculum >= 0 else None), hp=bool(args.curriculum_hp))
    if resumed is not None and (run_dir / "league.json").exists():
        league.restore(json.loads((run_dir / "league.json").read_text()))
    else:
        first = run_dir / "checkpoints" / f"{0:016d}.pt"
        torch.save({**ck, "model": net.state_dict(), "config": net.config, "vocab": vocab, "agent_steps": 0}, first)
        league.add_snapshot(first, 0)  # the clone itself: the first past opponent
    league.write()
    publish(net, resumed["update"] if resumed else 0, run_dir)
    cfg = {"run_dir": str(run_dir), "device": str(device), "vocab": str(run_dir / "vocab.json"), "races": races,
           "map": args.map, "handicap": args.handicap, "step_seconds": args.step_seconds,
           "max_minutes": args.max_minutes, "max_steps": int(args.max_minutes * 60 / args.step_seconds) + 20,
           "wait_floor_ms": args.wait_floor_ms, "games_per_actor": args.games_per_actor,
           "games_per_process": args.games_per_process, "chunk": args.chunk, "gamma": args.gamma, "lam": args.lam,
           "seed": args.seed, "slot": slot, "video_every": args.video_every, "scripted_reset": bool(args.scripted_reset),
           "agent_games_factor": args.agent_games_factor, "mirror": bool(args.mirror), "learner_pid": os.getpid(),
           "actors": args.actors, "values": unit_values(), "shaping": args.shaping, "shaping_scale": args.shaping_scale,
           "tie_break": args.tie_break, "compile": bool(args.compile), "native_obs": bool(args.native_obs),
           "real_share": args.real_share,
           "avail_mask": bool(args.avail_mask),
           "costs": order_costs(vocab, args.map).tolist() if args.avail_mask else None, "max_past": args.max_past,
           "order_names": vocab.get("order_names") or {str(k): v for k, v in fx.demo_order_names(args.runs).items()}}
    # the actors compile their network calls: without this each starts a pool of ~32 compile workers
    # (~100 processes, several GB, idle after the first seconds)
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    ctx = torch.multiprocessing.get_context("spawn")
    out_q, stop, render_q = ctx.Queue(maxsize=4096), ctx.Event(), ctx.Queue()
    actors = [ctx.Process(target=actor_main, args=(w, cfg, out_q, stop, render_q), daemon=True) for w in range(args.actors)]
    if args.video_every > 0:
        actors.append(ctx.Process(target=render_main, args=(cfg, render_q, stop), daemon=True))
    for p in actors:
        p.start()
    info.update(status="training", started=time.time())
    save_info()
    t0 = time.time()
    agent_steps, update, episodes = ((resumed["steps"], resumed["update"], resumed["episodes"]) if resumed
                                     else (0, 0, 0))
    buf: list[dict] = []
    chunks: list[list[dict]] = []
    stale = []
    try:
        while agent_steps < args.timesteps:
            t_collect = time.time()
            while len(buf) < args.batch_steps:
                try:
                    msg = out_q.get(timeout=600)
                except queue.Empty:
                    raise RuntimeError("no trajectories from the actors for 10 minutes")
                if "episode" in msg:
                    e = msg["episode"]
                    episodes += 1
                    e["episode"] = episodes
                    m = league.member(e["opponent"])
                    if m is not None:
                        m.record(e["outcome"])
                    if "ai_delay" in e:
                        league.curriculum(e["opponent"], e["outcome"])
                    with open(run_dir / "episodes.jsonl", "a") as f:
                        f.write(json.dumps(e) + "\n")
                    continue
                buf += msg["steps"]
                chunks.append(msg["steps"])
                stale += [update - s["version"] for s in msg["steps"] if s["version"] >= 0]
            t_train = time.time()
            steps, buf, batch_chunks, chunks = buf, [], chunks, []
            warmup = update < args.value_warmup
            stats = ppo_update(net, ref, opt, steps, args, warmup, device, batch_chunks)
            update += 1
            agent_steps += len(steps)
            publish(net, update, run_dir)
            if update % args.snapshot_every == 0:
                path = run_dir / "checkpoints" / f"{agent_steps:016d}.pt"
                torch.save({"model": net.state_dict(), "config": net.config, "vocab": vocab, "agent_steps": agent_steps,
                            "update": update}, path)
                league.add_snapshot(path, agent_steps)
            elif update % args.checkpoint_every == 0:
                torch.save({"model": net.state_dict(), "config": net.config, "vocab": vocab, "agent_steps": agent_steps,
                            "update": update}, run_dir / "checkpoints" / f"{agent_steps:016d}.pt")
            league.write()
            now = time.time()
            row = {"agent_steps": agent_steps, "epoch": update, "time": now, "uptime": now - t0,
                   "SPS": len(steps) / max(now - t_collect, 1e-9), "lr": args.lr, "value_warmup": warmup,
                   "perf/rollout": t_train - t_collect, "perf/train": now - t_train,
                   "staleness": sum(stale) / len(stale) if stale else 0.0, **stats, **league.train_keys()}
            stale = []
            with open(run_dir / "train.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"update {update}: {agent_steps} steps, {row['SPS']:.0f} steps/s, {episodes} games; "
                  f"policy {stats.get('loss/policy', 0):.4f} value {stats.get('loss/value', 0):.4f} "
                  f"kl {stats.get('loss/kl', 0):.4f} ref_kl {stats.get('loss/ref_kl', 0):.4f} "
                  f"{json.dumps(league.train_keys())}", flush=True)
        info["status"] = "finished"
    except KeyboardInterrupt:
        info["status"] = "stopped"
    except Exception as e:  # noqa: BLE001
        info["status"] = f"failed: {type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        info["finished"] = time.time()
        save_info()
        torch.save({"model": net.state_dict(), "config": net.config, "vocab": vocab, "agent_steps": agent_steps,
                    "update": update}, run_dir / "checkpoints" / f"{agent_steps:016d}.pt")
        stop.set()
        for p in actors:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
