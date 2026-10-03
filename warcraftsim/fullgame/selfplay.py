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
import types
import weakref
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import features as fx
from .collect import claim_slot
from .costs import order_costs
from .trace import DEAD_FLAG, Production, material, trace_step, unit_values  # noqa: F401
from .play import unit_rows
from .model import FullGameNet, act, evaluate, load
from ..rl.league import Member, pfsp_weight

RUNS = Path(__file__).resolve().parents[2] / "runs"


# ---- actors ---------------------------------------------------------------------------------------

LIVE = ("current", "exploiter")  # the networks the learner trains (published, reloaded); else past snapshots


class Nets:
    """An actor's networks: the learner's current weights and its exploiter's (reloaded when the
    learner publishes new ones) and past snapshots (a few, most recently used)."""

    def __init__(self, cfg: dict, device):
        self.cfg, self.device = cfg, device
        self.current_path = Path(cfg["run_dir"]) / "current.pt"
        self.exploiter_path = Path(cfg["run_dir"]) / "exploiter.pt"
        self.current: FullGameNet | None = None
        self.exploiter: FullGameNet | None = None
        self.version = self.exploiter_version = -1
        self.mtime = self.exploiter_mtime = 0.0
        self.checked = 0.0
        self.snapshots: dict[str, FullGameNet] = {}
        self.reload()

    def _live(self, path: Path, net: FullGameNet | None, version: int, mtime: float) -> tuple:
        try:
            m = path.stat().st_mtime
        except FileNotFoundError:
            return net, version, mtime
        if m == mtime:
            return net, version, mtime
        for _ in range(3):
            try:
                ck = torch.load(path, map_location=self.device, weights_only=False)
                break
            except (EOFError, RuntimeError, OSError):  # being replaced
                time.sleep(0.2)
        else:
            return net, version, mtime
        if net is None:
            net = FullGameNet(**ck["config"]).to(self.device).eval()
        net.load_state_dict(ck["model"])  # (in place: CUDA graphs captured from it stay valid)
        return net, ck["version"], m

    def reload(self) -> None:
        self.checked = time.time()
        self.current, self.version, self.mtime = self._live(self.current_path, self.current, self.version, self.mtime)
        if self.cfg.get("exploiter_share", 0) > 0:
            self.exploiter, self.exploiter_version, self.exploiter_mtime = self._live(
                self.exploiter_path, self.exploiter, self.exploiter_version, self.exploiter_mtime)

    def live_version(self, key: str) -> int:
        return self.version if key == "current" else self.exploiter_version if key == "exploiter" else -1

    def get(self, key: str) -> FullGameNet:
        if key in LIVE:
            if time.time() - self.checked > 5.0:
                self.reload()
            return self.current if key == "current" else self.exploiter
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

    def __init__(self, nets: Nets, device, max_batch: int, compile_: bool = True, buckets: tuple[int, ...] | None = None,
                 thread: bool = True, entities: tuple[int, ...] = (fx.MAX_ENT,)):
        """`buckets`: the batch sizes its compiled calls pad to (the smallest that fits; default: just
        max_batch); `entities`: likewise the entity counts (a call padded 4 rows of ~47 entities to
        16 x 160: ~50 times the attention). `thread` False: no batching thread of its own (the
        inference server calls _forward)."""
        self.nets, self.device, self.max_batch = nets, device, max_batch
        self.buckets = tuple(sorted(buckets or (max_batch,)))
        self.entities = tuple(sorted(set(entities) | {fx.MAX_ENT}))
        self._hosts: dict[tuple, torch.Tensor] = {}
        self._used: dict[tuple, int] = {}  # the host buffers taken since the last wait()
        # the calls as CUDA graphs (one launch instead of ~150 kernels: the network is small, its calls
        # latency-bound). Graphs need fixed shapes: the batch and the entities padded to a bucket.
        # Captured from the eager network, per network and shape, in well under a second. (torch.compile's
        # "reduce-overhead" graphs took 10-20 s a shape, the games waiting, and its guards 0.3 ms a call.)
        self.compile = compile_ and device.type == "cuda"
        self._fns = weakref.WeakKeyDictionary()  # net -> its packed act
        self._graphs = weakref.WeakKeyDictionary()  # net -> {(batch, width): (input, graph, output)}
        self.q: queue.Queue = queue.Queue()
        if thread:
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
                version = self.nets.live_version(key)
                for (key_, st, out, k, done, n), res in zip(items, results):
                    if res is not None:
                        res["version"] = version
                    out[k] = res
                    pending[id(out)] = pending.get(id(out), 0) + 1
                    if pending[id(out)] == n:
                        del pending[id(out)]
                        done.set()

    def _forward(self, net: FullGameNet, sts: list[dict]) -> list[dict]:
        handle = self._launch(net, sts)
        self.wait()
        return self._finish(handle)

    def _fn(self, net: FullGameNet):
        """The network's act() from one packed input to one packed output."""
        if net not in self._fns:
            G, NO, d = net.config["G"], net.config["n_orders"], net.config["d"] if net.memory else 0
            ref = weakref.ref(net)  # (not the network itself: the entry would keep it alive)

            def packed_act(x):
                # one tensor in and one out: each copy to or from the GPU is a driver call of its own
                # (seven inputs were seven, seven outputs seven waits)
                B = x.shape[0]
                E = (x.shape[1] - G - 1 - NO - d) // (fx.F + 3)
                a = E * fx.F
                ent, glob = x[:, :a].reshape(B, E, fx.F), x[:, a:a + G]
                a += G
                typ, cur, mask = x[:, a:a + E].long(), x[:, a + E:a + 2 * E].long(), x[:, a + 2 * E:a + 3 * E] > 0.5
                a += 3 * E
                n_own, avail = x[:, a].long(), x[:, a + 1:a + 1 + NO] > 0.5
                out = act(ref(), ent, typ, cur, mask, glob, n_own, avail, *([x[:, a + 1 + NO:]] if d else []))
                return torch.cat([out["order"].float(), out["tgt"].float(), out["bx"].float(), out["by"].float(),
                                  out["logp"].float(), out["value"].float()[:, None], out["entropy"].float()[:, None]]
                                 + ([out["h"].float()] if d else []), 1)
            self._fns[net] = packed_act
        return self._fns[net]

    def _graph(self, net: FullGameNet, host: torch.Tensor) -> tuple:
        """The call's CUDA graph at the host batch's shape: (its input, the graph, its output). The
        graph reads the network's weights where they are: new ones copied into them are used."""
        per = self._graphs.setdefault(net, {})
        if host.shape not in per:
            fn = self._fn(net)
            x = host.to(self.device)
            for _ in range(2):  # (outside the capture: what the first calls set up)
                fn(x)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn(x)
            per[host.shape] = (x, graph, out)
        return per[host.shape]

    @torch.no_grad()
    def _launch(self, net: FullGameNet, sts: list[dict]) -> tuple:
        """Start a call without waiting for it (-> what _finish needs, after wait()). The server
        starts a round's calls (the current network's and each past snapshot's) one after the other
        and waits once: next to the learner every wait cost a turn of the GPU (a call 2.5 ms, 0.8 alone)."""
        bucket = next((b for b in self.buckets if len(sts) <= b), None)
        fixed = self.compile and bucket is not None
        widest = max(st["n"] for st in sts)
        B, E = (bucket, next(e for e in self.entities if widest <= e)) if fixed else (len(sts), widest)
        G, NO, d = len(sts[0]["glob"]), net.config["n_orders"], net.config["d"] if net.memory else 0
        W = E * (fx.F + 3) + G + 1 + NO + d
        pinned = fixed and self.device.type == "cuda"
        if pinned:  # the same shapes every call: pinned host buffers, reused (a round's calls each their own)
            slot = self._used[(B, W)] = self._used.get((B, W), -1) + 1
            if (B, W, slot) not in self._hosts:
                self._hosts[(B, W, slot)] = torch.zeros(B, W).pin_memory()
            host = self._hosts[(B, W, slot)]
            host.zero_()
        else:
            host = torch.zeros(B, W)
        hb = host.numpy()
        a = E * fx.F
        ent = hb[:, :a]
        ent.shape = (B, E, fx.F)  # (a view)
        glob = hb[:, a:a + G]
        a += G
        typ, cur, mask = (hb[:, a + k * E:a + (k + 1) * E] for k in range(3))
        a += 3 * E
        n_own, avail, hs = hb[:, a], hb[:, a + 1:a + 1 + NO], hb[:, a + 1 + NO:]
        avail[:] = 1.0
        mask[:, 0] = 1.0  # (padding rows: one entity, or attention over nothing gives NaNs)
        for i, st in enumerate(sts):
            n = st["n"]
            ent[i, :n], typ[i, :n], cur[i, :n], mask[i, :n] = st["ent"], st["type"], st["cur"], 1.0
            glob[i], n_own[i] = st["glob"], min(st["n_own"], fx.MAX_OWN)
            if st.get("avail") is not None:
                avail[i] = st["avail"]
            if d and st.get("h") is not None:  # (none: the game's start, zeros)
                hs[i] = st["h"]
        own = n_own[:len(sts)].astype(np.int64)
        if pinned:  # queued one behind the other: the input in, the graph, its output out (before it runs again)
            x, graph, packed = self._graph(net, host)
            x.copy_(host, non_blocking=True)
            graph.replay()
            key = ("out", B, packed.shape[1], slot)
            if key not in self._hosts:
                self._hosts[key] = torch.zeros(B, packed.shape[1]).pin_memory()
            out = self._hosts[key]
            out.copy_(packed, non_blocking=True)
        else:
            out = self._fn(net)(host.to(self.device))
        return out, own, min(fx.MAX_OWN, E), bool(d)

    def wait(self) -> None:
        """Until the calls launched are done (sleeping: CUDA's default sync spins a core the games need)."""
        if self.device.type == "cuda":
            done = torch.cuda.Event(blocking=True)
            done.record()
            done.synchronize()
        self._used.clear()

    def _finish(self, handle: tuple) -> list[dict]:
        out, own, O, memory = handle
        p = out.cpu().numpy()
        order, tgt, bx, by = (p[:, k * O:(k + 1) * O].astype(np.int64) for k in range(4))
        logp, value, entropy = p[:, 4 * O:5 * O], p[:, 5 * O], p[:, 5 * O + 1]
        res = []
        for i, o in enumerate(own):
            res.append({"order": order[i, :o], "tgt": tgt[i, :o], "bx": bx[i, :o], "by": by[i, :o],
                        "logp": logp[i, :o].copy(), "value": float(value[i]), "entropy": float(entropy[i])})
            if memory:
                res[-1]["h"] = p[i, 5 * O + 2:].copy()
        return res


ST_FIELDS = ("ent", "type", "cur", "glob", "n", "n_own", "avail", "h")  # what a network call needs of a view step


class RemoteInference:
    """An actor's side of the inference server (inference_main): the same request() as Inference, the
    calls made in one process for all the actors' games. Every game thread has a pipe of its own to
    the server and waits on it itself. (Through queues a request took ~32 ms of a 59 ms step, the
    server's call 4.5 of them: a feeder thread and a shared lock on the way in, a feeder, a reader
    thread and an event on the way back, each a wait for some process's GIL.)"""

    def __init__(self, conns: list):
        self.free = list(conns)
        self.lock = threading.Lock()
        self.local = threading.local()
        self.nets = types.SimpleNamespace(version=-1)  # (the version the server answered with last)

    def request(self, items: list[tuple[str, dict]]) -> list[dict]:
        loc = self.local
        if not hasattr(loc, "conn"):
            with self.lock:
                loc.conn, loc.rid = self.free.pop(), 0
        loc.rid += 1
        loc.conn.send((loc.rid, [(key, {f: st.get(f) for f in ST_FIELDS}) for key, st in items]))
        while True:
            if not loc.conn.poll(300):
                raise RuntimeError("no answer from the inference server for 5 minutes")
            rid, out = loc.conn.recv()
            if rid == loc.rid:  # (not the answer to a request this thread gave up on)
                break
        for res in out:
            if res is not None and res.get("version", -1) >= 0:
                self.nets.version = res["version"]
        return out


class PastNet:
    """The past snapshots' calls through one network object (per architecture): a snapshot's weights
    are copied into it before its calls. Its parameters are views of one flat tensor, so that is one
    copy. (Compiled per snapshot, every new league member stalled all the games for 25-50 s: four
    shapes to compile, every 20 updates.)"""

    def __init__(self, device):
        self.device = device
        self.hosts: dict[str, list] = {}  # architecture -> [network, its flat weights, its buffers, the snapshot loaded]
        self.flats = weakref.WeakKeyDictionary()  # snapshot network -> (its weights, flat; its buffers)

    @torch.no_grad()
    def get(self, key: str, src: FullGameNet) -> FullGameNet:
        if src not in self.flats:
            self.flats[src] = (torch.cat([p.data.reshape(-1) for p in src.parameters()]), list(src.buffers()),
                               json.dumps(src.config, sort_keys=True, default=str))
        flat, buffers, arch = self.flats[src]
        host = self.hosts.get(arch)
        if host is None:
            net = FullGameNet(**src.config).to(self.device).eval()
            net.load_state_dict(src.state_dict())
            mine = torch.cat([p.data.reshape(-1) for p in net.parameters()])
            at = 0
            for p in net.parameters():
                p.data = mine[at:at + p.numel()].view(p.shape)
                at += p.numel()
            host = self.hosts[arch] = [net, mine, list(net.buffers()), key]
        if host[3] != key:
            host[1].copy_(flat)
            for mine, theirs in zip(host[2], buffers):
                mine.copy_(theirs)
            host[3] = key
        return host[0]


def inference_main(cfg: dict, conns: list, stop) -> None:
    """The inference server: every actor's network calls in one process. The games spent 53% of
    their time waiting for their actor's calls: four actors and the learner took turns on the GPU
    (a context each), and an actor's batches were mostly padding (its 8 games rarely ask at once).
    Here the games' requests batch together, in one context: whatever has arrived while the last
    call ran goes into the next one (no waiting for more)."""
    import signal
    from multiprocessing.connection import wait
    signal.signal(signal.SIGTERM, _exit)
    _pin(cfg, "server")
    torch.set_num_threads(2)
    device = torch.device(cfg["device"])
    nets = Nets(cfg, device)
    while nets.current is None and not stop.is_set() and os.getppid() == cfg["learner_pid"]:
        time.sleep(1)
        nets.reload()
    top = cfg["infer_batch"]
    infer = Inference(nets, device, top, cfg["compile"], buckets=tuple(b for b in (8, 24, 64) if b <= top) or (top,),
                      thread=False, entities=(48, 96))
    past = Inference(nets, device, 16, cfg["compile"], buckets=(4, 16), thread=False, entities=(64,))  # past snapshots: few calls
    swap = PastNet(device)
    def fresh():
        return {"calls": 0, "past": 0, "rows": 0, "busy": 0.0, "wait": 0.0, "ents": 0, "ents_max": 0, "rounds": 0,
                "t": time.time()}
    stats = fresh()  # -> inference.jsonl
    stats_path = Path(cfg["run_dir"]) / "inference.jsonl"
    live = list(conns)
    t_round = 0.0
    while not stop.is_set() and os.getppid() == cfg["learner_pid"]:
        if time.time() - stats["t"] >= 10.0 and stats["calls"]:
            dt = time.time() - stats["t"]
            with open(stats_path, "a") as f:
                f.write(json.dumps({"time": time.time(), "calls_per_s": round(stats["calls"] / dt, 1),
                                    "rows_per_s": round(stats["rows"] / dt, 1), "rows_per_call": round(stats["rows"] / stats["calls"], 2),
                                    "rounds_per_s": round(stats["rounds"] / dt, 1),
                                    "past_calls_per_s": round(stats["past"] / dt, 1),
                                    "round_ms": round(1000 * stats["busy"] / max(stats["rounds"], 1), 2),  # requests in to answers out
                                    "wait_ms": round(1000 * stats["wait"] / max(stats["rounds"], 1), 2),  # of it: for the GPU
                                    "busy": round(stats["busy"] / dt, 3), "entities": round(stats["ents"] / stats["rows"], 1),
                                    "entities_max_mean": round(stats["ents_max"] / stats["calls"], 1)}) + "\n")
            stats = fresh()
        ready = wait(live, timeout=1.0)
        if not ready:
            continue
        # a round at most every cfg["infer_period"] seconds: every round takes the GPU from the learner
        # for a turn (260 rounds a second left it half of it), and what arrives meanwhile shares the round
        pause = t_round + cfg.get("infer_period", 0.0) - time.time()
        if pause > 0:
            time.sleep(pause)
            ready = wait(live, timeout=0)
        t_busy = t_round = time.time()
        asked: dict = {}  # game pipe -> (its request's id, the answers)
        groups: dict[str, list] = {}  # network -> [(pipe, item, view step)]
        for c in ready:
            try:
                rid, items = c.recv()
            except (EOFError, OSError):  # its actor is gone
                live.remove(c)
                continue
            asked[c] = (rid, [None] * len(items))
            for k, (key, st) in enumerate(items):
                groups.setdefault(key, []).append((c, k, st))
        calls = []  # the round's calls, all started before the one wait for them
        for key, items in groups.items():
            fwd, size = (infer, top) if key in LIVE else (past, 16)
            for a in range(0, len(items), size):
                part = items[a:a + size]
                try:
                    net = nets.get(key)
                    handle = fwd._launch(net if key in LIVE else swap.get(key, net), [it[2] for it in part])
                except Exception:  # noqa: BLE001 (the games must not hang on it)
                    traceback.print_exc()
                    handle = None
                calls.append((fwd, handle, part, nets.live_version(key)))
                ns = [it[2]["n"] for it in part]
                stats["calls"] += 1
                stats["past"] += key not in LIVE
                stats["rows"] += len(part)
                stats["ents"] += sum(ns)
                stats["ents_max"] += max(ns)
        t_wait = time.time()
        infer.wait()
        past.wait()
        stats["wait"] += time.time() - t_wait
        for fwd, handle, part, version in calls:
            try:
                results = fwd._finish(handle) if handle is not None else [None] * len(part)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                results = [None] * len(part)
            for (c, k, _), res in zip(part, results):
                if res is not None:
                    res["version"] = version
                asked[c][1][k] = res
        for c, answer in asked.items():
            try:
                c.send(answer)
            except OSError:
                if c in live:
                    live.remove(c)
        stats["rounds"] += 1
        stats["busy"] += time.time() - t_busy


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


def label_step(obs, t: int, advised: list[int], spans: dict, sts: dict, held: dict, bots: dict, trajs: dict) -> None:
    """The built-in AI advising (protocol.ShadowAI): its orders given during step t (in the next
    observation, `obs`) label the step each advised side just recorded (its last trajectory step),
    as the demonstrations' encoder labels a recorded game (features.Encoder.step_labels: in the state
    the side acted in, what it could pay for, train orders that started something)."""
    rows = np.asarray([(t, o.unit, o.order, o.kind, o.x, o.y, o.target) for o in obs.issued], np.int64).reshape(-1, 7)
    later = np.asarray([(t + 1, int(e.kind), e.a, e.b, e.c) for e in obs.events], np.int64).reshape(-1, 5)
    for s in advised:
        if s not in spans or not trajs[s].steps or s not in held:
            continue
        bot, step = bots[s], trajs[s].steps[-1]
        y, _ = bot.enc.step_labels(bot.view, sts[s], rows, held[s], later, bot.trees)
        n = step["n_own"]
        for k, v in y.items():
            step[k] = v[:n]


def shaped(obs, mat: dict, penalty: float) -> dict:
    """The material the shaping's potential counts: `mat` (units and buildings, fullgame.trace.material)
    less `penalty` times the gold and lumber each player holds. With a penalty, spending is credited
    when the resources leave the bank (an order), not only when what they buy appears: on duelfast the
    learner held ~3,700 a game (the AI ~800) and its idle barracks trained 5% of the steps."""
    if not penalty:
        return mat
    out = dict(mat)
    for p in out:
        pl = obs.players.get(p)
        if pl is not None:
            out[p] -= penalty * (pl.gold + pl.lumber)
    return out


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


def game_rng(cfg: dict, wid: int, k: int) -> random.Random:
    """A game thread's draws (races, sides, kinds of launches, opponents). The update the run starts
    from is part of the seed: seeded by the thread alone, every restart of a run began with the same
    launches (on a day of 15 restarts, 43 launches of the real game for human against the normal AI
    and 6 for orc)."""
    return random.Random(f"{cfg['seed']}/{cfg.get('start_update', 0)}/{wid}/{k}")


def matchup(mirror: bool, race: str, opp: str) -> str:
    """The key of the curriculum's level for a learner playing `race` against `opp` (League)."""
    return race if mirror else f"{race}/{opp}"


def balance_tax(spec: dict, races: list[str]) -> tuple[int, float]:
    """A game between agents playing `races` (players 0, 1): (the side whose income is taxed, the
    share taken; 0: none), from the league's balance between the two races (League.balance)."""
    a, b = races
    t = spec.get("balance", {}).get(f"{a}/{b}")
    if t is None:
        t = -spec.get("balance", {}).get(f"{b}/{a}", 0.0)
    return (0 if t > 0 else 1), abs(t)


def game_loop(wid: int, k: int, cfg: dict, infer: Inference, out_q, stop, render_q) -> None:
    from ..runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup
    from .play import BCAgent

    rng = game_rng(cfg, wid, k)
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
        adv = cfg.get("opd_difficulty", "normal")  # (an agent slot's AI: the advisor's, protocol.ShadowAI)
        slots = [Agent(races[0], handicap=cfg["handicap"], difficulty=adv), Agent(races[1], handicap=cfg["handicap"], difficulty=adv)]
        real = bool(ai.get("real"))  # the real game (the yardstick, no curriculum)
        curr = ai["kind"] == "ai" and "by_race" in ai  # a curriculum game (League.knobs, by the learner's race)
        key = matchup(cfg["mirror"], races[side], races[1 - side])
        if curr:
            ai = {**ai, **ai["by_race"].get(key, {})}
        late = curr and ai.get("delay", 0.0) > 0  # the AI starts late: an agent slot till then
        if ai["kind"] == "ai":
            slots[1 - side] = (Agent(races[1 - side], handicap=cfg["handicap"], difficulty=ai["difficulty"]) if late
                               else BuiltinAI(races[1 - side], ai["difficulty"], handicap=cfg["handicap"]))
            if curr:  # (the handicap is the launch's: the level when it started)
                slots[side] = Agent(races[side], handicap=int(ai.get("handicap", cfg["handicap"])), difficulty=adv)
        agents_only = ai["kind"] != "ai"
        # a restart goes on to the next pair of players (--pairs), or reloads the map (1 s).
        # (--scripted-reset: games without the built-in AI reset by script for the same players instead,
        # 0.1 s, but not to a new game: see GameSetup.melee_reset)
        setup = GameSetup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"],
                          max_game_seconds=cfg["max_minutes"] * 60, victory="decisive", window=(320, 240),
                          record_ai_orders=cfg.get("opd_share", 0.0) > 0,
                          wait_floor_ms=cfg["wait_floor_ms"], melee_reset=agents_only and cfg["scripted_reset"],
                          native_obs=cfg["native_obs"], nice=cfg.get("game_nice", 0), d3d_thread=False,
                          render_threads=0, pairs=cfg.get("pairs", 1))
        per_launch = cfg["games_per_process"] * (cfg["agent_games_factor"] if agents_only else 1)
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
                           if ai["kind"] == "ai" else {"name": "exploiter", "kind": "exploiter"} if ai["kind"] == "exploit"
                           else dict(_choose(rng, spec["agents"])))
                    if agents_only:  # races apart: the stronger one's income taxed (League.balance)
                        taxed, share = balance_tax(spec, races)
                        if share > 0:
                            opp["tax"], opp["tax_side"] = share, taxed
                    # on-policy distillation: the built-in AI advises the learner (never in the real game)
                    opp["advise"] = not real and rng.random() < cfg.get("opd_share", 0.0)
                    if curr:  # the curriculum's current knobs for this AI
                        now = next((x.get("by_race", {}).get(key, ai) for x in spec["launch"]
                                    if x.get("difficulty") == ai["difficulty"] and not x.get("real")), ai)
                        opp["curriculum"] = now.get("level", 0.0)
                        if late:
                            opp["start"] = int(round(now.get("delay", 0.0) / cfg["step_seconds"]))
                        if now.get("tax", 0.0) > 0:
                            opp["tax"] = now["tax"]
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

    from ..protocol import SetResources, ShadowAI, StartAI
    race_ix = [fx.RACES.index(r) if r in fx.RACES else 0 for r in races]
    bots: dict[int, BCAgent] = {}
    keys: dict[int, str] = {}
    trajs: dict[int, Trajectory] = {}
    late = 1 - side if opp.get("start") is not None else None  # the built-in AI's side, idle until opp["start"]
    # the side losing opp["tax"] of what it gathers: the AI's (the curriculum), or an agent's (the balance)
    taxed = opp.get("tax_side", 1 - side) if opp.get("tax") else None
    gathered = None
    prod = Production()  # what both sides make (the episode row: the dashboard's Production tab)
    prod.step(obs)
    for s in (0, 1):
        if g.setup.slots[s].kind != "agent" or s == late:
            continue
        bot = BCAgent(None, vocab, s, None, costs=cfg.get("costs_array"))
        bot.begin(obs, race_ix)
        bots[s] = bot
        keys[s] = ("current" if s == side or opp["kind"] == "self" else "exploiter" if opp["kind"] == "exploiter"
                   else opp["path"])
        if keys[s] == "current":
            trajs[s] = Trajectory(cfg, out_q, {"worker": wid, "opponent": opp["name"]})
        elif keys[s] == "exploiter":  # (its steps train the exploiter: League, --exploiter-share)
            trajs[s] = Trajectory(cfg, out_q, {"worker": wid, "opponent": "main", "learner": "exploiter"})
    # the built-in AI advises the learner's sides (opp["advise"]): its orders label their steps
    advised = [s for s in trajs if keys[s] == "current"] if opp.get("advise") else []
    for s in advised:
        bots[s].repair_builds = True  # (a builder the advisor pulled away resumes with "repair")
    t, t0 = 0, time.time()
    values, scale = cfg["values"], cfg["shaping_scale"]
    mat = material(obs, values)
    pot = shaped(obs, mat, cfg.get("float_penalty", 0.0))
    phi = {s: (pot[s] - pot[1 - s]) / scale for s in trajs}  # at the last recorded state
    ret = {s: 0.0 for s in trajs}
    trace: list[dict] = []
    names = {int(k): v for k, v in (cfg.get("order_names") or {}).items()}
    while True:
        # the tax first: the step's commands run in order, and set after an agent's orders the
        # resources undid what they spent (a taxed agent trained for free); the taxed agent sees
        # what it has left
        cmds = []
        if taxed is not None and obs.players.get(taxed) is not None:
            p = obs.players[taxed]
            now = (p.gold_gathered, p.lumber_gathered)
            dg, dl = (p.gold, p.lumber) if gathered is None else (now[0] - gathered[0], now[1] - gathered[1])
            if dg > 0 or dl > 0:  # (at the start: its starting gold and lumber)
                p.gold, p.lumber = max(0, p.gold - int(opp["tax"] * dg)), max(0, p.lumber - int(opp["tax"] * dl))
                cmds.append(SetResources(taxed, p.gold, p.lumber))
            gathered = now
        if t == 0:
            cmds += [ShadowAI(s) for s in advised]
        held = {s: obs.players.get(s) for s in advised}  # (the labels count what the player could pay for)
        held = {s: np.asarray([t, s, p.gold, p.lumber, p.food_used, p.food_cap, p.upkeep]) for s, p in held.items() if p}
        rows = unit_rows(obs, t)  # once for both sides
        sts = {s: bot.observe(obs, t, rows) for s, bot in bots.items()}
        live = [s for s in bots if sts[s] is not None]
        for s in live:
            sts[s]["h"] = bots[s].h
        results = infer.request([(keys[s], sts[s]) for s in live]) if live else []
        spans = {}
        for s, res in zip(live, results):
            if res is None:
                continue
            c = bots[s].commands(sts[s], res["order"], res["tgt"], res["bx"], res["by"])
            bots[s].h = res.get("h")
            spans[s] = (len(cmds), len(cmds) + len(c))
            cmds += c
            if s in trajs:
                trajs[s].add(sts[s], res)
                phi[s] = (pot[s] - pot[1 - s]) / scale
        if record:
            chose = {s: Counter(fx.order_label(*bots[s].orders[int(c)], names) for c in r["order"] if c)
                     for s, r in zip(live, results) if r is not None}
            trace.append(trace_step(t, obs, mat, chose, value={s: r["value"] for s, r in zip(live, results) if r},
                                    entropy={s: r["entropy"] for s, r in zip(live, results) if r}))
        if late is not None and t == opp["start"]:  # the built-in AI takes over its side (after the bots' orders)
            cmds.append(StartAI(late))
        obs = g.step(cmds)
        if advised:  # the advisor's orders of the step just taken: the labels of the step recorded
            label_step(obs, t, advised, spans, sts, held, bots, trajs)
        for s, (a, b) in spans.items():
            bots[s].accepted(cmds[a:b], obs.command_results[a:b])
        t += 1
        prod.step(obs)
        mat = material(obs, values)
        pot = shaped(obs, mat, cfg.get("float_penalty", 0.0))
        if obs.game_over or t >= cfg["max_steps"]:
            break
        for s, tr in trajs.items():  # shaping: gamma * phi(s') - phi(s), for the step just taken
            if s in spans and tr.steps:  # (it recorded a step this time)
                r = cfg["shaping"] * (cfg["gamma"] * (pot[s] - pot[1 - s]) / scale - phi[s])
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
        terminal = (outcome[s] if outcome[s] != 0.0
                    else cfg.get("tie_value", 0.0) + cfg["tie_break"] * math.tanh(2.0 * lead[s]))
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
        "gold": prod.row(side, obs)["gold"], "opponent_gold": prod.row(1 - side, obs)["gold"],  # (this game's)
        "orders": bots[side].issued if side in bots else 0,
        **({"curriculum": opp["curriculum"], "handicap": g.setup.slots[side].handicap,
            "ai_delay": opp.get("start", 0) * cfg["step_seconds"], "ai_tax": opp.get("tax", 0.0)}
           if "curriculum" in opp else {}),
        # games between agents: the share of the learner's income taken (< 0: of the opponent's)
        **({"balance_tax": opp["tax"] if opp["tax_side"] == side else -opp["tax"]} if "tax_side" in opp else {}),
        "prod": prod.row(side, obs), "opp_prod": prod.row(1 - side, obs), **({"advised": True} if advised else {})}
    out_q.put({"episode": ep})
    if record:  # the video's panel: A = the learner's side
        who = {"self": "itself", "ai": "built-in AI"}.get(opp["kind"], opp["name"])
        other = (f"built-in AI {opp['name'].split('-', 1)[1]}" if opp["kind"] == "ai"
                 else "itself" if opp["kind"] == "self" else "its exploiter" if opp["kind"] == "exploiter"
                 else f"snapshot {opp['name']}")
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
    calib = {}
    if trace is not None:
        replay.with_suffix(".trace.json").write_text(json.dumps(trace))
        # value calibration (the dashboard's Learning tab): the learner's first value and the
        # discounted return that followed it
        steps, me = trace.get("steps") or [], str(ep["side"])
        v0 = (steps[0].get("value") or {}).get(me) if steps else None
        if v0 is not None:
            gamma = trace.get("gamma", cfg["gamma"])  # (not g: that's the game)
            ret = 0.0
            for st in reversed(steps):
                ret = (st.get("reward") or {}).get(me, 0.0) + gamma * ret
            calib = {"value0": v0, "return0": round(ret, 4)}
    row = {"episode": ep.get("episode_id", ""), "title": f"{ep['race']} vs {ep['opponent_race']} ({ep['opponent']})",
           "outcome": ep["outcome"], "return": ep["return"], **calib,
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
    _pin(cfg, "games")
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


def _pin(cfg: dict, who: str) -> None:
    """The calling process on its CPUs (--pin): the learner and the inference server a core each of
    their own, the actors and their games (which inherit it) on the rest."""
    cpus = (cfg.get("cpus") or {}).get(who)
    if cpus:
        os.sched_setaffinity(0, cpus)


def actor_main(wid: int, cfg: dict, out_q, stop, render_q, conns: list | None = None) -> None:
    import signal

    from ..runtime import reaper
    signal.signal(signal.SIGTERM, _exit)
    _pin(cfg, "games")
    out_q.cancel_join_thread()  # exiting must not wait to flush trajectories nobody reads any more
    torch.set_num_threads(1)
    try:
        if conns is not None:  # the inference server makes the network calls (no GPU context here)
            sys.setswitchinterval(0.001)  # (a game thread woken by its answer waits for the GIL: 5 ms by default)
            infer = RemoteInference(conns)
        else:
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
                 curriculum: tuple[float, float, float, int] | None = None, mode: str = "hp",
                 races: tuple[str, ...] = ("",), mirror: bool = True, balance: tuple[float, float] | None = None,
                 exploiter_share: float = 0.0, tie: float = 0.0):
        """`curriculum`: (start level, step, the knob at level 1, base handicap): games against the
        built-in AI get easier or harder towards a 50% score, a level per difficulty in [0, 1] that
        a loss raises by `step`, a win lowers (a tie leaves it). At 0 the game is the real one; None:
        no curriculum. The level's knobs (knobs()), by `mode`:
        * "hp": from 0 to 0.5 the learner's units get more hit points (its handicap from the base to
          100, the AI's staying at the base); from 0.5 to 1 also the AI starts late (its units idle
          till then; protocol.StartAI), up to the knob in seconds. Twice the hit points let a small
          army win fights: the learner stopped needing to spend, which it needs in the real game.
        * "delay": the late start alone. The learner learned to rush the idle AI: 65% of its wins
          came before the AI started.
        * "tax": the AI plays from the start, but loses this share of what it gathers (and of its
          starting gold and lumber), up to the knob (e.g. 0.9): a poorer opponent, nothing idle.
        A level per difficulty and learner's race (`races`): one per difficulty settled where night
        elf games (96% won) balanced human and orc ones (7-14%), and neither end taught anything.
        Without `mirror` matchups, a level per difficulty and matchup ("human/nightelf": the learner
        human, the AI night elf).

        `balance` (step, the tax at level 1), for games between agents of different races (the races
        are not balanced on duelrush: night elf won 78% of the built-in AI's games, undead 29%): a
        level in [-1, 1] per pair of races ("human/orc": > 0 the first one's income is taxed, < 0
        the second's), which a win of the first raises by `step` and a loss lowers. Only the
        learner's games against itself move it (the same player on both sides: the races alone
        differ); games against past snapshots are taxed the same.

        `tie`: how far a tie moves a curriculum level, as a share of a loss's step (0: not at all; on
        duelfast 78% of the curriculum games tied at the time limit, so the levels stood still while
        the learner won 1%).

        `exploiter_share` (AlphaStar's main exploiter): that share of launches are games between the
        learner and its exploiter, a second network that trains only against it (both sides train,
        each its own network). "exploiter" (a member) holds the learner's results against the
        current exploiter; the exploiter's snapshots that beat it join the league ("exploiter:N")."""
        self.run_dir, self.shares, self.max_past, self.pfsp = run_dir, shares, max_past, pfsp
        self.scripts = {f"script:ai-{d}": Member(f"script:ai-{d}") for d in ai}
        self.rule, self.mode, self.races, self.tie = curriculum, mode, tuple(races), tie
        self.mirror = mirror
        self.keys = self.races if mirror else tuple(matchup(False, a, b) for a in self.races for b in self.races)
        self.level = {(n, k): curriculum[0] for n in self.scripts for k in self.keys} if curriculum else {}
        self.balance_rule = balance
        self.balance = ({f"{a}/{b}": 0.0 for i, a in enumerate(self.races) for b in self.races[i + 1:]}
                        if balance and not mirror else {})
        # with a curriculum some launches play the real game (the yardstick: "script:ai-X (real)")
        self.real = {f"{n} (real)": Member(f"{n} (real)") for n in self.scripts} if curriculum else {}
        self.real_share = 0.0  # (set by the learner: --real-share)
        self.past: list[Member] = []
        self.self_member = Member("self")
        self.exploiter_share = exploiter_share
        self.exploiter = Member("exploiter") if exploiter_share > 0 else None
        self.exploiter_resets = 0

    def matchup(self, race: str, opp: str) -> str:
        return matchup(self.mirror, race, opp)

    def knobs(self, name: str, race: str = "") -> dict:
        """At the level of AI `name` and the learner's `race` (or matchup): the learner's handicap
        (hit points in percent), the AI's late start (seconds) and the share of the AI's income taken."""
        _, _, top, base = self.rule
        lv = self.level[(name, race)]
        if self.mode == "tax":
            return {"handicap": base, "delay": 0.0, "tax": round(lv * top, 3)}
        if self.mode == "delay":
            return {"handicap": base, "delay": round(lv * top, 1), "tax": 0.0}
        hp = base + int(min(1.0, 2 * lv) * (100 - base) / 10.0 + 0.5) * 10  # (handicaps in steps of 10)
        return {"handicap": hp, "delay": round(max(0.0, 2 * lv - 1) * top, 1), "tax": 0.0}

    def curriculum(self, name: str, outcome: float, race: str = "") -> None:
        """A game against the built-in AI `name` with the learner playing `race` ended (+1 / 0 / -1
        for the learner): that level moves."""
        key = (name, race)
        if key in self.level:
            move = -outcome if outcome != 0 else self.tie  # (a win lowers it, a loss raises it)
            self.level[key] = min(1.0, max(0.0, self.level[key] + self.rule[1] * move))

    def balanced(self, race: str, opp: str, outcome: float) -> None:
        """A game of the learner against itself, `race` against `opp`, ended (`outcome` for `race`)."""
        pair, sign = f"{race}/{opp}", 1.0
        if pair not in self.balance:
            pair, sign = f"{opp}/{race}", -1.0
        if pair in self.balance:
            self.balance[pair] = min(1.0, max(-1.0, self.balance[pair] + self.balance_rule[0] * sign * outcome))

    def balance_taxes(self) -> dict:
        """Per pair of races, the share of the first one's income taken (< 0: of the second's)."""
        return {k: round(v * self.balance_rule[1], 3) for k, v in self.balance.items()}

    def member(self, name: str) -> Member | None:
        if name == "self":
            return self.self_member
        if name == "exploiter":
            return self.exploiter
        return self.scripts.get(name) or self.real.get(name) or next((m for m in self.past if m.name == name), None)

    def add_snapshot(self, path: Path, steps: int, name: str | None = None) -> None:
        self.past.append(Member(name or f"past:{steps}", path=str(path), steps=steps))
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
        for n, v in (summary.get("level") or {}).items():  # "script:ai-X|race" (runs from before races: every race)
            name, _, race = n.partition("|")
            if not race:
                keys = [(name, k) for k in self.keys]
            elif not self.mirror and "/" not in race:  # (a run of mirror matchups goes on with mixed ones)
                keys = [(name, self.matchup(race, b)) for b in self.races]
            else:
                keys = [(name, race)]
            for key in keys:
                if key in self.level:
                    self.level[key] = v
        self.balance.update({k: v for k, v in (summary.get("balance") or {}).items() if k in self.balance})
        if self.exploiter is not None and summary.get("exploiter"):
            x = summary["exploiter"]
            self.exploiter.wins, self.exploiter.losses, self.exploiter.draws = x.get("wins", 0.0), x.get("losses", 0.0), x.get("draws", 0.0)
            self.exploiter.recent = list(x.get("recent") or [])
            self.exploiter_resets = summary.get("exploiter_resets", 0)
        me = summary.get("self") or {}
        self.self_member.wins, self.self_member.losses = me.get("wins", 0.0), me.get("losses", 0.0)
        self.self_member.draws, self.self_member.recent = me.get("draws", 0.0), list(me.get("recent") or [])

    def spec(self, real_share: float = 0.0) -> dict:
        """What the actors draw from: per launch built-in AI or agents; per agent game an opponent.
        With a curriculum, `real_share` of the AI launches play the real game ("real": its own launch
        kind; an extra draw after the races' made night elf rare: 1 of 38 real launches)."""
        ai_share = self.shares["ai"] if self.scripts else 0.0
        launch = [{"kind": "agents", "p": 1.0 - ai_share}]
        if self.exploiter is not None:  # (out of every other kind's share alike)
            launch = [{"kind": "agents", "p": (1.0 - ai_share) * (1 - self.exploiter_share)},
                      {"kind": "exploit", "p": self.exploiter_share}]
            ai_share *= 1 - self.exploiter_share
        if self.level and real_share > 0:
            launch += [{"kind": "ai", "difficulty": n.split("-", 1)[1], "real": True,
                        "p": ai_share * real_share / len(self.scripts)} for n in self.scripts]
            ai_share *= 1.0 - real_share
        launch += [{"kind": "ai", "difficulty": n.split("-", 1)[1], "p": ai_share / len(self.scripts),
                    **({"by_race": {k: dict(self.knobs(n, k), level=self.level[(n, k)]) for k in self.keys}}
                       if self.level else {})}
                   for n in self.scripts]
        agents = [{"name": "self", "kind": "self", "p": self.shares["self"] if self.past else 1.0}]
        if self.past:
            w = [pfsp_weight(m.win_rate(), self.pfsp) for m in self.past]
            total = sum(w) or 1.0
            agents += [{"name": m.name, "kind": "past", "path": m.path, "p": self.shares["past"] * x / total}
                       for m, x in zip(self.past, w)]
        return {"launch": launch, "agents": agents, **({"balance": self.balance_taxes()} if self.balance else {})}

    def write(self) -> None:
        spec = self.run_dir / "league_spec.json"
        tmp = spec.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.spec(self.real_share)))
        tmp.replace(spec)

        def row(m: Member) -> dict:
            p = m.win_rate()
            return {"name": m.name, "steps": m.steps, "games": m.games, "wins": m.wins, "losses": m.losses,
                    "draws": m.draws, "win_rate": p, "weight": pfsp_weight(p, self.pfsp) if m.path else None,
                    "path": m.path, "recent": m.recent}
        summary = {"pfsp": self.pfsp, "members": [row(m) for m in [*self.scripts.values(), *self.real.values(), *self.past]],
                   "self": row(self.self_member), "level": {f"{n}|{r}": v for (n, r), v in self.level.items()},
                   "balance": self.balance,
                   **({"exploiter": row(self.exploiter), "exploiter_resets": self.exploiter_resets} if self.exploiter else {})}
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
        used = {"tax": ("tax",), "delay": ("delay",)}.get(self.mode, ("handicap", "delay"))  # (the others stay put)
        for (n, r), lv in self.level.items():  # the curriculum: its levels and knobs
            short = f"{n.split(':', 1)[1]} {r}".strip()
            out[f"curriculum/{short} level"] = lv
            out.update({f"curriculum/{short} {k}": v for k, v in self.knobs(n, r).items() if k in used})
        out.update({f"balance/{k} tax": v for k, v in self.balance_taxes().items()} if self.balance else {})
        if self.exploiter is not None:
            if self.exploiter.win_rate() is not None:
                out["league/exploiter_vs_main"] = 1.0 - self.exploiter.win_rate()  # (the exploiter's score)
            out["league/exploiter_resets"] = self.exploiter_resets
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
    labels = {}
    if any("y_order" in s for s in steps):  # the advisor's labels (on-policy distillation; others: none)
        labels = {"y_order": np.zeros((B, O), np.int64), **{k: np.full((B, O), -1, np.int64) for k in ("y_ptr", "y_x", "y_y")}}
        has = np.zeros(B, bool)
        for i, s in enumerate(steps):
            if "y_order" in s:
                has[i] = True
                for k in labels:
                    labels[k][i, :s["n_own"]] = s[k]
        labels = {**{k: t(v) for k, v in labels.items()}, "y_has": t(has), "y_any": bool(has.any())}
    return {"ent": t(ent), "type": t(typ), "cur": t(cur), "mask": t(mask), "glob": t(glob), "n_own": t(n_own), **labels,
            **{k: t(v) for k, v in acts.items()}, "logp": t(logp), "avail": t(avail) if avail is not None else None,
            "adv": torch.tensor([s["adv"] for s in steps], device=device, dtype=torch.float32),
            "ret": torch.tensor([s["ret"] for s in steps], device=device, dtype=torch.float32),
            "own": torch.arange(O, device=device)[None] < t(n_own)[:, None]}


def minibatches(sizes: np.ndarray, size: int, group: int, rng=np.random) -> list[np.ndarray]:
    """An epoch's minibatches (indices). A minibatch is padded to its widest step, and at random
    that is twice the mean (91 entities for a mean of 41: attention costs the square). So the steps,
    in random order, are cut into groups of `group` minibatches, each group sorted by entity count
    and cut into its minibatches (49 entities with 8), which then come in random order. `group` 1:
    at random."""
    order = rng.permutation(len(sizes))
    if group <= 1:
        return [order[a:a + size] for a in range(0, len(order), size)]
    out = []
    for a in range(0, len(order), size * group):
        part = order[a:a + size * group]
        part = part[np.argsort(sizes[part], kind="stable")]
        out += [part[b:b + size] for b in range(0, len(part), size)]
    return [out[i] for i in rng.permutation(len(out))]


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


def distill_loss(net: FullGameNet, ev: dict, mb: dict, autocast) -> tuple[torch.Tensor, torch.Tensor]:
    """On-policy distillation: the cloning loss (bc.losses: the order, then its target given the
    teacher's order) on the learner's own steps that the built-in AI advised (mb["y_has"]), from
    the PPO pass's network outputs; and the share of the teacher's orders (not "none") the policy
    likes most."""
    own = mb["own"] & mb["y_has"][:, None]
    y = mb["y_order"]
    lp = torch.log_softmax(ev["logits"].float(), -1)
    lp_y = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    ok = (own & (lp_y > -1e4)).float()  # (an order the unit can't get now: none of its business)
    l_order = -(lp_y * ok).sum() / ok.sum().clamp(min=1)
    ptr, xl, z = net.target_logits(ev["g"], ev["u"], mb["mask"], y)  # (in the caller's autocast)
    yl = net.y_logits(z, mb["y_x"])
    # (masked means, not boolean indexing: every index or any() was a wait for the GPU)
    l_ptr = _masked_ce(ptr, mb["y_ptr"], ok)
    l_pt = _masked_ce(xl, mb["y_x"], ok) + _masked_ce(yl, mb["y_y"], ok)
    with torch.no_grad():
        issued = ok * (y > 0)
        acc = ((lp[..., 1:].argmax(-1) + 1 == y) * issued).sum() / issued.sum().clamp(min=1)
    return l_order + l_ptr + 0.5 * l_pt, acc


def _masked_ce(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Cross-entropy [N, O, K] logits against labels [N, O] (-1: none) averaged over the labelled
    entries with weight > 0 (0 when there are none), with no GPU sync."""
    has = (y >= 0).float() * weight
    ce = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.clamp(min=0).flatten(), reduction="none")
    return (ce * has.flatten()).sum() / has.sum().clamp(min=1)


def ppo_update(net: FullGameNet, ref: FullGameNet | None, opt, steps: list[dict], args, warmup: bool, device,
               chunks: list[list[dict]] | None = None, bc_iter=None) -> dict:
    """PPO epochs over the steps; for a network with memory over sequences cut from `chunks` (the
    actors' pieces of trajectories, in order), from the states the actors had. `bc_iter`: batches
    of demonstrations (bc.Steps) whose cloning loss joins each minibatch's, times args.bc_coef."""
    advs = np.array([s["adv"] for s in steps], np.float32)
    mean, std = float(advs.mean()), float(advs.std()) + 1e-8
    if getattr(args, "opd_coef", 0.0) > 0:  # the advisor's orders become orders the unit type can get (as cloning's do)
        typ_y = [(s["type"][:s["n_own"]], s["y_order"]) for s in steps if "y_order" in s]
        if typ_y:
            ty = torch.from_numpy(np.concatenate([a for a, _ in typ_y]).astype(np.int64))
            yy = torch.from_numpy(np.concatenate([b for _, b in typ_y]).astype(np.int64))
            keep = yy > 0
            net.allowed[ty[keep].to(net.allowed.device), yy[keep].to(net.allowed.device)] = True
    if ref is not None:  # the orders each unit type can get: the cloning loss adds the demonstrations' (a
        # pair only the learner allowed had the clone's logit at -1e9: a KL of 96 and a gradient norm
        # of 726 in one of fgself-11's first updates)
        ref.allowed.copy_(net.allowed)
    stats: dict[str, list[torch.Tensor]] = {}  # (kept on the GPU until the end: a float() each was a wait for it)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(device.type == "cuda" and args.bf16))
    net.train()
    seqs = pieces(chunks, args.seq_len) if net.memory else None
    per_mb = max(1, args.minibatch // args.seq_len)
    sizes = np.array([s["n"] for s in steps])
    for _ in range(args.epochs):
        if seqs is None:
            batches = (collate([steps[i] for i in idx], device)
                       for idx in minibatches(sizes, args.minibatch, args.pad_groups))
        else:  # (sequences of similar widths together too: a sequence pads to its minibatch's widest step)
            widths = np.array([max(s["n"] for s in q) for q in seqs])
            batches = (collate_seq([seqs[i] for i in idx], device) for idx in minibatches(widths, per_mb, args.pad_groups))
        for mb in batches:
            seq = mb.get("seq")
            # every forward pass of the minibatch in one autocast region: bf16 matmuls (the losses and
            # log-probabilities in fp32), and a weight cast to bf16 once, not once per pass
            autocast.__enter__()  # (left before the backward pass, below)
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
                with torch.no_grad():
                    ev_ref = evaluate(ref, mb["ent"], mb["type"], mb["cur"], mb["mask"], mb["glob"], mb["n_own"],
                                      mb["order"], mb["tgt"], mb["bx"], mb["by"], mb["avail"],
                                      seq=seq if ref.memory else None)
                lp, lp_ref = torch.log_softmax(ev["logits"].float(), -1), torch.log_softmax(ev_ref["logits"].float(), -1)
                kl = (lp.exp() * (lp - lp_ref)).sum(-1)
                ref_kl = (kl * own).sum() / n_units
                # also while the value warms up: it shares the network's trunk, so training it alone
                # moved the policy too (the KL to the clone doubled in the first five updates)
                loss = loss + args.ref_kl * ref_kl
            if getattr(args, "opd_coef", 0.0) > 0 and mb.get("y_any"):
                l_opd, opd_acc = distill_loss(net, ev, mb, autocast)
                loss = loss + args.opd_coef * l_opd
                stats.setdefault("loss/opd", []).append(l_opd.detach())
                stats.setdefault("opd/acc", []).append(opd_acc)
                stats.setdefault("opd/steps", []).append(mb["y_has"].float().mean())
            if bc_iter is not None and args.bc_coef > 0:  # DAgger's labels as an auxiliary loss (not a fine-tune:
                from .bc import losses as bc_losses  # cloning afterwards overwrote what RL had learned)
                l_bc, _ = bc_losses(net, next(bc_iter), device, None, value_coef=0.0, stats=False)
                loss = loss + args.bc_coef * l_bc
                stats.setdefault("loss/bc", []).append(l_bc.detach())
            autocast.__exit__(None, None, None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            opt.step()
            with torch.no_grad():
                approx_kl = (((ratio - 1) - log_ratio) * own).sum() / n_units
                clipfrac = (((ratio - 1).abs() > args.clip).float() * own).sum() / n_units
            for k, v in (("loss/policy", pg), ("loss/value", v_loss), ("loss/entropy", entropy), ("loss/kl", approx_kl),
                         ("loss/clipfrac", clipfrac), ("loss/ref_kl", ref_kl), ("grad_norm", gn)):
                stats.setdefault(k, []).append(v.detach())
    net.eval()
    return {k: float(torch.stack(v).float().mean()) for k, v in stats.items()}


def publish(net: FullGameNet, version: int, run_dir: Path, name: str = "current.pt") -> None:
    tmp = run_dir / f"{name}.tmp"
    torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "config": net.config,
                "version": version}, tmp)
    tmp.replace(run_dir / name)


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
    ap.add_argument("--pairs", type=int, default=5,
                    help="games per load of the map, each with two players of its own (GameSetup.pairs): the next "
                         "game starts in 0.1 s, and one in this many reloads the map (1: every game does)")
    ap.add_argument("--scripted-reset", type=int, default=0,
                    help="1: games without the built-in AI restart by script (0.1 s) instead of reloading the map (1 s). "
                         "Not the same game: a player's heroes stay counted through it (after the first game's hero the "
                         "next one needed the second tier, a third was refused), and its food count drifted")
    ap.add_argument("--agent-games-factor", type=int, default=4,
                    help="games without the built-in AI: this many times as many per launch (--ai-share is a share of "
                         "launches; with 0.75 and 4, 43%% of the games are against the AI)")
    ap.add_argument("--map", default="duelrush")
    ap.add_argument("--races", default="all")
    ap.add_argument("--mirror", type=int, default=1, help="both sides play the same race (on duelrush the races "
                                                           "are far from balanced: night elf beat the others 96%%)")
    ap.add_argument("--exploiter-share", type=float, default=0.0,
                    help="AlphaStar's main exploiter: this share of launches are games between the learner and a "
                         "second network that trains only against it (League; 0: none)")
    ap.add_argument("--exploiter-batch", type=int, default=4096, help="the exploiter's steps per update")
    ap.add_argument("--exploiter-reset", type=float, default=0.7,
                    help="the exploiter's score against the learner (its last 50 games) at which its weights join the "
                         "league and it starts over from the run's starting weights")
    ap.add_argument("--exploiter-timeout", type=int, default=300, help="updates after which it starts over anyway")
    ap.add_argument("--balance-step", type=float, default=0.02,
                    help="without --mirror: how far a game of the learner against itself moves the tax between "
                         "its two races (League: the winner's income taxed more, towards a 50%% score; 0: no tax)")
    ap.add_argument("--balance-max", type=float, default=0.9, help="the tax between two races at the end of its range")
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
    ap.add_argument("--curriculum-tie", type=float, default=0.0,
                    help="a tie raises the level by this share of a loss's step (0: a tie leaves it)")
    ap.add_argument("--curriculum-delay", type=float, default=120.0,
                    help="the knob at level 1: the AI's late start (seconds; modes hp and delay) or the share of "
                         "its income taken (mode tax, e.g. 0.9)")
    ap.add_argument("--bc-data", type=Path, nargs="*", default=[],
                    help="demonstration collections (e.g. takeover games from the policy's own states) whose cloning "
                         "loss joins each PPO minibatch, times --bc-coef")
    ap.add_argument("--bc-coef", type=float, default=0.02)
    ap.add_argument("--bc-batch", type=int, default=256)
    ap.add_argument("--bc-winners", type=int, default=0, help="the cloning loss: only the side that won each game")
    ap.add_argument("--bc-max-minutes", type=float, default=0.0, help="the cloning loss: only games decided within this many minutes")
    ap.add_argument("--bc-workers", type=int, default=2)
    ap.add_argument("--opd-share", type=float, default=0.0,
                    help="on-policy distillation: the share of games (not the real game) in which the built-in AI "
                         "advises the learner's side (protocol.ShadowAI: its orders are labels, undone); the "
                         "learner's steps there get its labels and a cloning loss towards them (--opd-coef)")
    ap.add_argument("--opd-coef", type=float, default=0.05, help="the distillation loss's weight (--opd-share)")
    ap.add_argument("--opd-difficulty", default="insane", help="the advising AI's difficulty")
    ap.add_argument("--real-share", type=float, default=0.1, help="with a curriculum: the share of launches against "
                                                                   "the built-in AI that play the real game (the yardstick)")
    ap.add_argument("--curriculum-mode", default="hp", choices=("hp", "delay", "tax"),
                    help="the curriculum's knobs (League): the learner's hit points then the AI's late start, the "
                         "late start alone, or a tax on the AI's income")
    ap.add_argument("--self-share", type=float, default=0.5, help="of the agent games: against itself")
    ap.add_argument("--snapshot-every", type=int, default=20, help="updates between league snapshots")
    ap.add_argument("--max-past", type=int, default=12)
    ap.add_argument("--pfsp", default="hard")
    ap.add_argument("--batch-steps", type=int, default=8192, help="agent steps per update")
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--pad-groups", type=int, default=8,
                    help="minibatches are cut from groups of this many, sorted by entity count: they pad to their widest "
                         "step (1: minibatches at random)")
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
    ap.add_argument("--compile", type=int, default=1, help="the actors' network calls as CUDA graphs")
    ap.add_argument("--infer-period-ms", type=float, default=4.0,
                    help="the inference server makes a round of calls at most this often (0: whenever requests wait)")
    ap.add_argument("--pin", type=int, default=0,
                    help="1: the learner and the inference server on a core each of their own, the games on the rest")
    ap.add_argument("--game-nice", type=int, default=10,
                    help="the games' niceness: they yield the CPU to the inference server, the actors and the learner")
    ap.add_argument("--central-inference", type=int, default=1,
                    help="1: one inference server makes every actor's network calls (one GPU context, batches over all "
                         "games); 0: each actor its own")
    ap.add_argument("--native-obs", type=int, default=1, help="observations parsed in C (warcraftsim.native)")
    ap.add_argument("--avail-mask", type=int, default=1, help="mask the orders the player can't pay for yet")
    ap.add_argument("--chunk", type=int, default=64, help="steps per trajectory piece an actor sends")
    ap.add_argument("--seq-len", type=int, default=16, help="a network with memory: steps per training sequence "
                                                            "(from the state the actor had at its first)")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--video-every", type=float, default=10.0,
                    help="minutes between game videos (0: none; the actors take turns). Rendering one takes ~6 minutes of "
                         "3.4 cores (the game at 960x540 on a software renderer, in real time)")
    ap.add_argument("--shaping", type=float, default=1.0, help="weight of the material-lead reward shaping (0: none)")
    ap.add_argument("--shaping-scale", type=float, default=2000.0, help="material (gold + lumber cost) worth 1 of potential")
    ap.add_argument("--float-penalty", type=float, default=0.0,
                    help="the shaping's potential counts the gold and lumber a player holds at minus this (0: not at "
                         "all): spending is credited at the order")
    ap.add_argument("--tie-break", type=float, default=0.5, help="a tie's reward: this times tanh(2 x material lead)")
    ap.add_argument("--tie-value", type=float, default=0.0,
                    help="added to a tie's reward (e.g. -0.5: a tie at the time limit counts almost as a loss)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", help="default: cuda if available (asking opens the GPU driver)")
    ap.add_argument("--resume", action="store_true", help="continue the run --name: its latest checkpoint, league and counts")
    ap.add_argument("--title", default="", help="with --resume: the restart's short name (the lineage's subtitle)")
    ap.add_argument("--note", default="", help="the run's notes (notes.md); with --resume, why it was restarted (the "
                                                "restart's marker on the dashboard's charts)")
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
    # the main exploiter (--exploiter-share): a second network, from the run's starting weights
    # (as AlphaStar's from the supervised agent), trained on its games against the learner alone
    x_init = {k: v.detach().clone() for k, v in net.state_dict().items()} if args.exploiter_share > 0 else None
    xnet = copy.deepcopy(net) if x_init is not None else None
    xopt = torch.optim.Adam(xnet.parameters(), lr=args.lr, eps=1e-5) if xnet is not None else None
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
            "inputs": {"init": str(args.init), "bc_data": [str(d) for d in (args.bc_data or [])],
                       **({"advisor": f"built-in AI ({args.opd_difficulty})"} if args.opd_share > 0 else {})},
            "launch": {"command": "python3 -m warcraftsim.fullgame.selfplay " + " ".join(sys.argv[1:] if argv is None else argv),
                       "argv": list(sys.argv[1:] if argv is None else argv), "git": git_info()}}

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
        # what this launch takes in (the lineage: a new anchor or new demonstrations enter the run here)
        info.update(created=old.get("created", info["created"]), init_from=old.get("init_from", info["init_from"]),
                    inputs=old.get("inputs") or info["inputs"],
                    resumes=old.get("resumes", []) + [{"time": time.time(), **resumed, **({"note": args.note} if args.note else {}),
                                                       **({"title": args.title} if args.title else {}),
                                                       "inputs": info["inputs"], "argv": info["launch"]["argv"]}])

    def save_info():
        try:  # notes written on the restarts since (by hand, in run.json) stay
            notes = {r["time"]: r["note"] for r in json.loads((run_dir / "run.json").read_text()).get("resumes", [])
                     if r.get("note")}
            for r in info.get("resumes", []):
                if not r.get("note") and r["time"] in notes:
                    r["note"] = notes[r["time"]]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            pass
        tmp = run_dir / "run.json.tmp"
        tmp.write_text(json.dumps(info, indent=1))
        tmp.replace(run_dir / "run.json")
    save_info()
    if args.note and resumed is None:  # (resuming, it is the restart's note: a marker on the dashboard's charts)
        (run_dir / "notes.md").write_text(args.note + "\n")

    league = League(run_dir, ai, {"ai": args.ai_share, "self": args.self_share, "past": 1.0 - args.self_share},
                    args.max_past, args.pfsp,
                    curriculum=((args.curriculum, args.curriculum_step, args.curriculum_delay, args.handicap)
                                if args.curriculum >= 0 else None), mode=args.curriculum_mode, races=tuple(races),
                    mirror=bool(args.mirror),
                    balance=(args.balance_step, args.balance_max) if args.balance_step > 0 else None,
                    exploiter_share=args.exploiter_share, tie=args.curriculum_tie)
    league.real_share = args.real_share
    if resumed is not None and (run_dir / "league.json").exists():
        league.restore(json.loads((run_dir / "league.json").read_text()))
    else:
        first = run_dir / "checkpoints" / f"{0:016d}.pt"
        torch.save({**ck, "model": net.state_dict(), "config": net.config, "vocab": vocab, "agent_steps": 0}, first)
        league.add_snapshot(first, 0)  # the clone itself: the first past opponent
    league.write()
    publish(net, resumed["update"] if resumed else 0, run_dir)
    x_update = x_since = 0  # the exploiter's updates (its weights' version), and since its last restart
    if xnet is not None:
        if resumed is not None and (run_dir / "exploiter.pt").exists():  # (it goes on too)
            xck = torch.load(run_dir / "exploiter.pt", map_location=device, weights_only=False)
            xnet.load_state_dict(xck["model"])
            x_update, x_since = xck["version"], xck.get("since", 0)
        publish(xnet, x_update, run_dir, "exploiter.pt")
    cfg = {"run_dir": str(run_dir), "device": str(device), "vocab": str(run_dir / "vocab.json"), "races": races,
           "map": args.map, "handicap": args.handicap, "step_seconds": args.step_seconds,
           "max_minutes": args.max_minutes, "max_steps": int(args.max_minutes * 60 / args.step_seconds) + 20,
           "wait_floor_ms": args.wait_floor_ms, "games_per_actor": args.games_per_actor,
           "games_per_process": args.games_per_process, "chunk": args.chunk, "gamma": args.gamma, "lam": args.lam,
           "seed": args.seed, "start_update": resumed["update"] if resumed else 0, "slot": slot,
           "video_every": args.video_every, "scripted_reset": bool(args.scripted_reset),
           "agent_games_factor": args.agent_games_factor, "mirror": bool(args.mirror), "learner_pid": os.getpid(),
           "actors": args.actors, "values": unit_values(), "shaping": args.shaping, "shaping_scale": args.shaping_scale,
           "tie_break": args.tie_break, "tie_value": args.tie_value, "compile": bool(args.compile), "native_obs": bool(args.native_obs),
           "real_share": args.real_share, "infer_batch": 64, "game_nice": args.game_nice, "pairs": args.pairs,
           "exploiter_share": args.exploiter_share, "float_penalty": args.float_penalty,
           "infer_period": args.infer_period_ms / 1000.0,
           "avail_mask": bool(args.avail_mask),
           "opd_share": args.opd_share, "opd_difficulty": args.opd_difficulty,
           "costs": order_costs(vocab, args.map).tolist() if args.avail_mask else None, "max_past": args.max_past,
           "order_names": vocab.get("order_names") or {str(k): v for k, v in fx.demo_order_names(args.runs).items()}}
    if args.pin and (os.cpu_count() or 0) >= 16:
        # the learner's Python is what bounds an update, and the inference server what the games wait
        # for: each gets a core (both of its threads) that no game runs on. (CPUs 2k and 2k + 1 are a core.)
        every = sorted(os.sched_getaffinity(0))
        cfg["cpus"] = {"learner": every[:2], "server": every[2:4], "games": every[4:]}
    ctx = torch.multiprocessing.get_context("spawn")
    # the actors wait when the learner is a batch behind (unbounded, faster actors piled up steps the
    # learner then trained on several updates late: 3 after ten minutes, and growing)
    out_q, stop, render_q = ctx.Queue(maxsize=max(32, args.batch_steps // args.chunk)), ctx.Event(), ctx.Queue()
    # a pipe per game to the inference server: [actor][game] -> (the game's end, the server's)
    pipes = [[ctx.Pipe() for _ in range(args.games_per_actor)] if args.central_inference else None
             for _ in range(args.actors)]
    actors = [ctx.Process(target=actor_main, args=(w, cfg, out_q, stop, render_q, pipes[w] and [a for a, _ in pipes[w]]),
                          daemon=True) for w in range(args.actors)]
    if args.central_inference:
        actors.append(ctx.Process(target=inference_main, args=(cfg, [b for ps in pipes for _, b in ps], stop), daemon=True))
    if args.video_every > 0:
        actors.append(ctx.Process(target=render_main, args=(cfg, render_q, stop), daemon=True))
    for p in actors:
        p.start()
    _pin(cfg, "learner")  # (after the others started: they would inherit it)
    info.update(status="training", started=time.time())
    save_info()
    t0 = time.time()
    agent_steps, update, episodes = ((resumed["steps"], resumed["update"], resumed["episodes"]) if resumed
                                     else (0, 0, 0))
    bc_iter = None
    if args.bc_data:  # demonstrations for the auxiliary cloning loss, encoded by loader workers, epoch after epoch
        from .bc import Steps, as_is
        bc_paths = sorted(p for d in args.bc_data for p in d.glob("game*.npz"))
        bc_set = Steps(bc_paths, vocab, args.bc_batch, cfg["values"], order_costs(vocab, args.map), arrays=True,
                       winners=bool(args.bc_winners), max_minutes=args.bc_max_minutes)

        def bc_cycle():
            epoch = 0
            while True:
                bc_set.epoch, epoch = epoch, epoch + 1
                for b in torch.utils.data.DataLoader(bc_set, batch_size=None, num_workers=args.bc_workers, collate_fn=as_is,
                                                     worker_init_fn=lambda _i: _pin(cfg, "games")):
                    yield {k: torch.from_numpy(v) for k, v in b.items()}
        bc_iter = bc_cycle()
        print(f"auxiliary cloning loss: {len(bc_paths)} games, x{args.bc_coef}", flush=True)
    buf: list[dict] = []
    chunks: list[list[dict]] = []
    stale = []
    xbuf: list[dict] = []
    xchunks: list[list[dict]] = []
    x_stats: dict = {}

    def exploiter_update() -> None:
        """A PPO update of the exploiter on its steps (no cloning loss: it is to find what beats the
        learner, not to play like the AI), then a restart when it beats the learner often enough
        (its weights join the league) or has not in --exploiter-timeout updates."""
        nonlocal xbuf, xchunks, x_update, x_since, xopt, x_stats
        x_stats = ppo_update(xnet, ref, xopt, xbuf, args, x_since < args.value_warmup, device, xchunks, None)
        xbuf, xchunks = [], []
        x_update += 1
        x_since += 1
        xm = league.exploiter
        score = None if len(xm.recent) < 50 else 1.0 - xm.win_rate()
        if (score is not None and score >= args.exploiter_reset) or x_since >= args.exploiter_timeout:
            if score is not None and score >= args.exploiter_reset:
                path = run_dir / "checkpoints" / f"exploiter-{agent_steps:016d}.pt"
                torch.save({"model": xnet.state_dict(), "config": xnet.config, "vocab": vocab, "agent_steps": agent_steps,
                            "update": x_update}, path)
                league.add_snapshot(path, agent_steps, name=f"exploiter:{agent_steps}")
            print(f"exploiter restarts after {x_since} updates (its score against the learner: "
                  f"{'-' if score is None else f'{score:.2f}'})", flush=True)
            xnet.load_state_dict(x_init)
            xopt = torch.optim.Adam(xnet.parameters(), lr=args.lr, eps=1e-5)
            xm.recent = []
            league.exploiter_resets += 1
            x_since = 0
        torch.save({"model": {k: v.detach().cpu() for k, v in xnet.state_dict().items()}, "config": xnet.config,
                    "version": x_update, "since": x_since}, run_dir / "exploiter.pt.tmp")
        (run_dir / "exploiter.pt.tmp").replace(run_dir / "exploiter.pt")
        league.write()

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
                    if "curriculum" in e:
                        league.curriculum(e["opponent"], e["outcome"], league.matchup(e["race"], e["opponent_race"]))
                    if e["opponent"] == "self" and e["race"] != e["opponent_race"]:
                        league.balanced(e["race"], e["opponent_race"], e["outcome"])
                    with open(run_dir / "episodes.jsonl", "a") as f:
                        f.write(json.dumps(e) + "\n")
                    continue
                if (msg.get("info") or {}).get("learner") == "exploiter":
                    xbuf += msg["steps"]
                    xchunks.append(msg["steps"])
                    if len(xbuf) >= args.exploiter_batch:
                        exploiter_update()
                    continue
                buf += msg["steps"]
                chunks.append(msg["steps"])
                stale += [update - s["version"] for s in msg["steps"] if s["version"] >= 0]
            t_train = time.time()
            steps, buf, batch_chunks, chunks = buf, [], chunks, []
            warmup = update < args.value_warmup
            stats = ppo_update(net, ref, opt, steps, args, warmup, device, batch_chunks, bc_iter)
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
            if x_stats:  # the exploiter's last update
                stats.update({f"exploiter/{k.split('/')[-1]}": v for k, v in x_stats.items()})
                stats["exploiter/updates"] = x_update
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
