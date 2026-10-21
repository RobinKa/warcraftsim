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
import contextlib
import copy
import json
import math
import os
import queue
import random
import struct
import subprocess
import sys
import threading
import time
import traceback
import types
import weakref
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import features as fx
from . import modes as zmodes
from . import scenarios
from . import strategy as strat
from .collect import claim_slot
from .costs import order_costs
from .trace import DEAD_FLAG, Production, material, trace_step, unit_values  # noqa: F401
from .play import unit_rows
from .model import FullGameNet, act, adapt_config, anchor_kl, arch_key, evaluate, load, trained
from .teacher import Teacher
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

    def vocab(self) -> dict:
        if not hasattr(self, "_vocab"):
            self._vocab = json.loads(Path(self.cfg["vocab"]).read_text())
        return self._vocab

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
        if net is None:  # (--neutral: its orders in the run's vocabulary's order, as the games decode them)
            net, _ = load(key, self.device, vocab=self.vocab() if fx.NEUTRAL else None)
        self.snapshots[key] = net  # most recently used last
        # the whole league stays loaded (~8 MB each on the GPU): with a cache of 4 and 8+ snapshots
        # it reloaded one from disk on almost every call (83% of an actor's time, the games waiting)
        while len(self.snapshots) > self.cfg["max_past"] + 2:
            self.snapshots.pop(next(iter(self.snapshots)))
        return net


class Inference:
    """Batches the network calls of an actor's agents (one per game side) on the GPU."""

    def __init__(self, nets: Nets, device, max_batch: int, compile_: bool = True, buckets: tuple[int, ...] | None = None,
                 thread: bool = True, entities: tuple[int, ...] = (fx.MAX_ENT,), shared: "SharedRows | None" = None,
                 fuse: bool = False, bf16: bool = False, generator: torch.Generator | None = None,
                 capture_mode: str = "global", guide: bool = False):
        """`buckets`: the batch sizes its compiled calls pad to (the smallest that fits; default: just
        max_batch); `entities`: likewise the entity counts (a call padded 4 rows of ~47 entities to
        16 x 160: ~50 times the attention). `thread` False: no batching thread of its own (the
        inference server calls _forward)."""
        self.nets, self.device, self.max_batch = nets, device, max_batch
        # guide: rows can carry a classifier-free guidance weight (model.act; --eval-guide): every call then runs the
        # network twice, so only the evaluation worker's (its games are all evaluation games)
        self.guide = guide
        self.shared = shared  # view steps given as {"n", "n_own", "row"}: their rows there
        # fuse: the call compiled (torch.compile, inductor: fused kernels) before its CUDA graph is captured.
        # A call was ~150 small kernels on ~10 rows: the GPU waited between them more than it computed.
        self.fuse = fuse and device.type == "cuda"
        if self.fuse:  # (one compile per network object and shape, of the same function: past 8 the rest ran unfused)
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 128)
        # bf16: the network's matmuls in bfloat16 (tensor cores); the sampling and log-probabilities stay fp32
        self.bf16 = bf16 and device.type == "cuda"
        # generator: the sampling's random numbers (an inference thread in the learner's process: the learner's
        # dropout uses the default one, and a capture in one thread must not see the other's draws);
        # capture_mode "thread_local": the other thread's CUDA calls go on while a graph is captured
        self.generator, self.capture_mode = generator, capture_mode
        self.buckets = tuple(sorted(buckets or (max_batch,)))
        self.entities = tuple(sorted(set(entities) | {fx.MAX_ENT}))
        self._hosts: dict[tuple, torch.Tensor] = {}
        self._used: dict[tuple, int] = {}  # the host buffers taken since the last wait() (or begin())
        self.bank = 0  # which set of host buffers: two rounds in flight use two (inference_main's pipeline)
        # the calls as CUDA graphs (one launch instead of ~150 kernels: the network is small, its calls
        # latency-bound). Graphs need fixed shapes: the batch and the entities padded to a bucket.
        # Captured from the eager network, per network and shape, in well under a second. (torch.compile's
        # "reduce-overhead" graphs took 10-20 s a shape, the games waiting, and its guards 0.3 ms a call.)
        self.compile = compile_ and device.type == "cuda"
        self._fns = weakref.WeakKeyDictionary()  # net -> its packed act
        self._graphs = weakref.WeakKeyDictionary()  # net -> {(batch, width): (input, graph, output)}
        # net -> the stream its graphs are captured on: a graph keeps the cuBLAS workspace of the stream it was
        # captured on, and two networks' graphs replayed side by side must not share one (CUDA 13's cuBLAS picks
        # split-K kernels that count in it: two at once and one waited forever, the GPU at 100%)
        self._capture = weakref.WeakKeyDictionary()
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
            bf16, gen, guided = self.bf16, self.generator, self.guide

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
                # (class 0, "none", always available, carries flags: 2 marks a shared-noise game's row; with guide,
                # + 10 * round(10 * w) a guidance weight w)
                flag = x[:, a + 1]
                tens = torch.floor(flag / 10.0) if guided else torch.zeros_like(flag)
                shared = flag - 10.0 * tens > 1.5
                guide = torch.where(tens > 0, tens / 10.0, torch.ones_like(tens)) if guided else None
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
                    out = act(ref(), ent, typ, cur, mask, glob, n_own, avail, x[:, a + 1 + NO:] if d else None,
                              generator=gen, shared=shared, guide=guide)
                return torch.cat([out["order"].float(), out["tgt"].float(), out["bx"].float(), out["by"].float(),
                                  out["logp"].float(), out["value"].float()[:, None], out["entropy"].float()[:, None]]
                                 + ([out["h"].float()] if d else []), 1)
            if self.fuse:  # (compiled at the shape's first calls, before its graph's capture)
                packed_act = torch.compile(packed_act, dynamic=False)
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
            if self.generator is not None:
                graph.register_generator_state(self.generator)
            if net not in self._capture:
                self._capture[net] = torch.cuda.Stream(device=self.device)
            with torch.cuda.graph(graph, stream=self._capture[net], capture_error_mode=self.capture_mode):
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
        G, NO, d = net.config["G"], net.config["n_orders"], net.config["d"] if net.memory else 0
        W = E * (fx.F + 3) + G + 1 + NO + d
        pinned = fixed and self.device.type == "cuda"
        if pinned:  # the same shapes every call: pinned host buffers, reused (a round's calls each their own)
            slot = self._used[(B, W)] = self._used.get((B, W), -1) + 1
            if (B, W, slot, self.bank) not in self._hosts:
                self._hosts[(B, W, slot, self.bank)] = torch.zeros(B, W).pin_memory()
            host = self._hosts[(B, W, slot, self.bank)]
            host.zero_()
        else:
            host = torch.zeros(B, W)
        hb = host.numpy()
        a = E * fx.F
        ent = hb[:, :a].reshape(B, E, fx.F, copy=False)  # (a view)
        glob = hb[:, a:a + G]
        a += G
        typ, cur, mask = (hb[:, a + k * E:a + (k + 1) * E] for k in range(3))
        a += 3 * E
        n_own, avail, hs = hb[:, a], hb[:, a + 1:a + 1 + NO], hb[:, a + 1 + NO:]
        avail[:] = 1.0
        mask[:, 0] = 1.0  # (padding rows: one entity, or attention over nothing gives NaNs)
        if self.shared is not None and "row" in sts[0]:  # the rows gathered at once (zeros past each's entities)
            sh, R = self.shared, len(sts)
            idx = np.fromiter((st["row"] for st in sts), np.int64, R)
            ns = np.fromiter((st["n"] for st in sts), np.int64, R)
            keep = np.arange(E)[None, :] < ns[:, None]
            mask[:R] = keep
            ent[:R] = sh.ent[idx, :E] * keep[..., None]
            typ[:R], cur[:R] = sh.typ[idx, :E] * keep, sh.cur[idx, :E] * keep
            glob[:R], avail[:R] = sh.glob[idx], sh.avail[idx]
            n_own[:R] = np.minimum(sh.n_own[idx], fx.MAX_OWN)
            if d:
                hs[:R] = sh.h[idx, :d]
        else:
            for i, st in enumerate(sts):
                n = st["n"]
                ent[i, :n], typ[i, :n], cur[i, :n], mask[i, :n] = st["ent"], st["type"], st["cur"], 1.0
                glob[i], n_own[i] = st["glob"], min(st["n_own"], fx.MAX_OWN)
                if st.get("avail") is not None:
                    avail[i] = st["avail"]
                if st.get("shared_noise"):  # (see SharedRows.write)
                    avail[i, 0] = 2.0
                if st.get("guide"):  # (--eval-guide: the weight with the flags, see packed_act)
                    avail[i, 0] += 10.0 * round(10.0 * st["guide"])
                if d and st.get("h") is not None:  # (none: the game's start, zeros)
                    hs[i] = st["h"]
        own = n_own[:len(sts)].astype(np.int64)
        if pinned:  # queued one behind the other: the input in, the graph, its output out (before it runs again)
            x, graph, packed = self._graph(net, host)
            x.copy_(host, non_blocking=True)
            graph.replay()
            key = ("out", B, packed.shape[1], slot, self.bank)
            if key not in self._hosts:
                self._hosts[key] = torch.zeros(B, packed.shape[1]).pin_memory()
            out = self._hosts[key]
            out.copy_(packed, non_blocking=True)
        else:
            out = self._fn(net)(host.to(self.device))
        return out, own, min(fx.MAX_OWN, E), bool(d)

    def begin(self, bank: int) -> None:
        """A round on this set of host buffers (the other one may still be in flight)."""
        self.bank = bank
        self._used.clear()

    def wait(self, streams: list | None = None) -> None:
        """Until the calls launched (on these streams; default: the current one) are done (sleeping:
        CUDA's default sync spins a core the games need)."""
        if self.device.type == "cuda":
            events = []
            for st in streams or [torch.cuda.current_stream()]:
                done = torch.cuda.Event(blocking=True)
                done.record(st)
                events.append(done)
            for done in events:
                done.synchronize()
        self._used.clear()

    def _finish_shared(self, handle: tuple, rows: list[int], version: int) -> list[tuple]:
        """A call's answers into its rows of the shared memory: [(version, O, memory width)] per row."""
        out, own, O, memory = handle
        p = out.numpy() if out.device.type == "cpu" else out.cpu().numpy()
        R, sh = len(rows), self.shared
        idx = np.asarray(rows, np.int64)
        sh.out[idx, :p.shape[1]] = p[:R]
        width = p.shape[1] - 5 * O - 2 if memory else 0
        return [(version, O, width)] * R

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


class SharedRows:
    """The games' requests to the inference server and its answers in shared memory (a file in
    /dev/shm, mapped by each process): a game writes its sides' view steps into rows of its own and
    sends only their numbers; the server gathers a call's rows with a few index operations and writes
    the answers back the same way. (Pickled through the pipes they were ~18% of the server's time and
    ~11% of an actor's, and the server copied its inputs in row by row.) A row is written before its
    message is sent and read after it arrives: the pipe orders them."""

    def __init__(self, path: str, rows: int, G: int, NO: int, d: int, create: bool = False):
        E, F, O = fx.MAX_ENT, fx.F, fx.MAX_OWN
        layout = [("ent", (rows, E, F)), ("typ", (rows, E)), ("cur", (rows, E)), ("glob", (rows, G)), ("n_own", (rows,)),
                  ("avail", (rows, NO)), ("h", (rows, max(d, 1))), ("out", (rows, 5 * O + 2 + max(d, 1)))]
        size = sum(int(np.prod(shape)) for _, shape in layout)
        if create:
            with open(path, "wb") as f:
                f.truncate(4 * size)
        self.mm = np.memmap(path, dtype=np.float32, mode="r+", shape=(size,))
        at = 0
        for name, shape in layout:
            n = int(np.prod(shape))
            setattr(self, name, self.mm[at:at + n].reshape(shape))
            at += n
        self.spec = (path, rows, G, NO, d)

    def write(self, r: int, st: dict) -> None:
        """A view step into row r (an actor's game)."""
        n = st["n"]
        self.ent[r, :n], self.typ[r, :n], self.cur[r, :n] = st["ent"], st["type"], st["cur"]
        self.glob[r], self.n_own[r] = st["glob"], st["n_own"]
        self.avail[r] = st["avail"] if st.get("avail") is not None else 1.0
        if st.get("shared_noise"):  # (class 0 is always available: 2 marks the row, see packed_act)
            self.avail[r, 0] = 2.0
        if st.get("guide"):  # (--eval-guide: the weight with the flags)
            self.avail[r, 0] += 10.0 * round(10.0 * st["guide"])
        h = st.get("h")
        self.h[r, :len(h) if h is not None else self.h.shape[1]] = h if h is not None else 0.0

    def answer(self, r: int, n_own: int, info: tuple) -> dict:
        """Row r's answer (an actor's game): info (version, O, memory) from the server."""
        version, O, memory = info
        p, o = self.out[r], min(n_own, fx.MAX_OWN, O)
        # (plain arrays: astype keeps the memmap subclass, and unpickling memmaps cost the learner ~5% of its time)
        res = {k: np.array(p[i * O:i * O + o], dtype=np.int64) for i, k in enumerate(("order", "tgt", "bx", "by"))}
        res.update(logp=np.array(p[4 * O:4 * O + o]), value=float(p[5 * O]), entropy=float(p[5 * O + 1]), version=version)
        if memory:
            res["h"] = np.array(p[5 * O + 2:5 * O + 2 + memory])
        return res


def pack_request(rid: int, items: list[tuple]) -> bytes:
    """A game's request with shared rows: [(network key, entities, own units, row)] (csrc/infer_loop.cpp's format)."""
    out = [struct.pack("<IB", rid & 0xFFFFFFFF, len(items))]
    for key, n, n_own, row in items:
        kb = key.encode()
        out.append(struct.pack("<HHIH", n, n_own, row, len(kb)) + kb)
    return b"".join(out)


def unpack_request(msg: bytes) -> tuple[int, list[tuple]]:
    rid, count = struct.unpack_from("<IB", msg)
    at, items = 5, []
    for _ in range(count):
        n, n_own, row, kl = struct.unpack_from("<HHIH", msg, at)
        items.append((msg[at + 10:at + 10 + kl].decode(), n, n_own, row))
        at += 10 + kl
    return rid, items


def pack_reply(rid: int, answers: list) -> bytes:
    """[(version, O, memory width) or None (no answer)] for a request's items."""
    return struct.pack("<IB", rid & 0xFFFFFFFF, len(answers)) + b"".join(
        struct.pack("<iHH", *(a if a is not None else (-2, 0, 0))) for a in answers)


def unpack_reply(msg: bytes) -> tuple[int, list]:
    rid, count = struct.unpack_from("<IB", msg)
    out = []
    for k in range(count):
        v, o, w = struct.unpack_from("<iHH", msg, 5 + 8 * k)
        out.append(None if v == -2 else (v, o, w))
    return rid, out


class RemoteInference:
    """An actor's side of the inference server (inference_main): the same request() as Inference, the
    calls made in one process for all the actors' games. Every game thread has a pipe of its own to
    the server and waits on it itself. (Through queues a request took ~32 ms of a 59 ms step, the
    server's call 4.5 of them: a feeder thread and a shared lock on the way in, a feeder, a reader
    thread and an event on the way back, each a wait for some process's GIL.)"""

    def __init__(self, conns: list, shared: SharedRows | None = None, rows: list[int] | None = None):
        """`shared`, `rows`: the view steps go through shared memory, each pipe's game in rows of its own
        (rows[i]: the first of conns[i]'s two)."""
        self.free = list(zip(conns, rows or [0] * len(conns)))
        self.shared = shared
        self.lock = threading.Lock()
        self.local = threading.local()
        self.nets = types.SimpleNamespace(version=-1)  # (the version the server answered with last)

    def request(self, items: list[tuple[str, dict]]) -> list[dict]:
        loc = self.local
        if not hasattr(loc, "conn"):
            with self.lock:
                (loc.conn, loc.row), loc.rid = self.free.pop(), 0
        loc.rid += 1
        sh = self.shared
        if sh is not None:  # (a game's two sides at most: its two rows; binary messages: pack_request)
            assert len(items) <= 2
            for k, (_, st) in enumerate(items):
                sh.write(loc.row + k, st)
            loc.conn.send_bytes(pack_request(loc.rid, [(key, st["n"], int(st["n_own"]), loc.row + k)
                                                       for k, (key, st) in enumerate(items)]))
        else:
            loc.conn.send((loc.rid, [(key, {f: st.get(f) for f in ST_FIELDS}) for key, st in items]))
        while True:
            if not loc.conn.poll(300):
                raise RuntimeError("no answer from the inference server for 5 minutes")
            rid, out = unpack_reply(loc.conn.recv_bytes()) if sh is not None else loc.conn.recv()
            if rid == loc.rid:  # (not the answer to a request this thread gave up on)
                break
        if sh is not None:
            out = [None if info is None else sh.answer(loc.row + k, int(items[k][1]["n_own"]), info)
                   for k, info in enumerate(out)]
        for res in out:
            if res is not None and res.get("version", -1) >= 0:
                self.nets.version = res["version"]
        return out


class PastNet:
    """The past snapshots' calls through a few network objects (slots, per architecture): a snapshot's
    weights are copied into a slot before its calls. Its parameters are views of one flat tensor, so
    that is one copy. Each slot has its CUDA graphs and stream: a round's snapshots run side by side.
    (Compiled per snapshot, every new league member stalled all the games for 25-50 s: four shapes to
    compile, every 20 updates.)"""

    def __init__(self, device, slots: int = 1):
        self.device, self.slots = device, slots
        self.hosts: dict[tuple, list] = {}  # (architecture, slot) -> [network, its flat weights, its buffers, the snapshot loaded]
        self.flats = weakref.WeakKeyDictionary()  # snapshot network -> (its weights, flat; its buffers)
        self.holding: dict[int, str] = {}  # slot -> the snapshot it holds

    def slot(self, key: str, taken: set) -> int:
        """A slot for this snapshot in a round (`taken`: the round's slots so far): the one that holds
        it, else a free one, else one already taken (its calls then wait for the other's)."""
        free = [k for k in range(self.slots) if k not in taken]
        held = [k for k in free if self.holding.get(k) == key]
        if held:
            return held[0]
        if free:
            empty = [k for k in free if k not in self.holding]
            return (empty or free)[0]
        return len(taken) % self.slots

    @torch.no_grad()
    def get(self, key: str, src: FullGameNet, slot: int = 0) -> FullGameNet:
        if src not in self.flats:
            self.flats[src] = (torch.cat([p.data.reshape(-1) for p in src.parameters()]), list(src.buffers()),
                               arch_key(src.config))
        flat, buffers, arch = self.flats[src]
        self.holding[slot] = key
        host = self.hosts.get((arch, slot))
        if host is None:
            net = FullGameNet(**src.config).to(self.device).eval()
            net.load_state_dict(src.state_dict())
            mine = torch.cat([p.data.reshape(-1) for p in net.parameters()])
            at = 0
            for p in net.parameters():
                p.data = mine[at:at + p.numel()].view(p.shape)
                at += p.numel()
            host = self.hosts[(arch, slot)] = [net, mine, list(net.buffers()), key]
        if host[3] != key:
            host[1].copy_(flat)
            for mine, theirs in zip(host[2], buffers):
                mine.copy_(theirs)
            host[3] = key
        return host[0]


def launch_round(groups: dict, nets: Nets, infer: Inference, past: Inference, swap: PastNet, top: int,
                 streams: dict | None, priority: int = 0) -> tuple[list, list]:
    """Start a round's calls without waiting for them (-> [(its Inference, handle, the part's items, the
    network's version)], the streams to wait for). `groups`: network key -> [(pipe, index, view step)].
    `streams` (a dict the server keeps them in): every network its own CUDA stream (the current
    network's, each past snapshot slot's), so a round's calls run side by side; one after the other
    they were ~3-4 small calls a round and ~45% of the server's time a wait for the GPU. None: all on
    the current stream."""
    cuda = streams is not None and infer.device.type == "cuda"
    base = torch.cuda.current_stream() if cuda else None
    calls, used, taken = [], [], set()
    for key, items in groups.items():
        live_key = key in LIVE
        fwd, size = (infer, top) if live_key else (past, 16)
        slot = 0 if live_key else swap.slot(key, taken)
        taken |= set() if live_key else {slot}
        st = None
        if cuda:
            st = streams.get((live_key, key if live_key else slot))
            if st is None:
                st = streams[(live_key, key if live_key else slot)] = torch.cuda.Stream(device=infer.device,
                                                                                         priority=priority)
            if st not in used:  # (after the current stream's work: weights reloaded, snapshots loaded)
                st.wait_stream(base)
                used.append(st)
        with torch.cuda.stream(st) if st is not None else contextlib.nullcontext():
            for a in range(0, len(items), size):
                part = items[a:a + size]
                try:
                    net = nets.get(key)
                    handle = fwd._launch(net if live_key else swap.get(key, net, slot), [it[2] for it in part])
                except Exception:  # noqa: BLE001 (the games must not hang on it)
                    traceback.print_exc()
                    handle = None
                calls.append((fwd, handle, part, nets.live_version(key)))
    return calls, used


def warm_inference(nets: Nets, infer: Inference, past: Inference, swap: PastNet, keys: list[str]) -> None:
    """Capture every shape's graph now (the current network's; each past snapshot slot's, for each
    architecture among `keys`): an inference thread next to the learner must not capture later."""
    def dummy(n: int, G: int) -> dict:
        return {"n": n, "n_own": 1, "ent": np.zeros((n, fx.F), np.float32), "type": np.zeros(n, np.int64),
                "cur": np.zeros(n, np.int64), "glob": np.zeros(G, np.float32), "avail": None, "h": None}
    net = nets.current
    G = net.config["G"]
    for B in infer.buckets:
        for E in infer.entities:
            infer.begin(0)
            infer._launch(net, [dummy(E, G)] * B)
            infer.wait()
    archs: dict[str, FullGameNet] = {}
    for key in [None, *keys]:
        src = net if key is None else nets.get(key)
        archs.setdefault(arch_key(src.config), src)
    for src in archs.values():
        for k in range(swap.slots):
            slot_net = swap.get(f"warm:{id(src)}", src, k)
            for B in past.buckets:
                for E in past.entities:
                    past.begin(0)
                    past._launch(slot_net, [dummy(E, src.config["G"])] * B)
                    past.wait()
            swap.holding.pop(k, None)  # (nothing real in it yet)


def inference_main(cfg: dict, conns: list, stop, in_thread: bool = False) -> None:
    """The inference server: every actor's network calls in one process. The games spent 53% of
    their time waiting for their actor's calls: four actors and the learner took turns on the GPU
    (a context each), and an actor's batches were mostly padding (its 8 games rarely ask at once).
    Here the games' requests batch together, in one context: whatever has arrived while the last
    call ran goes into the next one (no waiting for more)."""
    import signal
    from multiprocessing.connection import wait
    device = torch.device(cfg["device"])
    # in_thread: a thread of the learner's process (--infer-in-learner): one GPU context, so the learner's
    # kernels and these run side by side instead of taking turns (WSL has no MPS); its own high-priority
    # streams and its own random generator, and every graph captured before the loop starts
    priority, generator, mode = 0, None, "global"
    if in_thread:
        alive = lambda: True  # noqa: E731
        if device.type == "cuda":
            priority = torch.cuda.Stream.priority_range()[1]
            torch.cuda.set_stream(torch.cuda.Stream(device=device, priority=priority))
            generator = torch.Generator(device=device)
            generator.seed()
            mode = "thread_local"
    else:
        signal.signal(signal.SIGTERM, _exit)
        _pin(cfg, "server")
        torch.set_num_threads(2)
        alive = lambda: os.getppid() == cfg["learner_pid"]  # noqa: E731
    fx.set_inputs(cfg.get("inputs", 1), bool(cfg.get("privileged")))  # (the rows' width, the snapshots' inputs: model.load)
    fx.set_masks(cfg.get("tech_mask", 0), cfg.get("valid_mask", 0))
    fx.set_neutral(cfg.get("neutral", 0))
    nets = Nets(cfg, device)
    while nets.current is None and not stop.is_set() and alive():
        time.sleep(1)
        nets.reload()
    top = cfg["infer_batch"]
    shared = SharedRows(*cfg["shared_rows"]) if cfg.get("shared_rows") else None
    fuse = bool(cfg.get("infer_fuse", 0))
    infer = Inference(nets, device, top, cfg["compile"], buckets=tuple(b for b in (8, 24, 64) if b <= top) or (top,),
                      thread=False, entities=tuple(cfg.get("infer_entities", (48, 96))), shared=shared, fuse=fuse,
                      generator=generator, capture_mode=mode)
    past = Inference(nets, device, 16, cfg["compile"], buckets=(4, 16), thread=False,
                     entities=tuple(cfg.get("infer_past_entities", (64,))),
                     shared=shared, fuse=fuse, generator=generator, capture_mode=mode)  # past snapshots: few calls
    swap = PastNet(device, slots=cfg.get("infer_past_slots", 1))
    streams = {} if cfg.get("infer_streams", 0) else None
    if in_thread and device.type == "cuda" and cfg.get("compile", True):
        t0 = time.time()
        league = Path(cfg["run_dir"]) / "league.json"
        keys = [m["path"] for m in json.loads(league.read_text()).get("members", []) if m.get("path")] if league.exists() else []
        warm_inference(nets, infer, past, swap, keys + list(cfg.get("eval_checkpoints") or []))
        print(f"inference thread: graphs captured in {time.time() - t0:.0f} s", flush=True)
    def fresh():
        return {"calls": 0, "past": 0, "rows": 0, "busy": 0.0, "wait": 0.0, "idle": 0.0, "ents": 0, "ents_max": 0,
                "rounds": 0, "t": time.time()}
    stats = fresh()  # -> inference.jsonl
    stats_path = Path(cfg["run_dir"]) / "inference.jsonl"
    live = list(conns)
    t_round = 0.0
    # Rounds in a pipeline (--infer-pipeline): the next round is read and launched while the last one
    # is on the GPU (its host buffers another bank), then the last one is answered. A round was ~4 ms
    # of reading, launching and answering and ~4.5 ms of waiting for the GPU (behind the learner's
    # kernels): one after the other the games' requests waited for both.
    pipeline = bool(cfg.get("infer_pipeline", 0))
    pending = None  # the round in flight: (asked, calls, events, its start)
    bank = 0

    def complete(p) -> None:
        asked, calls, events, t0 = p
        t_wait = time.time()
        for ev in events:
            ev.synchronize()
        stats["wait"] += time.time() - t_wait
        for fwd, handle, part, version in calls:
            rows = [st.get("row") for _, _, st in part]
            try:
                if handle is None:
                    results = [None] * len(part)
                elif rows[0] is not None:  # the answers in the shared memory: (version, O, memory width) each
                    results = fwd._finish_shared(handle, rows, version)
                else:
                    results = fwd._finish(handle)
                    for res in results:
                        res["version"] = version
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                results = [None] * len(part)
            for (c, k, _), res in zip(part, results):
                asked[c][1][k] = res
        for c, answer in asked.items():
            try:
                c.send_bytes(pack_reply(answer[0], answer[1])) if shared is not None else c.send(answer)
            except OSError:
                if c in live:
                    live.remove(c)
        stats["rounds"] += 1
        stats["busy"] += time.time() - t0

    while not stop.is_set() and alive():
        if time.time() - stats["t"] >= 10.0 and stats["calls"]:
            dt = time.time() - stats["t"]
            with open(stats_path, "a") as f:
                f.write(json.dumps({"time": time.time(), "calls_per_s": round(stats["calls"] / dt, 1),
                                    "rows_per_s": round(stats["rows"] / dt, 1), "rows_per_call": round(stats["rows"] / stats["calls"], 2),
                                    "rounds_per_s": round(stats["rounds"] / dt, 1),
                                    "past_calls_per_s": round(stats["past"] / dt, 1),
                                    "round_ms": round(1000 * stats["busy"] / max(stats["rounds"], 1), 2),  # requests in to answers out
                                    "wait_ms": round(1000 * stats["wait"] / max(stats["rounds"], 1), 2),  # of it: for the GPU
                                    "busy": round(1 - stats["idle"] / dt, 3), "entities": round(stats["ents"] / stats["rows"], 1),
                                    "entities_max_mean": round(stats["ents_max"] / stats["calls"], 1)}) + "\n")
            stats = fresh()
        t_idle = time.time()
        ready = wait(live, timeout=0 if pending else 1.0)
        if pending is None:
            stats["idle"] += time.time() - t_idle
        if not ready:
            if pending:
                complete(pending)
                pending = None
            continue
        # a round at most every cfg["infer_period"] seconds: every round takes the GPU from the learner
        # for a turn (260 rounds a second left it half of it), and what arrives meanwhile shares the round
        pause = t_round + cfg.get("infer_period", 0.0) - time.time()
        if pause > 0:  # (the round in flight answered meanwhile: its games' requests may join this one)
            if pending:
                complete(pending)
                pending = None
                pause = t_round + cfg.get("infer_period", 0.0) - time.time()
            if pause > 0:
                time.sleep(pause)
            ready = wait(live, timeout=0)
        t_round = time.time()
        asked: dict = {}  # game pipe -> (its request's id, the answers)
        groups: dict[str, list] = {}  # network -> [(pipe, item, view step)]
        for c in ready:
            try:
                rid, items = unpack_request(c.recv_bytes()) if shared is not None else c.recv()
            except (EOFError, OSError):  # its actor is gone
                live.remove(c)
                continue
            asked[c] = (rid, [None] * len(items))
            for k, item in enumerate(items):
                if len(item) == 4:  # (key, entities, own units, its row in the shared memory)
                    key, st = item[0], {"n": item[1], "n_own": item[2], "row": item[3]}
                else:
                    key, st = item
                groups.setdefault(key, []).append((c, k, st))
        infer.begin(bank)
        past.begin(bank)
        calls, used = launch_round(groups, nets, infer, past, swap, top, streams, priority)  # all started before the one wait
        events = []
        if device.type == "cuda":  # (recorded now: the next round's work on these streams comes after them)
            for st in used or [torch.cuda.current_stream()]:
                ev = torch.cuda.Event(blocking=True)
                ev.record(st)
                events.append(ev)
        for fwd, _, part, _ in calls:
            ns = [it[2]["n"] for it in part]
            stats["calls"] += 1
            stats["past"] += fwd is past
            stats["rows"] += len(part)
            stats["ents"] += sum(ns)
            stats["ents_max"] += max(ns)
        this = (asked, calls, events, t_round)
        if pending:
            complete(pending)
        pending = None
        if pipeline:
            pending, bank = this, bank ^ 1
        else:
            complete(this)
    if pending:
        complete(pending)


class Trajectory:
    """One trained side's steps; complete chunks (GAE with the values it acted with) go to the learner.
    `hold` (GRPO, a branch of a game): every step is kept until release() adds the group's advantage."""

    def __init__(self, cfg: dict, out_q, info: dict, hold: bool = False):
        self.cfg, self.out_q, self.info = cfg, out_q, info
        self.steps: list[dict] = []
        self.hold, self.bonus = hold, 0.0

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
        if self.cfg.get("unit_credit", 0.0) > 0:  # (--unit-credit: each own unit's id, for its local reward)
            self.steps[-1]["ids"] = st["sel"][:self.steps[-1]["n_own"], fx.C_ID].astype(np.int64)
        if "pent" in st:  # (--opsd-coef: the privileged view of the step, the self-distillation's teacher's)
            self.steps[-1].update(pent=st["pent"].astype(np.float16), ptype=st["ptype"].astype(np.int16),
                                  pcur=st["pcur"].astype(np.int16), pn=st["pn"], pglob=st["pglob"])
        if self.cfg.get("reward_parts"):  # (--reward-parts: the reward by part, PART_NAMES; "reward" their sum)
            self.steps[-1]["rparts"] = np.zeros(len(PART_NAMES), np.float32)
        if len(self.steps) > self.cfg["chunk"] and not self.hold:
            self.flush(bootstrap=True)

    def end(self, reward: float, parts=None) -> None:
        if self.steps:
            self.steps[-1]["reward"] = reward
            if parts is not None and "rparts" in self.steps[-1]:
                self.steps[-1]["rparts"] = np.asarray(parts, np.float32)
            self.steps[-1]["done"] = True
            if not self.hold:
                self.flush(bootstrap=False)

    def release(self, bonus: float) -> None:
        """GRPO: a held game's steps go to the learner, each advantage plus `bonus` (the game's return
        less its group's mean: the branches of one state that differ only by the actions sampled)."""
        self.bonus = bonus
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
            s["adv"], s["ret"] = adv + self.bonus, adv + s["value"]
            next_v = s["value"]
        if self.cfg.get("unit_credit", 0.0) > 0:  # each own unit's short return of its own (unit_rewards), by id
            gl, ahead = self.cfg.get("unit_gamma", 0.8), {}
            for s in reversed(steps):
                ids = s.get("ids")
                if ids is None:
                    continue
                if s["done"]:
                    ahead = {}
                loc = s.get("local")
                g = np.array([(loc[k] if loc is not None else 0.0) + gl * ahead.get(int(u), 0.0)
                              for k, u in enumerate(ids.tolist())], np.float32)
                ahead = dict(zip(ids.tolist(), g.tolist()))
                s["ladv"] = g - g.mean() if len(g) else g  # (relative to the step's other units: the team's
                # credit stays the step's advantage)
        if self.cfg.get("upgo", 0.0) > 0:  # UPGO (AlphaStar): the return that follows the trajectory while the
            # actions after did at least as well as the value expected (Q >= V), else bootstraps from the value
            boot = self.steps[-1]["value"] if bootstrap else 0.0
            g, v1, q1 = boot, boot, -np.inf  # (the next step's UPGO return, value and action's value; the
            for s in reversed(steps):            # chunk's end bootstraps)
                G = s["reward"] if s["done"] else s["reward"] + gamma * (g if q1 >= v1 else v1)
                s["uadv"] = G - s["value"]
                q1 = s["reward"] + (0.0 if s["done"] else gamma * v1)
                g, v1 = G, s["value"]
        for i in range(0, len(steps), self.cfg["chunk"]):  # (a held game's: in pieces)
            self.out_q.put({"steps": steps[i:i + self.cfg["chunk"]], "info": self.info})
        self.steps = self.steps[-1:] if bootstrap else []


def merge_labels(step: dict, y: dict, teach: bool = False) -> None:
    """A teacher's labels (encoder step_labels: y_order, y_ptr, y_x, y_y) into a recorded step, only the units it
    ordered (y_only); another teacher's labels of the step stay where this one has none. `teach` (the heuristic
    teacher's, --teach): those units marked ("y_teach": their own loss weight, --teach-coef)."""
    if not (y["y_order"] > 0).any():
        return
    n = step["n_own"]
    if teach:
        step["y_teach"] = (step.get("y_teach", np.zeros(n, bool)) | (y["y_order"][:n] > 0))
    if "y_order" in step:
        on = y["y_order"][:n] > 0
        for k in ("y_order", "y_ptr", "y_x", "y_y"):
            step[k] = np.where(on, y[k][:n], step[k])
    else:
        for k, v in y.items():
            step[k] = v[:n]
    step["y_only"] = True


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
    and 6 for orc). So is the run's name: forks (all from update 0) drew the same evaluation games
    in the same order, and the current policy got the same matchups in every fork."""
    return random.Random(f"{cfg['seed']}/{Path(cfg.get('run_dir', '')).name}/{cfg.get('start_update', 0)}/{wid}/{k}")


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


_BRANCHES: list[dict] = []  # GRPO (--grpo-share): this actor's open groups (its game slots are threads)
_BRANCH_LOCK = threading.Lock()


def run_branch(wid: int, k: int, cfg: dict, vocab: dict, infer: Inference, out_q, name: str) -> bool:
    """GRPO: one branch of an open group of this actor's, if there is one (a slot's game between agents,
    replayed in a fresh launch to its branch step and played on with new samples). When the group's last
    branch ends its held steps go to the learner, each game's advantages plus its return from the branch
    less the group's mean (per trained side). -> whether it played one."""
    from ..runtime.instance import GameInstance
    with _BRANCH_LOCK:
        job = next((j for j in _BRANCHES if j["left"] > 0), None)
        if job is None:
            return False
        job["left"] -= 1
    got = None
    try:
        with GameInstance(job["setup"], name=name, timeout=120) as g:
            obs = g.start()
            grpo = {"replay": job["record"]}
            play_one(g, obs, cfg, vocab, job["races"], job["side"], copy.deepcopy(job["opp"]), infer, out_q, wid,
                     grpo=grpo)
            got = (grpo["trajs"], grpo["ret"], bool(grpo.get("same")))
    except Exception as e:  # noqa: BLE001 (the group goes on without it)
        print(f"actor {wid}/{k}: branch: {type(e).__name__}: {e}", flush=True)
    with _BRANCH_LOCK:
        job["done"].append(got)
        last = len(job["done"]) == job["n"]
        if last:
            _BRANCHES.remove(job)
    if last:
        release_group([d for d in job["done"] if d is not None], cfg.get("grpo_coef", 1.0))
    return True


def state_print(obs) -> int:
    """The players' state as a number (each of their units' type, owner, position and hit points): a branch's
    replay must reach its source's. (Not the neutral units': the critters wander apart from step ~10.)"""
    return hash(tuple(sorted((u.type_id, u.owner, u.x, u.y, u.hp) for u in obs.units if u.owner in (0, 1))))


def release_group(games: list[tuple[dict, dict, bool]], coef: float) -> None:
    """GRPO: each game's held trajectories (trained side -> Trajectory, its return from the branch, whether
    its replay reached the source's state) go to the learner, every advantage plus `coef` times the return
    less the mean of the group's (that side's) exact branches. A branch that is not exact gets nothing
    added: its game was another (fights with creeps do not replay alike, engine notes), and needs two."""
    for s in {s for trajs, _, _ in games for s in trajs}:
        rs = [(trajs[s], ret[s], same) for trajs, ret, same in games if s in trajs]
        exact = [r for _, r, same in rs if same]
        mean = sum(exact) / len(exact) if exact else 0.0
        print(f"grpo group: side {s}, returns {' '.join(f'{r:+.2f}' + ('' if same else '?') for _, r, same in rs)}, "
              f"steps {' '.join(str(len(tr.steps)) for tr, _, _ in rs)}", flush=True)
        for tr, r, same in rs:
            tr.release(coef * (r - mean) if same and len(exact) > 1 else 0.0)


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
    evaluating = k >= cfg["games_per_actor"]  # an evaluation slot (--eval-games-per-actor)
    if evaluating and cfg.get("eval_map"):  # (--eval-map: its games on another map, with its costs and strategies)
        cfg = {**cfg, "map": cfg["eval_map"], "z_pool": cfg.get("eval_z_pool") or cfg.get("z_pool"),
               "costs_array": np.asarray(cfg["eval_costs"], np.int64) if cfg.get("eval_costs") else None}
    paired: list[tuple] = []  # an evaluation slot's next launches: one matchup for every policy in turn
    mirrored: list[tuple] = []  # (--eval-vs) the next launch against a checkpoint: the last one's races swapped
    alive = Path(cfg["run_dir"]) / "remote-eval.alive"  # (scripts/remote_eval.py: its games still arrive)
    while not stop.is_set() and os.getppid() == cfg["learner_pid"]:
        if evaluating and cfg.get("eval_fallback", 0) > 0:  # --eval-fallback: another machine plays them while
            try:                                            # its games keep coming; these only when they stop
                if time.time() - alive.stat().st_mtime < cfg["eval_fallback"]:
                    time.sleep(15)
                    continue
            except FileNotFoundError:
                pass
        if not evaluating and cfg.get("grpo_share", 0.0) > 0 and run_branch(wid, k, cfg, vocab, infer, out_q, name):
            continue  # (a branch of another slot's game first: its group waits for it)
        try:
            spec = json.loads(spec_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)
            continue
        races = [rng.choice(cfg["races"]), rng.choice(cfg["races"])]
        if cfg["mirror"]:  # both sides the same race (balanced by construction)
            races[1] = races[0]
        side = rng.randrange(2)  # the learner's
        if cfg.get("learner_races") and not evaluating:  # (--learner-races: the learner's race weighted, training only)
            names, weights = zip(*cfg["learner_races"].items())
            races[side] = rng.choices(names, weights)[0]
            if cfg["mirror"]:
                races[1 - side] = races[side]
        ai = _choose(rng, spec["launch"])  # {"kind": "agents"} or {"kind": "ai", "difficulty"[, "delay"]}
        versus = None  # (--eval-vs: an evaluation launch of the current policy against a listed checkpoint)
        if mirrored or (evaluating and cfg.get("eval_vs", 0.0) > 0 and cfg.get("eval_checkpoints")
                        and rng.random() < cfg["eval_vs"]):
            # each matchup both ways round (the current policy with either race; drawn alone, the races decided:
            # 0 of 19 as human against the checkpoint's orc read as a weaker policy)
            if mirrored:
                races, side, versus = mirrored.pop()
            else:
                versus = rng.choice(cfg["eval_checkpoints"])
                mirrored.append(([races[1], races[0]], side, versus))
            ai = {"kind": "agents"}
        elif evaluating:  # (--eval-games-per-actor: the yardstick only)
            # which policy: the current one, or one of --eval-checkpoints (head to head, at the same time).
            # Each matchup (races, sides, the AI's difficulty) is played by every policy in turn: drawn
            # apart, the policies' matchups differed (the AI's mean record in them 0.45 against 0.50),
            # and the AI-record share depends on the mix.
            if not paired:
                keys = ["current"] + (["current+z"] if cfg.get("eval_z") else []) + list(cfg.get("eval_checkpoints") or [])
                if cfg.get("eval_z") and cfg.get("eval_guide", 0.0) > 0 and cfg.get("guide_capable"):
                    keys.insert(2, "current+zg")  # (--eval-guide: its target followed with guidance; the worker's only)
                rng.shuffle(keys)
                diff = rng.choice(cfg.get("eval_ai") or ["normal", "insane"])
                paired = [(list(races), side, diff, key) for key in keys]
            races, side, diff, eval_key = paired.pop(0)
            ai = {"kind": "ai", "difficulty": diff, "real": True}
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
        # GRPO's sources: games between agents with a seed (the built-in AI's games do not replay)
        seeded = agents_only and not evaluating and cfg.get("grpo_share", 0.0) > 0
        # a restart goes on to the next pair of players (--pairs), or reloads the map (1 s).
        # (--scripted-reset: games without the built-in AI reset by script for the same players instead,
        # 0.1 s, but not to a new game: see GameSetup.melee_reset)
        setup = GameSetup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"],
                          max_game_seconds=cfg["max_minutes"] * 60, victory="decisive", window=(320, 240),
                          record_ai_orders=cfg.get("opd_share", 0.0) > 0,
                          draw=False,  # (nobody looks: videos are played back from the replays, drawing)
                          wait_floor_ms=cfg["wait_floor_ms"], melee_reset=agents_only and cfg["scripted_reset"],
                          native_obs=cfg["native_obs"], nice=cfg.get("game_nice", 0), d3d_thread=False,
                          render_threads=0, pairs=cfg.get("pairs", 1), cuts=cfg.get("game_cuts") or None,
                          seed=rng.randint(1, cfg.get("grpo_seeds", 8)) if seeded else 0)
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
                           else {"name": f"eval-vs:{versus}", "kind": "past", "path": versus,
                                 "eval_tag": "vs " + "/".join(Path(versus).parts[-3::2])} if versus
                           else dict(_choose(rng, spec["agents"])))
                    if agents_only and not versus:  # races apart: the stronger one's income taxed (League.balance)
                        taxed, share = balance_tax(spec, races)
                        if share > 0:
                            opp["tax"], opp["tax_side"] = share, taxed
                    # on-policy distillation: the built-in AI advises the learner (never in the real game)
                    opp["advise"] = not real and rng.random() < cfg.get("opd_share", 0.0)
                    if evaluating and not versus and eval_key != "current":
                        opp["eval_key"] = eval_key
                    if "shaping" in spec:  # (--shaping-anneal: the weight now, the game's for its whole length)
                        opp["shaping"] = spec["shaping"]
                    if curr:  # the curriculum's current knobs for this AI
                        now = next((x.get("by_race", {}).get(key, ai) for x in spec["launch"]
                                    if x.get("difficulty") == ai["difficulty"] and not x.get("real")), ai)
                        opp["curriculum"] = now.get("level", 0.0)
                        if late:
                            opp["start"] = int(round(now.get("delay", 0.0) / cfg["step_seconds"]))
                        if now.get("tax", 0.0) != 0:  # (< 0: a bonus, --curriculum-floor)
                            opp["tax"] = now["tax"]
                    # a video: the first game of a launch (its replay then holds just this game)
                    film = films and fresh and time.time() >= next_video
                    # GRPO: the first game of a launch (a fresh start, as the branches' launches) may be a source
                    grpo = ({"source": int(rng.uniform(*cfg["grpo_branch"]) / cfg["step_seconds"])}
                            if seeded and n == 0 and not film and rng.random() < cfg["grpo_share"] else None)
                    src_opp = copy.deepcopy(opp) if grpo else None
                    ep = play_one(g, obs, cfg, vocab, races, side, opp, infer, out_q, wid, record=film,
                                  train=not evaluating, grpo=grpo)
                    if grpo and grpo.get("record"):  # (it reached its branch step: a group of branches from there)
                        with _BRANCH_LOCK:
                            _BRANCHES.append({"setup": setup, "record": grpo["record"], "races": list(races),
                                              "side": side, "opp": src_opp, "left": cfg["grpo_group"],
                                              "n": cfg["grpo_group"], "done": []})
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


_POOLS: dict[str, dict] = {}


def strategy_pool(cfg: dict) -> dict | None:
    """The run's strategies (--z-pool: fullgame/strategy.py), read once per process."""
    path = cfg.get("z_pool")
    if not path:
        return None
    if path not in _POOLS:
        pool = json.loads(Path(path).read_text())
        if fx.NEUTRAL == 2:  # (--neutral 2: no mercenaries to buy, none in the targets)
            from .neutral import MERCENARIES
            pool["strategies"] = [{**s, "t": {c: v for c, v in s["t"].items() if c not in MERCENARIES}}
                                  for s in pool["strategies"]]  # (the items' list stays: it is the inputs' slots)
        _POOLS[path] = pool
    return _POOLS[path]


HALLS = {int.from_bytes(c.encode(), "big") for c in
         ("htow", "hkee", "hcas", "ogre", "ostr", "ofrt", "unpl", "unp1", "unp2", "etol", "etoa", "etoe")}
AT_MINES = {"htow", "ogre", "unpl", "etol", "ugol", "egol"}  # (with a hall already: an expansion)
WORKER_TYPES = {"human": "hpea", "orc": "opeo", "undead": "uaco", "nightelf": "ewsp"}
WORKER_IDS = {int.from_bytes(c.encode(), "big") for c in WORKER_TYPES.values()}
_FOOD: dict[int, int] = {}


def unit_food() -> dict[int, int]:
    """Unit type id -> its food (the game's tables; --z-modes' army size)."""
    if not _FOOD:
        from ..data.objects import unit_table
        _FOOD.update({int.from_bytes(c.encode(), "big"): u.food for c, u in unit_table().items()})
    return _FOOD
SUPPLY_TYPES = {"human": "hhou", "orc": "otrb", "undead": "uzig", "nightelf": "emow"}
CANCEL = 851976  # a building's queue: its last unit


class TargetBuilder:  # (uses protocol.Build and ImmediateOrder, imported below with the other commands)
    """--z-build: for a side playing toward a target strategy, the harness starts what the agent leaves
    undone: the earliest target item due by now (its time in the strategy) that is neither made nor under
    way, or, while the player lacks what it needs (the tech tree: a requirement, or a building that makes
    it), the first of those it can start. A building with the worker nearest the town hall (one outside a
    mine; with none, a worker first); a unit, hero, research or a hall's next tier at a building that makes
    it, an idle one first (else a unit or research queued; the game refuses an upgrade or a hero behind
    others, so for its next tier a hall training workers has their queue cancelled, or, taught, gets the
    upgrade as its label). Out of supply, a farm first
    (a unit ordered then waits in its building's queue, never starting, and blocks a next tier behind it). At most one a step, when the player can pay for it; an order that
    starts nothing is given again after 20 steps, and after three refusals not that game. The game's
    events tell what is under way (the agent's own orders too) and which buildings are busy (a building's
    current order reads 0 while it trains). Expansions and everything else stay the agent's."""

    def __init__(self, cfg: dict, vocab: dict, side: int, st: dict, rng: random.Random):
        from . import techtree
        self.side, self.st, self.rng, self.race = side, st, rng, st["race"]
        self.teach = cfg.get("z_build", 0) == 2  # (its orders labels only: a busy building's decision too)
        self.costs = cfg.get("costs_array")
        self.cls = {(int(o), int(k)): c for c, (o, k) in enumerate(vocab["orders"], start=1)}
        self.tree = techtree.tree()
        self.upgraded = {to for tos in self.tree.upgrades.values() for to in tos}  # (a keep, a guard tower)
        self.tried: dict[str, int] = {}
        self.started: dict[str, float] = {}  # items under way -> since (game time)
        self.busy: dict[int, tuple[str, ...]] = {}  # the side's buildings at work -> what they make
        self.site: dict[int, str] = {}  # its buildings under construction -> their types
        self.log: dict[str, list[int]] = {}  # item -> [orders given, accepted by the game] (the episode row)
        self.last = ""
        self.refused: Counter = Counter()  # its orders the game refused, per item (carried out: --z-build 1)
        self.why: list = []  # the refused orders' context (the episode row's z_refusals: the first 40)
        self.agent_now: set = set()
        self.frames: dict[int, tuple[int, int]] = {}  # its buildings under construction -> (hit points, since)
        self.blocked: dict[str, Counter] = {}  # item -> why it was not ordered, per step (the episode row)
        self.why_not = ""
        self.reserve: tuple[int, int, int] | None = None  # (order class, gold, lumber): saved for (View.reserve)
        self.ctx: list = []

    def observe(self, obs) -> None:
        """The items the side started, finished or cancelled in the step just taken (its events)."""
        from .features import rawcode
        now = obs.game_time
        for c in [c for c, t0 in self.started.items() if now - t0 > 240]:  # (lost track: one destroyed unfinished)
            del self.started[c]
        if not obs.events:
            return
        owner = {u.id: u.owner for u in obs.units}
        for e in obs.events:
            k = int(e.kind)
            if owner.get(e.a) != self.side:
                continue
            if k == 2:  # construction: a=structure, b=type
                self.site[e.a] = rawcode(e.b)
                self.started[rawcode(e.b)] = now
            elif k in (5, 8, 11):  # training, research, upgrade (b: the type upgraded from)
                made = self.tree.upgrades.get(rawcode(e.b), ()) if k == 11 else (rawcode(e.b),)
                self.busy[e.a] = made
                self.started.update(dict.fromkeys(made, now))
            elif k in (3, 4) or (k == 1 and e.a in self.site):  # finished, cancelled or destroyed
                self.started.pop(self.site.pop(e.a, ""), None)
            elif k in (6, 7, 9, 10, 12, 13) or k == 1:
                for c in self.busy.pop(e.a, ()):
                    self.started.pop(c, None)

    def commands(self, obs, made: dict, t: int, step_s: float, ordered=()) -> list:
        """`ordered`: the agent's commands this step (the game refuses the same hero or next tier twice: the
        builder's order is then no refusal of its own)."""
        from . import techtree
        from ..protocol import UnitFlags
        self.observe(obs)
        self.agent_now = {fx.rawcode(o) for x in ordered
                          if (o := getattr(x, "building", None) or getattr(x, "item", None) or getattr(x, "order", 0)) >= fx.TYPE_CODE}
        p = obs.players.get(self.side)
        if p is None:
            return []
        if (cmd := self.resume(obs, t)) is not None:
            self.last = "resume"
            self.log.setdefault("resume", [0, 0])[0] += 1
            return [cmd]
        now = obs.game_time
        due = sorted((t0, c) for c, t0 in self.st["t"].items() if t0 <= now and c not in made and c not in self.started)
        food = SUPPLY_TYPES.get(self.race)  # out of supply: a unit ordered waits in its building's queue and
        if p.food_cap < 100 and p.food_used + 2 > p.food_cap and food not in self.started:  # blocks it
            due.insert(0, (0.0, food))  # (a next tier is refused behind it): supply first
        if not due:
            self.reserve = None
            return []
        mine = [u for u in obs.units if u.owner == self.side and u.alive]
        done = [u for u in mine if u.is_structure and not u.flags & UnitFlags.CONSTRUCTING]
        have = techtree.counts_as({u.type for u in done} | {c for c in made if c[0] == "R"}, self.race)
        heroes = len({c for c in (*made, *self.started) if techtree.is_hero(c)})
        save = None
        for _, c in due:
            item = self.next_step(c, have, {u.type for u in done}, heroes, t)
            if item is None:
                continue
            self.why_not = ""
            cmds = self.order(item, obs, mine, done, p, t)
            if not cmds and self.why_not:
                self.blocked.setdefault(item, Counter())[self.why_not] += 1
                if self.why_not in ("gold", "lumber") and save is None:  # (the first one short of money)
                    k = self._class(item)
                    save = (k, int(self.costs[k][0]), int(self.costs[k][1]))
            if cmds:  # (`item`, or a worker to build it; the order last)
                got = self._name(cmds[-1])
                self.tried[got], self.last = t, got
                self.log.setdefault(got, [0, 0])[0] += 1
                self.reserve = save
                return cmds
        self.reserve = save
        return []

    def resume(self, obs, t: int):
        """A human's or orc's building left unfinished (its worker taken away: no hit points gained for 20
        steps; the game keeps the frame, so the item stays under way): the nearest free worker resumes it."""
        from ..protocol import TargetOrder, UnitFlags
        from .play import REPAIR
        if self.race not in ("human", "orc"):
            return None
        frames = {u.id: u for u in obs.units if u.owner == self.side and u.alive and u.flags & UnitFlags.CONSTRUCTING}
        self.frames = {i: (u.hp, t if u.hp > self.frames.get(i, (-1, t))[0] else self.frames[i][1])
                       for i, u in frames.items()}
        stalled = [frames[i] for i, (_, since) in self.frames.items() if t - since >= 20]
        workers = [u for u in obs.units if u.owner == self.side and u.alive and u.is_worker and not u.is_structure
                   and not fx.building(int(u.flags), u.order) and u.order != REPAIR
                   and not u.flags & (UnitFlags.HIDDEN | UnitFlags.LOADED)]
        if not stalled or not workers:
            return None
        f = stalled[0]
        w = min(workers, key=lambda u: (u.x - f.x) ** 2 + (u.y - f.y) ** 2)
        self.frames[f.id] = (f.hp, t)  # (another try in 20 steps if it still gains nothing)
        return TargetOrder(w.id, REPAIR, f.id)

    def next_step(self, c: str, have: set, types: set, heroes: int, t: int, depth: int = 0) -> str | None:
        """The item to start toward `c` now: `c` when the player has what it needs (`have`: techtree.counts_as;
        `types`: its finished buildings, one of which must make `c`), else a missing need's own next step;
        None while every missing need is under way or was tried in the last 20 steps, or none can be made."""
        from . import techtree
        if c in self.started or t - self.tried.get(c, -99) < 20 or self.refused[c] >= 3:
            return None  # (under way, just tried, or refused three times: tech or a spot the game won't take)
        if c in AT_MINES and types & {h for tiers in techtree.TIERS.values() for h in tiers}:
            return None  # an expansion: the agent's (its gold mine is guarded by creeps)
        miss = techtree.missing(c, self.race, have, heroes if techtree.is_hero(c) else 0, self.tree)
        makers = self.tree.makers.get(c)
        if makers and not types & set(makers) and (int.from_bytes(c.encode(), "big"), fx.POINT) not in self.cls:
            miss.append(makers[0])  # (nothing that makes it: the first type that does)
        if not miss:
            return c
        if depth < 4:
            for m in miss:
                if (n := self.next_step(m, have, types, heroes, t, depth + 1)) is not None:
                    return n
        return None

    def order(self, c: str, obs, mine: list, done: list, p, t: int = 0) -> list | None:
        """The commands that start `c` (a Build or an ImmediateOrder), if the side can pay for it now; a
        building without a free worker (all in a mine: a night elf's five wisps in theirs): a worker first.
        A next tier at a hall training workers: carried out, the workers' queue cancelled first (the game
        takes no upgrade behind them, and a hall trains them back to back); taught, the upgrade alone (the
        hall's decision to learn)."""
        from . import techtree
        from ..protocol import Build, Buy, ImmediateOrder
        code = int.from_bytes(c.encode(), "big")
        if (k := self.cls.get((code, fx.UNIT))) is not None:  # (--neutral: a tavern hero, a mercenary: bought by the
            from .neutral import MERCENARIES  # worker nearest its building, the tavern or a camp)
            if c in MERCENARIES and fx.NEUTRAL == 2:  # (--neutral 2: none to buy)
                self.why_not = "off"
                return None
            kind = "nmer" if c in MERCENARIES else "ntav"
            shops = [u for u in obs.units if u.type == kind and u.alive]
            workers = [u for u in mine if u.is_worker and not u.is_structure and not fx.building(int(u.flags), u.order)]
            if not shops or not workers:
                self.why_not = "no shop" if not shops else "no worker"
                return None
            if not self._affords(k, p):
                return None
            hall = next((u for u in done if u.type_id in HALLS), workers[0])
            shop = min(shops, key=lambda u: (u.x - hall.x) ** 2 + (u.y - hall.y) ** 2)
            w = min(workers, key=lambda u: (u.x - shop.x) ** 2 + (u.y - shop.y) ** 2)
            return [Buy(w.id, shop.id, code)]
        if (k := self.cls.get((code, fx.POINT))) is not None:  # a building: the worker nearest the hall
            from ..protocol import UnitFlags
            hall = next((u for u in done if u.type_id in HALLS), None)
            workers = [u for u in mine if u.is_worker and not u.is_structure and not fx.building(int(u.flags), u.order)
                       and not u.flags & (UnitFlags.HIDDEN | UnitFlags.LOADED)]  # (not in a mine: refused)
            if not workers:
                self.why_not = "no worker"
                w = WORKER_TYPES.get(self.race)
                return self.order(w, obs, mine, done, p, t) if w and t - self.tried.get(w, -99) >= 20 else None
            if not self._affords(k, p):
                return None
            at = hall or workers[0]
            w = min(workers, key=lambda u: (u.x - at.x) ** 2 + (u.y - at.y) ** 2)
            a, r = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(450, 800)
            return [Build(w.id, code, at.x + r * math.cos(a), at.y + r * math.sin(a))]
        if (k := self.cls.get((code, fx.IMMEDIATE))) is not None:  # trained, researched or upgraded to
            makers = set(self.tree.makers.get(c, ()))
            at = [u for u in done if u.type in makers]  # an idle one, else queued (a barracks training units);
            queue = self.teach or (not techtree.is_hero(c) and c not in self.upgraded)  # (the game takes
            b = next((u for u in at if u.id not in self.busy), at[0] if at and queue else None)  # neither behind)
            cancel = []
            if b is None and c in self.upgraded:  # (a hall training workers: their queue cancelled first)
                b = next((u for u in at if set(self.busy[u.id]) <= set(WORKER_TYPES.values())), None)
                cancel = [ImmediateOrder(b.id, CANCEL)] * 6 if b is not None else []
            if b is not None:  # (a refusal's context: item, gold, lumber, food, cap, maker, busy with)
                self.ctx = [c, p.gold, p.lumber, p.food_used, p.food_cap, b.type, ",".join(self.busy.get(b.id, ()))]
            if b is not None and self._affords(k, p):
                return cancel + [ImmediateOrder(b.id, code)]
            self.why_not = self.why_not or ("busy" if at else "no maker")
        return None

    def _class(self, c: str) -> int:
        code = int.from_bytes(c.encode(), "big")
        return (self.cls.get((code, fx.POINT)) or self.cls.get((code, fx.IMMEDIATE))
                or self.cls.get((code, fx.UNIT)) or 0)  # (--neutral: a tavern hero, a mercenary: bought)

    @staticmethod
    def _name(cmd) -> str:
        o = cmd.building if hasattr(cmd, "building") else cmd.item if hasattr(cmd, "item") else cmd.order  # (a Buy: its ware)
        return fx.rawcode(o) if o >= fx.TYPE_CODE else str(o)

    def _affords(self, k: int, p) -> bool:
        if self.costs is None:
            return True
        g, l, f = (int(v) for v in self.costs[k])
        self.why_not = "gold" if p.gold < g else "lumber" if p.lumber < l else "food" if f and p.food_used + f > p.food_cap else ""
        return not self.why_not


def target(cfg: dict, view, st: dict, pool: dict, race: str, vocab: dict, modes: dict | None = None) -> dict:
    """A side's target strategy `st` in its view: the input (strategy.vector; with --z-modes, its modes'
    after it: modes.vector) and, with --z-enforce, the orders it allows for the strategies' horizon
    (strategy.allowed) -> `st`."""
    z = strat.vector(st, pool["items"][race])
    view.z = np.concatenate([z, zmodes.vector(modes)]) if fx.ZDIM > strat.ZDIM else z
    if cfg.get("z_enforce"):
        view.zmask = strat.allowed(st, pool["items"][race], vocab["orders"])
        view.zmask_until = int(strat.horizon(st) / cfg["step_seconds"])  # (`st` scaled: --z-time-scale)
    return st


def play_one(g, obs, cfg: dict, vocab: dict, races: list[str], side: int, opp: dict, infer: Inference,
             out_q, wid: int, record: bool = False, train: bool = True, grpo: dict | None = None) -> dict:
    """One game from its first observation; `record`: also a trace for the video's panel
    (fullgame.overlay), in the returned episode row's "trace". `grpo` (--grpo-share, a seeded game between
    agents): {"source": t} records the game's commands up to step t and the memory then (grpo["record"]);
    {"replay": record} replays them to t without asking the policy (the game is deterministic: the same
    state) and plays on from there, its trained sides' steps held (grpo["trajs"], their returns from t in
    grpo["ret"]) for the group's advantage (game_loop)."""
    from .play import BCAgent

    from ..protocol import Buy, SetResources, ShadowAI, StartAI
    replay = grpo.get("replay") if grpo else None
    source = grpo.get("source", 0) if grpo else 0
    rec = {"cmds": [], "spans": [], "res": [], "prints": [], "t": source} if source else None
    same = None  # (a branch: whether its replay reached its source's state)
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
        keys[s] = ("current" if s == side or opp["kind"] == "self" else exploiter_key(cfg) if opp["kind"] == "exploiter"
                   else opp["path"])
        if s == side and opp.get("eval_key"):  # (an evaluation game of a listed checkpoint: --eval-checkpoints)
            keys[s] = "current" if opp["eval_key"] in ("current+z", "current+zg") else opp["eval_key"]
            bot.view.active_f = fx.F  # (all features: a checkpoint's weights past its own are zeros)
        if keys[s] == "current" and train:  # (an evaluation game: no steps to train on)
            trajs[s] = Trajectory(cfg, out_q, {"worker": wid, "opponent": opp["name"]}, hold=replay is not None)
        elif opp["kind"] == "exploiter" and s != side:  # (its steps train the exploiter: League, --exploiter-share)
            trajs[s] = Trajectory(cfg, out_q, {"worker": wid, "opponent": "main", "learner": "exploiter"})
    # strategy conditioning (--z-pool): --z-share of the learner's trained sides play toward a strategy drawn for
    # their race (and their opponent's, when the pool has some): the global vector's strategy block, and a
    # reward for the items made by now, theirs against the strategy's (strategy.mismatch); none in evaluation
    zs: dict[int, dict] = {}
    zm: dict[int, dict] = {}  # (--z-modes: each side's modes, modes.draw)
    pool = strategy_pool(cfg) if train else None
    for s in trajs if pool is not None and replay is None else ():
        if keys[s] == "current" and (random.random() < cfg.get("z_share", 0.5) or (record and s == side)):  # (not the
            # exploiter's sides; a filmed game's learner always: its video shows how it plays a target)
            st = strat.sample(pool, races[s], races[1 - s], random, bool(cfg.get("z_winners", 1)), cfg.get("z_src", ""))
            if st is not None:
                zm[s] = zmodes.draw(random, cfg["z_modes"]) if cfg.get("z_modes") else {}
                zs[s] = target(cfg, bots[s].view, strat.scaled(st, cfg.get("z_time_scale", 1.0)), pool, races[s], vocab,
                               zm[s])
    if opp.get("eval_key") in ("current+z", "current+zg"):  # (--eval-z: the current policy's evaluation games with a target, AlphaStar's
        pool = strategy_pool(cfg)              # z at test time: drawn as in training, winners of the matchup first)
        st = strat.sample(pool, races[side], races[1 - side], random, True, cfg.get("z_src", "")) if pool else None
        if st is not None:
            zm[side] = zmodes.draw(random, cfg["z_modes"]) if cfg.get("z_modes") else {}
            zs[side] = target(cfg, bots[side].view, strat.scaled(st, cfg.get("z_time_scale", 1.0)), pool, races[side],
                              vocab, zm[side])
    if replay is not None:  # (the source's targets)
        zs = {s: target(cfg, bots[s].view, st, pool, races[s], vocab, replay.get("zm", {}).get(s))
              for s, st in replay["zs"].items()}
        zm = dict(replay.get("zm", {}))
    if rec is not None:
        rec["zs"], rec["zm"] = dict(zs), dict(zm)
    # --z-modes: each side's modes measured each step (modes.Tracker: their violations, the strategy potential's)
    trackers = {s: zmodes.Tracker(m, s, (obs.players[1 - s].start_x, obs.players[1 - s].start_y), unit_food(),
                                  WORKER_IDS) for s, m in zm.items() if m and obs.players.get(1 - s) is not None}
    viol = {s: 0.0 for s in trackers}
    rows_next = None  # (the step's unit rows, made for the trackers: the next step's)
    made = {s: {} for s in zs}  # each item's first game second, of the sides with a strategy
    builders = {s: TargetBuilder(cfg, vocab, s, zs[s], random.Random()) for s in zs} if cfg.get("z_build") else {}
    # --scenarios: a share of the training games start in a situation the harness sets up (scenarios.py)
    scenario = (random.choice(cfg["scenarios"]) if train and replay is None and cfg.get("scenarios")
                and random.random() < cfg.get("scenario_share", 0.0) else None)
    scenario_cmds = scenarios.commands(scenario, obs, side, races, random.Random()) if scenario else []
    if cfg.get("privileged") and train:  # (--opsd-*: a teacher game acts on the privileged view; the others give it beside)
        for s in trajs:
            if keys[s] == "current":
                if random.random() < cfg.get("opsd_share", 0.0):
                    bots[s].view.privileged_act = True
                elif cfg.get("opsd_coef", 0.0) > 0:
                    bots[s].view.dual = True
    # --teach: a heuristic teacher's lessons (teacher.py: a tavern hero, a town portal, a potion) label the learner's steps
    teachers = ({s: Teacher(s, races[s], random.Random(), zs.get(s)) for s in trajs if keys[s] == "current"}
                if cfg.get("teach") and replay is None else {})
    zphi = {s: 0.0 for s in zs}  # their strategy potentials at the last recorded step
    # the built-in AI advises the learner's sides (opp["advise"]): its orders label their steps
    advised = [s for s in trajs if keys[s] == "current"] if opp.get("advise") else []
    for s in advised:
        bots[s].repair_builds = True  # (a builder the advisor pulled away resumes with "repair")
    t, t0 = 0, time.time()
    own_rows: dict[int, np.ndarray] = {}  # each trained side's own units at its last recorded step (--unit-credit)
    # --shared-noise: this share of the games sample every policy side's units with one noise vector a step
    # (model._gumbel_argmax: the same marginals, coordinated choices); the episode row says which
    shared_noise = random.random() < cfg.get("shared_noise", 0.0) if replay is None else replay["shared_noise"]
    if rec is not None:
        rec["shared_noise"] = shared_noise
    values, scale = cfg["values"], cfg["shaping_scale"]
    shaping = opp.get("shaping", cfg["shaping"])  # (the potential goes back to 0 with the weight it was paid with)
    mat = material(obs, values, cfg.get("hero_level_value", 0.0), cfg.get("worker_cap", 0))
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
            cmds += scenario_cmds  # (--scenarios: the situation the game starts in)
        held = {s: obs.players.get(s) for s in advised}  # (the labels count what the player could pay for)
        held = {s: np.asarray([t, s, p.gold, p.lumber, p.food_used, p.food_cap, p.upkeep]) for s, p in held.items() if p}
        rows = rows_next if rows_next is not None else unit_rows(obs, t)  # once for both sides
        if cfg.get("unit_credit", 0.0) > 0 and t > 0:  # each own unit's reward of its own for the last step
            for s, tr in trajs.items():
                last = tr.steps[-1] if tr.steps else None
                if last is not None and last.get("ids") is not None and "local" not in last and s in own_rows:
                    last["local"] = unit_rewards(own_rows[s], rows, obs.events, values, scale)
        for s, b in builders.items():  # (--z-build: the agent's orders leave the target's next item's cost)
            if s in bots:
                bots[s].view.reserve = b.reserve
        sts = {s: bot.observe(obs, t, rows) for s, bot in bots.items()}
        replaying = replay is not None and t < replay["t"]
        if replay is not None and t == replay["t"]:  # (the branch: the memory as it was, the strategy's potential)
            same = state_print(obs) == replay["print"]
            for s, h in replay["h"].items():
                bots[s].h = h
            for s in zs:
                zphi[s] = -cfg.get("z_reward", 0.0) * strat.mismatch(made[s], zs[s], pool["items"][races[s]], obs.game_time)
        if rec is not None and t == rec["t"]:
            rec["h"], rec["print"] = {s: bots[s].h for s in bots}, state_print(obs)
        live = [s for s in bots if sts[s] is not None]
        for s in live:
            sts[s]["h"] = bots[s].h
            if shared_noise:
                sts[s]["shared_noise"] = True
            if opp.get("eval_key") == "current+zg" and s == side and s in zs:  # (--eval-guide)
                sts[s]["guide"] = cfg["eval_guide"]
        results = infer.request([(keys[s], sts[s]) for s in live]) if live and not replaying else []
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
                own_rows[s] = sts[s]["sel"][:min(sts[s]["n_own"], fx.MAX_OWN)]  # (its local reward: next step)
                phi[s] = (pot[s] - pot[1 - s]) / scale
        if record:
            chose = {s: Counter(fx.order_label(*bots[s].orders[int(c)], names) for c in r["order"] if c)
                     for s, r in zip(live, results) if r is not None}
            trace.append(trace_step(t, obs, mat, chose, value={s: r["value"] for s, r in zip(live, results) if r},
                                    entropy={s: r["entropy"] for s, r in zip(live, results) if r}))
        for s, tc in teachers.items():  # (--teach: never carried out, the side's step just recorded labelled)
            if s in spans and trajs[s].steps:
                c = tc.commands(rows, obs.players.get(s), obs.game_time, t)
                q = obs.players.get(s)
                if c and q is not None:
                    me = np.asarray([t, s, q.gold, q.lumber, q.food_used, q.food_cap, q.upkeep])
                    lessons = np.asarray([(t, x.unit, x.item, 2, 0, 0, x.shop) if isinstance(x, Buy)
                                          else (t, x.unit, x.order, 0, 0, 0, 0) for x in c], np.int64).reshape(-1, 7)
                    y, _ = bots[s].enc.step_labels(bots[s].view, sts[s], lessons, me, np.zeros((0, 5), np.int64),
                                                   bots[s].trees)
                    merge_labels(trajs[s].steps[-1], y, teach=True)
        built, given = {}, {}
        for s, b in builders.items():  # (--z-build: after the agents' orders, so a worker it takes is taken)
            c = b.commands(obs, made[s], t, cfg["step_seconds"], cmds[spans[s][0]:spans[s][1]] if s in spans else ())
            if c and cfg.get("z_build", 0) != 2:
                given[s] = len(cmds) + len(c) - 1  # (its order: last, after a queue's cancels)
            if c and cfg.get("z_build_labels") and s in trajs and s in spans and obs.players.get(s) is not None:
                q = obs.players[s]
                built[s] = (len(cmds) if cfg.get("z_build", 0) != 2 else None, c,
                            np.asarray([t, s, q.gold, q.lumber, q.food_used, q.food_cap, q.upkeep]))
            if cfg.get("z_build", 0) != 2:  # (--z-build 2: a teacher only, its orders labels and never carried out)
                cmds += c
        if late is not None and t == opp["start"]:  # the built-in AI takes over its side (after the bots' orders)
            cmds.append(StartAI(late))
        if replaying:  # (the source's commands, which its spans index; what they send workers to harvest)
            if same is None and state_print(obs) != replay["prints"][t]:  # (the first step it differs: a log line)
                same = False
                print(f"actor {wid}: grpo replay differs from step {t} (branch {replay['t']}): {len(obs.units)} units; "
                      f"the step before's commands {[type(c).__name__ for c in replay['cmds'][t - 1]] if t else []}", flush=True)
            cmds, spans, given, built = replay["cmds"][t], replay["spans"][t], {}, {}
            for s_, res_ in replay["res"][t].items():
                bots[s_].sent_resources = res_
        elif rec is not None and t < rec["t"]:
            rec["cmds"].append(list(cmds))
            rec["prints"].append(state_print(obs))
            rec["spans"].append(dict(spans))
            rec["res"].append({s_: list(bots[s_].sent_resources) for s_ in spans})
        obs = g.step(cmds)
        for s, at in given.items():  # (the builder's log: its order accepted?)
            if obs.command_results[at]:
                builders[s].log[builders[s].last][1] += 1
            elif builders[s].last not in builders[s].agent_now:  # (not a duplicate of the agent's own)
                builders[s].refused[builders[s].last] += 1
                if len(builders[s].why) < 40 and builders[s].ctx[:1] == [builders[s].last]:
                    builders[s].why.append([round(obs.game_time)] + builders[s].ctx)
        for s, (at, c, me) in built.items():  # (--z-build-labels: the builder's accepted orders label the side's step)
            ok = obs.command_results[at:at + len(c)] if at is not None else [True] * len(c)
            rows = np.asarray([(t, x.unit, x.item, 2, 0, 0, x.shop) if isinstance(x, Buy) else  # (a hire: on its building)
                               (t, x.unit, getattr(x, "building", getattr(x, "order", 0)),
                                1 if hasattr(x, "x") else 2 if hasattr(x, "target") else 0,
                                int(getattr(x, "x", 0)), int(getattr(x, "y", 0)), getattr(x, "target", 0))
                               for x, a in zip(c, ok) if a], np.int64).reshape(-1, 7)
            if len(rows) and trajs[s].steps:
                later = np.asarray([(t + 1, int(e.kind), e.a, e.b, e.c) for e in obs.events], np.int64).reshape(-1, 5)
                y, _ = bots[s].enc.step_labels(bots[s].view, sts[s], rows, me, later, bots[s].trees)
                merge_labels(trajs[s].steps[-1], y)
        if advised:  # the advisor's orders of the step just taken: the labels of the step recorded
            label_step(obs, t, advised, spans, sts, held, bots, trajs)
        for s, (a, b) in spans.items():
            bots[s].accepted(cmds[a:b], obs.command_results[a:b])
        t += 1
        prod.step(obs)
        for s in zs:  # (what each side with a strategy has made: its items' first times)
            for code in (*prod.trained[s], *prod.built[s], *prod.research[s]):
                made[s].setdefault(code, obs.game_time)
        rows_next = unit_rows(obs, t) if trackers else None
        for s, tk in trackers.items():  # (--z-modes: evaluation games too, their rows' report)
            p_ = obs.players.get(s)
            viol[s] = tk.step(rows_next, obs.events, p_.food_used if p_ is not None else 0, obs.game_time)
        mat = material(obs, values, cfg.get("hero_level_value", 0.0), cfg.get("worker_cap", 0))
        pot = shaped(obs, mat, cfg.get("float_penalty", 0.0))
        if obs.game_over or t >= cfg["max_steps"]:
            break
        # --shaping-decay (OpenAI Five, its eq. 11): every reward but the outcome times decay^(game minutes / 10),
        # so a long game's large material swings do not drown the win
        fade = cfg["shaping_decay"] ** (obs.game_time / 600.0) if cfg.get("shaping_decay", 0.0) > 0 else 1.0
        for s, tr in trajs.items():  # shaping: gamma * phi(s') - phi(s), for the step just taken
            if s in spans and tr.steps:  # (it recorded a step this time)
                r = fade * shaping * (cfg["gamma"] * (pot[s] - pot[1 - s]) / scale - phi[s])
                r_z = 0.0
                if s in zs:  # the strategy's potential (not returned at the end: following it pays)
                    z_new = -cfg.get("z_reward", 0.0) * strat.mismatch(made[s], zs[s], pool["items"][races[s]], obs.game_time)
                    z_new -= cfg.get("z_mode_reward", 0.0) * viol.get(s, 0.0)
                    r_z = fade * (cfg["gamma"] * z_new - zphi[s])
                    zphi[s] = z_new
                if tr.steps and "rparts" in tr.steps[-1]:
                    tr.steps[-1]["rparts"][1:] += (r, r_z)
                r += r_z
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
        fade = cfg["shaping_decay"] ** (obs.game_time / 600.0) if cfg.get("shaping_decay", 0.0) > 0 else 1.0
        last = terminal - fade * shaping * phi.get(s, 0.0)
        tr.end(last, (terminal, -fade * shaping * phi.get(s, 0.0), 0.0))
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
        "prod": prod.row(side, obs), "opp_prod": prod.row(1 - side, obs), **({"advised": True} if advised else {}),
        **({"max_own": bots[side].view.max_own, "max_ent": bots[side].view.max_ent} if side in bots else {}),
        **({"shared_noise": shared_noise} if cfg.get("shared_noise", 0.0) > 0 else {}),
        **({"z_build": builders[side].log, "z_refusals": builders[side].why,
            "z_blocked": {c: dict(v) for c, v in builders[side].blocked.items()}} if side in builders else {}),
        **({"z_modes": trackers[side].report()} if side in trackers else {}),
        **({"taught": teachers[side].taught} if side in teachers else {}),
        **({"buy_orders": bots[side].bought, "buys": dict(getattr(bots[side], "buys", {}))}
           if side in bots and getattr(bots[side], "bought", 0) else {}),
        **({"scenario": scenario} if scenario else {}),
        **({"z_t": {c: round(v) for c, v in zs[side]["t"].items()},
            "z_made": {c: round(v) for c, v in made[side].items()}} if side in zs else {}),
        **({"z": zs[side]["src"] + (" won" if zs[side]["won"] else ""), "z_items": len(zs[side]["t"]),
            "z_miss": strat.mismatch(made[side], zs[side], pool["items"][races[side]], obs.game_time),
            # (which: the strategy's items due by now and not made; made and not in it)
            "z_missed": sorted(c for c, t0 in zs[side]["t"].items() if t0 <= min(obs.game_time, strat.horizon(zs[side])) and made[side].get(c, 1e9) > min(obs.game_time, strat.horizon(zs[side]))),
            "z_extra": sorted(c for c, t0 in made[side].items() if t0 <= min(obs.game_time, strat.horizon(zs[side])) and zs[side]["t"].get(c, 1e9) > min(obs.game_time, strat.horizon(zs[side])) and c in pool["items"][races[side]])} if side in zs else {}),
        **({"branch": round(replay["t"] * cfg["step_seconds"]), "branch_same": same} if replay is not None else {}),
        **({} if train else {"eval": True, "eval_key": opp.get("eval_tag") or opp.get("eval_key") or "current"})}
    out_q.put({"episode": ep})
    if grpo is not None:
        grpo["same"] = same
        if rec is not None and "h" in rec and t > rec["t"] + 20:  # (the game reached its branch with a minute to go)
            grpo["record"] = rec
        grpo["trajs"], grpo["ret"] = trajs, ret
    if record:  # the video's panel: A = the learner's side
        who = {"self": "itself", "ai": "built-in AI"}.get(opp["kind"], opp["name"])
        other = (f"built-in AI {opp['name'].split('-', 1)[1]}" if opp["kind"] == "ai"
                 else "itself" if opp["kind"] == "self" else "its exploiter" if opp["kind"] == "exploiter"
                 else f"snapshot {opp['name']}")
        ep["trace"] = {"steps": trace, "gamma": cfg["gamma"], "shared_noise": shared_noise, "outcome": {str(s): v for s, v in outcome.items()},
                       # (the video's panel: the value without the shaping's potential, overlay.FullGameOverlay)
                       "shaping": {"weight": shaping, "scale": scale, "penalty": cfg.get("float_penalty", 0.0)},
                       "title": f"{Path(cfg['run_dir']).name} · policy v{infer.nets.version} · vs {who}",
                       # (the learner's target strategy, if it had one: the panel shows it, made and due)
                       **({"target": {"t": dict(zs[side]["t"]), "made": dict(made[side]), "src": zs[side].get("src", ""),
                                      "enforced": bool(cfg.get("z_enforce"))}} if side in zs else {}),
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
    fx.ACTIVE_F = cfg.get("features", fx.F)  # (a spawned process: the learner's setting, --features)
    fx.set_geometry(*cfg.get("map_geometry", (fx.STARTS, fx.BIN_RANGE)))  # (and the map's frame)
    fx.set_z(cfg.get("z_dim", 0))  # (and the global vector's strategy block)
    fx.set_inputs(cfg.get("inputs", 1), bool(cfg.get("privileged")))  # (and the inputs: --inputs, the masks, --opsd-*)
    fx.set_masks(cfg.get("tech_mask", 0), cfg.get("valid_mask", 0))
    fx.set_neutral(cfg.get("neutral", 0))
    out_q.cancel_join_thread()  # exiting must not wait to flush trajectories nobody reads any more
    torch.set_num_threads(1)
    try:
        if conns is not None:  # the inference server makes the network calls (no GPU context here)
            sys.setswitchinterval(0.001)  # (a game thread woken by its answer waits for the GIL: 5 ms by default)
            shared = SharedRows(*cfg["shared_rows"]) if cfg.get("shared_rows") else None
            per = cfg["games_per_actor"] + cfg.get("eval_games_per_actor", 0)
            infer = RemoteInference(conns, shared, [2 * (wid * per + k) for k in range(len(conns))])
        else:  # (the actor's own network calls on the GPU)
            device = torch.device(cfg["device"])
            nets = Nets(cfg, device)
            while nets.current is None and not stop.is_set() and os.getppid() == cfg["learner_pid"]:
                time.sleep(1)
                nets.reload()
            infer = Inference(nets, device, 2 * cfg["games_per_actor"], cfg["compile"])
        evals = cfg.get("eval_games_per_actor", 0)
        if cfg.get("eval_games"):  # (--eval-games: that many in all, the first actors one more than the rest)
            evals = cfg["eval_games"] // cfg["actors"] + (wid < cfg["eval_games"] % cfg["actors"])
        threads = [threading.Thread(target=game_loop, args=(wid, k, cfg, infer, out_q, stop, render_q), daemon=True)
                   for k in range(cfg["games_per_actor"] + evals)]
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
                 exploiter_share: float = 0.0, tie: float = 0.0, floor: float = 0.0):
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
        `floor` < 0 (mode "tax"): the level goes on below 0 as the learner keeps winning, and the
        tax turns into a bonus: the AI gets that share of what it gathers on top (level -0.5 at a
        knob of 0.9: 45% more income), a richer opponent and longer games. The real game's launches
        never have one.
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
        self.floor = floor if mode == "tax" else 0.0
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
        self.shaping: float | None = None  # (--shaping-anneal: the shaping's weight now, in the spec)

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
            self.level[key] = min(1.0, max(self.floor, self.level[key] + self.rule[1] * move))

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
        self._restore_levels(summary)
        if self.exploiter is not None and summary.get("exploiter"):
            x = summary["exploiter"]
            self.exploiter.wins, self.exploiter.losses, self.exploiter.draws = x.get("wins", 0.0), x.get("losses", 0.0), x.get("draws", 0.0)
            self.exploiter.recent = list(x.get("recent") or [])
            self.exploiter_resets = summary.get("exploiter_resets", 0)
        me = summary.get("self") or {}
        self.self_member.wins, self.self_member.losses = me.get("wins", 0.0), me.get("losses", 0.0)
        self.self_member.draws, self.self_member.recent = me.get("draws", 0.0), list(me.get("recent") or [])

    def inherit(self, summary: dict, run: str, usable: Callable[[str], bool]) -> list[str]:
        """A new run takes over the league of `run` (its league.json: a fork starts with its parent's
        opponents, not a league of one): its past snapshots that `usable` takes (this map's point
        bins, the run's orders), with their records (PFSP weighs them at once), and its curriculum
        levels and balance (the parent's last levels fit the fork's start, its weights). Snapshots
        are renamed "past:<run>/<steps>" (steps count per run); those named after a run keep their
        names. The records against the built-in AI and itself stay the parent's (the new run's
        charts start clean). Returns the members taken."""
        taken = []
        for row in summary.get("members", []):
            if not row.get("path") or row["name"] in self.scripts or row["name"] in self.real or not usable(row["path"]):
                continue
            kind, _, rest = row["name"].partition(":")
            m = Member(row["name"] if "/" in rest else f"{kind}:{run}/{rest}", path=row["path"], steps=row.get("steps", 0))
            m.wins, m.losses, m.draws = row.get("wins", 0.0), row.get("losses", 0.0), row.get("draws", 0.0)
            m.recent = list(row.get("recent") or [])
            self.past.append(m)
            taken.append(m.name)
        if len(self.past) > self.max_past - 1:  # (room for the new run's own first snapshot; the first kept)
            del self.past[1:len(self.past) - self.max_past + 2]
        self._restore_levels(summary)
        return [m.name for m in self.past]

    def _restore_levels(self, summary: dict) -> None:
        """The curriculum levels and the balance between races from a league.json."""
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
                    self.level[key] = min(1.0, max(self.floor, v))  # (a parent's bonus levels: this run's floor)
        self.balance.update({k: v for k, v in (summary.get("balance") or {}).items() if k in self.balance})

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
        return {"launch": launch, "agents": agents, **({"balance": self.balance_taxes()} if self.balance else {}),
                **({"shaping": self.shaping} if self.shaping is not None else {})}

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

def to_device(a, device) -> torch.Tensor:
    """A host array on the device through pinned memory. From pageable memory the CUDA driver stages the
    copy itself, synchronously, holding the context's lock while the stream's earlier kernels finish: the
    inference loop in the learner's process waited ~4 ms a round on it during updates."""
    x = torch.as_tensor(a)
    if torch.device(device).type == "cuda":
        x = x.pin_memory()
    return x.to(device, non_blocking=True)


def _ladv_std(steps: list[dict]) -> float:
    """The spread of the units' own advantages over a batch (1 when there are none)."""
    parts = [s["ladv"] for s in steps if s.get("ladv") is not None and len(s["ladv"])]
    return float(np.concatenate(parts).std()) + 1e-8 if parts else 1.0


def _uadv_stats(steps: list[dict]) -> tuple[float, float]:
    """The UPGO advantages' mean and spread over a batch (0, 1 when there are none)."""
    u = np.array([s["uadv"] for s in steps if "uadv" in s], np.float32)
    return (float(u.mean()), float(u.std()) + 1e-8) if len(u) else (0.0, 1.0)


def unit_rewards(own: np.ndarray, rows: np.ndarray, events, values: dict, scale: float) -> np.ndarray:
    """Each own unit's reward of its own for the step just taken (--unit-credit; OpenAI Five's per-hero
    reward): the material it lost or regained (its value times the change of its hit points' fraction;
    buildings do not count, their hit points grow as they are built), plus the material of the enemy units
    it killed (DEATH events' killer), over the shaping's scale. `own`: the step's own unit rows (features'
    columns), `rows`: every unit after the step."""
    now = {int(r[fx.C_ID]): r for r in rows}
    out = np.zeros(len(own), np.float32)
    index = {int(u): k for k, u in enumerate(own[:, fx.C_ID].tolist())}
    for k, r in enumerate(own):
        if int(r[fx.C_FLAGS]) & fx.STRUCTURE:
            continue
        v, m = values.get(str(int(r[fx.C_TYPE])), 0), max(int(r[fx.C_MAXHP]), 1)
        after = now.get(int(r[fx.C_ID]))
        hp1 = 0 if after is None or int(after[fx.C_FLAGS]) & fx.DEAD else int(after[fx.C_HP])
        out[k] += v * (hp1 - int(r[fx.C_HP])) / m
    for e in events:
        if int(e.kind) == 1 and int(e.b) in index:  # (EventKind.DEATH: a dead, b killer, c its type)
            out[index[int(e.b)]] += values.get(str(int(e.c)), 0)
    return out / scale


def collate(steps: list[dict], device, width: int | None = None, with_labels: bool = False) -> dict:
    """A minibatch of steps as padded tensors. `width`: pad the entities to it (CUDA graphs: a few
    fixed shapes); `with_labels`: the advisor's label arrays even when no step has any."""
    B, E = len(steps), max(max(s["n"] for s in steps), width or 0)
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
    t = lambda a: to_device(a, device)  # noqa: E731
    labels = {}
    if with_labels or any("y_order" in s for s in steps):  # the advisor's labels (on-policy distillation; others: none)
        labels = {"y_order": np.zeros((B, O), np.int64), **{k: np.full((B, O), -1, np.int64) for k in ("y_ptr", "y_x", "y_y")}}
        has, only, teach = np.zeros(B, bool), np.zeros(B, bool), np.zeros((B, O), bool)
        for i, s in enumerate(steps):
            if "y_order" in s:
                has[i] = True
                only[i] = bool(s.get("y_only", False))
                for k in labels:
                    labels[k][i, :s["n_own"]] = s[k]
                if "y_teach" in s:  # (the heuristic teacher's units, --teach-coef)
                    teach[i, :min(len(s["y_teach"]), O)] = s["y_teach"][:O]
        labels = {**{k: t(v) for k, v in labels.items()}, "y_has": t(has), "y_only": t(only), "y_teach": t(teach),
                  "y_any": bool(has.any())}
    priv = {}
    if any("pent" in s for s in steps):  # (--opsd-coef: the privileged views, their own width; none: phas False)
        PE = _width(max(O, *(s.get("pn", 1) for s in steps)))  # (at least the batch's own units: the orders' width)
        pent = np.zeros((B, PE, fx.F), np.float32)
        ptyp, pcur = np.zeros((B, PE), np.int64), np.zeros((B, PE), np.int64)
        pmask, pglob, phas = np.zeros((B, PE), bool), np.zeros((B, G), np.float32), np.zeros(B, bool)
        for i, s in enumerate(steps):
            if "pent" in s:
                n = s["pn"]
                pent[i, :n], ptyp[i, :n], pcur[i, :n], pmask[i, :n], pglob[i], phas[i] = s["pent"], s["ptype"], s["pcur"], True, s["pglob"], True
            else:
                pmask[i, 0] = True  # (attention over nothing gives NaNs)
        priv = {"pent": to_device(pent, device), "ptype": to_device(ptyp, device), "pcur": to_device(pcur, device),
                "pmask": to_device(pmask, device), "pglob": to_device(pglob, device), "phas": to_device(phas, device)}
    ladv = None
    if any(s.get("ladv") is not None for s in steps):  # (--unit-credit: each own unit's share of the credit)
        ladv = np.zeros((B, O), np.float32)
        for i, s in enumerate(steps):
            if s.get("ladv") is not None:
                o = min(len(s["ladv"]), O)
                ladv[i, :o] = s["ladv"][:o]
    return {"ent": t(ent), "type": t(typ), "cur": t(cur), "mask": t(mask), "glob": t(glob), "n_own": t(n_own), **labels, **priv,
            **({"ladv": t(ladv)} if ladv is not None else {}),
            **({"uadv": t(np.array([s.get("uadv", 0.0) for s in steps], np.float32))}  # (--upgo)
               if any("uadv" in s for s in steps) else {}),
            **{k: t(v) for k, v in acts.items()}, "logp": t(logp), "avail": t(avail) if avail is not None else None,
            "adv": t(np.array([s["adv"] for s in steps], np.float32)),
            "ret": t(np.array([s["ret"] for s in steps], np.float32)),
            **({"ret_parts": t(np.stack([s["ret_parts"] if s.get("ret_parts") is not None else np.zeros(len(PART_NAMES), np.float32)
                                         for s in steps]))} if any(s.get("ret_parts") is not None for s in steps) else {}),
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


def collate_seq(seqs: list[list[dict]], device, width: int | None = None, B: int | None = None, T: int | None = None,
                labels: bool = False) -> dict:
    """Sequences as collate() batches of B * T steps (T: the longest; shorter ones padded with steps
    that have no units, "valid" False), plus "seq" for evaluate(): (B, T, the first steps' states
    [B, d], no starts: a sequence stays in one game)."""
    T = max(T or 0, max(len(q) for q in seqs))
    seqs = seqs + [[] for _ in range((B or len(seqs)) - len(seqs))]  # (empty: all padding)
    B = len(seqs)
    first = next(q[0] for q in seqs if q)
    G = len(first["glob"])
    pad = {"ent": np.zeros((1, fx.F), np.float16), "type": np.zeros(1, np.int16), "cur": np.zeros(1, np.int16),
           "glob": np.zeros(G, np.float32), "n": 1, "n_own": 0, "logp": np.zeros(0, np.float32), "adv": 0.0, "ret": 0.0,
           "avail": None, **{k: np.zeros(0, np.int16) for k in ("order", "tgt", "bx", "by")}}
    flat = [q[t] if t < len(q) else pad for q in seqs for t in range(T)]
    mb = collate(flat, device, width, with_labels=labels)
    mb["valid"] = to_device(np.array([t < len(q) for q in seqs for t in range(T)], bool), device)
    d = next((len(q[0]["h"]) for q in seqs if q and q[0].get("h") is not None), None)
    h0 = None
    if d is not None:
        h0 = to_device(np.stack([q[0]["h"] if q and q[0].get("h") is not None else np.zeros(d, np.float32)
                                 for q in seqs]), device)
    mb["seq"] = (B, T, h0, None)
    return mb


def distill_loss(net: FullGameNet, ev: dict, mb: dict, autocast, teach_scale: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """On-policy distillation: the cloning loss (bc.losses: the order, then its target given the
    teacher's order) on the learner's own steps that the built-in AI advised (mb["y_has"]), from
    the PPO pass's network outputs; and the share of the teacher's orders (not "none") the policy
    likes most."""
    own = mb["own"] & mb["y_has"][:, None]
    y = mb["y_order"]
    lp = torch.log_softmax(ev["logits"].float(), -1)
    lp_y = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    every = (own & (lp_y > -1e4)).float()  # (an order the unit can't get now: none of its business)
    if "y_only" in mb:  # (--z-build-labels: those steps label only the units the builder ordered, and each
        own = own & ((y > 0) | ~mb["y_only"][:, None])  # counts as one of all the step's own units: averaged over
    ok = (own & (lp_y > -1e4)).float()                   # the labelled ones alone, a rare label pulled the policy away)
    denom = every.sum().clamp(min=1)
    l_order = -(lp_y * ok).sum() / denom
    ptr, xl, z = net.target_logits(ev["g"], ev["u"], mb["mask"], y)  # (in the caller's autocast)
    yl = net.y_logits(z, mb["y_x"])
    # (masked means, not boolean indexing: every index or any() was a wait for the GPU)
    share = ok.sum() / denom
    l_ptr = _masked_ce(ptr, mb["y_ptr"], ok) * share
    l_pt = (_masked_ce(xl, mb["y_x"], ok) + _masked_ce(yl, mb["y_y"], ok)) * share
    with torch.no_grad():
        issued = ok * (y > 0)
        acc = ((lp[..., 1:].argmax(-1) + 1 == y) * issued).sum() / issued.sum().clamp(min=1)
    out = l_order + l_ptr + 0.5 * l_pt
    if teach_scale and "y_teach" in mb:  # (the heuristic teacher's lessons: rare, their own weight, --teach-coef)
        tm = (mb["y_teach"] & mb["own"] & (lp_y > -1e4)).float()
        out = out + teach_scale * (-(lp_y * tm).sum() / tm.sum().clamp(min=1)
                                   + _masked_ce(ptr, mb["y_ptr"], tm))
    return out, acc


def opsd_loss(net: FullGameNet, ev: dict, mb: dict, seq) -> torch.Tensor:
    """--opsd-coef: KL(teacher || policy) over each own unit's order, the teacher the same network given the step's
    privileged view (no gradient through it; own units come first in both views), on the steps that have one."""
    with torch.no_grad():
        evt = evaluate(net, mb["pent"], mb["ptype"], mb["pcur"], mb["pmask"], mb["pglob"], mb["n_own"], mb["order"],
                       mb["tgt"].clamp(max=mb["pent"].shape[1] - 1),  # (the player's view's pointers: only the orders'
                       mb["bx"], mb["by"], mb["avail"], seq=seq)      # logits are used, but gathered within its width)
        lt = torch.log_softmax(evt["logits"].float(), -1)
    ls = torch.log_softmax(ev["logits"].float(), -1)
    kl = (lt.exp() * (lt - ls)).sum(-1)  # [N, O]
    w = mb["own"].float() * mb["phas"].float()[:, None]
    return (kl * w).sum() / w.sum().clamp(min=1)


def _masked_ce(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Cross-entropy [N, O, K] logits against labels [N, O] (-1: none) averaged over the labelled
    entries with weight > 0 (0 when there are none), with no GPU sync."""
    has = (y >= 0).float() * weight
    ce = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.clamp(min=0).flatten(), reduction="none")
    return (ce * has.flatten()).sum() / has.sum().clamp(min=1)


# the network's parts for their own noise scales (NoiseMeter): parameter name prefixes
NOISE_GROUPS = (("input", ("ent.", "type_emb.", "cur_emb.", "glob.", "priv_proj.")), ("memory", ("mem_",)),
                *((f"layer {i}", (f"transformer.layers.{i}.",)) for i in range(16)),
                ("order head", ("order.", "order_emb.")),
                ("target heads", ("cond.", "query.", "key.", "point_x.", "x_emb.", "point_y.")), ("value head", ("value_head.",)))


def noise_group(name: str) -> str:
    return next((g for g, pre in NOISE_GROUPS if name.startswith(pre)), "other")


class NoiseMeter:
    """A measurement pass's gradients (ppo_update(measure=True)) per key ("all": the whole loss; "term
    <name>": one of its terms), per weight tensor: the sum of the minibatches' squared norms and the
    sum of their gradients, plain and in Adam's metric (each gradient divided by Adam's denominator,
    sqrt(v / (1 - beta2^t)) + eps: the direction Adam steps in; the weights stay put, so the sum's
    preconditioned norm is the preconditioned sum's). parts() gives |G|^2 (the true gradient's) and
    tr(S) (one step's gradient noise) for the whole network and per part (NOISE_GROUPS): with |G_B|^2 =
    |G|^2 + tr(S)/B for a batch of B steps, the minibatches (b steps) and their mean (k*b) give
    unbiased estimates of both."""

    def __init__(self, net: FullGameNet, opt):
        named = list(net.named_parameters())
        self.params = [p for _, p in named]
        groups = [noise_group(n) for n, _ in named]
        self.groups = {g: torch.tensor([x == g for x in groups], device=self.params[0].device) for g in dict.fromkeys(groups)}
        hp = opt.param_groups[0]
        self.denom = []  # (a weight without Adam's state yet: out of Adam's metric)
        for p in self.params:
            st = opt.state.get(p, {})
            t = float(st["step"]) if "step" in st else 0.0
            self.denom.append((st["exp_avg_sq"] / (1 - hp["betas"][1] ** t)).sqrt() + hp["eps"]
                              if "exp_avg_sq" in st and t > 0 else None)
        self.adam = torch.tensor([d is not None for d in self.denom], device=self.params[0].device)
        self.acc: dict[str, list] = {}
        self.sq: dict[str, torch.Tensor] = {}
        self.sqa: dict[str, torch.Tensor] = {}
        self.k: dict[str, int] = {}

    def add(self, key: str, grads) -> None:
        if key not in self.acc:
            self.acc[key] = [torch.zeros_like(p) for p in self.params]
            self.sq[key], self.sqa[key] = (torch.zeros(len(self.params), device=self.params[0].device) for _ in range(2))
            self.k[key] = 0
        idx = [i for i, g in enumerate(grads) if g is not None]
        g = [grads[i].float() for i in idx]
        torch._foreach_add_([self.acc[key][i] for i in idx], g)
        ix = torch.tensor(idx, device=self.sq[key].device)
        self.sq[key][ix] += torch.stack(torch._foreach_norm(g)).pow(2)
        ia = [j for j, i in enumerate(idx) if self.denom[i] is not None]
        if ia:
            pre = torch._foreach_div([g[j] for j in ia], [self.denom[idx[j]] for j in ia])
            self.sqa[key][ix[ia]] += torch.stack(torch._foreach_norm(pre)).pow(2)
        self.k[key] += 1

    def parts(self, b: float) -> dict[str, tuple[float, float]]:
        """{name: (|G|^2, tr(S))}: "all" and "all adam", "group <part>" (plain), "term <name>" (plain)."""
        out = {}
        for key, acc in self.acc.items():
            k = self.k[key]
            if k < 2:
                continue
            big = torch.stack(torch._foreach_norm(acc)).pow(2) / (k * k)
            small = self.sq[key] / k
            ia = [i for i, d in enumerate(self.denom) if d is not None]
            bigA = torch.zeros_like(big)
            if ia:
                bigA[ia] = torch.stack(torch._foreach_norm(torch._foreach_div([acc[i] for i in ia],
                                                                              [self.denom[i] for i in ia]))).pow(2) / (k * k)

            def pair(sm: torch.Tensor, bg: torch.Tensor) -> tuple[float, float]:
                sm, bg = float(sm), float(bg)
                return (k * bg - sm) / (k - 1), (sm - bg) * k * b / (k - 1)
            if key == "all":
                out["all"] = pair(small.sum(), big.sum())
                out["all adam"] = pair((self.sqa[key] / k)[self.adam].sum(), bigA[self.adam].sum())
                for g, m in self.groups.items():
                    out[f"group {g}"] = pair(small[m].sum(), big[m].sum())
            else:
                out[key] = pair(small.sum(), big.sum())
        return out


class NoiseScale:
    """The gradient noise scale B_simple = tr(S) / |G|^2 (McCandlish et al. 2018, "An Empirical Model
    of Large-Batch Training", appendix A), in agent steps: the batch beyond which a larger one buys
    proportionally fewer steps (the critical batch size's estimate; OpenAI Five sized its batches by
    it). Measured every --noise-scale updates on the batch at the weights it starts from, an extra pass
    without steps (ppo_update(measure=True)): measured over the update's first epoch instead, Adam's
    steps between the minibatches made their gradients partly cancel, and |G|^2 came out < 0 in 95% of
    fgself-76's updates. The measurements' |G|^2 and tr(S) are averaged first (an exponential mean
    with its start's bias corrected), then divided: one batch's |G|^2 can be < 0."""

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.mean: dict[str, list] = {}  # name -> [measurements, |G|^2's mean, tr(S)'s mean]

    def add(self, parts: dict[str, tuple[float, float]], minibatch: int) -> dict:
        """The noise scales from a measurement's parts (NoiseMeter.parts) as train.jsonl keys: the whole
        network's (grad/noise_scale; in Adam's metric grad/noise_scale_adam), a step's signal to noise
        (grad/snr: the minibatch over the noise scale), each part's (grad/noise_group/<part>) and each
        loss term's (grad/noise_term/<term>); a scale whose |G|^2 averages <= 0 is left out (none yet)."""
        out = {}
        if "all" in parts:
            out["grad/g2"], out["grad/trace"] = parts["all"]
        for name, (g2, tr) in parts.items():
            m = self.mean.setdefault(name, [0, 0.0, 0.0])
            m[0] += 1
            m[1] = self.decay * m[1] + (1 - self.decay) * g2
            m[2] = self.decay * m[2] + (1 - self.decay) * tr
            if m[1] <= 0:
                continue
            scale = m[2] / m[1]  # (the start's bias, 1 - decay^n, is the same in both: it cancels)
            key = ("grad/noise_scale" if name == "all" else "grad/noise_scale_adam" if name == "all adam"
                   else f"grad/noise_group/{name[6:]}" if name.startswith("group ") else f"grad/noise_term/{name[5:]}")
            out[key] = scale
            if name == "all":
                out["grad/snr"] = minibatch / scale
        return out


PART_NAMES = ("outcome", "shaping", "strategy")  # --reward-parts: the rewards each with its value and advantage


@torch.no_grad()
def parts_pass(net: FullGameNet, chunks: list[list[dict]], args, device) -> dict:
    """--reward-parts (GDPO, the user's idea): each reward part's advantage on its own, then normalized on its own
    and summed (args.part_weights), so one part's scale does not drown the others'. The parts' values from the
    network now (a pass over the batch, no gradient), GAE per part over each chunk; a chunk's last step without
    its game's end only bootstraps (its next state is the next chunk's): neutral. A part whose advantages spread
    less than args.part_floor times the outcome's is not blown up to the others' size (saturated: SAW's point).
    Sets each step's "adv", "ret", "ret_parts" and with UPGO "uadv" (on the sums) -> statistics."""
    K = net.config["vparts"]
    seqs = pieces(chunks, args.seq_len)
    per_mb = max(1, args.minibatch // args.seq_len)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(device.type == "cuda" and args.bf16))
    was = net.training
    net.eval()
    for a in range(0, len(seqs), per_mb):
        part = seqs[a:a + per_mb]
        mb = collate_seq(part, device, B=len(part), T=args.seq_len)
        with autocast:
            g, _ = net.encode(mb["ent"], mb["type"], mb["cur"], mb["mask"], mb["glob"])
            if net.memory:
                B, T, h0, _ = mb["seq"]
                g = net.context_seq(g.view(B, T, -1), h0)[0].reshape(B * T, -1)
            vp = net.value_parts(g, mb["glob"]).float().cpu().numpy()
        T = mb["seq"][1]  # (collate_seq's rows: each sequence padded to T steps)
        for b, q in enumerate(part):
            for t, st in enumerate(q):
                st["vp"] = vp[b * T + t]
    net.train(was)
    gamma, lam = args.gamma, args.lam
    trained, neutral = [], []
    for c in chunks:
        last = c[-1]
        n = len(c) if last["done"] else len(c) - 1
        next_v, adv = (np.zeros(K, np.float32) if last["done"] else last["vp"]), np.zeros(K, np.float32)
        for st in reversed(c[:n]):
            nonterminal = 0.0 if st["done"] else 1.0
            delta = st["rparts"] + gamma * next_v * nonterminal - st["vp"]
            adv = delta + gamma * lam * nonterminal * adv
            st["advp"], st["ret_parts"] = adv.copy(), adv + st["vp"]
            next_v = st["vp"]
        if args.upgo > 0:  # (UPGO on the sums, as Trajectory.flush)
            boot = 0.0 if last["done"] else float(last["vp"].sum())
            g_, v1, q1 = boot, boot, -np.inf
            for st in reversed(c[:n]):
                rr, v = float(st["rparts"].sum()), float(st["vp"].sum())
                G = rr if st["done"] else rr + gamma * (g_ if q1 >= v1 else v1)
                st["uadv"] = G - v
                q1 = rr + (0.0 if st["done"] else gamma * v1)
                g_, v1 = G, v
        trained += c[:n]
        if not last["done"]:
            neutral.append(last)
    A = np.stack([st["advp"] for st in trained]) if trained else np.zeros((1, K), np.float32)
    mean, std = A.mean(0), A.std(0)
    scale = np.maximum(std, args.part_floor * std[0]) + 1e-8
    w = np.asarray([float(x) for x in args.part_weights.split(",")][:K], np.float32)
    for st in trained:
        st["adv"] = float((w * (st["advp"] - mean) / scale).sum())
        st["ret"] = float(st["ret_parts"].sum())
    um = float(np.mean([st["uadv"] for st in trained])) if trained and args.upgo > 0 else 0.0
    for st in neutral:  # (the batch's mean: no push either way)
        st["adv"], st["ret_parts"], st["ret"] = 0.0, st["vp"].copy(), float(st["vp"].sum())
        if args.upgo > 0:
            st["uadv"] = um
    out = {}
    for k, name in enumerate(PART_NAMES[:K]):
        out[f"parts/{name} adv std"] = float(std[k])
        out[f"parts/{name} return"] = float(np.mean([st["ret_parts"][k] for st in trained])) if trained else 0.0
    return out


def ppo_update(net: FullGameNet, ref: FullGameNet | None, opt, steps: list[dict], args, warmup: bool, device,
               chunks: list[list[dict]] | None = None, bc_iter=None, measure: bool = False, terms: bool = False) -> dict:
    """PPO epochs over the steps; for a network with memory over sequences cut from `chunks` (the
    actors' pieces of trajectories, in order), from the states the actors had. `bc_iter`: batches
    of demonstrations (bc.Steps) whose cloning loss joins each minibatch's, times args.bc_coef.
    `measure`: one pass and no steps, only the gradient noise's parts at these weights (NoiseMeter.parts;
    `terms`: also each loss term's, a backward pass each)."""
    advs = np.array([s["adv"] for s in steps], np.float32)
    mean, std = float(advs.mean()), float(advs.std()) + 1e-8
    lstd = _ladv_std(steps)
    umean, ustd = _uadv_stats(steps)
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
    params = list(net.parameters())
    meter = NoiseMeter(net, opt) if measure else None
    for epoch in range(1 if measure else args.epochs):
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
            if "ladv" in mb and getattr(args, "unit_credit", 0.0) > 0:  # (each unit's own share: --unit-credit)
                adv = adv + args.unit_credit * mb["ladv"] / lstd
            if "uadv" in mb and getattr(args, "upgo", 0.0) > 0:  # (UPGO's share: --upgo)
                adv = adv + args.upgo * ((mb["uadv"] - umean) / ustd)[:, None]
            log_ratio = (ev["logp"] - mb["logp"]) * own
            ratio = log_ratio.exp()
            pg = -(torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv) * own).sum() / n_units
            se = (((ev["vparts"].float() - mb["ret_parts"]) ** 2).sum(-1) if "ret_parts" in mb and "vparts" in ev
                  else (ev["value"] - mb["ret"]) ** 2)  # (--reward-parts: each part's value its own return)
            if "valid" in mb:  # (the padding steps: no units, no value target)
                valid = mb["valid"].float()
                v_loss = 0.5 * (se * valid).sum() / valid.sum()
            else:
                v_loss = 0.5 * se.mean()
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
                kl = anchor_kl(ev, ev_ref, getattr(args, "ref_kl_mode", "reverse"))
                ref_kl = (kl * own).sum() / n_units
                # also while the value warms up: it shares the network's trunk, so training it alone
                # moved the policy too (the KL to the clone doubled in the first five updates)
                loss = loss + args.ref_kl * ref_kl
            if getattr(args, "opsd_coef", 0.0) > 0 and "pent" in mb:  # (self-distillation, privileged information)
                l_opsd = opsd_loss(net, ev, mb, seq)
                loss = loss + args.opsd_coef * l_opsd
                stats.setdefault("loss/opsd", []).append(l_opsd.detach())
            if getattr(args, "opd_coef", 0.0) > 0 and mb.get("y_any"):
                l_opd, opd_acc = distill_loss(net, ev, mb, autocast, getattr(args, "teach_coef", 0.0) / args.opd_coef)
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
            if meter is not None:  # (the gradients at these weights, unclipped; no step)
                if terms:  # each term's, and the whole loss's as their sum
                    named = {"value": args.vf_coef * v_loss}
                    if not warmup:
                        named["policy"] = pg
                    if ref is not None and args.ref_kl > 0:
                        named["kl"] = args.ref_kl * ref_kl
                    if args.ent_coef or getattr(args, "opd_coef", 0.0) > 0 or (bc_iter is not None and args.bc_coef > 0):
                        named["other"] = loss - sum(named.values())
                    grads = {n: torch.autograd.grad(v, params, retain_graph=True, allow_unused=True) for n, v in named.items()}
                    for n, g in grads.items():
                        meter.add(f"term {n}", g)
                    whole = [None if all(g[i] is None for g in grads.values()) else
                             sum(g[i] for g in grads.values() if g[i] is not None) for i in range(len(params))]
                else:
                    whole = torch.autograd.grad(loss, params, allow_unused=True)
                meter.add("all", whole)
                continue
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
    if meter is not None:
        return meter.parts(len(steps) / max(1, meter.k.get("all", 1)))
    return {k: float(torch.stack(v).float().mean()) for k, v in stats.items()}


WIDTHS = (32, 48, 64, 96, 128, fx.MAX_ENT)  # the padded entity counts the update's CUDA graphs are captured for


def _width(n: int) -> int:
    return next(w for w in WIDTHS if w >= n)


class GraphedUpdate:
    """The update's minibatch step (every forward pass, the losses, the backward pass, the gradient's
    clip and Adam's step) as CUDA graphs, one per shape: the minibatch's entities padded to WIDTHS,
    the cloning batch's alike. A replay is a few launches where eager PyTorch made ~2,000: the update
    was bound by its own Python (3.7 s for 8,192 steps alone, ~10 s next to the games, with 1.8 s of
    kernels). The same losses as ppo_update: padding steps and entities are masked out (only dropout
    draws other numbers). Needs a network with memory (sequences), CUDA, and Adam(capturable=True)."""

    def __init__(self, net: FullGameNet, ref: FullGameNet | None, opt, args, device):
        self.net, self.ref, self.opt, self.args, self.device = net, ref, opt, args, device
        self.graphs: dict[tuple, tuple] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.mean = torch.zeros((), device=device)
        self.std = torch.ones((), device=device)
        self.pg_coef = torch.ones((), device=device)
        self.lstd = torch.ones((), device=device)  # (the units' own advantages' spread: --unit-credit)
        self.umean, self.ustd = torch.zeros((), device=device), torch.ones((), device=device)  # (--upgo)

    def _step(self, s: dict, b: dict | None) -> dict:
        """One minibatch's step on the static inputs `s` (and the cloning batch `b`): no GPU syncs."""
        net, ref, args = self.net, self.ref, self.args
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.bf16), cache_enabled=False):
            seq = (s["B"], s["T"], s["h0"], None)
            ev = evaluate(net, s["ent"], s["type"], s["cur"], s["mask"], s["glob"], s["n_own"],
                          s["order"], s["tgt"], s["bx"], s["by"], s["avail"], seq=seq)
            value = ev["value"].float()
            own = s["own"].float()
            n_units = own.sum().clamp(min=1)
            adv = ((s["adv"] - self.mean) / self.std)[:, None]
            if "ladv" in s and getattr(args, "unit_credit", 0.0) > 0:  # (each unit's own share: --unit-credit)
                adv = adv + args.unit_credit * s["ladv"] / self.lstd
            if "uadv" in s and getattr(args, "upgo", 0.0) > 0:  # (UPGO's share: --upgo)
                adv = adv + args.upgo * ((s["uadv"] - self.umean) / self.ustd)[:, None]
            log_ratio = (ev["logp"] - s["logp"]) * own
            ratio = log_ratio.exp()
            pg = -(torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv) * own).sum() / n_units
            valid = s["valid"].float()
            se = (((ev["vparts"].float() - s["ret_parts"]) ** 2).sum(-1) if "ret_parts" in s and "vparts" in ev
                  else (value - s["ret"]) ** 2)  # (--reward-parts: each part's value its own return)
            v_loss = 0.5 * (se * valid).sum() / valid.sum().clamp(min=1)
            entropy = (ev["entropy"] * own).sum() / n_units
            loss = args.vf_coef * v_loss - args.ent_coef * entropy + self.pg_coef * pg
            out = {"loss/policy": pg, "loss/value": v_loss, "loss/entropy": entropy}
            ref_kl = torch.zeros((), device=self.device)
            if ref is not None and args.ref_kl > 0:
                with torch.no_grad():
                    ev_ref = evaluate(ref, s["ent"], s["type"], s["cur"], s["mask"], s["glob"], s["n_own"],
                                      s["order"], s["tgt"], s["bx"], s["by"], s["avail"], seq=seq if ref.memory else None)
                ref_kl = (anchor_kl(ev, ev_ref, getattr(args, "ref_kl_mode", "reverse")) * own).sum() / n_units
                loss = loss + args.ref_kl * ref_kl
            out["loss/ref_kl"] = ref_kl
            if getattr(args, "opsd_coef", 0.0) > 0 and "pent" in s:  # (self-distillation, privileged information)
                l_opsd = opsd_loss(net, ev, s, seq)
                loss = loss + args.opsd_coef * l_opsd
                out["loss/opsd"] = l_opsd
            if getattr(args, "opd_coef", 0.0) > 0:
                l_opd, opd_acc = distill_loss(net, ev, s, None, getattr(args, "teach_coef", 0.0) / args.opd_coef)
                loss = loss + args.opd_coef * l_opd
                out.update({"loss/opd": l_opd, "opd/acc": opd_acc, "opd/steps": s["y_has"].float().sum() / valid.sum().clamp(min=1)})
            if b is not None:
                from .bc import losses as bc_losses
                l_bc, _ = bc_losses(net, {k: v for k, v in b.items() if k != "width"}, self.device, None, value_coef=0.0,
                                    stats=False, width=b["width"])
                loss = loss + args.bc_coef * l_bc
                out["loss/bc"] = l_bc
        loss.backward()
        out["grad_norm"] = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm, foreach=True)
        self.opt.step()
        with torch.no_grad():
            out["loss/kl"] = (((ratio - 1) - log_ratio) * own).sum() / n_units
            out["loss/clipfrac"] = (((ratio - 1).abs() > args.clip).float() * own).sum() / n_units
        return out

    def _static(self, mb: dict, b: dict | None) -> tuple[dict, dict | None]:
        s = {k: v.clone() for k, v in mb.items() if isinstance(v, torch.Tensor)}
        B, T, h0, _ = mb["seq"]
        s["B"], s["T"] = B, T
        s["h0"] = (h0.clone() if h0 is not None else torch.zeros(B, self.net.d, device=self.device))
        if b is not None:
            b = {**{k: v.to(self.device).clone() for k, v in b.items() if isinstance(v, torch.Tensor)}, "width": b["width"]}
        return s, b

    def _fill(self, s: dict, mb: dict, sb: dict | None, b: dict | None) -> None:
        for k, v in mb.items():
            if isinstance(v, torch.Tensor) and k in s:
                s[k].copy_(v, non_blocking=True)
        h0 = mb["seq"][2]
        s["h0"].copy_(h0) if h0 is not None else s["h0"].zero_()
        if sb is not None:
            for k, v in b.items():
                if isinstance(v, torch.Tensor):
                    sb[k].copy_(v, non_blocking=True)

    def _capture(self, key: tuple, mb: dict, b: dict | None) -> tuple:
        """Warm up on a side stream (cuBLAS, autotuning), put the weights and Adam's state back, then
        capture the step (capturing runs nothing)."""
        s, sb = self._static(mb, b)
        params = [p for p in self.net.parameters()]
        saved = [p.detach().clone() for p in params]
        state = {id(p): {k: v.clone() for k, v in self.opt.state[p].items() if isinstance(v, torch.Tensor)}
                 for p in params if p in self.opt.state}
        allowed = self.net.allowed.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.opt.zero_grad(set_to_none=True)
                self._step(s, sb)
        torch.cuda.current_stream().wait_stream(side)
        with torch.no_grad():
            for p, v in zip(params, saved):
                p.copy_(v)
            for p in params:
                for k, v in self.opt.state[p].items() if p in self.opt.state else ():
                    if isinstance(v, torch.Tensor):  # (as before; a state the warm-up made: zero)
                        v.copy_(state[id(p)][k]) if id(p) in state and k in state[id(p)] else v.zero_()
            self.net.allowed.copy_(allowed)
        self.opt.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):  # (an inference thread
            out = self._step(s, sb)                                                         # may run alongside)
        self.graphs[key] = (graph, s, sb, out)
        return self.graphs[key]

    def update(self, steps: list[dict], chunks: list[list[dict]], warmup: bool, bc_iter=None) -> dict:
        args, net, device = self.args, self.net, self.device
        advs = np.array([s["adv"] for s in steps], np.float32)
        self.mean.fill_(float(advs.mean()))
        self.std.fill_(float(advs.std()) + 1e-8)
        self.lstd.fill_(_ladv_std(steps))
        um, us = _uadv_stats(steps)
        self.umean.fill_(um)
        self.ustd.fill_(us)
        self.pg_coef.fill_(0.0 if warmup else 1.0)
        if getattr(args, "opd_coef", 0.0) > 0:  # the advisor's orders become orders the unit type can get
            typ_y = [(s["type"][:s["n_own"]], s["y_order"]) for s in steps if "y_order" in s]
            if typ_y:
                ty = to_device(np.concatenate([a for a, _ in typ_y]).astype(np.int64), device)
                yy = to_device(np.concatenate([c for _, c in typ_y]).astype(np.int64), device)
                net.allowed[ty, yy] = True  # ("none" sets class 0: always allowed)
        if self.ref is not None:
            self.ref.allowed.copy_(net.allowed)
        net.train()
        seqs = pieces(chunks, args.seq_len)
        per_mb = max(1, args.minibatch // args.seq_len)
        labels = getattr(args, "opd_coef", 0.0) > 0
        outs: list[dict] = []

        def prepared():  # the minibatches made on the CPU (their copies to the GPU queued)
            for _ in range(args.epochs):
                widths = np.array([max(s["n"] for s in q) for q in seqs])
                for idx in minibatches(widths, per_mb, args.pad_groups):
                    width = _width(int(widths[idx].max()))
                    mb = collate_seq([seqs[i] for i in idx], device, width=width, B=per_mb, T=args.seq_len, labels=labels)
                    b = None
                    if bc_iter is not None and args.bc_coef > 0:
                        b = dict(next(bc_iter))
                        bw = _width(int(b["mask"].any(0).nonzero().max()) + 1)  # (on the CPU: no wait)
                        b = {k: (v[:, :bw] if v.dim() >= 2 and v.shape[1] == b["mask"].shape[1] else v) for k, v in b.items()}
                        if b["mask"].shape[1] < bw:  # (pad to the width)
                            b = {k: (torch.nn.functional.pad(v, (0, 0) * (v.dim() - 2) + (0, bw - v.shape[1]))
                                     if v.dim() >= 2 and v.shape[1] == b["mask"].shape[1] else v) for k, v in b.items()}
                        b = {k: (v.contiguous().pin_memory() if isinstance(v, torch.Tensor) else v)  # (see to_device)
                             for k, v in b.items()}
                        b["width"] = bw
                    yield (width, b["width"] if b is not None else 0, mb["pent"].shape[1] if "pent" in mb else 0,
                           tuple(sorted(k for k, v in mb.items() if isinstance(v, torch.Tensor)))), mb, b

        pause = getattr(args, "update_pause_ms", 0)
        batches = prepared()
        nxt = next(batches, None)
        while nxt is not None:
            key, mb, b = nxt
            if key in self.graphs:
                graph, s, sb, out = self.graphs[key]
                self._fill(s, mb, sb, b)
            else:
                graph, s, sb, out = self._capture(key, mb, b)
            graph.replay()
            outs.append({k: v.detach().clone() for k, v in out.items()})
            nxt = next(batches, None)  # (made while the GPU runs this one)
            if pause > 0 and nxt is not None:  # then the GPU to the inference loop for a moment (--update-pause-ms)
                torch.cuda.current_stream().synchronize()
                time.sleep(pause / 1000)
        net.eval()
        keys = dict.fromkeys(k for o in outs for k in o)  # (some only in some minibatches: --opsd-coef's, with a privileged view)
        stats = {k: float(torch.stack([o[k] for o in outs if k in o]).float().mean()) for k in keys}
        return {**stats, "adv/std": float(advs.std())} if stats else stats  # (the scale a GRPO bonus meets)


def set_dropout(net: torch.nn.Module, p: float) -> None:
    """Every dropout of the network at p (the transformer's: attention, residual, feed-forward), saved
    with its config. The actors act in eval mode, without dropout; trained with it, an update's ratios
    and KL count the dropout's noise (--rl-dropout)."""
    for m in net.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p
        elif isinstance(m, torch.nn.MultiheadAttention):
            m.dropout = p
    if isinstance(getattr(net, "config", None), dict):
        net.config["dropout"] = p


def save_checkpoint(path: Path, net: FullGameNet, opt, vocab: dict, agent_steps: int, update: int, keep_opt: int = 3) -> None:
    """A checkpoint, and Adam's state next to it (<steps>.adam; the newest `keep_opt` kept): a fork or a
    resume from it goes on with the optimizer's moments instead of fresh ones (a fresh Adam's first steps
    are ~lr per weight whatever the gradient: forks lost ~10 points of the AI's record in their first
    20 minutes)."""
    for obj, dest in (({"model": net.state_dict(), "config": net.config, "vocab": vocab, "agent_steps": agent_steps,
                        "update": update}, path), (opt.state_dict(), path.with_suffix(".adam"))):
        tmp = dest.with_name(dest.name + ".tmp")  # (whole or not at all: a learner stopped while saving left a
        torch.save(obj, tmp)                       # truncated checkpoint, and the resume failed on it)
        os.replace(tmp, dest)
    for old in sorted(path.parent.glob("[0-9]*.adam"))[:-keep_opt]:
        old.unlink(missing_ok=True)


def load_optimizer(opt, checkpoint: Path, device, lr: float) -> bool:
    """Adam's state saved with `checkpoint` (save_checkpoint), if there is one that fits; this run's lr."""
    path = Path(checkpoint).with_suffix(".adam")
    if not path.exists():
        return False
    capturable = [g.get("capturable", False) for g in opt.param_groups]
    try:
        opt.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    except (ValueError, RuntimeError, KeyError) as e:
        print(f"note: Adam's state in {path} does not fit ({e}); a fresh one", flush=True)
        return False
    for g in opt.param_groups:  # inputs added since (model.load: columns at the end of a weight): their state zeros
        for p in g["params"]:
            for k, v in opt.state.get(p, {}).items():
                if torch.is_tensor(v) and v.dim() == p.dim() and v.dim() > 0 and v.shape != p.shape:
                    pad = v.new_zeros(p.shape)
                    pad[tuple(slice(0, s) for s in v.shape)] = v
                    opt.state[p][k] = pad
    for g, c in zip(opt.param_groups, capturable):
        g["lr"], g["capturable"] = lr, c
    print(f"Adam's state from {path}", flush=True)
    return True


def publish(net: FullGameNet, version: int, run_dir: Path, name: str = "current.pt") -> None:
    tmp = run_dir / f"{name}.tmp"
    torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "config": net.config,
                "version": version}, tmp)
    tmp.replace(run_dir / name)


def shaping_now(args, update: int) -> float:
    """--shaping-anneal: the shaping's weight at this run's `update` (linearly to 0)."""
    return round(args.shaping * max(0.0, 1.0 - update / args.shaping_anneal), 4)


def publish_exploiter(xnet: FullGameNet, version: int, run_dir: Path, vocab: dict, keep: int = 40) -> None:
    """The exploiter's weights as a numbered file (exploiter/<version>.pt, the newest `keep` kept) and
    exploiter.json naming the newest: an exploiter game plays the version named at its start, served
    like a past snapshot (by path: the native inference loop serves the current network and
    snapshots; a new path is loaded once). `keep` covers the longest game (40 game minutes, ~3 wall
    minutes) many times over: the loop copies a snapshot's weights from the learner's copy, which
    NativeServer drops for versions past the newest `keep`."""
    xdir = run_dir / "exploiter"
    xdir.mkdir(exist_ok=True)
    path = xdir / f"{version:08d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"model": {k: v.detach().cpu() for k, v in xnet.state_dict().items()}, "config": xnet.config,
                "vocab": vocab, "version": version}, tmp)
    tmp.replace(path)
    tmp = run_dir / "exploiter.json.tmp"
    tmp.write_text(json.dumps({"version": version, "path": str(path)}))
    tmp.replace(run_dir / "exploiter.json")
    for old in sorted(xdir.glob("[0-9]*.pt"))[:-keep]:
        old.unlink(missing_ok=True)


def exploiter_key(cfg: dict) -> str:
    """The exploiter's newest weights (publish_exploiter): an exploiter game's key for its whole length."""
    return json.loads((Path(cfg["run_dir"]) / "exploiter.json").read_text())["path"]


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
    ap.add_argument("--eval-games-per-actor", type=int, default=0,
                    help="each actor's extra games that only evaluate: the real game against the built-in AI (normal "
                         "or insane, races at random), never advised, nothing trained on; episodes have eval: true "
                         "(the real-game yardstick sooner and steadier, the training's mix unchanged)")
    ap.add_argument("--inputs", type=int, default=1, help="2: also each hero's abilities (level, ready, cooldown, "
                    "learnable) and items, what each building makes, and the time of day (features.set_inputs; older "
                    "checkpoints read them with zero weights)")
    ap.add_argument("--neutral", type=int, default=0, help="the neutral buildings' wares as orders (a tavern's heroes, "
                    "mercenaries, the goblin merchant's items: protocol.Buy) and the tavern heroes' skills and spells "
                    "(neutral.py: every network loaded grows to the extended vocabulary); 2: no mercenaries")
    ap.add_argument("--memory-fade", type=int, default=0, help="inputs 3: the remembered enemies fade into the learner's "
                    "attention over this many updates (its bias on them from -8 to 0; older networks and the anchor "
                    "keep them hidden at -30); 0: as loaded")
    ap.add_argument("--teach-coef", type=float, default=0.0, help="--teach: the cloning loss's weight on the heuristic "
                    "teacher's lessons alone (rare among the target builder's labels: under --opd-coef they barely moved "
                    "the new orders)")
    ap.add_argument("--opsd-share", type=float, default=0.0, help="self-distillation with privileged information (the "
                    "user's idea): this share of the learner's training games act on the privileged view (every enemy "
                    "unit, the opponent's resources: the network learns to use it); never in evaluation")
    ap.add_argument("--opsd-coef", type=float, default=0.0, help="the other games' steps: a KL from the same network's "
                    "orders given the privileged view (no gradient through it) to its orders given the player's")
    ap.add_argument("--scenarios", default="", help="scenarios (scenarios.py: clash, raid, rich; comma-separated) "
                    "a share of the training games start in")
    ap.add_argument("--scenario-share", type=float, default=0.0, help="--scenarios: the share of the training games")
    ap.add_argument("--teach", type=int, default=0, help="a heuristic teacher (teacher.py: a tavern hero with a worker, "
                    "a town portal, a potion when a hero is low) labels the learner's steps, never carried out; "
                    "--opd-coef the labels' weight; needs --neutral 1")
    ap.add_argument("--tech-mask", type=int, default=0, help="mask the orders the player lacks the tech for "
                    "(features.TechMask: buildings, tiers, research levels, the hero limit)")
    ap.add_argument("--valid-mask", type=int, default=0, help="mask each hero's orders it can't carry out now: an "
                    "ability not learned, cooling down or without the mana, no skill point, an empty item slot (needs "
                    "--inputs 2)")
    ap.add_argument("--features", type=int, default=fx.F, help="the unit features in use (features.ACTIVE_F): the "
                                                                "rest are zeros (29: as before 2026-10-12, 50: without "
                                                                "the last order, 55: all)")
    ap.add_argument("--eval-checkpoints", type=Path, nargs="+", help="the evaluation games also play these "
                                                                    "checkpoints (each game one of them or the current "
                                                                    "policy, at random; episodes' eval_key): head to "
                                                                    "head, at the same time")
    ap.add_argument("--eval-games", type=int, default=0, help="evaluation games in all instead (as --eval-games-per-"
                                                            "actor; 4: ~600 a hour, each actor's slot cost ~2.5%% of "
                                                            "the training's throughput)")
    ap.add_argument("--rl-dropout", type=float, default=-1.0, help="the network's dropout in the updates (the actors "
                                                                 "act without; -1: the checkpoint's, 0.1 from cloning)")
    ap.add_argument("--eval-ai", default="normal,insane", help="the evaluation games' built-in AI, one of these at "
                                                              "random (a name twice: twice as often)")
    ap.add_argument("--order-temperature", type=float, help="the policy's order head samples at this temperature, "
                                                            "acting and training alike (saved with its checkpoints; "
                                                            "default: the starting weights')")
    ap.add_argument("--target-temperature", type=float, help="the policy's target heads (unit, x, y) sample at this "
                                                            "temperature (model.FullGameNet, acting and training "
                                                            "alike; saved in its checkpoints); default: the "
                                                            "checkpoint's (1.0 before)")
    ap.add_argument("--start-weights", type=Path, help="a new run's starting weights: another run's checkpoint (the "
                                                      "lineage's parent; scripts/selfplay_fork.py); --init stays the "
                                                      "KL anchor and the vocabulary")
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
    ap.add_argument("--learner-races", default="", help="the learner's race in training games, weighted (e.g. "
                                                        "human:2,undead:2,orc:1,nightelf:1; evaluation games stay "
                                                        "uniform); '': as --races")
    ap.add_argument("--mirror", type=int, default=1, help="both sides play the same race (on duelrush the races "
                                                           "are far from balanced: night elf beat the others 96%%)")
    ap.add_argument("--eval-fallback", type=float, default=0.0, help="seconds: the evaluation games here wait while "
                    "another machine's games keep arriving (scripts/remote_eval.py marks remote-eval.alive) and play "
                    "when none came for this long (0: they always play)")
    ap.add_argument("--z-pool", type=Path, help="strategy conditioning (AlphaStar's z): the strategies "
                    "(fullgame/strategy.py) a trained side's target is drawn from; the network needs a strategy block "
                    "(scripts/grow_strategy.py)")
    ap.add_argument("--z-share", type=float, default=0.5, help="the share of the learner's trained sides given a "
                    "strategy (the rest and every evaluation game: none)")
    ap.add_argument("--z-modes", type=float, default=0.0, help="more kinds of strategy conditioning (modes.py: no rush, "
                    "rush, expand, workers, creeping, harass, scouting, ...): each kind's chance to be on in a game with a "
                    "target, its switch and parameter in the strategy block (older networks grow it with zero weights)")
    ap.add_argument("--z-mode-reward", type=float, default=0.1, help="--z-modes: the strategy potential's weight of "
                    "one whole violation (a build target's item: --z-reward)")
    ap.add_argument("--reward-parts", type=int, default=0, help="GDPO (the user's idea): a value and an advantage per "
                    "reward (outcome, shaping, strategy), each normalized on its own, summed with --part-weights "
                    "(parts_pass; the value head grows, model.split_value)")
    ap.add_argument("--part-weights", default="1,1,1", help="--reward-parts: each part's weight")
    ap.add_argument("--part-floor", type=float, default=0.1, help="--reward-parts: a part's advantages are not scaled up "
                    "past this share of the outcome's spread (a saturated part stays small)")
    ap.add_argument("--z-reward", type=float, default=0.02, help="the strategy reward: its potential is minus this "
                    "per item of the race's list made by one of the side and its strategy by now, not both")
    ap.add_argument("--z-winners", type=int, default=1, help="draw the strategies of players who won (if any)")
    ap.add_argument("--z-enforce", type=int, default=0, help="with --z-pool: a side with a target may only train, "
                    "build and research the target's items (and workers and supply) for the strategies' horizon "
                    "(the availability mask; strategy.allowed)")
    ap.add_argument("--eval-vs", type=float, default=0.0, help="this share of the evaluation launches play the current "
                    "policy against one of --eval-checkpoints (agents both sides, no tax; their rows' eval_key \"vs "
                    "<run>/<checkpoint>\"): a yardstick beyond the built-in AI")
    ap.add_argument("--eval-guide", type=float, default=0.0, help="with --eval-z: the evaluation worker also plays "
                    "current+zg, current+z's games with classifier-free guidance toward the target (model.act: the order "
                    "logits free + w * (with - free)) at this weight w (e.g. 2; 0: none)")
    ap.add_argument("--eval-map", default="", help="the evaluation games' map, if not the training's (e.g. train on "
                    "echofast, evaluate on (2)EchoIsles: the real game); the same map frame")
    ap.add_argument("--eval-z-pool", type=Path, default=None, help="with --eval-map: the strategies its current+z games "
                    "draw from (default --z-pool)")
    ap.add_argument("--grpo-share", type=float, default=0.0, help="GRPO from mid-game branches: this share of the "
                    "training games between agents (the first of a launch, seeded: such games replay exactly) are a "
                    "source; at a random step in --grpo-branch, --grpo-group fresh games replay its commands to "
                    "there and play on with new samples, and each branch's advantages get --grpo-coef times its "
                    "return from the branch less its group's mean (engine notes: determinism; run_branch)")
    ap.add_argument("--grpo-group", type=int, default=4, help="branches per group")
    ap.add_argument("--grpo-coef", type=float, default=1.0, help="the group-relative return's weight in a branch's advantages")
    ap.add_argument("--grpo-seeds", type=int, default=8, help="the seeds games between agents draw from (a built map each)")
    ap.add_argument("--grpo-branch", default="60-480", help="the branch step's range, game seconds")
    ap.add_argument("--z-build", type=int, default=0, help="with --z-pool: the harness starts the target's due "
                    "items that the agent leaves undone, and what they need first (TargetBuilder, the tech tree: "
                    "buildings with a worker near the town hall; units, heroes, research and tiers at an idle "
                    "building; training and current+z evaluation games); 2: a teacher only, its orders labels for "
                    "--z-build-labels and never carried out (the agent has to make its target itself)")
    ap.add_argument("--z-build-labels", type=int, default=0, help="with --z-build and --opd-coef > 0: the builder's "
                    "accepted orders label the agent's step for the distillation loss (only the units it ordered), so the "
                    "agent learns to make its target's buildings and heroes itself")
    ap.add_argument("--z-time-scale", type=float, default=1.0, help="the targets' times scaled (faster rules: e.g. 0.45 "
                    "on echofast, half costs and a third of the build times)")
    ap.add_argument("--eval-z", type=int, default=0, help="with --z-pool: the current policy also plays evaluation "
                    "games with a target strategy (eval_key current+z: the same weights, paired on the same matchups)")
    ap.add_argument("--z-src", default="", choices=["", "human", "ai"], help="draw only the strategies of people's "
                    "replays (human) or of the built-in AI's games (ai); default: both")
    ap.add_argument("--exploiter-share", type=float, default=0.0,
                    help="AlphaStar's main exploiter: this share of launches are games between the learner and a "
                         "second network that trains only against it (League; 0: none)")
    ap.add_argument("--exploiter-batch", type=int, default=4096, help="the exploiter's steps per update")
    ap.add_argument("--exploiter-reset", type=float, default=0.7,
                    help="the exploiter's score against the learner (its last 50 games) at which its weights join the "
                         "league and it starts over from the run's starting weights")
    ap.add_argument("--exploiter-timeout", type=int, default=300, help="updates after which it starts over anyway")
    ap.add_argument("--exploiter-publish", type=int, default=8,
                    help="the exploiter's updates between the versions new games play (and at a restart): a version a "
                         "game is each another network for the inference loop to run; one every update (~18 s) "
                         "kept ~10 in play and saturated the loop (the main policy's steps 1,600 -> 750 a second)")
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
    ap.add_argument("--curriculum-floor", type=float, default=0.0,
                    help="mode tax: the lowest level; below 0 the tax becomes a bonus to the AI's income (its share "
                         "the level times --curriculum-delay), so a learner that keeps winning meets a richer AI")
    ap.add_argument("--bc-data", type=Path, nargs="*", default=[],
                    help="demonstration collections (e.g. takeover games from the policy's own states) whose cloning "
                         "loss joins each PPO minibatch, times --bc-coef")
    ap.add_argument("--bc-coef", type=float, default=0.02)
    ap.add_argument("--bc-batch", type=int, default=256)
    ap.add_argument("--bc-winners", type=int, default=0, help="the cloning loss: only the side that won each game")
    ap.add_argument("--bc-z", type=int, default=1, help="with --z-pool: the cloning loss's sides with their own strategy "
                    "as the input (strategy.hindsight: what each made and when; AlphaStar's supervised policy learned "
                    "what a strategy means so), the sides the pool lacks left out")
    ap.add_argument("--bc-max-minutes", type=float, default=0.0, help="the cloning loss: only games decided within this many minutes")
    ap.add_argument("--bc-workers", type=int, default=4, help="the cloning loss's data workers (2 kept the learner "
                                                                 "waiting ~18%% of its time)")
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
    ap.add_argument("--league-from", help="a new run starts with this run's league (a name under --runs, or its directory): "
                    "its past snapshots that fit (this map's point bins, the same orders) with their records, "
                    "and its curriculum levels and balance (scripts/selfplay_fork.py: the parent's)")
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
    ap.add_argument("--ref-kl-mode", default="reverse", choices=["reverse", "forward", "forward-all"],
                    help="that term's direction and heads (model.anchor_kl): reverse KL over the orders (as before), "
                         "forward KL over the orders (AlphaStar's direction: rare orders keep their weight), "
                         "or forward over the orders and their targets")
    ap.add_argument("--value-warmup", type=int, default=-1, help="first updates: the value head only (default: 10, "
                                                                  "2 when BC trained the value head)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--noise-scale", type=int, default=10, help="every this many updates, the gradient noise scale "
                    "(grad/noise_scale, in agent steps: the critical batch size's estimate; NoiseScale) from an extra "
                    "pass over the batch without steps (0: never); also per part of the network and in Adam's metric")
    ap.add_argument("--noise-terms", type=int, default=50, help="every this many updates (a multiple of --noise-scale), "
                    "the noise scale of each loss term too (policy, value, KL: a backward pass each)")
    ap.add_argument("--bf16", type=int, default=1, help="the updates' matmuls in bfloat16")
    ap.add_argument("--game-cuts", type=int, default=0, help="the training games' W3SIM_CUTS (shim/cuts.c: equivalent "
                                                            "speedups of the game's own code; 1529: all verified ones, "
                                                            "~28%% less CPU a game step); 0: none")
    ap.add_argument("--unit-credit", type=float, default=0.0, help="each own unit's advantage also gets this much of "
                                                                 "a share of its own (OpenAI Five's team spirit: the "
                                                                 "material it lost and killed over the next few steps, "
                                                                 "relative to the step's other units, normalised); 0: "
                                                                 "the step's advantage for all its units")
    ap.add_argument("--unit-gamma", type=float, default=0.8, help="the discount of that share's return (0.8: ~5 steps)")
    ap.add_argument("--upgo", type=float, default=0.0, help="each step's advantage also gets this much of its UPGO "
                                                          "advantage (AlphaStar: the return follows the trajectory "
                                                          "while the actions after did at least as well as the value "
                                                          "expected, else the value; normalised); 0: off")
    ap.add_argument("--shared-noise", type=float, default=0.0, help="this share of the games (per game, at random) "
                                                                  "sample each policy side's units' targets (unit, x, y) "
                                                                  "with one noise vector a step (the same distributions, "
                                                                  "coordinated choices: a group's target points had "
                                                                  "scattered); the episode rows say which, for an A/B")
    ap.add_argument("--update-pause-ms", type=float, default=0.0, help="after each minibatch of the CUDA-graph update, "
                                                                     "wait for it and leave the GPU to the inference "
                                                                     "loop this long (during the update its rounds took "
                                                                     "~10x their GPU time; the learner waits for data "
                                                                     "about as long as it updates)")
    ap.add_argument("--cuda-graphs", type=int, default=1, help="the update's minibatch steps as CUDA graphs (GraphedUpdate; "
                                                             "networks with memory)")
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
    ap.add_argument("--infer-native", type=int, default=0, help="the inference server's loop in C++ in the learner's "
                                                                  "process (native_infer: one GPU context, no GIL)")
    ap.add_argument("--infer-in-learner", type=int, default=0, help="the inference server as a thread of the learner's "
                                                                     "process: one GPU context, no turns on the GPU")
    ap.add_argument("--infer-fuse", type=int, default=1, help="the inference server's calls compiled (fused kernels) "
                                                               "before their CUDA graphs are captured: 30%% less GPU a round")
    ap.add_argument("--infer-entities", default="48,96", help="the entity counts the current network's calls are "
                                                               "padded to (each a CUDA graph per batch bucket; up to 160 always)")
    ap.add_argument("--infer-past-entities", default="64", help="the same for the past snapshots' calls")
    ap.add_argument("--gpu-split", type=int, default=0, help="with --infer-native: this many of the GPU's SMs the "
                                                            "inference loop's alone, the rest the learner's (CUDA green "
                                                            "contexts, gpu_split.py); 0: both on all")
    ap.add_argument("--infer-pipeline", type=int, default=1, help="the inference server reads and launches the "
                                                                   "next round while the last one is on the GPU")
    ap.add_argument("--infer-shm", type=int, default=1, help="the games' view steps and answers through shared "
                                                              "memory (SharedRows), not pickled through the pipes")
    ap.add_argument("--infer-streams", type=int, default=1, help="the inference server runs a round's networks "
                                                                  "side by side, a CUDA stream each")
    ap.add_argument("--infer-past-slots", type=int, default=4, help="past snapshots a round runs side by side "
                                                                     "(a network copy each)")
    ap.add_argument("--wine", default="", help="another Wine's bin folder (e.g. GE-Proton's files/bin; "
                                              "runtime.wine.use_wine): its own prefixes")
    ap.add_argument("--fsync", type=int, default=0, help="with --wine (a Proton build): Wine's synchronization "
                                                         "on futexes, not wineserver (WINEFSYNC)")
    ap.add_argument("--ntsync", type=int, default=0, help="with --wine (a Proton build): Wine's synchronization "
                                                          "in the kernel's ntsync driver (WINENTSYNC; /dev/ntsync)")
    ap.add_argument("--avail-mask", type=int, default=1, help="mask the orders the player can't pay for yet")
    ap.add_argument("--chunk", type=int, default=64, help="steps per trajectory piece an actor sends")
    ap.add_argument("--seq-len", type=int, default=16, help="a network with memory: steps per training sequence "
                                                            "(from the state the actor had at its first)")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--video-every", type=float, default=10.0,
                    help="minutes between game videos (0: none; the actors take turns). Rendering one takes ~6 minutes of "
                         "3.4 cores (the game at 960x540 on a software renderer, in real time)")
    ap.add_argument("--shaping", type=float, default=1.0, help="weight of the material-lead reward shaping (0: none)")
    ap.add_argument("--shaping-decay", type=float, default=0.0, help="every reward but the outcome (the shaping, the "
                    "strategy's) times this ^ (game minutes / 10) (OpenAI Five: 0.6; 0: none)")
    ap.add_argument("--ref-update", type=int, default=0, help="every this many updates the KL anchor becomes the current "
                    "policy (a rolling anchor: the KL still stops sudden jumps, but no longer pulls back to the start's "
                    "habits; 0: the anchor stays --init)")
    ap.add_argument("--worker-cap", type=int, default=0, help="the shaping's material counts a player's workers only as "
                    "much as this many (0: all; trace.material)")
    ap.add_argument("--shaping-anneal", type=int, default=0, help="updates over which the shaping's weight falls to 0 "
                    "(linearly, from this run's start; 0: it stays): a game keeps the weight it started with, so its "
                    "potential still goes back to 0 at its end")
    ap.add_argument("--shaping-scale", type=float, default=2000.0, help="material (gold + lumber cost) worth 1 of potential")
    ap.add_argument("--hero-level-value", type=float, default=0.0, help="the shaping's material: a hero also counts this "
                                                                       "much for each level above the first (its cost "
                                                                       "alone does not grow as it levels)")
    ap.add_argument("--float-penalty", type=float, default=0.0,
                    help="the shaping's potential counts the gold and lumber a player holds at minus this (0: not at "
                         "all): spending is credited at the order")
    ap.add_argument("--tie-break", type=float, default=0.5, help="a tie's reward: this times tanh(2 x material lead)")
    ap.add_argument("--tie-value", type=float, default=0.0,
                    help="added to a tie's reward (e.g. -0.5: a tie at the time limit counts almost as a loss)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", help="default: cuda if available (asking opens the GPU driver)")
    ap.add_argument("--resume", action="store_true", help="continue the run --name: its latest checkpoint, league and counts")
    ap.add_argument("--resume-weights", type=Path, help="with --resume: this checkpoint's weights instead of the latest's "
                                                        "(a roll back; scripts/selfplay_restart.py --weights)")
    ap.add_argument("--title", default="", help="a new run's change in a few words (summary.txt: the lineage shows it); "
                    "with --resume: the restart's short name (the lineage's subtitle)")
    ap.add_argument("--note", default="", help="the run's notes (notes.md); with --resume, why it was restarted (the "
                                                "restart's marker on the dashboard's charts)")
    args = ap.parse_args(argv)
    fx.ACTIVE_F = args.features  # (the cloning loss's data workers fork from here: they inherit it)
    os.environ["WARCRAFTSIM_FEATURES"] = str(args.features)  # (and processes started anew: actors, videos)
    if args.valid_mask and args.inputs < 2:
        raise SystemExit("--valid-mask reads the hero features: --inputs 2")
    fx.set_inputs(args.inputs, args.opsd_share > 0 or args.opsd_coef > 0)  # (before any network is loaded: model.load
    # gives older ones the new inputs; --opsd-*: the privileged channels)
    fx.set_masks(args.tech_mask, args.valid_mask)
    fx.set_neutral(args.neutral)
    if args.z_modes > 0:  # (the strategy block with the modes' part: model.load grows older networks')
        fx.set_z(strat.ZDIM + zmodes.MDIM)
    from ..data.duelmap import parse_duel_name
    if parse_duel_name(args.map) is None:  # another map than the duel maps: its own frame for the positions
        from ..data.mapbuild import stock_map_path    # and point targets (features.set_geometry; the actors
        fx.set_geometry(*fx.map_geometry(stock_map_path(args.map)))  # inherit it through the environment)
        print(f"map frame: starts {fx.STARTS}, point bins over +-{fx.BIN_RANGE:g} ({fx.BINS})", flush=True)
    if args.eval_games:  # (in all: each actor's pipes and rows for the most any actor gets)
        args.eval_games_per_actor = -(-args.eval_games // args.actors)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.wine:  # (before any game or child process: they read the environment)
        from ..runtime import wine
        wine.use_wine(args.wine, bool(args.fsync), bool(args.ntsync))
        wine.ensure_template()  # (here, once: not by the actors' games at once)
    import signal
    signal.signal(signal.SIGTERM, _interrupt)  # kill: a clean stop (status, a checkpoint, the actors)

    run_dir = args.runs / args.name
    for sub_dir in ("checkpoints", "replays", "videos"):
        (run_dir / sub_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    split = None
    if args.gpu_split and device.type == "cuda" and args.central_inference and args.infer_native:
        from .gpu_split import GpuSplit
        split = GpuSplit(args.gpu_split)  # (before anything else on the GPU: the learner's work all on its part)
        torch.cuda.set_stream(split.stream(1))
        print(f"GPU split: {split.small_sms} SMs for the inference loop, {split.rest_sms} for the learner", flush=True)
    torch.set_num_threads(4)  # the games need the cores
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # --neutral: the vocabulary of the weights it trains (its last checkpoint's, or the start's), when they have one
    # extended already: the anchor's orders put in its order (one extended in another process could list a toggle's
    # two orders the other way round, neutral.reorder)
    weights_vocab = None
    if fx.NEUTRAL:
        from .neutral import extended
        own = sorted((run_dir / "checkpoints").glob("[0-9]*.pt")) if args.resume else []
        src = args.resume_weights or (own[-1] if own else None) or (args.start_weights if not own else None)
        if src is not None:
            weights_vocab = torch.load(src, map_location="cpu", weights_only=False).get("vocab")
            weights_vocab = weights_vocab if weights_vocab is not None and extended(weights_vocab) else None
    net, ck = load(args.init, device, None if args.memory < 0 else bool(args.memory), vocab=weights_vocab)
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
    if args.start_weights and not args.resume:  # a new run from another run's checkpoint (--init stays the anchor)
        start, _ = load(args.start_weights, device, None if args.memory < 0 else bool(args.memory), vocab=vocab if fx.NEUTRAL else None)
        if arch_key(start.config) != arch_key(net.config):  # a grown network (scripts/grow_model.py): its own
            net = start                                     # shapes (the anchor keeps the init's)
        else:
            net.load_state_dict(start.state_dict())
            net.config["target_temperature"] = start.config.get("target_temperature", 1.0)
            net.config["order_temperature"] = start.config.get("order_temperature", 1.0)
        del start
    if args.target_temperature is not None:  # (the policy's, saved with its checkpoints; the anchor keeps its own)
        net.target_temp.fill_(args.target_temperature)
        net.config["target_temperature"] = args.target_temperature
    if args.order_temperature is not None:  # (alike, the order head's)
        net.order_temp.fill_(args.order_temperature)
        net.config["order_temperature"] = args.order_temperature
    fx.set_z(net.config.get("z", 0))  # the global vector's strategy block (scripts/grow_strategy.py), as the policy's
    G = fx.Encoder(vocab).G
    for what, n in [("the starting weights" if args.start_weights else "--init", net), ("the anchor (--init)", ref)] + \
            [(str(c), FullGameNet(**adapt_config(torch.load(c, map_location="cpu", weights_only=False)["config"])))
             for c in args.eval_checkpoints or []]:
        if n is not None and n.bins != fx.BINS:  # (their point targets must be this map's bins)
            raise SystemExit(f"{what} has {n.bins} point bins, {args.map}'s frame {fx.BINS}: scripts/grow_bins.py")
        if n is not None and n.config["G"] != G:  # (a run's rows have one width: the inference loop packs them alike)
            raise SystemExit(f"{what} reads {n.config['G']} global numbers, the run's {G} (a strategy block of "
                             f"{fx.ZDIM}): scripts/grow_strategy.py")
    if args.ref_update > 0 and ref is not None and arch_key(ref.config) != arch_key(net.config):
        raise SystemExit("--ref-update copies the policy into the anchor: --init must have the starting weights' architecture")
    if args.z_pool and not fx.ZDIM:
        raise SystemExit("--z-pool needs a network with a strategy block: scripts/grow_strategy.py")
    if args.reward_parts:  # (a value per reward: the old value is the first, the others start at zero)
        from .model import split_value
        net = split_value(net, len(PART_NAMES))
    if args.rl_dropout >= 0:  # the updates' dropout (the actors act without: eval mode)
        set_dropout(net, args.rl_dropout)
    graphed = bool(args.cuda_graphs) and device.type == "cuda" and bool(net.memory)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5, capturable=graphed)
    if args.start_weights and not args.resume:
        load_optimizer(opt, args.start_weights, device, args.lr)
    # the main exploiter (--exploiter-share): a second network, from the run's starting weights
    # (as AlphaStar's from the supervised agent), trained on its games against the learner alone
    x_init = {k: v.detach().clone() for k, v in net.state_dict().items()} if args.exploiter_share > 0 else None
    xnet = copy.deepcopy(net) if x_init is not None else None
    xopt = torch.optim.Adam(xnet.parameters(), lr=args.lr, eps=1e-5, capturable=graphed) if xnet is not None else None
    slot, slot_lock = claim_slot(args.runs)
    races = list(fx.RACES) if args.races == "all" else args.races.split(",")
    ai = [d for d in args.ai.split(",") if d]
    # (the lineage's parent: the run the starting weights come from, else the anchor)
    init_rel = os.path.relpath((args.start_weights or args.init).resolve(), args.runs.resolve().parent)
    info = {"name": args.name, "kind": "run", "trainer": "fullgame", "task": f"fullgame_{args.map}",
            "status": "launching", "created": time.time(), "timesteps": int(args.timesteps),
            "envs": args.actors * args.games_per_actor, "init_from": init_rel, "init_steps": 0,
            "description": (f"Self-play on whole games ({args.map}) from a behavior-cloned policy: PPO per unit, "
                            f"a league of itself, past snapshots and the built-in AI ({', '.join(ai) or 'none'})"),
            "config": {k: str(v) for k, v in vars(args).items()},
            "spaces": fx.describe_spaces(vocab, None if vocab.get("order_names") else fx.demo_order_names(args.runs),
                                         "two per game against itself or a past snapshot, one against the built-in AI"),
            "inputs": {"init": str(args.init), **({"start_weights": str(args.start_weights)} if args.start_weights else {}),
                       "bc_data": [str(d) for d in (args.bc_data or [])],
                       **({"advisor": f"built-in AI ({args.opd_difficulty})"} if args.opd_share > 0 else {})},
            "launch": {"command": "python3 -m warcraftsim.fullgame.selfplay " + " ".join(sys.argv[1:] if argv is None else argv),
                       "argv": list(sys.argv[1:] if argv is None else argv), "git": git_info()}}

    resumed = None
    if args.resume and (run_dir / "run.json").exists():  # the run goes on: its weights, league and counts
        old = json.loads((run_dir / "run.json").read_text())
        rows = [json.loads(line) for line in open(run_dir / "train.jsonl")] if (run_dir / "train.jsonl").exists() else []
        cks = sorted((run_dir / "checkpoints").glob("[0-9]*.pt"))
        # --resume-weights: an earlier checkpoint's weights instead (a roll back; the counts and league go on)
        src = args.resume_weights if args.resume_weights else (cks[-1] if cks else None)
        if src:
            last = torch.load(src, map_location=device, weights_only=False)
            net.load_state_dict(trained(last["model"], net))
            load_optimizer(opt, src, device, args.lr)
        resumed = {"steps": rows[-1]["agent_steps"] if rows else 0, "update": rows[-1]["epoch"] if rows else 0,
                   "episodes": sum(1 for _ in open(run_dir / "episodes.jsonl")) if (run_dir / "episodes.jsonl").exists() else 0,
                   "checkpoint": Path(src).name if src else None,
                   **({"rolled_back_to": str(args.resume_weights)} if args.resume_weights else {})}
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
    if args.title and resumed is None:
        (run_dir / "summary.txt").write_text(args.title + "\n")

    league = League(run_dir, ai, {"ai": args.ai_share, "self": args.self_share, "past": 1.0 - args.self_share},
                    args.max_past, args.pfsp,
                    curriculum=((args.curriculum, args.curriculum_step, args.curriculum_delay, args.handicap)
                                if args.curriculum >= 0 else None), mode=args.curriculum_mode, races=tuple(races),
                    mirror=bool(args.mirror),
                    balance=(args.balance_step, args.balance_max) if args.balance_step > 0 else None,
                    exploiter_share=args.exploiter_share, tie=args.curriculum_tie, floor=args.curriculum_floor)
    league.real_share = args.real_share
    if args.shaping_anneal > 0:  # (the games from the start: the weight at this run's update)
        league.shaping = shaping_now(args, resumed["update"] if resumed else 0)
    if resumed is not None and (run_dir / "league.json").exists():
        league.restore(json.loads((run_dir / "league.json").read_text()))
    else:
        if args.league_from:  # (a fork: its parent's opponents, then its own start)
            src = Path(args.league_from) if Path(args.league_from).is_dir() else args.runs / args.league_from

            def usable(path: str) -> bool:
                if not Path(path).exists():
                    return False
                c = torch.load(path, map_location="cpu", weights_only=False)
                if fx.NEUTRAL and c.get("vocab"):  # (--neutral: model.load extends it)
                    from .neutral import extend_vocab
                    c["vocab"] = extend_vocab(c["vocab"])
                from .neutral import same_orders  # (--neutral: a reordering of the run's fits, Nets.get)
                return (c["config"].get("bins", 128) == fx.BINS and c.get("vocab") is not None
                        and (c["vocab"] == vocab or bool(fx.NEUTRAL) and same_orders(c["vocab"], vocab))
                        and adapt_config(c["config"])["G"] == net.config["G"])  # (the run's rows' width: a strategy block or none)
            taken = league.inherit(json.loads((src / "league.json").read_text()), src.name, usable)
            print(f"league from {src.name}: {len(taken)} past snapshots ({', '.join(taken)})", flush=True)
        first = run_dir / "checkpoints" / f"{0:016d}.pt"
        torch.save({**ck, "model": net.state_dict(), "config": net.config, "vocab": vocab, "agent_steps": 0}, first)
        league.add_snapshot(first, 0)  # the clone itself (a fork: its start), a past opponent
    league.write()
    publish(net, resumed["update"] if resumed else 0, run_dir)
    x_update = x_since = 0  # the exploiter's updates (its weights' version), and since its last restart
    if xnet is not None:
        if resumed is not None and (run_dir / "exploiter.pt").exists():  # (it goes on too)
            xck = torch.load(run_dir / "exploiter.pt", map_location=device, weights_only=False)
            xnet.load_state_dict(trained(xck["model"], xnet))
            x_update, x_since = xck["version"], xck.get("since", 0)
        publish(xnet, x_update, run_dir, "exploiter.pt")
        publish_exploiter(xnet, x_update, run_dir, vocab)
    cfg = {"run_dir": str(run_dir), "device": str(device), "vocab": str(run_dir / "vocab.json"), "races": races,
           "learner_races": {k: float(v) for k, v in (kv.split(":") for kv in args.learner_races.split(",") if kv)},
           "map": args.map, "handicap": args.handicap, "step_seconds": args.step_seconds,
           "max_minutes": args.max_minutes, "max_steps": int(args.max_minutes * 60 / args.step_seconds) + 20,
           "wait_floor_ms": args.wait_floor_ms, "games_per_actor": args.games_per_actor,
           "eval_games_per_actor": args.eval_games_per_actor, "eval_games": args.eval_games,
           "games_per_process": args.games_per_process, "chunk": args.chunk, "gamma": args.gamma, "lam": args.lam,
           "seed": args.seed, "start_update": resumed["update"] if resumed else 0, "slot": slot,
           "video_every": args.video_every, "scripted_reset": bool(args.scripted_reset),
           "agent_games_factor": args.agent_games_factor, "mirror": bool(args.mirror), "learner_pid": os.getpid(),
           "actors": args.actors, "values": unit_values(), "shaping": args.shaping, "shaping_scale": args.shaping_scale,
           "shaping_decay": args.shaping_decay,
           "tie_break": args.tie_break, "tie_value": args.tie_value, "compile": bool(args.compile), "native_obs": bool(args.native_obs),
           "real_share": args.real_share, "infer_batch": 64, "infer_streams": args.infer_streams, "infer_pipeline": args.infer_pipeline,
           "infer_fuse": args.infer_fuse, "game_cuts": args.game_cuts, "shared_noise": args.shared_noise,
           "unit_credit": args.unit_credit, "unit_gamma": args.unit_gamma, "upgo": args.upgo,
           "hero_level_value": args.hero_level_value,
           "features": args.features, "eval_checkpoints": [str(c) for c in args.eval_checkpoints or []],
           "inputs": args.inputs, "tech_mask": args.tech_mask, "valid_mask": args.valid_mask, "neutral": args.neutral, "teach": args.teach, "privileged": args.opsd_share > 0 or args.opsd_coef > 0,
           "opsd_share": args.opsd_share, "opsd_coef": args.opsd_coef,
           "scenarios": [x for x in args.scenarios.split(",") if x], "scenario_share": args.scenario_share,
           "eval_ai": [d for d in args.eval_ai.split(",") if d],
           "map_geometry": [fx.STARTS, fx.BIN_RANGE],
           "infer_entities": [int(x) for x in args.infer_entities.split(",")],
           "infer_past_entities": [int(x) for x in args.infer_past_entities.split(",")],
           "infer_past_slots": args.infer_past_slots, "game_nice": args.game_nice, "pairs": args.pairs,
           "exploiter_share": args.exploiter_share, "float_penalty": args.float_penalty,
           "infer_period": args.infer_period_ms / 1000.0,
           "avail_mask": bool(args.avail_mask),
           "opd_share": args.opd_share, "opd_difficulty": args.opd_difficulty,
           "costs": order_costs(vocab, args.map).tolist() if args.avail_mask else None, "max_past": args.max_past,
           "eval_fallback": args.eval_fallback, "worker_cap": args.worker_cap, "z_dim": fx.ZDIM, "z_pool": str(args.z_pool) if args.z_pool else "", "z_share": args.z_share,
           "z_reward": args.z_reward, "reward_parts": args.reward_parts, "z_modes": args.z_modes, "z_mode_reward": args.z_mode_reward, "z_winners": args.z_winners, "z_src": args.z_src, "eval_z": args.eval_z,
           "z_enforce": args.z_enforce, "z_time_scale": args.z_time_scale, "z_build": args.z_build, "z_build_labels": args.z_build_labels,
           "grpo_share": args.grpo_share, "grpo_group": args.grpo_group, "grpo_coef": args.grpo_coef,
           "grpo_seeds": args.grpo_seeds, "grpo_branch": [float(x) for x in args.grpo_branch.split("-")],
           "eval_map": args.eval_map, "eval_z_pool": str(args.eval_z_pool) if args.eval_z_pool else "", "eval_vs": args.eval_vs,
           "eval_guide": args.eval_guide,
           "eval_costs": order_costs(vocab, args.eval_map).tolist() if args.eval_map and args.avail_mask else None,
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
    pipes = [[ctx.Pipe() for _ in range(args.games_per_actor + args.eval_games_per_actor)] if args.central_inference
             else None for _ in range(args.actors)]
    shared_path = None
    if args.central_inference and args.infer_shm:  # the view steps and answers: two rows a game
        shared_path = f"/dev/shm/warcraftsim-{args.name}-{os.getpid()}.rows"
        c = net.config
        cfg["shared_rows"] = SharedRows(shared_path, 2 * args.actors * (args.games_per_actor + args.eval_games_per_actor),
                                        c["G"], c["n_orders"],
                                        c["d"] if net.memory else 0, create=True).spec
    actors = [ctx.Process(target=actor_main, args=(w, cfg, out_q, stop, render_q, pipes[w] and [a for a, _ in pipes[w]]),
                          daemon=True) for w in range(args.actors)]
    infer_thread = native = None
    if args.central_inference and args.infer_native:
        pass  # (set up below, after the actors started: its graphs take ~1 minute)
    elif args.central_inference and args.infer_in_learner:
        infer_thread = threading.Thread(target=inference_main, args=(cfg, [b for ps in pipes for _, b in ps], stop, True),
                                        daemon=True, name="inference")
    elif args.central_inference:
        actors.append(ctx.Process(target=inference_main, args=(cfg, [b for ps in pipes for _, b in ps], stop), daemon=True))
    if args.video_every > 0:
        actors.append(ctx.Process(target=render_main, args=(cfg, render_q, stop), daemon=True))
    for p in actors:
        p.start()
    if infer_thread is not None:
        infer_thread.start()
    if args.central_inference and args.infer_native:  # the server's loop in C++ in this process (native_infer)
        from .native_infer import NativeServer
        native = NativeServer(cfg, [b for ps in pipes for _, b in ps], device, split)
        native.start()
    _pin(cfg, "learner")  # (after the others started: they would inherit it)
    info.update(status="training", started=time.time())
    save_info()
    t0 = time.time()
    agent_steps, update, episodes = ((resumed["steps"], resumed["update"], resumed["episodes"]) if resumed
                                     else (0, 0, 0))
    bc_iter = None
    updater = GraphedUpdate(net, ref, opt, args, device) if graphed else None
    noise = NoiseScale()  # (--noise-scale)
    # the exploiter's updates as CUDA graphs too (eager, an update of 4,096 steps took the learner ~3-5 s
    # every ~15 s: a quarter of its time)
    x_updater = GraphedUpdate(xnet, ref, xopt, args, device) if graphed and xnet is not None else None
    if args.bc_data:  # demonstrations for the auxiliary cloning loss, encoded by loader workers, epoch after epoch
        from .bc import Steps, as_is
        bc_paths = sorted(p for d in args.bc_data for p in d.glob("game*.npz") if ".tmp" not in p.name)
        bc_set = Steps(bc_paths, vocab, args.bc_batch, cfg["values"], order_costs(vocab, args.map), arrays=True,
                       winners=bool(args.bc_winners), max_minutes=args.bc_max_minutes,
                       strategies=strategy_pool(cfg) if args.z_pool and args.bc_z else None)

        def bc_cycle():
            epoch = 0
            while True:
                bc_set.epoch, epoch = epoch, epoch + 1
                # (workers forked, as before Python 3.14 made forkserver the default: nothing pickled)
                for b in torch.utils.data.DataLoader(bc_set, batch_size=None, num_workers=args.bc_workers, collate_fn=as_is,
                                                     worker_init_fn=lambda _i: _pin(cfg, "games"),
                                                     multiprocessing_context="fork" if args.bc_workers else None):
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
        nonlocal xbuf, xchunks, x_update, x_since, xopt, x_stats, x_updater
        x_stats = None
        if x_updater is not None:
            try:
                x_stats = x_updater.update(xbuf, xchunks, x_since < args.value_warmup, None)
            except Exception:  # noqa: BLE001 (the eager update from here on)
                traceback.print_exc()
                print("the exploiter's CUDA-graph update failed: the eager one from now on", flush=True)
                x_updater = None
        if x_stats is None:
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
            xnet.load_state_dict(x_init)  # (in place: the graphs keep their weights' addresses)
            if x_updater is not None:  # (Adam's state zeroed in place, alike: a fresh optimizer)
                for st in xopt.state.values():
                    for v in st.values():
                        if isinstance(v, torch.Tensor):
                            v.zero_()
            else:
                xopt = torch.optim.Adam(xnet.parameters(), lr=args.lr, eps=1e-5)
            xm.recent = []
            league.exploiter_resets += 1
            x_since = 0
        torch.save({"model": {k: v.detach().cpu() for k, v in xnet.state_dict().items()}, "config": xnet.config,
                    "version": x_update, "since": x_since}, run_dir / "exploiter.pt.tmp")
        (run_dir / "exploiter.pt.tmp").replace(run_dir / "exploiter.pt")
        if x_since == 0 or x_update % args.exploiter_publish == 0:  # (a version for new games: see the flag)
            publish_exploiter(xnet, x_update, run_dir, vocab)
        league.write()

    try:
        while agent_steps < args.timesteps:
            if native is not None and native.failed():
                raise RuntimeError(f"the native inference loop failed: {native.failed()}")
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
            stats = None
            if native is not None:  # (the inference loop's statistics: rounds during the update apart)
                native.loop.learner_busy = 1
            measured = {}
            if args.memory_fade > 0 and net.config.get("ghost"):  # (published with the weights: the actors see it too)
                net.ghost_bias.fill_(-8.0 * max(0.0, 1.0 - update / args.memory_fade))
            part_stats = parts_pass(net, batch_chunks, args, device) if args.reward_parts else {}  # (--reward-parts)
            if args.noise_scale > 0 and update % args.noise_scale == 0:  # (at the weights the update starts from)
                t_noise = time.time()
                parts = ppo_update(net, ref, opt, steps, args, warmup, device, batch_chunks, None, measure=True,
                                   terms=args.noise_terms > 0 and update % args.noise_terms == 0)
                measured = {**noise.add(parts, args.minibatch), "perf/noise": time.time() - t_noise}
            if updater is not None:
                try:
                    stats = updater.update(steps, batch_chunks, warmup, bc_iter)
                except Exception:  # noqa: BLE001 (the eager update from here on)
                    traceback.print_exc()
                    print("CUDA-graph update failed: the eager one from now on", flush=True)
                    updater = None
            if stats is None:
                stats = ppo_update(net, ref, opt, steps, args, warmup, device, batch_chunks, bc_iter)
            stats.update(measured)
            stats.update(part_stats)
            if net.config.get("ghost"):
                stats["memory/ghost_bias"] = float(net.ghost_bias)
            update += 1
            agent_steps += len(steps)
            publish(net, update, run_dir)
            if native is not None:
                native.loop.learner_busy = 0
            if update % args.snapshot_every == 0:
                path = run_dir / "checkpoints" / f"{agent_steps:016d}.pt"
                save_checkpoint(path, net, opt, vocab, agent_steps, update)
                league.add_snapshot(path, agent_steps)
            elif update % args.checkpoint_every == 0:
                save_checkpoint(run_dir / "checkpoints" / f"{agent_steps:016d}.pt", net, opt, vocab, agent_steps, update)
            if args.ref_update > 0 and ref is not None and update % args.ref_update == 0:  # --ref-update: a rolling
                ref.load_state_dict(net.state_dict())  # anchor (in place: the CUDA graphs keep their weights' addresses)
                print(f"the anchor moved to update {update}", flush=True)
            if args.shaping_anneal > 0:  # the shaping's weight to 0 over --shaping-anneal updates of this run
                league.shaping = shaping_now(args, update)
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
        save_checkpoint(run_dir / "checkpoints" / f"{agent_steps:016d}.pt", net, opt, vocab, agent_steps, update)
        stop.set()
        for p in actors:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
        if native is not None:
            native.stop()
        if shared_path:
            Path(shared_path).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
