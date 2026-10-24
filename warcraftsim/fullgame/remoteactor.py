"""Remote actors: a self-play run's training games on another machine, for a learner that trains here and
there its inference (fullgame/gateway.py on the learner's machine, selfplay --remote-games). The games
and the actors' Python run here (selfplay.actor_main, game_loop and play_one, as the learner's own actors);
their network calls and their steps go to the gateway over TCP. No GPU here, no network's weights.

    python -m warcraftsim.fullgame.remoteactor --host <learner's machine> [--port 8770] --actors 8 --games-per-actor 7

* At its start (and whenever the learner comes back after a restart) the gateway sends the actors' settings
  (the learner's own, as its actors get them), vocab.json, the strategy pool and Wine's settings; they go
  into ~/.cache/warcraftsim/remote-actors/<run>/ with the league's launches (league_spec.json), asked for
  every few seconds.
* Each game thread has a connection of its own (NetInference): its calls (request(), as RemoteInference's)
  and its trajectories and episode rows (put(), as the learner's queue's). The learner gone: its calls wait
  for it (--wait seconds), then its game ends as a local actor's would.
* Settings of its own: no evaluation games (the run's machine and the evaluation worker play them), no
  videos, its own draws (the seed with this machine's name and the start's time), its own slot for the
  games' Wine prefixes.
* The shared key: wire.KEY_PATH on both machines (the gateway makes it at its first start: copy it here).

What the games need here, as for the evaluation worker: this repository's code at the learner's commit and
its built shim and native libraries, the game, and Wine (the run's, at the same path under this home).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import sys
import threading
import time
import types
from pathlib import Path

from . import wire


class NetInference:
    """A remote actor's link to the gateway: selfplay.RemoteInference's request() and the learner's queue's
    put(), each game thread over a connection of its own (opened at its first call: its slot at the gateway,
    so its two rows and its pipe to the inference server, are its own while it is open)."""

    def __init__(self, remote: dict):
        self.host, self.port = remote["host"], int(remote["port"])
        self.key, self.name = bytes.fromhex(remote["key"]), remote["name"]
        self.wait = float(remote.get("wait", 600.0))
        self.local = threading.local()
        self.nets = types.SimpleNamespace(version=-1)  # (the version the server answered with last)
        self.lost = 0  # messages a closed connection lost

    def _link(self):
        loc = self.local
        if getattr(loc, "sock", None) is not None:
            return loc.sock, loc.codec
        deadline, why = time.time() + self.wait, None
        while time.time() < deadline:
            sock = None
            try:
                sock = socket.create_connection((self.host, self.port), timeout=30)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                hello = wire.client_handshake(sock, self.key, {"kind": "game", "host": self.name})
                kind, _, _ = wire.recv_frame(sock)  # READY, or BUSY: no slot free (one closed not noticed yet)
                if kind == wire.READY:
                    sock.settimeout(self.wait)
                    loc.sock, loc.codec = sock, hello.get("codec")
                    return loc.sock, loc.codec
                why = "no free slot at the gateway"
            except wire.Refused:
                if sock is not None:
                    sock.close()
                raise
            except (OSError, wire.WireError) as e:
                why = f"{type(e).__name__}: {e}"
            if sock is not None:
                sock.close()
            time.sleep(2)
        raise wire.WireError(f"no gateway at {self.host}:{self.port} for {self.wait:.0f} s ({why})")

    def _drop(self) -> None:
        sock = getattr(self.local, "sock", None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        self.local.sock = None

    def request(self, items: list[tuple[str, dict]]) -> list[dict]:
        sock, codec = self._link()
        try:
            wire.send_frame(sock, wire.REQUEST, wire.pack_request(items), codec)
            kind, payload, _ = wire.recv_frame(sock)
            if kind != wire.REPLY:
                raise wire.WireError(f"frame {kind} instead of an answer")
            out = wire.unpack_reply(payload)
        except (OSError, wire.WireError):
            self._drop()  # (the game ends; the thread's next game connects again)
            raise
        for res in out:
            if res is not None and res.get("version", -1) >= 0:
                self.nets.version = res["version"]
        return out

    def put(self, msg: dict) -> None:
        """The learner's queue's put: a trajectory's chunk or an episode's row (lost if the connection is)."""
        sock, codec = self._link()
        try:
            wire.send_frame(sock, wire.PUT, wire.pack_value(msg), codec)
        except (OSError, wire.WireError):
            self._drop()
            self.lost += 1

    def cancel_join_thread(self) -> None:  # (the queue's: nothing to flush here)
        pass


def actor_settings(hello: dict, a, local_dir: Path, key: bytes, slot: int) -> dict:
    """The learner's actors' settings for this machine's actors (paths here, its own draws and games)."""
    cfg = dict(hello["cfg"])
    cfg.pop("shared_rows", None)
    cfg.update(run_dir=str(local_dir), vocab=str(local_dir / "vocab.json"),
               z_pool=str(local_dir / "z_pool.json") if hello["files"].get("z_pool") else "", eval_z_pool="",
               games_per_actor=a.games_per_actor, eval_games_per_actor=0, eval_games=0, actors=a.actors,
               video_every=0.0, cpus=None, device="cpu", slot=slot, learner_pid=os.getpid(),
               seed=f"{cfg.get('seed', 0)}/{a.name}", start_update=f"{hello.get('start_update', 0)}/{int(time.time())}",
               remote={"host": a.host, "port": a.port, "key": key.hex(), "name": a.name, "wait": a.wait})
    return cfg


class Control:
    """The control connection: the gateway's hello, then the league's launches when asked."""

    def __init__(self, a, key: bytes):
        self.a, self.key, self.sock, self.codec = a, key, None, None

    def connect(self) -> dict:
        while True:
            sock = None
            try:
                sock = socket.create_connection((self.a.host, self.a.port), timeout=30)
                hello = wire.client_handshake(sock, self.key, {"kind": "control", "host": self.a.name})
                kind, payload, _ = wire.recv_frame(sock)
                if kind != wire.HELLO:
                    raise wire.WireError(f"frame {kind} instead of the run's settings")
                self.sock, self.codec = sock, hello.get("codec")
                return wire.unpack_value(payload)
            except wire.Refused as e:
                raise SystemExit(f"the gateway refused this machine: {e}")
            except (OSError, wire.WireError) as e:
                if sock is not None:
                    sock.close()
                print(f"no gateway at {self.a.host}:{self.a.port} ({type(e).__name__}: {e}): again in 10 s", flush=True)
                time.sleep(10)

    def spec(self) -> bytes:
        wire.send_frame(self.sock, wire.SPEC_ASK, b"")
        kind, payload, _ = wire.recv_frame(self.sock)
        if kind != wire.SPEC:
            raise wire.WireError(f"frame {kind} instead of the league's launches")
        return payload


def write_files(hello: dict, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (("vocab.json", hello["files"]["vocab"]), ("z_pool.json", hello["files"].get("z_pool"))):
        if text:
            tmp = local_dir / f"{name}.tmp"
            tmp.write_text(text)
            tmp.replace(local_dir / name)


def write_spec(spec: bytes, local_dir: Path) -> None:
    if spec:
        tmp = local_dir / "league_spec.json.tmp"
        tmp.write_bytes(spec)
        tmp.replace(local_dir / "league_spec.json")


VOLATILE = ("learner_pid", "shared_rows", "slot", "start_update", "cpus", "device")  # (a learner's restart changes
# them, not what its actors play)


def fingerprint(hello: dict) -> str:
    """The run and the settings its actors play with: another, and this machine's actors start again."""
    cfg = {k: v for k, v in (hello.get("cfg") or {}).items() if k not in VOLATILE}
    return hashlib.sha256(wire.pack_value({"cfg": cfg, **{k: hello.get(k) for k in ("run", "files", "wine")}})).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", required=True, help="the learner's machine (its gateway: selfplay --remote-games)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--key", type=Path, default=wire.KEY_PATH, help="the shared key (the learner machine's, copied here)")
    ap.add_argument("--actors", type=int, default=8, help="actor processes (a process's games share its GIL)")
    ap.add_argument("--games-per-actor", type=int, default=7)
    ap.add_argument("--name", default=socket.gethostname(), help="this machine's name: the rows' host and part of the draws' seed")
    ap.add_argument("--wine", help="Wine's bin folder here (default: the run's, at the same path under this home)")
    ap.add_argument("--wait", type=float, default=600.0, help="seconds a game waits for the gateway before it ends")
    ap.add_argument("--spec-every", type=float, default=5.0, help="seconds between asks for the league's launches")
    a = ap.parse_args(argv)
    import multiprocessing as mp

    from .. import paths
    from ..runtime import reaper, wine
    from .collect import claim_slot
    from .evalworker import rehome
    from .selfplay import RUNS, actor_main
    key = wire.load_key(a.key)
    control = Control(a, key)
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(1))
    signal.signal(signal.SIGINT, lambda *_: stopping.append(1))
    slot, slot_lock = claim_slot(RUNS)
    ctx = mp.get_context("spawn")
    procs: list = []
    stop = None
    hello = control.connect()
    wine_set = False
    try:
        while not stopping:
            local_dir = paths.CACHE_DIR / "remote-actors" / hello["run"]
            write_files(hello, local_dir)
            try:
                write_spec(control.spec(), local_dir)
            except (OSError, wire.WireError):  # (gone again at once: wait for it)
                hello = control.connect()
                continue
            w = hello.get("wine") or {}
            if not wine_set and (a.wine or w.get("wine")):  # (before any game: they read the environment)
                wine.use_wine(a.wine or rehome(w["wine"]), bool(w.get("fsync")), bool(w.get("ntsync")))
                wine_set = True
            wine.ensure_template()
            cfg = actor_settings(hello, a, local_dir, key, slot)
            stop = ctx.Event()

            def start(i: int):
                p = ctx.Process(target=actor_main, args=(i, cfg, None, stop, None, None), daemon=True)
                p.start()
                return p
            procs = [start(i) for i in range(a.actors)]
            print(f"remote actors {a.name} for {hello['run']}: {a.actors} processes of {a.games_per_actor} games, "
                  f"gateway {a.host}:{a.port}", flush=True)
            mine = fingerprint(hello)
            while not stopping:
                time.sleep(a.spec_every)
                try:
                    write_spec(control.spec(), local_dir)
                except (OSError, wire.WireError):  # (the learner restarts: the games wait for its gateway)
                    print("gateway gone: waiting for it", flush=True)
                    hello = control.connect()
                    if fingerprint(hello) != mine:  # (another run or other settings: the actors start again)
                        print("the run's settings changed: actors restarted", flush=True)
                        break
                for i, p in enumerate(procs):
                    if not p.is_alive() and not stopping:  # (a crashed one; not one stopped with this process)
                        print(f"actor {i} exited ({p.exitcode}): restarted", flush=True)
                        procs[i] = start(i)
            stop.set()
            for p in procs:
                p.join(timeout=20)
            for p in procs:
                if p.is_alive():
                    p.terminate()
    finally:
        if stop is not None:
            stop.set()
        for p in procs:
            if p.is_alive():
                p.terminate()
        reaper.reap()
        try:
            wire.send_frame(control.sock, wire.BYE, b"")
        except (OSError, AttributeError):
            pass
        slot_lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
