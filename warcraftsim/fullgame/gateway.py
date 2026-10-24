"""Remote actors' gateway (selfplay --remote-games N): a process beside the learner's actors that stands in
for actors on other machines (fullgame/remoteactor.py there). Their games play there; their network calls
and their steps come here over TCP.

* It holds N game slots of the inference server's, as a local actor's games do: each a pipe to the server
  and two rows of the shared memory (SharedRows). A remote game's connection takes a free slot; its calls
  are written into the slot's rows and asked through its pipe, so the server (the C++ loop, or Python's)
  answers them as it answers the local games', with the same networks and the same past snapshots.
* Its trajectories and episode rows (selfplay's out_q messages) go into the learner's queue as a local
  actor's would (the episode rows with "host": the remote machine's name). The queue is bounded: when the
  learner is a batch behind, the gateway stops reading, and TCP holds the remote games back as the queue
  holds the local ones.
* A control connection a remote machine: what its actors need from the run (the actors' settings, vocab,
  the strategy pool, Wine's settings) when it connects, and the league's launches (league_spec.json) on
  request, every few seconds.
* A connection closed: its slot is free again (the call it was in finishes first, so no late answer reaches
  the next game in the slot); a slot asked for with none free is refused (BUSY).
* Only clients with the shared key (wire.KEY_PATH, on both machines) get past the handshake, and nothing they
  send is unpickled.

runs/<run>/remote-actors.json: the connections by host, calls and steps a second and bytes in and out.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from struct import error as struct_error

from . import wire


class Slots:
    """The gateway's game slots: (pipe to the inference server, its first shared row, its last call's id)."""

    def __init__(self, conns: list, first_row: int):
        self.free = [(c, first_row + 2 * i, 0) for i, c in enumerate(conns)][::-1]
        self.lock = threading.Lock()
        self.size = len(conns)

    def take(self):
        with self.lock:
            return self.free.pop() if self.free else None

    def give(self, slot) -> None:
        with self.lock:
            self.free.append(slot)


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = self.steps = self.puts = self.episodes = self.bytes_in = self.bytes_out = 0
        self.hosts: dict[str, int] = {}

    def add(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def host(self, name: str, d: int) -> None:
        with self.lock:
            self.hosts[name] = self.hosts.get(name, 0) + d
            if not self.hosts[name]:
                del self.hosts[name]


def call(conn, shared, row: int, rid: int, items: list[tuple[str, dict]], timeout: float = 300.0) -> list:
    """A remote game's network call through its slot (selfplay.RemoteInference.request's, for its rows)."""
    from .selfplay import pack_request, unpack_reply
    assert len(items) <= 2
    for k, (_, st) in enumerate(items):
        shared.write(row + k, st)
    conn.send_bytes(pack_request(rid, [(key, int(st["n"]), int(st["n_own"]), row + k) for k, (key, st) in enumerate(items)]))
    while True:
        if not conn.poll(timeout):
            raise RuntimeError("no answer from the inference server for 5 minutes")
        got, out = unpack_reply(conn.recv_bytes())
        if got == rid:  # (not an answer to a call given up on)
            break
    return [None if info is None else shared.answer(row + k, int(items[k][1]["n_own"]), info) for k, info in enumerate(out)]


class Gateway:
    def __init__(self, cfg: dict, conns: list, first_row: int, out_q, stop, port: int, bind: str = "0.0.0.0",
                 key: bytes | None = None, hello: dict | None = None, shared=None, code: str | None = "here"):
        """`hello`: what a remote machine's control connection is sent (selfplay.main: the actors' settings and
        files); `shared`: the SharedRows (default: cfg's); `code`: the code_version remote machines must run (here:
        this tree's, as the learner started; None: any)."""
        from .selfplay import SharedRows
        self.cfg, self.out_q, self.stop = cfg, out_q, stop
        self.slots = Slots(conns, first_row)
        self.shared = shared if shared is not None else SharedRows(*cfg["shared_rows"])
        self.key = key if key is not None else wire.load_key(create=True)
        self.hello = hello or {}
        self.code = wire.code_version() if code == "here" else code
        self.stats = Stats()
        self.run_dir = Path(cfg["run_dir"])
        self.server = socket.create_server((bind, port), reuse_port=False)
        self.server.settimeout(1.0)
        self.port = self.server.getsockname()[1]

    def serve_forever(self, alive=lambda: True) -> None:
        t_status = time.time()
        while not self.stop.is_set() and alive():
            try:
                sock, addr = self.server.accept()
            except socket.timeout:
                sock = None
            if sock is not None:
                threading.Thread(target=self.serve, args=(sock, addr), daemon=True).start()
            if time.time() - t_status > 30:
                t_status = time.time()
                self.status()
        self.server.close()

    def serve(self, sock: socket.socket, addr) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # (a machine gone without a word: its slots
        for opt, v in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 6)):  # free in ~2 minutes)
            if hasattr(socket, opt):
                sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), v)
        sock.settimeout(60.0)  # (the handshake's)
        try:
            peer = wire.server_handshake(sock, self.key, {"run": self.run_dir.name}, self.code)
        except (OSError, ValueError, wire.WireError) as e:
            print(f"gateway: {addr[0]} refused: {e}", flush=True)
            sock.close()
            return
        sock.settimeout(None)
        host = str(peer.get("host") or addr[0])
        try:
            if peer.get("kind") == "control":
                self.control(sock, host, peer["codec"])
            elif peer.get("kind") == "game":
                self.game(sock, host, peer["codec"])
        except (OSError, wire.WireError, ValueError, KeyError, RuntimeError, struct_error) as e:  # (struct: a frame cut short)
            if not isinstance(e, wire.WireError) or str(e) != "connection closed":
                print(f"gateway: {host}'s {peer.get('kind')} connection: {type(e).__name__}: {e}", flush=True)
        finally:
            sock.close()

    def control(self, sock: socket.socket, host: str, codec: str | None) -> None:
        """What the remote machine's actors need, then the league's launches when asked."""
        print(f"gateway: {host} connected", flush=True)
        wire.send_frame(sock, wire.HELLO, wire.pack_value(self.hello), codec)
        try:
            while True:
                kind, _, _ = wire.recv_frame(sock)
                if kind == wire.SPEC_ASK:
                    try:
                        spec = (self.run_dir / "league_spec.json").read_bytes()
                    except FileNotFoundError:
                        spec = b""
                    wire.send_frame(sock, wire.SPEC, spec, codec)
                elif kind == wire.BYE:
                    return
        finally:
            print(f"gateway: {host} gone", flush=True)

    def game(self, sock: socket.socket, host: str, codec: str | None) -> None:
        """A remote game's calls and steps through a slot of its own while it is connected."""
        slot = self.slots.take()
        if slot is None:
            wire.send_frame(sock, wire.BUSY, b"")
            return
        conn, row, rid = slot
        self.stats.host(host, 1)
        try:
            wire.send_frame(sock, wire.READY, b"")
            while True:
                kind, payload, n_in = wire.recv_frame(sock)
                if kind == wire.REQUEST:
                    items = wire.unpack_request(payload)
                    rid += 1
                    out = call(conn, self.shared, row, rid, items)
                    n_out = wire.send_frame(sock, wire.REPLY, wire.pack_reply(out), codec)
                    self.stats.add(calls=1, bytes_in=n_in, bytes_out=n_out)
                elif kind == wire.PUT:
                    msg = wire.unpack_value(payload)
                    if not isinstance(msg, dict) or not ("steps" in msg or "episode" in msg):
                        raise ValueError("not an actor's message")
                    if "episode" in msg:
                        msg["episode"]["host"] = host
                    self.out_q.put(msg)  # (blocks while the learner is a batch behind: so do the remote games)
                    self.stats.add(puts=1, steps=len(msg.get("steps", ())), episodes="episode" in msg, bytes_in=n_in)
                elif kind == wire.BYE:
                    return
                else:
                    raise ValueError(f"frame {kind} from a game")
        finally:
            self.stats.host(host, -1)
            self.slots.give((conn, row, rid))

    def status(self) -> None:
        s = self.stats
        with s.lock:
            now = time.time()
            last = getattr(self, "_last", None)
            row = {"time": now, "port": self.port, "slots": self.slots.size, "free": len(self.slots.free),
                   "games_by_host": dict(s.hosts), "calls": s.calls, "steps": s.steps, "episodes": s.episodes,
                   "bytes_in": s.bytes_in, "bytes_out": s.bytes_out}
            if last:
                dt = max(now - last["time"], 1e-9)
                row.update({f"{k}_per_s": round((row[k] - last[k]) / dt, 1)
                            for k in ("calls", "steps", "episodes", "bytes_in", "bytes_out")})
            self._last = row
        tmp = self.run_dir / "remote-actors.json.tmp"
        tmp.write_text(json.dumps(row, indent=1))
        tmp.replace(self.run_dir / "remote-actors.json")



def gateway_main(cfg: dict, conns: list, first_row: int, out_q, stop, opts: dict) -> None:
    """selfplay's process for --remote-games: `opts` port, bind, key path and the hello (gateway.Gateway)."""
    from .selfplay import _exit
    signal.signal(signal.SIGTERM, _exit)
    sys.setswitchinterval(0.001)  # (a call's thread woken by its answer waits for the GIL: 5 ms by default)
    out_q.cancel_join_thread()
    key = wire.load_key(opts.get("key"), create=True)
    gw = Gateway(cfg, conns, first_row, out_q, stop, opts["port"], opts.get("bind", "0.0.0.0"), key, opts.get("hello"))
    print(f"gateway: {len(conns)} remote game slots on port {gw.port} (key {opts.get('key') or wire.KEY_PATH})", flush=True)
    gw.serve_forever(lambda: os.getppid() == cfg["learner_pid"])
