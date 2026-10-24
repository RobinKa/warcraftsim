"""Remote actors (fullgame/wire.py, gateway.py, remoteactor.py): the wire format's round trips, a remote game's
calls through the gateway's slot to an inference server (a fake one here: no GPU, no games) and back, its steps
into the learner's queue, and slots freed and refused as they should be."""

import multiprocessing as mp
import queue
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import wire


def _st(rng, n=37, n_own=12, F=None, G=40, NO=203, d=16, h=True):
    F = F or fx.F
    return {"n": n, "n_own": n_own, "ent": rng.normal(size=(n, F)).astype(np.float32),
            "type": rng.integers(0, 150, n), "cur": rng.integers(0, 400, n),
            "glob": rng.normal(size=G).astype(np.float32), "avail": rng.random(NO) < 0.5,
            "h": rng.normal(size=d).astype(np.float32) if h else None}


def test_values_round_trip():
    x = {"steps": [{"ent": np.arange(12, dtype=np.float16).reshape(3, 4), "type": np.array([1, 2], np.int16),
                    "ok": np.array([True, False]), "value": 0.25, "v32": np.float32(1.5), "n": np.int64(7),
                    "h": None, "done": False}],
         "info": {"learner": "current", "pair": (1, "a"), 3: "int key", "__a": "a key like a tag"},
         "raw": b"\x00\x01", "path": Path("/x/y"), "nan": float("inf")}
    y = wire.unpack_value(wire.pack_value(x))
    s = y["steps"][0]
    assert s["ent"].dtype == np.float16 and np.array_equal(s["ent"], x["steps"][0]["ent"])
    assert s["type"].dtype == np.int16 and s["ok"].dtype == bool and s["ok"].tolist() == [True, False]
    assert type(s["v32"]) is np.float32 and s["v32"] == np.float32(1.5) and type(s["n"]) is np.int64
    assert s["value"] == 0.25 and s["h"] is None and s["done"] is False
    assert y["info"] == {"learner": "current", "pair": (1, "a"), 3: "int key", "__a": "a key like a tag"}
    assert y["raw"] == b"\x00\x01" and y["path"] == "/x/y" and y["nan"] == float("inf")
    with pytest.raises(TypeError):
        wire.pack_value({"x": object()})
    with pytest.raises(TypeError):
        wire.pack_value(np.array([object()]))


def test_request_round_trip():
    rng = np.random.default_rng(0)
    a, b = _st(rng), _st(rng, n=160, n_own=96, NO=201, h=False)
    b["shared_noise"], b["guide"] = True, 0.5
    out = wire.unpack_request(wire.pack_request([("current", a), ("past:runs/x/1.pt", b)]))
    assert [k for k, _ in out] == ["current", "past:runs/x/1.pt"]
    for (_, got), want in zip(out, (a, b)):
        n = want["n"]
        assert got["n"] == n and got["n_own"] == want["n_own"]
        assert np.array_equal(got["ent"], want["ent"][:n].astype(np.float16))  # (float16, as the learner keeps them)
        assert np.array_equal(got["type"], want["type"][:n]) and np.array_equal(got["cur"], want["cur"][:n])
        assert np.array_equal(got["glob"], want["glob"]) and np.array_equal(got["avail"], want["avail"])
        assert (got.get("h") is None) == (want["h"] is None)
    assert np.array_equal(out[0][1]["h"], a["h"])
    assert out[1][1]["shared_noise"] and out[1][1]["guide"] == 0.5 and "shared_noise" not in out[0][1]


def test_reply_round_trip():
    res = {"order": np.array([3, 0, 7], np.int64), "tgt": np.array([1, 2, 3], np.int64), "bx": np.array([5, 6, 7]),
           "by": np.array([9, 9, 9]), "logp": np.array([-0.5, -1.25, -3.0], np.float32),
           "value": float(np.float32(0.3)), "entropy": float(np.float32(1.7)), "version": 12,
           "h": np.arange(4, dtype=np.float32)}
    got = wire.unpack_reply(wire.pack_reply([res, None, {**res, "h": None}]))
    assert got[1] is None and "h" not in got[2]
    for k in ("order", "tgt", "bx", "by"):
        assert got[0][k].dtype == np.int64 and np.array_equal(got[0][k], res[k])
    assert got[0]["logp"].dtype == np.float32 and np.array_equal(got[0]["logp"], res["logp"])
    assert got[0]["value"] == res["value"] and got[0]["entropy"] == res["entropy"] and got[0]["version"] == 12
    assert np.array_equal(got[0]["h"], res["h"])


def test_handshake_needs_the_key():
    key = bytes(range(32))
    for theirs, ok in ((key, True), (bytes(32), False)):
        a, b = socket.socketpair()
        got = {}

        def server():
            try:
                got["hello"] = wire.server_handshake(a, key, {"run": "r"})
            except wire.WireError as e:
                got["error"] = e
                a.close()
        th = threading.Thread(target=server)
        th.start()
        if ok:
            assert wire.client_handshake(b, theirs, {"kind": "game", "host": "h"})["run"] == "r"
        else:
            with pytest.raises((wire.WireError, OSError)):
                wire.client_handshake(b, theirs, {"kind": "game", "host": "h"})
        th.join(5)
        assert ("hello" in got) == ok and (not ok or got["hello"]["host"] == "h")
        b.close()


class FakeServer:
    """The inference server's side of the slots' pipes (selfplay's binary messages): each answer computed from
    the rows the call was written to, so a test can tell the gateway wrote them where it said."""

    def __init__(self, shared, conns, version=3):
        self.shared, self.conns, self.version, self.calls = shared, conns, version, 0
        self.stop = threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        from multiprocessing.connection import wait

        from warcraftsim.fullgame.selfplay import pack_reply, unpack_request
        sh, O = self.shared, fx.MAX_OWN
        d = sh.h.shape[1]
        while not self.stop.is_set():
            for c in wait(self.conns, timeout=0.1):
                try:
                    rid, items = unpack_request(c.recv_bytes())
                except EOFError:
                    self.conns.remove(c)
                    continue
                for key, n, n_own, r in items:
                    o = min(n_own, O)
                    p = sh.out[r]
                    p[:] = 0
                    p[0:o] = (np.arange(o) + n) % 7
                    p[O:O + o] = np.arange(o) % max(n, 1)
                    p[4 * O:4 * O + o] = sh.ent[r, :o, 0]
                    p[5 * O], p[5 * O + 1] = sh.glob[r, 0], float(n)
                    p[5 * O + 2:5 * O + 2 + d] = 2 * sh.h[r]
                self.calls += 1
                c.send_bytes(pack_reply(rid, [(self.version, O, d)] * len(items)))


@pytest.fixture
def gateway(tmp_path):
    from warcraftsim.fullgame.gateway import Gateway
    from warcraftsim.fullgame.selfplay import SharedRows
    d, slots = 16, 2
    sh = SharedRows(str(tmp_path / "rows"), 2 * (1 + slots), 40, 203, d, create=True)  # (a local game's two rows first)
    pipes = [mp.Pipe() for _ in range(slots)]
    server = FakeServer(sh, [b for _, b in pipes])
    key = bytes(range(32))
    out_q, stop = queue.Queue(), threading.Event()
    gw = Gateway({"run_dir": str(tmp_path)}, [a for a, _ in pipes], 2, out_q, stop, 0, "127.0.0.1", key,
                 {"run": "test", "cfg": {"x": (1, 2)}}, shared=sh)
    th = threading.Thread(target=gw.serve_forever, daemon=True)
    th.start()
    yield gw, server, key, out_q
    stop.set()
    server.stop.set()
    th.join(5)


def _client(gw, key, wait=5.0):
    from warcraftsim.fullgame.remoteactor import NetInference
    return NetInference({"host": "127.0.0.1", "port": gw.port, "key": key.hex(), "name": "box", "wait": wait})


def test_calls_through_the_gateway(gateway):
    gw, server, key, out_q = gateway
    rng = np.random.default_rng(1)
    net = _client(gw, key)
    for _ in range(3):
        a, b = _st(rng, G=40), _st(rng, n=90, n_own=40, G=40)
        out = net.request([("current", a), ("current", b)])
        for res, st in zip(out, (a, b)):
            o = min(st["n_own"], fx.MAX_OWN)
            assert res["version"] == 3 and len(res["order"]) == o and res["order"].dtype == np.int64
            assert np.array_equal(res["order"], (np.arange(o) + st["n"]) % 7)
            assert np.array_equal(res["logp"], st["ent"][:o, 0].astype(np.float16).astype(np.float32))
            assert res["value"] == float(st["glob"][0]) and res["entropy"] == float(st["n"])
            assert np.array_equal(res["h"], 2 * st["h"])
    assert net.nets.version == 3 and server.calls == 3
    assert len(gw.slots.free) == 1  # (its slot held while connected)


def test_steps_and_rows_into_the_queue(gateway):
    gw, _, key, out_q = gateway
    net = _client(gw, key)
    steps = [{"ent": np.ones((5, fx.F), np.float16), "order": np.array([1, 2], np.int16), "value": 0.5,
              "reward": np.float64(1.0), "h": None, "done": True, "version": 4, "adv": 0.1, "ret": 0.6}]
    net.put({"steps": steps, "info": {"learner": "current", "opponent": "self"}})
    net.put({"episode": {"outcome": 1.0, "race": "human", "opponent": "self"}})
    msg = out_q.get(timeout=5)
    assert msg["info"]["learner"] == "current" and np.array_equal(msg["steps"][0]["ent"], steps[0]["ent"])
    assert msg["steps"][0]["ent"].dtype == np.float16 and msg["steps"][0]["reward"] == 1.0
    ep = out_q.get(timeout=5)["episode"]
    assert ep["outcome"] == 1.0 and ep["host"] == "box"  # (the remote machine's name in its rows)


def test_slots_freed_and_refused(gateway):
    gw, _, key, _ = gateway
    rng = np.random.default_rng(2)
    a = _client(gw, key)
    b = _client(gw, key)
    a.request([("current", _st(rng, G=40))])
    b.request([("current", _st(rng, G=40))])
    assert len(gw.slots.free) == 0
    c = _client(gw, key, wait=1.0)
    with pytest.raises(wire.WireError):  # (both slots taken: refused, then given up)
        c.request([("current", _st(rng, G=40))])
    a.local.sock.close()  # (a remote game gone: its slot free again)
    for _ in range(50):
        if len(gw.slots.free) == 1:
            break
        time.sleep(0.05)
    assert len(gw.slots.free) == 1
    c.request([("current", _st(rng, G=40))])  # (and taken by the next)
    assert len(gw.slots.free) == 0


def test_other_code_refused(tmp_path):
    """A machine whose code is not the learner's is told so, at once (no retries), and gets no slot."""
    from warcraftsim.fullgame.gateway import Gateway
    from warcraftsim.fullgame.remoteactor import NetInference
    from warcraftsim.fullgame.selfplay import SharedRows
    sh = SharedRows(str(tmp_path / "rows"), 2, 40, 203, 16, create=True)
    key, stop = bytes(range(32)), threading.Event()
    gw = Gateway({"run_dir": str(tmp_path)}, [mp.Pipe()[0]], 0, queue.Queue(), stop, 0, "127.0.0.1", key, {},
                 shared=sh, code="0" * 64)
    threading.Thread(target=gw.serve_forever, daemon=True).start()
    try:
        net = NetInference({"host": "127.0.0.1", "port": gw.port, "key": key.hex(), "name": "box", "wait": 30})
        t0 = time.time()
        with pytest.raises(wire.Refused, match="not the learner's"):
            net.request([("current", _st(np.random.default_rng(0), G=40))])
        assert time.time() - t0 < 5 and len(gw.slots.free) == 1
        assert wire.code_version() == wire.code_version() and len(wire.code_version()) == 64
    finally:
        stop.set()


def test_wrong_key_gets_nothing(gateway):
    gw, _, _, out_q = gateway
    net = _client(gw, bytes(32), wait=1.0)
    with pytest.raises(wire.WireError):
        net.put({"episode": {"outcome": 1.0}})
    assert out_q.empty() and len(gw.slots.free) == 2


def test_control_hello_and_spec(gateway, tmp_path):
    gw, _, key, _ = gateway
    from warcraftsim.fullgame.remoteactor import Control
    (tmp_path / "league_spec.json").write_text('{"launch": [], "agents": []}')
    a = type("A", (), {"host": "127.0.0.1", "port": gw.port, "name": "box"})()
    ctl = Control(a, key)
    hello = ctl.connect()
    assert hello == {"run": "test", "cfg": {"x": (1, 2)}}
    assert ctl.spec() == b'{"launch": [], "agents": []}'


def test_the_real_server_behind_the_gateway(tmp_path):
    """The inference server itself (selfplay.inference_main, a thread here, on the CPU) answers a remote game
    through the gateway's slot as it would a local one: the values and the memory's states of a direct call on
    the same (float16) features."""
    import os

    import torch

    from warcraftsim.fullgame.gateway import Gateway
    from warcraftsim.fullgame.model import FullGameNet
    from warcraftsim.fullgame.selfplay import Inference, SharedRows, inference_main, publish
    torch.manual_seed(0)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=64, layers=1, memory=True).eval()
    net.allowed[:] = True
    publish(net, 5, tmp_path)
    spec = SharedRows(str(tmp_path / "rows"), 6, 30, 30, 64, create=True).spec
    pipes = [mp.get_context("spawn").Pipe() for _ in range(3)]  # (a local game's, then the gateway's two slots)
    stop = threading.Event()
    cfg = {"run_dir": str(tmp_path), "device": "cpu", "learner_pid": os.getpid(), "infer_batch": 8, "compile": False,
           "max_past": 2, "shared_rows": spec, "infer_pipeline": 1}
    server = threading.Thread(target=inference_main, args=(cfg, [b for _, b in pipes], stop, True), daemon=True)
    server.start()
    key = bytes(range(32))
    gw = Gateway(cfg, [a for a, _ in pipes[1:]], 2, queue.Queue(), stop, 0, "127.0.0.1", key, {})
    threading.Thread(target=gw.serve_forever, daemon=True).start()
    try:
        remote = _client(gw, key)
        direct = Inference(None, torch.device("cpu"), 8, compile_=False, thread=False)
        rng = np.random.default_rng(3)
        for n in (9, 4, 13):
            st = {"n": n, "n_own": 3, "ent": rng.normal(size=(n, fx.F)).astype(np.float32),
                  "type": rng.integers(1, 20, n), "cur": rng.integers(0, 10, n),
                  "glob": rng.normal(size=30).astype(np.float32), "avail": None,
                  "h": rng.normal(size=64).astype(np.float32)}
            got = remote.request([("current", st)])[0]
            want = direct._finish(direct._launch(net, [{**st, "ent": st["ent"].astype(np.float16).astype(np.float32)}]))[0]
            assert got["version"] == 5 and len(got["order"]) == 3
            assert abs(got["value"] - want["value"]) < 1e-5
            assert np.allclose(got["h"], want["h"], atol=1e-5)
    finally:
        stop.set()
        server.join(timeout=20)


def test_actor_settings_and_restarts(tmp_path):
    """This machine's actors: the learner's settings with paths here, no evaluation games or videos, draws of
    their own; a learner's restart with the same settings keeps them, other settings start them again."""
    import os
    import types

    from warcraftsim.fullgame.remoteactor import actor_settings, fingerprint
    cfg = {"seed": 3, "shared_rows": ("/dev/shm/x", 4, 30, 30, 64), "learner_pid": 1, "games_per_actor": 7,
           "eval_games_per_actor": 1, "eval_games": 8, "video_every": 2.0, "z_pool": "/main/z.json", "map": "echofast",
           "vocab": "/main/v.json", "run_dir": "/main/runs/r", "slot": 0, "start_update": 5, "actors": 8}
    hello = {"run": "r", "cfg": cfg, "files": {"vocab": "{}", "z_pool": "{}"}, "wine": {"wine": "/w", "fsync": True}}
    a = types.SimpleNamespace(games_per_actor=3, actors=2, name="box", host="h", port=1, wait=5.0)
    got = actor_settings(hello, a, tmp_path, bytes(32), 4)
    assert "shared_rows" not in got and got["run_dir"] == str(tmp_path) and got["vocab"] == str(tmp_path / "vocab.json")
    assert got["z_pool"] == str(tmp_path / "z_pool.json") and got["eval_games"] == got["eval_games_per_actor"] == 0
    assert got["video_every"] == 0 and got["games_per_actor"] == 3 and got["actors"] == 2 and got["slot"] == 4
    assert got["learner_pid"] == os.getpid() and got["seed"] == "3/box" and got["remote"]["name"] == "box"
    assert cfg["learner_pid"] == 1  # (the hello's own left as it was)
    restarted = {**hello, "cfg": {**cfg, "learner_pid": 2, "start_update": 9, "shared_rows": ("/dev/shm/y", 4, 30, 30, 64)}}
    assert fingerprint(restarted) == fingerprint(hello)
    assert fingerprint({**hello, "cfg": {**cfg, "map": "echo100"}}) != fingerprint(hello)
