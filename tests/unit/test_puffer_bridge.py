"""The bridge server's wire protocol, driven like the C environment does, against a fake game."""

import json
import socket
import struct
import threading
import time

import numpy as np

from warcraftsim.puffer.bridge import MAGIC, VERSION, BridgeServer
from warcraftsim.puffer.tasks import Task


class FakeEnv:
    """Episode of 3 steps; reward = action; win when the action sum is >= 2."""

    def __init__(self, name):
        self.t = 0
        self.total = 0.0

    def reset(self, options=None):
        self.t, self.total = 0, 0.0
        return np.array([0.0, 1.0], np.float32), {"obs": None}

    def step(self, action):
        self.t += 1
        self.total += action
        done = self.t >= 3
        return np.array([self.t, 1.0], np.float32), float(action), done, False, {"obs": None, "total": self.total}

    def close(self):
        pass


def _task() -> Task:
    return Task(name="fake", obs_size=2, act_sizes=(3,), make_env=FakeEnv,
                flatten=lambda o: np.asarray(o, np.float32), to_action=lambda a: int(a[0]),
                outcome=lambda env, info: 1.0 if info["total"] >= 2 else -1.0)


def _recv(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        assert chunk, "bridge closed the connection"
        buf += chunk
    return buf


def test_bridge_protocol(tmp_path):
    sock_path = str(tmp_path / "b.sock")
    bridge = BridgeServer(_task(), 1, tmp_path / "run", sock_path, record_every=0, video_every=0)
    bridge.launch_games(log=lambda m: None)
    bridge.serve()
    try:
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.connect(sock_path)
        c.sendall(struct.pack("<4I", MAGIC, VERSION, 2, 1) + b"fake".ljust(32, b"\0"))
        assert struct.unpack("<II", _recv(c, 8)) == (1, 0)  # one agent, no action masks
        c.sendall(struct.pack("<I", 1))
        assert np.frombuffer(_recv(c, 8), np.float32).tolist() == [0.0, 1.0]
        results = []
        for action in (1, 1, 1):
            c.sendall(struct.pack("<If", 2, action))
            obs = np.frombuffer(_recv(c, 8), np.float32)
            reward, terminal = struct.unpack("<ff", _recv(c, 8))
            stats = struct.unpack("<4f", _recv(c, 16))
            results.append((obs.tolist(), reward, terminal, stats))
        assert results[0][2] == 0.0 and results[0][3][0] == 0.0
        obs, reward, terminal, stats = results[2]
        assert terminal == 1.0 and obs == [0.0, 1.0]  # auto-reset: first obs of the next episode
        assert stats == (1.0, 3.0, 3.0, 1.0)  # ended, return, length, win
        episodes = [json.loads(l) for l in (tmp_path / "run" / "episodes-0.jsonl").read_text().splitlines()]
        assert episodes[0]["return"] == 3.0 and episodes[0]["outcome"] == 1.0
        hello = struct.pack("<4I", MAGIC, VERSION, 2, 1) + b"fake".ljust(32, b"\0")
        # a second environment is refused while the only game is in use ...
        c2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c2.connect(sock_path)
        c2.sendall(hello)
        assert struct.unpack("<II", _recv(c2, 8)) == (0, 0)
        c2.close()
        # ... and served once the first trainer is gone (a restarted trainer reuses the games)
        c.close()
        for _ in range(100):
            with bridge._lock:
                if bridge._free:
                    break
            time.sleep(0.01)
        c3 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c3.connect(sock_path)
        c3.sendall(hello)
        assert struct.unpack("<II", _recv(c3, 8)) == (1, 0)
        c3.sendall(struct.pack("<I", 1))
        assert np.frombuffer(_recv(c3, 8), np.float32).tolist() == [0.0, 1.0]
        c3.close()
    finally:
        bridge.close()


def test_dashboard_jsonl_cache_rereads_replaced_files(tmp_path):
    import json
    from warcraftsim.dashboard.server import _JsonlCache
    f = tmp_path / "media.jsonl"
    cache = _JsonlCache()
    f.write_text(json.dumps({"episode": 1, "time": 1}) + "\n")
    assert cache.read(f) == [{"episode": 1, "time": 1}]
    f.unlink()  # a run deleted and started again under its name: a longer new file
    f.write_text(json.dumps({"episode": 2, "time": 2}) + "\n" + json.dumps({"episode": 1, "time": 3}) + "\n")
    assert [r["episode"] for r in cache.read(f)] == [2, 1]
    with open(f, "a") as out:
        out.write(json.dumps({"episode": 3, "time": 4}) + "\n")
    assert [r["episode"] for r in cache.read(f)] == [2, 1, 3]


def test_dashboard_api(tmp_path):
    import urllib.request

    from warcraftsim.dashboard.server import serve

    run = tmp_path / "r1"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"name": "r1", "task": "nav", "envs": 2, "timesteps": 100,
                                              "created": 1.0, "status": "training"}))
    (run / "train.jsonl").write_text('{"time": 1, "agent_steps": 64, "SPS": 10, "loss/policy": 0.1}\n{"partial')
    (run / "episodes.jsonl").write_text("".join(json.dumps({"time": i, "episode": i + 1, "env": 0, "return": i,
                                                            "length": 5, "outcome": 1 if i % 2 else -1,
                                                            "total_steps": 10 * i}) + "\n" for i in range(20)))
    (run / "renders").mkdir()
    (run / "renders" / "e.html").write_text("<canvas>")
    (run / "media.jsonl").write_text(json.dumps({"kind": "render", "file": "renders/e.html", "episode": 1,
                                                 "outcome": 1, "return": 1}) + "\n")
    server = serve(tmp_path, "127.0.0.1", 0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        runs = json.load(urllib.request.urlopen(base + "/api/runs"))
        assert runs[0]["name"] == "r1" and runs[0]["summary"]["episodes"] == 20
        assert runs[0]["summary"]["win_rate_100"] == 0.5
        data = json.load(urllib.request.urlopen(base + "/api/runs/r1"))
        assert len(data["train"]) == 1  # the partial last line is ignored
        assert data["media"][0]["file"] == "renders/e.html"
        assert urllib.request.urlopen(base + "/files/r1/renders/e.html").read() == b"<canvas>"
        assert b"warcraftsim" in urllib.request.urlopen(base + "/").read()
        try:
            urllib.request.urlopen(base + "/files/../../etc/passwd")
            raise AssertionError("path traversal must fail")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()


def test_bridge_recovers_from_a_failed_game(tmp_path):
    from warcraftsim.runtime.instance import GameCrashed

    class _Inst:
        def close(self):
            pass

    class _Game:
        instance = _Inst()

    class FlakyEnv(FakeEnv):
        failures = 1

        def __init__(self, name):
            super().__init__(name)
            self.game = _Game()

        def step(self, action):
            if FlakyEnv.failures:
                FlakyEnv.failures -= 1
                raise GameCrashed("game connection closed")
            return super().step(action)

    task = _task()
    task.make_env = FlakyEnv
    bridge = BridgeServer(task, 1, tmp_path / "run", str(tmp_path / "b.sock"), record_every=0, video_every=0)
    bridge.launch_games(log=lambda m: None)
    slot = bridge.slots[0]
    bridge._reset(slot)
    obs, rewards, done, stats = bridge._step(slot, [np.array([1.0], np.float32)])
    assert done and stats[0][0] == 1.0 and stats[0][3] == 0.0  # truncated, outcome "other"
    assert obs[0].tolist() == [0.0, 1.0]  # the relaunched game's first observation
    obs, rewards, done, stats = bridge._step(slot, [np.array([1.0], np.float32)])
    assert not done and rewards == [1.0]  # and play continues
    log = (tmp_path / "run" / "episodes-0.jsonl").read_text()
    assert "game_restart" in log
    bridge.close()


class FakeSelfPlayEnv:
    """Two agents; 2 steps; agent 0 wins. Mirrors MicroSelfPlayEnv's dict-per-agent API."""

    class _Player:
        def __init__(self, result):
            self.result = type("R", (), {"name": result})()

    def __init__(self, name):
        self.t = 0

    def reset(self, options=None):
        self.t = 0
        return {0: np.array([0.0], np.float32), 1: np.array([10.0], np.float32)}, {0: {"obs": None}, 1: {}}

    def step(self, actions):
        self.t += 1
        done = self.t >= 2
        obs = type("O", (), {"players": [self._Player("VICTORY"), self._Player("DEFEAT")], "game_time": 0.5 * self.t,
                             "game_over": done, "units": []})()
        return ({0: np.array([self.t], np.float32), 1: np.array([10.0 + self.t], np.float32)},
                {0: float(actions[0]), 1: float(actions[1])}, {0: done, 1: done}, {0: False, 1: False},
                {0: {"obs": obs}, 1: {}})

    def close(self):
        pass


def test_bridge_two_agents(tmp_path):
    task = Task(name="fake2", obs_size=1, act_sizes=(3,), make_env=FakeSelfPlayEnv,
                flatten=lambda o: np.asarray(o, np.float32), to_action=lambda a: int(a[0]), num_agents=2)
    sock_path = str(tmp_path / "b.sock")
    bridge = BridgeServer(task, 1, tmp_path / "run", sock_path, record_every=0, video_every=0)
    bridge.launch_games(log=lambda m: None)
    bridge.serve()
    try:
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.connect(sock_path)
        c.sendall(struct.pack("<4I", MAGIC, VERSION, 1, 1) + b"fake2".ljust(32, b"\0"))
        assert struct.unpack("<II", _recv(c, 8)) == (2, 0)
        c.sendall(struct.pack("<I", 1))
        assert np.frombuffer(_recv(c, 8), np.float32).tolist() == [0.0, 10.0]
        for t in (1, 2):
            c.sendall(struct.pack("<Iff", 2, 1.0, 2.0))
            obs = np.frombuffer(_recv(c, 8), np.float32).tolist()
            rewards = struct.unpack("<2f", _recv(c, 8))
            terminals = struct.unpack("<2f", _recv(c, 8))
            stats = [struct.unpack("<4f", _recv(c, 16)) for _ in range(2)]
            assert rewards == (1.0, 2.0)
        assert terminals == (1.0, 1.0) and obs == [0.0, 10.0]  # auto-reset
        assert stats == [(1.0, 2.0, 2.0, 1.0), (1.0, 4.0, 2.0, -1.0)]
        c.close()
    finally:
        bridge.close()


def test_bridge_action_masks(tmp_path):
    """With Task.action_mask, each observation is followed by the masks (a byte per option)."""
    task = Task(name="fakem", obs_size=2, act_sizes=(3, 2), make_env=FakeEnv,
                flatten=lambda o: np.asarray(o, np.float32), to_action=lambda a: int(a[0]),
                outcome=lambda env, info: 0.0,
                action_mask=lambda env: [np.array([1, 0, 1, 1, env.t % 2], np.uint8)])
    sock_path = str(tmp_path / "b.sock")
    bridge = BridgeServer(task, 1, tmp_path / "run", sock_path, record_every=0, video_every=0)
    bridge.launch_games(log=lambda m: None)
    bridge.serve()
    try:
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.connect(sock_path)
        c.sendall(struct.pack("<4I", MAGIC, VERSION, 2, 2) + b"fakem".ljust(32, b"\0"))
        assert struct.unpack("<II", _recv(c, 8)) == (1, 5)
        c.sendall(struct.pack("<I", 1))
        assert np.frombuffer(_recv(c, 8), np.float32).tolist() == [0.0, 1.0]
        assert list(_recv(c, 5)) == [1, 0, 1, 1, 0]
        c.sendall(struct.pack("<Iff", 2, 1.0, 0.0))
        assert np.frombuffer(_recv(c, 8), np.float32).tolist() == [1.0, 1.0]
        assert list(_recv(c, 5)) == [1, 0, 1, 1, 1]  # the state after the step
        struct.unpack("<ff4f", _recv(c, 24))
        c.close()
    finally:
        bridge.close()
