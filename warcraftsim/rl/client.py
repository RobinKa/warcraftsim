"""The bridge's environment protocol (puffer/wc3_bridge.h) from Python: many games stepped together.

Each environment is one game served by a bridge worker thread over a Unix socket. A step sends every
environment its actions first and then reads the results, so the games step in parallel. Numpy only.
"""

from __future__ import annotations

import socket
import struct
import time

import numpy as np

MAGIC = 0x46503357  # "W3PF"
VERSION = 2


def _recv(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("bridge connection closed")
        buf += chunk
    return bytes(buf)


class BridgeEnvs:
    """`n` environments of `task` (a name) spread round-robin over the bridge `sockets`.

    Arrays are per agent: an environment of a self-play task has 2 agents (agent a of environment
    e is row e * agents + a). The bridge resets an environment itself when its episode ends: the
    observations after a terminal step are the new episode's first."""

    def __init__(self, sockets: list[str], n: int, task: str, obs_size: int, num_atns: int, timeout: float = 600):
        self.obs_size, self.num_atns = obs_size, num_atns
        self.conns: list[socket.socket] = []
        agents, masks = set(), set()
        for e in range(n):
            path = sockets[e % len(sockets)]
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            t0 = time.time()
            while True:
                try:
                    conn.connect(path)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.time() - t0 > timeout:
                        raise
                    time.sleep(0.1)
            conn.sendall(struct.pack("<4I", MAGIC, VERSION, obs_size, num_atns) + task.encode()[:31].ljust(32, b"\0"))
            a, m = struct.unpack("<II", _recv(conn, 8))
            if a == 0:
                raise RuntimeError(f"the bridge refused environment {e} (task {task}, obs {obs_size}, atns {num_atns})")
            agents.add(a)
            masks.add(m)
            self.conns.append(conn)
        assert len(agents) == 1 and len(masks) == 1, "environments differ in agents or masks"
        self.agents, self.mask_size = agents.pop(), masks.pop()
        self.n = n * self.agents

    def _read_obs(self, conn) -> tuple[np.ndarray, np.ndarray]:
        a = self.agents
        obs = np.frombuffer(_recv(conn, a * self.obs_size * 4), np.float32).reshape(a, self.obs_size)
        masks = (np.frombuffer(_recv(conn, a * self.mask_size), np.uint8).reshape(a, self.mask_size)
                 if self.mask_size else np.ones((a, 0), np.uint8))
        return obs, masks

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        for conn in self.conns:
            conn.sendall(struct.pack("<I", 1))
        parts = [self._read_obs(conn) for conn in self.conns]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def step(self, actions: np.ndarray):
        """actions [n agents, num_atns] -> obs, masks, rewards, terminals, stats [n, 4] (episode
        ended, return, length, outcome)."""
        envs = range(len(self.conns))
        self.send(envs, actions)
        return self.recv(envs)

    def send(self, envs, actions: np.ndarray) -> None:
        """Starts a step of environments `envs` (their agents' actions [len(envs) * agents, num_atns])."""
        a = self.agents
        acts = np.asarray(actions, np.float32).reshape(len(envs), a * self.num_atns)
        for e, act in zip(envs, acts):
            self.conns[e].sendall(struct.pack("<I", 2) + act.tobytes())

    def recv(self, envs):
        """The results of the step `send` started for `envs` (as step returns them)."""
        a = self.agents
        obs, masks, rew, term, stats = [], [], [], [], []
        for conn in (self.conns[e] for e in envs):
            o, m = self._read_obs(conn)
            obs.append(o)
            masks.append(m)
            rew.append(np.frombuffer(_recv(conn, 4 * a), np.float32))
            term.append(np.frombuffer(_recv(conn, 4 * a), np.float32))
            stats.append(np.frombuffer(_recv(conn, 16 * a), np.float32).reshape(a, 4))
        return (np.concatenate(obs), np.concatenate(masks), np.concatenate(rew), np.concatenate(term),
                np.concatenate(stats))

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.close()
            except OSError:
                pass
