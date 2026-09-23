"""Bridge server: runs Warcraft III games for the PufferLib 5.0 trainer.

The trainer's C environment (puffer/wc3_bridge.h) connects once per environment over a Unix
socket; every connection gets its own game and thread, so all games step in parallel. Episodes
reset automatically. Besides serving the trainer the bridge writes the run's episode log and
media for the dashboard:

    <run>/episodes-<w>.jsonl   one line per finished episode (w = bridge worker process)
    <run>/bridge-<w>.jsonl     throughput every few seconds
    <run>/renders/*.html    trajectory animations (every `record_every` episodes of game 0)
    <run>/replays/*.w3g     single-episode replays every `video_every` episodes, with the agent
                            orders (.commands.json) and the policy's inputs/outputs (.steps.npz)
    <run>/videos/*.mp4      real game footage rendered from those replays in the background, with
                            the agent's orders drawn on it and a panel of what the policy thinks
                            (value, action probabilities; from the checkpoint of that time)
"""

from __future__ import annotations

import json
import os
import queue
import socket
import struct
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from collections import Counter

import numpy as np

from ..record import TrajectoryRecorder, render_html
from ..runtime.instance import GameError
from .tasks import Task, action_summary

MAGIC = 0x46503357
VERSION = 1


def combat_stats(first, last, player: int = 0) -> dict | None:
    """Damage and unit losses of `player` against everyone else over an episode (harness obs)."""
    if first is None or last is None:
        return None

    def side(o, mine: bool) -> list:
        return [u for u in o.units if u.alive and not u.is_structure
                and (u.owner == player if mine else u.owner != player and u.owner in o.players)]

    own0, enemy0, own1, enemy1 = side(first, True), side(first, False), side(last, True), side(last, False)
    if not own0 or not enemy0:
        return None
    hp = lambda us: sum(u.hp for u in us)  # noqa: E731
    max_hp = lambda us: max(sum(u.max_hp for u in us), 1)  # noqa: E731
    return {"dealt": round((hp(enemy0) - hp(enemy1)) / max_hp(enemy0), 4),
            "taken": round((hp(own0) - hp(own1)) / max_hp(own0), 4),
            "kills": len(enemy0) - len(enemy1), "losses": len(own0) - len(own1)}


def _recv(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("trainer disconnected")
        buf += chunk
    return bytes(buf)


class _Slot:
    """One game served to one trainer environment (with task.num_agents agents)."""

    def __init__(self, index: int, env, first_obs: list, first_info: dict):
        self.index = index
        self.env = env
        self.pending = (first_obs, first_info)  # the observations to hand out on the first reset
        self.ep_return: list[float] = []
        self.ep_length = 0
        self.episodes = 0
        self.recorder: TrajectoryRecorder | None = None
        self.replay_path: Path | None = None
        self.trace: dict[str, list] | None = None  # policy inputs/outputs of a video episode
        self.actions: Counter = Counter()
        self.first_obs = None  # the episode's first harness observation


class BridgeServer:
    def __init__(self, task: Task, num_envs: int, run_dir: str | os.PathLike, socket_path: str,
                 record_every: int = 25, video_every: int = 100, name: str = "wc3", worker: int = 0):
        self.task = task
        self.num_envs = num_envs
        self.run_dir = Path(run_dir)
        self.socket_path = socket_path
        self.record_every = record_every
        self.video_every = video_every
        self.name = name
        self.worker = worker
        self.slots: list[_Slot] = []
        self._next_slot = 0
        self._lock = threading.Lock()
        self._episodes = 0
        self._steps = 0
        self._server: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._render_queue: queue.Queue = queue.Queue()
        for sub in ("renders", "replays", "videos"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self._episode_log = open(self.run_dir / f"episodes-{worker}.jsonl", "a")
        self._bridge_log = open(self.run_dir / f"bridge-{worker}.jsonl", "a")

    # ---- setup --------------------------------------------------------------------------------

    def launch_games(self, log=print) -> None:
        def launch(i: int) -> _Slot:
            env = self.task.make_env(f"{self.name}{i}")
            obs, info = self.task.reset(env)
            return _Slot(i, env, obs, info)

        t0 = time.time()
        with ThreadPoolExecutor(self.num_envs) as pool:
            self.slots = list(pool.map(launch, range(self.num_envs)))
        log(f"bridge: {self.num_envs} games ready in {time.time() - t0:.0f}s")

    def serve(self) -> None:
        """Listen for trainer environments (in background threads)."""
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.socket_path)
        self._server.listen(self.num_envs + 8)
        for target in (self._accept_loop, self._stats_loop, self._render_loop):
            t = threading.Thread(target=target, daemon=True, name=f"bridge-{target.__name__}")
            t.start()
            self._threads.append(t)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            t = threading.Thread(target=self._serve_conn, args=(conn,), daemon=True)
            t.start()
            self._threads.append(t)

    # ---- per-environment protocol ---------------------------------------------------------------

    def _serve_conn(self, conn: socket.socket) -> None:
        slot = None
        try:
            magic, version, obs_size, num_atns = struct.unpack("<4I", _recv(conn, 16))
            task_name = _recv(conn, 32).split(b"\0")[0].decode()
            ok = (magic == MAGIC and version == VERSION and obs_size == self.task.obs_size
                  and num_atns == self.task.num_atns and task_name == self.task.name)
            with self._lock:
                if ok and self._next_slot < len(self.slots):
                    slot = self.slots[self._next_slot]
                    self._next_slot += 1
            if slot is None:
                print(f"bridge: refused environment (task {task_name!r}, obs {obs_size}, atns {num_atns})")
                conn.sendall(struct.pack("<I", 0))
                return
            n = self.task.num_agents
            conn.sendall(struct.pack("<I", n))
            act_bytes = 4 * self.task.num_atns
            while True:
                (cmd,) = struct.unpack("<I", _recv(conn, 4))
                if cmd == 1:
                    obs = self._reset(slot)
                    conn.sendall(b"".join(o.tobytes() for o in obs))
                elif cmd == 2:
                    actions = [np.frombuffer(_recv(conn, act_bytes), dtype=np.float32) for _ in range(n)]
                    obs, rewards, done, stats = self._step(slot, actions)
                    conn.sendall(b"".join(o.tobytes() for o in obs)
                                 + struct.pack(f"<{n}f", *rewards)
                                 + struct.pack(f"<{n}f", *([1.0 if done else 0.0] * n))
                                 + b"".join(struct.pack("<4f", *st) for st in stats))
                else:
                    raise ValueError(f"unknown command {cmd}")
        except ConnectionError:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            conn.close()

    def _begin_episode(self, slot: _Slot, obs: list, info) -> list:
        slot.ep_return, slot.ep_length = [0.0] * self.task.num_agents, 0
        slot.actions = Counter()
        slot.first_obs = info.get("obs")
        slot.trace = {"obs": [np.stack(obs)], "actions": [], "rewards": []} if slot.replay_path else None
        if slot.recorder is not None:
            slot.recorder.close()
            slot.recorder = None
        if slot.index == 0 and self.record_every and slot.episodes % self.record_every == 0:
            path = self.run_dir / "renders" / f"w{self.worker}-episode{self._episodes:06d}.jsonl"
            slot.recorder = TrajectoryRecorder(path, map_name=self.task.scenario.map if self.task.scenario else None)
            slot.recorder.add(info["obs"])
        return obs

    def _reset(self, slot: _Slot) -> np.ndarray:
        if slot.pending is not None:
            obs, info = slot.pending
            slot.pending = None
            return self._begin_episode(slot, obs, info)
        try:
            obs, info = self.task.reset(slot.env)
            return self._begin_episode(slot, obs, info)
        except GameError as e:
            return self._recover(slot, e)[0]

    def _recover(self, slot: _Slot, error: Exception):
        """A game crashed or hung: relaunch it and end the episode as truncated (outcome 0)."""
        print(f"bridge: game {self.worker}/{slot.index} failed ({error}); relaunching", flush=True)
        with self._lock:
            self._episode_log.write(json.dumps({"time": time.time(), "worker": self.worker, "env": slot.index,
                                                "event": "game_restart", "error": str(error)[:200]}) + "\n")
            self._episode_log.flush()
        if slot.recorder is not None:
            slot.recorder.close()
            slot.recorder = None
        slot.replay_path = None
        for attempt in range(5):
            try:
                slot.env.game.instance.close()
                obs, info = self.task.reset(slot.env)
                break
            except Exception as e:  # keep trying: one bad launch must not end the training run
                print(f"bridge: relaunch {attempt + 1} of game {self.worker}/{slot.index} failed: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"game {self.worker}/{slot.index} could not be relaunched")
        stats = [(1.0, r, float(slot.ep_length), 0.0) for r in slot.ep_return]
        slot.episodes += 1
        return self._begin_episode(slot, obs, info), [0.0] * self.task.num_agents, True, stats

    def _step(self, slot: _Slot, actions: list[np.ndarray]):
        try:
            return self._step_game(slot, actions)
        except GameError as e:
            return self._recover(slot, e)

    def _step_game(self, slot: _Slot, actions: list[np.ndarray]):
        if self.task.action_stats is not None:
            slot.actions.update(self.task.action_stats(slot.env, actions))
        obs, rewards, done, info, outcomes = self.task.step(slot.env, actions)
        if slot.trace is not None:
            slot.trace["actions"].append(np.stack(actions))
            slot.trace["rewards"].append(np.asarray(rewards, np.float32))
            if not done:
                slot.trace["obs"].append(np.stack(obs))
        slot.ep_return = [r + dr for r, dr in zip(slot.ep_return, rewards)]
        slot.ep_length += 1
        with self._lock:
            self._steps += 1
        if slot.recorder is not None:
            slot.recorder.add(info["obs"])
        if not done:
            return obs, rewards, False, [(0.0, 0.0, 0.0, 0.0)] * self.task.num_agents
        outcome = outcomes[0]
        self._finish_episode(slot, info, outcome, outcomes)
        # next episode: normally in-game; the episode chosen for a video starts in a fresh process
        # so that its replay holds exactly that episode
        slot.episodes += 1
        want_video = (slot.index == 0 and self.video_every and slot.episodes % self.video_every == 0)
        next_obs, next_info = self.task.reset(slot.env, options={"relaunch": True} if want_video else None)
        slot.replay_path = ((self.run_dir / "replays" / f"w{self.worker}-episode{self._episodes:06d}.w3g")
                            if want_video else None)
        stats = [(1.0, r, float(slot.ep_length), o) for r, o in zip(slot.ep_return, outcomes)]
        return self._begin_episode(slot, next_obs, next_info), rewards, True, stats

    def _finish_episode(self, slot: _Slot, info: dict, outcome: float, outcomes: list[float]) -> None:
        o = info.get("obs")
        with self._lock:
            self._episodes += 1
            episode = self._episodes
            row = {"time": time.time(), "episode": episode, "worker": self.worker, "env": slot.index,
                   "return": round(slot.ep_return[0], 4),
                   "length": slot.ep_length, "outcome": outcome, "game_time": o.game_time if o else None,
                   "total_steps": self._steps}
            if slot.actions:
                row["act"] = action_summary(slot.actions)
            combat = combat_stats(slot.first_obs, o)
            if combat:
                row["combat"] = combat
            self._episode_log.write(json.dumps(row) + "\n")
            self._episode_log.flush()
        if slot.recorder is not None:
            slot.recorder.close()
            html = render_html(slot.recorder.path)
            slot.recorder = None
            self._media_event("render", html, episode, outcome, slot.ep_return[0])
        if slot.replay_path is not None:
            try:
                inst = slot.env.game.instance
                replay = inst.save_replay(slot.replay_path)
                if slot.trace is not None:
                    t = slot.trace
                    np.savez_compressed(replay.with_suffix(".steps.npz"), obs=np.stack(t["obs"]),
                                        actions=np.stack(t["actions"]), rewards=np.stack(t["rewards"]),
                                        outcomes=np.asarray(outcomes, np.float32), time=time.time())
                self._render_queue.put((replay, episode, outcome, slot.ep_return[0]))
            except Exception as e:  # a missing video must not stop training
                print(f"bridge: replay not saved: {e}")
            slot.replay_path = None

    def _media_event(self, kind: str, path: Path, episode: int, outcome: float, ret: float, **extra) -> None:
        with self._lock, open(self.run_dir / f"media-{self.worker}.jsonl", "a") as f:
            f.write(json.dumps({"time": time.time(), "kind": kind, "file": str(path.relative_to(self.run_dir)),
                                "episode": episode, "outcome": outcome, "return": round(ret, 4), **extra}) + "\n")

    # ---- background work ----------------------------------------------------------------------

    def _render_loop(self) -> None:
        from ..video import render_replay

        while not self._stop.is_set():
            try:
                replay, episode, outcome, ret = self._render_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                setup = self.slots[0].env.setup
                out = self.run_dir / "videos" / (replay.stem + ".mp4")
                overlay = self._overlay(replay, episode)
                render_replay(setup, replay, out, name=f"{self.name}render{self.worker}", overlay=overlay)
                extra = {}
                if overlay is not None and overlay.values is not None:  # value calibration
                    extra = {"value0": round(float(overlay.values[0, 0]), 4),
                             "return0": round(float(overlay.returns[0, 0]), 4),
                             "policy_step": overlay.policy_step}
                self._media_event("video", out, episode, outcome, ret, **extra)
            except Exception as e:
                print(f"bridge: video not rendered: {e}")

    def _overlay(self, replay: Path, episode: int):
        """What the policy thought during a video episode, from the checkpoint of that time."""
        from ..overlay import EpisodeOverlay
        from .policy import PufferPolicy, checkpoint_at, checkpoint_step

        steps_file = replay.with_suffix(".steps.npz")
        if not steps_file.exists():
            return None
        trace = dict(np.load(steps_file))
        info = json.loads((self.run_dir / "run.json").read_text()) if (self.run_dir / "run.json").exists() else {}
        args = info.get("args", {})
        outputs, step = None, None
        ckpt = checkpoint_at(self.run_dir / "checkpoints", float(trace["time"]))
        if ckpt is not None:
            try:
                pol = PufferPolicy(ckpt, self.task.obs_size, self.task.act_sizes, hidden=args.get("hidden", 128),
                                   layers=args.get("layers", 2))
                outputs = [pol.run(trace["obs"][:, a]) for a in range(trace["obs"].shape[1])]
                step = checkpoint_step(ckpt)
            except (OSError, ValueError) as e:
                print(f"bridge: policy not evaluated for the video: {e}")
        return EpisodeOverlay(self.task, trace, outputs, gamma=args.get("gamma", 0.99),
                              title=f"{info.get('name', self.run_dir.name)} · episode {episode}",
                              policy_step=step)

    def _stats_loop(self) -> None:
        last_steps, last_t = 0, time.time()
        while not self._stop.wait(5):
            now = time.time()
            with self._lock:
                steps, episodes = self._steps, self._episodes
            sps = (steps - last_steps) / (now - last_t)
            last_steps, last_t = steps, now
            step_s = self.slots[0].env.setup.step_seconds if self.slots else 0.25
            self._bridge_log.write(json.dumps({"time": now, "worker": self.worker, "steps": steps,
                                               "episodes": episodes,
                                               "env_sps": round(sps, 1),
                                               "game_x_realtime": round(sps * step_s, 1)}) + "\n")
            self._bridge_log.flush()

    def close(self) -> None:
        self._stop.set()
        if self._server:
            self._server.close()
        for slot in self.slots:
            try:
                slot.env.close()
            except Exception:
                pass
        self._episode_log.close()
        self._bridge_log.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)


def run_worker(task_name: str, num_envs: int, run_dir: str, socket_path: str, record_every: int, video_every: int,
               name: str, worker: int, ready, stop) -> None:
    """Bridge worker process: its own games and GIL (multiprocessing target)."""
    from .tasks import get_task

    bridge = BridgeServer(get_task(task_name), num_envs, run_dir, socket_path, record_every, video_every, name,
                          worker)
    try:
        bridge.launch_games(log=lambda m: print(f"[worker {worker}] {m}", flush=True))
        bridge.serve()
        ready.set()
        stop.wait()
    finally:
        bridge.close()
