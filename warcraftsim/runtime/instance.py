"""One running Warcraft III process under Wine, driven step by step.

Lifecycle::

    inst = GameInstance(GameSetup(...), name="g0")
    obs = inst.start()                  # launches, returns the observation at game time 0
    obs = inst.step([...commands...])   # applies commands, runs step_seconds of game time
    obs = inst.restart()                # new episode (scenarios: in-game; melee: relaunch)
    inst.close()

Step protocol: the harness writes obs.txt and opens act.txt; the w3shim DLL
reports "OBS n" over TCP and blocks the game until we answer "GO". Before
answering we write act.txt, which the harness then executes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .. import paths
from ..data.mapbuild import HarnessConfig, build_map
from ..data.wgc import Difficulty, Race, Wgc, WgcSlot
from ..protocol import (Command, EndGame, Observation, ProtocolError, Restart, Snapshot, Unit, encode_commands,
                        merge_observation, parse_observation, write_action_file)
from . import wine
from .display import Xvfb

if False:  # typing only
    from ..scenario import Scenario  # noqa: F401

SHIM_DIR = paths.REPO_ROOT / "build" / "shim"
_map_lock = threading.Lock()
# Map loading renders the loading screen in software as fast as it can; many simultaneous loads
# starve each other. Limit concurrent launches per Python process.
LAUNCH_SLOTS = threading.BoundedSemaphore(int(os.environ.get("WARCRAFTSIM_PARALLEL_LAUNCHES", "4")))


class GameError(RuntimeError):
    pass


class GameCrashed(GameError):
    pass


SLOT_KINDS = ("agent", "ai", "scripted", "idle")


@dataclass(frozen=True)
class Slot:
    """A player slot.

    kind: "agent" (controlled from Python), "ai" (built-in melee AI), "scripted" (scenario
    opponent: idle units attack-move to the nearest enemy), "idle" (no controller; units only
    auto-acquire targets).
    """
    kind: str
    race: str = "human"
    difficulty: str = "normal"  # built-in AI: easy / normal / insane
    team: int | None = None

    def __post_init__(self):
        if self.kind not in SLOT_KINDS:
            raise ValueError(f"slot kind must be one of {SLOT_KINDS}, not {self.kind!r}")
        Race.parse(self.race)
        Difficulty.parse(self.difficulty)


def Agent(race: str = "human", team: int | None = None) -> Slot:  # noqa: N802 - factory reads like a type
    return Slot("agent", race, team=team)


def BuiltinAI(race: str = "orc", difficulty: str = "normal", team: int | None = None) -> Slot:  # noqa: N802
    return Slot("ai", race, difficulty, team)


def Scripted(race: str = "orc", team: int | None = None) -> Slot:  # noqa: N802
    return Slot("scripted", race, team=team)


def Idle(race: str = "orc", team: int | None = None) -> Slot:  # noqa: N802
    return Slot("idle", race, team=team)


@dataclass
class GameSetup:
    map: str = "(2)EchoIsles"
    slots: Sequence[Slot] = (Slot("agent", "human"), Slot("ai", "orc", "normal"))
    step_seconds: float = 0.25
    # Virtual clock multiplier. None = adaptive: about 2.5x the rate the simulation actually reaches.
    # Much higher than needed only makes the game's background threads wake up (and hit the
    # wineserver) more often; lower than the reachable rate throttles the game.
    speed: float | None = None
    turbo_ms: int = 0  # >0: simulate up to this much game time per frame, bypassing turn pacing
    max_game_seconds: float = 0.0  # 0 = unlimited; otherwise a tie when reached
    fog: bool | None = None  # None: on for melee, off for scenarios
    wgc_speed: int = 1
    window: tuple[int, int] = (800, 600)
    scenario: "Scenario | None" = None
    # True: the first agent is the local (user) player instead of a computer slot watched by an
    # observer. Only one slot can be a user in a local game.
    agent_is_user: bool = False
    # Melee only: keep a second game loaded and waiting at game time 0, so restart() is instant
    # instead of a ~8 s relaunch. Costs one more idle process (~400 MB) and its load time.
    warm_spare: bool = True

    def __post_init__(self):
        self.slots = tuple(self.slots)
        if self.scenario is not None:
            self.map = self.scenario.map
            if any(s.kind == "ai" for s in self.slots):
                raise ValueError("scenarios use 'agent', 'scripted' or 'idle' slots, not the melee AI")
        elif any(s.kind in ("scripted", "idle") for s in self.slots):
            raise ValueError("'scripted' and 'idle' slots need a scenario")

    @property
    def agent_players(self) -> tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.slots) if s.kind == "agent")

    @property
    def fog_enabled(self) -> bool:
        return self.fog if self.fog is not None else self.scenario is None

    def harness_config(self) -> HarnessConfig:
        if self.scenario is None:
            return HarnessConfig(self.step_seconds, self.agent_players, self.max_game_seconds)
        sc = self.scenario
        cx, cy = sc.resolved_center()
        return HarnessConfig(
            step_seconds=self.step_seconds,
            agent_players=self.agent_players,
            max_game_seconds=self.max_game_seconds or sc.max_game_seconds,
            scripted_players=tuple(i for i, s in enumerate(self.slots) if s.kind == "scripted"),
            scenario=True,
            victory=sc.victory,
            send_destructables=False,
            spawn=sc.absolute_units(),
            resources=sc.resources,
            clear_area=(cx, cy, sc.clear_radius),
        )

    def wgc(self, map_path: str) -> Wgc:
        slots = []
        user = self.agent_players[0] if self.agent_is_user and self.agent_players else None
        for i, s in enumerate(self.slots):
            team = s.team if s.team is not None else i
            if i == user:
                slots.append(WgcSlot.user(i, s.race, team=team))
            else:
                # agents use computer slots too; the harness does not start an AI for them
                slots.append(WgcSlot.computer(i, s.race, s.difficulty if s.kind == "ai" else "normal", team=team))
        if user is None:
            slots.append(WgcSlot.observer(len(self.slots)))
        flags = 0 if self.fog_enabled else 1
        return Wgc(map_path, slots, game_speed=self.wgc_speed, flags=flags)

    def map_key(self) -> str:
        from ..data.mapbuild import harness_source
        blob = json.dumps({"map": self.map, "harness": asdict(self.harness_config()),
                           "source": hashlib.sha1(harness_source().encode()).hexdigest()}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _shim_files() -> tuple[Path, Path]:
    dll, launcher = SHIM_DIR / "w3shim.dll", SHIM_DIR / "w3launch.exe"
    if not (dll.exists() and launcher.exists()):
        raise FileNotFoundError(f"w3shim not built in {SHIM_DIR}; run `make -C shim`")
    return dll, launcher


def commands_path(replay: Path) -> Path:
    """The agent-order log saved next to a replay."""
    return replay.with_name(replay.stem + ".commands.json")


def _winpath(p: Path) -> str:
    return "Z:" + str(p.resolve()).replace("/", "\\")


# Attributes that belong to one game process; a warm spare's are swapped in on restart.
_PROCESS_ATTRS = ("name", "_own_display", "_display", "proc", "_server", "_conn", "_rfile", "prefix", "ipc_dir",
                  "inst_dir", "_last_seq", "_units", "last_obs", "_sent_at", "_need_snapshot", "_cmd_log",
                  "_proc_episode")


class GameInstance:
    def __init__(self, setup: GameSetup, name: str = "g0", display: str | None = None,
                 timeout: float = 120.0, keep_logs: bool = True):
        self.setup = setup
        self.base_name = name
        self.name = name
        self.timeout = timeout
        self.keep_logs = keep_logs
        self._own_display: Xvfb | None = None
        self._display = display
        self.proc: subprocess.Popen | None = None
        self._server: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._rfile = None
        self._last_seq: int | None = None
        self._pending_speed: float | None = None
        self.prefix: Path | None = None
        self.ipc_dir = Path("/dev/shm/warcraftsim") / name
        self.inst_dir = paths.RUNTIME_DIR / "instances" / name
        self.last_obs: Observation | None = None
        self.episode = 0
        self.steps = 0
        self.damaged_records = 0
        self._units: dict[int, Unit] = {}
        self._need_snapshot = False
        self.order_names = setup.harness_config().resolved_order_names()
        self._speed = setup.speed or 32.0
        self._game_wall = 0.0  # wall seconds spent inside the game (not waiting for Python)
        self._game_ms0 = 0
        self._sent_at: float | None = None
        self._spare: GameInstance | None = None
        self._spare_thread: threading.Thread | None = None
        self._spare_error: BaseException | None = None
        self._ended = False  # the game was ended (save_replay); the next restart relaunches
        # Commands sent in the current game process, keyed "<episode in process>:<obs seq>", so a
        # .w3g replay can be played back with the agents' orders (see play_replay).
        self._cmd_log: dict[str, list[int]] = {}
        self._proc_episode = -1
        self._playback: dict[str, list[int]] | None = None

    # ---- launch ---------------------------------------------------------------------------

    def _prepare(self) -> None:
        wine.ensure_template()
        self.prefix = wine.clone_prefix(self.name)
        wine.kill_prefix(self.prefix)
        # IPC files live in RAM; the game sees them as Documents\Warcraft III\CustomMapData\w3sim
        self.ipc_dir.mkdir(parents=True, exist_ok=True)
        for f in self.ipc_dir.iterdir():
            f.unlink()
        cmd_dir = wine.documents_dir(self.prefix) / "CustomMapData"
        cmd_dir.mkdir(parents=True, exist_ok=True)
        link = cmd_dir / "w3sim"
        if link.is_symlink() or link.exists():
            link.unlink() if link.is_symlink() else shutil.rmtree(link)
        link.symlink_to(self.ipc_dir, target_is_directory=True)
        write_action_file(self.ipc_dir / "act.txt")
        # map + game config
        work = self.prefix / "drive_c" / wine.WORK_DIR
        # replays store the map as "..\\w3sim\\map.w3x" and resolve it from the Documents folder
        doc_link = wine.documents_dir(self.prefix) / wine.WORK_DIR
        if not doc_link.is_symlink():
            doc_link.symlink_to(work, target_is_directory=True)
        maps_cache = paths.CACHE_DIR / "maps"
        cached = maps_cache / f"{self.setup.map_key()}.w3x"
        with _map_lock:
            if not cached.exists():
                build_map(self.setup.map, cached, self.setup.harness_config())
        shutil.copyfile(cached, work / "map.w3x")
        # .wgc map paths are relative to the game directory (C:\Warcraft III)
        self.setup.wgc(f"..\\{wine.WORK_DIR}\\map.w3x").write(work / "game.wgc")
        self._set_window_size()

    def _set_window_size(self) -> None:
        w, h = self.setup.window
        user_reg = self.prefix / "user.reg"
        text = user_reg.read_text(encoding="latin-1")
        for key, val in (("reswidth", w), ("resheight", h)):
            text = text.replace(f'"{key}"=dword:{wine.VIDEO_SETTINGS[key]:08x}', f'"{key}"=dword:{val:08x}')
        user_reg.write_text(text, encoding="latin-1")

    warm_spare_child = False  # a spare never starts its own spare

    def start(self) -> Observation:
        """Launch the game and return the first observation (game time 0)."""
        if self.proc:
            raise GameError("already started")
        self._prepare()
        if not self._display:
            self._own_display = Xvfb()
            self._display = self._own_display.display
        self.episode = 0
        obs = self._launch()
        self._start_spare()
        return obs

    def _launch(self, attempts: int = 2) -> Observation:
        for attempt in range(attempts):
            with LAUNCH_SLOTS:
                try:
                    return self._launch_once()
                except GameError:
                    self._stop_process()
                    if attempt == attempts - 1:
                        raise
        raise AssertionError("unreachable")

    def play_replay(self, replay: str | os.PathLike) -> Observation:
        """Play a replay saved by save_replay() of a game with this setup, stepping it like a live
        game: the harness runs again, receives the recorded agent orders at the same steps and
        writes observations. Commands passed to step() are ignored (except Camera and Snapshot)."""
        log_file = commands_path(Path(replay))
        self._playback = json.loads(log_file.read_text())["commands"] if log_file.exists() else {}
        self._stop_process()
        if self.prefix is None:
            self._prepare()
        if not self._display:
            self._own_display = Xvfb()
            self._display = self._own_display.display
        shutil.copyfile(replay, self.prefix / "drive_c" / wine.WORK_DIR / "replay.w3g")
        for f in self.ipc_dir.iterdir():
            f.unlink()
        write_action_file(self.ipc_dir / "act.txt")
        self._ended = False
        self._loadfile = f"C:\\{wine.WORK_DIR}\\replay.w3g"
        try:
            return self._launch(attempts=1)
        finally:
            self._loadfile = None

    _loadfile: str | None = None

    def _launch_once(self) -> Observation:
        dll, launcher = _shim_files()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        port = self._server.getsockname()[1]
        self.inst_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.inst_dir / "shim.log"
        env = wine.wine_env(self.prefix, DISPLAY=self._display, W3SIM_PORT=str(port),
                            W3SIM_SPEED=str(self._speed), W3SIM_TURBO_MS=str(self.setup.turbo_ms),
                            W3SIM_LOG=_winpath(log_path))
        out = open(self.inst_dir / "wine.log", "wb") if self.keep_logs else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            ["wine", _winpath(launcher), _winpath(dll), f"C:\\{wine.GAME_LINK}\\Warcraft III.exe", "-window",
             "-loadfile", self._loadfile or f"C:\\{wine.WORK_DIR}\\game.wgc"],
            env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
        )
        self._server.settimeout(self.timeout)
        try:
            self._conn, _ = self._server.accept()
        except socket.timeout:
            if self._own_display:  # keep a picture of what the game was showing
                try:
                    self._own_display.screenshot(self.inst_dir / "launch-timeout.png")
                except Exception:
                    pass
            self._stop_process()
            raise GameError(f"game did not reach the harness within {self.timeout:.0f}s (see {self.inst_dir})")
        self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._conn.settimeout(self.timeout)
        self._rfile = self._conn.makefile("rb")
        self._last_seq = None
        self.last_obs = None
        self._sent_at = None
        self._cmd_log = {}
        self._proc_episode = -1
        self.episode += 1
        return self._await_observation(new_episode=True)

    # ---- stepping -------------------------------------------------------------------------

    def _read_obs_file(self) -> Observation:
        path = self.ipc_dir / "obs.txt"
        for attempt in range(50):
            try:
                return parse_observation(path.read_text(encoding="latin-1"), self.order_names)
            except (ProtocolError, FileNotFoundError):
                if attempt == 49:
                    raise
                time.sleep(0.002)
        raise AssertionError("unreachable")

    def _await_observation(self, new_episode: bool = False) -> Observation:
        while True:
            try:
                line = self._rfile.readline()
            except socket.timeout:
                raise GameError(f"no observation within {self.timeout:.0f}s")
            if not line:
                code = self.proc.poll() if self.proc else None
                raise GameCrashed(f"game connection closed (exit code {code}; see {self.inst_dir})")
            if not line.startswith(b"OBS"):
                continue
            obs = self._read_obs_file()
            if new_episode:
                if obs.seq != 0:
                    self._reply()  # still finishing the previous episode
                    continue
            elif self._last_seq is not None and obs.seq == self._last_seq:
                self._reply()  # the same observation reported twice: nothing new to act on
                continue
            self._last_seq = obs.seq
            if new_episode:
                self._proc_episode += 1
            if obs.damaged_records:
                # a unit delta may be lost: ask for a full snapshot with the next commands
                self.damaged_records += obs.damaged_records
                self._need_snapshot = True
            obs = merge_observation(self._units, obs)
            self.last_obs = obs
            return obs

    def _reply(self, extra: str = "") -> None:
        msg = "GO"
        if self._pending_speed is not None:
            msg += f" speed={self._pending_speed}"
            self._pending_speed = None
        if extra:
            msg += " " + extra
        self._conn.sendall(msg.encode() + b"\n")

    def step(self, commands: Iterable[Command] = ()) -> Observation:
        """Apply commands and advance one step (setup.step_seconds of game time)."""
        self.send(commands)
        return self.receive()

    def send(self, commands: Iterable[Command] = ()) -> None:
        """First half of step(): hand the commands to the game and let it run."""
        if not self._conn:
            raise GameError("not started")
        commands = list(commands)
        if self._need_snapshot:
            commands.append(Snapshot())
            self._need_snapshot = False
        key = f"{self._proc_episode}:{self._last_seq}"
        ints = encode_commands(commands)
        if self._playback is not None:
            # replay playback: the recorded orders; only observation/camera commands are added
            extra = [c for c in commands if type(c).__name__ in ("Snapshot", "Camera")]
            ints = self._playback.get(key, []) + encode_commands(extra)
        elif ints:
            self._cmd_log[key] = ints
        write_action_file(self.ipc_dir / "act.txt", ints=ints)
        self._sent_at = time.perf_counter()
        self._reply()

    def receive(self) -> Observation:
        """Second half of step(): wait for the next observation."""
        prev_ms = self.last_obs.game_ms if self.last_obs else 0
        obs = self._await_observation()
        write_action_file(self.ipc_dir / "act.txt")  # a stray re-read must not repeat commands
        self.steps += 1
        if self._sent_at is not None and obs.game_ms > prev_ms:
            self._game_wall += time.perf_counter() - self._sent_at
            self._game_ms0 += obs.game_ms - prev_ms
            if self.setup.speed is None and self._game_wall > 0.25:
                self._adapt_speed()
        return obs

    def _adapt_speed(self) -> None:
        rate = self._game_ms0 / 1000.0 / self._game_wall
        target = min(max(2.5 * rate, 16.0), 512.0)
        self._game_wall, self._game_ms0 = 0.0, 0
        if abs(target - self._speed) / self._speed > 0.15:
            self._speed = target
            self._pending_speed = round(target, 1)

    @property
    def speed(self) -> float:
        """Current virtual clock multiplier."""
        return self._speed

    def restart(self, relaunch: bool = False) -> Observation:
        """End the current episode and start a new one.

        Scenario maps reset inside the running game (units are removed and respawned). Melee
        games are relaunched: RestartGame/ChangeLevel/LoadGame all return a .wgc game to the
        main menu, and the .wgc is what sets exact slots and AI difficulty. relaunch=True always
        starts a fresh process (e.g. so a saved replay holds exactly one episode).
        """
        if self.setup.scenario is not None and not self._ended and self._playback is None and not relaunch:
            key = f"{self._proc_episode}:{self._last_seq}"
            self._cmd_log[key] = self._cmd_log.get(key, []) + Restart().encode()
            write_action_file(self.ipc_dir / "act.txt", [Restart()])
            self._reply()
            self._last_seq = None
            obs = self._await_observation(new_episode=True)
            write_action_file(self.ipc_dir / "act.txt")
            self.episode += 1
            return obs
        self._ended = False
        self._playback = None
        spare = self._take_spare()
        if spare is not None:
            return self._swap_in(spare)
        self._stop_process()
        for f in self.ipc_dir.iterdir():
            f.unlink()
        write_action_file(self.ipc_dir / "act.txt")
        return self._launch()

    # ---- warm spare (melee) ---------------------------------------------------------------

    def _wants_spare(self) -> bool:
        return (self.setup.warm_spare and self.setup.scenario is None and not self.warm_spare_child
                and not self._display_shared())

    def _display_shared(self) -> bool:
        return self._own_display is None and self._display is not None

    def _start_spare(self, recycled: "GameInstance | None" = None) -> None:
        """Load the next spare in the background. `recycled` holds the retired process: it is shut
        down first and its name (prefix, display slot) is reused."""
        if not self._wants_spare() or self._spare is not None:
            if recycled is not None:
                recycled.close()
            return
        if recycled is None:
            other = f"{self.base_name}.b" if self.name == self.base_name else self.base_name
            recycled = GameInstance(self.setup, name=other, timeout=self.timeout, keep_logs=self.keep_logs)
            recycled.warm_spare_child = True
        spare = recycled
        spare._speed = self._speed
        self._spare, self._spare_error = spare, None

        def run():
            try:
                spare.close()  # the retired process, if any
                spare.start()
            except BaseException as e:  # reported when the spare is needed
                self._spare_error = e

        self._spare_thread = threading.Thread(target=run, name=f"spare-{spare.name}", daemon=True)
        self._spare_thread.start()

    def _take_spare(self) -> "GameInstance | None":
        spare, thread = self._spare, self._spare_thread
        self._spare = self._spare_thread = None
        if spare is None:
            return None
        thread.join()
        if self._spare_error is not None or spare.last_obs is None:
            spare.close()
            return None
        return spare

    def _swap_in(self, spare: "GameInstance") -> Observation:
        """Continue with the spare's process (waiting at game time 0); the retired process is shut
        down in the background, which then loads the next spare under its name."""
        for attr in _PROCESS_ATTRS:
            mine, theirs = getattr(self, attr), getattr(spare, attr)
            setattr(self, attr, theirs)
            setattr(spare, attr, mine)
        self._pending_speed = self._speed
        self.episode += 1
        self._start_spare(recycled=spare)
        return self.last_obs

    def set_speed(self, speed: float | None) -> None:
        """Change the virtual clock speed from the next step on (None: adaptive)."""
        self.setup.speed = speed
        if speed is not None:
            self._speed = speed
            self._pending_speed = speed

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def screenshot(self, path: str | os.PathLike) -> None:
        if not self._own_display:
            raise GameError("screenshots need an instance-owned display")
        self._own_display.screenshot(path)

    def replay_dir(self) -> Path:
        return wine.documents_dir(self.prefix) / "Replay"

    def save_replay(self, dest: str | os.PathLike, timeout: float = 10.0) -> Path:
        """End the current game normally and copy the replay the engine writes to `dest`.

        Agent orders are issued by the map script, not through the engine's recorded command stream,
        so they are not in the .w3g itself; they are saved next to it (``<name>.commands.json``) and
        play_replay() feeds them back at the same steps. The stock game client shows such a replay
        without the agents' orders. The game is over afterwards; the next restart() relaunches.
        """
        replay = self.replay_dir() / "LastReplay.w3g"
        before = replay.stat().st_mtime if replay.exists() else 0.0
        log = dict(self._cmd_log)
        write_action_file(self.ipc_dir / "act.txt", [EndGame()])
        self._reply()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if replay.exists() and replay.stat().st_mtime > before and replay.stat().st_size > 0:
                time.sleep(0.2)  # let the writer finish
                dest = Path(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(replay, dest)
                commands_path(dest).write_text(json.dumps({"format": "warcraftsim-commands", "version": 1,
                                                           "commands": log}))
                self._ended = True
                return dest
            time.sleep(0.05)
        raise GameError("the game did not write a replay")

    # ---- teardown -------------------------------------------------------------------------

    def _stop_process(self) -> None:
        if self._conn:
            try:
                self._conn.sendall(b"QUIT\n")
            except OSError:
                pass
        for s in (self._rfile, self._conn, self._server):
            try:
                if s:
                    s.close()
            except OSError:
                pass
        self._rfile = self._conn = self._server = None
        if self.proc:
            try:
                self.proc.wait(3)
            except subprocess.TimeoutExpired:
                pass
        if self.prefix:
            wine.kill_prefix(self.prefix)
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None

    def close(self) -> None:
        spare = self._take_spare()
        if spare is not None:
            spare.close()
        self._stop_process()
        if self._own_display:
            self._own_display.close()
            self._own_display = None
            self._display = None

    def __enter__(self) -> "GameInstance":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
