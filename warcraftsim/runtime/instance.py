"""One running Warcraft III process under Wine, driven step by step.

Lifecycle::

    inst = GameInstance(GameSetup(...), name="g0")
    obs = inst.start()                  # launches, returns the observation at game time 0
    obs = inst.step([...commands...])   # applies commands, runs step_seconds of game time
    obs = inst.restart()                # new episode (scenarios: in-game; melee: relaunch)
    inst.close()

Step protocol: the harness writes obs.txt and calls its mailbox native; the w3shim
DLL reports "OBS n" over TCP and blocks the game until we answer "GO ... A <n>
<ints>", whose command integers the harness then reads from the mailbox and executes.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .. import paths
from ..data.mapbuild import HarnessConfig, build_map
from ..data.wgc import Difficulty, Race, Wgc, WgcSlot
from ..protocol import (Command, EndGame, Observation, Op, ProtocolError, Restart, Snapshot, Unit, command_ops,
                        encode_commands,
                        merge_observation, parse_observation, parse_token_lines)
from . import reaper, wine
from .display import Xvfb

if False:  # typing only
    from ..scenario import Scenario  # noqa: F401

SHIM_DIR = paths.REPO_ROOT / "build" / "shim"
_map_lock = threading.Lock()
# Map loading renders the loading screen in software as fast as it can; many simultaneous loads
# starve each other. Concurrent launches are limited machine-wide (file locks in /dev/shm, so the
# limit also holds across processes, e.g. several bridge workers).
PARALLEL_LAUNCHES = int(os.environ.get("WARCRAFTSIM_PARALLEL_LAUNCHES", "4"))


class _LaunchSlot:
    def __enter__(self):
        import fcntl

        lock_dir = Path("/dev/shm/warcraftsim/launch-slots")
        lock_dir.mkdir(parents=True, exist_ok=True)
        while True:
            for i in range(PARALLEL_LAUNCHES):
                f = open(lock_dir / f"{i}.lock", "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._f = f
                    return self
                except BlockingIOError:
                    f.close()
            time.sleep(0.1)

    def __exit__(self, *exc):
        self._f.close()  # releases the lock


LAUNCH_SLOTS = _LaunchSlot()


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
    # clock speed until the first observation (map loading); see _launch_once
    launch_speed: float = 1.0
    turbo_ms: int = 0  # >0: simulate up to this much game time per frame, bypassing turn pacing
    # Shortest real wait (ms) the virtual clock turns a timed wait into. Game threads that poll with
    # 100-1000 ms timeouts would otherwise spin (timeouts / clock speed round to 0), each wait a
    # wineserver round trip. 24 skirmish games: 0 ms ~2600, 1 ms 4180, 3 ms 4450, 5 ms 4640,
    # 10 ms 3260 env steps/s (longer floors start delaying waits the game's progress depends on).
    wait_floor_ms: int = 5
    # A virtual sound card (w3shim) instead of none: the game's audio, in step with the virtual
    # clock, delivered with captured video frames (set_frame_capture). Off for training.
    audio: bool = False
    music_volume: int = 50  # with audio: 0-100 (0: no music); sound effects play at full volume
    health_bars: bool = False  # the game's own health bars over every unit (Gameplay option; videos)
    mouse_scroll: bool = True  # the camera scrolls with the pointer at a screen edge (off for videos)
    max_game_seconds: float = 0.0  # 0 = unlimited; otherwise a tie when reached
    fog: bool | None = None  # None: on for melee, off for scenarios
    wgc_speed: int = 1
    # The virtual screen the game runs on. In windowed mode the game fills a small screen and
    # uses a 960x540 window on 1024x768 (the size replay videos and their overlay assume).
    # Training does not look at the pixels: a small screen saves most of the rendering CPU.
    window: tuple[int, int] = (1024, 768)
    scenario: "Scenario | None" = None
    # True: the first agent is the local (user) player instead of a computer slot watched by an
    # observer. Only one slot can be a user in a local game.
    agent_is_user: bool = False
    # Relaunch the game process at the next episode boundary after about this many steps (0 =
    # never; each game at its own point within ±30%, so games started together do not all reload
    # at once: 24 did, and the trainer stalled for 90 s). Reading actions with Preloader leaked
    # ~20 KB of JASS compiler memory per step ("Not enough memory" after ~20k steps) and dead units
    # stayed referenced until the harness flushed them; now it is only a safety net (each relaunch
    # stalls the trainer's batch for a game load).
    recycle_steps: int = 500_000
    # Melee only: keep a second game loaded and waiting at game time 0, so restart() is instant
    # instead of a ~8 s relaunch. Costs one more idle process (~400 MB) and its load time.
    warm_spare: bool = True
    # Scenarios too, for relaunch=True (a single-episode replay, e.g. for a video): the spare
    # takes over and the process that was running is parked as the next spare; the relaunch
    # after the replay was saved resumes it with an in-game restart. Neither costs a load.
    scenario_spare: bool = False

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

    def map_key(self, source: str | None = None) -> str:
        """The built map's cache key; `source`: another harness source than the current one."""
        from ..data.mapbuild import harness_source
        cfg = self.harness_config()
        # the tables generated from game data are part of the built script too
        tables = json.dumps([cfg.resolved_order_names(), cfg.resolved_hero_abilities()], sort_keys=True)
        blob = json.dumps({"map": self.map, "harness": asdict(cfg),
                           "source": hashlib.sha1((source or harness_source()).encode()).hexdigest(),
                           "tables": hashlib.sha1(tables.encode()).hexdigest()}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _shim_files() -> tuple[Path, Path]:
    dll, launcher = SHIM_DIR / "w3shim.dll", SHIM_DIR / "w3launch.exe"
    if not (dll.exists() and launcher.exists()):
        raise FileNotFoundError(f"w3shim not built in {SHIM_DIR}; run `make -C shim`")
    return dll, launcher


def _set_reg_values(text: str, key: str, values: dict[str, int]) -> str:
    """Set DWORD values of one key in a Wine .reg file's text (the key must exist)."""
    if f"[{key}]" not in text:  # a key the template prefix doesn't have: added
        text = text.rstrip("\n") + f"\n\n[{key}] {int(time.time())}\n"
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(f"[{key}]"))
    end = next((i for i in range(start + 1, len(lines)) if not lines[i].strip()), len(lines))
    body = [line for line in lines[start + 1:end] if not any(line.startswith(f'"{k}"=') for k in values)]
    body += [f'"{k}"=dword:{v:08x}' for k, v in values.items()]
    return "\n".join(lines[:start + 1] + body + lines[end:])


def recorded_map(setup: "GameSetup", replay: Path, key: str | None) -> Path | None:
    """The cached map a replay was recorded on: its key from the replay's command file, or for
    replays from before that was saved, the key with the harness source as git had it when the
    replay was written. None: not found (the current map is used)."""
    cache = paths.CACHE_DIR / "maps"
    if key:
        return cache / f"{key}.w3x" if (cache / f"{key}.w3x").exists() else None
    try:
        when = int(replay.stat().st_mtime)
        rev = subprocess.run(["git", "-C", str(paths.REPO_ROOT), "log", "-1", f"--before={when}", "--format=%H",
                              "--", "warcraftsim/harness/w3sim.j"], capture_output=True, text=True, timeout=10).stdout.strip()
        if not rev:
            return None
        src = subprocess.run(["git", "-C", str(paths.REPO_ROOT), "show", f"{rev}:warcraftsim/harness/w3sim.j"],
                             capture_output=True, text=True, timeout=10).stdout
        path = cache / f"{setup.map_key(src)}.w3x"
        return path if src and path.exists() else None
    except (OSError, subprocess.SubprocessError):
        return None


def replay_script(setup: "GameSetup", replay: str | os.PathLike) -> str:
    """The map script `replay` plays back with (the harness it was recorded with)."""
    log_file = commands_path(Path(replay))
    key = json.loads(log_file.read_text()).get("map_key") if log_file.exists() else None
    recorded = recorded_map(setup, Path(replay), key)
    if recorded is None:
        from ..data.mapbuild import harness_source
        return harness_source()
    from ..data.mpq import MpqArchive
    try:
        with MpqArchive(recorded) as m:
            return m.read("war3map.j").decode("latin-1")
    except OSError:
        return ""


def replay_markers(setup: "GameSetup", replay: str | os.PathLike) -> bool:
    """Whether the map `replay` plays back on draws video markers (protocol.VisMark etc.; harnesses
    from before them ignore the commands, and the overlay then draws the marks itself)."""
    return "function W3S_VisClear" in replay_script(setup, replay)


def commands_path(replay: Path) -> Path:
    """The agent-order log saved next to a replay."""
    return replay.with_name(replay.stem + ".commands.json")


def _winpath(p: Path) -> str:
    return "Z:" + str(p.resolve()).replace("/", "\\")


# Attributes that belong to one game process; a warm spare's are swapped in on restart.
_PROCESS_ATTRS = ("name", "_own_display", "_display", "proc", "_server", "_conn", "_rfile", "prefix", "ipc_dir",
                  "inst_dir", "_last_seq", "_units", "last_obs", "_sent_at", "_need_snapshot", "_cmd_log",
                  "_proc_episode", "_proc_steps", "_name_lock")


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
        self.game_wall_total = 0.0  # the same, never reset (profiling)
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
        self._proc_steps = 0  # steps taken by the current game process
        self._playback: dict[str, list[int]] | None = None
        self._name_lock = None  # machine-wide lock on `name` (its prefix, IPC dir) while in use
        self._pending_go: list[str] = []  # options for the next "GO" (see shim/sync.c)
        self._on_frame: Callable[[int], None] | None = None
        self._on_audio: Callable[[bytes, int, int, int, int], None] | None = None

    # ---- launch ---------------------------------------------------------------------------

    def _lock_name(self) -> None:
        """Two games must never share an instance name: they would share a Wine prefix."""
        if self._name_lock is not None:
            return
        import fcntl

        lock_dir = Path("/dev/shm/warcraftsim/instances")
        lock_dir.mkdir(parents=True, exist_ok=True)
        f = open(lock_dir / f"{self.name}.lock", "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            raise GameError(f"game instance name {self.name!r} is in use by another game") from None
        self._name_lock = f

    def _prepare(self) -> None:
        self._lock_name()
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
        # map + game config
        work = self.prefix / "drive_c" / wine.WORK_DIR
        # replays store the map as "..\\w3sim\\map.w3x" and resolve it from the Documents folder
        doc_link = wine.documents_dir(self.prefix) / wine.WORK_DIR
        if not doc_link.is_symlink():
            doc_link.symlink_to(work, target_is_directory=True)
        maps_cache = paths.CACHE_DIR / "maps"
        self._map_key = self.setup.map_key()  # replays name it: they play back only on this map
        cached = maps_cache / f"{self._map_key}.w3x"
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
        if self.setup.audio:
            music = self.setup.music_volume > 0
            text = _set_reg_values(text, r"Software\\Blizzard Entertainment\\Warcraft III\\Sound", {
                "sfx": 1, "sfxvolume": 100, "ambient": 1, "movement": 1, "unit": 1, "positional": 0,
                "music": int(music), "musicvolume": max(self.setup.music_volume, 0)})
        gameplay = {"healthbars": int(self.setup.health_bars), "mousescrolldisable": int(not self.setup.mouse_scroll)}
        text = _set_reg_values(text, r"Software\\Blizzard Entertainment\\Warcraft III\\Gameplay", gameplay)
        user_reg.write_text(text, encoding="latin-1")

    warm_spare_child = False  # a spare never starts its own spare

    def start(self) -> Observation:
        """Launch the game and return the first observation (game time 0)."""
        if self.proc:
            raise GameError("already started")
        self._prepare()
        if not self._display:
            self._own_display = Xvfb(*self.setup.window)
            self._display = self._own_display.display
        self.episode = 0
        obs = self._launch()
        self._start_spare()
        return obs

    def _launch(self, attempts: int = 2) -> Observation:
        for attempt in range(attempts):
            with _LaunchSlot():
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
        writes observations. Commands passed to step() are ignored, except those for watching
        (Command.playback: Camera, Snapshot, the video markers)."""
        log_file = commands_path(Path(replay))
        saved = json.loads(log_file.read_text()) if log_file.exists() else {}
        self._playback = saved.get("commands", {})
        self._stop_process()
        if self.prefix is None:
            self._prepare()
        # the map the replay was recorded on (a harness change since then builds another one)
        recorded = recorded_map(self.setup, Path(replay), saved.get("map_key"))
        if recorded is not None:
            shutil.copyfile(recorded, self.prefix / "drive_c" / wine.WORK_DIR / "map.w3x")
        if not self._display:
            self._own_display = Xvfb(*self.setup.window)
            self._display = self._own_display.display
        shutil.copyfile(replay, self.prefix / "drive_c" / wine.WORK_DIR / "replay.w3g")
        for f in self.ipc_dir.iterdir():
            f.unlink()
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
                            # real time until the game starts: a fast clock during loading becomes a
                            # backlog the engine simulates in one frame at the start (in replays too:
                            # videos then skip their first seconds); the step speed is sent with the
                            # first GO
                            W3SIM_SPEED=str(self.setup.launch_speed), W3SIM_TURBO_MS=str(self.setup.turbo_ms),
                            W3SIM_WAIT_FLOOR=str(self.setup.wait_floor_ms),
                            W3SIM_AUDIO="1" if self.setup.audio else "0",
                            W3SIM_LOG=_winpath(log_path))
        out = open(self.inst_dir / "wine.log", "wb") if self.keep_logs else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            ["wine", _winpath(launcher), _winpath(dll), f"C:\\{wine.GAME_LINK}\\Warcraft III.exe", "-window",
             "-loadfile", self._loadfile or f"C:\\{wine.WORK_DIR}\\game.wgc"],
            env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
        )
        proc, prefix = self.proc, self.prefix
        reaper.track(proc, lambda: (prefix and wine.kill_prefix(prefix), proc.kill()))
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
            for log in ("shim.log", "wine.log"):  # the retry would overwrite them
                if (self.inst_dir / log).exists():
                    shutil.copyfile(self.inst_dir / log, self.inst_dir / f"launch-timeout-{log}")
            raise GameError(f"game did not reach the harness within {self.timeout:.0f}s (see {self.inst_dir})")
        self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._conn.settimeout(self.timeout)
        self._rfile = self._conn.makefile("rb")
        self._last_seq = None
        self.last_obs = None
        self._sent_at = None
        self._cmd_log = {}
        self._proc_episode = -1
        self._proc_steps = 0
        self.episode += 1
        self._pending_speed = self._speed
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
            if line.startswith(b"FRAME"):  # frame capture: the game waits until we answer
                parts = line.split()  # FRAME n [pcm_bytes rate channels bits latency_bytes]
                if len(parts) >= 7:
                    pcm = self._rfile.read(int(parts[2]))
                    if self._on_audio is not None:
                        self._on_audio(pcm, *(int(v) for v in parts[3:7]))
                if self._on_frame is not None:
                    self._on_frame(int(parts[1]))
                self._conn.sendall(b"OK\n")
                continue
            if not line.startswith(b"OBS"):
                continue
            parts = line.split()
            if len(parts) >= 3:  # the observation follows in memory (w3shim obs capture)
                payload = self._rfile.read(int(parts[2]))
                obs = parse_token_lines(payload, self.order_names)
            else:
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

    def _reply(self, extra: str = "", ints: Sequence[int] = ()) -> None:
        """Let the game continue; `ints`: encoded commands for the harness to execute."""
        msg = "GO"
        if self._pending_speed is not None:
            msg += f" speed={self._pending_speed}"
            self._pending_speed = None
        if self._pending_go:
            msg += " " + " ".join(self._pending_go)
            self._pending_go.clear()
        if extra:
            msg += " " + extra
        if ints:
            msg += f" A {len(ints)} " + " ".join(str(int(v)) for v in ints)
        self._conn.sendall(msg.encode() + b"\n")

    def set_frame_capture(self, frame_ms: float | None, on_frame: Callable[[int], None] | None = None,
                          on_audio: Callable[[bytes, int, int, int, int], None] | None = None) -> None:
        """Video recording, from the next step on. The virtual clock advances exactly `frame_ms`
        per rendered frame instead of with wall time (None: wall time again), so each frame covers
        the same game time however loaded the machine is. `on_frame(n)` is called after every
        presented frame while the game waits, so the window shows exactly that frame. With
        setup.audio, `on_audio(pcm, rate, channels, bits, latency)` first receives the audio played
        during that frame (exactly frame_ms of it). Miles mixes ahead: a sound starting in this
        frame is `latency` bytes later in the audio stream (shift the stream by it to align)."""
        self._on_frame = on_frame
        self._on_audio = on_audio
        self._pending_go.append(f"frame={frame_ms or 0:g} capture={1 if on_frame else 0}")

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
            # replay playback: the recorded orders; only commands for watching are added
            extra = [c for c in commands if c.playback]
            ints = self._playback.get(key, []) + encode_commands(extra)
        elif ints:
            self._cmd_log[key] = ints
        # a scenario Restart among them (e.g. replayed from the command log): the next observation
        # starts a new episode (sequence 0 again, and the key of the recorded commands changes)
        self._restart_sent = Op.RESTART in command_ops(ints)
        self._sent_at = time.perf_counter()
        self._reply(ints=ints)

    _restart_sent = False

    def receive(self) -> Observation:
        """Second half of step(): wait for the next observation."""
        prev_ms = self.last_obs.game_ms if self.last_obs else 0
        if self._restart_sent:
            self._restart_sent = False
            self._last_seq = None
            obs = self._await_observation(new_episode=True)
            self.episode += 1
            return obs
        obs = self._await_observation()
        self.steps += 1
        self._proc_steps += 1
        if self._sent_at is not None and obs.game_ms > prev_ms:
            dt = time.perf_counter() - self._sent_at
            self._game_wall += dt
            self.game_wall_total += dt
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

    def restart(self, relaunch: bool = False, spawns: Sequence[Command] = ()) -> Observation:
        """End the current episode and start a new one; scenarios: with `spawns` (QueueSpawn) on
        top of the scenario's units, in the same in-game restart when there is one.

        Scenario maps reset inside the running game (units are removed and respawned). Melee
        games are relaunched: RestartGame/ChangeLevel/LoadGame all return a .wgc game to the
        main menu, and the .wgc is what sets exact slots and AI difficulty. relaunch=True always
        starts a fresh process (e.g. so a saved replay holds exactly one episode).
        """
        jitter = 0.7 + 0.6 * random.Random(self.base_name).random()  # fixed per game
        recycle = bool(self.setup.recycle_steps and self._proc_steps >= self.setup.recycle_steps * jitter)
        # a parked game (see scenario_spare) means this process only ran a fresh-process episode:
        # the next normal episode continues the parked game, even if that episode was cut short
        parked = self._spare is not None and self._spare._parked
        if (self.setup.scenario is not None and not self._ended and self._playback is None and not relaunch
                and not recycle and not parked):
            ints = encode_commands([*spawns, Restart()])
            key = f"{self._proc_episode}:{self._last_seq}"
            self._cmd_log[key] = self._cmd_log.get(key, []) + ints
            self._reply(ints=ints)
            self._last_seq = None
            obs = self._await_observation(new_episode=True)
            self.episode += 1
            return obs
        obs = self._restart_process(relaunch, recycle, parked)
        return self.respawn(spawns) if spawns else obs

    def _restart_process(self, relaunch: bool, recycle: bool, parked: bool) -> Observation:
        """restart() in another process: the parked game, the warm spare, or a fresh launch."""
        was_ended = self._ended
        self._ended = False
        self._playback = None
        spare = self._take_spare()
        if spare is not None and spare._parked:
            if not relaunch:  # back to the paused game: a new episode by in-game restart
                return self._resume_parked(spare)
            # a fresh process is wanted: never the parked game (its replay would hold every episode
            # it ever ran, and a video of one would re-simulate them all, desynced). It stays parked.
            self._spare, self._spare_thread, self._spare_error = spare, None, None
            spare = None
        elif spare is not None:
            # park the running game for a fresh-process episode (a video), to continue it afterwards;
            # a recycled one (too many steps in one process) is retired
            park = (self.setup.scenario is not None and relaunch and not recycle and not was_ended
                    and self.proc is not None and self.proc.poll() is None and self._conn is not None)
            return self._swap_in(spare, park=park)
        self._stop_process()
        for f in self.ipc_dir.iterdir():
            f.unlink()
        return self._launch()

    def respawn(self, commands: Sequence[Command]) -> Observation:
        """Scenarios: a new episode in the running game with extra units (protocol.QueueSpawn
        commands, spawned after the old units are removed)."""
        if self.setup.scenario is None:
            raise GameError("respawn needs a scenario")
        ints = encode_commands([*commands, Restart()])
        key = f"{self._proc_episode}:{self._last_seq}"
        self._cmd_log[key] = self._cmd_log.get(key, []) + ints
        self._reply(ints=ints)
        self._last_seq = None
        obs = self._await_observation(new_episode=True)
        self.episode += 1
        return obs

    # ---- warm spare (melee) ---------------------------------------------------------------

    def _wants_spare(self) -> bool:
        return (self.setup.warm_spare and (self.setup.scenario is None or self.setup.scenario_spare)
                and not self.warm_spare_child and not self._display_shared())

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

    _parked = False  # a spare that is a paused scenario game (see scenario_spare), not a fresh one

    def _take_spare(self) -> "GameInstance | None":
        spare, thread = self._spare, self._spare_thread
        self._spare = self._spare_thread = None
        if spare is None:
            return None
        if spare._parked:
            return spare
        thread.join()
        if self._spare_error is not None or spare.last_obs is None:
            spare.close()
            return None
        return spare

    def _swap_in(self, spare: "GameInstance", park: bool = False) -> Observation:
        """Continue with the spare's process (waiting at game time 0). The retired process is shut
        down in the background, which then loads the next spare under its name, or, with `park`
        (scenarios), kept as it is as the next spare."""
        for attr in _PROCESS_ATTRS:
            mine, theirs = getattr(self, attr), getattr(spare, attr)
            setattr(self, attr, theirs)
            setattr(spare, attr, mine)
        self._pending_speed = self._speed
        self.episode += 1
        if park:
            spare._parked = True
            self._spare, self._spare_thread, self._spare_error = spare, None, None
        else:
            spare._parked = False
            self._start_spare(recycled=spare)
        return self.last_obs

    def _resume_parked(self, parked: "GameInstance") -> Observation:
        """Continue with a parked scenario game: a new episode by in-game restart. The process
        that ran until now (its game ended to save a replay) reloads as the next spare."""
        for attr in _PROCESS_ATTRS:
            mine, theirs = getattr(self, attr), getattr(parked, attr)
            setattr(self, attr, theirs)
            setattr(parked, attr, mine)
        parked._parked = False
        self._pending_speed = self._speed
        key = f"{self._proc_episode}:{self._last_seq}"
        self._cmd_log[key] = self._cmd_log.get(key, []) + Restart().encode()
        self._reply(ints=Restart().encode())
        self._last_seq = None
        obs = self._await_observation(new_episode=True)
        self.episode += 1
        self._start_spare(recycled=parked)
        return obs

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
        self._reply(ints=EndGame().encode())
        deadline = time.time() + timeout
        while time.time() < deadline:
            if replay.exists() and replay.stat().st_mtime > before and replay.stat().st_size > 0:
                time.sleep(0.2)  # let the writer finish
                dest = Path(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(replay, dest)
                commands_path(dest).write_text(json.dumps({"format": "warcraftsim-commands", "version": 1,
                                                           "map_key": getattr(self, "_map_key", None),
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
        reaper.untrack(self.proc)
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
        if self._name_lock is not None:
            self._name_lock.close()
            self._name_lock = None

    def __enter__(self) -> "GameInstance":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
