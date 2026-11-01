"""Recording a video on another machine (a run's --video-host, the dashboard's choice of host): the games box has
idle cores, this one has a learner and its games.

The code goes to its own place there (~/wc3/video/code/<code version>: the checkout's package, native sources and built
shim, which is the run's pinned code when a run asks), apart from what the live actors run (~/warcraftsim there:
wire.code_version checks that) and using only that one's Python and its built libraries. Then the replay with its
trace and orders, the run's run.json and the map it was recorded on (~/.cache/warcraftsim/maps, added if missing, named
by its key). The render runs niced, on no GPU, under a lock (one at a time there, whoever asks) and a time limit, and
the MP4 comes back. Any failure is a RuntimeError with the reason; the caller skips the video, or says so.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .. import paths
from . import rerender

BASE = "wc3/video"  # (under the host's home)
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"]
RSH = "ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
PYTHON = "warcraftsim/.venv/bin/python"  # (the actors' environment there, used and not changed)
LOCK_WAIT = 3 * 3600  # seconds to wait for the recording ahead of this one
KEEP_CODE = 3  # code versions kept there


def _run(cmd: list[str], timeout: float, what: str) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{what}: no answer in {timeout:.0f} s") from None
    except OSError as e:
        raise RuntimeError(f"{what}: {e}") from None
    if p.returncode != 0:
        tail = "\n".join((p.stderr or p.stdout).strip().splitlines()[-6:])
        raise RuntimeError(f"{what} (exit code {p.returncode}): {tail}")
    return p.stdout


def _ssh(host: str, script: str, timeout: float, what: str) -> str:
    return _run([*SSH, host, script], timeout, what)


def _rsync(args: list[str], timeout: float, what: str) -> str:
    return _run(["rsync", "-a", "-e", RSH, *args], timeout, what)


def mirror_code(host: str, root: Path | None = None) -> str:
    """The checkout's code in its own place on the host, once per code version -> the version."""
    from .wire import code_version
    root = Path(root or paths.REPO_ROOT)
    tag = code_version(root)[:16]
    where = f"{BASE}/code/{tag}"
    if _ssh(host, f"mkdir -p {where} {BASE}/runs {BASE}/out; test -f {where}/.ok && touch {where} && echo yes || true", 60,
            "the host's code").strip() == "yes":
        return tag
    files = ["warcraftsim", "native", "pyproject.toml", "build/shim/w3shim.dll", "build/shim/w3launch.exe"]
    if (root / "build" / "native").is_dir():
        files.append("build/native")
    _run(["rsync", "-a", "-R", "--exclude=__pycache__", "-e", RSH, *[f"{root}/./{f}" for f in files], f"{host}:{where}/"], 300,
         "copying the code")
    # (the oldest versions go; never the one in use, the newest)
    _ssh(host, f"touch {where}/.ok; cd {BASE}/code && ls -1t | tail -n +{KEEP_CODE + 1} | xargs -r rm -rf", 60, "the host's code")
    return tag


def render_on_host(host: str, run_dir: Path, stem: str, part: Path, speed: float = 1.0, root: Path | None = None,
                   game_seconds: float = 2400.0, log=print) -> Path:
    """Record the replay of `run_dir`/replays/`stem` on `host` into the local file `part`. Raises RuntimeError."""
    if not rerender.STEM.fullmatch(stem):
        raise RuntimeError(f"not a game's replay: {stem}")
    run_dir, part = Path(run_dir), Path(part)
    why = rerender.cannot_play(run_dir, stem)
    if why:
        raise RuntimeError(f"cannot be played back: {why}")
    root = Path(root or paths.REPO_ROOT)
    flags = rerender.flags_dir(run_dir, stem)
    run = flags.name
    tag = mirror_code(host, root)
    there = f"{BASE}/runs/{run}"
    _ssh(host, f"mkdir -p {there}/replays .cache/warcraftsim/maps", 60, "the host's run")
    replay = run_dir / "replays" / f"{stem}.w3g"
    _rsync([str(flags / "run.json"), f"{host}:{there}/run.json"], 120, "copying run.json")
    _rsync([str(replay), str(replay.with_suffix(".trace.json")), str(replay.with_name(stem + ".commands.json")),
            f"{host}:{there}/replays/"], 300, "copying the replay")
    key = rerender.map_key(replay)
    if key:
        _rsync(["--ignore-existing", str(paths.CACHE_DIR / "maps" / f"{key}.w3x"), f"{host}:.cache/warcraftsim/maps/"], 300,
               "copying the map")
    # the render's limit: a game takes at most about its own length here, by a wide margin, plus the map's load
    limit = int(1800 + 3 * game_seconds)
    out = f"{BASE}/out/{stem}.mp4"
    cmd = (f"cd ~/{BASE}/code/{tag} && flock -w {LOCK_WAIT} ~/{BASE}/lock timeout -k 60 {limit} nice -n 10 "
           f"env CUDA_VISIBLE_DEVICES= WARCRAFTSIM_STORMLIB=$HOME/warcraftsim/build/stormlib/libstorm.so "
           f"WARCRAFTSIM_PJASS=$HOME/warcraftsim/build/pjass/pjass "
           f"$HOME/{PYTHON} -m warcraftsim.fullgame.rerender {shlex.quote(run)} {shlex.quote(stem)} "
           f"--runs $HOME/{BASE}/runs --speed {speed:g} --out $HOME/{out} --name videohost")
    try:
        _ssh(host, cmd, LOCK_WAIT + limit + 120, f"recording on {host}")
        part.parent.mkdir(parents=True, exist_ok=True)
        _rsync([f"{host}:{out}", str(part)], 1800, "copying the video back")
    finally:
        try:  # (its files there, and any left-over parts of this video)
            _ssh(host, f"rm -f {there}/replays/{stem}.* {BASE}/out/{stem}.*", 60, "cleaning up")
        except RuntimeError as e:
            log(f"video: {e}")
    return part
