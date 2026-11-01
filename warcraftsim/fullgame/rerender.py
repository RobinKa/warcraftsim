"""A whole game's video from its replay, on this machine: what scripts/rerender_fullgame.py, the dashboard's Re-render
button and Replays page (render_request.py) and a run's videos on another machine (video_host.py, which runs this
file there) have in common.

A replay needs the settings it was recorded with: its .commands.json (the orders and the map it was recorded on), its
.trace.json (the panel, the players), and the run's launch flags (run.json: step, handicap, game length, cuts, Wine).
A game rendered on request (runs/renders) has its run's flags: the request names the run.

    python -m warcraftsim.fullgame.rerender fgself-64 game-1792106881-a4 [--runs DIR] [--speed 4] [--out file.mp4]

Default output: the run's videos/STEM.mp4 (replacing it). --speed N: N game seconds a video second, every Nth frame,
and the sound N times as fast at its pitch. As fast as the machine draws: no real-time pacing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from .. import paths

RUNS = paths.REPO_ROOT / "runs"
STEM = re.compile(r"game-\d+-a\d+")
SPEEDS = (1.0, 2.0, 4.0, 8.0)  # (at 40 frames a second, a whole number of the engine's 25 ms turns a frame)
FPS = 40


def flags_dir(run_dir: Path, stem: str) -> Path:
    """Where the replay's launch flags are: its run, or for a game rendered on request (runs/renders) the request's run."""
    run_dir = Path(run_dir)
    if (run_dir / "run.json").exists():
        return run_dir
    m = re.fullmatch(r"game-\d+-a(\d+)", stem)
    if m:
        try:
            req = json.loads((run_dir / "requests" / f"{m.group(1)}.json").read_text())
            return run_dir.parent / req["request"]["run"]
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    return run_dir


def launch_argv(run_dir: Path, stem: str) -> list[str]:
    return json.loads((flags_dir(run_dir, stem) / "run.json").read_text())["launch"]["argv"]


def map_key(replay: Path) -> str | None:
    """The cached map the replay was recorded on, from its command file's first bytes (the orders follow: megabytes)."""
    with open(replay.with_name(replay.stem + ".commands.json")) as f:
        m = re.search(r'"map_key": "([^"]*)"', f.read(400))
    return m.group(1) if m else None


def cannot_play(run_dir: Path, stem: str) -> str:
    """Why the replay cannot be played back here ("": it can): what a render needs, checked without loading it."""
    replay = Path(run_dir) / "replays" / f"{stem}.w3g"
    for f, what in ((replay, "the replay"), (replay.with_suffix(".trace.json"), "its trace (the panel, the players)"),
                    (replay.with_name(stem + ".commands.json"), "its orders (.commands.json)")):
        if not f.exists():
            return f"{what} is gone"
    try:
        argv = launch_argv(run_dir, stem)
    except (OSError, KeyError, ValueError):
        return f"the run's launch flags are gone (no run.json with its launch for {Path(run_dir).name})"
    if not isinstance(argv, list):
        return "the run's launch flags are unreadable"
    key = map_key(replay)
    if key and not (paths.CACHE_DIR / "maps" / f"{key}.w3x").exists():
        return f"the map it was recorded on ({key}) is no longer in the map cache"
    return ""


def local_wine(path: str) -> str:
    """A run's Wine as this machine has it: another machine's home in the path replaced (.../wc3/ge/X/files/bin)."""
    if Path(path).exists() or "/wc3/" not in path:
        return path
    here = str(Path.home() / "wc3" / path.split("/wc3/", 1)[1])
    return here if Path(here).exists() else path


def game_setup(argv: list[str], trace: dict):
    """The game the replay was recorded in, as far as its playback depends on it."""
    from ..runtime.instance import Agent, BuiltinAI, GameSetup

    def flag(name: str, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    handicap = int(flag("--handicap", 50))
    slots = []
    for side in sorted(trace["sides"], key=lambda s: s["player"]):
        race = re.search(r"\((human|orc|undead|nightelf)\)", side["name"]).group(1)
        if side["kind"] == "ai":
            difficulty = re.search(r"(easy|normal|insane)", side["name"])
            slots.append(BuiltinAI(race, difficulty.group(1) if difficulty else "insane", handicap=handicap))
        else:
            slots.append(Agent(race, handicap=handicap))
    return GameSetup(map=flag("--map", "duelrush"), slots=slots, step_seconds=float(flag("--step-seconds", 0.5)),
                     max_game_seconds=float(flag("--max-minutes", 4.0)) * 60, victory="decisive", window=(320, 240),
                     draw=False, d3d_thread=False, render_threads=0, cuts=int(flag("--game-cuts", 0)) or None)


def render(run_dir: Path, stem: str, out: Path, speed: float = 1.0, name: str = "rerender", nice: int = 0) -> Path:
    """Render the replay of `run_dir`/replays/`stem` into `out`. This process's Wine is set here: once a process."""
    from ..runtime import wine
    argv = launch_argv(run_dir, stem)
    if "--wine" in argv:
        i = argv.index("--wine") + 1
        wine.use_wine(local_wine(argv[i]), bool(int(argv[argv.index("--fsync") + 1])) if "--fsync" in argv else True)
    from ..video import render_replay
    from .overlay import FullGameOverlay
    replay = Path(run_dir) / "replays" / f"{stem}.w3g"
    trace = json.loads(replay.with_suffix(".trace.json").read_text())
    return render_replay(game_setup(argv, trace), replay, out, speed=speed, name=name, crf=28,
                         overlay=FullGameOverlay(trace), fit_all=True, max_steps=len(trace.get("steps") or []) + 40,
                         nice=nice, fps=FPS)


def render_pinned(run_dir: Path, stem: str, out: Path, speed: float = 1.0) -> Path:
    """The same with the code of the run's own commit (codepin: its checkout, made if none is left; the script as it was
    then, its shim's pacing too), for a replay this code cannot play back. So far every run's that was tried plays on the
    current code, back to fgself-12. Raises RuntimeError with the reason."""
    from .. import codepin
    run_dir = Path(run_dir)
    if not (run_dir / "run.json").exists():
        raise RuntimeError("a game rendered on request has no commit of its own")
    commit = (json.loads((run_dir / "run.json").read_text())["launch"].get("git") or {}).get("commit")
    if not commit:
        raise RuntimeError("the run's commit is not recorded")
    try:
        d = codepin.prepare(paths.REPO_ROOT, commit, infer=False, log=lambda _: None)
    except Exception as e:  # noqa: BLE001 (git's or make's)
        raise RuntimeError(f"no checkout of its commit {commit[:12]}: {e}") from None
    script = d / "scripts" / "rerender_fullgame.py"
    if not script.exists():
        raise RuntimeError(f"its commit {commit[:12]} has no scripts/rerender_fullgame.py")
    argv = [sys.executable, str(script), run_dir.name, stem, "--out", str(out)]
    if speed != 1.0:
        if "--speed" not in script.read_text():
            raise RuntimeError(f"its commit {commit[:12]} records at real time only")
        argv += ["--speed", f"{speed:g}"]
    p = subprocess.run(argv, cwd=d, env={**os.environ, "PYTHONPATH": str(d), "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
    if p.returncode != 0 or not Path(out).exists():
        raise RuntimeError(f"its commit {commit[:12]} (exit code {p.returncode}): " + " / ".join((p.stderr or p.stdout).strip().splitlines()[-3:]))
    return Path(out)


def _row_from_trace(run_dir: Path, stem: str) -> dict:
    """A media row for a game no video was listed of: what its trace says."""
    trace = json.loads((Path(run_dir) / "replays" / f"{stem}.trace.json").read_text())
    sides = trace.get("sides") or []
    me = sides[0] if sides else {}
    other = sides[1] if len(sides) > 1 else {}
    race = lambda s: (re.search(r"\((\w+)\)", s.get("name", "")) or [None, "?"])[1]  # noqa: E731
    steps = trace.get("steps") or []
    gt = float(steps[-1].get("game_time") or 0) if steps else 0.0
    o = (trace.get("outcome") or {}).get(str(me.get("player")), 0)
    opponent = re.sub(r"\s*\(\w+\)$", "", other.get("name", "")).replace("snapshot ", "")
    return {"episode": "", "title": f"{race(me)} vs {race(other)} ({opponent})", "outcome": float(o or 0),
            "sub": f"{gt / 60:.1f} game minutes", "opponent": opponent, "game_time": round(gt, 3)}


def install(run_dir: Path, stem: str, part: Path, speed: float, host: str = "local") -> dict:
    """A finished render takes the video's place: only now is the old one replaced. Its media row: the video's latest
    row with this recording's speed (the row is the Videos page's and the Replays page's word on it) -> the row."""
    run_dir = Path(run_dir)
    file = f"videos/{stem}.mp4"
    row = None
    try:
        for line in (run_dir / "media.jsonl").read_text().splitlines():
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("file") == file:
                row = m
    except OSError:
        pass
    row = {**(row or _row_from_trace(run_dir, stem)), "time": time.time(), "kind": "video", "file": file,
           "recorded_speed": speed, "recorded_on": host}
    part.replace(run_dir / file)
    with open(run_dir / "media.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _term(*_):
    raise SystemExit(143)  # (the game, its display and the encoder go down with the finally blocks)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("stem", help="the replay's stem, e.g. game-1792106881-a4")
    ap.add_argument("--runs", type=Path, default=RUNS, help="the runs' directory")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--speed", type=float, default=1.0, help="game seconds a video second (1: real time; 2, 4 or 8 records "
                    "faster, with every 2nd, 4th or 8th frame, and the sound sped up)")
    ap.add_argument("--name", default="rerender", help="the game instance's name (one at a time with this name)")
    ap.add_argument("--nice", type=int, default=0)
    a = ap.parse_args(argv)
    if not STEM.fullmatch(a.stem):
        ap.error(f"not a game's replay: {a.stem}")
    signal.signal(signal.SIGTERM, _term)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # (nothing here needs the GPU; the game: display.SOFTWARE_GL)
    run_dir = a.runs / a.run
    why = cannot_play(run_dir, a.stem)
    if why:
        raise SystemExit(f"{a.run}/{a.stem} cannot be played back: {why}")
    from ..runtime import reaper
    try:
        out = render(run_dir, a.stem, a.out or run_dir / "videos" / f"{a.stem}.mp4", a.speed, a.name, a.nice)
    finally:
        reaper.reap()
    print(out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
