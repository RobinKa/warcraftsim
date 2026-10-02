"""Restart a whole-game self-play run with --resume: stop its learner (SIGTERM, waiting until it is
gone: its last checkpoint and run.json are written on the way out), clear its leftover games, and
launch it again with the flags of its last launch (run.json), some changed, and a note for the
restart's marker on the dashboard's charts.

    python3 scripts/selfplay_restart.py fgself-11 --note "why" [--set curriculum-step=0.04 --set exploiter-share=0.1]
    python3 scripts/selfplay_restart.py fgself-11 --stop          # only stop it
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def learner_pid(name: str) -> int | None:
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, _, args = line.strip().partition(" ")
        if "python" in args.split(" ")[0] and f"selfplay --name {name} " in args + " " and "selfplay_restart" not in args:
            return int(pid)
    return None


def stop(name: str, timeout: float = 600.0) -> None:
    pid = learner_pid(name)
    if pid is None:
        print(f"{name}: no learner running")
    else:
        os.kill(pid, signal.SIGTERM)
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(2)
        else:
            raise SystemExit(f"{name}: the learner ({pid}) did not stop within {timeout:.0f} s")
        print(f"{name}: stopped after {time.time() - t0:.0f} s")
    time.sleep(5)
    # the games (and Wine's services) of every self-play run: only one runs at a time on this machine
    for pattern in ("Warcraft III.exe", "wineserver", "C:.windows.system32"):  # (regular expressions)
        subprocess.run(["pkill", "-f", pattern], capture_output=True)
    subprocess.run(["pkill", "-x", "Xvfb"], capture_output=True)
    videos = ROOT / "runs" / name / "videos"
    for f in list(videos.glob("*.pcm")) + list(videos.glob("*.video.mp4")):  # a video cut off mid-render
        f.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name")
    ap.add_argument("--note", default="", help="why it restarts (the dashboard's restart marker)")
    ap.add_argument("--set", action="append", default=[], help="flag=value (without the dashes); flag= drops it")
    ap.add_argument("--stop", action="store_true", help="only stop it")
    args = ap.parse_args()
    run = ROOT / "runs" / args.name
    launch = json.loads((run / "run.json").read_text())["launch"]
    # (the command line as a list since it was recorded; before, joined with spaces: only a note has any)
    words = launch.get("argv") or launch["command"].split()[3:]  # (python3 -m warcraftsim.fullgame.selfplay ...)
    stop(args.name)
    if args.stop:
        return
    flags: list[list[str]] = []
    for w in words:  # [flag, values...] in order
        if w.startswith("--"):
            flags.append([w])
        elif flags:
            flags[-1].append(w)
    flags = [f for f in flags if f[0] not in ("--note", "--resume")]
    for s in args.set:
        k, _, v = s.partition("=")
        flags = [f for f in flags if f[0] != f"--{k}"]
        if v:
            flags.append([f"--{k}", *v.split()])
    argv = [sys.executable, "-m", "warcraftsim.fullgame.selfplay", *[w for f in flags for w in f], "--resume"]
    if args.note:
        argv += ["--note", args.note]
    with open(ROOT / "runs" / f"{args.name}.log", "a") as log:
        subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print("launched:", " ".join(shlex.quote(a) for a in argv[1:]))


if __name__ == "__main__":
    main()
