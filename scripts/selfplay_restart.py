"""Restart a whole-game self-play run with --resume: stop its learner (SIGTERM, waiting until it is
gone: its last checkpoint and run.json are written on the way out), clear its leftover games, and
launch it again with the flags of its last launch (run.json), some changed, and a note for the
restart's marker on the dashboard's charts.

A launch runs from a checkout of its commit (HEAD; --commit), not from this tree (warcraftsim/codepin.py): the
checkout is made and built before the run stops, and this tree can change while it runs. Uncommitted code here
would not be in it: refused (--allow-dirty: launched without it). --dry-run prepares the checkout and prints the
command, launching nothing (and stopping nothing).

    python3 scripts/selfplay_restart.py fgself-11 --note "why" [--set curriculum-step=0.04 --set exploiter-share=0.1]
    python3 scripts/selfplay_restart.py fgself-11 --stop          # only stop it
    python3 scripts/selfplay_restart.py fgself-11 --note "why" --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from warcraftsim import codepin  # noqa: E402


def learner_pid(name: str) -> int | None:
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, _, args = line.strip().partition(" ")
        if "python" in args.split(" ")[0] and f"selfplay --name {name} " in args + " " and "selfplay_restart" not in args:
            return int(pid)
    return None


# a self-play run's game instances ("fgsp<slot>_<actor>_<k>", videos "fgvid<slot>"): only one runs
# at a time on this machine. Collections ("demo...") and evaluations keep their games.
SELFPLAY_PREFIX = re.compile(rb"^WINEPREFIX=.*/instances/fg(sp|vid)\d")


def leftovers() -> list[int]:
    """The processes a stopped self-play run can leave: its games and their Wine services (by
    their WINEPREFIX), and Xvfb servers whose parent is gone."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            env = (d / "environ").read_bytes().split(b"\0")
            stat = (d / "stat").read_text()
        except OSError:
            continue
        comm, ppid = stat[stat.index("(") + 1:stat.rindex(")")], int(stat[stat.rindex(")") + 2:].split()[1])
        if any(SELFPLAY_PREFIX.match(kv) for kv in env) or (comm == "Xvfb" and ppid == 1):
            out.append(int(d.name))
    return out


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
    for p in leftovers():
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    videos = ROOT / "runs" / name / "videos"
    for f in list(videos.glob("*.pcm")) + list(videos.glob("*.video.mp4")):  # a video cut off mid-render
        f.unlink()


def add_launch_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--commit", default="HEAD", help="the commit the launch runs (default: this tree's HEAD; "
                    "restart: 'launch' is the run's last launch's)")
    ap.add_argument("--allow-dirty", action="store_true", help="launch although this tree has uncommitted code (not "
                    "in the launch's checkout)")
    ap.add_argument("--dry-run", action="store_true", help="prepare the launch's checkout (built) and print the "
                    "command: nothing stopped or launched")


def pinned(rev: str, allow_dirty: bool = False, dry_run: bool = False) -> Path:
    """The launch's code: the checkout of `rev`, built (codepin.prepare). Uncommitted code in this tree (not in the
    checkout) refuses, unless allowed (a dry run: a warning)."""
    changed = codepin.dirty(ROOT)
    if changed:
        msg = (f"this tree's uncommitted code is not in the launch's ({rev}):\n  " + "\n  ".join(changed[:20])
               + ("\n  ..." if len(changed) > 20 else ""))
        if not (allow_dirty or dry_run):
            raise SystemExit(msg + "\ncommit it, or --allow-dirty")
        print("warning:", msg, file=sys.stderr)
    return codepin.prepare(ROOT, rev, log=lambda m: print(m, file=sys.stderr))


def start(argv: list[str], code: Path, log_name: str, dry_run: bool = False) -> None:
    """The learner started in its checkout (its output appended to runs/<log_name>.log); a dry run prints it. Then
    the checkouts no process uses go (codepin.prune)."""
    env = codepin.launch_env(code)
    if dry_run:
        found = subprocess.run([argv[0], "-c", "import warcraftsim; print(warcraftsim.__file__)"], cwd=code, env=env,
                               capture_output=True, text=True).stdout.strip()
        print(f"dry run: nothing launched. It would run in {code} (PYTHONPATH={env['PYTHONPATH']}; warcraftsim "
              f"imports from {found}), its output appended to {ROOT / 'runs' / f'{log_name}.log'}:")
        print(f"cd {shlex.quote(str(code))} && PYTHONPATH={shlex.quote(str(code))} {shlex.join(argv)}")
        return
    with open(ROOT / "runs" / f"{log_name}.log", "a") as log:
        subprocess.Popen(argv, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        codepin.prune(ROOT, protect=[code], log=lambda m: print(m, file=sys.stderr))
    except Exception as e:  # noqa: BLE001 (launched all the same)
        print(f"warning: old checkouts not pruned: {e}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name")
    ap.add_argument("--note", default="", help="why it restarts (the dashboard's restart marker)")
    ap.add_argument("--title", default="", help="the restart's short name (a few words: the lineage's subtitle)")
    ap.add_argument("--set", action="append", default=[], help="flag=value (without the dashes); flag= drops it")
    ap.add_argument("--stop", action="store_true", help="only stop it")
    ap.add_argument("--weights", help="resume from this checkpoint's weights instead of the latest (a roll back; once)")
    add_launch_args(ap)
    args = ap.parse_args()
    run = ROOT / "runs" / args.name
    launch = json.loads((run / "run.json").read_text())["launch"]
    # (the command line as a list since it was recorded; before, joined with spaces: only a note has any)
    words = launch.get("argv") or launch["command"].split()[3:]  # (python3 -m warcraftsim.fullgame.selfplay ...)
    if args.stop:
        stop(args.name)
        return
    rev = ((launch.get("git") or {}).get("commit") or "HEAD") if args.commit == "launch" else args.commit
    code = pinned(rev, args.allow_dirty, args.dry_run)  # (before the run stops: a failed build leaves it be)
    flags: list[list[str]] = []
    for w in words:  # [flag, values...] in order
        m = re.fullmatch(r"(--[a-z0-9][a-z0-9-]*)(?:=(.*))?", w, re.S)  # (a flag, or --flag=value; a note that
        if m:                                                               # starts with "--" is a value)
            flags.append([m.group(1)] + ([m.group(2)] if m.group(2) is not None else []))
        elif flags:
            flags[-1].append(w)
    flags = [f for f in flags if f[0] not in ("--note", "--title", "--resume", "--resume-weights")]  # (one-shot flags)
    for s in args.set:
        k, _, v = s.partition("=")
        flags = [f for f in flags if f[0] != f"--{k}"]
        if v:
            flags.append([f"--{k}", *v.split()])
    argv = [sys.executable, "-m", "warcraftsim.fullgame.selfplay", *[w for f in flags for w in f], "--resume"]
    if args.weights:
        argv += ["--resume-weights", str(args.weights)]
    if args.note:
        argv += [f"--note={args.note}"]  # (as one word: a note that starts with "--" is no flag)
    if args.title:
        argv += [f"--title={args.title}"]
    if not args.dry_run:
        stop(args.name)
    start(argv, code, args.name, args.dry_run)
    if not args.dry_run:
        print(f"launched in {code}:", " ".join(shlex.quote(a) for a in argv[1:]))


if __name__ == "__main__":
    main()
