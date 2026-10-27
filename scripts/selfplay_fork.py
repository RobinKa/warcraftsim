"""Start a new self-play run from another run's latest checkpoint (an experiment: the dashboard's lineage
shows it as the parent's child): stop the parent, take the flags of its last launch (run.json), some
changed, and launch the new run with the parent's checkpoint as its starting weights (--start-weights;
--init stays the KL anchor). The new run takes over the parent's league (--league-from: its past snapshots
with their records, its curriculum levels and balance; --league none: a league of its own from scratch)
and counts its steps from 0. It runs from a checkout of its commit (HEAD; --commit), made and built before the
parent stops (selfplay_restart.pinned; --dry-run: only that, and the command printed).

    python3 scripts/selfplay_fork.py fgself-12 fgself-13 --title "the change in a few words" \
        --note "what changes and why" [--set shared-noise=0.5] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfplay_restart import ROOT, add_launch_args, pinned, start, stop  # noqa: E402


def as_read(p: Path) -> str:
    """A path as the learner reads it in its checkout: one in runs as runs/... (a link there to this tree's runs),
    another absolute."""
    runs = (ROOT / "runs").resolve()
    return str(Path("runs") / p.relative_to(runs)) if p.is_relative_to(runs) else str(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parent")
    ap.add_argument("child")
    ap.add_argument("--note", required=True, help="the new run's notes: what it changes and why (notes.md)")
    ap.add_argument("--title", default="", help="the change in a few words (summary.txt: the lineage's line for it)")
    ap.add_argument("--league", help="the run whose league the new one starts with (default: the parent's; none: "
                    "its own)")
    ap.add_argument("--set", action="append", default=[], help="flag=value (without the dashes); flag= drops it; \"start\" in a value: the fork's start weights (e.g. eval-checkpoints=start runs/fgself-130/...)")
    ap.add_argument("--weights", help="the parent's checkpoint to start from (default: its latest)")
    add_launch_args(ap)
    args = ap.parse_args()
    parent = ROOT / "runs" / args.parent
    child = ROOT / "runs" / args.child
    if child.exists():
        raise SystemExit(f"{child} exists: a fork starts a new run")
    launch = json.loads((parent / "run.json").read_text())["launch"]
    words = launch.get("argv") or launch["command"].split()[3:]
    code = pinned(args.commit, args.allow_dirty, args.dry_run)  # (before the parent stops: a failed build leaves it)
    if not args.dry_run:
        stop(args.parent)
    weights = Path(args.weights) if args.weights else max((parent / "checkpoints").glob("[0-9]*.pt"))
    weights = (weights if weights.is_absolute() else ROOT / weights).resolve()
    flags: list[list[str]] = []
    for w in words:  # [flag, values...] in order
        m = re.fullmatch(r"(--[a-z0-9][a-z0-9-]*)(?:=(.*))?", w, re.S)  # (a flag, or --flag=value; a note that
        if m:                                                               # starts with "--" is a value)
            flags.append([m.group(1)] + ([m.group(2)] if m.group(2) is not None else []))
        elif flags:
            flags[-1].append(w)
    flags = [f for f in flags if f[0] not in ("--name", "--note", "--title", "--resume", "--resume-weights",
                                               "--start-weights", "--league-from")]
    for s in args.set:
        k, _, v = s.partition("=")
        flags = [f for f in flags if f[0] != f"--{k}"]
        if v:  # ("start" in a value: the fork's start weights, as chosen after the parent stopped and saved its last;
            # a checkpoint named before that had been an older one: fgself-159's eval list; audit 17)
            flags.append([f"--{k}", *[as_read(weights) if x == "start" else x for x in v.split()]])
    argv = [sys.executable, "-m", "warcraftsim.fullgame.selfplay", "--name", args.child,
            *[w for f in flags for w in f], "--start-weights", as_read(weights), f"--note={args.note}",
            *([f"--title={args.title}"] if args.title else []),
            *(["--league-from", args.league or args.parent] if (args.league or "") != "none" else [])]
    start(argv, code, args.child, args.dry_run)
    if not args.dry_run:
        print(f"{args.child} from {as_read(weights)} in {code}:", " ".join(shlex.quote(a) for a in argv[3:]))


if __name__ == "__main__":
    main()
