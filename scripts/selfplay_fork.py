"""Start a new self-play run from another run's latest checkpoint (an experiment: the dashboard's lineage
shows it as the parent's child): stop the parent, take the flags of its last launch (run.json), some
changed, and launch the new run with the parent's checkpoint as its starting weights (--start-weights;
--init stays the KL anchor). The new run starts its own league and curriculum and counts its steps from 0.

    python3 scripts/selfplay_fork.py fgself-12 fgself-13 --note "what changes and why" [--set shared-noise=0.5]
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfplay_restart import ROOT, stop  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parent")
    ap.add_argument("child")
    ap.add_argument("--note", required=True, help="the new run's notes: what it changes and why (notes.md)")
    ap.add_argument("--set", action="append", default=[], help="flag=value (without the dashes); flag= drops it")
    ap.add_argument("--weights", help="the parent's checkpoint to start from (default: its latest)")
    args = ap.parse_args()
    parent = ROOT / "runs" / args.parent
    child = ROOT / "runs" / args.child
    if child.exists():
        raise SystemExit(f"{child} exists: a fork starts a new run")
    launch = json.loads((parent / "run.json").read_text())["launch"]
    words = launch.get("argv") or launch["command"].split()[3:]
    stop(args.parent)
    weights = Path(args.weights) if args.weights else max((parent / "checkpoints").glob("[0-9]*.pt"))
    weights = (weights if weights.is_absolute() else ROOT / weights).resolve()
    flags: list[list[str]] = []
    for w in words:  # [flag, values...] in order
        if w.startswith("--"):
            flags.append([w])
        elif flags:
            flags[-1].append(w)
    flags = [f for f in flags if f[0] not in ("--name", "--note", "--title", "--resume", "--resume-weights",
                                               "--start-weights")]
    for s in args.set:
        k, _, v = s.partition("=")
        flags = [f for f in flags if f[0] != f"--{k}"]
        if v:
            flags.append([f"--{k}", *v.split()])
    argv = [sys.executable, "-m", "warcraftsim.fullgame.selfplay", "--name", args.child,
            *[w for f in flags for w in f], "--start-weights", str(weights.relative_to(ROOT)), "--note", args.note]
    with open(ROOT / "runs" / f"{args.child}.log", "a") as log:
        subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"{args.child} from {weights.relative_to(ROOT)}:", " ".join(shlex.quote(a) for a in argv[3:]))


if __name__ == "__main__":
    main()
