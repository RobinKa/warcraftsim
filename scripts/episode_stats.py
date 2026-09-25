"""Action and combat statistics of a run's episodes over training, and wins against losses.

    python scripts/episode_stats.py abilmk2-1 [--parts 4]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

RUNS = Path(os.environ.get("WARCRAFTSIM_RUNS", Path(__file__).resolve().parent.parent / "runs"))
ACTS = ("noop", "stop", "retreat", "move", "attack", "cast", "attack_invalid", "attack_weakest", "focus_fire",
        "cast_invalid")
COMBAT = ("dealt", "taken", "kills", "losses")


def line(rows: list[dict], label: str) -> str:
    if not rows:
        return f"{label:12s} (none)"
    n = len(rows)
    acts = {k: sum(r.get("act", {}).get(k, 0) for r in rows) / n for k in ACTS}
    combat = {k: sum(r.get("combat", {}).get(k, 0) for r in rows) / n for k in COMBAT}
    win = sum(r["outcome"] > 0 for r in rows) / n
    return (f"{label:12s} n={n:5d} win {win:.2f} len {sum(r['length'] for r in rows) / n:3.0f} | "
            + " ".join(f"{k} {v:.2f}" for k, v in acts.items() if v) + " | "
            + " ".join(f"{k} {v:.2f}" for k, v in combat.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--parts", type=int, default=4, help="training split into this many parts")
    ap.add_argument("--window", type=int, default=2000, help="episodes per part and for the last ones")
    args = ap.parse_args()
    rows = []
    for f in glob.glob(str(RUNS / args.run / "episodes-*.jsonl")):
        rows += [json.loads(x) for x in open(f) if x.strip() and '"event"' not in x]
    rows.sort(key=lambda r: r["time"])
    step = max(1, len(rows) // args.parts)
    for i in range(0, len(rows), step):
        print(line(rows[i:i + min(step, args.window)], f"ep {i}"))
    last = rows[-2 * args.window:]
    print(line([r for r in last if r["outcome"] > 0], "last wins"))
    print(line([r for r in last if r["outcome"] <= 0], "last losses"))


if __name__ == "__main__":
    main()
