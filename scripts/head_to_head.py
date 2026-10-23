"""A run's evaluation games by key (its current policy, with a target, the listed checkpoints, head to head with each):
games, share won, and the share won by thirds of them in time order.

    python scripts/head_to_head.py fgself-130 [--since <unix time>]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--since", type=float, default=0.0, help="only games after this unix time")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    a = ap.parse_args(argv)
    games = collections.defaultdict(list)
    for name in ("episodes.jsonl", "episodes-remote.jsonl"):
        path = a.runs / a.run / name
        if not path.exists():
            continue
        for line in path.open():
            row = json.loads(line)
            if row.get("eval") and row.get("time", 0) >= a.since:
                games[row.get("eval_key", "current")].append((row["time"], row["outcome"]))
    for key, v in sorted(games.items()):
        v.sort()
        thirds = [v[i * len(v) // 3:(i + 1) * len(v) // 3] for i in range(3)]
        print(f"{key[:45]:45} {len(v):5} {sum(o > 0 for _, o in v) / len(v):.3f}  thirds",
              [round(sum(o > 0 for _, o in h) / max(len(h), 1), 2) for h in thirds])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
