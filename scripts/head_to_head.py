"""A run's evaluation games by key (its current policy, with a target, the listed checkpoints, head to head with each):
games, the score (a win 1, a tie half: ties had counted as losses; audit 4), wins / ties / losses, and the score by
thirds of them in time order.

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
    def score(part: list) -> float:
        return sum(1.0 if o > 0 else 0.5 if o == 0 else 0.0 for _, o in part) / max(len(part), 1)
    for key, v in sorted(games.items()):
        v.sort()
        thirds = [v[i * len(v) // 3:(i + 1) * len(v) // 3] for i in range(3)]
        w, t = sum(o > 0 for _, o in v), sum(o == 0 for _, o in v)
        print(f"{key[:45]:45} {len(v):5} {score(v):.3f} ({w}/{t}/{len(v) - w - t})  thirds",
              [round(score(h), 2) for h in thirds])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
