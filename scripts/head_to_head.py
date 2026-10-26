"""A run's evaluation games by key (its current policy, with a target, the listed checkpoints, head to head with each):
games, the score (a win 1, a tie half: ties had counted as losses; audit 4), wins / ties / losses, and the score by
thirds of them in time order.

    python scripts/head_to_head.py fgself-130 [--since <unix time>]
    python scripts/head_to_head.py fgself-155 --vs fgself-153/0000000038318278   # the current policy against insane
        # beside that checkpoint's games in the same hours, by its race: a fork judged against its start played in the
        # same windows (the raw thirds had read a dip shared by every policy as the fork's decline; audit 14)
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
    ap.add_argument("--vs", default="", help="a listed checkpoint (its key's part, e.g. fgself-153/0000000038318278): the "
                    "current policy's games against the AI beside its, hour by hour and by the policy's race")
    a = ap.parse_args(argv)
    games = collections.defaultdict(list)
    ai = collections.defaultdict(list)  # (key -> (time, race, outcome): the games against the AI, --vs)
    for name in ("episodes.jsonl", "episodes-remote.jsonl"):
        path = a.runs / a.run / name
        if not path.exists():
            continue
        for line in path.open():
            row = json.loads(line)
            if row.get("eval") and row.get("time", 0) >= a.since:
                games[row.get("eval_key", "current")].append((row["time"], row["outcome"]))
                if "ai-" in str(row.get("opponent", "")):
                    ai[row.get("eval_key", "current")].append((row["time"], row.get("race", "?"), row["outcome"]))
    def score(part: list) -> float:
        return sum(1.0 if o > 0 else 0.5 if o == 0 else 0.0 for _, o in part) / max(len(part), 1)
    for key, v in sorted(games.items()):
        v.sort()
        thirds = [v[i * len(v) // 3:(i + 1) * len(v) // 3] for i in range(3)]
        w, t = sum(o > 0 for _, o in v), sum(o == 0 for _, o in v)
        print(f"{key[:45]:45} {len(v):5} {score(v):.3f} ({w}/{t}/{len(v) - w - t})  thirds",
              [round(score(h), 2) for h in thirds])
    if a.vs:
        other = next((k for k in ai if all(x in k for x in a.vs.split("/")) and not k.startswith("vs ")), None)
        if other is None:
            print(f"--vs {a.vs}: no games against the AI under that key")
            return 1
        cur, ref = ai["current"], ai[other]
        t0 = min(t for t, _, _ in cur + ref)
        sc = lambda part: sum(1.0 if o > 0 else 0.5 if o == 0 else 0.0 for o in part) / max(len(part), 1)  # noqa: E731
        races = sorted({r for _, r, _ in cur + ref})
        print(f"\nagainst the AI, current vs {other[:40]} (score, games), by hour and by the policy's race:")
        hours = sorted({int((t - t0) // 3600) for t, _, _ in cur + ref})
        for h in hours + [None]:
            cell = []
            for r in races + [None]:
                c = [o for t, rr, o in cur if (h is None or int((t - t0) // 3600) == h) and (r is None or rr == r)]
                f = [o for t, rr, o in ref if (h is None or int((t - t0) // 3600) == h) and (r is None or rr == r)]
                cell.append(f"{(r or 'all')[:5]} {sc(c):.2f}/{sc(f):.2f} ({len(c)}/{len(f)})")
            print(f"  {'all' if h is None else f'h{h}':4}", "  ".join(cell))
        # the races weighed alike (a window with many games of one race reads that race's level, not the policy's)
        d = [sc([o for _, rr, o in cur if rr == r]) - sc([o for _, rr, o in ref if rr == r]) for r in races]
        print(f"  races weighed alike: current minus {other.split('/')[-1][:20]} = {sum(d) / max(len(d), 1):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
