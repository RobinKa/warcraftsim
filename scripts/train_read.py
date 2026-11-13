"""A fast read of runs against the insane AI from their own training games (2026-11-13): the launches of --real-share
("script:ai-insane (real)": the real game, the AI untaxed) without a strategy target or advice are evaluation games in
all but name, and a run plays ~500 of them an hour where the laptop's evaluation gives ~30 a policy. The policy moves
during the window (a window's average), it samples as in training, and the learner's races are weighted
(--learner-races), so each race is averaged alike. Compare runs of the same recipe with each other (an experiment
against an unchanged continuation), not with the evaluation games' figures, which read a few points apart.

    python scripts/train_read.py fgself-214 fgself-201 --skip-updates 120
    python scripts/train_read.py fgself-214:1794566515 fgself-201:1794572543     # RUN:SINCE (unix time)
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

WORKERS = {"hpea", "opeo", "uaco", "ewsp"}


def rows(run_dir: Path, since: float, until: float, opponent: str) -> list[dict]:
    out = []
    for name in ("episodes.jsonl", "episodes-remote.jsonl"):
        path = run_dir / name
        if not path.exists():
            continue
        for line in path.open(errors="replace"):
            if opponent not in line or '"eval": true' in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not since <= r.get("time", 0.0) < until or r.get("z") or r.get("advise") or r.get("eval"):
                continue
            out.append(r)
    return out


def first_update_time(run_dir: Path, skip: int) -> float:
    """The time of the run's update number `skip` (train.jsonl's rows, counted from its newest launch's first row)."""
    if skip <= 0 or not (run_dir / "train.jsonl").exists():
        return 0.0
    times = []
    for line in (run_dir / "train.jsonl").open(errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if times and d.get("uptime", 1e9) < 60 and d["time"] - times[-1] > 30:  # (a new launch: count from it)
            times = []
        times.append(d["time"])
    return times[skip] if len(times) > skip else float("inf")


def measure(rs: list[dict]) -> dict:
    """Per race: (games, wins, wins past 18 minutes of the games that long, expansions); race-averaged with errors."""
    by = collections.defaultdict(list)
    for r in rs:
        by[r["race"]].append(r)
    out = {"n": len(rs), "races": {}}
    keys = {"win": lambda r: 1.0 if r["outcome"] > 0 else 0.5 if r["outcome"] == 0 else 0.0,
            "win past 18 min": lambda r: (1.0 if r["outcome"] > 0 else 0.5 if r["outcome"] == 0 else 0.0)
            if r.get("game_time", 0) > 18 * 60 else None,
            "expanded": lambda r: 1.0 if ((r.get("prod") or {}).get("expansions") or 0) > 0 else 0.0,
            "minutes": lambda r: r.get("game_time", 0) / 60.0}
    for k, f in keys.items():
        means, var = [], 0.0
        for race, g in by.items():
            xs = [x for x in (f(r) for r in g) if x is not None]
            if len(xs) < 5:
                continue
            m = sum(xs) / len(xs)
            v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
            means.append(m)
            var += v / len(xs)
            out["races"].setdefault(race, {})[k] = (m, len(xs))
        if means:
            out[k] = (sum(means) / len(means), math.sqrt(var) / len(means))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="RUN, RUN:SINCE or RUN:SINCE:UNTIL (unix times); the first is compared with "
                    "each other")
    ap.add_argument("--until", type=float, default=float("inf"))
    ap.add_argument("--hours", type=float, default=0.0, help="only the last N hours of each run's rows (0: all)")
    ap.add_argument("--skip-updates", type=int, default=0, help="leave out the rows before the run's newest launch's "
                    "update N (a fork's value warm-up and label ramp: ~110 updates)")
    ap.add_argument("--opponent", default="ai-insane (real)")
    ap.add_argument("--root", type=Path, default=Path("runs"))
    a = ap.parse_args(argv)
    results = []
    for spec in a.runs:
        name, since, until = (spec.split(":") + ["", ""])[:3]
        d = a.root / name
        t0 = max(float(since or 0.0), first_update_time(d, a.skip_updates))
        rs = rows(d, t0, min(float(until or "inf"), a.until), a.opponent)
        if a.hours > 0 and rs:
            last = max(r["time"] for r in rs)
            rs = [r for r in rs if r["time"] >= last - a.hours * 3600]
        m = measure(rs)
        span = (max(r["time"] for r in rs) - min(r["time"] for r in rs)) / 3600 if rs else 0.0
        results.append((name, m))
        line = f"{name:12} {m['n']:5} games over {span:4.1f} h"
        for k in ("win", "win past 18 min", "expanded", "minutes"):
            if k in m:
                line += f"   {k} {m[k][0]:.3f} ±{1.96 * m[k][1]:.3f}"
        print(line)
        print("             " + "  ".join(f"{r[:2]} {v['win'][0]:.2f} ({v['win'][1]})" for r, v in sorted(m["races"].items())
                                      if "win" in v))
    base_name, base = results[0]
    for name, m in results[1:]:
        for k in ("win", "win past 18 min", "expanded"):
            if k in base and k in m:
                d = base[k][0] - m[k][0]
                se = math.hypot(base[k][1], m[k][1])
                flag = "*" if abs(d) > 1.96 * se else " "
                print(f"{base_name} minus {name}: {k} {d:+.3f}{flag} [{d - 1.96 * se:+.3f}, {d + 1.96 * se:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
