"""Summarize a sweep (or any runs): how fast each run's win rate crossed thresholds.

    python scripts/sweep_summary.py f2-sweep3            # runs f2-sweep3-1, -2, ...
    python scripts/sweep_summary.py f2-sweep1-4 f2-sweep3-1
    python scripts/sweep_summary.py abilsw --thresholds 0.3,0.4,0.45,0.5 --at 0.5,1,1.5

Win rate: the trainer's per-epoch env/win_rate, smoothed over a window of 1/15 of the epochs.
Times are wall time since the run's first epoch (throughput differences included).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

RUNS = Path(os.environ.get("WARCRAFTSIM_RUNS", Path(__file__).resolve().parent.parent / "runs"))
THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.98)


def summarize(run_dir: Path, thresholds=THRESHOLDS, at=()) -> str:
    info = json.loads((run_dir / "run.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines() if line.strip()]
    wins = [(r["agent_steps"], r["env/win_rate"], r["time"]) for r in rows if r.get("env/n")]
    if not wins:
        return f"{run_dir.name}: no episodes yet"
    k = max(1, len(wins) // 15)
    roll = []
    for j in range(len(wins)):
        window = wins[max(0, j - k + 1):j + 1]
        roll.append((wins[j][0], sum(w for _, w, _ in window) / len(window), wins[j][2]))
    t0 = rows[0]["time"]

    def first(th: float) -> str:
        return next((f"{s / 1e6:.2f}M {(t - t0) / 60:4.1f}m" for s, w, t in roll if w >= th), "      -     ")

    def win_at(steps: float) -> str:  # the smoothed win rate at a step budget (millions)
        hit = [w for s, w, _ in roll if s <= steps * 1e6]
        return f"{hit[-1]:.2f}" if hit and roll[-1][0] >= steps * 1e6 * 0.98 else "  - "

    label = (info.get("sweep") or {}).get("options", "") or "(base)"
    return (f"{run_dir.name:16s} {label[:34]:34s} {info['status'][:8]:8s} final {roll[-1][1]:.2f} "
            f"best {max(w for _, w, _ in roll):.2f} " + " ".join(f"{th:.2f}: {first(th)}" for th in thresholds)
            + "".join(f"  @{a:g}M {win_at(a)}" for a in at)
            + f"  SPS {statistics.median(r['SPS'] for r in rows):.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run names or sweep prefixes (default: all runs)")
    ap.add_argument("--thresholds", default=",".join(map(str, THRESHOLDS)), help="win rates to report the time to")
    ap.add_argument("--at", default="", help="step budgets in millions: the smoothed win rate there")
    args = ap.parse_args()
    thresholds = tuple(float(t) for t in args.thresholds.split(",") if t)
    at = tuple(float(a) for a in args.at.split(",") if a)
    names = args.runs or sorted(p.name for p in RUNS.iterdir() if (p / "run.json").exists())
    dirs = []
    for n in names:
        if (RUNS / n / "run.json").exists():
            dirs.append(RUNS / n)
        else:  # a sweep group
            dirs += sorted((p for p in RUNS.glob(f"{n}-*") if (p / "run.json").exists()),
                           key=lambda p: int(p.name.rsplit("-", 1)[1]) if p.name.rsplit("-", 1)[1].isdigit() else 0)
    print(f"{'run':16s} {'options':34s} {'status':8s} final/best  steps and minutes to win rate " +
          " ".join(f"{th:.2f}" for th in thresholds))
    for d in dirs:
        if (d / "train.jsonl").exists():
            print(summarize(d, thresholds, at))


if __name__ == "__main__":
    main()
