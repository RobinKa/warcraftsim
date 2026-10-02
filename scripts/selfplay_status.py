"""A whole-game self-play run at a glance: is it alive, throughput, the real game's win rates (the last
30 minutes and the whole run, by race, by the AI's race, mirror or mixed), the taxes between races,
the curriculum levels still above 0.1, heroes and altars per game.

    python3 scripts/selfplay_status.py fgself-10
"""

from __future__ import annotations

import collections
import json
import shutil
import statistics as s
import subprocess
import sys
import time
from pathlib import Path

RACES = ("human", "orc", "undead", "nightelf")
ALTARS = ("halt", "oalt", "uaod", "eate")


def main(name: str, window: float = 1800.0) -> None:
    run = Path(__file__).resolve().parent.parent / "runs" / name
    alive = subprocess.run(["pgrep", "-f", f"selfplay --name {name}( |$)"], capture_output=True, text=True).stdout.split()
    log = run.parent / f"{name}.log"
    tracebacks = log.read_text(errors="replace").count("Traceback") if log.exists() else 0
    rows = [json.loads(line) for line in open(run / "train.jsonl")]
    eps = [json.loads(line) for line in open(run / "episodes.jsonl")]
    now, r = time.time(), rows[-1]
    recent = [x for x in rows if x["time"] > now - window] or rows[-5:]
    print(f"alive: {bool(alive)}  tracebacks: {tracebacks}  steps {r['agent_steps'] / 1e6:.2f}M  "
          f"last update {now - r['time']:.0f}s ago  disk free {shutil.disk_usage('/').free / 1e9:.0f} GB")
    print({k: round(s.mean(x[k] for x in recent if k in x), 4) for k in
           ("SPS", "perf/train", "perf/rollout", "staleness", "loss/kl", "loss/ref_kl", "loss/value", "loss/bc") if k in r})

    def wins(v: list[float]) -> str:
        return f"{100 * sum(o > 0.5 for o in v) / len(v):3.0f}% ({len(v)})" if v else "-"
    for label, lo in (("last %d min" % (window // 60), now - window), ("since start", 0)):
        by = collections.defaultdict(list)
        for e in eps:
            if e["time"] <= lo or "(real)" not in e["opponent"]:
                continue
            for k in ("all", "me " + e["race"], "ai " + e["opponent_race"],
                      "mirror" if e["race"] == e["opponent_race"] else "mixed", e["opponent"].split("-")[1].split()[0]):
                by[k].append(e["outcome"])
        keys = ["all", "mirror", "mixed", "easy", "normal"] + [f"me {x}" for x in RACES] + [f"ai {x}" for x in RACES]
        print(label + ":", "  ".join(f"{k} {wins(by[k])}" for k in keys))
    print("taxes between races:", {k[8:-4]: v for k, v in r.items() if k.startswith("balance/")})
    levels = {k[11:-6]: round(v, 2) for k, v in r.items() if k.startswith("curriculum/") and k.endswith(" level") and v >= 0.1}
    print("curriculum levels >= 0.1:", dict(sorted(levels.items(), key=lambda kv: -kv[1])))
    prod = collections.defaultdict(list)
    for e in eps:
        if e["time"] > now - window:
            p = e.get("prod") or {}
            prod[e["race"]].append((p.get("heroes", 0), sum(p.get("built", {}).get(k, 0) for k in ALTARS)))
    print(f"heroes / altars per game ({window // 60:.0f} min):",
          {k: (round(s.mean(x[0] for x in v), 2), round(s.mean(x[1] for x in v), 2)) for k, v in sorted(prod.items())})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fgself-10")
