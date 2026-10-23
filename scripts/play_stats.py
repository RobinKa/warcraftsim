"""A run's real-game evaluation games: per policy (eval_key) the tier reached, expansions, creeps, tavern heroes, items
picked up off the ground. python scripts/play_stats.py fgself-134"""
import json, sys
from collections import defaultdict
run = sys.argv[1]
rows = []
for f in ("episodes.jsonl", "episodes-remote.jsonl"):
    try:
        rows += [json.loads(l) for l in open(f"runs/{run}/{f}")]
    except FileNotFoundError:
        pass
T2 = {"hkee", "ostr", "unp1", "etoa"}; T3 = {"hcas", "ofrt", "unp2", "etoe"}
agg = defaultdict(lambda: defaultdict(float))
for r in rows:
    if not r.get("eval") or str(r.get("eval_key", "")).startswith("vs "):
        continue
    p = r.get("prod") or {}
    b = set((p.get("built") or {}).keys())
    a = agg[r.get("eval_key")]
    a["n"] += 1
    a["won"] += r.get("outcome") == "win" or r.get("outcome") == 1 or r.get("outcome") == "VICTORY"
    a["t2"] += bool(b & (T2 | T3)); a["t3"] += bool(b & T3)
    a["exp"] += (p.get("expansions") or 0) > 0; a["home"] += (p.get("halls_home") or 0) > 0
    a["creeps"] += p.get("creeps") or 0
    a["tavern"] += any(k.startswith("N") for k in (p.get("trained") or {}))
    a["picked"] += r.get("picked_up", 0)
for k, a in agg.items():
    n = a["n"]
    print(f"{str(k)[-30:]:30s} n={int(n):4d} tier2 {a['t2']/n:.0%} tier3 {a['t3']/n:.0%} expanded {a['exp']/n:.0%} "
          f"hall at home {a['home']/n:.0%} creeps/game {a['creeps']/n:.1f} tavern hero {a['tavern']/n:.0%} items picked/game {a['picked']/n:.2f}")
