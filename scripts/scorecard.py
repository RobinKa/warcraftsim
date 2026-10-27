"""A run's behaviour scorecard (docs/proposals/five-changes.md, 3): its current policy's evaluation games against the AI
beside a listed checkpoint's (its start) in the same hours, each measure averaged per race and the races weighed alike,
with a 95% interval on the difference (a flag where it excludes 0). The measures are what the audits found by watching
games: expansions, workers, income, upgrades, creeping, one-unit armies, anti-air against air, refused orders, the late
game.

    python scripts/scorecard.py fgself-158 --vs fgself-155/0000000036007395 [--since <unix time>]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

WORKERS = {"hpea", "opeo", "uaco", "ewsp"}
AIR = {"hgry", "hdhw", "hgyr", "owyv", "otbr", "ugar", "ufro", "ehip", "ehpr", "efdr", "edtm"}
ANTI_AIR = {"hrif", "hgry", "hdhw", "hgyr", "ohun", "otbk", "owyv", "otbr", "ucry", "ugar", "earc", "ehip", "ehpr", "edry"}
UPGRADES = {"Rhme", "Rhar", "Rhra", "Rhla", "Rome", "Roar", "Rora", "Rosp", "Rume", "Ruar", "Rura", "Rucr", "Resm", "Resw",
            "Rema", "Rerh"}  # (weapons and armour: each race's melee, ranged and armour lines)


def is_hero(code: str) -> bool:
    return code[:1].isupper()


def measures(r: dict) -> dict[str, float]:
    """One game's measures (None where it does not apply)."""
    p, o = r.get("prod") or {}, r.get("opp_prod") or {}
    minutes = max(float(r.get("game_time", 0.0)) / 60.0, 1.0)
    trained = p.get("trained") or {}
    army = {c: n for c, n in trained.items() if c not in WORKERS and not is_hero(c)}
    total = sum(army.values())
    enemy_air = sum(n for c, n in (o.get("trained") or {}).items() if c in AIR)
    win = 1.0 if r["outcome"] > 0 else 0.5 if r["outcome"] == 0 else 0.0
    out = {
        "win": win,
        "win past 18 min": win if minutes > 18 else None,
        "expanded": 1.0 if (p.get("expansions") or 0) > 0 else 0.0,
        "workers trained": float(sum(n for c, n in trained.items() if c in WORKERS)),
        "gold / min": float(p.get("gold") or 0) / minutes,
        "lumber / min": float(p.get("lumber") or 0) / minutes,
        "weapon/armour levels": float(sum(n for c, n in (p.get("research") or {}).items() if c in UPGRADES)),
        "creeps killed": float(p.get("creeps") or 0),
        "top unit's share": max(army.values()) / total if total >= 10 else None,
        "anti-air share vs 4+ air": (sum(n for c, n in army.items() if c in ANTI_AIR) / max(total, 1)) if enemy_air >= 4 else None,
        "win vs 4+ air": win if enemy_air >= 4 else None,
        "heroes": float(p.get("heroes") or 0),
        "refused share": float(r.get("refused_share") or 0.0),
        "orders to loaded units": float(r.get("loaded_orders") or 0),
    }
    return out


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / max(n - 1, 1)
    return m, math.sqrt(v / n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--vs", required=True, help="a listed checkpoint's key part, e.g. fgself-155/0000000036007395")
    ap.add_argument("--since", type=float, default=0.0)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    a = ap.parse_args(argv)
    rows: dict[str, list[dict]] = collections.defaultdict(list)
    for name in ("episodes.jsonl", "episodes-remote.jsonl"):
        path = a.runs / a.run / name
        if not path.exists():
            continue
        for line in path.open():
            if '"eval": true' not in line:
                continue
            r = json.loads(line)
            if r.get("time", 0) >= a.since and "ai-" in str(r.get("opponent", "")) and r.get("prod"):
                rows[str(r.get("eval_key") or "current")].append(r)
    ref = next((k for k in rows if all(x in k for x in a.vs.split("/")) and not k.startswith("vs ")), None)
    if ref is None or not rows.get("current"):
        print(f"no games against the AI for current and {a.vs}")
        return 1
    cur, old = rows["current"], rows[ref]
    t0 = min(r["time"] for r in cur + old)
    hours = {int((r["time"] - t0) // 3600) for r in cur} & {int((r["time"] - t0) // 3600) for r in old}
    cur = [r for r in cur if int((r["time"] - t0) // 3600) in hours]  # (the same windows: hours both played in)
    old = [r for r in old if int((r["time"] - t0) // 3600) in hours]
    races = sorted({r["race"] for r in cur} & {r["race"] for r in old})
    names = list(measures(cur[0]).keys())
    print(f"{a.run}: current ({len(cur)} games) vs {ref.split('/')[-1][:24]} ({len(old)}) against the AI, hours {sorted(hours)}, "
          f"races weighed alike ({', '.join(races)}); * = the 95% interval of the difference excludes 0")
    print(f"{'measure':26} {'current':>9} {'start':>9} {'diff':>9}  95% interval     by race (current/start)")
    for m in names:
        per, dm, var, cs, ss = [], 0.0, 0.0, 0.0, 0.0
        for race in races:
            c = [x for x in (measures(r)[m] for r in cur if r["race"] == race) if x is not None]
            s = [x for x in (measures(r)[m] for r in old if r["race"] == race) if x is not None]
            if len(c) < 3 or len(s) < 3:
                continue
            (mc, ec), (ms, es) = mean_se(c), mean_se(s)
            per.append((race, mc, ms))
            dm, var, cs, ss = dm + mc - ms, var + ec ** 2 + es ** 2, cs + mc, ss + ms
        if not per:
            continue
        k = len(per)
        d, se = dm / k, math.sqrt(var) / k
        lo, hi = d - 1.96 * se, d + 1.96 * se
        flag = "*" if lo > 0 or hi < 0 else " "
        fmt = (lambda v: f"{v:.2f}") if max(abs(cs / k), abs(ss / k)) < 10 else (lambda v: f"{v:.0f}")
        races_txt = " ".join(f"{r[:2]} {fmt(c)}/{fmt(s)}" for r, c, s in per)
        print(f"{m:26} {fmt(cs / k):>9} {fmt(ss / k):>9} {fmt(d):>9}{flag} [{fmt(lo)}, {fmt(hi)}]  {races_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
