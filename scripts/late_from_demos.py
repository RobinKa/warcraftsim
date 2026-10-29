"""Late-game starts from people's games (fullgame/late.py, from_demo): the mid-game states of demonstrations of people's
replays played back in the real game (fullgame/replays.py; the arrays hold the real game's observations every 0.5 s:
units with hit points, mana and orders, heroes with levels, experience, skills and items, research levels, resources,
the mines' gold, the creeps), written for training on the real Echo Isles and on echofast2.

    python3 scripts/late_from_demos.py runs/fullgame/demos-human-129 --out runs/fullgame/late-states-human-129.jsonl
    python3 scripts/late_from_demos.py --describe runs/fgself-166/late-states.jsonl [--map echofast2]

Each state goes into --out once per map of --maps: as captured on the demonstrations' map, and as each other map's
whose layout it is (late.rescaled: the mines' gold and the sides' gold and lumber scaled; checked here: the maps' files
the same, but for the script's mines' gold and starting resources, the mines' gold at the rules' scale). Skipped: games
out of step with their replay (meta "sync", and the converter's desync/ is not read), minutes in a game's last one
(people leave a lost game), decided ones (late.capture: a side without a town hall, or under a quarter of the other's
material). --describe: what a states file holds, by matchup and minute, and per side its army mix (the top unit's share,
air units), expansions, hall tier, research, heroes and workers (a run's own states against people's).

Results (2026-10-29, the 95 Echo Isles games of runs/fullgame/demos-human-129): see docs/reference/human-replays.md,
"Late-game starts from people's games".
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from scorecard import AIR, WORKERS  # noqa: E402
from warcraftsim.fullgame import late  # noqa: E402

LAYOUT_FILES = ("war3map.w3i", "war3map.w3e", "war3map.wpm", "war3map.doo", "war3mapUnits.doo", "war3map.shd")
EXPANSION_FROM_START, EXPANSION_AT_MINE = 1500.0, 1000.0  # (as trace.Production's expansions)


def script_layout(j: str) -> str:
    """A map script without what data.duelmap.economy_script changes: the mines' gold, the starting resources."""
    j = j.replace("\r\n", "\n")
    j = re.sub(r"(?ms)^function W3S_DuelRules takes nothing returns nothing\n.*?^endfunction\n\n?", "", j)
    j = re.sub(r"\n\s*call W3S_DuelRules\(\s*\)", "", j)
    return re.sub(r"(call SetResourceAmount\(\s*u\s*,\s*)\d+", r"\1N", j)


def check_layout(src: str, dst: str) -> str:
    """Whether map `dst` has `src`'s layout (its terrain, pathing, doodads and trees, pre-placed units, start
    locations, script but for the economy; the mines' gold at the rules' scale): "" if so, else what differs."""
    from warcraftsim.data.duelmap import map_rules
    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.data.mpq import MpqArchive
    if late.rescaled({"map": src, "sides": [], "mines": []}, dst) is None:
        return f"{dst} is not a map of {src}'s layout by name (late.layout)"
    with MpqArchive(stock_map_path(src)) as a, MpqArchive(stock_map_path(dst)) as b:
        for name in LAYOUT_FILES:
            if a.read(name) != b.read(name):
                return f"{name} differs"
        ja, jb = (m.read("war3map.j").decode("latin-1") for m in (a, b))
    if script_layout(ja) != script_layout(jb):
        return "the map scripts differ beyond the economy"
    gold = lambda j: [int(g) for g in re.findall(r"call SetResourceAmount\(\s*u\s*,\s*(\d+)", j)]  # noqa: E731
    scale = map_rules(dst).mine_gold / map_rules(src).mine_gold
    if [round(g * scale) for g in gold(ja)] != gold(jb):
        return f"the mines' gold {gold(jb)} is not {gold(ja)} times {scale}"
    return ""


def side_measures(st: dict, s: int) -> dict:
    """One side of a state: its army (units but workers, harvesters at work and heroes), the top unit's share of it (an
    army of 5 or more), air units, expansions (a town hall 1500 from its start, 1000 from a mine), hall tier, research
    levels, heroes alive (and their mean level), workers, gold and lumber."""
    side, (sx, sy) = st["sides"][s], st["starts"][s]
    army = collections.Counter(u[0] for u in side["units"] if u[5] not in ("s", "g", "l") and u[0] not in WORKERS
                               and u[0] not in late.MINES)
    n = sum(army.values())
    halls = [u for u in side["units"] if u[0] in late.HALLS]
    mines = [(m[1], m[2]) for m in st.get("mines", ())]
    expansions = sum(1 for h in halls if math.hypot(h[1] - sx, h[2] - sy) > EXPANSION_FROM_START
                     and any(math.hypot(h[1] - x, h[2] - y) < EXPANSION_AT_MINE for x, y in mines))
    heroes = [h for h in side["heroes"] if not h[-1]]
    return {"army": n, "top share": max(army.values()) / n if n >= 5 else None,
            "air": sum(v for c, v in army.items() if c in AIR), "air any": float(any(c in AIR for c in army)),
            "expanded": float(expansions > 0), "expansions": expansions,
            "tier": max((late.HALLS[h[0]] for h in halls), default=0),
            "research": sum(side.get("research", {}).values()), "heroes": len(heroes),
            "hero level": float(np.mean([h[5] for h in heroes])) if heroes else None,
            "workers": sum(1 for u in side["units"] if u[0] in WORKERS), "gold": side["gold"], "lumber": side["lumber"],
            "top unit": army.most_common(1)[0][0] if n else ""}


def describe(states: list[dict], title: str) -> None:
    """A states file's matchups by minute, and its sides' mean measures (side_measures)."""
    print(f"{title}: {len(states)} states")
    pairs = collections.Counter()
    for st in states:
        pairs[("-".join(sorted(st["races"])), int(st["minute"] + 0.5))] += 1
    minutes = sorted({m for _, m in pairs})
    names = sorted({p for p, _ in pairs})
    print(f"  {'matchup':18}" + "".join(f"{m:>5}" for m in minutes) + "  all")
    for p in names:
        print(f"  {p:18}" + "".join(f"{pairs[(p, m)]:5d}" for m in minutes) + f"{sum(pairs[(p, m)] for m in minutes):5d}")
    rows = [(st, side_measures(st, s)) for st in states for s in (0, 1)]
    for label, sel in [("all", rows)] + [(f"minute {m}", [r for r in rows if int(r[0]["minute"] + 0.5) == m])
                                          for m in minutes]:
        if not sel:
            continue
        mean = lambda k: np.mean([x[k] for _, x in sel if x[k] is not None] or [np.nan])  # noqa: E731
        tiers = collections.Counter(x["tier"] for _, x in sel)
        print(f"  {label:10} sides {len(sel):4d}: army {mean('army'):5.1f}, top unit {mean('top share'):.0%}, air "
              f"{mean('air'):4.1f} (any {mean('air any'):.0%}), expanded {mean('expanded'):.0%}, tier 1/2/3 "
              f"{tiers[1] / len(sel):.0%}/{tiers[2] / len(sel):.0%}/{tiers[3] / len(sel):.0%}, research "
              f"{mean('research'):4.1f}, heroes {mean('heroes'):.1f} (level {mean('hero level'):.1f}), workers "
              f"{mean('workers'):4.1f}, gold {mean('gold'):5.0f}, lumber {mean('lumber'):5.0f}")
    tops = collections.Counter(x["top unit"] for _, x in rows if x["top unit"])
    print("  top units: " + ", ".join(f"{c} {n}" for c, n in tops.most_common(12)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("demos", nargs="*", type=Path, help="directories of demonstrations (game*.npz, collect.py's)")
    ap.add_argument("--out", type=Path, help="the states file (JSON lines; written anew)")
    ap.add_argument("--minutes", default="8,10,12,14,16", help="the game minutes captured")
    ap.add_argument("--maps", default="(2)EchoIsles,echofast2", help="the maps each state is written for (the "
                    "demonstrations' own and maps of its layout under other rules)")
    ap.add_argument("--before-end", type=float, default=1.0, help="no state in a game's last this many minutes")
    ap.add_argument("--describe", type=Path, help="only describe this states file")
    ap.add_argument("--map", default="", help="--describe: only this map's states")
    a = ap.parse_args(argv)
    if a.describe:
        states = [json.loads(line) for line in a.describe.open() if line.strip()]
        describe([s for s in states if not a.map or s.get("map") == a.map], str(a.describe))
        return 0
    if not a.demos or not a.out:
        ap.error("demonstration directories and --out (or --describe)")
    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame.features import map_geometry
    from warcraftsim.fullgame.trace import unit_values
    minutes = [float(m) for m in a.minutes.split(",")]
    maps = [m for m in a.maps.split(",") if m]
    values = unit_values()
    found, skipped = [], collections.Counter()
    geometry: dict[str, tuple] = {}
    for d in a.demos:
        for f in sorted(d.glob("game*.npz")):  # (not desync/: games out of step)
            with np.load(f) as z:
                meta = json.loads(str(z["meta"]))
                if meta["map"] not in geometry:
                    geometry[meta["map"]] = map_geometry(stock_map_path(meta["map"]))[0]
                got = late.from_demo({k: z[k] for k in ("units", "heroes", "players", "events")}, meta, minutes,
                                     geometry[meta["map"]], values, a.before_end)
            reached = sum(1 for m in minutes if m <= meta["steps"] * meta.get("step_seconds", 0.5) / 60.0 - a.before_end)
            skipped["minutes past the game's end"] += len(minutes) - reached
            skipped["decided, or out of step"] += reached - len(got)
            found += got
    for src in sorted({st["map"] for st in found}):
        for m in maps:
            if m != src:
                why = check_layout(src, m)
                if why:
                    raise SystemExit(f"{m} has not {src}'s layout: {why}")
                print(f"{m}: {src}'s layout (map files {', '.join(LAYOUT_FILES)} and the script the same; the mines' "
                      f"gold, the sides' gold and lumber scaled: late.rescaled)", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w") as fh:
        for st in found:
            for m in maps:
                o = late.rescaled(st, m)
                if o is not None:
                    fh.write(json.dumps(o, separators=(",", ":")) + "\n")
                    n += 1
    print(f"{len(found)} states ({n} lines: {', '.join(maps)}) -> {a.out}; not captured: {dict(skipped)}")
    describe(found, "people's states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
