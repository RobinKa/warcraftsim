"""Statistics of a directory of Warcraft III replays (.w3g, any version; warcraftsim.data.w3g): one row
per replay in a jsonl (version, build, map, players, matchup, length, winner, each player's build order,
APM and strategy statistics: w3g.strategy), and a summary: versions and builds, maps, matchups, and how
many are 1v1 on Echo Isles at build 6060 (the ones our client can play back: fullgame/replays.py).

    python scripts/replay_stats.py runs/fullgame/human-replays-129 [--out stats.jsonl] [--jobs 4]
    python scripts/replay_stats.py --ids     # (re)writes warcraftsim/data/w3ids.json from the 1.29 game data
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warcraftsim.data import w3g  # noqa: E402

IDS = Path(w3g.__file__).with_name("w3ids.json")


def map_name(path: str) -> str:
    """'Maps\\FrozenThrone\\(2)EchoIsles.w3x' -> 'EchoIsles' (folder, player count, extension and a
    ladder's version suffix off: 'w3c_EchoIsles_v1.2' -> 'EchoIsles')."""
    name = re.split(r"[\\/]", path)[-1]
    name = re.sub(r"\.w3[mx]$", "", name, flags=re.I)
    name = re.sub(r"^\(\d+\)", "", name)
    name = re.sub(r"^(\d+_)?(w3c|w3arena|gcl|nwc|wgl)_(\d+_)*", "", name, flags=re.I)
    name = re.sub(r"[-_\s]?v?\d+(\.\d+)*[a-z]?$", "", name, flags=re.I)
    return name.replace("_", " ").strip() or path


def row(path: Path) -> dict:
    """A replay's row: {"file", "error"} when it does not parse."""
    try:
        rep = w3g.read_replay(path)
    except (w3g.ReplayError, OSError, ValueError, IndexError) as e:
        return {"file": path.name, "error": f"{type(e).__name__}: {e}"}
    h = rep.header
    s = w3g.summarize(rep)
    players = [{"id": p.id, "name": p.name, "slot": p.slot, "team": p.team, "race": p.race, "random": p.random,
                "computer": p.computer, "ai": p.ai, "handicap": p.handicap} for p in rep.playing]
    teams = Counter(p.team for p in rep.playing)
    return {"file": path.name, "container": h.container, "patch": h.patch, "version": h.version, "build": h.build,
            "expansion": h.expansion, "map": rep.map_path, "map_name": map_name(rep.map_path),
            "map_checksum": f"{rep.map_checksum:08x}" if rep.map_checksum is not None else None,
            "map_sha1": rep.map_sha1, "game_name": rep.game_name, "speed": rep.settings.get("speed"),
            "type": "v".join(str(teams[t]) for t in sorted(teams)), "matchup": matchup(rep, s),
            "observers": sum(p.observer for p in rep.players.values()), "players": players,
            "length_ms": s["length_ms"], "winner_team": s["winner_team"], "winner_how": s["winner_how"],
            "leaves": [(lv.time, lv.player, lv.reason, lv.result) for lv in rep.leaves],
            "actions": len(rep.actions), "unknown_actions": rep.unknown_actions, "build_orders": s["players"],
            "strategy": {str(p["player"]): w3g.strategy(p) for p in s["players"]}}


def matchup(rep: w3g.Replay, summary: dict) -> str:
    """"human vs orc" (teams sorted; a random player's race as played, if its actions tell)."""
    played = {p["player"]: p["race_played"] for p in summary["players"]}
    teams: dict[int, list[str]] = {}
    for p in rep.playing:
        teams.setdefault(p.team, []).append(played.get(p.id) or p.race or "?")
    return " vs ".join(sorted("+".join(sorted(t)) for t in teams.values()))


def is_echo_1v1_6060(r: dict) -> bool:
    return (r.get("build") == 6060 and r.get("version") == 29 and r.get("map_name") == "EchoIsles" and r.get("type") == "1v1"
            and not any(p["computer"] for p in r.get("players", ())))


def write_ids(path: Path = IDS) -> int:
    """The fourcc -> (kind, name) table of w3g.object_kinds from the game's 1.29 data: units, heroes and
    buildings (UnitData/UnitBalance), researches (UpgradeData), items (ItemData), hero skills (AbilityData)."""
    from warcraftsim.data.mpq import GameArchives
    from warcraftsim.data.objects import parse_slk
    names: dict[str, str] = {}
    with GameArchives() as g:
        for fname in sorted(g.names("Units\\*Strings.txt"), key=str.lower):
            current = None
            for line in g.read(fname).decode("utf-8", "replace").splitlines():
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    current = line[1:-1]
                elif current and line.startswith("Name=") and current not in names:
                    names[current] = line[5:].split(",")[0].strip().strip('"')
        table = lambda f: parse_slk(g.read(f"Units\\{f}").decode("latin-1"))  # noqa: E731
        units, balance = table("UnitData.slk"), {r.get("unitBalanceID"): r for r in table("UnitBalance.slk")}
        upgrades, items, abilities = table("UpgradeData.slk"), table("ItemData.slk"), table("AbilityData.slk")
    out: dict[str, list[str]] = {}
    for r in abilities:
        if r.get("hero") == "1" and len(r.get("alias", "")) == 4:
            out[r["alias"]] = ["skill", names.get(r["alias"], "")]
    for r in items:
        if len(r.get("itemID", "")) == 4:
            out[r["itemID"]] = ["item", names.get(r["itemID"], "")]
    for r in upgrades:
        if len(r.get("upgradeid", "")) == 4:
            out[r["upgradeid"]] = ["upgrade", names.get(r["upgradeid"], "")]
    for r in units:  # (last: a unit's code wins over an item's)
        uid = r.get("unitID", "")
        if len(uid) != 4:
            continue
        kind = "building" if balance.get(uid, {}).get("isbldg") == "1" else "hero" if uid[0].isupper() else "unit"
        out[uid] = [kind, names.get(uid, "")]
    path.write_text(json.dumps(dict(sorted(out.items())), separators=(",", ":"), ensure_ascii=False) + "\n")
    return len(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replays", nargs="?", type=Path, help="a directory of .w3g files (or one file)")
    ap.add_argument("--out", type=Path, help="the rows (default: <replays>/stats.jsonl)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--ids", action="store_true", help="write the object table (w3ids.json) from the game data")
    args = ap.parse_args(argv)
    if args.ids:
        print(f"{IDS}: {write_ids()} types")
        return 0
    if args.replays is None:
        ap.error("a directory of replays")
    files = sorted(args.replays.glob("*.w3g")) if args.replays.is_dir() else [args.replays]
    out = args.out or (args.replays / "stats.jsonl" if args.replays.is_dir() else Path("stats.jsonl"))
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex, open(out, "w") as f:
        for r in ex.map(row, files, chunksize=8):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
    good = [r for r in rows if "error" not in r]
    print(f"{len(rows)} replays, {len(good)} parsed ({len(rows) - len(good)} not: "
          f"{Counter(r['error'].split(':')[0] for r in rows if 'error' in r).most_common(3)}) -> {out}")
    top = lambda c, n=12: ", ".join(f"{k} {v}" for k, v in c.most_common(n))  # noqa: E731
    print("versions:", top(Counter(f"{r['patch']} build {r['build']}" for r in good)))
    print("containers:", top(Counter(r["container"] for r in good)))
    print("types:", top(Counter(r["type"] for r in good)))
    print("maps:", top(Counter(r["map_name"] for r in good), 20))
    ones = [r for r in good if r["type"] == "1v1"]
    print("1v1 matchups:", top(Counter(r["matchup"] for r in ones), 10))
    decided = sum(r["winner_team"] is not None for r in ones)
    print(f"1v1 winner told: {decided} of {len(ones)} ({top(Counter(r['winner_how'] for r in ones), 4)})")
    if ones:
        mins = sorted(r["length_ms"] / 60000 for r in ones)
        print(f"1v1 length: median {mins[len(mins) // 2]:.1f} min, 10-90% {mins[len(mins) // 10]:.1f}-"
              f"{mins[len(mins) * 9 // 10]:.1f}")
        apm = sorted(p["apm"] for r in ones for p in r["build_orders"])
        if apm:
            print(f"1v1 APM: median {apm[len(apm) // 2]:.0f}, 10-90% {apm[len(apm) // 10]:.0f}-{apm[len(apm) * 9 // 10]:.0f}")
    echo = [r for r in good if is_echo_1v1_6060(r)]
    print(f"1v1 on Echo Isles at 1.29 build 6060 (people only): {len(echo)}; matchups: "
          f"{top(Counter(r['matchup'] for r in echo), 10)}; map checksums: {top(Counter(r['map_checksum'] for r in echo), 4)}")
    print(f"unknown actions: {sum(r['unknown_actions'] for r in good)} in {sum(r['unknown_actions'] > 0 for r in good)} replays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
