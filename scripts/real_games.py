"""A whole-game self-play run's real games (against the built-in AI without the curriculum's tax) since a
time and in the last 30 minutes: wins by difficulty and by the learner's race, and per difficulty the
tempo against the AI (food at one minute, heroes, tier 2, workers a game minute, kills per unit lost);
then the unadvised curriculum games and the ties against itself and past snapshots.

    python3 scripts/real_games.py fgself-12 --since "2026-10-03 18:36" [--recent 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from throughput import RUNS, when  # noqa: E402

TIER2 = ("hkee", "ostr", "unp1", "etoa")  # keep, stronghold, halls of the dead, tree of ages
# the built-in AI against itself on duelfast (3,000 games, both sides normal or both insane): how often a race
# wins a matchup. duelfast is far from balanced (mixed matchups: night elf ~78%, orc ~68%, undead ~35%, human ~20%).
AI_GAMES = Path(__file__).resolve().parents[1] / "runs" / "fullgame" / "demos-fast-1" / "games.jsonl"
# Echo Isles under the game's rules (1,500 games, normal and insane, each side's drawn apart; ~620 decided
# between equals): its races are far apart too, and not as on duelfast: the AI as human wins 14% of its mixed
# insane games there, as undead 88%
ECHO_AI_GAMES = Path(__file__).resolve().parents[1] / "runs" / "fullgame" / "demos-echo-1" / "games.jsonl"


def ai_games_for(map_name: str | None) -> Path:
    """The built-in AI's games on a run's map, for its record there (duelfast's for the duel maps and the
    Echo Isles variants with other rules)."""
    return ECHO_AI_GAMES if map_name in ("(2)EchoIsles", "EchoIsles", "echo100") else AI_GAMES


def run_map(name: str) -> str | None:
    """A run's --map (its last launch's)."""
    try:
        argv = json.loads((RUNS / name / "run.json").read_text())["launch"]["argv"]
        return argv[argv.index("--map") + 1] if "--map" in argv else None
    except (OSError, ValueError, KeyError):
        return None


def ai_rates(games: Path | None = None) -> dict:
    """(race, opponent's race, difficulty) -> the AI's win rate there (mirrors 0.5), from `games` (default:
    duelfast's)."""
    tab: dict = defaultdict(lambda: [0, 0])
    path = games or AI_GAMES
    if path.exists():
        for line in path.open():
            g = json.loads(line)
            d = g["difficulties"]
            if d[0] != d[1] or g.get("winner") is None:
                continue
            for s in (0, 1):
                k = (g["races"][s], g["races"][1 - s], d[0])
                tab[k][0] += g["winner"] == s
                tab[k][1] += 1
    return {k: 0.5 if k[0] == k[1] else w / n for k, (w, n) in tab.items() if n >= 10}


def difficulty(e: dict) -> str:
    return next((d for d in ("easy", "normal", "insane") if d in e["opponent"]), "?")


def standardised(eps: list[dict], field: str, rates: dict) -> tuple[dict, int]:
    """Each value of `field`'s AI-record share on one matchup mix: its win rate in each matchup (race,
    opponent's race, difficulty) that every value played, weighted by the matchup's games over all
    values, over the AI's record on that mix. (Drawn apart, the values' matchups differ, and a share
    depends on the mix: the policy is far above the AI's record as human, near it as night elf.)
    -> ({value: share %}, games counted)."""
    cell = lambda e: (e["race"], e["opponent_race"], difficulty(e))  # noqa: E731
    t: dict = {}
    for e in eps:
        if "(real)" in e["opponent"] and rates.get(cell(e)) is not None:
            a = t.setdefault((str(e[field]), cell(e)), [0, 0])
            a[0] += e["outcome"] > 0
            a[1] += 1
    vals = sorted({v for v, _ in t})
    common = [c for c in {c for _, c in t} if all((v, c) in t for v in vals)]
    w = {c: sum(t[(v, c)][1] for v in vals) for c in common}
    ai = sum(w[c] * rates[c] for c in common)
    if not ai:
        return {}, 0
    return {v: 100 * sum(w[c] * t[(v, c)][0] / t[(v, c)][1] for c in common) / ai for v in vals}, sum(w.values())


def tier2(prod: dict) -> bool:
    return any(k in prod["built"] or k in prod["trained"] for k in TIER2)


def report(eps: list[dict], label: str, rates: dict | None = None) -> None:
    real = [e for e in eps if "(real)" in e["opponent"]]
    won = sum(e["outcome"] > 0 for e in real)
    print(f"== {label}: real games {won} of {len(real)} won ({100 * won / max(1, len(real)):.0f}%)")
    if rates:  # against the AI's own record in the same matchups (normal and insane; easy has none)
        rated = [(e, rates.get((e["race"], e["opponent_race"], difficulty(e)))) for e in real]
        rated = [(e, p) for e, p in rated if p is not None]
        if rated:
            w = sum(e["outcome"] > 0 for e, _ in rated)
            x = sum(p for _, p in rated)
            print(f"   vs normal+insane: {w} of {len(rated)} won; the built-in AI in the same matchups would win "
                  f"{x:.0f} ({100 * w / max(x, 1e-9):.0f}% of the AI's record)")
    for key, f in (("by difficulty", difficulty), ("by own race", lambda e: e["race"]),
                   ("by own race vs normal+insane", lambda e: e["race"] if difficulty(e) != "easy" else None)):
        c: dict = defaultdict(lambda: [0, 0])
        for e in real:
            k = f(e)
            if k:
                c[k][0] += e["outcome"] > 0
                c[k][1] += 1
        print(f"   {key}: " + ", ".join(f"{k} {w}/{n}" for k, (w, n) in sorted(c.items())))
    for d in ("easy", "normal", "insane"):
        D = [e for e in real if difficulty(e) == d]
        if not D:
            continue
        def m(k: str, side: str = "prod") -> float:  # (a game over before its first minute has no food then)
            v = [e[side][k] for e in D if e[side].get(k) is not None]
            return sum(v) / max(len(v), 1)
        t2 = lambda side: 100 * sum(tier2(e[side]) for e in D) / len(D)  # noqa: E731
        wpm = lambda side: sum(e[side]["workers"] for e in D) / sum(e["game_time"] / 60 for e in D)  # noqa: E731
        print(f"   {d:6s} (learner vs AI): food at 1 min {m('food_1min'):.1f} vs {m('food_1min', 'opp_prod'):.1f}, "
              f"heroes {m('heroes'):.1f} vs {m('heroes', 'opp_prod'):.1f}, tier 2 {t2('prod'):.0f}% vs {t2('opp_prod'):.0f}%, "
              f"workers a minute {wpm('prod'):.2f} vs {wpm('opp_prod'):.2f}, kills per unit lost {m('kills') / max(m('lost'), 1e-9):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name")
    ap.add_argument("--since", help="HH:MM or 'YYYY-mm-dd HH:MM' (default: 24 hours ago)")
    ap.add_argument("--until", help="the end of the first window (default: now)")
    ap.add_argument("--recent", type=float, default=30.0, help="minutes for the recent window")
    ap.add_argument("--split", help="an episode field (e.g. shared_noise, an A/B's flag): the first window's report "
                                    "per value of it, and the ties per value")
    a = ap.parse_args()
    eps = [json.loads(line) for line in (RUNS / a.name / "episodes.jsonl").open()]
    now = time.time()
    lo = when(a.since) if a.since else now - 86400
    rates = ai_rates(ai_games_for(run_map(a.name)))
    hi = when(a.until) if a.until else now
    report([e for e in eps if lo < e["time"] <= hi], f"since {time.strftime('%Y-%m-%d %H:%M', time.localtime(lo))}"
           + (f" until {time.strftime('%Y-%m-%d %H:%M', time.localtime(hi))}" if a.until else ""), rates)
    if a.split:  # an A/B within the window: each value's games apart
        window = [e for e in eps if lo < e["time"] <= hi and a.split in e]
        for v in sorted({str(e[a.split]) for e in window}):
            part = [e for e in window if str(e[a.split]) == v]
            report(part, f"{a.split} = {v}", rates)
            for kind in ("self", "past"):
                X = [e for e in part if e["opponent"].split(":")[0] == kind]
                print(f"   against {kind}: {len(X)} games, {100 * sum(e['outcome'] == 0 for e in X) / max(1, len(X)):.0f}% ties")
        shares, n = standardised(window, a.split, rates)
        if shares:
            print(f"== on one matchup mix ({n} games in the matchups every value played): "
                  + ", ".join(f"{v} {s:.0f}%" for v, s in shares.items()))
        return 0
    recent = [e for e in eps if e["time"] > now - 60 * a.recent]
    report(recent, f"last {a.recent:.0f} min", rates)
    curr = [e for e in recent if e["opponent"].startswith("script:") and "(real)" not in e["opponent"] and not e.get("advised")]
    o = Counter(e["outcome"] for e in curr)
    print(f"unadvised curriculum games ({len(curr)}): won {o[1.0]}, tied {o[0.0]}, lost {o[-1.0]}; "
          f"mean tax {sum(e['ai_tax'] for e in curr) / max(1, len(curr)):.2f}")
    for kind in ("self", "past"):
        X = [e for e in recent if e["opponent"].split(":")[0] == kind]
        print(f"against {kind}: {len(X)} games, {100 * sum(e['outcome'] == 0 for e in X) / max(1, len(X)):.0f}% ties")
    return 0


if __name__ == "__main__":
    sys.exit(main())
