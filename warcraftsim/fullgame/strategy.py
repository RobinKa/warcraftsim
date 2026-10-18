"""Strategies for strategy conditioning (AlphaStar's statistic z): what a player made and when, from the
built-in AI's demonstrations and people's replays; a strategy as the policy's input (the global vector's
strategy block, features.ZDIM) and a reward for following it.

    python -m warcraftsim.fullgame.strategy --demos runs/fullgame/demos-echo-1 \\
        --replays runs/fullgame/human-replays-129/stats.jsonl --out runs/fullgame/strategies-echo.json

A strategy: a player's race and its opponent's, where it comes from ("ai" or "human"), whether it won,
and the first time (game seconds) it made each item of its race's list (the race's SLOTS most common:
units, heroes, buildings, research; workers left out) within HORIZON. The policy's input (vector()):
per slot whether the strategy makes the item and when (time / HORIZON; 1 if never). The reward
(mismatch()): the items made by now, the player's against the strategy's, counted where they differ
(made by one and not the other): the policy is pulled to the strategy's tech path on its schedule,
and away from items it does not make.

The built-in AI's times are the items' (an event when an item is done); people's replays give the
order's (a click), so their times get CLICK_LAG, about an item's build time.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

SLOTS = 64  # items per race
HORIZON = 900.0  # game seconds a strategy covers (Echo Isles: wins ~10 minutes, losses ~24)
CLICK_LAG = 45.0  # people's times are their orders', the game's (and the AI's) the finished items'
ZDIM = 2 * SLOTS  # the policy's input: per slot, made and when
WORKERS = {"hpea", "opeo", "uaco", "ewsp"}
RACES = ("human", "orc", "undead", "nightelf")
MADE_EVENTS = (6, 3, 12, 9)  # trained (c: the type), built / upgraded (b), research done (b): trace.Production


def _first_times(pairs) -> dict[str, float]:
    out: dict[str, float] = {}
    for code, t in pairs:
        if code not in WORKERS and (code not in out or t < out[code]):
            out[code] = t
    return out


def from_demos(path: Path) -> list[dict]:
    """The two players' strategies of one demonstration game (collect.py's .npz)."""
    from .features import rawcode
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    units, ev, dt = z["units"], z["events"], float(meta.get("step_seconds", 0.5))
    owner = dict(zip(units[:, 1].tolist(), units[:, 3].tolist()))  # (step id type owner ...)
    made: dict[int, list] = {0: [], 1: []}
    for step, kind, a, b, c in ev.tolist():
        if kind in MADE_EVENTS and owner.get(a) in (0, 1):
            made[owner[a]].append((rawcode(c if kind == 6 else b), step * dt))
    result = meta.get("result") or {}
    races = meta["races"]
    return [{"race": races[p], "opp": races[1 - p], "src": "ai", "won": result.get(str(p)) == "VICTORY",
             "t": _first_times(made[p]), "game": f"{path.parent.name}/{path.name}", "player": p} for p in (0, 1)]


def from_replays(stats: Path) -> list[dict]:
    """Every 1v1 player's strategy in replay_stats.py's rows (people's replays)."""
    out = []
    for line in stats.open():
        r = json.loads(line)
        if r.get("type") != "1v1":
            continue
        bos = r.get("build_orders") or []
        races = [b.get("race_played") for b in bos]
        if len(bos) != 2 or any(x not in RACES for x in races):
            continue
        for i, b in enumerate(bos):
            pairs = [(code, t_ms / 1000.0 + CLICK_LAG) for t_ms, kind, code, *_ in b["build"]
                     if kind in ("train", "build", "hero", "research", "upgrade")]
            out.append({"race": races[i], "opp": races[1 - i], "src": "human", "won": bool(b.get("won")),
                        "t": _first_times(pairs), "replay": r.get("file"), "slot": b.get("slot")})
    return out


def build_pool(strategies: list[dict], min_items: int = 5) -> dict:
    """The pool: each race's item list (its SLOTS items most strategies make within HORIZON) and the
    strategies, their times cut to the horizon and the lists."""
    items = {}
    for race in RACES:
        n = Counter(code for s in strategies if s["race"] == race for code, t in s["t"].items() if t <= HORIZON)
        items[race] = [code for code, _ in n.most_common(SLOTS)]
    kept = []
    for s in strategies:
        t = {code: round(v, 1) for code, v in s["t"].items() if v <= HORIZON and code in items[s["race"]]}
        if len(t) >= min_items:
            kept.append({**s, "t": t})
    return {"slots": SLOTS, "horizon": HORIZON, "items": items, "strategies": kept}


def hindsight(pool: dict, path: Path, meta: dict) -> dict[int, dict]:
    """A demonstration game's sides' own strategies (what each made and when: the target it followed, in
    hindsight; an auxiliary cloning loss with it as the input teaches the policy what a strategy means, as
    AlphaStar's supervised policy learned with each replay's own): the pool's entries of that game, the
    built-in AI's by the game's file, people's by its replay and the players' slots. A side the pool left
    out (too few items) has none."""
    if "_games" not in pool:  # (an index, once)
        pool["_games"] = {(s.get("game"), s.get("player")): s for s in pool["strategies"] if "game" in s}
        pool["_replays"] = {(s.get("replay"), s.get("slot")): s for s in pool["strategies"] if "replay" in s}
    src = meta.get("source") or {}
    if src.get("replay"):
        slots = src.get("slots") or []
        return {p: pool["_replays"][(src["replay"], slots[p])] for p in (0, 1)
                if p < len(slots) and (src["replay"], slots[p]) in pool["_replays"]}
    key = f"{Path(path).parent.name}/{Path(path).name}"
    return {p: pool["_games"][(key, p)] for p in (0, 1) if (key, p) in pool["_games"]}


def vector(strategy: dict | None, items: list[str]) -> np.ndarray:
    """The policy's input for a strategy (zeros: none): per slot, made (1) and its time / HORIZON
    (1: never) in the second half."""
    z = np.zeros(ZDIM, np.float32)
    if strategy is None:
        return z
    z[SLOTS:SLOTS + len(items)] = 1.0
    for k, code in enumerate(items):
        t = strategy["t"].get(code)
        if t is not None and t <= HORIZON:
            z[k], z[SLOTS + k] = 1.0, t / HORIZON
    return z


def mismatch(made: dict[str, float], strategy: dict, items: list[str], now: float) -> int:
    """The race's items made by game second `now` (up to HORIZON) by one of the player (`made`: each
    item's first time) and the strategy, not both."""
    now = min(now, HORIZON)
    return sum((made.get(c, 1e9) <= now) != (strategy["t"].get(c, 1e9) <= now) for c in items)


def sample(pool: dict, race: str, opp: str, rng: random.Random, winners: bool = True, src: str = "") -> dict | None:
    """A strategy of `race`, against `opp` when the pool has some (else any opponent), won ones when
    `winners` and there are any; only those of one source when `src` ("human": people's replays, "ai": the
    built-in AI's games)."""
    mine = [s for s in pool["strategies"] if s["race"] == race and (not src or s["src"] == src)]
    tiers = [[s for s in mine if s["opp"] == opp and s["won"]], [s for s in mine if s["won"]]] if winners else []
    tiers += [[s for s in mine if s["opp"] == opp], mine]
    return next((rng.choice(c) for c in tiers if c), None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demos", type=Path, nargs="*", default=[], help="directories of demonstrations (game*.npz)")
    ap.add_argument("--replays", type=Path, nargs="*", default=[], help="replay_stats.py's stats.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    strategies = []
    for d in a.demos:
        for f in sorted(d.glob("game*.npz")):
            if not f.name.endswith(".tmp.npz"):
                strategies += from_demos(f)
    for f in a.replays:
        strategies += from_replays(f)
    pool = build_pool(strategies)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(pool))
    by = Counter((s["race"], s["src"], s["won"]) for s in pool["strategies"])
    print(f"{a.out}: {len(pool['strategies'])} strategies of {len(strategies)}")
    for race in RACES:
        n = {f"{src}{' won' if won else ''}": by[(race, src, won)] for src in ("ai", "human") for won in (True, False)}
        print(f"  {race}: {n}; items {' '.join(pool['items'][race][:16])} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
