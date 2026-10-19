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


def owners_at(units: np.ndarray, events: np.ndarray) -> list[int]:
    """Each event's unit's (column 2) owner at the event's step, from the nearest row of that unit at or
    before it (else after): the game reuses a dead unit's id, so a whole game's last owner of an id gave
    one player's buildings to the other (a human's barracks in undead strategies)."""
    order = np.lexsort((units[:, 0], units[:, 1]))
    ids, steps, own = units[order, 1], units[order, 0], units[order, 3]
    out = []
    for step, a in zip(events[:, 0].tolist(), events[:, 2].tolist()):
        lo, hi = np.searchsorted(ids, a, "left"), np.searchsorted(ids, a, "right")
        if lo == hi:
            out.append(-1)
            continue
        k = lo + int(np.searchsorted(steps[lo:hi], step, "right")) - 1
        out.append(int(own[max(k, lo)]))
    return out


RACE_LETTER = {"human": "h", "orc": "o", "undead": "u", "nightelf": "e"}


def of_race(code: str, race: str) -> bool:
    """A melee item of `race` (its id's letter: units and buildings first, heroes upper case, research
    after the R)."""
    return (code[1] if code[0] == "R" else code[0]).lower() == RACE_LETTER[race]


def from_demos(path: Path) -> list[dict]:
    """The two players' strategies of one demonstration game (collect.py's .npz)."""
    from .features import rawcode
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    units, ev, dt = z["units"], z["events"], float(meta.get("step_seconds", 0.5))
    made: dict[int, list] = {0: [], 1: []}
    for (step, kind, a, b, c), p in zip(ev.tolist(), owners_at(units, ev)):
        if kind in MADE_EVENTS and p in (0, 1):
            made[p].append((rawcode(c if kind == 6 else b), step * dt))
    result = meta.get("result") or {}
    races = meta["races"]
    diff = meta.get("difficulties") or [None, None]
    return [{"race": races[p], "opp": races[1 - p], "src": "ai", "won": result.get(str(p)) == "VICTORY",
             "t": _first_times(made[p]), "game": f"{path.parent.name}/{path.name}", "player": p,
             "difficulty": diff[p], "map": meta.get("map"), "end": round(float(units[:, 0].max()) * dt, 1)}
            for p in (0, 1)]


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
                        "t": _first_times(pairs), "replay": r.get("file"), "slot": b.get("slot"),
                        "end": round(r.get("length_ms", HORIZON * 1000) / 1000.0, 1)})
    return out


def orderable(vocab: dict) -> set[str]:
    """The items the policy can order at all (its vocabulary's train / build / research orders: those the
    built-in AI gave in the demonstrations). Tavern heroes, mercenaries and the neutral shops' units, and
    what the AI never made, are not among them: a target with one is missed for certain."""
    from .features import TYPE_CODE, rawcode
    return {rawcode(o) for o, _ in vocab["orders"] if o >= TYPE_CODE}


def consistent(strategy: dict, tree=None) -> dict:
    """`strategy` with no item before what it needs (techtree: its requirements, a hero's per its place
    among the strategy's heroes, and the building that makes it) where the strategy makes those: moving
    single items later (rescale_people's floor) can put a unit before its barracks. In place."""
    from . import techtree
    tree = tree or techtree.tree()
    t = strategy["t"]
    heroes = sorted((v, c) for c, v in t.items() if techtree.is_hero(c))
    place = {c: i for i, (_, c) in enumerate(heroes)}
    for _ in range(4):
        changed = False
        for c in sorted(t, key=t.get):
            per = tree.needs.get(c, ())
            needs = [techtree.resolve(n, strategy["race"]) for n in (per[min(place.get(c, 0), len(per) - 1)] if per else ())]
            needs += [m for m in tree.makers.get(c, ()) if m in t][:1]
            first = max((t[n] for n in needs if n in t), default=0.0)
            if t[c] < first:
                t[c], changed = first, True
        if not changed:
            break
    return strategy


def rescale_people(strategies: list[dict], quantile: float = 0.1) -> dict[str, float]:
    """People's times (the real rules, an order's click plus CLICK_LAG) onto the built-in AI's demonstrations'
    rules (e.g. echofast): per race, the factor between the medians of each item's first time, AI over people
    (items both make, median over items); then no item earlier than the AI's `quantile` of first times for it
    (a person's build order is kept, its schedule made one an agent with a normal economy can keep).
    In place; -> the factors."""
    factors = {}
    for race in RACES:
        ai = [s for s in strategies if s["race"] == race and s["src"] == "ai"]
        hu = [s for s in strategies if s["race"] == race and s["src"] == "human"]
        if not ai or not hu:
            continue
        t_ai = {c: [s["t"][c] for s in ai if c in s["t"]] for c in {c for s in ai for c in s["t"]}}
        t_hu = {c: [s["t"][c] for s in hu if c in s["t"]] for c in {c for s in hu for c in s["t"]}}
        both = [c for c in t_ai if c in t_hu and len(t_ai[c]) >= 20 and len(t_hu[c]) >= 20]
        if not both:
            continue
        f = float(np.median([np.median(t_ai[c]) / np.median(t_hu[c]) for c in both]))
        floor = {c: float(np.quantile(v, quantile)) for c, v in t_ai.items() if len(v) >= 10}
        factors[race] = round(f, 3)
        for s in hu:
            s["t"] = {c: max(t * f, floor.get(c, 0.0)) for c, t in s["t"].items()}
            s["end"] = round(s.get("end", HORIZON) * f, 1)
            consistent(s)
    return factors


def build_pool(strategies: list[dict], min_items: int = 5, keep: set[str] | None = None,
               items: dict[str, list[str]] | None = None) -> dict:
    """The pool: each race's item list (its SLOTS items most strategies make within HORIZON; or `items`,
    another pool's: the slots a policy learned) and the strategies, their times cut to the horizon and the
    lists."""
    items = dict(items or {})
    for race in [r for r in RACES if r not in items]:
        n = Counter(code for s in strategies if s["race"] == race for code, t in s["t"].items()
                    if t <= HORIZON and (keep is None or code in keep) and of_race(code, race))
        items[race] = [code for code, _ in n.most_common(SLOTS)]
    kept = []
    for s in strategies:
        t = {code: round(v, 1) for code, v in s["t"].items()
             if v <= HORIZON and code in items[s["race"]] and of_race(code, s["race"])}
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


# made whatever the target: each race's workers and supply (a target without farms would block itself)
ALWAYS = frozenset({"hpea", "opeo", "uaco", "ewsp", "hhou", "otrb", "uzig", "emow"})


def allowed(strategy: dict, items: list[str], orders: list) -> np.ndarray:
    """--z-enforce: [1 + n_orders] bool, the order classes a side playing toward `strategy` may give (row 0:
    no order): a train / build / research order of one of the race's `items` only when the strategy makes
    it, or what it needs on the way (techtree.closure: requirements, the building that makes it), or it is a
    worker or supply; every other order (moves, attacks, harvesting, spells, items outside the list) stays."""
    from . import features as fx
    from .techtree import closure
    keep = (closure(strategy["t"], strategy["race"]) if "race" in strategy else set(strategy["t"])) | ALWAYS
    listed = set(items)
    out = np.ones(len(orders) + 1, bool)
    for c, (oid, kind) in enumerate(orders, start=1):
        if oid >= fx.TYPE_CODE and kind in (fx.IMMEDIATE, fx.POINT):
            code = fx.rawcode(oid)
            out[c] = code not in listed or code in keep
    return out


def horizon(strategy: dict) -> float:
    """The game seconds `strategy` covers: HORIZON, or less when its game ended sooner ("end")."""
    return min(HORIZON, strategy.get("end", HORIZON))


def scaled(strategy: dict, scale: float) -> dict:
    """`strategy` with its items' times scaled (--z-time-scale: faster rules reach each item sooner)."""
    return strategy if scale == 1.0 else {**strategy, "t": {k: v * scale for k, v in strategy["t"].items()},
                                          "end": horizon(strategy) * scale}


def mismatch(made: dict[str, float], strategy: dict, items: list[str], now: float) -> int:
    """The race's items made by game second `now` (up to the strategy's horizon) by one of the player
    (`made`: each item's first time) and the strategy, not both."""
    now = min(now, horizon(strategy))
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
    ap.add_argument("--vocab", type=Path, help="keep only the items this vocabulary can order (strategy.orderable)")
    ap.add_argument("--difficulty", help="keep only the built-in AI's sides at this difficulty (e.g. normal: the "
                    "insane AI gathers twice as much, so its schedule is not an agent's)")
    ap.add_argument("--items-from", type=Path, help="another pool whose item lists to keep (a policy's slots)")
    ap.add_argument("--rescale-people", type=float, default=0.0, help="people's times onto the demonstrations' rules, "
                    "no item earlier than this quantile of the AI's (strategy.rescale_people; 0: as recorded)")
    a = ap.parse_args(argv)
    strategies = []
    for d in a.demos:
        for f in sorted(d.glob("game*.npz")):
            if not f.name.endswith(".tmp.npz"):
                strategies += from_demos(f)
    if a.difficulty:
        strategies = [s for s in strategies if s.get("difficulty") == a.difficulty]
    for f in a.replays:
        strategies += from_replays(f)
    if a.rescale_people > 0:
        print("people's times, factor per race:", rescale_people(strategies, a.rescale_people))
    keep = orderable(json.loads(a.vocab.read_text())) if a.vocab else None
    if keep is not None:
        strategies = [{**s, "t": {c: t for c, t in s["t"].items() if c in keep}} for s in strategies]
    pool = build_pool(strategies, keep=keep, items=json.loads(a.items_from.read_text())["items"] if a.items_from else None)
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
