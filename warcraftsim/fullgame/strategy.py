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

Narrow strategies (--narrow; the user's idea, 2026-10-20: the whole games' targets were big and alike): a
game's own heroes and main army (the unit types with FOOD_SHARE of its army's food, at most 3) and what they
need (techtree.closure), nothing else; their buildings and research outside it are free (mismatch, allowed),
heroes and army units outside it are not. Drawn evenly over their signatures (heroes, main army: "sig"), not
over games, so a rare build comes up as often as a common one.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from functools import lru_cache
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
    after the R), or one any race hires (a tavern hero, a mercenary: neutral.py)."""
    from .neutral import MERCENARIES, TAVERN_HEROES
    return (code[1] if code[0] == "R" else code[0]).lower() == RACE_LETTER[race] or code in TAVERN_HEROES or code in MERCENARIES


def from_demos(path: Path) -> list[dict]:
    """The two players' strategies of one demonstration game (collect.py's .npz)."""
    from .features import rawcode
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    units, ev, dt = z["units"], z["events"], float(meta.get("step_seconds", 0.5))
    made: dict[int, list] = {0: [], 1: []}
    count: dict[int, Counter] = {0: Counter(), 1: Counter()}  # each unit type trained (the main army: narrow)
    for (step, kind, a, b, c), p in zip(ev.tolist(), owners_at(units, ev)):
        if kind in MADE_EVENTS and p in (0, 1):
            made[p].append((rawcode(c if kind == 6 else b), step * dt))
            if kind == 6:
                count[p][rawcode(c)] += 1
    result = meta.get("result") or {}
    races = meta["races"]
    diff = meta.get("difficulties") or [None, None]
    return [{"race": races[p], "opp": races[1 - p], "src": "ai", "won": result.get(str(p)) == "VICTORY",
             "t": _first_times(made[p]), "n": dict(count[p]), "game": f"{path.parent.name}/{path.name}", "player": p,
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
            n = Counter(code for t_ms, kind, code, *_ in b["build"] if kind == "train")
            out.append({"race": races[i], "opp": races[1 - i], "src": "human", "won": bool(b.get("won")),
                        "t": _first_times(pairs), "n": dict(n), "replay": r.get("file"), "slot": b.get("slot"),
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


FOOD_SHARE = 0.2  # a narrow strategy's main army: the unit types with this share of its army's food


@lru_cache(None)
def restricted(code: str) -> bool:
    """A hero or an army unit (not a worker, building or research): what a narrow strategy makes only when it
    lists it."""
    from ..data.objects import unit_table
    from .techtree import is_hero
    if is_hero(code):
        return True
    u = unit_table().get(code)
    return u is not None and not u.is_building and code not in WORKERS and code[:1].islower()


def narrow(strategy: dict, tree=None) -> dict | None:
    """`strategy` (with its unit counts "n") narrowed to its heroes, its main army and what they need, at the
    times it made them; None when its main army is not among its items (none, or made past the horizon), or it has
    no hero among them."""
    from ..data.objects import unit_table
    from .techtree import closure, is_hero
    units = unit_table()
    food = {c: k * max(units[c].food, 1) for c, k in (strategy.get("n") or {}).items()
            if c in units and restricted(c) and not is_hero(c)}
    total = sum(food.values())
    core = [c for c, f in sorted(food.items(), key=lambda kv: (-kv[1], kv[0])) if f >= FOOD_SHARE * total][:3]
    t = strategy["t"]
    if not core or any(c not in t for c in core):
        return None
    heroes = sorted((c for c in t if is_hero(c)), key=t.get)
    if not heroes:  # (a player whose heroes were the tavern's, not orderable: its target would forbid any)
        return None
    keep = closure(set(core) | set(heroes), strategy["race"], tree)
    return {**strategy, "t": {c: v for c, v in t.items() if c in keep}, "narrow": True,
            "sig": f"{'+'.join(heroes) or '-'}|{'+'.join(sorted(core))}"}


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
    free = strategy.get("narrow")  # (a narrow strategy: buildings and research it does not list are free)
    out = np.ones(len(orders) + 1, bool)
    for c, (oid, kind) in enumerate(orders, start=1):
        if oid >= fx.TYPE_CODE and kind in (fx.IMMEDIATE, fx.POINT):
            code = fx.rawcode(oid)
            out[c] = code not in listed or code in keep or bool(free and not restricted(code))
    return out


WORKER_CODES = ("hpea", "opeo", "uaco", "ewsp")


def is_army(code: str) -> bool:
    """An army unit's code (a race's unit, no worker, hero, building or research): lower case and trained."""
    return code[:1].islower() and code not in WORKER_CODES


def mix(strategy: dict | None, items: list[str]) -> np.ndarray:
    """--z-mix: the policy's input for a strategy's army: per slot, the item's share of the army units it made (its
    counts "n"; zeros: none, or no army)."""
    z = np.zeros(SLOTS, np.float32)
    n = {c: v for c, v in ((strategy or {}).get("n") or {}).items() if is_army(c) and v > 0}
    total = sum(n.values())
    if not total:
        return z
    for k, code in enumerate(items[:SLOTS]):
        z[k] = n.get(code, 0) / total
    return z


_PER_STRATEGY: dict = {}  # (what, id(a), id(b)) -> (a, b, a value computed from them once; a few games' worth)


def _once(what: str, a, b, make):
    """make() for these two objects (a strategy's dicts, a race's items), computed once while they live (per step
    they had been ~3% of an actor's time)."""
    key = (what, id(a), id(b))
    hit = _PER_STRATEGY.get(key)
    if hit is None or hit[0] is not a or hit[1] is not b:
        if len(_PER_STRATEGY) > 64:
            _PER_STRATEGY.clear()
        hit = _PER_STRATEGY[key] = (a, b, make())
    return hit[2]


def mix_gap(trained: dict[str, int], strategy: dict, least: int = 5) -> float:
    """--z-mix: how far the army a side trained is from its strategy's (the total variation of the two mixes, 0 to
    1); 0 while it trained fewer than `least` army units or the strategy has no army."""
    n = strategy.get("n") or {}
    want = _once("want", n, None, lambda: {c: v for c, v in n.items() if is_army(c) and v > 0}) if n else {}
    have = {c: v for c, v in trained.items() if is_army(c) and v > 0}
    tw, th = sum(want.values()), sum(have.values())
    if not tw or th < least:
        return 0.0
    return 0.5 * sum(abs(have.get(c, 0) / th - want.get(c, 0) / tw) for c in set(want) | set(have))


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
    t = strategy["t"]
    if strategy.get("narrow"):  # (its own items as any strategy's; outside them, heroes and army units only)
        counted = _once("narrow", t, items, lambda: [c for c in items if c in t or restricted(c)])
        return sum((made.get(c, 1e9) <= now) != (t.get(c, 1e9) <= now) for c in counted)
    return sum((made.get(c, 1e9) <= now) != (t.get(c, 1e9) <= now) for c in items)


def sample(pool: dict, race: str, opp: str, rng: random.Random, winners: bool = True, src: str = "") -> dict | None:
    """A strategy of `race`, against `opp` when the pool has some (else any opponent), won ones when
    `winners` and there are any; only those of one source when `src` ("human": people's replays, "ai": the
    built-in AI's games)."""
    mine = [s for s in pool["strategies"] if s["race"] == race and (not src or s["src"] == src)]
    tiers = [[s for s in mine if s["opp"] == opp and s["won"]], [s for s in mine if s["won"]]] if winners else []
    tiers += [[s for s in mine if s["opp"] == opp], mine]
    c = next((c for c in tiers if c), None)
    if c is None:
        return None
    if "sig" in c[0]:  # (narrow strategies: a signature first, each as likely, then one of its games)
        sig = rng.choice(sorted({s["sig"] for s in c}))
        c = [s for s in c if s["sig"] == sig]
    return rng.choice(c)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demos", type=Path, nargs="*", default=[], help="directories of demonstrations (game*.npz)")
    ap.add_argument("--replays", type=Path, nargs="*", default=[], help="replay_stats.py's stats.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, help="keep only the items this vocabulary can order (strategy.orderable)")
    ap.add_argument("--difficulty", help="keep only the built-in AI's sides at this difficulty (e.g. normal: the "
                    "insane AI gathers twice as much, so its schedule is not an agent's)")
    ap.add_argument("--items-from", type=Path, help="another pool whose item lists to keep (a policy's slots)")
    ap.add_argument("--tavern", type=int, default=0, help="each race's this many least common item slots swapped for the "
                    "tavern heroes its people pick most (an extended vocabulary: --neutral, neutral.extend_vocab)")
    ap.add_argument("--narrow", action="store_true", help="each strategy narrowed to its heroes, its main army and "
                    "what they need (narrow): buildings and research outside them free; drawn evenly per signature")
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
    items = json.loads(a.items_from.read_text())["items"] if a.items_from else None
    if a.tavern and items is not None:  # (the slots: the old ones, the rarest swapped for the race's tavern picks)
        from .techtree import is_hero
        items = {r: list(v) for r, v in items.items()}
        for race in RACES:
            picks = Counter(c for st in strategies if st["race"] == race and st["src"] == "human"
                            for c in st["t"] if is_hero(c) and c[0] == "N")
            new = [c for c, _ in picks.most_common(a.tavern) if c not in items[race]]
            if new:
                items[race] = items[race][:len(items[race]) - len(new)] + new
                print(f"  {race}: tavern heroes in the last slots: {new}")
    pool = build_pool(strategies, keep=keep, items=items)
    if a.narrow:
        from .techtree import tree
        pool["strategies"] = [n for n in (narrow(s, tree()) for s in pool["strategies"]) if n is not None]
        sigs = Counter((s["race"], s["sig"]) for s in pool["strategies"])
        for race in RACES:
            mine = Counter({k[1]: v for k, v in sigs.items() if k[0] == race})
            print(f"  {race}: {len(mine)} signatures, the most common {mine.most_common(5)}")
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
