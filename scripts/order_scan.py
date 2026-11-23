"""Orders a policy wastes, from a run's saved replays and episode rows (2026-11-23, bug finder 11: the habits the user
spots in replay videos, the ghoul's back and forth, an army's attack-move every step, found from the data instead).

    python scripts/order_scan.py runs/fgself-249 runs/burst-248 [--top 25] [--unit GAME UNIT] [--episodes 4000]

Per replay (runs/RUN/replays/STEM.trace.json and .commands.json: the orders the policy chose a step, by side and
label, and the commands the agents sent; read only) and per unit:
  alternations   two orders in turn (A, B, A, B; each within --gap steps of the last): the orders in such runs
  repeats        the same order with the same aim re-sent with no other between (each within --gap steps)
  re-aims        the same order, another aim, a step or two apart, 4 or more in a row (a point re-picked every step)
  countermands   an order replaced by another kind within --gap steps (A -> B: the share of A's so replaced)
  bursts         units ordered at least every second step for 20 s (20 orders in 40 steps), their orders
  not sent       per label: chosen by the policy (the trace) and not sent as a command (dropped by the agent: a
                 harvest for a worker harvesting, an order for a builder, one the masks' checks know refused)
ranked by the orders they cost, per game, and split by race, by a target strategy or none, and by the policy that
played the side (an evaluation's replays mix its listed checkpoints: a habit may be one checkpoint's alone; --policy).
A unit's side comes from the trace (the labels each side chose that step), its kind from its orders over the game
(a building trains or researches, a worker builds, harvests or repairs, a hero learns skills or uses items; "army"
the rest). --unit GAME UNIT prints one unit's orders step by step (GAME: a replay's stem or a part of it).

Per run's episode rows (runs/RUN/episodes.jsonl and episodes-remote.jsonl, their last --episodes): the orders the game
refused, by kind and race, with the refused orders' contexts (the unit's type and current order, the target), and the
agent's own counters of orders it did not send (noop_repairs, futile_orders, over_budget, ...) a game.

What it found (2026-11-23; bursts 241-248's and fgself-241..250's replays, ~110 games):
  * heroes attack-moved at one place every step in target games (110-155 repeats a side-game; the game starts the
    swing over each time, scripts/order_probe.py): --order-masks bit 8
  * 2-16% of every policy's attack-moves, moves and patrols aimed above the map (the old point bins' top edge bin),
    and a third to a half of the ziggurats (refused): docs/reference/engine-notes.md; bit 16 for the buildings
  * fgself-244's average cancels and re-trains undead's first hero in every game without a target (12 of 12 sides:
    9-32 cancels, the hero 40-197 s late); no other checkpoint does (0 of 22 sides)
  * harvesters moved off their work and sent back: undead's ghoul (before --econ-masks bit 256, also in games
    without a target: 17-44 orders a side-game), wisps, and human peasants on gold in single long games (5 peasants
    with ~80 moves each in burst-244's game-1795437945-a13)
  * chosen and not sent (the agent's own checks): orc "stop" 98% (peons in burrows: loaded_orders 109-168 a game),
    undead "restoration" 97% and orc "repair" 87% (nothing to repair), orc "harvest (unit)" 80% (harvesting already)
  * refused most (a game, evaluation): undead cannibalize 24 and build uzig 13, night elf recharge 19 and entangle
    8, human and orc smart on a point 10-11 (refused always), 851973 on a unit 2-7 (refused always)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TYPE_CODE = 0x1000000
KIND_NAMES = {1: "point", 2: "unit", 6: "tree"}
RACES = ("human", "orc", "undead", "nightelf")
COUNTERS = ("noop_repairs", "futile_orders", "over_budget", "frame_cancels", "frame_orders", "full_queue",
            "upgrading_refused", "loaded_orders", "toggles_masked", "militia_skipped", "stood_down_n")


def rawcode(v: int) -> str:
    try:
        return int(v).to_bytes(4, "big").decode("latin-1")
    except (OverflowError, ValueError):
        return str(v)


def decode(ints: list[int], names: dict[int, str]) -> list[tuple]:
    """A step's encoded commands -> the unit orders among them: (unit, label, aim, op). The label as the trace's
    (features.order_label), the aim what the order is on (a unit's or tree's id, a point), None for one without."""
    from warcraftsim.protocol import COORD_BIAS, _OP_LENGTHS, Op
    out, i = [], 0
    while i < len(ints):
        op = int(ints[i])
        n = _OP_LENGTHS.get(op)
        if n is None:
            break
        a = [int(v) for v in ints[i + 1:i + n]]
        i += n

        def name(o: int, kind: str = "") -> str:
            if o >= TYPE_CODE:
                code = rawcode(o)
                verb = {"": "research" if code.startswith("R") else "train", "point": "build"}.get(kind, "")
                return f"{verb} {code}".strip()
            base = names.get(o, str(o))
            return f"{base} ({kind})" if kind else base
        if op == Op.POINT:
            out.append((a[0], name(a[1], "point"), (a[2] - COORD_BIAS, a[3] - COORD_BIAS), op))
        elif op == Op.TARGET:
            out.append((a[0], name(a[1], "unit"), a[2], op))
        elif op == Op.IMMEDIATE:
            out.append((a[0], name(a[1]), None, op))
        elif op == Op.BUILD:
            out.append((a[0], f"build {rawcode(a[1])}", (a[2] - COORD_BIAS, a[3] - COORD_BIAS), op))
        elif op == Op.LEARN:
            out.append((a[0], f"learn {rawcode(a[1])}", None, op))
        elif op == Op.TARGET_DESTRUCTABLE:
            out.append((a[0], name(a[1], "tree"), a[2], op))
        elif op == Op.ITEM:
            out.append((a[0], f"use item {a[2]}", a[5] or None, op))
        elif op == Op.BUY:
            out.append((a[0], f"buy {rawcode(a[2])}", a[1], op))
        elif op in (Op.PICK_UP, Op.PICK_UP_AT):
            out.append((a[0], "pick up", None, op))
        elif op == Op.DROP_ITEM:
            out.append((a[0], f"drop item {a[1]}", a[4] or None, op))
        elif op == Op.NEUTRAL_POINT:
            out.append((a[0], name(a[2], "point"), (a[3] - COORD_BIAS, a[4] - COORD_BIAS), op))
    return out


def unit_kind(labels: Counter) -> str:
    """A unit's kind from the orders it got over the game."""
    if any(l.startswith(("train ", "research ")) or l in ("cancel", "setrally (point)", "setrally (unit)") for l in labels):
        return "building"
    if any(l.startswith(("learn ", "use item", "drop item")) or l == "pick up" for l in labels):
        return "hero"
    if any(l.startswith(("build ", "harvest", "repair", "smart (tree)", "returnresources")) for l in labels):
        return "worker"
    return "army"


def policy_name(who: str) -> str:
    """A side's policy in one spelling: "runs/fgself-236/averaged/swa-43-53.pt" and "fgself-236 avg swa-43-53" ->
    "fgself-236 avg"; a run's current weights -> the run's name."""
    who = re.sub(r"^.*runs/", "", str(who))
    m = re.match(r"([\w-]+)[ /]+(?:averaged/|avg )", who)
    if m:
        return f"{m.group(1)} avg"
    return who.split("/")[0]


def load_game(trace_path: Path, names: dict[int, str]) -> dict | None:
    """A replay's orders: the header's facts, each agent side's chosen labels a step, each unit's orders in order."""
    cmd_path = trace_path.with_name(trace_path.name.replace(".trace.json", ".commands.json"))
    try:
        trace = json.loads(trace_path.read_text())
        commands = json.loads(cmd_path.read_text())["commands"]
    except (OSError, ValueError, KeyError):
        return None
    sides = {}
    for s in trace.get("sides", []):
        m = re.search(r"\((human|orc|undead|nightelf)\)", s.get("name", ""))
        sides[int(s["player"])] = {"race": m.group(1) if m else "?", "kind": s.get("kind"), "name": s.get("name", "")}
    agents = [p for p, s in sides.items() if s["kind"] == "agent"]
    learner = next((p for p, s in sides.items() if s["name"].startswith("learner")), agents[0] if agents else None)
    if learner is None:
        return None
    other = next((s for p, s in sides.items() if p != learner), {"kind": "?", "name": ""})
    title = trace.get("title", "")
    opp = ("ai" if other["kind"] == "ai" else "itself" if "itself" in other["name"] else
           "exploiter" if "exploit" in other["name"] else "snapshot")
    # who plays each agent side: the learner's side the title's policy ("policy vN": the run's current weights, a
    # burst's burst.json "current"), the other side an evaluation's listed checkpoint, the exploiter or itself
    run_dir = trace_path.parent.parent
    current = run_dir.name
    try:
        current = json.loads((run_dir / "burst.json").read_text())["current"]
    except (OSError, ValueError, KeyError):
        pass
    for p, s in sides.items():
        if s["kind"] != "agent":
            continue
        if p == learner:
            m = re.search(r"· (.*?) · vs", title)
            who = m.group(1) if m else "?"
            who = current if who.startswith("policy v") else who
        else:
            m = re.search(r"eval-vs:(\S+)", title)
            who = m.group(1) if m else (current if "itself" in s["name"] else s["name"].split(" (")[0])
        s["policy"] = policy_name(who)
    chosen = {int(st["t"]): {int(p): c for p, c in (st.get("orders") or {}).items()} for st in trace["steps"]}
    steps: dict[int, list] = {}
    for key, ints in commands.items():
        ep, seq = key.split(":")
        if ep != "0":
            continue
        got = decode(ints, names)
        if got:
            steps[int(seq)] = got
    # each unit's side: the side whose chosen labels of the step hold the command's label, when only one does
    votes: dict[int, Counter] = defaultdict(Counter)
    for t, got in steps.items():
        ch = chosen.get(t, {})
        for unit, label, _, _ in got:
            if len(agents) == 1:
                votes[unit][agents[0]] += 1
                continue
            alt = label.replace("harvest (tree)", "smart (tree)")
            hit = [p for p in agents if label in ch.get(p, {}) or alt in ch.get(p, {})]
            if len(hit) == 1:
                votes[unit][hit[0]] += 1
    side_of = {u: v.most_common(1)[0][0] for u, v in votes.items()}
    units: dict[int, list] = defaultdict(list)
    for t in sorted(steps):
        for k, (unit, label, aim, op) in enumerate(steps[t]):
            units[unit].append((t, label, aim, op))
    return {"stem": trace_path.name.replace(".trace.json", ""), "run": trace_path.parent.parent.name, "title": title,
            "kind": trace.get("kind", "?"), "target": bool(trace.get("target")), "sides": sides, "learner": learner,
            "opp": opp, "steps": len(trace["steps"]), "chosen": chosen, "sent": steps, "units": dict(units),
            "side_of": side_of, "agents": agents}


def runs_of(events: list, same, gap: int, least: int) -> list[tuple[int, int]]:
    """Maximal runs [i, j) of a unit's events where same(i) holds for every i past the first and each is within `gap`
    steps of the one before; `least` events at least."""
    out, i = [], 0
    while i < len(events):
        j = i + 1
        while j < len(events) and events[j][0] - events[j - 1][0] <= gap and same(j):
            j += 1
        if j - i >= least:
            out.append((i, j))
        i = max(j, i + 1) if j - i >= least else i + 1
    return out


def alternating(events: list, gap: int, least: int = 4) -> list[tuple[int, int]]:
    """Maximal runs [i, j) of a unit's events whose labels go A, B, A, B (two different ones in turn), each within
    `gap` steps of the one before; `least` events at least."""
    out, i, n = [], 0, len(events)
    while i < n - 1:
        j = i + 1
        if events[j][0] - events[i][0] > gap or events[j][1] == events[i][1]:
            i += 1
            continue
        j += 1
        while j < n and events[j][0] - events[j - 1][0] <= gap and events[j][1] == events[j - 2][1]:
            j += 1
        if j - i >= least:
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def scan(game: dict, gap: int) -> dict:
    """One game's patterns: {pattern: Counter of (side, key) -> orders}."""
    pat = {k: Counter() for k in ("alternations", "repeats", "re-aims", "countermands", "bursts", "orders", "by label")}
    kinds = {}
    for unit, ev in game["units"].items():
        side = game["side_of"].get(unit)
        if side is None:
            continue
        kind = kinds[unit] = unit_kind(Counter(l for _, l, _, _ in ev))
        pat["orders"][(side, kind)] += len(ev)
        for _, l, _, _ in ev:
            pat["by label"][(side, f"{kind}: {l}")] += 1
        # alternations: A, B, A, B (labels), 4 or more
        for i, j in alternating(ev, gap):
            pat["alternations"][(side, f"{kind}: {' <-> '.join(sorted((ev[i][1], ev[i + 1][1])))}")] += j - i
        # repeats: the same label and aim again and again (3 or more: the orders past the first)
        for i, j in runs_of(ev, lambda k: ev[k][1] == ev[k - 1][1] and ev[k][2] == ev[k - 1][2], gap, 3):
            pat["repeats"][(side, f"{kind}: {ev[i][1]}")] += j - i - 1
        # re-aims: the same label, another aim, within 2 steps, 4 or more in a row
        for i, j in runs_of(ev, lambda k: ev[k][1] == ev[k - 1][1] and ev[k][2] != ev[k - 1][2], 2, 4):
            pat["re-aims"][(side, f"{kind}: {ev[i][1]}")] += j - i - 1
        # countermands: another label within `gap` steps
        for k in range(1, len(ev)):
            if ev[k][1] != ev[k - 1][1] and 0 < ev[k][0] - ev[k - 1][0] <= gap:
                pat["countermands"][(side, f"{kind}: {ev[k - 1][1]} -> {ev[k][1]}")] += 1
        # bursts: 20 orders within 40 steps (the orders in such windows, each counted once)
        ts = [e[0] for e in ev]
        hot, lo = set(), 0
        for hi in range(len(ts)):
            while ts[hi] - ts[lo] >= 40:
                lo += 1
            if hi - lo + 1 >= 20:
                hot.update(range(lo, hi + 1))
        if hot:
            mix = Counter(ev[k][1] for k in hot).most_common(2)
            pat["bursts"][(side, f"{kind}: " + " + ".join(l for l, _ in mix))] += len(hot)
    # chosen and not sent, per label (a side's)
    sent = Counter()
    for t, got in game["sent"].items():
        for unit, label, _, _ in got:
            s = game["side_of"].get(unit)
            if s is not None:
                sent[(s, label)] += 1
    chose = Counter()
    for t, ch in game["chosen"].items():
        for s, c in ch.items():
            for label, n in c.items():
                chose[(s, label)] += n
    pat["chosen"] = chose
    pat["sent"] = sent
    game["kinds"] = kinds
    return pat


def split_key(game: dict, side: int) -> tuple:
    s = game["sides"][side]
    return (s["race"], "target" if game["target"] and side == game["learner"] else "no target", s.get("policy", "?"))


def report(games: list[dict], gap: int, top: int, policy: str = "") -> None:
    """The patterns ranked; `policy`: only the sides played by a policy whose name holds this."""
    total = {k: Counter() for k in ("alternations", "repeats", "re-aims", "countermands", "bursts")}
    by_split = {k: defaultdict(Counter) for k in total}
    orders, side_games, side_steps = Counter(), Counter(), Counter()
    by_label, chosen, sent = Counter(), Counter(), Counter()
    label_split = defaultdict(Counter)
    for g in games:
        pat = scan(g, gap)
        keep = {s for s in g["agents"] if not policy or policy in g["sides"][s].get("policy", "")}
        for s in keep:
            sk = split_key(g, s)
            side_games[sk] += 1
            side_steps[sk] += g["steps"]
        for name in total:
            for (s, key), n in pat[name].items():
                if s in keep:
                    total[name][key] += n
                    by_split[name][key][split_key(g, s)] += n
        for (s, kind), n in pat["orders"].items():
            if s in keep:
                orders[kind] += n
        for (s, key), n in pat["by label"].items():
            if s in keep:
                by_label[key] += n
                label_split[key][split_key(g, s)[0]] += n
        for (s, label), n in pat["chosen"].items():
            if s in keep:
                chosen[(g["sides"][s]["race"], label)] += n
        for (s, label), n in pat["sent"].items():
            if s in keep:
                sent[(g["sides"][s]["race"], label)] += n
    n_sides = sum(side_games.values())
    print(f"{len(games)} games, {n_sides} agent sides, {sum(side_steps.values())} side-steps; orders sent by unit kind: "
          + ", ".join(f"{k} {n}" for k, n in orders.most_common()))
    print("sides by split (race, target, kind): " + "; ".join(f"{' / '.join(k)}: {n}" for k, n in sorted(side_games.items())))
    for name, what in (("alternations", "orders in A, B, A, B runs"), ("repeats", "orders re-sent with the same aim"),
                       ("re-aims", "the same order re-aimed a step or two apart"),
                       ("countermands", f"orders replaced by another kind within {gap} steps"),
                       ("bursts", "orders of units ordered every second step for 20 s")):
        print(f"\n== {name}: {what} ({sum(total[name].values())} in all, {sum(total[name].values()) / max(n_sides, 1):.1f} a side-game)")
        for key, n in total[name].most_common(top):
            base = by_label.get(key.split(" -> ")[0], 0) if name == "countermands" else by_label.get(key, 0)
            share = f" ({n / base:.0%} of its {base})" if base and name in ("countermands", "repeats", "re-aims") else ""
            races = Counter()
            targ = Counter()
            opp = Counter()
            for (race, tg, kd), m in by_split[name][key].items():
                races[race] += m
                targ[tg] += m
                opp[kd] += m
            per = lambda c, idx: " ".join(  # noqa: E731 (a side-game of that split's)
                f"{k}={m / max(sum(v for sk, v in side_games.items() if sk[idx] == k), 1):.1f}" for k, m in c.most_common(4))
            print(f"  {n:6d}{share}  {key}\n          a side-game: {per(races, 0)} | {per(targ, 1)} | {per(opp, 2)}")
    print("\n== not sent: chosen by the policy and not sent as a command (race, label: chosen, sent, share dropped)")
    gapc = Counter({k: chosen[k] - sent.get(k, 0) for k in chosen if chosen[k] - sent.get(k, 0) > 0})
    for (race, label), n in gapc.most_common(top):
        print(f"  {n:6d}  {race:<9} {label:<28} chosen {chosen[(race, label)]:6d}  sent {sent.get((race, label), 0):6d}  "
              f"dropped {n / chosen[(race, label)]:.0%}")
    extra = Counter({k: sent[k] - chosen.get(k, 0) for k in sent if sent[k] - chosen.get(k, 0) > 0})
    print("== sent and not chosen under that label (the agent's own forms: a revival, a tier's cancels, a macro's order)")
    for (race, label), n in extra.most_common(max(top // 2, 5)):
        print(f"  {n:6d}  {race:<9} {label:<28} chosen {chosen.get((race, label), 0):6d}  sent {sent[(race, label)]:6d}")


def show_unit(games: list[dict], stem: str, unit: int, gap: int) -> None:
    for g in games:
        if stem in g["stem"] and unit in g["units"]:
            scan(g, gap)
            side = g["side_of"].get(unit)
            print(f"{g['run']}/{g['stem']} unit {unit}: side {side} ({g['sides'].get(side, {}).get('race')}), "
                  f"{g['kinds'].get(unit)}, {len(g['units'][unit])} orders; {g['title']}")
            for t, label, aim, _ in g["units"][unit]:
                print(f"  t={t:5d} {label:<26} {aim if aim is not None else ''}")


def worst_units(games: list[dict], gap: int, top: int, policy: str = "") -> None:
    """The units with the most orders in alternations and repeats: where to look (--unit)."""
    rows = []
    for g in games:
        scan(g, gap)
        for unit, ev in g["units"].items():
            side = g["side_of"].get(unit)
            if side is None or (policy and policy not in g["sides"][side].get("policy", "")):
                continue
            rep = sum(j - i - 1 for i, j in runs_of(ev, lambda k: ev[k][1] == ev[k - 1][1] and ev[k][2] == ev[k - 1][2], gap, 3))
            alt = sum(j - i for i, j in alternating(ev, gap))
            if rep + alt:
                mix = ", ".join(f"{l} {n}" for l, n in Counter(l for _, l, _, _ in ev).most_common(3))
                rows.append((rep + alt, alt, rep, len(ev), g["run"], g["stem"], unit, g["sides"][side]["race"],
                             g["kinds"].get(unit), "target" if g["target"] and side == g["learner"] else "", mix,
                             g["sides"][side].get("policy", "?")))
    print("\n== the units with the most alternating and repeated orders (alternating, repeated, all its orders)")
    for tot, alt, rep, n, run, stem, unit, race, kind, tg, mix, who in sorted(rows, reverse=True)[:top]:
        print(f"  {alt:4d} {rep:4d} {n:5d}  {run}/{stem} unit {unit} {race} {kind} {tg} [{who}]: {mix}")


def episodes(run: Path, last: int, top: int) -> None:
    """The game's refusals and the agent's own unsent orders from a run's episode rows."""
    rows = []
    for name in ("episodes.jsonl", "episodes-remote.jsonl"):
        p = run / name
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    rows = [r for r in rows if "race" in r][-last:]
    if not rows:
        return
    print(f"\n== {run.name}: {len(rows)} episode rows")
    by_race = defaultdict(list)
    for r in rows:
        by_race[r["race"]].append(r)
    for race in RACES:
        rs = by_race.get(race, [])
        if not rs:
            continue
        n = len(rs)
        refused, sent_total = Counter(), sum(r.get("orders", 0) for r in rs)
        for r in rs:
            refused.update(r.get("refused") or {})
        ctx = defaultdict(Counter)
        for r in rs:
            for c in r.get("refused_ctx") or []:
                if len(c) >= 4:
                    tail = f" on {c[5]} {c[4]}" if len(c) >= 8 and isinstance(c[5], str) else ""
                    ctx[c[0]][f"{c[1]} cur={c[3]}{tail}"] += 1
        share = sum(r.get("refused_share", 0.0) for r in rs) / n
        print(f"  {race}: {n} games, {sent_total / n:.0f} orders a game, refused share {share:.2%}, "
              f"refused {sum(refused.values()) / n:.1f} a game; the agent's unsent a game: "
              + ", ".join(f"{k} {sum(r.get(k, 0) or 0 for r in rs) / n:.1f}" for k in COUNTERS
                          if any(r.get(k) for r in rs)))
        for key, m in refused.most_common(top):
            where = "; ".join(f"{k} x{v}" for k, v in ctx.get(key, Counter()).most_common(3))
            print(f"    {m / n:6.2f} a game  {key:<30} {where}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run directories (their replays/ and episode rows)")
    ap.add_argument("--gap", type=int, default=3, help="steps between two orders of a unit that still count as in a row")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--unit", nargs=2, metavar=("GAME", "UNIT"), help="one unit's orders step by step")
    ap.add_argument("--episodes", type=int, default=4000, help="the episode rows read a run (its last; 0: none)")
    ap.add_argument("--only", default="", help="replays whose title or stem holds this")
    ap.add_argument("--policy", default="", help="only the sides played by a policy whose name holds this "
                    "(\"fgself-236 avg\": an evaluation's listed checkpoint; a run's name: its current weights)")
    ap.add_argument("--no-replays", action="store_true")
    a = ap.parse_args()
    games = []
    for run in a.runs if not a.no_replays else ():
        try:
            names = {int(k): v for k, v in (json.loads((run / "vocab.json").read_text()).get("order_names") or {}).items()}
        except (OSError, ValueError):
            names = {}
        for tp in sorted((run / "replays").glob("*.trace.json")):
            g = load_game(tp, names)
            if g is not None and (not a.only or a.only in g["title"] or a.only in g["stem"]):
                games.append(g)
    if a.unit:
        show_unit(games, a.unit[0], int(a.unit[1]), a.gap)
        return 0
    if games:
        report(games, a.gap, a.top, a.policy)
        worst_units(games, a.gap, a.top, a.policy)
    for run in a.runs if a.episodes else ():
        episodes(run, a.episodes, a.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
