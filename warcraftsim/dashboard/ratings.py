"""Skill ratings of the policies from their evaluation games (the user's idea, 2026-10-20: the insane AI's win rate
is spent near 95%, so a rating from every game a policy played tells more): a Bradley-Terry model on the Elo scale,
the insane built-in AI in the real game as its anchor (ANCHOR_RATING).

The players: each listed checkpoint (`--eval-checkpoints`, "fgself-102/11841833"), each run's current policy by the
hour since the run started ("fgself-107 h2"; with a target strategy "+z"), and the anchor. The games: a run's
evaluation games against the insane AI in the real game, and its head-to-head games (`--eval-vs`: the current
policy against a listed checkpoint). Ties count half a win each. A checkpoint from a run without the neutral buildings'
wares, evaluated in one with them (selfplay --neutral) since they could be bought (SHOPS_SINCE), is a player of its own
("fgself-109/4933795 +shops"): its new buy orders went through untrained (its heroes walked off to buy items), and it
won 71% against insane there to 97% elsewhere.

The scan reads only what was added to the runs' episode files since the last one (their offsets in CACHE), so the
dashboard keeps the ratings current cheaply; the fit runs on the pairs' totals."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path

ANCHOR = "insane AI"
ANCHOR_RATING = 1500.0
SCALE = 400.0 / math.log(10.0)  # Elo points per unit of log-strength
PRIOR_GAMES = 1.0  # each player's virtual tie against the anchor: finite ratings for a few games
CACHE = ".ratings-cache-v2.json"  # (v2: the "+shops" players)
# the evaluation games could buy from the neutral buildings from here on (2026-10-21 14:27: the laptop's evaluation
# worker started with the harness that reaches a shop in the fog; before, every buy order was refused)
SHOPS_SINCE = 1792589220.0
_CKPT = re.compile(r"(fgself-[\w.-]+)/(?:checkpoints/)?0*(\d+)\.pt")


def checkpoint_name(path: str) -> str | None:
    """'runs/fgself-102/checkpoints/0000000011841833.pt' or 'fgself-102/0000000011841833.pt' -> 'fgself-102/11841833'."""
    m = _CKPT.search(path)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def players_of(row: dict, run: str, start: float, untrained=lambda run: False) -> tuple[str, str] | None:
    """An evaluation row's (player, opponent), or None when it is no rated game. `untrained(run)`: whether that
    checkpoint's run lacked the neutral wares the game is played with (its checkpoints then "+shops")."""
    if not row.get("eval"):
        return None
    key, opp = str(row.get("eval_key") or "current"), str(row.get("opponent") or "")
    late = float(row.get("time", 0.0)) >= SHOPS_SINCE

    def named(path: str) -> str | None:
        c = checkpoint_name(path)
        return c + " +shops" if c is not None and late and untrained(c.split("/")[0]) else c
    if key.startswith("vs "):
        b = named(key)
        if b is None:
            return None
    elif "ai-insane" in opp and "(real)" in opp:
        b = ANCHOR
    else:
        return None
    if key.startswith("current") or key.startswith("vs "):
        hour = max(0, int((float(row.get("time", start)) - start) // 3600))
        suffix = {"current+z": " +z", "current+zg": " +zg"}.get(key, "")
        a = f"{run} h{hour}{suffix}"
    else:
        a = named(key)
        if a is None:
            return None
    return (a, b) if a != b else None


class Ratings:
    """The pairs' totals from every run's evaluation games (scan) and the fit (table)."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.path = self.runs_dir / CACHE
        self.lock = threading.Lock()
        try:
            self.state = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.state = {}
        self.state.setdefault("files", {})  # file -> offset read to
        self.state.setdefault("pairs", {})  # "a\tb" (a < b) -> [a's wins, b's wins, ties]
        self.state.setdefault("meta", {})  # player -> {"run", "kind", "first", "last"}
        self._table: dict | None = None

    def scan(self) -> bool:
        """Read the runs' new evaluation rows -> whether anything changed."""
        changed = False
        for d in sorted(self.runs_dir.iterdir()):
            info = d / "run.json"
            if not d.is_dir() or not info.exists():
                continue
            files = [f for f in (d / "episodes.jsonl", d / "episodes-remote.jsonl") if f.exists()]
            if not files:
                continue
            start = None
            shops = self._neutral(d.name) > 0
            untrained = (lambda r: self._neutral(r) == 0) if shops else (lambda r: False)
            for f in files:
                key, size = str(f.relative_to(self.runs_dir)), f.stat().st_size
                at = self.state["files"].get(key, 0)
                if size < at:  # (rewritten: from the start)
                    at = 0
                if size == at:
                    continue
                if start is None:
                    try:
                        meta = json.loads(info.read_text())
                        start = float(meta.get("started") or meta.get("created") or 0.0)
                    except (OSError, ValueError):
                        start = 0.0
                with f.open("rb") as fh:
                    fh.seek(at)
                    data = fh.read(size - at)
                end = data.rfind(b"\n") + 1  # (complete lines only)
                for line in data[:end].split(b"\n"):
                    if b'"eval": true' not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    pair = players_of(row, d.name, start, untrained)
                    if pair is not None:
                        self._add(pair, float(row.get("outcome", 0.0)), float(row.get("time", 0.0)), d.name)
                        changed = True
                self.state["files"][key] = at + end
        if changed:
            self._table = None
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state))
            tmp.replace(self.path)
        return changed

    def _neutral(self, run: str) -> int:
        """A run's --neutral (0: none, or no run.json)."""
        cache = self.__dict__.setdefault("_neutrals", {})
        if run not in cache:
            try:
                cache[run] = int(json.loads((self.runs_dir / run / "run.json").read_text()).get("config", {}).get("neutral") or 0)
            except (OSError, ValueError, TypeError):
                cache[run] = 0
        return cache[run]

    def _add(self, pair: tuple[str, str], outcome: float, t: float, run: str) -> None:
        a, b = pair
        flip = b < a
        k = f"{b}\t{a}" if flip else f"{a}\t{b}"
        w = self.state["pairs"].setdefault(k, [0, 0, 0])
        won = outcome > 0 if not flip else outcome < 0
        lost = outcome < 0 if not flip else outcome > 0
        w[0 if won else 1 if lost else 2] += 1
        for p in pair:
            m = self.state["meta"].setdefault(p, {"run": p.split()[0].split("/")[0] if p != ANCHOR else "",
                                                  "kind": ("anchor" if p == ANCHOR else "+z" if p.endswith("+z") or
                                                           p.endswith("+zg") else "checkpoint" if "/" in p else "current"),
                                                  "first": t, "last": t})
            m["first"], m["last"] = min(m["first"], t), max(m["last"], t)

    def table(self) -> dict:
        """The fit: {"players": [{name, rating, se, games, score, run, kind, first, last}], "games", "pairs"}."""
        if self._table is not None:
            return self._table
        pairs = [(k.split("\t"), v) for k, v in self.state["pairs"].items()]
        names = sorted({p for (a, b), _ in pairs for p in (a, b)} | {ANCHOR})
        ix = {n: i for i, n in enumerate(names)}
        n = len(names)
        wins = [0.0] * n  # (ties half each; the prior's virtual tie against the anchor too)
        games: dict[tuple[int, int], float] = {}
        count = [0] * n
        for (a, b), (wa, wb, ties) in pairs:
            i, j = ix[a], ix[b]
            g = wa + wb + ties
            wins[i] += wa + 0.5 * ties
            wins[j] += wb + 0.5 * ties
            games[(i, j)] = games.get((i, j), 0.0) + g
            count[i] += g
            count[j] += g
        anchor = ix[ANCHOR]
        for i in range(n):
            if i != anchor:
                games[(i, anchor)] = games.get((i, anchor), 0.0) + PRIOR_GAMES
                wins[i] += 0.5 * PRIOR_GAMES
                wins[anchor] += 0.5 * PRIOR_GAMES
        nbr: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for (i, j), g in games.items():
            nbr[i].append((j, g))
            nbr[j].append((i, g))
        gamma = [1.0] * n
        for _ in range(500):  # (minorization-maximization, Hunter 2004; the anchor's strength held at 1)
            new = [wins[i] / max(sum(g / (gamma[i] + gamma[j]) for j, g in nbr[i]), 1e-12) if i != anchor else 1.0
                   for i in range(n)]
            delta = max(abs(math.log(max(x, 1e-12)) - math.log(max(y, 1e-12))) for x, y in zip(new, gamma))
            gamma = new
            if delta < 1e-7:
                break
        out = []
        for i, name in enumerate(names):
            info = sum(g * gamma[i] * gamma[j] / (gamma[i] + gamma[j]) ** 2 for j, g in nbr[i])
            m = self.state["meta"].get(name, {"run": "", "kind": "anchor", "first": 0, "last": 0})
            out.append({"name": name, "rating": round(ANCHOR_RATING + SCALE * math.log(max(gamma[i], 1e-12)), 1),
                        "se": None if i == anchor else round(SCALE / math.sqrt(max(info, 1e-12)), 1),
                        "games": count[i], "score": round((wins[i] - (0.5 * PRIOR_GAMES if i != anchor else 0.0))
                                                          / max(count[i], 1), 3), **m})
        out.sort(key=lambda r: -r["rating"])
        self._table = {"players": out, "games": sum(sum(v) for _, v in pairs), "pairs": len(pairs),
                       "anchor": ANCHOR, "anchor_rating": ANCHOR_RATING, "updated": time.time()}
        return self._table


def background(r: Ratings, every: float = 300.0) -> None:
    """Scan now and every `every` seconds (a daemon thread of the dashboard)."""
    def loop():
        while True:
            try:
                with r.lock:
                    r.scan()
            except Exception as e:  # noqa: BLE001 (the dashboard must stay up)
                print(f"ratings: {type(e).__name__}: {e}", flush=True)
            time.sleep(every)
    threading.Thread(target=loop, daemon=True).start()
