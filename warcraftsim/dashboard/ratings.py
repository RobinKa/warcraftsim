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
dashboard keeps the ratings current cheaply; the fit runs on the pairs' totals.

The same pass keeps each policy's record by the folder that played the game and by opponent ("played"), its games
against the AI by race ("races") and each folder's rated games ("dirs"): what a run's Checkpoints section shows of one
policy (policies.py). After each scan that changed something `view` is made anew: the table and those records as plain
data of their own, read by the pages without the scan's lock (the first scan of every run's games: half a minute)."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path

ANCHOR = "insane AI"
# the insane AI's games against a policy at the same double income (an agent slot took the advisor's difficulty, insane,
# until 2026-10-26: docs/reference/engine-notes.md): another opponent than the anchor, which plays a person's income
EQUAL = "insane AI, equal income"
ANCHOR_RATING = 1500.0
SCALE = 400.0 / math.log(10.0)  # Elo points per unit of log-strength
PRIOR_GAMES = 1.0  # each player's virtual tie against the anchor: finite ratings for a few games
CACHE = ".ratings-cache-v10.json"  # (v2: the "+shops" players; v3: Echo Isles runs only, hours from the creation; v4: the
# pairs' games by the hour they were played; v5: games at the doubled income against EQUAL; v6: no games of a policy
# with the composition kinds' inputs played by a run without them, blind). Every game counts alike, however old (the user, 2026-10-25: a decay could
# lose a good past policy); the top rated ones are played anew under today's rules by the evaluation worker's gauntlet
# the evaluation games could buy from the neutral buildings from here on (2026-10-21 14:27: the laptop's evaluation
# worker started with the harness that reaches a shop in the fog; before, every buy order was refused)
SHOPS_SINCE = 1792589220.0
_CKPT = re.compile(r"(fgself-[\w.-]+)/(?:checkpoints/)?0*(\d+)\.pt")
# (v7, 2026-11-20: a weight average of a run's checkpoints, scripts/average_checkpoints.py, is a player of its own
# ("fgself-236 avg swa-43-53"), and an evaluation burst's current policy is the checkpoint its burst.json names, not
# "burst-237 h2": the same policy kept one rating across bursts, the user's "climb the rating")
_AVG = re.compile(r"(fgself-[\w.-]+)/averaged/([\w.-]+)\.pt")
# (v10, 2026-11-21: each policy's games by folder and opponent, by race against the AI, the folders' games; and a
# head-to-head opponent that is a weight average is rated: selfplay tags the game "vs fgself-236/swa-43-53.pt", the
# run and the file's name without its folder, which read as no checkpoint, so 8,366 such games had been dropped)
_VS_AVG = re.compile(r"vs (fgself-[\w.-]+)/([\w.-]+)\.pt")
_SOUP = re.compile(r"(?:soups|fullgame)/(soup[\w.-]*)\.pt")  # (a model soup, the mean of several runs' averages:
# scripts/soup.py, runs/fullgame/soups/; "soup avg <name>", an average of its own)
_HANDICAP = re.compile(r"\(real, agent -(\d+)%\)")  # (selfplay --eval-handicap's opponent name)
# a policy's name in parts: "fgself-236/53275744" (a checkpoint), "fgself-236 avg swa-43-53" (a weight average),
# "fgself-238 h3" (a run's policy in its hour 3), each with " +shops", " +z", " +zg" or " +human" after it or not
_POLICY = re.compile(r"(?P<run>[\w.-]+?)(?:/(?P<steps>\d+)| avg (?P<avg>[\w.-]+)| h(?P<hour>\d+))(?P<variant> \+\w+)?")
RACES = ("human", "orc", "undead", "nightelf")
BURST = "burst.json"  # (a burst's {"current": the checkpoint it plays as "current", ...}: scripts/burst_prep.py)
# a checkpoint the evaluation worker's gauntlet plays for a run's policy by the hour ("fgself-102/11882922" for
# "fgself-102 h2": the run's last checkpoint saved in that hour), written by scripts/remote_eval.py: its games count
# for that policy, so the top rated ones move as they are played anew (the user, 2026-10-25: the gauntlet's games had
# made a player of their own, and the stale hourly one stayed first)
ALIASES = ".gauntlet-aliases.json"


def checkpoint_name(path: str) -> str | None:
    """'runs/fgself-102/checkpoints/0000000011841833.pt' or 'fgself-102/0000000011841833.pt' -> 'fgself-102/11841833'."""
    a = _AVG.search(path)
    if a:
        return f"{a.group(1)} avg {a.group(2)}"
    u = _SOUP.search(path)
    if u:
        return f"soup avg {u.group(1)}"  # (also a burst's head-to-head tag of one: "vs fullgame/soup-...pt")
    m = _CKPT.search(path)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    v = _VS_AVG.fullmatch(path)  # (a head-to-head tag of a weight average: its folder left out)
    return f"{v.group(1)} avg {v.group(2)}" if v else None


def _new_lines(f: Path, at: int, size: int, step: int = 8 << 20):
    """A file's complete lines from offset `at` up to `size`, a few megabytes at a time: (lines, the offset read to)
    for each piece (a run's episodes are up to 300 MB: the first scan of every run had each whole file in memory, twice,
    beside the training)."""
    with f.open("rb") as fh:
        fh.seek(at)
        rest = b""
        while at + len(rest) < size:
            chunk = fh.read(min(step, size - at - len(rest)))
            if not chunk:
                break
            data = rest + chunk
            end = data.rfind(b"\n") + 1
            rest, at = data[end:], at + end
            if end:
                yield data[:end].split(b"\n"), at


def _inverse_diagonal(A):
    """The diagonal of a symmetric positive definite matrix's inverse (each rating's variance from the information
    matrix), by Gauss-Jordan elimination in numpy's own elementwise operations; None: not positive definite.
    No LAPACK: numpy's pinv of the ratings' 900 x 900 matrix took 6 s and ~150 CPU-seconds on this machine, its
    threads spinning beside the training (0.2 s on one thread; measured 2026-11-21), at every fit."""
    import numpy as np
    A = np.array(A, dtype=np.float64)
    n = len(A)
    tmp = np.empty_like(A)
    for k in range(n):
        p = A[k, k]
        if not p > 0.0 or not np.isfinite(p):
            return None
        col, row = A[:, k].copy(), A[k] / p
        np.multiply(col[:, None], row[None, :], out=tmp)
        A -= tmp
        A[k], A[:, k] = row, -col / p
        A[k, k] = 1.0 / p
    return np.diag(A).copy()


def kind_of(name: str) -> str:
    """A player's kind: "anchor", "ai" (the insane AI at another income), "heuristic", "+z" (with a target strategy),
    "average" (a weight average), "checkpoint" or "current" (a run's policy by the hour)."""
    if name == ANCHOR:
        return "anchor"
    if name == EQUAL or name.startswith(ANCHOR + " vs -"):
        return "ai"
    if name.startswith("heuristic:"):
        return "heuristic"
    if name.endswith("+z") or name.endswith("+zg"):
        return "+z"
    return "average" if " avg " in name else "checkpoint" if "/" in name else "current"


def policy_of(name: str) -> dict | None:
    """A policy's name in parts: {"run", "id" (within its run: the steps, the average's name, "h3"), "kind"
    ("checkpoint", "average", "hour"), "variant" ("", "+z", "+shops", ...)}; None for the AI's and the heuristic's."""
    m = _POLICY.fullmatch(name)
    if m is None or kind_of(name) in ("anchor", "ai", "heuristic"):
        return None
    pid, kind = ((m["steps"], "checkpoint") if m["steps"] is not None else (m["avg"], "average") if m["avg"] is not None
                 else (f"h{m['hour']}", "hour"))
    return {"run": m["run"], "id": pid, "kind": kind, "variant": (m["variant"] or "").strip()}


def players_of(row: dict, run: str, start: float, untrained=lambda run: False,
               aliases: dict | None = None, slot: str = "normal", current: str | None = None) -> tuple[str, str] | None:
    """An evaluation row's (player, opponent), or None when it is no rated game. `untrained(run)`: whether that
    checkpoint's run lacked the neutral wares the game is played with (its checkpoints then "+shops"). `aliases`:
    checkpoint -> the run's policy by the hour it stands for (the gauntlet's; ALIASES). `slot`: the run's agent slots'
    difficulty for rows without their own ("slot_difficulty"; --opd-difficulty then)."""
    if not row.get("eval"):
        return None
    key, opp = str(row.get("eval_key") or "current"), str(row.get("opponent") or "")
    late = float(row.get("time", 0.0)) >= SHOPS_SINCE

    def named(path: str) -> str | None:
        c = checkpoint_name(path)
        if c is not None and aliases and c in aliases:
            return aliases[c]
        return c + " +shops" if c is not None and late and untrained(re.split(r"[ /]", c)[0]) else c  # (the run's
        # name: "fgself-236/53275744" or "fgself-236 avg swa-43-53")
    if key.startswith("vs heuristic:"):  # (--eval-heuristic: the hand-written player, a player of its own that does not
        b = key[3:]                       # change: fullgame/heuristic.py)
    elif key.startswith("vs "):
        b = named(key)
        if b is None:
            return None
    elif "ai-insane" in opp and (m := _HANDICAP.search(opp)):  # (the handicap ladder: the insane AI against a policy
        b = f"{ANCHOR} vs -{m.group(1)}%"                         # that gathers that much less: a player of its own)
        key = key.split(" @-")[0]
    elif "ai-insane" in opp and "(real)" in opp:
        doubled = (row.get("slot_difficulty") or ("normal" if key == "current+human" else slot)) == "insane"
        b = EQUAL if doubled else ANCHOR
    else:
        return None
    if current is not None and (key == "current" or key.startswith("vs ")):  # (a burst's: the checkpoint it plays)
        a = current
    elif key.startswith("current") or key.startswith("vs "):
        hour = max(0, int((float(row.get("time", start)) - start) // 3600))
        suffix = {"current+z": " +z", "current+zg": " +zg", "current+human": " +human"}.get(key, "")  # (+human: from a slot at a
        # person's income against the insane AI's double)
        a = f"{run} h{hour}{suffix}"
    else:
        a = named(key)
        if a is None:
            return None
    return (a, b) if a != b else None


class Ratings:
    """The pairs' totals from every run's evaluation games (scan), the fit (table) and what the pages read (view)."""

    def __init__(self, runs_dir: Path, cache: Path | str | None = None):
        self.runs_dir = Path(runs_dir)
        self.path = Path(cache) if cache else self.runs_dir / CACHE  # (`cache`: the file elsewhere, e.g. a second
        self.lock = threading.Lock()                                  # dashboard reading the same runs)
        try:
            self.state = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.state = {}
        self._defaults()
        self._table: dict | None = None
        self._infos: dict[str, dict] = {}  # folder -> what the scan needs of its run.json, by the file's time (_facts)
        self.links: dict = {}  # who started from, is anchored on or bursts a policy (_links)
        self.view: dict | None = None  # the table and the policies' records for the pages (_view); None: no scan yet
        self.ready = threading.Event()  # set after the first scan (background)

    def _defaults(self) -> None:
        self.state.setdefault("files", {})  # file -> offset read to
        self.state.setdefault("pairs", {})  # "a\tb\thour" (a < b, the hour since the epoch) -> [a's wins, b's wins, ties]
        self.state.setdefault("meta", {})  # player -> {"run", "kind", "first", "last"}
        self.state.setdefault("played", {})  # policy -> the folder that played -> opponent -> [won, tied, lost]
        self.state.setdefault("races", {})  # policy -> an AI opponent -> "its race/the AI's" -> [won, tied, lost]
        self.state.setdefault("dirs", {})  # folder -> {"games", "first", "last"} of its rated games

    def _facts(self, d: Path) -> dict | None:
        """What the scan needs of a folder's run.json, read again only when the file changed: whether its evaluation
        games are the real Echo Isles (the anchor's games: the duel maps' insane AI is another opponent, and one anchor
        over both mixed them; audit 4), when the run was created, the policy it started from and its KL anchor, and
        the burst it is (BURST). None: no run."""
        try:
            stamp = ((d / "run.json").stat().st_mtime_ns,
                     (d / BURST).stat().st_mtime_ns if (d / BURST).exists() else 0)
        except OSError:
            self._infos.pop(d.name, None)
            return None
        got = self._infos.get(d.name)
        if got is None or got["stamp"] != stamp:
            try:
                meta = json.loads((d / "run.json").read_text())
            except (OSError, ValueError):
                meta = {}
            c, inputs = meta.get("config") or {}, meta.get("inputs") or {}
            got = {"stamp": stamp, "echo": "EchoIsles" in str(c.get("eval_map") or c.get("map") or ""),
                   # (a resume resets "started": every game before it had clamped into h0; audit 4)
                   "created": float(meta.get("created") or meta.get("started") or 0.0),
                   "start": self._named(meta.get("init_from") or inputs.get("start_weights")),
                   "anchor": self._named(inputs.get("init") or c.get("init")), "burst": None}
            if stamp[1]:
                try:  # (an evaluation burst: its current policy by the checkpoint it plays, its listed ones)
                    b = json.loads((d / BURST).read_text())
                    got["burst"] = {"policy": checkpoint_name(str(b.get("current") or "")),
                                    "evals": [n for n in map(checkpoint_name, map(str, b.get("evals") or [])) if n]}
                    got["start"] = got["anchor"] = None  # (its run.json is a copy of that policy's run's: no fork)
                except (OSError, ValueError, AttributeError):
                    pass
            self._infos[d.name] = got
        return got

    def _named(self, path) -> str | None:
        """A checkpoint path's policy; a derived checkpoint's (grown, a variant: its .json sidecar's "from") is the one
        it came from."""
        path = str(path or "")
        for _ in range(5):
            name = checkpoint_name(path) if path else None
            if name or not path.endswith(".pt"):
                return name
            p = Path(path)
            try:
                path = str(json.loads((p if p.is_absolute() else self.runs_dir.parent / p).with_suffix(".json")
                                      .read_text()).get("from") or "")
            except (OSError, ValueError, AttributeError):
                return None
        return None

    def _links(self) -> dict:
        """policy -> the runs started from it ("forks"), anchored on it ("anchors"), the bursts whose current policy it
        is ("bursts") and those that list it ("listed"); and folder -> its burst ("burst")."""
        out: dict = {"forks": {}, "anchors": {}, "bursts": {}, "listed": {}, "burst": {}}
        for name, f in sorted(self._infos.items()):
            if f["burst"]:
                out["burst"][name] = f["burst"]
                if f["burst"]["policy"]:
                    out["bursts"].setdefault(f["burst"]["policy"], []).append(name)
                for e in f["burst"]["evals"]:
                    out["listed"].setdefault(e, []).append(name)
                continue
            if f["start"] and (policy_of(f["start"]) or {}).get("run") != name:
                out["forks"].setdefault(f["start"], []).append(name)
            if f["anchor"] and f["anchor"] != f["start"]:
                out["anchors"].setdefault(f["anchor"], []).append(name)
        return out

    def scan(self) -> bool:
        """Read the runs' new evaluation rows -> whether anything changed. New gauntlet aliases: every row read again
        (its earlier games had counted for the checkpoint's own name)."""
        changed = False
        try:
            aliases = json.loads((self.runs_dir / ALIASES).read_text())
        except (OSError, ValueError):
            aliases = {}
        if aliases != self.state.get("aliases", {}):
            self.state = {"aliases": aliases}
            self._defaults()
            changed = True
        seen = set()
        for d in sorted(self.runs_dir.iterdir()):
            facts = self._facts(d) if d.is_dir() else None
            if facts is None:
                continue
            seen.add(d.name)
            files = [f for f in (d / "episodes.jsonl", d / "episodes-remote.jsonl") if f.exists()]
            if not files or not facts["echo"]:
                continue
            start = facts["created"]
            current = (facts["burst"] or {}).get("policy")  # (a burst's current policy: the checkpoint it plays)
            shops = self._neutral(d.name) > 0
            untrained = (lambda r: self._neutral(r) == 0) if shops else (lambda r: False)
            for f in files:
                key, size = str(f.relative_to(self.runs_dir)), f.stat().st_size
                at = self.state["files"].get(key, 0)
                if size < at:  # (rewritten: from the start)
                    at = 0
                if size == at:
                    continue
                for lines, at in _new_lines(f, at, size):
                    for line in lines:
                        if b'"eval": true' not in line:
                            continue
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        key_ = str(row.get("eval_key") or "")
                        m_ = _CKPT.search(key_) if key_.startswith("vs ") else None
                        if m_ is not None and self._comp(m_.group(1)) and not self._comp(d.name):
                            continue  # (a policy reading the composition kinds' inputs, played by a run that gives none:
                            # its strategy block shifted, it played blind; fgself-154's lost 60 of 60 to fgself-155)
                        pair = players_of(row, d.name, start, untrained, aliases, self._slot(d.name), current)
                        if pair is not None:
                            self._add(pair, float(row.get("outcome", 0.0)), float(row.get("time", 0.0)), d.name,
                                      f"{row.get('race') or '?'}/{row.get('opponent_race') or '?'}")
                            changed = True
                    self.state["files"][key] = at
                    time.sleep(0.001)  # (a pause for the dashboard's requests: this thread holds the GIL while parsing)
        for gone in set(self._infos) - seen:
            del self._infos[gone]
        links = self._links()
        if changed:
            self._table = None
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state))
            tmp.replace(self.path)
        if changed or links != self.links or self.view is None:  # (a new fork or burst of a policy shows too)
            self.links = links
            self.view = self._view()
        return changed

    def _neutral(self, run: str) -> int:
        """A run's --neutral (0: none, or no run.json; a soup's: its runs', all with the wares)."""
        cache = self.__dict__.setdefault("_neutrals", {})
        if run == "soup":  # (scripts/soup.py's "soup avg <name>": the mean of runs that bought from the shops; with
            return 2       # 0 its games as an evaluation were a "+shops" player of their own)
        if run not in cache:
            try:
                cache[run] = int(json.loads((self.runs_dir / run / "run.json").read_text()).get("config", {}).get("neutral") or 0)
            except (OSError, ValueError, TypeError):
                cache[run] = 0
        return cache[run]

    def _comp(self, run: str) -> bool:
        """Whether a run's policies read the composition kinds (--z-comp)."""
        cache = self.__dict__.setdefault("_comps", {})
        if run not in cache:
            try:
                cache[run] = float(json.loads((self.runs_dir / run / "run.json").read_text()).get("config", {})
                                   .get("z_comp") or 0) > 0
            except (OSError, ValueError, TypeError):
                cache[run] = False
        return cache[run]

    def _slot(self, run: str) -> str:
        """A run's agent slots' difficulty before its rows said (the advisor's: --opd-difficulty)."""
        cache = self.__dict__.setdefault("_slots", {})
        if run not in cache:
            try:
                cache[run] = str(json.loads((self.runs_dir / run / "run.json").read_text()).get("config", {})
                                 .get("opd_difficulty") or "normal")
            except (OSError, ValueError):
                cache[run] = "normal"
        return cache[run]

    def _add(self, pair: tuple[str, str], outcome: float, t: float, run: str, races: str = "") -> None:
        """One rated game: `pair` = (the policy, its opponent), the outcome the policy's; `run`: the folder whose
        rows hold it; `races`: "the policy's race/its opponent's"."""
        a, b = pair
        flip = b < a
        k = (f"{b}\t{a}" if flip else f"{a}\t{b}") + f"\t{int(t // 3600)}"
        w = self.state["pairs"].setdefault(k, [0, 0, 0])
        won = outcome > 0 if not flip else outcome < 0
        lost = outcome < 0 if not flip else outcome > 0
        w[0 if won else 1 if lost else 2] += 1
        for p in pair:
            m = self.state["meta"].setdefault(p, {"run": ("" if kind_of(p) in ("anchor", "ai", "heuristic")
                                                          else p.split()[0].split("/")[0]),
                                                  "kind": kind_of(p), "first": t, "last": t})
            m["first"], m["last"] = min(m["first"], t), max(m["last"], t)
        # each policy's record against the other in this folder's games, [won, tied, lost] (the AI's own: not kept)
        res = 0 if outcome > 0 else 2 if outcome < 0 else 1
        for p, q, i in ((a, b, res), (b, a, 2 - res)):
            if kind_of(p) not in ("anchor", "ai", "heuristic"):
                self.state["played"].setdefault(p, {}).setdefault(run, {}).setdefault(q, [0, 0, 0])[i] += 1
        if races and kind_of(b) in ("anchor", "ai"):
            self.state["races"].setdefault(a, {}).setdefault(b, {}).setdefault(races, [0, 0, 0])[res] += 1
        d = self.state["dirs"].setdefault(run, {"games": 0, "first": t, "last": t})
        d["games"], d["first"], d["last"] = d["games"] + 1, min(d["first"], t), max(d["last"], t)

    def _view(self) -> dict:
        """What the pages read, made after a scan (plain data of its own: read without the lock): {"table": the fit,
        "runs": run -> policy id -> {"id", "kind", "name", "rated": [each variant it was rated as: its rating, its
        record against each opponent, the folders that played it], "forked_by", "anchor_of", "bursts", "listed_in"},
        "bursts": folder -> {"policy", "evals", "games", "first", "last", "rows": each policy's records there}}."""
        table = self.table()
        rated = {p["name"]: p for p in table["players"]}
        runs: dict[str, dict[str, dict]] = {}
        bursts = {name: {**b, **(self.state["dirs"].get(name) or {"games": 0}), "rows": []}
                  for name, b in self.links.get("burst", {}).items()}

        def entry(pol: dict) -> dict:
            return runs.setdefault(pol["run"], {}).setdefault(pol["id"], {
                "id": pol["id"], "kind": pol["kind"], "rated": [],
                "name": {"checkpoint": f"{pol['run']}/{pol['id']}", "average": f"{pol['run']} avg {pol['id']}"}
                .get(pol["kind"], f"{pol['run']} {pol['id']}")})

        def opponent(o: str, rec: list[int], races: dict | None = None) -> dict:
            row = {"name": o, "kind": kind_of(o), "record": list(rec), "rating": (rated.get(o) or {}).get("rating")}
            op = policy_of(o)
            if op:
                row.update(run=op["run"], policy=op["id"])
            if races:
                row["races"] = {a: {b: list(races.get(f"{a}/{b}") or (0, 0, 0)) for b in RACES} for a in RACES}
            return row
        order = {"anchor": 0, "ai": 1, "heuristic": 2}
        for name, by_dir in self.state["played"].items():
            pol = policy_of(name)
            if pol is None:
                continue
            total: dict[str, list[int]] = {}
            where = []
            for folder, recs in by_dir.items():
                for o, rec in recs.items():
                    t = total.setdefault(o, [0, 0, 0])
                    for i in range(3):
                        t[i] += rec[i]
                di = self.state["dirs"].get(folder) or {}
                where.append({"run": folder, "games": sum(map(sum, recs.values())), "burst": folder in bursts,
                              "first": di.get("first"), "last": di.get("last")})
                if folder in bursts:
                    bursts[folder]["rows"].append({
                        "name": name, "run": pol["run"], "policy": pol["id"], "kind": pol["kind"],
                        "rating": (rated.get(name) or {}).get("rating"), "se": (rated.get(name) or {}).get("se"),
                        "opponents": sorted((opponent(o, rec) for o, rec in recs.items()),
                                            key=lambda r: (order.get(r["kind"], 3), r["name"]))})
            races = self.state["races"].get(name) or {}
            p = rated.get(name) or {}
            entry(pol)["rated"].append({
                "name": name, "variant": pol["variant"], **{k: p.get(k) for k in ("rating", "se", "games", "score", "first", "last")},
                "opponents": sorted((opponent(o, rec, races.get(o)) for o, rec in total.items()),
                                    key=lambda r: (order.get(r["kind"], 3), -sum(r["record"]), r["name"])),
                "played_by": sorted(where, key=lambda x: -(x["last"] or 0))})
        for key, field in (("forks", "forked_by"), ("anchors", "anchor_of"), ("bursts", "bursts"), ("listed", "listed_in")):
            for name, names in self.links.get(key, {}).items():
                pol = policy_of(name)
                if pol is not None:
                    entry(pol)[field] = sorted(names)
        for policies in runs.values():
            for e in policies.values():
                e["rated"].sort(key=lambda r: r["variant"])
        for b in bursts.values():  # (the burst's own policy first, then by rating)
            b["rows"].sort(key=lambda r: (r["name"] != b.get("policy"), -(r["rating"] or 0)))
        aliases: dict[str, list[str]] = {}  # a run's policy by the hour -> the checkpoints the gauntlet plays for it
        for ck, hour in (self.state.get("aliases") or {}).items():
            aliases.setdefault(hour, []).append(ck)
        return {"table": table, "runs": runs, "bursts": bursts, "aliases": aliases, "updated": table["updated"]}

    def table(self) -> dict:
        """The fit: {"players": [{name, rating, se, games, score, run, kind, first, last}], "games", "pairs"}."""
        if self._table is not None:
            return self._table
        merged: dict[tuple[str, str], list[float]] = {}
        for k, (wa, wb, ties) in self.state["pairs"].items():
            a, b, _ = k.split("\t")
            m = merged.setdefault((a, b), [0, 0, 0])
            m[0], m[1], m[2] = m[0] + wa, m[1] + wb, m[2] + ties
        pairs = [((a, b), v) for (a, b), v in merged.items()]
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
        # each rating's standard error from the whole information matrix, the anchor held (each player's own games
        # alone had read them too small: fgself-130 4.5 against 8.5; audit 11)
        import numpy as np
        H = np.zeros((n, n))
        for (i, j), g in games.items():
            w = g * gamma[i] * gamma[j] / (gamma[i] + gamma[j]) ** 2
            H[i, i] += w
            H[j, j] += w
            H[i, j] -= w
            H[j, i] -= w
        keep = [i for i in range(n) if i != anchor]
        diag = _inverse_diagonal(H[np.ix_(keep, keep)] + 1e-9 * np.eye(len(keep)))
        var = dict(zip(keep, diag.tolist())) if diag is not None else {}
        out = []
        for i, name in enumerate(names):
            info = sum(g * gamma[i] * gamma[j] / (gamma[i] + gamma[j]) ** 2 for j, g in nbr[i])
            v = var.get(i, 1.0 / max(info, 1e-12))
            m = self.state["meta"].get(name, {"run": "", "kind": "anchor", "first": 0, "last": 0})
            if name.startswith(ANCHOR + " vs -"):  # (a handicap rung, "insane AI vs -50%": the AI's, not a run's policy;
                m = {**m, "run": "", "kind": "ai"}  # the page had drawn it as a run's current policy, in green)
            pol = policy_of(name)
            if pol is not None:  # (its place in its run's Checkpoints section: what a click on it opens)
                m = {**m, "run": pol["run"], "policy": pol["id"]}
            out.append({"name": name, "rating": round(ANCHOR_RATING + SCALE * math.log(max(gamma[i], 1e-12)), 1),
                        "se": None if i == anchor else round(SCALE * math.sqrt(max(v, 1e-12)), 1),
                        "games": count[i], "score": round((wins[i] - 0.5 * PRIOR_GAMES * (1 if i != anchor else n - 1))
                                                          / max(count[i], 1), 3), **m})  # (the prior's ties out)
        out.sort(key=lambda r: -r["rating"])
        self._table = {"players": out, "games": sum(sum(v) for _, v in pairs), "pairs": len(pairs),
                       "anchor": ANCHOR, "anchor_rating": ANCHOR_RATING, "updated": time.time()}
        return self._table


def background(r: Ratings, every: float = 300.0) -> None:
    """Scan now and every `every` seconds (a daemon thread of the dashboard); `r.ready` is set after the first."""
    def loop():
        while True:
            try:
                with r.lock:
                    r.scan()
            except Exception as e:  # noqa: BLE001 (the dashboard must stay up)
                print(f"ratings: {type(e).__name__}: {e}", flush=True)
            r.ready.set()
            time.sleep(every)
    threading.Thread(target=loop, daemon=True).start()
