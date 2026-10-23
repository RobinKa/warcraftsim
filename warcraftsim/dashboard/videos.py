"""Game videos with what happened in them (the dashboard's Videos page, the user's idea, 2026-10-23: "search for
neutral heroes etc."; and the Replays tab's panel beside a video: its target without watching it).

For each video of a self-play run (media.jsonl rows of kind "video": videos/game-<time>-a<w>.mp4) two files tell:
its replay's trace (replays/<stem>.trace.json: the learner's target strategy and when it made each item, both sides'
heroes and their levels) and its episode row (the learner's and its opponent's production: tiers, expansions, camps
crept, items picked up and bought). Summed up once into videos/<stem>.info.json beside it (a trace is ~1.5 MB, the
episode files 1.6 GB in all): a thread fills those in, the newest runs first, and picks up new videos every minute.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

INFO_VERSION = 2  # (2: the target's end, the game's length)
TIER2 = {"hkee", "ostr", "unp1", "etoa"}
TIER3 = {"hcas", "ofrt", "unp2", "etoe"}
RACES = ("human", "orc", "undead", "nightelf")
_TIME = re.compile(r'"time": (\d+)')


def stem_of(file: str) -> str | None:
    """'videos/game-1792782843-a1.mp4' -> 'game-1792782843-a1' (another kind of file: None)."""
    m = re.fullmatch(r"videos/(game-\d+-a\d+)\.mp4", file)
    return m.group(1) if m else None


def _names() -> dict[str, str]:
    try:
        from ..data.objects import unit_names, upgrade_names
        return {**unit_names(), **upgrade_names()}
    except Exception:  # noqa: BLE001 (no game files: the codes)
        return {}


def is_hero(code: str) -> bool:
    return len(code) == 4 and code[0].isupper() and code[0] != "R" and code[1:].islower()


def _tier(built: dict) -> int:
    b = set(built or {})
    return 3 if b & TIER3 else 2 if b & TIER2 else 1


def _side(race: str | None, heroes: dict[str, int], prod: dict | None, names: dict) -> dict:
    """One side's summary: its heroes (code, name, highest level), tier, expansions, camps crept, army and workers."""
    prod = prod or {}
    for code in (prod.get("trained") or {}):  # (heroes the trace missed, or no trace)
        if is_hero(code):
            heroes.setdefault(code, 0)
    out = {"race": race, "heroes": [[c, names.get(c, c), lvl] for c, lvl in heroes.items()]}
    if prod:
        out.update(tier=_tier(prod.get("built")), expanded=int(prod.get("expansions") or 0),
                   halls_home=int(prod.get("halls_home") or 0), creeps=int(prod.get("creeps") or 0),
                   army=int(prod.get("army") or 0), workers=int(prod.get("workers") or 0))
    return out


def summarize(m: dict, trace: dict | None, ep: dict | None, names: dict | None = None) -> dict:
    """A video's info: both sides ("me": the learner, "opp"), the learner's target (its items, when due and when made;
    the share made of those due by the game's end) and tags to search by."""
    names = _names() if names is None else names
    title = m.get("title") or ""
    words = title.split()
    race = (ep or {}).get("race") or (words[0] if words and words[0] in RACES else None)
    opp_race = (ep or {}).get("opponent_race") or (words[2] if len(words) > 2 and words[2] in RACES else None)
    heroes: dict[str, dict[str, int]] = {"me": {}, "opp": {}}
    target = None
    if trace:
        sides = trace.get("sides") or []
        me = next((s for s in sides if "learner" in str(s.get("name", ""))), sides[0] if sides else None)
        mp = str(me["player"]) if me else str((ep or {}).get("side", 0))
        for st in trace.get("steps") or []:
            for p, hs in (st.get("heroes") or {}).items():
                into = heroes["me" if p == mp else "opp"]
                for h in hs:
                    into[h[0]] = max(into.get(h[0], 0), int(h[1]) if len(h) > 1 else 0)
        tg = trace.get("target")
        if tg and tg.get("t"):
            made = tg.get("made") or {}
            end = float(m.get("game_time") or (ep or {}).get("game_time") or 1e9)
            items = sorted(([c, names.get(c, c), round(float(t), 1), round(float(made[c]), 1) if c in made else None]
                            for c, t in tg["t"].items()), key=lambda x: x[2])
            due = [x for x in items if x[2] <= end]
            target = {"src": tg.get("src"), "items": items, "end": round(end, 1),
                      "followed": round(sum(x[3] is not None for x in due) / len(due), 2) if due else None}
    elif ep and ep.get("z_t"):
        made = ep.get("z_made") or {}
        items = sorted(([c, names.get(c, c), round(float(t), 1), round(float(made[c]), 1) if c in made else None]
                        for c, t in ep["z_t"].items()), key=lambda x: x[2])
        end = float(ep.get("game_time") or 1e9)
        due = [x for x in items if x[2] <= end]
        target = {"src": None, "items": items, "end": round(end, 1),
                  "followed": round(sum(x[3] is not None for x in due) / len(due), 2) if due else None}
    me = _side(race, heroes["me"], (ep or {}).get("prod"), names)
    opp = _side(opp_race, heroes["opp"], (ep or {}).get("opp_prod"), names)
    if ep:
        me["picked"] = int(ep.get("picked_up") or 0)
        bought: dict[str, int] = {}
        for k, n in (ep.get("buys") or {}).items():  # ("Npbm by hpea": the ware, by its buyer's type)
            c = k.split()[0]
            bought[names.get(c, c)] = bought.get(names.get(c, c), 0) + int(n)
        if bought:
            me["bought"] = bought
    tags = []
    for who, s in (("", me), ("opponent: ", opp)):
        neutral = [h[1] for h in s["heroes"] if h[0].startswith("N")]
        if neutral:
            tags.append(f"{who}neutral hero")
        if s.get("tier", 1) >= 2:
            tags.append(f"{who}tier {s['tier']}")
        if s.get("expanded"):
            tags.append(f"{who}expansion")
    if me.get("picked"):
        tags.append("items picked up")
    if me.get("creeps"):
        tags.append("creeping")
    if target:
        tags.append("target")
    return {"v": INFO_VERSION, "me": me, "opp": opp, "target": target, "tags": tags}


def _episodes(run_dir: Path, times: set[int]) -> dict[int, list[dict]]:
    """The run's episode rows whose time (in whole seconds) is one of `times` (a video's stem names its episode's):
    one pass over its episode file, a row parsed only when its time matches."""
    out: dict[int, list[dict]] = {}
    f = run_dir / "episodes.jsonl"
    if not times or not f.exists():
        return out
    with f.open() as fh:
        for line in fh:
            m = _TIME.match(line, 1)
            if m and int(m.group(1)) in times:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.setdefault(int(r["time"]), []).append(r)
    return out


def _match(m: dict, rows: list[dict]) -> dict | None:
    """The episode row of a video (several games end in one second): its race, opponent and length."""
    words = (m.get("title") or "").split()
    best = None
    for r in rows:
        if r.get("eval"):
            continue
        if words and r.get("race") != words[0]:
            continue
        if m.get("opponent") and r.get("opponent") != m.get("opponent"):
            continue
        d = abs(float(r.get("game_time") or 0) - float(m.get("game_time") or r.get("game_time") or 0))
        if best is None or d < best[0]:
            best = (d, r)
    return best[1] if best and best[0] < 2.0 else None


class VideoIndex:
    """Every run's videos and their info (info.json files, made by fill())."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.lock = threading.Lock()
        self.info: dict[str, dict] = {}  # "run/file" -> its info
        self.videos: list[dict] = []  # the media rows, with "run" (newest first)
        self.pending = 0
        self.updated = 0.0
        self._names: dict[str, str] | None = None

    def _media(self, d: Path) -> list[dict]:
        rows, seen = [], set()
        try:
            lines = (d / "media.jsonl").read_text().splitlines()
        except OSError:
            return rows
        for line in reversed(lines):
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            f = m.get("file", "")
            if m.get("kind") != "video" or f in seen or not stem_of(f) or not (d / f).exists():
                continue
            seen.add(f)
            rows.append({**m, "run": d.name})
        return rows

    def fill(self, budget: float = 20.0) -> None:
        """List the videos and make the info of those without one (for `budget` seconds at most a call)."""
        runs = sorted((p.parent for p in self.runs_dir.glob("*/media.jsonl")), key=lambda d: -(d / "media.jsonl").stat().st_mtime)
        videos, todo = [], []
        for d in runs:
            for m in self._media(d):
                videos.append(m)
                key = f"{d.name}/{m['file']}"
                if key in self.info:
                    continue
                side = d / m["file"].replace(".mp4", ".info.json")
                try:
                    info = json.loads(side.read_text())
                    if info.get("v") == INFO_VERSION:
                        self.info[key] = info
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
                todo.append((d, m))
        videos.sort(key=lambda m: -float(m.get("time") or 0))
        with self.lock:
            self.videos = videos
            self.pending = len(todo)
        t0 = time.time()
        by_run: dict[Path, list[dict]] = {}
        for d, m in todo:
            by_run.setdefault(d, []).append(m)
        for d, ms in by_run.items():
            if time.time() - t0 > budget:
                break
            if self._names is None:
                self._names = _names()
            eps = _episodes(d, {int(stem_of(m["file"]).split("-")[1]) for m in ms})
            for m in ms:
                stem = stem_of(m["file"])
                trace = None
                try:
                    trace = json.loads((d / "replays" / f"{stem}.trace.json").read_text())
                except (OSError, json.JSONDecodeError):
                    pass
                ep = _match(m, eps.get(int(stem.split("-")[1]), []))
                info = summarize(m, trace, ep, self._names)
                try:
                    (d / m["file"].replace(".mp4", ".info.json")).write_text(json.dumps(info))
                except OSError:
                    pass
                with self.lock:
                    self.info[f"{d.name}/{m['file']}"] = info
                    self.pending -= 1
        self.updated = time.time()

    def get(self, run: str, file: str) -> dict | None:
        with self.lock:
            return self.info.get(f"{run}/{file}")

    def listing(self) -> dict:
        with self.lock:
            return {"videos": [{k: m.get(k) for k in ("run", "file", "title", "outcome", "sub", "time", "game_time", "opponent")}
                               | {"info": self.info.get(f"{m['run']}/{m['file']}")} for m in self.videos],
                    "pending": self.pending, "updated": self.updated}


def background(index: VideoIndex, every: float = 60.0) -> None:
    """Fill the index now and every `every` seconds (a daemon thread of the dashboard)."""
    def loop():
        while True:
            try:
                index.fill()
                while index.pending > 0:  # (the backlog: on, in slices)
                    index.fill()
            except Exception as e:  # noqa: BLE001 (the dashboard must stay up)
                print(f"videos: {type(e).__name__}: {e}", flush=True)
            time.sleep(every)
    threading.Thread(target=loop, daemon=True).start()
