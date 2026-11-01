"""Every run's replays that can become a video (the dashboard's Replays page: the user's idea, 2026-11-01, "a filterable
list of all replays ... to request a render of any of them").

A replay is listed when it has its trace and its orders (replays/game-<time>-a<w>.{w3g,trace.json,commands.json}). What
the list shows of it comes from its trace (0.3 MB on average, up to 2.5): read once, by a thread, summed up in a small row
kept in a file of its own (`cache`, ~/.cache/warcraftsim/replay-index.json, never in the runs), so the server holds a few
hundred bytes per replay and looks at a run's files again only when its folder's modification time changes. Whether it
has a video, and at what speed, comes from the videos' folder and media.jsonl (again by their modification times).
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from .. import paths

VERSION = 1
_NAME = re.compile(r"game-(\d+)-a\d+")
_RACE = re.compile(r"\((human|orc|undead|nightelf)\)")
OUTCOMES = {"win": 1, "loss": -1, "tie": 0}
WORDS = {v: k for k, v in OUTCOMES.items()}


def summarize(stem: str, trace: dict) -> dict:
    """A replay's row: when, who played whom, how it ended for the first side (the learner) and how long it took."""
    sides = trace.get("sides") or []
    races = [(_RACE.search(str(s.get("name", ""))) or [None, None])[1] for s in sides]
    names = [_RACE.sub("", str(s.get("name", ""))).strip() for s in sides]
    steps = trace.get("steps") or []
    me = str(sides[0].get("player")) if sides else ""
    o = (trace.get("outcome") or {}).get(me)
    return {"stem": stem, "time": int(_NAME.fullmatch(stem).group(1)), "races": races, "names": names,
            "title": str(trace.get("title") or ""), "outcome": 0 if o is None else (o > 0) - (o < 0),
            "game_time": round(float(steps[-1].get("game_time") or 0), 1) if steps else 0.0}


class ReplayIndex:
    """The replays of `runs_dir`'s runs; fill() brings it up to date (cheap when nothing changed)."""

    def __init__(self, runs_dir: Path, cache: Path | None = None):
        self.runs_dir = Path(runs_dir)
        self.cache = Path(cache) if cache else paths.CACHE_DIR / "replay-index.json"
        self.lock = threading.Lock()
        self.runs: dict[str, dict] = {}  # run -> {"mtime": its replays folder's, "rows": {stem: row}}
        self.videos: dict[str, dict] = {}  # run -> {"mtime": (videos', media.jsonl's), "speeds": {stem: speed}}
        self.pending = 0
        self.updated = 0.0
        self._loaded = False

    def _load(self) -> None:
        self._loaded = True
        try:
            d = json.loads(self.cache.read_text())
            if d.get("v") == VERSION and d.get("root") == str(self.runs_dir):
                self.runs = d["runs"]
        except (OSError, ValueError, KeyError):
            pass

    def _save(self) -> None:
        try:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache.with_suffix(f".{id(self)}.tmp")
            tmp.write_text(json.dumps({"v": VERSION, "root": str(self.runs_dir), "runs": self.runs}))
            tmp.replace(self.cache)
        except OSError:
            pass

    def _video_speeds(self, d: Path) -> dict[str, float]:
        """The run's videos: stem -> the speed it was recorded at (the file's latest media row; none says: real time)."""
        have = {p.stem for p in (d / "videos").glob("game-*.mp4") if _NAME.fullmatch(p.stem)}
        out = dict.fromkeys(have, 1.0)
        try:
            for line in (d / "media.jsonl").read_text().splitlines():
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stem = Path(str(m.get("file", ""))).stem
                if stem in have and m.get("kind") == "video":
                    out[stem] = float(m.get("recorded_speed") or 1.0)
        except OSError:
            pass
        return out

    def fill(self, budget: float = 20.0) -> None:
        """Look at each run's folders; for at most `budget` seconds a call, sum up replays new since the last look."""
        if not self._loaded:
            self._load()
        t0 = time.time()
        pending, changed = 0, False
        dirs = [p.parent for p in self.runs_dir.glob("*/replays")]
        seen = {d.name for d in dirs}
        with self.lock:
            for gone in set(self.runs) - seen:
                del self.runs[gone]
                changed = True
        # (the newest runs first: the first look at thousands of traces takes a while)
        for d in sorted(dirs, key=lambda d: -(d / "replays").stat().st_mtime):
            run = d.name
            replays = d / "replays"
            try:
                vm = ((d / "videos").stat().st_mtime if (d / "videos").exists() else 0,
                      (d / "media.jsonl").stat().st_mtime if (d / "media.jsonl").exists() else 0)
                if self.videos.get(run, {}).get("mtime") != list(vm):
                    speeds = self._video_speeds(d)
                    with self.lock:
                        self.videos[run] = {"mtime": list(vm), "speeds": speeds}
                m = replays.stat().st_mtime
            except OSError:
                continue
            old = self.runs.get(run)
            if old and old.get("mtime") == m:
                continue
            rows = dict((old or {}).get("rows") or {})
            stems = {p.name[:-len(".trace.json")] for p in replays.glob("game-*.trace.json")}
            todo = [s for s in stems if s not in rows and _NAME.fullmatch(s)]
            rows = {s: r for s, r in rows.items() if s in stems}
            complete = True
            for s in sorted(todo, reverse=True):
                if time.time() - t0 > budget:
                    pending += len(todo) - sum(1 for x in todo if x in rows)
                    complete = False
                    break
                if not ((replays / f"{s}.w3g").exists() and (replays / f"{s}.commands.json").exists()):
                    continue  # (its files come one by one; with its folder's next change: looked at again)
                try:
                    rows[s] = summarize(s, json.loads((replays / f"{s}.trace.json").read_text()))
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    pass  # (a trace of another shape: no video can be made of it, not listed)
                time.sleep(0.001)  # (a pause for the dashboard's requests: this thread holds the GIL while parsing)
            with self.lock:
                self.runs[run] = {"mtime": m if complete else -1, "rows": rows}
            changed = True
        with self.lock:
            self.pending = pending
            self.updated = time.time()
        if changed:
            self._save()

    def listing(self, q: str = "", outcome: str = "all", race: str = "all", video: str = "all", offset: int = 0,
                limit: int = 100) -> dict:
        """The replays matching (every word of `q` in its run, title, names, races or outcome; outcome: win, loss, tie;
        race: the first side's; video: "yes", "no"), newest first, a page of them."""
        words = q.lower().split()
        want = OUTCOMES.get(outcome)
        out = []
        total = 0
        with self.lock:
            for run, d in self.runs.items():
                speeds = (self.videos.get(run) or {}).get("speeds") or {}
                for r in (d.get("rows") or {}).values():
                    total += 1
                    has = r["stem"] in speeds
                    if want is not None and r["outcome"] != want:
                        continue
                    if race != "all" and (not r["races"] or r["races"][0] != race):
                        continue
                    if (video == "yes" and not has) or (video == "no" and has):
                        continue
                    if words:
                        hay = " ".join([run, r["title"], r["stem"], *map(str, r["names"]), *map(str, r["races"]),
                                        WORDS[r["outcome"]]]).lower()
                        if not all(w in hay for w in words):
                            continue
                    out.append((r, run, speeds.get(r["stem"])))
            pending, updated = self.pending, self.updated
        out.sort(key=lambda x: -x[0]["time"])
        page = out[max(0, offset):max(0, offset) + max(1, min(limit, 500))]
        return {"total": total, "matched": len(out), "offset": offset, "pending": pending, "updated": updated,
                "rows": [{**r, "run": run, "video": speed is not None, "speed": speed} for r, run, speed in page]}


def background(index: ReplayIndex, every: float = 20.0) -> None:
    """Fill the index now and every `every` seconds (a daemon thread of the dashboard)."""
    def loop():
        while True:
            try:
                index.fill()
                while index.pending > 0:  # (the backlog: on, in slices)
                    index.fill()
            except Exception as e:  # noqa: BLE001 (the dashboard must stay up)
                print(f"replays: {type(e).__name__}: {e}", flush=True)
            time.sleep(every)
    threading.Thread(target=loop, daemon=True).start()
