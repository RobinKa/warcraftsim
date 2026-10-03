"""Training dashboard: a small web server over the runs/ directory.

    python -m warcraftsim dashboard --port 8765        # then open http://localhost:8765

Reads what the training runs write (run.json, train.jsonl, episodes[-w].jsonl, bridge[-w].jsonl,
media[-w].jsonl from each bridge worker w, renders/, videos/) and serves it as JSON plus one
self-contained page. Episodes are placed on the trainer's step axis by their timestamps.

Lineage: a run started from a checkpoint (run.json init_from) has a parent, another run or a
behavior-cloning dataset (runs/bc/<name>, its meta.json); the parent lists it as a child. Notes
live in runs/<name>/notes.md (train.py --note; POST /api/runs/<name>/notes edits them).
Behavior cloning datasets (runs/bc/<name>: bc.json, meta.json, episodes-*.jsonl, fit.jsonl,
evals.jsonl; warcraftsim.puffer.bc) are runs named bc/<name> of kind "bc". evals.jsonl in a run's
directory (bc eval of its checkpoints) shows with the run.
Sweeps (/api/sweeps): the runs of one train.py launch, with runs/sweeps/<group>/sweep.json (how it
was launched) and notes.md (its description; POST /api/sweeps/<group>/notes). Older sweeps get
theirs rebuilt from their runs.
"""

from __future__ import annotations

import json
import mimetypes
from collections import OrderedDict
import os
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

import numpy as np

TRAIN_KEYS = ("agent_steps", "SPS", "epoch", "uptime", "env/win_rate", "env/loss_rate", "env/episode_return",
              "env/episode_length", "env/n", "loss/policy", "loss/value", "loss/entropy", "loss/kl",
              "loss/old_kl", "loss/clipfrac", "loss/bc", "importance", "perf/rollout", "perf/eval_env", "perf/eval_model",
              "perf/eval_copy", "perf/train", "util/gpu_percent", "util/vram_used_gb", "util/cpu_mem_gb", "time",
              "loss/ref_kl", "lr")
# per-episode series (rolling means): name -> value of an episode row (None: not recorded)
EPISODE_SERIES = {
    "win_rate": lambda e: 1.0 if e.get("outcome", 0) > 0 else 0.0,
    "loss_rate": lambda e: 1.0 if e.get("outcome", 0) < 0 else 0.0,
    "draw_rate": lambda e: 1.0 if e.get("outcome", 0) == 0 else 0.0,
    "return": lambda e: float(e.get("return", 0)),
    "length": lambda e: float(e.get("length", 0)),
    "game_time": lambda e: e.get("game_time"),
    **{f"act_{k}": (lambda k: lambda e: e.get("act", {}).get(k))(k)
       for k in ("noop", "stop", "hold", "retreat", "move", "attack_move", "attack", "cast", "attack_invalid", "attack_weakest",
                 "focus_fire", "cast_invalid")},
    **{f"combat_{k}": (lambda k: lambda e: e.get("combat", {}).get(k))(k)
       for k in ("dealt", "taken", "kills", "losses", "focus_dealt", "focus_taken")},
    # whole-game self-play (fullgame/selfplay.py): what the learner (p_) and the built-in AI (o_) make,
    # in the games against the AI
    **{f"{pre}_{k}": (lambda side, k: lambda e: ((e.get(side) or {}).get(k)
                                                  if str(e.get("opponent", "")).startswith("script:") else None))(side, k)
       for pre, side in (("p", "prod"), ("o", "opp_prod"))
       for k in ("army", "workers", "heroes", "food_1min", "held", "lumber", "gold", "kills", "lost", "max_food")},
}
EPISODE_SERIES.update({  # whole-game self-play, over all its games (the Behaviour tab)
    "orders": lambda e: e.get("orders") if e.get("prod") is not None else None,
    "f_kills": lambda e: (e.get("prod") or {}).get("kills"),
    "f_lost": lambda e: (e.get("prod") or {}).get("lost"),
})
PROD_SCALARS = ("army", "workers", "heroes", "food_1min", "max_food", "held", "gold", "lumber", "kills", "lost")


def _trained_seconds(rows: list[dict]) -> float:
    """The learner's time over all its sessions: its uptime restarts when a run is resumed."""
    total, last = 0.0, 0.0
    for r in rows:
        u = r.get("uptime")
        if u is None:
            continue
        if u < last:  # a new session
            total += last
        last = u
    return total + last


def _real_game(episodes: list[dict], n: int = 100) -> dict | None:
    """Whole-game self-play: the last `n` real games against the built-in AI (no curriculum)."""
    real = [e for e in episodes if "(real)" in str(e.get("opponent", ""))][-n:]
    if not real:
        return None
    return {"games": len(real), "wins": sum(e.get("outcome", 0) > 0 for e in real) / len(real),
            "ties": sum(e.get("outcome", 0) == 0 for e in real) / len(real)}


MATCHUP_RACES = ("human", "orc", "undead", "nightelf")
SPARSE_GAMES = 40
SERIES_KEYS = tuple(EPISODE_SERIES)


def _matchup_kind(e: dict) -> list[str]:
    """The kinds of whole-game self-play game an episode counts for in the matchup tables."""
    o = str(e.get("opponent", ""))
    if "(real)" in o:
        return ["real", "real " + o.split("-", 1)[-1].split()[0]]
    if o.startswith("script:"):
        return ["curriculum"]
    if o == "self":
        return ["self"]
    return ["past"] if o.startswith("past:") else []


class _EpisodeSeries:
    """A run's episodes as columns of their EPISODE_SERIES values (NaN: none), and their results by
    matchup, kept between requests and extended with the new episodes only: a live run's page
    refreshes every few seconds, and reading 135k episodes anew took 2 s, binning them 3 s."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.seen: dict[Path, tuple[int, int]] = {}  # file -> (its rows list's id, rows read)
        self.times = np.empty(0)
        self.cols = np.empty((0, len(SERIES_KEYS)))
        # sparse series of whole-game self-play: series name -> (rows, values): curriculum games won
        # by AI and matchup ("cwin/ai-easy human/orc"), the real game won by the learner's race ("rwin/human")
        self.extra: dict[str, tuple[list[int], list[float]]] = {}
        # kind -> [(race, opponent race, 0 win / 1 tie / 2 loss)], and the whole run's counts
        self.games: dict[str, list[tuple[int, int, int]]] = {}
        self.totals: dict[str, np.ndarray] = {}

    def update(self, files: list[tuple[Path, list[dict]]]) -> None:
        """`files`: the run's episode files and their rows (as _JsonlCache keeps them: one list per
        file that grows; a new list when the file was replaced)."""
        if any(f in self.seen and (self.seen[f][0] != id(rows) or self.seen[f][1] > len(rows)) for f, rows in files) \
                or set(self.seen) - {f for f, _ in files}:
            self.reset()  # (a file replaced or gone: read anew)
        new = []
        for f, rows in files:
            new += [r for r in rows[self.seen.get(f, (0, 0))[1]:] if "event" not in r]
            self.seen[f] = (id(rows), len(rows))
        if not new:
            return
        block = np.full((len(new), len(SERIES_KEYS)), np.nan)
        gets = list(EPISODE_SERIES.values())
        n0 = len(self.times)
        for i, e in enumerate(new):
            for j, get in enumerate(gets):
                v = get(e)
                if v is not None:
                    block[i, j] = float(v)
            r, b = e.get("race"), e.get("opponent_race")
            if r in MATCHUP_RACES and b in MATCHUP_RACES:
                x = e.get("outcome", 0)
                g = (MATCHUP_RACES.index(r), MATCHUP_RACES.index(b), 0 if x > 0 else 1 if x == 0 else 2)
                for k in _matchup_kind(e):
                    self.games.setdefault(k, []).append(g)
                    self.totals.setdefault(k, np.zeros((4, 4, 3), np.int64))[g] += 1
                o = str(e.get("opponent", ""))
                key = (f"rwin/{r}" if "(real)" in o else f"cwin/{o.split(':', 1)[1]} {r}/{b}" if o.startswith("script:") else None)
                if key:
                    rows, vals = self.extra.setdefault(key, ([], []))
                    rows.append(n0 + i)
                    vals.append(1.0 if x > 0 else 0.0)
        self.cols = np.concatenate([self.cols, block])
        self.times = np.concatenate([self.times, [float(e.get("time", 0)) for e in new]])

    def binned(self, train: list[dict], in_order: bool = False) -> tuple[list[dict], dict]:
        """The series binned on the trainer's step axis (the episodes placed by their times; the
        file's order is not quite theirs: games end in one order and arrive in another), or with
        `in_order` by their number; and the sparse series, each binned on its own: name -> [[x, mean,
        games], ...], at least `SPARSE_GAMES` games a point (a matchup's games are a 30th of all: in
        the shared bins a point held one or two)."""
        order = np.argsort(self.times, kind="stable")
        times = self.times[order]
        if in_order:
            xs = np.arange(1.0, len(times) + 1)
        else:
            pts = np.array([(r["time"], r.get("agent_steps", 0)) for r in train if "time" in r], float).reshape(-1, 2)
            if len(pts):  # as _interp_steps: linear between the trainer's lines, 0 before the first, the last after it
                xs = np.where(times < pts[0, 0], 0.0, np.interp(times, pts[:, 0], pts[:, 1]))
            else:
                xs = np.zeros(len(times))
        x_of = np.empty(len(times))
        x_of[order] = xs  # (each row's step, in the file's order)
        sparse = {}
        for key, (rows, vals) in self.extra.items():
            x, v = x_of[np.asarray(rows)], np.asarray(vals)
            o = np.argsort(x, kind="stable")
            x, v = x[o], v[o]
            size = max(SPARSE_GAMES, -(-len(x) // 120))
            cuts = list(range(0, len(x), size))
            if len(cuts) > 1 and len(x) - cuts[-1] < size // 2:
                cuts.pop()  # (a short last piece joins the one before)
            ends = cuts[1:] + [len(x)]
            sparse[key] = [[float(x[a:b].mean()), float(v[a:b].mean()), b - a] for a, b in zip(cuts, ends)]
        return _binned_columns(xs, self.cols[order]), sparse

    def matchups(self, recent: int = 1500) -> dict | None:
        """Win rates by matchup (the learner's race by the opponent's), [wins, ties, losses] a cell,
        for each kind of game: the real game against the built-in AI (also by difficulty),
        curriculum games (the AI taxed), the learner against itself (each game counted from both
        sides) and against past snapshots; over the kind's last `recent` games and the whole run."""
        if not self.games:
            return None

        def table(m: np.ndarray, both: bool) -> dict:
            if both:  # (against itself: the other side's result too)
                m = m + m.transpose(1, 0, 2)[:, :, ::-1]
            return {a: {b: [int(v) for v in m[i, j]] for j, b in enumerate(MATCHUP_RACES)} for i, a in enumerate(MATCHUP_RACES)}
        out = {}
        for k, gs in self.games.items():
            m = np.zeros((4, 4, 3), np.int64)
            for g in gs[-recent:]:
                m[g] += 1
            out[k] = {"recent": table(m, k == "self"), "all": table(self.totals[k], k == "self"),
                      "games": len(gs), "recent_games": min(recent, len(gs))}
        return out


def _binned_columns(xs: list[float], cols: np.ndarray, n: int | None = None) -> list[dict]:
    """_binned() over columns (SERIES_KEYS, NaN: no value): each bin's mean of each column over the
    rows that have it, x = the bin's mean x."""
    n, N = n or MAX_POINTS, len(xs)
    if not N:
        return []
    size = max(1, -(-N // n))
    b = np.arange(N) // size
    nb = int(b[-1]) + 1
    cnt = np.bincount(b, minlength=nb)
    sx = np.bincount(b, weights=np.asarray(xs, float), minlength=nb)
    out = [{"steps": float(sx[i] / cnt[i]), "n": int(cnt[i])} for i in range(nb)]
    have = ~np.isnan(cols)
    for j, key in enumerate(SERIES_KEYS):
        m = have[:, j]
        if not m.any():
            continue
        c = np.bincount(b[m], minlength=nb)
        s = np.bincount(b[m], weights=cols[m, j], minlength=nb)
        for i in np.nonzero(c)[0]:
            out[i][key] = float(s[i] / c[i])
    return out


MAX_POINTS = 600
MAX_NOTES = 64 * 1024
MAX_DOC = 512 * 1024
DOC_NAME = __import__("re").compile(r"((reference|proposals|related)/)?[a-z0-9][a-z0-9-]{0,63}")
DOC_GROUPS = ("", "proposals", "related", "reference")  # the tab's sections: docs/*.md, then docs/<group>/*.md
# the Docs tab: docs/*.md (short, first the ones a reader starts with), then docs/reference/*.md (the details)
DOC_ORDER = ("overview", "environments", "model", "experiments", "road-to-the-real-game", "optimizations", "architecture")


class Docs:
    """The repository's documents (docs/*.md) for the dashboard's Docs tab: listed, read, saved."""

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def title(text: str, name: str) -> str:
        for line in text.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return name

    def list(self) -> list[dict]:
        out = []
        for group in DOC_GROUPS:
            files = (self.root / group).glob("*.md") if group else self.root.glob("*.md")
            for f in sorted(files):
                text = f.read_text(errors="replace")
                name = f"{group}/{f.stem}" if group else f.stem
                out.append({"name": name, "group": group, "title": self.title(text, f.stem), "updated": f.stat().st_mtime,
                            "bytes": f.stat().st_size, "lines": text.count("\n") + 1})
        rank = {n: i for i, n in enumerate(DOC_ORDER)}
        return sorted(out, key=lambda d: (DOC_GROUPS.index(d["group"]), rank.get(d["name"], len(rank)), d["title"].lower()))

    def read(self, name: str) -> dict | None:
        f = self.root / f"{name}.md"
        if not DOC_NAME.fullmatch(name) or not f.is_file():
            return None
        text = f.read_text(errors="replace")
        return {"name": name, "title": self.title(text, name), "text": text, "updated": f.stat().st_mtime}

    def save(self, name: str, text: str) -> bool:
        """Writes docs/<name>.md (a new document too); the name: lowercase letters, digits, dashes."""
        if not DOC_NAME.fullmatch(name) or len(text) > MAX_DOC:
            return False
        target = self.root / f"{name}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_text(text)
        tmp.replace(target)
        return True
# train.py options in the order its command line gives them (a command rebuilt for older runs)
_ARG_ORDER = ("task", "envs", "workers", "timesteps", "step_seconds", "horizon", "minibatch", "replay_ratio",
              "buffers", "lr", "ent_coef", "gamma", "hidden", "layers", "checkpoint_interval", "record_every",
              "video_every", "init_from")



def _production(episodes: list[dict]) -> dict | None:
    """Whole-game self-play: what the learner and its opponent make per game, by kind of game (the
    real game against the built-in AI, a curriculum game, self-play) and the learner's race."""
    rows = [e for e in episodes if e.get("prod")]
    if not rows:
        return None
    try:
        from ..data.objects import unit_names, upgrade_names
        names = {**unit_names(), **upgrade_names()}
    except Exception:  # noqa: BLE001 (no game data: codes only)
        names = {}

    def kind(e: dict) -> str:
        o = str(e.get("opponent", ""))
        return "real game" if "(real)" in o else "curriculum" if o.startswith("script:") else "self-play"

    groups: dict[tuple, list] = {}
    for e in rows:
        groups.setdefault((kind(e), e.get("race", "?")), []).append(e)
    order = {"real game": 0, "curriculum": 1, "self-play": 2}
    out, used = [], set()
    for (k, race), es in sorted(groups.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])):
        n = len(es)

        def side(key: str) -> dict:
            per = [e.get(key) or {} for e in es]
            res = {x: (sum(p.get(x) or 0 for p in per) / n) for x in PROD_SCALARS}
            f1 = [p["food_1min"] for p in per if p.get("food_1min") is not None]
            res["food_1min"] = sum(f1) / len(f1) if f1 else None
            for field in ("trained", "built"):
                tot: dict[str, float] = {}
                for p in per:
                    for code, c in (p.get(field) or {}).items():
                        tot[code] = tot.get(code, 0) + c / n
                res[field] = dict(sorted(tot.items(), key=lambda kv: -kv[1])[:14])
                used.update(res[field])
            res["research"] = {}
            for p in per:
                for code in p.get("research") or {}:
                    res["research"][code] = res["research"].get(code, 0) + 1 / n
            res["research"] = dict(sorted(res["research"].items(), key=lambda kv: -kv[1])[:10])
            used.update(res["research"])
            return res
        out.append({"kind": k, "race": race, "games": n,
                    "wins": sum(e.get("outcome", 0) > 0 for e in es) / n, "ties": sum(e.get("outcome", 0) == 0 for e in es) / n,
                    "tax": (sum(e.get("ai_tax", 0) for e in es) / n) if k == "curriculum" else None,
                    "learner": side("prod"), "opponent": side("opp_prod")})
    return {"groups": out, "names": {c: names[c] for c in used if c in names}, "games": len(rows)}


def _mean_side(rows: list[dict], side: str, key: str) -> float | None:
    """The mean of a play result's sides[side][key] over the games that record it (None: none do)."""
    v = [((r.get("sides") or {}).get(side) or {}).get(key) for r in rows]
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None

class _JsonlCache:
    """Incrementally read JSON-lines files (they only grow; a file that is replaced, e.g. a run
    deleted and started again under its name, is read anew). Keeps the `max_files` most recently
    read files: the runs being looked at, not every run ever."""

    def __init__(self, max_files: int = 64):
        self._files: "OrderedDict[Path, tuple[int, list[dict], bytes]]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_files = max_files

    def read(self, path: Path) -> list[dict]:
        with self._lock:
            offset, rows, head = self._files.get(path, (0, [], b""))
            if path in self._files:
                self._files.move_to_end(path)
            try:
                st = path.stat()
            except FileNotFoundError:
                self._files.pop(path, None)
                return []
            size = st.st_size
            if size < offset or (head and _head(path, len(head)) != head):  # rewritten, or another file now
                offset, rows, head = 0, [], b""
            if size > offset:
                with open(path, "rb") as f:
                    f.seek(offset)
                    data = f.read(size - offset)
                end = data.rfind(b"\n") + 1  # ignore a partially written last line
                for line in data[:end].splitlines():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                offset += end
                if len(head) < 256:
                    head = _head(path, min(offset, 256))
            self._files[path] = (offset, rows, head)
            while len(self._files) > self.max_files:
                self._files.popitem(last=False)
            return rows


def _head(path: Path, n: int) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _tail_rows(path: Path, n: int, max_bytes: int = 48 * 1024) -> list[dict]:
    """The last n parseable rows of a JSON-lines file, reading only its end."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    rows = []
    for line in data.splitlines()[-(n + 1):]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows[-n:]


def _downsample(rows: list, n: int = MAX_POINTS) -> list:
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)] + [rows[-1]]


def _binned(xs: list[float], rows: list[dict], n: int = MAX_POINTS) -> list[dict]:
    """At most n points: consecutive rows averaged per bin (each value over the rows that have it),
    x = the bin's mean x. The page smooths on top of this."""
    if not rows:
        return []
    size = max(1, -(-len(rows) // n))
    out = []
    for i in range(0, len(rows), size):
        chunk, cx = rows[i:i + size], xs[i:i + size]
        point = {"steps": sum(cx) / len(cx), "n": len(chunk)}
        for key in chunk[-1].keys() | chunk[0].keys():
            vals = [r[key] for r in chunk if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
            if vals:
                point[key] = sum(vals) / len(vals)
        out.append(point)
    return out


def _rolling(values: list[float], window: int) -> list[float]:
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def _interp_steps(times: list[float], train: list[dict]) -> list[float]:
    """Trainer agent_steps at the given wall times (linear interpolation on train.jsonl)."""
    pts = [(r["time"], r.get("agent_steps", 0)) for r in train if "time" in r]
    if not pts:
        return [0.0] * len(times)
    out, j = [], 0
    for t in times:  # times are sorted
        while j + 1 < len(pts) and pts[j + 1][0] <= t:
            j += 1
        if t < pts[0][0]:
            out.append(0.0)  # before the trainer's first log line
        elif j + 1 < len(pts):
            (t0, s0), (t1, s1) = pts[j], pts[j + 1]
            out.append(s0 + (s1 - s0) * (t - t0) / max(t1 - t0, 1e-9))
        else:
            out.append(pts[-1][1])
    return out


def _options(info: dict) -> dict:
    """A run's train.py options as flag -> value (PufferLib --section.key=value options included)."""
    args, out = info.get("args", {}), {}
    for key in _ARG_ORDER:
        v = args.get(key)
        if v is None or v == "" or (key in ("buffers", "step_seconds", "minibatch") and not v):
            continue
        out[f"--{key.replace('_', '-')}"] = str(int(v) if key == "timesteps" else v)
    for e in info.get("extra", []):
        k, _, v = e.partition("=")
        if k != "--base.load_model_path":
            out[k] = v
    return out


def _flags(opts: dict) -> list[str]:
    return [p for k, v in opts.items() for p in ((f"{k}={v}",) if "." in k else (k, v))]


def _rebuilt_sweep_command(infos: list[dict], group: str) -> str:
    """A sweep launch equivalent to its runs: the options they share, then each run's own."""
    import shlex

    opts = [_options(i) for i in infos]
    shared = {k: v for k, v in opts[0].items() if all(o.get(k) == v for o in opts[1:])}
    parts = ["python", "-m", "warcraftsim.puffer.train", *_flags(shared), "--name", group]
    for o in opts:
        parts += ["--sweep", shlex.join(_flags({k: v for k, v in o.items() if k not in shared})) or ""]
    return shlex.join(parts)


_SPACES: dict[str, dict | None] = {}


def _fullgame_spaces(d: Path, runs_dir: Path) -> dict | None:
    """Whole games (self-play runs, fits): described from the run's vocabulary (vocab.json)."""
    vocab = _read_json(d / "vocab.json")
    if not vocab:
        return None
    try:
        from ..fullgame import features as fx
        names = None if vocab.get("order_names") else fx.demo_order_names(runs_dir)
        return {**fx.describe_spaces(vocab, names), "from_current_code": True, "sizes_match": True}
    except Exception:  # noqa: BLE001
        return None


def _spaces(info: dict, d: Path | None = None, runs_dir: Path | None = None) -> dict | None:
    """A run's observation and action spaces: recorded at launch, else described by the current
    code (flagged, and whether its sizes match the run's)."""
    if info.get("spaces"):
        return info["spaces"]
    task = info.get("task")
    if not task:
        return None
    if str(task).startswith("fullgame") and d is not None:
        return _fullgame_spaces(d, runs_dir)
    if task not in _SPACES:
        try:
            from ..puffer.tasks import describe_spaces, get_task
            _SPACES[task] = describe_spaces(get_task(task))
        except Exception:  # a task that no longer exists
            _SPACES[task] = None
    d = _SPACES[task]
    if d is None:
        return None
    same = (info.get("obs_size") in (None, d["observation"]["size"])
            and info.get("act_sizes") in (None, d["actions"]["sizes"]))
    return {**d, "from_current_code": True, "sizes_match": same}


def _checkpoints(d: Path, repo: Path, evals: list[dict], league: dict | None = None, roles: dict | None = None) -> list[dict]:
    """A run's saved checkpoints, newest first: checkpoints/**/*.pt|.bin (the torch trainer and
    self-play: <agent steps>.pt; PufferLib: <env>/<time>/<steps>.bin) and the run directory's
    own policy files (behavior cloning: policy.pt = the best epoch, last.pt; self-play: current.pt,
    the weights the actors load). Each with its steps (from the name), size, time, its path
    from the repository, the league member it is (self-play) and its evaluations."""
    files = []
    ck_dir = d / "checkpoints"
    if ck_dir.is_dir():
        files += [f for f in ck_dir.rglob("*") if f.suffix in (".pt", ".bin") and f.is_file()]
    files += [d / n for n in ("policy.pt", "policy.bin", "last.pt", "current.pt") if (d / n).is_file()]
    members = {}
    for m in (league or {}).get("members", []):
        if m.get("path"):
            members[Path(m["path"]).name] = m
    by_ck: dict[str, list[dict]] = {}
    for e in evals:
        if e.get("checkpoint"):
            by_ck.setdefault(Path(e["checkpoint"]).name if "/checkpoints/" in e["checkpoint"] else e["checkpoint"], []).append(e)
    out = []
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        rel = f.relative_to(d).as_posix()
        try:
            path = f.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            path = str(f)
        row = {"file": rel, "path": path, "bytes": st.st_size, "time": st.st_mtime,
               "steps": int(f.stem) if f.stem.isdigit() else None, "role": (roles or {}).get(f.name)}
        m = members.get(f.name)
        if m is not None:
            row["league"] = {"name": m.get("name"), "games": m.get("games"), "win_rate": m.get("win_rate")}
        ev = by_ck.get(f.name if f.parent.name == "checkpoints" or f.parent.parent.name == "checkpoints" else path, [])
        if ev:
            last = ev[-1]
            row["eval"] = {"win_rate": last.get("win_rate"), "episodes": last.get("episodes"), "n": len(ev)}
        out.append(row)
    out.sort(key=lambda r: (r["steps"] is not None, r["steps"] or 0, r["time"]), reverse=True)
    return out


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _parent_key(p: dict | None) -> str | None:
    """The run name of a parent: a run's name, or bc/<dataset>."""
    if not p:
        return None
    return p["name"] if p["kind"] == "run" else f"bc/{p['name']}" if p["kind"] == "bc" else None


def _rebuilt_command(info: dict) -> str:
    """An equivalent command line for a run recorded before launches were (its options)."""
    import shlex

    args = info.get("args", {})
    parts = ["python", "-m", "warcraftsim.puffer.train"]
    for key in _ARG_ORDER:
        v = args.get(key)
        if v is None or v == "" or (key in ("buffers", "step_seconds", "minibatch") and not v):
            continue
        if key == "timesteps":
            v = int(v)
        parts += [f"--{key.replace('_', '-')}", str(v)]
    extra = [e for e in info.get("extra", []) if not e.startswith("--base.load_model_path=")]
    return shlex.join([*parts, *extra, "--name", info.get("name", "")])


class Dashboard:
    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.cache = _JsonlCache()
        self._series: dict[Path, _EpisodeSeries] = {}  # run dir -> its episodes as columns
        self._summaries: dict[Path, tuple[tuple, dict]] = {}  # run dir -> (file mtimes, summary)
        self._bc_meta: dict[str, tuple[float, dict]] = {}  # dataset -> (mtime, meta.json)
        self._runs_cache: tuple[float, list[dict]] = (0.0, [])  # the list and the sweeps from one scan

    def _sweep_dir(self, group: str) -> Path | None:
        if not group or group.startswith(".") or "/" in group or "\\" in group:
            return None
        return self.runs_dir / "sweeps" / group

    def sweeps(self) -> list[dict]:
        """Every sweep: its runs (in sweep order), description, launch, task, and when."""
        runs = self._recent_runs()
        groups: dict[str, dict] = {}
        for r in runs:
            sw = r.get("sweep")
            if not sw or not sw.get("group"):
                continue
            g = groups.setdefault(sw["group"], {"group": sw["group"], "task": r.get("task"), "of": sw.get("of"),
                                                "created": r.get("created", 0), "members": []})
            g["created"] = min(g["created"], r.get("created", 0))
            g["members"].append((sw.get("index", 0), r))
        out = []
        for g in groups.values():
            members = [r for _, r in sorted(g.pop("members"), key=lambda m: m[0])]
            g["runs"] = [r["name"] for r in members]
            d = self._sweep_dir(g["group"])
            g["notes"] = self.notes(d) if d else ""
            meta = {}
            try:
                meta = json.loads((d / "sweep.json").read_text())
            except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError, TypeError):
                pass
            if meta.get("launch"):
                g["launch"] = meta["launch"]
            else:
                infos = []
                for r in members:
                    try:
                        infos.append(json.loads((self.runs_dir / r["name"] / "run.json").read_text()))
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
                g["launch"] = {"command": _rebuilt_sweep_command(infos, g["group"]) if infos else "", "rebuilt": True}
            parents: dict[str, int] = {}
            for r in members:
                p = r.get("parent")
                key = json.dumps(p, sort_keys=True) if p else ""
                parents[key] = parents.get(key, 0) + 1
            g["parents"] = [{"parent": json.loads(k) if k else None, "runs": n} for k, n in parents.items()]
            out.append(g)
        return sorted(out, key=lambda g: g["created"], reverse=True)

    def set_sweep_notes(self, group: str, text: str) -> bool:
        d = self._sweep_dir(group)
        if d is None or len(text) > MAX_NOTES or not any(
                (r.get("sweep") or {}).get("group") == group for r in self._recent_runs()):
            return False
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "notes.md.tmp"
        tmp.write_text(text)
        tmp.replace(d / "notes.md")
        return True

    def _recent_runs(self, max_age: float = 2.0) -> list[dict]:
        """runs(), reused for a moment (the page asks for the runs and the sweeps together)."""
        t, runs = self._runs_cache
        return runs if time.time() - t < max_age else self.runs()

    def notes(self, d: Path) -> str:
        try:
            return (d / "notes.md").read_text()
        except (FileNotFoundError, UnicodeDecodeError):
            return ""

    def set_notes(self, name: str, text: str) -> bool:
        d = (self._bc_dir(name) if name.startswith("bc/") else self._collect_dir(name) if name.startswith("fullgame/")
             else self.runs_dir / name)
        if d is None or len(text) > MAX_NOTES or not (
                d.parent in (self.runs_dir / "bc", self.runs_dir / "fullgame")
                or (d.parent == self.runs_dir and (d / "run.json").exists())):
            return False
        tmp = d / "notes.md.tmp"
        tmp.write_text(text)
        tmp.replace(d / "notes.md")
        return True

    # ---- behavior cloning datasets ----------------------------------------------------------------
    def _bc_dir(self, name: str) -> Path | None:
        """runs/bc/<dataset> for the run name bc/<dataset>."""
        if not name.startswith("bc/"):
            return None
        ds = name[3:]
        if not ds or ds.startswith(".") or "/" in ds or "\\" in ds:
            return None
        d = self.runs_dir / "bc" / ds
        return d if (d / "meta.json").exists() or (d / "bc.json").exists() else None

    def _bc_info(self, d: Path) -> dict:
        def read(f):
            try:
                return json.loads((d / f).read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                return {}
        info, meta = read("bc.json"), read("meta.json")
        collect = info.get("collect") or {}
        status = info.get("status") or ("fitted" if (d / "policy.bin").exists() else "collected")
        return {**info, "kind": "bc", "name": f"bc/{d.name}", "dataset": d.name,
                "task": info.get("task") or meta.get("task"), "policy": info.get("policy") or meta.get("policy"),
                "status": status, "created": info.get("created") or (d / "meta.json").stat().st_mtime,
                "envs": collect.get("games"), "meta": meta,
                "description": ("Behavior cloning of the built-in AI's whole games (fullgame/bc.py): a transformer "
                                "over what a player sees, an order for each of its units"
                                if (info.get("task") or meta.get("task")) == "fullgame" else
                                f"Behavior cloning: the {info.get('policy') or meta.get('policy')} script's "
                                f"demonstrations, and PufferLib's network fitted to them.")}

    def _bc_summary(self, d: Path) -> dict:
        files = [f for f in [d / "bc.json", d / "meta.json", d / "fit.jsonl", d / "evals.jsonl",
                             *sorted(d.glob("episodes-*.jsonl")), *sorted(d.glob("play*.jsonl"))] if f.exists()]
        stamp = tuple((f.name, f.stat().st_mtime) for f in files)
        cached = self._summaries.get(d)
        if cached and cached[0] == stamp:
            return cached[1]
        if (_read_json(d / "bc.json") or {}).get("task") == "fullgame":
            fit = self.cache.read(d / "fit.jsonl")
            plays = self._plays(d)
            games = (_read_json(d / "bc.json") or {}).get("games") or {}
            summary = {"episodes": sum(games.values()) if isinstance(games, dict) else 0, "agent_steps": 0,
                       "win_rate_100": plays[-1]["win_rate"] if plays else None, "script_win_rate": None,
                       "fit_epochs": fit[-1]["epoch"] if fit else None,
                       "fit_acc": (fit[-1].get("val") or {}).get("order_acc") if fit else None,
                       "updated": max(f.stat().st_mtime for f in files) if files else None}
            self._summaries[d] = (stamp, summary)
            return summary
        eps = [r for f in d.glob("episodes-*.jsonl") for r in self.cache.read(f)]
        fit = self.cache.read(d / "fit.jsonl")
        evals = self.cache.read(d / "evals.jsonl")
        sampled = [e for e in evals if not e.get("greedy") and not e.get("forbid") and not e.get("script_casts")]
        summary = {
            "episodes": len(eps), "agent_steps": sum(e.get("length", 0) for e in eps),
            "script_win_rate": (sum(1 for e in eps if e.get("outcome", 0) > 0) / len(eps)) if eps else None,
            # the list's win column: the fitted policy's (latest plain evaluation)
            "win_rate_100": sampled[-1]["win_rate"] if sampled else None,
            "fit_epochs": fit[-1]["epoch"] if fit else None,
            "fit_acc": (fit[-1].get("acc") or {}).get("first") if fit else None,
            "updated": max(f.stat().st_mtime for f in files) if files else None,
        }
        self._summaries[d] = (stamp, summary)
        return summary

    def bc_runs(self) -> list[dict]:
        root = self.runs_dir / "bc"
        out = []
        if root.is_dir():
            for d in sorted(root.iterdir()):
                if (d / "meta.json").exists() or (d / "bc.json").exists():
                    info = self._bc_info(d)
                    info.pop("meta", None)
                    info["summary"] = self._bc_summary(d)
                    out.append(info)
        return out

    def _plays(self, d: Path) -> list[dict]:
        """A whole-game fit's play*.jsonl (fullgame/play.py: one row per game against the built-in
        AI), grouped into evaluations: one per play.py launch (its `eval`, or the file for older ones)."""
        groups: dict[str, list[dict]] = {}
        for f in sorted(d.glob("play*.jsonl")):
            for r in self.cache.read(f):
                groups.setdefault(r.get("eval") or f.stem, []).append(r)
        out = []
        for key, rows in groups.items():
            n = len(rows)
            res = lambda r: 0 if r.get("outcome") == "VICTORY" else 1 if r.get("outcome") == "TIE" else 2  # noqa: E731
            by_race: dict[str, list[int]] = {}
            matrix: dict[str, dict[str, list[int]]] = {}
            for r in rows:
                by_race.setdefault(r.get("race", "?"), [0, 0, 0])[res(r)] += 1
                matrix.setdefault(r.get("race", "?"), {}).setdefault(r.get("ai_race", "?"), [0, 0, 0])[res(r)] += 1
            sent = sum(r.get("orders", 0) + r.get("failed", 0) for r in rows)
            first = rows[0]
            out.append({"eval": key, "label": first.get("label") or key, "checkpoint": first.get("checkpoint"),
                        "epoch": first.get("epoch"), "difficulty": first.get("difficulty"), "map": first.get("map"),
                        "time": max(r.get("time", 0) for r in rows), "games": n,
                        "wins": sum(res(r) == 0 for r in rows), "ties": sum(res(r) == 1 for r in rows),
                        "losses": sum(res(r) == 2 for r in rows),
                        "win_rate": sum(res(r) == 0 for r in rows) / n,
                        "minutes": sum(r.get("minutes", 0) for r in rows) / n,
                        "gold": sum(((r.get("sides") or {}).get("agent") or {}).get("gold", 0) for r in rows) / n,
                        "ai_gold": sum(((r.get("sides") or {}).get("ai") or {}).get("gold", 0) for r in rows) / n,
                        "refused": sum(r.get("failed", 0) for r in rows) / max(sent, 1),
                        **{f"{pre}{k}": _mean_side(rows, side, k) for pre, side in (("", "agent"), ("ai_", "ai"))
                           for k in ("held", "food_1min")},
                        "orders": sum(r.get("orders", 0) for r in rows) / n,
                        "temperature": first.get("temperature"), "order_temperature": first.get("order_temperature"),
                        "by_race": by_race, "matrix": matrix})
        return sorted(out, key=lambda e: e["time"])

    def bc_run(self, name: str) -> dict | None:
        d = self._bc_dir(name)
        if d is None:
            return None
        info = self._bc_info(d)
        if info.get("task") == "fullgame":  # a whole-game fit (fullgame/bc.py)
            info["notes"] = self.notes(d)
            info["spaces"] = _fullgame_spaces(d, self.runs_dir)
            if info["spaces"]:
                info["spaces"] = {**info["spaces"], "from_current_code": False,
                                  "reward": "none: behavior cloning (labels: the built-in AI's orders)"}
            info["children"] = sorted(r["name"] for r in self._runs_from(name))
            info["datasets"] = [f"fullgame/{Path(x).name}" for x in str(info.get("data", "")).split()
                                if (self.runs_dir / "fullgame" / Path(x).name / "collect.json").exists()]
            media = self.cache.read(d / "media.jsonl") if (d / "media.jsonl").exists() else []
            best = info.get("best_epoch")
            fit = self.cache.read(d / "fit.jsonl")
            last_epoch = fit[-1].get("epoch") if fit else None
            roles = {"policy.pt": "the epoch with the lowest validation loss" + (f" (epoch {best})" if best else ""),
                     "last.pt": "the last epoch" + (f" (epoch {last_epoch})" if last_epoch else "")}
            plays = self._plays(d)
            ck = _checkpoints(d, self.runs_dir.parent, [], None, roles)
            for row in ck:  # its play.py evaluations
                ev = [p for p in plays if p.get("checkpoint") and Path(p["checkpoint"]).name == Path(row["file"]).name]
                if ev:
                    row["eval"] = {"win_rate": ev[-1]["win_rate"], "episodes": ev[-1]["games"], "n": len(ev)}
            return {"kind": "bc", "info": info, "fit": self.cache.read(d / "fit.jsonl"), "plays": plays, "checkpoints": ck,
                    "summary": self._bc_summary(d), "media": _latest_media(d, media),
                    "replay_hint": "the clone against the built-in AI (fullgame/play.py --videos)"}
        info["notes"] = self.notes(d)
        info["spaces"] = _spaces({**info, "obs_size": info.get("obs_size") or info["meta"].get("obs_size"),
                                  "act_sizes": info.get("act_sizes") or info["meta"].get("act_sizes")})
        info["children"] = sorted(r["name"] for r in self._runs_from(name))
        eps = sorted((r for f in d.glob("episodes-*.jsonl") for r in self.cache.read(f)),
                     key=lambda r: (r.get("time", 0), r.get("episode", 0)))
        rows = []
        for e in eps:
            row = {}
            for key, get in EPISODE_SERIES.items():
                v = get(e)
                if v is not None:
                    row[key] = float(v)
            rows.append(row)
        return {"kind": "bc", "info": info, "fit": self.cache.read(d / "fit.jsonl"),
                "checkpoints": _checkpoints(d, self.runs_dir.parent, self.cache.read(d / "evals.jsonl"), None,
                                            {"policy.pt": "the fitted network", "policy.bin": "the fitted network (PufferLib)"}),
                "episodes": _binned(list(range(1, len(rows) + 1)), rows), "evals": self.cache.read(d / "evals.jsonl"),
                "summary": self._bc_summary(d)}

    # ---- whole-game demonstrations: runs/fullgame/<name> (collect.json, games.jsonl; fullgame/collect.py)
    def _collect_dir(self, name: str) -> Path | None:
        if not name.startswith("fullgame/"):
            return None
        n = name[len("fullgame/"):]
        if not n or n.startswith(".") or "/" in n or "\\" in n:
            return None
        d = self.runs_dir / "fullgame" / n
        return d if (d / "collect.json").exists() else None

    def _collect_info(self, d: Path) -> dict:
        info = _read_json(d / "collect.json") or {}
        status = info.get("status") or "finished"
        if status == "collecting" and info.get("pid") and not _pid_alive(int(info["pid"])):
            status = "stopped"
        return {"kind": "collect", "name": f"fullgame/{d.name}", "task": "fullgame", "status": status,
                "created": info.get("time") or (d / "collect.json").stat().st_mtime, "finished": info.get("finished"),
                "map": Path(str(info.get("map", "duelrush"))).name, "planned": info.get("games"), "parallel": info.get("parallel"),
                "races": info.get("races"), "difficulty": info.get("difficulty"), "handicap": info.get("handicap"),
                "step_seconds": info.get("step_seconds"), "max_minutes": info.get("max_minutes"),
                "command": info.get("command"), "dir": f"runs/fullgame/{d.name}",
                "description": "Demonstrations: the built-in AI against itself, every step's state and the orders "
                               "it gave (fullgame/collect.py)"}

    def _collect_summary(self, d: Path) -> dict:
        f = d / "games.jsonl"
        stamp = (f.stat().st_mtime if f.exists() else 0, (d / "collect.json").stat().st_mtime)
        cached = self._summaries.get(d)
        if cached and cached[0] == stamp:
            return cached[1]
        rows = self.cache.read(f) if f.exists() else []
        recent = [r for r in rows if r["time"] > rows[-1]["time"] - 1800] if rows else []
        span = recent[-1]["time"] - recent[0]["time"] if len(recent) > 1 else 0
        summary = {"games": len(rows), "rate_per_hour": (len(recent) - 1) / span * 3600 if span > 0 else None,
                   "minutes": sum(r.get("minutes", 0) for r in rows) / len(rows) if rows else None,
                   "ties": sum(r.get("winner") is None for r in rows) / len(rows) if rows else None,
                   "updated": rows[-1]["time"] if rows else None}
        self._summaries[d] = (stamp, summary)
        return summary

    def collections(self) -> list[dict]:
        root = self.runs_dir / "fullgame"
        out = []
        if root.is_dir():
            for d in sorted(root.iterdir()):
                if (d / "collect.json").exists():
                    out.append({**self._collect_info(d), "summary": self._collect_summary(d)})
        return out

    def collection(self, name: str) -> dict | None:
        d = self._collect_dir(name)
        if d is None:
            return None
        info = self._collect_info(d)
        info["notes"] = self.notes(d)
        info["used_by"] = sorted(b["name"] for b in self.bc_runs()
                                 if any(Path(x).name == d.name for x in str(b.get("data", "")).split()))
        rows = self.cache.read(d / "games.jsonl") if (d / "games.jsonl").exists() else []
        per_game = [{"minutes": r.get("minutes", 0.0), "orders": r.get("orders") or 0,
                     "tie": 1.0 if r.get("winner") is None else 0.0} for r in rows]
        # games per hour, in 5-minute buckets of wall time
        buckets: dict[int, int] = {}
        for r in rows:
            buckets[int(r["time"] // 300)] = buckets.get(int(r["time"] // 300), 0) + 1
        rate = [{"steps": (b - min(buckets)) * 5.0 + 2.5, "rate": n * 12.0} for b, n in sorted(buckets.items())]
        # results by race: [race][other race] = [wins, ties, losses] of the first against the second
        races = ("human", "orc", "undead", "nightelf")
        matrix = {a: {b: [0, 0, 0] for b in races} for a in races}
        lengths = {a: [] for a in races}
        for r in rows:
            rc = r.get("races") or []
            if len(rc) != 2 or rc[0] not in matrix or rc[1] not in matrix:
                continue
            w = r.get("winner")
            for side in (0, 1):
                a, b = rc[side], rc[1 - side]
                matrix[a][b][0 if w == side else 1 if w is None else 2] += 1
                lengths[a].append(r.get("minutes", 0.0))
        media = self.cache.read(d / "media.jsonl") if (d / "media.jsonl").exists() else []
        return {"kind": "collect", "info": info, "summary": self._collect_summary(d),
                "media": _latest_media(d, media),
                "replay_hint": "the built-in AI against itself: the first game of a launch, at most every "
                               f"{(_read_json(d / 'collect.json') or {}).get('video_every', 10)} minutes",
                "games": _binned([float(i + 1) for i in range(len(per_game))], per_game), "rate": rate,
                "matrix": matrix, "recent": rows[-15:][::-1]}

    def parent(self, info: dict) -> dict | None:
        """Where a run started from: {"kind": "run", "name", "steps"} for another run's checkpoint,
        {"kind": "bc", "name", "policy", "task", "episodes", "win_rate"} for a fitted script."""
        init = info.get("init_from")
        if not init:
            return None
        path = Path(init)
        if not path.is_absolute():  # given relative to the repository (runs/...), or a run name
            path = self.runs_dir.parent / path
        try:
            parts = path.resolve().relative_to(self.runs_dir.resolve()).parts
        except ValueError:
            parts = (init,) if "/" not in init else ()
        if not parts:
            return {"kind": "file", "name": init}
        if parts[0] == "bc" and len(parts) > 1:
            meta_file = self.runs_dir / "bc" / parts[1] / "meta.json"
            if not meta_file.exists():  # a whole-game fit (fullgame/bc.py): its bc.json
                meta_file = self.runs_dir / "bc" / parts[1] / "bc.json"
            out = {"kind": "bc", "name": parts[1]}
            try:
                mtime = meta_file.stat().st_mtime
                cached = self._bc_meta.get(parts[1])
                if not cached or cached[0] != mtime:
                    cached = (mtime, json.loads(meta_file.read_text()))
                    self._bc_meta[parts[1]] = cached
                out.update({k: cached[1].get(k) for k in ("policy", "task", "episodes", "win_rate")})
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            return out
        steps = info.get("init_steps")
        if steps is None and len(parts) > 1 and Path(parts[-1]).stem.isdigit():
            steps = int(Path(parts[-1]).stem)
        return {"kind": "run", "name": parts[0], "steps": steps}

    def _lineage(self, runs: list[dict]) -> None:
        """Adds parent, children and notes to each run of the list."""
        children: dict[str, list[str]] = {}
        for r in runs:
            r["parent"] = self.parent(r) if r.get("kind") != "bc" else None
            key = _parent_key(r["parent"])
            if key:
                children.setdefault(key, []).append(r["name"])
        for r in runs:
            r["children"] = sorted(children.get(r.get("name"), []))

    @staticmethod
    def _files(d: Path, stem: str) -> list[Path]:
        return [f for f in sorted(d.glob(f"{stem}*.jsonl")) if f.stem == stem or f.stem.startswith(stem + "-")]

    def _merged(self, d: Path, stem: str) -> list[dict]:
        """<stem>.jsonl plus <stem>-<worker>.jsonl files, ordered by time."""
        rows: list[dict] = []
        for f in self._files(d, stem):
            rows.extend(r for r in self.cache.read(f) if "event" not in r)
        return sorted(rows, key=lambda r: r.get("time", 0))

    def lineage(self) -> dict:
        """What each run came from, as a graph: runs (their parent: a run's checkpoint or a cloning fit),
        cloning fits (the checkpoint they started from, the demonstrations they fit) and demonstration
        collections (the policy that played the takeover games). Nodes: {name, kind, track, status,
        result}; edges: {from, to, why}."""
        runs = self.runs()
        names = {r["name"] for r in runs}

        def node_of(path: str | None) -> str | None:  # a checkpoint or data path -> the run it belongs to
            if not path:
                return None
            parts = Path(str(path)).parts
            if "runs" not in parts:
                return None
            rest = parts[parts.index("runs") + 1:]
            if not rest:
                return None
            name = "/".join(rest[:2]) if rest[0] in ("bc", "fullgame") and len(rest) > 1 else rest[0]
            return name if name in names else None
        nodes, edges = [], []
        for r in runs:
            kind = r.get("kind") or "run"
            task = str(r.get("task") or "")
            track = "whole game" if task.startswith("fullgame") or kind == "collect" else "micro"
            s = r.get("summary") or {}
            result = (r.get("real_game") or {}).get("wins") if isinstance(r.get("real_game"), dict) else None
            nodes.append({"name": r["name"], "kind": kind if kind != "run" or not task.startswith("fullgame") else "selfplay",
                          "track": track, "status": r.get("status"), "win_rate": s.get("win_rate_100"),
                          "steps": s.get("agent_steps"), "result": result})
            par = r.get("parent")
            if isinstance(par, dict) and par.get("name"):
                src = ("bc/" + par["name"]) if par.get("kind") == "bc" and not par["name"].startswith("bc/") else par["name"]
                if src in names:
                    edges.append({"from": src, "to": r["name"], "why": "start"})
            if kind == "bc":
                src = node_of(r.get("init_from"))
                if src:
                    edges.append({"from": src, "to": r["name"], "why": "start"})
                for d in str(r.get("data") or "").split():
                    src = node_of(d)
                    if src:
                        edges.append({"from": src, "to": r["name"], "why": "data"})
            if kind == "collect":
                info = _read_json(self.runs_dir / r["name"] / "collect.json") or {}
                src = node_of(info.get("policy"))
                if src:
                    edges.append({"from": src, "to": r["name"], "why": "policy"})
            if kind == "match":  # its two players: runs (a run's latest checkpoint, or "run@checkpoint")
                for pl in (r.get("players") or {}).values():
                    src = str((pl or {}).get("name", "")).split("@")[0]
                    src = src if src in names else node_of((pl or {}).get("spec")) or (("bc/" + src) if "bc/" + src in names else None)
                    if src:
                        edges.append({"from": src, "to": r["name"], "why": "played"})
        seen, unique = set(), []
        for e in edges:
            k = (e["from"], e["to"])
            if k not in seen and e["from"] != e["to"]:
                seen.add(k)
                unique.append(e)
        return {"nodes": nodes, "edges": unique}

    def runs(self) -> list[dict]:
        out = []
        if not self.runs_dir.exists():
            return out
        for d in self.runs_dir.iterdir():
            info_file = d / "run.json"
            if not info_file.exists():
                continue
            try:
                info = json.loads(info_file.read_text())
            except json.JSONDecodeError:
                continue
            files = [d / "run.json", *sorted(d.glob("*.jsonl"))]
            stamp = tuple((f.name, f.stat().st_mtime) for f in files if f.exists())
            cached = self._summaries.get(d)
            if cached and cached[0] == stamp:  # nothing changed since (a finished run): reuse
                info["summary"] = cached[1]
                out.append(info)
                continue
            # the run list only needs the latest numbers: read the ends of the files, keep nothing
            tails = [[r for r in _tail_rows(f, 100) if "event" not in r] for f in d.glob("episodes*.jsonl")]
            recent = sorted((r for t in tails for r in t), key=lambda r: r.get("time", 0))[-100:]
            train = _tail_rows(d / "train.jsonl", 1)
            info["summary"] = {
                # episode numbers count per bridge worker: the total is the sum of each file's latest
                "episodes": sum(max((r.get("episode", 0) for r in t), default=0) for t in tails),
                "win_rate_100": (sum(1 for e in recent if e.get("outcome", 0) > 0) / len(recent)) if recent else None,
                "return_100": (sum(e.get("return", 0) for e in recent) / len(recent)) if recent else None,
                "agent_steps": train[-1].get("agent_steps") if train else 0,
                "sps": train[-1].get("SPS") if train else None,
                "updated": max(f.stat().st_mtime for f in [d / "run.json", *d.glob("*.jsonl")]),
            }
            self._summaries[d] = (stamp, info["summary"])
            out.append(info)
        out += self.bc_runs()
        out += self.collections()
        for info in out:
            info["notes"] = self.notes(self.runs_dir / info.get("name", ""))
            for k in ("description", "command", "spaces"):  # the list doesn't show these: less to send
                info.pop(k, None)
        self._lineage(out)
        out = sorted(out, key=lambda r: r.get("created", 0), reverse=True)
        self._runs_cache = (time.time(), out)
        return out

    def _runs_from(self, name: str) -> list[dict]:
        """The runs started from a checkpoint of `name` (a run, or bc/<dataset>)."""
        out = []
        for d in self.runs_dir.iterdir():
            try:
                info = json.loads((d / "run.json").read_text())
            except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
                continue
            if _parent_key(self.parent(info)) == name:
                out.append(info)
        return out

    def run(self, name: str) -> dict | None:
        if name.startswith("bc/"):
            return self.bc_run(name)
        if name.startswith("fullgame/"):
            return self.collection(name)
        d = self.runs_dir / name
        if not (d / "run.json").exists() or d.parent != self.runs_dir:
            return None
        info = json.loads((d / "run.json").read_text())
        info["notes"] = self.notes(d)
        info["spaces"] = _spaces(info, d, self.runs_dir)
        info["parent"] = self.parent(info)
        info["children"] = sorted(r["name"] for r in self._runs_from(name))
        if "launch" not in info:
            info["launch"] = {"run_command": _rebuilt_command(info), "rebuilt": True}
        train_rows = self.cache.read(d / "train.jsonl")
        train = [{k: v for k, v in r.items() if k in TRAIN_KEYS or k.startswith(("league/", "curriculum/", "balance/"))} for r in train_rows]
        episodes = self._merged(d, "episodes")
        series = self._series.setdefault(d, _EpisodeSeries())
        with series.lock:
            series.update([(f, self.cache.read(f)) for f in self._files(d, "episodes")])
            ep_series, sparse = series.binned(train_rows, in_order=info.get("kind") == "match")  # (a match: no trainer)
            matchups = series.matchups() if info.get("trainer") == "fullgame" else None
        # throughput: sum the bridge workers' latest rates in 5 s buckets
        buckets: dict[int, dict[int, float]] = {}
        for r in self._merged(d, "bridge"):
            buckets.setdefault(int(r["time"] // 5), {})[r.get("worker", 0)] = r.get("game_x_realtime", 0)
        bridge_times = sorted(buckets)
        bridge_steps = _interp_steps([t * 5.0 for t in bridge_times], train_rows)
        bridge = [{"time": t * 5.0, "steps": bridge_steps[i], "game_x_realtime": sum(buckets[t].values())}
                  for i, t in enumerate(bridge_times)]
        media = self._merged(d, "media")
        calib = [m for m in media if m.get("value0") is not None]
        calib_steps = _interp_steps([m["time"] for m in calib], train_rows)
        calibration = [{"steps": st, "episode": m["episode"], "value0": m["value0"], "return0": m["return0"]}
                       for st, m in zip(calib_steps, calib)]
        league = _read_json(d / "league.json")
        evals = self.cache.read(d / "evals.jsonl")
        fullgame = info.get("trainer") == "fullgame"
        production = _production(episodes[-800:]) if fullgame else None
        if fullgame:
            info["trained_seconds"] = _trained_seconds(train_rows)
            info["real_game"] = _real_game(episodes)
        return {
            "production": production,
            "matchups": matchups,
            "sparse": sparse,
            "checkpoints": _checkpoints(d, self.runs_dir.parent, evals, league,
                                        {"current.pt": "the weights the actors play with"}),
            "calibration": calibration,
            "info": info,
            "train": _binned([r.get("agent_steps", 0) for r in train], train),
            "episodes": ep_series,
            "recent_episodes": [dict(e, episode=len(episodes) - k) for k, e in enumerate(episodes[-15:][::-1])],
            "bridge": _downsample(bridge),
            "media": _latest_media(d, media),
            "evals": self.cache.read(d / "evals.jsonl"),
            "league": _read_json(d / "league.json"),
        }


def _latest_media(d: Path, media: list[dict], n: int = 40) -> list[dict]:
    """The last n media rows whose file exists, newest first; one per file (a re-rendered video
    appends another row for its file: the latest counts)."""
    seen: set[str] = set()
    out = []
    for m in reversed(media):
        if m["file"] in seen or not (d / m["file"]).exists():
            continue
        seen.add(m["file"])
        out.append(m)
        if len(out) >= n:
            break
    return out


def _page() -> bytes:
    return resources.files("warcraftsim.dashboard").joinpath("index.html").read_bytes()


def make_handler(dash: Dashboard, docs: Docs | None = None):
    docs = docs or Docs(dash.runs_dir.parent / "docs")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(json.dumps(obj).encode(), "application/json", status)

        def do_GET(self):
            path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
            if path in ("/", "/index.html", "/doc", "/lineage") or path.startswith(("/run/", "/sweep/", "/doc/")):
                return self._send(_page(), "text/html; charset=utf-8")  # (the page's own addresses: it routes)
            if path == "/api/runs":
                return self._json(dash.runs())
            if path == "/api/sweeps":
                return self._json(dash.sweeps())
            if path.startswith("/api/runs/"):
                data = dash.run(path[len("/api/runs/"):])
                return self._json(data) if data else self._json({"error": "no such run"}, 404)
            if path == "/api/lineage":
                return self._json(dash.lineage())
            if path == "/api/docs":
                return self._json(docs.list())
            if path.startswith("/api/docs/"):
                doc = docs.read(path[len("/api/docs/"):])
                return self._json(doc) if doc else self._json({"error": "no such document"}, 404)
            if path.startswith("/docs/"):  # a document's pictures (docs/media/...)
                target = (docs.root / path[len("/docs/"):]).resolve()
                if docs.root.resolve() not in target.parents or not target.is_file():
                    return self._send(b"not found", "text/plain", 404)
                return self._file(target)
            if path.startswith("/files/"):
                rel = Path(path[len("/files/"):])
                target = (dash.runs_dir / rel).resolve()
                if dash.runs_dir.resolve() not in target.parents or not target.is_file():
                    return self._send(b"not found", "text/plain", 404)
                return self._file(target)
            self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
            if path.startswith("/api/runs/") and path.endswith("/notes"):
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_NOTES * 4:
                    return self._json({"error": "too long"}, 413)
                try:
                    text = json.loads(self.rfile.read(length) or b"{}").get("notes", "")
                except (json.JSONDecodeError, AttributeError):
                    return self._json({"error": "expected {\"notes\": text}"}, 400)
                name = path[len("/api/runs/"):-len("/notes")]
                if not isinstance(text, str) or not dash.set_notes(name, text):
                    return self._json({"error": "no such run, or notes too long"}, 400)
                return self._json({"ok": True})
            if path.startswith("/api/docs/"):
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_DOC * 4:
                    return self._json({"error": "too long"}, 413)
                try:
                    text = json.loads(self.rfile.read(length) or b"{}").get("text", "")
                except (json.JSONDecodeError, AttributeError):
                    return self._json({"error": "expected {\"text\": markdown}"}, 400)
                if not isinstance(text, str) or not docs.save(path[len("/api/docs/"):], text):
                    return self._json({"error": "a name of lowercase letters, digits and dashes, and at most 512 KB"}, 400)
                return self._json({"ok": True})
            if path.startswith("/api/sweeps/") and path.endswith("/notes"):
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_NOTES * 4:
                    return self._json({"error": "too long"}, 413)
                try:
                    text = json.loads(self.rfile.read(length) or b"{}").get("notes", "")
                except (json.JSONDecodeError, AttributeError):
                    return self._json({"error": "expected {\"notes\": text}"}, 400)
                group = path[len("/api/sweeps/"):-len("/notes")]
                if not isinstance(text, str) or not dash.set_sweep_notes(group, text):
                    return self._json({"error": "no such sweep, or notes too long"}, 400)
                return self._json({"ok": True})
            self._send(b"not found", "text/plain", 404)

        def _file(self, target: Path) -> None:
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            size = target.stat().st_size
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):  # video seeking
                start_s, _, end_s = rng[6:].partition("-")
                start = int(start_s or 0)
                end = min(int(end_s) if end_s else size - 1, size - 1)
                with open(target, "rb") as f:
                    f.seek(start)
                    body = f.read(end - start + 1)
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                body = target.read_bytes()
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(runs_dir: str | os.PathLike, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(Dashboard(Path(runs_dir))))
    server.daemon_threads = True
    return server
