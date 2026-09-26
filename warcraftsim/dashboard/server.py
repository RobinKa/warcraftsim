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

TRAIN_KEYS = ("agent_steps", "SPS", "epoch", "uptime", "env/win_rate", "env/loss_rate", "env/episode_return",
              "env/episode_length", "env/n", "loss/policy", "loss/value", "loss/entropy", "loss/kl",
              "loss/old_kl", "loss/clipfrac", "importance", "perf/rollout", "perf/eval_env", "perf/eval_model",
              "perf/eval_copy", "perf/train", "util/gpu_percent", "util/vram_used_gb", "util/cpu_mem_gb", "time")
# per-episode series (rolling means): name -> value of an episode row (None: not recorded)
EPISODE_SERIES = {
    "win_rate": lambda e: 1.0 if e.get("outcome", 0) > 0 else 0.0,
    "loss_rate": lambda e: 1.0 if e.get("outcome", 0) < 0 else 0.0,
    "draw_rate": lambda e: 1.0 if e.get("outcome", 0) == 0 else 0.0,
    "return": lambda e: float(e.get("return", 0)),
    "length": lambda e: float(e.get("length", 0)),
    "game_time": lambda e: e.get("game_time"),
    **{f"act_{k}": (lambda k: lambda e: e.get("act", {}).get(k))(k)
       for k in ("noop", "stop", "retreat", "move", "attack", "cast", "attack_invalid", "attack_weakest",
                 "focus_fire", "cast_invalid")},
    **{f"combat_{k}": (lambda k: lambda e: e.get("combat", {}).get(k))(k)
       for k in ("dealt", "taken", "kills", "losses", "focus_dealt", "focus_taken")},
}
MAX_POINTS = 600
MAX_NOTES = 64 * 1024
# train.py options in the order its command line gives them (a command rebuilt for older runs)
_ARG_ORDER = ("task", "envs", "workers", "timesteps", "step_seconds", "horizon", "minibatch", "replay_ratio",
              "buffers", "lr", "ent_coef", "gamma", "hidden", "layers", "checkpoint_interval", "record_every",
              "video_every", "init_from")


class _JsonlCache:
    """Incrementally read JSON-lines files (they only grow). Keeps the `max_files` most recently
    read files: the runs being looked at, not every run ever."""

    def __init__(self, max_files: int = 64):
        self._files: "OrderedDict[Path, tuple[int, list[dict]]]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_files = max_files

    def read(self, path: Path) -> list[dict]:
        with self._lock:
            offset, rows = self._files.get(path, (0, []))
            if path in self._files:
                self._files.move_to_end(path)
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                return []
            if size < offset:  # rewritten
                offset, rows = 0, []
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
                self._files[path] = (offset, rows)
                while len(self._files) > self.max_files:
                    self._files.popitem(last=False)
            return rows


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


def _spaces(info: dict) -> dict | None:
    """A run's observation and action spaces: recorded at launch, else described by the current
    code (flagged, and whether its sizes match the run's)."""
    if info.get("spaces"):
        return info["spaces"]
    task = info.get("task")
    if not task:
        return None
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
        d = self._bc_dir(name) if name.startswith("bc/") else self.runs_dir / name
        if d is None or len(text) > MAX_NOTES or not (
                d.parent == self.runs_dir / "bc" or (d.parent == self.runs_dir and (d / "run.json").exists())):
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
                "description": f"Behavior cloning: the {info.get('policy') or meta.get('policy')} script's "
                               f"demonstrations, and PufferLib's network fitted to them."}

    def _bc_summary(self, d: Path) -> dict:
        files = [f for f in [d / "bc.json", d / "meta.json", d / "fit.jsonl", d / "evals.jsonl",
                             *sorted(d.glob("episodes-*.jsonl"))] if f.exists()]
        stamp = tuple((f.name, f.stat().st_mtime) for f in files)
        cached = self._summaries.get(d)
        if cached and cached[0] == stamp:
            return cached[1]
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

    def bc_run(self, name: str) -> dict | None:
        d = self._bc_dir(name)
        if d is None:
            return None
        info = self._bc_info(d)
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
                "episodes": _binned(list(range(1, len(rows) + 1)), rows), "evals": self.cache.read(d / "evals.jsonl"),
                "summary": self._bc_summary(d)}

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

    def _merged(self, d: Path, stem: str) -> list[dict]:
        """<stem>.jsonl plus <stem>-<worker>.jsonl files, ordered by time."""
        rows: list[dict] = []
        for f in sorted(d.glob(f"{stem}*.jsonl")):
            if f.stem == stem or f.stem.startswith(stem + "-"):
                rows.extend(r for r in self.cache.read(f) if "event" not in r)
        return sorted(rows, key=lambda r: r.get("time", 0))

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
        d = self.runs_dir / name
        if not (d / "run.json").exists() or d.parent != self.runs_dir:
            return None
        info = json.loads((d / "run.json").read_text())
        info["notes"] = self.notes(d)
        info["spaces"] = _spaces(info)
        info["parent"] = self.parent(info)
        info["children"] = sorted(r["name"] for r in self._runs_from(name))
        if "launch" not in info:
            info["launch"] = {"run_command": _rebuilt_command(info), "rebuilt": True}
        train_rows = self.cache.read(d / "train.jsonl")
        train = [{k: r[k] for k in TRAIN_KEYS if k in r} for r in train_rows]
        episodes = self._merged(d, "episodes")
        steps = _interp_steps([e["time"] for e in episodes], train_rows)
        ep_rows = []
        for e in episodes:
            row = {}
            for name, get in EPISODE_SERIES.items():
                v = get(e)
                if v is not None:
                    row[name] = float(v)
            ep_rows.append(row)
        ep_series = _binned(steps, ep_rows)
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
        return {
            "calibration": calibration,
            "info": info,
            "train": _binned([r.get("agent_steps", 0) for r in train], train),
            "episodes": ep_series,
            "recent_episodes": [dict(e, episode=len(episodes) - k) for k, e in enumerate(episodes[-15:][::-1])],
            "bridge": _downsample(bridge),
            "media": [m for m in media if (d / m["file"]).exists()][-40:][::-1],
            "evals": self.cache.read(d / "evals.jsonl"),
        }


def _page() -> bytes:
    return resources.files("warcraftsim.dashboard").joinpath("index.html").read_bytes()


def make_handler(dash: Dashboard):
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
            if path in ("/", "/index.html"):
                return self._send(_page(), "text/html; charset=utf-8")
            if path == "/api/runs":
                return self._json(dash.runs())
            if path == "/api/sweeps":
                return self._json(dash.sweeps())
            if path.startswith("/api/runs/"):
                data = dash.run(path[len("/api/runs/"):])
                return self._json(data) if data else self._json({"error": "no such run"}, 404)
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
