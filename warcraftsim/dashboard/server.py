"""Training dashboard: a small web server over the runs/ directory.

    python -m warcraftsim dashboard --port 8765        # then open http://localhost:8765

Reads what the training runs write (run.json, train.jsonl, episodes[-w].jsonl, bridge[-w].jsonl,
media[-w].jsonl from each bridge worker w, renders/, videos/) and serves it as JSON plus one
self-contained page. Episodes are placed on the trainer's step axis by their timestamps.
"""

from __future__ import annotations

import json
import mimetypes
import os
import threading
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
       for k in ("noop", "stop", "move", "attack", "attack_invalid", "attack_weakest", "focus_fire")},
    **{f"combat_{k}": (lambda k: lambda e: e.get("combat", {}).get(k))(k)
       for k in ("dealt", "taken", "kills", "losses", "focus_dealt", "focus_taken")},
}
MAX_POINTS = 600


class _JsonlCache:
    """Incrementally read JSON-lines files (they only grow)."""

    def __init__(self):
        self._files: dict[Path, tuple[int, list[dict]]] = {}
        self._lock = threading.Lock()

    def read(self, path: Path) -> list[dict]:
        with self._lock:
            offset, rows = self._files.get(path, (0, []))
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
            return rows


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


class Dashboard:
    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.cache = _JsonlCache()

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
            episodes = self._merged(d, "episodes")
            train = self.cache.read(d / "train.jsonl")
            recent = episodes[-100:]
            info["summary"] = {
                "episodes": len(episodes),
                "win_rate_100": (sum(1 for e in recent if e.get("outcome", 0) > 0) / len(recent)) if recent else None,
                "return_100": (sum(e.get("return", 0) for e in recent) / len(recent)) if recent else None,
                "agent_steps": train[-1].get("agent_steps") if train else 0,
                "sps": train[-1].get("SPS") if train else None,
                "updated": max(f.stat().st_mtime for f in [d / "run.json", *d.glob("*.jsonl")]),
            }
            out.append(info)
        return sorted(out, key=lambda r: r.get("created", 0), reverse=True)

    def run(self, name: str) -> dict | None:
        d = self.runs_dir / name
        if not (d / "run.json").exists() or d.parent != self.runs_dir:
            return None
        info = json.loads((d / "run.json").read_text())
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


def serve(runs_dir: str | os.PathLike, host: str = "0.0.0.0", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(Dashboard(Path(runs_dir))))
    server.daemon_threads = True
    return server
