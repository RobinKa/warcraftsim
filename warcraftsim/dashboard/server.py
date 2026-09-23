"""Training dashboard: a small web server over the runs/ directory.

    python -m warcraftsim dashboard --port 8765        # then open http://localhost:8765

Reads what the training runs write (run.json, train.jsonl, episodes.jsonl, bridge.jsonl,
media.jsonl, renders/, videos/) and serves it as JSON plus one self-contained page.
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
              "loss/clipfrac", "util/gpu_percent", "util/vram_used_gb", "time")
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


def _rolling(values: list[float], window: int) -> list[float]:
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


class Dashboard:
    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.cache = _JsonlCache()

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
            episodes = self.cache.read(d / "episodes.jsonl")
            train = self.cache.read(d / "train.jsonl")
            recent = episodes[-100:]
            info["summary"] = {
                "episodes": len(episodes),
                "win_rate_100": (sum(1 for e in recent if e.get("outcome", 0) > 0) / len(recent)) if recent else None,
                "return_100": (sum(e.get("return", 0) for e in recent) / len(recent)) if recent else None,
                "agent_steps": train[-1].get("agent_steps") if train else 0,
                "sps": train[-1].get("SPS") if train else None,
                "updated": max((d / f).stat().st_mtime for f in ("run.json", "episodes.jsonl", "train.jsonl")
                               if (d / f).exists()),
            }
            out.append(info)
        return sorted(out, key=lambda r: r.get("created", 0), reverse=True)

    def run(self, name: str) -> dict | None:
        d = self.runs_dir / name
        if not (d / "run.json").exists() or d.parent != self.runs_dir:
            return None
        info = json.loads((d / "run.json").read_text())
        train = [{k: r[k] for k in TRAIN_KEYS if k in r} for r in self.cache.read(d / "train.jsonl")]
        episodes = self.cache.read(d / "episodes.jsonl")
        window = max(10, min(100, len(episodes) // 20 or 10))
        wins = _rolling([1.0 if e.get("outcome", 0) > 0 else 0.0 for e in episodes], window)
        rets = _rolling([float(e.get("return", 0)) for e in episodes], window)
        lens = _rolling([float(e.get("length", 0)) for e in episodes], window)
        ep_series = [{"episode": e["episode"], "steps": e.get("total_steps", 0), "time": e["time"],
                      "win_rate": wins[i], "return": rets[i], "length": lens[i]} for i, e in enumerate(episodes)]
        media = self.cache.read(d / "media.jsonl")
        return {
            "info": info,
            "train": _downsample(train),
            "episodes": _downsample(ep_series),
            "episode_window": window,
            "recent_episodes": episodes[-15:][::-1],
            "bridge": _downsample(self.cache.read(d / "bridge.jsonl")),
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
