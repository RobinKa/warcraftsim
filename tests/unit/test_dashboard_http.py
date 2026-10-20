"""The dashboard's answers over HTTP: the slim run list, ETags (an unchanged answer is a 304), gzip, and a run's
page kept until its files change (the page polls every 5 s, a phone too)."""

import gzip
import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from warcraftsim.dashboard.server import LIST_DROP, serve


@pytest.fixture()
def dash(tmp_path: Path):
    runs = tmp_path / "runs"
    d = runs / "r1"
    d.mkdir(parents=True)
    info = {"name": "r1", "kind": "run", "trainer": "fullgame", "task": "fullgame_echofast", "status": "training",
            "created": 1.0, "timesteps": 100, "envs": 1, "launch": {"argv": ["--x"] * 500}, "config": {"a": "b" * 2000}}
    (d / "run.json").write_text(json.dumps(info))
    (d / "train.jsonl").write_text(json.dumps({"time": 1.0, "agent_steps": 10, "SPS": 5.0}) + "\n")
    server = serve(runs, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", d
    server.shutdown()


def _get(url: str, **headers):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:  # (a 304 comes as one)
        return e.code, dict(e.headers), b""


def test_slim_list_etag_and_gzip(dash):
    base, _ = dash
    status, h, body = _get(base + "/api/runs")
    rows = json.loads(body)
    assert status == 200 and rows[0]["name"] == "r1" and not set(LIST_DROP) & set(rows[0])
    status, h2, body = _get(base + "/api/runs", **{"If-None-Match": h["ETag"]})
    assert status == 304 and body == b""
    status, h3, body = _get(base + "/api/runs/r1", **{"Accept-Encoding": "gzip"})
    assert status == 200 and h3.get("Content-Encoding") == "gzip"
    page = json.loads(gzip.decompress(body))
    assert page["info"]["launch"]["argv"][0] == "--x"  # (the run's own page keeps them)


def test_run_page_cached_until_its_files_change(dash):
    base, d = dash
    _, h1, _ = _get(base + "/api/runs/r1")
    _, h2, _ = _get(base + "/api/runs/r1")
    assert h1["ETag"] == h2["ETag"]  # (nothing changed: the same answer, no clock in it)
    time.sleep(0.01)
    with open(d / "train.jsonl", "a") as f:
        f.write(json.dumps({"time": 2.0, "agent_steps": 20, "SPS": 6.0}) + "\n")
    _, h3, body = _get(base + "/api/runs/r1")
    assert h3["ETag"] != h1["ETag"] and json.loads(body)["train"][-1]["agent_steps"] == 20
