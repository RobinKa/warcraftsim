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


def test_a_burst_is_listed_as_itself_and_a_runs_policies_are_served(tmp_path, monkeypatch):
    """An evaluation burst's run.json is a copy of the run's whose policy it plays: the list had that run's name once
    more for each burst (fgself-236: four entries, one card). /api/policies/<run>: the run's rated policies."""
    runs = tmp_path / "runs"
    monkeypatch.setenv("WARCRAFTSIM_RATINGS_CACHE", str(tmp_path / "ratings.json"))  # (not in the runs' folder)
    src = {"name": "fgself-5", "kind": "run", "trainer": "fullgame", "task": "fullgame_(2)EchoIsles", "status": "training",
           "created": 100.0, "timesteps": 100, "envs": 1, "config": {"eval_map": "(2)EchoIsles"},
           "init_from": "runs/fgself-4/checkpoints/0000000000000009.pt",
           "launch": {"argv": ["--lr", "2e-5", "--resume"], "git": {"commit": "b" * 40, "subject": "two"}},
           "first_launch": {"argv": ["--lr", "3e-5"], "git": {"commit": "a" * 40, "subject": "one"}},
           "resumes": [{"time": 4000.0, "steps": 30, "title": "a control", "note": "why", "argv": ["--lr", "2e-5", "--resume"]}]}
    row = lambda key, x, t, opp="script:ai-insane (real)": json.dumps(  # noqa: E731
        {"eval": True, "eval_key": key, "outcome": x, "time": t, "opponent": opp, "race": "orc", "opponent_race": "human"})
    d = runs / "fgself-5"
    (d / "checkpoints").mkdir(parents=True)
    (d / "run.json").write_text(json.dumps(src))
    (d / "train.jsonl").write_text("".join(json.dumps({"time": 100.0 + 60 * k, "agent_steps": k, "uptime": 60.0 * k}) + "\n"
                                           for k in range(1, 80)))
    (d / "episodes.jsonl").write_text("\n".join([row("current", 1, 200.0), row("current", -1, 4100.0)]) + "\n")
    (d / "checkpoints" / "0000000000000040.pt").write_bytes(b"x")
    b = runs / "burst-5"
    (b / "checkpoints").mkdir(parents=True)
    (b / "run.json").write_text(json.dumps(src))  # (the copy: the run's name and all)
    (b / "burst.json").write_text(json.dumps({"current": "runs/fgself-5/checkpoints/0000000000000040.pt", "evals": []}))
    (b / "episodes-remote.jsonl").write_text("\n".join([row("current", 1, 9000.0)] * 3 + [row("current", -1, 9010.0)]) + "\n")
    server = serve(runs, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, _, body = _get(base + "/api/policies/fgself-5")  # (waits for the ratings' first scan)
        P = json.loads(body)
        assert status == 200 and P["launches"] == 2 and [p["id"] for p in P["policies"]] == ["h1", "h0", "40"]
        ck = P["policies"][2]  # (saved in hour 0: below it, the newest end first)
        assert ck["bursts"] == ["burst-5"] and ck["launch"] == 1 and ck["file"] == "checkpoints/0000000000000040.pt"
        assert ck["rated"][0]["opponents"][0]["record"] == [3, 0, 1] and ck["rated"][0]["played_by"][0]["run"] == "burst-5"
        assert _get(base + "/api/policies/fgself-5", **{"If-None-Match": _get(base + "/api/policies/fgself-5")[1]["ETag"]})[0] == 304
        assert _get(base + "/api/policies/none")[0] == 404 and _get(base + "/api/policies/bc/x")[0] == 404
        rows = {r["name"]: r for r in json.loads(_get(base + "/api/runs")[2])}
        assert set(rows) == {"fgself-5", "burst-5"}
        burst = rows["burst-5"]["burst"]
        assert (burst["policy"], burst["run"], burst["id"], burst["games"], burst["insane"]) == ("fgself-5/40", "fgself-5", "40", 4, [3, 0, 1])
        assert rows["burst-5"]["parent"] == {"kind": "run", "name": "fgself-5", "steps": 40, "policy": {"id": "40", "kind": "checkpoint"}}
        assert rows["burst-5"]["status"] == "prepared" and rows["fgself-5"]["launches"]["n"] == 2
        assert rows["fgself-5"]["launches"]["title"] == "a control" and "burst" not in rows["fgself-5"]
        page = json.loads(_get(base + "/api/runs/burst-5")[2])  # (its own page: before, the name opened the copied run's)
        assert page["info"]["name"] == "burst-5" and page["info"]["burst"]["policy"] == "fgself-5/40" and page["launches"] is None
        assert "resumes" not in page["info"] and json.loads(_get(base + "/api/policies/burst-5")[2])["burst"]["games"] == 4
        page = json.loads(_get(base + "/api/runs/fgself-5")[2])
        assert page["info"]["bursts"] == ["burst-5"] and page["info"]["children"] == []
        a, c = page["launches"]
        assert (a["commit"], a["insane"], c["title"], c["insane"]) == ("a" * 12, [1, 0, 0], "a control", [0, 0, 1])
        assert c["flags"] == [{"flag": "--lr", "from": "3e-5", "to": "2e-5"}] and c["steps"] == 30
        T = {p["name"]: p for p in json.loads(_get(base + "/api/ratings")[2])["players"]}
        assert T["fgself-5/40"]["policy"] == "40" and T["fgself-5 h1"]["policy"] == "h1" and T["fgself-5/40"]["games"] == 4
        names = {n["name"] for n in json.loads(_get(base + "/api/lineage")[2])["nodes"]}
        assert "fgself-5" in names and "burst-5" not in names  # (nothing comes from a burst)
        assert not list(runs.glob(".ratings-cache*"))
    finally:
        server.shutdown()


def test_challenge_api_checks_its_request(dash):
    import urllib.request
    base, d = dash
    status, _, body = _get(base + "/api/challenges")
    data = json.loads(body)
    assert status == 200 and set(data) == {"status", "history", "checkpoints"} and data["history"] == []

    def post(req):
        r = urllib.request.Request(base + "/api/challenge", data=json.dumps(req).encode(), method="POST")
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    assert post({"checkpoint": "../etc/passwd"}) == (400, {"ok": False, "message": "no such checkpoint"})
    (d / "checkpoints").mkdir()
    (d / "checkpoints" / "0000000000000001.pt").write_bytes(b"x")
    assert post({"checkpoint": "runs/r1/checkpoints/0000000000000001.pt", "you": "gnoll"})[1]["message"].startswith("races")
