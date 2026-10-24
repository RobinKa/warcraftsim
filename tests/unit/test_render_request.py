"""Games rendered on request (fullgame/render_request.py, the dashboard's Render button) without the game: the
requests' checks, the queue and its runner, the form's options, the targets drawn, and the dashboard's API."""

import json
import random
import threading
import urllib.request
from pathlib import Path

import pytest

from warcraftsim.dashboard.server import serve
from warcraftsim.fullgame import render_request as rr

POOL = {"items": {}, "strategies": [
    {"race": "human", "opp": "orc", "src": "ai", "won": True, "t": {"Hamg": 60.0, "hfoo": 90.0}, "sig": "Hamg|hfoo"},
    {"race": "human", "opp": "undead", "src": "ai", "won": False, "t": {"Hamg": 70.0, "hfoo": 80.0}, "sig": "Hamg|hfoo"},
    {"race": "human", "opp": "orc", "src": "ai", "won": True, "t": {"Hpal": 60.0, "hrif": 300.0}, "sig": "Hpal|hfoo+hrif"},
    {"race": "orc", "opp": "human", "src": "ai", "won": True, "t": {"Obla": 50.0, "ogru": 200.0}, "sig": "Obla|ogru"},
]}


@pytest.fixture()
def runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    for name in ("fgself-1", "fgself-2"):
        d = runs / name
        (d / "checkpoints").mkdir(parents=True)
        (d / "checkpoints" / "0000000000000100.pt").write_bytes(b"x")
        (d / "current.pt").write_bytes(b"x")
        (d / "vocab.json").write_text("{}")
    (tmp_path / "pools").mkdir()
    (tmp_path / "pools" / "real.json").write_text(json.dumps(POOL))
    cfg = {"map": "echofast", "eval_map": "(2)EchoIsles", "eval_z_pool": "pools/real.json", "z_pool": "None"}
    (runs / "fgself-1" / "run.json").write_text(json.dumps({"config": cfg}))
    (runs / "fgself-2" / "run.json").write_text(json.dumps({"config": {"map": "echofast", "z_pool": "None"}}))
    (tmp_path / "elsewhere.pt").write_bytes(b"x")
    return runs


def ok(**kw):
    return {"run": "fgself-1", "a": "current", "b": {"kind": "ai", "difficulty": "insane"}, "races": ["human", "orc"],
            "targets": ["sig:Hamg|hfoo", "none"], "follow": "a", "minutes": 5, "map": "real", **kw}


def test_validate_a_good_request_and_defaults(runs):
    q, err = rr.validate(ok(), runs)
    assert err == "" and q == {"run": "fgself-1", "a": "current", "b": {"kind": "ai", "difficulty": "insane"},
                               "races": ["human", "orc"], "targets": ["sig:Hamg|hfoo", "none"], "follow": "a",
                               "minutes": 5.0, "map": "real"}
    q, err = rr.validate({"run": "fgself-1"}, runs)  # (the form's defaults: current vs the insane AI, random races)
    assert err == "" and q["b"] == {"kind": "ai", "difficulty": "insane"} and q["races"] == ["random", "random"]
    q, _ = rr.validate(ok(targets=["random", "random"]), runs)
    assert q["targets"] == ["random", "none"]  # (the built-in AI plays its own)
    q, _ = rr.validate(ok(a="runs/fgself-2/checkpoints/0000000000000100.pt",
                          b={"kind": "policy", "policy": "current"}, follow="b", targets=["none", "random"]), runs)
    assert q["b"] == {"kind": "policy", "policy": "current"} and q["follow"] == "b" and q["targets"] == ["none", "random"]


@pytest.mark.parametrize("bad, words", [
    ({"run": "fgself-9"}, "no run"),
    ({"run": "../runs/fgself-1"}, "no run"),
    ({"a": "runs/fgself-1/checkpoints/0000000000000999.pt"}, "no policy"),
    ({"a": "elsewhere.pt"}, "no policy"),  # (outside runs/)
    ({"a": "runs/fgself-1/vocab.json"}, "no policy"),
    ({"b": {"kind": "ai", "difficulty": "hard"}}, "easy, normal or insane"),
    ({"b": {"kind": "human"}}, "the opponent"),
    ({"b": {"kind": "policy", "policy": "nope.pt"}}, "no policy"),
    ({"races": ["human"]}, "races"),
    ({"races": ["human", "naga"]}, "races"),
    ({"targets": ["sig:", "none"]}, "targets"),
    ({"targets": ["Hamg", "none"]}, "targets"),
    ({"follow": "b"}, "follows a policy"),
    ({"minutes": 0.5}, "minutes"),
    ({"minutes": 41}, "minutes"),
    ({"minutes": "soon"}, "minutes"),
    ({"map": "lordaeron"}, "map"),
    ({"run": "fgself-2", "targets": ["random", "none"]}, "no strategies"),
])
def test_validate_refuses(runs, bad, words):
    q, err = rr.validate(ok(**bad), runs)
    assert q is None and words in err


def test_queue_listing_and_a_dead_runner(runs, monkeypatch):
    started = []
    monkeypatch.setattr(rr.subprocess, "Popen", lambda argv, **kw: started.append((argv, kw)))
    assert rr.submit(ok(), runs) == (True, "1")
    assert rr.submit(ok(races=["orc", "orc"], targets=["none", "none"]), runs) == (True, "2")
    assert rr.submit(ok(minutes=99), runs)[0] is False
    assert len(started) == 2 and started[0][0][-2:] == ["--runs", str(runs)] and started[0][1]["cwd"] == runs.parent
    L = rr.listing(runs)
    assert [r["id"] for r in L["requests"]] == [2, 1] and L["queued"] == 2 and not L["runner"]
    r = rr.requests(runs)[0]
    rr.save({**r, "state": "playing"}, runs)  # (its runner gone: failed)
    assert rr.listing(runs)["requests"][1]["state"] == "failed"
    held = rr._lock(runs)  # (a runner holds the lock: no second one starts, the state stands)
    try:
        assert rr.runner_alive(runs) and rr.listing(runs)["requests"][1]["state"] == "playing"
        assert rr.submit(ok(), runs) == (True, "3") and len(started) == 2
    finally:
        held.close()


def test_runner_plays_the_queue_in_order_in_processes_of_their_own(runs, monkeypatch):
    for _ in range(3):
        rr.submit(ok(), runs, start=False)
    played = []

    def child(argv, cwd, stdout, **kw):  # (the request's process: done, or dies without a word for request 2)
        rid = int(argv[argv.index("--one") + 1])
        played.append(rid)
        r = next(x for x in rr.requests(runs) if x["id"] == rid)
        if rid == 2:
            stdout.write("Traceback (most recent call last):\nRuntimeError: the game crashed\n")
            stdout.flush()
            rr.save({**r, "state": "playing"}, runs)
            return -9
        rr.save({**r, "state": "done", "video": f"videos/game-1-a{rid}.mp4", "started": 1.0, "finished": 2.0}, runs)
        return 0
    monkeypatch.setattr(rr.subprocess, "call", child)
    assert rr.runner(runs) == 0
    rows = {r["id"]: r for r in rr.requests(runs)}
    assert played == [1, 2, 3] and rows[1]["state"] == rows[3]["state"] == "done"
    assert rows[2]["state"] == "failed" and rows[2]["error"].endswith("RuntimeError: the game crashed")
    assert not rr.runner_alive(runs)  # (it let go of its lock)


def test_options_and_builds(runs):
    o = rr.options("fgself-1", runs)
    assert [p["value"] for p in o["policies"]][:2] == ["current", "runs/fgself-1/checkpoints/0000000000000100.pt"]
    assert o["policies"][1]["label"] == "fgself-1/100" and o["policies"][-1]["label"] == "fgself-2/100 · its last"
    assert o["maps"] == {"real": "(2)EchoIsles", "training": "echofast"}
    names = {"Hamg": "Archmage", "hfoo": "Footman"}
    b = rr.builds(POOL, names)
    assert b["human"][0] == ["Hamg|hfoo", 2, 1, ["Archmage"], ["Footman"]] and b["orc"][0][0] == "Obla|ogru"
    assert b["nightelf"] == [] and rr.options("fgself-1", runs, "training")["builds"] == {}  # (no training pool)
    assert "error" in rr.options("../x", runs)


def test_pick_target():
    rng = random.Random(0)
    assert rr.pick_target(POOL, "none", "human", "orc", rng) is None
    assert rr.pick_target(POOL, "sig:Hamg|hfoo", "human", "orc", rng) is POOL["strategies"][0]  # (won vs the race)
    assert rr.pick_target(POOL, "sig:Hamg|hfoo", "human", "nightelf", rng) is POOL["strategies"][0]  # (won)
    assert rr.pick_target(POOL, "random", "orc", "human", rng) is POOL["strategies"][3]
    with pytest.raises(ValueError, match="no undead strategy"):
        rr.pick_target(POOL, "sig:Hamg|hfoo", "undead", "orc", rng)


def test_label():
    assert rr.label("current", "fgself-142") == "fgself-142 current"
    assert rr.label("runs/fgself-140/checkpoints/0000000002631004.pt", "fgself-142") == "fgself-140/2631004"


def test_api(runs, monkeypatch):
    started = []
    monkeypatch.setattr(rr.subprocess, "Popen", lambda argv, **kw: started.append(argv))
    server = serve(runs, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(body):
        req = urllib.request.Request(base + "/api/render", data=json.dumps(body).encode(), method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        assert post(ok()) == (200, {"ok": True, "message": "1"}) and len(started) == 1
        code, j = post(ok(follow="b"))
        assert code == 400 and not j["ok"] and "follows a policy" in j["message"]
        with urllib.request.urlopen(base + "/api/renders") as r:
            L = json.loads(r.read())
        assert [x["id"] for x in L["requests"]] == [1] and L["requests"][0]["state"] == "queued"
        with urllib.request.urlopen(base + "/api/render-options?run=fgself-1&map=real") as r:
            o = json.loads(r.read())
        assert o["policies"][0]["value"] == "current" and o["builds"]["human"][0][0] == "Hamg|hfoo"
    finally:
        server.shutdown()
