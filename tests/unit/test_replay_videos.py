"""Recording a replay's video again (fullgame/rerender.py, video_host.py, render_request.py's kind "replay"), the Replays
page's list (dashboard/replays.py) and the recording's parts that need no game (video.py)."""

import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from warcraftsim import paths
from warcraftsim.dashboard.replays import ReplayIndex, summarize
from warcraftsim.dashboard.server import serve
from warcraftsim.fullgame import render_request as rr
from warcraftsim.fullgame import rerender, video_host
from warcraftsim.video import _atempo, _Writer

TRACE = {"title": "r1 · policy v9 · vs built-in AI",
         "sides": [{"player": 1, "name": "learner (orc)", "kind": "agent"}, {"player": 0, "name": "built-in AI insane (human)", "kind": "ai"}],
         "outcome": {"1": 1.0, "0": -1.0}, "steps": [{"game_time": 0.5}, {"game_time": 90.0}]}
ARGV = ["--name", "r1", "--wine", "/nowhere/wc3/ge/GE/files/bin", "--fsync", "1", "--handicap", "100", "--map", "echofast",
        "--max-minutes", "40", "--step-seconds", "0.5", "--game-cuts", "1529"]


def add_replay(d: Path, stem: str, trace: dict = TRACE, key: str = "k1") -> None:
    (d / "replays").mkdir(parents=True, exist_ok=True)
    (d / "replays" / f"{stem}.w3g").write_bytes(b"x")
    (d / "replays" / f"{stem}.commands.json").write_text(json.dumps({"format": "warcraftsim-commands", "map_key": key, "commands": {}}))
    (d / "replays" / f"{stem}.trace.json").write_text(json.dumps(trace))


@pytest.fixture()
def runs(tmp_path: Path, monkeypatch) -> Path:
    runs = tmp_path / "runs"
    d = runs / "r1"
    add_replay(d, "game-100-a0")
    add_replay(d, "game-300-a1", {**TRACE, "outcome": {"1": -1.0, "0": 1.0}})
    (d / "run.json").write_text(json.dumps({"launch": {"argv": ARGV}}))
    (d / "videos").mkdir()
    (d / "videos" / "game-100-a0.mp4").write_bytes(b"old")
    (d / "media.jsonl").write_text(json.dumps({"time": 1.0, "kind": "video", "file": "videos/game-100-a0.mp4", "title": "orc vs human (x)",
                                               "outcome": 1.0, "sub": "the row", "game_time": 90.0}) + "\n")
    e = runs / "r2"
    add_replay(e, "game-200-a0", {**TRACE, "sides": [{"player": 0, "name": "learner (undead)", "kind": "agent"},
                                                     {"player": 1, "name": "snapshot past:5 (nightelf)", "kind": "agent"}],
                                  "outcome": {"0": 0.0, "1": 0.0}})
    (e / "run.json").write_text(json.dumps({"launch": {"argv": ARGV}}))
    (runs / "r2" / "replays" / "game-999-a0.w3g").write_bytes(b"x")  # (no trace or orders: not listed)
    cache = tmp_path / "cache"
    (cache / "maps").mkdir(parents=True)
    (cache / "maps" / "k1.w3x").write_bytes(b"map")
    monkeypatch.setattr(paths, "CACHE_DIR", cache)
    monkeypatch.setattr(rr, "VIDEO_HOSTS", ["box@box"])
    return runs


# ---- the Replays page's list ----------------------------------------------------------------------------------
def test_summary_from_a_trace():
    r = summarize("game-100-a0", TRACE)
    assert r == {"stem": "game-100-a0", "time": 100, "races": ["orc", "human"], "names": ["learner", "built-in AI insane"],
                 "title": TRACE["title"], "outcome": 1, "game_time": 90.0}


def test_index_list_filters_and_pages(runs, tmp_path):
    ix = ReplayIndex(runs, tmp_path / "index.json")
    ix.fill()
    assert [(r["run"], r["stem"]) for r in ix.listing()["rows"]] == [("r1", "game-300-a1"), ("r2", "game-200-a0"), ("r1", "game-100-a0")]
    assert ix.listing()["total"] == 3
    assert [r["stem"] for r in ix.listing(outcome="win")["rows"]] == ["game-100-a0"]
    assert [r["stem"] for r in ix.listing(outcome="tie")["rows"]] == ["game-200-a0"]
    assert [r["stem"] for r in ix.listing(race="undead")["rows"]] == ["game-200-a0"]
    assert [r["stem"] for r in ix.listing(q="past:5 nightelf")["rows"]] == ["game-200-a0"]
    assert ix.listing(q="past:5 orc")["matched"] == 0
    has = ix.listing(video="yes")["rows"]
    assert [r["stem"] for r in has] == ["game-100-a0"] and has[0]["video"] and has[0]["speed"] == 1.0  # (a row without a speed: real time)
    assert {r["stem"] for r in ix.listing(video="no")["rows"]} == {"game-300-a1", "game-200-a0"}
    page = ix.listing(offset=1, limit=1)
    assert [r["stem"] for r in page["rows"]] == ["game-200-a0"] and page["matched"] == 3


def test_index_reads_a_trace_once_and_follows_its_folders(runs, tmp_path):
    index = tmp_path / "index.json"
    ix = ReplayIndex(runs, index)
    ix.fill()
    for p in (runs / "r1" / "replays").glob("*.trace.json"):  # (a change in a file, not in the folder: not looked at)
        p.write_text("not json")
    ix.fill()
    assert ix.listing()["total"] == 3
    again = ReplayIndex(runs, index)  # (a new server: from its file, no traces read)
    again.fill()
    assert again.listing()["total"] == 3
    add_replay(runs / "r1", "game-400-a0")
    (runs / "r1" / "videos" / "game-400-a0.mp4").write_bytes(b"new")
    with open(runs / "r1" / "media.jsonl", "a") as f:
        f.write(json.dumps({"kind": "video", "file": "videos/game-400-a0.mp4", "recorded_speed": 4.0}) + "\n")
    ix.fill()
    row = ix.listing()["rows"][0]
    assert row["stem"] == "game-400-a0" and row["video"] and row["speed"] == 4.0
    (runs / "r2" / "replays" / "game-200-a0.trace.json").unlink()
    ix.fill()
    assert "game-200-a0" not in [r["stem"] for r in ix.listing()["rows"]]


def test_a_trace_of_another_shape_is_left_out(runs, tmp_path):
    add_replay(runs / "r1", "game-500-a0", trace={"sides": "what"})
    ix = ReplayIndex(runs, tmp_path / "index.json")
    ix.fill()
    assert ix.listing()["total"] == 3


def test_api_and_the_page(runs, tmp_path, monkeypatch):
    monkeypatch.setenv("WARCRAFTSIM_REPLAY_INDEX", str(tmp_path / "index.json"))
    (paths.CACHE_DIR / "maps" / "k1.w3x").unlink()  # (the map gone: each row says so)
    server = serve(runs, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path) as r:
            return r.status, r.read()
    try:
        assert get("/replays")[0] == 200 and b"openReplayList" in get("/replays")[1]
        d = {}
        for _ in range(100):  # (a thread fills the index)
            d = json.loads(get("/api/replays?limit=2&offset=0")[1])
            if d["total"] == 3:
                break
            threading.Event().wait(0.05)
        assert d["total"] == 3 and len(d["rows"]) == 2 and "no longer in the map cache" in d["rows"][0]["why"]
        d = json.loads(get("/api/replays?outcome=win&limit=bad")[1])
        assert [r["stem"] for r in d["rows"]] == ["game-100-a0"]
        assert json.loads(get("/api/renders")[1])["hosts"][-1]["value"] == "local"
    finally:
        server.shutdown()


# ---- a replay's settings and what it needs ------------------------------------------------------------------------
def test_cannot_play_says_why(runs):
    assert rerender.cannot_play(runs / "r1", "game-100-a0") == ""
    assert "the replay is gone" in rerender.cannot_play(runs / "r1", "game-777-a0")
    (runs / "r1" / "replays" / "game-100-a0.commands.json").unlink()
    assert "orders" in rerender.cannot_play(runs / "r1", "game-100-a0")
    (paths.CACHE_DIR / "maps" / "k1.w3x").unlink()
    assert "map cache" in rerender.cannot_play(runs / "r1", "game-300-a1")
    (runs / "r2" / "run.json").write_text("{}")
    assert "launch flags" in rerender.cannot_play(runs / "r2", "game-200-a0")


def test_a_requested_games_flags_are_its_runs(runs):
    d = runs / "renders"
    add_replay(d, "game-100-a7")
    (d / "requests").mkdir()
    (d / "requests" / "7.json").write_text(json.dumps({"id": 7, "request": {"run": "r2"}}))
    assert rerender.flags_dir(d, "game-100-a7") == runs / "r2"
    assert rerender.flags_dir(runs / "r1", "game-100-a0") == runs / "r1"
    assert rerender.cannot_play(d, "game-100-a7") == ""


def test_setup_from_the_flags():
    s = rerender.game_setup(ARGV, TRACE)
    assert s.map == "echofast" and s.step_seconds == 0.5 and s.max_game_seconds == 2400.0 and s.cuts == 1529
    # (the slots by player: the AI is player 0)
    assert [x.kind for x in s.slots] == ["ai", "agent"] and [x.race for x in s.slots] == ["human", "orc"] and s.slots[0].handicap == 100
    assert not s.audio_free  # (GameSetup's default: only video.render_replay asks for it)


def test_local_wine(tmp_path, monkeypatch):
    monkeypatch.setattr(rerender.Path, "home", lambda: tmp_path)
    (tmp_path / "wc3" / "ge" / "G" / "files" / "bin").mkdir(parents=True)
    assert rerender.local_wine("/home/other/wc3/ge/G/files/bin") == str(tmp_path / "wc3" / "ge" / "G" / "files" / "bin")
    assert rerender.local_wine("/home/other/wc3/ge/H/files/bin") == "/home/other/wc3/ge/H/files/bin"  # (not here: as it is)
    assert rerender.local_wine("/usr/bin") == "/usr/bin"


def test_install_replaces_only_now_and_says_its_speed(runs):
    d = runs / "r1"
    part = d / "videos" / "game-100-a0.part.mp4"
    part.write_bytes(b"new")
    row = rerender.install(d, "game-100-a0", part, 4.0, "box@box")
    assert (d / "videos" / "game-100-a0.mp4").read_bytes() == b"new" and not part.exists()
    assert row["recorded_speed"] == 4.0 and row["recorded_on"] == "box@box" and row["sub"] == "the row"
    last = json.loads((d / "media.jsonl").read_text().splitlines()[-1])
    assert last == row and last["time"] > 1.0
    # a game that never had a video: its row from the trace
    part = d / "videos" / "game-300-a1.part.mp4"
    part.write_bytes(b"new")
    row = rerender.install(d, "game-300-a1", part, 1.0)
    assert row["title"] == "orc vs human (built-in AI insane)" and row["outcome"] == -1.0 and row["game_time"] == 90.0


# ---- requests of kind "replay" -----------------------------------------------------------------------------------
def test_validate_a_replay_request(runs):
    q, err = rr.validate({"kind": "replay", "run": "r1", "stem": "game-100-a0", "speed": 4, "host": "box@box"}, runs)
    assert err == "" and q == {"kind": "replay", "run": "r1", "stem": "game-100-a0", "speed": 4.0, "host": "box@box"}
    q, _ = rr.validate({"kind": "replay", "run": "r1", "stem": "game-100-a0"}, runs)
    assert q["speed"] == 1.0 and q["host"] == "local"


@pytest.mark.parametrize("bad, words", [
    ({"run": "r9"}, "no run"),
    ({"run": "../r1"}, "no run"),
    ({"stem": "../../secret"}, "not a game's replay"),
    ({"stem": "game-5-a5"}, "no replay"),
    ({"speed": 3}, "speed"),
    ({"speed": "fast"}, "speed"),
    ({"host": "evil@elsewhere"}, "no host"),
])
def test_validate_a_replay_request_refuses(runs, bad, words):
    q, err = rr.validate({"kind": "replay", "run": "r1", "stem": "game-100-a0", **bad}, runs)
    assert q is None and words in err


def test_validate_a_replay_that_cannot_play(runs):
    (paths.CACHE_DIR / "maps" / "k1.w3x").unlink()
    q, err = rr.validate({"kind": "replay", "run": "r1", "stem": "game-100-a0"}, runs)
    assert q is None and "map cache" in err


def test_queue_holds_replay_requests_and_lists_the_hosts(runs, monkeypatch):
    monkeypatch.setattr(rr.subprocess, "Popen", lambda argv, **kw: None)
    ok, msg = rr.submit({"kind": "replay", "run": "r1", "stem": "game-100-a0", "speed": 2, "host": "box@box"}, runs)
    assert ok and msg == "1"
    L = rr.listing(runs)
    assert L["requests"][0]["request"]["kind"] == "replay" and L["requests"][0]["state"] == "queued"
    assert [h["value"] for h in L["hosts"]] == ["box@box", "local"]


def _request(**q):
    return {"id": 5, "time": 1.0, "state": "queued", "request": {"kind": "replay", "run": "r1", "stem": "game-100-a0",
                                                                 "speed": 1.0, "host": "local", **q}}


def test_record_replay_here_replaces_the_video_when_done(runs, monkeypatch):
    seen = []

    def render(run_dir, stem, out, speed, name):
        seen.append((Path(run_dir).name, stem, Path(out).name, speed))
        Path(out).write_bytes(b"fresh")
    monkeypatch.setattr(rerender, "render", render)
    states = []
    got = rr.record_replay(_request(speed=4.0), runs, states.append)
    assert seen == [("r1", "game-100-a0", "game-100-a0.part.mp4", 4.0)] and "rendering" in states
    assert got["video"] == "r1/videos/game-100-a0.mp4" and got["outcome"] == 1.0
    assert (runs / "r1" / "videos" / "game-100-a0.mp4").read_bytes() == b"fresh"
    assert json.loads((runs / "r1" / "media.jsonl").read_text().splitlines()[-1])["recorded_speed"] == 4.0


def test_record_replay_failing_leaves_the_old_video(runs, monkeypatch):
    def render(run_dir, stem, out, speed, name):
        Path(out).write_bytes(b"half a vid")
        raise RuntimeError("the game crashed")

    def pinned(run_dir, stem, out, speed):
        raise RuntimeError("no checkout")
    monkeypatch.setattr(rerender, "render", render)
    monkeypatch.setattr(rerender, "render_pinned", pinned)
    rows = (runs / "r1" / "media.jsonl").read_text()
    with pytest.raises(RuntimeError, match="crashed.*no checkout"):
        rr.record_replay(_request(), runs, lambda s: None)
    assert (runs / "r1" / "videos" / "game-100-a0.mp4").read_bytes() == b"old"
    assert not (runs / "r1" / "videos" / "game-100-a0.part.mp4").exists()
    assert (runs / "r1" / "media.jsonl").read_text() == rows


def test_record_replay_falls_back_on_the_runs_own_code(runs, monkeypatch):
    def render(run_dir, stem, out, speed, name):
        raise RuntimeError("this harness is not that one")

    def pinned(run_dir, stem, out, speed):
        Path(out).write_bytes(b"from its own code")
    monkeypatch.setattr(rerender, "render", render)
    monkeypatch.setattr(rerender, "render_pinned", pinned)
    rr.record_replay(_request(), runs, lambda s: None)
    assert (runs / "r1" / "videos" / "game-100-a0.mp4").read_bytes() == b"from its own code"


def test_render_pinned(runs, monkeypatch, tmp_path):
    from warcraftsim import codepin
    d = runs / "r1"
    with pytest.raises(RuntimeError, match="commit is not recorded"):
        rerender.render_pinned(d, "game-100-a0", tmp_path / "x.mp4")
    (d / "run.json").write_text(json.dumps({"launch": {"argv": ARGV, "git": {"commit": "abcdef0123456789"}}}))
    pin = tmp_path / "pin"
    (pin / "scripts").mkdir(parents=True)
    (pin / "scripts" / "rerender_fullgame.py").write_text("old script")
    monkeypatch.setattr(codepin, "prepare", lambda root, commit, **kw: pin)
    with pytest.raises(RuntimeError, match="real time only"):
        rerender.render_pinned(d, "game-100-a0", tmp_path / "x.mp4", speed=4.0)
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        Path(argv[argv.index("--out") + 1]).write_bytes(b"v")
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(rerender.subprocess, "run", run)
    rerender.render_pinned(d, "game-100-a0", tmp_path / "x.mp4")
    argv, kw = calls[0]
    assert argv[1:4] == [str(pin / "scripts" / "rerender_fullgame.py"), "r1", "game-100-a0"]
    assert kw["cwd"] == pin and kw["env"]["CUDA_VISIBLE_DEVICES"] == "" and kw["env"]["PYTHONPATH"] == str(pin)


def test_record_replay_on_a_host(runs, monkeypatch):
    calls = {}

    def on_host(host, run_dir, stem, part, speed=1.0, **kw):
        calls.update(host=host, stem=stem, speed=speed)
        Path(part).write_bytes(b"from the box")
    monkeypatch.setattr(video_host, "render_on_host", on_host)
    rr.record_replay(_request(host="box@box", speed=8.0), runs, lambda s: None)
    assert calls == {"host": "box@box", "stem": "game-100-a0", "speed": 8.0}
    assert (runs / "r1" / "videos" / "game-100-a0.mp4").read_bytes() == b"from the box"
    assert json.loads((runs / "r1" / "media.jsonl").read_text().splitlines()[-1])["recorded_on"] == "box@box"


# ---- recording on a host: what it asks of the machines -----------------------------------------------------------
class FakeSubprocess:
    def __init__(self, fail_on: str | None = None, ok_marker: bool = False):
        self.cmds: list[list[str]] = []
        self.fail_on, self.ok_marker = fail_on, ok_marker

    def __call__(self, cmd, capture_output=True, text=True, timeout=None):
        self.cmds.append(cmd)
        joined = " ".join(cmd)
        if self.fail_on and self.fail_on in joined:
            return type("P", (), {"returncode": 1, "stdout": "", "stderr": "it broke\n"})()
        if cmd[0] == "rsync" and "out/" in joined and "@" in cmd[-2]:  # (the video back)
            Path(cmd[-1]).write_bytes(b"video")
        out = "yes\n" if self.ok_marker and ".ok && touch" in joined else ""
        return type("P", (), {"returncode": 0, "stdout": out, "stderr": ""})()


@pytest.fixture()
def code_root(tmp_path):
    root = tmp_path / "checkout"
    (root / "warcraftsim").mkdir(parents=True)
    (root / "warcraftsim" / "video.py").write_text("x")
    (root / "native").mkdir()
    (root / "build" / "shim").mkdir(parents=True)
    (root / "build" / "shim" / "w3shim.dll").write_bytes(b"x")
    (root / "build" / "shim" / "w3launch.exe").write_bytes(b"x")
    return root


def test_host_mirrors_its_code_apart_and_records_under_a_lock(runs, code_root, monkeypatch, tmp_path):
    fake = FakeSubprocess()
    monkeypatch.setattr(video_host.subprocess, "run", fake)
    part = runs / "r1" / "videos" / "game-100-a0.part.mp4"
    video_host.render_on_host("box@box", runs / "r1", "game-100-a0", part, speed=4.0, root=code_root)
    assert part.read_bytes() == b"video"
    text = [" ".join(c) for c in fake.cmds]
    # only its own place on the host, not the actors' tree (only that one's Python and libraries)
    code = next(t for t in text if t.startswith("rsync") and "warcraftsim" in t and "w3shim.dll" in t)
    assert " box@box:wc3/video/code/" in code and "box@box:warcraftsim" not in code
    assert f"{code_root}/./warcraftsim" in code  # (whatever this process's folder is)
    assert not any("rsync" in t and " box@box:warcraftsim" in t for t in text)
    rec = next(t for t in text if "fullgame.rerender" in t)
    assert "flock" in rec and "nice -n 10" in rec and "CUDA_VISIBLE_DEVICES=" in rec and "timeout" in rec
    assert "--speed 4 " in rec and "--runs $HOME/wc3/video/runs" in rec
    assert any("--ignore-existing" in t and "k1.w3x" in t and ".cache/warcraftsim/maps" in t for t in text)  # (the map)
    assert any("rm -f" in t and "game-100-a0" in t for t in text)  # (its files there go)


def test_host_mirrors_its_code_once(runs, code_root, monkeypatch):
    fake = FakeSubprocess(ok_marker=True)
    monkeypatch.setattr(video_host.subprocess, "run", fake)
    video_host.mirror_code("box@box", code_root)
    assert not any(c[0] == "rsync" for c in fake.cmds)


@pytest.mark.parametrize("step", ["fullgame.rerender", "run.json", "commands.json"])
def test_host_failing_says_why_and_cleans_up(runs, code_root, monkeypatch, step):
    fake = FakeSubprocess(fail_on=step)
    monkeypatch.setattr(video_host.subprocess, "run", fake)
    part = runs / "r1" / "videos" / "game-100-a0.part.mp4"
    with pytest.raises(RuntimeError, match="it broke"):
        video_host.render_on_host("box@box", runs / "r1", "game-100-a0", part, root=code_root)
    assert not part.exists()


def test_host_refuses_a_replay_that_cannot_play(runs, code_root, monkeypatch):
    (paths.CACHE_DIR / "maps" / "k1.w3x").unlink()
    fake = FakeSubprocess()
    monkeypatch.setattr(video_host.subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="map cache"):
        video_host.render_on_host("box@box", runs / "r1", "game-100-a0", runs / "x.mp4", root=code_root)
    assert not fake.cmds


# ---- the run's renderer: its flags ------------------------------------------------------------------------------
def _job(tmp: Path):
    from warcraftsim.runtime.instance import Agent, GameSetup
    setup = GameSetup(map="echofast", slots=[Agent("orc"), Agent("human")])
    return setup, str(tmp / "replays" / "game-100-a0.w3g"), "game-100-a0", None, {"title": "t", "game_time": 90.0}, 1.0


def test_run_renderer_flag_off_changes_nothing(runs, monkeypatch):
    import warcraftsim.video
    from warcraftsim.fullgame import selfplay
    assert rr.NICE == selfplay.RENDER_NICE
    seen = {}

    def fake(setup, replay, out, **kw):
        seen.update(kw)
        return Path(out)
    monkeypatch.setattr(warcraftsim.video, "render_replay", fake)
    monkeypatch.setattr(video_host, "render_on_host", lambda *a, **k: pytest.fail("no host was asked for"))
    d = runs / "r1"
    cfg = {"run_dir": str(d), "slot": 0, "max_steps": 100}
    selfplay.render_video(cfg, _job(d))
    assert seen["speed"] == 1.0 and seen["nice"] == selfplay.RENDER_NICE
    row = json.loads((d / "media.jsonl").read_text().splitlines()[-1])
    assert "recorded_speed" not in row and "recorded_on" not in row  # (as it was: real time, here)


def test_run_renderer_on_a_host(runs, monkeypatch):
    import warcraftsim.video
    from warcraftsim.fullgame import selfplay
    calls = {}

    def on_host(host, run_dir, stem, part, speed=1.0, **kw):
        calls.update(host=host, speed=speed)
        Path(part).write_bytes(b"from the box")
    monkeypatch.setattr(video_host, "render_on_host", on_host)
    monkeypatch.setattr(warcraftsim.video, "render_replay", lambda *a, **k: pytest.fail("not here"))
    d = runs / "r1"
    cfg = {"run_dir": str(d), "slot": 0, "max_steps": 100, "video_host": "box@box", "video_speed": 4.0}
    selfplay.render_video(cfg, _job(d))
    assert calls == {"host": "box@box", "speed": 4.0}
    assert (d / "videos" / "game-100-a0.mp4").read_bytes() == b"from the box"
    row = json.loads((d / "media.jsonl").read_text().splitlines()[-1])
    assert row["recorded_speed"] == 4.0 and row["recorded_on"] == "box@box"


def test_run_renderer_a_failure_lists_no_video(runs, monkeypatch):
    from warcraftsim.fullgame import selfplay

    def on_host(*a, **k):
        raise RuntimeError("no route to host")
    monkeypatch.setattr(video_host, "render_on_host", on_host)
    d = runs / "r1"
    rows = (d / "media.jsonl").read_text()
    with pytest.raises(RuntimeError):  # (render_main logs it and goes on: one video fewer)
        selfplay.render_video({"run_dir": str(d), "slot": 0, "max_steps": 100, "video_host": "box@box"}, _job(d))
    assert (d / "media.jsonl").read_text() == rows and (d / "videos" / "game-100-a0.mp4").read_bytes() == b"old"


# ---- the recording's parts --------------------------------------------------------------------------------------
@pytest.mark.parametrize("speed, chain", [(1.0, ""), (2.0, ",atempo=2"), (1.5, ",atempo=1.5"), (4.0, ",atempo=2,atempo=2"),
                                          (8.0, ",atempo=2,atempo=2,atempo=2"), (3.0, ",atempo=2,atempo=1.5")])
def test_atempo_chains_steps_of_at_most_two(speed, chain):
    assert _atempo(speed) == chain


def test_writer_keeps_the_order_and_raises_its_failure():
    got = []
    w = _Writer(got.append, depth=2)
    for i in range(20):
        w.put(i)
    w.close()
    w.check()
    assert got == list(range(20))

    def boom(job):
        raise BrokenPipeError("ffmpeg is gone")
    w = _Writer(boom, depth=1)
    with pytest.raises(BrokenPipeError):
        for i in range(1000):  # (a failed writer goes on taking jobs off the queue: this never blocks)
            w.put(i)
    w.close()
    with pytest.raises(BrokenPipeError):
        w.check()


def test_overlay_composes_bgrx_frames_beside_its_panel(monkeypatch):
    from warcraftsim.fullgame.overlay import FullGameOverlay
    from warcraftsim.protocol import Observation
    trace = {"steps": [{"t": 0, "game_time": 0.5, "material": {"0": 1.0, "1": 1.0}, "res": {}, "orders": {}, "heroes": {},
                        "value": {}, "entropy": {}}] * 3,
             "gamma": 0.99, "sides": [{"player": 0, "name": "a (orc)", "kind": "agent"}, {"player": 1, "name": "b (human)", "kind": "agent"}],
             "outcome": {"0": 1.0, "1": -1.0}, "shaping": {}, "title": "t"}
    ov = FullGameOverlay(trace)
    ov.begin(None, {}, 8, 4)
    assert ov.size == (8 + ov.PANEL_W, 4) and ov.pix_fmt == "bgr0"
    panel = np.zeros((4, ov.PANEL_W, 3), np.uint8)
    panel[..., 0] = 200  # (red, in the panel's RGB)
    from PIL import Image
    monkeypatch.setattr(ov, "_panel", lambda t, obs: Image.fromarray(panel))
    a, b = bytes([1, 2, 3, 0]) * 32, bytes([4, 5, 6, 0]) * 32
    frames = [bytes(f) for f in ov.render_step([a, b], 0, Observation.__new__(Observation), None, [])]
    assert len(frames) == 2 and all(len(f) == 4 * 4 * (8 + ov.PANEL_W) for f in frames)
    img = np.frombuffer(frames[1], np.uint8).reshape(4, 8 + ov.PANEL_W, 4)
    assert img[0, 0, :3].tolist() == [4, 5, 6]  # (the game's frame as it came)
    assert img[0, 8, :3].tolist() == [0, 0, 200]  # (the panel's red, in BGR)
    end = ov.render_end(a, 40)
    assert len(end) == int(ov.END_SECONDS * 40) and len(end[0]) == len(frames[0])
