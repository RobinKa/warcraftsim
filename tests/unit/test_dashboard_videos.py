"""The dashboard's videos with what happened in them (dashboard/videos.py): a video's info from its trace and its
episode row (both sides' heroes, the learner's target, tags), kept beside it, listed by /api/videos and given with
a run's media rows."""

import json
import threading
import time
import urllib.request
from pathlib import Path

from warcraftsim.dashboard.server import serve
from warcraftsim.dashboard.videos import VideoIndex, summarize

MEDIA = {"time": 1792782900.0, "kind": "video", "file": "videos/game-1792782843-a1.mp4", "episode": "",
         "title": "orc vs nightelf (script:ai-insane)", "outcome": 1.0, "return": 0.7, "sub": "8.0 game minutes",
         "opponent": "script:ai-insane", "game_time": 480.0}
TRACE = {"sides": [{"player": 1, "name": "learner (orc)", "kind": "agent"}, {"player": 0, "name": "built-in AI", "kind": "ai"}],
         "target": {"t": {"oalt": 30.0, "Npbm": 150.0, "ostr": 300.0, "Otch": 600.0}, "made": {"oalt": 25.0, "Npbm": 200.0},
                    "src": "human"},
         "steps": [{"heroes": {"1": [["Npbm", 1, 1, [1, 0, 0, 0]]], "0": [["Ekee", 1, 1, [1, 0, 0, 0]]]}},
                   {"heroes": {"1": [["Npbm", 4, 1, [2, 1, 0, 0]], ["Ofar", 2, 1, [1, 0, 0, 0]]], "0": [["Ekee", 3, 1, [2, 0, 0, 0]]]}}]}
EPISODE = {"time": 1792782843.5, "race": "orc", "opponent_race": "nightelf", "opponent": "script:ai-insane",
           "game_time": 480.2, "side": 1, "picked_up": 2, "buys": {"Npbm by opeo": 1},
           "prod": {"trained": {"Npbm": 1, "Ofar": 1, "ogru": 9}, "built": {"ostr": 1}, "expansions": 1, "creeps": 3,
                    "army": 20, "workers": 18},
           "opp_prod": {"trained": {"Ekee": 1}, "built": {}, "army": 10, "workers": 12}}
NAMES = {"Npbm": "Pandaren Brewmaster", "Ofar": "Far Seer", "Ekee": "Keeper of the Grove", "oalt": "Altar of Storms"}


def _run(tmp_path: Path) -> Path:
    d = tmp_path / "runs" / "r1"
    for sub in ("videos", "replays"):
        (d / sub).mkdir(parents=True)
    (d / "videos" / "game-1792782843-a1.mp4").write_bytes(b"\0")
    (d / "replays" / "game-1792782843-a1.trace.json").write_text(json.dumps(TRACE))
    (d / "media.jsonl").write_text(json.dumps(MEDIA) + "\n")
    other = {**EPISODE, "race": "human", "game_time": 99.0}  # (another game that ended in that second)
    (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (other, EPISODE)))
    info = {"name": "r1", "kind": "run", "trainer": "fullgame", "task": "fullgame_echofast", "status": "training",
            "created": 1.0, "timesteps": 100, "envs": 1}
    (d / "run.json").write_text(json.dumps(info))
    (d / "train.jsonl").write_text(json.dumps({"time": 1.0, "agent_steps": 10, "SPS": 5.0}) + "\n")
    return d


def test_summarize_both_sides_the_target_and_tags():
    info = summarize(MEDIA, TRACE, EPISODE, NAMES)
    assert info["me"]["race"] == "orc" and info["me"]["heroes"] == [["Npbm", "Pandaren Brewmaster", 4], ["Ofar", "Far Seer", 2]]
    assert info["opp"]["heroes"] == [["Ekee", "Keeper of the Grove", 3]]
    assert (info["me"]["tier"], info["me"]["expanded"], info["me"]["picked"]) == (2, 1, 2)
    assert info["me"]["bought"] == {"Pandaren Brewmaster": 1}
    t = info["target"]
    assert [x[0] for x in t["items"]] == ["oalt", "Npbm", "ostr", "Otch"] and t["items"][1][3] == 200.0
    assert t["followed"] == round(2 / 3, 2) and t["end"] == 480.0  # (Otch due at 10:00: after the end)
    assert {"neutral hero", "tier 2", "expansion", "items picked up", "creeping", "target"} <= set(info["tags"])
    assert summarize(MEDIA, None, None, NAMES)["target"] is None  # (no trace, no row: the title's races only)


def test_index_writes_info_beside_the_video(tmp_path):
    d = _run(tmp_path)
    ix = VideoIndex(tmp_path / "runs")
    ix.fill()
    listing = ix.listing()
    assert listing["pending"] == 0 and len(listing["videos"]) == 1
    v = listing["videos"][0]
    assert v["run"] == "r1" and v["info"]["me"]["picked"] == 2  # (the orc's row, not the human's of that second)
    assert json.loads((d / "videos" / "game-1792782843-a1.info.json").read_text()) == v["info"]


def test_api_videos_and_run_media(tmp_path):
    _run(tmp_path)
    server = serve(tmp_path / "runs", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for _ in range(100):  # (the index's thread sums the video up)
            with urllib.request.urlopen(base + "/api/videos") as r:
                listing = json.loads(r.read())
            if listing["videos"] and listing["videos"][0]["info"]:
                break
            time.sleep(0.1)
        assert listing["videos"][0]["info"]["me"]["heroes"][0][0] == "Npbm"
        with urllib.request.urlopen(base + "/api/runs/r1") as r:
            page = json.loads(r.read())
        assert page["media"][0]["info"]["target"]["src"] == "human"
        with urllib.request.urlopen(base + "/videos") as r:
            assert b"nav-videos" in r.read()
    finally:
        server.shutdown()
