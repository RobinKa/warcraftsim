import json

from warcraftsim.protocol import Event, EventKind, Observation, PlayerState, Race, Result, Unit, UnitFlags
from warcraftsim.record import TrajectoryRecorder, render_html


def _obs(t: int, x: int, dead: bool = False) -> Observation:
    flags = UnitFlags.DEAD if dead else UnitFlags(0)
    units = [Unit(1, 0x68666F6F, 0, x, 0, 0, 420, 420, 0, 0, 0, flags, 1, 0)]
    players = {0: PlayerState(0, Race.HUMAN, True, 500, 150, 2, 12, 0, 0, 0, 0, Result.PLAYING, 0, 0)}
    events = [Event(EventKind.DEATH, 1, 0, 0)] if dead else []
    return Observation(t, t, False, players, units, events, [])


def test_record_and_render(tmp_path):
    path = tmp_path / "ep.jsonl"
    with TrajectoryRecorder(path) as rec:
        rec.add(_obs(0, 100))
        rec.add(_obs(250, 150))
        rec.add(_obs(500, 150, dead=True))
    lines = path.read_text().splitlines()
    assert json.loads(lines[0])["format"] == "warcraftsim-trajectory"
    frames = [json.loads(line) for line in lines[1:]]
    assert [f["t"] for f in frames] == [0, 250, 500]
    assert frames[0]["u"][0][:5] == [1, "hfoo", 0, 100, 0]
    assert frames[2]["u"] == [] and frames[2]["d"] == [1]
    html = render_html(path).read_text()
    assert "<canvas" in html and '"frames"' in html
