import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame.selfplay import League, Trajectory  # noqa: E402


class _Q(list):
    def put(self, x):
        self.append(x)


def _step(value: float) -> tuple[dict, dict]:
    st = {"n": 2, "n_own": 1, "ent": np.zeros((2, fx.F), np.float32), "type": np.zeros(2, np.int64),
          "cur": np.zeros(2, np.int64), "glob": np.zeros(4, np.float32)}
    res = {"order": np.zeros(1, np.int64), "tgt": np.zeros(1, np.int64), "bx": np.zeros(1, np.int64),
           "by": np.zeros(1, np.int64), "logp": np.zeros(1, np.float32), "value": value, "version": 0}
    return st, res


def test_trajectory_gae_chunks_and_terminal_reward():
    q = _Q()
    tr = Trajectory({"chunk": 2, "gamma": 0.5, "lam": 1.0}, q, {})
    for v in (0.0, 0.0, 0.8, 0.0):
        tr.add(*_step(v))
    tr.end(1.0)
    steps = [s for msg in q for s in msg["steps"]]
    assert len(steps) == 4  # every step once, the chunk boundary's bootstrap step not twice
    # the first chunk bootstraps from the value at its end (0.8), the last one ends in the win
    assert [round(s["ret"], 4) for s in steps] == [0.2, 0.4, 0.5, 1.0]
    assert steps[-1]["done"] and not steps[0]["done"]


def test_league_spec(tmp_path):
    lg = League(tmp_path, ["normal"], {"ai": 0.25, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard")
    spec = lg.spec()
    assert sum(o["p"] for o in spec["launch"]) == pytest.approx(1.0)
    assert spec["agents"] == [{"name": "self", "kind": "self", "p": 1.0}]  # no snapshots yet
    for steps in (0, 10, 20):
        lg.add_snapshot(tmp_path / f"{steps}.pt", steps)
    assert [m.name for m in lg.past] == ["past:0", "past:20"]  # the first and the newest
    lg.member("past:0").record(1.0)
    lg.member("script:ai-normal").record(-1.0)
    spec = lg.spec()
    assert sum(o["p"] for o in spec["agents"]) == pytest.approx(1.0)
    lg.write()
    summary = json.loads((tmp_path / "league.json").read_text())
    assert {m["name"] for m in summary["members"]} == {"script:ai-normal", "past:0", "past:20"}
    assert lg.train_keys()["league/script:ai-normal"] == 0.0


def test_dashboard_collection_and_plays(tmp_path):
    from warcraftsim.dashboard.server import Dashboard

    d = tmp_path / "fullgame" / "demos-x"
    d.mkdir(parents=True)
    (d / "collect.json").write_text(json.dumps({"games": 4, "map": "duelrush", "status": "finished", "time": 1.0}))
    rows = [{"time": 100.0 + i, "game": i, "races": ["human", "orc"], "winner": w, "minutes": 2.0, "orders": 10}
            for i, w in enumerate((0, 1, None))]
    (d / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    dash = Dashboard(tmp_path)
    (c,) = dash.collections()
    assert c["name"] == "fullgame/demos-x" and c["summary"]["games"] == 3
    detail = dash.run("fullgame/demos-x")
    assert detail["matrix"]["human"]["orc"] == [1, 1, 1] and detail["matrix"]["orc"]["human"] == [1, 1, 1]
    b = tmp_path / "bc" / "fit-x"
    b.mkdir(parents=True)
    (b / "bc.json").write_text(json.dumps({"task": "fullgame", "status": "fitted", "data": str(d), "games": {"train": 3},
                                               "created": 1.0}))
    plays = [{"eval": "e1", "outcome": o, "race": "human", "ai_race": "orc", "minutes": 2, "orders": 5, "failed": 1,
              "time": 5.0} for o in ("VICTORY", "DEFEAT")]
    (b / "play.jsonl").write_text("".join(json.dumps(r) + "\n" for r in plays))
    fit = dash.run("bc/fit-x")
    assert fit["info"]["datasets"] == ["fullgame/demos-x"]
    (p,) = fit["plays"]
    assert (p["games"], p["wins"], p["losses"], p["win_rate"]) == (2, 1, 1, 0.5)
    assert p["matrix"]["human"]["orc"] == [1, 0, 1]


def test_describe_spaces_names_orders():
    code = lambda s: int.from_bytes(s.encode(), "big")  # noqa: E731
    vocab = {"types": [code("hpea")], "current_orders": [851986], "upgrades": [code("Rhma")],
             "orders": [[851983, fx.UNIT], [code("hpea"), fx.IMMEDIATE], [code("Rhma"), fx.IMMEDIATE],
                        [code("hbar"), fx.POINT]],
             "order_names": {"851983": "attack"}}
    sp = fx.describe_spaces(vocab)
    order = sp["actions"]["heads"][0]
    assert order["options"] == ["none", "attack (unit)", "train hpea", "research Rhma", "build hbar"]
    assert order["size"] == 5 and [h["name"] for h in sp["actions"]["heads"]][1:] == ["target unit (pointer)", "point x", "point y (given x)"]
    glob = sp["observation"]["blocks"][0]
    assert len(glob["features"]) == fx.Encoder(vocab).G  # every global feature named
    assert len(sp["observation"]["blocks"][1]["features"]) == fx.F


def test_material_potential():
    from types import SimpleNamespace as U

    from warcraftsim.fullgame.selfplay import potential
    foot, hall = int.from_bytes(b"hfoo", "big"), int.from_bytes(b"htow", "big")
    values = {str(foot): 135, str(hall): 590}
    obs = U(units=[U(flags=0, owner=0, type_id=foot, hp=210, max_hp=420),       # half a footman: 67.5
                   U(flags=2, owner=1, type_id=hall, hp=1500, max_hp=1500),     # a whole town hall: 590
                   U(flags=1024 | 2, owner=0, type_id=hall, hp=0, max_hp=1500),  # dead: nothing
                   U(flags=0, owner=12, type_id=foot, hp=420, max_hp=420)])     # neutral: nothing
    assert potential(obs, 0, values, 100.0) == pytest.approx((67.5 - 590) / 100)
    assert potential(obs, 1, values, 100.0) == pytest.approx(-(67.5 - 590) / 100)  # zero-sum
