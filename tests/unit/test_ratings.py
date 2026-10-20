"""Skill ratings from the runs' evaluation games (dashboard/ratings.py): who played whom, the fit on the Elo scale with
the insane AI at 1500, and the scan reading only what was added."""

import json
import math

import pytest

from warcraftsim.dashboard import ratings


def _row(key, outcome, t, opp="script:ai-insane (real)"):
    return json.dumps({"eval": True, "eval_key": key, "outcome": outcome, "time": t, "opponent": opp})


def test_players_of_a_row():
    p = ratings.players_of
    assert p({"eval": True, "eval_key": "current", "opponent": "script:ai-insane (real)", "time": 7300}, "fgself-9", 0) \
        == ("fgself-9 h2", "insane AI")
    assert p({"eval": True, "eval_key": "current+z", "opponent": "script:ai-insane (real)", "time": 10}, "r", 0) == ("r h0 +z", "insane AI")
    assert p({"eval": True, "eval_key": "runs/fgself-102/checkpoints/0000000011841833.pt",
              "opponent": "script:ai-insane (real)"}, "r", 0) == ("fgself-102/11841833", "insane AI")
    assert p({"eval": True, "eval_key": "vs fgself-102/0000000011841833.pt", "opponent": "eval-vs:x", "time": 5}, "r", 0) \
        == ("r h0", "fgself-102/11841833")
    assert p({"eval": True, "eval_key": "current", "opponent": "script:ai-insane"}, "r", 0) is None  # (the taxed AI)
    assert p({"eval_key": "current", "opponent": "script:ai-insane (real)"}, "r", 0) is None  # (a training game)


def test_fit_and_incremental_scan(tmp_path):
    run = tmp_path / "fgself-1"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"created": 0.0}))
    rows = [_row("current", 1, 10.0)] * 300 + [_row("current", -1, 10.0)] * 100  # 75% against the anchor
    rows += [_row("vs fgself-0/0000000000000042.pt", 1, 20.0)] * 50 + [_row("vs fgself-0/0000000000000042.pt", 0, 20.0)] * 100
    (run / "episodes.jsonl").write_text("\n".join(rows) + "\n")
    r = ratings.Ratings(tmp_path)
    assert r.scan() and not r.scan()  # (nothing new the second time)
    T = {p["name"]: p for p in r.table()["players"]}
    assert T["insane AI"]["rating"] == 1500.0 and T["insane AI"]["se"] is None
    a, b = T["fgself-1 h0"], T["fgself-0/42"]
    assert a["rating"] == pytest.approx(1500 + 400 * math.log10(3), abs=15) and a["games"] == 550
    assert 0 < a["rating"] - b["rating"] < 400 * math.log10(2) + 20  # (2 of 3 points: about a third above)
    with (run / "episodes.jsonl").open("a") as f:
        f.write(_row("current", 1, 4000.0) + "\n")
    assert r.scan() and "fgself-1 h1" in {p["name"] for p in r.table()["players"]}
    assert ratings.Ratings(tmp_path).state["files"]  # (the offsets kept for the next start)
