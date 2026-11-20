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
    assert p({"eval": True, "eval_key": "vs heuristic:orc", "opponent": "heuristic:orc", "time": 3700}, "r", 0) \
        == ("r h1", "heuristic:orc")  # (--eval-heuristic: the hand-written player)
    assert p({"eval": True, "eval_key": "current", "opponent": "script:ai-insane"}, "r", 0) is None  # (the taxed AI)
    assert p({"eval_key": "current", "opponent": "script:ai-insane (real)"}, "r", 0) is None  # (a training game)
    # a checkpoint of a run without the shops, in a run with them since they could be bought: a player of its own
    late = ratings.SHOPS_SINCE + 1
    old = lambda run: run == "fgself-102"  # noqa: E731
    assert p({"eval": True, "eval_key": "vs fgself-102/0000000011841833.pt", "opponent": "x", "time": late}, "r", late, old) \
        == ("r h0", "fgself-102/11841833 +shops")
    assert p({"eval": True, "eval_key": "runs/fgself-102/checkpoints/0000000011841833.pt", "opponent": "script:ai-insane (real)",
              "time": late - 2}, "r", 0, old) == ("fgself-102/11841833", "insane AI")


def test_fit_and_incremental_scan(tmp_path):
    run = tmp_path / "fgself-1"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"created": 0.0, "config": {"eval_map": "(2)EchoIsles"}}))
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


def test_averages_and_bursts_keep_one_name():
    """v7: a weight average is a player of its own; a burst's current policy is the checkpoint its burst.json names."""
    assert ratings.checkpoint_name("runs/fgself-236/averaged/swa-43-53.pt") == "fgself-236 avg swa-43-53"
    assert ratings.checkpoint_name("runs/fgself-236/checkpoints/0000000053275744.pt") == "fgself-236/53275744"
    row = {"eval": True, "eval_key": "current", "opponent": "script:ai-insane (real)", "outcome": 1.0, "time": 2e9}
    cur = ratings.checkpoint_name("runs/fgself-236/averaged/swa-43-53.pt")
    assert ratings.players_of(row, "burst-confirm", 0.0, current=cur) == ("fgself-236 avg swa-43-53", ratings.ANCHOR)
    vs = {**row, "eval_key": "vs runs/fgself-236/averaged/swa-43-53.pt", "opponent": "eval-vs:x"}
    assert ratings.players_of(vs, "burst-x", 0.0, current="fgself-239/26000000") == ("fgself-239/26000000", "fgself-236 avg swa-43-53")
    assert ratings.players_of(row, "fgself-238", 0.0)[0].startswith("fgself-238 h")  # (a run: by the hour, as before)
    trained = lambda run: run != "fgself-236"  # (an averaged checkpoint's run read from its name, not the whole name)
    avg = {**row, "eval_key": "runs/fgself-236/averaged/swa-43-53.pt"}
    assert ratings.players_of(avg, "burst-x", 0.0, untrained=lambda r: r == "fgself-236 avg swa-43-53")[0] == "fgself-236 avg swa-43-53"
