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


def test_handicap_ladder_rows():
    """--eval-handicap: a policy against the insane AI at 75% of its income plays "insane AI vs -25%", a player of its
    own; the untaxed anchor's games are not touched."""
    row = {"eval": True, "eval_key": "current @-25%", "opponent": "script:ai-insane (real, agent -25%)", "outcome": 1.0,
           "time": 2e9, "handicap": 25}
    assert ratings.players_of(row, "burst-x", 0.0, current="fgself-240/26000000") == ("fgself-240/26000000", "insane AI vs -25%")
    ck = {**row, "eval_key": "runs/fgself-236/averaged/swa-43-53.pt @-50%", "opponent": "script:ai-insane (real, agent -50%)"}
    assert ratings.players_of(ck, "burst-x", 0.0) == ("fgself-236 avg swa-43-53", "insane AI vs -50%")
    plain = {**row, "eval_key": "current", "opponent": "script:ai-insane (real)"}
    assert ratings.players_of(plain, "burst-x", 0.0, current="fgself-240/26000000")[1] == ratings.ANCHOR


def test_a_head_to_head_opponent_that_is_a_weight_average_is_rated():
    """selfplay tags a head-to-head game "vs <run>/<file>": an average's folder is not in it ("vs fgself-236/swa-43-53.pt"),
    which read as no checkpoint: 8,366 such games were not rated (2026-11-21)."""
    assert ratings.checkpoint_name("vs fgself-236/swa-43-53.pt") == "fgself-236 avg swa-43-53"
    assert ratings.checkpoint_name("vs fgself-236/0000000027296274.pt") == "fgself-236/27296274"
    assert ratings.checkpoint_name("runs/fgself-236/current.pt") is None and ratings.checkpoint_name("vs bc/policy.pt") is None
    row = {"eval": True, "eval_key": "vs fgself-236/swa-43-53.pt", "outcome": 1.0, "time": 10.0,
           "opponent": "eval-vs:/home/x/runs/fgself-236/averaged/swa-43-53.pt"}
    assert ratings.players_of(row, "fgself-238", 0.0) == ("fgself-238 h0", "fgself-236 avg swa-43-53")
    assert ratings.players_of(row, "burst-240", 0.0, current="fgself-240/26966747") == ("fgself-240/26966747", "fgself-236 avg swa-43-53")


def test_kinds_and_a_policys_name_in_parts():
    k, p = ratings.kind_of, ratings.policy_of
    assert [k(n) for n in ("insane AI", "insane AI, equal income", "insane AI vs -25%", "heuristic:orc", "fgself-9 h2 +z",
                           "fgself-236 avg swa-43-53", "fgself-236/53275744", "fgself-238 h3", "fgself-109/4933795 +shops")] == \
        ["anchor", "ai", "ai", "heuristic", "+z", "average", "checkpoint", "current", "checkpoint"]
    assert p("fgself-236/53275744") == {"run": "fgself-236", "id": "53275744", "kind": "checkpoint", "variant": ""}
    assert p("fgself-236 avg swa-43-53") == {"run": "fgself-236", "id": "swa-43-53", "kind": "average", "variant": ""}
    assert p("fgself-238 h3") == {"run": "fgself-238", "id": "h3", "kind": "hour", "variant": ""}
    assert p("fgself-107 h2 +z")["variant"] == "+z" and p("fgself-109/4933795 +shops") == \
        {"run": "fgself-109", "id": "4933795", "kind": "checkpoint", "variant": "+shops"}
    assert p("burst-201 h4")["run"] == "burst-201" and p("r h0 +human") == {"run": "r", "id": "h0", "kind": "hour", "variant": "+human"}
    assert all(p(n) is None for n in ("insane AI", "insane AI vs -25%", "insane AI, equal income", "heuristic:orc"))


def test_view_keeps_each_policys_records_by_folder_opponent_and_race(tmp_path):
    """What a run's Checkpoints section shows of one policy: made by the scan, read without its lock."""
    echo = {"config": {"eval_map": "(2)EchoIsles"}}
    run = tmp_path / "fgself-1"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"created": 0.0, **echo}))
    row = lambda key, x, t, opp="script:ai-insane (real)", race="orc", vs="human": json.dumps(  # noqa: E731
        {"eval": True, "eval_key": key, "outcome": x, "time": t, "opponent": opp, "race": race, "opponent_race": vs})
    (run / "episodes.jsonl").write_text("\n".join(
        [row("current", 1, 10.0)] * 3 + [row("current", -1, 20.0, race="human", vs="undead")] + [row("current", 0, 30.0)]
        + [row("vs fgself-0/swa-1-2.pt", 1, 40.0, "eval-vs:x")] * 2 + [row("vs fgself-0/swa-1-2.pt", 0, 50.0, "eval-vs:x")]) + "\n")
    # a burst: its run.json a copy of fgself-1's (the same start), its current policy fgself-1's checkpoint 7
    burst = tmp_path / "burst-1"
    burst.mkdir()
    (burst / "run.json").write_text(json.dumps({"created": 0.0, "init_from": "runs/fgself-0/checkpoints/0000000000000005.pt", **echo}))
    (burst / "burst.json").write_text(json.dumps({"current": "runs/fgself-1/checkpoints/0000000000000007.pt",
                                                  "evals": ["runs/fgself-0/averaged/swa-1-2.pt"]}))
    (burst / "episodes-remote.jsonl").write_text("\n".join(
        [row("current", 1, 5000.0)] * 2 + [row("current @-25%", -1, 5010.0, "script:ai-insane (real, agent -25%)")]
        + [row("runs/fgself-0/averaged/swa-1-2.pt", 1, 5020.0)] + [row("vs fgself-0/swa-1-2.pt", -1, 5030.0, "eval-vs:x")]) + "\n")
    elsewhere = tmp_path / "cache" / "ratings.json"
    elsewhere.parent.mkdir()
    r = ratings.Ratings(tmp_path, elsewhere)
    assert r.view is None and r.scan() and elsewhere.exists() and not (tmp_path / ratings.CACHE).exists()
    v = r.view
    T = {p["name"]: p for p in v["table"]["players"]}
    assert T["fgself-0 avg swa-1-2"]["kind"] == "average" and T["fgself-0 avg swa-1-2"]["policy"] == "swa-1-2"
    assert (T["fgself-1 h0"]["policy"], T["fgself-1/7"]["policy"], T["fgself-1/7"]["run"]) == ("h0", "7", "fgself-1")
    assert "policy" not in T["insane AI"] and T["insane AI vs -25%"]["kind"] == "ai" and T["insane AI vs -25%"]["run"] == ""
    hour = v["runs"]["fgself-1"]["h0"]
    assert (hour["kind"], hour["name"]) == ("hour", "fgself-1 h0")
    rated = hour["rated"][0]
    opp = {o["name"]: o for o in rated["opponents"]}
    assert list(opp) == ["insane AI", "fgself-0 avg swa-1-2"] and rated["games"] == 8  # (the anchor first)
    assert opp["insane AI"]["record"] == [3, 1, 1] and opp["fgself-0 avg swa-1-2"]["record"] == [2, 1, 0]  # [won, tied, lost]
    assert opp["insane AI"]["races"]["orc"]["human"] == [3, 1, 0] and opp["insane AI"]["races"]["human"]["undead"] == [0, 0, 1]
    assert (opp["fgself-0 avg swa-1-2"]["run"], opp["fgself-0 avg swa-1-2"]["policy"]) == ("fgself-0", "swa-1-2")
    assert rated["played_by"] == [{"run": "fgself-1", "games": 8, "burst": False, "first": 10.0, "last": 50.0}]
    # the average: its games from both folders, each from its own side; who lists it
    avg = v["runs"]["fgself-0"]["swa-1-2"]
    assert avg["listed_in"] == ["burst-1"] and {x["run"]: x["games"] for x in avg["rated"][0]["played_by"]} == {"fgself-1": 3, "burst-1": 2}
    assert {o["name"]: o["record"] for o in avg["rated"][0]["opponents"]} == \
        {"insane AI": [1, 0, 0], "fgself-1 h0": [0, 1, 2], "fgself-1/7": [1, 0, 0]}
    # the burst's policy by the checkpoint it plays, its rung; the burst is no fork of what its copied run.json started from
    ck = v["runs"]["fgself-1"]["7"]
    assert ck["bursts"] == ["burst-1"] and [o["name"] for o in ck["rated"][0]["opponents"]] == \
        ["insane AI", "insane AI vs -25%", "fgself-0 avg swa-1-2"]
    assert "5" not in v["runs"]["fgself-0"]
    b = v["bursts"]["burst-1"]
    assert (b["policy"], b["evals"], b["games"], b["first"], b["last"]) == ("fgself-1/7", ["fgself-0 avg swa-1-2"], 5, 5000.0, 5030.0)
    assert [x["name"] for x in b["rows"]] == ["fgself-1/7", "fgself-0 avg swa-1-2"]  # (its own policy first)
    assert b["rows"][0]["opponents"][0] == {"name": "insane AI", "kind": "anchor", "record": [2, 0, 0], "rating": 1500.0}
    # nothing new: the same view; a run started from a policy since: the view anew, though no game was read
    assert not r.scan() and r.view is v
    fork = tmp_path / "fgself-2"
    fork.mkdir()
    (fork / "run.json").write_text(json.dumps({"init_from": "runs/fgself-1/checkpoints/0000000000000007.pt",
                                               "inputs": {"init": "runs/fgself-0/checkpoints/0000000000000005.pt"}, **echo}))
    assert not r.scan() and r.view is not v
    assert r.view["runs"]["fgself-1"]["7"]["forked_by"] == ["fgself-2"] and r.view["runs"]["fgself-0"]["5"]["anchor_of"] == ["fgself-2"]
    assert r.view["runs"]["fgself-0"]["5"]["rated"] == []
    # the records come back from the cache; a file is read in pieces to the same end
    again = ratings.Ratings(tmp_path, elsewhere)
    assert not again.scan() and again.view["runs"]["fgself-1"]["h0"]["rated"][0]["opponents"] == rated["opponents"]
    f = run / "episodes.jsonl"
    whole = [x for lines, _ in ratings._new_lines(f, 0, f.stat().st_size) for x in lines]
    pieces = list(ratings._new_lines(f, 0, f.stat().st_size, step=100))
    assert len(pieces) > 3 and [x for lines, _ in pieces for x in lines if x] == [x for x in whole if x]
    assert pieces[-1][1] == f.stat().st_size and list(ratings._new_lines(f, 0, f.stat().st_size - 5, step=64))[-1][1] < f.stat().st_size


def test_variances_without_lapack():
    """The ratings' standard errors: the diagonal of the information matrix's inverse, by elimination in numpy's own
    operations (numpy's pinv had its threads spinning for ~150 CPU-seconds a fit beside the training)."""
    import numpy as np
    rng = np.random.default_rng(1)
    for n in (1, 2, 7, 60):
        a = rng.random((n, n))
        h = a @ a.T + n * np.eye(n)
        assert np.allclose(ratings._inverse_diagonal(h), np.diag(np.linalg.inv(h)), rtol=1e-10)
    assert ratings._inverse_diagonal(np.array([[1.0, 2.0], [2.0, 1.0]])) is None  # (not positive definite)


def test_soup_names():
    from warcraftsim.dashboard.ratings import checkpoint_name, policy_of
    assert checkpoint_name("runs/fullgame/soups/soup-2026-11-26.pt") == "soup avg soup-2026-11-26"
    assert checkpoint_name("vs soups/soup-2026-11-26.pt") == "soup avg soup-2026-11-26"
    assert checkpoint_name("vs fullgame/soup-2026-11-26.pt") == "soup avg soup-2026-11-26"
    assert checkpoint_name("vs fullgame/other.pt") is None
    assert policy_of("soup avg soup-2026-11-26")["kind"] == "average"


def test_soup_is_trained_on_shops(tmp_path):
    from warcraftsim.dashboard.ratings import Ratings
    assert Ratings(tmp_path)._neutral("soup") > 0
