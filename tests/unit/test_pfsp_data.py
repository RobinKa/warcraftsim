"""--pfsp-data: a past snapshot's games with a target or in a scenario count in its record (PFSP's data); untaxed share."""
from __future__ import annotations

from pathlib import Path

from warcraftsim.fullgame.selfplay import League


def _league(tmp_path: Path, pfsp_data: float) -> League:
    lg = League(tmp_path, ["insane"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=4, pfsp="hard", win_balance=0.5)
    lg.pfsp_data = pfsp_data
    lg.add_snapshot(tmp_path / "a.pt", 1000, name="past:1000")
    return lg


def _row(**kw) -> dict:
    return {"opponent": "past:1000", "outcome": 1.0, **kw}


def test_off_as_before(tmp_path):
    lg = _league(tmp_path, 0.0)
    assert lg.win_balance is not None
    m = lg.member("past:1000")
    assert lg.recorded(_row()) is m
    assert lg.recorded(_row(scenario="base2")) is None
    assert lg.recorded(_row(z_t=1)) is None
    assert lg.recorded(_row(balance_tax=0.3)) is None
    assert lg.recorded(_row(eval=True)) is None


def test_on_counts_target_and_scenario_games_not_taxed_ones(tmp_path):
    lg = _league(tmp_path, 0.5)
    m = lg.member("past:1000")
    assert lg.recorded(_row(scenario="base2")) is m
    assert lg.recorded(_row(z_t=1)) is m
    assert lg.recorded(_row(scenario="late", z_t=1)) is m
    assert lg.recorded(_row(balance_tax=0.3)) is None  # (a taxed game still tells nothing of the snapshot's strength)
    assert lg.recorded(_row(eval=True)) is None


def test_on_leaves_other_members_as_before(tmp_path):
    lg = _league(tmp_path, 0.5)
    assert lg.recorded({"opponent": "self", "outcome": 1.0, "scenario": "base2"}) is None


def test_league_json_weight_is_the_specs_draw(tmp_path):
    """league.json's weight (the dashboard's PFSP column): each snapshot's chance as league_spec.json draws it (smoothed
    scores, the floor, the fixed members' share), not pfsp_weight of the raw score (an unplayed snapshot had shown 1.0
    and one won 4 of 4 had shown 0.0 while the spec drew them at a half's weight and at the floor)."""
    import json

    lg = _league(tmp_path, 0.5)
    lg.add_snapshot(tmp_path / "b.pt", 2000, name="past:2000")
    lg.add_snapshot(tmp_path / "c.pt", 3000, name="past:3000")
    lg.add_snapshot(tmp_path / "f.pt", 0, name="fixed:f")
    for _ in range(4):
        lg.member("past:1000").record(1.0)  # (won 4 of 4)
    for o in (1.0, -1.0, -1.0):
        lg.member("past:2000").record(o)
    lg.write()  # (past:3000 unplayed)
    spec = {a["name"]: a["p"] for a in json.loads((tmp_path / "league_spec.json").read_text())["agents"]}
    rows = {m["name"]: m for m in json.loads((tmp_path / "league.json").read_text())["members"]}
    for name in ("past:1000", "past:2000", "past:3000", "fixed:f"):
        assert rows[name]["weight"] == spec[name], name
    assert rows["script:ai-insane"]["weight"] is None
    assert 0 < rows["past:1000"]["weight"] < rows["past:3000"]["weight"] < rows["past:2000"]["weight"] < 1.0
