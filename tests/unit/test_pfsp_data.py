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
