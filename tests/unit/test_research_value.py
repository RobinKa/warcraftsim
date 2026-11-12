"""--research-value: the shaping's material counts what each player's finished research cost (each level)."""
from __future__ import annotations

from types import SimpleNamespace

from warcraftsim.fullgame import selfplay, trace

TABLE = {"Rhme": (150, 200), "Rhar": (200, 125)}


def test_levels_cost_each_step():
    assert trace.research_material({}, TABLE) == 0.0
    assert trace.research_material({"Rhme": 1}, TABLE) == 150
    assert trace.research_material({"Rhme": 3}, TABLE) == 150 + 350 + 550
    assert trace.research_material({"Rhme": 2, "Rhar": 1, "Xxxx": 2}, TABLE) == 150 + 350 + 200  # (unknown: nothing)


def test_with_research(monkeypatch):
    monkeypatch.setattr(selfplay, "research_values", lambda: TABLE)
    prod = SimpleNamespace(research={0: {"Rhme": 2}, 1: {"Rhar": 1}})
    mat = {0: 1000.0, 1: 800.0}
    assert selfplay.with_research(mat, prod, 0.0) is mat  # (off: as before)
    assert selfplay.with_research(mat, prod, 1.0) == {0: 1500.0, 1: 1000.0}
    assert selfplay.with_research(mat, prod, 0.5) == {0: 1250.0, 1: 900.0}


def test_game_tables():
    t = trace.research_values()
    assert t["Rhme"] == (150, 200)  # (iron forged swords: 100/50, then 175/175, then 250/300)
    assert trace.research_material({"Rhme": 3}, t) == 1050
