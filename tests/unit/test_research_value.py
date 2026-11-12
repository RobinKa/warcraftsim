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


def test_late_start_levels_counted_once(monkeypatch):
    """A late start's placed levels (no event; Production reset at step 2) count from the start, as its placed units do,
    and the next level's event (the game's level: the placed ones in it) adds only its own cost: without them, Rhme 3 on
    a placed 2 had added 150 + 350 + 550 at its event, a windfall of the placed levels."""
    monkeypatch.setattr(selfplay, "research_values", lambda: TABLE)
    mat = {0: 1000.0, 1: 800.0}
    placed = {0: {"Rhme": 2}, 1: {}}
    before = selfplay.with_research(mat, SimpleNamespace(research={0: {}, 1: {}}), 1.0, placed)
    assert before == {0: 1000.0 + 150 + 350, 1: 800.0}
    after = selfplay.with_research(mat, SimpleNamespace(research={0: {"Rhme": 3}, 1: {"Rhar": 1}}), 1.0, placed)
    assert after[0] - before[0] == 550 and after[1] - before[1] == 200
    lower = selfplay.with_research(mat, SimpleNamespace(research={0: {"Rhar": 1}, 1: {}}), 1.0, placed)
    assert lower[0] == 1000.0 + 150 + 350 + 200  # (another research beside the placed one)
    assert selfplay.with_research(mat, SimpleNamespace(research={0: {}, 1: {}}), 0.0, placed) is mat


def test_late_start_passes_its_levels():
    """play_one's both material calls give the late start's levels (one of them alone had counted them only at times)."""
    from pathlib import Path
    src = Path(selfplay.__file__).read_text()
    assert src.count('cfg.get("research_value", 0.0), placed_research)') == 2


def test_game_tables():
    t = trace.research_values()
    assert t["Rhme"] == (150, 200)  # (iron forged swords: 100/50, then 175/175, then 250/300)
    assert trace.research_material({"Rhme": 3}, t) == 1050
