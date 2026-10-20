"""Scenarios (scenarios.py): the first step's commands set up a situation, the same value for both sides."""

import random
from types import SimpleNamespace

import pytest

from warcraftsim.fullgame import scenarios
from warcraftsim.protocol import SetResources, Spawn, SpawnHero


def _obs():
    p = lambda x: SimpleNamespace(start_x=x, start_y=0, gold=500, lumber=150)  # noqa: E731
    return SimpleNamespace(players={0: p(-5000), 1: p(5000)})


def test_scenarios_commands():
    try:
        from warcraftsim.data.objects import unit_table
        unit_table()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no game data: {e}")
    rng = random.Random(0)
    rich = scenarios.commands("rich", _obs(), 0, ["human", "orc"], rng)
    assert rich == [SetResources(0, 2000, 850), SetResources(1, 2000, 850)]
    clash = scenarios.commands("clash", _obs(), 0, ["human", "orc"], rng)
    heroes = [c for c in clash if isinstance(c, SpawnHero)]
    assert [h.player for h in heroes] == [0, 1] and all(h.level == 3 for h in heroes)
    assert heroes[0].x < 0 < heroes[1].x  # (each a third of the way from its own base)
    units = [c for c in clash if isinstance(c, Spawn)]
    assert {u.player for u in units} == {0, 1} and all(u.unit_type in scenarios.UNITS["human"] for u in units if u.player == 0)
    raid = scenarios.commands("raid", _obs(), 1, ["human", "orc"], rng)
    xs = {p: [u.x for u in raid if u.player == p] for p in (0, 1)}
    assert all(2500 < x < 4600 for x in xs[0]) and all(3500 < x < 4700 for x in xs[1])  # (both at the learner's base)


def test_weight_averaging():
    import importlib.util
    from pathlib import Path

    import torch
    spec = importlib.util.spec_from_file_location("average_weights", Path(__file__).parents[2] / "scripts" / "average_weights.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    base = {"w": torch.zeros(2), "mask": torch.tensor([True, False])}
    a, b = {"w": torch.tensor([2.0, 0.0]), "mask": base["mask"]}, {"w": torch.tensor([0.0, 4.0]), "mask": base["mask"]}
    assert m.merge([a, b])["w"].tolist() == [1.0, 2.0]  # (a soup)
    assert m.merge([a, b], base, 0.5)["w"].tolist() == [1.0, 2.0] and m.merge([a, b], base, 1.0)["w"].tolist() == [2.0, 4.0]
    assert m.merge([a, b])["mask"].tolist() == [True, False]
