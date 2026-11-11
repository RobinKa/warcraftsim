"""--mana-value: each living unit's mana counts in the shaping's material (so a wasted spell is no longer free)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from warcraftsim.fullgame import trace


def _unit(owner, mana, max_mana, dead=False, type_id=1):
    return SimpleNamespace(owner=owner, mana=mana, max_mana=max_mana, flags=trace.DEAD_FLAG if dead else 0,
                           type_id=type_id, hp=100, max_hp=100, hero_level=0)


def test_off_is_as_before():
    obs = SimpleNamespace(units=[_unit(0, 200, 300), _unit(1, 50, 100)])
    assert trace.material(obs, {}) == trace.material_units(obs, {})
    assert trace.material(obs, {}, mana_value=0.0) == trace.material_units(obs, {})


def test_units_path_counts_living_mana():
    obs = SimpleNamespace(units=[_unit(0, 200, 300), _unit(0, 80, 0), _unit(0, 90, 300, dead=True), _unit(1, 50, 100)])
    base = trace.material_units(obs, {})
    got = trace.material(obs, {}, mana_value=0.5)
    assert got[0] == base[0] + 0.5 * 200  # (a unit without max mana and a dead one count nothing)
    assert got[1] == base[1] + 0.5 * 50


def test_native_path_counts_living_mana(monkeypatch):
    # columns: 0 id, 1 type, 2 owner, 6 hp, 7 max hp, 8 mana, 9 max mana, 11 flags
    ua = np.zeros((3, 32), np.int64)
    ua[:, 6] = ua[:, 7] = 100
    ua[0, [2, 8, 9]] = (0, 200, 300)
    ua[1, [2, 8, 9]] = (1, 70, 100)
    ua[2, [2, 8, 9, 11]] = (1, 90, 100, trace.DEAD_FLAG)
    obs = SimpleNamespace(unit_array=ua, units=[])
    monkeypatch.setattr(trace, "material_units", lambda *a, **k: {0: 1000.0, 1: 500.0})
    got = trace.material(obs, {}, mana_value=1.0)
    assert got == {0: 1200.0, 1: 570.0}
