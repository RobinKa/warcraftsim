"""Audit 15's fixes: the expansion lesson once the main mine runs low (--expand-lesson), whatever the target."""

import random
from types import SimpleNamespace

import numpy as np

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame.teacher import Teacher


def _rows(gold: int, dist: float) -> np.ndarray:
    r = np.zeros((1, fx.ROW_COLS), np.int64)
    r[0, fx.C_ID], r[0, fx.C_TYPE], r[0, fx.C_OWNER] = 7, int.from_bytes(b"ngol", "big"), 15
    r[0, fx.C_X], r[0, fx.C_Y], r[0, fx.C_RESOURCE] = int(dist), 0, gold
    return r


def test_main_mine_low_triggers_the_expansion_lesson():
    player = SimpleNamespace(start_x=0.0, start_y=0.0)
    t = Teacher(0, "human", random.Random(0), None, main_low=0.4, mine_full=6250.0)
    assert t._main_low(_rows(2000, 600.0), player, 0)  # (2000 of 6250: below 0.4)
    t = Teacher(0, "human", random.Random(0), None, main_low=0.4, mine_full=6250.0)
    assert not t._main_low(_rows(4000, 600.0), player, 0)  # (4000: above)
    t = Teacher(0, "human", random.Random(0), None, main_low=0.4, mine_full=6250.0)
    assert not t._main_low(_rows(2000, 3000.0), player, 0)  # (a mine far from its start: not its main)
    t = Teacher(0, "human", random.Random(0), None, main_low=0.0)
    assert not t._main_low(_rows(10, 600.0), player, 0)  # (off by default)


def test_surplus_workers_on_one_mine():
    """--saturation-lesson: more workers than a finished hall's mine takes (SATURATED), not chopping: surplus."""
    from warcraftsim.fullgame.teacher import SATURATED
    n = SATURATED + 3
    r = np.zeros((n + 1, fx.ROW_COLS), np.int64)
    r[:, fx.C_OWNER] = 0
    r[:n, fx.C_ID], r[:n, fx.C_TYPE] = np.arange(1, n + 1), int.from_bytes(b"hpea", "big")
    r[n, fx.C_ID], r[n, fx.C_TYPE], r[n, fx.C_FLAGS] = 99, int.from_bytes(b"htow", "big"), fx.STRUCTURE
    t = Teacher(0, "human", random.Random(0), None, lumber=True, saturation=True)
    assert t._surplus(r, set())
    assert not t._surplus(r, {1, 2, 3})  # (three of them chopping: SATURATED left on gold)
    assert not Teacher(0, "undead", random.Random(0), None, saturation=True).saturation
