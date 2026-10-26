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
