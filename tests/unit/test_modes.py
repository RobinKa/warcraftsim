"""More kinds of strategy conditioning (modes.py): draws, the input, and each kind's violations."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from warcraftsim.fullgame import modes


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


def _row(i, owner, x, y, flags=0, typ="hfoo", level=0):
    r = np.zeros(32, np.int64)
    r[1], r[modes.C_TYPE], r[modes.C_OWNER], r[modes.C_X], r[modes.C_Y] = i, code(typ), owner, x, y
    r[modes.C_FLAGS], r[modes.C_HLEVEL] = flags, level
    return r


def test_draw_and_vector():
    rng = random.Random(0)
    for _ in range(200):
        m = modes.draw(rng, 0.6)
        assert not any(a in m and b in m for a, b in modes.EXCLUSIVE)
        for k, v in m.items():
            (lo, hi), _ = modes.PARAMS[k]
            assert lo <= v <= hi
    z = modes.vector({"no_rush": 450.0, "workers": 24.0})
    i, j = modes.KINDS.index("no_rush"), modes.KINDS.index("workers")
    assert z.shape == (modes.MDIM,) and z[2 * i] == 1 and z[2 * i + 1] == pytest.approx(0.5) and z[2 * j + 1] == pytest.approx(0.8)
    assert z.sum() == pytest.approx(1 + 0.5 + 1 + 0.8) and not modes.vector(None).any()


def test_violations():
    food = {code("hfoo"): 2}
    tk = modes.Tracker({"no_rush": 300.0, "expand": 240.0, "scout": 0.0, "workers": 4.0, "army": 4.0, "harass": 2.0},
                       0, (3000.0, 0.0), food, {code("hpea")})
    home = [_row(1, 0, -3000, 0, modes.STRUCTURE, "htow"), _row(2, 0, -2900, 0, modes.WORKER, "hpea"),
            _row(3, 0, -2800, 0)]
    assert tk.step(np.stack(home), [], 10, 60.0) == 0.0
    there = home[:2] + [_row(3, 0, 2000, 0)]  # an army unit at the enemy's base for 60 s of the 300
    assert tk.step(np.stack(there), [], 10, 120.0) == pytest.approx(0.2)  # (also near its start: scouted)
    enemy_peon = _row(9, 1, 2900, 0, modes.WORKER, "hpea")
    kill = SimpleNamespace(kind=modes.DEATH, a=9, b=3)
    tk.step(np.stack(there + [enemy_peon]), [kill], 10, 180.0)
    assert tk.kills == 1
    v = tk.step(np.stack(home), [], 10, 250.0)  # no second hall by 240: a whole violation
    assert tk.v["expand"] == 1.0 and tk.v["no_rush"] == pytest.approx(0.4)
    v = tk.step(np.stack(home + [_row(4, 0, -2700, 0, modes.WORKER, "hpea")]), [], 10, 600.0)  # 2 of 4 workers at 480
    assert tk.v["workers"] == pytest.approx(0.5) and tk.v["army"] == pytest.approx(0.5) and tk.v["harass"] == pytest.approx(0.5)
    assert v == pytest.approx(sum(tk.v.values())) and tk.report()["workers"] == [4.0, 0.5]


def test_expand_needs_a_hall_at_another_mine():
    """With its own start known, "expand" is met by a hall far from it at a gold mine, not by a second one at home."""
    tk = modes.Tracker({"expand": 240.0}, 0, (3000.0, 0.0), {}, {code("hpea")}, own_start=(-3000.0, 0.0))
    home = [_row(1, 0, -3000, 0, modes.STRUCTURE, "htow"), _row(2, 15, -3300, 0, modes.STRUCTURE, "ngol"),
            _row(3, 15, 0, 1500, modes.STRUCTURE, "ngol")]
    tk.step(np.stack(home + [_row(4, 0, -2600, 300, modes.STRUCTURE, "htow")]), [], 10, 200.0)
    assert "expand" not in tk.done
    tk.step(np.stack(home + [_row(5, 0, 300, 1400, modes.STRUCTURE, "htow")]), [], 10, 220.0)
    assert "expand" in tk.done
