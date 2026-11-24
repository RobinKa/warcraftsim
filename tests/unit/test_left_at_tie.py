"""selfplay.left_at_tie: what each side has left when a game ends undecided (the episode row's "left")."""

import numpy as np
from test_spell_lesson import _unit

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame.selfplay import left_at_tie


def test_left_at_tie():
    rows = np.stack([
        _unit(1, 0, 0, 0, "uzg2", fx.STRUCTURE, vis=3),          # side 0: a nerubian tower side 1 sees
        _unit(2, 0, 0, 0, "uzg2", fx.STRUCTURE, vis=1),          # and one it does not
        _unit(3, 0, 0, 0, "ugol", fx.STRUCTURE | fx.DEAD),       # a dead building: not left
        _unit(4, 0, 0, 0, "ugho"), _unit(5, 0, 0, 0, "Udea", fx.HERO), _unit(6, 0, 0, 0, "uaco", fx.WORKER),
        _unit(7, 1, 0, 0, "emow", fx.STRUCTURE, vis=2),          # side 1: a moonwell side 0 does not see
        _unit(8, 1, 0, 0, "earc"), _unit(9, 1, 0, 0, "earc", fx.DEAD),
        _unit(10, 12, 0, 0, "nftr"),                             # a creep: nobody's
    ])
    out = left_at_tie(rows)
    assert out["0"] == {"buildings": {"uzg2": 2}, "seen": 1, "army": 2, "workers": 1}
    assert out["1"] == {"buildings": {"emow": 1}, "seen": 0, "army": 1, "workers": 0}
