"""Audit 12's economy fixes: --econ-masks (no idle tech building's copy, no worker past what the mines need), the
target builder's workers (--z-workers) and the teacher's burrow and lumber lessons."""

import numpy as np
import pytest

from warcraftsim.fullgame import features as fx


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


@pytest.fixture()
def econ():
    fx.set_econ_masks(1)
    yield
    fx.set_econ_masks(0)


def _rows(units) -> np.ndarray:
    """(id, type, owner, flags[, current order]) -> the view's rows."""
    rows = np.zeros((len(units), fx.ROW_COLS), np.int64)
    for k, (i, t, owner, flags, *order) in enumerate(units):
        rows[k, fx.C_ID], rows[k, fx.C_TYPE], rows[k, fx.C_OWNER], rows[k, fx.C_FLAGS] = i, code(t), owner, flags
        rows[k, fx.C_HP] = rows[k, fx.C_MAXHP] = 500
        rows[k, fx.C_ORDER] = order[0] if order else 0
    return rows


VOCAB = {"types": [code(t) for t in ("unpl", "uslh", "ugol", "uaco", "ugho")], "current_orders": [], "upgrades": [],
         "orders": [[code("uslh"), fx.POINT], [code("usep"), fx.POINT], [code("uaco"), fx.IMMEDIATE],
                    [code("ugho"), fx.IMMEDIATE]],
         "order_names": {}}
SLH, SEP, ACO, GHO = 1, 2, 3, 4  # (their classes)
NONE = np.zeros((0, 5), np.int64)


def test_worker_cap():
    assert fx.worker_cap("undead", 1) == 8 and fx.worker_cap("undead", 2) == 13 and fx.worker_cap("undead", 0) == 3
    assert fx.worker_cap("human", 1) == 18 and fx.worker_cap("orc", 1) == 17 and fx.worker_cap("nightelf", 1) == 20


def test_econ_masks_off_by_default():
    view = fx.Encoder(VOCAB, np.zeros((5, 3), np.int64)).view(0, 1.0, [2, 0])
    rows = _rows([(1, "unpl", 0, fx.STRUCTURE), (2, "uslh", 0, fx.STRUCTURE)])
    me = np.array([0, 0, 5000, 5000, 0, 100, 0, 0, 0, 0, 0])
    assert view.step(rows, me, NONE, 0)["avail"].all()


def test_a_second_tech_building_only_while_the_first_is_at_work(econ):
    view = fx.Encoder(VOCAB, np.zeros((5, 3), np.int64)).view(0, 1.0, [2, 0])  # (undead)
    me = np.array([0, 0, 5000, 5000, 0, 100, 0, 0, 0, 0, 0])
    base = [(1, "unpl", 0, fx.STRUCTURE), (3, "ugol", 0, fx.STRUCTURE)]
    st = view.step(_rows(base), me, NONE, 0)
    assert st["avail"][SLH] and st["avail"][SEP]  # none yet
    slh = (2, "uslh", 0, fx.STRUCTURE)
    st = view.step(_rows(base + [slh]), me, NONE, 1)
    assert not st["avail"][SLH] and st["avail"][SEP]  # one idle: no second (a crypt is no tech building)
    st = view.step(_rows(base + [slh]), me, np.array([[2, fx.TRAIN_START, 2, code("uabo"), 0]], np.int64), 2)
    assert st["avail"][SLH]  # at work: a second one
    st = view.step(_rows(base + [slh, (4, "uslh", 0, fx.STRUCTURE | fx.CONSTRUCTING)]), me, NONE, 3)
    assert not st["avail"][SLH]  # the second still building: no third
    st = view.step(_rows(base + [slh, (5, "uaco", 0, fx.WORKER, code("uslh"))]), me, NONE, 4)
    assert not st["avail"][SLH]  # an acolyte on its way to build one
    st = view.step(_rows(base + [slh]), me, np.array([[5, fx.TRAIN_FINISH, 2, 0, code("uabo")]], np.int64), 5)
    assert not st["avail"][SLH]  # done training: idle again


def test_no_worker_past_the_mines_need(econ):
    view = fx.Encoder(VOCAB, np.zeros((5, 3), np.int64)).view(0, 1.0, [2, 0])
    me = np.array([0, 0, 5000, 5000, 0, 100, 0, 0, 0, 0, 0])
    base = [(1, "unpl", 0, fx.STRUCTURE), (3, "ugol", 0, fx.STRUCTURE)]
    aco = [(10 + k, "uaco", 0, fx.WORKER) for k in range(7)]
    st = view.step(_rows(base + aco), me, NONE, 0)
    assert st["avail"][ACO] and st["avail"][GHO]  # 7 of 8
    st = view.step(_rows(base + aco), me, np.array([[1, fx.TRAIN_START, 1, code("uaco"), 0]], np.int64), 1)
    assert not st["avail"][ACO] and st["avail"][GHO]  # the 8th in training
    st = view.step(_rows(base + aco + [(4, "ugol", 0, fx.STRUCTURE | fx.CONSTRUCTING)]), me, NONE, 2)
    assert st["avail"][ACO]  # a second haunted mine: 13
    enemy = [(20 + k, "uaco", 1, fx.WORKER) for k in range(20)]
    st = view.step(_rows(base + aco[:3] + enemy), me, NONE, 3)
    assert st["avail"][ACO]  # (the enemy's do not count)


TREE = """
[htow]
Upgrade=hkee
Trains=hpea
[hbar]
Trains=hfoo
"""


def _builder(monkeypatch, st, z_workers=1):
    import random

    from warcraftsim.fullgame import selfplay as sp
    from warcraftsim.fullgame import techtree
    monkeypatch.setattr(techtree, "tree", lambda: techtree.parse([TREE]))
    orders = [("hbar", fx.POINT), ("hpea", fx.IMMEDIATE), ("hfoo", fx.IMMEDIATE)]
    cfg = {"costs_array": np.asarray([[0, 0, 0], [300, 0, 0], [75, 0, 1], [135, 0, 2]]), "z_workers": z_workers}
    return sp.TargetBuilder(cfg, {"orders": [[code(c), k] for c, k in orders]}, 0, {"race": "human", **st},
                            random.Random(0))


def _obs(gold=1000, t=65.0, events=()):
    from types import SimpleNamespace as NS
    return NS(players={0: NS(gold=gold, lumber=500, food_used=10, food_cap=40)}, units=[NS(id=1, owner=0)], game_time=t,
              events=[NS(kind=k, a=a, b=code(b), c=0) for k, a, b in events])


def _hall_rows(n_workers, hall_flags=fx.STRUCTURE):
    return _rows([(1, "htow", 0, hall_flags)] + [(10 + k, "hpea", 0, fx.WORKER) for k in range(n_workers)])


def test_builder_trains_workers_up_to_its_targets(monkeypatch):
    b = _builder(monkeypatch, {"t": {}, "n": {"hpea": 9}})  # (5 + 9 = 14)
    rows = _hall_rows(6)
    c = b.commands(_obs(), {}, 10, 0.5, rows=rows)
    assert len(c) == 1 and c[0].unit == 1 and c[0].order == code("hpea") and b.log["hpea"] == [1, 0]
    assert b.last == "hpea"  # (the caller's log of accepted orders)
    assert b.commands(_obs(t=66.0), {}, 12, 0.5, rows=rows) == []  # (one a training time apart)
    o = _obs(t=71.0, events=[(5, 1, "hpea")])
    c = b.commands(o, {}, 22, 0.5, rows=rows)
    assert c == [] and b.worker_goal(o) == (7, 14)  # (the hall at work: 6 and one in training)
    b.busy.clear()
    assert b.commands(_obs(t=80.0), {}, 30, 0.5, rows=_hall_rows(14)) == []  # (as many as its target's)
    assert b.commands(_obs(gold=50, t=90.0), {}, 40, 0.5, rows=_hall_rows(6)) == []  # (nothing to pay with)
    assert b.commands(_obs(t=100.0), {}, 50, 0.5, rows=_hall_rows(6, fx.STRUCTURE | fx.CONSTRUCTING)) == []  # (no hall)
    assert len(b.commands(_obs(t=110.0), {}, 60, 0.5, rows=_hall_rows(13))) == 1


def test_builder_workers_capped_by_the_mines_and_off_by_default(monkeypatch):
    b = _builder(monkeypatch, {"t": {}, "n": {"hpea": 40}})
    o = _obs()
    b.commands(o, {}, 10, 0.5, rows=_hall_rows(6))
    assert b.worker_goal(o) == (6, fx.worker_cap("human", 1)) and fx.worker_cap("human", 1) == 18  # (one hall)
    assert _builder(monkeypatch, {"t": {}, "n": {"hpea": 9}}, z_workers=0).commands(
        _obs(), {}, 10, 0.5, rows=_hall_rows(6)) == []


def test_builder_due_items_before_workers(monkeypatch):
    """A due item first; one short of money keeps its money from the workers (held: the agent's mask's view)."""
    b = _builder(monkeypatch, {"t": {"hbar": 50.0}, "n": {"hpea": 9}})
    rows = _hall_rows(6)
    assert b.commands(_obs(gold=250), {}, 10, 0.5, rows=rows) == []  # (the barracks short: no worker with its money)
    assert b.held is not None and b.held[1] == 300
    c = b.commands(_obs(gold=400, t=66.0), {}, 11, 0.5, rows=rows)
    assert len(c) == 1 and c[0].building == code("hbar")
    c = b.commands(_obs(gold=400, t=67.0), {"hbar": 66.0}, 12, 0.5, rows=rows)
    assert len(c) == 1 and c[0].order == code("hpea")


def _placed(units) -> np.ndarray:
    """(id, type, owner, flags, x, y) -> rows."""
    rows = _rows([u[:4] for u in units])
    rows[:, fx.C_X] = [u[4] for u in units]
    rows[:, fx.C_Y] = [u[5] for u in units]
    return rows


def test_burrows_stand_down_when_no_enemy_is_near():
    import random

    from warcraftsim.fullgame import teacher as tm
    from warcraftsim.protocol import ImmediateOrder
    assert not tm.Teacher(0, "human", random.Random(0), burrows=True).burrows  # (orc only)
    tc = tm.Teacher(0, "orc", random.Random(0), burrows=True)
    base = [(1, "otrb", 0, fx.STRUCTURE, 0, 0), (2, "otrb", 0, fx.STRUCTURE, 3000, 0),
            (3, "otrb", 0, fx.STRUCTURE | fx.CONSTRUCTING, 0, 3000)]
    inside = [(10, "opeo", 0, fx.WORKER | tm.LOADED, 50, 0), (11, "opeo", 0, fx.WORKER | tm.LOADED, 40, 20),
              (12, "opeo", 0, fx.WORKER, 3000, 50)]
    assert tc._burrows(_placed(base + inside), 0) == [ImmediateOrder(1, tm.STANDDOWN)]  # (the one with peons inside)
    assert tc._burrows(_placed(base + inside), 5) == []  # (a lesson every GAP steps)
    enemy = [(30, "hfoo", 1, 0, 800, 0)]
    assert tc._burrows(_placed(base + inside + enemy), 40) == []  # (an enemy near: they stay inside)
    far = [(30, "hfoo", 1, 0, 6000, 0), (31, "htow", 1, fx.STRUCTURE, 500, 0), (32, "hfoo", 1, fx.DEAD, 100, 0)]
    assert tc._burrows(_placed(base + inside + far), 40) == [ImmediateOrder(1, tm.STANDDOWN)]
    assert tc._burrows(_placed(base + inside[2:]), 80) == []  # (none inside)
    assert tm.Teacher(0, "orc", random.Random(0))._burrows is not None and not tm.Teacher(0, "orc", random.Random(0)).burrows


def test_lumber_short_without_a_target():
    from warcraftsim.fullgame.teacher import lumber_short
    assert lumber_short(500, 150) and lumber_short(1000, 300)  # (the start; a gold-rich bank)
    assert not lumber_short(300, 300) and not lumber_short(0, 0) and not lumber_short(200, 100)
