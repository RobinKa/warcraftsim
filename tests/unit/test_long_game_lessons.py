"""Audit 16's long-game lessons (teacher.py): --army-lesson (anti-air, tier units), --upgrade-lesson, --creep-lesson."""

import random
from types import SimpleNamespace

import numpy as np

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import teacher as tm
from warcraftsim.fullgame.teacher import ATTACK, Teacher
from warcraftsim.protocol import Build, ImmediateOrder, PointOrder


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


def _unit(i: int, owner: int, x: int, y: int, typ: str, flags: int = 0, vis: int = 3, hp: int = 400,
          maxhp: int = 400, level: int = 0) -> np.ndarray:
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = i, code(typ), owner, x, y
    r[fx.C_HP], r[fx.C_MAXHP], r[fx.C_FLAGS], r[fx.C_VIS], r[fx.C_HLEVEL] = hp, maxhp, flags, vis, level
    return r


S = fx.STRUCTURE
PLAYER = SimpleNamespace(gold=1000, lumber=500, food_used=30, food_cap=60, start_x=-2000.0, start_y=0.0)


def _base(hall: str = "htow", army=("hfoo",) * 6, buildings=("hbar",), extra=()):
    rows = [_unit(1, 0, -2000, 0, hall, S), _unit(2, 0, -2000, 300, "hpea", fx.WORKER)]
    rows += [_unit(10 + k, 0, -1600 + 50 * k, 400, b, S) for k, b in enumerate(buildings)]
    rows += [_unit(30 + k, 0, -1500, 50 * k, u) for k, u in enumerate(army)]
    return np.stack(rows + list(extra))


def _orders(rows, now=300.0, can=None, busy=(), levels=None, t=100, **kw):
    tc = Teacher(0, "human", random.Random(0), **kw)
    tc.busy, tc.levels = set(busy), dict(levels or {})
    return tc, [c for c in tc.commands(rows, PLAYER, now, t, can) if not isinstance(c, Build)]  # (not the altar's)


def test_army_lesson_anti_air():
    gryphon = _unit(90, 1, 3000, 3000, "hgry", vis=1)  # (seen by side 0)
    tc, out = _orders(_base(extra=[gryphon]), army=True)
    assert out == [ImmediateOrder(10, code("hrif"))] and tc.taught["anti-air"] == 1
    assert _orders(_base(extra=[gryphon]))[1] == []  # (off)
    assert _orders(_base(extra=[_unit(90, 1, 3000, 3000, "hgry", vis=2)]), army=True)[1] == []  # (unseen)
    assert _orders(_base(extra=[gryphon]), army=True, busy={10})[1] == []  # (the barracks busy)
    assert _orders(_base(extra=[gryphon]), army=True, can=lambda u, k=fx.IMMEDIATE: u != "hrif")[1] == []  # (masked)
    # (enough of the army hits air: 4 riflemen against one air unit)
    assert _orders(_base(army=("hfoo",) * 4 + ("hrif",) * 4, extra=[gryphon]), army=True)[1] == []
    # (the aviary's dragonhawks: the anti-air unit it has least of, riflemen it has)
    out = _orders(_base(army=("hfoo",) * 4 + ("hrif",) * 2, buildings=("hbar", "hgra"), extra=[gryphon]), army=True)[1]
    assert out == [ImmediateOrder(11, code("hdhw"))]


def test_army_lesson_tier_unit():
    can = lambda u, k=fx.IMMEDIATE: u in ("hsor", "hmpr")  # noqa: E731 (no knights: castle)
    out = _orders(_base("hkee", ("hfoo",) * 8, ("hbar", "hars")), can=can, army=True)[1]
    assert out == [ImmediateOrder(11, code("hsor"))]
    assert _orders(_base("htow", ("hfoo",) * 8, ("hbar", "hars")), can=can, army=True)[1] == []  # (tier 1)
    assert _orders(_base("hkee", ("hfoo",) * 5 + ("hrif",) * 3, ("hbar", "hars")), can=can, army=True)[1] == []
    assert _orders(_base("hkee", ("hfoo",) * 4, ("hbar", "hars")), can=can, army=True)[1] == []  # (a small army)
    target = {"t": {}, "n": {"hfoo": 6, "hrif": 2}}  # (a target's mix: the army mix lesson's business)
    assert _orders(_base("hkee", ("hfoo",) * 8, ("hbar", "hars")), can=can, army=True, target=target)[1] == []


def test_upgrade_lesson():
    rows = _base(buildings=("hbar", "hbla"))
    assert _orders(rows, upgrades=True, pace=0.5)[1] == [ImmediateOrder(11, code("Rhme"))]
    assert _orders(rows, now=200.0, upgrades=True, pace=0.5)[1] == []  # (before 8 minutes x 0.5)
    assert _orders(rows, upgrades=True)[1] == []  # (a real game's 8 minutes)
    assert _orders(rows, upgrades=True, pace=0.5, levels={code("Rhme"): 1})[1] == [ImmediateOrder(11, code("Rhar"))]
    assert _orders(rows, upgrades=True, pace=0.5, busy={11})[1] == []
    ranged = _base(army=("hrif",) * 4 + ("hfoo",) * 2, buildings=("hbar", "hbla"))
    assert _orders(ranged, upgrades=True, pace=0.5)[1] == [ImmediateOrder(11, code("Rhra"))]
    melee_masked = lambda u, k=fx.IMMEDIATE: u not in ("Rhme", "Rhar")  # noqa: E731
    assert _orders(rows, upgrades=True, pace=0.5, can=melee_masked)[1] == [ImmediateOrder(11, code("Rhra"))]
    assert _orders(rows, upgrades=True, pace=0.5, can=lambda u, k=fx.IMMEDIATE: False)[1] == []
    assert _orders(_base(), upgrades=True, pace=0.5)[1] == []  # (no blacksmith)


def _creep_rows(hero_hp=600, creep_hp=240, extra=()):
    hero = _unit(50, 0, -1500, 0, "Hpal", fx.HERO, hp=hero_hp, maxhp=650, level=2)
    camp = [_unit(80 + k, 12, -500 + 100 * k, 1500, "nftr", hp=creep_hp, maxhp=creep_hp) for k in range(2)]
    far = [_unit(85, 12, 4000, 4000, "nftr")]  # (beyond CREEP_REACH of its start)
    return _base(extra=[hero] + camp + far + list(extra))


def test_creep_lesson():
    tc, out = _orders(_creep_rows(), creep=True, pace=0.5)
    ids = [50] + [30 + k for k in range(6)]
    assert out == [PointOrder(i, ATTACK, -450.0, 1500.0) for i in ids] and tc.taught["creep"] == 1
    assert _orders(_creep_rows(), creep=2, pace=0.5)[1] == [PointOrder(50, ATTACK, -450.0, 1500.0)]  # (heroes only)
    assert _orders(_creep_rows(), creep=True, now=200.0)[1] == []  # (before a real game's 3.5 minutes)
    assert _orders(_creep_rows(), creep=True, now=60.0, pace=0.5)[1] == []
    assert _orders(_creep_rows(hero_hp=300), creep=True, pace=0.5)[1] == []  # (a hurt hero)
    assert _orders(_creep_rows(creep_hp=20000), creep=True, pace=0.5)[1] == []  # (a camp far stronger)
    raid = [_unit(91, 1, -1800, 600, "ogru", vis=1)]  # (an enemy army unit near its hall)
    assert _orders(_creep_rows(extra=raid), creep=True, pace=0.5)[1] == []
    small = _base(army=("hfoo",) * 3, extra=list(_creep_rows()[-4:]))  # (an army of 3)
    assert _orders(small, creep=True, pace=0.5)[1] == []


def test_camps_and_pace():
    creeps = np.stack([_unit(1, 12, 0, 0, "nftr"), _unit(2, 12, 400, 0, "nftr"), _unit(3, 12, 800, 0, "nftr"),
                       _unit(4, 12, 3000, 0, "nftr")])
    assert tm._camps(creeps).tolist() == [0, 0, 0, 1]  # (single link: 0-1-2 chained)
    assert tm.pace("echofast2") == 0.5 and tm.pace("(2)EchoIsles") == 1.0


def test_lessons_label_the_step():
    """The lessons' orders as the encoder's labels (taught: never given) at the units they name."""
    from warcraftsim.fullgame.selfplay import label_rows
    types = ("htow", "hpea", "hbar", "hbla", "hfoo", "Hpal", "nftr", "hgry")
    vocab = {"types": [code(c) for c in types], "current_orders": [], "upgrades": [],
             "orders": [[code("hrif"), fx.IMMEDIATE], [code("Rhme"), fx.IMMEDIATE], [ATTACK, fx.POINT]]}
    enc = fx.Encoder(vocab, np.zeros((len(vocab["orders"]) + 1, 3), np.int64))
    view = enc.view(0, 1.0, [0, 1])
    rows = _creep_rows(extra=[_unit(90, 1, 3000, 3000, "hgry", vis=1)])
    rows = np.concatenate([rows, _base(buildings=("hbar", "hbla"))[3:4]])  # (the blacksmith, id 11)
    me = np.asarray([0, 0, 1000, 500, 30, 60, 0])
    st = view.step(rows, me, np.zeros((0, 5), np.int64), 0)
    out = _orders(rows, army=True, upgrades=True, creep=True, pace=0.5)[1]
    y, _ = enc.step_labels(view, st, label_rows(0, out), me, np.zeros((0, 5), np.int64), {}, taught=True)
    got = {i: int(y["y_order"][st["index"][i]]) for i in (10, 11, 50, 30)}
    assert got == {10: 1, 11: 2, 50: 3, 30: 3}  # (riflemen at the barracks, the weapons, the heroes' and army's attack)
