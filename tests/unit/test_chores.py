"""Audit 16's chores (2026-10-26, scripts/militia_probe.py in the game): human's militia orders in the forms the game
takes, the --econ-masks bits 4 (night elf's lower worker allowance), 8 (Detonate), 16 (repair), 32 (the call to arms)
and the --militia-lesson."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from warcraftsim.fullgame import features as fx
from warcraftsim.protocol import ImmediateOrder, TargetOrder


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


VOCAB = {"types": [code(c) for c in ("hpea", "hmil", "htow", "ngol", "hfoo", "ewsp", "etol", "hbar", "hwat")],
         "current_orders": [], "upgrades": [],
         "orders": [[fx.MILITIA_UNCONVERT, fx.UNIT], [fx.HARVEST, fx.UNIT], [fx.MILITIA_CONVERT, fx.UNIT],
                    [fx.TOWNBELL_OFF, fx.IMMEDIATE], [fx.TOWNBELL_ON, fx.IMMEDIATE], [fx.REPAIR, fx.UNIT],
                    [fx.DETONATE, fx.POINT], [fx.DETONATE, fx.UNIT]]}
UNCONVERT, HARVEST, CONVERT, BELL_OFF, BELL_ON, REPAIR, DET_P, DET_U = range(1, 9)


def _unit(i: int, owner: int, x: int, y: int, typ: str, flags: int = 0, vis: int = 3, hp: int = 400) -> np.ndarray:
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = i, code(typ), owner, x, y
    r[fx.C_HP], r[fx.C_MAXHP] = hp, 400
    r[fx.C_FLAGS], r[fx.C_VIS] = flags, vis
    return r


def _base(militia: bool = True, extra=()):
    rows = [_unit(1, 0, -2000, 0, "htow", fx.STRUCTURE), _unit(4, 0, -2000, 300, "hpea", fx.WORKER),
            _unit(7, 15, -2000, 700, "ngol", fx.STRUCTURE)]
    if militia:
        rows.append(_unit(5, 0, -1800, 200, "hmil"))
    return np.stack(rows + list(extra))


def _bot(rows):
    from warcraftsim.fullgame.play import BCAgent
    bot = BCAgent(None, VOCAB, 0, None)
    bot.enc = fx.Encoder(VOCAB, np.zeros((len(VOCAB["orders"]) + 1, 3), np.int64))
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.trees, bot.building_since, bot.t = {}, {}, 0
    bot.me = me = np.asarray([0, 0, 500, 500, 5, 30, 0])
    st = bot.view.step(rows, me, np.zeros((0, 5), np.int64), 0)
    return bot, st


def _send(bot, st, unit: int, cls: int, target: int = 0) -> list:
    n = st["n_own"]
    order, tgt = [0] * n, [0] * n
    order[st["index"][unit]], tgt[st["index"][unit]] = cls, st["index"].get(target, 0)
    return bot.commands(st, order, tgt, [0] * n, [0] * n)


@pytest.mark.parametrize("unit,cls,target,want", [
    (5, UNCONVERT, 4, ImmediateOrder(5, fx.MILITIA_OFF)),   # back to work pointed at a peasant (refused as it was)
    (5, UNCONVERT, 1, ImmediateOrder(5, fx.MILITIA_OFF)),   # at the hall (taken; the immediate one too, anywhere)
    (5, HARVEST, 7, ImmediateOrder(5, fx.MILITIA_OFF)),     # a militia's harvest (refused): back to work
    (4, CONVERT, 1, ImmediateOrder(4, fx.MILITIA_CALL)),    # a peasant's call to arms: its own immediate one
    (4, UNCONVERT, 1, None),                                # a peasant's back to work: refused, not sent
    (5, CONVERT, 1, None),                                  # a militia's call to arms
    (1, BELL_OFF, 0, ImmediateOrder(1, fx.TOWNBELL_OFF)),   # the bell's end with militia
    (1, BELL_ON, 0, ImmediateOrder(1, fx.TOWNBELL_ON)),
    (4, HARVEST, 7, TargetOrder(4, fx.HARVEST, 7)),         # a peasant's harvest as before
])
def test_militia_orders_as_the_game_takes_them(unit, cls, target, want):
    bot, st = _bot(_base())
    assert _send(bot, st, unit, cls, target) == ([want] if want else [])


def test_bell_end_without_militia_not_sent():
    bot, st = _bot(_base(militia=False))
    assert _send(bot, st, 1, BELL_OFF) == []
    bot, st = _bot(_base())  # (a militia on its way back already: not ordered again)
    st["sel"][st["index"][5], fx.C_ORDER] = fx.MILITIA_OFF
    assert _send(bot, st, 5, UNCONVERT, 1) == []


@pytest.fixture()
def bits():
    yield fx.set_econ_masks
    fx.set_econ_masks(0)


def _avail(rows, race: int = 0) -> tuple:
    enc = fx.Encoder(VOCAB, np.zeros((len(VOCAB["orders"]) + 1, 3), np.int64))
    view = enc.view(0, 1.0, [race, 1])
    st = view.step(rows, np.asarray([0, 0, 500, 500, 5, 30, 0]), np.zeros((0, 5), np.int64), 0)
    return st["avail"], view


def test_call_to_arms_mask(bits):
    bits(32)
    av, _ = _avail(_base(militia=False))
    assert not av[BELL_ON] and not av[CONVERT] and not av[BELL_OFF] and not av[UNCONVERT]
    av, _ = _avail(_base(militia=True))  # (militia: their way back open)
    assert not av[BELL_ON] and av[BELL_OFF] and av[UNCONVERT]
    enemy = _unit(9, 1, -1200, 0, "hfoo", vis=1)  # (seen within MILITIA_NEAR of the hall)
    av, _ = _avail(_base(militia=False, extra=[enemy]))
    assert av[BELL_ON] and av[CONVERT]
    unseen = _unit(9, 1, -1200, 0, "hfoo", vis=2)
    av, _ = _avail(_base(militia=False, extra=[unseen]))
    assert not av[BELL_ON]
    bits(0)
    av, _ = _avail(_base(militia=False))
    assert av[BELL_ON] and av[BELL_OFF]  # (off: as before)


def test_repair_mask(bits):
    bits(16)
    barracks = lambda hp: _unit(8, 0, -2400, 0, "hbar", fx.STRUCTURE, hp=hp)  # noqa: E731
    av, view = _avail(_base(extra=[barracks(360)]))  # (90%: masked)
    assert not av[REPAIR] and view.repair_ids == set()
    av, view = _avail(_base(extra=[barracks(240)]))  # (60%)
    assert av[REPAIR] and view.repair_ids == {8}
    av, view = _avail(_base(extra=[barracks(360), _unit(9, 1, -2400, 500, "hfoo", vis=1)]))  # (90%, attacked)
    assert av[REPAIR] and view.repair_ids == {8}
    av, view = _avail(_base(extra=[_unit(8, 0, -2400, 0, "hbar", fx.STRUCTURE | fx.CONSTRUCTING, hp=100)]))
    assert not av[REPAIR]  # (a frame: no helpers)
    bot, st = _bot(_base(extra=[barracks(360), _unit(10, 0, -2600, 0, "hwat", fx.STRUCTURE, hp=200)]))
    assert bot.view.repair_ids == {10}
    assert _send(bot, st, 4, REPAIR, 8) == []  # (the 90% barracks: not sent)
    assert _send(bot, st, 4, REPAIR, 10) == [TargetOrder(4, fx.REPAIR, 10)]


def test_detonate_mask_and_lean_wisps(bits):
    bits(8 | 4)
    ne = [_unit(1, 0, -2000, 0, "etol", fx.STRUCTURE), _unit(4, 0, -2000, 300, "ewsp", fx.WORKER)]
    av, _ = _avail(np.stack(ne + [_unit(9, 1, -1500, 300, "hwat", vis=1)]), race=3)  # (a footman-like non-summon)
    assert not av[DET_P] and not av[DET_U]
    av, _ = _avail(np.stack(ne + [_unit(9, 1, -1500, 300, "hwat", fx.SUMMONED, vis=1)]), race=3)
    assert av[DET_P] and av[DET_U]
    assert fx.worker_cap("nightelf", 1) == 15 and fx.worker_cap("human", 1) == 18
    bits(1)
    assert fx.worker_cap("nightelf", 1) == 20


def test_militia_lesson():
    from warcraftsim.fullgame.teacher import Teacher
    player = SimpleNamespace(gold=300, lumber=100, food_used=5, food_cap=30, start_x=-2000.0, start_y=0.0)

    def taught(rows, on=True):
        out = Teacher(0, "human", random.Random(0), militia=on).commands(rows, player, 300.0, 100)
        return [c for c in out if getattr(c, "order", None) == fx.TOWNBELL_OFF]
    assert taught(_base()) == [ImmediateOrder(1, fx.TOWNBELL_OFF)]
    assert taught(_base(extra=[_unit(9, 1, -1200, 0, "hfoo", vis=1)])) == []  # (an enemy near: they fight)
    assert taught(_base(militia=False)) == []
    assert taught(_base(), on=False) == []


def test_burrow_mask_and_stand_down():
    """--econ-masks 128: battle stations only with a seen enemy near a hall or burrow; a burrow holding peons stands down
    once no seen enemy has been near it for BURROW_SAFE_STEPS steps."""
    import numpy as np
    from warcraftsim.fullgame import features as fx
    fx.set_econ_masks(128)
    try:
        assert fx.BURROW_MASK
        r = np.zeros((3, fx.ROW_COLS), np.int64)
        r[0, [fx.C_ID, fx.C_TYPE, fx.C_FLAGS]] = 1, fx.BURROW_T, fx.STRUCTURE
        r[1, [fx.C_ID, fx.C_TYPE, fx.C_FLAGS, fx.C_X]] = 2, fx.PEON_T, fx.LOADED_FLAG, 50
        r[2, [fx.C_ID, fx.C_TYPE, fx.C_OWNER, fx.C_X]] = 3, int.from_bytes(b"hfoo", "big"), 1, 6000
        r[2, fx.C_VIS] = 1
        from types import SimpleNamespace
        from warcraftsim.fullgame.play import BCAgent
        bot = SimpleNamespace(view=SimpleNamespace(burrows_safe={1}, races=[1, 0]), player=0, t=0)
        bot.__dict__.update({})
        out = BCAgent._stand_down(bot, r)
        assert out == []  # (safe only from now)
        bot.t = fx.BURROW_SAFE_STEPS
        out = BCAgent._stand_down(bot, r)
        assert len(out) == 1 and out[0].order == fx.STANDDOWN and out[0].unit == 1
        bot.t += 1
        assert BCAgent._stand_down(bot, r) == []  # (not again at once)
    finally:
        fx.set_econ_masks(0)
