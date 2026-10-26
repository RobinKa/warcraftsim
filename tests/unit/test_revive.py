"""A dead hero's revival (2026-10-26, scripts/revive_probe.py in the game): its price as the game takes it, its money
paid first in the agent's step, the --econ-masks bit 2 reserve and the --revive-lesson."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from warcraftsim.fullgame import features as fx


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


AUDIT_VOCAB = {"types": [code(c) for c in ("uaco", "ngol", "ugol", "ntav", "ngme", "Hpal", "halt", "hpea", "htow", "unpl")],
               "current_orders": [], "upgrades": [],
               "orders": [[code("ugol"), fx.UNIT], [fx.HARVEST, fx.UNIT], [code("Hpal"), fx.IMMEDIATE],
                          [code("hfoo"), fx.IMMEDIATE], [code("Npbm"), fx.UNIT]]}  # (test_inputs_v2's)


@pytest.fixture()
def v2():
    try:
        from warcraftsim.fullgame.herodata import hero_table, item_table
        hero_table(), item_table()
    except Exception as e:  # noqa: BLE001 (the game's tables: not on every machine)
        pytest.skip(f"no game data: {e}")
    fx.set_inputs(2)
    yield
    fx.set_inputs(1)


def _unit(i: int, owner: int, x: int, y: int, vis: int, flags: int = 0, typ: str = "hfoo") -> np.ndarray:
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = i, code(typ), owner, x, y
    r[fx.C_HP] = r[fx.C_MAXHP] = 400
    r[fx.C_FLAGS], r[fx.C_VIS] = flags, vis
    return r


def test_revive_gold_as_the_game_takes_it():
    """Probed: the real game's 425 gold heroes at levels 1 / 2 / 3 / 5 for 170 / 210 / 255 / 340; echofast2's 212 gold
    ones for 85 / 105 / 125 / 170 (the mask's 84 / 106 / 127 / 169 had let it through short)."""
    assert [fx.revive_gold(425, lv) for lv in (1, 2, 3, 4, 5, 10)] == [170, 210, 255, 295, 340, 550]
    assert [fx.revive_gold(212, lv) for lv in (1, 2, 3, 5)] == [85, 105, 125, 170]
    assert fx.revive_gold(1000, 10) == 700  # (HeroMaxReviveCostGold)


def _costs():
    costs = np.zeros((len(AUDIT_VOCAB["orders"]) + 1, 3), np.int64)
    costs[3], costs[4], costs[5] = (212, 50, 5), (68, 0, 2), (212, 68, 5)  # (Hpal, hfoo, Npbm: echofast2's)
    return costs


def _rows(food_dead="Hpal"):
    return np.stack([_unit(1, 0, -2000, 0, 1, fx.STRUCTURE, "htow"), _unit(3, 0, -2500, 0, 1, fx.STRUCTURE, "halt"),
                     _unit(4, 0, -2000, 100, 1, fx.WORKER, "hpea"), _unit(2, 0, -2100, 0, 1, fx.HERO | fx.DEAD, food_dead)])


def test_revival_paid_first_in_the_step(v2):
    """The policy's revival and a footman in one step with gold for one: the revival goes (its price kept before the
    step's other orders; it had paid nothing, and an earlier order's spending got it refused); short of its price,
    nothing of it is sent."""
    from warcraftsim.fullgame.play import REVIVE, BCAgent
    from warcraftsim.protocol import ImmediateOrder, TargetOrder
    bot = BCAgent(None, AUDIT_VOCAB, 0, None)
    bot.enc = fx.Encoder(AUDIT_VOCAB, _costs())
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.trees, bot.building_since, bot.t = {}, {}, 0
    rows = _rows()
    for gold, want in ((100, True), (84, False)):
        bot.me = me = np.asarray([0, 0, gold, 500, 5, 30, 0])
        bot.view = bot.enc.view(0, 1.0, [0, 1])
        st = bot.view.step(rows, me, np.zeros((0, 5), np.int64), 0)
        assert int(bot.view.costs_now[3][0]) == 85
        n = st["n_own"]
        order = [0] * n
        order[st["index"][1]], order[st["index"][3]] = 4, 3  # (a footman at the hall, Hpal at the altar)
        out = bot.commands(st, order, [0] * n, [0] * n, [0] * n)
        assert (TargetOrder(3, REVIVE, 2) in out) == want
        assert (ImmediateOrder(1, code("hfoo")) in out) == (not want)  # (100 - 85 < 68; at 84 the footman's)


def test_revival_reserve_masks_the_rest(v2):
    """--econ-masks bit 2: a dead hero the altar would take (REVIVE_AFTER_STEPS on, food free) keeps its revival's gold
    from the other paid orders; not before, not without the food, not with the bit off."""
    try:
        for bits, t, fu, kept in ((2, 0, 5, False), (2, 20, 5, True), (2, 20, 27, False), (0, 20, 5, False), (3, 20, 5, True)):
            fx.set_econ_masks(bits)
            enc = fx.Encoder(AUDIT_VOCAB, _costs())
            view = enc.view(0, 1.0, [0, 1])
            me = np.asarray([0, 0, 120, 500, fu, 30, 0])
            view.step(_rows(), me, np.zeros((0, 5), np.int64), 0)
            st = view.step(_rows(), me, np.zeros((0, 5), np.int64), t)
            assert (view.revive_kept == (3, 85, 0)) == kept, (bits, t, fu)
            assert bool(st["avail"][4]) == (not kept)  # (the footman: 120 - 85 < 68)
            assert bool(st["avail"][3]) == (fu + 5 <= 30)  # (the revival itself)
    finally:
        fx.set_econ_masks(0)


@pytest.mark.parametrize("dead,want", [("Hpal", "immediate"), ("Npbm", "buy"), ("Hpal", None)])
def test_revive_lesson(v2, dead, want):
    """--revive-lesson: a dead hero the policy could revive (its order's mask) taught at the finished altar: an altar
    hero's own order, a tavern hero's hire (play.BCAgent sends either as the revival); none the mask holds back."""
    from warcraftsim.fullgame.teacher import Teacher
    from warcraftsim.protocol import Buy, ImmediateOrder
    tc = Teacher(0, "human", random.Random(0), revive=True)
    player = SimpleNamespace(gold=300, lumber=100, food_used=5, food_cap=30, start_x=-2000.0, start_y=0.0)
    can = (lambda u, k=fx.IMMEDIATE: True) if want else (lambda u, k=fx.IMMEDIATE: False)
    out = tc.commands(_rows(dead), player, 300.0, 100, can)
    assert out == ([ImmediateOrder(3, code(dead))] if want == "immediate" else [Buy(3, 3, code(dead))] if want else [])
    assert Teacher(0, "human", random.Random(0)).commands(_rows(dead), player, 300.0, 100, can) == []  # (off)
