"""--order-masks bits 8 and 16 (features.REPEAT_ORDERS, EDGE_ORDERS; bug finder 11, 2026-11-23; play.BCAgent.commands).

Bit 8: the game starts a unit's swing over at every new attack-move (scripts/order_probe.py: attack-moved again every
0.5 s step a footman, an archer, a crypt fiend and a death knight never hit), and the policy picks every unit's order
every step: an attack-move or a move for a unit that reads that order already, to within REPEAT_NEAR of the point its
last one went to, is not sent. Bit 16: a point order or a building aimed outside the map (the old point bins' top
edge) is not sent. Off (every run before them): the commands as they were."""

import numpy as np
import pytest

from warcraftsim.fullgame import features as fx
from warcraftsim.protocol import Build, PointOrder, TargetOrder


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


ATTACK_ID, MOVE_ID, PATROL_ID = 851983, 851986, 851990
TYPES = ("htow", "hpea", "hfoo", "Hamg", "hhou")
VOCAB = {"types": [code(c) for c in TYPES], "current_orders": [], "upgrades": [],
         "orders": [[ATTACK_ID, fx.POINT], [MOVE_ID, fx.POINT], [PATROL_ID, fx.POINT], [ATTACK_ID, fx.UNIT],
                    [code("hhou"), fx.POINT]]}
ATTACK, MOVE, PATROL, ATTACK_UNIT, FARM = 1, 2, 3, 4, 5
ECHO = (((-5184.0, 2944.0), (4672.0, 2944.0)), 2.75)  # (Echo Isles' starts and bins' reach: features.map_geometry)
ECHO_BOUNDS = (-6912.0, -5376.0, 6912.0, 4864.0)  # (its camera bounds: features.map_bounds)
START = ECHO[0][0]


@pytest.fixture(autouse=True)
def echo_frame():
    was = (fx.STARTS, fx.BIN_RANGE)
    fx.set_geometry(*ECHO)
    yield
    fx.set_order_masks(0)
    fx.set_bounds(None)
    fx.set_geometry(*was)


def _unit(i: int, typ: str, flags: int = 0, order: int = 0, x: float = START[0] + 300, y: float = START[1], owner: int = 0):
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = i, code(typ), owner, x, y
    r[fx.C_HP], r[fx.C_MAXHP], r[fx.C_ORDER], r[fx.C_FLAGS], r[fx.C_VIS] = 300, 300, order, flags, 3
    return r


def _bot(rows, start=START):
    from warcraftsim.fullgame.play import BCAgent
    bot = BCAgent(None, VOCAB, 0, None)
    bot.enc = fx.Encoder(VOCAB, np.zeros((len(VOCAB["orders"]) + 1, 3), np.int64))
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.view.start = start
    bot.trees, bot.building_since, bot.t = {}, {}, 0
    bot.me = me = np.asarray([0, 0, 500, 500, 5, 30, 0])
    return bot, bot.view.step(np.stack(rows), me, np.zeros((0, 5), np.int64), 0)


def _send(bot, st, unit: int, cls: int, x: float, y: float, tgt: int = 0) -> list:
    n = st["n_own"]
    order = [0] * n
    order[st["index"][unit]] = cls
    bx, by = fx.world_bins(x, y, bot.view.sign)
    return bot.commands(st, order, [tgt] * n, [int(bx)] * n, [int(by)] * n)


def _point(x: float, y: float) -> tuple[float, float]:
    """The point the agent sends for (x, y): its bin's centre."""
    return fx.bins_world(*(int(v) for v in fx.world_bins(x, y, 1.0)), 1.0)


def _base(*units):
    return [_unit(1, "htow", fx.STRUCTURE, x=START[0], y=START[1]), *units]


CAMP = (-4636.0, -2984.0)  # (the natural's camp: where the lesson's heroes were attack-moved every step)


def test_the_bits_are_the_agents_alone():
    assert (fx.REPEAT_ORDERS, fx.EDGE_ORDERS) == (8, 16)
    for v in (0, 7, 8, 16, 31):
        fx.set_order_masks(v)
        assert fx.ORDER_MASKS == v


@pytest.mark.parametrize("masks, again", [(0, True), (7, True), (16, True), (8, False), (15, False), (31, False)])
def test_an_attack_move_again_for_a_unit_attack_moving_there(masks, again):
    fx.set_order_masks(masks)
    bot, st = _bot(_base(_unit(5, "Hamg", fx.HERO)))
    first = _send(bot, st, 5, ATTACK, *CAMP)
    assert first == [PointOrder(5, ATTACK_ID, *_point(*CAMP))]  # (a unit that reads no order: sent)
    bot.accepted(first, [True])
    bot2, st2 = bot, bot.view.step(np.stack(_base(_unit(5, "Hamg", fx.HERO, order=ATTACK_ID))), bot.me,
                                   np.zeros((0, 5), np.int64), 1)
    bot2.t = 1
    for dx, dy in ((0, 0), (48, -96), (-150, 150)):  # (the policy's scatter around the place)
        got = _send(bot2, st2, 5, ATTACK, CAMP[0] + dx, CAMP[1] + dy)
        assert bool(got) == again
    assert getattr(bot2, "repeat_orders", 0) == (0 if again else 3)


def test_what_still_goes_out_with_bit_8():
    fx.set_order_masks(8)
    bot, st = _bot(_base(_unit(5, "Hamg", fx.HERO)))
    first = _send(bot, st, 5, ATTACK, *CAMP)
    bot.accepted(first, [True])
    rows = _base(_unit(5, "Hamg", fx.HERO, order=ATTACK_ID), _unit(6, "hfoo", order=ATTACK_ID), _unit(9, "hfoo", owner=1))
    st = bot.view.step(np.stack(rows), bot.me, np.zeros((0, 5), np.int64), 1)
    # another place (a new aim), another order kind, another unit that reads attack with no attack-move of its own
    assert _send(bot, st, 5, ATTACK, CAMP[0] + 600, CAMP[1]) == [PointOrder(5, ATTACK_ID, *_point(CAMP[0] + 600, CAMP[1]))]
    assert _send(bot, st, 5, MOVE, *CAMP) == [PointOrder(5, MOVE_ID, *_point(*CAMP))]
    assert _send(bot, st, 5, PATROL, *CAMP) == [PointOrder(5, PATROL_ID, *_point(*CAMP))]
    assert _send(bot, st, 6, ATTACK, *CAMP) == [PointOrder(6, ATTACK_ID, *_point(*CAMP))]
    # the unit's order done (it reads none again): the same attack-move goes out
    st = bot.view.step(np.stack(_base(_unit(5, "Hamg", fx.HERO))), bot.me, np.zeros((0, 5), np.int64), 2)
    assert _send(bot, st, 5, ATTACK, *CAMP) == [PointOrder(5, ATTACK_ID, *_point(*CAMP))]


def test_another_order_taken_forgets_the_place():
    """An attack on a unit, then the unit reads "attack" too: the next attack-move to the old place is a new order."""
    fx.set_order_masks(8)
    bot, st = _bot(_base(_unit(5, "Hamg", fx.HERO), _unit(9, "hfoo", owner=1)))
    bot.accepted(_send(bot, st, 5, ATTACK, *CAMP), [True])
    on_unit = [TargetOrder(5, ATTACK_ID, 9)]
    bot.accepted(on_unit, [True])
    rows = _base(_unit(5, "Hamg", fx.HERO, order=ATTACK_ID), _unit(9, "hfoo", owner=1))
    st = bot.view.step(np.stack(rows), bot.me, np.zeros((0, 5), np.int64), 1)
    assert _send(bot, st, 5, ATTACK, *CAMP) == [PointOrder(5, ATTACK_ID, *_point(*CAMP))]
    # an attack-move the game refused is no order of the unit's
    bot, st = _bot(_base(_unit(5, "Hamg", fx.HERO)))
    bot.accepted(_send(bot, st, 5, ATTACK, *CAMP), [False])
    st = bot.view.step(np.stack(_base(_unit(5, "Hamg", fx.HERO, order=ATTACK_ID))), bot.me, np.zeros((0, 5), np.int64), 1)
    assert len(_send(bot, st, 5, ATTACK, *CAMP)) == 1


def test_a_move_again_for_a_unit_moving_there():
    fx.set_order_masks(8)
    bot, st = _bot(_base(_unit(5, "hfoo")))
    bot.accepted(_send(bot, st, 5, MOVE, *CAMP), [True])
    st = bot.view.step(np.stack(_base(_unit(5, "hfoo", order=MOVE_ID))), bot.me, np.zeros((0, 5), np.int64), 1)
    assert _send(bot, st, 5, MOVE, CAMP[0] + 96, CAMP[1]) == []
    assert _send(bot, st, 5, ATTACK, *CAMP) == [PointOrder(5, ATTACK_ID, *_point(*CAMP))]  # (a move, then an attack-move)


TOP = (START[0] + 548.0, START[1] + 3048.0)  # (the old bins' top edge: y 5992, the camera's bound 4864)


def test_outside_map():
    assert not fx.outside_map(*TOP)  # (no bounds: nothing is outside)
    fx.set_bounds(ECHO_BOUNDS)
    assert fx.outside_map(*TOP) and fx.outside_map(7500.0, 0.0) and fx.outside_map(0.0, -6000.0)
    assert not fx.outside_map(START[0], 5300.0) and not fx.outside_map(*CAMP) and not fx.outside_map(6912.0, 4864.0)
    import os
    assert os.environ["WARCRAFTSIM_MAP_BOUNDS"] == ",".join(str(v) for v in ECHO_BOUNDS)
    fx.set_bounds(None)
    assert fx.MAP_BOUNDS is None and "WARCRAFTSIM_MAP_BOUNDS" not in os.environ


@pytest.mark.parametrize("masks, bounds, sent", [(0, True, True), (15, True, True), (16, False, True), (16, True, False),
                                                  (31, True, False)])
def test_a_building_aimed_above_the_map(masks, bounds, sent):
    fx.set_order_masks(masks)
    fx.set_bounds(ECHO_BOUNDS if bounds else None)
    bot, st = _bot(_base(_unit(5, "hfoo"), _unit(6, "hpea", fx.WORKER)))
    got = _send(bot, st, 6, FARM, *TOP)
    assert got == ([Build(6, code("hhou"), *_point(*TOP))] if sent else [])
    assert getattr(bot, "edge_orders", 0) == (0 if sent else 1)
    assert len(_send(bot, st, 6, FARM, START[0] + 576, START[1] + 480)) == 1  # (inside the map: as ever)
    # a unit's point orders there go out with either setting: the game takes them (the unit walks to the map's edge),
    # and what they do the policy has learned; a probe game's policy asked for them every step when they were dropped
    for cls in (ATTACK, MOVE, PATROL):
        assert len(_send(bot, st, 5, cls, *TOP)) == 1


def test_a_dropped_building_leaves_the_steps_money():
    """An outside farm took its cost from the step's budget and its kind's one build a step: a second worker's farm at
    a good place, sampled in the same step, was not sent."""
    from warcraftsim.fullgame.play import BCAgent
    fx.set_bounds(ECHO_BOUNDS)
    costs = np.zeros((len(VOCAB["orders"]) + 1, 3), np.int64)
    costs[FARM] = (80, 20, 0)

    def both(masks: int) -> list:
        fx.set_order_masks(masks)
        bot = BCAgent(None, VOCAB, 0, None)
        bot.enc = fx.Encoder(VOCAB, costs)
        bot.view = bot.enc.view(0, 1.0, [0, 1])
        bot.view.start = START
        bot.trees, bot.building_since, bot.t = {}, {}, 0
        bot.me = me = np.asarray([0, 0, 100, 100, 5, 30, 0])  # (the money for one farm)
        st = bot.view.step(np.stack(_base(_unit(6, "hpea", fx.WORKER), _unit(7, "hpea", fx.WORKER))), me,
                           np.zeros((0, 5), np.int64), 0)
        n = st["n_own"]
        order, bx, by = [0] * n, [0] * n, [0] * n
        for unit, (x, y) in ((6, TOP), (7, (START[0] + 576, START[1] + 480))):
            k = st["index"][unit]
            order[k] = FARM
            bx[k], by[k] = (int(v) for v in fx.world_bins(x, y, 1.0))
        return bot.commands(st, order, [0] * n, bx, by)
    assert [c.unit for c in both(0)] == [6]  # (the outside one first, the good one over the budget)
    assert [c.unit for c in both(16)] == [7]


def test_only_on_the_bounds_own_map():
    """A game whose player's start is not one of the frame's (a --map-pool launch, a start unknown): nothing dropped."""
    fx.set_order_masks(16)
    fx.set_bounds(ECHO_BOUNDS)
    for start in (None, (1000.0, 1000.0)):
        bot, st = _bot(_base(_unit(6, "hpea", fx.WORKER)), start=start)
        assert len(_send(bot, st, 6, FARM, *TOP)) == 1
    bot, st = _bot(_base(_unit(6, "hpea", fx.WORKER)), start=ECHO[0][1])  # (the right base's player: the same map)
    assert _send(bot, st, 6, FARM, *TOP) == []
