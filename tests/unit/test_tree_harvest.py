"""--econ-masks bit 256 (features.TREE_HARVEST; 2026-11-23, scripts/ghoul_probe.py in the game): a lumber harvester's
right-click on a tree sent as the harvest order. Right-clicked, the game shows its current order as "smart" for its
whole first trip: no harvester to features.redundant, and the policy moved undead's first ghoul away and sent it back
every step in target games. Off (every run before it): the commands as they were."""

import numpy as np
import pytest

from warcraftsim.fullgame import features as fx
from warcraftsim.protocol import PointOrder, TargetDestructable


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


MOVE_ID = 851986
TYPES = ("unpl", "uaco", "ugho", "hpea", "opeo", "ewsp", "hfoo")
VOCAB = {"types": [code(c) for c in TYPES], "current_orders": [], "upgrades": [],
         "orders": [[fx.SMART, fx.TREE], [fx.HARVEST, fx.TREE], [MOVE_ID, fx.POINT]]}
SMART_TREE, HARVEST_TREE, MOVE = 1, 2, 3
TREE = 900


@pytest.fixture(autouse=True)
def masks_off_after():
    yield
    fx.set_econ_masks(0)


def _unit(i: int, typ: str, flags: int = 0, order: int = 0, x: int = -2000, y: int = 0) -> np.ndarray:
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = i, code(typ), 0, x, y
    r[fx.C_HP], r[fx.C_MAXHP], r[fx.C_ORDER], r[fx.C_FLAGS], r[fx.C_VIS] = 300, 300, order, flags, 3
    return r


def _bot(rows, assign=None):
    from warcraftsim.fullgame.play import BCAgent
    bot = BCAgent(None, VOCAB, 0, None)
    bot.enc = fx.Encoder(VOCAB, np.zeros((len(VOCAB["orders"]) + 1, 3), np.int64))
    bot.view = bot.enc.view(0, 1.0, [2, 1])  # (undead)
    bot.view.assign.update(assign or {})
    bot.trees, bot.building_since, bot.t = {TREE: (-2400, 300), TREE + 1: (2400, 300)}, {}, 0
    bot.me = me = np.asarray([0, 0, 500, 500, 5, 30, 0])
    st = bot.view.step(np.stack(rows), me, np.zeros((0, 5), np.int64), 0)
    return bot, st


def _send(bot, st, unit: int, cls: int) -> list:
    n = st["n_own"]
    order = [0] * n
    order[st["index"][unit]] = cls
    bx, by = fx.world_bins(-2400, 300, bot.view.sign)  # (the tree's bins: the tree nearest that point)
    return bot.commands(st, order, [0] * n, [int(bx)] * n, [int(by)] * n)


def _base(*units):
    return [_unit(1, "unpl", fx.STRUCTURE), *units]


def test_the_flag_is_bit_256_alone():
    for v in (0, 1, 64, 255):
        fx.set_econ_masks(v)
        assert not fx.TREE_HARVEST
    fx.set_econ_masks(256)
    assert fx.TREE_HARVEST and not (fx.ECON_MASK or fx.REVIVE_RESERVE or fx.NE_LEAN or fx.DETONATE_MASK
                                     or fx.REPAIR_MASK or fx.MILITIA_MASK or fx.FUTILE_MASK or fx.BURROW_MASK)
    fx.set_econ_masks(511)
    assert fx.TREE_HARVEST and fx.ECON_MASK and fx.FUTILE_MASK and fx.BURROW_MASK


def test_tree_order_only_a_harvesters_right_click():
    fx.set_econ_masks(255)  # (off: every order as given)
    for t in TYPES:
        assert fx.tree_order(fx.SMART, code(t)) == fx.SMART
    fx.set_econ_masks(511)
    for t in ("ugho", "hpea", "opeo", "ewsp"):
        assert fx.tree_order(fx.SMART, code(t)) == fx.HARVEST
        assert fx.tree_order(fx.HARVEST, code(t)) == fx.HARVEST
    for t in ("uaco", "hfoo", "unpl"):  # (no lumber harvester: its right-click is a walk to the tree, as the game has it)
        assert fx.tree_order(fx.SMART, code(t)) == fx.SMART
    assert fx.tree_order(MOVE_ID, code("ugho")) == MOVE_ID


@pytest.mark.parametrize("masks, sent", [(0, fx.SMART), (255, fx.SMART), (256, fx.HARVEST), (511, fx.HARVEST)])
def test_a_ghouls_right_click_on_a_tree(masks, sent):
    fx.set_econ_masks(masks)
    bot, st = _bot(_base(_unit(5, "ugho"), _unit(6, "uaco", fx.WORKER)))
    cmds = _send(bot, st, 5, SMART_TREE)
    assert cmds == [TargetDestructable(5, sent, TREE)]
    assert bot.sent_resources == ["lumber"]  # (what the view notes it harvests: either form)
    bot.accepted(cmds, [True])
    assert bot.view.assign == {5: "lumber"}
    # (an acolyte harvests no lumber: its right-click stays one; the harvest order class itself as it was)
    assert _send(bot, st, 6, SMART_TREE) == [TargetDestructable(6, fx.SMART, TREE)]
    assert _send(bot, st, 5, HARVEST_TREE) == [TargetDestructable(5, fx.HARVEST, TREE)]
    assert _send(bot, st, 5, MOVE)[0] == PointOrder(5, MOVE_ID, *fx.bins_world(*fx.world_bins(-2400, 300, 1.0), 1.0))


def test_why_the_current_order_matters():
    """The first trip after each form, as the game shows it (scripts/ghoul_probe.py --engine): reading "smart", the
    ghoul sent to the trees is no harvester to the agent and a repeated tree order goes out again; reading "harvest"
    (the harvest order's, at once) it is dropped."""
    fx.set_econ_masks(511)
    bot, st = _bot(_base(_unit(5, "ugho", order=fx.SMART)), {5: "lumber"})
    assert _send(bot, st, 5, SMART_TREE) == [TargetDestructable(5, fx.HARVEST, TREE)]
    bot, st = _bot(_base(_unit(5, "ugho", order=fx.HARVEST)), {5: "lumber"})
    assert _send(bot, st, 5, SMART_TREE) == []
    assert _send(bot, st, 5, HARVEST_TREE) == []
