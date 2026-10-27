"""Audit 17's orders the game refuses (2026-10-27, scripts/undead_probe.py in the game): the --econ-masks bit 64 masks
(View.futile), the orders play.BCAgent then leaves unsent or sends in the form the game takes, and the morphed forms'
rows of the network's table of orders by type (model.prune_allowed)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import herodata
from warcraftsim.protocol import ImmediateOrder, PointOrder, TargetOrder, UseItem


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


TYPES = ("uaco", "ugol", "unpl", "Udea", "ugho", "ogru", "ucry", "ucrm", "Hpal", "hfoo", "uabo")
VOCAB = {"types": [code(c) for c in TYPES], "current_orders": [], "upgrades": [],
         "orders": [[fx.HARVEST, fx.UNIT], [fx.DEATH_COIL, fx.UNIT], [fx.HOLY_LIGHT, fx.UNIT], [851986, fx.POINT],
                    [852534, fx.IMMEDIATE], [fx.USESLOT, fx.IMMEDIATE], [fx.USESLOT + 1, fx.IMMEDIATE]]}
HARVEST, COIL, LIGHT, MOVE, UNBURROW, SLOT1, SLOT2 = range(1, 8)
DUST, PORTAL = code("dust"), code("stwp")


@pytest.fixture(autouse=True)
def tables(monkeypatch):
    """The game's tables the masks read, as the game data has them for these types and items."""
    monkeypatch.setattr(herodata, "unit_kinds", lambda: {"undead": frozenset(code(c) for c in ("uaco", "ugol", "unpl",
                        "Udea", "ugho", "ucry", "ucrm", "uabo")), "mechanical": frozenset((code("ugol"), code("unpl"))),
                        "ancient": frozenset()})
    monkeypatch.setattr(herodata, "item_table", lambda: {DUST: (1, 0, 0, 0), PORTAL: (1, 0, 0, 1)})
    monkeypatch.setattr(fx, "_KINDS", None)
    fx.set_econ_masks(64)
    yield
    fx.set_econ_masks(0)


def _unit(i: int, owner: int, x: int, y: int, typ: str, flags: int = 0, vis: int = 3, hp: int = 400,
          order: int = 0, items=()) -> np.ndarray:
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = i, code(typ), owner, x, y
    r[fx.C_HP], r[fx.C_MAXHP], r[fx.C_ORDER] = hp, 400, order
    r[fx.C_FLAGS], r[fx.C_VIS] = flags, vis
    for k, it in enumerate(items):
        r[fx.C_ITEM0 + k] = it
    return r


def _base(miners: int, extra=()):
    rows = [_unit(1, 0, -2000, 0, "unpl", fx.STRUCTURE), _unit(2, 0, -2000, 700, "ugol", fx.STRUCTURE)]
    rows += [_unit(10 + k, 0, -2000 + 40 * k, 500, "uaco", fx.WORKER, order=fx.HARVEST) for k in range(miners)]
    rows += [_unit(20, 0, -2600, 0, "uaco", fx.WORKER)]  # (an idle one, away from the mine)
    return np.stack(rows + list(extra))


def _bot(rows, t: int = 0, full=None):
    from warcraftsim.fullgame.play import BCAgent
    bot = BCAgent(None, VOCAB, 0, None)
    bot.enc = fx.Encoder(VOCAB, np.zeros((len(VOCAB["orders"]) + 1, 3), np.int64))
    bot.view = bot.enc.view(0, 1.0, [2, 1])  # (undead)
    if full:
        bot.view.full_mines = full
    bot.trees, bot.building_since, bot.t = {}, {}, t
    bot.me = me = np.asarray([0, 0, 500, 500, 5, 30, 0])
    st = bot.view.step(rows, me, np.zeros((0, 5), np.int64), t)
    return bot, st


def _send(bot, st, unit: int, cls: int, target: int = 0) -> list:
    n = st["n_own"]
    order, tgt = [0] * n, [0] * n
    order[st["index"][unit]], tgt[st["index"][unit]] = cls, st["index"].get(target, 0)
    return bot.commands(st, order, tgt, [0] * n, [0] * n)


def test_harvest_masked_at_full_or_refusing_mines():
    _, st = _bot(_base(5))
    assert not st["avail"][HARVEST]
    bot, st = _bot(_base(4))
    assert st["avail"][HARVEST]
    assert _send(bot, st, 20, HARVEST, 2) == [TargetOrder(20, fx.HARVEST, 2)]
    # (the game refused one: its miners mining with their current order 0; full for MINE_FULL_STEPS)
    bot.accepted([TargetOrder(20, fx.HARVEST, 2)], [False])
    assert bot.view.full_mines == {2: 0}
    bot, st = _bot(_base(4), t=30, full={2: 0})
    assert not st["avail"][HARVEST]
    assert _send(bot, st, 20, HARVEST, 2) == []
    _, st = _bot(_base(4), t=fx.MINE_FULL_STEPS, full={2: 0})
    assert st["avail"][HARVEST]


def test_death_coil_targets():
    dk = _unit(30, 0, -1500, 0, "Udea", fx.HERO)
    full, hurt = _unit(31, 0, -1500, 100, "ugho"), _unit(32, 0, -1500, 200, "ugho", hp=200)
    grunt, foe_ghoul = _unit(40, 1, -1200, 0, "ogru", vis=3), _unit(41, 1, -1200, 100, "ugho", vis=3)
    unseen = _unit(42, 1, 3000, 0, "ogru", vis=2)
    _, st = _bot(_base(5, [dk, full, foe_ghoul, unseen]))  # (own at full health, enemy undead, an unseen grunt)
    assert not st["avail"][COIL]
    _, st = _bot(_base(5, [_unit(30, 0, -1500, 0, "Udea", fx.HERO, hp=100), full]))  # (only the knight itself hurt)
    assert not st["avail"][COIL]
    for extra in ([hurt], [grunt]):
        _, st = _bot(_base(5, [dk, full, foe_ghoul] + extra))
        assert st["avail"][COIL]
    bot, st = _bot(_base(5, [dk, full, hurt, grunt, foe_ghoul]))
    assert _send(bot, st, 30, COIL, 31) == []  # (own, full health)
    assert _send(bot, st, 30, COIL, 41) == []  # (enemy undead)
    assert _send(bot, st, 30, COIL, 30) == []  # (itself)
    assert _send(bot, st, 30, COIL, 32) == [TargetOrder(30, fx.DEATH_COIL, 32)]
    assert _send(bot, st, 30, COIL, 40) == [TargetOrder(30, fx.DEATH_COIL, 40)]
    assert _send(bot, st, 30, COIL, 2) == []  # (a building)
    assert bot.futile_orders == 4


def test_holy_light_heals_the_living():
    pal = _unit(30, 0, -1500, 0, "Hpal", fx.HERO, hp=100)  # (hurt: no target of its own)
    foot, ghoul = _unit(31, 0, -1500, 100, "hfoo", hp=200), _unit(41, 1, -1200, 100, "ugho", vis=3)
    bot, st = _bot(_base(5, [pal, foot, ghoul]))
    assert _send(bot, st, 30, LIGHT, 31) == [TargetOrder(30, fx.HOLY_LIGHT, 31)]
    assert _send(bot, st, 30, LIGHT, 41) == [TargetOrder(30, fx.HOLY_LIGHT, 41)]
    assert _send(bot, st, 30, LIGHT, 30) == []


def test_burrowed_fiend_takes_only_unburrow():
    rows = _base(5, [_unit(50, 0, -1500, 0, "ucrm"), _unit(51, 0, -1500, 100, "ucry")])
    bot, st = _bot(rows)
    assert _send(bot, st, 50, MOVE) == []
    assert _send(bot, st, 50, UNBURROW) == [ImmediateOrder(50, 852534)]
    assert _send(bot, st, 51, UNBURROW) == []
    assert isinstance(_send(bot, st, 51, MOVE)[0], PointOrder)


def test_item_slots_sent_as_the_items_use():
    hero = _unit(30, 0, -1500, 0, "Udea", fx.HERO, items=(PORTAL, DUST))
    bot, st = _bot(_base(5, [hero]))
    assert not st["avail"][SLOT1] and st["avail"][SLOT2]  # (a town portal needs a target; dust of appearance not)
    assert _send(bot, st, 30, SLOT2) == [UseItem(30, 1)]
    assert _send(bot, st, 30, SLOT1) == []
    _, st = _bot(_base(5, [_unit(30, 0, -1500, 0, "Udea", fx.HERO)]))  # (empty slots)
    assert not st["avail"][SLOT2]


def test_prune_allowed_morphed_forms(monkeypatch):
    from warcraftsim.fullgame import model, techtree
    monkeypatch.setattr(techtree, "tree", lambda: SimpleNamespace(makers={}))
    net = model.FullGameNet(n_types=len(TYPES) + 1, n_cur=2, n_orders=len(VOCAB["orders"]) + 1, G=fx.Encoder(VOCAB).G,
                            d=16, layers=1)
    net.allowed.fill_(True)
    model.prune_allowed(net, VOCAB)
    ucrm, ucry = TYPES.index("ucrm") + 1, TYPES.index("ucry") + 1
    assert [k for k in range(1, net.allowed.shape[1]) if net.allowed[ucrm, k]] == [UNBURROW]
    assert not net.allowed[ucry, UNBURROW] and net.allowed[ucry, MOVE]
    fx.set_econ_masks(63)  # (without the bit: as before)
    net.allowed.fill_(True)
    model.prune_allowed(net, VOCAB)
    assert bool(net.allowed[ucrm].all())
    assert torch.is_tensor(net.allowed)
