"""Inputs v2 (selfplay --inputs 2): a hero's abilities and items, what a building makes and the clock as
features; the masks of orders the player lacks the tech for (--tech-mask) and a hero can't carry out now
(--valid-mask); and older checkpoints loaded with zero weights for the new inputs."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from warcraftsim.fullgame import features as fx  # noqa: E402


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


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
    fx.set_masks(False, False)


VOCAB = {"types": [code("Hpal"), code("hbar"), code("halt"), code("htow")], "current_orders": [], "upgrades": [],
         "orders": [[852092, fx.UNIT], [code("AHhb"), fx.SKILL], [852008, fx.IMMEDIATE], [851983, fx.UNIT],
                    [code("hfoo"), fx.IMMEDIATE], [code("Hamg"), fx.IMMEDIATE]],
         "order_names": {"852092": "holybolt", "851983": "attack"}}


def _hero(owner: int, mana=100, cd=0, level=3, points=1, items=("phea", "stwp")) -> np.ndarray:
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER] = 1048576 + owner, code("Hpal"), owner
    r[fx.C_HP] = r[fx.C_MAXHP] = 700
    r[fx.C_MANA], r[fx.C_MAXMANA], r[fx.C_FLAGS], r[fx.C_VIS] = mana, 300, fx.HERO, 3
    r[fx.C_HLEVEL], r[fx.C_SKILLPTS] = level, points
    for k, it in enumerate(items):
        r[fx.C_ITEM0 + k] = code(it)
    r[fx.C_ABIL0], r[fx.C_ABIL0 + 1] = 1, cd  # holy light learned
    r[fx.C_ABIL0 + 6] = 1  # devotion aura
    return r


def test_hero_features(v2):
    view = fx.Encoder(VOCAB).view(0, 1.0, [0, 1])
    st = view.step(np.stack([_hero(0), _hero(1)]), None, np.zeros((0, 5), np.int64), 0)
    f, h = st["ent"], fx.A_HERO
    ab = lambda i, k: f[i, h + 4 * k:h + 4 * k + 4].tolist()  # noqa: E731 (level, ready, cooldown, learnable)
    assert ab(0, 0) == pytest.approx([1 / 3, 1, 0, 1])  # holy light: ready, level 2 at hero level 3
    assert ab(0, 1) == pytest.approx([0, 0, 0, 1])  # divine shield: not learned, learnable
    assert ab(0, 2) == pytest.approx([0, 0, 0, 0])  # resurrection: hero level 6
    assert ab(0, 3) == pytest.approx([1 / 3, 0, 0, 1])  # devotion aura: passive
    it = h + 16
    assert f[0, it:it + 10].tolist() == [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]  # a healing potion, a town portal
    assert ab(1, 0) == pytest.approx([1 / 3, 0, 0, 0])  # the enemy's: its levels only (and items)
    assert f[1, it:it + 5].tolist() == [1, 1, 1, 0, 0]
    st = view.step(np.stack([_hero(0, mana=50), _hero(1)]), None, np.zeros((0, 5), np.int64), 1)
    assert st["ent"][0, h + 1] == 0  # 50 mana: no holy light (65)
    st = view.step(np.stack([_hero(0, cd=30, points=0)]), None, np.zeros((0, 5), np.int64), 2)
    assert st["ent"][0, h:h + 4].tolist() == pytest.approx([1 / 3, 0, 3.0 / 60, 0])


def test_production_and_clock(v2):
    enc = fx.Encoder(VOCAB)
    view = enc.view(0, 1.0, [0, 1])
    rows = np.zeros((2, fx.ROW_COLS), np.int64)
    rows[:, fx.C_ID] = 10, 11
    rows[:, fx.C_TYPE] = code("hbar"), code("halt")
    rows[:, fx.C_FLAGS] = fx.STRUCTURE
    rows[:, fx.C_HP] = rows[:, fx.C_MAXHP] = 1000
    ev = np.array([[0, 5, 10, code("hfoo"), 0], [0, 5, 11, code("Hpal"), 0]], np.int64)
    st = view.step(rows, None, ev, 0)
    p = fx.A_PROD
    assert st["ent"][0, p:p + 5].tolist() == [0, 1, 0, 0, 0] and st["ent"][1, p:p + 5].tolist() == [0, 0, 1, 0, 0]
    x = enc.G0 + enc.ZD
    assert st["glob"][x:x + 3].tolist() == pytest.approx([np.sin(2 * np.pi / 3), -0.5, 0.0], abs=1e-6)  # 8 o'clock
    st = view.step(rows, None, np.array([[400, 6, 10, 77, code("hfoo")]], np.int64), 400)  # 200 s: dusk
    assert st["ent"][0, p:p + 5].tolist() == [0] * 5 and st["ent"][1, p + 2] == 1
    assert st["glob"][x + 2] == 1.0


def test_tech_mask(v2):
    from warcraftsim.fullgame.techtree import tree
    tree()
    vocab = {**VOCAB, "orders": [[code("hfoo"), fx.IMMEDIATE], [code("hgry"), fx.IMMEDIATE], [code("Rhme"), fx.IMMEDIATE],
                                 [code("Hpal"), fx.IMMEDIATE], [code("Hamg"), fx.IMMEDIATE], [851983, fx.UNIT]]}
    m = fx.TechMask(vocab, "human")
    base = frozenset({"htow", "hbar", "halt"})
    ok = m.allowed(base, {}, frozenset(), frozenset())
    assert ok.tolist() == [True, True, False, True, True, True, True]  # gryphons: a castle
    ok = m.allowed(base, {"Rhme": 1}, frozenset({"Hpal"}), frozenset())
    assert ok.tolist() == [True, True, False, False, False, False, True]  # level 2 and a 2nd hero: a keep; one paladin
    ok = m.allowed(base | {"hcas"}, {"Rhme": 1}, frozenset({"Hpal"}), frozenset({"Rhme"}))
    assert ok.tolist() == [True, True, True, False, False, True, True]  # (a castle is a keep; Rhme under way)
    assert not m.allowed(base | {"hcas"}, {"Rhme": 3}, frozenset({"Hpal", "Hmkg", "Hblm"}), frozenset())[[3, 4, 5]].any()


def test_tech_mask_in_the_view(v2):
    fx.set_masks(True, False)
    vocab = {**VOCAB, "orders": [[code("hfoo"), fx.IMMEDIATE], [code("hgry"), fx.IMMEDIATE]]}
    view = fx.Encoder(vocab, np.zeros((3, 3), np.int64)).view(0, 1.0, [0, 1])
    rows = np.zeros((1, fx.ROW_COLS), np.int64)
    rows[0, fx.C_ID], rows[0, fx.C_TYPE], rows[0, fx.C_FLAGS] = 10, code("hcas"), fx.STRUCTURE | fx.CONSTRUCTING
    rows[0, fx.C_HP] = rows[0, fx.C_MAXHP] = 1000
    me = np.array([0, 0, 500, 500, 0, 10, 0, 0, 0, 0, 0])
    assert view.step(rows, me, np.zeros((0, 5), np.int64), 0)["avail"].tolist() == [True, True, False]
    rows[0, fx.C_FLAGS] = fx.STRUCTURE  # (finished)
    assert view.step(rows, me, np.zeros((0, 5), np.int64), 1)["avail"].tolist() == [True, True, True]


def test_valid_mask(v2):
    from warcraftsim.fullgame.model import FullGameNet
    net = FullGameNet(n_types=5, n_cur=2, n_orders=7, G=40, d=32, layers=1).eval()
    net.allowed[:] = True
    net.set_valid(VOCAB)
    assert net.config["valid"] == 3  # holy light, learning it, the first item slot
    ent = torch.zeros(1, 2, fx.F)
    typ = torch.tensor([[1, 2]])  # a paladin, a barracks
    h = fx.A_HERO

    def ok():
        lg = net.order_logits(torch.zeros(1, 32), torch.zeros(1, 2, 32), typ, torch.tensor([2]), ent=ent)
        return (lg[0] > -1e8).tolist()
    assert ok() == [[True, False, False, False, True, True, True], [True, True, True, False, True, True, True]]
    ent[0, 0, h + 1] = 1  # holy light ready
    ent[0, 0, h + 3] = 1  # a point for it
    ent[0, 0, h + 16 + 1] = 1  # a usable item in slot 1
    assert ok()[0] == [True] * 7


def test_older_checkpoint_gets_the_new_inputs(v2, tmp_path):
    from warcraftsim.fullgame.model import FullGameNet, load
    fx.set_inputs(1)
    vocab = {**VOCAB, "upgrades": [code("Rhme")]}
    G = fx.Encoder(vocab).G
    torch.manual_seed(0)
    old = FullGameNet(n_types=5, n_cur=2, n_orders=7, G=G, d=32, layers=1, priv=fx.PRIV).eval()
    torch.save({"config": old.config, "model": old.state_dict(), "vocab": vocab}, tmp_path / "old.pt")
    fx.set_inputs(2)
    fx.set_masks(False, True)
    net, _ = load(tmp_path / "old.pt")
    assert net.config["G"] == G + 3 and net.config["gx"] == 3 and net.config["valid"] and net.ent.in_features == fx.F
    ent, glob = torch.randn(1, 3, fx.F), torch.randn(1, G + 3)
    typ, cur, mask = torch.tensor([[1, 2, 3]]), torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():  # (what it read before; the new inputs weigh nothing yet)
        g_new, _ = net.encode(ent, typ, cur, mask, glob)
        keep = torch.cat([glob[:, :G - fx.PRIV], glob[:, -fx.PRIV:]], 1)
        g_old, _ = old.encode(ent[..., :fx.F_BASE], typ, cur, mask, keep)
        assert torch.allclose(g_new, g_old, atol=1e-5)
        assert torch.allclose(net.value(g_new, glob), old.value(g_old, keep), atol=1e-5)


def test_demo_rows_get_their_heroes(v2):
    units = np.zeros((3, 18), np.int64)
    units[:, fx.C_STEP] = 0, 0, 1
    units[:, fx.C_ID] = 5, 6, 5
    heroes = np.zeros((2, 16), np.int64)
    heroes[:, 0], heroes[:, 1] = (0, 1), 5
    heroes[:, 2] = code("phea")
    heroes[1, 8], heroes[1, 12] = 2, 45  # step 1: slot 1 at level 2, 4.5 s to go
    rows = fx.hero_rows(units, heroes)
    assert rows.shape == (3, fx.ROW_COLS) and rows[:, fx.C_ITEM0].tolist() == [code("phea"), 0, code("phea")]
    assert rows[2, fx.C_ABIL0:fx.C_ABIL0 + 2].tolist() == [2, 45]


def test_live_rows_have_the_hero_columns():
    from types import SimpleNamespace

    from warcraftsim.fullgame.play import unit_rows
    from warcraftsim.protocol import Unit, UnitFlags
    u = Unit(7, code("Hpal"), 0, 1, 2, 0, 700, 700, 100, 300, 0, UnitFlags.HERO, 3, 0, 3, 200, 1,
             (code("phea"),), ((1, 2.5), (0, 0.0)))
    rows = unit_rows(SimpleNamespace(units=[u]), 4)
    assert rows.shape == (1, fx.ROW_COLS) and rows[0, fx.C_ITEM0] == code("phea") and rows[0, fx.C_ITEM0 + 1] == 0
    assert rows[0, fx.C_ABIL0:fx.C_ABIL0 + 4].tolist() == [1, 25, 0, 0]


def _unit(i: int, owner: int, x: int, y: int, vis: int, flags: int = 0, typ: str = "hfoo") -> np.ndarray:
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = i, code(typ), owner, x, y
    r[fx.C_HP] = r[fx.C_MAXHP] = 400
    r[fx.C_FLAGS], r[fx.C_VIS] = flags, vis
    return r


def test_remembered_enemies(v2):
    fx.set_inputs(3)
    view = fx.Encoder(VOCAB).view(0, 1.0, [0, 1])
    ev = np.zeros((0, 5), np.int64)
    mine = _unit(1, 0, -2000, 0, 1)
    seen = [_unit(2, 1, 2000, 0, 3), _unit(3, 1, 2500, 500, 3, fx.STRUCTURE, "hbar")]
    st = view.step(np.stack([mine, *seen]), None, ev, 0)
    assert st["ghost0"] == 3 and st["ent"][1:, fx.A_MEM].tolist() == [0, 0]
    gone = [_unit(2, 1, 2100, 0, 2), _unit(3, 1, 2500, 500, 2, fx.STRUCTURE, "hbar")]  # (out of sight: player 1's only)
    st = view.step(np.stack([mine, *gone]), None, ev, 4)
    assert st["n"] == 3 and st["ghost0"] == 1
    assert st["sel"][1:, fx.C_ID].tolist() == [3, 2] and st["sel"][2, fx.C_X] == 2000  # buildings first; where it was seen
    assert st["ent"][1:, fx.A_MEM].tolist() == [1, 1] and st["ent"][1, fx.A_MEM + 1] == pytest.approx(2.0 / 60)
    st = view.step(np.stack([mine, *gone]), None, ev, 4 + 121)  # a minute later: the unit forgotten, the building not
    assert st["sel"][1:, fx.C_ID].tolist() == [3]
    near = _unit(1, 0, 2300, 500, 1)  # its place in sight: forgotten
    st = view.step(np.stack([near, *gone]), None, ev, 130)
    assert st["n"] == 1


def test_attack_on_a_remembered_enemy_goes_where_it_was(v2):
    from warcraftsim.fullgame.play import BCAgent
    from warcraftsim.protocol import PointOrder, TargetOrder
    fx.set_inputs(3)
    bot = BCAgent(None, VOCAB, 0, None)
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.trees, bot.building_since, bot.t = {}, {}, 0
    ev = np.zeros((0, 5), np.int64)
    mine = _unit(1, 0, -2000, 0, 1)
    bot.view.step(np.stack([mine, _unit(2, 1, 2000, 0, 3), _unit(3, 1, 0, 0, 3)]), None, ev, 0)
    st = bot.view.step(np.stack([mine, _unit(2, 1, 2100, 0, 2), _unit(3, 1, 0, 0, 3)]), None, ev, 1)
    attack = 4  # (VOCAB's class of attack on a unit)
    cmds = bot.commands(st, [attack], [st["ghost0"]], [0], [0])
    assert cmds == [PointOrder(1, 851983, 2000, 0)]
    cmds = bot.commands(st, [attack], [1], [0], [0])
    assert cmds == [TargetOrder(1, 851983, 3)]
