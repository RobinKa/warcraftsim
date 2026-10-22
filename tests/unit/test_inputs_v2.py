"""Inputs v2 (selfplay --inputs 2): a hero's abilities and items, what a building makes and the clock as
features; the masks of orders the player lacks the tech for (--tech-mask) and a hero can't carry out now
(--valid-mask); and older checkpoints loaded with zero weights for the new inputs."""

import json

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


def test_adam_state_grows_with_the_inputs(v2, tmp_path):
    from warcraftsim.fullgame.model import FullGameNet, load
    from warcraftsim.fullgame.selfplay import load_optimizer
    fx.set_inputs(1)
    vocab = {**VOCAB, "upgrades": []}
    old = FullGameNet(n_types=5, n_cur=2, n_orders=7, G=fx.Encoder(vocab).G, d=16, layers=1, priv=fx.PRIV)
    opt = torch.optim.Adam(old.parameters(), lr=1e-3)
    old.value(old.encode(torch.randn(1, 2, fx.F), torch.tensor([[1, 2]]), torch.zeros(1, 2, dtype=torch.long),
                         torch.ones(1, 2, dtype=torch.bool), torch.randn(1, old.config["G"]))[0]).sum().backward()
    opt.step()
    torch.save({"config": old.config, "model": old.state_dict(), "vocab": vocab}, tmp_path / "c.pt")
    torch.save(opt.state_dict(), tmp_path / "c.adam")
    fx.set_inputs(3)
    net, _ = load(tmp_path / "c.pt")
    opt2 = torch.optim.Adam(net.parameters(), lr=1e-3)
    assert load_optimizer(opt2, tmp_path / "c.pt", torch.device("cpu"), 1e-3)
    net.value(net.encode(torch.randn(1, 2, fx.F), torch.tensor([[1, 2]]), torch.zeros(1, 2, dtype=torch.long),
                         torch.ones(1, 2, dtype=torch.bool), torch.randn(1, net.config["G"]))[0]).sum().backward()
    opt2.step()  # (the state's shapes fit the grown weights)
    assert opt2.state[net.ent.weight]["exp_avg"].shape == net.ent.weight.shape


def test_strategy_block_grows_with_the_modes(v2, tmp_path):
    from warcraftsim.fullgame import modes, strategy
    from warcraftsim.fullgame.model import FullGameNet, load
    fx.set_inputs(2)
    fx.set_z(strategy.ZDIM)
    try:
        vocab = {**VOCAB, "upgrades": [code("Rhme")]}
        enc = fx.Encoder(vocab)
        torch.manual_seed(0)
        old = FullGameNet(n_types=5, n_cur=2, n_orders=7, G=enc.G, d=32, layers=1, priv=fx.PRIV, z=strategy.ZDIM,
                          gx=fx.GX).eval()
        torch.save({"config": old.config, "model": old.state_dict(), "vocab": vocab}, tmp_path / "z.pt")
        fx.set_z(strategy.ZDIM + modes.MDIM)
        net, _ = load(tmp_path / "z.pt")
        assert net.config["z"] == strategy.ZDIM + modes.MDIM and net.config["G"] == enc.G + modes.MDIM
        G0 = enc.G0
        glob_old = torch.randn(1, enc.G)
        glob = torch.cat([glob_old[:, :G0 + strategy.ZDIM], torch.randn(1, modes.MDIM), glob_old[:, G0 + strategy.ZDIM:]], 1)
        ent, typ = torch.randn(1, 2, fx.F), torch.tensor([[1, 2]])
        cur, mask = torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool)
        with torch.no_grad():  # (the modes' part weighs nothing yet; the clock and the rest read as before)
            assert torch.allclose(net.encode(ent, typ, cur, mask, glob)[0], old.encode(ent, typ, cur, mask, glob_old)[0], atol=1e-5)
    finally:
        fx.set_z(0)


def test_neutral_wares_as_orders(v2, tmp_path):
    """--neutral: a checkpoint's vocabulary gets the neutral buildings' wares and the tavern heroes' orders, its
    network grows (old orders and types as they were); a buy order on a shop is a Buy, on anything else nothing;
    a ware is masked until it is in stock."""
    from warcraftsim.fullgame import neutral
    from warcraftsim.fullgame.model import FullGameNet, load
    from warcraftsim.fullgame.play import BCAgent
    from warcraftsim.protocol import Buy
    fx.set_inputs(2)
    vocab = {**VOCAB, "types": VOCAB["types"] + [code("hpea"), code("hbar")], "upgrades": []}
    torch.manual_seed(0)
    old = FullGameNet(n_types=7, n_cur=2, n_orders=7, G=fx.Encoder(vocab).G, d=16, layers=1, priv=fx.PRIV).eval()
    old.allowed[:] = True
    torch.save({"config": old.config, "model": old.state_dict(), "vocab": vocab}, tmp_path / "o.pt")
    fx.set_neutral(True)
    try:
        net, ck = load(tmp_path / "o.pt")
        v = ck["vocab"]
        assert neutral.extended(v) and v["orders"][:6] == vocab["orders"] and v["types"][:6] == vocab["types"]
        assert net.config["n_orders"] == len(v["orders"]) + 1 and net.config["n_types"] == len(v["types"]) + 1
        oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
        tix = {t: i + 1 for i, t in enumerate(v["types"])}
        hire, learn = oix[(code("Npbm"), fx.UNIT)], oix[(code("ANbf"), fx.SKILL)]
        assert net.allowed[tix[code("hpea")], hire] and not net.allowed[tix[code("hbar")], hire]
        portal = oix[(code("stwp"), fx.UNIT)]  # (the merchant's items: a hero's, not a worker's)
        assert net.allowed[tix[code("Npbm")], portal] and not net.allowed[tix[code("hpea")], portal]
        assert net.allowed[tix[code("Npbm")], learn] and net.allowed[tix[code("Npbm")], oix[(852580, fx.POINT)]]
        g, glob = torch.randn(1, 16), torch.zeros(1, net.config["G"])
        assert torch.allclose(net.order[2](torch.ones(1, 16))[0, :7], old.order[2](torch.ones(1, 16))[0, :7])
        bot = BCAgent(None, v, 0, None)
        bot.view = bot.enc.view(0, 1.0, [0, 1])
        bot.trees, bot.building_since, bot.t = {}, {}, 0
        peon = _unit(1, 0, -100, 0, 1, fx.WORKER, "hpea")
        tav = _unit(9, 15, 0, 0, 1, fx.STRUCTURE, "ntav")
        st = bot.view.step(np.stack([peon, tav]), None, np.zeros((0, 5), np.int64), 0)
        assert bot.commands(st, [hire], [1], [0], [0]) == [Buy(1, 9, code("Npbm"))]
        assert bot.commands(st, [hire], [0], [0], [0]) == []  # (on its way: committed, no other order)
        bot.buying_since = {}
        assert bot.commands(st, [hire], [0], [0], [0]) == [Buy(1, 9, code("Npbm"))]  # (pointed elsewhere: the tavern)
        m = fx.TechMask(v, "human")
        early = m.allowed(frozenset({"htow", "halt"}), {}, frozenset(), frozenset(), now=60.0)
        late = m.allowed(frozenset({"htow", "halt"}), {}, frozenset(), frozenset(), now=200.0)
        assert not early[hire] and late[hire]  # (the tavern's heroes from 135 s)
        assert not m.allowed(frozenset({"htow"}), {}, frozenset(), frozenset(), now=200.0)[hire]  # (an altar first)
    finally:
        fx.set_neutral(False)


def test_teacher_lessons_and_merged_labels(v2):
    import random
    from types import SimpleNamespace

    from warcraftsim.fullgame.selfplay import merge_labels
    from warcraftsim.fullgame.teacher import Teacher
    from warcraftsim.protocol import Buy, ImmediateOrder
    tc = Teacher(0, "human", random.Random(0))
    rows = np.stack([_unit(1, 0, 0, 0, 1, fx.WORKER, "hpea"), _unit(2, 0, -3000, 0, 1, fx.STRUCTURE, "halt"),
                     _unit(9, 15, 1000, 0, 1, fx.STRUCTURE, "ntav"), _unit(8, 15, 0, -2000, 1, fx.STRUCTURE, "ngme")])
    rich = SimpleNamespace(gold=900, lumber=300, food_used=10, food_cap=30)
    assert tc.commands(rows, rich, 100.0, 0) == []  # (the tavern sells from 135 s)
    assert tc.commands(rows, rich, 200.0, 1) == [Buy(1, 9, tc.pick)]
    assert tc.commands(rows, rich, 201.0, 2) == []  # (a lesson every GAP steps)
    hero = _hero(0, items=("phea", "stwp"))
    hero[fx.C_HP], hero[fx.C_X], hero[fx.C_Y] = 200, 0, -1500
    lessons = tc.commands(np.stack([*rows, hero]), rich, 500.0, 40)
    assert ImmediateOrder(int(hero[fx.C_ID]), 852008) in lessons  # (its healing potion: slot 1)
    hero2 = _hero(0, items=("phea",))
    hero2[fx.C_X], hero2[fx.C_Y] = 0, -1500
    assert Buy(int(hero2[fx.C_ID]), 8, int.from_bytes(b"stwp", "big")) in tc.commands(np.stack([*rows, hero2]), rich, 500.0, 80)
    step = {"n_own": 3, "y_order": np.array([0, 4, 0]), "y_ptr": np.array([-1, 2, -1]), "y_x": np.full(3, -1), "y_y": np.full(3, -1)}
    merge_labels(step, {"y_order": np.array([7, 0, 0]), "y_ptr": np.array([1, -1, -1]), "y_x": np.full(3, -1), "y_y": np.full(3, -1)})
    assert step["y_order"].tolist() == [7, 4, 0] and step["y_ptr"].tolist() == [1, 2, -1] and step["y_only"]


def test_remembered_enemies_hidden_until_they_fade_in(v2, tmp_path):
    """inputs 3: a network from before the memory (its config gets "ghost", bias -30) plays as if the remembered
    enemies were not there; at bias 0 they count."""
    from warcraftsim.fullgame.model import FullGameNet, load
    fx.set_inputs(2)
    vocab = {**VOCAB, "upgrades": []}
    torch.manual_seed(0)
    old = FullGameNet(n_types=5, n_cur=2, n_orders=7, G=fx.Encoder(vocab).G, d=16, layers=2, priv=fx.PRIV).eval()
    torch.save({"config": old.config, "model": old.state_dict(), "vocab": vocab}, tmp_path / "g.pt")
    fx.set_inputs(3)
    net, _ = load(tmp_path / "g.pt")
    assert net.config["ghost"] and float(net.ghost_bias) == -30.0
    ent = torch.randn(1, 4, fx.F)
    ent[..., fx.A_MEM] = torch.tensor([0.0, 0.0, 1.0, 1.0])  # (the last two remembered)
    typ, cur, mask, glob = torch.tensor([[1, 2, 3, 4]]), torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.bool), torch.randn(1, net.config["G"])
    with torch.no_grad():
        g, u = net.encode(ent, typ, cur, mask, glob)
        g2, u2 = net.encode(ent[:, :2], typ[:, :2], cur[:, :2], mask[:, :2], glob)
        assert torch.allclose(g, g2, atol=1e-5) and torch.allclose(u[:, :2], u2, atol=1e-5)
        net.ghost_bias.fill_(0.0)
        g3, _ = net.encode(ent, typ, cur, mask, glob)
        assert not torch.allclose(g, g3, atol=1e-4)


def test_privileged_views_and_self_distillation(v2):
    """--opsd-*: the privileged view has every enemy unit (those out of sight marked) and its global flag; a dual view
    gives it beside the player's; the policy's privileged path starts at zero; the distillation's KL in an update."""
    import types

    from warcraftsim.fullgame.model import FullGameNet, act
    from warcraftsim.fullgame.selfplay import collate, opsd_loss
    fx.set_inputs(2, True)
    try:
        enc = fx.Encoder(VOCAB)
        rows = np.stack([_unit(1, 0, -2000, 0, 1), _unit(2, 1, 2000, 0, 3), _unit(3, 1, 2500, 0, 2)])  # (3: out of sight)
        view = enc.view(0, 1.0, [0, 1])
        view.dual = True
        st = view.step(rows, None, np.zeros((0, 5), np.int64), 0)
        assert st["n"] == 2 and st["pn"] == 3 and st["pent"][:, fx.A_HID].tolist() == [0, 0, 1]
        flag = enc.G0 + enc.ZD + enc.GX - 1
        assert st["glob"][flag] == 0 and st["pglob"][flag] == 1
        # (inputs 2: no remembered enemies, though the privileged column makes the rows as wide as inputs 3's)
        seen = enc.view(0, 1.0, [0, 1])
        seen.dual = True
        seen.step(np.stack([rows[0], rows[1], _unit(3, 1, 2500, 0, 3)]), None, np.zeros((0, 5), np.int64), 0)
        assert seen.step(rows, None, np.zeros((0, 5), np.int64), 1)["n"] == 2
        actor = enc.view(0, 1.0, [0, 1])
        actor.privileged_act = True
        st2 = actor.step(rows, None, np.zeros((0, 5), np.int64), 0)
        assert st2["n"] == 3 and st2["glob"][flag] == 1 and "pent" not in st2
        torch.manual_seed(0)
        net = FullGameNet(n_types=5, n_cur=2, n_orders=7, G=enc.G, d=16, layers=1, priv=fx.PRIV, privileged=True).eval()
        net.allowed[:] = True
        with torch.no_grad():
            net.glob[0].weight[:, flag] = 0.0  # (as model.load gives a network from before it)
        g_flag, _ = net.encode(torch.as_tensor(st["ent"])[None], torch.as_tensor(st["type"])[None], torch.as_tensor(st["cur"])[None],
                               torch.ones(1, 2, dtype=torch.bool), torch.as_tensor(st["pglob"])[None])
        g_none, _ = net.encode(torch.as_tensor(st["ent"])[None], torch.as_tensor(st["type"])[None], torch.as_tensor(st["cur"])[None],
                               torch.ones(1, 2, dtype=torch.bool), torch.as_tensor(st["glob"])[None])
        assert torch.allclose(g_flag, g_none)  # (the privileged path: zero at first)
        with torch.no_grad():
            a = act(net, torch.as_tensor(st["ent"])[None], torch.as_tensor(st["type"])[None].long(), torch.as_tensor(st["cur"])[None].long(),
                    torch.ones(1, 2, dtype=torch.bool), torch.as_tensor(st["glob"])[None], torch.tensor([1]))
        step = {**{k: st[k] for k in ("ent", "type", "cur", "glob", "n", "pent", "ptype", "pcur", "pn", "pglob")}, "n_own": 1,
                "avail": None, "adv": 0.0, "ret": 0.0, **{k: a[k][0, :1].numpy() for k in ("order", "tgt", "bx", "by", "logp")}}
        mb = collate([step, {**step, "pent": None}][:1], torch.device("cpu"))
        assert mb["pent"].shape[1] >= 3 and bool(mb["phas"][0])
        from warcraftsim.fullgame.model import evaluate
        ev = evaluate(net, mb["ent"], mb["type"], mb["cur"], mb["mask"], mb["glob"], mb["n_own"], mb["order"], mb["tgt"],
                      mb["bx"], mb["by"], mb["avail"])
        assert float(opsd_loss(net, ev, mb, None)) >= 0.0
        # a batch whose widest step (its own units) has no privileged view: the privileged width covers its orders
        units = [_unit(10 + k, 0, -2000, 10 * k, 1) for k in range(40)] + [_unit(100 + k, 1, -1900, 10 * k, 3) for k in range(70)]
        big = enc.view(0, 1.0, [0, 1]).step(np.stack(units), None, np.zeros((0, 5), np.int64), 0)
        wide = {**{k: big[k] for k in ("ent", "type", "cur", "glob", "n")}, "n_own": 40, "avail": None, "adv": 0.0,
                "ret": 0.0, **{k: np.zeros(40, np.float32 if k == "logp" else np.int64) for k in ("order", "tgt", "bx", "by", "logp")}}
        wide["tgt"] = np.full(40, big["n"] - 1)  # (an enemy past the privileged views' width)
        mb = collate([step, wide], torch.device("cpu"))
        assert mb["pent"].shape[1] >= mb["order"].shape[1] and mb["pent"].shape[1] <= wide["tgt"][0]
        ev = evaluate(net, mb["ent"], mb["type"], mb["cur"], mb["mask"], mb["glob"], mb["n_own"], mb["order"], mb["tgt"],
                      mb["bx"], mb["by"], mb["avail"])
        assert float(opsd_loss(net, ev, mb, None)) >= 0.0
    finally:
        fx.set_inputs(1, False)


def test_no_mercenaries(v2, tmp_path):
    """--neutral 2: no unit type may buy a mercenary (who_buys), the target builder buys none, and the targets have
    none (their items' list, the inputs' slots, unchanged)."""
    from warcraftsim.fullgame import neutral, selfplay as sp
    from warcraftsim.fullgame.model import FullGameNet
    vocab = neutral.extend_vocab(VOCAB)
    net = FullGameNet(n_types=len(vocab["types"]) + 1, n_orders=len(vocab["orders"]) + 1, n_cur=2, G=8, d=16, layers=1)
    tix = {int(t): i + 1 for i, t in enumerate(vocab["types"])}
    oix = {tuple(o): c for c, o in enumerate(vocab["orders"], start=1)}
    merc, tav = oix[(code("nfsp"), fx.UNIT)], oix[(code("Npbm"), fx.UNIT)]
    pool = {"items": {"human": ["hfoo", "nfsp"]}, "strategies": [{"race": "human", "t": {"hfoo": 60.0, "nfsp": 200.0}}]}
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool))
    try:
        for v in (1, 2):
            fx.set_neutral(v)
            neutral.who_buys(net, vocab)
            assert bool(net.allowed[tix[code("Hpal")], merc]) == (v == 1) and bool(net.allowed[tix[code("Hpal")], tav])
        sp._POOLS.clear()
        got = sp.strategy_pool({"z_pool": str(path)})
        assert got["strategies"][0]["t"] == {"hfoo": 60.0} and got["items"]["human"] == ["hfoo", "nfsp"]
    finally:
        fx.set_neutral(0)
        sp._POOLS.clear()


def test_vocabulary_extension_is_deterministic_and_reorderable(v2):
    """--neutral: extend_vocab lists a toggle's two orders sorted (a set's order was the process's hash seed's: runs
    extended in different processes listed robogoblin and unrobogoblin the other way round), and a network of one
    order is put in another's (neutral.reorder: every order-indexed weight and table follows its order)."""
    import copy
    import subprocess
    import sys

    from warcraftsim.fullgame import neutral
    from warcraftsim.fullgame.model import FullGameNet
    code_ = ("import json; from warcraftsim.fullgame.neutral import extend_vocab; "
             "print(json.dumps(extend_vocab({'types': [], 'orders': [], 'upgrades': [], 'current_orders': []})['orders']))")
    outs = {subprocess.run([sys.executable, "-c", code_], capture_output=True, text=True,
                           env={**__import__("os").environ, "PYTHONHASHSEED": s}).stdout for s in ("1", "2", "3")}
    assert len(outs) == 1 and len(json.loads(next(iter(outs)))) > 50
    vocab = neutral.extend_vocab(VOCAB)
    net = FullGameNet(n_types=len(vocab["types"]) + 1, n_orders=len(vocab["orders"]) + 1, n_cur=2, G=8, d=16, layers=1)
    torch.nn.init.normal_(net.order[2].bias)
    net.allowed[:] = torch.rand(net.allowed.shape) > 0.5
    net.order_kind.copy_(torch.as_tensor([0] + [k for _, k in vocab["orders"]]))
    other = {**vocab, "orders": vocab["orders"][:-2] + vocab["orders"][-1:] + vocab["orders"][-2:-1]}
    moved = neutral.reorder(copy.deepcopy(net), vocab, other)
    n = len(vocab["orders"])
    for i, j in ((n, n - 1), (n - 1, n), (1, 1)):  # (class i in `other` is class j in `vocab`)
        assert torch.equal(moved.order[2].bias[i], net.order[2].bias[j])
        assert torch.equal(moved.order_emb.weight[i], net.order_emb.weight[j])
        assert torch.equal(moved.allowed[:, i], net.allowed[:, j]) and moved.order_kind[i] == net.order_kind[j]
    assert neutral.reorder(copy.deepcopy(net), vocab, {**vocab, "orders": vocab["orders"][:-1]}) is None


def test_view_first_hero_free(v2):
    """The view's mask lets a player with no hero (alive or dead) order its first at 50 gold; one with a hero not."""
    costs = np.zeros((len(VOCAB["orders"]) + 1, 3), np.int64)
    costs[6] = (425, 100, 5)  # (Hamg)
    enc = fx.Encoder(VOCAB, costs)
    me = np.asarray([0, 0, 50, 20, 5, 12, 0])
    none = np.stack([_unit(1, 0, -2000, 0, 1)])
    assert enc.view(0, 1.0, [0, 1]).step(none, me, np.zeros((0, 5), np.int64), 0)["avail"][6]
    dead = np.stack([_unit(1, 0, -2000, 0, 1), _unit(2, 0, -2100, 0, 1, fx.HERO | fx.DEAD, "Hpal")])
    assert not enc.view(0, 1.0, [0, 1]).step(dead, me, np.zeros((0, 5), np.int64), 0)["avail"][6]


AUDIT_VOCAB = {"types": [code(c) for c in ("uaco", "ngol", "ugol", "ntav", "ngme", "Hpal", "halt", "hpea", "htow", "unpl")],
               "current_orders": [], "upgrades": [],
               "orders": [[code("ugol"), fx.UNIT], [fx.HARVEST, fx.UNIT], [code("Hpal"), fx.IMMEDIATE], [code("hfoo"), fx.IMMEDIATE],
                          [code("Npbm"), fx.UNIT]]}


def test_audit_fixes_orders(v2):
    """The audit of 2026-10-22: a haunted gold mine is built on a gold mine (a target order), not bought; a ware goes
    to a building that sells it; a buyer in reach of its shop a few steps is free again, a refused one at once; a
    worker may be sent from its gold mine to another one."""
    from warcraftsim.fullgame.play import BUY_ARRIVED, BCAgent
    from warcraftsim.protocol import Buy, TargetOrder
    v = AUDIT_VOCAB
    bot = BCAgent(None, v, 0, None)
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.trees, bot.building_since, bot.t = {}, {}, 0
    aco = _unit(1, 0, -100, 0, 1, fx.WORKER, "uaco")
    home = _unit(5, 15, -300, 0, 1, fx.STRUCTURE, "ugol")
    far = _unit(6, 15, 3000, 0, 1, fx.STRUCTURE, "ngol")
    tav, gme = _unit(9, 15, 0, 2000, 1, fx.STRUCTURE, "ntav"), _unit(8, 15, 0, -2000, 1, fx.STRUCTURE, "ngme")
    st = bot.view.step(np.stack([aco, home, far, tav, gme]), None, np.zeros((0, 5), np.int64), 0)
    idx = st["index"]
    haunt, harvest, hire = 1, 2, 5
    assert bot.commands(st, [haunt], [idx[5]], [0], [0]) == [TargetOrder(1, code("ugol"), 6)]  # (the unhaunted mine)
    assert bot.commands(st, [hire], [idx[8]], [0], [0]) == [Buy(1, 9, code("Npbm"))]  # (the merchant sells no heroes)
    bot.accepted([Buy(1, 9, code("Npbm"))], [False])
    assert not bot.buying_since  # (refused: free again)
    bot.commands(st, [hire], [idx[9]], [0], [0])
    near = _unit(1, 0, 0, 1700, 1, fx.WORKER, "uaco")
    for t in range(BUY_ARRIVED):
        st2 = bot.view.step(np.stack([near, home, far, tav, gme]), None, np.zeros((0, 5), np.int64), t + 1)
        bot.commands(st2, [0], [0], [0], [0])
    assert not bot.buying_since  # (in reach a while: free again, not 60 s)
    miner = _unit(1, 0, -100, 0, 1, fx.WORKER, "uaco")
    miner[fx.C_ORDER] = 852018  # (harvesting)
    st3 = bot.view.step(np.stack([miner, home, far, tav, gme]), None, np.zeros((0, 5), np.int64), 20)
    bot.view.mine_of[1] = 5
    assert bot.commands(st3, [harvest], [st3["index"][5]], [0], [0]) == []  # (its own mine: redundant)
    assert bot.commands(st3, [harvest], [st3["index"][6]], [0], [0]) == [TargetOrder(1, fx.HARVEST, 6)]  # (another)


def test_audit_fixes_revival_labels_and_production(v2):
    """A dead hero may be revived (the order that trained it, at its revival's price); the target builder's and
    teacher's labels at an idle building are kept; a building's "making" reads 1 with no worker on lumber."""
    v = AUDIT_VOCAB
    costs = np.zeros((len(v["orders"]) + 1, 3), np.int64)
    costs[3], costs[4] = (425, 100, 5), (135, 0, 2)
    enc = fx.Encoder(v, costs)
    m = fx.TechMask(v, "human")
    k = 3
    assert not m.allowed(frozenset({"htow", "halt"}), {}, frozenset({"Hpal"}), frozenset())[k]  # (alive: not again)
    assert m.allowed(frozenset({"htow", "halt"}), {}, frozenset({"Hpal"}), frozenset(), dead=frozenset({"Hpal"}))[k]
    pal = _unit(2, 0, -2100, 0, 1, fx.HERO | fx.DEAD, "Hpal")
    pal[fx.C_HLEVEL] = 3
    rows = np.stack([_unit(1, 0, -2000, 0, 1, fx.WORKER, "hpea"), pal, _unit(3, 0, -2500, 0, 1, fx.STRUCTURE, "halt")])
    assert fx.revive_costs(enc, rows, 0)[k].tolist() == [255, 0, 5]  # (425 x (0.4 + 0.1 x 2))
    view = enc.view(0, 1.0, [0, 1])
    me = np.asarray([0, 0, 1000, 500, 5, 30, 0])
    st = view.step(rows, me, np.zeros((0, 5), np.int64), 0)
    label = np.asarray([[0, 3, code("hfoo"), 0, 0, 0, 0]], np.int64)
    y, _ = enc.step_labels(view, st, label, me, np.zeros((0, 5), np.int64), {})
    yt, _ = enc.step_labels(view, st, label, me, np.zeros((0, 5), np.int64), {}, taught=True)
    i = st["index"][3]
    assert y["y_order"][i] == 0 and yt["y_order"][i] == 4  # (an order never given starts nothing: kept when taught)
    view.busy = {3}
    st2 = view.step(rows, me, np.zeros((0, 5), np.int64), 1)
    making = 9 + len(fx.FLAG_BITS) + 7  # (the base block's "being made" column)
    assert not view.assign and st2["ent"][st2["index"][3], making] == 1.0


def test_dead_hero_revived_with_the_revive_order(v2):
    """An altar's order for a dead hero's type is its revival: the revive order on the dead hero (tested in the game:
    the type's own order is refused)."""
    from warcraftsim.fullgame.play import REVIVE, BCAgent
    from warcraftsim.protocol import ImmediateOrder, TargetOrder
    bot = BCAgent(None, AUDIT_VOCAB, 0, None)
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.trees, bot.building_since, bot.t = {}, {}, 0
    alt = _unit(3, 0, -2500, 0, 1, fx.STRUCTURE, "halt")
    pal = _unit(2, 0, -2100, 0, 1, fx.HERO | fx.DEAD, "Hpal")
    st = bot.view.step(np.stack([alt, pal]), None, np.zeros((0, 5), np.int64), 0)
    i = st["index"][3]
    order = [0] * (i + 1)
    order[i] = 3  # (Hpal at the altar)
    assert bot.commands(st, order, [0] * (i + 1), [0] * (i + 1), [0] * (i + 1)) == [TargetOrder(3, REVIVE, 2)]
    st2 = bot.view.step(np.stack([alt]), None, np.zeros((0, 5), np.int64), 1)
    i2 = st2["index"][3]
    order2 = [0] * (i2 + 1)
    order2[i2] = 3
    assert bot.commands(st2, order2, [0] * (i2 + 1), [0] * (i2 + 1), [0] * (i2 + 1)) == [ImmediateOrder(3, code("Hpal"))]


def test_teacher_takes_the_natural_expansion(v2):
    """With its target's expansion due and the natural's camp gone, the teacher has a worker build the hall where
    people do (undead: an acolyte haunts the mine), then three workers mine there; while the camp stands, nothing."""
    import random
    from types import SimpleNamespace

    from warcraftsim.fullgame.teacher import HARVEST, Teacher
    from warcraftsim.protocol import Build, TargetOrder
    player = SimpleNamespace(gold=500, lumber=300, food_used=30, food_cap=60, start_x=-5184.0, start_y=2944.0)
    main, nat = _unit(50, 15, -5632, 3584, 1, fx.STRUCTURE, "ngol"), _unit(51, 15, -4864, -3200, 1, fx.STRUCTURE, "ngol")
    base = [_unit(1, 0, -5184, 2944, 1, fx.STRUCTURE, "htow"), main, nat] + [_unit(10 + i, 0, -5300, 3300, 1, fx.WORKER, "hpea") for i in range(5)]
    army = [_unit(30 + i, 0, -4000, 1000, 1, 0, "hfoo") for i in range(6)]
    camp = [_unit(60, 12, -4700, -3000, 1, 0, "nomg")]
    tc = Teacher(0, "human", random.Random(0), {"t": {"htow": 300.0}})
    assert tc.commands(np.stack(base + army + camp), player, 200.0, 1) == []  # (not due yet)
    assert tc.commands(np.stack(base + army + camp), player, 320.0, 2) == []  # (its camp stands)
    c = tc.commands(np.stack(base + army), player, 330.0, 30)
    assert len(c) == 1 and isinstance(c[0], Build) and c[0].building == code("htow") and (c[0].x, c[0].y) == (-4764.0, -2400.0)
    done = [_unit(70, 0, -4764, -2400, 1, fx.STRUCTURE, "htow")]
    c = tc.commands(np.stack(base + army + done), player, 340.0, 60)
    assert len(c) == 3 and all(isinstance(x, TargetOrder) and x.order == HARVEST and x.target == 51 for x in c)
    ud = Teacher(0, "undead", random.Random(0), {"t": {"ugol": 300.0}})
    acolytes = [_unit(10 + i, 0, -5300, 3300, 1, fx.WORKER, "uaco") for i in range(3)]
    c = ud.commands(np.stack(base[:3] + acolytes), player, 320.0, 1)
    assert c == [TargetOrder(10, code("ugol"), 51)]
    assert Teacher(0, "nightelf", random.Random(0), {"t": {"etol": 300.0}}).expand_at is None  # (no entangle order yet)


def test_expand_macro(v2):
    """--macros: the expand order (a worker's) joins the vocabulary and the network; it may be given while the
    natural is free and its hall affordable; the agent builds the hall where people do (undead: haunts the mine);
    the teacher's expansion lesson is that order."""
    import random
    from types import SimpleNamespace

    import torch

    from warcraftsim.fullgame import macros
    from warcraftsim.fullgame.model import FullGameNet
    from warcraftsim.fullgame.play import BCAgent
    from warcraftsim.fullgame.teacher import Teacher
    from warcraftsim.protocol import Build, ImmediateOrder, TargetOrder
    vocab = {**AUDIT_VOCAB, "orders": AUDIT_VOCAB["orders"] + [[code("htow"), fx.POINT], [code("ugol"), fx.UNIT]]}
    v = macros.extend(vocab)
    assert macros.extended(v) and v["orders"][-1] == [macros.EXPAND, fx.IMMEDIATE] and macros.extend(v) is v
    net = FullGameNet(n_types=len(v["types"]) + 1, n_orders=len(vocab["orders"]) + 1, n_cur=2, G=fx.Encoder(vocab).G, d=16, layers=1)
    grown = macros.grow(net, vocab, v)
    k = len(v["orders"])
    tix = {t: i + 1 for i, t in enumerate(v["types"])}
    assert grown.config["n_orders"] == k + 1 and bool(grown.allowed[tix[code("uaco")], k]) and not bool(grown.allowed[tix[code("halt")], k])
    assert torch.equal(grown.order[2].weight[:k], net.order[2].weight) and grown.order[2].bias[k] < net.order[2].bias.min()
    start = (-5184.0, 2944.0)
    main, nat = _unit(50, 15, -5632, 3584, 1, fx.STRUCTURE, "ngol"), _unit(51, 15, -4864, -3200, 1, fx.STRUCTURE, "ngol")
    home = [_unit(1, 0, -5184, 2944, 1, fx.STRUCTURE, "htow"), main, nat, _unit(2, 0, -5300, 3300, 1, fx.WORKER, "hpea")]
    assert macros.site(np.stack(home), start) == (51, -4864.0, -3200.0, -4764.0, -2400.0)
    assert macros.site(np.stack(home + [_unit(60, 12, -4700, -3000, 1, 0, "nomg")]), start) is None  # (its camp)
    assert macros.site(np.stack(home + [_unit(70, 1, -4764, -2400, 1, fx.STRUCTURE, "htow")]), start) is None  # (taken)
    costs = np.zeros((k + 1, 3), np.int64)
    costs[k - 2] = (193, 103, 0)  # (htow's build)
    fx.set_macros(True)
    try:
        enc = fx.Encoder(v, costs)
        bot = BCAgent(None, v, 0, None, costs=costs)
        bot.view = bot.enc.view(0, 1.0, [0, 1])
        bot.view.start = start
        bot.trees, bot.building_since, bot.t = {}, {}, 0
        rich, poor = np.asarray([0, 0, 500, 300, 5, 12, 0]), np.asarray([0, 0, 50, 300, 5, 12, 0])
        assert not bot.view.step(np.stack(home), poor, np.zeros((0, 5), np.int64), 0)["avail"][k]
        st = bot.view.step(np.stack(home), rich, np.zeros((0, 5), np.int64), 1)
        assert st["avail"][k]
        i = st["index"][2]
        order = [0] * (i + 1)
        order[i] = k
        assert bot.commands(st, order, [0] * (i + 1), [0] * (i + 1), [0] * (i + 1)) == [Build(2, code("htow"), -4764.0, -2400.0)]
        bot.view.races = [2, 1]  # (undead: the mine haunted)
        st = bot.view.step(np.stack(home), rich, np.zeros((0, 5), np.int64), 2)
        assert bot.commands(st, order, [0] * (i + 1), [0] * (i + 1), [0] * (i + 1)) == [TargetOrder(2, code("ugol"), 51)]
        player = SimpleNamespace(gold=500, lumber=300, food_used=5, food_cap=12, start_x=start[0], start_y=start[1])
        tc = Teacher(0, "human", random.Random(0), {"t": {"htow": 300.0}})
        assert tc.commands(np.stack(home), player, 320.0, 1) == [ImmediateOrder(2, macros.EXPAND)]
    finally:
        fx.set_macros(False)
