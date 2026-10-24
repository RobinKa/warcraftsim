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
    tc = Teacher(0, "human", random.Random(0), {"t": {"Npbm": 150.0}})  # (a target's tavern hero: the first one)
    rows = np.stack([_unit(1, 0, 0, 0, 1, fx.WORKER, "hpea"), _unit(2, 0, -3000, 0, 1, fx.STRUCTURE, "halt"),
                     _unit(9, 15, 1000, 0, 1, fx.STRUCTURE, "ntav"), _unit(8, 15, 0, -2000, 1, fx.STRUCTURE, "ngme")])
    rich = SimpleNamespace(gold=900, lumber=300, food_used=10, food_cap=30)
    assert tc.commands(rows, rich, 20.0, 0) == [Buy(1, 9, tc.pick)]  # (its first hero: as soon as the altar stands)
    assert tc.commands(rows, rich, 21.0, 2) == []  # (a lesson every GAP steps)
    assert tc.commands(rows, rich, 200.0, 21) == [Buy(1, 9, tc.pick)]
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
    bot.view.races = [2, 1]  # (undead: they mine their own haunted mines)
    own, other = _unit(5, 0, -300, 0, 1, fx.STRUCTURE, "ugol"), _unit(7, 0, 3000, 300, 1, fx.STRUCTURE, "ugol")
    st3 = bot.view.step(np.stack([miner, own, far, other, tav, gme]), None, np.zeros((0, 5), np.int64), 20)
    bot.view.mine_of[1] = 5
    assert bot.commands(st3, [harvest], [st3["index"][5]], [0], [0]) == []  # (its own mine: redundant)
    assert bot.commands(st3, [harvest], [st3["index"][7]], [0], [0]) == [TargetOrder(1, fx.HARVEST, 7)]  # (another)
    # (a plain gold mine pointed at: the player's haunted one nearest it; an undead harvest there was refused)
    assert bot.commands(st3, [harvest], [st3["index"][6]], [0], [0]) == [TargetOrder(1, fx.HARVEST, 7)]


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



def test_dead_tavern_hero_revived_at_the_altar(v2):
    """A dead tavern hero's hire is its revival at an altar of the player's (tested in the game, 2026-10-23: the
    altar takes it, at 0.4 of its gold); the mask allows it with an altar, its price the revival's."""
    from warcraftsim.fullgame.play import REVIVE, BCAgent
    from warcraftsim.protocol import TargetOrder
    v = AUDIT_VOCAB
    k = 5  # (Npbm, a tavern's)
    m = fx.TechMask(v, "human")
    assert not m.allowed(frozenset({"htow"}), {}, frozenset({"Npbm"}), frozenset(), dead=frozenset({"Npbm"}))[k]  # (no altar)
    assert m.allowed(frozenset({"htow", "halt"}), {}, frozenset({"Npbm"}), frozenset(), dead=frozenset({"Npbm"}))[k]
    costs = np.zeros((len(v["orders"]) + 1, 3), np.int64)
    costs[k] = (425, 135, 5)
    enc = fx.Encoder(v, costs)
    alt, peon = _unit(3, 0, -2500, 0, 1, fx.STRUCTURE, "halt"), _unit(1, 0, -2000, 0, 1, fx.WORKER, "hpea")
    panda = _unit(2, 0, -2100, 0, 1, fx.HERO | fx.DEAD, "Npbm")
    assert fx.revive_costs(enc, np.stack([alt, peon, panda]), 0)[k].tolist() == [170, 0, 5]
    bot = BCAgent(None, v, 0, None)
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.trees, bot.building_since, bot.t = {}, {}, 0
    st = bot.view.step(np.stack([alt, peon, panda]), None, np.zeros((0, 5), np.int64), 0)
    i = st["index"][1]
    order = [0] * (i + 1)
    order[i] = k  # (the worker's hire of it)
    assert bot.commands(st, order, [0] * (i + 1), [0] * (i + 1), [0] * (i + 1)) == [TargetOrder(3, REVIVE, 2)]
    st = bot.view.step(np.stack([peon, panda]), None, np.zeros((0, 5), np.int64), 1)
    i = st["index"][1]
    assert bot.commands(st, order[:i + 1], [0] * (i + 1), [0] * (i + 1), [0] * (i + 1)) == []  # (no altar: nothing)


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
    ne = Teacher(0, "nightelf", random.Random(0), {"t": {"etol": 300.0}})
    assert ne.expand_at == 300.0  # (its tree of life entangles the mine: play.BCAgent._entangle)
    wisps = [_unit(10 + i, 0, -5300, 3300, 1, fx.WORKER, "ewsp") for i in range(3)]
    tree = [_unit(70, 0, -4736, -2368, 1, fx.STRUCTURE, "etol")]
    assert ne.commands(np.stack(base[:3] + wisps + tree), player, 340.0, 60) == []  # (not entangled yet)
    held = [_unit(71, 0, -4864, -3200, 1, fx.STRUCTURE, "egol")]
    c = ne.commands(np.stack(base[:3] + wisps + tree + held), player, 350.0, 90)
    assert len(c) == 3 and all(isinstance(x, TargetOrder) and x.order == HARVEST and x.target == 71 for x in c)


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
    from warcraftsim.protocol import Build, ImmediateOrder, PointOrder, TargetOrder
    vocab = {**AUDIT_VOCAB, "orders": AUDIT_VOCAB["orders"] + [[code("htow"), fx.POINT], [code("ugol"), fx.UNIT]]}
    v = macros.extend(vocab)
    assert macros.extended(v) and v["orders"][-3:] == [[macros.EXPAND, fx.IMMEDIATE], [macros.CREEP, fx.IMMEDIATE],
                                                       [macros.MIX, fx.IMMEDIATE]]
    assert macros.extend(v) is v
    net = FullGameNet(n_types=len(v["types"]) + 1, n_orders=len(vocab["orders"]) + 1, n_cur=2, G=fx.Encoder(vocab).G, d=16, layers=1)
    grown = macros.grow(net, vocab, v)
    k, kc = len(v["orders"]) - 2, len(v["orders"]) - 1
    tix = {t: i + 1 for i, t in enumerate(v["types"])}
    assert grown.config["n_orders"] == kc + 2 and bool(grown.allowed[tix[code("uaco")], k]) and not bool(grown.allowed[tix[code("halt")], k])
    assert torch.equal(grown.order[2].weight[:k], net.order[2].weight) and grown.order[2].bias[k] < net.order[2].bias.min()
    start = (-5184.0, 2944.0)
    main, nat = _unit(50, 15, -5632, 3584, 1, fx.STRUCTURE, "ngol"), _unit(51, 15, -4864, -3200, 1, fx.STRUCTURE, "ngol")
    home = [_unit(1, 0, -5184, 2944, 1, fx.STRUCTURE, "htow"), main, nat, _unit(2, 0, -5300, 3300, 1, fx.WORKER, "hpea")]
    assert macros.site(np.stack(home), start) == (51, -4864.0, -3200.0, -4764.0, -2400.0)
    assert macros.site(np.stack(home + [_unit(60, 12, -4700, -3000, 1, 0, "nomg")]), start) is None  # (its camp)
    assert macros.site(np.stack(home + [_unit(70, 1, -4764, -2400, 1, fx.STRUCTURE, "htow")]), start) is None  # (taken)
    costs = np.zeros((kc + 2, 3), np.int64)
    costs[k - 2] = (193, 103, 0)  # (htow's build)
    costs[4] = (100, 0, 0)  # (hfoo)
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
        st = bot.view.step(np.stack(home), rich, np.zeros((0, 5), np.int64), 2)
        assert not st["avail"][k]  # (one under way: no other expand meanwhile, its hall's cost kept)
        assert not bot.view.step(np.stack(home), np.asarray([0, 0, 250, 300, 5, 12, 0]), np.zeros((0, 5), np.int64), 3)["avail"][4]
        bot.view.expanding = None
        bot.view.races = [2, 1]  # (undead: the mine haunted)
        st = bot.view.step(np.stack(home), rich, np.zeros((0, 5), np.int64), 2)
        assert bot.commands(st, order, [0] * (i + 1), [0] * (i + 1), [0] * (i + 1)) == [TargetOrder(2, code("ugol"), 51)]
        player = SimpleNamespace(gold=500, lumber=300, food_used=5, food_cap=12, start_x=start[0], start_y=start[1])
        tc = Teacher(0, "human", random.Random(0), {"t": {"htow": 300.0}})
        assert tc.commands(np.stack(home), player, 320.0, 1) == [ImmediateOrder(2, macros.EXPAND)]
        # creep: while the natural's camp stands, to it; the teacher shows it to the heroes once the army is 6 strong
        guarded = home + [_unit(60, 12, -4700, -3000, 1, 0, "nomg")]
        assert macros.camp(np.stack(guarded), start) == (-4864.0, -3200.0) and macros.camp(np.stack(home), start) is None
        bot.view.races = [0, 1]
        pal = _unit(80, 0, -5000, 2900, 1, fx.HERO, "Hpal")
        army = [_unit(81 + j, 0, -5000, 2800, 1, 0, "hfoo") for j in range(5)]
        assert not bot.view.step(np.stack(guarded + army), rich, np.zeros((0, 5), np.int64), 400)["avail"][kc]  # (5 units)
        assert not bot.view.step(np.stack(guarded + [pal] + army), rich, np.zeros((0, 5), np.int64), 100)["avail"][kc]  # (50 s)
        st = bot.view.step(np.stack(guarded + [pal] + army), rich, np.zeros((0, 5), np.int64), 400)
        assert st["avail"][kc] and not st["avail"][k]
        i = st["index"][2]
        order = [0] * (i + 1)
        order[i] = kc
        assert bot.commands(st, order, [0] * (i + 1), [0] * (i + 1), [0] * (i + 1)) == [PointOrder(2, 851983, -4864, -3200)]
        tc2 = Teacher(0, "human", random.Random(0), {"t": {"htow": 300.0}})
        assert tc2.commands(np.stack(guarded + [pal] + army), player, 320.0, 1) == [ImmediateOrder(80, macros.CREEP)]
        assert Teacher(0, "human", random.Random(0), {"t": {"htow": 300.0}}).commands(np.stack(guarded + [pal]), player, 320.0, 1) == []
    finally:
        fx.set_macros(False)



def test_own_units_past_the_cap_keep_heroes_army_and_producers(v2):
    """Past MAX_OWN own units: heroes, the army, buildings that make something, workers, then farms are kept, in that
    order (the rows' first-seen order cut the newest)."""
    farms = [_unit(1000 + j, 0, -3000, j, 1, fx.STRUCTURE, "hhou") for j in range(60)]
    workers = [_unit(2000 + j, 0, -3000, j, 1, fx.WORKER, "hpea") for j in range(50)]
    late = [_unit(3000, 0, -3000, 0, 1, fx.STRUCTURE, "hbar"), _unit(3001, 0, -3000, 0, 1, 0, "hfoo"),
            _unit(3002, 0, -3000, 0, 1, fx.HERO, "Hpal")]
    st = fx.Encoder(VOCAB).view(0, 1.0, [0, 1]).step(np.stack(farms + workers + late), None, np.zeros((0, 5), np.int64), 0)
    ids = set(st["sel"][:st["n_own"], fx.C_ID].tolist())
    assert st["n_own"] == fx.MAX_OWN and {3000, 3001, 3002} <= ids and all(2000 + j in ids for j in range(50))


def test_tier_order_cancels_a_busy_halls_queue(v2):
    """A hall's next tier while it trains workers: its queue cancelled first (the game takes no upgrade behind them);
    a hall already upgrading, or idle, gets the order alone."""
    from warcraftsim.fullgame.play import CANCEL, BCAgent
    from warcraftsim.protocol import ImmediateOrder
    v = {**AUDIT_VOCAB, "types": AUDIT_VOCAB["types"] + [code("hkee")], "orders": AUDIT_VOCAB["orders"] + [[code("hkee"), fx.IMMEDIATE]]}
    k = len(v["orders"])
    bot = BCAgent(None, v, 0, None)
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.trees, bot.building_since, bot.t = {}, {}, 0
    hall = _unit(1, 0, -5184, 2944, 1, fx.STRUCTURE, "htow")
    st = bot.view.step(np.stack([hall]), None, np.zeros((0, 5), np.int64), 0)
    assert bot.commands(st, [k], [0], [0], [0]) == [ImmediateOrder(1, code("hkee"))]  # (idle)
    bot.view.busy, bot.view.making = {1}, {1: (0, code("hpea"))}
    assert bot.commands(st, [k], [0], [0], [0]) == [ImmediateOrder(1, CANCEL)] * 6 + [ImmediateOrder(1, code("hkee"))]
    bot.view.making = {1: (4, code("hkee"))}  # (upgrading already: not cancelled)
    assert bot.commands(st, [k], [0], [0], [0]) == [ImmediateOrder(1, code("hkee"))]


def test_teacher_teaches_the_target_army_mix(v2):
    """With its target's army mix known, the teacher has a building that makes it train the unit the target has the
    largest share of more than its own army (by 0.15 at least), from 120 s and 5 army units."""
    import random
    from types import SimpleNamespace

    from warcraftsim.fullgame.teacher import Teacher
    from warcraftsim.protocol import ImmediateOrder
    player = SimpleNamespace(gold=500, lumber=300, food_used=20, food_cap=40, start_x=-5184.0, start_y=2944.0)
    tc = Teacher(0, "human", random.Random(0), {"t": {"hfoo": 60.0}, "n": {"hfoo": 5, "hrif": 5, "hpea": 20}})
    base = [_unit(1, 0, -5184, 2944, 1, fx.STRUCTURE, "htow"), _unit(2, 0, -5000, 2900, 1, fx.STRUCTURE, "hbar")]
    feet = [_unit(10 + i, 0, -5000, 2800, 1, 0, "hfoo") for i in range(6)]
    assert tc.commands(np.stack(base + feet), player, 100.0, 1) == []  # (before 120 s)
    assert tc.commands(np.stack(base + feet), player, 200.0, 2) == [ImmediateOrder(2, code("hrif"))]
    mixed = feet[:3] + [_unit(20 + i, 0, -5000, 2800, 1, 0, "hrif") for i in range(3)]
    assert Teacher(0, "human", random.Random(0), {"t": {}, "n": {"hfoo": 5, "hrif": 5}}).commands(np.stack(base + mixed), player, 200.0, 3) == []


def test_army_mix_lesson_only_for_what_the_policy_can_train(v2):
    """The army mix teaches only units the policy could train now (can): the most lacking of those."""
    import random
    from types import SimpleNamespace

    from warcraftsim.fullgame.teacher import Teacher
    from warcraftsim.protocol import ImmediateOrder
    player = SimpleNamespace(gold=500, lumber=300, food_used=20, food_cap=40, start_x=-5184.0, start_y=2944.0)
    base = [_unit(1, 0, -5184, 2944, 1, fx.STRUCTURE, "htow"), _unit(2, 0, -5000, 2900, 1, fx.STRUCTURE, "hbar")]
    feet = [_unit(10 + i, 0, -5000, 2800, 1, 0, "hfoo") for i in range(6)]
    target = {"t": {}, "n": {"hfoo": 2, "hrif": 6, "hkni": 4}}
    rows = np.stack(base + feet)
    assert Teacher(0, "human", random.Random(0), target).commands(rows, player, 200.0, 1, lambda u: u != "hrif") == \
        [ImmediateOrder(2, code("hkni"))]  # (riflemen lack most but can't be trained now: knights)
    assert Teacher(0, "human", random.Random(0), target).commands(rows, player, 200.0, 1, lambda u: False) == []



def test_mix_macro_trains_the_target_army_mix(v2):
    """--macros' mix: a building trains the unit its target's army lacks most of those it makes and the player can
    train now; the order is available while such a unit lacks."""
    from warcraftsim.fullgame import macros
    from warcraftsim.fullgame.play import BCAgent
    from warcraftsim.protocol import ImmediateOrder
    assert macros.mix_choice({"hfoo": 0.3, "hrif": 0.7}, {"hfoo": 1.0}, "hbar", lambda u: True) == "hrif"
    assert macros.mix_choice({"hfoo": 0.3, "hrif": 0.7}, {"hfoo": 1.0}, "hbar", lambda u: u != "hrif") is None  # (footmen
    assert macros.mix_choice({"hfoo": 0.3, "hkni": 0.7}, {}, "hbar", lambda u: True) in ("hkni", "hfoo")      # enough)
    base = {**AUDIT_VOCAB, "types": AUDIT_VOCAB["types"] + [code("hrif")],
            "orders": AUDIT_VOCAB["orders"] + [[code("hrif"), fx.IMMEDIATE]]}
    v = macros.extend(base)
    km = len(v["orders"])
    costs = np.zeros((km + 1, 3), np.int64)
    fx.set_macros(True)
    try:
        bot = BCAgent(None, v, 0, None, costs=costs)
        bot.view = bot.enc.view(0, 1.0, [0, 1])
        bot.view.start = (-5184.0, 2944.0)
        bot.view.mix_target = {"hfoo": 0.3, "hrif": 0.7}
        bot.trees, bot.building_since, bot.t = {}, {}, 0
        rows = np.stack([_unit(2, 0, -5000, 2900, 1, fx.STRUCTURE, "hbar")] + [_unit(10 + j, 0, -5000, 2800, 1, 0, "hfoo") for j in range(4)])
        me = np.asarray([0, 0, 500, 300, 10, 40, 0])
        st = bot.view.step(rows, me, np.zeros((0, 5), np.int64), 0)
        assert st["avail"][km]
        i = st["index"][2]
        order = [0] * (i + 1)
        order[i] = km
        assert bot.commands(st, order, [0] * (i + 1), [0] * (i + 1), [0] * (i + 1)) == [ImmediateOrder(2, code("hrif"))]
        bot.view.mix_target = {"hfoo": 1.0}
        assert not bot.view.step(rows, me, np.zeros((0, 5), np.int64), 1)["avail"][km]  # (nothing lacks)
    finally:
        fx.set_macros(False)


def test_pickup_macro_takes_the_items_lying_near(v2):
    """--macros 2: pick up joins the vocabulary (a hero's order); it is available while a hero has an item lying near
    that its player sees (and a free slot, but for a powerup); the agent sends the harness's pick-up and keeps the
    hero on it until it holds one; the teacher shows it when no enemy is near."""
    import random
    from types import SimpleNamespace

    import torch

    from warcraftsim.fullgame import macros
    from warcraftsim.fullgame.model import FullGameNet
    from warcraftsim.fullgame.play import PICKUP_COMMIT_STEPS, BCAgent
    from warcraftsim.fullgame.teacher import Teacher
    from warcraftsim.protocol import Event, EventKind, ImmediateOrder, Observation, PickUp
    assert macros.extend(AUDIT_VOCAB, 1)["orders"][-1] == [macros.MIX, fx.IMMEDIATE]
    v1 = macros.extend(AUDIT_VOCAB, 1)
    v = macros.extend(v1, 2)
    assert v["orders"][:-1] == v1["orders"] and v["orders"][-1] == [macros.PICKUP, fx.IMMEDIATE] and v["macros"] == 2
    assert macros.extended(v, 2) and not macros.extended(v1, 2) and macros.extend(v, 1) is v
    net = FullGameNet(n_types=len(v1["types"]) + 1, n_orders=len(v1["orders"]) + 1, n_cur=2, G=fx.Encoder(v1).G, d=16, layers=1)
    grown = macros.grow(net, v1, v)
    kp = len(v["orders"])
    tix = {t: i + 1 for i, t in enumerate(v["types"])}
    assert bool(grown.allowed[tix[code("Hpal")], kp]) and not bool(grown.allowed[tix[code("hpea")], kp])
    assert torch.equal(grown.order[2].weight[:kp], net.order[2].weight)
    pal = _unit(80, 0, -5000, 2900, 1, fx.HERO, "Hpal")
    full = pal.copy()
    full[fx.C_ITEM0:fx.C_ITEM0 + 6] = code("rat6")
    rows = np.stack([_unit(1, 0, -5184, 2944, 1, fx.STRUCTURE, "htow"), pal])
    near, far, unseen = (code("rat6"), -4600, 2900, 1, 0), (code("rat6"), -3900, 2900, 1, 0), (code("rat6"), -4900, 2900, 2, 0)
    tome = (code("tdex"), -4900, 2950, 1, 1)
    assert macros.pickable(rows, 0, [near]) == [80] and macros.pickable(rows, 0, [far, unseen]) == []
    assert macros.pickable(np.stack([rows[0], full]), 0, [near]) == [] and macros.pickable(np.stack([rows[0], full]), 0, [tome]) == [80]
    costs = np.zeros((kp + 1, 3), np.int64)
    fx.set_macros(2)
    try:
        bot = BCAgent(None, v, 0, None, costs=costs)
        bot.view = bot.enc.view(0, 1.0, [0, 1])
        bot.view.start = (-5184.0, 2944.0)
        bot.trees, bot.building_since, bot.t = {}, {}, 0
        me, ev = np.asarray([0, 0, 500, 300, 10, 40, 0]), np.zeros((0, 5), np.int64)
        bot.view.ground = [far]
        assert not bot.view.step(rows, me, ev, 0)["avail"][kp]
        bot.view.ground = [near]
        st = bot.view.step(rows, me, ev, 1)
        assert st["avail"][kp]
        i = st["index"][80]
        order = [0] * (i + 1)
        order[i] = kp
        assert bot.commands(st, order, [0] * (i + 1), [0] * (i + 1), [0] * (i + 1)) == [PickUp(80, int(macros.PICKUP_NEAR))]
        assert bot.committed(80)  # (walking to it: the next orders wait)
        bot.t = PICKUP_COMMIT_STEPS
        assert not bot.committed(80)
        bot.picking_since[80] = bot.t
        obs = Observation(0, 0, False, {}, [], [Event(EventKind.ITEM_PICKUP, 80, code("rat6"), 1)], [])
        bot.observe(obs, bot.t, rows)
        assert not bot.committed(80) and bot.picked_up == 1  # (it holds one: free again)
        player = SimpleNamespace(gold=500, lumber=300, food_used=5, food_cap=12, start_x=-5184, start_y=2944)
        tc = Teacher(0, "human", random.Random(0))
        assert tc.commands(rows, player, 300.0, 1, None, [80]) == [ImmediateOrder(80, macros.PICKUP)]
        foe = _unit(90, 1, -4800, 2900, 1, 0, "ogru")
        tc = Teacher(0, "human", random.Random(0))
        assert tc.commands(np.vstack([rows, foe]), player, 300.0, 1, None, [80]) == []  # (a fight: later)
    finally:
        fx.set_macros(0)


def test_night_elves_entangle_their_expansion(v2):
    """Night elves expand too (--macros): a finished tree of life near a gold mine no one entangled entangles it
    (tested in the game: from 840 away), again after ENTANGLE_EVERY steps until it is; their main's mine is left be."""
    from warcraftsim.fullgame.play import ENTANGLE, ENTANGLE_EVERY, BCAgent
    from warcraftsim.protocol import TargetOrder
    bot = BCAgent(None, AUDIT_VOCAB, 0, None)
    bot.view = bot.enc.view(0, 1.0, [3, 1])
    bot.t = 0
    main, nat = _unit(50, 15, -5632, 3584, 1, fx.STRUCTURE, "ngol"), _unit(51, 15, -4864, -3200, 1, fx.STRUCTURE, "ngol")
    home = [_unit(1, 0, -5300, 3300, 1, fx.STRUCTURE, "etol"), main, _unit(52, 0, -5632, 3584, 1, fx.STRUCTURE, "egol"), nat]
    tree = _unit(2, 0, -4736, -2368, 1, fx.STRUCTURE, "etol")
    assert bot._entangle(np.stack(home)) == []  # (the main's: entangled)
    assert bot._entangle(np.stack(home + [_unit(2, 0, -4736, -2368, 1, fx.STRUCTURE | fx.CONSTRUCTING, "etol")])) == []
    assert bot._entangle(np.stack(home + [tree])) == [TargetOrder(2, ENTANGLE, 51)]
    bot.t = ENTANGLE_EVERY - 1
    assert bot._entangle(np.stack(home + [tree])) == []  # (given: a while to walk there)
    bot.t = ENTANGLE_EVERY
    assert bot._entangle(np.stack(home + [tree])) == [TargetOrder(2, ENTANGLE, 51)]
    held = _unit(53, 0, -4864, -3200, 1, fx.STRUCTURE, "egol")
    assert bot._entangle(np.stack(home + [tree, held])) == []


def test_teacher_first_hero_without_a_target(v2):
    """A game without a target: a side with no hero is shown its altar (a worker near its hall builds one) and then
    its first hero at it, when the policy could order them; with a target the teacher leaves it to the target."""
    import random
    from types import SimpleNamespace

    from warcraftsim.fullgame.teacher import FIRST_HEROES, Teacher
    from warcraftsim.protocol import Build, ImmediateOrder
    player = SimpleNamespace(gold=500, lumber=300, food_used=10, food_cap=22, start_x=-5184, start_y=2944)
    hall, peon = _unit(1, 0, -5184, 2944, 1, fx.STRUCTURE, "htow"), _unit(2, 0, -5000, 2900, 1, fx.WORKER, "hpea")
    tc = Teacher(0, "human", random.Random(0))
    assert tc.commands(np.stack([hall, peon]), player, 30.0, 1) == []  # (not yet)
    c = tc.commands(np.stack([hall, peon]), player, 60.0, 2)
    assert len(c) == 1 and isinstance(c[0], Build) and c[0].unit == 2 and c[0].building == code("halt")
    assert 450 <= np.hypot(c[0].x + 5184, c[0].y - 2944) <= 800
    frame = _unit(3, 0, -4500, 2900, 1, fx.STRUCTURE | fx.CONSTRUCTING, "halt")
    assert tc.commands(np.stack([hall, peon, frame]), player, 90.0, 60) == []  # (being built)
    altar = _unit(3, 0, -4500, 2900, 1, fx.STRUCTURE, "halt")
    c = tc.commands(np.stack([hall, peon, altar]), player, 120.0, 90)
    assert len(c) == 1 and isinstance(c[0], ImmediateOrder) and c[0].unit == 3
    assert fx.rawcode(c[0].order) in FIRST_HEROES["human"]
    assert tc.commands(np.stack([hall, peon, altar]), player, 130.0, 120, lambda u, k=fx.IMMEDIATE: False) == []  # (masked)
    pal = _unit(4, 0, -4600, 2900, 1, fx.HERO, "Hpal")
    assert tc.commands(np.stack([hall, peon, altar, pal]), player, 140.0, 150) == []  # (it has one)
    targeted = Teacher(0, "human", random.Random(0), {"t": {"hfoo": 100.0}})
    assert targeted.commands(np.stack([hall, peon]), player, 60.0, 2) == []


def test_audit3_skills_dead_buildings_held_naturals(v2):
    """Audit 3: --macros 3 spends a skill point the policy left unspent 10 s (the AI's order: the Mountain King's
    second Storm Bolt before its Avatar); the view forgets a building that died (its research masked no more) and
    an enemy's production; a keep holds the natural; the mix order's units are those a train order makes."""
    from warcraftsim.fullgame import macros
    from warcraftsim.fullgame.play import SKILL_WAIT_STEPS, BCAgent
    from warcraftsim.protocol import LearnSkill
    mk = _unit(5, 0, -5000, 2900, 1, fx.HERO, "Hmkg")
    mk[fx.C_HLEVEL], mk[fx.C_SKILLPTS] = 6, 1
    for k, lv in enumerate((3, 1, 1, 0)):  # (slots: Thunder Clap, Storm Bolt, Bash, Avatar)
        mk[fx.C_ABIL0 + 2 * k] = lv
    bot = BCAgent(None, AUDIT_VOCAB, 0, None)
    fx.set_macros(3)
    try:
        bot.t = 0
        assert bot._auto_skills(np.stack([mk]), []) == []  # (the policy's to spend first)
        bot.t = SKILL_WAIT_STEPS
        assert bot._auto_skills(np.stack([mk]), []) == [LearnSkill(5, code("AHtb"))]
        mk[fx.C_ABIL0 + 2], mk[fx.C_ABIL0 + 4] = 3, 1  # (Storm Bolt 3, Bash 1: the AI's next is its ultimate)
        bot.t = 2 * SKILL_WAIT_STEPS
        assert bot._auto_skills(np.stack([mk]), []) == [LearnSkill(5, code("AHav"))]
        bot.t = 3 * SKILL_WAIT_STEPS
        assert bot._auto_skills(np.stack([mk]), [LearnSkill(5, code("AHbh"))]) == []  # (the policy spends it)
    finally:
        fx.set_macros(0)
    view = fx.Encoder(AUDIT_VOCAB).view(0, 1.0, [0, 1])
    altar = _unit(3, 0, -2500, 0, 1, fx.STRUCTURE, "halt")
    ev = np.asarray([[0, 8, 3, code("Rhst"), 0], [0, 5, 9, code("ogru"), 0]], np.int64)  # (research at it; an enemy's)
    enemy = _unit(9, 1, 2500, 0, 1, fx.STRUCTURE, "obar")
    view.step(np.stack([altar, enemy]), None, ev, 0)
    assert 3 in view.making and 3 in view.busy and 9 not in view.busy
    dead = altar.copy()
    dead[fx.C_FLAGS] |= fx.DEAD
    view.step(np.stack([dead, enemy]), None, np.zeros((0, 5), np.int64), 1)
    assert 3 not in view.making and 3 not in view.busy
    start = (-5184.0, 2944.0)
    nat = _unit(51, 15, -4864, -3200, 1, fx.STRUCTURE, "ngol")
    keep = _unit(70, 0, -4764, -2400, 1, fx.STRUCTURE, "hkee")
    assert macros.site(np.stack([nat]), start) is not None and macros.site(np.stack([nat, keep]), start) is None


def test_repair_only_what_needs_it(v2):
    """A repair order goes out only for a building or machine of the player's that is damaged or unfinished (the
    game refused the others: ~1,000-1,500 a game as human and orc); the rest are counted, not sent."""
    from warcraftsim.fullgame.play import REPAIR, BCAgent
    from warcraftsim.protocol import TargetOrder
    v = {**AUDIT_VOCAB, "orders": AUDIT_VOCAB["orders"] + [[REPAIR, fx.UNIT]]}
    k = len(v["orders"])
    bot = BCAgent(None, v, 0, None)
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.trees, bot.building_since, bot.t = {}, {}, 0
    peon = _unit(1, 0, -100, 0, 1, fx.WORKER, "hpea")
    whole, hurt = _unit(2, 0, -500, 0, 1, fx.STRUCTURE, "htow"), _unit(3, 0, -800, 0, 1, fx.STRUCTURE, "halt")
    hurt[fx.C_HP] = 100
    st = bot.view.step(np.stack([peon, whole, hurt]), None, np.zeros((0, 5), np.int64), 0)
    i = st["index"][1]
    order = [0] * (i + 1)
    order[i] = k
    tgt = [0] * (i + 1)
    tgt[i] = st["index"][2]
    assert bot.commands(st, order, tgt, [0] * (i + 1), [0] * (i + 1)) == [] and bot.noop_repairs == 1
    tgt[i] = st["index"][3]
    assert bot.commands(st, order, tgt, [0] * (i + 1), [0] * (i + 1)) == [TargetOrder(1, REPAIR, 3)]


def test_power_build_needs_lumber(v2):
    """A helper's repair at a frame goes out only while the player has lumber (gold for a building that costs no
    lumber): the game refused it without (tested 2026-10-24), ~275 a game as human."""
    from warcraftsim.fullgame.play import REPAIR, BCAgent
    hbar, hlum = (int.from_bytes(c.encode(), "big") for c in ("hbar", "hlum"))
    v = {**AUDIT_VOCAB, "orders": AUDIT_VOCAB["orders"] + [[REPAIR, fx.UNIT], [hbar, fx.POINT], [hlum, fx.POINT]]}
    bot = BCAgent(None, v, 0, None)
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    costs = np.zeros((len(v["orders"]) + 1, 3), np.int64)
    costs[bot.enc.order_index[(hbar, fx.POINT)]] = (160, 60, 0)
    costs[bot.enc.order_index[(hlum, fx.POINT)]] = (120, 0, 0)
    bot.enc.costs = costs
    frame, mill = _unit(2, 0, -500, 0, 1, fx.STRUCTURE | fx.CONSTRUCTING, "hbar"), \
        _unit(3, 0, -800, 0, 1, fx.STRUCTURE | fx.CONSTRUCTING, "hlum")
    me = lambda g, l: np.asarray([0, 0, g, l, 5, 12, 0, 0, 0, 2, 0])  # noqa: E731
    for (g, l), ok_frame, ok_mill in (((5, 0), False, True), ((0, 5), True, False), ((1, 1), True, True), ((0, 0), False, False)):
        bot.me = me(g, l)
        assert bot._repairable(frame) == ok_frame and bot._repairable(mill) == ok_mill, (g, l)


def test_teacher_tavern_hero_at_its_place(v2):
    """A target's tavern hero is taught when the side has as many heroes as come before it in the target (its next
    hero), with the tier that hero count needs and the money (the mask's); without a target, the second hero from
    135 s (audit 7's sequel: as the first or second only, from 135 s, the lesson fired 68 times in 3,000 games)."""
    import random
    from types import SimpleNamespace

    from warcraftsim.fullgame.teacher import Teacher
    from warcraftsim.protocol import Buy
    player = SimpleNamespace(gold=900, lumber=300, food_used=10, food_cap=40)
    base = [_unit(1, 0, 0, 0, 1, fx.WORKER, "hpea"), _unit(2, 0, -3000, 0, 1, fx.STRUCTURE, "halt"),
            _unit(9, 15, 1000, 0, 1, fx.STRUCTURE, "ntav")]
    h1, h2 = _hero(0), _hero(0)
    h1[fx.C_ID], h2[fx.C_ID] = 101, 102
    h2[fx.C_TYPE] = code("Hmkg")
    keep, castle = _unit(3, 0, -3500, 0, 1, fx.STRUCTURE, "hkee"), _unit(4, 0, -3500, 500, 1, fx.STRUCTURE, "hcas")
    third = {"t": {"Hamg": 60.0, "Hmkg": 300.0, "Npbm": 600.0}}  # (its tavern hero the third)
    tc = Teacher(0, "human", random.Random(0), third)
    assert tc.commands(np.stack([*base, keep, h1]), player, 300.0, 0) == []  # (one hero: the target's next is Hmkg)
    assert tc.commands(np.stack([*base, keep, h1, h2]), player, 400.0, 40) == []  # (two: tier 3 needed)
    assert tc.commands(np.stack([*base, castle, h1, h2]), player, 400.0, 80) == [Buy(1, 9, code("Npbm"))]
    poor = lambda u, k=fx.IMMEDIATE: False  # (the mask: not the money)
    assert tc.commands(np.stack([*base, castle, h1, h2]), player, 400.0, 120, poor) == []
    second = Teacher(0, "human", random.Random(0), {"t": {"Hamg": 60.0, "Nalc": 250.0}})
    assert second.commands(np.stack([*base, keep, h1]), player, 150.0, 0) == [Buy(1, 9, code("Nalc"))]
    free = Teacher(0, "human", random.Random(0))  # (no target: the second hero, from 135 s)
    assert free.commands(np.stack([*base, keep, h1]), player, 100.0, 0) == []
    assert free.commands(np.stack([*base, keep, h1]), player, 140.0, 40) == [Buy(1, 9, free.pick)]


def test_teacher_target_first_hero(v2):
    """--teach 2: with a target, its first hero from the altar is taught as soon as the altar stands (the builder had
    taught it only once due, after the policy had made another); a target whose first hero is a tavern's: the
    tavern's lesson, none at the altar."""
    import random
    from types import SimpleNamespace

    from warcraftsim.fullgame.teacher import Teacher
    from warcraftsim.protocol import ImmediateOrder
    player = SimpleNamespace(gold=500, lumber=300, food_used=10, food_cap=22, start_x=-5184, start_y=2944)
    hall, peon = _unit(1, 0, -5184, 2944, 1, fx.STRUCTURE, "htow"), _unit(2, 0, -5000, 2900, 1, fx.WORKER, "hpea")
    altar = _unit(3, 0, -4500, 2900, 1, fx.STRUCTURE, "halt")
    target = {"t": {"halt": 30.0, "Hamg": 70.0, "hbar": 90.0, "Hmkg": 400.0}}
    assert Teacher(0, "human", random.Random(0), target).first is None  # (--teach 1: as before)
    tc = Teacher(0, "human", random.Random(0), target, target_hero=True)
    assert tc.commands(np.stack([hall, peon]), player, 20.0, 40) == []  # (no altar: the builder's, not before 45 s)
    c = tc.commands(np.stack([hall, peon, altar]), player, 33.0, 66)  # (before FIRST_HERO_FROM: 45 s had held 71%
    assert c == [ImmediateOrder(3, code("Hamg"))]                     # of these lessons back, audit 7)
    tavern_first = Teacher(0, "human", random.Random(0), {"t": {"Nalc": 60.0, "Hamg": 300.0}}, target_hero=True)
    assert tavern_first.first is None and tavern_first.pick == code("Nalc")
