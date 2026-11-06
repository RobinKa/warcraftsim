"""--shop-items (fullgame/shopitems.py): the race shops' items as orders. Old behaviour with the flag off, a grown network's
old orders as they were, who may buy, the masks (the shop, what an item needs, a free slot, a slot's form), the forms of
an item's use, the stock against the game's own tables, the episode row's counts."""

import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.protocol import Buy, UseItem  # noqa: E402


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


VOCAB = {"types": [code("Hpal"), code("hbar"), code("halt"), code("htow"), code("hpea"), code("hvlt"), code("Hamg")],
         "current_orders": [], "upgrades": [],
         "orders": [[852092, fx.UNIT], [code("AHhb"), fx.SKILL], [852008, fx.IMMEDIATE], [852008, fx.UNIT], [851983, fx.UNIT],
                    [code("hfoo"), fx.IMMEDIATE], [code("Hamg"), fx.IMMEDIATE]],
         "order_names": {"852092": "holybolt", "851983": "attack"}}


def _row(i: int, typ: str, flags: int = 0, owner: int = 0, x: int = 0, y: int = 0, hp=(400, 400), mana=(0, 0), items=()) -> np.ndarray:
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = i, code(typ), owner, x, y
    r[fx.C_HP], r[fx.C_MAXHP], r[fx.C_MANA], r[fx.C_MAXMANA] = *hp, *mana
    r[fx.C_FLAGS], r[fx.C_VIS] = flags, 3
    for k, it in enumerate(items):
        r[fx.C_ITEM0 + k] = code(it) if it else 0
    return r


@pytest.fixture()
def game_tables():
    try:
        from warcraftsim.fullgame.herodata import hero_table, item_table
        from warcraftsim.fullgame.techtree import tree
        hero_table(), item_table(), tree()
    except Exception as e:  # noqa: BLE001 (the game's tables: not on every machine)
        pytest.skip(f"no game data: {e}")
    fx.set_inputs(2)
    fx.set_masks(True, True)
    yield
    fx.set_inputs(1)
    fx.set_masks(False, False)
    fx.set_neutral(0)
    fx.set_shop_items(0)


def _nets(tmp_path):
    from warcraftsim.fullgame.model import FullGameNet
    torch.manual_seed(0)
    old = FullGameNet(n_types=len(VOCAB["types"]) + 1, n_cur=2, n_orders=len(VOCAB["orders"]) + 1, G=fx.Encoder(VOCAB).G,
                      d=16, layers=1, priv=fx.PRIV).eval()
    old.allowed[:] = True
    torch.save({"config": old.config, "model": old.state_dict(), "vocab": VOCAB}, tmp_path / "o.pt")
    return old


def test_flag_off_is_as_before(game_tables, tmp_path):
    """Off: a network loaded has the neutral vocabulary alone, no slot's new form, and the valid table's old needs."""
    from warcraftsim.fullgame import neutral
    from warcraftsim.fullgame.herodata import order_needs
    from warcraftsim.fullgame.model import load
    _nets(tmp_path)
    fx.set_neutral(1)
    net, ck = load(tmp_path / "o.pt")
    v = ck["vocab"]
    assert v == neutral.extend_vocab(VOCAB) and not v.get("shop_items")
    assert (code("phea"), fx.UNIT) not in {tuple(o) for o in v["orders"]}
    assert 19 not in order_needs(v).values() and int(net.valid_need.max()) < 19
    assert fx.item_use_now(code("ssan"), _row(1, "Hpal", fx.HERO))  # (usable, not in TARGETED_ITEMS: as ever)


def test_grow_keeps_the_old_network(game_tables, tmp_path):
    """On: the old orders first, their logits as they were; a new item's far below no order, a new slot form's less so."""
    from warcraftsim.fullgame import neutral, shopitems
    from warcraftsim.fullgame.model import load
    _nets(tmp_path)
    fx.set_neutral(1)
    base, ck0 = load(tmp_path / "o.pt")
    fx.set_shop_items(1)
    net, ck = load(tmp_path / "o.pt")
    v = ck["vocab"]
    n0 = len(ck0["vocab"]["orders"])
    assert shopitems.extended(v) and neutral.extended(v) and v["orders"][:n0] == ck0["vocab"]["orders"]
    assert v["types"] == ck0["vocab"]["types"] and net.config["n_orders"] == len(v["orders"]) + 1
    x = torch.randn(4, 16)
    a, b = base.order[2](x), net.order[2](x)
    assert torch.allclose(a, b[:, :n0 + 1])
    oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
    assert torch.allclose(b[:, oix[(code("phea"), fx.UNIT)]], b[:, 0] - shopitems.BUY_BIAS)
    assert torch.allclose(b[:, oix[(852013, fx.POINT)]], b[:, 0] - shopitems.USE_BIAS)
    assert (oix[(852009, fx.IMMEDIATE)] and oix[(852013, fx.UNIT)])  # (the slots the demonstrations lacked)
    assert net.order_kind[oix[(852008, fx.POINT)]] == fx.POINT and net.order_kind[oix[(code("phea"), fx.UNIT)]] == fx.UNIT
    assert int(net.valid_need.max()) == 19  # (a free inventory slot: what an item bought needs)
    # again: a vocabulary with them stays, a network loads into its orders (a snapshot of the run)
    torch.save({"config": net.config, "model": net.state_dict(), "vocab": v}, tmp_path / "n.pt")
    again, ck2 = load(tmp_path / "n.pt", vocab=v)
    assert ck2["vocab"]["orders"] == v["orders"] and again.config["n_orders"] == net.config["n_orders"]


def test_who_buys(game_tables, tmp_path):
    """A hero its items and its slots' forms; a worker, a building never; what can't be relied on is no order."""
    from warcraftsim.fullgame import shopitems
    from warcraftsim.fullgame.model import load
    _nets(tmp_path)
    fx.set_neutral(1)
    fx.set_shop_items(1)
    net, ck = load(tmp_path / "o.pt")
    v = ck["vocab"]
    oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
    tix = {t: i + 1 for i, t in enumerate(v["types"])}
    for c in shopitems.ITEMS:
        k = oix[(code(c), fx.UNIT)]
        assert net.allowed[tix[code("Hpal")], k] and net.allowed[tix[code("Npbm")], k]
        assert not net.allowed[tix[code("hpea")], k] and not net.allowed[tix[code("hvlt")], k]
    for s in shopitems.SLOTS:
        for kind in shopitems.SLOT_KINDS:
            assert net.allowed[tix[code("Hpal")], oix[(s, kind)]]
    assert not net.allowed[tix[code("hpea")], oix[(852013, fx.POINT)]]
    assert (code("tgrh"), fx.UNIT) not in oix and (code("mcri"), fx.UNIT) not in oix
    assert net.allowed[tix[code("Hpal")], oix[(code("stel"), fx.UNIT)]]  # (the merchant's staff: its use is known now)


def test_masks_shop_needs_and_stock(game_tables):
    """An item needs its race's finished shop and what the item requires; another race's nothing; what the merchant sells
    too also from its stock time on."""
    from warcraftsim.fullgame import neutral, shopitems
    v = shopitems.extend(neutral.extend_vocab(VOCAB))
    oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
    c = lambda s: oix[(code(s), fx.UNIT)]  # noqa: E731
    m = fx.TechMask(v, "human")
    none = m.allowed(frozenset({"htow"}), {}, frozenset(), frozenset(), now=10.0)
    assert not none[c("plcl")] and not none[c("sreg")] and not none[c("phea")]
    vault = m.allowed(frozenset({"htow", "hvlt"}), {}, frozenset(), frozenset(), now=11.0)
    assert vault[c("plcl")] and vault[c("sreg")] and not vault[c("phea")]  # (a potion: tier 2, no stock time here)
    keep = m.allowed(frozenset({"htow", "hkee", "hvlt"}), {}, frozenset(), frozenset(), now=12.0)
    assert keep[c("phea")] and keep[c("pman")] and keep[c("stwp")] and keep[c("tsct")] and not keep[c("ssan")]
    assert not keep[c("hslv")] and not keep[c("rnec")] and not keep[c("moon")]  # (other races' shops)
    castle = m.allowed(frozenset({"htow", "hkee", "hcas", "hvlt"}), {}, frozenset(), frozenset(), now=13.0)
    assert castle[c("ssan")] and castle[c("ofir")]
    merchant = m.allowed(frozenset({"htow"}), {}, frozenset(), frozenset(), now=500.0)  # (no vault: the merchant's)
    assert merchant[c("stwp")] and merchant[c("dust")] and not merchant[c("phea")]
    assert merchant[c("stel")]
    undead = fx.TechMask(v, "undead").allowed(frozenset({"unpl", "utom"}), {}, frozenset(), frozenset(), now=14.0)
    assert undead[c("rnec")] and not undead[c("skul")] and not undead[c("shea")]
    assert fx.TechMask(v, "undead").allowed(frozenset({"unpl", "utom", "ugrv"}), {}, frozenset(), frozenset(), now=14.0)[c("skul")]
    assert fx.TechMask(v, "orc").allowed(frozenset({"ogre", "ovln"}), {}, frozenset(), frozenset(), now=14.0)[c("hslv")]
    assert fx.TechMask(v, "nightelf").allowed(frozenset({"etol", "eden"}), {}, frozenset(), frozenset(), now=14.0)[c("moon")]


def test_valid_mask_free_slot(game_tables, tmp_path):
    """A hero with all six slots full may buy no item (with none the sale goes through and the item lands on the ground);
    a tavern hero's hire is not an item."""
    from warcraftsim.fullgame.model import load
    _nets(tmp_path)
    fx.set_neutral(1)
    fx.set_shop_items(1)
    net, ck = load(tmp_path / "o.pt")
    v = ck["vocab"]
    oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
    enc = fx.Encoder(v)
    view = enc.view(0, 1.0, [0, 1])
    free = _row(1, "Hpal", fx.HERO, items=("stwp",))
    full = _row(2, "Hpal", fx.HERO, x=100, items=("stwp", "phea", "pman", "plcl", "sreg", "tsct"))
    st = view.step(np.stack([free, full]), None, np.zeros((0, 5), np.int64), 0)
    ok = net.unit_valid(torch.as_tensor(st["ent"]).unsqueeze(0).float(), torch.as_tensor(st["type"]).unsqueeze(0).long())
    at = net.valid_cols.tolist().index(oix[(code("phea"), fx.UNIT)])
    assert ok[0, 0, at] and not ok[0, 1, at]
    assert ok[0, 1, net.valid_cols.tolist().index(oix[(code("Npbm"), fx.UNIT)])]


def test_free_slot_mask_in_the_actors_network(game_tables, tmp_path):
    """The actors' networks (selfplay.Nets: the published config and state dict; PastNet likewise) mask a full
    inventory's buy as the learner's (model.load, set_valid) does: the free slot had been an attribute of set_valid's
    alone, so the actors sampled buys the learner held at log-probability -1e9 (an update's approximate KL ~1e5)."""
    from warcraftsim.fullgame.model import FullGameNet, evaluate, load
    _nets(tmp_path)
    fx.set_neutral(1)
    fx.set_shop_items(1)
    net, ck = load(tmp_path / "o.pt")
    live = FullGameNet(**net.config).eval()  # (as Nets._live builds "current" from current.pt)
    live.load_state_dict(net.state_dict())
    v = ck["vocab"]
    oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
    view = fx.Encoder(v).view(0, 1.0, [0, 1])
    full = _row(2, "Hpal", fx.HERO, items=("stwp", "phea", "pman", "plcl", "sreg", "tsct"))
    st = view.step(np.stack([full]), None, np.zeros((0, 5), np.int64), 0)
    ent, typ = torch.as_tensor(st["ent"]).unsqueeze(0).float(), torch.as_tensor(st["type"]).unsqueeze(0).long()
    assert torch.equal(live.unit_valid(ent, typ), net.unit_valid(ent, typ))
    buy = oix[(code("phea"), fx.UNIT)]
    assert not live.unit_valid(ent, typ)[0, 0, live.valid_cols.tolist().index(buy)]
    # the learner's log-probability of an order the actor's network may sample: finite, never the mask's -1e9
    with torch.no_grad():
        g, u = live.encode(ent, typ, torch.zeros_like(typ), torch.ones_like(typ, dtype=torch.bool), torch.zeros(1, live.config["G"]))
        lg = live.order_logits(g, u, typ, torch.tensor([1]), ent=ent)
        allowed = (lg[0, 0] > -1e8).nonzero().flatten().tolist()
        zero = torch.zeros(1, 1, dtype=torch.long)
        for c in allowed:
            lp = evaluate(net, ent, typ, torch.zeros_like(typ), torch.ones_like(typ, dtype=torch.bool), torch.zeros(1, net.config["G"]),
                          torch.tensor([1]), torch.tensor([[c]]), zero, zero, zero)["logp"]
            assert float(lp) > -1e4, c


def test_slot_forms_mask_and_commands(game_tables):
    """Each slot's form is open only while a hero holds an item there that takes it, and goes out as UseItem in that form."""
    from warcraftsim.fullgame import neutral, shopitems
    from warcraftsim.fullgame.play import BCAgent
    fx.set_neutral(1)
    fx.set_shop_items(1)
    v = shopitems.extend(neutral.extend_vocab(VOCAB))
    oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
    bot = BCAgent(None, v, 0, None)
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.trees, bot.building_since, bot.t = {}, {}, 0
    hero = _row(1, "Hpal", fx.HERO, hp=(100, 400), mana=(10, 300), items=("phea", "stwp", "ofir", "ssan", "hslv", "skul"))
    foot = _row(2, "hfoo", 0, x=50)
    hall = _row(3, "htow", fx.STRUCTURE, x=2000)
    rows = np.stack([hero, foot, hall])
    avail = np.ones(len(v["orders"]) + 1, bool)
    bot.view.item_slots(rows, avail)
    open_ = {(s - 852008, k): bool(avail[oix[(s, k)]]) for s in shopitems.SLOTS for k in shopitems.SLOT_KINDS}
    N, U, P = fx.IMMEDIATE, fx.UNIT, fx.POINT
    assert open_[(0, N)] and not open_[(0, U)] and not open_[(0, P)]  # (a potion)
    assert not open_[(1, N)] and open_[(1, P)] and not open_[(1, U)]  # (a town portal)
    assert not any(open_[(2, k)] for k in (N, U, P))  # (an orb: it acts held)
    assert not open_[(3, N)] and open_[(3, U)] and not open_[(3, P)]  # (a staff of sanctuary)
    assert open_[(4, N)] and open_[(4, U)]  # (a salve)
    assert open_[(5, P)] and not open_[(5, N)]  # (a sacrificial skull)
    st = bot.view.step(rows, None, np.zeros((0, 5), np.int64), 0)
    sel = st["sel"]
    at = {int(i): k for k, i in enumerate(sel[:, fx.C_ID].tolist())}
    i = at[1]

    def send(slot, kind, tgt=0):
        order = [0] * st["n_own"]
        order[i] = oix[(852008 + slot, kind)]
        target = [tgt] * st["n_own"]
        bins = [0] * st["n_own"]
        bot.casting_until = {}
        return bot.commands(st, order, target, bins, bins)
    assert send(0, N) == [UseItem(1, 0)]
    got = send(1, P)[0]  # (a portal's point, near none of the player's structures here: at its nearest)
    assert (got.slot, got.x, got.y) == (1, 2000, 0) and got.target == 0
    assert send(1, U, at[3]) == []  # (the game's portal takes no unit)
    assert send(3, U, at[2]) == [UseItem(1, 3, target=2)]  # (the staff on the footman)
    assert send(3, U, i) == [UseItem(1, 3, target=2)]  # (the game refuses it on the hero itself: the unit nearest)
    assert send(4, N) == [UseItem(1, 4)] and send(4, U, at[2]) == [UseItem(1, 4, target=2)] and send(4, U, at[3]) == [UseItem(1, 4, target=1)]
    assert send(2, N) == [] and send(0, U) == []  # (no such form: not sent, counted)
    assert bot.futile_orders == 3
    staffed = hero.copy()
    staffed[fx.C_ITEM0 + 2] = code("stel")
    st2 = {**st, "sel": sel.copy()}
    st2["sel"][i, fx.C_ITEM0 + 2] = code("stel")
    order = [0] * st["n_own"]
    order[i] = oix[(852010, U)]
    assert bot.commands(st2, order, [at[2]] * st["n_own"], order, order) == [UseItem(1, 2, target=3)]  # (no building pointed at)
    assert bot.commands(st2, order, [at[3]] * st["n_own"], order, order) == [UseItem(1, 2, target=3)]
    # a full-health hero wastes no heal, a full one no mana potion; other forms alike
    well = _row(1, "Hpal", fx.HERO, hp=(400, 400), mana=(300, 300), items=("phea", "pman"))
    assert not fx.item_use_now(code("phea"), well) and not fx.item_use_now(code("pman"), well)
    assert fx.item_use_now(code("phea"), hero) and fx.item_use_now(code("plcl"), hero)
    assert not fx.item_use_now(code("stwp"), hero) and not fx.item_use_now(code("ssan"), hero)


def test_buy_goes_to_the_sides_own_shop(game_tables):
    """Its own finished shop, nearest, not an enemy's beside it nor a frame; the merchant for what it sells; pointed at
    the merchant it goes there; no free slot or no such shop: nothing sent."""
    from warcraftsim.fullgame import neutral, shopitems
    from warcraftsim.fullgame.play import BCAgent
    fx.set_neutral(1)
    fx.set_shop_items(1)
    v = shopitems.extend(neutral.extend_vocab(VOCAB))
    oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
    bot = BCAgent(None, v, 0, None)
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.trees, bot.building_since, bot.t = {}, {}, 2000
    hero = _row(1, "Hpal", fx.HERO, items=("stwp",))
    keep = _row(2, "hkee", fx.STRUCTURE, x=500)
    vault = _row(3, "hvlt", fx.STRUCTURE, x=900)
    frame = _row(4, "hvlt", fx.STRUCTURE | fx.CONSTRUCTING, x=100)
    enemy = _row(5, "hvlt", fx.STRUCTURE, owner=1, x=50)
    merchant = _row(6, "ngme", fx.STRUCTURE, owner=15, x=4000)
    rows = np.stack([hero, keep, vault, frame, enemy, merchant])
    st = bot.view.step(rows, None, np.zeros((0, 5), np.int64), 2000)
    sel = st["sel"]
    at = {int(i): k for k, i in enumerate(sel[:, fx.C_ID].tolist())}
    i = at[1]

    def buy(item, tgt, row=None):
        order = [0] * st["n_own"]
        order[i] = oix[(code(item), fx.UNIT)]
        s = st if row is None else {**st, "sel": row}
        bot.buying_since = {}
        return bot.commands(s, order, [tgt] * st["n_own"], [0] * st["n_own"], [0] * st["n_own"])
    assert buy("phea", at[5]) == [Buy(1, 3, code("phea"))]  # (pointed at the enemy's vault: its own)
    assert buy("phea", at[3]) == [Buy(1, 3, code("phea"))]
    assert buy("stwp", at[6]) == [Buy(1, 6, code("stwp"))]  # (the merchant, pointed at)
    assert buy("stwp", i) == [Buy(1, 3, code("stwp"))]  # (the nearer)
    assert bot.buying_at[1] == (code("stwp"), "hvlt")
    assert buy("ofir", at[3]) == []  # (a castle's)
    assert buy("rnec", at[3]) == []  # (undead's)
    fuller = sel.copy()
    fuller[i, fx.C_ITEM0:fx.C_ITEM0 + 6] = code("plcl")
    assert buy("phea", at[3], fuller) == []  # (no free slot)


def test_episode_counts(game_tables):
    """What reached a hero from a purchase (by where it was ordered), not off the ground; what left a living hero; a use's
    order answered, a point's however it was answered (the game says False, and acts)."""
    from types import SimpleNamespace as NS

    from warcraftsim.fullgame import neutral, shopitems
    from warcraftsim.fullgame.play import BCAgent
    from warcraftsim.protocol import EventKind
    fx.set_shop_items(1)
    bot = BCAgent(None, shopitems.extend(neutral.extend_vocab(VOCAB)), 0, None)
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.trees, bot.t = {}, 0
    bot.buying_at = {1: (code("phea"), "hvlt")}
    ev = NS(kind=EventKind.ITEM_PICKUP, a=1, b=code("phea"), c=0)
    rows = np.stack([_row(1, "Hpal", fx.HERO, items=("stwp", "phea"))])
    bot._count_items(rows, [NS(kind=EventKind.ITEM_PICKUP, a=1, b=code("stwp"), c=0), ev])
    assert bot.shop_bought == {"phea": 1} and not bot.buying_at
    bot.buying_at = {1: (code("stwp"), "ngme")}
    bot._count_items(rows, [NS(kind=EventKind.ITEM_PICKUP, a=1, b=code("stwp"), c=1)])  # (off the ground)
    assert "merchant_bought" not in bot.__dict__
    bot._count_items(rows, [NS(kind=EventKind.ITEM_PICKUP, a=1, b=code("stwp"), c=0)])
    assert bot.merchant_bought == {"stwp": 1}
    bot._count_items(np.stack([_row(1, "Hpal", fx.HERO, items=("stwp",))]), [])
    assert bot.items_gone == {"phea": 1}
    dead = _row(1, "Hpal", fx.HERO | fx.DEAD)  # (its items fall as it dies: no use)
    bot._count_items(np.stack([dead]), [])
    assert bot.items_gone == {"phea": 1}
    bot.use_sent = {(1, 0): ("phea", fx.IMMEDIATE), (1, 1): ("stwp", fx.POINT), (1, 2): ("plcl", fx.IMMEDIATE)}
    bot.accepted([UseItem(1, 0), UseItem(1, 1, x=5, y=5), UseItem(1, 2)], [True, False, False])
    assert bot.item_uses == {"phea": 1, "stwp": 1}


def test_stock_against_the_game(game_tables):
    """STOCK, ITEM_NEEDS and the costs against the game's tables."""
    from warcraftsim.data.mpq import GameArchives
    from warcraftsim.fullgame import costs, neutral, shopitems
    needs, stock = {}, {}
    with GameArchives() as g:
        for race in ("Human", "Orc", "Undead", "NightElf"):
            for m in re.finditer(r"\[(\w{4})\][^\[]*?Makeitems=([^\r\n]*)", g.read(f"Units\\{race}UnitFunc.txt").decode("latin-1")):
                stock[m.group(1)] = m.group(2).split(",")
        sec = None
        for line in g.read("Units\\ItemFunc.txt").decode("latin-1").splitlines():
            m = re.match(r"\[(\w{4})\]", line)
            sec = m.group(1) if m else sec
            if line.startswith("Requires="):
                needs[sec] = line.split("=")[1].strip()
    for shop, have in shopitems.STOCK.items():
        assert set(have) <= set(stock[shop]) and set(stock[shop]) - set(have) <= {"mcri", "tgrh"}
    for c in shopitems.ITEMS:
        assert shopitems.ITEM_NEEDS.get(c) == needs.get(c), c
    v = shopitems.extend(neutral.extend_vocab(VOCAB))
    table = costs.order_costs(v, "echofast2")
    oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
    assert table[oix[(code("phea"), fx.UNIT)]].tolist() == [150, 0, 0]
    assert table[oix[(code("tsct"), fx.UNIT)]].tolist() == [40, 20, 0]
    assert set(shopitems.ITEMS) <= set(shopitems.USES)


def test_scorecard_measures():
    """scripts/scorecard.py: the totals a game; a run without the flag has no such lines (None)."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "scripts" / "scorecard.py"
    spec = importlib.util.spec_from_file_location("scorecard", path)
    card = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(card)
    row = {"outcome": 1, "game_time": 600.0, "prod": {}, "opp_prod": {}}
    got = card.measures({**row, "shop_bought": {"phea": 2, "rnec": 1}, "item_uses": {"phea": 1}, "items_gone": {"phea": 2}})
    assert (got["race-shop items bought"], got["items used (orders)"], got["items gone (used up)"]) == (3.0, 1.0, 2.0)
    old = card.measures(row)
    assert old["race-shop items bought"] is None and old["items used (orders)"] is None and old["items gone (used up)"] is None
