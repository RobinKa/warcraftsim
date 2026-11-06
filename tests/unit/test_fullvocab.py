"""--full-vocab (fullgame/fullvocab.py): every order a 1.29 melee player can give. Old behaviour with the flag off, a grown
network's old logits exactly as they were, the coverage checker's zero after extend, who may give the new orders, and the
masks (tech, mana, a toggle's state, an item in the slot, night, the siege engine's research)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from warcraftsim.fullgame import features as fx  # noqa: E402


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


VOCAB = {"types": [code("Hpal"), code("hbar"), code("halt"), code("htow"), code("hpea"), code("hvlt"), code("hsor"),
                   code("hars"), code("edoc")],
         "current_orders": [], "upgrades": [],
         "orders": [[852092, fx.UNIT], [code("AHhb"), fx.SKILL], [852008, fx.IMMEDIATE], [851983, fx.UNIT],
                    [851986, fx.POINT], [851972, fx.IMMEDIATE], [code("hfoo"), fx.IMMEDIATE], [code("Hamg"), fx.IMMEDIATE],
                    [852075, fx.UNIT]],
         "order_names": {"852092": "holybolt", "851983": "attack", "851986": "move", "851972": "stop", "852075": "slow"}}


@pytest.fixture()
def game_tables():
    try:
        from warcraftsim.fullgame import fullvocab
        from warcraftsim.fullgame.herodata import hero_table, item_table
        from warcraftsim.fullgame.techtree import tree
        hero_table(), item_table(), tree(), fullvocab.tables(), fullvocab.universe("human")
    except Exception as e:  # noqa: BLE001 (the game's tables: not on every machine)
        pytest.skip(f"no game data: {e}")
    fx.set_inputs(2)
    fx.set_masks(True, True)
    yield
    fx.set_inputs(1)
    fx.set_masks(False, False)
    fx.set_neutral(0)
    fx.set_spells(0)
    fx.set_shop_items(0)
    fx.set_full_vocab(0)


def _flags(full: int) -> None:
    fx.set_neutral(1)
    fx.set_spells(1)
    fx.set_shop_items(1)
    fx.set_full_vocab(full)


def _save(tmp_path):
    from warcraftsim.fullgame.model import FullGameNet
    torch.manual_seed(0)
    old = FullGameNet(n_types=len(VOCAB["types"]) + 1, n_cur=2, n_orders=len(VOCAB["orders"]) + 1, G=fx.Encoder(VOCAB).G,
                      d=16, layers=1, priv=fx.PRIV).eval()
    old.allowed[:] = True
    torch.save({"config": old.config, "model": old.state_dict(), "vocab": VOCAB}, tmp_path / "o.pt")


def test_flag_off_is_as_before(game_tables, tmp_path):
    """Off: a network loaded has --shop-items' vocabulary alone, the valid table's old bits, no mask of its own."""
    from warcraftsim.fullgame import fullvocab, shopitems
    from warcraftsim.fullgame.model import load
    _save(tmp_path)
    _flags(0)
    net, ck = load(tmp_path / "o.pt")
    v = ck["vocab"]
    assert shopitems.extended(v) and not fullvocab.extended(v) and "full_vocab_from" not in v
    assert (fullvocab.DROP0, fx.POINT) not in {tuple(o) for o in v["orders"]}
    assert net.valid_mana == () and "vmana" not in net.config
    from warcraftsim.fullgame.costs import order_costs
    assert order_costs(v, "echofast2").shape[0] == len(v["orders"]) + 1


def test_grow_keeps_every_old_logit(game_tables, tmp_path):
    """On: the old orders and types first; every old logit exactly as before (same state, same inputs), each new one no
    order's less its category's bias; the old types' old orders allowed as before."""
    from warcraftsim.fullgame import fullvocab
    from warcraftsim.fullgame.model import load
    _save(tmp_path)
    _flags(0)
    base, cb = load(tmp_path / "o.pt")
    _flags(1)
    net, cn = load(tmp_path / "o.pt")
    vb, vn = cb["vocab"], cn["vocab"]
    O0, T0 = len(vb["orders"]) + 1, len(vb["types"]) + 1
    assert fullvocab.extended(vn) and vn["orders"][:O0 - 1] == vb["orders"] and vn["types"][:T0 - 1] == vb["types"]
    assert vn["full_vocab_from"] == {"orders": O0 - 1, "types": T0 - 1}
    assert torch.equal(net.allowed[:T0, :O0], base.allowed)
    x = torch.randn(5, 32)
    a, b = base.order(x), net.order(x)
    assert torch.equal(a, b[:, :O0])
    want = torch.tensor([fullvocab.bias_of(o, k) for o, k in vn["orders"][O0 - 1:]])
    assert torch.allclose(b[:, :1] - b[:, O0:], want.expand(5, -1), atol=1e-5)
    assert set(want.tolist()) <= set(fullvocab.BIASES.values()) | {9.0}
    # again: a run's vocabulary with them stays as it is, a snapshot of the run loads into it
    torch.save({"config": net.config, "model": net.state_dict(), "vocab": vn}, tmp_path / "n.pt")
    again, ck2 = load(tmp_path / "n.pt", vocab=vn)
    assert ck2["vocab"]["orders"] == vn["orders"] and torch.equal(again.allowed, net.allowed)


def test_coverage_is_complete_after_extend(game_tables):
    """The checker: a small vocabulary misses much of every race's universe; extended, nothing (EXCLUDED aside), and every
    castable ability of the universe has an order kind."""
    from warcraftsim.fullgame import fullvocab
    before = fullvocab.coverage(VOCAB)
    assert all(before[r]["missing"] and before[r]["missing_types"] for r in fullvocab.RACES)
    v = fullvocab.extend(VOCAB)
    after = fullvocab.coverage(v)
    for r in fullvocab.RACES:
        assert after[r]["missing"] == [] and after[r]["missing_types"] == [], r
        assert after[r]["have"] == after[r]["orders"]
    assert not [e.name for r in fullvocab.RACES for e in fullvocab.universe(r).values() if e.kind < 0]
    assert fullvocab.extend(v) is v  # (once)
    # what the probe found: the forms the game takes
    ids = fullvocab.order_ids()
    names = {(o, k) for o, k in v["orders"]}
    assert (ids["requestsacrifice"], fx.UNIT) in names and (ids["avengerform"], fx.IMMEDIATE) in names
    assert (ids["unavengerform"], fx.IMMEDIATE) not in names and (ids["undivineshield"], fx.IMMEDIATE) not in names
    assert (ids["freezingbreath"], fx.UNIT) not in names and (ids["root"], fx.POINT) not in names
    assert (code("tgrh"), fx.UNIT) not in names and (code("mcri"), fx.UNIT) in names


def test_who_gives_the_new_orders(game_tables, tmp_path):
    """A sorceress polymorph, a druid of the claw its bear form and the bear form its undo only, a hero the item orders,
    every unit patrol and hold position; a new unit type its old generic orders."""
    from warcraftsim.fullgame import fullvocab
    from warcraftsim.fullgame.model import load
    _save(tmp_path)
    _flags(1)
    net, ck = load(tmp_path / "o.pt")
    v = ck["vocab"]
    ids = fullvocab.order_ids()
    oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
    tix = {t: i + 1 for i, t in enumerate(v["types"])}
    A = net.allowed
    assert A[tix[code("hsor")], oix[(ids["polymorph"], fx.UNIT)]] and not A[tix[code("hpea")], oix[(ids["polymorph"], fx.UNIT)]]
    assert A[tix[code("edoc")], oix[(ids["bearform"], fx.IMMEDIATE)]]
    assert not A[tix[code("edcm")], oix[(ids["bearform"], fx.IMMEDIATE)]]
    assert A[tix[code("edcm")], oix[(ids["unbearform"], fx.IMMEDIATE)]]
    assert not A[tix[code("edoc")], oix[(ids["unbearform"], fx.IMMEDIATE)]]
    assert A[tix[code("Hpal")], oix[(fullvocab.DROP0, fx.POINT)]] and not A[tix[code("hpea")], oix[(fullvocab.DROP0, fx.POINT)]]
    for unit in ("hpea", "Hpal", "edoc"):
        assert A[tix[code(unit)], oix[(ids["patrol"], fx.POINT)]] and A[tix[code(unit)], oix[(ids["holdposition"], fx.IMMEDIATE)]]
    assert A[tix[code("ospm")], oix[(851986, fx.POINT)]] and A[tix[code("ospm")], oix[(ids["spiritlink"], fx.UNIT)]]
    assert not A[tix[code("hbar")], oix[(ids["patrol"], fx.POINT)]]


def test_tech_and_valid_masks(game_tables):
    """Tech: polymorph after sorceress training's 2nd level, the siege engine hmtt before the barrage and hrtt after it,
    the critter at a finished arcane vault only. Valid: polymorph's mana bit for a sorceress, the slot bits of the drops."""
    from warcraftsim.fullgame import fullvocab
    from warcraftsim.fullgame.herodata import order_needs
    _flags(1)
    v = fullvocab.extend(VOCAB)
    ids = fullvocab.order_ids()
    oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
    m = fx.TechMask(v, "human")
    poly, hmtt, hrtt = oix[(ids["polymorph"], fx.UNIT)], oix[(code("hmtt"), fx.IMMEDIATE)], oix[(code("hrtt"), fx.IMMEDIATE)]
    critter = oix[(code("mcri"), fx.UNIT)]
    done = frozenset({"htow", "hkee", "hcas", "harm", "hbla", "hars", "hlum"})
    ok = m.allowed(done, {"Rhst": 1}, frozenset(), frozenset())
    assert not ok[poly] and ok[hmtt] and not ok[hrtt] and not ok[critter]
    ok = m.allowed(done | {"hvlt"}, {"Rhst": 2, "Rhrt": 1}, frozenset(), frozenset())
    assert ok[poly] and not ok[hmtt] and ok[hrtt] and ok[critter]
    assert not fx.TechMask(v, "orc").allowed(frozenset({"ogre", "ovln"}), {}, frozenset(), frozenset())[critter]
    needs = order_needs(v)
    tix = {t: i + 1 for i, t in enumerate(v["types"])}
    steps = fullvocab.mana_steps()
    assert needs[(tix[code("hsor")], poly)] == fullvocab.MANA0 + steps.index(220.0)
    assert needs[(-1, oix[(fullvocab.DROP0 + 2, fx.POINT)])] == fullvocab.HELD0 + 2
    assert needs[(-1, oix[(fullvocab.SELL0 + 5, fx.UNIT)])] == fullvocab.HELD0 + 5
    assert needs[(-1, oix[(fullvocab.PICKUP_AT, fx.POINT)])] == fullvocab.FREE_BIT


def test_unit_valid_reads_mana_and_slots(game_tables, tmp_path):
    """The network's mask: a sorceress with 219 mana can't polymorph, with 220 she can; a hero's drop of slot 1 only while it
    holds an item there."""
    from warcraftsim.fullgame import fullvocab
    from warcraftsim.fullgame.model import load
    _save(tmp_path)
    _flags(1)
    net, ck = load(tmp_path / "o.pt")
    v = ck["vocab"]
    ids = fullvocab.order_ids()
    oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
    tix = {t: i + 1 for i, t in enumerate(v["types"])}
    cols = net.valid_cols.tolist()
    poly, drop = cols.index(oix[(ids["polymorph"], fx.UNIT)]), cols.index(oix[(fullvocab.DROP0, fx.POINT)])
    ent = torch.zeros(1, 3, fx.F)
    ent[0, 0, 7], ent[0, 0, 8] = 219.0 / 400.0, 0.4  # (mana / max, max / 1000)
    ent[0, 1, 7], ent[0, 1, 8] = 220.0 / 400.0, 0.4
    it = fx.A_HERO + 4 * fx.HERO_ABIL_F
    ent[0, 2, it] = 1.0  # (a hero's slot 1 holds an item)
    typ = torch.tensor([[tix[code("hsor")], tix[code("hsor")], tix[code("Hpal")]]])
    ok = net.unit_valid(ent, typ)
    assert not ok[0, 0, poly] and ok[0, 1, poly] and ok[0, 2, drop]
    ent[0, 2, it] = 0.0
    assert not net.unit_valid(ent, typ)[0, 2, drop]


def _row(i, typ, flags=0, owner=0, mana=(0, 0), items=()):
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_HP], r[fx.C_MAXHP] = i, code(typ), owner, 100, 100
    r[fx.C_MANA], r[fx.C_MAXMANA] = mana
    r[fx.C_FLAGS], r[fx.C_VIS] = flags, 3
    for k, it in enumerate(items):
        r[fx.C_ITEM0 + k] = code(it) if it else 0
    return r


def test_checks_mask_toggles_items_night(game_tables):
    """The player's mask (fullvocab.Checks): an autocast's off only while a giver has it on (a priest's heal starts on, inner
    fire off), on once the agent turned it on; a drop while a hero holds an item there, a sale with a shop to sell to;
    shadow meld by night only; polymorph with the mana, not on cooldown."""
    from warcraftsim.fullgame import fullvocab
    _flags(1)
    v = fullvocab.extend(VOCAB)
    ids = fullvocab.order_ids()
    oix = {tuple(o): c for c, o in enumerate(v["orders"], start=1)}
    ch = fullvocab.Checks(v)

    class View:
        player, step_seconds, ground = 0, 0.5, ()
        cooling: dict = {}
        autocast: dict = {}
    view = View()
    rows = np.stack([_row(1, "hmpr", mana=(200, 200)), _row(2, "Hpal", fx.HERO, items=("phea",)), _row(3, "hsor", mana=(219, 400)),
                     _row(4, "Emoo", fx.HERO)])
    avail = np.ones(len(v["orders"]) + 1, bool)
    ch.mask(view, rows, avail, 0)
    healoff, ifon, ifoff = (oix[(ids[n], fx.IMMEDIATE)] for n in ("healoff", "innerfireon", "innerfireoff"))
    assert avail[healoff] and avail[ifon] and not avail[ifoff]
    assert avail[oix[(fullvocab.DROP0, fx.POINT)]] and not avail[oix[(fullvocab.DROP0 + 1, fx.POINT)]]
    assert not avail[oix[(fullvocab.SELL0, fx.UNIT)]] and avail[oix[(fullvocab.GIVE0, fx.UNIT)]]  # (no shop; two heroes)
    assert not avail[oix[(fullvocab.GIVE0 + 1, fx.UNIT)]]  # (nothing in slot 2)
    assert not avail[oix[(ids["ambush"], fx.IMMEDIATE)]]  # (day at the start)
    assert not avail[oix[(ids["polymorph"], fx.UNIT)]]  # (219 mana)
    view.autocast[(1, "Ainf")] = True
    rows[2, fx.C_MANA] = 220
    night = int((24 - fx.START_HOUR + fx.DAWN - 1) / 24 * fx.DAY_SECONDS / 0.5)  # (5 in the morning: night still)
    avail[:] = True
    ch.mask(view, np.vstack([rows, _row(5, "hvlt")]), avail, night)
    assert avail[ifoff] and avail[oix[(fullvocab.SELL0, fx.UNIT)]] and avail[oix[(ids["ambush"], fx.IMMEDIATE)]]
    assert avail[oix[(ids["polymorph"], fx.UNIT)]]
    view.cooling[(3, oix[(ids["polymorph"], fx.UNIT)])] = night + 5
    avail[:] = True
    ch.mask(view, rows, avail, night)
    assert not avail[oix[(ids["polymorph"], fx.UNIT)]]


def test_bias_by_category(game_tables):
    from warcraftsim.fullgame import fullvocab
    ids = fullvocab.order_ids()
    assert fullvocab.bias_of(ids["patrol"], fx.POINT) == fullvocab.BIASES["generic"]
    assert fullvocab.bias_of(ids["polymorph"], fx.UNIT) == fullvocab.BIASES["ability"]
    assert fullvocab.bias_of(ids["repairon"], fx.IMMEDIATE) == fullvocab.BIASES["build"]  # (a worker's: as its builds)
    assert fullvocab.bias_of(code("hmtt"), fx.IMMEDIATE) == fullvocab.BIASES["train"]


def test_item_commands_encode():
    from warcraftsim.protocol import _OP_LENGTHS, DropItem, NeutralPointOrder, Op, PickUpAt, encode_commands
    cmds = [DropItem(7, 2, 10.0, -20.0), DropItem(7, 3, target=9), PickUpAt(7, 100.0, 200.0, 400),
            NeutralPointOrder(7, 11, 852023, 0.0, 5.0)]
    for c in cmds:
        assert len(c.encode()) == _OP_LENGTHS[Op(c.encode()[0])]
    ints = encode_commands(cmds)
    assert ints[:6] == [Op.DROP_ITEM, 7, 2, 65546, 65516, 0] and ints[6:12] == [Op.DROP_ITEM, 7, 3, 65536, 65536, 9]


def test_actor_network_masks_as_the_learner(game_tables, tmp_path):
    """On: a network built as the actors build theirs (FullGameNet(**config) and a state dict, never set_valid) has the
    learner's mana bits too, and the same unit_valid on the same inputs (a bit only the learner checked had given the
    update -1e9 log-probabilities: the free inventory slot's, 2026-11-06)."""
    from warcraftsim.fullgame.model import FullGameNet, load
    _save(tmp_path)
    _flags(1)
    net, _ = load(tmp_path / "o.pt")
    assert net.valid_mana and net.config.get("vmana") == 1
    actor = FullGameNet(**net.config)
    actor.load_state_dict(net.state_dict())
    assert actor.valid_mana == net.valid_mana
    torch.manual_seed(1)
    n_types = net.type_emb.num_embeddings
    ent = torch.rand(3, 8, fx.ENT_F if hasattr(fx, "ENT_F") else net.ent.in_features)
    typ = torch.randint(0, n_types, (3, 8))
    assert torch.equal(actor.unit_valid(ent, typ), net.unit_valid(ent, typ))
