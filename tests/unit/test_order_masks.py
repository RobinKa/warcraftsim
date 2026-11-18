"""--order-masks (fullgame/ordermasks.py): orders the game refuses that no other mask caught (bug finder 10's refusals of
fgself-224/225). Bit 1 the pointer of a unit-target spell by its AbilityData targs1, and the order while an entity in
view takes it (in the network: acting and the update alike); bit 2 toggles against their state (the view's avail, from
obs-full's buff events) and forms (the allowed table); bit 4 learns by a hero type with the skill (the valid mask's
needs). Off, nothing changes."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame import ordermasks as om  # noqa: E402


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


IDS = {"frostnova": 852226, "thunderbolt": 852095, "holybolt": 852092, "heal": 852063, "impale": 852555,
       "blizzard": 852089, "attack": 851983, "harvest": 852018, "immolation": 852177, "unimmolation": 852178,
       "manashieldon": 852589, "manashieldoff": 852590, "robogoblin": 852656, "unrobogoblin": 852657}
TYPES = ("Ulic", "ugho", "uzig", "ogru", "opeo", "ocat", "hgry", "nfrl", "Hpal", "Edem", "Nngs", "Ntin", "Nrob")
VOCAB = {"types": [code(c) for c in TYPES], "current_orders": [], "upgrades": [],
         "orders": [[IDS["frostnova"], fx.UNIT], [IDS["thunderbolt"], fx.UNIT], [IDS["holybolt"], fx.UNIT],
                    [IDS["heal"], fx.UNIT], [IDS["impale"], fx.UNIT], [IDS["blizzard"], fx.POINT],
                    [IDS["attack"], fx.UNIT], [IDS["harvest"], fx.UNIT], [code("AUfn"), fx.SKILL],
                    [IDS["immolation"], fx.IMMEDIATE], [IDS["unimmolation"], fx.IMMEDIATE],
                    [IDS["manashieldon"], fx.IMMEDIATE], [IDS["manashieldoff"], fx.IMMEDIATE],
                    [IDS["robogoblin"], fx.IMMEDIATE], [IDS["unrobogoblin"], fx.IMMEDIATE]],
         "order_names": {str(v): k for k, v in IDS.items()}}
(NOVA, BOLT, LIGHT, HEAL, IMPALE, BLIZZARD, ATTACK, HARVEST, LEARN_NOVA, IMMO, UNIMMO, SHIELD, UNSHIELD, ROBO,
 UNROBO) = range(1, 16)
T = {c: i for i, c in enumerate(TYPES, start=1)}


@pytest.fixture(autouse=True)
def tables():
    try:
        from warcraftsim.fullgame import fullvocab
        from warcraftsim.fullgame.herodata import hero_table, unit_kinds
        fullvocab.tables(), unit_kinds(), hero_table()
    except Exception as e:  # noqa: BLE001 (the game's tables: not on every machine)
        pytest.skip(f"no game data: {e}")
    yield
    fx.set_order_masks(0)
    fx.set_inputs(1)
    fx.set_masks(False, False)


def test_targs_rules():
    nova = om.targs_rule("ground,enemy,air,neutral,organic")  # (AUfn)
    assert nova == om.OWN | om.SELF | om.STRUCT | om.MECH | om.UNSEEN
    light = om.targs_rule("air,ground,organic,notself,invu,vuln,nonancient")  # (AHhb: any side, not the paladin)
    assert light == om.SELF | om.STRUCT | om.MECH | om.ANCIENT | om.UNSEEN
    heal = om.targs_rule("air,ground,friend,vuln,invu,self,organic,nonancient,neutral")  # (Ahea)
    assert heal == om.ENEMY | om.STRUCT | om.MECH | om.ANCIENT | om.UNSEEN
    assert om.targs_rule("air,enemy,neutral") == om.OWN | om.SELF | om.STRUCT | om.GROUND | om.UNSEEN  # (web)
    assert om.targs_rule("air,ground,ward") & om.GROUND == 0  # (a ward is a ground unit to the features)
    assert om.targs_rule("dead") == om.ALIVE | om.UNSEEN  # (raise dead: corpses only)
    assert om.targs_rule("_") == om.targs_rule("") == om.targs_rule("none") == 0
    # (the abilities sharing an order: an entity refused only when all of them refuse it)
    assert om.order_rule("thunderbolt") & om.STRUCT == 0 and om.order_rule("thunderbolt") & om.OWN


def test_rules_only_for_spells_cast_on_a_unit():
    rules = om.pointer_rules(VOCAB)
    assert len(rules) == len(VOCAB["orders"]) + 1 and rules[0] == 0
    for c in (NOVA, BOLT, LIGHT, HEAL):
        assert rules[c], c
    # (an area spell aimed at a unit's place, a point order, the generic orders, harvest, the rest: none)
    for c in (IMPALE, BLIZZARD, ATTACK, HARVEST, LEARN_NOVA, IMMO, UNIMMO, SHIELD, UNSHIELD, ROBO, UNROBO):
        assert rules[c] == 0, c
    kinds = om.type_kinds(VOCAB)
    assert kinds[T["ocat"]] & om.MECH and not kinds[T["ogru"]] and kinds[0] == 0


def _net(omask=True, **kw):
    from warcraftsim.fullgame.model import FullGameNet
    torch.manual_seed(0)
    net = FullGameNet(n_types=len(TYPES) + 1, n_cur=2, n_orders=len(VOCAB["orders"]) + 1, G=40, d=32, layers=1, **kw)
    net.allowed[:] = True
    if omask:
        net.set_order_masks(VOCAB)
    return net.eval()


# the entities: the lich (own hero), an own ghoul, an own ziggurat, an enemy grunt, a peon in a mine, a catapult (a
# machine), a gryphon rider (flying), a creep
ENTS = (("Ulic", 0, fx.HERO), ("ugho", 0, 0), ("uzig", 0, fx.STRUCTURE), ("ogru", 1, 0), ("opeo", 1, fx.WORKER | 16),
        ("ocat", 1, 0), ("hgry", 1, 2048), ("nfrl", 2, 0))


def _ents(ents=ENTS):
    ent = torch.zeros(1, len(ents), fx.F)
    for i, (_, side, fl) in enumerate(ents):
        ent[0, i, side] = 1
        for k, b in enumerate(fx.FLAG_BITS):
            ent[0, i, 9 + k] = float(fl & b > 0)
    typ = torch.tensor([[T[t] for t, _, _ in ents]])
    return ent, typ, torch.ones(1, len(ents), dtype=torch.bool)


def test_pointer_takes_what_the_spell_takes():
    net = _net()
    ent, typ, mask = _ents()
    E = ent.shape[1]
    order = torch.tensor([[NOVA, BOLT, ATTACK]])
    ok = net.pointer_ok(ent, typ, mask, order)[0]
    # frost nova (the lich): the grunt, the peon in the mine (the game takes it: scripts/order_masks_probe.py), the
    # gryphon rider, the creep; not itself, its own, a building, a machine
    assert ok[0].tolist() == [False, False, False, True, True, False, True, True]
    assert ok[1].tolist() == [False, False, False, True, True, True, True, True]  # (storm bolt: machines too)
    assert ok[2].all()  # (attack: no rule)
    u = torch.randn(1, E, 32)
    ptr, _, _ = net.target_logits(torch.zeros(1, 32), u, mask, order, ent, typ)
    assert ((ptr[0] > -1e8) == ok).all()
    with pytest.raises(ValueError):  # (a network with the rule needs the types: every path must give the same mask)
        net.target_logits(torch.zeros(1, 32), u, mask, order, ent)


def test_holy_light_never_on_the_paladin_itself():
    net = _net()
    ents = (("Hpal", 0, fx.HERO), ("Hpal", 0, fx.HERO), ("uzig", 1, fx.STRUCTURE))
    ent, typ, mask = _ents(ents)
    ok = net.pointer_ok(ent, typ, mask, torch.tensor([[LIGHT, LIGHT, 0]]))[0]
    assert ok[0].tolist() == [False, True, False] and ok[1].tolist() == [True, False, False]
    # the order: each paladin has the other to take it; alone, none (a building is no target)
    lg = net.order_logits(torch.zeros(1, 32), torch.zeros(1, 3, 32), typ, torch.tensor([2]), ent=ent, emask=mask)
    assert (lg[0, :2, LIGHT] > -1e8).all()
    lone = (ents[0], ents[2])
    ent, typ, mask = _ents(lone)
    lg = net.order_logits(torch.zeros(1, 32), torch.zeros(1, 2, 32), typ, torch.tensor([1]), ent=ent, emask=mask)
    assert lg[0, 0, LIGHT] < -1e8 and lg[0, 0, ATTACK] > -1e8


def test_the_order_only_with_a_target_in_view():
    net = _net()
    mine = ENTS[:3] + (("ocat", 1, 0),)  # (the lich, its own, and an enemy catapult: a machine)
    ent, typ, mask = _ents(mine)
    lg = net.order_logits(torch.zeros(1, 32), torch.zeros(1, 4, 32), typ, torch.tensor([3]), ent=ent, emask=mask)
    assert lg[0, 0, NOVA] < -1e8 and lg[0, 0, BOLT] > -1e8 and lg[0, 0, HEAL] > -1e8
    # (and a unit given it all the same, a label: its pointer as without the rule, no -1e9 everywhere)
    ok = net.pointer_ok(ent, typ, mask, torch.tensor([[NOVA, 0, 0]]))
    assert ok[0, 0].all()
    ent, typ, mask = _ents(mine + (ENTS[3],))  # (a grunt in sight)
    lg = net.order_logits(torch.zeros(1, 32), torch.zeros(1, 5, 32), typ, torch.tensor([3]), ent=ent, emask=mask)
    assert lg[0, 0, NOVA] > -1e8 and lg[0, 0, BOLT] > -1e8
    assert (net.order_logits(torch.zeros(1, 32), torch.zeros(1, 5, 32), typ, torch.tensor([3]), ent=ent)
            == lg).all()  # (no emask, no order rule: bc's path, as before)


def test_flag_off_changes_nothing(tmp_path):
    """Without the flag a network has no rule, no buffers, no config key; with rules that refuse nothing it gives the
    same outputs bit for bit; a network loaded with the flag off is the checkpoint's (and with it on, the actors' path,
    its config and state dict, gives the learner's outputs)."""
    from warcraftsim.fullgame import model
    off = _net(omask=False)
    assert "omask" not in off.config and not off.omask
    assert not {"ptr_rule", "type_kind", "ptr_cols"} & {n for n, _ in off.named_buffers()}
    on = _net(omask=True)
    on.load_state_dict({**off.state_dict(), **{k: on.state_dict()[k] for k in ("ptr_rule", "type_kind", "ptr_cols")}})
    on.ptr_rule.zero_()  # (no rule: as without)
    g = torch.Generator().manual_seed(1)
    ent, typ, mask = _ents()
    ent = ent + 0.01 * torch.rand(ent.shape, generator=g) * (ent == 0)  # (the other channels: any values)
    E = ent.shape[1]
    glob, cur = torch.rand(1, 40, generator=g), torch.zeros(1, E, dtype=torch.long)
    n_own, avail = torch.tensor([3]), torch.ones(1, len(VOCAB["orders"]) + 1, dtype=torch.bool)
    for gen_seed in range(3):
        a0 = model.act(off, ent, typ, cur, mask, glob, n_own, avail, generator=torch.Generator().manual_seed(gen_seed))
        a1 = model.act(on, ent, typ, cur, mask, glob, n_own, avail, generator=torch.Generator().manual_seed(gen_seed))
        for k in a0:
            assert torch.equal(a0[k], a1[k]), k
    order = torch.tensor([[NOVA, BOLT, LIGHT]])
    tgt = torch.tensor([[3, 0, 1]])
    e0 = model.evaluate(off, ent, typ, cur, mask, glob, n_own, order, tgt, tgt, tgt, avail=avail)
    e1 = model.evaluate(on, ent, typ, cur, mask, glob, n_own, order, tgt, tgt, tgt, avail=avail)
    for k in ("logits", "logp", "ptr", "xl", "yl", "value", "entropy"):
        assert torch.equal(e0[k], e1[k]), k
    w = torch.ones(1, 3)
    assert model.pointer_label_ok(off, e0["ptr"], tgt, w) is w
    # load(): off, the checkpoint's own network; on, the rules (and the actors' copy of it alike)
    path = tmp_path / "ck.pt"
    torch.save({"model": off.state_dict(), "config": off.config, "vocab": VOCAB}, path)
    fx.set_order_masks(0)
    net0, _ = model.load(path)
    assert "omask" not in net0.config and set(net0.state_dict()) == set(off.state_dict())
    fx.set_order_masks(1)
    net1, _ = model.load(path)
    assert net1.config["omask"] == int((net1.ptr_rule != 0).sum()) == 4
    actor = model.FullGameNet(**net1.config)
    actor.load_state_dict(net1.state_dict())
    actor.eval()
    assert [n for n, _ in actor.named_buffers()] == [n for n, _ in net1.named_buffers()]  # (a slot copies by position)
    e1 = model.evaluate(net1, ent, typ, cur, mask, glob, n_own, order, tgt, tgt, tgt, avail=avail)
    e2 = model.evaluate(actor, ent, typ, cur, mask, glob, n_own, order, tgt, tgt, tgt, avail=avail)
    for k in ("logits", "logp", "ptr", "value"):
        assert torch.equal(e1[k], e2[k]), k
    # a checkpoint of a run with the flag, loaded with it off: its own rules (it was trained with them)
    torch.save({"model": net1.state_dict(), "config": net1.config, "vocab": VOCAB}, path)
    fx.set_order_masks(0)
    net3, _ = model.load(path)
    assert net3.config.get("omask") == 4 and torch.equal(net3.ptr_rule, net1.ptr_rule)


def test_labels_on_refused_targets_get_no_pointer_loss():
    from warcraftsim.fullgame import model
    net = _net()
    ent, typ, mask = _ents()
    order = torch.tensor([[NOVA, NOVA, ATTACK]])
    ptr, _, _ = net.target_logits(torch.zeros(1, 32), torch.randn(1, ent.shape[1], 32), mask, order, ent, typ)
    y = torch.tensor([[3, 1, 1]])  # (frost nova on the grunt; on its own ghoul; attack on its own ghoul)
    assert model.pointer_label_ok(net, ptr, y, torch.ones(1, 3)).tolist() == [[1.0, 0.0, 1.0]]


def _view(rows, events, t, vocab=VOCAB):
    enc = fx.Encoder(vocab, np.zeros((len(vocab["orders"]) + 1, 3), np.int64))
    view = enc.view(0, 1.0, [3, 1])
    me = np.asarray([0, 0, 500, 500, 5, 30, 0])
    return view, view.step(rows, me, events, t)


def _row(i, typ, owner=0, flags=0):
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_FLAGS], r[fx.C_VIS] = i, code(typ), owner, flags, 3
    r[fx.C_HP] = r[fx.C_MAXHP] = 400
    return r


def _events(*evs):
    return np.asarray([(0, k, a, b, 0) for k, a, b in evs], np.int64).reshape(-1, 5)


def test_toggles_against_their_state():
    dh = _row(7, "Edem", flags=fx.HERO)
    rows = np.stack([dh, _row(8, "ogru", owner=1)])
    fx.set_order_masks(0)
    view, st = _view(rows, _events((101, 7, 0)), 0)
    assert st["avail"][[IMMO, UNIMMO]].tolist() == [True, True]  # (off: as before)
    fx.set_order_masks(2)
    view, st = _view(rows, _events(), 0)
    assert st["avail"][[IMMO, UNIMMO]].tolist() == [True, True]  # (its state unknown: either)
    st = view.step(rows, None, _events((101, 7, 0), (101, 8, 0)), 1)  # (its state: no buffs)
    assert st["avail"][[IMMO, UNIMMO]].tolist() == [True, False]
    st = view.step(rows, None, _events((101, 7, 0), (102, 7, code("BEim"))), 2)  # (immolation on)
    assert st["avail"][[IMMO, UNIMMO]].tolist() == [False, True]
    st = view.step(rows, None, _events(), 3)  # (no change: as it was)
    assert st["avail"][[IMMO, UNIMMO]].tolist() == [False, True]
    st = view.step(rows[1:], None, _events(), 4)  # (no demon hunter: neither)
    assert st["avail"][[IMMO, UNIMMO]].tolist() == [False, False]
    assert view.toggles.masked >= 5
    # (the sea witch's mana shield, beside the demon hunter's immolation: each by its own buff)
    naga = _row(9, "Nngs", flags=fx.HERO)
    view, st = _view(np.stack([dh, naga]), _events((101, 7, 0), (102, 7, code("BEim")), (101, 9, 0),
                                                   (102, 9, code("BNms")), (102, 9, code("BHab"))), 0)
    assert st["avail"][[IMMO, UNIMMO, SHIELD, UNSHIELD]].tolist() == [False, True, False, True]
    st = view.step(np.stack([dh, naga]), None, _events((101, 9, 0), (102, 9, code("BHab"))), 1)
    assert st["avail"][[IMMO, UNIMMO, SHIELD, UNSHIELD]].tolist() == [False, True, True, False]


def test_forms_and_learns(tmp_path):
    from warcraftsim.fullgame import herodata, model
    assert sorted(om.form_refused(VOCAB)) == [(T["Ntin"], UNROBO), (T["Nrob"], ROBO)]
    fx.set_inputs(2)
    fx.set_masks(True, True)
    fx.set_order_masks(0)
    needs0 = herodata.order_needs(VOCAB)
    fx.set_order_masks(4)
    needs4 = herodata.order_needs(VOCAB)
    assert needs0[(T["Ulic"], LEARN_NOVA)] == needs4[(T["Ulic"], LEARN_NOVA)] == 8  # (the lich: frost nova's slot)
    assert (0, LEARN_NOVA) not in needs0 and needs4[(0, LEARN_NOVA)] == 8  # (an unknown type: a point it never has)
    assert (T["ugho"], LEARN_NOVA) not in needs0 and needs4[(T["ugho"], LEARN_NOVA)] == 8
    assert (T["Hpal"], LEARN_NOVA) not in needs4  # (another hero: the allowed table's)
    assert all(needs4[k] == v for k, v in needs0.items())  # (the rest as before; only learns added)
    assert all(c == LEARN_NOVA for _, c in set(needs4) - set(needs0))
    # in the network: a unit of an unknown type (a hexed hero, a clockwerk goblin) the allowed table lets learn
    net = model.FullGameNet(n_types=len(TYPES) + 1, n_cur=2, n_orders=len(VOCAB["orders"]) + 1, G=40, d=32, layers=1)
    net.allowed[:] = True
    net.set_valid(VOCAB)
    ent = torch.zeros(1, 2, fx.F)
    ent[0, :, fx.A_HERO + 3] = 1  # (a point it may spend, as a hero's slot would say)
    typ = torch.tensor([[0, T["Ulic"]]])
    lg = net.order_logits(torch.zeros(1, 32), torch.zeros(1, 2, 32), typ, torch.tensor([2]), ent=ent)
    assert (lg[0, :, LEARN_NOVA] > -1e8).tolist() == [True, True]
    ent[0, 0, fx.A_HERO + 3] = 0  # (an unknown type's: no hero slots read, the bit 0)
    fx.set_order_masks(4)
    net.set_valid(VOCAB)
    lg = net.order_logits(torch.zeros(1, 32), torch.zeros(1, 2, 32), typ, torch.tensor([2]), ent=ent)
    assert (lg[0, :, LEARN_NOVA] > -1e8).tolist() == [False, True]
    # load(): bit 2's forms out of the allowed table
    path = tmp_path / "ck.pt"
    torch.save({"model": net.state_dict(), "config": net.config, "vocab": VOCAB}, path)
    fx.set_order_masks(2)
    got, _ = model.load(path)
    assert not got.allowed[T["Ntin"], UNROBO] and not got.allowed[T["Nrob"], ROBO]
    assert got.allowed[T["Ntin"], ROBO] and got.allowed[T["Nrob"], UNROBO]
    fx.set_order_masks(0)
    got, _ = model.load(path)
    assert got.allowed[T["Ntin"], UNROBO] and got.allowed[T["Nrob"], ROBO]


CKPT = Path("/home/tora/warcraftsim/runs/fgself-225/averaged/swa-57-67.pt")


@pytest.mark.skipif(not CKPT.exists(), reason="fgself-225's averaged checkpoint")
def test_a_runs_network_with_the_masks_only_masks():
    """fgself-225's network under its run's flags, loaded with --order-masks 0 and 7: the same weights; with the masks
    every logit left open is the one without them, bit for bit, and only the ruled orders (bit 1), the learns (bit 4)
    and the ruled orders' pointers are masked more."""
    from warcraftsim.fullgame import model

    def load(bits):
        fx.set_inputs(2)
        fx.set_masks(True, True)
        fx.set_neutral(2)
        fx.set_macros(3)
        fx.set_spells(1)
        fx.set_shop_items(1)
        fx.set_full_vocab(1)
        fx.set_obs_full(1)
        fx.set_econ_masks(255)
        fx.set_order_masks(bits)
        return model.load(CKPT)
    try:
        net0, _ = load(0)
        net7, ck = load(7)
        state0, state7 = net0.state_dict(), net7.state_dict()
        assert set(state7) - set(state0) == {"ptr_rule", "type_kind", "ptr_cols"}
        assert all(torch.equal(v, state7[k]) for k, v in state0.items() if k not in ("valid_need", "allowed"))
        C, T_ = net0.config["n_orders"], net0.config["n_types"]
        g = torch.Generator().manual_seed(1)
        N, E, O = 6, 64, 24
        ent = torch.rand(N, E, fx.F, generator=g)
        ent[..., :3] = 0
        ent[:, :O, 0], ent[:, O:48, 1], ent[:, 48:, 2] = 1, 1, 1
        ent[..., 9:20] = (torch.rand(N, E, 11, generator=g) < 0.15).float()
        ent[..., fx.A_OBS:fx.A_OBS + 3] = 0
        typ = torch.randint(0, T_, (N, E), generator=g)
        cur = torch.randint(0, net0.config["n_cur"], (N, E), generator=g)
        mask = torch.ones(N, E, dtype=torch.bool)
        mask[0, 40:] = False
        glob = torch.rand(N, net0.config["G"], generator=g)
        n_own, avail = torch.full((N,), O), torch.rand(N, C, generator=g) < 0.8
        rules = net7.ptr_rule != 0
        ruled = net7.ptr_cols[torch.randint(0, net7.omask, (N, O), generator=g)]
        order = torch.where(torch.rand(N, O, generator=g) < 0.5, ruled, torch.randint(0, C, (N, O), generator=g))
        tgt = torch.randint(0, 30, (N, O), generator=g)
        bx, by = (torch.randint(0, net0.bins, (N, O), generator=g) for _ in range(2))
        h = torch.rand(N, net0.config["d"], generator=g)
        with torch.no_grad():
            e0, e7 = (model.evaluate(n, ent, typ, cur, mask, glob, n_own, order, tgt, bx, by, avail=avail,
                                     seq=(N, 1, h, None)) for n in (net0, net7))
        open7 = e7["logits"] > model.NEG / 2
        assert torch.equal(e7["logits"][open7], e0["logits"][open7])
        shut = (e0["logits"] > model.NEG / 2) & ~open7  # (masked by the flag alone)
        learns = torch.tensor([False] + [int(k) == fx.SKILL for _, k in ck["vocab"]["orders"]])
        forms = torch.zeros(C, dtype=torch.bool)  # (bit 2: the robo-goblin's morph, the tinker's way back)
        forms[[c for _, c in om.form_refused(ck["vocab"])]] = True
        assert shut.any() and (rules | learns | forms)[shut.nonzero()[:, 2]].all()
        p7 = e7["ptr"] > model.NEG / 2
        assert torch.equal(e7["ptr"][p7], e0["ptr"][p7])
        more = ((e0["ptr"] > model.NEG / 2) & ~p7).any(-1)  # (units whose pointer the flag masked more)
        assert more.any() and rules[order[more]].all()
        for k in ("xl", "yl", "value"):
            assert torch.equal(e0[k], e7[k]), k
    finally:
        fx.set_obs_full(0)
        fx.set_inputs(1)
        fx.set_masks(False, False)
        fx.set_neutral(0)
        fx.set_macros(0)
        fx.set_spells(0)
        fx.set_shop_items(0)
        fx.set_full_vocab(0)
        fx.set_econ_masks(0)
