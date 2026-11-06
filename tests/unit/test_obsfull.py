"""--obs-full (fullgame/obsfull.py): the features as before with the flag off; the block's channels and the new entity
kinds from the harness's events 101-109 and the research, production and ground-item records; a network loaded with it
plays exactly as it did (fgself-182's checkpoint: the learner's evaluate and the actors' FullGameNet(**config) + state
dict) until its new weights move."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from warcraftsim.fullgame import features as fx  # noqa: E402

CKPT = Path("/home/tora/warcraftsim/runs/fgself-182/checkpoints/0000000050150172.pt")


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


VOCAB = {"types": [code(c) for c in ("hfoo", "hbar", "Hamg", "ogru", "obar", "ugho")], "current_orders": [],
         "upgrades": [], "orders": [[851983, fx.UNIT], [851986, fx.POINT], [851972, fx.IMMEDIATE]],
         "order_names": {"851983": "attack", "851986": "move", "851972": "stop"}}


@pytest.fixture()
def obs_on():
    try:
        from warcraftsim.fullgame import obsfull
        obsfull.tables()
    except Exception as e:  # noqa: BLE001 (the game's tables: not on every machine)
        pytest.skip(f"no game data: {e}")
    fx.set_inputs(2)
    fx.set_obs_full(1)
    yield
    fx.set_obs_full(0)
    fx.set_inputs(1)


def test_flag_off_layout_unchanged():
    fx.set_obs_full(0)
    for inputs in (1, 2, 3):
        fx.set_inputs(inputs)
        before = fx.F
        assert fx.A_OBS == -1
        fx.set_obs_full(1)
        assert fx.A_OBS == before and fx.F == before + fx.OBS_F  # (the block last: every old channel where it was)
        fx.set_obs_full(0)
        assert fx.F == before and fx.A_OBS == -1
    fx.set_inputs(1)
    from warcraftsim.fullgame import obsfull
    assert obsfull.OBS_F == fx.OBS_F


def _row(uid, typ, owner, x, y, flags=0, vis=3, hero_items=()):
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = uid, code(typ), owner, x, y
    r[fx.C_HP] = r[fx.C_MAXHP] = 400
    r[fx.C_FLAGS], r[fx.C_VIS] = flags, vis
    for i, it in enumerate(hero_items):
        r[fx.C_ITEM0 + i] = code(it)
    return r


def test_block_and_new_kinds(obs_on):
    from warcraftsim.fullgame import obsfull as of
    enc = fx.Encoder(VOCAB)
    view = enc.view(0, 1.0, [0, 1])
    rows = np.stack([_row(1, "hfoo", 0, -4000, 2900), _row(2, "hbar", 0, -4200, 2600, flags=fx.STRUCTURE),
                     _row(3, "Hamg", 0, -4100, 2900, flags=fx.HERO, hero_items=("phea", "pman")),
                     _row(4, "ogru", 1, -3800, 2900), _row(5, "obar", 1, 3000, 2900, flags=fx.STRUCTURE, vis=2)])
    speed_armor = (188 << 12) | (130 + 2048)  # (a defending footman with inner fire: speed 188, armor 13)
    ev = [(0, of.EV_STATE, 1, 2 | (1 << 8) | (2 << 16) | (1 << 20), speed_armor),
          (0, of.EV_BUFF, 1, code("Binf"), 0), (0, of.EV_BUFF, 1, code("Beng"), 1),
          (0, of.EV_STATE, 3, 0, (320 << 12) | 2093), (0, of.EV_CHARGES, 3, 1 | (2 << 10), 0),
          (0, of.EV_STATE, 4, 8, (125 << 12) | 2118),  # (a polymorphed grunt)
          (0, 1, 6, 4, code("ugho")), (0, of.EV_CORPSE, 6, -3900, 2950), (0, of.EV_CORPSE_SEEN, 6, 1, 0),
          (0, of.EV_SPELL_AT, 3, -3850, 2900), (0, of.EV_SPELL_SEEN, 3, code("AHbz"), 1),
          (0, fx.RESEARCH_FINISH, 5, code("Rome"), 2),  # (the enemy's weapons: what its grunt's tooltip shows)
          (0, 5, 2, code("hfoo"), 0)]  # (the barracks trains a footman)
    view.ground = [(code("tdex"), -3950, 2800, 1, 1), (code("ratc"), 3000, 2800, 2, 0)]  # (the second: player 1's)
    st = view.step(rows, None, np.asarray(ev, np.int64), 4)
    A, f, sel = fx.A_OBS, st["ent"], st["sel"]
    at = {int(u): k for k, u in enumerate(sel[:, fx.C_ID].tolist())}
    foot, mage, grunt, bar = f[at[1]], f[at[3]], f[at[4]], f[at[2]]
    assert foot[A + of.K_CLASS + 1] == 1 and foot[A + of.K_CLASS] == 0  # snared
    assert foot[A + of.K_MAGIC] == pytest.approx(1 / 3) and foot[A + of.K_HAVE] == 1
    assert foot[A + of.K_ARMOR] == pytest.approx(1.3) and foot[A + of.K_SPEED] == pytest.approx(188 / 400)
    assert foot[A + of.K_SPEED_R] == pytest.approx(188 / 270)
    want = of.code(code("Binf"), of.BUFF_D) + of.code(code("Beng"), of.BUFF_D)
    assert np.allclose(foot[A + of.K_BUFF:A + of.K_BUFF + of.BUFF_D], want)
    assert grunt[A + of.K_CLASS + 3] == 1  # polymorphed (its type and hit points as before)
    assert grunt[A + of.K_UPG] == pytest.approx(2 / 3)  # the enemy's Rome 2 on its grunt
    assert foot[A + of.K_UPG] == 0  # (the player's own: none researched)
    assert np.allclose(mage[A + of.K_CHARGE:A + of.K_CHARGE + 3], [0.2, 0.4, 0.0])
    items = of.code(code("phea"), of.ITEM_D) + of.code(code("pman"), of.ITEM_D)
    assert np.allclose(mage[A + of.K_ITEM:A + of.K_ITEM + of.ITEM_D], items)
    assert bar[A + of.K_PROD] == 0.0 and bar[A + of.K_PROD + 1] > 0  # (it just started: its seconds left)
    assert 5 not in at  # (the enemy's barracks out of sight)
    kinds = f[:, A:A + 3]
    assert kinds[:, 0].sum() == 1 and kinds[:, 1].sum() == 1 and kinds[:, 2].sum() == 1  # a corpse, an item, an area
    corpse = int(np.flatnonzero(kinds[:, 0])[0])
    assert sel[corpse, fx.C_TYPE] == code("ugho") and st["type"][corpse] == VOCAB["types"].index(code("ugho")) + 1
    area = f[int(np.flatnonzero(kinds[:, 2])[0])]
    assert area[A + of.K_RADIUS] == pytest.approx(0.2) and area[A + of.K_AGE] > 0
    assert st["n"] == len(rows) - 1 + 3
    # the corpse is used: gone the next step; the area runs out
    st = view.step(rows, None, np.asarray([(5, of.EV_CORPSE_GONE, 6, 0, 0)], np.int64), 5)
    assert st["ent"][:, A].sum() == 0
    st = view.step(rows, None, np.zeros((0, 5), np.int64), 400)
    assert st["ent"][:, A + 2].sum() == 0


def test_creep_types_grow_as_unknown(obs_on):
    from warcraftsim.fullgame import obsfull as of
    from warcraftsim.fullgame.model import FullGameNet
    new = of.extend(VOCAB)
    assert of.extended(new) and new["types"][:len(VOCAB["types"])] == VOCAB["types"]
    added = [fx.rawcode(t) for t in new["types"][len(VOCAB["types"]):]]
    assert {"nmrl", "nmrr", "nmrm", "nomg", "nkot", "ntrg", "nwzg"} <= set(added)  # (Echo Isles' unknown creeps)
    net = FullGameNet(len(VOCAB["types"]) + 1, 3, 4, 20)
    grown = of.grow(net, VOCAB, new)
    te = grown.type_emb.weight
    assert torch.equal(te[:len(VOCAB["types"]) + 1], net.type_emb.weight)
    assert all(torch.equal(te[k], net.type_emb.weight[0]) for k in range(len(VOCAB["types"]) + 1, te.shape[0]))


def _load(obs_full: int):
    from warcraftsim.fullgame import model
    fx.set_inputs(2)
    fx.set_masks(True, True)
    fx.set_neutral(2)
    fx.set_macros(3)
    fx.set_spells(1)
    fx.set_shop_items(1)
    fx.set_obs_full(obs_full)
    net, ck = model.load(CKPT)
    return net.eval(), ck


@pytest.mark.skipif(not CKPT.exists(), reason="fgself-182's checkpoint")
def test_identity_fgself182():
    """A network loaded with --obs-full gives the same orders, targets and value as without (the learner's evaluate and
    the actors' network built from its config and state dict), whatever the new channels and kinds hold."""
    from warcraftsim.fullgame import model
    from warcraftsim.fullgame import obsfull as of
    from warcraftsim.fullgame.model import FullGameNet
    try:
        net0, ck0 = _load(0)
        F0 = fx.F
        net1, ck1 = _load(1)
        F1, A = fx.F, fx.A_OBS
        assert F1 == F0 + of.OBS_F and A == F0 and net1.config.get("obsfull") == 1 and not net0.config.get("obsfull")
        assert net1.kind_bias.tolist() == [-30.0] * 3
        T0 = len(ck0["vocab"]["types"]) + 1
        assert net1.config["n_types"] > T0  # (the creep types)
        g = torch.Generator().manual_seed(0)
        N, E0, extra = 6, 40, 5
        E1 = E0 + extra
        O = 24
        ent0 = torch.rand(N, E0, F0, generator=g)
        ent0[..., :3] = 0
        ent0[:, :O, 0] = 1  # (own units first, then the others)
        ent0[:, O:, 1] = 1
        ent1 = torch.zeros(N, E1, F1)
        ent1[:, :E0, :F0] = ent0
        ent1[:, :E0, A + 3:] = torch.rand(N, E0, F1 - A - 3, generator=g)  # (the block on the units: any values)
        ent1[:, E0:, :] = torch.rand(N, extra, F1, generator=g)
        ent1[:, E0:, A:A + 3] = 0
        for k in range(extra):  # (the new kinds after the units)
            ent1[:, E0 + k, A + k % 3] = 1.0
        typ0 = torch.randint(0, T0, (N, E0), generator=g)
        typ0[:, O:O + 3] = 0  # (creeps of a type the old vocabulary lacked: unknown)
        typ1 = torch.cat([typ0, torch.zeros(N, extra, dtype=torch.long)], 1)
        typ1[:, O:O + 3] = T0  # (now their own type, embedded as unknown)
        cur0 = torch.randint(0, net0.config["n_cur"], (N, E0), generator=g)
        cur1 = torch.cat([cur0, torch.zeros(N, extra, dtype=torch.long)], 1)
        mask0 = torch.ones(N, E0, dtype=torch.bool)
        mask0[0, 30:] = False  # (padding)
        mask1 = torch.cat([mask0, torch.ones(N, extra, dtype=torch.bool)], 1)
        mask1[0, 30:E0] = False
        glob = torch.rand(N, net0.config["G"], generator=g)
        n_own = torch.full((N,), O)
        order = torch.randint(0, net0.config["n_orders"], (N, O), generator=g)
        tgt = torch.randint(0, 30, (N, O), generator=g)
        bx = torch.randint(0, net0.bins, (N, O), generator=g)
        by = torch.randint(0, net0.bins, (N, O), generator=g)
        seq = (N, 1, None, None) if net0.memory else None
        with torch.no_grad():
            ev0 = model.evaluate(net0, ent0, typ0, cur0, mask0, glob, n_own, order, tgt, bx, by, seq=seq)
            ev1 = model.evaluate(net1, ent1, typ1, cur1, mask1, glob, n_own, order, tgt, bx, by, seq=seq)
            actor = FullGameNet(**net1.config)  # (the actors' path: the config and a state dict, nothing else)
            actor.load_state_dict(net1.state_dict())
            actor.eval()
            ev2 = model.evaluate(actor, ent1, typ1, cur1, mask1, glob, n_own, order, tgt, bx, by, seq=seq)
        for k in ("logits", "logp", "value", "xl", "yl"):
            assert torch.allclose(ev0[k], ev1[k], atol=2e-5, rtol=1e-5), (k, (ev0[k] - ev1[k]).abs().max())
            assert torch.equal(ev1[k], ev2[k]), k
        assert torch.allclose(ev0["ptr"], ev1["ptr"][..., :E0], atol=2e-5, rtol=1e-5)
        assert (ev1["ptr"][..., E0:] <= model.NEG / 2).all()  # (the new kinds: no target)
        # the attention bias is what hides them: raised, the same inputs give other outputs
        net1.kind_bias.fill_(0.0)
        with torch.no_grad():
            ev3 = model.evaluate(net1, ent1, typ1, cur1, mask1, glob, n_own, order, tgt, bx, by, seq=seq)
        assert not torch.allclose(ev3["value"], ev1["value"], atol=1e-4)
    finally:
        fx.set_obs_full(0)
        fx.set_inputs(1)
        fx.set_masks(False, False)
        fx.set_neutral(0)
        fx.set_macros(0)
        fx.set_spells(0)
        fx.set_shop_items(0)
