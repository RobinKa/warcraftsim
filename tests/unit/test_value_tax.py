"""--value-tax: the value (never the policy) reads each training game's tax, [the share of its income the opponent
loses, the share it loses itself], two numbers after the privileged block (features.View.tax, model tax_proj)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame import model as fm  # noqa: E402
from warcraftsim.fullgame.model import FullGameNet, act, evaluate, load  # noqa: E402
from warcraftsim.fullgame.selfplay import collate, load_optimizer, value_tax  # noqa: E402


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


VOCAB = {"types": [code("hpea"), code("hfoo")], "current_orders": [], "upgrades": [],
         "orders": [[851983, fx.UNIT], [code("hpea"), fx.IMMEDIATE], [code("hbar"), fx.POINT]]}


@pytest.fixture()
def off(monkeypatch):
    """No --value-tax at first, and none after; model.load without the game's tech tree (prune_allowed)."""
    monkeypatch.setattr(fm, "prune_allowed", lambda net, vocab: 0)
    fx.set_value_tax(0)
    yield
    fx.set_value_tax(0)


def _net(enc, seed: int = 0, **kw) -> FullGameNet:
    torch.manual_seed(seed)
    net = FullGameNet(n_types=enc.n_types, n_cur=enc.n_cur, n_orders=enc.n_orders, G=enc.G, d=16, layers=1,
                      priv=fx.PRIV + fx.VTAX, vtax=fx.VTAX, **kw).eval()
    net.allowed[:] = True
    net.order_kind.copy_(torch.as_tensor([fx.IMMEDIATE] + [k for _, k in VOCAB["orders"]]))
    with torch.no_grad():  # (trained: the zero-initialised parts too)
        for name, p in net.named_parameters():
            if name.startswith(("priv_proj.2", "priv_pol.2", "tax_proj")):
                p.normal_(0.0, 0.5)
    return net


def _inputs(G: int, n: int = 3, E: int = 6, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, E, fx.F, generator=g), torch.randint(1, 3, (n, E), generator=g),
            torch.zeros(n, E, dtype=torch.long), torch.ones(n, E, dtype=torch.bool), torch.randn(n, G, generator=g),
            torch.tensor([2, 1, 3]))


def test_the_taxes_as_each_side_sees_them():
    def taxed_side(opp: dict, side: int):  # (play_one's: the AI's side unless the tax names another)
        return opp.get("tax_side", 1 - side) if opp.get("tax") else None
    for side in (0, 1):  # against the built-in AI: its income taxed (the curriculum), or a bonus (< 0)
        for t in (0.3, -0.2):
            opp = {"kind": "ai", "tax": t}
            assert value_tax(opp, taxed_side(opp, side), side) == pytest.approx([t, 0.0])
    opp = {"kind": "self", "tax": 0.15, "tax_side": 1}  # between agents (--win-balance): the favoured side's income
    taxed = taxed_side(opp, 0)
    assert value_tax(opp, taxed, 0) == pytest.approx([0.15, 0.0])
    assert value_tax(opp, taxed, 1) == pytest.approx([0.0, 0.15])
    assert value_tax({"kind": "ai"}, taxed_side({"kind": "ai"}, 0), 0).tolist() == [0.0, 0.0]  # untaxed (--real-share)
    assert value_tax({"kind": "ai", "tax": 0.3}, 1, 0, train=False).tolist() == [0.0, 0.0]  # an evaluation game
    assert value_tax(opp, taxed, 1).dtype == np.float32


@pytest.mark.parametrize("privileged", [False, True])
def test_the_policy_never_reads_the_taxes(off, privileged):
    fx.set_value_tax(1)
    enc = fx.Encoder(VOCAB)
    net = _net(enc, privileged=privileged)
    ent, typ, cur, mask, glob, n_own = _inputs(enc.G)
    glob[:, enc.G - fx.PRIV - fx.VTAX - 1] = 1.0  # (the privileged path's flag on: it reads the privileged block)
    glob[:, -2:] = 0.0
    taxed = glob.clone()
    taxed[:, -2:] = torch.tensor([[0.3, 0.0], [0.0, 0.15], [-0.2, 0.0]])
    a0 = act(net, ent, typ, cur, mask, glob, n_own, generator=torch.Generator().manual_seed(5))
    a1 = act(net, ent, typ, cur, mask, taxed, n_own, generator=torch.Generator().manual_seed(5))
    for k in ("order", "tgt", "bx", "by", "logp", "entropy"):
        assert torch.equal(a0[k], a1[k]), k
    assert not torch.allclose(a0["value"], a1["value"])
    taxed.requires_grad_(True)
    ev = evaluate(net, ent, typ, cur, mask, taxed, n_own, a1["order"], a1["tgt"], a1["bx"], a1["by"])
    ev0 = evaluate(net, ent, typ, cur, mask, glob, n_own, a1["order"], a1["tgt"], a1["bx"], a1["by"])
    for k in ("logits", "logp", "ptr", "xl", "yl", "entropy"):
        assert torch.equal(ev[k], ev0[k]), k
    pol = torch.autograd.grad(ev["logp"].sum() + ev["logits"].sum() + ev["xl"].sum(), taxed, retain_graph=True)[0]
    assert float(pol[:, -2:].abs().max()) == 0.0 and float(pol[:, :-2].abs().max()) > 0.0
    val = torch.autograd.grad(ev["value"].sum(), taxed)[0]
    assert float(val[:, -2:].abs().max()) > 0.0  # (the value does read them)


def test_an_old_checkpoint_plays_and_values_as_before(off, tmp_path):
    enc0 = fx.Encoder(VOCAB)
    old = _net(enc0)
    assert not hasattr(old, "tax_proj")
    torch.save({"config": old.config, "model": old.state_dict(), "vocab": VOCAB}, tmp_path / "old.pt")
    fx.set_value_tax(1)
    net, _ = load(tmp_path / "old.pt")
    assert fx.Encoder(VOCAB).G == net.config["G"] == old.config["G"] + 2
    assert net.config["priv"] == fx.PRIV + 2 and net.config["vtax"] == 2
    assert not net.tax_proj.weight.any() and not net.tax_proj.bias.any()
    ent, typ, cur, mask, glob, n_own = _inputs(old.config["G"])
    wide = torch.cat([glob, torch.tensor([[0.3, 0.0], [0.0, 0.15], [-0.2, 0.0]])], 1)  # (the run's rows: the taxes)
    with torch.no_grad():
        a = act(old, ent, typ, cur, mask, glob, n_own, generator=torch.Generator().manual_seed(3))
        ev_old = evaluate(old, ent, typ, cur, mask, glob, n_own, a["order"], a["tgt"], a["bx"], a["by"])
        ev_new = evaluate(net, ent, typ, cur, mask, wide, n_own, a["order"], a["tgt"], a["bx"], a["by"])
    for k in ("logits", "logp", "ptr", "xl", "yl", "value"):  # (the same numbers, read from rows 2 wider: another
        assert float((ev_old[k] - ev_new[k]).abs().max()) <= 1e-5, k  # stride, so the float rounding may differ)
    # and back: a network trained with the taxes, in a run without them, plays as it did (its value without them)
    with torch.no_grad():
        net.tax_proj.weight.normal_()
    torch.save({"config": net.config, "model": net.state_dict(), "vocab": VOCAB}, tmp_path / "taxed.pt")
    fx.set_value_tax(0)
    back, _ = load(tmp_path / "taxed.pt")
    assert not hasattr(back, "tax_proj") and back.config["G"] == old.config["G"] and "vtax" not in back.config
    with torch.no_grad():
        ev_back = evaluate(back, ent, typ, cur, mask, glob, n_own, a["order"], a["tgt"], a["bx"], a["by"])
    for k in ("logits", "logp", "ptr", "xl", "yl", "value"):
        assert torch.equal(ev_old[k], ev_back[k]), k


def test_a_restart_and_adam_with_the_taxes_turned_on(off, tmp_path):
    enc0 = fx.Encoder(VOCAB)
    old = _net(enc0)
    old.train()
    opt = torch.optim.Adam(old.parameters(), lr=1e-3)
    ent, typ, cur, mask, glob, n_own = _inputs(old.config["G"])
    old.value(old.encode(ent, typ, cur, mask, glob)[0], glob).sum().backward()
    opt.step()
    torch.save({"config": old.config, "model": old.state_dict(), "vocab": VOCAB}, tmp_path / "c.pt")
    torch.save(opt.state_dict(), tmp_path / "c.adam")
    fx.set_value_tax(1)
    net, _ = load(tmp_path / "c.pt")
    net.train()
    net.load_state_dict(fm.trained(old.state_dict(), net))  # (a resume's: the checkpoint has no tax_proj)
    assert not net.tax_proj.weight.any()
    opt2 = torch.optim.Adam(net.parameters(), lr=1e-3)
    assert load_optimizer(opt2, tmp_path / "c.pt", torch.device("cpu"), 1e-3, net)
    assert torch.equal(opt2.state[net.priv_proj[0].weight]["exp_avg"], opt.state[old.priv_proj[0].weight]["exp_avg"])
    assert net.tax_proj.weight not in opt2.state  # (from zero)
    wide = torch.cat([glob, torch.full((glob.shape[0], 2), 0.25)], 1)
    net.value(net.encode(ent, typ, cur, mask, wide)[0], wide).sum().backward()
    opt2.step()
    assert net.tax_proj.weight.any()  # (it learns at once)
    # and a resume of a run with the taxes, without them: the checkpoint's tax_proj left out
    fx.set_value_tax(0)
    plain = _net(enc0)
    plain.load_state_dict(fm.trained(net.state_dict(), plain))


def test_the_view_carries_the_taxes_to_the_learner(off, monkeypatch):
    fx.set_value_tax(1)
    enc = fx.Encoder(VOCAB)
    G0 = enc.G0 + enc.ZD + enc.GX
    assert enc.G == G0 + fx.PRIV + 2
    assert len(fx.describe_spaces(VOCAB)["observation"]["blocks"][0]["features"]) == enc.G  # (every number named)
    rows = np.zeros((2, fx.ROW_COLS), np.int64)
    for i, (owner, x) in enumerate(((0, -2000), (1, 2000))):
        rows[i, fx.C_ID], rows[i, fx.C_TYPE], rows[i, fx.C_OWNER] = 1048576 + i, code("hfoo"), owner
        rows[i, fx.C_X], rows[i, fx.C_MAXHP], rows[i, fx.C_HP], rows[i, fx.C_VIS] = x, 420, 420, 3
    me = np.array([0, 0, 50, 10, 10, 20, 0, 0, 0, 0, 0])
    opp_row = np.array([0, 1, 450, 80, 12, 20, 0, 0, 0, 0, 0])
    plain = enc.view(0, 1.0, [0, 1]).step(rows, me, np.zeros((0, 5), np.int64), 0, opp_row)
    assert plain["glob"][-2:].tolist() == [0.0, 0.0]  # (no tax given: zeros)
    view = enc.view(0, 1.0, [0, 1])
    view.tax = value_tax({"kind": "ai", "tax": 0.3}, 1, 0)
    st = view.step(rows, me, np.zeros((0, 5), np.int64), 0, opp_row)
    assert st["glob"][-2:] == pytest.approx([0.3, 0.0])
    assert np.array_equal(st["glob"][:-2], plain["glob"][:-2])  # (the privileged block where it was)
    assert st["glob"][G0] == pytest.approx(0.45)  # (the opponent's gold / 1000: the block's first number)
    from warcraftsim.fullgame import featnative
    if featnative.lib() is not None:  # (the numpy path, without the C features: the same numbers)
        monkeypatch.setattr(featnative, "_lib", False)
        py = enc.view(0, 1.0, [0, 1])
        py.tax = view.tax
        assert np.allclose(py.step(rows, me, np.zeros((0, 5), np.int64), 0, opp_row)["glob"], st["glob"])
        monkeypatch.setattr(featnative, "_lib", None)
    # the value the actor acted with is the one the learner computes from the stored step
    net = _net(enc)
    t = lambda a, dt=None: torch.as_tensor(a)[None] if dt is None else torch.as_tensor(a)[None].to(dt)  # noqa: E731
    with torch.no_grad():
        a = act(net, t(st["ent"]), t(st["type"], torch.long), t(st["cur"], torch.long),
                torch.ones(1, st["n"], dtype=torch.bool), t(st["glob"]), torch.tensor([st["n_own"]]))
    o = st["n_own"]
    step = {**{k: st[k] for k in ("ent", "type", "cur", "glob", "n", "n_own")}, "avail": None, "adv": 0.0, "ret": 0.0,
            **{k: a[k][0, :o].numpy() for k in ("order", "tgt", "bx", "by", "logp")}}
    mb = collate([step], torch.device("cpu"))
    assert mb["glob"][0, -2:].tolist() == pytest.approx([0.3, 0.0])
    with torch.no_grad():
        ev = evaluate(net, mb["ent"], mb["type"], mb["cur"], mb["mask"], mb["glob"], mb["n_own"], mb["order"],
                      mb["tgt"], mb["bx"], mb["by"])
        untaxed = net.value(net.encode(mb["ent"], mb["type"], mb["cur"], mb["mask"], torch.as_tensor(plain["glob"])[None])[0],
                            torch.as_tensor(plain["glob"])[None])
    assert torch.allclose(ev["value"], a["value"], atol=1e-5)
    assert not torch.allclose(ev["value"], untaxed)
