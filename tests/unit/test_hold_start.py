"""A fine-tune held near its start: bc.py --hold-start (the act/no-act decision's KL from the start network),
--hold-all (the forward KL over every head) and --train-heads (the trunk frozen)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import bc
from test_acting_only import _batch, _net

CPU = torch.device("cpu")


def test_act_kl_is_the_bernoulli_kl_of_none():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(4, 3, 12, generator=g), torch.randn(4, 3, 12, generator=g)
    q, p = a.softmax(-1)[..., 0], b.softmax(-1)[..., 0]
    want = q * (q / p).log() + (1 - q) * ((1 - q) / (1 - p)).log()
    assert torch.allclose(bc.act_kl(a, b), want, atol=1e-5)
    assert float(bc.act_kl(a, a).abs().max()) < 1e-6
    shuffled = torch.cat([a[..., :1], a[..., 1:].flip(-1)], -1)  # (the orders given one reordered: the same rate)
    assert float(bc.act_kl(a, shuffled).abs().max()) < 1e-5


@pytest.mark.parametrize("memory", [False, True])
def test_hold_terms_are_zero_at_the_start(memory):
    net = _net()
    if memory:
        from warcraftsim.fullgame.model import FullGameNet
        torch.manual_seed(0)
        net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=32, layers=1, memory=True)
        net.allowed[:] = True
        net.order_kind.copy_(_net().order_kind)
    ref = copy.deepcopy(net).eval()
    y = np.random.default_rng(1).integers(0, 12, (5, 6))
    net.train()  # (no dropout in these nets: train mode computes what eval does)
    loss, st = bc.losses(net, _batch(y), CPU, None, value_coef=0.0, acting_only=True, ref=ref, hold_start=1.0,
                         hold_all=1.0)
    assert abs(st["kl_act"]) < 1e-6 and abs(st["kl_all"]) < 1e-5
    base, _ = bc.losses(copy.deepcopy(net), _batch(y), CPU, None, value_coef=0.0, acting_only=True)
    assert torch.allclose(loss, base, atol=1e-5)


def _fine_tune(hold_start: float = 0.0, hold_all: float = 0.0, heads: bool = False, steps: int = 40):
    """An acting-only fine-tune of a small net on labels it does not give (most units: none) -> (start, after) of the
    rate of giving orders as played, the order accuracy given one before and after, and the net."""
    net = _net()
    ref = copy.deepcopy(net).eval()
    rng = np.random.default_rng(2)
    y = rng.integers(10, 14, (16, 6))
    y[rng.random((16, 6)) < 0.9] = 0  # (people: an order in a tenth of the steps)
    b = _batch(y, seed=3)
    if heads:
        bc.train_heads(net)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-2)

    def stats():  # (the KL from the start over every head, whatever the fine-tune held)
        with torch.no_grad():
            return bc.summarize([bc.losses(net.eval(), b, CPU, None, value_coef=0.0, acting_only=True, ref=ref,
                                           hold_all=1.0)[1]])
    s0 = stats()
    for _ in range(steps):
        net.train()
        loss, _ = bc.losses(net, b, CPU, None, value_coef=0.0, acting_only=True, ref=ref, hold_start=hold_start,
                            hold_all=hold_all)
        opt.zero_grad()
        loss.backward()
        opt.step()
    s1 = stats()
    return s0, s1, net


def test_hold_start_keeps_the_rate_while_the_orders_are_learned():
    s0, free, _ = _fine_tune()
    _, held, _ = _fine_tune(hold_start=10.0)
    drift_free = abs(free["order_rate_played"] - s0["order_rate_played"])
    drift_held = abs(held["order_rate_played"] - s0["order_rate_played"])
    assert drift_held < 0.02 and drift_held < drift_free / 3, (s0["order_rate_played"], free, held)
    assert held["order_acc"] > s0["order_acc"] + 0.3  # (what to order learned all the same)


def test_hold_all_keeps_the_heads_near_the_start():
    s0, free, _ = _fine_tune(hold_start=10.0)
    _, held, _ = _fine_tune(hold_start=10.0, hold_all=1.0)
    assert s0["kl_all"] < 1e-5 and held["kl_all"] < free["kl_all"] / 2, (free["kl_all"], held["kl_all"])


def test_train_heads_freezes_the_trunk():
    net0 = _net()
    _, _, net = _fine_tune(heads=True, steps=5)
    before, after = net0.state_dict(), net.state_dict()
    for name, p in net.named_parameters():
        moved = not torch.equal(before[name], after[name])
        if name.startswith(bc.HEADS):
            assert p.requires_grad
        else:
            assert not p.requires_grad and not moved, name
    assert not torch.equal(before["order.2.weight"], after["order.2.weight"])
    assert any(n.startswith("transformer.") for n, _ in net.named_parameters())


def test_hold_all_is_the_forward_all_anchor_kl():
    """--hold-all's KL (the targets' terms over the units that use them only) equals model.anchor_kl's forward-all
    over every own unit."""
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame.model import anchor_kl
    net, ref = _net(0).eval(), _net(1).eval()
    y = np.random.default_rng(4).integers(0, 12, (5, 6))
    b = _batch(y, seed=5)
    with torch.no_grad():
        st = bc.losses(net, b, CPU, None, value_coef=0.0, ref=ref, hold_all=1.0)[1]
        O = min(fx.MAX_OWN, y.shape[1])
        args = (b["ent"], b["type"], b["cur"], b["mask"], b["glob"])
        order, y_x = b["y_order"][:, :O].long(), b["y_x"][:, :O].long()
        own = torch.arange(O)[None] < b["n_own"][:, None]
        ev = {}
        for k, m in (("net", net), ("ref", ref)):
            g, u = m.encode(*args)
            logits = m.order_logits(g, u[:, :O], b["type"], b["n_own"], by_type=False, avail=b["avail"])
            ptr, xl, z = m.target_logits(g, u, b["mask"], order.clamp(min=0))
            ev[k] = {"logits": bc.played(net, logits, b["type"][:, :O]), "ptr": ptr, "xl": xl,
                     "yl": m.y_logits(z, y_x)}
        ev["net"]["uses_ptr"] = ((b["y_ptr"][:, :O] >= 0) & own).float()
        ev["net"]["uses_pt"] = ((y_x >= 0) & own).float()
        want = (anchor_kl(ev["net"], ev["ref"], "forward-all") * own).sum() / own.sum()
    assert st["kl_all"] > 0.01 and abs(st["kl_all"] - float(want)) < 1e-4
