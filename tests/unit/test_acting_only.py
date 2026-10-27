"""People's clicks without their "none": bc.py --acting-only (the order learned only given one) and selfplay's
--ref-kl-mode forward-acting (the anchor's pull over the orders given one, weighted by the policy's chance of acting)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import bc
from warcraftsim.fullgame import features as fx


def _net(seed: int = 0):
    from warcraftsim.fullgame.model import FullGameNet
    torch.manual_seed(seed)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=32, layers=1)
    net.allowed[:] = True
    net.order_kind.copy_(torch.randint(0, 5, (30,)))
    net.order_kind[0] = fx.IMMEDIATE
    net.order_kind[1:6] = fx.POINT
    net.order_kind[6:10] = fx.UNIT
    return net


def _batch(y: np.ndarray, seed: int = 0):
    """A cloning batch (bc.losses) with order labels `y` [B, E]; a point label only with an order at a point."""
    rng = np.random.default_rng(seed)
    B, E = y.shape
    kind = _net().order_kind.numpy()
    at_pt = (y > 0) & np.isin(kind[np.clip(y, 0, None)], [fx.POINT, fx.TREE])
    on_unit = (y > 0) & (kind[np.clip(y, 0, None)] == fx.UNIT)
    y_x = np.where(at_pt, rng.integers(0, 128, (B, E)), -1)
    y_y = np.where(at_pt, rng.integers(0, 128, (B, E)), -1)
    y_ptr = np.where(on_unit, rng.integers(0, E, (B, E)), -1)
    return {"ent": torch.from_numpy(rng.normal(size=(B, E, fx.F)).astype(np.float32)),
            "type": torch.from_numpy(rng.integers(0, 20, (B, E))), "cur": torch.from_numpy(rng.integers(0, 10, (B, E))),
            "mask": torch.ones(B, E, dtype=torch.bool), "glob": torch.from_numpy(rng.normal(size=(B, 30)).astype(np.float32)),
            "n_own": torch.full((B,), 4), "y_order": torch.from_numpy(y), "y_ptr": torch.from_numpy(y_ptr),
            "y_x": torch.from_numpy(y_x), "y_y": torch.from_numpy(y_y),
            "avail": torch.ones(B, 30, dtype=torch.bool), "ret": torch.zeros(B)}


@pytest.mark.parametrize("stats", [True, False])
def test_acting_only_loss_ignores_none_labels(stats):
    rng = np.random.default_rng(1)
    y = rng.integers(1, 12, (5, 6))
    y[rng.random((5, 6)) < 0.5] = 0  # (half the units: no order)
    y[0, 0] = 0
    dropped = np.where(y == 0, bc.IGNORE, y)  # (the same batch with the "none" labels left out)
    cpu = torch.device("cpu")

    def loss(labels, acting_only):
        net = _net().train()
        out, _ = bc.losses(net, _batch(labels), cpu, None, value_coef=0.0, stats=stats, acting_only=acting_only)
        out.backward()
        return out.detach(), net.order[2]

    l_none, head = loss(y, True)
    l_drop, _ = loss(dropped, True)
    assert torch.isfinite(l_none) and torch.allclose(l_none, l_drop, atol=1e-6)
    assert float(head.weight.grad[0].abs().max()) == 0.0 and float(head.bias.grad[0].abs()) == 0.0  # (none untouched)
    assert float(head.weight.grad[1:].abs().max()) > 0.0
    plain, plain_head = loss(y, False)  # (without it: the none labels count, and the none class learns)
    assert not torch.allclose(plain, loss(dropped, False)[0], atol=1e-6)
    assert float(plain_head.bias.grad[0].abs()) > 0.0


def _evals(seed_ref: int = 1):
    from warcraftsim.fullgame.model import act, evaluate
    net, ref = _net(0).eval(), _net(seed_ref).eval()
    g = torch.Generator().manual_seed(3)
    N, E = 3, 10
    x = [torch.randn(N, E, fx.F, generator=g), torch.randint(0, 20, (N, E), generator=g),
         torch.randint(0, 10, (N, E), generator=g), torch.ones(N, E, dtype=torch.bool), torch.randn(N, 30, generator=g),
         torch.tensor([5, 3, E])]
    with torch.no_grad():
        a = act(net, *x, generator=torch.Generator().manual_seed(1))
        args = (a["order"], a["tgt"], a["bx"], a["by"])
        return evaluate(net, *x, *args), evaluate(ref, *x, *args)


def _kl(a, b):
    la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
    return (la.exp() * (la - lb)).sum(-1)


def test_forward_acting_kl_is_zero_for_identical_nets_and_ignores_none():
    from warcraftsim.fullgame.model import anchor_kl
    ev, ev_ref = _evals()
    assert float(anchor_kl(ev, ev, "forward-acting").abs().max()) < 1e-6
    kl = anchor_kl(ev, ev_ref, "forward-acting")
    assert bool((kl >= -1e-6).all()) and float(kl.max()) > 1e-3
    shifted = dict(ev_ref, logits=ev_ref["logits"].clone())  # the anchor's none logit alone changed: no difference
    shifted["logits"][..., 0] += 7.0
    assert torch.allclose(anchor_kl(ev, shifted, "forward-acting"), kl, atol=1e-6)
    # the targets' terms given the step's order (as forward-all), and the order's: the policy's chance of acting
    # times the KL of the orders given one
    targets = ev["uses_ptr"] * _kl(ev_ref["ptr"], ev["ptr"]) + ev["uses_pt"] * (_kl(ev_ref["xl"], ev["xl"])
                                                                              + _kl(ev_ref["yl"], ev["yl"]))
    act_p = 1 - torch.softmax(ev["logits"].float(), -1)[..., 0]
    assert torch.allclose(kl, act_p * _kl(ev_ref["logits"][..., 1:], ev["logits"][..., 1:]) + targets, atol=1e-5)
    passive = dict(ev, logits=ev["logits"].clone())  # a policy that never acts: no pull on its orders at all
    passive["logits"][..., 0] = 1e4
    assert torch.allclose(anchor_kl(passive, ev_ref, "forward-acting"), targets, atol=1e-6)
    # no gradient to the policy's none logit (its weight is a constant)
    logits = ev["logits"].clone().requires_grad_(True)
    anchor_kl(dict(ev, logits=logits), ev_ref, "forward-acting").sum().backward()
    assert float(logits.grad[..., 0].abs().max()) == 0.0 and float(logits.grad[..., 1:].abs().max()) > 0.0


def test_old_kl_modes_unchanged():
    from warcraftsim.fullgame.model import anchor_kl
    ev, ev_ref = _evals()
    assert torch.allclose(anchor_kl(ev, ev_ref, "reverse"), _kl(ev["logits"], ev_ref["logits"]))
    fwd = _kl(ev_ref["logits"], ev["logits"])
    assert torch.allclose(anchor_kl(ev, ev_ref, "forward"), fwd)
    pt = _kl(ev_ref["xl"], ev["xl"]) + _kl(ev_ref["yl"], ev["yl"])
    want = fwd + ev["uses_ptr"] * _kl(ev_ref["ptr"], ev["ptr"]) + ev["uses_pt"] * pt
    assert torch.allclose(anchor_kl(ev, ev_ref, "forward-all"), want)
