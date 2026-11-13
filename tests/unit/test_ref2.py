"""--ref2 (selfplay.ANCHOR2, second_anchor_kl): a second KL anchor beside --init with its own weight and direction. None
set: the update's loss as before, bit for bit; set: its term joins the loss (loss/ref2_kl), 0 when the anchor is the
policy itself, and in both update paths (GraphedUpdate and ppo_update)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from test_update_fuse import make_chunks, make_net, settings, statics, updater

from warcraftsim.fullgame import selfplay


@pytest.fixture(autouse=True)
def no_anchor2():
    selfplay.ANCHOR2.clear()
    yield
    selfplay.ANCHOR2.clear()


def frozen(net):
    r = copy.deepcopy(net).eval()
    for p in r.parameters():
        p.requires_grad_(False)
    return r


def moved(net, by=0.05, seed=1):
    torch.manual_seed(seed)
    r = frozen(net)
    with torch.no_grad():
        for p in r.parameters():
            p.add_(by * torch.randn_like(p))
    return r


def loss_of(net, ref, args):
    u = updater(net, args, ref)
    s, _ = u._static(statics(make_chunks(np.random.default_rng(0)), False), None)
    return u._loss(s, None)


def test_without_it_the_loss_is_unchanged():
    net = make_net()
    ref = moved(net)
    a = loss_of(net, ref, settings())
    selfplay.ANCHOR2.append(moved(net, seed=2))
    b = loss_of(net, ref, settings(ref2_kl=0.0, ref2_kl_mode="forward-acting"))
    assert torch.equal(a[0].detach(), b[0].detach()) and "loss/ref2_kl" not in b[1]


@pytest.mark.parametrize("mode", ["forward-acting", "forward-all", "reverse"])
def test_its_term_joins_the_loss(mode):
    net = make_net()
    ref = moved(net)
    base = loss_of(net, ref, settings())[0].detach()
    selfplay.ANCHOR2.append(moved(net, seed=2))
    loss, out = loss_of(net, ref, settings(ref2_kl=0.5, ref2_kl_mode=mode))
    k2 = float(out["loss/ref2_kl"].detach())
    assert k2 > 1e-5
    assert float(loss.detach()) == pytest.approx(float(base) + 0.5 * k2, rel=1e-5, abs=1e-6)


def test_zero_when_the_second_anchor_is_the_policy():
    net = make_net()
    selfplay.ANCHOR2.append(frozen(net))
    _, out = loss_of(net, moved(net), settings(ref2_kl=1.0, ref2_kl_mode="forward-all"))
    assert float(out["loss/ref2_kl"].detach()) < 1e-6


def test_ppo_update_reads_it_too():
    net = make_net()
    ref = moved(net)
    selfplay.ANCHOR2.append(moved(net, seed=2))
    chunks = make_chunks(np.random.default_rng(0))
    steps = [s for c in chunks for s in c]
    args = settings(ref2_kl=0.5, ref2_kl_mode="forward-acting", epochs=1, lr=1e-3, label_w=1.0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    stats = selfplay.ppo_update(net, ref, opt, steps, args, False, torch.device("cpu"), chunks, None)
    assert stats.get("loss/ref2_kl", 0.0) > 1e-5
