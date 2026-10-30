"""The update's opt-in speedups, on the CPU (selfplay.GraphedUpdate): --update-grad-fuse gives backward()'s gradients
and None for the weights the loss doesn't reach, dynamo traces it whole, --adam-fused takes the same steps and survives
a resume, --update-widths pads as asked."""

from __future__ import annotations

import copy
import types

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame.model import FullGameNet
from warcraftsim.fullgame.selfplay import GraphedUpdate, collate_seq, load_optimizer, save_checkpoint, update_widths


def make_net(**kw) -> FullGameNet:
    torch.manual_seed(0)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=32, layers=1, memory=True, dropout=0.0, **kw)
    net.allowed[:] = True
    net.order_kind[10:20] = fx.UNIT
    net.order_kind[20:] = fx.POINT
    return net


def make_chunks(rng, labels: bool = False, steps: int = 32, seq: int = 8) -> list:
    """Steps of one game side as the learner gets them: its entity count changes, a few units, some labels."""
    out = []
    for t in range(steps):
        n = int(rng.integers(3, 30))
        o = int(rng.integers(1, min(n, 6)))
        st = {"ent": rng.normal(size=(n, fx.F)).astype(np.float16), "type": rng.integers(0, 20, n).astype(np.int16),
              "cur": rng.integers(0, 10, n).astype(np.int16), "glob": rng.normal(size=30).astype(np.float32), "n": n, "n_own": o,
              "logp": (rng.normal(size=o) * 0.1 - 1).astype(np.float32), "adv": float(rng.normal()), "ret": float(rng.normal()),
              "avail": None, "order": rng.integers(0, 30, o).astype(np.int16), "tgt": rng.integers(0, n, o).astype(np.int16),
              "bx": rng.integers(0, 50, o).astype(np.int16), "by": rng.integers(0, 50, o).astype(np.int16),
              "uadv": float(rng.normal()), "h": rng.normal(size=32).astype(np.float32)}
        if labels and t % 2 == 0:
            st.update({"y_order": rng.integers(0, 30, o).astype(np.int16), "y_ptr": rng.integers(0, n, o).astype(np.int16),
                       "y_x": rng.integers(0, 50, o).astype(np.int16), "y_y": rng.integers(0, 50, o).astype(np.int16)})
        out.append(st)
    return [out[a:a + seq] for a in range(0, len(out), seq)]


def settings(**kw) -> types.SimpleNamespace:
    return types.SimpleNamespace(**{**dict(bf16=0, clip=0.2, vf_coef=0.5, ent_coef=0.01, ref_kl=0.1, ref_kl_mode="forward-all", upgo=1.0,
                                           unit_credit=0.0, opd_coef=0.0, opsd_coef=0.0, bc_coef=0.0, max_grad_norm=1.0, minibatch=16,
                                           seq_len=8, pad_groups=1), **kw})


def updater(net, args, ref=None, opt=None):
    opt = opt or torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-5)  # (capturable: GPU only)
    return GraphedUpdate(net, ref, opt, args, torch.device("cpu"))


def statics(chunks, labels: bool) -> dict:
    mb = collate_seq([c for c in chunks], torch.device("cpu"), width=32, B=len(chunks), T=8, labels=labels)
    return mb


def backward_grads(net, ref, args, chunks, labels):
    u = updater(net, args, ref)
    s, _ = u._static(statics(chunks, labels), None)
    net.train()
    net.zero_grad(set_to_none=True)
    u._backward(s, None)
    return [None if p.grad is None else p.grad.clone() for p in net.parameters()], s


@pytest.mark.parametrize("bf16,labels", [(0, False), (1, False), (0, True)])
def test_grad_fuse_equals_backward(bf16, labels):
    """torch.func's gradients are backward()'s, the anchor's KL, UPGO's and the labels' terms among them, under bf16 too.
    A fine head the run's points don't use: None from both, not zeros (Adam would move it by its momentum)."""
    rng = np.random.default_rng(0)
    net, ref = make_net(fine=2), make_net()
    args = settings(bf16=bf16, opd_coef=0.1 if labels else 0.0, teach_coef=0.0, label_norm="units")
    chunks = make_chunks(rng, labels)
    want, s = backward_grads(net, ref, args, chunks, labels)
    assert any(g is None for g in want), "the test needs a weight the loss doesn't reach"
    u = updater(net, settings(**{**vars(args), "update_grad_fuse": 1}), ref)
    u._find_unused(s, None)
    net.zero_grad(set_to_none=True)
    out = u._backward(s, None)
    assert set(out) >= {"loss/policy", "loss/value", "loss/entropy", "loss/kl", "loss/ref_kl"}
    assert all(not v.requires_grad for v in out.values())
    whole = float(torch.cat([g.flatten() for g in want if g is not None]).norm())
    for (name, p), g in zip(net.named_parameters(), want):
        assert (p.grad is None) == (g is None), name
        if g is None:
            continue
        if bf16:  # (bf16's rounding, 8 bits: the two sum a weight's parts in another order, 0.3% apart on the mean,
            assert float((p.grad - g).norm()) <= 0.02 * float(g.norm()) + 1e-4 * whole, name  # bf16 and fp32 0.5-0.9%)
        else:
            assert torch.allclose(p.grad, g, rtol=1e-4, atol=1e-6 * whole), name


def test_grad_fuse_traced_by_dynamo_whole():
    """dynamo traces the gradients with the network (minGRU memory, attention masks, the statistics as aux) without a
    graph break, and the traced function's gradients are backward()'s. (The inductor's compile itself: scripts/bench_gpu_round.py.)"""
    rng = np.random.default_rng(1)
    net, ref = make_net(), make_net()
    args = settings()
    chunks = make_chunks(rng)
    want, s = backward_grads(net, ref, args, chunks, False)
    u = updater(net, settings(update_grad_fuse=1), ref)
    u._find_unused(s, None)
    params = {n: p.detach() for n, p in u._named}
    info = torch._dynamo.explain(u._grads)(params, s, None)
    assert info.graph_count == 1 and info.graph_break_count == 0, info.break_reasons
    u._grads = torch.compile(u._grads, backend="eager", dynamic=False)
    net.zero_grad(set_to_none=True)
    u._backward(s, None)
    for (name, p), g in zip(net.named_parameters(), want):
        assert torch.allclose(p.grad, g, rtol=1e-5, atol=1e-7), name


def test_grad_fuse_takes_backwards_step():
    """A whole step (the clip, Adam) from either gradient: the same weights."""
    rng = np.random.default_rng(2)
    a, ref = make_net(), make_net()
    b = copy.deepcopy(a)
    chunks = make_chunks(rng)
    mb = statics(chunks, False)
    ua, ub = updater(a, settings(), ref), updater(b, settings(update_grad_fuse=1), ref)
    s, _ = ua._static(mb, None)
    ub._find_unused(s, None)
    for u in (ua, ub):
        u.net.train()
        for _ in range(3):
            u.opt.zero_grad(set_to_none=True)
            out = u._step(s, None)
            assert float(out["grad_norm"]) > 0
    for (name, x), y in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-6), name


def test_adam_fused_takes_the_same_steps():
    """Adam(fused=True, capturable=True) as the foreach path it replaces, the step count on the weights' device."""
    a = make_net()
    b = copy.deepcopy(a)
    oa = torch.optim.Adam(a.parameters(), lr=1e-3, eps=1e-5)  # (capturable: GPU only)
    ob = torch.optim.Adam(b.parameters(), lr=1e-3, eps=1e-5, capturable=True, fused=True)
    torch.manual_seed(3)
    for _ in range(5):
        for p, q in zip(a.parameters(), b.parameters()):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        oa.step()
        ob.step()
    for (name, x), y in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(x, y, rtol=1e-5, atol=1e-6), name
    p = next(b.parameters())
    assert float(ob.state[p]["step"]) == 5 and ob.state[p]["step"].dtype == torch.float32


@pytest.mark.parametrize("saved_fused,wanted", [(False, True), (True, False), (False, False)])
def test_load_optimizer_keeps_this_runs_fused(tmp_path, saved_fused, wanted):
    """A saved state's param group carries its run's "fused": a resume keeps this run's setting (and its moments)."""
    net = make_net()
    kw = {"fused": True} if saved_fused else {}
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-5, **kw)
    for p in net.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    path = tmp_path / "0000000000000001.pt"
    save_checkpoint(path, net, opt, {}, 1, 1)
    kw = {"fused": True} if wanted else {}
    fresh = torch.optim.Adam(net.parameters(), lr=3e-5, eps=1e-5, **kw)
    assert load_optimizer(fresh, path, torch.device("cpu"), 3e-5, net)
    g = fresh.param_groups[0]
    assert bool(g["fused"]) == wanted and g["lr"] == 3e-5
    assert float(fresh.state[next(net.parameters())]["step"]) == 1
    for p in net.parameters():
        p.grad = torch.ones_like(p)
    fresh.step()


def test_update_widths():
    assert update_widths("") == (32, 48, 64, 96, 128, fx.MAX_ENT)
    assert update_widths("32,40,48") == (32, 40, 48, fx.MAX_ENT)
    assert update_widths("48,32,48") == (32, 48, fx.MAX_ENT)
    with pytest.raises(ValueError):
        update_widths(f"32,{fx.MAX_ENT + 1}")
    rng = np.random.default_rng(4)
    chunks = make_chunks(rng, steps=64)
    u = updater(make_net(), settings(update_widths="20,30,40"))
    assert u.widths == (20, 30, 40, fx.MAX_ENT)
    keys = [key for key, mb, _ in u._batches(chunks, None, 1)]
    assert keys and all(k[0] in u.widths for k in keys)
    assert updater(make_net(), settings()).widths == (32, 48, 64, 96, 128, fx.MAX_ENT)
