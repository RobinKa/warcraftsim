"""--ref-ema (selfplay.EmaAnchor): at 0 the anchor stays as it was, bit for bit; a step moves each weight tau of the way
to the policy's, buffers and weights of another shape left alone; the anchor is saved with a checkpoint (<steps>.ref) and
taken up again on a resume."""

from __future__ import annotations

import copy

import numpy as np
import torch
from test_update_fuse import make_chunks, make_net, settings, statics, updater

from warcraftsim.fullgame.model import FullGameNet
from warcraftsim.fullgame.selfplay import EmaAnchor, save_checkpoint


def make_pair():
    torch.manual_seed(0)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=32, layers=1, memory=True, dropout=0.0)
    ref = copy.deepcopy(net).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(3):  # (the policy trained away from the anchor)
        opt.zero_grad()
        sum((p * torch.randn_like(p)).sum() for p in net.parameters()).backward()
        opt.step()
    return net, ref, opt


def snapshot(m: torch.nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def test_tau_zero_leaves_the_anchor_bit_identical():
    net, ref, _ = make_pair()
    before = snapshot(ref)
    ema = EmaAnchor(ref, net, 0.0)
    for _ in range(5):
        ema.step()
    after = ref.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert ema.distances()["ref_ema/from_init"] == 0.0


def test_a_step_moves_the_anchor_tau_towards_the_policy():
    net, ref, _ = make_pair()
    with torch.no_grad():  # (a weight the policy has in another shape: a grown network's; it stays --init's)
        grown = net.value_head[-1]
        grown.weight = torch.nn.Parameter(torch.randn(grown.weight.shape[0] + 1, grown.weight.shape[1]))
        ref.target_temp.fill_(0.5)  # (a buffer of the anchor's own: stays)
    tau = 1 - 0.5 ** (1 / 6000)
    before, pol = snapshot(ref), snapshot(net)
    ema = EmaAnchor(ref, net, tau)
    assert ema.left == 1 and "value_head.2.weight" not in ema.names
    gap0 = ema.distances()["ref_ema/to_policy"]
    ema.step()
    moved = 0
    for k, v in ref.state_dict().items():
        if k in ema.names:
            want = before[k] + tau * (pol[k] - before[k])
            assert torch.allclose(v, want, rtol=0, atol=1e-7), k
            moved += int(not torch.equal(v, before[k]))
        else:
            assert torch.equal(v, before[k]), k
    assert moved > 0 and float(ref.target_temp) == 0.5
    d = ema.distances()
    assert d["ref_ema/from_init"] > 0
    assert abs(d["ref_ema/to_policy"] - (1 - tau) * gap0) < 1e-6 * gap0


def test_the_anchor_is_saved_with_the_checkpoint_and_resumed(tmp_path, capsys):
    net, ref, opt = make_pair()
    init = snapshot(ref)
    ema = EmaAnchor(ref, net, 0.1, "runs/x/init.pt")
    for _ in range(3):
        ema.step()
    path = tmp_path / f"{100:016d}.pt"
    save_checkpoint(path, net, opt, {}, 100, 3, ema=ema)
    assert path.with_suffix(".ref").exists()
    assert [p.name for p in tmp_path.glob("[0-9]*.pt")] == [path.name]  # (not among the checkpoints)
    save_checkpoint(tmp_path / f"{200:016d}.pt", net, opt, {}, 200, 4)  # (without --ref-ema: no anchor saved)
    assert not (tmp_path / f"{200:016d}.ref").exists()

    def fresh(init_path: str):  # (a new process: the anchor at --init again)
        r = copy.deepcopy(ref)
        r.load_state_dict(init)
        return r, EmaAnchor(r, net, 0.1, init_path)

    r, e = fresh("runs/x/init.pt")
    assert e.resume(path) == path.with_suffix(".ref").name
    assert all(torch.equal(a, b) for a, b in zip(r.parameters(), ref.parameters()))
    assert e.distances() == ema.distances()  # (its distance from --init: from the saved --init's weights)
    r, e = fresh("runs/x/init.pt")  # (a checkpoint without one: the newest saved before it)
    assert e.resume(tmp_path / f"{200:016d}.pt") == path.with_suffix(".ref").name
    assert all(torch.equal(a, b) for a, b in zip(r.parameters(), ref.parameters()))
    r, e = fresh("runs/x/init.pt")  # (none saved at or before it: --init)
    assert e.resume(tmp_path / f"{50:016d}.pt") is None
    r, e = fresh("runs/y/other.pt")  # (another --init: the anchor from it, not the saved one)
    assert e.resume(path) is None and "not this run's --init" in capsys.readouterr().out
    assert all(torch.equal(r.state_dict()[k], v) for k, v in init.items())


def test_the_update_reads_the_moved_anchor():
    """The update's KL term reads the anchor's weights where they are (GraphedUpdate's, as its CUDA graphs do): moved all
    the way to the policy (tau 1), the KL to it is 0."""
    net = make_net()
    ref = copy.deepcopy(net).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    u = updater(net, settings(), ref)
    s, _ = u._static(statics(make_chunks(np.random.default_rng(0)), False), None)
    assert float(u._loss(s, None)[1]["loss/ref_kl"].detach()) > 1e-4
    EmaAnchor(ref, net, 1.0).step()
    assert float(u._loss(s, None)[1]["loss/ref_kl"].detach()) < 1e-6
