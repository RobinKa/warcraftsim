"""--update-label-rows K (selfplay.with_label_rows, distill_loss): the labels' target heads on a minibatch's labelled steps
alone give the same loss, statistics and gradients as on every step (the floors' paths, the teacher's lessons and both
label norms among them), on the CPU; more labelled steps than K: the minibatch as before."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import selfplay as S
from warcraftsim.fullgame.model import FullGameNet

C, BINS = 30, 50


def make_net(seed: int = 0) -> FullGameNet:
    torch.manual_seed(seed)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=C, G=30, d=32, layers=1, memory=True, bins=BINS)
    net.allowed[:] = True
    net.order_kind[10:20] = fx.UNIT
    net.order_kind[20:] = fx.POINT
    return net


def make_seqs(rng, seqs: int = 6, T: int = 8, every: int = 5) -> list[list[dict]]:
    """Sequences of steps with labels (orders, pointers, points, the teacher's lessons) on about one step in `every`."""
    out = []
    for _ in range(seqs):
        q = []
        for t in range(T):
            n = int(rng.integers(3, 28))
            o = int(rng.integers(1, min(n, 7)))
            st = {"ent": rng.normal(size=(n, fx.F)).astype(np.float16), "type": rng.integers(0, 20, n).astype(np.int16),
                  "cur": rng.integers(0, 10, n).astype(np.int16), "glob": rng.normal(size=30).astype(np.float32), "n": n,
                  "n_own": o, "logp": (rng.normal(size=o) * 0.1 - 1).astype(np.float32), "adv": float(rng.normal()),
                  "ret": float(rng.normal()), "avail": None, "order": rng.integers(0, C, o).astype(np.int16),
                  "tgt": rng.integers(0, n, o).astype(np.int16), "bx": rng.integers(0, BINS, o).astype(np.int16),
                  "by": rng.integers(0, BINS, o).astype(np.int16), "uadv": float(rng.normal()),
                  "h": rng.normal(size=32).astype(np.float32) if t == 0 else None}
            if rng.integers(every) == 0:
                y = rng.integers(0, C, o)
                y[rng.random(o) < 0.3] = 0
                st.update({"y_order": y.astype(np.int16),
                           "y_ptr": np.where(rng.random(o) < 0.6, rng.integers(0, n, o), -1).astype(np.int16),
                           "y_x": np.where(rng.random(o) < 0.7, rng.integers(0, BINS, o), -1).astype(np.int16),
                           "y_teach": rng.random(o) < 0.4})
                st["y_y"] = np.where(st["y_x"] >= 0, rng.integers(0, BINS, o), -1).astype(np.int16)
            q.append(st)
        out.append(q)
    return out


@pytest.fixture()
def floors():
    """The floors' paths on (moves' place, harvest's pointer), as the run's --teach-floor-orders: the orders 20.. and 10..
    put back after."""
    keep = (S.TEACH_FLOOR_PLACE, S.TEACH_FLOOR_POINTER, S.TEACH_POINT)
    place = torch.zeros(C, dtype=torch.bool)
    place[20:25] = True
    pointer = torch.zeros(C, dtype=torch.bool)
    pointer[10:13] = True
    S.TEACH_FLOOR_PLACE, S.TEACH_FLOOR_POINTER, S.TEACH_POINT = (place, math.log(0.25)), (pointer, math.log(0.3)), 0.5
    yield
    S.TEACH_FLOOR_PLACE, S.TEACH_FLOOR_POINTER, S.TEACH_POINT = keep


def settings(norm: str):
    import types
    return types.SimpleNamespace(bf16=0, clip=0.2, vf_coef=0.5, ent_coef=0.01, ref_kl=0.1, ref_kl_mode="forward-all", upgo=1.0,
                                 unit_credit=0.0, opd_coef=0.1, teach_coef=0.5, label_norm=norm, opsd_coef=0.0, bc_coef=0.0,
                                 max_grad_norm=1.0, minibatch=48, seq_len=8, pad_groups=1, ref2_kl=0.0)


def _loss_and_grads(net, ref, args, mb):
    up = S.GraphedUpdate(net, ref, torch.optim.Adam(net.parameters(), lr=1e-3), args, torch.device("cpu"))
    s, _ = up._static(mb, None)
    net.train()
    net.zero_grad(set_to_none=True)
    loss, out = up._loss(s, None)
    loss.backward()
    return float(loss), {k: float(v) for k, v in out.items()}, [None if p.grad is None else p.grad.clone() for p in net.parameters()]


@pytest.mark.parametrize("norm", ["units", "lessons"])
def test_label_rows_same_loss_and_gradients(floors, norm):
    rng = np.random.default_rng(1)
    seqs = make_seqs(rng)
    net, ref = make_net(0), make_net(1)
    args = settings(norm)
    B, T = 6, 8
    full = S.collate_seq(seqs, torch.device("cpu"), width=32, B=B, T=T, labels=True)
    n_lab = sum("y_order" in st for q in seqs for st in q)
    assert 2 <= n_lab <= 16, n_lab
    sub = S.collate_seq(seqs, torch.device("cpu"), width=32, B=B, T=T, labels=True)
    assert S.with_label_rows(sub, seqs, B, T, 16, torch.device("cpu"))
    assert int(sub["lab_ok"].sum()) == n_lab and torch.equal(sub["y_has"][sub["lab_idx"][:n_lab]], torch.ones(n_lab, dtype=torch.bool))
    l0, o0, g0 = _loss_and_grads(net, ref, args, full)
    l1, o1, g1 = _loss_and_grads(net, ref, args, sub)
    assert o0["loss/opd"] > 0 and o0["loss/teach"] != 0  # (the labels' and the lessons' terms are in it)
    assert abs(l0 - l1) <= 1e-5 * max(1.0, abs(l0)), (l0, l1)
    for k in o0:
        assert abs(o0[k] - o1[k]) <= 1e-5 * max(1.0, abs(o0[k])), (k, o0[k], o1[k])
    whole = float(torch.cat([g.flatten() for g in g0 if g is not None]).norm())
    for (name, _), a, b in zip(net.named_parameters(), g0, g1):
        assert (a is None) == (b is None), name
        if a is not None:
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-6 * whole), name


def test_more_labelled_steps_than_k_keep_the_whole_minibatch():
    rng = np.random.default_rng(2)
    seqs = make_seqs(rng, every=1)  # (every step labelled)
    mb = S.collate_seq(seqs, torch.device("cpu"), width=32, B=8, T=8, labels=True)
    assert not S.with_label_rows(mb, seqs, 8, 8, 16, torch.device("cpu")) and "lab_idx" not in mb
    none = make_seqs(rng, every=10 ** 9)  # (no labels: K dummy rows, all weighing 0)
    mb = S.collate_seq(none, torch.device("cpu"), width=32, B=8, T=8, labels=True)
    assert S.with_label_rows(mb, none, 8, 8, 4, torch.device("cpu")) and not bool(mb["lab_ok"].any())


def test_batches_add_label_rows_only_when_asked():
    """GraphedUpdate._batches: lab_idx in the minibatch with --update-label-rows and labels on (a graph key of its own)."""
    rng = np.random.default_rng(3)
    seqs = make_seqs(rng, seqs=12)
    chunks = [[st for q in seqs[a:a + 2] for st in q] for a in range(0, 12, 2)]
    net = make_net()
    for K, want in ((0, False), (32, True)):
        args = settings("units")
        args.update_label_rows = K
        args.epochs = 1
        up = S.GraphedUpdate(net, None, torch.optim.Adam(net.parameters(), lr=1e-3), args, torch.device("cpu"))
        keys = [k for k, mb, _ in up._batches(chunks, None, 1)]
        assert all(("lab_idx" in k[3]) == want for k in keys), (K, keys)
