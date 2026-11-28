"""--infer-past-stack (paststack.PastStack), on the CPU: a stacked call gives each slot's rows exactly what that slot's own
call gives with the same draws (actions bit for bit, numbers to float rounding), eager and compiled; a snapshot copied
into a slot's places as the C++ loop copies it (dst(), raw bytes) is what the slot computes with; the sampling's draws made
by the caller are the draws act() would make; the vmap-safe forms in model.py (the mask's columns, the encoder's token
layout) give the same numbers as before. The C++ loop itself and the speed: scripts/check_native_infer.py --past-stack
and scripts/bench_speed10.py on the GPU."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import model as M
from warcraftsim.fullgame.model import FullGameNet, act
from warcraftsim.fullgame.paststack import PastStack, blank_rows, pack_rows, row_width, stack_rows
from warcraftsim.fullgame.selfplay import pack_flat, packed_call

T, C, G, D = 20, 30, 30, 32  # (unit types, order classes, global numbers, width)


@pytest.fixture(autouse=True)
def inputs2():
    """Hero features in the entities (features.set_inputs 2: unit_valid reads them), put back after."""
    was = fx.INPUTS
    fx.set_inputs(2)
    yield
    fx.set_inputs(was)


def _net(seed: int, memory: bool = True, valid: bool = True, omask: bool = True, layers: int = 2) -> FullGameNet:
    torch.manual_seed(seed)
    net = FullGameNet(n_types=T, n_cur=10, n_orders=C, G=G, d=D, layers=layers, memory=memory, bins=40,
                      valid=6 if valid else 0, omask=5 if omask else 0).eval()
    g = torch.Generator().manual_seed(100 + seed)
    with torch.no_grad():
        net.allowed.copy_(torch.rand(T, C, generator=g) < 0.7)
        net.order_kind.copy_(torch.randint(0, 5, (C,), generator=g))
        if memory:
            torch.nn.init.normal_(net.mem_out.weight, std=0.3)
        net.order[2].weight.mul_(4.0)
        net.key.weight.mul_(3.0)
        if valid:  # (distinct columns, as set_valid's sorted set; needs of each type, -1 none)
            net.valid_cols.copy_(torch.randperm(C - 1, generator=g)[:6] + 1)
            net.valid_need.copy_(torch.randint(-1, M.VALID_BITS, (T, 6), generator=g).to(torch.int8))
        if omask:
            net.ptr_rule.copy_(torch.randint(0, 64, (C,), generator=g, dtype=torch.int32))
            net.type_kind.copy_(torch.randint(0, 4, (T,), generator=g, dtype=torch.int32))
            net.ptr_cols.copy_(torch.randperm(C - 1, generator=g)[:5] + 1)
    return net


def _step(rng, n: int, own: int, memory: bool = True) -> dict:
    avail = rng.random(C) < 0.7
    avail[0] = True
    return {"n": n, "n_own": own, "ent": rng.random((n, fx.F)).astype(np.float32), "type": rng.integers(0, T, n),
            "cur": rng.integers(0, 10, n), "glob": rng.normal(size=G).astype(np.float32), "avail": avail,
            "h": rng.normal(size=D).astype(np.float32) if memory and rng.random() < 0.7 else None,
            "shared_noise": bool(rng.random() < 0.5)}


def _rows(net, S: int, B: int, E: int, rng, counts=None) -> torch.Tensor:
    """[S * B, W] packed rows: slot s's first counts[s] rows real steps, the rest blank padding."""
    x = blank_rows(net, S * B, E)
    for s in range(S):
        k = B if counts is None else counts[s]
        pack_rows(net, x, [_step(rng, int(rng.integers(2, E + 1)), int(rng.integers(0, 9)), net.memory) for _ in range(k)],
                  E, s * B)
    return x


def _own_call(net, x, noise, s: int) -> torch.Tensor:
    c = net.config
    pairs = {r: (noise[r][s], noise[r + "1"][s]) for r in noise if not r.endswith("1")}
    return packed_call(net, x, c["G"], c["n_orders"], c["d"] if net.memory else 0, False, None, False, False, noise=pairs)


def _same(want: torch.Tensor, got: torch.Tensor, O: int, tol: float = 1e-5) -> None:
    assert torch.equal(want[:, :4 * O], got[:, :4 * O])  # (order, target, x, y: the same draws, the same samples)
    assert (want[:, 4 * O:] - got[:, 4 * O:]).abs().max().item() < tol  # (log-probabilities, value, entropy, memory)


def test_stack_rows():
    assert stack_rows("") == (1, 2, 4) and stack_rows(None) == (1, 2, 4)
    assert stack_rows("4,1,2,2") == (1, 2, 4) and stack_rows([8, 2]) == (2, 8)
    for bad in ("0,2", "17"):
        with pytest.raises(ValueError):
            stack_rows(bad)


def test_and_cols_is_the_indexed_form():
    g = torch.Generator().manual_seed(0)
    allowed = torch.rand(3, 7, 30, generator=g) < 0.6
    cols = torch.randperm(30, generator=g)[:9]
    ok = torch.rand(3, 7, 9, generator=g) < 0.5
    want = allowed.clone()
    want[..., cols] &= ok
    assert torch.equal(M._and_cols(allowed, cols, ok), want)


@pytest.mark.parametrize("memory", [False, True])
def test_drawn_noise_is_acts_own(memory):
    """act(noise=...) with the uniform draws act() would make itself (the same generator, the same order) gives the same
    samples and numbers bit for bit."""
    net = _net(0, memory=memory)
    rng = np.random.default_rng(1)
    B, E = 5, 12
    x = blank_rows(net, B, E)
    pack_rows(net, x, [_step(rng, int(rng.integers(2, E + 1)), int(rng.integers(0, 9)), memory) for _ in range(B)], E)
    c = net.config
    d = c["d"] if memory else 0
    with torch.no_grad():
        want = packed_call(net, x, c["G"], c["n_orders"], d, False, torch.Generator().manual_seed(7), False, False)
        g = torch.Generator().manual_seed(7)
        shapes = M.noise_shapes(net, B, E)
        noise = {"order": (torch.rand(shapes["order"][0], generator=g), None)}
        for role in ("tgt", "x", "y"):  # (act's order: each draw, then its shared one)
            u = torch.rand(shapes[role][0], generator=g)
            noise[role] = (u, torch.rand(shapes[role][1], generator=g))
        got = packed_call(net, x, c["G"], c["n_orders"], d, False, None, False, False, noise=noise)
    assert torch.equal(want, got)


@pytest.mark.parametrize("memory", [False, True])
def test_stacked_call_is_each_slots_own(memory):
    """Four slots, three snapshots in them (one left with the stack's first network), two rows a slot (one slot with one
    real row, padding in the other): each slot's rows answered as that slot's network answers them alone."""
    nets = [_net(k, memory=memory) for k in range(3)]
    S, B, E = 4, 2, 12
    stack = PastStack(nets[0], S, "cpu", rows=(1, 2), entities=(E,))
    for s, n in enumerate(nets):
        stack.load(s, n, f"k{s}")
    rng = np.random.default_rng(2)
    x = _rows(nets[0], S, B, E, rng, counts=[2, 1, 2, 2])
    noise = stack.draw(B, E)
    with torch.no_grad():
        out = stack.call(x, noise)
        for s in range(S):
            net = nets[s] if s < len(nets) else nets[0]
            _same(_own_call(net, x[s * B:(s + 1) * B], noise, s), out[s * B:(s + 1) * B], min(fx.MAX_OWN, E))


def test_a_snapshot_copied_into_a_slots_places():
    """The C++ loop's copy (NativeServer._snapshot's places: pack_flat of the parameters, then each buffer) into dst(s),
    byte for byte (emulated with memmove on the CPU): the slot then answers as the snapshot."""
    base, snap = _net(0), _net(5)
    S, B, E = 3, 1, 10
    stack = PastStack(base, S, "cpu", rows=(1,), entities=(E,))
    flat = pack_flat(snap.parameters())
    src = [(flat.data_ptr(), flat.numel() * flat.element_size())] + \
        [(b.data_ptr(), b.numel() * b.element_size()) for b in snap.buffers()]
    dst = stack.dst(1)
    assert [n for _, n in dst] == [n for _, n in src]  # (the same places, in the same order)
    for (d, n), (s_, _) in zip(dst, src):
        ctypes.memmove(d, s_, n)
    x = _rows(base, S, B, E, np.random.default_rng(3))
    noise = stack.draw(B, E)
    with torch.no_grad():
        out = stack.call(x, noise)
        _same(_own_call(snap, x[B:2 * B], noise, 1), out[B:2 * B], min(fx.MAX_OWN, E))
        _same(_own_call(base, x[:B], noise, 0), out[:B], min(fx.MAX_OWN, E))  # (the others untouched)


def test_token_major_encoder_is_the_same():
    """The stack's network runs its encoder over [L, N, d] tokens (FullGameNet.encode; batch_first off): the same
    numbers as the batch-first encoder's."""
    net = _net(0)
    other = FullGameNet(**net.config).eval()
    other.load_state_dict(net.state_dict())
    for layer in other.transformer.layers:
        layer.self_attn.batch_first = False
    g = torch.Generator().manual_seed(4)
    N, E = 6, 11
    ent, typ = torch.rand(N, E, fx.F, generator=g), torch.randint(0, T, (N, E), generator=g)
    cur, glob = torch.randint(0, 10, (N, E), generator=g), torch.randn(N, G, generator=g)
    mask = torch.arange(E)[None] < torch.randint(1, E + 1, (N, 1), generator=g)
    with torch.no_grad():
        for a, b in zip(net.encode(ent, typ, cur, mask, glob), other.encode(ent, typ, cur, mask, glob)):
            assert (a - b).abs().max().item() < 1e-5


def test_stack_rows_layout_and_blank_rows_are_valid():
    """A blank row (the padding of a stacked call's input) has one entity in view and every order available: its answer is
    finite (no attention over nothing)."""
    net = _net(1)
    S, B, E = 2, 2, 8
    stack = PastStack(net, S, "cpu", rows=(2,), entities=(E,))
    x = blank_rows(net, S * B, E)
    assert x.shape == (S * B, row_width(net, E))
    with torch.no_grad():
        out = stack.call(x)
    assert torch.isfinite(out).all()


def test_compiled_stack_is_the_eager_one():
    """The stacked call compiled as the server compiles it (torch.compile over vmap: no graph break, the draws made
    outside it) answers as the eager one."""
    import torch._dynamo
    nets = [_net(k, layers=1) for k in range(2)]
    S, B, E = 2, 2, 10
    stack = PastStack(nets[0], S, "cpu", rows=(2,), entities=(E,))
    stack.load(1, nets[1], "k1")
    x = _rows(nets[0], S, B, E, np.random.default_rng(5))
    noise = stack.draw(B, E)
    with torch.no_grad():
        eager = stack.call(x, noise)
        torch._dynamo.reset()
        explained = torch._dynamo.explain(stack._core_fn())(stack.params, stack.buffers, x.view(S, B, x.shape[1]), noise)
        assert explained.graph_break_count == 0, explained.break_reasons
        stack.fuse, stack._core = True, None  # (the CPU: PastStack compiles only on CUDA unless asked)
        compiled = stack.call(x, noise)
    _same(eager, compiled, min(fx.MAX_OWN, E), tol=1e-4)
