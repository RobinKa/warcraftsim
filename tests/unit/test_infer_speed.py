"""The inference calls' speed options, on the CPU: --infer-rows (finer row buckets) pads a call with more or fewer rows,
and a row's answer does not depend on the rows beside it; --infer-precision tf32 sets TF32 only while the inference
calls are compiled and captured, and puts the process's setting back exactly; both are off by default.
--infer-precision bf16-mixed (the trunk in bf16, the heads and the value in fp32): the default path is the same calls as
before, bit for bit; with CUDA's bf16 autocast emulated on the CPU (scripts/infer_cpu_study.py's CudaAutocast) its
log-probabilities and values stay within what the CPU study measured on fgself-182's network, well inside plain bf16's;
and the past snapshots' slots compute with the weights copied into them."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import model as M
from warcraftsim.fullgame.model import FullGameNet
from warcraftsim.fullgame.selfplay import (INFER_PRECISIONS, INFER_ROWS, Inference, PastNet, infer_rows, matmul_precision,
                                           pack_flat, precision_flags)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def test_infer_rows():
    assert INFER_ROWS == (8, 24, 64) and infer_rows("", 64) == (8, 24, 64) and infer_rows(None, 64) == (8, 24, 64)
    assert infer_rows("8,16,24,32,48,64", 64) == (8, 16, 24, 32, 48, 64)
    assert infer_rows("32,16,16", 64) == (16, 32, 64)  # (sorted, the call's most always)
    assert infer_rows([8, 24, 64], 64) == (8, 24, 64)  # (as the actors' config carries it)
    assert infer_rows("", 16) == (8, 16)  # (a smaller most: every call still has a graph)
    for bad in ("0,8", "8,65"):
        with pytest.raises(ValueError):
            infer_rows(bad, 64)


def _state() -> tuple:
    b = torch.backends
    return (torch.get_float32_matmul_precision(), b.cuda.matmul.fp32_precision, b.mkldnn.matmul.fp32_precision,
            torch._C._get_cublas_allow_tf32())


def test_tf32_only_inside_and_put_back():
    before = _state()
    with matmul_precision(False):
        assert _state() == before
    with matmul_precision(True):
        assert torch.backends.cuda.matmul.fp32_precision == "tf32"
        assert torch.backends.cuda.matmul.allow_tf32  # (torch's legacy reads agree: no "mix of APIs" error)
        assert torch.get_float32_matmul_precision() == "high"
    assert _state() == before
    with pytest.raises(RuntimeError):
        with matmul_precision(True):
            raise RuntimeError("a failed capture")
    assert _state() == before


def test_precision_off_by_default():
    assert INFER_PRECISIONS[0] == "fp32"
    plain = Inference(None, torch.device("cpu"), 8, compile_=False, thread=False)
    assert not plain.tf32 and not plain.bf16
    cpu = Inference(None, torch.device("cpu"), 8, compile_=False, thread=False, tf32=True)
    assert not cpu.tf32  # (the CUDA graphs' setting: nothing on the CPU)


@pytest.mark.parametrize("memory", [False, True])
def test_a_rows_answer_does_not_depend_on_the_padding_rows(memory):
    """A call's rows beside a game's own (another bucket's padding: rows an earlier call left in the input) leave its
    value, entropy and memory state as they were; only the sampling's noise is drawn for another shape."""
    torch.manual_seed(0)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=64, layers=2, memory=memory).eval()
    net.allowed[:] = True
    net.order_kind.copy_(torch.randint(0, 5, (30,)))
    if memory:
        torch.nn.init.normal_(net.mem_out.weight, std=0.3)
    rng = np.random.default_rng(3)

    def step(n: int, own: int) -> dict:
        avail = rng.random(30) < 0.6
        avail[0] = True
        return {"n": n, "n_own": own, "ent": rng.normal(size=(n, fx.F)).astype(np.float32), "type": rng.integers(0, 20, n),
                "cur": rng.integers(0, 10, n), "glob": rng.normal(size=30).astype(np.float32), "avail": avail,
                "h": rng.normal(size=64).astype(np.float32) if memory else None}
    real = [step(n, o) for n, o in ((14, 5), (7, 7), (14, 1), (3, 2), (11, 6))]
    infer = Inference(None, torch.device("cpu"), 64, compile_=False, thread=False)
    answers = []
    for pad in (0, 3, 19, 27):  # (5 rows alone, and padded to 8, 24 and 32 rows)
        out = infer._finish(infer._launch(net, real + [step(int(rng.integers(1, 15)), int(rng.integers(0, 4)))
                                                       for _ in range(pad)]))
        answers.append(out[:len(real)])
    for got in answers[1:]:
        for a, b in zip(answers[0], got):
            assert abs(a["value"] - b["value"]) < 1e-6 and abs(a["entropy"] - b["entropy"]) < 1e-5
            assert ("h" in a) == memory and (not memory or np.abs(a["h"] - b["h"]).max() < 1e-6)
            assert len(a["order"]) == len(b["order"])


def test_precision_flags():
    assert INFER_PRECISIONS == ("fp32", "tf32", "bf16", "bf16-mixed")
    assert precision_flags(None) == precision_flags("") == precision_flags("fp32") == {"bf16": False, "tf32": False,
                                                                                       "mixed": False}
    assert precision_flags("bf16-mixed") == {"bf16": False, "tf32": False, "mixed": True}
    with pytest.raises(ValueError):
        precision_flags("fp16")
    cuda = types.SimpleNamespace(type="cuda")  # (the flags as a GPU's Inference sets them; nothing runs)
    for prec, want in (("fp32", (False, False, False)), ("tf32", (False, True, False)), ("bf16", (True, False, False)),
                       ("bf16-mixed", (True, True, True))):  # (bf16, tf32: the heads' fp32 multiplies, mixed)
        inf = Inference(None, cuda, 8, compile_=False, thread=False, **precision_flags(prec))
        assert (inf.bf16, inf.tf32, inf.mixed) == want, prec
        cpu = Inference(None, torch.device("cpu"), 8, compile_=False, thread=False, **precision_flags(prec))
        assert not (cpu.bf16 or cpu.tf32 or cpu.mixed)  # (the GPU's settings: the CPU's calls stay fp32)


def _trained_like(seed: int = 0) -> FullGameNet:
    """A small network whose heads' logits are as large as a trained one's (fgself-182: the order's ~20 and up to ~100,
    the point's x and y bins ~25-32, where bf16 rounds in steps of 0.125-0.5) and a value near 0.5."""
    torch.manual_seed(seed)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=40, G=30, d=64, layers=2, memory=True, bins=48).eval()
    net.allowed[:] = True
    net.order_kind.copy_(torch.arange(40) % 3)  # (IMMEDIATE, POINT, UNIT alike)
    with torch.no_grad():
        torch.nn.init.normal_(net.mem_out.weight, std=0.3)
        net.order[2].weight.mul_(8.0)
        net.order[2].bias.normal_(-20.0, 8.0)
        for lin in (net.point_x, net.point_y[2]):
            lin.weight.mul_(4.0)
            lin.bias.normal_(-27.0, 1.5)
        net.key.weight.mul_(3.0)
        net.value_head[2].bias.fill_(0.5)
    return net


def _inputs(B: int = 16, E: int = 24, seed: int = 1, d: int = 64) -> tuple:
    g = torch.Generator().manual_seed(seed)
    ent = torch.randn(B, E, fx.F, generator=g)
    typ, cur = torch.randint(0, 20, (B, E), generator=g), torch.randint(0, 10, (B, E), generator=g)
    n = torch.randint(4, E + 1, (B,), generator=g)
    mask = torch.arange(E)[None] < n[:, None]
    glob = torch.randn(B, 30, generator=g)
    n_own = torch.minimum(n, torch.randint(1, E + 1, (B,), generator=g))
    avail = torch.rand(B, 40, generator=g) < 0.7
    avail[:, 0] = True
    return ent, typ, cur, mask, glob, n_own, avail, torch.randn(B, d, generator=g) * 0.5


def _act(net, x, monkeypatch, actions=None, **kw) -> tuple[dict, list]:
    """model.act on inputs x: its own draws (actions None: returned, in act's order: order, unit, x, y) or these."""
    drawn = []
    real = M._gumbel_argmax

    def draw(logits, generator=None, shared=None):
        got = real(logits, generator, shared) if actions is None else actions[len(drawn)]
        drawn.append(got)
        return got
    monkeypatch.setattr(M, "_gumbel_argmax", draw)
    with torch.no_grad():
        out = M.act(net, *x[:7], x[7], generator=torch.Generator().manual_seed(5), **kw)
    monkeypatch.setattr(M, "_gumbel_argmax", real)
    return out, drawn


def _rows(x) -> list[dict]:
    """Inputs x as view steps (Inference's)."""
    out = []
    for i in range(x[0].shape[0]):
        n = int(x[3][i].sum())
        out.append({"n": n, "n_own": int(x[5][i]), "ent": x[0][i, :n].numpy(), "type": x[1][i, :n].numpy(),
                    "cur": x[2][i, :n].numpy(), "glob": x[4][i].numpy(), "avail": x[6][i].numpy(), "h": x[7][i].numpy()})
    return out


def test_heads_fp32_changes_nothing_without_autocast(monkeypatch):
    """The default path (heads_fp32 off, the default) is the same calls as before; with it on but no autocast, the
    same numbers bit for bit, the draws too (the fp32 inference calls: --infer-precision fp32, the evaluation worker)."""
    net, x = _trained_like(), _inputs()
    base, acts = _act(net, x, monkeypatch)
    for kw in ({"heads_fp32": True}, {"heads_fp32": False}):
        out, again = _act(net, x, monkeypatch, **kw)
        assert all(torch.equal(a, b) for a, b in zip(acts, again))
        for k in ("order", "tgt", "bx", "by", "logp", "value", "entropy", "h"):
            assert torch.equal(out[k], base[k]), k
    # Inference's packed call (the fp32 default: the graphs' function) answers as act() itself, padding zeroed as it does
    inf = Inference(None, torch.device("cpu"), 16, compile_=False, thread=False)
    assert not inf.bf16 and not inf.mixed
    got = inf._finish(inf._launch(net, _rows(x)))
    E = int(x[3].sum(1).max())  # (the call's width: its widest row)
    m = x[3][:, :E]
    want, _ = _act(net, (x[0][:, :E] * m[..., None], x[1][:, :E] * m, x[2][:, :E] * m, m, *x[4:]), monkeypatch)
    assert np.array_equal(np.array([r["value"] for r in got], np.float32), want["value"].numpy())
    assert np.array_equal(np.array([r["entropy"] for r in got], np.float32), want["entropy"].numpy())


def _study():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import infer_cpu_study
    finally:
        sys.path.remove(str(SCRIPTS))
    return infer_cpu_study


def test_bf16_mixed_within_the_measured_bounds(monkeypatch):
    """CUDA's bf16 autocast emulated on the CPU (tensor cores' bf16 products summed in fp32, outputs rounded to bf16;
    layer norms, softmax and sums in fp32; the attention as the memory-efficient kernel). bf16-mixed (the heads in fp32,
    their multiplies TF32) scores the same actions within the CPU study's bounds for it on fgself-182's network (a unit's
    log-probability p99 2.1e-2, max 0.22; the value max 6.8e-3; docs/reference/optimization-log.md), and well inside
    plain bf16's, whose rounding of the large logits it leaves out."""
    study = _study()
    torch.backends.mha.set_fastpath_enabled(False)  # (as under the GPU's autocast: no fused encoder layer)
    try:
        net, x = _trained_like(), _inputs()
        ref, acts = _act(net, x, monkeypatch)
        own = torch.arange(x[0].shape[1])[None] < x[5][:, None]
        err = {}
        for name, kind, mixed in (("bf16", "bf16", False), ("bf16-mixed", "bf16+tf32", True)):
            with study.cudaish(kind):
                out, _ = _act(net, x, monkeypatch, actions=acts, heads_fp32=mixed)
            assert out["logp"].dtype == torch.float32 and torch.isfinite(out["logp"]).all()
            assert torch.isfinite(out["value"]).all() and torch.isfinite(out["h"]).all()
            dlp = (out["logp"] - ref["logp"]).abs()[own].double().numpy()
            dv = (out["value"].float() - ref["value"]).abs().double().numpy()
            err[name] = (np.percentile(dlp, 99), dlp.max(), dv.max())
        lp99, lmax, vmax = err["bf16-mixed"]
        assert lp99 <= 2.1e-2 and lmax <= 0.22 and vmax <= 6.8e-3, err
        b99, bmax, _ = err["bf16"]
        assert lp99 <= 0.25 * b99 and lmax <= 0.5 * bmax, err  # (bf16's rounding of the large logits: gone)
        assert b99 > 5e-2, err  # (and the emulation did round: plain bf16 moved them)
    finally:
        torch.backends.mha.set_fastpath_enabled(True)


def test_bf16_mixed_past_slots_compute_with_their_weights():
    """A past snapshot's slot (PastNet: its weights views of one flat fp32 tensor, a snapshot copied in as one flat copy,
    as the C++ loop copies them) under bf16-mixed: autocast casts the weights at each call (no cache), so after another
    snapshot is copied in the slot answers as that snapshot does, not with stale bf16 copies of the first's. On the CPU,
    the CPU's own autocast stands in for the GPU's (the flags forced: a CPU Inference keeps fp32)."""
    a, b = _trained_like(0), _trained_like(1)
    swap = PastNet(torch.device("cpu"), slots=2)
    inf = Inference(None, torch.device("cpu"), 16, compile_=False, thread=False)
    inf.bf16 = inf.mixed = True  # (before its packed functions are made: _fn reads them)
    plain = Inference(None, torch.device("cpu"), 16, compile_=False, thread=False)
    steps = _rows(_inputs(B=6, E=30, seed=3))

    def answers(infer, net):
        return np.array([[r["value"], r["entropy"]] for r in infer._finish(infer._launch(net, steps))])
    slot = swap.get("a", a, 0)
    assert all(q.dtype == torch.float32 for q in slot.parameters())
    first = answers(inf, slot)
    assert np.array_equal(first, answers(inf, a))  # (the slot's views: the snapshot's own numbers)
    assert np.abs(first - answers(plain, a)).max() > 1e-4  # (and the bf16 path did run)
    assert swap.get("b", b, 0) is slot  # (b into the same slot: one flat copy)
    second = answers(inf, slot)
    assert np.array_equal(second, answers(inf, b)) and np.abs(second - first).max() > 1e-2
    _, mine, buffers, _ = swap.hosts[next(iter(swap.hosts))]
    mine.copy_(pack_flat(a.parameters()))  # (the C++ loop's way: a's packed weights byte for byte, and its buffers)
    for x, y in zip(buffers, a.buffers()):
        x.copy_(y)
    assert np.array_equal(answers(inf, slot), first)
