"""The inference calls' speed options, on the CPU: --infer-rows (finer row buckets) pads a call with more or fewer rows,
and a row's answer does not depend on the rows beside it; --infer-precision tf32 sets TF32 only while the inference
calls are compiled and captured, and puts the process's setting back exactly; both are off by default."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame.model import FullGameNet
from warcraftsim.fullgame.selfplay import INFER_PRECISIONS, INFER_ROWS, Inference, infer_rows, matmul_precision


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
