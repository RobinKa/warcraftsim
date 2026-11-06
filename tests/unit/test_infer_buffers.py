"""The inference calls' pinned host buffers (selfplay.Inference._launch): a round launches several calls before it
waits (launch_round: the current network's and each past snapshot's), and each call's input and answers must have a
host buffer of their own. Reproduced on the CPU: the CUDA parts (pinned memory, the graphs) stood in for."""

import types

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import selfplay

G, NO = 6, 5


class _Graph:
    """A captured call's stand-in: replay() computes the packed answers from the static input (value = the row's first
    global number, entropy = its second), as the graph would on the GPU."""

    def __init__(self, x: torch.Tensor, out: torch.Tensor, E: int):
        self.x, self.out, self.E = x, out, E

    def replay(self):
        O = min(fx.MAX_OWN, self.E)
        g0 = self.E * fx.F
        self.out.zero_()
        self.out[:, 5 * O] = self.x[:, g0]
        self.out[:, 5 * O + 1] = self.x[:, g0 + 1]


def _infer(monkeypatch):
    """An Inference that takes the pinned path on the CPU (a device that says "cuda"; pinning a no-op)."""
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    dev = types.SimpleNamespace(type="cuda")
    inf = selfplay.Inference(None, dev, 16, True, buckets=(4, 16), thread=False, entities=(32, 48, 64, 96))
    graphs = {}

    def graph(net, host):  # (per network and shape, as Inference._graph: a static input and output)
        if (id(net), host.shape) not in graphs:
            E = next(e for e in inf.entities if host.shape[1] == e * (fx.F + 3) + G + 1 + NO)
            x, out = torch.zeros(host.shape), torch.zeros(host.shape[0], 5 * min(fx.MAX_OWN, E) + 2)
            graphs[(id(net), host.shape)] = (x, _Graph(x, out, E), out)
        return graphs[(id(net), host.shape)]
    monkeypatch.setattr(inf, "_graph", graph)
    return inf


def _net():
    return types.SimpleNamespace(config={"G": G, "n_orders": NO, "d": 0}, memory=False)


def _step(n: int, value: float) -> dict:
    glob = np.zeros(G, np.float32)
    glob[0], glob[1] = value, -value
    return {"n": n, "n_own": 1, "ent": np.zeros((n, fx.F), np.float32), "type": np.zeros(n, np.int64),
            "cur": np.zeros(n, np.int64), "glob": glob, "avail": None, "h": None}


def test_each_call_of_a_round_reads_its_own_answers(monkeypatch):
    """Two past snapshots' calls in one round, the same row bucket (4), one padded to 96 entities and one to 160
    (MAX_ENT, always a bucket): both answer 5 x min(MAX_OWN = 96, E) + 2 numbers a row, and their answers had shared one
    pinned buffer (keyed by that width, the input's slot counted per input width): the round's last copy won, and the
    first call read the second's rows (fgself-194's bench: rows off by up to 2.7, even fp32 against fp32)."""
    inf = _infer(monkeypatch)
    a, b = _net(), _net()
    inf.begin(0)
    h1 = inf._launch(a, [_step(80, 1.0), _step(70, 2.0)])  # (widest 80: E = 96)
    h2 = inf._launch(b, [_step(150, 7.0)])  # (widest 150: E = 160)
    inf.wait()
    first, second = inf._finish(h1), inf._finish(h2)
    assert [r["value"] for r in first] == [1.0, 2.0] and [r["entropy"] for r in first] == [-1.0, -2.0]
    assert [r["value"] for r in second] == [7.0]
    assert h1[0] is not h2[0]


def test_every_call_of_a_round_has_its_own_buffers(monkeypatch):
    """Any mix of a round's calls (row buckets, entity buckets, repeats): no two share an input or an answer buffer; a
    new round (after wait) takes them again."""
    inf = _infer(monkeypatch)
    rng = np.random.default_rng(0)
    for r in range(20):
        inf.begin(r % 2)
        handles, want = [], []
        for c in range(int(rng.integers(2, 9))):
            rows = [_step(int(rng.integers(1, fx.MAX_ENT + 1)), float(1000 * r + 10 * c + i))
                    for i in range(int(rng.integers(1, 17)))]
            handles.append(inf._launch(_net(), rows))
            want.append([st["glob"][0] for st in rows])
        assert len({id(h[0]) for h in handles}) == len(handles)
        inf.wait()
        for h, w in zip(handles, want):
            assert [x["value"] for x in inf._finish(h)] == pytest.approx(w)
