"""--act-ema (ActEma): the games play a moving average of the learner's weights."""
from pathlib import Path

import torch

from warcraftsim.fullgame.selfplay import ActEma


class Tiny(torch.nn.Module):
    def __init__(self, width: int = 3):
        super().__init__()
        self.config = {"width": width}
        self.lin = torch.nn.Linear(width, width)
        self.register_buffer("table", torch.arange(width))  # (a non-float buffer: copied, not averaged)


def test_half_life_and_copy():
    net = Tiny()
    a = ActEma(net, half_life=100.0)
    assert all(torch.equal(x, y) for x, y in zip(a.act.state_dict().values(), net.state_dict().values()))
    with torch.no_grad():
        net.lin.weight.add_(1.0)
        net.table.add_(5)
    w0 = a.act.lin.weight.clone()
    tau = a.step(100)  # one half-life: half the way
    assert abs(tau - 0.5) < 1e-9
    assert torch.allclose(a.act.lin.weight, w0 + 0.5)
    assert torch.equal(a.act.table, net.table)
    assert 0.0 < a.distance() < 1.0
    a.step(10 ** 9)  # (all the way)
    assert torch.allclose(a.act.lin.weight, net.lin.weight) and a.distance() < 1e-6
    assert not any(q.requires_grad for q in a.act.parameters())


def test_save_and_resume(tmp_path: Path):
    net = Tiny()
    a = ActEma(net, half_life=10.0)
    with torch.no_grad():
        net.lin.weight.add_(2.0)
    a.step(10)
    dest = a.save(tmp_path, {"v": 1}, agent_steps=1234, update=7)
    assert dest == tmp_path / "averaged" / f"ema-{1234:016d}.pt"
    saved = a.act.lin.weight.clone()
    b = ActEma(net, half_life=10.0)  # (from the weights)
    assert not torch.allclose(b.act.lin.weight, saved)
    assert b.resume(tmp_path, f"{1234:016d}.pt") == dest
    assert torch.allclose(b.act.lin.weight, saved)
    assert b.resume(tmp_path, f"{99:016d}.pt") is None and b.resume(tmp_path, None) is None


def test_ratings_name_it():
    from warcraftsim.dashboard.ratings import checkpoint_name
    assert checkpoint_name(f"runs/fgself-267/averaged/ema-{26967036:016d}.pt") == f"fgself-267 avg ema-{26967036:016d}"
