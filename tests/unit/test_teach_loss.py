"""The heuristic teacher's part of the label loss (selfplay.distill_loss, --teach-coef): a lesson at a point (the hall at
the natural, the camp's attack-move) teaches its place as well as its order and its unit (bug finder 2026-11-10: the
teach term had the order and the pointer only, so the hall's place had --opd-coef's share of l_pt alone)."""

import pytest
import torch

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame.model import FullGameNet, evaluate
from warcraftsim.fullgame.selfplay import distill_loss

B, E, O, BINS = 2, 6, 3, 40
POINT_ORDER = 25  # (a build at a point)


def _net() -> FullGameNet:
    torch.manual_seed(0)
    net = FullGameNet(n_types=8, n_cur=4, n_orders=30, G=12, d=32, layers=1, dropout=0.0, bins=BINS).eval()
    net.allowed[:] = True
    net.order_kind[20:] = fx.POINT
    return net


def _batch(y_x: int, teach: bool) -> tuple[dict, dict, FullGameNet]:
    net = _net()
    g = torch.Generator().manual_seed(1)
    ent = torch.randn(B, E, fx.F, generator=g)
    typ = torch.randint(1, 8, (B, E), generator=g)
    cur = torch.zeros(B, E, dtype=torch.long)
    mask = torch.ones(B, E, dtype=torch.bool)
    glob = torch.randn(B, 12, generator=g)
    n_own = torch.full((B,), O)
    zeros = torch.zeros(B, O, dtype=torch.long)
    ev = evaluate(net, ent, typ, cur, mask, glob, n_own, zeros, zeros, zeros, zeros)
    y_order = torch.zeros(B, O, dtype=torch.long)
    y_order[0, 0] = POINT_ORDER  # (one lesson: a build at a point)
    y_x_t = torch.full((B, O), -1, dtype=torch.long)
    y_y_t = torch.full((B, O), -1, dtype=torch.long)
    y_x_t[0, 0], y_y_t[0, 0] = y_x, 7
    mb = {"own": torch.arange(O)[None].expand(B, O) < n_own[:, None], "y_has": torch.tensor([True, False]),
          "y_order": y_order, "y_ptr": torch.full((B, O), -1, dtype=torch.long), "y_x": y_x_t, "y_y": y_y_t,
          "y_only": torch.tensor([True, False]), "y_teach": (y_order > 0) & teach, "mask": mask, "ent": ent}
    return ev, mb, net


@pytest.mark.parametrize("norm", ["units", "lessons"])
def test_a_point_lesson_teaches_its_place(norm):
    """The teacher's term depends on the lesson's point: a label at another bin is another loss."""
    got = []
    for y_x in (3, 30):
        ev, mb, net = _batch(y_x, teach=True)
        _, _, teach = distill_loss(net, ev, mb, None, teach_scale=10.0, norm=norm)
        got.append(float(teach))
    assert got[0] != pytest.approx(got[1])


@pytest.mark.parametrize("norm", ["units", "lessons"])
def test_the_place_has_the_teach_weight(norm):
    """With the teacher's weight (teach_scale), the point head's gradient from a lesson grows with it, as its order's."""
    grads = []
    for scale in (0.0, 10.0):
        ev, mb, net = _batch(30, teach=True)
        out, _, _ = distill_loss(net, ev, mb, None, teach_scale=scale, norm=norm)
        net.zero_grad()
        out.backward()
        grads.append(float(net.point_x.weight.grad.norm()))
    assert grads[1] > 2.0 * grads[0]


def test_no_lesson_no_teach_term():
    ev, mb, net = _batch(30, teach=False)
    _, _, teach = distill_loss(net, ev, mb, None, teach_scale=10.0, norm="units")
    assert float(teach) == 0.0
