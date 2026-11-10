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


def _y(orders, x=-1):
    import numpy as np
    n = len(orders)
    return {"y_order": np.asarray(orders, np.int16), "y_ptr": np.full(n, -1, np.int16),
            "y_x": np.full(n, x, np.int16), "y_y": np.full(n, x, np.int16)}


def test_a_later_label_on_a_lesson_s_unit_is_no_lesson():
    """merge_labels: the builder's label after the teacher's on the same unit replaces it, and with it the teacher's
    weight (a building's mix lesson, then its next target item: the item had kept --teach-coef)."""
    from warcraftsim.fullgame.selfplay import merge_labels
    step = {"n_own": 3}
    merge_labels(step, _y([5, 0, 6]), teach=True)  # (lessons for units 0 and 2)
    assert step["y_teach"].tolist() == [True, False, True]
    merge_labels(step, _y([7, 8, 0], x=4))  # (the builder: units 0 and 1)
    assert step["y_order"].tolist() == [7, 8, 6]
    assert step["y_teach"].tolist() == [False, False, True]  # (unit 2's lesson stays)
    merge_labels(step, _y([9, 0, 0]), teach=True)  # (a lesson again on unit 0)
    assert step["y_order"].tolist() == [9, 8, 6] and step["y_teach"].tolist() == [True, False, True]


def test_no_lesson_no_teach_term():
    ev, mb, net = _batch(30, teach=False)
    _, _, teach = distill_loss(net, ev, mb, None, teach_scale=10.0, norm="units")
    assert float(teach) == 0.0


@pytest.mark.parametrize("norm", ["units", "lessons"])
def test_teach_point_zero_leaves_the_place_out(norm, monkeypatch):
    """--teach-point 0: the lesson's place is not in the teacher's term (as before 2026-11-10): a label at another bin is
    the same loss; with it, another."""
    from warcraftsim.fullgame import selfplay
    for w, differs in ((0.5, True), (0.0, False)):
        monkeypatch.setattr(selfplay, "TEACH_POINT", w)
        got = []
        for y_x in (3, 30):
            ev, mb, net = _batch(y_x, teach=True)
            with torch.no_grad():
                _, _, teach = selfplay.distill_loss(net, ev, mb, None, teach_scale=10.0, norm=norm)
            got.append(float(teach))
        assert (got[0] != pytest.approx(got[1])) is differs
