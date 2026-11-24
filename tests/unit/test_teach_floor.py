"""--teach-floor P (selfplay.teach_floored, floor_classes): the labels of attacks on a unit and of the spell lesson's
spells push only while the unit's chance of the order is under P; other labels, and everything without the flag, as
before."""

import math

import numpy as np
import pytest
import torch
from test_update_fuse import make_chunks, make_net, settings, statics, updater

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import selfplay
from warcraftsim.fullgame.teacher import ATTACK

BLIZZARD = 852089


@pytest.fixture(autouse=True)
def no_floor():
    selfplay.TEACH_FLOOR = selfplay.TEACH_FLOOR_PLACE = selfplay.TEACH_FLOOR_POINTER = selfplay.TEACH_FLOOR_LESSONS = None
    yield
    selfplay.TEACH_FLOOR = selfplay.TEACH_FLOOR_PLACE = selfplay.TEACH_FLOOR_POINTER = selfplay.TEACH_FLOOR_LESSONS = None


def test_floor_classes():
    vocab = {"orders": [[851986, fx.POINT], [ATTACK, fx.POINT], [ATTACK, fx.UNIT], [BLIZZARD, fx.UNIT], [BLIZZARD, fx.POINT]]}
    assert selfplay.floor_classes(vocab).tolist() == [False, False, False, True, True, True]


def test_floor_groups():
    code = lambda c: int.from_bytes(c.encode(), "big")
    vocab = {"orders": [[ATTACK, fx.UNIT], [code("htow"), fx.POINT], [code("ugol"), fx.UNIT], [code("unpl"), fx.POINT],
                        [code("hbar"), fx.POINT], [BLIZZARD, fx.UNIT]]}
    assert selfplay.floor_classes(vocab).tolist() == [False, True, False, False, False, False, True]
    assert selfplay.floor_classes(vocab, ("halls",)).tolist() == [False, False, True, True, False, False, False]
    assert selfplay.floor_classes(vocab, ("attack", "halls")).tolist() == [False, True, True, True, False, False, False]
    with pytest.raises(SystemExit):
        selfplay.floor_classes(vocab, ("walls",))
    army = {"orders": [[code("hgry"), fx.IMMEDIATE], [code("hfoo"), fx.IMMEDIATE], [code("hgry"), fx.POINT]]}
    assert selfplay.floor_classes(army, ("army",)).tolist() == [False, True, False, False]
    up = {"orders": [[code("Rhme"), fx.IMMEDIATE], [code("Rhar"), fx.IMMEDIATE], [code("Rhpm"), fx.IMMEDIATE]]}
    assert selfplay.floor_classes(up, ("upgrades",)).tolist() == [False, True, True, False]
    # ("workers": the races' workers' train orders, the early expansion lesson's "a worker": floored on the teacher's own
    # lessons only, floor_lesson_classes, so not among the classes floored whoever labels them)
    w = {"orders": [[code("hpea"), fx.IMMEDIATE], [code("hfoo"), fx.IMMEDIATE], [code("uaco"), fx.IMMEDIATE], [code("ewsp"), fx.POINT],
                    [code("opeo"), fx.IMMEDIATE], [code("ewsp"), fx.IMMEDIATE]]}
    assert selfplay.floor_lesson_classes(w, ("workers",)).tolist() == [False, True, False, True, False, True, True]
    assert selfplay.floor_lesson_classes(w, ("army", "halls", "moves", "harvest")) is None
    assert selfplay.floor_classes(w, ("workers", "army", "halls", "moves", "harvest")).tolist() == [False] * 7


def test_capped_only_for_the_floored_classes():
    lp = torch.log(torch.tensor([0.001, 0.5, 0.001, 0.5]))
    y = torch.tensor([1, 1, 2, 2])
    assert torch.equal(selfplay.teach_floored(lp, y), lp)  # (off)
    selfplay.TEACH_FLOOR = (torch.tensor([False, True, False]), math.log(0.05))
    lp.requires_grad_(True)
    out = selfplay.teach_floored(lp, y)
    o, l = out.detach(), lp.detach()
    assert o[0] == l[0] and float(o[1]) == pytest.approx(math.log(0.05)) and o[2] == l[2] and o[3] == l[3]
    (-out.sum()).backward()
    assert lp.grad.tolist() == [-1.0, 0.0, -1.0, -1.0]  # (no push past the floor)


def test_the_update_runs_with_it_and_a_floor_of_one_changes_nothing():
    net = make_net()
    args = settings(opd_coef=0.05, label_norm="units", teach_coef=0.5)
    chunks = make_chunks(np.random.default_rng(0), labels=True)
    u = updater(net, args)
    s, _ = u._static(statics(chunks, True), None)
    base = u._loss(s, None)[0].detach()
    selfplay.TEACH_FLOOR = (torch.ones(31, dtype=torch.bool), 0.0)  # (P = 1: log-chances are never above it)
    assert torch.equal(u._loss(s, None)[0].detach(), base)
    selfplay.TEACH_FLOOR = (torch.ones(31, dtype=torch.bool), math.log(1e-9))  # (every label capped: no push)
    assert not torch.equal(u._loss(s, None)[0].detach(), base)


def test_floor_place_classes():
    """--teach-floor-orders moves: the attack-move's class alone, and not among the order-level floor's."""
    vocab = {"orders": [[851986, fx.POINT], [ATTACK, fx.POINT], [ATTACK, fx.UNIT], [BLIZZARD, fx.POINT]]}
    assert selfplay.floor_place_classes(vocab, ("spells",)) is None
    assert selfplay.floor_place_classes(vocab, ("spells", "moves")).tolist() == [False, False, True, False, False]
    assert selfplay.floor_classes(vocab, ("moves",)).tolist() == [False] * 5
    assert selfplay.floor_classes(vocab, ("spells", "moves")).tolist() == selfplay.floor_classes(vocab, ("spells",)).tolist()


def test_the_attack_moves_lessons_teach_their_place_alone():
    """--teach-floor-orders moves: no class of it: the loss as before; a class of it: its lessons' order is never
    pushed (the loss differs), and with a floor no place is under, nothing of them is (the loss differs again)."""
    net = make_net()
    args = settings(opd_coef=0.05, label_norm="units", teach_coef=0.5)
    chunks = make_chunks(np.random.default_rng(0), labels=True)
    u = updater(net, args)
    s, _ = u._static(statics(chunks, True), None)
    base = u._loss(s, None)[0].detach()
    selfplay.TEACH_FLOOR_PLACE = (torch.zeros(31, dtype=torch.bool), math.log(0.03))  # (no class of it)
    assert torch.allclose(u._loss(s, None)[0].detach(), base, rtol=1e-6, atol=1e-7)
    selfplay.TEACH_FLOOR_PLACE = (torch.ones(31, dtype=torch.bool), 1.0)  # (every point label: place only, always)
    place_only = u._loss(s, None)[0].detach()
    assert not torch.allclose(place_only, base, rtol=1e-6, atol=1e-7)
    selfplay.TEACH_FLOOR_PLACE = (torch.ones(31, dtype=torch.bool), math.log(1e-30))  # (no place under the floor)
    nothing = u._loss(s, None)[0].detach()
    assert not torch.allclose(nothing, place_only, rtol=1e-6, atol=1e-7)


def test_no_gradient_to_the_order_from_an_attack_moves_lesson():
    """The order's logits get no gradient from the labelled rows of the floored class (distill_loss)."""
    net = make_net()
    args = settings(opd_coef=0.05, label_norm="units", teach_coef=0.5)
    chunks = make_chunks(np.random.default_rng(0), labels=True)
    u = updater(net, args)
    s, _ = u._static(statics(chunks, True), None)
    selfplay.TEACH_FLOOR_PLACE = (torch.ones(31, dtype=torch.bool), math.log(1e-30))
    loss = u._loss(s, None)[0]
    assert torch.isfinite(loss)


# --teach-floor-orders harvest: a harvest lesson (the expansion lesson's workers) teaches its mine alone, under a floor

HARVEST = 852018


def test_floor_pointer_classes():
    """The harvest order on a unit alone (not a tree's), and not among the order-level floor's or the place's."""
    vocab = {"orders": [[851986, fx.POINT], [HARVEST, fx.UNIT], [HARVEST, fx.TREE], [ATTACK, fx.POINT], [851971, fx.UNIT]]}
    assert selfplay.floor_pointer_classes(vocab, ("spells", "moves")) is None
    assert selfplay.floor_pointer_classes(vocab, ("moves", "harvest")).tolist() == [False, False, True, False, False, False]
    assert selfplay.floor_classes(vocab, ("harvest",)).tolist() == [False] * 6
    assert selfplay.floor_place_classes(vocab, ("harvest",)) is None
    assert selfplay.floor_place_classes(vocab, ("moves", "harvest")).tolist() == [False, False, False, False, True, False]


def test_the_update_with_a_harvest_floor():
    """No class of it: the loss as before; every unit-target label of it: another loss (the order no longer pushed), and
    with a floor no pointer is under, another again (nothing of them pushed)."""
    net = make_net()
    args = settings(opd_coef=0.05, label_norm="units", teach_coef=0.5)
    chunks = make_chunks(np.random.default_rng(0), labels=True)
    u = updater(net, args)
    s, _ = u._static(statics(chunks, True), None)
    base = u._loss(s, None)[0].detach()
    selfplay.TEACH_FLOOR_POINTER = (torch.zeros(31, dtype=torch.bool), math.log(0.25))  # (no class of it)
    assert torch.equal(u._loss(s, None)[0].detach(), base)
    selfplay.TEACH_FLOOR_POINTER = (torch.ones(31, dtype=torch.bool), 1.0)  # (every pointer label: the pointer only, always)
    pointer_only = u._loss(s, None)[0].detach()
    assert not torch.allclose(pointer_only, base, rtol=1e-6, atol=1e-7)
    selfplay.TEACH_FLOOR_POINTER = (torch.ones(31, dtype=torch.bool), math.log(1e-30))  # (no pointer under the floor)
    nothing = u._loss(s, None)[0].detach()
    assert not torch.allclose(nothing, pointer_only, rtol=1e-6, atol=1e-7)


UNIT_ORDER = 12  # (an order on a unit in _lesson's network: the harvest's stand-in)


def _lesson(teach: bool = True, order: int = UNIT_ORDER, ptr: int | None = 4):
    """One labelled unit in a batch of two steps: an order on a unit with its pointer (test_teach_loss's batch), or
    another order without one."""
    from warcraftsim.fullgame.model import FullGameNet, evaluate
    B, E, O = 2, 6, 3
    torch.manual_seed(0)
    net = FullGameNet(n_types=8, n_cur=4, n_orders=30, G=12, d=32, layers=1, dropout=0.0, bins=40).eval()
    net.allowed[:] = True
    net.order_kind[10:20] = fx.UNIT
    net.order_kind[20:] = fx.POINT
    g = torch.Generator().manual_seed(1)
    ent = torch.randn(B, E, fx.F, generator=g)
    typ = torch.randint(1, 8, (B, E), generator=g)
    mask = torch.ones(B, E, dtype=torch.bool)
    n_own = torch.full((B,), O)
    zeros = torch.zeros(B, O, dtype=torch.long)
    ev = evaluate(net, ent, typ, torch.zeros(B, E, dtype=torch.long), mask, torch.randn(B, 12, generator=g), n_own, zeros, zeros,
                  zeros, zeros)
    ev["logits"].retain_grad()
    y_order = torch.zeros(B, O, dtype=torch.long)
    y_order[0, 0] = order
    y_ptr = torch.full((B, O), -1, dtype=torch.long)
    if ptr is not None:
        y_ptr[0, 0] = ptr
    none = torch.full((B, O), -1, dtype=torch.long)
    mb = {"own": torch.arange(O)[None].expand(B, O) < n_own[:, None], "y_has": torch.tensor([True, False]), "y_order": y_order,
          "y_ptr": y_ptr, "y_x": none, "y_y": none.clone(), "y_only": torch.tensor([True, False]),
          "y_teach": (y_order > 0) & teach, "mask": mask, "ent": ent}
    return ev, mb, net


def _floor(on: bool, p: float):
    m = torch.zeros(31, dtype=torch.bool)
    m[UNIT_ORDER] = on
    return m, math.log(p)


@pytest.mark.parametrize("norm", ["units", "lessons"])
@pytest.mark.parametrize("teach", [True, False])
def test_a_harvest_lesson_teaches_its_mine_alone(norm, teach):
    """The labelled order's logits get no gradient from a lesson of the floored class; the pointer's head does while the
    mine's chance given the order is under the floor, and nothing does once it is over. Off, or another class: as before."""
    def grads(floor):
        selfplay.TEACH_FLOOR_POINTER = floor
        ev, mb, net = _lesson(teach)
        out, _, t = selfplay.distill_loss(net, ev, mb, None, teach_scale=10.0, norm=norm)
        net.zero_grad()
        if out.requires_grad:
            out.backward()
        g_order = ev["logits"].grad
        g_ptr = net.query.weight.grad
        return (float(out.detach()), float(t), 0.0 if g_order is None else float(g_order.abs().sum()),
                0.0 if g_ptr is None else float(g_ptr.abs().sum()))
    off = grads(None)
    assert off[2] > 0 and off[3] > 0  # (as before: the order and the pointer pushed)
    assert grads(_floor(False, 0.25)) == off  # (another class floored: the same numbers)
    under = grads(_floor(True, 1.0))  # (the mine's chance is under 1: the pointer alone)
    assert under[2] == 0.0 and under[3] > 0 and under[0] < off[0]
    over = grads(_floor(True, 1e-30))  # (over the floor: nothing)
    assert over == (0.0, 0.0, 0.0, 0.0)
    if teach:
        assert under[1] > 0  # (the teacher's term is the pointer's)


def test_the_workers_floor_is_on_the_teacher_s_lessons_only():
    """--teach-floor-orders workers: a teacher's lesson of a floored class pushes its order only under the floor; a
    builder's label of the same class (no "y_teach") as before."""
    def push(teach, floor):
        selfplay.TEACH_FLOOR_LESSONS = floor
        ev, mb, net = _lesson(teach, order=5, ptr=None)
        out, _, _ = selfplay.distill_loss(net, ev, mb, None, teach_scale=10.0, norm="units")
        if out.requires_grad:
            out.backward()
        g = ev["logits"].grad
        return 0.0 if g is None else float(g.abs().sum())
    m = torch.zeros(31, dtype=torch.bool)
    m[5] = True
    over = (m, math.log(1e-30))  # (any chance is over it)
    try:
        assert push(True, None) > 0 and push(False, None) > 0
        assert push(True, over) == 0.0  # (the teacher's lesson: no push over the floor)
        assert push(False, over) == push(False, None)  # (a builder's label: as before)
        assert push(True, (m, 0.0)) == push(True, None)  # (a floor of 1: every chance is under it)
        other = torch.zeros(31, dtype=torch.bool)
        other[6] = True
        assert push(True, (other, math.log(1e-30))) == push(True, None)  # (another class)
    finally:
        selfplay.TEACH_FLOOR_LESSONS = None


def test_the_harvest_floor_is_the_pointer_s_chance_given_the_order():
    """Pushed exactly while log P(mine | harvest) is under the floor."""
    ev, mb, net = _lesson()
    with torch.no_grad():
        ptr, _, _ = net.target_logits(ev["g"], ev["u"], mb["mask"], mb["y_order"], mb["ent"], None)
        p = float(torch.softmax(ptr[0, 0].float(), -1)[4])
    for floor, pushed in ((p * 1.05, True), (p * 0.95, False)):
        selfplay.TEACH_FLOOR_POINTER = _floor(True, floor)
        ev, mb, net = _lesson()
        out, _, _ = selfplay.distill_loss(net, ev, mb, None, teach_scale=10.0, norm="units")
        assert (float(out.detach()) > 0) is pushed
