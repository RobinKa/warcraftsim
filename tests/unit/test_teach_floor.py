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
    selfplay.TEACH_FLOOR = None
    yield
    selfplay.TEACH_FLOOR = None


def test_floor_classes():
    vocab = {"orders": [[851986, fx.POINT], [ATTACK, fx.POINT], [ATTACK, fx.UNIT], [BLIZZARD, fx.UNIT], [BLIZZARD, fx.POINT]]}
    assert selfplay.floor_classes(vocab).tolist() == [False, False, False, True, True, True]


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
