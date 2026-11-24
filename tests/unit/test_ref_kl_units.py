"""--ref-kl-units macro (selfplay.anchor_units): the anchor's KL term over workers and buildings only; "all": as before."""

import numpy as np
import pytest
import torch
from test_update_fuse import make_chunks, make_net, settings, statics, updater

from warcraftsim.fullgame import ordermasks as om
from warcraftsim.fullgame import selfplay


@pytest.fixture(autouse=True)
def all_units():
    selfplay.REF_KL_UNITS = "all"
    yield
    selfplay.REF_KL_UNITS = "all"


def test_anchor_units():
    ent = torch.zeros(1, 5, 64)
    ent[0, 0, om.CH_STRUCT] = 1.0   # a building
    ent[0, 1, om.CH_WORKER] = 1.0   # a worker
    ent[0, 2, om.CH_HERO] = 1.0     # a hero
    own = torch.ones(1, 4)          # (four own units; the fifth entity another's)
    assert torch.equal(selfplay.anchor_units(ent, own), own)
    selfplay.REF_KL_UNITS = "macro"
    assert selfplay.anchor_units(ent, own).tolist() == [[1.0, 1.0, 0.0, 0.0]]


def test_the_loss_with_and_without():
    """In the graphed update: "all" as before; "macro" a smaller anchor term (the army's share left out)."""
    net = make_net()
    ref = make_net()
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.05 * torch.randn_like(p))
    args = settings(ref_kl=0.5, ref_kl_mode="forward-all")
    u = updater(net, args, ref)
    s, _ = u._static(statics(make_chunks(np.random.default_rng(0)), False), None)
    base = u._loss(s, None)[1]["loss/ref_kl"].detach()
    selfplay.REF_KL_UNITS = "macro"
    macro = u._loss(s, None)[1]["loss/ref_kl"].detach()
    assert float(macro) < float(base)
