"""--fine-points: point indices (bin * S + sub-cell) through features, the model, the inference server's packing and
the losses; a network without a fine head plays its bins' centres exactly as before."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import features as fx

ECHO = (((-5184.0, 2944.0), (4672.0, 2944.0)), 2.75)  # (Echo Isles' starts and bins' reach: map_geometry)


@pytest.fixture
def frame():
    """Echo Isles' frame (its x squeezed between the bases) and FINE, restored afterwards (and their environment)."""
    saved = (fx.STARTS, fx.MAP_CX, fx.BIN_RANGE, fx.BINS, fx.FINE)
    env = {k: os.environ.get(k) for k in ("WARCRAFTSIM_MAP_GEOMETRY", "WARCRAFTSIM_FINE_POINTS")}
    fx.set_geometry(*ECHO)
    try:
        yield
    finally:
        fx.STARTS, fx.MAP_CX, fx.BIN_RANGE, fx.BINS, fx.FINE = saved
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _net(seed: int = 0, bins: int = 352, **kw):
    from warcraftsim.fullgame.model import FullGameNet
    torch.manual_seed(seed)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=64, layers=1, bins=bins, **kw).eval()
    net.allowed[:] = True
    net.order_kind.copy_(torch.randint(0, 5, (30,)))
    net.order_kind[0] = fx.IMMEDIATE
    net.order_kind[1:6] = fx.POINT  # (some orders at a point, for sure)
    return net


def _inputs(seed: int = 1, N: int = 3, E: int = 12):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N, E, fx.F, generator=g), torch.randint(0, 20, (N, E), generator=g),
            torch.randint(0, 10, (N, E), generator=g), torch.ones(N, E, dtype=torch.bool), torch.randn(N, 30, generator=g),
            torch.tensor([5, 3, E][:N])]


def _trained(net):
    """The fine head as if trained: its last layer at random (the prior alone draws nearly always the middle)."""
    with torch.no_grad():
        net.point_f[2].weight.normal_(0, 0.3)
        net.point_f[2].bias.normal_(0, 0.5)
    return net


def test_world_points_round_trip_within_a_sub_cell(frame):
    """A label's point -> its index (bin and sub-cell) -> the point: within half a sub-cell (48 / S / 2 world units
    near the bases and in y, ~206 / S / 2 in x between the bases); its bin is the bin without --fine-points."""
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-6000, 5500, 4000), rng.uniform(-5400, 5400, 4000)
    for S in (7, 9):
        for sign in (1.0, -1.0):
            fx.set_fine(1)
            cx, cy = fx.world_bins(x, y, sign)
            fx.set_fine(S)
            bx, by = fx.world_bins(x, y, sign)
            assert np.array_equal(bx // S, cx) and np.array_equal(by // S, cy)
            assert bx.max() < 32767 and bx.min() >= 0
            back = np.array([fx.bins_world(int(a), int(b), sign) for a, b in zip(bx, by)])
            ox = fx.own_start(sign)[0]
            near = np.abs(x - ox) < fx.NEAR - 1
            assert np.abs(back[:, 1] - y).max() <= 48 / S / 2 + 1e-6
            assert np.abs(back[near, 0] - x[near]).max() <= 48 / S / 2 + 1e-6
            assert np.abs(back[:, 0] - x).max() <= 205.7 / S / 2 + 1e-6
            assert np.allclose(fx.bin_center(bx), fx.frame_xy(back[:, 0], back[:, 1], sign)[0])


def test_middle_sub_cell_is_the_old_point_exactly(frame):
    """A network without a fine head plays index bin * S + S // 2 (both axes): the old point, bit for bit."""
    for sign in (1.0, -1.0):
        fx.set_fine(1)
        old = [fx.bins_world(b, (b * 7) % fx.BINS, sign) for b in range(fx.BINS)]
        fx.set_fine(7)
        new = [fx.bins_world(b * 7 + 3, ((b * 7) % fx.BINS) * 7 + 3, sign) for b in range(fx.BINS)]
        assert old == new


def test_without_the_flag_a_fine_head_changes_nothing(frame):
    """FINE 1: a network with a fine head acts (the same draws) and evaluates as the one it grew from."""
    from warcraftsim.fullgame.model import act, evaluate, grow_fine
    fx.set_fine(1)
    net = _net()
    big = _trained(grow_fine(net, 7))
    x = _inputs()
    with torch.no_grad():
        a = act(net, *x, generator=torch.Generator().manual_seed(5))
        b = act(big, *x, generator=torch.Generator().manual_seed(5))
        for k in ("order", "tgt", "bx", "by", "logp", "value"):
            assert torch.equal(a[k], b[k]), k
        assert int(a["bx"].max()) < fx.BINS
        ea = evaluate(net, *x, a["order"], a["tgt"], a["bx"], a["by"])
        eb = evaluate(big, *x, a["order"], a["tgt"], a["bx"], a["by"])
    assert torch.equal(ea["logp"], eb["logp"]) and "fl" not in eb


def test_grown_network_keeps_the_bins_and_starts_at_the_prior(frame):
    """The grown network's bins' logits are the old ones; the sub-cell's are the prior alone (the middle with
    probability --fine-centre at the target temperature); an old network under FINE plays the bins' centres."""
    from warcraftsim.fullgame.model import act, grow_fine
    fx.set_fine(7)
    net = _net(target_temperature=0.8)
    big = grow_fine(net, 7, centre=0.9)
    assert big.config["fine"] == 7 and "fine" not in net.config
    x = _inputs()
    with torch.no_grad():
        g, u = big.encode(*x[:5])
        order = torch.randint(1, 6, (3, 12))
        _, xl0, z0 = net.target_logits(g, u, x[3], order)
        _, xl1, z1 = big.target_logits(g, u, x[3], order)
        assert torch.equal(xl0, xl1)
        cx = xl1.argmax(-1)
        assert torch.equal(net.y_logits(z0, cx), big.y_logits(z1, cx))
        p = torch.softmax(big.fine_logits(z1, cx, cx.flip(-1)).float(), -1)
        assert torch.allclose(p[..., 24], torch.full_like(p[..., 24], 0.9), atol=1e-5)
        assert torch.allclose(p[..., :24], torch.full_like(p[..., :24], 0.1 / 48), atol=1e-6)
        # the old network under FINE: its draws as without, at the middle sub-cell
        a = act(net, *x, generator=torch.Generator().manual_seed(3))
        fx.set_fine(1)
        b = act(net, *x, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a["bx"], b["bx"] * 7 + 3) and torch.equal(a["by"], b["by"] * 7 + 3)
    assert torch.equal(a["logp"], b["logp"]) and torch.equal(a["order"], b["order"])


def test_act_and_evaluate_log_probs_agree_with_fine_points(frame):
    from warcraftsim.fullgame.model import act, evaluate, grow_fine
    fx.set_fine(7)
    net = _trained(grow_fine(_net(), 7))
    x = _inputs()
    with torch.no_grad():
        seen = set()
        for s in range(8):
            a = act(net, *x, generator=torch.Generator().manual_seed(s))
            ev = evaluate(net, *x, a["order"], a["tgt"], a["bx"], a["by"])
            own = torch.arange(a["order"].shape[1])[None] < x[5][:, None]
            assert float(((ev["logp"] - a["logp"]) * own).abs().max()) < 1e-5
            assert "fl" in ev and int(a["bx"].max()) < fx.BINS * 7
            seen |= set((a["bx"] % 7).flatten().tolist())
    assert len(seen) > 3  # (the trained head draws many sub-cells)


def test_inference_server_packing_carries_point_indices(frame):
    """The inference server's packed call (the function its CUDA graphs and the C++ loop run; its output's columns
    copied as they are): each unit's log-probability is evaluate()'s for the actions it answers."""
    from warcraftsim.fullgame.model import evaluate, grow_fine
    from warcraftsim.fullgame.selfplay import Inference
    fx.set_fine(7)
    net = _trained(grow_fine(_net(), 7))
    direct = Inference(None, torch.device("cpu"), 8, compile_=False, thread=False)
    rng = np.random.default_rng(0)
    sts = []
    for n in (5, 9, 12):
        sts.append({"n": n, "n_own": min(4, n), "ent": rng.normal(size=(n, fx.F)).astype(np.float32),
                    "type": rng.integers(1, 20, n), "cur": rng.integers(0, 10, n),
                    "glob": rng.normal(size=30).astype(np.float32), "avail": None, "h": None})
    res = direct._finish(direct._launch(net, sts))
    with torch.no_grad():
        for st, r in zip(sts, res):
            n, o = st["n"], len(r["order"])
            x = [torch.as_tensor(st["ent"])[None], torch.as_tensor(st["type"])[None], torch.as_tensor(st["cur"])[None],
                 torch.ones(1, n, dtype=torch.bool), torch.as_tensor(st["glob"])[None], torch.tensor([st["n_own"]])]
            pad = lambda v: torch.as_tensor(np.pad(v, (0, n - o)))[None]  # noqa: E731
            ev = evaluate(net, *x, pad(r["order"]), pad(r["tgt"]), pad(r["bx"]), pad(r["by"]))
            assert np.abs(ev["logp"][0, :o].numpy() - r["logp"]).max() < 1e-4
            assert int(r["bx"].max()) < fx.BINS * 7


def test_labels_split_into_bins_and_sub_cell_and_train_the_fine_head(frame):
    from warcraftsim.fullgame.bc import _masked_ce
    from warcraftsim.fullgame.model import grow_fine, point_parts
    fx.set_fine(7)
    y_x, y_y = torch.tensor([[-1, 7 * 100 + 6, 3]]), torch.tensor([[-1, 7 * 50 + 1, 7 * 2]])
    cx, cy, f = point_parts(y_x, y_y)
    assert cx.tolist() == [[-1, 100, 0]] and cy.tolist() == [[-1, 50, 2]] and f.tolist() == [[-1, 6 * 7 + 1, 3 * 7]]
    net = grow_fine(_net(), 7).train()
    opt = torch.optim.Adam([p for n, p in net.named_parameters() if n.startswith(("fine_", "point_f."))], lr=1e-2)
    z = torch.randn(1, 3, 64)
    for _ in range(60):
        loss = _masked_ce(net.fine_logits(z, cx, cy), f, torch.ones(1, 3))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        assert net.fine_logits(z, cx, cy)[0, 1:].argmax(-1).tolist() == f[0, 1:].tolist()


def test_anchor_kl_takes_the_sub_cell_only_from_an_anchor_with_a_fine_head(frame):
    from warcraftsim.fullgame.model import act, anchor_kl, evaluate, grow_fine
    fx.set_fine(7)
    old = _net()
    net = _trained(grow_fine(old, 7))
    x = _inputs()
    with torch.no_grad():
        a = act(net, *x, generator=torch.Generator().manual_seed(1))
        args = (a["order"], a["tgt"], a["bx"], a["by"])
        ev, ev_old = evaluate(net, *x, *args), evaluate(old, *x, *args)
        ev_ref = evaluate(grow_fine(old, 7), *x, *args)
        assert "fl" not in ev_old and "fl" in ev_ref
        base = anchor_kl({k: v for k, v in ev.items() if k != "fl"}, ev_old, "forward-all")
        assert torch.equal(anchor_kl(ev, ev_old, "forward-all"), base)
        assert bool((anchor_kl(ev, ev_ref, "forward-all") >= base - 1e-6).all())
        assert float((anchor_kl(ev, ev_ref, "forward-all") - base).abs().max()) > 1e-3


def test_checkpoint_round_trip_keeps_the_fine_head(frame, tmp_path):
    from warcraftsim.fullgame.model import grow_fine, load
    from warcraftsim.fullgame.selfplay import publish
    fx.set_fine(7)
    net = _trained(grow_fine(_net(), 7))
    publish(net, 1, tmp_path)
    back, _ = load(tmp_path / "current.pt")
    assert back.fine == 7 and torch.equal(back.point_f[2].weight, net.point_f[2].weight)
