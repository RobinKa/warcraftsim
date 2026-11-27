"""--value-bins (HL-Gauss, Farebrother et al., ICML 2024, "Stop Regressing"): the target histogram, the cross-entropy and
its fit, the expectation every consumer reads (act, evaluate, the update's statistics), the scalar head's conversion
for a fork (model.bin_value) and its Adam state, checkpoints and inference slots, and the off path (no bins) as before.
CPU only: the CUDA-graph updater's loss (GraphedUpdate._loss) runs eagerly here, and dynamo traces it whole."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame.model import (FullGameNet, act, arch_key, bin_value, evaluate, gaussian_logits, hl_gauss_target,
                                        load, split_value, value_ce)
from warcraftsim.fullgame.selfplay import (GraphedUpdate, PastNet, collate_seq, load_optimizer, ppo_update, publish,
                                           save_checkpoint, value_bins_error)

K, LO, HI = 101, -6.0, 6.0
WIDTH = (HI - LO) / K  # (a bin's)
SIGMA = 0.75 * WIDTH  # (the target's, the returns' units)


def edges() -> torch.Tensor:
    return torch.linspace(LO, HI, K + 1)


def centers() -> torch.Tensor:
    e = edges()
    return (e[:-1] + e[1:]) / 2


def spread(t: torch.Tensor) -> torch.Tensor:
    c = centers()
    m = (t * c).sum(-1)
    return ((t * c * c).sum(-1) - m * m).clamp(min=0).sqrt()


def make_net(bins: int = 0, seed: int = 0, **kw) -> FullGameNet:
    torch.manual_seed(seed)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=32, layers=1, memory=True, dropout=0.0,
                      **({"vbins": bins} if bins else {}), **kw)
    net.allowed[:] = True
    net.order_kind[10:20] = fx.UNIT
    net.order_kind[20:] = fx.POINT
    return net


def make_chunks(rng, steps: int = 32, seq: int = 8, ret_scale: float = 1.0) -> list:
    """Steps of one game side as the learner gets them (test_update_fuse's): entity counts change, a few units."""
    out = []
    for t in range(steps):
        n = int(rng.integers(3, 30))
        o = int(rng.integers(1, min(n, 6)))
        out.append({"ent": rng.normal(size=(n, fx.F)).astype(np.float16), "type": rng.integers(0, 20, n).astype(np.int16),
                    "cur": rng.integers(0, 10, n).astype(np.int16), "glob": rng.normal(size=30).astype(np.float32),
                    "n": n, "n_own": o, "logp": (rng.normal(size=o) * 0.1 - 1).astype(np.float32),
                    "adv": float(rng.normal()), "ret": float(rng.normal() * ret_scale), "avail": None,
                    "order": rng.integers(0, 30, o).astype(np.int16), "tgt": rng.integers(0, n, o).astype(np.int16),
                    "bx": rng.integers(0, 50, o).astype(np.int16), "by": rng.integers(0, 50, o).astype(np.int16),
                    "uadv": float(rng.normal()), "h": rng.normal(size=32).astype(np.float32)})
    return [out[a:a + seq] for a in range(0, len(out), seq)]


def settings(**kw) -> types.SimpleNamespace:
    return types.SimpleNamespace(**{**dict(bf16=0, clip=0.2, vf_coef=0.5, ent_coef=0.01, ref_kl=0.1, ref_kl_mode="forward-all",
                                           upgo=1.0, unit_credit=0.0, opd_coef=0.0, opsd_coef=0.0, bc_coef=0.0,
                                           max_grad_norm=1.0, minibatch=16, seq_len=8, pad_groups=1, epochs=1), **kw})


def static(net, ref, chunks, args):
    u = GraphedUpdate(net, ref, torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-5), args, torch.device("cpu"))
    mb = collate_seq(chunks, torch.device("cpu"), width=32, B=len(chunks), T=8, labels=False)
    s, _ = u._static(mb, None)
    return u, s


def evaluate_static(net, s) -> dict:
    return evaluate(net, s["ent"], s["type"], s["cur"], s["mask"], s["glob"], s["n_own"], s["order"], s["tgt"], s["bx"],
                    s["by"], s.get("avail"), seq=(s["B"], s["T"], s["h0"], None))


# the target


def test_target_sums_to_one_and_its_mean_is_the_return():
    y = torch.tensor([-4.0, -1.3, 0.0, 0.37, 2.9, 4.5, 6.0 - 4 * SIGMA])
    t = hl_gauss_target(y, edges(), SIGMA)
    assert t.shape == (7, K) and bool((t >= 0).all())
    assert torch.allclose(t.sum(-1), torch.ones(7), atol=1e-5)
    assert torch.allclose((t * centers()).sum(-1), y, atol=1e-4)
    # its spread: the Gaussian's, and the bins' own (Sheppard: sigma^2 + width^2 / 12)
    assert torch.allclose(spread(t), torch.full((7,), math.sqrt(SIGMA ** 2 + WIDTH ** 2 / 12)), rtol=0.02)
    # a wider sigma, a wider histogram; its mean still the return
    wide = hl_gauss_target(y[1:4], edges(), 6 * SIGMA)
    assert bool((spread(wide) > 4 * spread(t[1:4])).all())
    assert torch.allclose((wide * centers()).sum(-1), y[1:4], atol=1e-4)


def test_target_at_the_edges_is_clipped_and_cut():
    t = hl_gauss_target(torch.tensor([HI, HI + 3.0, LO - 100.0, LO]), edges(), SIGMA)
    assert bool(torch.isfinite(t).all()) and torch.allclose(t.sum(-1), torch.ones(4), atol=1e-5)
    assert torch.equal(t[0], t[1]) and torch.equal(t[2], t[3])  # (beyond the span: as at its edge)
    mean = (t * centers()).sum(-1)
    inward = float(HI - mean[0])  # (the Gaussian cut at the edge, its mass inside: the mean falls in, ~0.8-0.9 sigma)
    assert 0.6 * SIGMA < inward < 1.1 * SIGMA
    assert float(mean[3] - LO) == pytest.approx(inward, abs=1e-5)  # (alike at the bottom)
    assert float(t[0, :-6].sum()) < 1e-4  # (all within the last few bins)


# the loss and the expectation


def head_net(d: int = 16, **kw) -> FullGameNet:
    torch.manual_seed(0)
    return FullGameNet(n_types=4, n_cur=2, n_orders=3, G=8, d=d, layers=1, vbins=K, **kw)


def test_cross_entropy_fits_returns_and_the_expectation_matches():
    net = head_net()
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(256, 16, generator=gen)
    y = (0.9 * x[:, :3].sum(-1)).clamp(-4.5, 4.5)
    opt = torch.optim.Adam(net.value_head.parameters(), lr=3e-3)
    losses = []
    for _ in range(1500):
        loss = value_ce(net, net.value_logits(x), y).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    floor = float(-(hl_gauss_target(y, edges(), SIGMA).clamp(min=1e-30).log() * hl_gauss_target(y, edges(), SIGMA)).sum(-1).mean())
    assert losses[-1] < 0.5 * losses[0] and losses[-1] < floor + 0.6  # (the target's own entropy is the minimum)
    with torch.no_grad():
        v = net.value(x)
    assert float((v - y).abs().mean()) < 0.06 and float(np.corrcoef(v.numpy(), y.numpy())[0, 1]) > 0.995


def test_noisy_returns_give_their_mean_and_a_histogram_with_both_outcomes():
    """A state won 70% and lost 30% of the time (returns +1 and -1): the expectation goes to 0.4 and the histogram has
    both modes, where a squared error's scalar knows the mean alone."""
    net = head_net()
    x = torch.randn(1, 16).expand(400, 16)
    y = torch.where(torch.arange(400) < 280, torch.tensor(1.0), torch.tensor(-1.0))
    opt = torch.optim.Adam(net.value_head.parameters(), lr=3e-3)
    for _ in range(600):
        loss = value_ce(net, net.value_logits(x), y).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.softmax(net.value_logits(x[:1])[0, 0], -1)
        v = float(net.value(x[:1])[0])
    c = centers()
    assert v == pytest.approx(0.4, abs=0.02)
    assert float(p[(c - 1).abs() < 0.3].sum()) == pytest.approx(0.7, abs=0.05)
    assert float(p[(c + 1).abs() < 0.3].sum()) == pytest.approx(0.3, abs=0.05)


def test_act_and_evaluate_read_the_expectation_in_fp32():
    net = make_net(K)
    B, E = 4, 9
    gen = torch.Generator().manual_seed(2)
    x = [torch.randn(B, E, fx.F, generator=gen), torch.randint(0, 20, (B, E), generator=gen),
         torch.randint(0, 10, (B, E), generator=gen), torch.ones(B, E, dtype=torch.bool), torch.randn(B, 30, generator=gen),
         torch.full((B,), 3)]
    with torch.no_grad():
        out = act(net, *x, generator=torch.Generator().manual_seed(3))
        g, _ = net.encode(*x[:5])
        g, _ = net.context(g, None)
        p = torch.softmax(net.value_logits(g, x[4]), -1)
        assert torch.allclose(out["value"], (p * centers()).sum(-1).sum(-1), atol=1e-6)
        assert out["value"].shape == (B,)  # (a scalar per row, as the packed output and the actors' GAE read it)
        ev = evaluate(net, *x, out["order"][:, :3], out["tgt"][:, :3], out["bx"][:, :3], out["by"][:, :3],
                      seq=(B, 1, None, None))
        assert ev["vlogits"].shape == (B, 1, K) and torch.allclose(ev["value"], out["value"], atol=1e-6)
        h = torch.randn(5, 32)
        with torch.autocast("cpu", dtype=torch.bfloat16):  # (the update's region: the bins' head stays fp32)
            v16, l16 = net.value_out(h)
        v32, l32 = net.value_out(h)
    assert l16.dtype == torch.float32 and torch.equal(l16, l32) and torch.equal(v16, v32)


# the conversion of a scalar head (a fork)


def test_gaussian_logits_expectation_is_the_value_inside_the_span():
    v = torch.linspace(-4.5, 4.5, 91, dtype=torch.float64)
    W, B = gaussian_logits(torch.zeros(91, 3, dtype=torch.float64), v, centers().double(), 0.5)
    p = torch.softmax(B.view(91, K).float(), -1)
    err = ((p * centers()).sum(-1) - v.float()).abs()
    assert float(err[v.abs() <= HI - 4 * 0.5].max()) < 1e-4  # (|v| <= vmax - 4 widths; the last centre is half a bin in)
    assert float(err.max()) < 3e-3  # (3 widths from the edge)
    assert bool((spread(p) - 0.5).abs().max() < 0.01)
    _, Bc = gaussian_logits(torch.zeros(1, 3, dtype=torch.float64), torch.tensor([5.8], dtype=torch.float64),
                            centers().double(), 0.5)
    assert float((torch.softmax(Bc.float(), -1) * centers()).sum()) < 5.8 - 0.05  # (at the edge the Gaussian is cut)


@pytest.mark.parametrize("width", [0.5, 0.2])
def test_bin_value_keeps_the_scalar_value_state_by_state(width):
    old = make_net()
    with torch.no_grad():  # (values spread over a few returns, as a trained head's)
        old.value_head[2].weight.mul_(5.0)
        old.value_head[2].bias.fill_(0.3)
    g, glob = torch.randn(200, 32), torch.randn(200, 30)
    v0 = old.value(g, glob).detach()
    assert float(v0.std()) > 0.5
    new = bin_value(old, K, width=width)
    assert new.vbins == K and "vbins" not in old.config
    assert {k: new.config[k] for k in ("vbins", "vmin", "vmax", "vsigma")} == {"vbins": K, "vmin": LO, "vmax": HI,
                                                                              "vsigma": 0.75}
    v1 = new.value(g, glob).detach()
    inside = v0.abs() <= HI - 4 * width
    assert float(inside.float().mean()) > 0.95
    assert float((v1 - v0)[inside].abs().max()) < 1e-4
    ns, os_ = new.state_dict(), old.state_dict()
    assert set(ns) == set(os_)
    for k, v in os_.items():  # (every other weight and buffer as it was)
        if not k.startswith("value_head.2."):
            assert torch.equal(ns[k], v), k
    assert ns["value_head.2.weight"].shape == (K, 32) and ns["value_head.2.bias"].shape == (K,)


def test_bin_value_pulls_on_the_trunk_as_the_squared_error_over_width_squared():
    """For the converted layer the cross-entropy's gradient into the head's input is the squared error's times
    1 / width^2 (--vf-coef 0.5 * width^2 pulls on the trunk as --vf-coef 0.5 did)."""
    old = make_net()
    new = bin_value(old, K, width=0.5)
    g = torch.randn(16, 32, requires_grad=True)
    y = (old.value(g).detach() + 0.05 * torch.randn(16)).clamp(-3, 3)
    mse = torch.autograd.grad((0.5 * (old.value(g) - y) ** 2).sum(), g)[0]
    ce = torch.autograd.grad(value_ce(new, new.value_logits(g), y).sum(), g)[0]
    assert float((ce - mse / 0.25).norm()) < 1e-2 * float((mse / 0.25).norm())


# the update's loss


def test_update_paths_train_the_bins():
    rng = np.random.default_rng(0)
    chunks = make_chunks(rng, ret_scale=2.0)
    chunks[1][2]["ret"] = 9.0  # (beyond the span: clipped, and counted)
    net, ref = make_net(K), make_net()
    for p in ref.parameters():
        p.requires_grad_(False)
    args = settings()
    u, s = static(net, ref, chunks, args)
    net.train()
    loss, out = u._loss(s, None)
    ev = evaluate_static(net, s)
    valid = s["valid"].float()
    want = (value_ce(net, ev["vlogits"], s["ret"]) * valid).sum() / valid.sum()
    assert torch.allclose(out["loss/value"], want, rtol=1e-6)
    assert set(out) >= {"value/mse", "value/width", "value/clipped"}
    assert float(out["value/clipped"]) == pytest.approx(1.0 / float(valid.sum()), rel=1e-5)
    assert float(out["value/mse"]) == pytest.approx(float(0.5 * (((ev["value"] - s["ret"]) ** 2) * valid).sum() / valid.sum()),
                                                    rel=1e-5)
    assert float(out["value/width"]) == pytest.approx(float((spread(torch.softmax(ev["vlogits"][:, 0], -1)) * valid).sum()
                                                            / valid.sum()), rel=1e-5)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.value_head.parameters())
    net.zero_grad(set_to_none=True)
    steps = [st for c in chunks for st in c]
    before = net.value_head[2].weight.detach().clone()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-5)
    res = ppo_update(net, ref, opt, steps, args, False, torch.device("cpu"), chunks)
    assert all(np.isfinite(v) for v in res.values()) and {"value/mse", "value/width", "value/clipped"} <= set(res)
    assert float((net.value_head[2].weight - before).abs().max()) > 0
    warm = ppo_update(net, ref, opt, steps, args, True, torch.device("cpu"), chunks)  # (--value-warmup: the value alone)
    assert np.isfinite(warm["loss/value"])


def test_bins_loss_traced_by_dynamo_whole():
    """The update's loss with bins as GraphedUpdate captures and compiles it: one graph, no break (no data-dependent
    Python, no host syncs), and the traced gradients are backward()'s."""
    rng = np.random.default_rng(1)
    chunks = make_chunks(rng)
    net, ref = make_net(K), make_net()
    args = settings(update_grad_fuse=1)
    u, s = static(net, ref, chunks, args)
    net.train()
    u._find_unused(s, None)
    params = {n: p.detach() for n, p in u._named}
    info = torch._dynamo.explain(u._grads)(params, s, None)
    assert info.graph_count == 1 and info.graph_break_count == 0, info.break_reasons
    net.zero_grad(set_to_none=True)
    loss, _ = u._loss(s, None)
    loss.backward()
    want = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    u._grads = torch.compile(u._grads, backend="eager", dynamic=False)
    net.zero_grad(set_to_none=True)
    u._backward(s, None)
    for n, p in net.named_parameters():
        if n in want:
            assert torch.allclose(p.grad, want[n], rtol=1e-5, atol=1e-7), n


# the off path


def test_off_path_unchanged():
    net = make_net()
    assert net.vbins == 0 and not {"vbins", "vmin", "vmax", "vsigma"} & set(net.config)
    assert [(k, tuple(v.shape)) for k, v in net.state_dict().items() if k.startswith("value_head")] == [
        ("value_head.0.weight", (32, 32)), ("value_head.0.bias", (32,)), ("value_head.2.weight", (1, 32)),
        ("value_head.2.bias", (1,))]
    assert not {"value_edges", "value_centers"} & set(dict(net.named_buffers()))
    torch.manual_seed(0)
    plain = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=32, layers=1, memory=True, dropout=0.0)
    assert all(torch.equal(a, b) for a, b in zip(plain.parameters(), make_net().parameters()))  # (the same draws)
    g = torch.randn(5, 32)
    assert torch.equal(net.value_parts(g), net.value_head(g)) and net.value_logits(g) is None
    rng = np.random.default_rng(2)
    chunks = make_chunks(rng)
    u, s = static(net, None, chunks, settings(ref_kl=0.0))
    net.train()
    _, out = u._loss(s, None)
    ev = evaluate_static(net, s)
    assert "vlogits" not in ev and "value/mse" not in out
    valid = s["valid"].float()
    assert torch.equal(out["loss/value"], 0.5 * (((ev["value"].float() - s["ret"]) ** 2) * valid).sum() / valid.sum().clamp(min=1))


# checkpoints, inference slots, Adam


def test_checkpoints_round_trip(tmp_path):
    scalar = make_net()
    torch.save({"model": scalar.state_dict(), "config": scalar.config}, tmp_path / "scalar.pt")
    old, _ = load(tmp_path / "scalar.pt")  # (an old checkpoint: as it was)
    g = torch.randn(6, 32)
    assert old.vbins == 0 and torch.equal(old.value(g), scalar.value(g))
    binned = bin_value(old, K)
    publish(binned, 3, tmp_path)  # (current.pt: the actors' and the league's)
    back, _ = load(tmp_path / "current.pt")
    assert back.vbins == K and back.value_sigma == pytest.approx(SIGMA)
    assert torch.equal(back.value_logits(g), binned.value_logits(g)) and torch.equal(back.value(g), binned.value(g))
    ck = torch.load(tmp_path / "current.pt", weights_only=False)
    assert not any(k.startswith("value_edges") or k.startswith("value_centers") for k in ck["model"])  # (from the config)
    slot = FullGameNet(**ck["config"])  # (an actor's network and an inference slot: the config, the weights strictly)
    slot.load_state_dict(ck["model"])
    assert torch.equal(slot.value(g), binned.value(g))
    assert bin_value(back, K) is back
    with pytest.raises(ValueError):
        bin_value(back, 51)
    assert arch_key(binned.config) != arch_key(old.config) and arch_key(back.config) == arch_key(binned.config)
    past = PastNet(torch.device("cpu"), slots=1)  # (past snapshots: the binned ones share a slot, the scalar its own)
    other = bin_value(make_net(seed=1), K)
    for key, src in (("a", binned.eval()), ("b", other.eval()), ("s", old.eval()), ("a", binned)):
        host = past.get(key, src, 0)
        with torch.no_grad():
            assert torch.equal(host.value(g), src.value(g)), key
    assert len(past.hosts) == 2


def test_fork_adam_state_for_the_new_head(tmp_path):
    old = make_net()
    opt = torch.optim.Adam(old.parameters(), lr=1e-3)
    for p in old.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    path = tmp_path / f"{5:016d}.pt"
    save_checkpoint(path, old, opt, {}, 5, 5)
    new = bin_value(old, K)
    fresh = torch.optim.Adam(new.parameters(), lr=5e-5)
    assert load_optimizer(fresh, path, "cpu", 5e-5, new)
    for p in new.value_head[2].parameters():  # (the new layer: a fresh Adam's moments, step 0)
        st = fresh.state[p]
        assert st["exp_avg"].shape == p.shape and not bool(st["exp_avg"].any()) and not bool(st["exp_avg_sq"].any())
        assert float(st["step"]) == 0
    olds, news = dict(old.named_parameters()), dict(new.named_parameters())
    for n in ("order.2.weight", "value_head.0.weight"):  # (the others' as saved)
        assert torch.equal(fresh.state[news[n]]["exp_avg"], opt.state[olds[n]]["exp_avg"]), n
    for p in new.parameters():
        p.grad = torch.randn_like(p)
    fresh.step()
    assert all(torch.isfinite(p).all() for p in new.parameters())
    path2 = tmp_path / f"{6:016d}.pt"  # (a binned run's own state on a resume: as saved)
    save_checkpoint(path2, new, fresh, {}, 6, 6)
    again = torch.optim.Adam(new.parameters(), lr=5e-5)
    assert load_optimizer(again, path2, "cpu", 5e-5, new)
    w = new.value_head[2].weight
    assert torch.equal(again.state[w]["exp_avg"], fresh.state[w]["exp_avg"]) and float(again.state[w]["step"]) == 1


def test_flags_and_refusals():
    def ns(**kw):
        return types.SimpleNamespace(**{**dict(value_bins=101, value_min=-6.0, value_max=6.0, value_sigma=0.75,
                                               value_init_width=0.5, reward_parts=0), **kw})
    assert value_bins_error(ns()) is None and value_bins_error(ns(value_bins=0, reward_parts=3, value_max=-9.0)) is None
    for bad in (dict(value_bins=1), dict(value_bins=-2), dict(value_max=-6.0), dict(value_sigma=0.0),
                dict(value_init_width=0.0), dict(reward_parts=3)):
        assert value_bins_error(ns(**bad)), bad
    with pytest.raises(ValueError):
        make_net(1)
    with pytest.raises(ValueError):
        split_value(make_net(K), 3)
