"""A checkpoint's network with more point bins (a larger map's frame: features.map_geometry), playing as
before: the bins it had keep their weights at their place in the wider range (the same positions: 48 world
units a bin), each new outer one starts as the nearest old edge bin's row with its bias lowered by LOW (and
a little more the farther out), so in every state it is far less likely than that edge (a constant low bias
was not: where every old logit was low, the outer bins took the point: builds 10,000 units north and
south); learning raises them. Their x embeddings: the nearest old edge's. Adam's state carried (zero for
the new rows).

    python scripts/grow_bins.py runs/fgself-52/checkpoints/X.pt runs/fullgame/grown/x-bins416.pt --map echofast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame.model import FullGameNet, load  # noqa: E402

LOW = 10.0  # the new bins' logits: this far below the nearest old edge's (e^-10 each)
STEP = 0.02  # and this much lower for each bin farther out


def grow(net: FullGameNet, bins: int) -> tuple[FullGameNet, int]:
    """(the network with `bins` point bins, the old bins' offset in the new range)."""
    old = net.bins
    if bins < old or (bins - old) % 2:
        raise ValueError(f"{old} -> {bins} bins: more, and as many on each side")
    off = (bins - old) // 2
    big = FullGameNet(**{**net.config, "bins": bins}).to(next(net.parameters()).device)
    state = net.state_dict()
    new = big.state_dict()
    with torch.no_grad():
        for k, v in state.items():
            if k in ("point_x.weight", "point_y.2.weight", "point_x.bias", "point_y.2.bias"):
                w = torch.empty_like(new[k])
                w[:off] = v[0]
                w[off + old:] = v[-1]
                w[off:off + old] = v
                if k.endswith("bias"):  # (the outer bins: the edge's logit, lowered)
                    far = torch.arange(1, off + 1, dtype=w.dtype)
                    w[:off] -= LOW + STEP * far.flip(0)
                    w[off + old:] -= LOW + STEP * far[:bins - off - old]
                new[k] = w
            elif k == "x_emb.weight":
                w = torch.empty_like(new[k])
                w[:off] = v[0]
                w[off + old:] = v[-1]
                w[off:off + old] = v
                new[k] = w
            else:
                new[k] = v
    big.load_state_dict(new)
    return big.eval(), off


def grow_adam(state: dict, net: FullGameNet, big: FullGameNet, off: int) -> dict:
    """Adam's state for `big`: each tensor of a grown parameter at the old rows' place, zero elsewhere."""
    names = [n for n, _ in big.named_parameters()]
    old = {n: i for i, (n, _) in enumerate(net.named_parameters())}
    params = dict(big.named_parameters())
    out = {}
    for i, n in enumerate(names):
        s = state["state"].get(old.get(n, -1))
        if s is None:
            continue
        p = params[n]
        if tuple(s["exp_avg"].shape) == tuple(p.shape):
            out[i] = s
            continue
        st = dict(s)
        for key in ("exp_avg", "exp_avg_sq"):
            t = torch.zeros(p.shape, dtype=s[key].dtype)
            t[off:off + s[key].shape[0]] = s[key]
            st[key] = t
        out[i] = st
    return {"state": out, "param_groups": [{**g, "params": list(range(len(names)))} for g in state["param_groups"]]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--map", required=True, help="the map whose frame's bins it gets (e.g. echofast)")
    a = ap.parse_args()
    from warcraftsim.data.mapbuild import stock_map_path
    fx.set_geometry(*fx.map_geometry(stock_map_path(a.map)))
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    net, _ = load(a.src)
    big, off = grow(net, fx.BINS)
    # the same play: the same point logits on the old bins, the new ones far below
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4, 3, net.config["d"], generator=g)
    with torch.no_grad():
        x0, x1 = net.point_x(z), big.point_x(z)
        err = float((x0 - x1[..., off:off + net.bins]).abs().max())
        p_new = float(torch.softmax(x1, -1)[..., :off].sum(-1).max() + torch.softmax(x1, -1)[..., off + net.bins:].sum(-1).max())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ck, "model": {k: v.detach().cpu() for k, v in big.state_dict().items()}, "config": big.config,
                "bins_from": str(a.src)}, a.out)
    a.out.with_suffix(".json").write_text(json.dumps({"from": str(a.src), "how": f"{fx.BINS} point bins"}))  # (the lineage)
    adam = a.src.with_suffix(".adam")
    if adam.exists():
        torch.save(grow_adam(torch.load(adam, map_location="cpu", weights_only=False), net, big, off),
                   a.out.with_suffix(".adam"))
    print(f"{a.out}: {net.bins} -> {fx.BINS} point bins (old ones from {off}; their logits within {err:.1e}, the new "
          f"ones' probability at most {p_new:.1e}){'; Adam state carried' if adam.exists() else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
