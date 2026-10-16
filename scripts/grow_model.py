"""A checkpoint's network grown deeper, playing as before: new transformer layers after its last, each
starting as the identity (its attention's output projection and its feed-forward's second layer zero: a
pre-norm layer then adds nothing to its input), and Adam's state (<steps>.adam) carried for the weights
it had, by parameter name (the new layers' starts empty). A run starts from it with selfplay_fork.py
--weights; its anchor (--init) keeps its own architecture.

    python scripts/grow_model.py runs/fgself-41/checkpoints/0000000006414566.pt runs/fullgame/grown/x.pt --layers 5
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


def grow(net: FullGameNet, layers: int) -> FullGameNet:
    """A copy of `net` with `layers` transformer layers, the new ones the identity at first."""
    old = net.config["layers"]
    if layers <= old:
        raise ValueError(f"the network has {old} layers already")
    big = FullGameNet(**{**net.config, "layers": layers}).to(next(net.parameters()).device)
    missing, unexpected = big.load_state_dict(net.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(tuple(f"transformer.layers.{i}." for i in range(old, layers)))
                                  for k in missing), (missing, unexpected)
    with torch.no_grad():
        for i in range(old, layers):
            layer = big.transformer.layers[i]
            for p in (layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias, layer.linear2.weight,
                      layer.linear2.bias):
                p.zero_()
    return big.eval()


def grow_adam(state: dict, net: FullGameNet, big: FullGameNet) -> dict:
    """Adam's state_dict for `net`'s parameters, re-indexed for `big`'s (by name; new ones without state)."""
    old = {n: i for i, (n, _) in enumerate(net.named_parameters())}
    names = [n for n, _ in big.named_parameters()]
    out = {i: state["state"][old[n]] for i, n in enumerate(names) if n in old and old[n] in state["state"]}
    groups = [{**g, "params": list(range(len(names)))} for g in state["param_groups"]]
    assert len(groups) == 1, len(groups)
    return {"state": out, "param_groups": groups}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layers", type=int, required=True, help="the grown network's transformer layers")
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    net, _ = load(a.src)
    big = grow(net, a.layers)
    # the same play: the same outputs on random inputs (the encoder's tokens)
    g = torch.Generator().manual_seed(0)
    N, E = 4, 20
    ent = torch.randn(N, E, fx.F, generator=g)
    typ = torch.randint(0, net.config["n_types"], (N, E), generator=g)
    cur = torch.randint(0, net.config["n_cur"], (N, E), generator=g)
    mask = torch.ones(N, E, dtype=torch.bool)
    mask[:, 15:] = False
    glob = torch.randn(N, net.config["G"], generator=g)
    with torch.no_grad():
        a0, b0 = net.encode(ent, typ, cur, mask, glob)
        a1, b1 = big.encode(ent, typ, cur, mask, glob)
    err = max(float((a0 - a1).abs().max()), float((b0 - b1)[mask].abs().max()))
    assert err < 1e-4, err
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ck, "model": {k: v.detach().cpu() for k, v in big.state_dict().items()}, "config": big.config,
                "grown_from": str(a.src)}, a.out)
    a.out.with_suffix(".json").write_text(json.dumps({"from": str(a.src), "how": f"grown to {a.layers} layers"}))  # (the lineage)
    adam = a.src.with_suffix(".adam")
    if adam.exists():
        torch.save(grow_adam(torch.load(adam, map_location="cpu", weights_only=False), net, big), a.out.with_suffix(".adam"))
    n0, n1 = (sum(p.numel() for p in m.parameters()) for m in (net, big))
    print(f"{a.out}: {net.config['layers']} -> {a.layers} layers, {n0:,} -> {n1:,} weights (encoder outputs within "
          f"{err:.1e}){'; Adam state carried' if adam.exists() else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
