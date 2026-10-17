"""A checkpoint's network with the strategy block in its global input (strategy conditioning: features.ZDIM,
fullgame/strategy.py), playing as before: the global layer's new input columns zero, so any strategy and
none give the same outputs until training uses them; Adam's state carried, the new columns' moments zero.
A strategy run starts from it (selfplay_fork.py --weights), and its anchor and evaluation checkpoints must
be grown too: a run's rows have one width, its networks' (the inference loop packs every network's rows
alike).

    python scripts/grow_strategy.py runs/fgself-73/checkpoints/0000000008386478.pt runs/fullgame/grown/fgself-73-z.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame import strategy  # noqa: E402
from warcraftsim.fullgame.model import FullGameNet, load  # noqa: E402


def grow(net: FullGameNet, zdim: int) -> FullGameNet:
    """A copy of `net` reading `zdim` more global numbers (the strategy block) with zero weights."""
    if net.config.get("z"):
        raise ValueError(f"the network has a strategy block already ({net.config['z']})")
    big = FullGameNet(**{**net.config, "G": net.config["G"] + zdim, "z": zdim}).to(next(net.parameters()).device)
    state = dict(net.state_dict())
    w = state["glob.0.weight"]  # [d, what the player knows]: the strategy's columns after it
    state["glob.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], zdim)], 1)
    big.load_state_dict(state)
    return big.eval()


def grow_adam(state: dict, net: FullGameNet, zdim: int) -> dict:
    """Adam's state_dict with the global layer's moments widened by zero columns."""
    i = [n for n, _ in net.named_parameters()].index("glob.0.weight")
    if i not in state["state"]:
        return state
    st = dict(state["state"][i])  # (a copy: an optimizer's state_dict holds its live tensors)
    for k in ("exp_avg", "exp_avg_sq"):
        if k in st:
            st[k] = torch.cat([st[k], st[k].new_zeros(st[k].shape[0], zdim)], 1)
    return {**state, "state": {**state["state"], i: st}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--zdim", type=int, default=strategy.ZDIM)
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    net, _ = load(a.src)
    big = grow(net, a.zdim)
    # the same play: the same outputs on random inputs, with no strategy and with a random one
    g = torch.Generator().manual_seed(0)
    N, E = 4, 20
    ent = torch.randn(N, E, fx.F, generator=g)
    typ = torch.randint(0, net.config["n_types"], (N, E), generator=g)
    cur = torch.randint(0, net.config["n_cur"], (N, E), generator=g)
    mask = torch.ones(N, E, dtype=torch.bool)
    mask[:, 15:] = False
    glob = torch.randn(N, net.config["G"], generator=g)
    G0 = net.config["G"] - net.config["priv"]
    errs = []
    with torch.no_grad():
        a0, b0 = net.encode(ent, typ, cur, mask, glob)
        v0 = net.value(a0, glob)
        for z in (torch.zeros(N, a.zdim), torch.rand(N, a.zdim, generator=g)):
            wide = torch.cat([glob[:, :G0], z, glob[:, G0:]], 1)
            a1, b1 = big.encode(ent, typ, cur, mask, wide)
            errs += [float((a0 - a1).abs().max()), float((b0 - b1)[mask].abs().max()),
                     float((v0 - big.value(a1, wide)).abs().max())]
    err = max(errs)
    assert err < 1e-5, errs
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ck, "model": {k: v.detach().cpu() for k, v in big.state_dict().items()}, "config": big.config,
                "grown_from": str(a.src)}, a.out)
    a.out.with_suffix(".json").write_text(json.dumps({"from": str(a.src), "how": f"a strategy block ({a.zdim})"}))
    adam = a.src.with_suffix(".adam")
    if adam.exists():
        torch.save(grow_adam(torch.load(adam, map_location="cpu", weights_only=False), net, a.zdim), a.out.with_suffix(".adam"))
    print(f"{a.out}: global input {net.config['G']} -> {big.config['G']} (a strategy block of {a.zdim}; outputs within "
          f"{err:.1e} with and without a strategy){'; Adam state carried' if adam.exists() else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
