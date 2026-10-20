"""Merge specialists by averaging their weights (the user's idea, 2026-10-20: runs forked from one checkpoint, each
trained on one family of scenarios, combined): a model soup (the mean of their weights) or task arithmetic (the
start plus `--scale` times the sum of each run's change from it). Works for networks fine-tuned from the same weights
(the forks' shared start); buffers (masks, temperatures) come from the first checkpoint.

    python scripts/average_weights.py runs/fgself-120/checkpoints/<a>.pt runs/fgself-121/checkpoints/<b>.pt \\
        --out runs/merged/soup.pt [--base runs/fgself-119/checkpoints/<start>.pt --scale 0.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def merge(states: list[dict], base: dict | None = None, scale: float = 1.0) -> dict:
    """The mean of `states` (a soup), or with `base` the base plus `scale` times the sum of each one's change (task
    arithmetic). Tensors that are not floats (buffers of counts and masks) come from the first state."""
    out = {}
    for k, v in states[0].items():
        if not torch.is_floating_point(v) or any(k not in s or s[k].shape != v.shape for s in states):
            out[k] = v
        elif base is not None and k in base and base[k].shape == v.shape:
            out[k] = base[k] + scale * sum(s[k].float() - base[k].float() for s in states).to(v.dtype)
        else:
            out[k] = (sum(s[k].float() for s in states) / len(states)).to(v.dtype)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", type=Path, help="the specialists' shared start: task arithmetic instead of a soup")
    ap.add_argument("--scale", type=float, default=1.0, help="task arithmetic: the changes' weight")
    a = ap.parse_args(argv)
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in a.checkpoints]
    if any(c["config"] != cks[0]["config"] for c in cks[1:]):
        raise SystemExit("the checkpoints' networks differ (config)")
    base = torch.load(a.base, map_location="cpu", weights_only=False)["model"] if a.base else None
    merged = merge([c["model"] for c in cks], base, a.scale)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**cks[0], "model": merged, "merged_from": [str(p) for p in a.checkpoints],
                **({"merged_base": str(a.base), "merged_scale": a.scale} if a.base else {})}, a.out)
    print(f"{a.out}: {'task arithmetic' if base is not None else 'a soup'} of {len(cks)} checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
