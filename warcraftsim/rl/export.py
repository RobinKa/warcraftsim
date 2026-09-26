"""Converts a torch-trainer checkpoint (.pt) to numpy (.npz) for numpy_model.py.

    python warcraftsim/rl/export.py CHECKPOINT.pt OUT.npz      (run with the torch Python)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch


def main() -> None:
    src, out = sys.argv[1:3]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    arrays = {k: v.detach().float().numpy() for k, v in ck["model"].items()}
    # the trainer learns values of rewards times reward_scale (default 1 / the task's, which the bridge
    # applied): value_scale converts them back to the bridge's scale
    scale = (ck.get("args") or {}).get("reward_scale") or 1.0 / ck["spec"].get("reward_scale", 1.0)
    np.savez(out, **arrays, __spec__=np.array(json.dumps(ck["spec"])), __config__=np.array(json.dumps(ck["config"])),
             __steps__=np.array(ck.get("steps", 0)), __value_scale__=np.array(1.0 / scale))


if __name__ == "__main__":
    main()
