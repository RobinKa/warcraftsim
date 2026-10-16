"""A copy of a checkpoint that samples at other temperatures (the order head's, the target heads'), the
same weights: as an evaluation checkpoint (selfplay --eval-checkpoints) it measures a temperature's
effect on play head to head against the original, in the same run's paired games.

    python scripts/checkpoint_variant.py runs/fgself-41/checkpoints/X.pt runs/fullgame/variants/x-t0.8.pt --order 0.8 --target 0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--order", type=float, help="the order head's temperature")
    ap.add_argument("--target", type=float, help="the target heads' temperature (unit, x, y)")
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    model, config = dict(ck["model"]), dict(ck["config"])
    if a.order is not None:
        model["order_temp"], config["order_temperature"] = torch.tensor(a.order), a.order
    if a.target is not None:
        model["target_temp"], config["target_temperature"] = torch.tensor(a.target), a.target
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ck, "model": model, "config": config, "variant_of": str(a.src)}, a.out)
    a.out.with_suffix(".json").write_text(json.dumps({"from": str(a.src), "how": "other temperatures"}))  # (the lineage)
    print(f"{a.out}: order temperature {config.get('order_temperature', 1.0)}, target {config.get('target_temperature', 1.0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
