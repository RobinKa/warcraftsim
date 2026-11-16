"""Weight averaging of a run's checkpoints (2026-11-16): the mean of their float weights, the newest's tables (the
orders each type may give, the order kinds, the valid masks) and config, as one checkpoint. A continuation's level
against insane swings by ~8 points between checkpoints 20-30M steps apart (burst-216, burst-220: both starts 0.838,
every continuation 0.68-0.76), and an average along one trajectory (SWA) often sits above its points.

    python scripts/average_checkpoints.py runs/fgself-212/averaged/swa-118808740-144699411.pt \\
        --run fgself-212 --from 118808740 --to 144699411 [--every 1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def average(paths: list[Path]) -> dict:
    total, n, last = None, 0, None
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = ck["model"]
        if total is None:
            total = {k: v.double().clone() for k, v in m.items() if v.is_floating_point()}
        else:
            if set(k for k, v in m.items() if v.is_floating_point()) != set(total):
                raise SystemExit(f"{p}: its weights' names differ from the first's")
            for k in total:
                total[k] += m[k].double()
        n, last = n + 1, ck
    model = {k: (total[k] / n).to(v.dtype) if k in total else v for k, v in last["model"].items()}
    return {**last, "model": model, "averaged": {"n": n, "from": str(paths[0]), "to": str(paths[-1])}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--run", required=True)
    ap.add_argument("--from", dest="lo", type=int, required=True, help="the first checkpoint's steps (inclusive)")
    ap.add_argument("--to", dest="hi", type=int, required=True, help="the last checkpoint's steps (inclusive)")
    ap.add_argument("--every", type=int, default=1, help="every Nth checkpoint in the range")
    ap.add_argument("--root", type=Path, default=Path("runs"))
    a = ap.parse_args(argv)
    cks = sorted(p for p in (a.root / a.run / "checkpoints").glob("[0-9]*.pt") if a.lo <= int(p.stem) <= a.hi)[::a.every]
    if not cks:
        raise SystemExit("no checkpoints in that range")
    ck = average(cks)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    torch.save(ck, a.out)
    print(f"{a.out}: the mean of {len(cks)} checkpoints, {cks[0].stem} to {cks[-1].stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
