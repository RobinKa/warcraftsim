"""A model soup (Wortsman et al., 2022): the mean of several runs' weight averages that share one start, as one
checkpoint. The forks of 2026-11-23..26 all start from fgself-236's or fgself-252's weight average and train ~27M
steps each with another change; each one's 17-27M average leans ~+20 over its start and falls back with more
training. Their mean combines what each found along a different direction from the same point, as weight averaging
along one trajectory did (+100-250 rating over single checkpoints). Float weights are averaged; the tables (allowed
orders, masks) and config are the last file's; every file must have the same weights' names and shapes and config.

    python scripts/soup.py runs/fullgame/soups/soup-a.pt runs/fgself-249/averaged/swa-17-27.pt runs/fgself-252/... [...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def soup(paths: list[Path]) -> dict:
    total, n, last, cfg0 = None, 0, None, None
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = ck["model"]
        if total is None:
            total = {k: v.double().clone() for k, v in m.items() if v.is_floating_point()}
            shapes = {k: tuple(v.shape) for k, v in m.items()}
            cfg0 = ck.get("config")
        else:
            if {k: tuple(v.shape) for k, v in m.items()} != shapes:
                raise SystemExit(f"{p}: its weights' names or shapes differ from the first's")
            if ck.get("config") != cfg0:
                diff = {k for k in set(cfg0) | set(ck.get("config") or {}) if cfg0.get(k) != (ck.get("config") or {}).get(k)}
                raise SystemExit(f"{p}: its config differs from the first's in {sorted(diff)}")
            for k in total:
                total[k] += m[k].double()
        n, last = n + 1, ck
    model = {k: (total[k] / n).to(v.dtype) if k in total else v for k, v in last["model"].items()}
    return {**last, "model": model, "soup": {"n": n, "from": [str(p) for p in paths]}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("paths", type=Path, nargs="+")
    a = ap.parse_args(argv)
    out = soup(a.paths)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    a.out.with_suffix(".json").write_text(json.dumps({"how": "model soup", "from": [str(p) for p in a.paths]}))
    print(f"{a.out}: the mean of {len(a.paths)} checkpoints: {', '.join(p.parts[-3] for p in a.paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
