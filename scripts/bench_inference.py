"""The inference server's rounds (selfplay.launch_round) on the GPU: one after the other against side
by side (a CUDA stream per network, past snapshots in slots). Every row's value and entropy (what the
call computes without sampling) must agree; prints milliseconds a round. It needs the GPU to itself.

    python3 scripts/bench_inference.py --run runs/fgself-12 --data runs/fullgame/demos-fast-1 [--rounds 300]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warcraftsim.fullgame import bc, features as fx  # noqa: E402
from warcraftsim.fullgame.costs import order_costs  # noqa: E402
from warcraftsim.fullgame.selfplay import Inference, Nets, PastNet, launch_round  # noqa: E402
from warcraftsim.fullgame.trace import unit_values  # noqa: E402


def view_steps(paths: list[Path], vocab: dict, costs, n: int) -> list[dict]:
    enc, out = fx.Encoder(vocab, costs), []
    for p in paths:
        g = fx.load_game(p)
        for side in (0, 1):
            d = bc.side_data(enc, g, side, unit_values())
            for t in range(0, len(d["n_own"]), 7):
                k = max(1, int(d["mask"][t].sum()))
                out.append({"ent": d["ent"][t, :k].astype(np.float32), "type": d["type"][t, :k], "cur": d["cur"][t, :k],
                            "glob": d["glob"][t].astype(np.float32), "n": k, "n_own": int(d["n_own"][t]),
                            "avail": d["avail"][t], "h": None})
                if len(out) >= n:
                    return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--past", type=int, default=6, help="past snapshots to draw from")
    ap.add_argument("--slots", type=int, default=4)
    a = ap.parse_args()
    device = torch.device("cuda")
    league = json.loads((a.run / "league.json").read_text())
    keys = [m["path"] for m in league["members"] if m.get("path")][-a.past:]
    cfg = {"run_dir": str(a.run), "max_past": 32, "exploiter_share": 0}
    nets = Nets(cfg, device)
    latest = max((a.run / "checkpoints").glob("*.pt"))
    vocab = torch.load(latest, map_location="cpu", weights_only=False)["vocab"]
    steps = view_steps(sorted(a.data.glob("game*.npz"))[:6], vocab, order_costs(vocab, "duelfast"), 3000)
    print(f"{len(steps)} view steps, {len(keys)} past snapshots", flush=True)
    rng = random.Random(0)
    rounds = []
    for _ in range(a.rounds):  # like the run's: ~10 rows of the current network, ~3-4 past snapshots with a few each
        groups = {"current": [(None, i, rng.choice(steps)) for i in range(rng.randint(3, 20))]}
        for key in rng.sample(keys, rng.randint(1, min(len(keys), 6))):
            groups[key] = [(None, i, rng.choice(steps)) for i in range(rng.randint(1, 6))]
        rounds.append(groups)

    def run(parallel: bool) -> tuple[list, float]:
        infer = Inference(nets, device, 64, True, buckets=(8, 24, 64), thread=False, entities=(48, 96))
        past = Inference(nets, device, 16, True, buckets=(4, 16), thread=False, entities=(64,))
        swap = PastNet(device, slots=a.slots if parallel else 1)
        streams = {} if parallel else None
        out, times = [], []
        for r, groups in enumerate(rounds + rounds[:100]):  # (the first 100 again, timed: shapes captured)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            calls, used = launch_round(groups, nets, infer, past, swap, 64, streams)
            t1 = time.perf_counter()
            infer.wait(used or None)
            past.wait(used or None)
            t2 = time.perf_counter()
            res = [fwd._finish(h) for fwd, h, _, _ in calls]
            t3 = time.perf_counter()
            if r >= len(rounds):
                times.append((t1 - t0, t2 - t1, t3 - t2))
            else:
                out.append([(x["value"], x["entropy"]) for part in res for x in part])
        return out, 1000 * np.mean(times, axis=0)

    seq, t_seq = run(False)
    par, t_par = run(True)
    worst = max(abs(x[0] - y[0]) + abs(x[1] - y[1]) for rs, rp in zip(seq, par) for x, y in zip(rs, rp))
    rows = sum(len(r) for r in seq)
    print(f"{rows} rows in {len(seq)} rounds: largest difference in value + entropy {worst:.2e}")
    for name, t in (("one after the other", t_seq), ("side by side", t_par)):
        print(f"ms a round, {name}: {t.sum():.2f} (launching {t[0]:.2f}, waiting for the GPU {t[1]:.2f}, "
              f"the answers {t[2]:.2f})")
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
