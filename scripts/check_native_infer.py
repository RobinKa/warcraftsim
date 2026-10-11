"""The C++ inference loop (selfplay --infer-native, native_infer.py) against plain Python calls: games ask from
threads of their own (the current network and past snapshots, through shared rows and binary messages);
every answer's value and entropy must equal a direct call's, the versions be right, and the sampling differ
from call to call. Then requests per second from many threads. Uses the GPU briefly.

    python3 scripts/check_native_infer.py --run runs/fgself-12 --data runs/fullgame/demos-fast-1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from warcraftsim.fullgame.costs import order_costs  # noqa: E402
from warcraftsim.fullgame.model import load  # noqa: E402
from warcraftsim.fullgame.native_infer import NativeServer  # noqa: E402
from warcraftsim.fullgame.selfplay import Inference, RemoteInference, SharedRows, publish  # noqa: E402
from bench_inference import view_steps  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--entities", default="48,96", help="selfplay --infer-entities")
    ap.add_argument("--past-entities", default="64", help="selfplay --infer-past-entities")
    ap.add_argument("--gpu-split", type=int, default=0, help="selfplay --gpu-split: the loop's streams in a part of "
                                                            "the GPU of this many SMs")
    a = ap.parse_args()
    device = torch.device("cuda")
    tmp = Path(tempfile.mkdtemp(prefix="native_infer_"))
    latest = max((a.run / "checkpoints").glob("*.pt"))
    net, ck = load(latest, device)
    net.eval()
    publish(net, 7, tmp)
    league = json.loads((a.run / "league.json").read_text())
    past = [m["path"] for m in league["members"] if m.get("path")][-3:]
    (tmp / "league.json").write_text(json.dumps({"members": [{"path": p} for p in past]}))
    c = net.config
    spec = SharedRows(str(tmp / "rows"), 2 * a.games, c["G"], c["n_orders"], c["d"] if net.memory else 0, create=True).spec
    pipes = [mp.get_context("spawn").Pipe() for _ in range(a.games)]
    cfg = {"run_dir": str(tmp), "device": "cuda", "learner_pid": os.getpid(), "infer_batch": 64, "compile": True,
           "max_past": 32, "shared_rows": spec, "infer_fuse": 1, "infer_past_slots": 4, "infer_period": 0.004,
           "exploiter_share": 0, "infer_entities": [int(x) for x in a.entities.split(",")],
           "infer_past_entities": [int(x) for x in a.past_entities.split(",")]}
    split = None
    if a.gpu_split:
        from warcraftsim.fullgame.gpu_split import GpuSplit
        split = GpuSplit(a.gpu_split)
        print(f"GPU split: {split.small_sms} SMs for the loop", flush=True)
    server = NativeServer(cfg, [b for _, b in pipes], device, split)
    server.start()
    remote = RemoteInference([x for x, _ in pipes], SharedRows(*spec), [2 * k for k in range(a.games)])
    steps = view_steps(sorted(a.data.glob("game*.npz"))[:3], ck["vocab"], order_costs(ck["vocab"], "duelfast"), 400)
    direct = Inference(None, device, 64, compile_=False, thread=False)
    keys = ["current"] + past
    nets = {k: (net if k == "current" else load(k, device)[0].eval()) for k in keys}
    errors, counts, rng0 = [], {"calls": 0}, np.random.default_rng(0)
    lock = threading.Lock()

    def game(seed: int, check: bool, until: float):
        rng = np.random.default_rng(seed)
        while time.time() < until:
            key = keys[int(rng.integers(len(keys)))]
            st = dict(steps[int(rng.integers(len(steps)))])
            st["h"] = rng.normal(size=c["d"]).astype(np.float32) if net.memory and rng.random() < 0.5 else None
            got = remote.request([(key, st)])[0]
            with lock:
                counts["calls"] += 1
            if check:
                want = direct._finish(direct._launch(nets[key], [st]))[0]
                bad = got is None or abs(got["value"] - want["value"]) > 1e-3 or abs(got["entropy"] - want["entropy"]) > 1e-3 \
                    or got["version"] != (7 if key == "current" else -1)
                if bad:
                    errors.append((key, None if got is None else (got["value"], got["entropy"], got["version"]),
                                   (want["value"], want["entropy"])))
            if server.failed():
                errors.append(("server", server.failed()))
                return

    # 1. correctness: a few threads, every answer checked
    threads = [threading.Thread(target=game, args=(s, True, time.time() + 8)) for s in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(f"checked {counts['calls']} calls: {len(errors)} wrong" + (f", e.g. {errors[:3]}" if errors else ""), flush=True)
    # sampling: the same view step twice gives different orders (somewhere)
    st = dict(steps[0], h=None)
    o1, o2 = remote.request([("current", st)])[0], remote.request([("current", st)])[0]
    print("the same step twice:", "different samples" if not np.array_equal(o1["order"], o2["order"]) or
          not np.array_equal(o1["tgt"], o2["tgt"]) else "the SAME samples", flush=True)
    # 2. throughput: every game thread asking as fast as it can
    counts["calls"] = 0
    t0 = time.time()
    threads = [threading.Thread(target=game, args=(100 + s, False, t0 + a.seconds)) for s in range(a.games - 5)]  # (a pipe per thread: 5 used above)
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(f"{counts['calls'] / (time.time() - t0):.0f} requests a second from {a.games - 5} threads", flush=True)
    server.stop()
    print("stats:", (tmp / "inference.jsonl").read_text().splitlines()[-1:] if (tmp / "inference.jsonl").exists() else "-")
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
