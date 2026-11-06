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

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame.costs import order_costs  # noqa: E402
from warcraftsim.fullgame.model import load  # noqa: E402
from warcraftsim.fullgame.native_infer import NativeServer  # noqa: E402
from warcraftsim.fullgame.selfplay import Inference, RemoteInference, SharedRows, precision_flags, publish  # noqa: E402
from bench_inference import view_steps  # noqa: E402

# the largest (value, entropy) difference against a direct fp32 call that passes, per --infer-precision (fp32: graphs
# against eager; the others about twice the largest scripts/infer_cpu_study.py mixed found on ~110k units, the value's
# per row and the order entropy's per unit, which a row with one own unit is: rounding, not a bug)
TOL = {"fp32": (1e-3, 1e-3), "tf32": (5e-3, 2.5e-2), "bf16-mixed": (2e-2, 0.2), "bf16": (5e-2, 0.25)}


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
    ap.add_argument("--past-period-ms", type=float, default=0.0, help="selfplay --infer-past-period-ms")
    ap.add_argument("--rows", default="", help="selfplay --infer-rows (the current network's padded row counts)")
    ap.add_argument("--precision", default="fp32", help="selfplay --infer-precision: the loop's calls in fp32, tf32, "
                                                        "bf16 or bf16-mixed, checked against direct fp32 calls within --tol "
                                                        "(and reported against direct calls of the same precision)")
    ap.add_argument("--tol", type=float, help="the largest difference in value or entropy that passes (default by "
                                              "precision, the value's and the entropy's: TOL)")
    ap.add_argument("--setup", action="store_true", help="the run's input settings first (scripts/bench_view.setup: a "
                                                          "recent run needs them)")
    a = ap.parse_args()
    tol_v, tol_e = (a.tol, a.tol) if a.tol is not None else TOL[a.precision]
    device = torch.device("cuda")
    if a.setup:
        from bench_view import setup
        setup(str(a.run.resolve()))
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
           "exploiter_share": 0, "infer_entities": [int(x) for x in a.entities.split(",")], "vocab": str(a.run / "vocab.json"),
           "infer_past_period": a.past_period_ms / 1000.0, "infer_precision": a.precision,
           **({"infer_rows": [int(x) for x in a.rows.split(",") if x]} if a.rows else {}),
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
    # (and plain calls of the loop's precision: what the compiled graphs and the loop add to the rounding itself)
    alike = Inference(None, device, 64, compile_=False, thread=False, **precision_flags(a.precision)) \
        if a.precision != "fp32" else None
    keys = ["current"] + past
    nets = {k: (net if k == "current" else load(k, device, vocab=ck["vocab"] if fx.NEUTRAL else None)[0].eval()) for k in keys}
    errors, counts, rng0 = [], {"calls": 0}, np.random.default_rng(0)
    diffs: list[tuple[float, float]] = []  # (value, entropy) against the direct fp32 call
    same: list[tuple[float, float]] = []  # (value, entropy) against the direct call of the same precision
    past_diffs: list[tuple[float, float]] = []  # (the past snapshots' slots' rows alone, against fp32)
    lock, ref_lock = threading.Lock(), threading.Lock()
    slowest = [0.0]

    def game(seed: int, check: bool, until: float):
        rng = np.random.default_rng(seed)
        while time.time() < until:
            # a game's request: one side's row or both sides' (the current network with a past snapshot, or two
            # networks: a mixed request's answers may come from different rounds with --past-period-ms)
            asked = [keys[int(rng.integers(len(keys)))] for _ in range(1 + int(rng.integers(2)))]
            st = dict(steps[int(rng.integers(len(steps)))])
            st["h"] = rng.normal(size=c["d"]).astype(np.float32) if net.memory and rng.random() < 0.5 else None
            t_asked = time.time()
            answers = remote.request([(key, dict(st)) for key in asked])
            with lock:
                counts["calls"] += 1
                slowest[0] = max(slowest[0], time.time() - t_asked)
            if check:
                for key, got in zip(asked, answers):
                    with ref_lock:  # (one direct call at a time: a tf32 or bf16-mixed one sets the process's matmul
                        want = direct._finish(direct._launch(nets[key], [st]))[0]  # precision while it runs)
                        w2 = alike._finish(alike._launch(nets[key], [st]))[0] if alike is not None else None
                    if got is not None:
                        diffs.append((abs(got["value"] - want["value"]), abs(got["entropy"] - want["entropy"])))
                        if key != "current":
                            past_diffs.append(diffs[-1])
                        if w2 is not None:
                            same.append((abs(got["value"] - w2["value"]), abs(got["entropy"] - w2["entropy"])))
                    bad = got is None or abs(got["value"] - want["value"]) > tol_v or abs(got["entropy"] - want["entropy"]) > tol_e \
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
    for name, got in ((f"direct fp32 calls (tol {tol_v:g}, {tol_e:g})", diffs), ("direct fp32 calls, past snapshots' slots alone",
                                                                     past_diffs),
                      (f"direct {a.precision} calls (eager)", same)):
        if got:
            d = np.array(got)
            print(f"{a.precision} against {name}, {len(d)} rows: value difference mean {d[:, 0].mean():.1e} p99 "
                  f"{np.percentile(d[:, 0], 99):.1e} max {d[:, 0].max():.1e}, entropy mean {d[:, 1].mean():.1e} p99 "
                  f"{np.percentile(d[:, 1], 99):.1e} max {d[:, 1].max():.1e}", flush=True)
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
    print(f"{counts['calls'] / (time.time() - t0):.0f} requests a second from {a.games - 5} threads; the slowest answer "
          f"{1000 * slowest[0]:.0f} ms", flush=True)
    server.stop()
    print("stats:", (tmp / "inference.jsonl").read_text().splitlines()[-1:] if (tmp / "inference.jsonl").exists() else "-")
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
