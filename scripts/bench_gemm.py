"""Matrix multiply throughput on the GPU by precision (TFLOPS): bf16, fp16 (tensor cores, fp32 accumulation), tf32 and
fp32, at the README's size of gpu-unlock (8192^3) and at the learner's sizes (a minibatch's entity tokens times the
model width). Before and after a change to the card (gpu-unlock, a power limit): its numbers side by side.

    python scripts/bench_gemm.py --out runs/bench/gpu-before.json
    python scripts/bench_gemm.py --out runs/bench/gpu-after.json --compare runs/bench/gpu-before.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

SIZES = ((8192, 8192, 8192), (65536, 256, 1024), (65536, 1024, 256), (16384, 256, 256))  # (m, k, n)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "tf32": torch.float32, "fp32": torch.float32}


def tflops(m: int, k: int, n: int, name: str, seconds: float = 3.0) -> float:
    torch.backends.cuda.matmul.allow_tf32 = name == "tf32"
    a = torch.randn(m, k, device="cuda", dtype=DTYPES[name])
    b = torch.randn(k, n, device="cuda", dtype=DTYPES[name])
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    reps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        reps += 10
    return 2.0 * m * k * n * reps / (time.perf_counter() - t0) / 1e12


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path, help="an earlier run's results: each beside its ratio")
    a = ap.parse_args()
    out = {"gpu": torch.cuda.get_device_name(0), "time": time.time(), "results": {}}
    before = json.loads(a.compare.read_text())["results"] if a.compare else {}
    for m, k, n in SIZES:
        for name in DTYPES:
            key = f"{name} {m}x{k}x{n}"
            v = tflops(m, k, n, name)
            out["results"][key] = round(v, 2)
            ratio = f"  x{v / before[key]:.2f}" if key in before else ""
            print(f"{key:24s} {v:7.2f} TFLOPS{ratio}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
