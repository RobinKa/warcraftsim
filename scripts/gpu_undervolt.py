"""Undervolt the GPU the way Linux allows it: lock the top clock and shift the
voltage/frequency curve up, so the top clock runs at a lower voltage.

NVIDIA's Linux driver has no voltage control. The clock offset
(nvmlDeviceSetGpcClkVfOffset) shifts the whole curve: with +150 MHz, each
voltage step runs 150 MHz faster. With the top clock locked
(nvmlDeviceSetGpuLockedClocks) the card then holds the same clock at the
voltage it used to need for 150 MHz less, which draws less power.

  status                       clocks, offset, power, Xid lines since boot
  apply --max-clock 1725 --offset 150   (root; lost on reboot)
  reset                        (root) offset 0, clocks unlocked
  test --max-clock 1725 --offsets 0,100,150,200 [--seconds 60]
      (root; nothing else on the GPU) for each offset: a bf16 GEMM load and an
      fp32 GEMM checked bit for bit against its first result; power, clock,
      throughput; stops at the first mismatch or new Xid line, then resets

docs/reference/optimization-log.md has the measurements.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import pynvml as nv


def handle():
    nv.nvmlInit()
    return nv.nvmlDeviceGetHandleByIndex(0)


def xid_lines() -> list[str]:
    out = subprocess.run(["journalctl", "-k", "-b", "--no-pager", "-q"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "NVRM: Xid" in line]


def status(h) -> dict:
    return {"clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_GRAPHICS),
            "offset_mhz": nv.nvmlDeviceGetGpcClkVfOffset(h),
            "power_w": round(nv.nvmlDeviceGetPowerUsage(h) / 1000, 1),
            "power_limit_w": nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000,
            "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
            "xid_lines": len(xid_lines())}


def apply(h, max_clock: int, offset: int) -> None:
    nv.nvmlDeviceSetGpuLockedClocks(h, 210, max_clock)
    nv.nvmlDeviceSetGpcClkVfOffset(h, offset)


def reset(h) -> None:
    nv.nvmlDeviceSetGpcClkVfOffset(h, 0)
    nv.nvmlDeviceResetGpuLockedClocks(h)


def load(h, seconds: float, shape=(8192, 8192, 8192)) -> dict:
    """bf16 GEMMs for throughput and power; an fp32 GEMM every ~0.5 s compared
    bit for bit with its first result (cuBLAS gives the same bits for the same
    shapes on the same card, so any difference is a computing error)."""
    import torch
    torch.manual_seed(0)
    m, k, n = shape
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    x, y = (torch.randn(4096, 4096, device="cuda") for _ in range(2))
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = x @ y
    torch.cuda.synchronize()
    flops, power, clock, checks, bad = 0.0, [], [], 0, 0
    t0 = time.time()
    last = t0
    while time.time() - t0 < seconds:
        for _ in range(20):
            a @ b
        flops += 20 * 2 * m * k * n
        torch.cuda.synchronize()
        if time.time() - last > 0.5:
            last = time.time()
            checks += 1
            bad += int(not torch.equal(x @ y, ref))
            if time.time() - t0 > 5:  # past the ramp
                power.append(nv.nvmlDeviceGetPowerUsage(h) / 1000)
                clock.append(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_GRAPHICS))
    dt = time.time() - t0
    return {"tflops": round(flops / dt / 1e12, 1), "power_w": round(sum(power) / max(1, len(power)), 1),
            "clock_mhz": round(sum(clock) / max(1, len(clock))), "checks": checks, "mismatches": bad}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "apply", "reset", "test"])
    ap.add_argument("--max-clock", type=int, default=1725)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--offsets", default="0,100,150,200")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--shape", default="8192,8192,8192", help="test: the bf16 load's m,k,n (65536,256,1024 is "
                    "learner-sized: light enough to reach the locked clock under the power limit)")
    ap.add_argument("--baseline", type=int, default=1, help="test: first a run with nothing locked (stock)")
    args = ap.parse_args()
    h = handle()
    if args.cmd == "status":
        print(json.dumps(status(h)))
    elif args.cmd == "apply":
        apply(h, args.max_clock, args.offset)
        time.sleep(1)
        print(json.dumps(status(h)))
    elif args.cmd == "reset":
        reset(h)
        print(json.dumps(status(h)))
    else:
        xid0 = len(xid_lines())
        shape = tuple(int(v) for v in args.shape.split(","))
        rows = []
        try:
            if args.baseline:
                reset(h)
                rows.append({"setting": "stock", **load(h, args.seconds, shape)})
                print(json.dumps(rows[-1]), flush=True)
            for off in (int(o) for o in args.offsets.split(",")):
                apply(h, args.max_clock, off)
                r = {"setting": f"{args.max_clock} MHz +{off}", **load(h, args.seconds, shape)}
                r["new_xid"] = len(xid_lines()) - xid0
                rows.append(r)
                print(json.dumps(r), flush=True)
                if r["mismatches"] or r["new_xid"]:
                    print("unstable: stopped", flush=True)
                    break
        finally:
            reset(h)
        sys.exit(0)


if __name__ == "__main__":
    main()
