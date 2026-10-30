"""One sitting with the GPU to itself (~25 minutes, the first time: the compiles stay in ~/.cache/torchinductor; stop the
run first): what each opt-in speedup of the learner's update and the inference round gives on a run's newest checkpoint.

    python3 scripts/bench_gpu_round.py --run fgself-169 --data runs/fullgame/demos-echo-1

The update (scripts/bench_update.py --graphs, the run's settings, seconds per update of 8,192 steps, each variant a
process of its own): the baseline, A --update-grad-fuse 1, B --adam-fused 1, C --update-fuse max-autotune-no-cudagraphs,
D --update-widths with finer steps, and combinations; each one's losses and gradient norms against the baseline's. The
round (scripts/bench_inference.py --compare): today's buckets against finer ones. The C++ loop
(scripts/check_native_infer.py): answers against direct calls, as today and with --infer-past-period-ms.

--infer-past-period-ms itself needs a real run to judge (the games' waits against the GPU freed): fork a run with it, 33
then 50 ms, and read in `inference.jsonl` past_calls_per_s, past_rows_per_call, past_held_ms, gpu_idle, lat_idle, beside
the run's steps a second over the same few minutes (scripts/throughput.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINER = "32,40,48,56,64,80,96,128"  # the update's (160 always)
FINER_INFER = "32,40,48,56,64,80,96"  # the inference calls'
VARIANTS = {  # name -> (bench_update's arguments, seconds it may take: a compile per entity width, ~1 minute each)
    "baseline": ([], 300),
    "A grad-fuse": (["--set", "update_grad_fuse=1"], 480),
    "B adam-fused": (["--adam-fused"], 240),
    "C autotune": (["--set", "update_fuse=max-autotune-no-cudagraphs"], 600),
    "D widths": (["--set", f"update_widths={FINER}"], 300),
    "B+D": (["--adam-fused", "--set", f"update_widths={FINER}"], 180),
    "A+B+D": (["--set", "update_grad_fuse=1", "--adam-fused", "--set", f"update_widths={FINER}"], 480),
}
# fgself-169's settings of the update (scripts/bench_update.py's own defaults are fgself-12's)
SETTINGS = ["epochs=1", "minibatch=256", "pad_groups=8", "seq_len=16", "bf16=1", "clip=0.2", "vf_coef=0.5", "ent_coef=0", "ref_kl=0.025",
            "ref_kl_mode=forward-all", "bc_coef=0", "opd_coef=0.05", "teach_coef=0.5", "label_norm=units", "upgo=1.0", "max_grad_norm=1.0",
            "update_fuse=default"]
COMPARED = ("loss/policy", "loss/value", "loss/entropy", "loss/ref_kl", "loss/opd", "grad_norm")


def run(cmd: list[str], limit: float, log: Path) -> tuple[bool, float]:
    print("$", " ".join(cmd), flush=True)
    t = time.time()
    with open(log, "w") as out:
        try:
            ok = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, timeout=limit).returncode == 0
        except subprocess.TimeoutExpired:
            out.write(f"\n(stopped after {limit:.0f} s)\n")
            ok = False
    took = time.time() - t
    print(f"  {'ok' if ok else 'FAILED'} in {took:.0f} s (log {log})", flush=True)
    if not ok:
        print("  " + "\n  ".join(log.read_text().splitlines()[-12:]), flush=True)
    return ok, took


def difference(a: dict, b: dict) -> float:
    """The largest relative difference of the statistics both have (with a floor: a term near zero)."""
    return max((abs(a[k] - b[k]) / (abs(b[k]) + 1e-3) for k in COMPARED if k in a and k in b), default=float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="fgself-169", help="a run's name under runs/, or its path")
    ap.add_argument("--data", type=Path, default=Path("runs/fullgame/demos-echo-1"), help="demonstrations: the steps")
    ap.add_argument("--only", default="", help="comma-separated: of the variants above (their first letters do), infer, loop")
    ap.add_argument("--out", type=Path, help="where the logs and numbers stay (default: a folder under /tmp)")
    ap.add_argument("--updates", type=int, default=5, help="timed updates of each variant, after the first")
    ap.add_argument("--tolerance", type=float, default=0.02, help="a variant's losses and gradient norm against the baseline's")
    ap.add_argument("--force", action="store_true", help="even with other processes on the GPU (the numbers mean little)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands")
    a = ap.parse_args()
    run_dir = Path(a.run) if "/" in a.run else ROOT / "runs" / a.run
    run_dir = run_dir.resolve()
    data = (a.data if a.data.is_absolute() else ROOT / a.data).resolve()
    ckpts = sorted((run_dir / "checkpoints").glob("[0-9]*.pt"))
    if not ckpts or not data.exists():
        raise SystemExit(f"no checkpoints in {run_dir}, or no {data}")
    if not a.force and not a.dry_run and shutil.which("nvidia-smi"):
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()
        if apps:
            raise SystemExit(f"processes on the GPU: {', '.join(apps)} (stop the run first, or --force)")
    out = a.out or Path("(a new folder)" if a.dry_run else tempfile.mkdtemp(prefix="gpu_round_"))
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    only = [x.strip() for x in a.only.split(",") if x.strip()]
    wanted = lambda name: not only or any(name == w or name.split()[0] == w for w in only)  # noqa: E731
    py = sys.executable
    print(f"the newest checkpoint {ckpts[-1].name}, the anchor {ckpts[0].name} (the same architecture, its memory as in a run); logs in {out}", flush=True)
    base = [py, "scripts/bench_update.py", "--checkpoint", str(ckpts[-1]), "--ref", str(ckpts[0]), "--ref-memory", "--run", str(run_dir),
            "--data", str(data), "--games", "24", "--steps", "8192", "--updates", str(a.updates), "--graphs", "--no-dropout"]
    for kv in SETTINGS:
        base += ["--set", kv]
    results: dict[str, dict] = {}
    took: dict[str, float] = {}
    for name, (extra, limit) in VARIANTS.items():
        if not wanted(name) and name != "baseline":
            continue
        if name == "baseline" and only and not any(wanted(n) for n in VARIANTS):
            continue
        slug = name.replace(" ", "_").replace("+", "-")
        js = out / f"update_{slug}.json"
        cmd = base + extra + ["--json", str(js)]
        if a.dry_run:
            print("$", " ".join(cmd))
            continue
        ok, took[name] = run(cmd, limit, out / f"update_{slug}.log")
        results[name] = json.loads(js.read_text()) if ok and js.exists() else {}
    finer = "" if not (wanted("infer") or wanted("D")) else FINER_INFER
    infer = None
    if finer:
        js = out / "infer.json"
        cmd = [py, "scripts/bench_inference.py", "--run", str(run_dir), "--data", str(data), "--setup", "--entities", "32,48,64,96", "--past-entities",
               "32,48,64,96", "--compare", finer, finer, "--live-rows", "10,30", "--past-rows", "1,5", "--rounds", "300", "--json", str(js)]
        if a.dry_run:
            print("$", " ".join(cmd))
        else:
            ok, _ = run(cmd, 420, out / "infer.log")
            infer = json.loads(js.read_text()) if ok and js.exists() else {}
    loops = {}
    if wanted("loop"):
        for period in (0, 33):
            cmd = [py, "scripts/check_native_infer.py", "--run", str(run_dir), "--data", str(data), "--setup", "--seconds", "20",
                   "--entities", "32,48,64,96", "--past-entities", "32,48,64,96", "--past-period-ms", str(period)]
            if a.dry_run:
                print("$", " ".join(cmd))
            else:
                log = out / f"loop_{period}.log"
                ok, _ = run(cmd, 420, log)
                loops[period] = (ok, [ln for ln in log.read_text().splitlines() if ln.startswith(("checked", "the same step", "stats:"))
                                      or "requests a second" in ln])
    if a.dry_run:
        return 0

    # the table
    print("\nthe update (8,192 steps, seconds after the first update; alone on the GPU):")
    print(f"{'variant':16s} {'s/update':>9s} {'vs base':>8s} {'1st update':>11s} {'whole':>7s} {'GPU GB':>7s} {'losses vs base':>15s}")
    b = results.get("baseline") or {}
    t_base = sum(b["seconds"][1:]) / max(len(b["seconds"]) - 1, 1) if b else None
    bad = 0
    for name, r in results.items():
        if not r:
            print(f"{name:16s} {'failed: see its log above':>40s}")
            bad += 1
            continue
        t = sum(r["seconds"][1:]) / max(len(r["seconds"]) - 1, 1)
        worst = max(difference(x, y) for x, y in zip(r["stats"], b["stats"])) if b else float("nan")
        flag = "" if not (worst > a.tolerance) else "  DIFFERENT"
        bad += bool(flag)
        print(f"{name:16s} {t:9.3f} {(t / t_base - 1) * 100 if t_base else 0:+7.1f}% {r['seconds'][0]:10.1f}s {took[name]:6.0f}s {r.get('gpu_gb', 0):7.2f} {worst:15.4f}{flag}")
    if infer is not None:
        print("\nan inference round (ms, launching + GPU + answers; alone):")
        if infer:
            print(f"entities 32,48,64,96 (today) {sum(infer['base_ms']):.2f} ms; {FINER_INFER} {sum(infer['other_ms']):.2f} ms; "
                  f"largest difference in value + entropy {infer['worst']:.1e}; GPU {infer['gpu_gb']:.2f} GB at its peak")
        else:
            print("failed: see its log above")
            bad += 1
    for period, (ok, lines) in loops.items():
        print(f"\nthe C++ loop, --infer-past-period-ms {period}: {'ok' if ok else 'FAILED'}")
        print("\n".join("  " + ln[:300] for ln in lines))
        bad += not ok
    print(f"\nlogs and numbers in {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
