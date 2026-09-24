"""Training throughput for several trainer (vec) configurations on the same running games.

    python scripts/bench_train.py --task micro_mirror --envs 24 --workers 6 --steps 150000 \
        --config buffers=2 --config buffers=6 --config "buffers=6 omp=passive"

The bridge workers launch the games once; each configuration then runs the trainer for
`--steps` agent steps (the bridge hands the games to the next trainer). Reports the median SPS
after a warm-up and the trainer's time split. Configuration keys: buffers, threads (default:
envs), horizon, omp (OMP_WAIT_POLICY), and any --section.key=value trainer argument.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from warcraftsim.puffer.bridge import run_worker
from warcraftsim.puffer.build import PUFFER_BUILD, build_trainer
from warcraftsim.puffer.tasks import get_task
from warcraftsim.puffer.train import claim_slot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="micro_mirror")
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--config", action="append", required=True)
    args = ap.parse_args()

    task = get_task(args.task)
    binary = build_trainer(task)
    run_dir = Path(tempfile.mkdtemp(prefix="bench-train-"))
    counts = [args.envs // args.workers + (1 if w < args.envs % args.workers else 0) for w in range(args.workers)]
    slot, _lock = claim_slot()
    games = "train" if slot == 0 else f"train{slot}_"
    sockets = [f"/dev/shm/warcraftsim/bench-train-{os.getpid()}-{w}.sock" for w in range(args.workers)]
    ctx = mp.get_context("spawn")
    stop, readies = ctx.Event(), [ctx.Event() for _ in counts]
    procs = [ctx.Process(target=run_worker, daemon=True, args=(task.name, c, str(run_dir), sockets[w], 0, 0,
                                                              f"{games}{w}-", w, readies[w], stop))
             for w, c in enumerate(counts)]
    t0 = time.time()
    for p in procs:
        p.start()
    for r in readies:
        r.wait()
    print(f"{args.envs} games ready in {time.time() - t0:.0f}s", flush=True)
    try:
        for cfg in args.config:
            kv = dict(item.split("=", 1) for item in cfg.split() if not item.startswith("--"))
            extra = [item for item in cfg.split() if item.startswith("--")]
            buffers = int(kv.get("buffers", 2))
            horizon = int(kv.get("horizon", 128))
            agents = args.envs * task.num_agents
            batch = agents * horizon
            log = run_dir / f"train-{len(list(run_dir.glob('train-*.jsonl')))}.jsonl"
            cmd = [str(binary), "train", f"--vec.total_agents={agents}", f"--vec.num_buffers={buffers}",
                   f"--vec.num_threads={int(kv.get('threads', max(args.envs, buffers)))}",
                   f"--train.total_timesteps={args.steps}", f"--train.horizon={horizon}",
                   f"--train.minibatch_size={batch}", "--base.eval_episodes=0", "--base.checkpoint_interval=0",
                   f"--base.checkpoint_dir={run_dir / 'ck'}", f"--base.log_dir={run_dir / 'logs'}", *extra]
            env = dict(os.environ, WC3_BRIDGE=";".join(sockets), PUFFER_JSONL=str(log))
            if "omp" in kv:
                env["OMP_WAIT_POLICY"] = kv["omp"]
            t = time.time()
            res = subprocess.run(cmd, cwd=PUFFER_BUILD, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                 text=True)
            rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            rows = rows[max(3, len(rows) // 5):]  # skip the warm-up
            if res.returncode != 0 or not rows:
                print(f"{cfg}: failed (exit {res.returncode}) {res.stderr[-300:]}", flush=True)
                continue
            med = lambda k: statistics.median(r.get(k, 0.0) for r in rows)  # noqa: E731
            print(f"{cfg:40s} SPS {med('SPS'):6.0f}  per epoch: rollout {med('perf/rollout'):.3f}s "
                  f"env {med('perf/eval_env'):.3f}s model {med('perf/eval_model'):.3f}s "
                  f"train {med('perf/train'):.3f}s  ({time.time() - t:.0f}s)", flush=True)
    finally:
        stop.set()
        for p in procs:
            p.join(60)


if __name__ == "__main__":
    main()
