"""Train a PufferLib 5.0 policy on a Warcraft III task.

    python -m warcraftsim.puffer.train --task micro --envs 16 --timesteps 2_000_000

A run lives in runs/<name>/:
    run.json         configuration and status (read by the dashboard)
    train.jsonl      trainer log, one line per epoch (SPS, losses, env/win_rate, ...)
    episodes-*.jsonl every finished episode (one file per bridge worker)
    trainer.log      the trainer's terminal output
    checkpoints/     PufferLib checkpoints
    renders/, replays/, videos/, media.jsonl   see bridge.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .. import paths
from .bridge import run_worker
from .build import PUFFER_BUILD, build_trainer
from .tasks import get_task

RUNS_DIR = Path(os.environ.get("WARCRAFTSIM_RUNS", paths.REPO_ROOT / "runs"))


def claim_slot():
    """A machine-wide training slot (held until exit): its game names (and Wine prefixes) are
    reused by later runs, while concurrent runs get their own."""
    import fcntl

    lock_dir = Path("/dev/shm/warcraftsim/train-slots")
    lock_dir.mkdir(parents=True, exist_ok=True)
    for k in range(64):
        f = open(lock_dir / f"{k}.lock", "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return k, f
        except BlockingIOError:
            f.close()
    raise RuntimeError("no free training slot")


class Run:
    def __init__(self, run_dir: Path, info: dict):
        self.dir = run_dir
        self.info = info
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save()

    def save(self, **updates) -> None:
        self.info.update(updates)
        tmp = self.dir / "run.json.tmp"
        tmp.write_text(json.dumps(self.info, indent=2))
        tmp.replace(self.dir / "run.json")


def trainer_args(args, envs: int, agents_per_env: int = 1) -> list[str]:
    horizon = args.horizon
    agents = envs * agents_per_env  # PufferLib creates environments until it has this many agents
    batch = agents * horizon
    minibatch = min(args.minibatch or batch, batch)
    return [
        "train",
        f"--vec.total_agents={agents}", f"--vec.num_buffers={args.buffers}",
        f"--vec.num_threads={max(envs, args.buffers)}",
        f"--train.total_timesteps={int(args.timesteps)}", f"--train.horizon={horizon}",
        f"--train.minibatch_size={minibatch}", f"--train.learning_rate={args.lr}",
        f"--train.ent_coef={args.ent_coef}", f"--train.gamma={args.gamma}",
        f"--policy.hidden_size={args.hidden}", f"--policy.num_layers={args.layers}",
        "--base.eval_episodes=0", f"--base.checkpoint_interval={args.checkpoint_interval}",
        *args.extra,
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="micro")
    ap.add_argument("--envs", type=int, default=16, help="parallel games")
    ap.add_argument("--workers", type=int, default=4, help="bridge processes (each runs envs/workers games)")
    ap.add_argument("--timesteps", type=float, default=1_000_000)
    ap.add_argument("--name", help="run name (default: task + timestamp)")
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--minibatch", type=int, default=0, help="default: the whole batch (envs * horizon)")
    ap.add_argument("--buffers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--ent-coef", type=float, default=0.001,
                    help="PufferLib 5 does not normalize advantages: keep this small for small rewards")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--checkpoint-interval", type=int, default=20,
                    help="epochs; videos show the policy of the latest checkpoint before their episode")
    ap.add_argument("--record-every", type=int, default=20, help="trajectory render every N episodes of game 0")
    ap.add_argument("--video-every", type=int, default=60, help="replay video every N episodes of game 0 (0: off)")
    ap.add_argument("extra", nargs="*", help="extra PufferLib arguments, e.g. --train.clip_coef=0.1")
    args = ap.parse_args(argv)

    task = get_task(args.task)
    name = args.name or f"{task.name}-{datetime.now():%Y%m%d-%H%M%S}"
    run = Run(RUNS_DIR / name, {
        "name": name, "task": task.name, "description": task.description, "envs": args.envs,
        "timesteps": int(args.timesteps), "agents_per_env": task.num_agents, "obs_size": task.obs_size,
        "act_sizes": list(task.act_sizes),
        "args": {k: v for k, v in vars(args).items() if k != "extra"}, "extra": args.extra,
        "created": time.time(), "status": "building",
    })
    print(f"run {name}: {run.dir}", flush=True)
    binary = build_trainer(task)
    run.save(status="launching games", trainer=str(binary))
    workers = max(1, min(args.workers, args.envs))
    counts = [args.envs // workers + (1 if w < args.envs % workers else 0) for w in range(workers)]
    Path("/dev/shm/warcraftsim").mkdir(parents=True, exist_ok=True)
    sockets = [f"/dev/shm/warcraftsim/bridge-{name}-{w}.sock" for w in range(workers)]
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    readies = [ctx.Event() for _ in range(workers)]
    slot, _slot_lock = claim_slot()
    games = "train" if slot == 0 else f"train{slot}_"
    procs = [ctx.Process(target=run_worker, daemon=True, args=(
        task.name, counts[w], str(run.dir), sockets[w], args.record_every if w == 0 else 0,
        args.video_every if w == 0 else 0, f"{games}{w}-", w, readies[w], stop_event)) for w in range(workers)]
    trainer = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if trainer and trainer.poll() is None:
            trainer.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        t0 = time.time()
        for p in procs:
            p.start()
        for p, ready in zip(procs, readies):
            while not ready.wait(1):
                if not p.is_alive():
                    raise RuntimeError(f"bridge worker {procs.index(p)} failed to start its games")
        print(f"bridge: {args.envs} games in {workers} workers ready in {time.time() - t0:.0f}s", flush=True)
        env = dict(os.environ, WC3_BRIDGE=";".join(sockets), PUFFER_JSONL=str(run.dir / "train.jsonl"))
        cmd = [str(binary), *trainer_args(args, args.envs, task.num_agents), f"--base.checkpoint_dir={run.dir / 'checkpoints'}",
               f"--base.log_dir={run.dir / 'logs'}"]
        run.save(status="training", started=time.time(), command=cmd)
        with open(run.dir / "trainer.log", "wb") as log:
            trainer = subprocess.Popen(cmd, cwd=PUFFER_BUILD, env=env, stdout=log, stderr=subprocess.STDOUT)
            code = trainer.wait()
        status = "stopped" if stopping else ("finished" if code == 0 else f"failed (exit {code})")
        run.save(status=status, finished=time.time())
        print(f"run {name}: {status}", flush=True)
        return 0 if code == 0 or stopping else 1
    except Exception as e:
        run.save(status=f"failed: {e}", finished=time.time())
        raise
    finally:
        stop_event.set()
        for p in procs:
            p.join(30)
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    sys.exit(main())
