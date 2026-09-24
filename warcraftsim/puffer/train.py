"""Train a PufferLib 5.0 policy on a Warcraft III task.

    python -m warcraftsim.puffer.train --task micro --envs 24 --timesteps 2_000_000
    python -m warcraftsim.puffer.train --task footmen2 --name f2-lr --timesteps 1e6 \
        --sweep "--lr 0.003" --sweep "--lr 0.01" --sweep "--lr 0.01 --train.gae_lambda=0.95"

A sweep launches the games once and trains one run per --sweep after another (runs
<name>-1, <name>-2, ...; the dashboard shows them like any other run).

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
import shlex
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


def default_buffers(envs: int) -> int:
    """Two games per buffer. Each buffer's thread steps its games together and waits for the
    slowest before its next model call, so small buffers keep games busy (24 games: 2 buffers
    2150 SPS, 12 buffers 2920, 24 buffers 2790; scripts/bench_train.py)."""
    return max(1, envs // 2) if envs % 2 == 0 else envs


def trainer_args(args, envs: int, agents_per_env: int = 1) -> list[str]:
    horizon = args.horizon
    agents = envs * agents_per_env  # PufferLib creates environments until it has this many agents
    batch = agents * horizon
    minibatch = min(args.minibatch or batch, batch)
    buffers = args.buffers or default_buffers(envs)
    return [
        "train",
        f"--vec.total_agents={agents}", f"--vec.num_buffers={buffers}",
        f"--vec.num_threads={max(envs, buffers)}",
        f"--train.total_timesteps={int(args.timesteps)}", f"--train.horizon={horizon}",
        f"--train.minibatch_size={minibatch}", f"--train.replay_ratio={args.replay_ratio}",
        f"--train.learning_rate={args.lr}",
        f"--train.ent_coef={args.ent_coef}", f"--train.gamma={args.gamma}",
        f"--policy.hidden_size={args.hidden}", f"--policy.num_layers={args.layers}",
        "--base.eval_episodes=0", f"--base.checkpoint_interval={args.checkpoint_interval}",
        *args.extra,
    ]


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="micro")
    ap.add_argument("--envs", type=int, default=24, help="parallel games")
    ap.add_argument("--workers", type=int, default=6, help="bridge processes (each runs envs/workers games)")
    ap.add_argument("--timesteps", type=float, default=1_000_000)
    ap.add_argument("--name", help="run name (default: task + timestamp); with --sweep, the runs' prefix")
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--minibatch", type=int, default=0, help="default: the whole batch (envs * horizon)")
    ap.add_argument("--replay-ratio", type=float, default=1.0,
                    help="updates per epoch = replay_ratio * batch / minibatch")
    ap.add_argument("--buffers", type=int, default=0, help="trainer buffers (default: one per two games)")
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
    ap.add_argument("--step-seconds", type=float, default=0, help="game time per step (default: the task's, 0.25)")
    ap.add_argument("--init-from", help="start from a checkpoint: a .bin file, or a run name (its latest)")
    ap.add_argument("--sweep", action="append", default=[], metavar="OPTIONS",
                    help="a sweep: one run per --sweep, each with these options on top of the others "
                         "(e.g. --sweep '--lr 0.01' --sweep '--lr 0.003 --train.gae_lambda=0.95'); "
                         "the games are launched once and serve the runs one after another")
    return ap


def parse(ap: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    """Our options, plus any --section.key=value PufferLib options (kept in args.extra)."""
    joined, i = [], 0
    while i < len(argv):  # --sweep "--x.y=1": argparse would take the value for an option
        if argv[i] == "--sweep" and i + 1 < len(argv):
            joined.append(f"--sweep={argv[i + 1]}")
            i += 2
        else:
            joined.append(argv[i])
            i += 1
    args, unknown = ap.parse_known_args(joined)
    bad = [u for u in unknown if not (u.startswith("--") and "." in u.split("=")[0] and "=" in u)]
    if bad:
        ap.error(f"unrecognized arguments: {' '.join(bad)}")
    args.extra = unknown
    return args


def _resolve_init(args) -> Path | None:
    if not args.init_from:
        return None
    init = Path(args.init_from)
    if not init.is_file():
        found = sorted((RUNS_DIR / args.init_from / "checkpoints").rglob("*.bin"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise SystemExit(f"--init-from: no checkpoint file or run with checkpoints named {args.init_from!r}")
        init = found[-1]
    args.extra = [*args.extra, f"--base.load_model_path={init.resolve()}"]
    return init


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = make_parser()
    base = parse(ap, argv)
    task = get_task(base.task)
    group = base.name or f"{task.name}-{datetime.now():%Y%m%d-%H%M%S}"
    configs = base.sweep or [""]
    plans = []  # (args, run name, sweep info)
    for i, cfg in enumerate(configs):
        args = parse(ap, [*argv, *shlex.split(cfg)])
        if (args.task, args.envs, args.workers, args.step_seconds) != (base.task, base.envs, base.workers,
                                                                       base.step_seconds):
            ap.error("--sweep options cannot change --task, --envs, --workers or --step-seconds (the runs share the games)")
        name = f"{group}-{i + 1}" if base.sweep else group
        plans.append((args, name, {"group": group, "index": i + 1, "of": len(configs), "options": cfg}
                      if base.sweep else None))

    runs = []
    for args, name, sweep in plans:
        init = _resolve_init(args)
        runs.append(Run(RUNS_DIR / name, {
            "name": name, "task": task.name, "description": task.description, "envs": args.envs,
            "timesteps": int(args.timesteps), "agents_per_env": task.num_agents, "obs_size": task.obs_size,
            "act_sizes": list(task.act_sizes),
            "args": {k: v for k, v in vars(args).items() if k not in ("extra", "sweep")}, "extra": args.extra,
            "created": time.time(), "status": "queued" if sweep and sweep["index"] > 1 else "building",
            "init_from": str(init) if init else None, "sweep": sweep,
        }))
        print(f"run {name}: {RUNS_DIR / name}" + (f"  [{sweep['options']}]" if sweep else ""), flush=True)
    binary = build_trainer(task)
    first = runs[0]
    first.save(status="launching games", trainer=str(binary))
    workers = max(1, min(base.workers, base.envs))
    counts = [base.envs // workers + (1 if w < base.envs % workers else 0) for w in range(workers)]
    Path("/dev/shm/warcraftsim").mkdir(parents=True, exist_ok=True)
    sockets = [f"/dev/shm/warcraftsim/bridge-{group}-{w}.sock" for w in range(workers)]
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    readies = [ctx.Event() for _ in range(workers)]
    controls = [ctx.Queue() for _ in range(workers)]
    acks = ctx.Queue()
    slot, _slot_lock = claim_slot()
    games = "train" if slot == 0 else f"train{slot}_"
    a0 = plans[0][0]
    procs = [ctx.Process(target=run_worker, daemon=True, args=(
        task.name, counts[w], str(first.dir), sockets[w], a0.record_every if w == 0 else 0,
        a0.video_every if w == 0 else 0, f"{games}{w}-", w, readies[w], stop_event, controls[w], acks,
        base.step_seconds or None))
        for w in range(workers)]
    trainer = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if trainer and trainer.poll() is None:
            trainer.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    current = first
    code = 0
    try:
        t0 = time.time()
        for p in procs:
            p.start()
        for p, ready in zip(procs, readies):
            while not ready.wait(1):
                if not p.is_alive():
                    raise RuntimeError(f"bridge worker {procs.index(p)} failed to start its games")
        print(f"bridge: {base.envs} games in {workers} workers ready in {time.time() - t0:.0f}s", flush=True)
        for (args, name, sweep), run in zip(plans, runs):
            if stopping:
                run.save(status="stopped (sweep ended)")
                continue
            current = run
            if run is not first:  # point the bridge workers at this run
                for w, q in enumerate(controls):
                    q.put(("run", str(run.dir), args.record_every if w == 0 else 0, args.video_every if w == 0 else 0))
                for _ in controls:
                    acks.get(timeout=60)
            env = dict(os.environ, WC3_BRIDGE=";".join(sockets), PUFFER_JSONL=str(run.dir / "train.jsonl"))
            # OpenMP threads that finished their game spin at the barrier by default: with 2 buffers
            # the trainer burned 10.7 cores (more than 24 games); passive, 2.5
            env.setdefault("OMP_WAIT_POLICY", "passive")
            cmd = [str(binary), *trainer_args(args, args.envs, task.num_agents),
                   f"--base.checkpoint_dir={run.dir / 'checkpoints'}", f"--base.log_dir={run.dir / 'logs'}"]
            run.save(status="training", started=time.time(), command=cmd, trainer=str(binary))
            with open(run.dir / "trainer.log", "wb") as log:
                trainer = subprocess.Popen(cmd, cwd=PUFFER_BUILD, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = trainer.wait()
            status = "stopped" if stopping else ("finished" if code == 0 else f"failed (exit {code})")
            run.save(status=status, finished=time.time())
            print(f"run {name}: {status}", flush=True)
        return 0 if code == 0 or stopping else 1
    except Exception as e:
        current.save(status=f"failed: {e}", finished=time.time())
        raise
    finally:
        stop_event.set()
        for p in procs:
            p.join(30)
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    sys.exit(main())
