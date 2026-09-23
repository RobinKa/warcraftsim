"""Environment throughput benchmark: N games of a training task with random actions, spread over
W processes like the training bridge. Reports env steps/s and where a step's time goes.

    python scripts/bench_env.py --task micro_mirror -n 16 -w 4 --seconds 60
    LP_NUM_THREADS=1 python scripts/bench_env.py ...   # the environment reaches the games (Wine)
    python scripts/bench_env.py ... --profile          # also the game-side split (shim profiler)

Per step, "game" is the wall time from sending the actions until the next observation file is
ready (simulation, rendering, harness I/O); "python" is everything else (observation parsing,
feature encoding, action encoding, thread scheduling under the GIL).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def _worker(w: int, task_name: str, n: int, warmup: float, seconds: float, setup_kw: dict, ready, go, out):
    import numpy as np

    from warcraftsim.puffer.tasks import get_task

    task = get_task(task_name)
    rng = np.random.default_rng(w)

    def make(i: int):
        env = task.make_env(f"bench{w}-{i}")
        for k, v in setup_kw.items():
            setattr(env.game.setup if hasattr(env.game, "setup") else env.setup, k, v)
        task.reset(env)
        return env

    with ThreadPoolExecutor(n) as pool:
        envs = list(pool.map(make, range(n)))
    ready.set()
    go.wait()

    def run(env):
        inst = env.game.instance
        t_end_warm = time.time() + warmup
        while time.time() < t_end_warm:
            _step(env)
        steps, game0, t0 = 0, inst.game_wall_total, time.time()
        while time.time() - t0 < seconds:
            _step(env)
            steps += 1
        return steps, time.time() - t0, inst.game_wall_total - game0, inst.speed

    def _step(env):
        acts = [np.array([rng.integers(0, k) for k in task.act_sizes], np.float32) for _ in range(task.num_agents)]
        done = task.step(env, acts)[2]
        if done:
            task.reset(env)

    with ThreadPoolExecutor(n) as pool:
        results = list(pool.map(run, envs))
    out.put((w, results))
    for env in envs:
        env.close()


def _kind(comm: str) -> str:
    return ("game" if comm.startswith("Warcraft") else "wineserver" if comm.startswith("wineserver")
            else "python" if comm.startswith("python") else "Xvfb" if comm == "Xvfb"
            else "wine other" if comm.endswith(".exe") or comm.startswith("wine") else "other")


def _cpu_snapshot() -> dict[tuple, tuple[str, str, float]]:
    """CPU seconds per thread: {(pid, tid): (process kind, thread kind, seconds)}."""
    tick = os.sysconf("SC_CLK_TCK")
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            kind = _kind(open(f"/proc/{pid}/comm").read().strip())
            for tid in os.listdir(f"/proc/{pid}/task"):
                stat = open(f"/proc/{pid}/task/{tid}/stat").read()
                tname = stat[stat.index("(") + 1:stat.rindex(")")]
                fields = stat[stat.rindex(")") + 2:].split()
                tkind = ("llvmpipe" if tname.startswith("llvmpipe") else "main+wine threads"
                         if tname.startswith("Warcraft") else tname)
                out[(pid, tid)] = (kind, tkind, (int(fields[11]) + int(fields[12])) / tick)
        except (OSError, ValueError):
            continue
    return out


def _cpu_delta(a: dict, b: dict, seconds: float) -> tuple[dict[str, float], dict[str, float]]:
    """Cores busy per process kind, and per thread kind inside the games (threads alive at the end)."""
    procs: dict[str, float] = {}
    threads: dict[str, float] = {}
    for key, (kind, tkind, cpu) in b.items():
        d = (cpu - a[key][2] if key in a else cpu) / seconds
        procs[kind] = procs.get(kind, 0.0) + d
        if kind == "game":
            threads[tkind] = threads.get(tkind, 0.0) + d
    return procs, threads


def _shim_profile(names: list[str]) -> dict[str, float]:
    """Average per-second totals from the games' shim logs (W3SIM_PROFILE=1)."""
    from warcraftsim import paths

    rows = []
    pat = re.compile(r"profile/s: updates=([\d.]+) \(([\d.]+) ms\) presents=([\d.]+) \(([\d.]+) ms\) "
                     r"sync_wait=([\d.]+) ms")
    for name in names:
        log = paths.RUNTIME_DIR / "instances" / name / "shim.log"
        if log.exists():
            for m in pat.finditer(log.read_text(errors="replace")):
                rows.append(tuple(float(x) for x in m.groups()))
    if not rows:
        return {}
    rows = rows[len(rows) // 3:]  # skip launch and warm-up
    cols = list(zip(*rows))
    return {"frames/s": statistics.mean(cols[2]), "updates/s": statistics.mean(cols[0]),
            "update ms/s": statistics.mean(cols[1]),
            "present ms/s": statistics.mean(cols[3]), "waiting for python ms/s": statistics.mean(cols[4])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="micro_mirror")
    ap.add_argument("-n", type=int, default=16, help="games")
    ap.add_argument("-w", type=int, default=4, help="worker processes")
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--warmup", type=float, default=10)
    ap.add_argument("--profile", action="store_true", help="shim profiler in every game (W3SIM_PROFILE=1)")
    ap.add_argument("--speed", type=float, help="fixed clock speed (default: adaptive)")
    ap.add_argument("--window", help="game window WxH (default: the setup's)")
    args = ap.parse_args()
    if args.profile:
        os.environ["W3SIM_PROFILE"] = "1"
    setup_kw = {"speed": args.speed} if args.speed else {}
    if args.window:
        setup_kw["window"] = tuple(int(v) for v in args.window.split("x"))
    counts = [args.n // args.w + (1 if i < args.n % args.w else 0) for i in range(args.w)]
    ctx = mp.get_context("spawn")
    readies, go, out = [ctx.Event() for _ in counts], ctx.Event(), ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(w, args.task, c, args.warmup, args.seconds, setup_kw, readies[w], go, out))
             for w, c in enumerate(counts)]
    t0 = time.time()
    for p in procs:
        p.start()
    for r in readies:
        r.wait()
    print(f"{args.n} games ready in {time.time() - t0:.0f}s; measuring {args.seconds:.0f}s after {args.warmup:.0f}s warm-up",
          flush=True)
    go.set()
    time.sleep(args.warmup + 1)  # CPU accounting over the measured window only
    cpu0, t_cpu = _cpu_snapshot(), time.time()
    time.sleep(max(args.seconds - 3, 1))
    cpu1, t_cpu = _cpu_snapshot(), time.time() - t_cpu
    results = [out.get() for _ in procs]
    for p in procs:
        p.join()
    flat = [r for _, rs in results for r in rs]
    rate = sum(s / t for s, t, _, _ in flat)
    steps = sum(s for s, _, _, _ in flat)
    game = sum(g for _, _, g, _ in flat) / max(steps, 1)
    wall = sum(t for _, t, _, _ in flat) / max(steps, 1)
    print(f"env steps/s: {rate:.0f} total, {rate / args.n:.1f} per game "
          f"(game time {rate * 0.25:.0f}x realtime total)")
    print(f"per step: {1000 * wall:.1f} ms wall = {1000 * game:.1f} ms in the game + {1000 * (wall - game):.1f} ms python")
    print(f"clock speeds: {sorted(round(s) for _, _, _, s in flat)}")
    used, thr = _cpu_delta(cpu0, cpu1, t_cpu)
    print(f"CPU (cores busy, of {os.cpu_count()}): " + ", ".join(f"{k} {v:.1f}" for k, v in
                                                          sorted(used.items(), key=lambda kv: -kv[1]) if v >= 0.05)
          + f"; total {sum(used.values()):.1f}")
    print("game process threads (cores): " + ", ".join(f"{k} {v:.2f}" for k, v in
                                                       sorted(thr.items(), key=lambda kv: -kv[1])[:8] if v >= 0.02))
    if args.profile:
        prof = _shim_profile([f"bench{w}-{i}" for w, c in enumerate(counts) for i in range(c)])
        print("game side (per game, per wall second): " + ", ".join(f"{k} {v:.0f}" for k, v in prof.items()))


if __name__ == "__main__":
    main()
