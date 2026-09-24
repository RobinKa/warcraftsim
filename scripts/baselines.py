"""Scripted baselines on micro tasks: how winnable is a scenario?

    python scripts/baselines.py footmen2 footmen3v4 footmen4v5 --episodes 20 --games 4
    python scripts/baselines.py mirror_mix_hp400 --policies noop,pull35 --step-seconds 0.5

Policies: see warcraftsim/agents/micro.py (noop, focus, range, sticky, [base]pull<L>).
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from warcraftsim.agents.micro import micro_action as act
from warcraftsim.puffer.tasks import get_task


def run(job):
    task_name, policy, idx, episodes, step_seconds = job
    task = get_task(task_name)
    env = task.make_env(f"base-{idx}")
    env.setup.window = (320, 240)
    if step_seconds:
        env.setup.step_seconds = step_seconds
    max_units = task.num_atns // 3
    out = []
    try:
        for _ in range(episodes):
            env.reset()
            state: dict = {}
            done, steps = False, 0
            while not done:
                _, _, term, trunc, info = env.step(act(policy, env, state, max_units))
                done = term or trunc
                steps += 1
            o = info["obs"]
            left = sum(1 for u in o.units if u.alive and u.owner == 0)
            out.append((task.outcome(env, info), steps * env.setup.step_seconds, left))
    finally:
        env.close()
    return task_name, policy, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--policies", default="noop,focus,pull35,pull50")
    ap.add_argument("--episodes", type=int, default=20, help="per game")
    ap.add_argument("--games", type=int, default=2, help="per task and policy")
    ap.add_argument("--step-seconds", type=float, default=0, help="default: the task's (0.25 s)")
    args = ap.parse_args()
    jobs, k = [], 0
    for t in args.tasks:
        for p in args.policies.split(","):
            for _ in range(args.games):
                jobs.append((t, p, k, args.episodes, args.step_seconds))
                k += 1
    res: dict = {}
    with ThreadPoolExecutor(len(jobs)) as ex:
        for t, p, out in ex.map(run, jobs):
            res.setdefault((t, p), []).extend(out)
    for (t, p), out in res.items():
        c = Counter(o[0] for o in out)
        print(f"{t:18s} {p:8s} win {c[1.0] / len(out):4.0%} ({c[1.0]}/{len(out)})  loss {c[-1.0]}  draw {c[0.0]}  "
              f"episode {np.mean([o[1] for o in out]):4.1f} s  units left {np.mean([o[2] for o in out]):.1f}", flush=True)


if __name__ == "__main__":
    main()
