"""Scripted baselines on micro tasks: how winnable is a scenario?

    python scripts/baselines.py footmen2 footmen3v4 footmen4v5 --episodes 20 --games 4

Policies (per own unit, every step):
    noop      units fight on their own (auto-acquire)
    focus     everyone attacks the weakest enemy
    pull<L>   focus, but a unit below L% hit points that is being hit (lost hit points in the
              last second) and is not the healthiest walks away from the enemies until it is no
              longer being hit, then rejoins (the enemies switch to another target meanwhile)
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from warcraftsim.puffer.tasks import get_task


def act(policy: str, env, state: dict, max_units: int) -> np.ndarray:
    own, enemy = env._own, env._enemy
    a = np.zeros((max_units, 3), np.int64)
    if not enemy or policy == "noop":
        return a
    weakest = min(range(len(enemy)), key=lambda i: enemy[i].hp)
    ex = sum(e.x for e in enemy) / len(enemy)
    ey = sum(e.y for e in enemy) / len(enemy)
    healthiest = max(u.hp for u in own) if own else 0
    for i, u in enumerate(own[:max_units]):
        a[i] = (3, 0, weakest)
        if not policy.startswith("pull"):
            continue
        low = int(policy[4:]) / 100
        hist = state.setdefault(u.id, [u.hp] * 4)
        hit = u.hp < hist[0]
        hist.append(u.hp)
        del hist[0]
        if hit and u.hp < low * u.max_hp and u.hp < healthiest:
            ang = math.atan2(u.y - ey, u.x - ex)
            a[i] = (2, round(ang / (math.pi / 4)) % 8, 0)
    return a


def run(job):
    task_name, policy, idx, episodes = job
    task = get_task(task_name)
    env = task.make_env(f"base-{idx}")
    env.setup.window = (320, 240)
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
    args = ap.parse_args()
    jobs, k = [], 0
    for t in args.tasks:
        for p in args.policies.split(","):
            for _ in range(args.games):
                jobs.append((t, p, k, args.episodes))
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
