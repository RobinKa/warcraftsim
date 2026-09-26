"""Which enemy a general-order policy attacks, compared with the weakest one.

    python scripts/target_choice.py mirror_mix_gen_hp400 runs/genft-1/checkpoints/X.pt --episodes 60 --games 2

Plays a torch-trainer checkpoint (numpy) and, for every attack order, compares the target with the
enemy alternatives: hit points, damage per second, DPS per hit point left (threat), hero, distance.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from warcraftsim.data.objects import combat_stats
from warcraftsim.env import GENERAL_KINDS
from warcraftsim.puffer.tasks import get_task
from warcraftsim.rl.numpy_model import load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--games", type=int, default=2)
    args = ap.parse_args()
    task, net, stats = get_task(args.task), load(args.checkpoint), combat_stats()
    attack = GENERAL_KINDS.index("attack")

    def run(g: int) -> Counter:
        c: Counter = Counter()
        rng = np.random.default_rng(g)
        env = task.make_env(f"tgt{g}")
        env.setup.window = (320, 240)
        env.setup.step_seconds = 0.5
        try:
            for _ in range(args.episodes // args.games):
                obs, _ = task.reset(env)
                h, done = net.initial_state(1), False
                while not done:
                    m = task.action_mask(env)[0]
                    a, _, _, h, _ = net.step(obs[0][None], h, m[None], rng=rng)
                    a = a[0].reshape(net.k, -1)
                    live = [(j, e) for j, e in enumerate(env._enemy) if e is not None]
                    if len(live) >= 2:
                        weakest = min(live, key=lambda je: je[1].hp)[0]
                        for i, u in enumerate(env._own):
                            if u is None or a[i, 0] != attack:
                                continue
                            j = int(a[i, 3]) - env.max_own
                            t = env._enemy[j] if 0 <= j < len(env._enemy) else None
                            if t is None:
                                continue
                            c["attacks"] += 1
                            if j == weakest:
                                c["weakest"] += 1
                                continue
                            c["other"] += 1
                            def dps(e):
                                s_ = stats.get(e.type)
                                return s_.dps if s_ else 0.0
                            threat = {jj: dps(e) / max(e.hp, 1) for jj, e in live}
                            dist = {jj: e.dist(u.x, u.y) for jj, e in live}
                            c["other: highest threat (DPS / HP)"] += j == max(threat, key=threat.get)
                            c["other: highest DPS"] += j == max(live, key=lambda je: dps(je[1]))[0]
                            c["other: a hero"] += t.is_hero
                            c["other: nearest to the attacker"] += j == min(dist, key=dist.get)
                            c["other: lowest HP share"] += j == min(live, key=lambda je: je[1].hp / je[1].max_hp)[0]
                    obs, _, done, _, _ = task.step(env, [a.ravel()])
        finally:
            env.close()
        return c

    total: Counter = Counter()
    with ThreadPoolExecutor(args.games) as ex:
        for c in ex.map(run, range(args.games)):
            total += c
    n, o = total["attacks"], max(total["other"], 1)
    print(f"attacks with 2+ enemies alive: {n}; on the weakest (lowest HP): {total['weakest'] / n:.0%}, on another: {total['other'] / n:.0%}")
    for key in sorted(k for k in total if k.startswith("other:")):
        print(f"  {key[7:]}: {total[key] / o:.0%} of the others")


if __name__ == "__main__":
    main()
