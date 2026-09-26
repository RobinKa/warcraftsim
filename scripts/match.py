"""Head-to-head matches between two players in self-play mirror games (general orders).

    python scripts/match.py runs/genft-1/checkpoints/X.pt runs/genleague/checkpoints/Y.pt --episodes 100
    python scripts/match.py runs/genft-1/checkpoints/X.pt script:pull35 --games 4

A player is a torch-trainer checkpoint (.pt, played in numpy) or a script (script:noop,
script:focus, script:pull35; warcraftsim/rl/league.py). Each game swaps sides every episode, so
neither player keeps the same side. Prints player A's win rate against B (draws count half).
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from warcraftsim.puffer.tasks import get_task, task_spec
from warcraftsim.rl.league import Scripts
from warcraftsim.rl.numpy_model import load


class Player:
    def __init__(self, spec: str, task, rng):
        self.name, self.rng = spec, rng
        self.script = spec.split(":", 1)[1] if spec.startswith("script:") else None
        self.net = None if self.script else load(spec)
        self.scripts = Scripts(task_spec(task)) if self.script else None
        self.h = None

    def reset(self) -> None:
        self.h = self.net.initial_state(1) if self.net else None

    def act(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.script:
            return self.scripts.act(self.script, obs[None])[0]
        a, _, _, self.h, _ = self.net.step(obs[None], self.h, mask[None], rng=self.rng)
        return a[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--task", default="mirror_mix_gen_self_hp400")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--games", type=int, default=4)
    args = ap.parse_args()
    task = get_task(args.task)
    assert task.num_agents == 2, "a self-play task"

    def run(g: int) -> Counter:
        rng = np.random.default_rng(g)
        players = [Player(args.a, task, rng), Player(args.b, task, rng)]
        env = task.make_env(f"match{g}")
        env.setup.window = (320, 240)
        env.setup.step_seconds = 0.5
        c: Counter = Counter()
        try:
            for ep in range(args.episodes // args.games):
                side_of_a = (ep + g) % 2  # swap sides every episode
                seats = [players[0], players[1]] if side_of_a == 0 else [players[1], players[0]]
                obs, _ = task.reset(env)
                for p in seats:
                    p.reset()
                done = False
                while not done:
                    masks = task.action_mask(env)
                    acts = [seats[s].act(obs[s], masks[s]) for s in range(2)]
                    obs, _, done, _, outcomes = task.step(env, acts)
                o = outcomes[side_of_a]
                c["a_wins" if o > 0.5 else "b_wins" if o < -0.5 else "draws"] += 1
        finally:
            env.close()
        return c

    total: Counter = Counter()
    with ThreadPoolExecutor(args.games) as ex:
        for c in ex.map(run, range(args.games)):
            total += c
    n = sum(total.values())
    score = (total["a_wins"] + 0.5 * total["draws"]) / max(n, 1)
    print(f"{args.a}  vs  {args.b}: {score:.0%} for A over {n} episodes "
          f"(A wins {total['a_wins']}, B wins {total['b_wins']}, draws {total['draws']})", flush=True)


if __name__ == "__main__":
    main()
