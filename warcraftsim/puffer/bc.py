"""Behavior cloning warm start: record a scripted policy on a task, fit the trainer's network to
it, and start PPO from the result instead of from a random policy.

    python -m warcraftsim.puffer.bc collect mirror_mix_sem_hp400 --policy pull35 --episodes 1500 --games 12
    python -m warcraftsim.puffer.bc fit runs/bc/mirror_mix_sem_hp400-pull35      # -> .../policy.bin
    python -m warcraftsim.puffer.bc eval mirror_mix_sem_hp400 runs/bc/mirror_mix_sem_hp400-pull35/policy.bin
    python -m warcraftsim.puffer.train --task mirror_mix_sem_hp400 --init-from runs/bc/.../policy.bin ...

`collect` plays the scripted policy (warcraftsim.agents.micro) in games like the trainer's and saves
what the policy saw and did: observations as the trainer gets them, actions, scaled rewards, and
the task's action masks (fit trains on the masked choices, as the trainer samples them).
`fit` trains PufferLib's default network on that (bc_train.py; it needs torch, and runs with
WC3_TORCH_PYTHON, default the first Python that has it). `eval` plays a checkpoint (sampled
actions, as in training) and reports its win rate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import paths
from .tasks import Task, get_task

BC_DIR = Path(os.environ.get("WARCRAFTSIM_RUNS", paths.REPO_ROOT / "runs")) / "bc"  # no run.json: not a run


def _make_env(task: Task, name: str, step_seconds: float):
    env = task.make_env(name)
    env.setup.window = (320, 240)
    env.setup.step_seconds = step_seconds
    return env


def _play(task: Task, env, choose):
    """One episode; choose(obs_flat, env) -> flat action. Returns per step the observations,
    actions, rewards, which own unit slots were alive (their actions count), and the outcome."""
    obs, _ = task.reset(env)
    o = obs[0]
    obs_l, act_l, rew_l, live_l, mask_l = [], [], [], [], []
    while True:
        if task.action_mask is not None:  # before choose: the state the action is chosen in
            mask_l.append(task.action_mask(env)[0])
        a = np.asarray(choose(o, env), np.int64).ravel()
        obs_l.append(o)
        act_l.append(a)
        own = env._own
        live_l.append([i < len(own) and own[i] is not None for i in range(task.num_atns // task.group_size)])
        obs, rewards, done, info, outcomes = task.step(env, [a])
        rew_l.append(rewards[0])
        o = obs[0]
        if done:
            return obs_l, act_l, rew_l, live_l, mask_l, outcomes[0]


def collect(task_name: str, policy: str, episodes: int, games: int, step_seconds: float, out: Path) -> Path:
    from ..agents.micro import micro_action

    task = get_task(task_name)
    out.mkdir(parents=True, exist_ok=True)
    max_units = task.num_atns // task.group_size
    per_game = [episodes // games + (i < episodes % games) for i in range(games)]
    t0 = time.time()

    def run(i: int) -> Counter:
        env = _make_env(task, f"bc{i}", step_seconds)
        results: Counter = Counter()
        obs, act, rew, live, masks, ends = [], [], [], [], [], []
        try:
            for ep in range(per_game[i]):
                state: dict = {}
                o, a, r, lv, m, outcome = _play(task, env, lambda _o, e: micro_action(policy, e, state, max_units))
                obs += o
                act += a
                rew += r
                live += lv
                masks += m
                ends.append(len(obs))
                results[outcome] += 1
                if (ep + 1) % 25 == 0 or ep + 1 == per_game[i]:  # keep what we have if a game dies later
                    extra = {"masks": np.asarray(masks, np.uint8)} if masks else {}
                    np.savez_compressed(out / f"game{i}.npz", obs=np.asarray(obs, np.float32),
                                        act=np.asarray(act, np.int16), rew=np.asarray(rew, np.float32),
                                        live=np.asarray(live, bool), ends=np.asarray(ends, np.int32), **extra)
        finally:
            env.close()
        return results

    total: Counter = Counter()
    with ThreadPoolExecutor(games) as ex:
        for c in ex.map(run, range(games)):
            total += c
    n = sum(total.values())
    meta = {"task": task_name, "policy": policy, "episodes": n, "step_seconds": step_seconds,
            "obs_size": task.obs_size, "act_sizes": list(task.act_sizes), "reward_scale": task.reward_scale,
            "group_size": task.group_size, "detail_heads": {str(k): v for k, v in task.detail_heads.items()},
            "win_rate": total[1.0] / max(n, 1), "loss_rate": total[-1.0] / max(n, 1),
            "minutes": round((time.time() - t0) / 60, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"collected {n} episodes of {policy} on {task_name} in {meta['minutes']} min: "
          f"win {meta['win_rate']:.0%}, loss {meta['loss_rate']:.0%} -> {out}", flush=True)
    return out


def _torch_python() -> str:
    env = os.environ.get("WC3_TORCH_PYTHON")
    if env:
        return env
    for py in (sys.executable, shutil.which("python3"), "/usr/bin/python3"):
        if py and subprocess.run([py, "-c", "import torch"], capture_output=True).returncode == 0:
            return py
    raise SystemExit("bc fit needs torch: install it or point WC3_TORCH_PYTHON at a Python that has it")


def fit(data: Path, out: Path | None, epochs: int, hidden: int, layers: int, gamma: float, lr: float,
        smoothing: float = 0.1) -> Path:
    out = out or data / "policy.bin"
    script = Path(__file__).with_name("bc_train.py")
    cmd = [_torch_python(), str(script), str(data), str(out), f"--epochs={epochs}", f"--hidden={hidden}",
           f"--layers={layers}", f"--gamma={gamma}", f"--lr={lr}", f"--smoothing={smoothing}"]
    subprocess.run(cmd, check=True)
    return out


def evaluate(task_name: str, checkpoint: Path, episodes: int, games: int, step_seconds: float,
             hidden: int, layers: int, greedy: bool = False, script_casts: bool = False,
             forbid: tuple[int, ...] = ()) -> dict:
    """`script_casts`: heroes cast what MicroEnv.scripted_cast picks instead of what the policy
    chose (a diagnostic: is the policy's casting what separates it from the scripts?). `forbid`:
    unit action kinds (first-head values) masked for every unit (does the policy need them?)."""
    from .policy import PufferPolicy

    task = get_task(task_name)
    pol = PufferPolicy(checkpoint, task.obs_size, task.act_sizes, hidden=hidden, layers=layers)
    per_game = [episodes // games + (i < episodes % games) for i in range(games)]

    by_type: dict[str, Counter] = {}  # unit type in the (mirror) composition -> outcomes

    def run(i: int) -> Counter:
        rng = np.random.default_rng(i)
        env = _make_env(task, f"bceval{i}", step_seconds)
        results: Counter = Counter()
        try:
            for _ in range(per_game[i]):
                state = pol.initial_state()
                comp: set[str] = set()

                def choose(o, env):
                    if not comp:
                        comp.update(u.type for u in getattr(env, "_own", ()) if u is not None)
                    dec = pol.step(o, state)
                    mask = task.action_mask(env)[0] if task.action_mask is not None else None
                    if forbid:
                        mask = np.ones(sum(task.act_sizes), np.uint8) if mask is None else mask.copy()
                        at = 0
                        for h, n in enumerate(task.act_sizes):
                            if h % task.group_size == 0:
                                mask[[at + k for k in forbid]] = 0
                                mask[at] = 1  # noop stays
                            at += n
                    a, at = [], 0
                    for n in task.act_sizes:
                        logits = dec[at:at + n].astype(np.float64)
                        if mask is not None and mask[at:at + n].any():  # as the trainer samples
                            logits = np.where(mask[at:at + n] > 0, logits, -np.inf)
                        at += n
                        if greedy:
                            a.append(int(np.argmax(logits)))
                        else:
                            p = np.exp(logits - logits.max())
                            a.append(int(rng.choice(n, p=p / p.sum())))
                    if script_casts:
                        g = task.group_size
                        for i, u in enumerate(env._own):
                            if u is not None and u.is_hero and g > 3:
                                choice = env.scripted_cast(u, env._own, env._enemy)
                                if choice is not None:
                                    a[i * g:i * g + 4] = [4, 0, choice[1], choice[0]]
                    return a

                *_, outcome = _play(task, env, choose)
                results[outcome] += 1
                for t in comp:
                    by_type.setdefault(t, Counter())[outcome] += 1
        finally:
            env.close()
        return results

    total: Counter = Counter()
    with ThreadPoolExecutor(games) as ex:
        for c in ex.map(run, range(games)):
            total += c
    n = sum(total.values())
    res = {"episodes": n, "win_rate": total[1.0] / max(n, 1), "loss_rate": total[-1.0] / max(n, 1),
           "by_type": {t: c[1.0] / sum(c.values()) for t, c in by_type.items()}}
    mode = ("greedy" if greedy else "sampled") + (", scripted casts" if script_casts else "") + (
        f", kinds {list(forbid)} forbidden" if forbid else "")
    print(f"{checkpoint} on {task_name} ({mode}): win {res['win_rate']:.0%} "
          f"({total[1.0]}/{n}), loss {total[-1.0]}, draw {total[0.0]}", flush=True)
    if by_type:
        print("win rate in episodes with the unit type: " + ", ".join(
            f"{t} {c[1.0] / sum(c.values()):.0%} ({sum(c.values())})"
            for t, c in sorted(by_type.items(), key=lambda kv: kv[1][1.0] / sum(kv[1].values()))), flush=True)
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="record a scripted policy")
    c.add_argument("task")
    c.add_argument("--policy", default="pull35")
    c.add_argument("--episodes", type=int, default=1500)
    c.add_argument("--games", type=int, default=12)
    c.add_argument("--step-seconds", type=float, default=0.5)
    c.add_argument("--out", type=Path, help="default: runs/bc/<task>-<policy>")
    f = sub.add_parser("fit", help="train the network on recorded episodes")
    f.add_argument("data", type=Path)
    f.add_argument("--out", type=Path, help="default: <data>/policy.bin")
    f.add_argument("--epochs", type=int, default=30)
    f.add_argument("--hidden", type=int, default=128)
    f.add_argument("--layers", type=int, default=2)
    f.add_argument("--gamma", type=float, default=0.99)
    f.add_argument("--lr", type=float, default=0.003)
    f.add_argument("--smoothing", type=float, default=0.1, help="label smoothing (keeps unused choices possible)")
    e = sub.add_parser("eval", help="play a checkpoint and report its win rate")
    e.add_argument("task")
    e.add_argument("checkpoint", type=Path)
    e.add_argument("--episodes", type=int, default=120)
    e.add_argument("--games", type=int, default=12)
    e.add_argument("--step-seconds", type=float, default=0.5)
    e.add_argument("--hidden", type=int, default=128)
    e.add_argument("--layers", type=int, default=2)
    e.add_argument("--greedy", action="store_true", help="most likely actions instead of sampling")
    e.add_argument("--script-casts", action="store_true", help="heroes cast by the scripted rule instead")
    e.add_argument("--forbid", default="", help="unit action kinds to mask for every unit, e.g. retreat")
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        collect(args.task, args.policy, args.episodes, args.games, args.step_seconds,
                args.out or BC_DIR / f"{args.task}-{args.policy}")
    elif args.cmd == "fit":
        fit(args.data, args.out, args.epochs, args.hidden, args.layers, args.gamma, args.lr, args.smoothing)
    else:
        task = get_task(args.task)
        kinds = task.head_labels[0] if task.head_labels else ()
        forbid = tuple(kinds.index(k) if k in kinds else int(k) for k in args.forbid.split(",") if k)
        evaluate(args.task, args.checkpoint, args.episodes, args.games, args.step_seconds, args.hidden,
                 args.layers, args.greedy, args.script_casts, forbid)


if __name__ == "__main__":
    main()
