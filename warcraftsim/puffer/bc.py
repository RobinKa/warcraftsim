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

The dashboard shows a dataset as run bc/<name>: bc.json (status, commands, settings),
episodes-<game>.jsonl (the demonstrations: outcome, actions, combat, as training episodes),
fit.jsonl (per epoch: losses, accuracy, precision and recall per unit order), evals.jsonl
(`eval` results, also written to a training run's directory for its checkpoints) and notes.md.
`backfill` writes bc.json and the episode log of a dataset recorded before these existed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import paths
from .tasks import Task, action_summary, describe_spaces, get_task

BC_DIR = Path(os.environ.get("WARCRAFTSIM_RUNS", paths.REPO_ROOT / "runs")) / "bc"  # no run.json: not a run


def _make_env(task: Task, name: str, step_seconds: float):
    env = task.make_env(name)
    env.setup.window = (320, 240)
    env.setup.step_seconds = step_seconds
    return env


def _play(task: Task, env, choose, stats: dict | None = None):
    """One episode; choose(obs_flat, env) -> flat action. Returns per step the observations,
    actions, rewards, which own unit slots were alive (their actions count), and the outcome.
    `stats` (a dict) gets the episode's statistics, as the training episode log has them."""
    from .bridge import combat_stats

    obs, info = task.reset(env)
    first = info.get("obs") if isinstance(info, dict) else None
    o = obs[0]
    obs_l, act_l, rew_l, live_l, mask_l = [], [], [], [], []
    counts: Counter = Counter()
    while True:
        if task.action_mask is not None:  # before choose: the state the action is chosen in
            mask_l.append(task.action_mask(env)[0])
        a = np.asarray(choose(o, env), np.int64).ravel()
        obs_l.append(o)
        act_l.append(a)
        own = env._own
        live_l.append([i < len(own) and own[i] is not None for i in range(task.num_atns // task.group_size)])
        if stats is not None and task.action_stats is not None:
            counts.update(task.action_stats(env, [a]))
        obs, rewards, done, info, outcomes = task.step(env, [a])
        rew_l.append(rewards[0])
        o = obs[0]
        if done:
            if stats is not None:
                last = info.get("obs") if isinstance(info, dict) else None
                stats.update({"length": len(rew_l), "return": round(float(sum(rew_l)), 4), "outcome": outcomes[0],
                              "game_time": last.game_time if last is not None else None})
                if counts:
                    stats["act"] = action_summary(counts)
                combat = combat_stats(first, last)
                if combat:
                    stats["combat"] = combat
            return obs_l, act_l, rew_l, live_l, mask_l, outcomes[0]


class _Status:
    """bc.json of a dataset directory: what the dashboard shows as the run bc/<name>."""

    def __init__(self, d: Path):
        self.path = d / "bc.json"
        self._lock = threading.Lock()

    def update(self, **fields) -> dict:
        with self._lock:
            try:
                info = json.loads(self.path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                info = {"kind": "bc", "name": self.path.parent.name, "created": time.time()}
            for k, v in fields.items():
                if isinstance(v, dict) and isinstance(info.get(k), dict):
                    info[k] = {**info[k], **v}
                else:
                    info[k] = v
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(info, indent=2))
            tmp.replace(self.path)
            return info


def _command() -> str:
    return shlex.join(["python", "-m", "warcraftsim.puffer.bc", *sys.argv[1:]])


def _git() -> dict:
    from .train import _git as git
    return git()


def collect(task_name: str, policy: str, episodes: int, games: int, step_seconds: float, out: Path) -> Path:
    from ..agents.micro import micro_action

    task = get_task(task_name)
    out.mkdir(parents=True, exist_ok=True)
    max_units = task.num_atns // task.group_size
    per_game = [episodes // games + (i < episodes % games) for i in range(games)]
    t0 = time.time()
    for f in out.glob("episodes-*.jsonl"):  # a new collection replaces the old
        f.unlink()
    status = _Status(out)
    status.update(kind="bc", name=out.name, task=task_name, policy=policy, created=time.time(), status="collecting",
                  spaces=describe_spaces(task), obs_size=task.obs_size, act_sizes=list(task.act_sizes),
                  collect={"episodes": episodes, "games": games, "step_seconds": step_seconds, "command": _command(),
                           "git": _git(), "started": time.time()})
    done_eps = [0]
    lock = threading.Lock()

    def run(i: int) -> Counter:
        env = _make_env(task, f"bc{i}", step_seconds)
        results: Counter = Counter()
        obs, act, rew, live, masks, ends = [], [], [], [], [], []
        log = open(out / f"episodes-{i}.jsonl", "a")
        try:
            for ep in range(per_game[i]):
                state: dict = {}
                stats: dict = {}
                o, a, r, lv, m, outcome = _play(task, env, lambda _o, e: micro_action(policy, e, state, max_units), stats)
                with lock:
                    done_eps[0] += 1
                    log.write(json.dumps({"time": time.time(), "episode": done_eps[0], "game": i, **stats}) + "\n")
                    log.flush()
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
            log.close()
            env.close()
        return results

    total: Counter = Counter()
    try:
        with ThreadPoolExecutor(games) as ex:
            for c in ex.map(run, range(games)):
                total += c
    except BaseException as e:
        status.update(status=f"failed: {e!r}"[:200])
        raise
    n = sum(total.values())
    meta = {"task": task_name, "policy": policy, "episodes": n, "step_seconds": step_seconds,
            "obs_size": task.obs_size, "act_sizes": list(task.act_sizes), "reward_scale": task.reward_scale,
            "group_size": task.group_size, "detail_heads": {str(k): v for k, v in task.detail_heads.items()},
            "kind_names": list(task.head_labels[0]) if task.head_labels else None,
            "win_rate": total[1.0] / max(n, 1), "loss_rate": total[-1.0] / max(n, 1),
            "minutes": round((time.time() - t0) / 60, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    status.update(status="collected", collect={"finished": time.time(), "episodes_done": n,
                                               "win_rate": meta["win_rate"], "loss_rate": meta["loss_rate"]})
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
    status = _Status(data)
    status.update(status="fitting", fit={"epochs": epochs, "hidden": hidden, "layers": layers, "gamma": gamma, "lr": lr,
                                         "smoothing": smoothing, "out": str(out), "command": _command(),
                                         "trainer": shlex.join(cmd), "git": _git(), "started": time.time(),
                                         "finished": None})
    try:
        subprocess.run(cmd, check=True)
    except BaseException as e:
        status.update(status=f"fit failed: {e!r}"[:200])
        raise
    status.update(status="fitted", fit={"finished": time.time()})
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
    owner = _checkpoint_owner(checkpoint)
    if owner is not None:  # the dashboard shows it with the dataset or the run
        kinds = task.head_labels[0] if task.head_labels else ()
        row = {"time": time.time(), "task": task_name, "checkpoint": str(checkpoint), "episodes": n,
               "win_rate": res["win_rate"], "loss_rate": res["loss_rate"],
               "draw_rate": total[0.0] / max(n, 1), "greedy": greedy, "script_casts": script_casts,
               "forbid": [kinds[k] if k < len(kinds) else k for k in forbid], "step_seconds": step_seconds,
               "by_type": {t: [c[1.0], sum(c.values())] for t, c in by_type.items()}, "command": _command()}
        steps = Path(checkpoint).stem
        if steps.isdigit():
            row["steps"] = int(steps)
        try:  # a dataset's policy.bin is refitted in place: which fit this was
            fit_info = json.loads((owner / "bc.json").read_text()).get("fit") or {}
            if fit_info.get("epochs") and Path(checkpoint).resolve() == Path(fit_info.get("out", "")).resolve():
                row["fit_epochs"] = fit_info["epochs"]
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        with open(owner / "evals.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    print(f"{checkpoint} on {task_name} ({mode}): win {res['win_rate']:.0%} "
          f"({total[1.0]}/{n}), loss {total[-1.0]}, draw {total[0.0]}", flush=True)
    if by_type:
        print("win rate in episodes with the unit type: " + ", ".join(
            f"{t} {c[1.0] / sum(c.values()):.0%} ({sum(c.values())})"
            for t, c in sorted(by_type.items(), key=lambda kv: kv[1][1.0] / sum(kv[1].values()))), flush=True)
    return res


def _checkpoint_owner(checkpoint: Path) -> Path | None:
    """The dataset or run directory a checkpoint belongs to (runs/bc/<name>, runs/<run>)."""
    runs = BC_DIR.parent.resolve()
    try:
        parts = Path(checkpoint).resolve().relative_to(runs).parts
    except ValueError:
        return None
    if len(parts) >= 3 and parts[0] == "bc":
        return runs / "bc" / parts[1]
    if len(parts) >= 2 and (runs / parts[0] / "run.json").exists():
        return runs / parts[0]
    return None


def backfill(data: Path) -> None:
    """bc.json and the episode log for a dataset collected before they were written: episodes
    from the recorded rewards (the outcome is the sign of the last step's, which carries the
    ±1 × reward scale) and actions (the order kinds of live units)."""
    meta = json.loads((data / "meta.json").read_text())
    task = get_task(meta["task"])
    kinds = list(task.head_labels[0]) if task.head_labels else []
    meta.setdefault("kind_names", kinds)
    (data / "meta.json").write_text(json.dumps(meta, indent=2))
    g = meta.get("group_size", 1)
    created = min(f.stat().st_mtime for f in data.glob("game*.npz"))
    n = 0
    for f in sorted(data.glob("game*.npz")):
        i = int(f.stem[4:]) if f.stem[4:].isdigit() else 0
        with np.load(f) as d:
            act, rew, ends = d["act"], d["rew"], d["ends"]
            live = d["live"] if "live" in d else None
        rows, start = [], 0
        for end in ends:
            r = rew[start:end]
            c: Counter = Counter()
            for t in range(start, end):
                for u in range(act.shape[1] // g):
                    if live is None or (u < live.shape[1] and live[t, u]):
                        c["unit_steps"] += 1
                        c[kinds[act[t, u * g]] if act[t, u * g] < len(kinds) else str(act[t, u * g])] += 1
            last = float(r[-1]) if len(r) else 0.0
            outcome = 1.0 if last > 0.5 * meta.get("reward_scale", 1) else -1.0 if last < -0.5 * meta.get("reward_scale", 1) else 0.0
            n += 1
            rows.append({"time": created, "episode": n, "game": i, "length": int(end - start),
                         "return": round(float(r.sum()), 4), "outcome": outcome,
                         # the order kinds only: targets were not resolved when recording
                         "act": {k: v for k, v in action_summary(c).items() if k != "attack_invalid"}})
            start = end
        (data / f"episodes-{i}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    status = _Status(data)
    status.update(kind="bc", name=data.name, task=meta["task"], policy=meta["policy"], created=created,
                  status="fitted" if (data / "policy.bin").exists() else "collected", backfilled=True,
                  obs_size=meta.get("obs_size"), act_sizes=meta.get("act_sizes"),
                  collect={"episodes": meta["episodes"], "episodes_done": meta["episodes"], "win_rate": meta.get("win_rate"),
                           "loss_rate": meta.get("loss_rate"), "step_seconds": meta.get("step_seconds"),
                           "command": shlex.join(["python", "-m", "warcraftsim.puffer.bc", "collect", meta["task"],
                                                  "--policy", meta["policy"], "--episodes", str(meta["episodes"])])})
    if (data / "policy.bin").exists():
        status.update(fit={"out": str(data / "policy.bin"), "finished": (data / "policy.bin").stat().st_mtime})
    print(f"{data}: {n} episodes, bc.json", flush=True)


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
    b = sub.add_parser("backfill", help="bc.json and episode logs for datasets recorded before them")
    b.add_argument("data", type=Path, nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        collect(args.task, args.policy, args.episodes, args.games, args.step_seconds,
                args.out or BC_DIR / f"{args.task}-{args.policy}")
    elif args.cmd == "backfill":
        for d in args.data:
            backfill(d)
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
