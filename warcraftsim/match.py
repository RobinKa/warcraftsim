"""Match runs: two players against each other in self-play mirror games, with a replay video of
every episode (or every k-th), shown by the dashboard like any run.

    python -m warcraftsim.match genleague3 genft-1 --episodes 20 --name final-vs-specialist
    python -m warcraftsim.match runs/genleague3/checkpoints/X.pt script:amove --task mirror_mix_gen_self_hp400
    python -m warcraftsim.match genleague3 micro:pull35 --videos 5

A player is:
* a run name (its latest checkpoint), bc/<dataset> (its fitted policy), or a checkpoint file:
  torch-trainer checkpoints (.pt, played in numpy) and PufferLib ones (.bin, their task's layout);
* script:NAME: the league's scripts from observations (general orders: noop, focus, pull35, amove);
* micro:POLICY: the scripted policies of warcraftsim/agents/micro.py (any action space).

The players swap sides every episode. A recorded episode starts in a fresh game process (a replay
re-simulates everything its process played) and is rendered in the background with both sides'
policy panels. The run's episodes log A's outcome (+1: A won).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .puffer.bridge import combat_stats
from .puffer.policy import PolicyOutput, PufferPolicy
from .puffer.tasks import Task, describe_spaces, get_task, task_spec
from .puffer.train import RUNS_DIR, Run, _git


def resolve(spec: str) -> tuple[str, Path | None]:
    """A player spec -> (display name, checkpoint file or None for scripts)."""
    if spec.startswith(("script:", "micro:")):
        return spec, None
    path = Path(spec)
    if path.is_file():
        run = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name
        steps = path.stem.split("-")[-1]
        return (f"{run}@{int(steps) / 1e6:.2f}M" if steps.isdigit() else f"{run}/{path.name}"), path
    d = RUNS_DIR / spec
    if spec.startswith("bc/"):
        for name in ("policy.pt", "policy.bin"):
            if (d / name).exists():
                return spec, d / name
    found = sorted([*(d / "checkpoints").rglob("*.pt"), *(d / "checkpoints").rglob("*.bin")],
                   key=lambda p: p.stat().st_mtime)
    found = [f for f in found if f.stem.isdigit()]
    if not found:
        raise SystemExit(f"no checkpoint for player {spec!r} (a run name, bc/<dataset>, a file, script:, micro:)")
    ck = found[-1]
    return f"{spec}@{int(ck.stem) / 1e6:.2f}M", ck


class Player:
    def __init__(self, spec: str, task: Task, rng: np.random.Generator):
        self.spec, self.task, self.rng = spec, task, rng
        self.name, self.path = resolve(spec)
        self.kind = spec.split(":", 1)[0] if self.path is None else self.path.suffix.lstrip(".")
        if self.kind == "script":
            from .rl.league import Scripts
            self.scripts = Scripts(task_spec(task))
        elif self.kind in ("pt", "npz"):
            from .rl.numpy_model import load
            self.net = load(self.path).adapt(task_spec(task))
        elif self.kind == "bin":  # the network's size from its run (checkpoints sit below runs/<run>/)
            run_json = next((d / "run.json" for d in self.path.parents if (d / "run.json").exists()), None)
            args = json.loads(run_json.read_text()).get("args", {}) if run_json else {}
            self.pol = PufferPolicy(self.path, task.obs_size, task.act_sizes, hidden=args.get("hidden", 128),
                                    layers=args.get("layers", 2))
        self.state = None

    def reset(self) -> None:
        self.state = ({} if self.kind == "micro" else self.net.initial_state(1) if self.kind in ("pt", "npz")
                      else self.pol.initial_state() if self.kind == "bin" else None)

    def act(self, obs: np.ndarray, mask: np.ndarray, side_env) -> np.ndarray:
        if self.kind == "script":
            return self.scripts.act(self.spec.split(":", 1)[1], obs[None])[0]
        if self.kind == "micro":
            from .agents.micro import micro_action
            k = self.task.num_atns // self.task.group_size
            return micro_action(self.spec.split(":", 1)[1], side_env, self.state, k).ravel()
        if self.kind in ("pt", "npz"):
            a, _, _, self.state, _ = self.net.step(obs[None], self.state, mask[None], rng=self.rng)
            return a[0]
        dec = self.pol.step(obs, self.state)
        a, at = [], 0
        for n in self.task.act_sizes:
            logits = dec[at:at + n].astype(np.float64)
            if mask[at:at + n].any():
                logits = np.where(mask[at:at + n] > 0, logits, -np.inf)
            at += n
            p = np.exp(logits - logits.max())
            a.append(int(self.rng.choice(n, p=p / p.sum())))
        return np.asarray(a)

    def outputs(self, obs: np.ndarray, masks: np.ndarray, actions: np.ndarray) -> PolicyOutput:
        """What it thought over an episode (the video panel); scripts: their choices, certain."""
        if self.kind in ("pt", "npz"):
            from .rl.numpy_model import episode_outputs
            values, probs, ent = episode_outputs(self.net, obs, masks, actions)
            return PolicyOutput(values=values, probs=probs, entropy=ent)
        if self.kind == "bin":
            return self.pol.run(obs, masks)
        T = len(actions)
        probs = []
        for h, n in enumerate(self.task.act_sizes):
            one = np.zeros((T, n))
            one[np.arange(T), np.clip(actions[:, h].astype(int), 0, n - 1)] = 1.0
            probs.append(one)
        return PolicyOutput(values=np.full(T, np.nan), probs=probs, entropy=np.zeros((T, len(self.task.act_sizes))))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="player A")
    ap.add_argument("b", help="player B")
    ap.add_argument("--task", default="mirror_mix_gen_self_hp400", help="a self-play task (2 agents per game)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--games", type=int, default=4, help="games played at once")
    ap.add_argument("--videos", type=int, default=1, help="a video of every k-th episode (0: none; 1: all)")
    ap.add_argument("--render-workers", type=int, default=2, help="videos rendered at once")
    ap.add_argument("--step-seconds", type=float, default=0.5)
    ap.add_argument("--name", help="the run's name (default: match-A-vs-B-<time>)")
    ap.add_argument("--note", action="append", default=[])
    args = ap.parse_args(argv)
    task = get_task(args.task)
    if task.num_agents != 2:
        raise SystemExit(f"{args.task}: not a self-play task (two agents per game)")
    rng0 = np.random.default_rng()
    names = [resolve(args.a)[0], resolve(args.b)[0]]
    name = args.name or f"match-{names[0].split('@')[0]}-vs-{names[1].split('@')[0]}-{time.strftime('%H%M%S')}".replace(":", "_").replace("/", "_")
    run_dir = RUNS_DIR / name
    run = Run(run_dir, {
        "name": name, "kind": "match", "task": task.name, "players": {"A": {"spec": args.a, "name": names[0]},
                                                                      "B": {"spec": args.b, "name": names[1]}},
        "description": f"Match: A {names[0]} against B {names[1]} ({task.name}; they swap sides every episode).",
        "envs": args.games, "timesteps": args.episodes, "created": time.time(), "status": "playing",
        "args": {k: v for k, v in vars(args).items() if k != "note"}, "spaces": describe_spaces(task),
        "launch": {"command": shlex.join(["python", "-m", "warcraftsim.match", *(argv if argv is not None else sys.argv[1:])]),
                   "cwd": os.getcwd(), "git": _git(), "time": time.time()}})
    if args.note:
        (run_dir / "notes.md").write_text("\n".join(args.note) + "\n")
    (run_dir / "replays").mkdir(exist_ok=True)
    print(f"match {name}: A {names[0]} vs B {names[1]}, {args.episodes} episodes -> {run_dir}", flush=True)

    lock = threading.Lock()
    counter = [0]
    score = {"a": 0, "b": 0, "draw": 0}
    renders = ThreadPoolExecutor(max(1, args.render_workers))
    pending = []

    def render(replay: Path, episode: int, a_side: int, outcome_a: float, ret_a: float, g: int, setup) -> None:
        from .overlay import EpisodeOverlay
        from .video import render_replay
        try:
            trace = dict(np.load(replay.with_suffix(".steps.npz")))
            seats = [None, None]
            seats[a_side], seats[1 - a_side] = Player(args.a, task, rng0), Player(args.b, task, rng0)
            outputs = [seats[s].outputs(trace["obs"][:, s], trace["masks"][:, s], trace["actions"][:, s]) for s in range(2)]
            side_names = [names[0] if s == a_side else names[1] for s in range(2)]
            overlay = EpisodeOverlay(task, trace, outputs, title=f"{name} · episode {episode}", agent_names=side_names)
            out = render_replay(setup, replay, run_dir / "videos" / f"{replay.stem}.mp4", name=f"{name[:28]}-r{episode}",
                                overlay=overlay)
            with lock, open(run_dir / "media.jsonl", "a") as f:
                f.write(json.dumps({"time": time.time(), "kind": "video", "file": str(out.relative_to(run_dir)),
                                    "episode": episode, "outcome": outcome_a, "return": round(ret_a, 4),
                                    "a_side": a_side}) + "\n")
        except Exception as e:  # noqa: BLE001 (one video fewer)
            print(f"match: video of episode {episode} not rendered: {e}", flush=True)

    def play(g: int) -> None:
        rng = np.random.default_rng(g)
        players = [Player(args.a, task, rng), Player(args.b, task, rng)]
        env = task.make_env(f"{name}-g{g}"[:40])
        env.setup.window = (320, 240)
        env.setup.step_seconds = args.step_seconds
        log = open(run_dir / f"episodes-{g}.jsonl", "a")
        try:
            while True:
                with lock:
                    if counter[0] >= args.episodes:
                        break
                    counter[0] += 1
                    episode = counter[0]
                record = args.videos > 0 and (episode - 1) % args.videos == 0
                a_side = (episode - 1) % 2  # swap sides every episode
                seats = [players[0], players[1]] if a_side == 0 else [players[1], players[0]]
                # a recorded episode starts in a fresh process: its replay then holds just this episode
                obs, info = task.reset(env, options={"relaunch": True} if record else None)
                first = info.get("obs") if isinstance(info, dict) else None
                for p in seats:
                    p.reset()
                masks = task.action_mask(env)
                trace = {"obs": [np.stack(obs)], "actions": [], "rewards": [], "masks": [np.stack(masks)]}
                ret = np.zeros(2)
                done, steps = False, 0
                while not done:
                    acts = [np.asarray(seats[s].act(obs[s], masks[s], env.sides[s]), np.int64) for s in range(2)]
                    obs, rew, done, info, outcomes = task.step(env, acts)
                    ret += rew
                    steps += 1
                    trace["actions"].append(np.stack(acts))
                    trace["rewards"].append(np.asarray(rew, np.float32))
                    if not done:
                        masks = task.action_mask(env)
                        trace["obs"].append(np.stack(obs))
                        trace["masks"].append(np.stack(masks))
                last = info.get("obs") if isinstance(info, dict) else None
                outcome_a = float(outcomes[a_side])
                row = {"time": time.time(), "episode": episode, "worker": g, "env": 0, "outcome": outcome_a,
                       "return": round(float(ret[a_side]), 4), "length": steps,
                       "game_time": last.game_time if last is not None else None, "a_side": a_side}
                combat = combat_stats(first, last, player=a_side)
                if combat:
                    row["combat"] = combat
                with lock:
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    score["a" if outcome_a > 0.5 else "b" if outcome_a < -0.5 else "draw"] += 1
                    print(f"episode {episode}: {'A' if outcome_a > 0.5 else 'B' if outcome_a < -0.5 else 'nobody'} wins "
                          f"(A on side {a_side}; score A {score['a']} B {score['b']} draws {score['draw']})", flush=True)
                if record:
                    try:
                        replay = env.game.instance.save_replay(run_dir / "replays" / f"g{g}-episode{episode:04d}.w3g")
                        arrays = {"obs": np.stack(trace["obs"]), "actions": np.stack(trace["actions"]),
                                  "rewards": np.stack(trace["rewards"]), "masks": np.stack(trace["masks"]),
                                  "outcomes": np.asarray(outcomes, np.float32)}
                        np.savez_compressed(replay.with_suffix(".steps.npz"), time=time.time(), **arrays)
                        pending.append(renders.submit(render, replay, episode, a_side, outcome_a,
                                                      float(ret[a_side]), g, env.setup))
                    except Exception as e:  # noqa: BLE001
                        print(f"match: replay of episode {episode} not saved: {e}", flush=True)
        finally:
            log.close()
            env.close()

    status = "finished"
    try:
        with ThreadPoolExecutor(args.games) as ex:
            list(ex.map(play, range(args.games)))
        run.save(status="rendering videos")
        for f in pending:
            f.result()
    except BaseException as e:
        status = f"failed: {e!r}"[:200]
        raise
    finally:
        renders.shutdown(wait=True)
        n = sum(score.values())
        run.save(status=status, finished=time.time(), score=score,
                 summary_line=f"A {names[0]} {score['a']} - {score['b']} B {names[1]} ({score['draw']} draws)")
        print(f"A {names[0]} {score['a']} - {score['b']} B {names[1]}, {score['draw']} draws "
              f"(A {(score['a'] + 0.5 * score['draw']) / max(n, 1):.0%})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
