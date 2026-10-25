"""A remote evaluation worker: a self-play run's evaluation games (selfplay --eval-games: the real game
against the untaxed built-in AI, each matchup played by the current policy and every --eval-checkpoints
in turn), played on another machine for a run that trains elsewhere, with this machine's own inference.
A verdict needs ~300 games a policy, and the run's own 8 evaluation games took 1.5-2 hours for it.

    python -m warcraftsim.fullgame.evalworker --run fgself-76 --games 16 [--processes 8] [--host laptop-wsl]

scripts/remote_eval.py (on the run's machine) starts it, mirrors the run's code there, sends it the run's
run.json and vocab.json, its newest checkpoint every few minutes and the evaluation checkpoints, and takes
back runs/<run>/episodes-remote.jsonl. What it reads, all in runs/<run>/ here:
* run.json: the run's settings (its "config", vars(args) of its last launch: map, handicap, time limit,
  features, game cuts, Wine, races, evaluation AI and checkpoints, step settings).
* checkpoints/<steps>.pt: the newest is the policy "current" (published here as current.pt, the file the
  actors' Nets reads, its version the checkpoint's update): a few minutes behind the learner's live weights.
* the --eval-checkpoints at their paths relative to the repository (a path under the run machine's
  repository is taken relative to it).

The games are selfplay.game_loop's evaluation slots (every slot of a process evaluates) and play_one's:
the same launches, pairing and rows as the run's own. The draws are seeded by the run's name, this
worker's name (--host), the update it started with and its games so far: other matchups than the run
machine's, and new ones after every restart (as game_rng's docstring explains for the learner). Each
process runs its games as threads and batches their network calls (selfplay.Inference: CUDA graphs on a
GPU, eager on the CPU).

Rows: the learner's evaluation rows ("eval": true, "eval_key"), plus "host", "version" (in games of
"current": the update of the weights it had at the game's end) and "episode" (this file's own count),
in runs/<run>/episodes-remote.jsonl; runs/<run>/remote-worker.json is its status.

SIGUSR1 drains it: the games in flight finish (their rows written), no new ones start, then it exits (at most
DRAIN_MINUTES). A row is written only when its game ends, and losses and ties last 3-5x as long as wins: games
cut by a relaunch were mostly long ones, and 58 relaunches on 2026-10-24 had read a true 84% as ~89.5% (audit 7).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from .. import paths

RUNS = paths.REPO_ROOT / "runs"
EPISODES = "episodes-remote.jsonl"
STATUS = "remote-worker.json"
STEPS = re.compile(r"\d+\.pt")
DRAIN_MINUTES = 15.0  # (a game's longest: 40 game minutes, ~6 minutes here)


def local_path(path: str, main_root: str | None) -> Path:
    """A path in the run's settings -> here: relative ones to this repository (the run machine's working
    directory is its repository), absolute ones under the run machine's repository likewise."""
    p = Path(path)
    if p.is_absolute() and main_root and p.is_relative_to(main_root):
        p = p.relative_to(main_root)
    return p if p.is_absolute() else paths.REPO_ROOT / p


def rehome(path: str) -> str:
    """A path on the run's machine under its user's home (/home/<user>/...) -> the same under this home,
    when it is not here as it is (the Wine build: ~/wc3/ge/...)."""
    p = Path(path).expanduser()
    if p.exists() or len(p.parts) < 4 or p.parts[1] != "home":
        return str(p)
    return str(Path.home().joinpath(*p.parts[3:]))


def _flag(v) -> bool:
    return str(v).strip().lower() not in ("0", "false", "none", "")


def newest_checkpoint(run_dir: Path) -> Path | None:
    cks = sorted(p for p in (run_dir / "checkpoints").glob("[0-9]*.pt") if STEPS.fullmatch(p.name))
    return cks[-1] if cks else None


def publish_newest(run_dir: Path, last: str | None) -> tuple[str | None, int]:
    """The newest checkpoint as current.pt (selfplay.publish's format: what Nets reloads, in place), if it
    is not `last` -> (its name, its update; -1: none yet)."""
    import torch

    ck_path = newest_checkpoint(run_dir)
    if ck_path is None or ck_path.name == last:
        return last, -1
    try:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    except (EOFError, RuntimeError, OSError) as e:  # (being copied in: the next look)
        print(f"checkpoint {ck_path.name} not read yet: {e}", flush=True)
        return last, -1
    version = int(ck.get("update", 0))
    tmp = run_dir / "current.pt.tmp"
    torch.save({"model": {k: v.cpu() for k, v in ck["model"].items()}, "config": ck["config"], "version": version}, tmp)
    tmp.replace(run_dir / "current.pt")
    print(f"current: {ck_path.name} (update {version})", flush=True)
    return ck_path.name, version


def run_config(run_dir: Path, host: str, games: int, start_update) -> tuple[dict, dict]:
    """The settings game_loop and play_one need for the run's evaluation games, as selfplay.main makes them
    from its arguments (here from run.json's config: its arguments as strings) -> (cfg, {local evaluation
    checkpoint path: the run's own}). Training-only settings are left out: no trajectories are made."""
    info = json.loads((run_dir / "run.json").read_text())
    c = info["config"]

    def num(key: str, default, cast=float):
        v = c.get(key)
        return default if v in (None, "None", "") else cast(v)
    main_root = str(Path(c["runs"]).parent) if c.get("runs") else None
    evals = re.findall(r"PosixPath\('((?:[^'\\]|\\.)*)'\)", c.get("eval_checkpoints") or "")
    local = {str(local_path(p, main_root)): p for p in evals}
    from . import features as fx
    races = list(fx.RACES) if c.get("races", "all") == "all" else c["races"].split(",")
    cfg = {"run_dir": str(run_dir), "vocab": str(run_dir / "vocab.json"), "map": c.get("eval_map") or c["map"], "races": races,  # (--eval-map)
           "eval_fallback": 0,  # (the worker always plays: it is the other machine)
           "learner_races": {}, "mirror": _flag(c.get("mirror", 0)), "handicap": num("handicap", 100, int),
           "step_seconds": num("step_seconds", 0.5), "max_minutes": num("max_minutes", 40.0),
           "wait_floor_ms": num("wait_floor_ms", 5, int), "games_per_process": num("games_per_process", 2, int),
           "agent_games_factor": num("agent_games_factor", 1, int), "pairs": num("pairs", 1, int),
           "scripted_reset": _flag(c.get("scripted_reset", 0)), "native_obs": _flag(c.get("native_obs", 1)),
           "game_nice": num("game_nice", 10, int), "game_cuts": num("game_cuts", 0, int) or None,
           "features": num("features", fx.F, int), "inputs": num("inputs", 1, int),  # (--inputs, the masks)
           "tech_mask": num("tech_mask", 0, int), "valid_mask": num("valid_mask", 0, int), "neutral": num("neutral", 0, int), "macros": num("macros", 0, int),
           "privileged": num("opsd_share", 0.0) > 0 or num("opsd_coef", 0.0) > 0,  # (the networks' channels only)
           "scenarios": [], "scenario_share": 0.0,  # (training games only)
           "eval_ai": [d for d in c.get("eval_ai", "normal,insane").split(",") if d],
           "eval_checkpoints": list(local), "eval_z": _flag(c.get("eval_z", 0)),
           "z_pool": c.get("eval_z_pool") or c.get("z_pool") or "", "z_src": c.get("z_src") or "", "z_enforce": _flag(c.get("z_enforce", 0)),
           # (a number: 2 is the builder as a teacher only, never carrying out; as a flag it was True, == 1, and the
           # worker's current+z games had the builder carry out the target until 2026-10-19)
           "z_build": num("z_build", 0, int), "z_modes": num("z_modes", 0.0), "z_mix": num("z_mix", 0, int),  # (--z-modes: drawn and measured)
           "grpo_share": 0.0, "grpo_group": 4, "grpo_branch": [60.0, 480.0],  # (evaluation games never branch)
           "eval_map": "", "eval_costs": None,  # (its map is the evaluation games' already: --eval-map)
           "eval_vs": num("eval_vs", 0.0),  # (--eval-vs: the current policy against a listed checkpoint)
           "eval_heuristic": num("eval_heuristic", 0.0), "heuristic_race": c.get("heuristic_race") or "orc", "heuristic_share": 0.0,  # (the
           # current policy against the hand-written player: --eval-heuristic)
           "eval_guide": num("eval_guide", 0.0), "guide_capable": True,  # (--eval-guide: current+zg, this worker's only)
           "z_time_scale": num("z_time_scale", 1.0), "opd_share": 0.0, "opd_difficulty": c.get("opd_difficulty", "normal"),
           "shared_noise": num("shared_noise", 0.0), "float_penalty": num("float_penalty", 0.0),
           "hero_level_value": num("hero_level_value", 0.0), "shaping": num("shaping", 0.0), "shaping_decay": 0.0,
           "shaping_scale": num("shaping_scale", 1.0), "gamma": num("gamma", 0.999), "lam": num("lam", 0.95),
           "tie_break": num("tie_break", 0.0), "tie_value": num("tie_value", 0.0), "unit_credit": 0.0,
           "upgo": 0.0, "chunk": num("chunk", 64, int), "max_past": num("max_past", 12, int),
           "avail_mask": _flag(c.get("avail_mask", 1)), "infer_entities": [int(x) for x in c.get("infer_entities", "48,96").split(",")],
           "exploiter_share": 0.0, "video_every": 0.0,
           # every slot evaluates (game_loop: k >= games_per_actor); the draws: this worker's, from this start
           "games_per_actor": 0, "eval_games_per_actor": games, "eval_games": 0,
           "seed": f"{num('seed', 0, int)}/{host}", "start_update": start_update}
    cfg["max_steps"] = int(cfg["max_minutes"] * 60 / cfg["step_seconds"]) + 20
    wine = {"wine": rehome(c["wine"]) if c.get("wine") not in (None, "None", "") else None,
            "fsync": _flag(c.get("fsync", 0)), "ntsync": _flag(c.get("ntsync", 0))}
    return {**cfg, **{f"_{k}": v for k, v in wine.items()}}, local


class Rows:
    """play_one's out_q in a worker process: each episode row, with this host and the weights' version,
    and the evaluation checkpoint by the run's own name for it, to the main process."""

    def __init__(self, q, host: str, keys: dict, infer):
        self.q, self.host, self.keys, self.infer = q, host, keys, infer

    def put(self, msg: dict) -> None:
        e = msg.get("episode")
        if e is None:  # (trajectories: none in evaluation games)
            return
        if e.get("eval_key") in self.keys:
            e["eval_key"] = self.keys[e["eval_key"]]
        e["host"] = self.host
        if e.get("eval_key") in ("current", "current+z", "current+zg") or str(e.get("eval_key", "")).startswith("vs "):
            e["version"] = self.infer.nets.version  # (the current policy's games: with a target too; audit 8)
        self.q.put(e)


def worker_main(wid: int, cfg: dict, games: int, keys: dict, host: str, q, stop) -> None:
    """A process of the worker: `games` evaluation games (game_loop's threads) and their inference."""
    import torch

    from ..runtime import reaper
    from . import features as fx
    from .selfplay import Inference, Nets, _exit, game_loop
    signal.signal(signal.SIGTERM, _exit)
    fx.ACTIVE_F = cfg["features"]
    fx.set_geometry(*cfg["map_geometry"])
    fx.set_z(cfg.get("z_dim", 0))
    fx.set_inputs(cfg.get("inputs", 1), bool(cfg.get("privileged")))
    fx.set_masks(cfg.get("tech_mask", 0), cfg.get("valid_mask", 0))
    fx.set_neutral(cfg.get("neutral", 0))
    fx.set_macros(int(cfg.get("macros", 0) or 0))
    torch.set_num_threads(1)
    sys.setswitchinterval(0.001)  # (a game thread woken by its answer waits for the GIL: 5 ms by default)
    try:
        device = torch.device(cfg["device"])
        nets = Nets(cfg, device)
        while nets.current is None and not stop.is_set() and os.getppid() == cfg["learner_pid"]:
            time.sleep(1)
            nets.reload()
        cuda = device.type == "cuda"
        # the calls padded to a few batch sizes and entity counts (each shape a CUDA graph, captured at its first call)
        buckets = tuple(sorted({b for b in (2, 4, 8) if b < games} | {games}))
        infer = Inference(nets, device, games, compile_=cuda, buckets=buckets, entities=tuple(cfg["infer_entities"]),
                          guide=cfg.get("eval_guide", 0.0) > 0)
        rows = Rows(q, host, keys, infer)
        threads = [threading.Thread(target=game_loop, args=(wid, k, cfg, infer, rows, stop, None), daemon=True)
                   for k in range(games)]
        for th in threads:
            th.start()
            time.sleep(3)  # launches spread out
        while not stop.is_set() and os.getppid() == cfg["learner_pid"]:
            time.sleep(1)
        for th in threads:  # (stop set: each thread ends after the game it plays; a hard stop terminates this process)
            while th.is_alive() and os.getppid() == cfg["learner_pid"]:
                th.join(timeout=1)
    finally:
        reaper.reap()


def _status(run_dir: Path, **fields) -> None:
    tmp = run_dir / f"{STATUS}.tmp"
    tmp.write_text(json.dumps({**fields, "time": time.time()}, indent=1))
    tmp.replace(run_dir / STATUS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="the run's name: runs/<run> here holds its run.json, vocab.json, checkpoints")
    ap.add_argument("--games", type=int, default=16, help="evaluation games at a time")
    ap.add_argument("--processes", type=int, default=0, help="processes the games are split over (default: one per 2 games: "
                                                             "a process's games share its GIL, ~10 ms of Python a step)")
    ap.add_argument("--host", default=socket.gethostname(), help="this worker's name: the rows' host and part of the draws' seed")
    ap.add_argument("--device", help="the inference's device (default: cuda if available)")
    ap.add_argument("--hours", type=float, default=0.0, help="stop after this long (0: when stopped)")
    a = ap.parse_args(argv)
    import multiprocessing as mp

    import torch

    from ..runtime import reaper, wine
    from . import features as fx
    from .collect import claim_slot
    run_dir = RUNS / a.run
    published, version = publish_newest(run_dir, None)
    while published is None:  # (the manager sends the newest checkpoint first)
        print("waiting for a checkpoint in", run_dir / "checkpoints", flush=True)
        time.sleep(30)
        published, version = publish_newest(run_dir, None)
    out = run_dir / EPISODES
    count = sum(1 for _ in out.open()) if out.exists() else 0
    # (the draws from the checkpoint it starts with and the games so far: a restart draws anew)
    cfg, keys = run_config(run_dir, a.host, a.games, f"{max(version, 0)}+{count}")
    for path in keys:
        if not Path(path).exists():
            raise SystemExit(f"evaluation checkpoint {path} missing ({keys[path]} in the run)")
    w = {k[1:]: cfg.pop(k) for k in list(cfg) if k.startswith("_")}
    if w["wine"]:  # (before any game or child process: they read the environment)
        wine.use_wine(w["wine"], w["fsync"], w["ntsync"])
    wine.ensure_template()
    fx.ACTIVE_F = cfg["features"]
    os.environ["WARCRAFTSIM_FEATURES"] = str(cfg["features"])
    fx.set_inputs(cfg["inputs"], bool(cfg["privileged"]))  # (before the networks: model.load gives older ones the new inputs)
    fx.set_masks(cfg["tech_mask"], cfg["valid_mask"])
    fx.set_neutral(cfg["neutral"])
    fx.set_macros(int(cfg.get("macros", 0) or 0))
    from ..data.duelmap import parse_duel_name
    if parse_duel_name(cfg["map"]) is None:  # (as selfplay.main: another map's frame; set_geometry exports it)
        from ..data.mapbuild import stock_map_path
        fx.set_geometry(*fx.map_geometry(stock_map_path(cfg["map"])))
    cfg["map_geometry"] = [fx.STARTS, fx.BIN_RANGE]
    first = newest_checkpoint(run_dir)  # the run's networks' strategy block (selfplay: the rows' width)
    cfg["z_dim"] = torch.load(first, map_location="cpu", weights_only=False)["config"].get("z", 0) if first else 0
    fx.set_z(cfg["z_dim"])
    vocab = json.loads(Path(cfg["vocab"]).read_text())
    from .costs import order_costs
    from .trace import unit_values
    cfg["values"] = unit_values()
    cfg["costs"] = order_costs(vocab, cfg["map"]).tolist() if cfg["avail_mask"] else None
    cfg["order_names"] = vocab.get("order_names") or {}  # (only for videos' traces: none here)
    cfg["device"] = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg["slot"], slot_lock = claim_slot(RUNS)
    cfg["learner_pid"] = os.getpid()  # (game_loop and the processes stop when this process is gone)
    spec = run_dir / "league_spec.json"
    if not spec.exists():  # (game_loop reads the league's launches; evaluation games replace what it draws)
        spec.write_text(json.dumps({"launch": [{"kind": "ai", "difficulty": cfg["eval_ai"][0], "p": 1.0}], "agents": []}))
    n_proc = a.processes or max(1, -(-a.games // 2))
    split = [a.games // n_proc + (i < a.games % n_proc) for i in range(n_proc)]
    cfg["actors"] = n_proc
    ctx = mp.get_context("spawn")
    q, stop = ctx.Queue(), ctx.Event()

    def start(i: int):
        p = ctx.Process(target=worker_main, args=(i, cfg, split[i], keys, a.host, q, stop), daemon=True)
        p.start()
        return p
    stopping, draining = [], []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(1))
    signal.signal(signal.SIGINT, lambda *_: stopping.append(1))
    signal.signal(signal.SIGUSR1, lambda *_: draining.append(time.time()))
    started, done, last_ep = time.time(), 0, None
    procs = [start(i) for i in range(n_proc)]
    print(f"evaluation worker {a.host} for {a.run}: {a.games} games in {n_proc} processes ({split}) on {cfg['device']}, "
          f"policies current + {list(keys.values())}, against {cfg['eval_ai']} on {cfg['map']}", flush=True)
    t_status = t_ck = 0.0
    try:
        while not stopping and not (a.hours and time.time() - started > 3600 * a.hours):
            if draining and not stop.is_set():
                stop.set()  # (game_loop: no new game; the processes end when their games did)
                print("draining: the games in flight finish, then the worker exits", flush=True)
            if draining and (time.time() - draining[0] > 60 * DRAIN_MINUTES
                             or not any(p.is_alive() for p in procs) and q.empty()):
                break
            try:
                e = q.get(timeout=2)
                count += 1
                done += 1
                e["episode"] = count
                last_ep = e["time"]
                with out.open("a") as f:
                    f.write(json.dumps(e) + "\n")
            except queue.Empty:
                pass
            now = time.time()
            if now - t_ck > 20:
                t_ck = now
                published, v = publish_newest(run_dir, published)
                version = v if v >= 0 else version
                for i, p in enumerate(procs):
                    if not p.is_alive() and not draining:  # (a crashed process: its games again)
                        print(f"process {i} exited ({p.exitcode}): restarted", flush=True)
                        procs[i] = start(i)
            if now - t_status > 30:
                t_status = now
                _status(run_dir, host=a.host, run=a.run, pid=os.getpid(), started=started, games=a.games,
                        processes=n_proc, device=cfg["device"], episodes=done, episodes_total=count,
                        last_episode=last_ep, current=published, version=version)
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=20)
        for p in procs:
            if p.is_alive():
                p.terminate()
        reaper.reap()
        _status(run_dir, host=a.host, run=a.run, pid=None, stopped=time.time(), episodes=done, episodes_total=count)
        slot_lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
