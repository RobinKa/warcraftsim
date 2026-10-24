"""An actor's per-step Python on a recorded game, without the game or the network: selfplay.play_one and all it
calls (the observation's parse, the view, the policy's orders as commands, the teacher, the target builder, the
trajectory and its chunks to the learner), timed per side-step, and its outputs kept to compare another
implementation's with (every step's commands, every chunk put to the learner, the episode row).

    python scripts/bench_actor.py record --run fgself-140 --races human undead --out runs/bench/actor-hu.pkl
    python scripts/bench_actor.py replay runs/bench/actor-hu.pkl [--reps 3] [--profile] [--save ref.pkl | --compare ref.pkl]

record plays one real game between the run's current policy and itself on the run's training map (the policy on the
CPU; its answers and every observation the game sent are kept); replay feeds play_one those observations (through the
same parser) and answers, so it makes the same steps again."""

from __future__ import annotations

import argparse
import copy
import cProfile
import json
import pickle
import pstats
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUNS = Path(__file__).resolve().parents[1] / "runs"


class _Seeded(random.Random):
    """random.Random() with a fixed seed: play_one's teacher and target builder draw alike in a record and a replay."""

    def __init__(self, x=None):
        super().__init__(12345 if x is None else x)


def settings(run: str) -> tuple[dict, dict]:
    """The run's actor settings (as selfplay.main makes them for its training games) and its vocabulary; the
    features' module set up as an actor's."""
    import torch

    from warcraftsim.fullgame import evalworker
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame.costs import order_costs
    from warcraftsim.fullgame.trace import unit_values
    run_dir = RUNS / run
    c = json.loads((run_dir / "run.json").read_text())["config"]
    cfg, _ = evalworker.run_config(run_dir, "bench", 2, "0+0")
    w = {k[1:]: cfg.pop(k) for k in list(cfg) if k.startswith("_")}
    num = lambda k, d=0.0: float(c.get(k)) if c.get(k) not in (None, "None", "") else d  # noqa: E731
    cfg.update(map=c["map"], z_pool=c.get("z_pool") or "", teach=int(num("teach")), z_share=num("z_share", 0.5),
               z_reward=num("z_reward"), z_build=int(num("z_build")), z_build_labels=int(num("z_build_labels")),
               z_mix_reward=num("z_mix_reward"), z_mode_reward=num("z_mode_reward"), z_winners=int(num("z_winners", 1)),
               upgo=num("upgo"), opd_share=0.0, scenarios=[], grpo_share=0.0, win_balance=0.0, video_every=0.0,
               learner_pid=0, actors=1, games_per_actor=1, eval_games_per_actor=0)
    fx.ACTIVE_F = cfg["features"]
    fx.set_inputs(cfg["inputs"], bool(cfg["privileged"]))
    fx.set_masks(cfg["tech_mask"], cfg["valid_mask"])
    fx.set_neutral(cfg["neutral"])
    fx.set_macros(int(cfg.get("macros", 0) or 0))
    from warcraftsim.data.duelmap import parse_duel_name
    if parse_duel_name(cfg["map"]) is None:
        from warcraftsim.data.mapbuild import stock_map_path
        fx.set_geometry(*fx.map_geometry(stock_map_path("(2)EchoIsles" if cfg["map"].startswith("echo") else cfg["map"])))
    cfg["map_geometry"] = [fx.STARTS, fx.BIN_RANGE]
    cfg["z_dim"] = torch.load(run_dir / "current.pt", map_location="cpu", weights_only=False)["config"].get("z", 0)
    fx.set_z(cfg["z_dim"])
    vocab = json.loads((run_dir / "vocab.json").read_text())
    cfg["values"] = unit_values()
    cfg["costs"] = order_costs(vocab, cfg["map"]).tolist() if cfg["avail_mask"] else None
    cfg["costs_array"] = np.asarray(cfg["costs"], np.int64) if cfg["costs"] is not None else None
    cfg["device"] = "cpu"
    cfg["_wine"] = w
    return cfg, vocab


class Queue:
    """play_one's out_q: what it puts, kept."""

    def __init__(self):
        self.items: list = []

    def put(self, msg) -> None:
        self.items.append(msg)

    def cancel_join_thread(self) -> None:
        pass


def record(a) -> int:
    import torch

    from warcraftsim.fullgame import selfplay
    from warcraftsim.runtime import wine
    from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup
    cfg, vocab = settings(a.run)
    w = cfg.pop("_wine")
    if w.get("wine"):
        wine.use_wine(w["wine"], w["fsync"], w["ntsync"])
    wine.ensure_template()
    torch.set_num_threads(4)
    nets = selfplay.Nets(cfg, torch.device("cpu"))
    infer = selfplay.Inference(nets, torch.device("cpu"), 2, compile_=False, entities=tuple(cfg["infer_entities"]))
    answers: list = []
    request = infer.request

    def kept(items):
        out = request(items)
        answers.append(copy.deepcopy(out))
        return out
    infer.request = kept
    races, side = list(a.races), 0
    setup = GameSetup(map=cfg["map"], slots=[Agent(races[0], handicap=cfg["handicap"]), Agent(races[1], handicap=cfg["handicap"])],
                      step_seconds=cfg["step_seconds"], max_game_seconds=cfg["max_minutes"] * 60, victory="decisive",
                      window=(320, 240), draw=False, wait_floor_ms=cfg["wait_floor_ms"], native_obs=True, d3d_thread=False,
                      render_threads=0, pairs=1, cuts=cfg.get("game_cuts") or None)
    groups: list[list[bytes]] = []  # the payloads each step parsed (a repeat or another episode's is parsed, then skipped)
    random.seed(a.seed)
    random.Random = _Seeded
    with GameInstance(setup, name="bench_actor", timeout=120) as g:
        parse = g._native.parse

        def parsed(payload, names):
            groups[-1].append(bytes(payload))
            return parse(payload, names)
        g._native.parse = parsed
        step = g.step

        def stepped(cmds=()):
            groups.append([])
            return step(cmds)
        g.step = stepped
        groups.append([])
        obs = g.start()
        order_names = list(g.order_names)
        q = Queue()
        t0 = time.time()
        ep = selfplay.play_one(g, obs, cfg, vocab, races, side, {"name": "self", "kind": "self"}, infer, q, 0, train=True)
        print(f"recorded {ep['length']} steps of {races} in {time.time() - t0:.0f} s: outcome {ep['outcome']}, "
              f"{len(answers)} answers, {sum(len(x) for x in groups)} payloads", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"run": a.run, "races": races, "side": side, "seed": a.seed, "groups": groups, "answers": answers,
                     "order_names": order_names, "setup": setup}, f)
    print("saved", a.out)
    return 0


class Game:
    """GameInstance's part play_one uses, replaying a record's observations through the same parser."""

    def __init__(self, rec: dict):
        from warcraftsim.native import NativeObs
        self.setup, self.groups, self.names = rec["setup"], rec["groups"], rec["order_names"]
        self._native = NativeObs()
        self.k = 0
        self.sent: list = []
        self.game_wall_total = 0.0

    def _next(self):
        payloads = self.groups[self.k]
        self.k += 1
        for p in payloads[:-1]:  # (parsed, then skipped: GameInstance._await_observation)
            self._native.parse(p, self.names)
        return self._native.merge(self._native.parse(payloads[-1], self.names))

    def start(self):
        return self._next()

    def step(self, cmds=()):
        self.sent.append(list(cmds))
        return self._next()


class Answers:
    """The inference's answers, in the order they were asked."""

    def __init__(self, answers: list):
        self.answers, self.k = answers, 0

    def request(self, items):
        out = self.answers[self.k]
        self.k += 1
        return out


def replay_once(rec: dict, cfg: dict, vocab: dict, prof=None) -> tuple[float, dict]:
    from warcraftsim.fullgame import selfplay
    random.seed(rec["seed"])
    random.Random = _Seeded
    g, q = Game(rec), Queue()
    answers = Answers(copy.deepcopy(rec["answers"]))
    obs = g.start()
    t0 = time.process_time()  # (this process's CPU: the machine is busy with a run, wall time is noisy)
    if prof:
        prof.enable()
    ep = selfplay.play_one(g, obs, cfg, vocab, rec["races"], rec["side"], {"name": "self", "kind": "self"}, answers, q, 0,
                           train=True)
    if prof:
        prof.disable()
    return time.process_time() - t0, {"cmds": g.sent, "puts": q.items, "episode": ep}


def unpacked(msg: dict) -> dict:
    """A chunk message's steps as dicts, packed or not (selfplay.unpack_steps)."""
    if "packed" in msg:
        from warcraftsim.fullgame.selfplay import unpack_steps
        return {**{k: v for k, v in msg.items() if k != "packed"}, "steps": unpack_steps(msg["packed"])}
    return msg


def same(a, b, path="") -> list[str]:
    """Where two outputs differ (arrays exactly, dtypes too)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
                and np.array_equal(a, b, equal_nan=True)):
            return [path]
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}.{k} (only one has it)" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += same(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} vs {len(b)})"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += same(x, y, f"{path}[{i}]")
        return out
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return [f"{path} ({type(a).__name__} vs {type(b).__name__})"]
    return [] if a == b or (a != a and b != b) else [path]


def replay(a) -> int:
    rec = pickle.loads(a.record.read_bytes())
    cfg, vocab = settings(rec["run"])
    cfg.pop("_wine")
    prof = cProfile.Profile() if a.profile else None
    times, out = [], None
    for r in range(a.reps):
        dt, out = replay_once(rec, cfg, vocab, prof if r == a.reps - 1 else None)
        times.append(dt)
    n = sum(len(x) for x in rec["answers"])
    best = min(times)
    print(f"{len(rec['answers'])} steps, {n} side-steps: {1e6 * best / n:.0f} us of CPU a side-step (best of {a.reps}, {Path(sys.modules['warcraftsim'].__file__).parent.parent.name}: "
          f"{', '.join(f'{t:.2f}' for t in times)} s)")
    if prof:
        pstats.Stats(prof).sort_stats("cumulative").print_stats(40)
        pstats.Stats(prof).sort_stats("tottime").print_stats(30)
    for k in ("time", "wall_seconds", "step_wall_ms", "step_game_ms"):  # (the clock's)
        out["episode"].pop(k, None)
    out["puts"] = [unpacked(m) for m in out["puts"]]
    for m in out["puts"]:
        for k in ("time", "wall_seconds", "step_wall_ms", "step_game_ms"):
            (m.get("episode") or {}).pop(k, None)
    if a.save:
        a.save.write_bytes(pickle.dumps(out))
        print("saved", a.save)
    if a.compare:
        ref = pickle.loads(a.compare.read_bytes())
        diff = same(ref["cmds"], out["cmds"], "cmds") + same(ref["puts"], out["puts"], "puts") \
            + same(ref["episode"], out["episode"], "episode")
        print(f"compared {len(out['cmds'])} steps' commands, {len(out['puts'])} puts, the episode row: "
              f"{len(diff)} differences" + "".join(f"\n  {d}" for d in diff[:12]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--run", default="fgself-140")
    r.add_argument("--races", nargs=2, default=["human", "undead"])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("replay")
    p.add_argument("record", type=Path)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--save", type=Path)
    p.add_argument("--compare", type=Path)
    p.add_argument("--root", type=Path, help="another tree's warcraftsim to time (e.g. the code before a change)")
    a = ap.parse_args()
    if getattr(a, "root", None):
        sys.path.insert(0, str(a.root.resolve()))
    return record(a) if a.cmd == "record" else replay(a)


if __name__ == "__main__":
    raise SystemExit(main())
