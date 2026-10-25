"""An actor's per-step Python on a recorded game, without the game or the network: selfplay.play_one and all it
calls (the observation's parse, the view, the policy's orders as commands, the teacher, the target builder, the
trajectory and its chunks to the learner, pickled for its queue), timed per side-step, and its outputs kept to compare another
implementation's with (every step's commands, every chunk put to the learner, the episode row).

    python scripts/bench_actor.py record --run fgself-140 --races human undead --out runs/bench/actor-hu.pkl
    python scripts/bench_actor.py replay runs/bench/actor-hu.pkl [--reps 3] [--profile] [--save ref.pkl | --compare ref.pkl]
        [--instructions] [--root another-tree]

record plays one real game between the run's current policy and itself on the run's training map (the policy on the
CPU; its answers and every observation the game sent are kept); replay feeds play_one those observations (through the
same parser) and answers, so it makes the same steps again."""

from __future__ import annotations

import argparse
import copy
import cProfile
import json
import os
import pickle
import pstats
import random
import sys
import time
from multiprocessing.reduction import ForkingPickler
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
    """play_one's out_q: what it puts, pickled as the learner's queue does in the actor (its feeder thread), and
    unpickled as the learner does when compared."""

    def __init__(self):
        self.items: list = []

    def put(self, msg) -> None:
        self.items.append(ForkingPickler.dumps(msg))

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
    with GameInstance(setup, name=a.name, timeout=120) as g:
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
        opp = {"name": "self", "kind": "self", **({"tax": a.tax, "tax_side": a.tax_side} if a.tax else {})}
        ep = selfplay.play_one(g, obs, cfg, vocab, races, side, dict(opp), infer, q, 0, train=True)
        print(f"recorded {ep['length']} steps of {races} in {time.time() - t0:.0f} s: outcome {ep['outcome']}, "
              f"{len(answers)} answers, {sum(len(x) for x in groups)} payloads", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"run": a.run, "races": races, "side": side, "seed": a.seed, "groups": groups, "answers": answers,
                     "order_names": order_names, "setup": setup, "opp": opp}, f)
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


class PerfControl:
    """perf stat's counters on only around play_one (BENCH_PERF_CTL="ctl fifo,ack fifo": perf stat -D -1 --control
    on those fifos; bench-out's driver): the record's loading, its copies and the comparison left out."""

    def __init__(self):
        spec = os.environ.get("BENCH_PERF_CTL")
        self.fds = (os.open(spec.split(",")[0], os.O_WRONLY), os.open(spec.split(",")[1], os.O_RDONLY)) if spec else None

    def __call__(self, cmd: str) -> None:
        if self.fds:
            os.write(self.fds[0], cmd.encode() + b"\n")
            os.read(self.fds[1], 16)  # (its ack: counting has started / stopped)


PERF: PerfControl | None = None


def replay_once(rec: dict, cfg: dict, vocab: dict, prof=None, count: bool = False) -> tuple[float, dict]:
    from warcraftsim.fullgame import selfplay
    random.seed(rec["seed"])
    random.Random = _Seeded
    g, q = Game(rec), Queue()
    answers = Answers(copy.deepcopy(rec["answers"]))
    obs = g.start()
    if count:
        PERF("enable")
    t0 = time.process_time()  # (this process's CPU: the machine is busy with a run, wall time is noisy)
    if prof:
        prof.enable()
    opp = dict(rec.get("opp") or {"name": "self", "kind": "self"})
    ep = selfplay.play_one(g, obs, cfg, vocab, rec["races"], rec["side"], opp, answers, q, 0,
                           train=True)
    if prof:
        prof.disable()
    dt = time.process_time() - t0
    if count:
        PERF("disable")
    return dt, {"cmds": g.sent, "puts": q.items, "episode": ep}


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
    global PERF
    PERF = PerfControl()
    times, out = [], None
    for r in range(a.reps):  # (counted by perf: the reps after the first, the caches warm as in an actor's later games)
        dt, out = replay_once(rec, cfg, vocab, prof if r == a.reps - 1 else None, count=r > 0)
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
    out["puts"] = [pickle.loads(m) for m in out["puts"]]  # (as the learner gets them)
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
              f"{len(diff)} differences" + "".join(f"\n  {d}" for d in diff[:int(os.environ.get("BENCH_DIFFS", 12))]))
    return 0


def counted(argv: list[str]) -> int:
    """This replay again under perf stat, counting play_one's warm reps only (PerfControl): user-space instructions a
    side-step, steadier than CPU time on a machine a run keeps busy (math libraries on one thread, string hashes fixed:
    their spinning and set orders had moved the counts by up to 20%)."""
    import re
    import subprocess
    import tempfile
    a = [x for x in argv if x != "--instructions"]
    with tempfile.TemporaryDirectory() as d:
        ctl, ack = os.path.join(d, "ctl"), os.path.join(d, "ack")
        os.mkfifo(ctl)
        os.mkfifo(ack)
        cfd, afd = os.open(ctl, os.O_RDWR), os.open(ack, os.O_RDWR)
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "PYTHONHASHSEED": "0", "BENCH_PERF_CTL": f"{ctl},{ack}"}
        r = subprocess.run(["perf", "stat", "-x,", "-e", "instructions:u", "-D", "-1", "--control", f"fd:{cfd},{afd}",
                            sys.executable, __file__, *a], capture_output=True, text=True, pass_fds=(cfd, afd), env=env)
        os.close(cfd)
        os.close(afd)
    print(r.stdout, end="")
    m = re.search(r"^(\d+),,instructions", r.stderr, re.M)
    sides = re.search(r"(\d+) side-steps", r.stdout)
    reps = re.search(r"best of (\d+)", r.stdout)
    if not (m and sides and reps) or int(reps.group(1)) < 2:
        print(r.stderr[-2000:], "(no count: perf stat, and --reps 2 or more)")
        return 1
    print(f"{int(m.group(1)) / (int(reps.group(1)) - 1) / int(sides.group(1)) / 1e6:.3f} M instructions a side-step "
          f"(user space, play_one's warm reps)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--run", default="fgself-140")
    r.add_argument("--races", nargs=2, default=["human", "undead"])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--name", default="bench_actor", help="the game instance's name (two recordings at once: two names)")
    r.add_argument("--tax", type=float, default=0.0, help="the share of what one side gathers taken (a game between "
                   "agents' balance tax: longer games)")
    r.add_argument("--tax-side", type=int, default=0)
    p = sub.add_parser("replay")
    p.add_argument("record", type=Path)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--save", type=Path)
    p.add_argument("--compare", type=Path)
    p.add_argument("--root", type=Path, help="another tree's warcraftsim to time (e.g. the code before a change)")
    p.add_argument("--instructions", action="store_true", help="count instructions instead (perf stat)")
    a = ap.parse_args()
    if getattr(a, "instructions", False):
        return counted(sys.argv[1:])
    if getattr(a, "root", None):
        sys.path.insert(0, str(a.root.resolve()))
    return record(a) if a.cmd == "record" else replay(a)


if __name__ == "__main__":
    raise SystemExit(main())
