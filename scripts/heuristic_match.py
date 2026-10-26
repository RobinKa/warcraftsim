"""Games of the hand-written player (fullgame/heuristic.py) against the built-in AI or a policy checkpoint, with the
evaluation games' settings (Echo Isles, a step of 0.5 s, 40 minutes at most, the run's handicap and game cuts), CPU
inference, niced games named heur*. Wins, losses and ties per matchup at the end; a row per game in --out.

    python scripts/heuristic_match.py --vs insane --games 8 --race orc --parallel 4
    python scripts/heuristic_match.py --vs runs/fgself-145/checkpoints/0000000013576303.pt --games 8 --race orc

--vs a checkpoint: the policy plays as in the run's evaluation games (its run.json's settings: features, inputs,
masks, macros, neutral wares; evalworker.run_config), through selfplay.play_one with the heuristic as its
opponent ({"kind": "heuristic"}), without a target. --opp-race: the opponent's race (default: each in turn).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RACES = ("human", "orc", "undead", "nightelf")
GAME_CUTS = 17179866617
WINE = "/home/tora/wc3/ge/GE-Proton10-34/files/bin"


def variant_class(spec: str | None):
    """--variant module:Class (a module of warcraftsim.fullgame, e.g. heuristic_old:HeuristicAgent) -> the class, or
    None for fullgame.heuristic's."""
    if not spec:
        return None
    import functools
    import importlib
    mod, _, rest = spec.partition(":")
    name, _, opts = rest.partition(":")
    cls = getattr(importlib.import_module(f"warcraftsim.fullgame.{mod}"), name or "HeuristicAgent")
    def value(v: str):
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
        return v
    kw = {k: value(v) for k, v in (o.split("=") for o in opts.split(",") if o)}
    return functools.partial(cls, **kw) if kw else cls


def play_scripted(name: str, race: str, opp_race: str, opp: str, side: int, args, strategy: str = "",
                  opp_strategy: str = "", record: Path | None = None) -> dict:
    """One game of the heuristic (`strategy`, "" its race's default) on `side` against the built-in AI (`opp` a
    difficulty) or another heuristic (`opp` "heuristic", playing `opp_strategy`) -> its row; `record`: the game's
    .npz there (collect.Recorder: both sides' orders as labels)."""
    from warcraftsim.fullgame.collect import Recorder
    from warcraftsim.fullgame.heuristic import HeuristicAgent
    from warcraftsim.fullgame.trace import Production
    from warcraftsim.runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup
    scripted = opp == "heuristic"
    mine = Agent(race, handicap=args.handicap, difficulty=args.slot_difficulty)
    theirs = (Agent(opp_race, handicap=args.handicap, difficulty=args.slot_difficulty) if scripted
              else BuiltinAI(opp_race, opp, handicap=args.handicap))
    slots = [mine, theirs] if side == 0 else [theirs, mine]
    setup = GameSetup(map=args.map, slots=slots, step_seconds=args.step_seconds, max_game_seconds=args.max_minutes * 60,
                      victory="decisive", window=(320, 240), draw=False, d3d_thread=False, render_threads=0,
                      native_obs=True, nice=args.nice, cuts=args.cuts or None, wait_floor_ms=5,
                      record_ai_orders=record is not None and not scripted)
    cls = variant_class(args.variant) or HeuristicAgent
    t0 = time.time()
    with GameInstance(setup, name=name, timeout=180) as g:
        obs = g.start()
        agents = {side: cls(side, race, map_name=args.map, step_seconds=args.step_seconds,
                            **({"strategy": strategy} if strategy else {}))}
        if scripted:
            agents[1 - side] = HeuristicAgent(1 - side, opp_race, map_name=args.map, step_seconds=args.step_seconds,
                                              **({"strategy": opp_strategy} if opp_strategy else {}))
        for a in agents.values():
            a.begin(obs)
        rec = Recorder(obs) if record is not None else None
        prod = Production()
        prod.step(obs)
        t = 0
        while not obs.game_over:
            if rec is not None:
                rec.state(obs, t)
            cmds, spans = [], {}
            for s_, a in agents.items():
                a.observe(obs, t)
                c = a.commands()
                spans[s_] = (len(cmds), len(cmds) + len(c))
                cmds += c
            obs = g.step(cmds)
            for s_, (a0, a1) in spans.items():
                agents[s_].accepted(cmds[a0:a1], obs.command_results[a0:a1])
            if rec is not None:
                rec.commands(cmds, obs.command_results[:len(cmds)])
                rec.issued(obs)
            prod.step(obs)
            t += 1
        if rec is not None:
            rec.state(obs, t)
    res = obs.players[side].result.name if side in obs.players else "TIE"
    outcome = 1.0 if res == "VICTORY" else -1.0 if res == "DEFEAT" else 0.0
    agent = agents[side]
    names = [""] * 2
    names[side] = agent.plan.name or strategy
    if scripted:
        names[1 - side] = agents[1 - side].plan.name or opp_strategy
    if record is not None:
        races = [""] * 2
        races[side], races[1 - side] = race, opp_race
        diffs = ["heuristic"] * 2
        if not scripted:
            diffs[1 - side] = opp
        record.parent.mkdir(parents=True, exist_ok=True)
        rec.save(record, {"races": races, "difficulties": diffs, "handicap": args.handicap, "map": args.map, "map_file": args.map,
                          "victory": "decisive", "step_seconds": args.step_seconds,
                          "strategies": [n or None for n in names] if scripted else
                          [names[0] or None if side == 0 else None, names[1] or None if side == 1 else None],
                          "slot_difficulty": args.slot_difficulty}, obs)
    return {"time": time.time(), "vs": "heuristic" if scripted else f"ai-{opp}", "race": race, "opponent_race": opp_race,
            "side": side, "strategy": names[side], **({"opp_strategy": names[1 - side]} if scripted else {}),
            "outcome": outcome, "game_time": round(obs.game_time, 1), "wall_seconds": round(time.time() - t0, 1),
            "prod": prod.row(side, obs), "opp_prod": prod.row(1 - side, obs), "heuristic": agent.summary(),
            **({"record": str(record)} if record is not None else {})}


class PolicyMatch:
    """The run's settings and a CPU inference for one checkpoint, as render_request.play_and_render makes them."""

    def __init__(self, checkpoint: Path, out_dir: Path, args):
        import numpy as np
        import torch

        from warcraftsim.fullgame import features as fx
        from warcraftsim.fullgame.costs import order_costs
        from warcraftsim.fullgame.evalworker import run_config
        from warcraftsim.fullgame.model import load
        from warcraftsim.fullgame.selfplay import Inference, Nets, publish
        from warcraftsim.fullgame.trace import unit_values
        run_dir = checkpoint.resolve().parent.parent
        cfg, _ = run_config(run_dir, "heur", 1, 0)
        for k in [k for k in cfg if k.startswith("_")]:
            cfg.pop(k)
        cfg["run_dir"] = str(out_dir)  # (Nets' current.pt: here, never in the run)
        cfg["map"] = args.map
        cfg["max_minutes"] = args.max_minutes
        cfg["max_steps"] = int(args.max_minutes * 60 / cfg["step_seconds"]) + 20
        cfg["eval_checkpoints"] = []
        cfg["handicap"] = args.handicap if args.handicap is not None else cfg["handicap"]
        fx.ACTIVE_F = cfg["features"]
        os.environ["WARCRAFTSIM_FEATURES"] = str(cfg["features"])
        fx.set_inputs(cfg["inputs"], bool(cfg["privileged"]))
        fx.set_masks(cfg["tech_mask"], cfg["valid_mask"])
        fx.set_neutral(cfg["neutral"])
        fx.set_macros(int(cfg.get("macros", 0) or 0))
        fx.set_spells(int(cfg.get("spells", 0) or 0))
        from warcraftsim.data.mapbuild import stock_map_path
        fx.set_geometry(*fx.map_geometry(stock_map_path(cfg["map"])))
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        version = int(ck.get("update", ck.get("version", 0)) or 0)
        cfg["z_dim"] = ck["config"].get("z", 0)
        fx.set_z(cfg["z_dim"])
        del ck
        vocab = json.loads(Path(cfg["vocab"]).read_text())
        out_dir.mkdir(parents=True, exist_ok=True)
        publish(load(checkpoint, "cpu", vocab=vocab if fx.NEUTRAL else None)[0], version, out_dir)
        cfg["values"] = unit_values()
        cfg["costs"] = order_costs(vocab, cfg["map"]).tolist() if cfg["avail_mask"] else None
        if cfg["avail_mask"]:
            cfg["costs_array"] = np.asarray(cfg["costs"], np.int64)
        cfg["order_names"] = vocab.get("order_names") or {}
        cfg["device"] = "cpu"
        cfg["learner_pid"] = os.getpid()
        cfg["z_pool"] = ""  # (no targets)
        torch.set_num_threads(args.threads)
        self.cfg, self.vocab, self.version = cfg, vocab, version
        nets = Nets(cfg, torch.device("cpu"))
        self.infer = Inference(nets, torch.device("cpu"), max(args.parallel, 1), compile_=False,
                               buckets=tuple(sorted({1, 2, 4, max(args.parallel, 1)})),
                               entities=tuple(cfg["infer_entities"]))

    def play(self, name: str, race: str, opp_race: str, side: int, args, plan=None, strategy: str = "") -> dict:
        """One game: the heuristic (`race`) on `side`, the policy (`opp_race`) on the other."""
        from warcraftsim.fullgame.selfplay import play_one
        from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup
        cfg = self.cfg
        races = [""] * 2
        races[side], races[1 - side] = race, opp_race
        # (both agent slots at --slot-difficulty: an insane computer slot gathers twice as much)
        diff = args.slot_difficulty
        slots = [Agent(races[s], handicap=cfg["handicap"], difficulty=diff) for s in (0, 1)]
        setup = GameSetup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"],
                          max_game_seconds=args.max_minutes * 60, victory="decisive", window=(320, 240), draw=False,
                          wait_floor_ms=cfg["wait_floor_ms"], native_obs=cfg["native_obs"], d3d_thread=False,
                          render_threads=0, cuts=args.cuts or None, nice=args.nice)

        class Rows:
            def put(self, msg):
                pass
        t0 = time.time()
        with GameInstance(setup, name=name, timeout=180) as g:
            obs = g.start()
            from warcraftsim.fullgame.heuristic import STRATEGIES
            opp = {"name": f"heuristic:{race}", "kind": "heuristic", "race": race, "log": True,
                   "plan": plan or (STRATEGIES[strategy] if strategy else None), "cls": variant_class(args.variant)}
            ep = play_one(g, obs, cfg, self.vocab, races, 1 - side, opp, self.infer, Rows(), 0, record=False, train=False)
        # the row from the heuristic's side (play_one's is the policy's)
        return {"time": time.time(), "vs": "policy", "version": self.version, "race": race, "opponent_race": opp_race,
                "side": side, "strategy": strategy or (plan.name if plan else ""), "outcome": -ep["outcome"], "game_time": round(ep["game_time"], 1),
                "wall_seconds": round(time.time() - t0, 1), "prod": ep.get("opp_prod"), "opp_prod": ep.get("prod"),
                "heuristic": ep.get("heuristic"), "policy_refused": ep.get("refused_share")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vs", default="insane", help="a built-in AI's difficulty (easy, normal, insane), 'heuristic' (two "
                    "hand-written players, both sides' strategies drawn by --strategy) or a checkpoint")
    ap.add_argument("--strategy", default="default", help="the heuristic's strategy (heuristic.STRATEGIES): a name or "
                    "several (a,b), 'default' (its race's tuned one), 'all' (each of the race's in turn) or 'random'")
    ap.add_argument("--record", type=Path, help="a folder: each game's demonstration (.npz, collect.py's format, both "
                    "sides' orders as labels) and games.jsonl; bc.py --data and strategy.from_demos read it")
    ap.add_argument("--race", default="orc", help="the heuristic's race (or 'all': each in turn)")
    ap.add_argument("--opp-race", default="all", help="the opponent's race (or 'all': each in turn)")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=4, help="games at a time (at most 4 on the live machine)")
    ap.add_argument("--map", default="(2)EchoIsles")
    ap.add_argument("--handicap", type=int, default=100)
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--step-seconds", type=float, default=0.5)
    ap.add_argument("--cuts", type=int, default=GAME_CUTS)
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--threads", type=int, default=2, help="torch's CPU threads (policy games)")
    ap.add_argument("--slot-difficulty", default="normal",
                    help="the agent slots' difficulty: the engine gives an insane computer slot double income; normal is "
                         "a person's, as the runs' agent slots since 2026-10-26 (selfplay.agent_difficulty)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="rows (.jsonl); default runs/heuristic/matches.jsonl")
    ap.add_argument("--label", default="", help="a name for this batch in the rows")
    ap.add_argument("--first", type=int, default=0, help="--record: the first game's number (game<first>.npz)")
    ap.add_argument("--variant", default="", help="another HeuristicAgent class and its options (module:Class[:key=value,...] in warcraftsim.fullgame)")
    ap.add_argument("--wine", default=WINE)
    a = ap.parse_args(argv)
    if a.parallel > 4:
        raise SystemExit("at most 4 games at a time on the live machine")
    from warcraftsim.runtime import reaper, wine
    if a.wine:
        wine.use_wine(a.wine, True, False)
    wine.ensure_template()
    rng = random.Random(a.seed)
    races = RACES if a.race == "all" else (a.race,)
    opps = RACES if a.opp_race == "all" else (a.opp_race,)
    from warcraftsim.fullgame.heuristic import DEFAULT_STRATEGY, strategies

    def pick(race: str, i: int) -> str:
        if a.strategy == "default":
            return ""
        options = strategies(race) if a.strategy in ("all", "random") else [x for x in a.strategy.split(",")
                                                                             if x in strategies(race)]
        if not options:
            return ""
        return rng.choice(options) if a.strategy == "random" else options[(i // max(len(races), 1)) % len(options)]
    plans = []
    for i in range(a.games):
        race, opp_race = races[i % len(races)], opps[(i // len(races)) % len(opps)]
        plans.append((i, race, opp_race, rng.randrange(2), pick(race, i), pick(opp_race, i + 1)))
    out = a.out or Path("runs/heuristic/matches.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    policy = None
    if a.vs not in ("easy", "normal", "insane", "heuristic"):
        policy = PolicyMatch(Path(a.vs), out.parent / f"policy-{os.getpid()}", a)
    names = queue.Queue()
    for k in range(a.parallel):
        names.put(f"heur{os.getpid() % 1000}_{k}")
    lock = threading.Lock()
    rows: list[dict] = []

    def run(p) -> None:
        i, race, opp_race, side, strategy, opp_strategy = p
        name = names.get()
        try:
            if policy is None:
                rec = a.record / f"game{i + a.first:05d}.npz" if a.record else None
                r = play_scripted(name, race, opp_race, a.vs, side, a, strategy, opp_strategy, rec)
                if a.record:
                    with lock, open(a.record / "games.jsonl", "a") as f:
                        f.write(json.dumps({"time": time.time(), "game": i + a.first, "races": [r["race"], r["opponent_race"]][::1 - 2 * side],
                                            "strategies": [r["strategy"], r.get("opp_strategy", a.vs)][::1 - 2 * side],
                                            "result": r["outcome"] if side == 0 else -r["outcome"],
                                            "minutes": round(r["game_time"] / 60, 2)}) + "\n")
            else:
                r = policy.play(name, race, opp_race, side, a, strategy=strategy)
                r["checkpoint"] = a.vs
            r.update(game=i, label=a.label)
        except Exception as e:  # noqa: BLE001 (a failed game: noted, not counted)
            import traceback
            traceback.print_exc()
            print(f"game {i}: failed: {type(e).__name__}: {e}", flush=True)
            return
        finally:
            names.put(name)
        with lock:
            rows.append(r)
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
        res = {1.0: "WIN", -1.0: "LOSS"}.get(r["outcome"], "TIE")
        pr, op = r.get("prod") or {}, r.get("opp_prod") or {}
        print(f"game {i}: heuristic {race} {r.get('strategy') or ''} vs {opp_race} {r.get('opp_strategy') or ''} "
              f"({a.vs if policy is None else 'policy'}): {res} after "
              f"{r['game_time'] / 60:.1f} min ({r['wall_seconds']:.0f} s) · army {pr.get('army')} kills {pr.get('kills')} "
              f"lost {pr.get('lost')} · theirs army {op.get('army')} kills {op.get('kills')}", flush=True)

    try:
        with ThreadPoolExecutor(a.parallel) as ex:
            list(ex.map(run, plans))
    finally:
        reaper.reap()
    by = defaultdict(Counter)
    by_strategy = defaultdict(Counter)
    for r in rows:
        by[(r["race"], r["opponent_race"])][{1.0: "W", -1.0: "L"}.get(r["outcome"], "T")] += 1
        by_strategy[r.get("strategy") or "default"][{1.0: "W", -1.0: "L"}.get(r["outcome"], "T")] += 1
    if len(by_strategy) > 1:
        for k in sorted(by_strategy):
            c = by_strategy[k]
            print(f"strategy {k:22}: {c['W']}W {c['L']}L {c['T']}T of {sum(c.values())}")
    tot = Counter()
    for k in sorted(by):
        c = by[k]
        tot.update(c)
        print(f"{k[0]:9} vs {k[1]:9}: {c['W']}W {c['L']}L {c['T']}T of {sum(c.values())}")
    print(f"total vs {a.vs}: {tot['W']}W {tot['L']}L {tot['T']}T of {sum(tot.values())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
