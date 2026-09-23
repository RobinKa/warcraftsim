"""Command line: python -m warcraftsim <command>

  setup          create the Wine template prefix and check the native helpers
  build-map      write a harness map (for inspection or manual testing)
  ai-vs-ai       built-in AI against built-in AI, report the result
  play           the scripted Human bot against the built-in AI
  scenario       run a skirmish scenario with a scripted opponent
  bench          measure throughput of N parallel games
  view           render a recorded trajectory (.jsonl) as an HTML animation
  dashboard      serve the training dashboard (runs/) on http://localhost:8765
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def _setup(args) -> int:
    from . import paths
    from .runtime import wine
    from .runtime.instance import SHIM_DIR

    ok = True
    for what, p in (("game", paths.GAME_DIR / "Warcraft III.exe"), ("StormLib", paths.STORMLIB_PATH),
                    ("pjass", paths.PJASS_PATH), ("w3shim.dll", SHIM_DIR / "w3shim.dll"),
                    ("w3launch.exe", SHIM_DIR / "w3launch.exe")):
        status = "ok" if p.exists() else "MISSING"
        ok &= p.exists()
        print(f"{what:13s} {status:8s} {p}")
    if not ok:
        print("build native helpers with: scripts/build_native.sh && make -C shim")
        return 1
    prefix = wine.ensure_template(force_registry=args.force)
    print(f"template prefix ready: {prefix}")
    return 0


def _build_map(args) -> int:
    from .data.mapbuild import HarnessConfig, build_map

    agents = tuple(int(p) for p in args.agents.split(",")) if args.agents else ()
    out = build_map(args.map, args.out, HarnessConfig(step_seconds=args.step, agent_players=agents))
    print(out)
    return 0


def _ai_vs_ai(args) -> int:
    from .runtime.instance import BuiltinAI, GameInstance, GameSetup

    setup = GameSetup(map=args.map, slots=[BuiltinAI(args.race0, args.difficulty), BuiltinAI(args.race1, args.difficulty)],
                      step_seconds=2.0, speed=args.speed, max_game_seconds=args.max_minutes * 60)
    with GameInstance(setup, name="aivsai") as g:
        t0 = time.time()
        obs = g.start()
        while not obs.game_over:
            obs = g.step()
            if g.steps % 60 == 0:
                sup = {p: f"{s.food_used}/{s.food_cap}" for p, s in obs.players.items()}
                print(f"  t={obs.game_time / 60:5.1f} min  units={len(obs.units)}  supply={sup}", flush=True)
        wall = time.time() - t0
        results = {p: s.result.name for p, s in obs.players.items()}
        print(f"game over at {obs.game_time / 60:.1f} min game time, {wall:.0f}s wall "
              f"({obs.game_time / wall:.1f}x): {results}")
    return 0


def _play(args) -> int:
    from .agents.scripted import HumanRushBot
    from .client import Wc3Game
    from .runtime.instance import Agent, BuiltinAI, GameSetup

    setup = GameSetup(map=args.map, slots=[Agent("human"), BuiltinAI(args.race, args.difficulty)],
                      speed=args.speed, step_seconds=args.step, max_game_seconds=args.max_minutes * 60)
    from .record import TrajectoryRecorder, render_html

    with Wc3Game(setup, name="play") as game:
        bot = HumanRushBot(game)
        for ep in range(args.episodes):
            t0 = time.time()
            obs = game.reset()
            rec = TrajectoryRecorder(f"{args.record}/episode{ep + 1}.jsonl", map_name=args.map,
                                     every=4) if args.record else None
            bot.on_reset(obs)
            while not obs.game_over:
                if rec:
                    rec.add(obs)
                bot.act(obs)
                obs = game.step()
                if game.instance.steps % 240 == 0:
                    me = obs.players[game.player]
                    kinds = {}
                    for u in obs.units_of(game.player):
                        kinds[u.type] = kinds.get(u.type, 0) + 1
                    print(f"  t={obs.game_time / 60:5.1f} min gold={me.gold} lumber={me.lumber} "
                          f"food={me.food_used}/{me.food_cap} units={kinds}", flush=True)
            wall = time.time() - t0
            print(f"episode {ep + 1}: {obs.players[game.player].result.name} at {obs.game_time / 60:.1f} min "
                  f"({wall:.0f}s wall, {obs.game_time / wall:.1f}x)", flush=True)
            if rec:
                rec.add(obs)
                rec.close()
                print(f"  recorded: {render_html(rec.path)}")
    return 0


def _scenario(args) -> int:
    from .client import Wc3Game
    from .runtime.instance import Agent, GameSetup, Scripted
    from .scenario import Scenario

    from .record import TrajectoryRecorder, render_html

    sc = Scenario.skirmish(args.mine.split(","), args.theirs.split(","))
    with Wc3Game(GameSetup(slots=[Agent("human"), Scripted("orc")], scenario=sc, speed=args.speed),
                 name="scenario") as game:
        for ep in range(args.episodes):
            t0 = time.time()
            obs = game.reset()
            rec = TrajectoryRecorder(f"{args.record}/episode{ep + 1}.jsonl") if args.record else None
            while not obs.game_over:
                if rec:
                    rec.add(obs)
                enemies = game.enemies()
                for u in game.my_units():
                    if u.idle and enemies:
                        game.attack(u, min(enemies, key=lambda e: e.hp))
                obs = game.step()
            print(f"episode {ep + 1}: {obs.players[0].result.name} at t={obs.game_time:.1f}s "
                  f"({time.time() - t0:.2f}s wall)", flush=True)
            if rec:
                rec.add(obs)
                rec.close()
                print(f"  recorded: {render_html(rec.path)}")
    return 0


def _bench(args) -> int:
    from .runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup, Scripted
    from .scenario import Scenario

    if args.scenario:
        setup = GameSetup(slots=[Agent("human"), Scripted("orc")],
                          scenario=Scenario.skirmish(["hfoo"] * 6, ["ogru"] * 4, max_game_seconds=600),
                          speed=args.speed, step_seconds=args.step)
    else:
        setup = GameSetup(slots=[Agent("human"), BuiltinAI("orc", "normal")], speed=args.speed,
                          step_seconds=args.step)
    games = [GameInstance(setup, name=f"bench{i}") for i in range(args.n)]
    pool = ThreadPoolExecutor(args.n)
    t0 = time.time()
    list(pool.map(lambda g: g.start(), games))
    print(f"{args.n} games started in {time.time() - t0:.1f}s", flush=True)

    def run(g):
        n = 0
        t = time.time()
        worst = 0.0
        while time.time() - t < args.seconds:
            s0 = time.time()
            obs = g.step()
            worst = max(worst, time.time() - s0)
            n += 1
            if obs.game_over:
                obs = g.restart()
        return n, g.last_obs.game_time, worst, g.speed, time.time() - t

    results = list(pool.map(run, games))
    steps = sum(r[0] for r in results)
    rate = sum(r[0] / r[4] for r in results)  # steps per second, each game over its own run time
    print(f"{steps} steps: {rate:.0f} steps/s total, {rate / args.n:.0f} per game; "
          f"{rate * args.step:.0f}x realtime total ({rate * args.step / args.n:.1f}x per game)")
    per = sorted(r[0] for r in results)
    print(f"steps per game: min {per[0]} median {per[len(per) // 2]} max {per[-1]}; "
          f"slowest single step {max(r[2] for r in results):.2f}s; clock speeds "
          f"{sorted(round(r[3]) for r in results)}")
    list(pool.map(lambda g: g.close(), games))
    return 0


def _view(args) -> int:
    from .record import render_html

    print(render_html(args.trajectory, args.out))
    return 0


def _dashboard(args) -> int:
    from .dashboard.server import serve
    from .puffer.train import RUNS_DIR

    server = serve(args.runs or RUNS_DIR, args.host, args.port)
    print(f"dashboard: http://localhost:{args.port}  (runs: {args.runs or RUNS_DIR})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="warcraftsim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--force", action="store_true", help="re-apply registry settings")
    p.set_defaults(fn=_setup)
    p = sub.add_parser("build-map")
    p.add_argument("--map", default="(2)EchoIsles")
    p.add_argument("--out", required=True)
    p.add_argument("--agents", default="0")
    p.add_argument("--step", type=float, default=0.25)
    p.set_defaults(fn=_build_map)
    p = sub.add_parser("ai-vs-ai")
    p.add_argument("--map", default="(2)EchoIsles")
    p.add_argument("--race0", default="human")
    p.add_argument("--race1", default="orc")
    p.add_argument("--difficulty", default="insane")
    p.add_argument("--speed", type=float, default=None, help="clock multiplier (default: adaptive)")
    p.add_argument("--max-minutes", type=float, default=60)
    p.set_defaults(fn=_ai_vs_ai)
    p = sub.add_parser("play")
    p.add_argument("--map", default="(2)EchoIsles")
    p.add_argument("--race", default="orc")
    p.add_argument("--difficulty", default="easy")
    p.add_argument("--speed", type=float, default=None, help="clock multiplier (default: adaptive)")
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-minutes", type=float, default=40)
    p.add_argument("--record", help="directory for trajectories + HTML viewers")
    p.set_defaults(fn=_play)
    p = sub.add_parser("scenario")
    p.add_argument("--mine", default="hfoo,hfoo,hfoo,hfoo")
    p.add_argument("--theirs", default="ogru,ogru,ogru")
    p.add_argument("--record", help="directory for trajectories + HTML viewers")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--speed", type=float, default=None, help="clock multiplier (default: adaptive)")
    p.set_defaults(fn=_scenario)
    p = sub.add_parser("bench")
    p.add_argument("-n", type=int, default=4)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--speed", type=float, default=None, help="clock multiplier (default: adaptive)")
    p.add_argument("--step", type=float, default=0.25)
    p.add_argument("--scenario", action="store_true")
    p.set_defaults(fn=_bench)
    p = sub.add_parser("dashboard")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--runs", help="runs directory (default: <repo>/runs)")
    p.set_defaults(fn=_dashboard)
    p = sub.add_parser("view")
    p.add_argument("trajectory")
    p.add_argument("-o", "--out")
    p.set_defaults(fn=_view)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
