"""The two-base start (fullgame/scenarios.py, the scenario "base2") in the real game.

    python3 scripts/base2_probe.py gold --races undead,nightelf [--base2 1] [--ai 0] [--left 1] [--minutes 3]
    python3 scripts/base2_probe.py play --races orc,nightelf [--ai 0] --run runs/fgself-172 [--checkpoint PT] [--minutes 2]

gold: a game with no policy. The setup as selfplay.play_one sends it (scenarios.base2 at step 0, late.again at steps 1 and
2, late.jobs at step 2), every start worker sent to its main mine at step 2 (the same with --base2 0, to compare), then
--minutes of play: what the game answered, what stands at the natural, its camp, the workers at each mine, gold gathered,
each mine's gold, Production as play_one keeps it (from step 2). --ai 1: the second side the built-in AI (insane), no setup
for it. --left 1: reload until side 0 starts on the left (each race on each start).

play: a training game through selfplay.play_one (scenario base2 drawn always, a run's policy on the CPU, targets with an
expansion): the first rewards against the material lead the setup gave, the episode row's scenario, prod and z fields.

Instance names start with "base2probe" (--name: a suffix).

Results (2026-10-31, echofast2, nothing refused, each race on both starts; gold gathered in 3 game minutes, its race's start
workers on the main mine, without / with): human 1780 / 3470-3530, orc 1780 / 3470-3530, undead (3 acolytes on the main)
1070 / 2850-2860, night elf 1790 / 3580-3590; both mines mined (natural: 1720-1800). Production stayed empty, the camp 3 to
0. Against the insane AI (double income) only the agent's side set up, the AI's camp stands: 3470 to the AI's 3460-3500.
play: first rewards -0.00007 to +0.0075 where the setup moved the lead by +0.39 of the scale (fgself-172's policy, undead and
human against the AI, night elf against undead); targets' halls credited at their own time, in neither z_missed nor z_extra.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.pop("WAYLAND_DISPLAY", None)  # (games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"

import numpy as np  # noqa: E402

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame import late, scenarios  # noqa: E402
from warcraftsim.fullgame.trace import Production  # noqa: E402
from warcraftsim.protocol import TargetOrder, UnitFlags  # noqa: E402
from warcraftsim.runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup  # noqa: E402


def rows_of(obs, t: int):
    from warcraftsim.fullgame.play import unit_rows
    return unit_rows(obs, t)


def setup(map_name: str, slots, shim: str, seconds: float = 0, native: bool = False) -> GameSetup:
    return GameSetup(map=map_name, slots=slots, step_seconds=0.5, draw=False, native_obs=native, d3d_thread=False,
                     render_threads=0, nice=19, warm_spare=False, window=(320, 240), max_game_seconds=seconds,
                     victory="decisive", shim_dir=shim)


def natural_of(obs, s: int) -> tuple[int, int] | None:
    sx, sy = obs.players[s].start_x, obs.players[s].start_y
    far = sorted((math.hypot(u.x - sx, u.y - sy), u.x, u.y) for u in obs.units
                 if u.type in late.MINES and u.alive and math.hypot(u.x - sx, u.y - sy) > scenarios.NATURAL_FROM_START)
    return (far[0][1], far[0][2]) if far else None


def main_of(obs, s: int, race: str):
    """The side's main mine it harvests (its own type, nearest its start)."""
    sx, sy = obs.players[s].start_x, obs.players[s].start_y
    want = late.OWN_MINE.get(race, "ngol")
    return min((u for u in obs.units if u.type == want and u.alive), key=lambda u: math.hypot(u.x - sx, u.y - sy))


def creeps_at(obs, x: float, y: float, r: float = 900.0) -> int:
    return sum(1 for u in obs.units if u.owner == late.NEUTRAL_HOSTILE and u.alive and not u.is_structure
               and math.hypot(u.x - x, u.y - y) < r)


def mines_now(obs) -> dict:
    return {(u.type, u.x, u.y, u.owner): u.resource for u in obs.units if u.type in late.MINES and u.alive}


def forced_start(g, a) -> object:
    for tries in range(16):
        obs = g.start() if tries == 0 else g.restart()
        if not a.left or obs.players[0].start_x < 0:
            return obs
    raise SystemExit("no start on the left in 16 reloads")


def gold(a) -> int:
    races = a.races.split(",")
    slots = [Agent(races[0], handicap=100, difficulty="normal"),
             BuiltinAI(races[1], "insane") if a.ai else Agent(races[1], handicap=100, difficulty="normal")]
    agents = [0] if a.ai else [0, 1]
    with GameInstance(setup(a.map, slots, a.shim), name=f"base2probe{a.name}", timeout=600) as g:
        obs = forced_start(g, a)
        print(f"map {a.map}, races {races}, base2 {a.base2}, ai {a.ai}: start {[(obs.players[s].start_x, obs.players[s].start_y) for s in (0, 1)]}",
              flush=True)
        nat = {s: natural_of(obs, s) for s in (0, 1)}
        print(f"naturals {nat}; creeps within 900: {[creeps_at(obs, *nat[s]) for s in (0, 1)]}; mines {sorted(mines_now(obs).items())}",
              flush=True)
        got = scenarios.base2(obs, agents, races) if a.base2 else ([], None)
        if got is None:
            print("base2: None (no natural)", flush=True)
            return 1
        cmds, state = got
        from late_probe import answers
        print(f"step 0: {len(cmds)} commands: {[(type(c).__name__, getattr(c, 'unit_type', '')) for c in cmds]}", flush=True)
        obs = g.step(cmds)
        res = answers(cmds, obs.command_results)
        refused = [(type(c).__name__, getattr(c, "unit_type", "")) for c, ok in zip(cmds, res) if not ok]
        print(f"step 0's answers: refused {refused}", flush=True)
        bad = len(refused)
        print(f"step 1: creeps within 900 of each natural {[creeps_at(obs, *nat[s]) for s in (0, 1)]}", flush=True)
        if state is not None:
            fix = late.again(state, rows_of(obs, 1))
            obs = g.step(fix)
            print(f"step 1's follow-up: {sum(obs.command_results)} of {len(fix)} taken ({[(type(c).__name__, getattr(c, 'unit_type', getattr(c, 'gold', ''))) for c in fix]})",
                  flush=True)
            bad += len(fix) - sum(obs.command_results)
            rows2 = rows_of(obs, 2)
            fix = late.again(state, rows2)
            given = {s: late.jobs(state, rows2, s, races[s], {}) for s in agents}
        else:
            obs = g.step([])
            fix, given = [], {}
        prod = Production()
        prod.step(obs)
        # (the start workers: to the main mine, a harvest as the policy would order)
        main = {}
        for s in agents:
            placed = {uid for _c, uid, *_r in given.get(s, ())}
            m = main_of(obs, s, races[s])
            main[s] = m
            sx, sy = obs.players[s].start_x, obs.players[s].start_y
            workers = [u for u in obs.units if u.owner == s and u.alive and u.type == scenarios.WORKERS[races[s]]
                       and u.id not in placed and math.hypot(u.x - sx, u.y - sy) < 1500]
            given.setdefault(s, [])
            print(f"side {s} {races[s]}: {len(given[s])} placed workers, {len(workers)} start workers", flush=True)
            given[s] = [(c, uid, job, tg) for c, uid, job, tg in given[s]]
            mains = [TargetOrder(u.id, fx.HARVEST, m.id) for u in workers]
            given[s] = (given[s], mains)
        jc = [c for s in agents for c, *_ in given[s][0]]
        mc = [c for s in agents for c in given[s][1]]
        obs = g.step(fix + jc + mc)
        took = obs.command_results
        n1, n2 = len(fix), len(jc)
        print(f"step 2: follow-up {sum(took[:n1])} of {n1} taken, placed workers' jobs {sum(took[n1:n1 + n2])} of {n2}, start workers {sum(took[n1 + n2:])} of {len(mc)}",
              flush=True)
        bad += (n1 + n2 + len(mc)) - sum(took)
        prod.step(obs)
        for s in agents:
            units = Counter(u.type for u in obs.units if u.owner == s and u.alive)
            at_nat = [(u.type, u.x, u.y, u.hp, u.max_hp, bool(int(u.flags) & int(UnitFlags.CONSTRUCTING))) for u in obs.units
                      if u.owner == s and u.alive and math.hypot(u.x - nat[s][0], u.y - nat[s][1]) < 1200]
            p = obs.players[s]
            print(f"step 3, side {s} {races[s]}: units {dict(units)}, food {p.food_used}/{p.food_cap}, gold {p.gold} lumber {p.lumber}; at the natural {nat[s]}: {at_nat}",
                  flush=True)
        print(f"step 3: mines {sorted(mines_now(obs).items())}; halls {[(u.owner, u.type, u.x, u.y, u.order) for u in obs.units if u.type in late.HALLS]}", flush=True)
        gold0 = {s: obs.players[s].gold_gathered for s in (0, 1)}
        mines0 = mines_now(obs)
        start_time, t = obs.game_time, 3
        mark = start_time
        while obs.game_time - start_time < a.minutes * 60 and not obs.game_over:
            obs = g.step()
            t += 1
            prod.step(obs)
            if obs.game_time - mark >= 30:
                mark = obs.game_time
                line = []
                for s in agents:
                    near = [u for u in obs.units if u.owner == s and u.alive and u.type == scenarios.WORKERS[races[s]]
                            and math.hypot(u.x - nat[s][0], u.y - nat[s][1]) < 1300]
                    hid = sum(1 for u in near if int(u.flags) & int(UnitFlags.HIDDEN))
                    line.append(f"{s}: gathered {obs.players[s].gold_gathered - gold0[s]} natural's workers seen {len(near)} (in the mine {hid})")
                print(f"{(obs.game_time - start_time) / 60:4.1f} min  " + " | ".join(line), flush=True)
        window = (obs.game_time - start_time) / 60.0
        delta = {k: mines0[k] - v for k, v in mines_now(obs).items() if k in mines0 and mines0[k] != v}
        print(f"over {obs.game_over}; {window:.2f} game minutes; mines' gold taken {delta}", flush=True)
        for s in agents:
            gone_nat = sum(v for k, v in delta.items() if math.hypot(k[1] - nat[s][0], k[2] - nat[s][1]) < 200 and k[0] != "ngol" or (
                k[0] == "ngol" and races[s] in ("human", "orc") and math.hypot(k[1] - nat[s][0], k[2] - nat[s][1]) < 200))
            gone_main = sum(v for k, v in delta.items() if math.hypot(k[1] - main[s].x, k[2] - main[s].y) < 200 and k[0] == main[s].type)
            row = prod.row(s, obs)
            print(f"RESULT side {s} {races[s]} base2={a.base2} ai={a.ai}: gold gathered {obs.players[s].gold_gathered - gold0[s]} in {window:.2f} min "
                  f"(main mine -{gone_main}, natural -{gone_nat}); prod expansions {row['expansions']} halls_home {row['halls_home']} "
                  f"workers {row['workers']} trained {row['trained']} built {row['built']}", flush=True)
        if a.ai:
            print(f"RESULT AI side {races[1]}: units {dict(Counter(u.type for u in obs.units if u.owner == 1 and u.alive))} "
                  f"gathered {obs.players[1].gold_gathered - gold0[1]}", flush=True)
    return 0 if bad == 0 else 2


def play(a) -> int:
    import torch

    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame import selfplay, strategy as strat
    from warcraftsim.fullgame.costs import order_costs
    from warcraftsim.fullgame.evalworker import run_config
    from warcraftsim.fullgame.model import load
    from warcraftsim.fullgame.trace import unit_values
    run = Path(a.run).resolve()
    races = a.races.split(",")
    c = json.loads((run / "run.json").read_text())["config"]
    root = run.parent.parent
    ck = Path(a.checkpoint) if a.checkpoint else sorted((run / "checkpoints").glob("*.pt"))[-2]
    out = Path(tempfile.mkdtemp(prefix="base2probe", dir=a.tmp or None))
    cfg, _ = run_config(run, "probe", 1, 0)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg.update(run_dir=str(out), map=a.map, z_pool=str(root / c["z_pool"]), z_share=1.0, z_reward=float(c["z_reward"]),
               z_mix_reward=float(c.get("z_mix_reward", 0.0)), z_mode_reward=float(c.get("z_mode_reward", 0.0)),
               z_build=int(c.get("z_build", 0)), z_workers=int(c.get("z_workers", 0)), z_modes=float(c.get("z_modes", 0.0)),
               z_build_labels=int(c.get("z_build_labels", 0)), teach=int(c.get("teach", 0)),
               win_balance=float(c.get("win_balance", 0.0)), scenarios=["base2"], scenario_share=1.0, late_states=[], late_z=0,
               eval_checkpoints=[], max_minutes=a.minutes, max_steps=int(a.minutes * 60 / cfg["step_seconds"]) + 20, device="cpu",
               values=unit_values(), order_names={})
    fx.ACTIVE_F = cfg["features"]
    os.environ["WARCRAFTSIM_FEATURES"] = str(cfg["features"])
    fx.set_inputs(cfg["inputs"], bool(cfg["privileged"]))
    fx.set_masks(cfg["tech_mask"], cfg["valid_mask"])
    fx.set_neutral(cfg["neutral"])
    fx.set_macros(int(cfg.get("macros", 0) or 0))
    fx.set_spells(int(cfg.get("spells", 0) or 0))
    fx.set_macros_off(str(cfg.get("macros_off") or "").split(","))
    fx.set_econ_masks(cfg.get("econ_masks", 0))
    fx.set_geometry(*fx.map_geometry(stock_map_path(cfg["map"])))
    checkpoint = torch.load(ck, map_location="cpu", weights_only=False)
    cfg["z_dim"] = checkpoint["config"].get("z", 0)
    fx.set_z(cfg["z_dim"])
    version = int(checkpoint.get("update", checkpoint.get("version", 0)) or 0)
    del checkpoint
    vocab = json.loads(Path(cfg["vocab"]).read_text())
    selfplay.publish(load(ck, "cpu", vocab=vocab if fx.NEUTRAL else None)[0], version, out)
    cfg["costs"] = order_costs(vocab, cfg["map"]).tolist() if cfg["avail_mask"] else None
    cfg["costs_array"] = np.asarray(cfg["costs"], np.int64) if cfg["avail_mask"] else None
    cfg["learner_pid"] = os.getpid()
    torch.set_num_threads(a.threads)
    infer = selfplay.Inference(selfplay.Nets(cfg, torch.device("cpu")), torch.device("cpu"), 2, compile_=False,
                               buckets=(1, 2), entities=tuple(cfg["infer_entities"]))

    # (a target with an expansion, for each side that draws one: the case of a hall that must count as done)
    sample = strat.sample

    def with_expansion(pool, race, opp, rng, winners=True, src=""):
        ex = [s for s in pool["strategies"] if s["race"] == race and set(s["t"]) & set(late.EXPANSION_ITEMS)]
        got = min(ex, key=lambda s: min(s["t"][k] for k in s["t"] if k in late.EXPANSION_ITEMS)) if ex else sample(pool, race, opp, rng, winners, src)
        print(f"target for {race}: its expansion {({k: v for k, v in got['t'].items() if k in late.EXPANSION_ITEMS})} of {len(got['t'])} items", flush=True)
        return got
    strat.sample = with_expansion

    leads: list = []
    shaped = selfplay.shaped

    def shaped_(obs, mat, penalty):
        got = shaped(obs, mat, penalty)
        if len(leads) < 8:
            leads.append(round(obs.game_time, 1))
            print(f"  game {obs.game_time:5.1f} s  material {mat[0]:7.0f} {mat[1]:7.0f}  lead over the scale (side 0) {(got[0] - got[1]) / cfg['shaping_scale']:+.4f}", flush=True)
        return got
    selfplay.shaped = shaped_

    added = Counter()
    add = selfplay.Trajectory.add

    def add_(self, st, res):
        k = id(self)
        if self.steps and added[k] < 6:
            print(f"  trajectory {k % 1000}: its step {added[k] - 1}'s reward {self.steps[-1].get('reward'):+.5f}", flush=True)
        added[k] += 1
        add(self, st, res)
    selfplay.Trajectory.add = add_

    noted = late.noted

    def noted_(view, given, results):
        n = noted(view, given, results)
        print(f"  the view took {n} of {len(given)} placed workers' jobs (gold {sum(v == 'gold' for v in view.assign.values())})", flush=True)
        return n
    late.noted = noted_

    class Rows:
        def __init__(self):
            self.ep = None

        def put(self, msg):
            if "episode" in msg:
                self.ep = msg["episode"]

    rows = Rows()
    slots = [Agent(races[0], handicap=100, difficulty="normal"),
             BuiltinAI(races[1], "insane") if a.ai else Agent(races[1], handicap=100, difficulty="normal")]
    g_setup = setup(cfg["map"], slots, a.shim, a.minutes * 60, bool(cfg["native_obs"]))  # (as the training's games)
    opp = {"name": "ai-insane", "kind": "ai"} if a.ai else {"name": "self", "kind": "self"}
    try:
        with GameInstance(g_setup, name=f"base2probe_play{a.name}", timeout=600) as g:
            obs = g.start()
            ep = selfplay.play_one(g, obs, cfg, vocab, races, 0, opp, infer, rows, 0)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    print(f"RESULT over at {ep['game_time']:.0f} s, outcome {ep['outcome']}, scenario {ep.get('scenario')}, late_minute {ep.get('late_minute')}, gold {ep['gold']} (the other {ep['opponent_gold']})", flush=True)
    for k in ("z", "z_items", "z_miss", "z_t", "z_made", "z_missed", "z_extra", "z_modes", "taught", "refused_share"):
        print(f"  {k}: {ep.get(k)}", flush=True)
    for name, p in (("learner", ep["prod"]), ("other", ep["opp_prod"])):
        print(f"  {name}: expansions {p['expansions']} halls_home {p['halls_home']} workers {p['workers']} trained {p['trained']} built {p['built']}", flush=True)
    z_log = ep.get("z_build") or {}
    print(f"  builder's log of halls: { {k: v for k, v in z_log.items() if k in late.EXPANSION_ITEMS} if isinstance(z_log, dict) else z_log}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("gold", "play"))
    ap.add_argument("--map", default="echofast2")
    ap.add_argument("--races", default="human,orc")
    ap.add_argument("--base2", type=int, default=1)
    ap.add_argument("--ai", type=int, default=0)
    ap.add_argument("--left", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--name", default="")
    ap.add_argument("--run", default="runs/fgself-172")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tmp", default="")
    ap.add_argument("--shim", default=str(Path(__file__).resolve().parents[1] / "build" / "shim"))
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    random.seed()
    return gold(a) if a.what == "gold" else play(a)


if __name__ == "__main__":
    raise SystemExit(main())
