"""Late-game starts (fullgame/late.py) in the real game: capture a mid-game state, start a new game from it, and check
what the game shows at step 1 against the capture; then whether the economy runs and what the built-in AI does from it.

    python3 scripts/late_probe.py capture --races human,undead --minute 10 --out STATE.json
    python3 scripts/late_probe.py start STATE.json [--ai 1] [--minutes 4]
    python3 scripts/late_probe.py play STATE.json [--run runs/fgself-168] [--minutes 5]

play: a training game of a state with plans under --late-z 1 (see its docstring).

capture: a game between two built-in AIs (insane) to the minute, its state (late.capture: the jobs guessed, as for an AI's
side) written to --out. start: a game of the state's races until its start locations fit the state (as captured or
swapped; a reload until they do), the state set up at step 0 (late.commands), then the comparison at step 1 (units by
type, hall tiers, research levels as the game answered SetResearch, heroes' levels / skills / items, resources, mines,
creeps, the commands the game refused), the follow-ups at steps 1 and 2 (late.again) and the harvesters' jobs at step 2
(late.jobs) as selfplay.play_one sends them, the mines and anything under construction at step 3, and --minutes of play:
both sides agents that give no orders, or with --ai 1 the second side the built-in AI (insane), every 30 s each side's
food, income and how far its army is from home, and what was started. Instance names start with "lateprobe" (--name: a
suffix, for two probes at once).

Results (2026-10-29, echofast2, states of two insane AIs): human (castle, 54 units, 3 heroes) against undead (black
citadel, haunted expansion, 52 units, 2 of 3 heroes dead) at minute 10, orc (fortress, 53) against night elf (tree of
eternity, 28, its main mined out) at minute 12, and night elf (tree of eternity on its entangled main, 47) against human
(56) at minute 8: at step 1 every unit type's count (but for a mountain giant killed in the fight under way), hall tier,
hero level, experience, skill, item and death, all research levels (29, 29, 22), the creeps; the mines' gold at step 3
(after the follow-ups), nothing left under construction, the night elves' main mined from the first 30 s; gold and
lumber as captured (human and undead 1 of each short at step 1: a repair under way). The built-in AI from the undead
state floated 7,500 gold by 5 minutes and 11,700 by 15 while training 15 units: late starts are for games between
agents only.

People's states (scripts/late_from_demos.py, rescaled to echofast2; a line of its file as STATE.json), 2026-10-29: orc
(fortress, 39 units) against human (castle, 61, a tavern hero) at minute 16, human (62) against night elf (43) at 14,
human (62) against undead (40, a haunted expansion) at 16: every unit type's count, hall tier, expansion (both sides'),
hero level, experience, skill, item, research level (10, 6, 5) and creep at step 1, gold and lumber at echofast2's scale
(undead 1 of each short); the entangled and haunted mines set by the follow-ups as for the AI's states; all 25, 23 and
21 harvest orders taken.

play, 2026-10-30 (fgself-168's policy on the CPU, echofast2, 5 minutes): night elf against undead from minute 10, human
against undead from minute 14: targets of 4 and 5, 12 and 11 items; the input 212 numbers in 0..1; mismatch 0 at every step
to 13 and 18 s (placed units and expansions no windfall, more footmen of a held type no change); items credited (the
human's castle due at 100 s, made at 105 s: 6 to 5). See docs/reference/human-replays.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("WAYLAND_DISPLAY", None)  # (games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"

import numpy as np  # noqa: E402

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame import late  # noqa: E402
from warcraftsim.protocol import EventKind, Place, SetResearch, UnitFlags, encode_commands  # noqa: E402
from warcraftsim.runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup  # noqa: E402


def rows_of(obs, t: int):
    from warcraftsim.fullgame.play import unit_rows
    return unit_rows(obs, t)


def setup(map_name: str, slots, shim: str) -> GameSetup:
    return GameSetup(map=map_name, slots=slots, step_seconds=0.5, draw=False, native_obs=False, d3d_thread=False,
                     render_threads=0, nice=10, warm_spare=False, window=(320, 240), max_game_seconds=0,
                     victory="decisive", shim_dir=shim, record_ai_orders=all(s.kind == "ai" for s in slots))


def capture(a) -> int:
    races = a.races.split(",")
    with GameInstance(setup(a.map, [BuiltinAI(races[0], "insane"), BuiltinAI(races[1], "insane")], a.shim),
                      name=f"lateprobe_cap{a.name}", timeout=300) as g:
        obs = g.start()
        seen, t = {0: {}, 1: {}}, 0
        while obs.game_time < a.minute * 60 and not obs.game_over:
            obs = g.step()
            t += 1
            late.note_research(seen, obs.events, rows_of(obs, t))
        rows = rows_of(obs, t)
        st = late.capture(rows, obs.players, races, a.map, seen, obs.game_time)
        idle = Counter((fx.rawcode(int(r[fx.C_TYPE])), int(r[fx.C_ORDER]), int(r[fx.C_FLAGS])) for r in rows
                       if fx.rawcode(int(r[fx.C_TYPE])) in late.HARVESTERS and int(r[fx.C_OWNER]) in (0, 1)
                       and late._role(r, np.zeros((0, 2)), {}) == "")
        print(f"harvesters without a job (type, order, flags): {dict(idle)}", flush=True)
        if st is None:
            print(f"decided at {obs.game_time / 60:.1f} min: no state", flush=True)
            return 1
    Path(a.out).write_text(json.dumps(st))
    for s in (0, 1):
        side = st["sides"][s]
        print(f"side {s} {races[s]}: {len(side['units'])} units, {len(side['heroes'])} heroes, research {side['research']}, "
              f"gold {side['gold']} lumber {side['lumber']}", flush=True)
    return 0


def counts(rows, side: int) -> Counter:
    fl = rows[:, fx.C_FLAGS]
    keep = (rows[:, fx.C_OWNER] == side) & ((fl & (fx.DEAD | late.SKIP)) == 0)
    return Counter(fx.rawcode(int(t)) for t in rows[keep, fx.C_TYPE])


def expected(st: dict, side: int) -> Counter:
    s = st["sides"][side]
    return Counter([u[0] for u in s["units"]] + [h[0] for h in s["heroes"] if not h[-1]])


def tier(c: Counter) -> int:
    return max((late.HALLS[t] for t in c if t in late.HALLS), default=0)


def expansions(halls, mines, start) -> int:
    """Town halls ((type, x, y)) 1500 from the start and 1000 from a mine ((x, y)): expansions (trace.Production's)."""
    return sum(1 for t, x, y in halls if t in late.HALLS and math.hypot(x - start[0], y - start[1]) > 1500
               and any(math.hypot(x - mx, y - my) < 1000 for mx, my in mines))


def answers(cmds: list, results: list) -> list[bool]:
    """Each command's answer: all of its ops' (a Place with skills, items or a death encodes to several)."""
    out, k = [], 0
    for c in cmds:
        n = 1 + (len(c.skills) + len(c.items) + bool(c.dead) if isinstance(c, Place) else 0)
        out.append(all(results[k:k + n]))
        k += n
    return out


def check(st: dict, obs, rows, cmds: list) -> int:
    """The game at step 1 against the state: returns the number of mismatches."""
    bad = 0
    res = answers(cmds, obs.command_results)
    refused = Counter(type(c).__name__ + (":" + str(getattr(c, "unit_type", "")) if isinstance(c, Place) else "")
                      for c, ok in zip(cmds, res) if not ok)
    print(f"commands {len(cmds)} ({len(encode_commands(cmds))} ints), refused {dict(refused)}", flush=True)
    for s in (0, 1):
        want, have = expected(st, s), counts(rows, s)
        diff = {t: (want[t], have[t]) for t in set(want) | set(have) if want[t] != have[t]}
        p = obs.players[s]
        print(f"side {s} {st['races'][s]}: {sum(have.values())} units (captured {sum(want.values())}), hall tier "
              f"{tier(have)} (captured {tier(want)}), gold {p.gold}/{st['sides'][s]['gold']} lumber "
              f"{p.lumber}/{st['sides'][s]['lumber']}, food {p.food_used}/{p.food_cap}", flush=True)
        if diff:
            print(f"  by type (captured, game): {diff}", flush=True)
            bad += len(diff)
        bad += (p.gold != st["sides"][s]["gold"]) + (p.lumber != st["sides"][s]["lumber"]) + (tier(have) != tier(want))
        ex_want = expansions([u[:3] for u in st["sides"][s]["units"]], [m[1:3] for m in st["mines"]], st["starts"][s])
        ex_have = expansions([(u.type, u.x, u.y) for u in obs.units if u.owner == s and u.alive],
                             [(u.x, u.y) for u in obs.units if u.type in late.MINES and u.alive], st["starts"][s])
        print(f"  expansions {ex_have} (captured {ex_want})", flush=True)
        bad += ex_have != ex_want
        abil = __import__("warcraftsim.data.abilities", fromlist=["x"]).hero_abilities()
        for t, x, y, hp, mana, level, xp, skills, items, dead in st["sides"][s]["heroes"]:
            hs = [u for u in obs.units if u.owner == s and u.type == t]
            if not hs:
                print(f"  hero {t}: missing", flush=True)
                bad += 1
                continue
            u = hs[0]
            lv = {c: lvl for c, (lvl, _cd) in zip(abil.get(t, ()), u.abilities)}
            got = sorted(c for c, n in lv.items() for _ in range(n))
            its = sorted(fx.rawcode(i) for i in u.items if i)
            ok = u.hero_level == level and got == sorted(skills) and its == sorted(items) and (not u.alive) == bool(dead)
            print(f"  hero {t}: level {u.hero_level}/{level} xp {u.hero_xp}/{xp} skills {got == sorted(skills)} "
                  f"items {its}/{sorted(items)} dead {not u.alive}/{bool(dead)} hp {u.hp}/{u.max_hp} ({hp} per mille)",
                  flush=True)
            bad += not ok
    rs = [(c, ok) for c, ok in zip(cmds, res) if isinstance(c, SetResearch)]
    print(f"research levels set: {sum(ok for _, ok in rs)} of {len(rs)}"
          + (f", refused {[(c.player, c.tech, c.level) for c, ok in rs if not ok]}" if not all(ok for _, ok in rs) else ""),
          flush=True)
    bad += mines_off(st, obs, 1)
    cw = Counter(c[0] for c in st["creeps"])
    ch = Counter(u.type for u in obs.units if u.owner == late.NEUTRAL_HOSTILE and u.alive and not u.is_structure)
    print(f"creeps: {sum(ch.values())} (captured {sum(cw.values())})" + (f" by type {dict(cw - ch)} short, "
                                                                          f"{dict(ch - cw)} more" if cw != ch else ""),
          flush=True)
    bad += sum(((cw - ch) + (ch - cw)).values())
    return bad


def mines_off(st: dict, obs, t: int) -> int:
    """The mines standing (hidden ones too) whose gold is not a captured one's of their type there."""
    now = [(u.type, u.x, u.y, u.resource) for u in obs.units if u.type in late.MINES and u.alive]
    off = [(t_, x, y, r) for t_, x, y, r in now
           if not any(m[0] == t_ and math.hypot(m[1] - x, m[2] - y) < late.MINE_SAME and m[3] == r for m in st["mines"])]
    print(f"step {t} mines: {len(now)} standing (captured {len(st['mines'])}), gold off: {off}", flush=True)
    return len(off)


def start(a) -> int:
    st0 = json.loads(Path(a.state).read_text())
    races = list(st0["races"])
    slots = [Agent(races[0]), BuiltinAI(races[1], "insane") if a.ai else Agent(races[1])]
    with GameInstance(setup(st0["map"], slots, a.shim), name=f"lateprobe_start{a.name}", timeout=300) as g:
        for tries in range(12):
            obs = g.start() if tries == 0 else g.restart()
            starts = [(obs.players[s].start_x, obs.players[s].start_y) for s in (0, 1)]
            st = late.fit(st0, st0["map"], races, starts)
            if st is not None:
                break
            print(f"start locations {starts}: not the state's {st0['starts']}; again", flush=True)
        else:
            return 1
        print(f"fits {'as captured' if st is st0 else 'swapped'} on try {tries + 1}", flush=True)
        trees = {d.id: (d.x, d.y) for d in obs.destructables or [] if d.life > 0}
        rows0 = rows_of(obs, 0)
        cmds = late.commands(st, rows0)
        obs = g.step(cmds)
        rows1 = rows_of(obs, 1)
        bad = check(st, obs, rows1, cmds)
        print(f"step 1: {bad} mismatches", flush=True)
        fix = late.again(st, rows1)  # (step 1: the follow-up, as selfplay.play_one's)
        obs = g.step(fix)
        print(f"step 1's follow-up: {sum(obs.command_results)} of {len(fix)} taken "
              f"({dict(Counter(type(c).__name__ for c in fix))})", flush=True)
        rows2 = rows_of(obs, 2)
        agents = [0] if a.ai else [0, 1]
        given = {s: late.jobs(st, rows2, s, races[s], trees) for s in agents}
        jc = [c for s in agents for c, *_ in given[s]]
        fix = late.again(st, rows2)
        p0 = {s: (obs.players[s].gold_gathered, obs.players[s].lumber_gathered) for s in (0, 1)}
        obs = g.step(fix + jc)
        took = obs.command_results
        print(f"step 2's follow-up: {sum(took[:len(fix)])} of {len(fix)} taken; jobs: {sum(took[len(fix):])} of "
              f"{len(jc)} harvest orders taken ({ {s: dict(Counter(j for _, _, j, _ in given[s])) for s in agents} })",
              flush=True)
        bad += mines_off(st, obs, 3)
        building = [(u.type, u.owner) for u in obs.units if u.alive and int(u.flags) & int(UnitFlags.CONSTRUCTING)]
        print(f"under construction at step 3: {building}; gold, lumber: "
              f"{[(obs.players[s].gold, obs.players[s].lumber) for s in (0, 1)]}", flush=True)
        bad += len(building)
        home = {s: starts[s] for s in (0, 1)}
        started = Counter()
        t, last = 3, None
        while obs.game_time < a.minutes * 60 and not obs.game_over:
            obs = g.step()
            t += 1
            for e in obs.events:
                if int(e.kind) in (EventKind.TRAIN_START, EventKind.RESEARCH_START, EventKind.CONSTRUCT_START,
                                   EventKind.UPGRADE_START):
                    started[(EventKind(int(e.kind)).name, fx.rawcode(int(e.b)))] += 1
            if last is None or obs.game_time - last >= 30:
                last = obs.game_time
                line = []
                for s in (0, 1):
                    p = obs.players[s]
                    army = [u for u in obs.units if u.owner == s and u.alive and not u.is_structure and not u.is_worker]
                    far = sorted(math.hypot(u.x - home[s][0], u.y - home[s][1]) for u in army)
                    line.append(f"{s}: food {p.food_used}/{p.food_cap} gold {p.gold} +{p.gold_gathered - p0[s][0]} "
                                f"lumber +{p.lumber_gathered - p0[s][1]} army {len(army)} "
                                f"median from home {far[len(far) // 2] if far else 0:.0f}")
                print(f"{obs.game_time / 60:5.1f} min  " + " | ".join(line), flush=True)
        print(f"started: {dict(started)}", flush=True)
        print(f"over: {obs.game_over} results {[obs.players[s].result.name for s in (0, 1)]}", flush=True)
    return 0 if bad == 0 else 2


def play(a) -> int:
    """A game of a state with plans (a line of its file, STATE.json: late_from_demos.py --plans 1) as selfplay.play_one
    plays a training game with --late-z 1, both sides a run's policy on the CPU: each side's target (its items and their
    times, what it holds, the input's range), the strategy potential's parts at the first steps and each time they change,
    what was credited and when, the episode row's z fields."""
    import shutil
    import tempfile

    import torch

    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame import selfplay, strategy as strat
    from warcraftsim.fullgame.costs import order_costs
    from warcraftsim.fullgame.evalworker import run_config
    from warcraftsim.fullgame.model import load
    from warcraftsim.fullgame.trace import unit_values
    from warcraftsim.runtime.instance import Agent, GameSetup as Setup
    run = Path(a.run).resolve()
    st0 = json.loads(Path(a.state).read_text())
    races = list(st0["races"])
    c = json.loads((run / "run.json").read_text())["config"]
    root = run.parent.parent
    ck = Path(a.checkpoint) if a.checkpoint else sorted((run / "checkpoints").glob("*.pt"))[-1]
    out = Path(tempfile.mkdtemp(prefix="lateprobe", dir=a.tmp or None))
    cfg, _ = run_config(run, "probe", 1, 0)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    states = out / "state.jsonl"
    states.write_text(json.dumps(st0) + "\n")
    cfg.update(run_dir=str(out), map=st0["map"], z_pool=str(root / c["z_pool"]), z_share=1.0, z_reward=float(c["z_reward"]),
               z_mix_reward=float(c.get("z_mix_reward", 0.0)), z_mode_reward=float(c.get("z_mode_reward", 0.0)),
               z_build_labels=int(c.get("z_build_labels", 0)), teach=int(c.get("teach", 0)), win_balance=float(c.get("win_balance", 0.0)),
               scenarios=["late"], scenario_share=1.0, late_states=[str(states)], late_z=1, eval_checkpoints=[],
               max_minutes=a.minutes, max_steps=int(a.minutes * 60 / cfg["step_seconds"]) + 20, device="cpu",
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

    target, miss, gap = selfplay.target, strat.mismatch, strat.mix_gap
    seen: dict = {}  # (each target's times by id: its side)
    last: dict = {}
    starts = []

    def targets(cfg_, view, st, *rest, **kw):  # (the target play_one sets, and the view's input as the network reads it)
        got = target(cfg_, view, st, *rest, **kw)
        seen[id(got["t"])] = len(seen)
        print(f"target {len(seen) - 1}: {got['race']} against {got['opp']}, src {got['src']}, won {got['won']}, end {got.get('end')}: "
              f"{len(got['t'])} items {sorted(got['t'].items(), key=lambda kv: kv[1])}; army {({k: v for k, v in got['n'].items() if strat.is_army(k)})}; "
              f"holds {sorted(got.get('held') or ())}", flush=True)
        print(f"  input: {len(view.z)} numbers (the network's {fx.ZDIM}), range {view.z.min():.3f}..{view.z.max():.3f}", flush=True)
        return got

    def mismatch(made, st, items, now):
        v = miss(made, st, items, now)
        k = seen.get(id(st["t"]), "?")
        if k not in last or last[k][0] != v or now < 4.0:
            print(f"  {now:6.1f} s  target {k}: mismatch {v}, potential {-cfg['z_reward'] * v:.3f}", flush=True)
        last[k] = (v, dict(made))
        return v

    def mix_gap(trained, st, least=5):
        g = gap(trained, st, least)
        k = seen.get(id(st["t"]), "?")
        if g and (k not in starts):
            starts.append(k)
            print(f"  target {k}: the mix's gap first {g:.3f} ({dict(trained)})", flush=True)
        return g
    selfplay.target, strat.mismatch, strat.mix_gap = targets, mismatch, mix_gap

    class Rows:
        def __init__(self):
            self.ep = None

        def put(self, msg):
            if "episode" in msg:
                self.ep = msg["episode"]

    rows = Rows()
    slots = [Agent(races[s], handicap=100, difficulty="normal") for s in (0, 1)]
    g_setup = Setup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"], max_game_seconds=a.minutes * 60,
                    victory="decisive", window=(320, 240), draw=False, wait_floor_ms=cfg["wait_floor_ms"],
                    native_obs=cfg["native_obs"], d3d_thread=False, render_threads=0, nice=10, shim_dir=a.shim)
    try:
        with GameInstance(g_setup, name=f"lateprobe_play{a.name}", timeout=300) as g:
            for tries in range(16):
                obs = g.start() if tries == 0 else g.restart()
                if late.fit(st0, st0["map"], races, [(obs.players[s].start_x, obs.players[s].start_y) for s in (0, 1)]):
                    break
            else:
                return 1
            print(f"try {tries + 1}: fits", flush=True)
            ep = selfplay.play_one(g, obs, cfg, vocab, races, 0, {"name": "self", "kind": "self"}, infer, rows, 0)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    print(f"over at {ep['game_time']:.0f} s, outcome {ep['outcome']}, scenario {ep.get('scenario')}, late {ep.get('late_minute')}, "
          f"balance features {ep.get('tb_x')}", flush=True)
    for k in ("z", "z_items", "z_miss", "z_t", "z_made", "z_missed", "z_extra", "z_mix_gap", "z_n", "z_build", "taught"):
        print(f"  {k}: {ep.get(k)}", flush=True)
    print(f"  trained {ep['prod']['trained']}, built {ep['prod']['built']}, research {ep['prod']['research']}, "
          f"expansions {ep['prod']['expansions']}", flush=True)
    print(f"  the other side: trained {ep['opp_prod']['trained']}, built {ep['opp_prod']['built']}, expansions "
          f"{ep['opp_prod']['expansions']}; both as credited {[(k, v[1]) for k, v in last.items()]}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("capture", "start", "play"))
    ap.add_argument("state", nargs="?")
    ap.add_argument("--run", default="runs/fgself-168", help="play: the run whose recipe and policy play")
    ap.add_argument("--checkpoint", default="", help="play: its checkpoint (default: its newest)")
    ap.add_argument("--threads", type=int, default=2, help="play: torch's threads")
    ap.add_argument("--tmp", default="", help="play: where its temporary run goes")
    ap.add_argument("--map", default="echofast2")
    ap.add_argument("--races", default="human,undead")
    ap.add_argument("--minute", type=float, default=10.0)
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--ai", type=int, default=0)
    ap.add_argument("--out", default="late-state.json")
    ap.add_argument("--name", default="", help="the instance names' suffix (two probes at once)")
    ap.add_argument("--shim", default=str(Path(__file__).resolve().parents[1] / "build" / "shim"))
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    random.seed()
    return capture(a) if a.what == "capture" else start(a) if a.what == "start" else play(a)


if __name__ == "__main__":
    raise SystemExit(main())
