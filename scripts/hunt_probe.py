"""Why games between agents tie at the limit with one side far ahead, and whether a lesson could end them.

    python3 scripts/hunt_probe.py watch --races undead,human --run runs/fgself-173 --checkpoint PT --out DIR [--scenario base2] [--hunt 1] [--ai 1]
    python3 scripts/hunt_probe.py follow --races human,orc --run runs/fgself-173 --out DIR [--minutes 12]

watch: one real training game (selfplay.play_one, its scenario always: base2, natural, late or none; both sides the run's policy
on the CPU, --ai 1: the second the built-in AI) with every step logged from the outside: a proxy of the game instance sees each
step's orders and rows, a wrapper of BCAgent.commands each step's samples. Written to DIR/<tag>.json and a line a minute to
stdout, for each side: the enemy's buildings (where, when first and last in its sight, when dead), where its army stands against
the enemy's natural and main, what its army was ordered (a point's zone against the truth: the natural, the dead main, near a
building it saw or did not, its own base, elsewhere), what it could see, its workers' orders and deaths by zone. With the point
head's reach of every gold mine (the bins' round trip) printed first. --hunt 1: the hunt lesson on (labels only, as in training):
each lesson given, by either side, with what was in its view then.

follow: no policy. A real game with base2 for both sides, the first side's army (--army units of its race's first unit) spawned
at the second's main with 4 of its farms beside (the hall alone is too little to see gone: HUNT_CLEARED), the second's main then
removed in its sight (RemoveUnit: it saw them gone), the second giving no orders.
A Teacher with the hunt lesson looks on for the first side and its lessons are carried out as orders: the labels the network
would get (order, bins, the point they mean), when the natural's base comes in view; then all of the army attack-moves to what
it remembers until the game is over or --minutes.

Instance names start with "base2probe" (--name: a suffix). Nice 19. No learner, no GPU.

Results (2026-11-01, fgself-173's 50855953, echofast2, the run's settings, no balance tax). watch, base2 between agents, 16 games:
13 ties at 40 minutes. In those, the last 10 minutes of the side ahead (medians over games): 19.6 army units, 9.8 at the enemy's
dead main, 0 at its natural, no enemy building in sight, 187 of its 363 point orders a minute to that main, 0 to the natural;
over the whole games 44 of 170,502 point orders named the natural. Its attack-move at a point was open at all 62,374 looks, and
the bins name every mine to within 27. In 8 of the 13 it never saw a hall that survived; in 5 it saw one and left. The side behind
had no army and no workers from minute 8 to 32 (median 16), having trained 15 to 116 workers; of 325 worker deaths logged (8 games)
228 at its own dead main, 97 elsewhere, none at its natural.
follow, each race hunting and hunted once: human > orc, undead > human, orc > night elf, night elf > undead: labels of the attack-move
at the taught point (open, bins meaning a point within 27), the natural's base in view at 141-173 s after 1-2 lessons, won at 163-201 s.
watch --hunt 1, labels only, the policy not trained on them: no start 6 games 0 lessons, natural 6 games 18 (all in its one tie), late 6
games 89 (in 2 of its 4 ties), base2 8 games 222 (in 4 of its 5 ties and 2 of its 3 decided games), against the AI 8 games 0; none of the
329 with an enemy building in view. The 3 ties without a lesson: the side ahead's army, heroes counted, was 2-6 units (HUNT_ARMY: 6
without heroes), so most likely that gate; not logged directly.
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
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.pop("WAYLAND_DISPLAY", None)  # (games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame import late, scenarios  # noqa: E402
from warcraftsim.protocol import ImmediateOrder, PointOrder, RemoveUnit, Spawn, TargetOrder  # noqa: E402
from warcraftsim.runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup  # noqa: E402

ATTACK, MOVE = 851983, 851986
NEAR_NATURAL, NEAR_MAIN, NEAR_STRUCT = 1300.0, 2000.0, 800.0


def setup(map_name: str, slots, shim: str, seconds: float, native: bool = False) -> GameSetup:
    return GameSetup(map=map_name, slots=slots, step_seconds=0.5, draw=False, native_obs=native, d3d_thread=False,
                     render_threads=0, nice=19, warm_spare=False, window=(320, 240), max_game_seconds=seconds,
                     victory="decisive", shim_dir=shim)


def code_of(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


MINE_TYPES = {code_of(c) for c in ("ngol", "ugol", "egol")}
HALLS = {code_of(c) for c in ("htow", "hkee", "hcas", "ogre", "ostr", "ofrt", "unpl", "unp1", "unp2", "etol", "etoa", "etoe")}
OWN_MINES = {code_of(c) for c in ("ugol", "egol")}


def map_facts(rows: np.ndarray, starts: dict) -> dict:
    """Each side's start, main mine, natural mine; all the mines (from a first step's rows: every unit is listed)."""
    m = rows[np.isin(rows[:, fx.C_TYPE], list(MINE_TYPES))]
    xy = sorted({(int(r[fx.C_X]), int(r[fx.C_Y])) for r in m})
    out = {"mines": xy, "starts": starts, "natural": {}, "main": {}}
    for s, (sx, sy) in starts.items():
        d = sorted((math.hypot(x - sx, y - sy), x, y) for x, y in xy)
        out["main"][s] = d[0][1:]
        out["natural"][s] = next(p[1:] for p in d if p[0] > scenarios.NATURAL_FROM_START)
    return out


def reach(facts: dict) -> None:
    """Whether the point head can name each mine, in each side's frame: its bins, the point they mean, the error."""
    for s, (sx, sy) in facts["starts"].items():
        sign = 1.0 if sx == fx.STARTS[0][0] else -1.0
        for x, y in facts["mines"]:
            bx, by = fx.world_bins(float(x), float(y), sign)
            wx, wy = fx.bins_world(int(bx), int(by), sign)
            print(f"  side {s} frame: mine ({x}, {y}) bins ({int(bx)}, {int(by)}) of {fx.BINS * fx.FINE} -> ({wx:.0f}, {wy:.0f}), "
                  f"off by {math.hypot(wx - x, wy - y):.0f}", flush=True)


class Watch:
    """A game instance's stand-in: the same step, each one's orders and rows logged."""

    def __init__(self, g, obs, races: list, values, hero_level_value: float, worker_cap: int):
        from warcraftsim.fullgame.play import unit_rows
        self.g, self.unit_rows, self.races = g, unit_rows, races
        self.values, self.hlv, self.cap = values, hero_level_value, worker_cap
        self.t = 0
        self.rows = unit_rows(obs, 0)
        starts = {s: (obs.players[s].start_x, obs.players[s].start_y) for s in (0, 1)}
        self.facts = map_facts(self.rows, starts)
        print(f"starts {starts}; naturals {self.facts['natural']}; mines {self.facts['mines']}", flush=True)
        reach(self.facts)
        self.structs: dict = {0: {}, 1: {}}  # (owner -> id -> what happened to it)
        self.minute_at = 0.0
        self.log: list = []
        self.acc = {s: Counter() for s in (0, 1)}
        self.dest = {s: Counter() for s in (0, 1)}  # (a coarse grid of the army's point orders: the minute's)
        self.sampled = {s: Counter() for s in (0, 1)}
        self.workers = {s: Counter() for s in (0, 1)}  # (the minute's: workers' orders by zone, deaths by zone)
        self.dead_workers: set = set()
        self.first_near = {0: None, 1: None}  # (the first time an army unit stood near the enemy's natural)

    def __getattr__(self, name):
        return getattr(self.g, name)

    def zone(self, owner: int, x: float, y: float) -> str:
        nx, ny = self.facts["natural"][owner]
        sx, sy = self.facts["starts"][owner]
        if math.hypot(x - nx, y - ny) < NEAR_NATURAL:
            return "natural"
        if math.hypot(x - sx, y - sy) < NEAR_MAIN:
            return "main"
        return "other"

    def _orders(self, cmds) -> None:
        rows = self.rows
        ids = {int(i): k for k, i in enumerate(rows[:, fx.C_ID].tolist())}
        structs = rows[(rows[:, fx.C_FLAGS] & fx.STRUCTURE > 0) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]
        for c in cmds or ():
            unit = getattr(c, "unit", None)
            k = ids.get(int(unit)) if unit is not None else None
            if k is None:
                continue
            r = rows[k]
            s = int(r[fx.C_OWNER])
            if s in (0, 1) and int(r[fx.C_FLAGS]) & fx.WORKER:  # (a worker's order: where it is sent, by its owner's zones)
                j = ids.get(int(c.target)) if isinstance(c, TargetOrder) else None
                x, y = ((c.x, c.y) if isinstance(c, PointOrder)
                        else (float(rows[j, fx.C_X]), float(rows[j, fx.C_Y])) if j is not None else (None, None))
                if x is not None:
                    self.workers[s][f"to {self.zone(s, x, y)}"] += 1
                continue
            if s not in (0, 1) or int(r[fx.C_FLAGS]) & fx.STRUCTURE:
                continue
            acc = self.acc[s]
            e = 1 - s
            if isinstance(c, PointOrder):
                kind = {ATTACK: "attack", MOVE: "move"}.get(c.order, f"o{c.order}")
                ex, ey = self.facts["natural"][e]
                sx, sy = self.facts["starts"][e]
                es = structs[structs[:, fx.C_OWNER] == e]
                near = len(es) and float(np.hypot(es[:, fx.C_X] - c.x, es[:, fx.C_Y] - c.y).min()) < NEAR_STRUCT
                where = ("enemy natural" if math.hypot(c.x - ex, c.y - ey) < NEAR_NATURAL
                         else "enemy main" if math.hypot(c.x - sx, c.y - sy) < NEAR_MAIN
                         else "enemy building" if near
                         else "own base" if self.zone(s, c.x, c.y) != "other" else "elsewhere")
                acc[f"{kind} -> {where}"] += 1
                self.dest[s][(int(round(c.x / 1000)) * 1000, int(round(c.y / 1000)) * 1000)] += 1
            elif isinstance(c, TargetOrder):
                j = ids.get(int(c.target))
                if j is None:
                    acc["target: gone"] += 1
                    continue
                t = rows[j]
                who = int(t[fx.C_OWNER])
                st = int(t[fx.C_FLAGS]) & fx.STRUCTURE
                acc["target: " + ("enemy building" if who == e and st else "enemy unit" if who == e else "own" if who == s
                                  else "neutral")] += 1
            elif isinstance(c, ImmediateOrder):
                acc["immediate"] += 1

    def step(self, cmds=None):
        self._orders(cmds)
        obs = self.g.step(cmds) if cmds is not None else self.g.step()
        self.t += 1
        rows = self.rows = self.unit_rows(obs, self.t)
        fl = rows[:, fx.C_FLAGS]
        for owner in (0, 1):
            viewer = 1 - owner
            b = rows[(rows[:, fx.C_OWNER] == owner) & (fl & fx.STRUCTURE > 0)]
            tab = self.structs[owner]
            for i, typ, x, y, f, vis in b[:, [fx.C_ID, fx.C_TYPE, fx.C_X, fx.C_Y, fx.C_FLAGS, fx.C_VIS]].tolist():
                e = tab.get(i)
                if e is None:
                    e = tab[i] = {"type": fx.rawcode(int(typ)), "x": x, "y": y, "made": round(obs.game_time),
                                  "zone": self.zone(owner, x, y), "seen": None, "last": None, "dead": None}
                if f & fx.DEAD:
                    if e["dead"] is None:
                        e["dead"] = round(obs.game_time)
                elif (vis >> viewer) & 1:
                    if e["seen"] is None:
                        e["seen"] = round(obs.game_time)
                    e["last"] = round(obs.game_time)
        for i, s, x, y in rows[(fl & fx.WORKER > 0) & (fl & fx.DEAD > 0)][:, [fx.C_ID, fx.C_OWNER, fx.C_X, fx.C_Y]].tolist():
            if i not in self.dead_workers and s in (0, 1):
                self.dead_workers.add(i)
                self.workers[s][f"died at {self.zone(s, x, y)}"] += 1
        for s in (0, 1):
            nx, ny = self.facts["natural"][1 - s]
            army = rows[(rows[:, fx.C_OWNER] == s) & (fl & (fx.STRUCTURE | fx.WORKER | fx.DEAD) == 0)]
            if len(army) and self.first_near[s] is None and (np.hypot(army[:, fx.C_X] - nx, army[:, fx.C_Y] - ny) < NEAR_NATURAL).any():
                self.first_near[s] = round(obs.game_time)
        if obs.game_time - self.minute_at >= 60.0 or obs.game_over:
            self.minute_at = obs.game_time
            self._minute(obs, rows)
        return obs

    def _minute(self, obs, rows) -> None:
        from warcraftsim.fullgame import selfplay
        fl = rows[:, fx.C_FLAGS]
        mat = selfplay.material(obs, self.values, self.hlv, self.cap)
        row = {"minute": round(obs.game_time / 60.0, 1), "material": [round(float(mat[0])), round(float(mat[1]))], "sides": {}}
        parts = []
        for s in (0, 1):
            e = 1 - s
            nx, ny = self.facts["natural"][e]
            ex, ey = self.facts["starts"][e]
            ox, oy = self.facts["starts"][s]
            own = rows[(rows[:, fx.C_OWNER] == s) & (fl & fx.DEAD == 0)]
            of = own[:, fx.C_FLAGS]
            army = own[of & (fx.STRUCTURE | fx.WORKER) == 0]
            ax, ay = army[:, fx.C_X].astype(float), army[:, fx.C_Y].astype(float)
            enemy_b = rows[(rows[:, fx.C_OWNER] == e) & (fl & fx.STRUCTURE > 0) & (fl & fx.DEAD == 0)]
            seen_b = enemy_b[(enemy_b[:, fx.C_VIS] >> s) & 1 == 1]
            zones = Counter(self.zone(s, float(x), float(y)) for x, y in own[of & fx.STRUCTURE > 0][:, [fx.C_X, fx.C_Y]].tolist())
            p = obs.players.get(s)
            d = {"army": len(army), "workers": int(np.count_nonzero(of & fx.WORKER)), "buildings": dict(zones),
                 "centre": [round(float(ax.mean())), round(float(ay.mean()))] if len(army) else None,
                 "at enemy natural": int((np.hypot(ax - nx, ay - ny) < NEAR_NATURAL).sum()),
                 "at enemy main": int((np.hypot(ax - ex, ay - ey) < NEAR_MAIN).sum()),
                 "at home": int((np.hypot(ax - ox, ay - oy) < NEAR_MAIN).sum()),
                 "nearest enemy natural": round(float(np.hypot(ax - nx, ay - ny).min())) if len(army) else None,
                 "idle": int((army[:, fx.C_ORDER] == 0).sum()),
                 "enemy buildings": len(enemy_b), "enemy buildings in sight": len(seen_b),
                 "enemy halls": int(np.isin(enemy_b[:, fx.C_TYPE], list(HALLS | OWN_MINES)).sum()),
                 "gold": p.gold if p else 0, "lumber": p.lumber if p else 0,
                 "orders": dict(self.acc[s]), "points": {f"{k[0]},{k[1]}": v for k, v in self.dest[s].most_common(6)},
                 "sampled": dict(self.sampled[s]), "workers_log": dict(self.workers[s])}
            row["sides"][s] = d
            parts.append(f"{self.races[s][:2]} army {d['army']:3d} w {d['workers']:2d} b {d['buildings']} at-nat {d['at enemy natural']:2d} "
                         f"at-main {d['at enemy main']:2d} home {d['at home']:2d} idle {d['idle']:2d} sees {len(seen_b)}/{len(enemy_b)}")
            self.acc[s], self.dest[s], self.sampled[s], self.workers[s] = Counter(), Counter(), Counter(), Counter()
        self.log.append(row)
        print(f"{row['minute']:5.1f} mat {row['material']} | " + " | ".join(parts), flush=True)


def _late_states(c: dict, run: Path) -> list:
    """The run's late-game states that this machine still has (absolute paths; a share's ":0.5" kept)."""
    got, root = [], run.parent.parent
    for p in str(c.get("late_states") or "").split(","):
        path, _, share = p.strip().partition(":")
        if path and (root / path).exists():
            got.append(str(root / path) + (f":{share}" if share else ""))
    return got


def run_settings(a) -> tuple[dict, dict, Path]:
    """The run's settings as play_one needs them, the features' module set up as its learner has it -> (cfg, vocab, the
    checkpoint)."""
    import torch

    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame.costs import order_costs
    from warcraftsim.fullgame.evalworker import run_config
    from warcraftsim.fullgame.trace import unit_values
    run = Path(a.run).resolve()
    c = json.loads((run / "run.json").read_text())["config"]
    root = run.parent.parent
    ck = Path(a.checkpoint) if a.checkpoint else sorted((run / "checkpoints").glob("*.pt"))[-2]
    cfg, _ = run_config(run, "probe", 1, 0)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg.update(map=c["map"], z_pool=str(root / c["z_pool"]), z_share=float(c["z_share"]), z_reward=float(c["z_reward"]),
               z_mix_reward=float(c.get("z_mix_reward", 0.0)), z_mode_reward=float(c.get("z_mode_reward", 0.0)),
               z_build=int(c.get("z_build", 0)), z_workers=int(c.get("z_workers", 0)), z_modes=float(c.get("z_modes", 0.0)),
               z_build_labels=int(c.get("z_build_labels", 0)), teach=int(c.get("teach", 0)), hunt_lesson=a.hunt,
               win_balance=float(c.get("win_balance", 0.0)), scenarios=[a.scenario] if a.scenario != "none" else [],
               scenario_share=1.0, late_states=(_late_states(c, run) if a.scenario == "late" else []), late_z=0,
               eval_checkpoints=[], max_minutes=a.minutes,
               max_steps=int(a.minutes * 60 / cfg["step_seconds"]) + 20, device="cpu", values=unit_values(), order_names={})
    if a.no_z:
        cfg["z_share"] = 0.0
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
    cfg["checkpoint_version"] = int(checkpoint.get("update", checkpoint.get("version", 0)) or 0)
    del checkpoint
    vocab = json.loads(Path(cfg["vocab"]).read_text())
    cfg["costs"] = order_costs(vocab, cfg["map"]).tolist() if cfg["avail_mask"] else None
    cfg["costs_array"] = np.asarray(cfg["costs"], np.int64) if cfg["avail_mask"] else None
    cfg["learner_pid"] = os.getpid()
    torch.set_num_threads(a.threads)
    return cfg, vocab, ck


def play_watch(a) -> int:
    import torch

    from warcraftsim.fullgame import selfplay
    from warcraftsim.fullgame.model import load
    from warcraftsim.fullgame.play import BCAgent
    from warcraftsim.fullgame.teacher import Teacher
    races = a.races.split(",")
    tmp = Path(tempfile.mkdtemp(prefix="huntwatch", dir=a.out))
    cfg, vocab, ck = run_settings(a)
    cfg["run_dir"] = str(tmp)
    selfplay.publish(load(ck, "cpu", vocab=vocab if fx.NEUTRAL else None)[0], cfg["checkpoint_version"], tmp)
    infer = selfplay.Inference(selfplay.Nets(cfg, torch.device("cpu")), torch.device("cpu"), 2, compile_=False,
                               buckets=(1, 2), entities=tuple(cfg["infer_entities"]))
    watch = {}
    commands = BCAgent.commands

    def commands_(self, st, order, tgt, bx, by):
        """What the policy sampled for its army, before the agent's filters; whether its order is open."""
        w = watch.get("w")
        if w is not None:
            n = min(st["n_own"], fx.MAX_OWN, len(order))
            sel = st["sel"]
            acc = w.sampled[self.player]
            army = sel[:n, fx.C_FLAGS] & (fx.STRUCTURE | fx.WORKER) == 0
            acc["army"] += int(army.sum())
            acc["given none"] += int((np.asarray(order[:n])[army] == 0).sum())
            av = st.get("avail")
            for key, kind in ((ATTACK, fx.POINT), (ATTACK, fx.UNIT)):
                c_ = self.enc.order_index.get((key, kind))
                if c_ is not None:
                    acc[f"attack kind {kind} chosen"] += int((np.asarray(order[:n])[army] == c_).sum())
                    acc[f"attack kind {kind} open"] += int(av is None or bool(av[c_]))
                    acc[f"attack kind {kind} looked"] += 1
                else:
                    acc[f"attack kind {kind} not in vocabulary"] += 1
            e = sel[st["n_own"]:]
            acc["enemy entities"] += int((e[:, fx.C_OWNER] == 1 - self.player).sum())
        return commands(self, st, order, tgt, bx, by)
    BCAgent.commands = commands_
    fired: list = []
    hunt = Teacher._hunt

    def hunt_(self, rows, now, t):
        """Each lesson given, for both sides; and whether a building of the enemy was in the side's view then."""
        out = hunt(self, rows, now, t)
        if out:
            v = rows[(rows[:, fx.C_OWNER] == 1 - self.side) & (rows[:, fx.C_FLAGS] & fx.STRUCTURE > 0)
                     & (rows[:, fx.C_FLAGS] & fx.DEAD == 0) & ((rows[:, fx.C_VIS] >> self.side) & 1 == 1)]
            fired.append({"side": self.side, "at": round(now), "units": len(out), "x": out[0].x, "y": out[0].y,
                          "in_view": len(v), "known": len(self.hunt_known), "gone": self.hunt_gone})
        return out
    Teacher._hunt = hunt_

    class Rows:
        def __init__(self):
            self.ep = None

        def put(self, msg):
            if "episode" in msg:
                self.ep = msg["episode"]

    rows = Rows()
    slots = [Agent(races[0], handicap=100, difficulty="normal"),
             BuiltinAI(races[1], "insane") if a.ai else Agent(races[1], handicap=100, difficulty="normal")]
    g_setup = setup(cfg["map"], slots, a.shim, a.minutes * 60, bool(cfg["native_obs"]))
    t0 = time.time()
    tag = a.tag or f"{races[0]}-{races[1]}-{int(t0)}"
    try:
        with GameInstance(g_setup, name=f"base2probe{a.name}", timeout=600) as g:
            obs = g.start()
            w = watch["w"] = Watch(g, obs, races, cfg["values"], cfg.get("hero_level_value", 0.0), cfg.get("worker_cap", 0))
            opp = {"name": "script:ai-insane", "kind": "ai"} if a.ai else {"name": "self", "kind": "self"}
            ep = selfplay.play_one(w, obs, cfg, vocab, races, 0, opp, infer, rows, 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = {"races": races, "checkpoint": str(ck), "wall": round(time.time() - t0), "facts": {k: v for k, v in w.facts.items()},
           "structs": w.structs, "first_near": w.first_near, "log": w.log, "fired": fired,
           "ep": {k: ep.get(k) for k in ("outcome", "material_lead", "game_time", "scenario", "z", "taught", "taught_kept", "prod", "opp_prod",
                                         "race", "opponent_race", "hunt_lessons", "hunt_found_at")}}
    path = Path(a.out) / f"{tag}.json"
    path.write_text(json.dumps(out))
    print(f"RESULT {races} scenario {ep.get('scenario')} outcome {ep['outcome']} lead {ep['material_lead']} at {ep['game_time'] / 60:.1f} min, "
          f"lessons {len(fired)}, wall {out['wall']} s: {path}", flush=True)
    return 0


def follow(a) -> int:
    """(The module's docstring.)"""
    from warcraftsim.fullgame import selfplay
    from warcraftsim.fullgame.play import BCAgent, unit_rows
    from warcraftsim.fullgame.teacher import HALL_CODES, OWN_MINE_CODES, Teacher
    races = a.races.split(",")
    a.hunt = 1
    cfg, vocab, _ = run_settings(a)
    slots = [Agent(races[0], handicap=100, difficulty="normal"), Agent(races[1], handicap=100, difficulty="normal")]
    with GameInstance(setup(cfg["map"], slots, a.shim, a.minutes * 60, False), name=f"base2probe{a.name}", timeout=600) as g:
        obs = g.start()
        rows = unit_rows(obs, 0)
        starts = {s: (obs.players[s].start_x, obs.players[s].start_y) for s in (0, 1)}
        facts = map_facts(rows, starts)
        nat = facts["natural"][1]
        print(f"{races[0]} hunts {races[1]}: starts {starts}, the enemy's natural mine {nat}", flush=True)
        cmds, state = scenarios.base2(obs, [0, 1], races)
        obs = g.step(cmds)
        print(f"setup refused {len(obs.command_results) - sum(obs.command_results)} of {len(obs.command_results)}", flush=True)
        obs = g.step(late.again(state, unit_rows(obs, 1)))
        fix = late.again(state, unit_rows(obs, 2))
        ex, ey = starts[1]
        unit = scenarios.UNITS[races[0]][0]
        bot = BCAgent(None, vocab, 0, None, costs=cfg.get("costs_array"), steps=selfplay.research_steps(vocab, cfg["map"]))
        bot.enc.mine_full = 12500.0 * selfplay.mine_scale(cfg["map"])
        bot.begin(obs, [fx.RACES.index(r) for r in races])
        army = [Spawn(0, unit, *scenarios._ring(ex + (300.0 if ex < 0 else -300.0), ey - 300.0, k)) for k in range(1, a.army + 1)]
        farm = {"human": "hhou", "orc": "otrb", "undead": "uzig", "nightelf": "emow"}[races[1]]  # (buildings to see gone)
        away = 600.0 if ex < 0 else -600.0
        farms = [Spawn(1, farm, ex + away, ey + 250.0 * k) for k in (-1, 0, 1, 2)]
        obs = g.step(fix + army + farms)
        print(f"step 3: army taken {sum(obs.command_results[len(fix):])} of {len(army) + len(farms)}", flush=True)
        tc = Teacher(0, races[0], random.Random(0), hunt=1)
        tc.first, tc.pick = None, None
        t, removed, lessons, end_order, first_view = 3, False, 0, -100, None
        while obs.game_time < a.minutes * 60 and not obs.game_over:
            rows = unit_rows(obs, t)
            st = bot.observe(obs, t, rows)
            c = [x for x in tc.commands(rows, obs.players.get(0), obs.game_time, t) if isinstance(x, PointOrder) and x.order == ATTACK]
            cmds = list(c)
            fl = rows[:, fx.C_FLAGS]
            enemy = rows[(rows[:, fx.C_OWNER] == 1) & (fl & fx.DEAD == 0)]
            seen = enemy[(enemy[:, fx.C_VIS] & 1 == 1) & (enemy[:, fx.C_FLAGS] & fx.STRUCTURE > 0)]
            base = seen[fx.isin(seen[:, fx.C_TYPE], HALL_CODES) | fx.isin(seen[:, fx.C_TYPE], OWN_MINE_CODES)]
            near_nat = np.hypot(base[:, fx.C_X] - nat[0], base[:, fx.C_Y] - nat[1]) < NEAR_NATURAL
            if not removed and t >= 12:  # (in sight of the army: its main goes, its workers too; the natural stays)
                d = np.hypot(enemy[:, fx.C_X] - ex, enemy[:, fx.C_Y] - ey)
                keep = np.hypot(enemy[:, fx.C_X] - nat[0], enemy[:, fx.C_Y] - nat[1]) < NEAR_NATURAL
                gone = enemy[(d < NEAR_MAIN) & ~keep]
                print(f"{obs.game_time:6.1f} s: in view {len(seen)} buildings; removing the main's {len(gone)} units: "
                      f"{Counter(fx.rawcode(int(v)) for v in gone[:, fx.C_TYPE].tolist()).most_common(6)}", flush=True)
                cmds += [RemoveUnit(int(i)) for i in gone[:, fx.C_ID].tolist()]
                removed = True
            if c:
                lessons += 1
                p = obs.players[0]
                me = np.asarray([t, 0, p.gold, p.lumber, p.food_used, p.food_cap, p.upkeep])
                y, _ = bot.enc.step_labels(bot.view, st, selfplay.label_rows(t, c), me, np.zeros((0, 5), np.int64), bot.trees, taught=True)
                lab = np.flatnonzero(y["y_order"] > 0)
                av = st.get("avail")
                k = lab[0] if len(lab) else None
                pts = {fx.bins_world(int(y["y_x"][i]), int(y["y_y"][i]), bot.view.sign) for i in lab}
                print(f"{obs.game_time:6.1f} s: lesson {lessons}: {len(c)} units to ({c[0].x:.0f}, {c[0].y:.0f}); labels {len(lab)}: order "
                      f"{bot.orders[int(y['y_order'][k])] if k is not None else None} open "
                      f"{bool(av[y['y_order'][k]]) if k is not None and av is not None else None}, "
                      f"bins {(int(y['y_x'][k]), int(y['y_y'][k])) if k is not None else None}, they mean "
                      f"{[(round(x_), round(y_)) for x_, y_ in pts]}; in view {len(seen)} known {len(tc.hunt_known)} gone {tc.hunt_gone}", flush=True)
            if near_nat.any() and first_view is None:
                first_view = obs.game_time
                print(f"{obs.game_time:6.1f} s: THE NATURAL'S BASE IN VIEW: "
                      f"{[(fx.rawcode(int(u[fx.C_TYPE])), int(u[fx.C_X]), int(u[fx.C_Y])) for u in base[near_nat]]}; "
                      f"the row's hunt_found_at {tc.hunt_found_at}, lessons {tc.hunt_lessons}", flush=True)
            if first_view is not None and t - end_order >= 20:  # (found: the rest of the way, as the policy would have to)
                places = list(tc.hunt_known.values())
                own = rows[(rows[:, fx.C_OWNER] == 0) & (fl & (fx.STRUCTURE | fx.WORKER | fx.DEAD) == 0)]
                if places and len(own):
                    ax, ay = float(own[:, fx.C_X].mean()), float(own[:, fx.C_Y].mean())
                    px, py = min(places, key=lambda p: math.hypot(p[0] - ax, p[1] - ay))
                    cmds += [PointOrder(int(i), ATTACK, float(px), float(py)) for i in own[:, fx.C_ID].tolist()]
                    end_order = t
            obs = g.step(cmds)
            t += 1
        rows = unit_rows(obs, t)
        left = [fx.rawcode(int(v)) for v in rows[(rows[:, fx.C_OWNER] == 1) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)][:, fx.C_TYPE].tolist()]
        print(f"RESULT {races}: found at {first_view} s (row {tc.hunt_found_at}), lessons {tc.hunt_lessons}, game over {obs.game_over} at "
              f"{obs.game_time:.0f} s, side 0 {obs.players[0].result.name if obs.players.get(0) else None}, the enemy's left "
              f"{Counter(left).most_common(5)}", flush=True)
    return 0 if first_view is not None else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("watch", "follow"))
    ap.add_argument("--races", default="undead,human")
    ap.add_argument("--run", default="runs/fgself-173")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--scenario", default="base2")
    ap.add_argument("--no-z", type=int, default=0)
    ap.add_argument("--hunt", type=int, default=0)
    ap.add_argument("--ai", type=int, default=0, help="watch: the second side the built-in AI (insane)")
    ap.add_argument("--army", type=int, default=20, help="follow: the hunter's units")
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--out", default=".")
    ap.add_argument("--tag", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--shim", default=str(Path(__file__).resolve().parents[1] / "build" / "shim"))
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    random.seed()
    return play_watch(a) if a.what == "watch" else follow(a)


if __name__ == "__main__":
    raise SystemExit(main())
