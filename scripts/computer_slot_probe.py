"""What a computer slot does for our side by itself: every order our units get that our side did not give.

    python3 scripts/computer_slot_probe.py play --slot computer,user --matchups human:orc,orc:undead --out DIR [--minutes 30]
    python3 scripts/computer_slot_probe.py units --slot user --races human,orc,undead,nightelf --out DIR [--place 1]
    python3 scripts/computer_slot_probe.py report DIR [DIR ...]

Our agent slots are computer slots with no AI script (GameSetup: WgcSlot.computer; agent_is_user=True makes the first agent
the local user slot instead). The engine still acts for a computer player's units. To see it, the harness is patched in
this process only (mapbuild.harness_source; the patched map has its own cache key): its order events (W3S_Issued,
GameSetup.record_ai_orders) also record an agent's unit's order when it was given outside the harness's command window
(w3s_quiet false: not the agent's command, not the harness's), its kind + 8. The probe takes those records out of the
observation before the policy sees it (Observation.issued as without the patch: the built-in AI's orders) and logs each with
its context: the unit's type, hit points and mana, the enemy units within 800, whether it lost hit points in the last 2 s,
its age, the agent's last command to it. Also logged: our units' spell effects (event 15: who gave the order, the agent or
the engine) and the items that left our heroes' inventories.

play: games of a run's checkpoint on the CPU (selfplay.play_one, an evaluation game: "current", no target, no tax) against
the untaxed insane AI on the run's evaluation map, our side player 0. --slot: computer, user or both (each matchup played
in each); the games run as threads of one process (--parallel), sharing its inference. Also logged: each game minute's gold
and lumber gathered, the idle workers (no order, not in a mine or burrow) summed over steps.
units: no policy. Player 0 (--slot) and player 1 (always a computer slot) are both agents of this script on the evaluation
map. Step 2: each side gets the same group 2,000 from player 0's start toward player 1 (creeps cleared): 4 footmen and 4 grunts,
the casters with autocasts (priest, sorceress, necromancer, druid of the talon, dryad, banshee, shaman, spell breaker; their
adept trainings researched), 4 heroes at level 3 with their three skills and 6 items (healing and mana potion, salve, lesser
clarity, scroll of regeneration, town portal); player 1's attack-moves into player 0's, which is given no order. Orc: a burrow
placed by the hall. Neither side's start workers get an order; at second 20 their orders are noted and player 0's idle ones
sent to its mine. Second 30: two workers of player 0's race spawned by its hall (as if trained), never ordered; their orders
noted at second 50. Second 60: 5 grunts and 3 huntresses of player 1 spawned at player 0's mine attack-move to its hall.
Second 100 (--away): player 0's group's survivors moved 900 toward its base; where they stand noted 30 s later. Over at
--minutes (3 for this mode). --place 1: the groups placed (protocol.Place: the harness removes an agent's placed unit's AI
guard post) instead of spawned as trained units are.
report: the logs of DIRs as tables per mode and slot: engine orders by order and unit kind (per game minute, their context),
units sent back to where they were made (the AI guard post), workers sent to work that our side never ordered, idle workers,
gold by minute 10, spells by who ordered them, items gone.

Instance names start with "cslotprobe". Nice 19, no GPU, no learner. The patch's order records are those of
EVENT_PLAYER_UNIT_ISSUED_*ORDER: an order the engine carries out without an event (auto-acquiring a target) is not seen.

Results (2026-11-03, fgself-179/50301674, (2)EchoIsles; 18 play games per slot kind on the same 8 matchups, 4 units games
each; docs/reference/engine-notes.md, "What a computer slot does by itself"): only as a computer slot did the engine put idle
and new workers to work, send trained units back to where they were made (1,209 orders in 16 of 18 games, up to 8,500
away), cast non-autocast spells (raise dead, purge, unholy frenzy, defend, cannibalize...), cast hero spells (~116, about as
many as the policy), use hero items, board burrows and ring the town bell, walk hurt night elves to moon wells. Default
autocasts (heal, slow, faerie fire, curse, spell steal) and the harvest cycle: both. Won 11 of 18 as a computer slot, 7 of 18
as the user slot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("WAYLAND_DISPLAY", None)  # (games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.protocol import (ClearCreeps, Place, PointOrder, SetResearch, Spawn, TargetOrder,  # noqa: E402
                                  rawcode)
from warcraftsim.runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup, exact_cuts  # noqa: E402

ENGINE = 8  # (the patched records' kind offset)
NEAR = 800.0
HARVEST, ATTACK, MOVE, SMART, STOP = 852018, 851983, 851986, 851971, 851972
USESLOT = {852008 + k: f"useslot{k + 1}" for k in range(6)}
BASIC = {"smart", "stop", "attack", "move", "patrol", "holdposition", "harvest", "returnresources", "resumeharvesting",
         "AImove"}

OLD = """    if w3s_agent[p] then
        if not w3s_shadow[p] or w3s_quiet then
            return
        endif
"""
NEW = """    if w3s_agent[p] then
        if w3s_quiet then
            return
        endif
        if not w3s_shadow[p] then
            if w3s_niss < 1300 then
                set w3s_iss[k] = GetHandleId(u)
                set w3s_iss[k + 1] = order
                set w3s_iss[k + 2] = kind + 8
                set w3s_iss[k + 3] = R2I(x)
                set w3s_iss[k + 4] = R2I(y)
                set w3s_iss[k + 5] = target
                set w3s_niss = w3s_niss + 1
            endif
            return
        endif
"""


def patch_harness() -> None:
    """The harness source as this process builds maps from it: an agent's units' orders from outside the command window
    recorded (kind + 8)."""
    from warcraftsim.data import mapbuild
    src = mapbuild.harness_source()
    if src.count(OLD) != 1:
        raise SystemExit("the harness's W3S_Issued changed: the probe's patch no longer applies")
    patched = src.replace(OLD, NEW)
    mapbuild.harness_source = lambda: patched


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


class Log:
    """A game instance's stand-in: each step's engine orders to `players`' units taken out of the observation (the
    policy sees the AI's orders only, as without the patch) and logged with their context."""

    def __init__(self, g, obs, players: dict[int, str]):
        from warcraftsim.fullgame.play import unit_rows
        self.g, self.unit_rows, self.players = g, unit_rows, players  # (player -> its slot: computer / user)
        self.names = {oid: name for name, oid in (obs.orders or {}).items()}
        self.names.update({k: v for k, v in USESLOT.items() if k not in self.names})
        self.t = 0
        self.rows = unit_rows(obs, 0)
        self.born: dict[int, int] = {}
        self.kind: dict[int, int] = {}
        self.born_xy: dict[int, tuple[int, int]] = {}  # (where each unit was first seen)
        self.hp: dict[int, deque] = {}
        self.items: dict[int, tuple] = {}
        self.given: dict[int, tuple[int, int]] = {}  # unit -> (step, order) of our side's last command to it
        self.engine_at: dict[int, tuple[int, int]] = {}  # unit -> (step, order) of the engine's last order to it
        self.recs: list[dict] = []
        self.spells: list[dict] = []
        self.gone: list[dict] = []
        self.commands = Counter()  # (player -> our side's commands)
        self.worker_steps, self.idle = Counter(), Counter()  # (player -> its living workers, idle ones, summed over steps)
        self.gathered: dict[int, list] = {}  # (player -> (minute, gold gathered, lumber gathered, workers) each minute)
        self.minutes = 0.0
        self._index(self.rows)

    def __getattr__(self, name):
        return getattr(self.g, name)

    def name(self, order: int) -> str:
        if order >= fx.TYPE_CODE:
            return rawcode(order)
        return self.names.get(order, str(order))

    def _index(self, rows) -> None:
        self.ix = {int(i): k for k, i in enumerate(rows[:, fx.C_ID].tolist())}
        fl = rows[:, fx.C_FLAGS]
        alive = fl & fx.DEAD == 0
        self.mobile = rows[alive & (fl & fx.STRUCTURE == 0)]
        self.structs = rows[alive & (fl & fx.STRUCTURE > 0)]
        for i, ty, hp, x, y in rows[:, [fx.C_ID, fx.C_TYPE, fx.C_HP, fx.C_X, fx.C_Y]].tolist():
            if self.kind.get(i) != ty:  # (a new unit: the game reuses a dead unit's id)
                self.kind[i] = ty
                self.born[i] = self.t
                self.born_xy[i] = (x, y)
                self.hp[i] = deque(maxlen=5)
                self.items.pop(i, None)
                self.given.pop(i, None)
                self.engine_at.pop(i, None)
            self.hp[i].append(hp)

    def owner(self, unit: int) -> int | None:
        k = self.ix.get(int(unit))
        return None if k is None else int(self.rows[k, fx.C_OWNER])

    def context(self, unit: int) -> dict:
        if int(unit) not in self.ix:
            return {"type": "?", "hero": False, "worker": False, "structure": False, "summoned": False, "hp": 1.0,
                    "mana": None, "enemies": 0, "creeps": 0, "hurt": False, "age": 0, "x": 0, "y": 0}
        r = self.rows[self.ix[int(unit)]]
        p = int(r[fx.C_OWNER])
        m = self.mobile
        foe = (m[:, fx.C_OWNER] != p) & (m[:, fx.C_OWNER] < 12)
        d = np.hypot(m[:, fx.C_X] - r[fx.C_X], m[:, fx.C_Y] - r[fx.C_Y])
        creeps = (m[:, fx.C_OWNER] >= 12) & (d < NEAR)
        h = self.hp.get(unit) or [int(r[fx.C_HP])]
        fl = int(r[fx.C_FLAGS])
        return {"type": rawcode(int(r[fx.C_TYPE])), "hero": bool(fl & 1), "worker": bool(fl & fx.WORKER),
                "structure": bool(fl & fx.STRUCTURE), "summoned": bool(fl & fx.SUMMONED),
                "hp": round(float(r[fx.C_HP]) / max(1, int(r[fx.C_MAXHP])), 2),
                "mana": round(float(r[fx.C_MANA]) / max(1, int(r[fx.C_MAXMANA])), 2) if r[fx.C_MAXMANA] > 0 else None,
                "enemies": int((foe & (d < NEAR)).sum()), "creeps": int(creeps.sum()),
                "hurt": bool(max(h) > int(r[fx.C_HP])), "age": self.t - self.born.get(unit, 0),
                "x": int(r[fx.C_X]), "y": int(r[fx.C_Y])}

    def target(self, target: int, unit: int, kind: int, p: int) -> str:
        if target == 0:
            return "point" if kind == 1 else "-"
        k = self.ix.get(int(target))
        if k is None:
            return "tree/item"
        r = self.rows[k]
        q = int(r[fx.C_OWNER])
        rel = "self" if int(r[fx.C_ID]) == unit else "own" if q == p else "neutral" if q >= 12 else "enemy"
        return f"{rel} {rawcode(int(r[fx.C_TYPE]))}"

    def step(self, cmds=None):
        cmds = list(cmds or [])
        for c in cmds:
            u = getattr(c, "unit", None)
            if u is not None:
                self.given[int(u)] = (self.t, int(getattr(c, "order", 0) or 0))
                p = self.owner(u)
                if p is not None:
                    self.commands[p] += 1
        obs = self.g.step(cmds)
        self.t += 1
        self.rows = rows = self.unit_rows(obs, self.t)
        self._index(rows)
        self.minutes = obs.game_ms / 60000.0
        ours = [o for o in obs.issued if o.kind >= ENGINE]
        obs.issued = [o for o in obs.issued if o.kind < ENGINE]
        for o in ours:
            p = self.owner(o.unit)
            if p not in self.players:
                continue
            c = self.context(o.unit)
            g = self.given.get(o.unit)
            dist = (round(math.hypot(o.x - c["x"], o.y - c["y"])) if o.kind - ENGINE in (1, 2) else None)
            self.recs.append({"t": self.t, "s": round(obs.game_ms / 1000), "player": p, "slot": self.players[p],
                              "order": o.order, "name": self.name(o.order), "kind": o.kind - ENGINE,
                              "target": self.target(o.target, o.unit, o.kind - ENGINE, p), "dist": dist,
                              "agent_ago": None if g is None else self.t - 1 - g[0],
                              "agent_same": g is not None and g[0] == self.t - 1 and g[1] == o.order,
                              "px": o.x if o.kind - ENGINE in (1, 2) else None, "py": o.y if o.kind - ENGINE in (1, 2) else None,
                              "born_xy": self.born_xy.get(o.unit), **c})
            self.engine_at[o.unit] = (self.t, o.order)
        for e in obs.events:
            if int(e.kind) != 15:
                continue
            p = self.owner(e.a)
            if p not in self.players:
                continue
            g, en = self.given.get(e.a), self.engine_at.get(e.a)
            by = ("engine" if en is not None and self.t - en[0] <= 4 and self.name(en[1]) not in BASIC
                  else "agent" if g is not None and self.t - 1 - g[0] <= 8 and self.name(g[1]) not in BASIC
                  else "?")
            c = self.context(e.a)
            self.spells.append({"t": self.t, "s": round(obs.game_ms / 1000), "player": p, "slot": self.players[p],
                                "ability": rawcode(e.b), "by": by,
                                "engine_order": self.name(en[1]) if en is not None and self.t - en[0] <= 4 else None,
                                "target": self.target(e.c, e.a, 2, p), **c})
        for r in rows[(rows[:, fx.C_FLAGS] & 1 > 0) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]:
            u, p = int(r[fx.C_ID]), int(r[fx.C_OWNER])
            if p not in self.players:
                continue
            now = tuple(int(v) for v in r[fx.C_ITEM0:fx.C_ITEM0 + 6])
            before = self.items.get(u)
            self.items[u] = now
            if before is None:
                continue
            for it in Counter(before) - Counter(now):
                if it:
                    g = self.given.get(u)
                    self.gone.append({"t": self.t, "s": round(obs.game_ms / 1000), "player": p, "slot": self.players[p],
                                      "item": rawcode(it), "agent_ago": None if g is None else self.t - 1 - g[0],
                                      "agent_order": None if g is None else self.name(g[1]), **self.context(u)})
        fl = rows[:, fx.C_FLAGS]
        for p in self.players:
            w = rows[(rows[:, fx.C_OWNER] == p) & (fl & fx.WORKER > 0) & (fl & fx.DEAD == 0)]
            # (idle: no order and not inside a mine or burrow; LOADED 32 / HIDDEN 16)
            self.worker_steps[p] += len(w)
            self.idle[p] += int(((w[:, fx.C_ORDER] == 0) & (w[:, fx.C_FLAGS] & 48 == 0)).sum())
            q = obs.players.get(p)
            if q is not None and self.t % 120 == 0:  # (each game minute: gathered so far)
                self.gathered.setdefault(p, []).append((self.t // 120, q.gold_gathered, q.lumber_gathered, len(w)))
        return obs

    def dump(self) -> dict:
        return {"steps": self.t, "minutes": round(self.minutes, 2), "players": self.players, "records": self.recs,
                "spells": self.spells, "items_gone": self.gone, "commands": dict(self.commands),
                "worker_steps": dict(self.worker_steps), "idle_worker_steps": dict(self.idle), "gathered": self.gathered}


def setup(cfg: dict, slots, user: bool, a) -> GameSetup:
    return GameSetup(map=cfg["map"], slots=slots, step_seconds=0.5, max_game_seconds=a.minutes * 60, victory="decisive",
                     window=(320, 240), record_ai_orders=True, draw=False, wait_floor_ms=cfg.get("wait_floor_ms", 5),
                     native_obs=bool(cfg.get("native_obs", True)), nice=19, d3d_thread=False, render_threads=0,
                     cuts=exact_cuts(cfg.get("game_cuts")) or None, warm_spare=False, agent_is_user=user, shim_dir=a.shim)


def run_settings(a) -> tuple[dict, dict, Path]:
    """The run's evaluation games' settings (evalworker.run_config: its --eval-map), the features' module set up as its
    evaluation worker has it -> (cfg, vocab, the checkpoint)."""
    import torch

    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame.costs import order_costs
    from warcraftsim.fullgame.evalworker import run_config
    from warcraftsim.fullgame.trace import unit_values
    run = Path(a.run).resolve()
    c = json.loads((run / "run.json").read_text())["config"]
    root = run.parent.parent
    ck = Path(a.checkpoint)
    cfg, _ = run_config(run, "probe", 1, 0)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg.update(z_pool=str(root / (c.get("eval_z_pool") or c["z_pool"])), z_share=0.0, scenarios=[], scenario_share=0.0,
               late_states=[], late_z=0, eval_checkpoints=[], max_minutes=a.minutes,
               max_steps=int(a.minutes * 60 / cfg["step_seconds"]) + 20, device="cpu", values=unit_values(), order_names={})
    fx.ACTIVE_F = cfg["features"]
    os.environ["WARCRAFTSIM_FEATURES"] = str(cfg["features"])
    fx.set_inputs(cfg["inputs"], bool(cfg["privileged"]))
    fx.set_masks(cfg["tech_mask"], cfg["valid_mask"])
    fx.set_neutral(cfg["neutral"])
    fx.set_macros(int(cfg.get("macros", 0) or 0))
    fx.set_spells(int(cfg.get("spells", 0) or 0))
    fx.set_shop_items(int(cfg.get("shop_items", 0) or 0))
    fx.set_fine(int(cfg.get("fine_points", 0) or 0))
    fx.set_macros_off(str(cfg.get("macros_off") or "").split(","))
    fx.set_econ_masks(cfg.get("econ_masks", 0))
    fx.set_value_tax(cfg.get("value_tax", 0))
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


def play(a) -> int:
    import torch

    from warcraftsim.fullgame import selfplay
    from warcraftsim.fullgame.model import load
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="cslotprobe"))
    cfg, vocab, ck = run_settings(a)
    cfg["run_dir"] = str(tmp)
    selfplay.publish(load(ck, "cpu", vocab=vocab if fx.NEUTRAL else None)[0], cfg["checkpoint_version"], tmp)
    n = max(1, a.parallel)
    infer = selfplay.Inference(selfplay.Nets(cfg, torch.device("cpu")), torch.device("cpu"), n, compile_=False,
                               buckets=tuple(range(1, n + 1)), entities=tuple(cfg["infer_entities"]))
    jobs: queue.Queue = queue.Queue()
    for k in range(a.repeat):
        for m in a.matchups.split(","):
            race, foe = m.split(":")
            for slot in a.slot.split(","):
                jobs.put((race, foe, k, slot))

    class Rows:
        def put(self, msg):
            pass

    def worker(w: int) -> None:
        while True:
            try:
                race, foe, k, slot = jobs.get_nowait()
            except queue.Empty:
                return
            tag = f"play-{slot}-{race}-{foe}-{k}-{int(time.time())}"
            name = f"cslotprobe{a.name}{race[:2]}{foe[:2]}{slot[0]}{k}"  # (a name of its own per game, as in units)
            t0 = time.time()
            slots = [Agent(race, handicap=100, difficulty="normal"), BuiltinAI(foe, "insane", handicap=100)]
            try:
                with GameInstance(setup(cfg, slots, slot == "user", a), name=name, timeout=600) as g:
                    obs = g.start()
                    log = Log(g, obs, {0: slot})
                    ep = selfplay.play_one(log, obs, cfg, vocab, [race, foe], 0,
                                           {"name": "script:ai-insane (real)", "kind": "ai"}, infer, Rows(), w, train=False)
            except Exception as e:  # (a game that failed: said, the others go on)
                print(f"FAILED {tag}: {e!r}", flush=True)
                continue
            res = {"mode": "play", "slot": slot, "races": [race, foe], "checkpoint": str(ck), "outcome": ep.get("outcome"),
                   "wall": round(time.time() - t0), **log.dump()}
            (out / f"{tag}.json").write_text(json.dumps(res))
            print(f"RESULT {tag}: {ep.get('outcome')} at {log.minutes:.1f} min, {len(log.recs)} engine orders, "
                  f"{len(log.spells)} spells, {len(log.gone)} items gone, wall {res['wall']} s", flush=True)

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(n)]
    try:
        for th in threads:
            th.start()
            time.sleep(5)
        for th in threads:
            th.join()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


# units: the same group for both sides (spawned as trained units are: Spawn, no harness tweak), the heroes placed
# (Place: level, skills, items; the harness removes a placed agent unit's AI guard post)
GROUP = ["hfoo"] * 4 + ["ogru"] * 4 + ["hmpr", "hmpr", "hsor", "hsor", "unec", "unec", "edot", "edot", "edry", "uban", "oshm",
                                       "hspt"]
ADEPT = ["Rhpt", "Rhst", "Rune", "Redt", "Ruba", "Rost"]
HEROES = [("Hpal", ("AHhb", "AHds", "AHad")), ("Ulic", ("AUfn", "AUfu", "AUdr")), ("Emoo", ("AHfa", "AEst", "AEar")),
          ("Ofar", ("AOcl", "AOfs", "AOsf"))]
ITEMS = ("phea", "pman", "hslv", "plcl", "sreg", "stwp")
RAID = ["ogru"] * 5 + ["ehun"] * 3
WORKERS = {"human": "hpea", "orc": "opeo", "undead": "uaco", "nightelf": "ewsp"}


def units(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from warcraftsim.data.mapbuild import stock_map_path
    cfg = {"map": a.map, "native_obs": True, "game_cuts": a.cuts}
    fx.set_geometry(*fx.map_geometry(stock_map_path(a.map)))
    for race in a.races.split(","):
        foe = a.foe
        tag = f"units{'-placed' if a.place else ''}-{a.slot}-{race}-{int(time.time())}"
        slots = [Agent(race, handicap=100, difficulty="normal"), Agent(foe, handicap=100, difficulty="normal")]
        t0 = time.time()
        with GameInstance(setup(cfg, slots, a.slot == "user", a), name=f"cslotprobe{a.name}{race[:2]}", timeout=600) as g:
            # (a name of its own per game: the next game under the same name lost its connection at its first step)
            obs = g.start()
            log = Log(g, obs, {0: a.slot, 1: "computer"})
            s0 = np.array([obs.players[0].start_x, obs.players[0].start_y], float)
            s1 = np.array([obs.players[1].start_x, obs.players[1].start_y], float)
            way = (s1 - s0) / max(1.0, float(np.linalg.norm(s1 - s0)))
            spot = s0 + way * a.spot
            side = np.array([-way[1], way[0]])
            rows = log.rows
            mine = next(r for r in sorted(rows.tolist(), key=lambda r: math.hypot(r[fx.C_X] - s0[0], r[fx.C_Y] - s0[1]))
                        if r[fx.C_TYPE] in fx.GOLD_MINES)
            hall = next(r for r in rows.tolist() if r[fx.C_OWNER] == 0 and r[fx.C_FLAGS] & fx.STRUCTURE
                        and math.hypot(r[fx.C_X] - s0[0], r[fx.C_Y] - s0[1]) < 600)
            workers = [r for r in rows.tolist() if r[fx.C_OWNER] == 0 and r[fx.C_FLAGS] & fx.WORKER]
            cmds = [ClearCreeps(float(spot[0]), float(spot[1]), 1600)]
            if race == "orc":  # (a burrow by the hall, complete)
                bx, by = np.array([hall[fx.C_X], hall[fx.C_Y]], float) - way * 450 + side * 250
                cmds.append(Place(0, "otrb", float(bx), float(by)))
            for p in (0, 1):
                for t in ADEPT:
                    cmds.append(SetResearch(p, t, 1))
            log.step(cmds)
            cmds = []
            for p, at in ((0, spot), (1, spot + way * a.gap)):
                for k, t in enumerate(GROUP):
                    xy = at + side * ((k % 8) - 3.5) * 90 + way * (k // 8) * 90 * (1 if p else -1)
                    # (--place 1: placed instead, as late-game starts place units: the harness removes an agent's placed
                    # unit's AI guard post, RemoveGuardPosition)
                    cmds.append(Place(p, t, float(xy[0]), float(xy[1])) if a.place else Spawn(p, t, float(xy[0]), float(xy[1])))
                for k, (h, skills) in enumerate(HEROES):
                    xy = at + side * (k - 1.5) * 150 - way * 300 * (1 if p == 0 else -1)
                    cmds.append(Place(p, h, float(xy[0]), float(xy[1]), hero_level=3, skills=skills, items=ITEMS))
            log.step(cmds)
            group1 = [r for r in log.rows.tolist() if r[fx.C_OWNER] == 1 and not r[fx.C_FLAGS] & (fx.STRUCTURE | fx.WORKER)
                      and math.hypot(r[fx.C_X] - spot[0] - way[0] * a.gap, r[fx.C_Y] - spot[1] - way[1] * a.gap) < 900]
            log.step([PointOrder(int(r[fx.C_ID]), ATTACK, float(spot[0]), float(spot[1])) for r in group1])
            raid_at = int(a.raid / 0.5)
            snaps: dict = {}
            spawned: list = []

            def orders_of(ids=None, p=0) -> dict:
                rs = [r for r in log.rows.tolist() if r[fx.C_OWNER] == p and r[fx.C_FLAGS] & fx.WORKER
                      and not r[fx.C_FLAGS] & fx.DEAD and (ids is None or r[fx.C_ID] in ids)]
                return dict(Counter("idle" if r[fx.C_ORDER] == 0 else log.name(int(r[fx.C_ORDER])) for r in rs))
            while log.t < int(a.minutes * 120) and not getattr(log.g, "_ended", False):
                cmds = []
                if log.t == 40:  # (20 s: our workers given no order so far; then the idle ones sent to the mine)
                    snaps["20 s, start workers (no order given)"] = {p: orders_of(p=p) for p in (0, 1)}
                    cmds = [TargetOrder(int(r[fx.C_ID]), HARVEST, int(mine[fx.C_ID])) for r in log.rows.tolist()
                            if r[fx.C_OWNER] == 0 and r[fx.C_FLAGS] & fx.WORKER and r[fx.C_ORDER] == 0]
                elif log.t == 60:  # (30 s: two workers spawned by the hall, as if trained; never given an order)
                    wt = WORKERS[race]
                    cmds = [Spawn(0, wt, float(hall[fx.C_X] - way[0] * 300 + side[0] * d), float(hall[fx.C_Y] - way[1] * 300 + side[1] * d))
                            for d in (-120, 120)]
                elif log.t == 61:
                    spawned = [r[fx.C_ID] for r in log.rows.tolist() if r[fx.C_OWNER] == 0 and r[fx.C_TYPE] == code(WORKERS[race])
                               and log.born.get(r[fx.C_ID], 0) >= 60]
                    snaps["30 s, workers spawned"] = {0: {"found": len(spawned)}}
                elif log.t == 100:
                    snaps["50 s, workers spawned at 30 s (no order given)"] = {0: orders_of(set(spawned))}
                elif log.t == int(a.away / 0.5):  # (the group's survivors moved 900 back toward the base: sent back by itself?)
                    away = spot - way * 900
                    moved = [r for r in log.rows.tolist() if r[fx.C_OWNER] == 0 and not r[fx.C_FLAGS] & (fx.STRUCTURE | fx.WORKER | fx.DEAD)
                             and math.hypot(r[fx.C_X] - spot[0], r[fx.C_Y] - spot[1]) < 1500]
                    cmds = [PointOrder(int(r[fx.C_ID]), MOVE, float(away[0]), float(away[1])) for r in moved]
                    snaps[f"{a.away:.0f} s, group moved away"] = {0: {"units": len(moved), "to": [round(float(v)) for v in away]}}
                elif log.t == int(a.away / 0.5) + 60:
                    near = sum(1 for r in log.rows.tolist() if r[fx.C_OWNER] == 0 and not r[fx.C_FLAGS] & (fx.STRUCTURE | fx.WORKER | fx.DEAD)
                               and math.hypot(r[fx.C_X] - away[0], r[fx.C_Y] - away[1]) < 500)
                    back = sum(1 for r in log.rows.tolist() if r[fx.C_OWNER] == 0 and not r[fx.C_FLAGS] & (fx.STRUCTURE | fx.WORKER | fx.DEAD)
                               and math.hypot(r[fx.C_X] - spot[0], r[fx.C_Y] - spot[1]) < 600)
                    snaps[f"{a.away + 30:.0f} s, the moved group"] = {0: {"at the point": near, "back at the fight's spot": back}}
                if log.t == raid_at:
                    mx, my = mine[fx.C_X], mine[fx.C_Y]
                    cmds = [Spawn(1, t, float(mx + 200 * math.cos(k)), float(my + 200 * math.sin(k))) for k, t in enumerate(RAID)]
                elif log.t == raid_at + 1:
                    raid = [r for r in log.rows.tolist() if r[fx.C_OWNER] == 1 and r[fx.C_TYPE] in (code("ogru"), code("ehun"))
                            and math.hypot(r[fx.C_X] - mine[fx.C_X], r[fx.C_Y] - mine[fx.C_Y]) < 600]
                    cmds = [PointOrder(int(r[fx.C_ID]), ATTACK, float(hall[fx.C_X]), float(hall[fx.C_Y])) for r in raid]
                obs = log.step(cmds)
                if obs.game_over:
                    break
        res = {"mode": "units-placed" if a.place else "units", "slot": a.slot, "races": [race, foe], "wall": round(time.time() - t0),
               "spot": [round(float(v)) for v in spot], "mine": [int(mine[fx.C_X]), int(mine[fx.C_Y])], "snapshots": snaps,
               **log.dump()}
        (out / f"{tag}.json").write_text(json.dumps(res))
        print(f"RESULT {tag}: {len(log.recs)} engine orders, {len(log.spells)} spells, {len(log.gone)} items gone, "
              f"wall {res['wall']} s", flush=True)
    return 0


def kind_of(r: dict) -> str:
    return ("hero" if r["hero"] else "worker" if r["worker"] else "building" if r["structure"] else
            "summoned" if r["summoned"] else "unit")


def report(a) -> int:
    games = []
    for d in a.dirs:
        for p in sorted(Path(d).glob("*.json")):
            games.append(json.loads(p.read_text()))
    by = defaultdict(list)
    for gm in games:
        by[(gm["mode"], gm["slot"])].append(gm)
    for (mode, slot), gs in sorted(by.items()):
        mins = sum(gm["minutes"] for gm in gs)
        print(f"\n== {mode}, our side's slot {slot}: {len(gs)} games, {mins:.0f} game minutes; outcomes "
              f"{Counter(str(gm.get('outcome')) for gm in gs)}")
        for p in sorted({int(k) for gm in gs for k in gm["players"]}):
            sl = gs[0]["players"][str(p)]
            recs = [r for gm in gs for r in gm["records"] if r["player"] == p]
            cmds = sum(gm["commands"].get(str(p), 0) for gm in gs)
            same = sum(r["agent_same"] for r in recs)
            ws = sum((gm.get("worker_steps") or {}).get(str(p), 0) for gm in gs)
            idle = sum((gm.get("idle_worker_steps") or {}).get(str(p), 0) for gm in gs)
            g10 = [row[1] for gm in gs for row in (gm.get("gathered") or {}).get(str(p), []) if row[0] == 10]
            print(f"-- player {p} ({sl} slot): {len(recs)} engine orders, our side's commands {cmds}, "
                  f"engine orders equal to our command of the step: {same}; workers idle {idle / max(ws, 1):.1%} of "
                  f"worker-steps; gold gathered by 10 min {np.mean(g10) if g10 else float('nan'):.0f} ({len(g10)} games)")
            home = [r for r in recs if not r["worker"] and not r["structure"] and r["kind"] == 1 and r.get("born_xy")
                    and r["name"] in ("attack", "move") and (r["dist"] or 0) > 600
                    and math.hypot(r["px"] - r["born_xy"][0], r["py"] - r["born_xy"][1]) < 400]
            never = {(r["type"], tuple(r["born_xy"])) for r in recs if r["worker"] and r["name"] in ("harvest", "smart")
                     and r["agent_ago"] is None and r.get("born_xy")}
            print(f"   sent back to where they were made (a point within 400 of it, from 600 or more away): {len(home)} orders, "
                  f"{len({(r['type'], tuple(r['born_xy'])) for r in home})} units, median distance "
                  f"{float(np.median([r['dist'] for r in home])) if home else 0:.0f}; workers sent to work that our side never "
                  f"ordered: {len(never)}")
            tab = defaultdict(list)
            for r in recs:
                tab[(r["name"], kind_of(r))].append(r)
            print(f"   {'order':<22}{'unit kind':<10}{'n':>6}{'/min':>7}{'games':>6}  enemies near / hurt / hp<.5 / age<=2 / "
                  f"agent cmd <=2 steps / types")
            for (name, k), rs in sorted(tab.items(), key=lambda kv: -len(kv[1])):
                n = len(rs)
                ng = sum(1 for gm in gs if any(r["player"] == p and r["name"] == name and kind_of(r) == k
                                               for r in gm["records"]))
                print(f"   {name:<22}{k:<10}{n:>6}{n / max(mins, 1e-9):>7.2f}{ng:>6}  "
                      f"{sum(r['enemies'] > 0 for r in rs) / n:.2f} / {sum(r['hurt'] for r in rs) / n:.2f} / "
                      f"{sum(r['hp'] < 0.5 for r in rs) / n:.2f} / {sum(r['age'] <= 2 for r in rs) / n:.2f} / "
                      f"{sum(r['agent_ago'] is not None and r['agent_ago'] <= 2 for r in rs) / n:.2f} / "
                      f"{dict(Counter(r['type'] for r in rs).most_common(4))} {dict(Counter(r['target'].split(' ')[0] for r in rs).most_common(3))}")
            for gm in gs:
                for k, v in (gm.get("snapshots") or {}).items():
                    if str(p) in v:
                        print(f"   {gm['races'][0]}: {k}: {v[str(p)]}")
            sp = [s for gm in gs for s in gm["spells"] if s["player"] == p]
            if sp:
                print("   spells (ability, caster, by): " + ", ".join(
                    f"{ab} {ty} {b} x{n}" for (ab, ty, b), n in Counter((s["ability"], s["type"], s["by"]) for s in sp).most_common(30)))
            it = [s for gm in gs for s in gm["items_gone"] if s["player"] == p]
            if it:
                print("   items gone (item, by agent cmd <=2 steps, hp<.5, enemies near): " + ", ".join(
                    f"{i} {c} {h} {e} x{n}" for (i, c, h, e), n in Counter(
                        (s["item"], s["agent_ago"] is not None and s["agent_ago"] <= 2, s["hp"] < 0.5, s["enemies"] > 0)
                        for s in it).most_common(30)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("play", "units", "report"))
    ap.add_argument("dirs", nargs="*", help="report: directories of logs")
    ap.add_argument("--slot", default="computer", help="computer or user (play: a comma separated list, each matchup in each)")
    ap.add_argument("--run", default="/home/tora/warcraftsim/runs/fgself-179")
    ap.add_argument("--checkpoint", default="/home/tora/warcraftsim/runs/fgself-179/checkpoints/0000000050301674.pt")
    ap.add_argument("--matchups", default="human:orc,orc:undead,undead:nightelf,nightelf:human",
                    help="play: our race:the AI's race, comma separated")
    ap.add_argument("--repeat", type=int, default=1, help="play: games per matchup")
    ap.add_argument("--parallel", type=int, default=2, help="play: games at a time (one process, threads)")
    ap.add_argument("--minutes", type=float, default=25.0)
    ap.add_argument("--races", default="human,orc,undead,nightelf", help="units: player 0's races, a game each")
    ap.add_argument("--foe", default="orc", help="units: player 1's race")
    ap.add_argument("--map", default="(2)EchoIsles", help="units: the map")
    ap.add_argument("--cuts", type=int, default=17592186041849, help="units: the game cuts (exact ones only)")
    ap.add_argument("--spot", type=float, default=2000.0, help="units: the groups' fight this far from player 0's start")
    ap.add_argument("--gap", type=float, default=900.0, help="units: player 1's group this far beyond")
    ap.add_argument("--raid", type=float, default=60.0, help="units: the raid at player 0's mine at this second")
    ap.add_argument("--place", type=int, default=0, help="units: the groups placed (no AI guard post) instead of spawned")
    ap.add_argument("--away", type=float, default=100.0, help="units: player 0's group moved 900 toward its base at this second")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=".")
    ap.add_argument("--name", default="")
    ap.add_argument("--shim", default=str(Path(__file__).resolve().parents[1] / "build" / "shim"))
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    if any(s not in ("computer", "user") for s in a.slot.split(",")) or (a.what == "units" and "," in a.slot):
        ap.error("--slot: computer or user (play: or both, comma separated)")
    if a.what == "report":
        return report(a)
    os.nice(19)
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    patch_harness()
    return play(a) if a.what == "play" else units(a)


if __name__ == "__main__":
    raise SystemExit(main())
