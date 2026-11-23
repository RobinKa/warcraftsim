"""A policy's order habits in one live game, with what the replays do not hold: each unit's current order and place,
what the game answered, and the spells that took effect (2026-11-23, bug finder 11; scripts/order_scan.py finds the
habits in the replays, scripts/order_probe.py asks the engine what they cost).

    python scripts/habit_probe.py --weights runs/fgself-236/averaged/swa-43-53.pt --run runs/fgself-249 \\
        --race undead --opp orc [--target 0] [--steps 1600] [--seed 1] [--set key=value ...] [--json out.json]

Plays ONE game as selfplay.play_one plays it (scripts/ghoul_probe.py's setup: the run's settings from its run.json,
read only; the policy on the CPU; the built-in AI as the opponent) and counts, for the policy's side:
  spells      each spell's orders sent and taken, and the casts that took effect (the game's spell events by caster);
              of the orders taken: how many had another order sent to the caster one or two steps later, and how many
              of those took effect (an order within the cast's turn and wind-up cancels it: order_probe's frost nova)
  attacks     attack-moves sent to a unit already attack-moving (its current order "attack"), by how far the new
              point is from the last one and how long ago that was; those with an enemy in reach (a swing at stake)
  edge        orders aimed above the map (the old point bins' top edge): the unit's kind, whether an enemy was near,
              how far the unit was from where it stood 20 steps later
  harvest     harvesters given another order while they read a harvest order, by type and the order given
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ATTACK, MOVE, PATROL, SMART, STOP = 851983, 851986, 851990, 851971, 851972
GENERIC = {ATTACK, MOVE, PATROL, SMART, STOP, 851993, 852018, 852017, 852020, 852024, 851976, 851980, 852161, 852202,
           852460, 851973}
MAP_TOP = 5376.0  # ((2)EchoIsles: its playable area ends here; the camera's bound is 4864)


class Habits:
    """The game's stand-ins (the inference, the learner's queue) and the side's record a step."""

    def __init__(self, net, vocab: dict, side: int, seed: int):
        import torch
        from warcraftsim.fullgame import features as fx
        self.net, self.vocab, self.side = net, vocab, side
        self.gen = torch.Generator().manual_seed(seed)
        self.names = {int(k): v for k, v in (vocab.get("order_names") or {}).items()}
        self.fx = fx
        self.t = 0
        self.units: dict[int, list] = defaultdict(list)  # unit -> [(t, x, y, cur, hp)]
        self.type: dict[int, str] = {}
        self.flags: dict[int, int] = {}
        self.enemy: dict[int, list] = {}  # step -> [(x, y)] of the enemy's and creeps' living units in sight
        self.sent: dict[int, list] = {}  # step -> [(unit, kind, order, x, y, target)]
        self.taken: dict[int, list] = {}
        self.effects: list = []  # (t, caster, ability)
        self.last_point: dict[int, tuple] = {}  # unit -> (order, x, y, t) of its last accepted point order
        self.repeat = Counter()
        self.harvest = Counter()
        self.steps = 0

    def put(self, msg: dict) -> None:
        pass

    def request(self, items: list) -> list:
        return [self.forward(st) for _, st in items]

    def forward(self, st: dict) -> dict:
        import torch
        from warcraftsim.fullgame.model import act
        fx = self.fx
        n = st["n"]
        ent = torch.from_numpy(np.asarray(st["ent"], np.float32)[None, :n])
        typ = torch.from_numpy(np.asarray(st["type"], np.int64)[None, :n])
        cur = torch.from_numpy(np.asarray(st["cur"], np.int64)[None, :n])
        mask = torch.ones(1, n, dtype=torch.bool)
        glob = torch.from_numpy(np.asarray(st["glob"], np.float32)[None])
        n_own = torch.tensor([min(st["n_own"], fx.MAX_OWN)])
        avail = torch.from_numpy(np.asarray(st["avail"], bool)[None]) if st.get("avail") is not None else None
        h = torch.from_numpy(np.asarray(st["h"], np.float32)[None]) if st.get("h") is not None else None
        shared = torch.tensor([bool(st.get("shared_noise"))])
        with torch.no_grad():
            out = act(self.net, ent, typ, cur, mask, glob, n_own, avail, h, generator=self.gen, shared=shared)
        o = int(n_own[0])
        res = {"order": out["order"][0, :o].numpy().astype(np.int64), "tgt": out["tgt"][0, :o].numpy().astype(np.int64),
               "bx": out["bx"][0, :o].numpy().astype(np.int64), "by": out["by"][0, :o].numpy().astype(np.int64),
               "logp": out["logp"][0, :o].numpy().copy(), "value": float(out["value"][0]),
               "entropy": float(out["entropy"][0]), "version": 0}
        if self.net.memory:
            res["h"] = out["h"][0].numpy().copy()
        return res

    # the wrappers' notes
    def on_observe(self, bot, obs, t: int) -> None:
        fx = self.fx
        self.t = t
        self.steps = max(self.steps, t + 1)
        rows = bot.rows
        fl = rows[:, fx.C_FLAGS]
        alive = fl & fx.DEAD == 0
        own = rows[alive & (rows[:, fx.C_OWNER] == self.side)]
        for i, ty, x, y, cur, hp, f in own[:, [fx.C_ID, fx.C_TYPE, fx.C_X, fx.C_Y, fx.C_ORDER, fx.C_HP, fx.C_FLAGS]].tolist():
            self.units[i].append((t, x, y, cur, hp))
            self.type[i] = fx.rawcode(ty)
            self.flags[i] = f
        foe = rows[alive & (rows[:, fx.C_OWNER] != self.side) & ((rows[:, fx.C_VIS] >> self.side) & 1 == 1)
                   & (fl & fx.STRUCTURE == 0)]
        foe = foe[(foe[:, fx.C_OWNER] == 1 - self.side) | (foe[:, fx.C_OWNER] == 12)]
        self.enemy[t] = foe[:, [fx.C_X, fx.C_Y]].tolist()
        for e in obs.events:
            if int(e.kind) == 15:
                self.effects.append((t, int(e.a), fx.rawcode(int(e.b))))

    def on_commands(self, bot, out: list) -> None:
        rec = []
        for c in out:
            unit = getattr(c, "unit", getattr(c, "hero", None))
            order = getattr(c, "order", None) or getattr(c, "building", None) or getattr(c, "ability", None) or 0
            rec.append((unit, type(c).__name__, int(order) if isinstance(order, int) else 0, getattr(c, "x", None),
                        getattr(c, "y", None), getattr(c, "target", None)))
        self.sent[self.t] = rec

    def on_accepted(self, cmds: list, results: list) -> None:
        self.taken[self.t - 1] = [bool(r) for r in results]


def install(p: Habits) -> None:
    from warcraftsim.fullgame.play import BCAgent
    b_observe, b_commands, b_accepted = BCAgent.observe, BCAgent.commands, BCAgent.accepted

    def observe(self, obs, t, rows=None):
        st = b_observe(self, obs, t, rows)
        if self.player == p.side:
            p.on_observe(self, obs, t)
        return st

    def commands(self, st, order, tgt, bx, by):
        out = b_commands(self, st, order, tgt, bx, by)
        if self.player == p.side:
            p.on_commands(self, out)
        return out

    def accepted(self, cmds, results):
        if self.player == p.side:
            p.on_accepted(cmds, results)
        return b_accepted(self, cmds, results)
    BCAgent.observe, BCAgent.commands, BCAgent.accepted = observe, commands, accepted


def kind_of(p: Habits, unit: int) -> str:
    f = p.flags.get(unit, 0)
    return "hero" if f & 1 else "building" if f & 2 else "worker" if f & 4 else "army"


def report(p: Habits) -> dict:
    from warcraftsim.data.abilities import ability_info
    names = p.names
    order_of = {code: i.order for code, i in ability_info().items() if i.order}
    pos = {u: {t: (x, y, cur, hp) for t, x, y, cur, hp in rec} for u, rec in p.units.items()}
    by_unit: dict[int, list] = defaultdict(list)  # unit -> [(t, kind, order, x, y, target, taken)]
    for t, rec in sorted(p.sent.items()):
        res = p.taken.get(t, [])
        for k, (unit, kind, order, x, y, target) in enumerate(rec):
            if unit is not None:
                by_unit[unit].append((t, kind, order, x, y, target, res[k] if k < len(res) else None))
    out: dict = {"steps": p.steps}
    # spells
    eff = defaultdict(list)
    for t, caster, abil in p.effects:
        eff[caster].append((t, order_of.get(abil, abil)))
    spells: dict = defaultdict(Counter)
    for unit, ev in by_unit.items():
        for k, (t, kind, order, x, y, target, ok) in enumerate(ev):
            if kind not in ("PointOrder", "TargetOrder", "ImmediateOrder") or order in GENERIC or order >= 0x1000000:
                continue
            name = names.get(order, str(order))
            c = spells[name]
            c["sent"] += 1
            if not ok:
                continue
            c["taken"] += 1
            cast = any(t <= te <= t + 8 and n == name for te, n in eff.get(unit, ()))
            c["took effect"] += cast
            nxt = next((e for e in ev[k + 1:] if e[6]), None)  # (the caster's next order the game took)
            gap = nxt[0] - t if nxt is not None else 99
            if gap <= 2:
                c["another order within 2 steps"] += 1
                c["another order within 2 steps, took effect"] += cast
            else:
                c["left alone 2 steps"] += 1
                c["left alone 2 steps, took effect"] += cast
    out["spells"] = {k: dict(v) for k, v in spells.items()}
    print("\n== spells: orders sent, taken by the game, casts that took effect; of the taken ones, with another order "
          "to the caster within 2 steps: took effect; left alone: took effect")
    for name, c in sorted(spells.items(), key=lambda kv: -kv[1]["sent"]):
        print(f"  {name:<18} sent {c['sent']:4d} taken {c['taken']:4d} took effect {c['took effect']:4d} | "
              f"re-ordered {c['another order within 2 steps']:4d}: {c['another order within 2 steps, took effect']:4d} | "
              f"left alone {c['left alone 2 steps']:4d}: {c['left alone 2 steps, took effect']:4d}")
    # attack-moves to a unit attack-moving already
    rep = Counter()
    last: dict[int, tuple] = {}
    for unit, ev in by_unit.items():
        for t, kind, order, x, y, target, ok in ev:
            if kind == "PointOrder" and order == ATTACK and ok:
                at = pos.get(unit, {}).get(t)
                prev = last.get(unit)
                rep["attack-moves"] += 1
                if at is not None and at[2] == ATTACK:
                    rep["to a unit that reads attack"] += 1
                    if prev is not None:
                        d, age = math.hypot(x - prev[0], y - prev[1]), t - prev[2]
                        if d <= 256:
                            rep["... its last attack-move within 256"] += 1
                            rep[f"... within 256, {'the step before' if age <= 1 else 'within 10 steps' if age <= 10 else 'longer ago'}"] += 1
                            if any((ex - at[0]) ** 2 + (ey - at[1]) ** 2 < 700 ** 2 for ex, ey in p.enemy.get(t, ())):
                                rep["... within 256, an enemy within 700 of the unit"] += 1
                            rep[f"... within 256: {kind_of(p, unit)}"] += 1
                last[unit] = (x, y, t)
            elif ok and kind in ("PointOrder", "TargetOrder", "ImmediateOrder", "Build", "TargetDestructable"):
                last.pop(unit, None)
    out["attacks"] = dict(rep)
    print("\n== attack-moves the game took:")
    for k, v in rep.items():
        print(f"  {k:<52} {v:5d}")
    # orders above the map
    edge = Counter()
    walked = []
    for unit, ev in by_unit.items():
        for k, (t, kind, order, x, y, target, ok) in enumerate(ev):
            if kind == "PointOrder" and y is not None and y > MAP_TOP and ok:
                at = pos.get(unit, {}).get(t)
                edge[f"{names.get(order, order)} by a {kind_of(p, unit)}"] += 1
                if at is None:
                    continue
                near = any((ex - at[0]) ** 2 + (ey - at[1]) ** 2 < 900 ** 2 for ex, ey in p.enemy.get(t, ()))
                edge["an enemy within 900 of the unit"] += near
                later = pos.get(unit, {}).get(t + 20)
                nxt = ev[k + 1][0] - t if k + 1 < len(ev) else 999
                if later is not None:
                    walked.append((math.hypot(later[0] - at[0], later[1] - at[1]), later[1] - at[1], nxt))
    out["edge"] = dict(edge)
    print("\n== orders aimed above the map (y > %d) the game took:" % MAP_TOP)
    for k, v in edge.most_common():
        print(f"  {k:<52} {v:5d}")
    if walked:
        w = sorted(d for d, _, _ in walked)
        north = sorted(dy for _, dy, _ in walked)
        print(f"  20 steps later the unit stood {w[len(w) // 2]:.0f} from where it was (median; p90 {w[9 * len(w) // 10]:.0f}), "
              f"{north[len(north) // 2]:.0f} to the north (median); its next order after "
              f"{sorted(n for _, _, n in walked)[len(walked) // 2]} steps (median)")
    # harvesters given another order while harvesting
    hv = Counter()
    for unit, ev in by_unit.items():
        if p.type.get(unit) not in ("hpea", "opeo", "uaco", "ugho", "ewsp"):
            continue
        for t, kind, order, x, y, target, ok in ev:
            at = pos.get(unit, {}).get(t)
            if at is not None and at[2] in (852018, 852017, 852020) and ok and kind != "Build" and order < 0x1000000:
                hv[f"{p.type[unit]}: {names.get(order, order)} ({kind})"] += 1
    out["harvest"] = dict(hv)
    print("\n== harvesters (reading harvest, resumeharvesting or returnresources) given another order the game took:")
    for k, v in hv.most_common(12):
        print(f"  {k:<52} {v:5d}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="the run whose settings the game is played with")
    ap.add_argument("--target", type=int, default=0, help="1: a game with a target strategy; 0: without (as evaluated)")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--race", default="undead")
    ap.add_argument("--opp", default="orc")
    ap.add_argument("--ai", default="insane")
    ap.add_argument("--side", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], help="settings changed for this game: key=value")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--shim", type=Path, default=None)
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    import torch
    torch.set_num_threads(2)
    import ghoul_probe as gp
    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame import selfplay as sp
    from warcraftsim.fullgame import strategy as strat
    from warcraftsim.fullgame.model import load
    from warcraftsim.runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup
    vocab = json.loads((a.run / "vocab.json").read_text())
    over = dict(kv.split("=", 1) for kv in a.set)
    over.setdefault("z_share", 1.0 if a.target else 0.0)
    over["max_steps"] = a.steps
    cfg = gp.run_cfg(a.run, vocab, over)
    fx.ACTIVE_F = cfg["features"]
    fx.set_inputs(cfg["inputs"], bool(cfg["privileged"]))
    fx.set_masks(cfg["tech_mask"], cfg["valid_mask"])
    fx.set_neutral(cfg["neutral"])
    fx.set_macros(int(cfg["macros"] or 0))
    fx.set_spells(int(cfg["spells"] or 0))
    fx.set_shop_items(int(cfg["shop_items"] or 0))
    fx.set_full_vocab(int(cfg["full_vocab"] or 0))
    fx.set_obs_full(int(cfg["obs_full"] or 0))
    fx.set_fine(int(cfg["fine_points"] or 0))
    fx.set_macros_off(str(cfg["macros_off"] or "").split(","))
    fx.set_econ_masks(cfg["econ_masks"])
    fx.set_order_masks(cfg["order_masks"])
    fx.set_value_tax(cfg["value_tax"])
    if (width := strat.block_width(cfg["z_modes"], cfg["z_mix"], cfg["z_comp"], cfg["z_shop"])):
        fx.set_z(width)
    fx.set_geometry(*fx.map_geometry(stock_map_path(cfg["map"])))
    if cfg["order_masks"] & fx.EDGE_ORDERS:  # (--set order_masks=31: bit 16's bounds, as selfplay.main)
        fx.set_bounds(fx.map_bounds(stock_map_path(cfg["map"])))
    net, _ = load(a.weights, "cpu", vocab=vocab)
    fx.set_z(net.config.get("z", 0))
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    side = a.side
    races = [a.opp, a.opp]
    races[side] = a.race
    p = Habits(net, vocab, side, a.seed)
    install(p)
    slots = [None, None]
    slots[side] = Agent(races[side], handicap=cfg["handicap"], difficulty="normal")
    slots[1 - side] = BuiltinAI(races[1 - side], a.ai, handicap=cfg["handicap"])
    setup = GameSetup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"], max_game_seconds=cfg["max_minutes"] * 60,
                      victory="decisive", window=(320, 240), draw=False, wait_floor_ms=cfg["wait_floor_ms"],
                      native_obs=True, nice=19, d3d_thread=False, render_threads=0, warm_spare=False,
                      cuts=cfg.get("game_cuts") or None, obs_full=bool(cfg.get("obs_full")),
                      **({"shim_dir": a.shim} if a.shim else {}))
    opp = {"name": f"script:ai-{a.ai} (real)", "kind": "ai", "advise": False}
    with GameInstance(setup, name="habitprobe", timeout=600) as g:
        obs = g.start()
        ep = sp.play_one(g, obs, cfg, vocab, races, side, opp, p, p, 0, record=False, train=True)
    print(f"GAME race={a.race} vs {a.ai} {a.opp} target={a.target} steps={p.steps} outcome={ep.get('outcome')} "
          f"orders={ep.get('orders')} refused_share={ep.get('refused_share')} z={ep.get('z')} "
          f"order_masks={cfg['order_masks']} repeat_orders={ep.get('repeat_orders', 0)} edge_orders={ep.get('edge_orders', 0)} "
          f"creeps={(ep.get('prod') or {}).get('creeps')} kills={(ep.get('prod') or {}).get('kills')}", flush=True)
    out = report(p)
    if a.json:
        out["episode"] = {k: v for k, v in ep.items() if k != "trace"}
        a.json.write_text(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
