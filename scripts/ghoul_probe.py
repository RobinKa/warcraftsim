"""Why a policy's starting ghoul flips between "move (point)" and "smart (tree)" (2026-11-23, the user saw it in the
replay videos: undead target games against the built-in AI, 40-67 moves in the first 240 steps since fgself-236).

1. The policy's game. Plays ONE training game as selfplay.play_one plays it (a run's settings from its run.json, read
only; its teachers, the target builder, the masks), the policy on the CPU, and logs the traced unit's every step (the
race's lumber harvester, its first by id: undead's starting ghoul): its current order, what the view says it harvests,
how many order classes the masks leave it, the policy's order probabilities (the top ones and "none"), the order
sampled, what the agent sent for it (or dropped), whether the game took it, and the teacher's lesson for it that step.
Then every harvester of the side by state (type, current order, the view's "sent to the trees" flag): the policy's mean
probabilities of none, a move and a tree order there.

    python scripts/ghoul_probe.py --weights runs/fgself-236/averaged/swa-43-53.pt --run runs/fgself-245 \\
        --target 1 [--steps 130] [--race undead] [--opp orc] [--seed 1] [--set key=value ...] [--json out.json]

--target 0: a game without a target strategy (as an evaluation game: no builder, so no lumber lesson). --set: a
setting of the run's changed for this game (econ_masks=511: with --econ-masks bit 256, features.TREE_HARVEST).
--counter: per step also the traced unit's probabilities and the value with one input changed (the strategy block
zeroed, or one of its parts; its "sent to the trees" flag flipped; the bank's gold or lumber set; no memory): which
input the choice hangs on.

2. The game's own answer (--engine RACE RACE, no policy): each race's harvester right-clicked onto a tree (smart) and
another sent with the harvest order, left alone: the current order each reads, step by step, and the lumber.

What it found (fgself-236's averaged weights swa-43-53, fgself-245's settings, undead against insane orc on Echo Isles):
  * the game: a right-clicked harvester reads "smart" for its whole first trip (a ghoul, a peasant and a peon 25-29
    steps; a wisp for as long as it stays), then "resumeharvesting" and "harvest"; sent with harvest it reads
    "harvest" at once. The lumber is the same.
  * the policy, a target game (3 games of 240 steps): the ghoul free (no order, or moving) gets "smart (tree)" with
    0.92-0.97; sent to the trees and reading "smart" it gets "move (point)" with 0.86-0.93 (to a point near its hall),
    which frees it: 61-76 moves, 66-77 tree orders, 40-60 lumber. It goes on until ~75 s, then the ghoul is left to
    chop. In those states with the strategy block zeroed (or only its items): move 0.44-0.48, "harvest (tree)"
    0.42-0.44; with the "sent to the trees" flag flipped: none 0.48, move 0.29; the modes' part, the bank's lumber, the
    time and the memory change nothing. The lumber lesson gave no label in any of these games (no item was short of
    lumber in their first two minutes).
  * the policy, no target (3 games): the same first step, then "harvest (tree)" (0.15-0.45) beside the move
    (0.1-0.5), and once it reads "harvest": none 0.96. 0 moves in 130 steps; 0 and 8 moves, 160 and 120 lumber in 240.
  * with --econ-masks bit 256 (--set econ_masks=511: the right-click sent as harvest): a target game's ghoul chops
    from step 1 (none 0.96-0.98 for the first ~20 s). 23-27 moves and 100 lumber in 240 steps. What stays is a second
    habit: from ~20 s to ~80 s of a target game the ghoul sent to the trees gets "move (point)" with 0.1-0.8 while it
    reads "harvest" too (the strategy block zeroed in those states: 0.01; only its items: 0.02; the bank's gold set
    to 500: 0.05; the flag flipped: 0.06), and is sent back at once. Why the policy learned either habit this did not
    find.
  * the other races' harvesters (one game each way, 400 steps): a peasant or a peon reading "smart" gets a move with
    0.00-0.02 (a tree order again 0.05-0.2, sent again without the bit: 3-5 a game); a wisp reading "smart" 0.10
    (without a target). In a target game night elf's chopping wisps get a move with 0.03-0.05 a wisp-step while they
    read "harvest" (35 and 61 moves in 400 steps; 0.01 and 5 moves without a target): the second habit, not the
    right-click's.
  * fgself-247's weights at 5.9M steps (2 target games each way): the ghoul reads "smart" in 15-27 steps only (the
    policy gives "harvest (tree)" more), move 0.72-0.77 there; 20 and 33 moves without the bit, 16 and 20 with it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HARVESTERS = {"human": "hpea", "orc": "opeo", "undead": "ugho", "nightelf": "ewsp"}
SMART, HARVEST = 851971, 852018


def fourcc(c: str) -> int:
    return int.from_bytes(c.encode(), "big")


def run_cfg(run_dir: Path, vocab: dict, overrides: dict) -> dict:
    """play_one's settings from a run's run.json (its last launch's arguments, as strings), as selfplay.main makes
    them for a training game."""
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame.costs import order_costs
    from warcraftsim.fullgame.trace import unit_values
    c = json.loads((run_dir / "run.json").read_text())["config"]

    def num(key, default, cast=float):
        v = c.get(key)
        return default if v in (None, "None", "") else cast(v)
    i = int
    cfg = {"run_dir": str(run_dir), "vocab": str(run_dir / "vocab.json"), "map": c.get("eval_map") or c["map"],
           "handicap": num("handicap", 100, i), "step_seconds": num("step_seconds", 0.5), "max_minutes": num("max_minutes", 40.0),
           "wait_floor_ms": num("wait_floor_ms", 5, i), "chunk": num("chunk", 64, i), "gamma": num("gamma", 0.999),
           "lam": num("lam", 0.95), "values": unit_values(), "shaping": num("shaping", 0.0),
           "shaping_scale": num("shaping_scale", 2000.0), "shaping_decay": num("shaping_decay", 0.0),
           "tie_break": num("tie_break", 0.0), "tie_value": num("tie_value", 0.0), "native_obs": True,
           "game_cuts": num("game_cuts", 0, i) or None, "shared_noise": num("shared_noise", 0.0),
           "unit_credit": 0.0, "upgo": num("upgo", 0.0), "hero_level_value": num("hero_level_value", 0.0),
           "features": num("features", fx.F, i), "inputs": num("inputs", 1, i), "tech_mask": num("tech_mask", 0, i),
           "valid_mask": num("valid_mask", 0, i), "neutral": num("neutral", 0, i), "macros": num("macros", 0, i),
           "spells": num("spells", 0, i), "shop_items": num("shop_items", 0, i), "full_vocab": num("full_vocab", 0, i),
           "obs_full": num("obs_full", 0, i), "fine_points": num("fine_points", 0, i),
           "macros_off": "" if c.get("macros_off") in (None, "None") else str(c.get("macros_off")),
           "econ_masks": num("econ_masks", 0, i), "order_masks": num("order_masks", 0, i), "value_tax": num("value_tax", 0, i),
           "expand_lesson": num("expand_lesson", 0.0), "saturation_lesson": num("saturation_lesson", 0, i),
           "z_decided": num("z_decided", 0.0), "burrow_lesson": num("burrow_lesson", 0, i),
           "militia_lesson": num("militia_lesson", 0, i), "army_lesson": num("army_lesson", 0, i),
           "upgrade_lesson": num("upgrade_lesson", 0, i), "creep_lesson": num("creep_lesson", 0, i),
           "creep_lesson_cap": num("creep_lesson_cap", 0, i), "creep_lesson_until": num("creep_lesson_until", 0.0),
           "hunt_lesson": num("hunt_lesson", 0, i), "spell_lesson": num("spell_lesson", 0.0),
           "attack_lesson": num("attack_lesson", 0.0), "first_hero_weights": c.get("first_hero_weights") or "",
           "worker_value": num("worker_value", 1.0), "hero_xp_value": num("hero_xp_value", 0.0),
           "building_hp_value": num("building_hp_value", 1.0), "unit_hp_value": num("unit_hp_value", 1.0),
           "lesson_time_scale": num("lesson_time_scale", 0.0), "revive_lesson": num("revive_lesson", 0, i),
           "teach": num("teach", 0, i), "privileged": num("opsd_share", 0.0) > 0 or num("opsd_coef", 0.0) > 0,
           "opsd_share": 0.0, "opsd_coef": 0.0, "scenarios": [], "scenario_share": 0.0, "late_capture": 0.0,
           "research_value": num("research_value", 0.0), "float_penalty": num("float_penalty", 0.0),
           "avail_mask": True, "opd_share": 0.0, "worker_cap": num("worker_cap", 0, i), "mana_value": num("mana_value", 0.0),
           "z_pool": c.get("eval_z_pool") or c.get("z_pool") or "", "z_share": num("z_share", 0.5),
           "win_balance": 0.0, "z_reward": num("z_reward", 0.0), "z_mix": num("z_mix", 0, i),
           "z_mix_reward": num("z_mix_reward", 0.0), "reward_parts": 0, "z_modes": num("z_modes", 0.0),
           "z_mode_reward": num("z_mode_reward", 0.0), "z_mode_time": num("z_mode_time", 1.0), "z_comp": num("z_comp", 0.0),
           "z_forbid_mask": num("z_forbid_mask", 0, i), "z_winners": num("z_winners", 1, i), "z_src": c.get("z_src") or "",
           "z_enforce": num("z_enforce", 0, i), "z_time_scale": num("z_time_scale", 1.0), "z_build": num("z_build", 0, i),
           "z_build_labels": num("z_build_labels", 0, i), "z_workers": num("z_workers", 0, i), "z_shop": num("z_shop", 0, i),
           "teach_heuristic": 0, "order_names": vocab.get("order_names") or {}}
    for k, v in overrides.items():
        cfg[k] = type(cfg[k])(v) if k in cfg and isinstance(cfg[k], (int, float)) and not isinstance(cfg[k], bool) else v
    if cfg["z_pool"] and not Path(cfg["z_pool"]).is_absolute():  # (the run's paths: relative to its repository)
        cfg["z_pool"] = str(run_dir.resolve().parent.parent / cfg["z_pool"])
    cfg["costs"] = order_costs(vocab, cfg["map"]).tolist()
    cfg["costs_array"] = np.asarray(cfg["costs"], np.int64)
    return cfg


class Probe:
    """The game's stand-ins (the inference, the learner's queue) and the wrappers' notes, a record a step."""

    def __init__(self, net, vocab: dict, side: int, seed: int, counter: bool, unit: str):
        import torch
        from warcraftsim.fullgame import features as fx
        self.net, self.vocab, self.side, self.counter = net, vocab, side, counter
        self.nets = SimpleNamespace(version=0)
        self.gen = torch.Generator().manual_seed(seed)
        self.names = {int(k): v for k, v in (vocab.get("order_names") or {}).items()}
        self.orders = [(0, 0)] + [tuple(o) for o in vocab["orders"]]
        self.labels = ["none"] + [fx.order_label(o, k, self.names) for o, k in self.orders[1:]]
        self.steps: list[dict] = []
        self.chunks: list = []
        self.traced = None  # (the traced unit's id: the first of its type, by id)
        self.bot = None
        self.unit = unit
        self.chop_types = np.asarray([fourcc(c) for c in HARVESTERS.values()], np.int64)
        self.agg: dict = {}  # ((type, current order, sent to the trees) -> the policy's probabilities summed)
        self.c_move = next((k for k, n in enumerate(self.labels) if n == "move (point)"), 0)
        self.c_trees = [k for k, (o_, k_) in enumerate(self.orders) if k and k_ == fx.TREE and o_ in (fx.SMART, fx.HARVEST)]

    # the learner's queue
    def put(self, msg: dict) -> None:
        if "steps" in msg:
            self.chunks.append(list(msg["steps"]))

    # the inference
    def request(self, items: list) -> list:
        return [self.forward(st) for _, st in items]

    def _tensors(self, st: dict):
        import torch
        from warcraftsim.fullgame import features as fx
        n = st["n"]
        ent = torch.from_numpy(np.asarray(st["ent"], np.float32)[None, :n])
        typ = torch.from_numpy(np.asarray(st["type"], np.int64)[None, :n])
        cur = torch.from_numpy(np.asarray(st["cur"], np.int64)[None, :n])
        mask = torch.ones(1, n, dtype=torch.bool)
        glob = torch.from_numpy(np.asarray(st["glob"], np.float32)[None])
        n_own = torch.tensor([min(st["n_own"], fx.MAX_OWN)])
        avail = torch.from_numpy(np.asarray(st["avail"], bool)[None]) if st.get("avail") is not None else None
        h = torch.from_numpy(np.asarray(st["h"], np.float32)[None]) if st.get("h") is not None else None
        return ent, typ, cur, mask, glob, n_own, avail, h

    def _logits(self, ent, typ, cur, mask, glob, n_own, avail, h):
        from warcraftsim.fullgame import features as fx
        net = self.net
        g, u = net.encode(ent, typ, cur, mask, glob)
        g, _ = net.context(g, h)
        O = min(fx.MAX_OWN, u.shape[1])
        return net.order_logits(g, u[:, :O], typ, n_own, avail=avail, ent=ent, emask=mask if net.omask else None)

    def forward(self, st: dict) -> dict:
        import torch
        from warcraftsim.fullgame import features as fx
        from warcraftsim.fullgame.model import NEG, act
        ent, typ, cur, mask, glob, n_own, avail, h = self._tensors(st)
        shared = torch.tensor([bool(st.get("shared_noise"))])
        with torch.no_grad():
            out = act(self.net, ent, typ, cur, mask, glob, n_own, avail, h, generator=self.gen, shared=shared)
            lg = self._logits(ent, typ, cur, mask, glob, n_own, avail, h)[0]
        o = int(n_own[0])
        res = {"order": out["order"][0, :o].numpy().astype(np.int64), "tgt": out["tgt"][0, :o].numpy().astype(np.int64),
               "bx": out["bx"][0, :o].numpy().astype(np.int64), "by": out["by"][0, :o].numpy().astype(np.int64),
               "logp": out["logp"][0, :o].numpy().copy(), "value": float(out["value"][0]),
               "entropy": float(out["entropy"][0]), "version": 0}
        if self.net.memory:
            res["h"] = out["h"][0].numpy().copy()
        sel = st["sel"]
        own = sel[:st["n_own"]]
        rec = {"t": len(self.steps), "value": round(res["value"], 4)}
        self.steps.append(rec)
        # every harvester of the side (the four races' types), by state (its current order, the view's "sent to the
        # trees" flag): the policy's mean probabilities there, and what was sampled
        a8 = 9 + len(fx.FLAG_BITS) + 8
        pr = torch.softmax(lg[:o].float(), -1).numpy()
        for k in np.flatnonzero(np.isin(own[:o, fx.C_TYPE], self.chop_types)).tolist():
            key = (fx.rawcode(int(own[k, fx.C_TYPE])),
                   self.names.get(int(own[k, fx.C_ORDER]), str(int(own[k, fx.C_ORDER]))), int(st["ent"][k, a8]))
            g_ = self.agg.setdefault(key, {"n": 0, "none": 0.0, "move": 0.0, "tree": 0.0, "sampled move": 0,
                                           "sampled tree": 0})
            g_["n"] += 1
            g_["none"] += float(pr[k, 0])
            g_["move"] += float(pr[k, self.c_move]) if self.c_move else 0.0
            g_["tree"] += float(pr[k, self.c_trees].sum())
            c_ = int(res["order"][k])
            g_["sampled move"] += int(c_ == self.c_move)
            g_["sampled tree"] += int(c_ in self.c_trees)
        of_type = np.flatnonzero(own[:, fx.C_TYPE] == fourcc(self.unit))
        if len(of_type):
            if self.traced is None:
                self.traced = int(own[of_type, fx.C_ID].min())
            at = np.flatnonzero(own[:, fx.C_ID] == self.traced)
            i = int(at[0]) if len(at) else None
            if i is not None and i < o:
                p = pr[i]
                ok = (lg[i] > NEG / 2).numpy()
                top = np.argsort(-p)[:5]
                c = int(res["order"][i])
                oid, kind = self.orders[c]
                rec.update(
                    unit=self.traced, x=int(own[i, fx.C_X]), y=int(own[i, fx.C_Y]), cur=int(own[i, fx.C_ORDER]),
                    cur_name=self.names.get(int(own[i, fx.C_ORDER]), str(int(own[i, fx.C_ORDER]))),
                    flags=int(own[i, fx.C_FLAGS]), lumber_feat=float(st["ent"][i, a8]),
                    n_allowed=int(ok.sum()), allowed=[self.labels[k] for k in np.flatnonzero(ok)],
                    top=[(self.labels[k], round(float(p[k]), 4)) for k in top], p_none=round(float(p[0]), 4),
                    sampled=self.labels[c], gold=int(round(float(st["glob"][0]) * 1000)),
                    bank_lumber=int(round(float(st["glob"][1]) * 1000)))
                if kind in (fx.POINT, fx.TREE) and self.bot is not None:
                    rec["point"] = [int(v) for v in fx.bins_world(int(res["bx"][i]), int(res["by"][i]), self.bot.view.sign)]
                elif kind == fx.UNIT:
                    rec["target"] = fx.rawcode(int(sel[int(res["tgt"][i]), fx.C_TYPE]))
                if self.counter:
                    rec["counter"] = self._counter(i, (ent, typ, cur, mask, glob, n_own, avail, h))
                    # (the order head's own ranking for this unit, without the masks by unit type and by what the player
                    # can pay for: what the mask's "allowed" leaves is what gets sampled)
                    with torch.no_grad():
                        g_, u_ = self.net.encode(ent, typ, cur, mask, glob)
                        g_, _ = self.net.context(g_, h)
                        raw = self.net.order_logits(g_, u_[:, :o], typ, n_own, by_type=False)[0, i]
                        q = torch.softmax(raw.float(), -1).numpy()
                    rec["raw_top"] = [(self.labels[k], round(float(q[k]), 3)) for k in np.argsort(-q)[:6]]
        return res

    def _counter(self, i: int, t) -> dict:
        """The traced unit's probabilities (none, a move, a tree order...) and the value with one input changed."""
        import torch
        from warcraftsim.fullgame import features as fx
        from warcraftsim.fullgame import modes as zmodes, strategy as strat
        ent, typ, cur, mask, glob, n_own, avail, h = t
        net = self.net
        a8 = 9 + len(fx.FLAG_BITS) + 8
        z0 = net.config["G"] - net.priv - net.config.get("gx", 0) - net.config["z"]
        want = {n: k for k, n in enumerate(self.labels) if n in ("none", "move (point)", "smart (tree)", "harvest (tree)",
                                                                 "attack (point)", "stop", "holdposition")}

        def probs(ent=ent, glob=glob, h=h):
            with torch.no_grad():
                p = torch.softmax(self._logits(ent, typ, cur, mask, glob, n_own, avail, h)[0, i].float(), -1)
                g, _ = net.encode(ent, typ, cur, mask, glob)
                g, _ = net.context(g, h)
                v = float(net.value(g, glob)[0])
            return {**{n: round(float(p[k]), 3) for n, k in want.items()}, "value": round(v, 4)}

        def with_glob(at: int, width: int, value: float):
            g2 = glob.clone()
            g2[:, at:at + width] = value
            return g2
        out = {"as is": probs()}
        blank = with_glob(z0, net.config["z"], 0.0)
        out["no target (z zeroed)"] = probs(glob=blank)
        e2 = ent.clone()
        e2[0, i, a8] = 1.0 - e2[0, i, a8]
        out["lumber flag flipped"] = probs(ent=e2)
        out["no target, flag flipped"] = probs(ent=e2, glob=blank)
        # the strategy block's parts (strategy.vector's items made and their times, the modes, the army mix)
        parts = (("items", 0, strat.ZDIM), ("modes", strat.ZDIM, zmodes.MDIM), ("mix", strat.ZDIM + zmodes.MDIM, strat.SLOTS))
        for name, at, w in parts if net.config["z"] >= strat.ZDIM + zmodes.MDIM + strat.SLOTS else ():
            out[f"target without its {name}"] = probs(glob=with_glob(z0 + at, w, 0.0))
        out["bank lumber 0"] = probs(glob=with_glob(1, 1, 0.0))
        out["bank lumber 500"] = probs(glob=with_glob(1, 1, 0.5))
        out["bank gold 500"] = probs(glob=with_glob(0, 1, 0.5))
        out["time 0"] = probs(glob=with_glob(5, 1, 0.0))
        out["time 120 s"] = probs(glob=with_glob(5, 1, 240 / 1800.0))  # (the clock's steps / 1800)
        out["time 300 s"] = probs(glob=with_glob(5, 1, 600 / 1800.0))
        out["food 9 of 10"] = probs(glob=torch.cat([glob[:, :2], glob.new_tensor([[0.09, 0.10]]), glob[:, 4:]], 1))
        out["no memory"] = probs(h=None)
        return out


def install(probe: Probe) -> None:
    """The wrappers: what the agent sent for the traced unit, the teacher's lessons, the builder's state."""
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame import selfplay as sp
    from warcraftsim.fullgame.play import BCAgent
    from warcraftsim.fullgame.teacher import Teacher

    b_begin, b_commands, b_accepted = BCAgent.begin, BCAgent.commands, BCAgent.accepted

    def begin(self, obs, races):
        b_begin(self, obs, races)
        if self.player == probe.side:
            probe.bot = self

    def commands(self, st, order, tgt, bx, by):
        out = b_commands(self, st, order, tgt, bx, by)
        if self.player == probe.side and probe.steps and probe.traced is not None:
            rec = probe.steps[-1]
            rec["sent"] = [(type(c).__name__, probe.names.get(getattr(c, "order", 0), getattr(c, "order", None)),
                            getattr(c, "x", None), getattr(c, "y", None), getattr(c, "target", None),
                            getattr(c, "destructable", None))
                           for c in out if getattr(c, "unit", None) == probe.traced]
            rec["assign"] = self.view.assign.get(probe.traced)
            rec["choppers"] = len(self.view.lumber_now)
            rec["all_sent"] = len(out)
        return out

    def accepted(self, cmds, results):
        if self.player == probe.side and probe.steps and probe.traced is not None:
            probe.steps[-1]["taken"] = [bool(ok) for c, ok in zip(cmds, results) if getattr(c, "unit", None) == probe.traced]
        return b_accepted(self, cmds, results)
    BCAgent.begin, BCAgent.commands, BCAgent.accepted = begin, commands, accepted

    t_commands = Teacher.commands

    def lessons(self, rows, player, now, t, can=None, pickers=()):
        out = t_commands(self, rows, player, now, t, can, pickers)
        if self.side == probe.side and probe.steps:
            rec = probe.steps[-1]
            eco = getattr(self, "economy", None)
            rec["lumber_short"] = bool(eco[0]) if eco else None
            rec["lessons"] = [(type(c).__name__, probe.names.get(getattr(c, "order", 0), fx.rawcode(getattr(c, "order", 0))
                                                                 if getattr(c, "order", 0) >= fx.TYPE_CODE else getattr(c, "order", None)),
                               getattr(c, "unit", None) == probe.traced) for c in out]
        return out
    Teacher.commands = lessons

    tb_commands = sp.TargetBuilder.commands

    def builder(self, obs, made, t, step_s, ordered=(), rows=None):
        out = tb_commands(self, obs, made, t, step_s, ordered, rows)
        if self.side == probe.side and probe.steps:
            probe.steps[-1]["builder"] = {"why_not": self.why_not, "short": self.lumber_short,
                                          "reserve": list(self.reserve) if self.reserve else None,
                                          "labels": [sp.TargetBuilder._name(c) for c in out]}
        return out
    sp.TargetBuilder.commands = builder


def line(r: dict) -> str:
    if "unit" not in r:
        return f"t={r['t']:3d} (no such unit)"
    sent = "; ".join(f"{n}:{o}" + (f"@({x:.0f},{y:.0f})" if x is not None else "") + (f"->{d or tg}" if (d or tg) else "")
                     for n, o, x, y, tg, d in r.get("sent", [])) or "-"
    less = ",".join(f"{o}{'*' if mine else ''}" for _, o, mine in r.get("lessons", [])) or "-"
    top = " ".join(f"{n}={p:.2f}" for n, p in r["top"][:4])
    return (f"t={r['t']:3d} cur={r['cur_name']:<16} assign={str(r.get('assign')):<6} lumf={r['lumber_feat']:.0f} "
            f"bank={r['gold']:4d}/{r['bank_lumber']:3d} allowed={r['n_allowed']:2d} none={r['p_none']:.2f} | {top} | "
            f"sampled={r['sampled']}{r.get('point', '')} sent={sent} taken={r.get('taken', [])} "
            f"short={r.get('lumber_short')} lessons={less}")


def engine(a) -> int:
    """--engine: each race's harvester right-clicked onto a tree and another sent with harvest (undead's one ghoul:
    right-clicked), left alone for 120 steps: the current order each reads when it changes, the lumber its player has
    gathered."""
    from warcraftsim.protocol import TargetDestructable
    from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup
    names = {SMART: "smart", HARVEST: "harvest", 852017: "resumeharvesting", 852020: "returnresources", 0: "none"}
    slots = [Agent(r, handicap=100, difficulty="normal") for r in a.engine]
    gs = GameSetup(map="(2)EchoIsles", slots=slots, step_seconds=0.5, draw=False, native_obs=False, d3d_thread=False,
                   render_threads=0, nice=19, warm_spare=False, window=(320, 240), max_game_seconds=0, victory="none",
                   cuts=a.cuts, **({"shim_dir": a.shim} if a.shim else {}))
    with GameInstance(gs, name="ghoulprobe", timeout=600) as g:
        obs = g.start()
        watch, cmds = {}, []
        for s, r in enumerate(a.engine):
            ws = [u for u in obs.units if u.owner == s and u.type == HARVESTERS[r]]
            hall = next(u for u in obs.units if u.owner == s and u.is_structure)
            trees = sorted(obs.destructables, key=lambda d: (d.x - hall.x) ** 2 + (d.y - hall.y) ** 2)
            for k, (u, order) in enumerate(zip(ws[:2], (SMART, HARVEST))):
                cmds.append(TargetDestructable(u.id, order, trees[3 * k].id))
                watch[u.id] = (r, HARVESTERS[r], names[order], s)
        obs = g.step(cmds)
        print("taken:", list(obs.command_results), flush=True)
        last: dict = {}
        for k in range(120):
            obs = g.step([])
            for u in obs.units:
                if u.id in watch:
                    r, w, how, s = watch[u.id]
                    state = (names.get(u.order, u.order), obs.players[s].lumber_gathered)
                    if state != last.get(u.id):
                        print(f"  {r} {w} sent with {how}: step {k:3d} reads {state[0]}, its player's lumber gathered "
                              f"{state[1]}", flush=True)
                        last[u.id] = state
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--run", type=Path, help="the run whose settings the game is played with (run.json, vocab.json)")
    ap.add_argument("--target", type=int, default=1, help="1: a game with a target strategy; 0: without")
    ap.add_argument("--steps", type=int, default=130)
    ap.add_argument("--race", default="undead")
    ap.add_argument("--opp", default="orc")
    ap.add_argument("--ai", default="insane")
    ap.add_argument("--side", type=int, default=1, help="the learner's player (the replay's: 1)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], help="settings changed for this game: key=value (econ_masks=511)")
    ap.add_argument("--unit", default="", help="the unit type traced step by step (its first unit by id; default: the "
                    "race's lumber harvester, undead's ghoul)")
    ap.add_argument("--quiet", action="store_true", help="no line a step: the states' table and the result only")
    ap.add_argument("--counter", action="store_true", help="per step the probabilities with one input changed")
    ap.add_argument("--json", type=Path, help="the steps' records, the states' table and the episode row written here")
    ap.add_argument("--engine", nargs=2, metavar="RACE", help="no policy: these two races' harvesters right-clicked "
                    "onto a tree and sent with harvest, the current order each reads")
    ap.add_argument("--cuts", type=int, default=17592186041849, help="--engine: W3SIM_CUTS (the runs')")
    ap.add_argument("--shim", type=Path, default=None)
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    if a.engine:
        return engine(a)
    if a.weights is None or a.run is None:
        ap.error("--weights and --run (or --engine RACE RACE)")
    import torch
    torch.set_num_threads(2)
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
    cfg = run_cfg(a.run, vocab, over)
    # the features' settings, as selfplay.main sets them before any network is loaded
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
    net, _ = load(a.weights, "cpu", vocab=vocab)
    fx.set_z(net.config.get("z", 0))
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    side = a.side
    races = [a.opp, a.opp]
    races[side] = a.race
    a.unit = a.unit or HARVESTERS[a.race]
    probe = Probe(net, vocab, side, a.seed, a.counter, a.unit)
    install(probe)
    slots = [None, None]
    slots[side] = Agent(races[side], handicap=cfg["handicap"], difficulty="normal")
    slots[1 - side] = BuiltinAI(races[1 - side], a.ai, handicap=cfg["handicap"])
    setup = GameSetup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"], max_game_seconds=cfg["max_minutes"] * 60,
                      victory="decisive", window=(320, 240), draw=False, wait_floor_ms=cfg["wait_floor_ms"],
                      native_obs=True, nice=19, d3d_thread=False, render_threads=0, warm_spare=False,
                      cuts=cfg.get("game_cuts") or None, obs_full=bool(cfg.get("obs_full")),
                      **({"shim_dir": a.shim} if a.shim else {}))
    opp = {"name": f"script:ai-{a.ai} (real)", "kind": "ai", "advise": False}
    with GameInstance(setup, name="ghoulprobe", timeout=600) as g:
        obs = g.start()
        ep = sp.play_one(g, obs, cfg, vocab, races, side, opp, probe, probe, 0, record=False, train=True)
    for r in probe.steps if not a.quiet else ():
        print(line(r), flush=True)
    moves = sum(1 for r in probe.steps if any(o == "move" for _, o, *_ in r.get("sent", [])))
    trees = sum(1 for r in probe.steps if any(n == "TargetDestructable" for n, *_ in r.get("sent", [])))
    print("the side's harvesters by state (type, current order, sent to the trees): steps, the policy's mean P(none), "
          "P(move on a point), P(a tree order); moves sampled, tree orders sampled")
    for key, v in sorted(probe.agg.items(), key=lambda kv: -kv[1]["n"]):
        n = v["n"]
        print(f"  {key[0]} cur={key[1]:<16} lumber={key[2]}: {n:5d}  none={v['none'] / n:.2f} move={v['move'] / n:.2f} "
              f"tree={v['tree'] / n:.2f}  moves={v['sampled move']} trees={v['sampled tree']}")
    print(f"RESULT race={a.race} target={a.target} econ_masks={cfg['econ_masks']} steps={len(probe.steps)} "
          f"{a.unit}_moves={moves} {a.unit}_tree_orders={trees} lumber_gathered={(ep.get('prod') or {}).get('lumber')} "
          f"z={ep.get('z')} z_t={ep.get('z_t')} taught={ep.get('taught')}", flush=True)
    if a.json:
        flat = [s for ch in probe.chunks for s in ch]
        out = {"steps": probe.steps, "episode": {k: v for k, v in ep.items() if k != "trace"},
               "rewards": [float(s.get("reward", 0.0)) for s in flat], "states": [[*k, v] for k, v in probe.agg.items()]}
        a.json.write_text(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
