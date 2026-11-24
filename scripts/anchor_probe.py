"""What the KL anchor pulls the self-play policy toward (2026-11-24; a measurement, no training change).

The runs add --ref-kl times a forward KL (--ref-kl-mode forward-all) between a frozen anchor (--init: fgself-100's
checkpoint, five weeks old) and the learner. The anchor "holds the macro up" (no anchor, a rolling one, one at the run's
start: worse or level). Does it also hold bad habits in place (an army called home by the side far ahead, no second base,
no creeping)?

    python scripts/anchor_probe.py --weights runs/fgself-252/averaged/swa-17-27.pt --run runs/fgself-253 \\
        --races orc orc --seed 1 --out OUT/orc-orc-1.npz [--every 4] [--steps 4800]
    python scripts/anchor_probe.py --report OUT/*.npz

Plays ONE game of the policy against itself as an evaluation game plays (selfplay.play_one, train=False: no target
strategy, no teachers, the run's map and masks from its run.json, read only; the exact game's cuts; both networks on
the CPU) and, on a share of each side's steps, evaluates the policy AND the anchor on the same observation, with the
step's own action, as the update does (model.evaluate, model.anchor_kl "forward-all"):
  * the anchor is loaded as selfplay.main loads it (model.load, the run's vocabulary and masks; its table of allowed
    orders copied from the policy's);
  * the update cuts a side's steps into sequences of --seq-len 16 and starts BOTH networks' memory from the state the
    actor had at the sequence's first step, the anchor's then runs on its own gates: so does this. Of every --every
    sequences one is evaluated whole (--every 4: a quarter of the steps, 16 in a row, then 48 not);
  * a unit's term: KL(anchor || policy) over the orders, plus, when the step's order uses them, over the target unit,
    the point's x, and its y given the step's x (the first evaluated steps and every 50th after are checked against
    model.anchor_kl).

What it keeps (--out, a small .npz):
  1. where the KL is: its four heads' sums by game phase (0-5, 5-15, 15-25, 25+ minutes), unit class (worker: the
     worker types and a unit the view has on lumber, undead's ghouls; army; hero; building) and the anchor's top order;
     with each cell's summed order probabilities of both networks (the direction: 3).
     1b. the point heads' units (an army unit or hero whose own order of the step is an attack-move or a move): where
     the anchor would point that order and where the policy does, by where the unit stands.
  2. what the anchor wants where the policy misbehaves:
     a. late states (minute 15+) by "far ahead" (own material >= 2x the opponent's; 1.5-2x apart) and "an enemy army
        unit in view": for the army units and heroes, both networks' chance of an attack-move, a move, no order, and
        where their attack-moves and moves point (the point's x and y joint, given the order; up to --points army units
        a step): within 1,500 of its own start, of the enemy's start, elsewhere (and: within 1,000 of a living enemy
        building, outside the map's camera bounds);
     b. minutes 4-12 with 6+ army units and a hero: the chance of an attack-move to a cell within 600 of a living
        creep, of an attack on a creep in view, and of the expansion orders (a worker's hall, undead's haunted mine);
        the hall's order also on every evaluated step, by race and phase (how often the mask leaves it open).
  3. the pull's direction: from 1's sums, the anchor's and the policy's chance of each cell's orders.

What it found (fgself-252's averaged weights swa-17-27 against themselves, fgself-253's settings, 16 games: 3 orc
mirrors, 3 orc/undead, 3 human/undead, 2 undead mirrors, 2 night elf/undead, human/orc, human/night elf, a night elf
mirror; 9 to 40 minutes, 5 past minute 25, 3 ties at 40; 21,474 of 85,660 side-steps evaluated, 596,116 unit-steps;
the check's largest difference 7e-7, the two networks' masks never differ). "a / p": the anchor's / the policy's.
  * the mean KL a unit-step is 0.109 (the run logs ~0.13 over its training games; 0.09-0.14 a game). By head: the order
    66%, the point's x 17%, its y 15%, the target unit 2%. By class: army units 51%, heroes 22% (3% of the unit-steps:
    0.84 a hero-step), buildings 14%, workers 13%. By phase as the unit-steps are (0-5: 17%, 5-15: 42%, 15-25: 30%,
    25+: 11%). So about three quarters of the term is on the army's and the heroes' orders and their points, a quarter
    on the macro (workers and buildings).
  * 70% of the term is on unit-steps where the anchor's top order is "none" (98% of the unit-steps), and there the
    policy's is too: the two give an attack-move or a move at about the same rate (2-3% an army unit-step) but on other
    steps. Where the anchor's top order is an order: a move 13%, an attack-move 8%, a worker's training 4%, an army
    unit's 2.5%.
  * the direction, orders a 1,000 unit-steps (a / p): every class of the policy's gives fewer orders than the anchor
    would. Buildings: a worker's training 5.2 / 0.4, an army unit's 6.4 / 2.3. Heroes: a move 148 / 51, an attack-move
    90 / 59. Army units: an attack-move 28 / 20, a move 31 / 29, a ghoul's tree order 8 / 2. Workers: harvest orders
    again 11 / 4. What the policy gives and the anchor does not: a hero's carrion swarm 0.4 / 4.2 and patrol, undead's
    haunt order, a worker's attack-move.
  * far ahead (>= 2x material) at minute 15+ with no enemy army unit in view (631 side-steps of 9 games, 7,346 army
    unit-steps): no order 0.965 / 0.958, an attack-move 0.016 / 0.025, a move 0.014 / 0.015. The attack-moves point at
    the enemy's start 0.93 / 0.91, at the own start 0.01 / 0.00: neither network sends the army home, the anchor
    presses a little less. The few moves point home 0.38 / 0.25 (4.4 / 2.2 a 1,000 unit-steps). Heroes (862): the anchor
    would move them home (66% of its moves; ~90 / 1 a 1,000 hero-steps) or attack-move them to the enemy's start (0.91
    of its attack-moves); the policy points 41% of their attack-moves and 74% of their moves outside the camera bounds
    (the anchor 1-2%). The army stood at home in 126 of those unit-steps only: the replay's pattern (an army at home
    attack-moved home again) did not occur in these games. With an enemy in view and not ahead, a unit at home: the
    anchor's attack-moves stay at home 0.32 / 0.19 and go to the enemy's start 0.04 / 0.20.
  * minutes 4-12 with 6+ army units and a hero (6,012 side-steps): an attack on a creep in view 0.0000 / 0.0000; an
    attack-move within 600 of a living creep 0.5 / 0.5 a 1,000 army unit-steps, 2.6 / 1.8 a 1,000 hero-steps. Neither
    creeps.
  * the hall's order is open to a worker (the bank covers it) on 0% of orc's side-steps, 5% of human's, 9% of night
    elf's, 4-20% of undead's by phase (mean gold 130-210). Where open, a side-step: human 0.009 / 0.0003, the anchor's
    at home (0.92), the policy's at another mine (0.97); night elf 0.0005 / 0.006-0.026, the policy's at another mine
    (0.68-0.94), the anchor's at home; undead 0.001-0.011 / 0.000-0.002, both at home (no expansion). Undead's haunt
    order 0.0000 / 0.003-0.008 a side-step from minute 5. No game had an expansion.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PHASES = ("0-5", "5-15", "15-25", "25+")
CLASSES = ("worker", "army", "hero", "building")
HEADS = ("order", "pointer", "x", "y")
HALLS = {"human": "htow", "orc": "ogre", "undead": "unpl", "nightelf": "etol"}
NEAR_BASE, NEAR_CREEP, NEAR_BUILDING, NEAR_MINE = 1500.0, 600.0, 1000.0, 1000.0
SEQ = 16  # (--seq-len: the update's sequences)
W2 = 60  # (the width of 2's sums)


def phase_of(minute: float) -> int:
    return 0 if minute < 5 else 1 if minute < 15 else 2 if minute < 25 else 3


def group_of(label: str) -> str:
    """An order class's kind, for the tables."""
    if label == "none":
        return "none"
    if label == "attack (point)":
        return "attack-move"
    if label == "move (point)":
        return "move"
    if label == "attack (unit)":
        return "attack a unit"
    if label.startswith("smart"):
        return "smart (right-click)"
    if label.startswith("harvest"):
        return "harvest"
    if label in ("stop", "holdposition") or label.startswith("patrol"):
        return "stop/hold/patrol"
    if label.startswith("train "):
        code = label[6:]
        if code in ("hpea", "opeo", "uaco", "ewsp"):
            return "train a worker"
        if code[:1].isupper():
            return "train a hero"
        if code in ("hkee", "hcas", "ostr", "ofrt", "unp1", "unp2", "etoa", "etoe"):
            return "tier"
        return "train an army unit"
    if label.startswith("research "):
        return "research"
    if label.startswith("build "):
        return "build a hall" if label[6:] in HALLS.values() else "build"
    if label == "ugol":
        return "haunt a mine"
    if label.startswith("learn "):
        return "learn a skill"
    if label.startswith("repair") or label.startswith("revive"):
        return "repair/revive"
    return "spell/ability/item"


class Cells:
    """Sums by (phase, class, the anchor's top order)."""

    def __init__(self, C: int):
        self.C = C
        self.at: dict[tuple, int] = {}
        self.n = np.zeros(0, np.int64)
        self.kl = np.zeros((0, 4), np.float64)
        self.A = np.zeros((0, C), np.float64)  # the anchor's order probabilities, summed
        self.P = np.zeros((0, C), np.float64)  # the policy's
        self.T = np.zeros((0, C), np.float64)  # the order KL's terms a (log a - log p), summed by order

    def row(self, key: tuple) -> int:
        i = self.at.get(key)
        if i is None:
            i = self.at[key] = len(self.at)
            if i >= len(self.n):
                grow = max(64, len(self.n))
                self.n = np.concatenate([self.n, np.zeros(grow, np.int64)])
                self.kl = np.concatenate([self.kl, np.zeros((grow, 4))])
                for k in ("A", "P", "T"):
                    setattr(self, k, np.concatenate([getattr(self, k), np.zeros((grow, self.C))]))
        return i


class Probe:
    """The game's stand-ins (the inference, the learner's queue) and the measurement."""

    def __init__(self, net, ref, vocab: dict, cfg: dict, seed: int, every: int, points: int, bounds):
        import torch
        from warcraftsim.fullgame import features as fx
        self.net, self.ref, self.cfg, self.every, self.points = net, ref, cfg, max(1, every), points
        self.fx = fx
        self.nets = type("Nets", (), {"version": 0})()
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        names = {int(k): v for k, v in (vocab.get("order_names") or {}).items()}
        self.labels = ["none"] + [fx.order_label(int(o), int(k), names) for o, k in vocab["orders"]]
        ix = {l: i for i, l in enumerate(self.labels)}
        self.c_am, self.c_mv, self.c_atk = ix["attack (point)"], ix["move (point)"], ix["attack (unit)"]
        self.c_smart = [i for i, l in enumerate(self.labels) if l.startswith("smart")]
        self.c_hold = [i for i, l in enumerate(self.labels) if l in ("stop", "holdposition") or l.startswith("patrol")]
        self.c_hall = {r: ix.get(f"build {c}") for r, c in HALLS.items()}
        self.c_ugol = ix.get("ugol")
        self.cells = Cells(len(self.labels))
        self.k = {0: 0, 1: 0}  # each side's steps so far
        self.ref_h = {0: None, 1: None}  # the anchor's memory inside the sequence
        self.on = {0: False, 1: False}
        self.minute_kl = np.zeros((41, 5))  # per game minute: the four heads' sums, the units
        self.pos_kl = np.zeros((SEQ, 2))  # by the step's place in its sequence: the KL's sum, the units
        self.late = defaultdict(lambda: np.zeros(W2))  # 2a's sums, by (far ahead, enemy army in view, class, where)
        self.mid = defaultdict(lambda: np.zeros(W2))  # 2b's
        self.pts = defaultdict(lambda: np.zeros(20))  # 1b's: the point heads' units, by (phase, class, order, where)
        self.exp = defaultdict(lambda: np.zeros(W2))  # the expansion orders on every evaluated step, by (race, phase)
        self.checked = 0
        self.check_max = 0.0
        self.mask_differs = 0
        self.big = 0
        self.evaluated = {0: 0, 1: 0}
        self.unit_steps = 0
        self.mat_minute: dict[int, list] = {}  # game minute -> [material side 0, side 1]
        self.eval_ms = 0.0
        self._t, self._shared = -1, None
        self.bounds = bounds
        self.grids: dict[float, tuple] = {}
        self.values = cfg["values"]

    # the learner's queue
    def put(self, msg: dict) -> None:
        pass

    # the inference
    def request(self, items: list) -> list:
        return [self.forward(st) for _, st in items]

    # ---- the wrappers' note: what the step's state is (both sides share the rows)
    def note(self, bot, obs, t: int, st: dict | None) -> None:
        if st is None:
            return
        fx = self.fx
        if self._t != t:
            from warcraftsim.fullgame.trace import material
            rows = bot.rows
            rows = rows[rows[:, fx.C_FLAGS] & fx.DEAD == 0]
            minute = float(obs.game_time) / 60.0
            mat = material(obs, self.values)
            self._t, self._shared = t, (rows, minute, mat)
            self.mat_minute.setdefault(int(minute), [round(mat[0]), round(mat[1])])
        rows, minute, mat = self._shared
        side = bot.player
        fl, owner = rows[:, fx.C_FLAGS], rows[:, fx.C_OWNER]
        seen = (rows[:, fx.C_VIS] >> side) & 1 == 1
        foe = owner == 1 - side
        foe_army = foe & seen & (fl & (fx.STRUCTURE | fx.WORKER) == 0)
        st["_probe"] = {
            "side": side, "minute": minute, "sign": float(bot.view.sign), "mat": (float(mat[side]), float(mat[1 - side])),
            "foe_army_seen": int(foe_army.sum()),
            "foe_buildings": rows[foe & (fl & fx.STRUCTURE > 0)][:, [fx.C_X, fx.C_Y]].astype(np.float64),
            "creeps": rows[(owner == 12) & (fl & fx.STRUCTURE == 0)][:, [fx.C_X, fx.C_Y]].astype(np.float64),
            "mines": rows[np.isin(rows[:, fx.C_TYPE], fx.GOLD_MINES)][:, [fx.C_X, fx.C_Y]].astype(np.float64),
            "race": getattr(bot, "_race", None)}

    def _tensors(self, st: dict):
        import torch
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
        return ent, typ, cur, mask, glob, n_own, avail, h

    def forward(self, st: dict) -> dict:
        import torch
        from warcraftsim.fullgame.model import act
        ent, typ, cur, mask, glob, n_own, avail, h = t = self._tensors(st)
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
        note = st.get("_probe")
        if note is not None:
            side = note["side"]
            k = self.k[side]
            self.k[side] = k + 1
            if k % SEQ == 0:  # (a sequence's first step: the anchor's memory from the actor's state, as the update's)
                self.ref_h[side] = h
                self.on[side] = (k // SEQ + side * (self.every // 2)) % self.every == 0
            if self.on[side] and o > 0:
                t0 = time.perf_counter()
                with torch.no_grad():
                    self.measure(st, note, t, out, k % SEQ)
                self.eval_ms += 1000 * (time.perf_counter() - t0)
        return res

    @staticmethod
    def _eval(net, ent, typ, cur, mask, glob, n_own, order, bx, avail, h) -> dict:
        """model.evaluate's passes for one step (its seq = (1, 1, h, None)), and the memory's new state."""
        g, u = net.encode(ent, typ, cur, mask, glob)
        g, h_new = net.context(g, h)
        O = order.shape[1]
        logits = net.order_logits(g, u[:, :O], typ, n_own, avail=avail, ent=ent, emask=mask if net.omask else None)
        ptr, xl, z = net.target_logits(g, u, mask, order, ent, typ)
        yl = net.y_logits(z, bx)
        uses_ptr, uses_pt = net.uses(order)
        return {"logits": logits, "ptr": ptr, "xl": xl, "yl": yl, "uses_ptr": uses_ptr, "uses_pt": uses_pt, "g": g, "u": u,
                "h": h_new}

    def measure(self, st: dict, note: dict, t, out: dict, pos: int) -> None:
        import torch
        from warcraftsim.fullgame.model import NEG, anchor_kl, evaluate
        fx = self.fx
        ent, typ, cur, mask, glob, n_own, avail, h = t
        side = note["side"]
        order, tgt, bx, by = out["order"], out["tgt"], out["bx"], out["by"]
        o = int(n_own[0])
        ev = self._eval(self.net, ent, typ, cur, mask, glob, n_own, order, bx, avail, h)
        h_ref = self.ref_h[side]
        er = self._eval(self.ref, ent, typ, cur, mask, glob, n_own, order, bx, avail, h_ref)
        self.ref_h[side] = er["h"]

        def kl(a, b):
            la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
            return (la.exp() * (la - lb)).sum(-1)
        la, lb = torch.log_softmax(er["logits"].float(), -1), torch.log_softmax(ev["logits"].float(), -1)
        pa, pp = la.exp(), lb.exp()
        terms = pa * (la - lb)  # [1, O, C]
        parts = torch.stack([terms.sum(-1), ev["uses_ptr"] * kl(er["ptr"], ev["ptr"]), ev["uses_pt"] * kl(er["xl"], ev["xl"]),
                             ev["uses_pt"] * kl(er["yl"], ev["yl"])], -1)[0, :o].double().numpy()  # [o, 4]
        if self.checked < 24 or (self.evaluated[0] + self.evaluated[1]) % 50 == 0:  # (the split against the update's
            # own functions: the first steps, then every 50th evaluated one)
            self.checked += 1
            seq_p, seq_r = (1, 1, h, None), (1, 1, h_ref, None)
            e1 = evaluate(self.net, ent, typ, cur, mask, glob, n_own, order, tgt, bx, by, avail, seq=seq_p)
            e2 = evaluate(self.ref, ent, typ, cur, mask, glob, n_own, order, tgt, bx, by, avail, seq=seq_r)
            want = anchor_kl(e1, e2, "forward-all")[0, :o].double().numpy()
            d = float(np.abs(want - parts.sum(-1)).max()) if o else 0.0
            self.check_max = max(self.check_max, d)
            if d > 1e-3 * (1.0 + float(np.abs(want).max())):
                raise RuntimeError(f"the probe's KL differs from model.anchor_kl's by {d}")
        ok_p, ok_r = ev["logits"][0, :o] > NEG / 2, er["logits"][0, :o] > NEG / 2
        self.mask_differs += int((ok_p != ok_r).sum())
        self.big += int((parts.sum(-1) > 50).sum())
        pa, pp, terms = pa[0, :o].double().numpy(), pp[0, :o].double().numpy(), terms[0, :o].double().numpy()
        sel = st["sel"]
        fl = sel[:o, fx.C_FLAGS].astype(np.int64)
        lumber = np.asarray(st["ent"])[:o, 9 + len(fx.FLAG_BITS) + 8] > 0.5  # (the view's "a worker on lumber")
        cls = np.where(fl & fx.STRUCTURE > 0, 3, np.where(fl & fx.HERO > 0, 2, np.where((fl & fx.WORKER > 0) | lumber, 0, 1)))
        minute = note["minute"]
        ph = phase_of(minute)
        top = pa.argmax(-1)
        cells = self.cells
        for i in range(o):
            r = cells.row((ph, int(cls[i]), int(top[i])))
            cells.n[r] += 1
            cells.kl[r] += parts[i]
            cells.A[r] += pa[i]
            cells.P[r] += pp[i]
            cells.T[r] += terms[i]
        m = min(int(minute), 40)
        self.minute_kl[m, :4] += parts.sum(0)
        self.minute_kl[m, 4] += o
        self.pos_kl[pos] += (parts.sum(), o)
        self.evaluated[side] += 1
        self.unit_steps += o
        # ---- 2: the states in question
        xy = sel[:o][:, [fx.C_X, fx.C_Y]].astype(np.float64)
        own_start, foe_start = fx.own_start(note["sign"]), fx.own_start(-note["sign"])
        d_home = np.hypot(xy[:, 0] - own_start[0], xy[:, 1] - own_start[1])
        d_foe = np.hypot(xy[:, 0] - foe_start[0], xy[:, 1] - foe_start[1])
        where = np.where(d_home <= NEAR_BASE, 0, np.where(d_foe <= NEAR_BASE, 1, 2))  # home, the enemy's base, away
        army, hero = np.flatnonzero(cls == 1), np.flatnonzero(cls == 2)
        late, mid = minute >= 15.0, 4.0 <= minute < 12.0 and len(army) >= 6 and len(hero) >= 1
        grid = self._grid(note["sign"])
        own_m, foe_m, out_m = grid[2], grid[3], grid[4]
        near: dict = {}

        def shares(j):  # [K, BINS, BINS] -> [K, 6]: own base, the enemy's, elsewhere; an enemy building, outside, a creep
            if not near:
                near["b"] = self._near(grid, note["foe_buildings"], NEAR_BUILDING)
                near["c"] = self._near(grid, note["creeps"], NEAR_CREEP)
            own_s, foe_s = (j * own_m).sum((1, 2)), (j * foe_m).sum((1, 2))
            return np.stack([own_s, foe_s, 1.0 - own_s - foe_s, (j * near["b"]).sum((1, 2)), (j * out_m).sum((1, 2)),
                             (j * near["c"]).sum((1, 2))], -1)
        # ---- 1b: the point heads' units (the step's own order an attack-move or a move, of an army unit or a hero):
        # where the anchor would point it and where the policy does, by where the unit stands
        acted = order[0, :o].numpy()
        for kind, c in ((0, self.c_am), (1, self.c_mv)):
            units = np.flatnonzero((acted == c) & ((cls == 1) | (cls == 2)))
            self.pts[("n", ph, kind)][0] += len(units)
            if len(units) > 16:
                units = self.rng.permutation(units)[:16]
            if not len(units):
                continue
            sa_, sp_ = (shares(self._joint(n_, e_, ent, typ, mask, c, o, units)) for n_, e_ in ((self.ref, er), (self.net, ev)))
            px = grid[0][out["bx"][0, units].numpy()]
            py = grid[1][out["by"][0, units].numpy()]
            went = np.where(np.hypot(px - own_start[0], py - own_start[1]) <= NEAR_BASE, 0,
                            np.where(np.hypot(px - foe_start[0], py - foe_start[1]) <= NEAR_BASE, 1, 2))
            for k, i in enumerate(units.tolist()):
                v = self.pts[(ph, int(cls[i]), kind, int(where[i]))]
                v[0] += 1
                v[1] += parts[i, 2] + parts[i, 3]
                v[2:8] += sa_[k]
                v[8:14] += sp_[k]
                v[14 + int(went[k])] += 1
        self._expansion(st, note, ev, er, ent, typ, mask, pa, pp, fl, o, grid, own_start, self.exp, ph)
        if not (late or mid) or not (len(army) or len(hero)):
            return
        pick = np.concatenate([self.rng.permutation(army)[:self.points], hero]).astype(np.int64)
        ja = self._joint(self.ref, er, ent, typ, mask, self.c_am, o, pick)
        jp = self._joint(self.net, ev, ent, typ, mask, self.c_am, o, pick)
        sa, sp = shares(ja), shares(jp)
        ma = shares(self._joint(self.ref, er, ent, typ, mask, self.c_mv, o, pick))  # (and the move order's points)
        mp = shares(self._joint(self.net, ev, ent, typ, mask, self.c_mv, o, pick))

        def probs(p):  # [o, C] -> [o, 6]: none, attack-move, move, attack a unit, smart, stop/hold/patrol
            return np.stack([p[:, 0], p[:, self.c_am], p[:, self.c_mv], p[:, self.c_atk], p[:, self.c_smart].sum(-1),
                             p[:, self.c_hold].sum(-1)], -1)
        qa, qp = probs(pa), probs(pp)
        # (an attack on a creep in view: the attack order's pointer on the entities that are creeps)
        E = ent.shape[1]
        is_creep = (sel[:E, fx.C_OWNER] == 12) & (sel[:E, fx.C_FLAGS] & fx.STRUCTURE == 0)
        if fx.A_OBS >= 0:
            is_creep &= np.asarray(st["ent"])[:E, fx.A_OBS:fx.A_OBS + 3].sum(-1) < 0.5
        ca = cp = np.zeros(o)
        if is_creep.any():
            forced = torch.full_like(order, self.c_atk)
            m_ = torch.from_numpy(is_creep)
            ca = (torch.softmax(self.ref.target_logits(er["g"], er["u"], mask, forced, ent, typ)[0].float(), -1)[0, :o]
                  * m_).sum(-1).double().numpy()
            cp = (torch.softmax(self.net.target_logits(ev["g"], ev["u"], mask, forced, ent, typ)[0].float(), -1)[0, :o]
                  * m_).sum(-1).double().numpy()
        ratio = note["mat"][0] / max(note["mat"][1], 1.0)
        ahead = 1 if ratio >= 2.0 else "h" if ratio >= 1.5 else 0  # (far ahead; ahead by half; neither)
        seen = note["foe_army_seen"] > 0
        at = {int(u): k for k, u in enumerate(pick)}
        for table, on, tag in ((self.late, late, (ahead, int(seen))), (self.mid, mid, ())):
            if not on:
                continue
            table[("steps",) + tag][0] += 1
            for c, units in ((1, army), (2, hero)):
                for i in units.tolist():
                    v = table[tag + (c, int(where[i]))]
                    v[0] += 1
                    v[1:7] += qa[i]
                    v[7:13] += qp[i]
                    v[13] += qa[i, 3] * ca[i]  # (an attack on a creep in view)
                    v[14] += qp[i, 3] * cp[i]
                    k = at.get(i)
                    if k is not None:
                        v[15] += 1
                        v[16] += qa[i, 1]  # (the attack-move's chance of the units whose points were read)
                        v[17] += qp[i, 1]
                        v[18:24] += qa[i, 1] * sa[k]  # (weighted by the chance of the order: the attack-moves' places)
                        v[24:30] += qp[i, 1] * sp[k]
                        v[30:36] += sa[k]  # (unweighted: where an attack-move would point, given one)
                        v[36:42] += sp[k]
                        v[42] += qa[i, 2]  # (the move's chance, and its places weighted by it)
                        v[43] += qp[i, 2]
                        v[44:50] += qa[i, 2] * ma[k]
                        v[50:56] += qp[i, 2] * mp[k]
                        v[56] += 1
        if mid:
            self._expansion(st, note, ev, er, ent, typ, mask, pa, pp, fl, o, grid, own_start, self.mid, None)

    def _expansion(self, st, note, ev, er, ent, typ, mask, pa, pp, fl, o, grid, own_start, table, tag) -> None:
        """The expansion orders' chance (a worker's hall; undead's haunted mine), and where the hall would stand; into
        `table` (2b's states: self.mid; every evaluated step by phase `tag`: self.exp)."""
        fx = self.fx
        race = note["race"]
        workers = np.flatnonzero(fl & fx.WORKER > 0)
        c_h = self.c_hall.get(race)
        v = table[("expand", race) + (() if tag is None else (tag,))]
        v[0] += 1
        if c_h is None or not len(workers):
            return
        av = st.get("avail")
        ok = bool(av is None or av[c_h]) and bool(pa[workers, c_h].sum() + pp[workers, c_h].sum() > 0)
        v[1] += ok
        gold, lumber = 1000.0 * float(st["glob"][0]), 1000.0 * float(st["glob"][1])
        v[16] += gold
        v[17] += lumber
        costs = st.get("costs")
        if costs is not None:  # (the bank covers the hall's cost: the mask also keeps reserves and the tech)
            v[18] += bool(gold >= float(costs[c_h][0]) and lumber >= float(costs[c_h][1]))
        v[2] += pa[workers, c_h].sum()  # (the hall orders expected this step)
        v[3] += pp[workers, c_h].sum()
        v[4] += 1.0 - np.prod(1.0 - pa[workers, c_h])  # (the chance of one or more)
        v[5] += 1.0 - np.prod(1.0 - pp[workers, c_h])
        if self.c_ugol is not None and race == "undead":
            ok_u = bool(av is None or av[self.c_ugol]) and bool(pa[workers, self.c_ugol].sum() + pp[workers, self.c_ugol].sum() > 0)
            v[6] += ok_u
            v[7] += pa[workers, self.c_ugol].sum()
            v[8] += pp[workers, self.c_ugol].sum()
        if ok and float(pa[workers, c_h].sum() + pp[workers, c_h].sum()) > 1e-4:  # (the likeliest builder's place for
            # it: at home, at another mine, elsewhere)
            w = workers[np.argmax(pa[workers, c_h] + pp[workers, c_h])]
            pick = np.asarray([w], np.int64)
            ja = self._joint(self.ref, er, ent, typ, mask, c_h, o, pick)
            jp = self._joint(self.net, ev, ent, typ, mask, c_h, o, pick)
            mines = note["mines"]
            far = mines[np.hypot(mines[:, 0] - own_start[0], mines[:, 1] - own_start[1]) > 2000.0] if len(mines) else mines
            mine_m = self._near(grid, far, NEAR_MINE)
            home_m = self._near(grid, np.asarray([own_start], np.float64), NEAR_BASE)
            v[9] += 1
            v[10] += pa[w, c_h]
            v[11] += pp[w, c_h]
            v[12] += pa[w, c_h] * float((ja[0] * home_m).sum())
            v[13] += pa[w, c_h] * float((ja[0] * mine_m).sum())
            v[14] += pp[w, c_h] * float((jp[0] * home_m).sum())
            v[15] += pp[w, c_h] * float((jp[0] * mine_m).sum())

    def _joint(self, net, e: dict, ent, typ, mask, c: int, o: int, pick: np.ndarray) -> np.ndarray:
        """The point's distribution [K, BINS (x), BINS (y)] of the units `pick`, were their order class `c`."""
        import torch
        O = e["u"].shape[1] if e["u"].shape[1] < self.fx.MAX_OWN else self.fx.MAX_OWN
        forced = torch.full((1, O), int(c), dtype=torch.long)
        _, xl, z = net.target_logits(e["g"], e["u"], mask, forced, ent, typ)
        idx = torch.from_numpy(pick)
        px = torch.softmax(xl[0, idx].float(), -1)  # [K, B]
        K, B = px.shape
        zz = z[0, idx][:, None, :].expand(K, B, z.shape[-1]).reshape(1, K * B, -1)
        xs = torch.arange(B).repeat(K)[None]
        py = torch.softmax(net.y_logits(zz, xs).float(), -1).reshape(K, B, B)
        return (px[:, :, None] * py).double().numpy()

    def _grid(self, sign: float) -> tuple:
        """The bins' world places in a player's view: (x of each x bin, y of each y bin, the cells within NEAR_BASE of
        its own start, of the enemy's, outside the map's camera bounds)."""
        g = self.grids.get(sign)
        if g is None:
            fx = self.fx
            B = fx.BINS
            wx = np.asarray([fx.bins_world(b, 0, sign)[0] for b in range(B)])
            wy = np.asarray([fx.bins_world(0, b, sign)[1] for b in range(B)])
            (ox, oy), (ex, ey) = fx.own_start(sign), fx.own_start(-sign)
            X, Y = wx[:, None], wy[None, :]
            own = (np.hypot(X - ox, Y - oy) <= NEAR_BASE).astype(np.float64)
            foe = (np.hypot(X - ex, Y - ey) <= NEAR_BASE).astype(np.float64)
            if self.bounds is not None:
                x0, y0, x1, y1 = self.bounds
                outside = ((X < x0) | (X > x1) | (Y < y0) | (Y > y1)).astype(np.float64)
            else:
                outside = np.zeros((B, B))
            g = self.grids[sign] = (wx, wy, own, foe, outside)
        return g

    @staticmethod
    def _near(grid: tuple, pts: np.ndarray, r: float) -> np.ndarray:
        """The cells within r of any of the points [M, 2]."""
        wx, wy = grid[0], grid[1]
        m = np.zeros((len(wx), len(wy)))
        for px, py in np.asarray(pts, np.float64).reshape(-1, 2):
            ix, iy = np.flatnonzero(np.abs(wx - px) <= r), np.flatnonzero(np.abs(wy - py) <= r)
            if len(ix) and len(iy):
                sub = np.hypot(wx[ix][:, None] - px, wy[iy][None, :] - py) <= r
                m[np.ix_(ix, iy)] = np.maximum(m[np.ix_(ix, iy)], sub)
        return m

    def save(self, path: Path, meta: dict) -> None:
        c = self.cells
        n = len(c.at)
        keys = np.zeros((n, 3), np.int64)
        for key, i in c.at.items():
            keys[i] = key
        tab = lambda d: json.dumps({"|".join(map(str, k)): v.tolist() for k, v in d.items()})  # noqa: E731
        meta = {**meta, "evaluated": self.evaluated, "steps": self.k, "unit_steps": self.unit_steps,
                "check_steps": self.checked, "check_max_diff": self.check_max, "mask_differs": self.mask_differs,
                "kl_over_50": self.big, "eval_ms": round(self.eval_ms), "material_by_minute": self.mat_minute,
                "every": self.every, "points": self.points,
                "late_steps": {"|".join(map(str, k[1:])): int(v[0]) for k, v in self.late.items() if k[0] == "steps"},
                "ahead_half": True}  # (the late states' "ahead by half" kept apart: the games since)
        np.savez_compressed(path, keys=keys, n=c.n[:n], kl=c.kl[:n], A=c.A[:n].astype(np.float32),
                            P=c.P[:n].astype(np.float32), T=c.T[:n].astype(np.float32), labels=np.asarray(self.labels),
                            minute_kl=self.minute_kl, pos_kl=self.pos_kl, late=tab(self.late), mid=tab(self.mid),
                            pts=tab(self.pts), exp=tab(self.exp), meta=json.dumps(meta, default=str))


def install(p: Probe, races: list[str]) -> None:
    from warcraftsim.fullgame.play import BCAgent
    b_observe = BCAgent.observe

    def observe(self, obs, t, rows=None):
        st = b_observe(self, obs, t, rows)
        self._race = races[self.player]
        p.note(self, obs, t, st)
        return st
    BCAgent.observe = observe


# ---------------------------------------------------------------- the report
def report(paths: list[Path]) -> int:
    files = [np.load(p, allow_pickle=False) for p in paths]
    labels = [str(x) for x in files[0]["labels"]]
    C = len(labels)
    groups = [group_of(l) for l in labels]
    cells: dict[tuple, list] = {}
    late, mid = defaultdict(lambda: np.zeros(W2)), defaultdict(lambda: np.zeros(W2))
    pts, exp = defaultdict(lambda: np.zeros(20)), defaultdict(lambda: np.zeros(W2))
    minute_kl, pos_kl = np.zeros((41, 5)), np.zeros((SEQ, 2))
    with_pts, tot_pts = 0, 0.0  # (the games with 1b's sums, and their KL's total)
    halves = 0  # (the games that keep the late states' "ahead by half" apart)
    print("== the games")
    for p, z in zip(paths, files):
        m = json.loads(str(z["meta"]))
        print(f"  {Path(p).name}: {'/'.join(m['races'])} seed {m['seed']}: {m['game_minutes']:.1f} min, outcome (side 0) "
              f"{m['outcome']:+.0f}, evaluated {sum(m['evaluated'].values())} of {sum(m['steps'].values())} side-steps, "
              f"{m['unit_steps']} unit-steps, mean KL {float(z['kl'].sum()) / max(int(z['n'].sum()), 1):.3f}; "
              f"expansions {m.get('expansions')} creeps killed {m.get('creeps')}; late side-steps far ahead with no enemy "
              f"army in view: {(m.get('late_steps') or {}).get('1|0', 0)}; check: {m['check_steps']} steps, max "
              f"difference {m['check_max_diff']:.1e}; masks differ {m['mask_differs']}, KL > 50: {m['kl_over_50']}")
        mm = m.get("material_by_minute") or {}
        print("     material (side 0 / side 1) at minute " + ", ".join(
            f"{k}: {mm[str(k)][0]}/{mm[str(k)][1]}" for k in (5, 10, 15, 20, 25, 30, 35, 39) if str(k) in mm))
        with_pts += "pts" in z.files
        halves += bool(m.get("ahead_half"))
        tot_pts += float(z["kl"].sum()) if "pts" in z.files else 0.0
        for name, tab in (("pts", pts), ("exp", exp)):
            for k, v in (json.loads(str(z[name])) if name in z.files else {}).items():
                tab[tuple(k.split("|"))][:len(v)] += np.asarray(v)
        for i, key in enumerate(z["keys"].tolist()):
            c = cells.setdefault(tuple(key), [0, np.zeros(4), np.zeros(C), np.zeros(C), np.zeros(C)])
            c[0] += int(z["n"][i])
            c[1] += z["kl"][i]
            c[2] += z["A"][i]
            c[3] += z["P"][i]
            c[4] += z["T"][i]
        for name, tab in (("late", late), ("mid", mid)):
            for k, v in json.loads(str(z[name])).items():
                tab[tuple(k.split("|"))][:len(v)] += np.asarray(v)  # (a file from before the moves' places: narrower)
        minute_kl += z["minute_kl"]
        pos_kl += z["pos_kl"]
    N = sum(c[0] for c in cells.values())
    KL = sum(c[1] for c in cells.values())
    tot = float(KL.sum())
    print(f"\n== 1. where the KL is: {N} unit-steps, mean KL a unit-step {tot / N:.4f} (the run's ref_kl: ~0.13 over its "
          f"training games); by head: " + ", ".join(f"{h} {KL[k] / N:.4f} ({100 * KL[k] / tot:.0f}%)" for k, h in enumerate(HEADS)))
    print("   the share of the total KL (%), [the share of the unit-steps %], (mean KL a unit-step) by phase and class")
    print("   " + " " * 10 + "".join(f"{c:>26}" for c in CLASSES) + f"{'all':>26}")
    for ph in range(4):
        line = f"   {PHASES[ph]:<10}"
        for cl in list(range(4)) + [None]:
            sel = [c for k, c in cells.items() if k[0] == ph and (cl is None or k[1] == cl)]
            n, s = sum(c[0] for c in sel), sum(float(c[1].sum()) for c in sel)
            line += f"{100 * s / tot:8.1f} [{100 * n / N:4.1f}] ({s / max(n, 1):.3f})".rjust(26)
        print(line)
    line = f"   {'all':<10}"
    for cl in list(range(4)) + [None]:
        sel = [c for k, c in cells.items() if cl is None or k[1] == cl]
        n, s = sum(c[0] for c in sel), sum(float(c[1].sum()) for c in sel)
        line += f"{100 * s / tot:8.1f} [{100 * n / N:4.1f}] ({s / max(n, 1):.3f})".rjust(26)
    print(line)
    print("   by head and class (% of the total KL):")
    for k, h in enumerate(HEADS):
        print(f"   {h:<10}" + "".join(
            f"{100 * sum(float(c[1][k]) for key, c in cells.items() if key[1] == cl) / tot:10.1f}" for cl in range(4)))
    print("   by the anchor's top order (its kind): % of the total KL [% of the unit-steps] (mean KL), and by head "
          "(order / pointer / x / y, % of the total), by class (worker / army / hero / building, % of the total)")
    by_g: dict[str, list] = {}
    for (ph, cl, top), c in cells.items():
        g = by_g.setdefault(groups[top], [0, np.zeros(4), np.zeros(4)])
        g[0] += c[0]
        g[1] += c[1]
        g[2][cl] += float(c[1].sum())
    for g, (n, k4, c4) in sorted(by_g.items(), key=lambda kv: -kv[1][1].sum()):
        print(f"   {g:<22}{100 * k4.sum() / tot:6.1f} [{100 * n / N:5.1f}] ({k4.sum() / max(n, 1):.3f})   heads "
              + " / ".join(f"{100 * v / tot:4.1f}" for v in k4) + "   classes " + " / ".join(f"{100 * v / tot:4.1f}" for v in c4))
    print("   by game minute (mean KL a unit-step): " + " ".join(
        f"{m}:{minute_kl[m, :4].sum() / minute_kl[m, 4]:.2f}" for m in range(0, 41, 2) if minute_kl[m, 4] > 0))
    print("   by the step's place in its sequence of 16 (mean KL a unit-step; place 0: both memories the actor's): "
          + " ".join(f"{pos_kl[k, 0] / max(pos_kl[k, 1], 1):.3f}" for k in range(SEQ)))

    # ---- 1b
    print(f"\n== 1b. the point heads ({with_pts} of the {len(files)} games): the army units and heroes whose own order of "
          "the step is an attack-move or a move (the x and y terms' units): where the anchor would point that order / "
          "where the policy does (the point's whole distribution): within 1,500 of its own start, of the enemy's start, "
          "elsewhere [within 1,000 of a living enemy building; outside the map; within 600 of a living creep]")
    for phases, pname in ((("0", "1", "2", "3"), "all phases"), (("0", "1"), "minutes 0-15"), (("2", "3"), "minutes 15+")):
        for kind, kname in (("0", "attack-move"), ("1", "move")):
            for cl, cname in (("1", "army"), ("2", "hero")):
                rows = {w: sum((pts[(ph, cl, kind, w)] for ph in phases if (ph, cl, kind, w) in pts), np.zeros(20))
                        for w in ("0", "1", "2")}
                rows["all"] = sum(rows.values(), np.zeros(20))
                if rows["all"][0] < 1:
                    continue
                for w, wname in (("0", "the unit at home"), ("1", "at the enemy's base"), ("2", "elsewhere"), ("all", "ALL")):
                    v = rows[w]
                    if v[0] < 1:
                        continue
                    n = v[0]
                    a, p_ = v[2:8] / n, v[8:14] / n
                    print(f"   {pname:<13}{kname:<12}{cname:<5}{wname:<20} n={int(n):5d} x+y KL {v[1] / n:5.2f} ({100 * v[1] / max(tot_pts, 1e-9):4.1f}% of "
                          f"their games' KL)  own base {a[0]:.2f}/{p_[0]:.2f}  enemy base {a[1]:.2f}/{p_[1]:.2f}  elsewhere "
                          f"{a[2]:.2f}/{p_[2]:.2f}  [enemy building {a[3]:.2f}/{p_[3]:.2f}; outside {a[4]:.2f}/{p_[4]:.2f}; "
                          f"creep {a[5]:.2f}/{p_[5]:.2f}]  the step's own point: home {v[14] / n:.2f} enemy base {v[15] / n:.2f} "
                          f"elsewhere {v[16] / n:.2f}")

    # ---- 2a
    names6 = ("none", "attack-move", "move", "attack unit", "smart", "stop/hold")
    print("\n== 2a. late states (minute 15+): the army units' orders, the anchor's chance / the policy's, a unit-step")
    print("   where an attack-move points: the share of the attack-moves (weighted by their chance) to within 1,500 of "
          "its own start / of the enemy's start / elsewhere; [to within 1,000 of a living enemy building; outside the map]")
    for ahead, seen, title in (("1", "0", "FAR AHEAD (>= 2x material), no enemy army unit in view"),
                               ("1", "1", "far ahead, an enemy army unit in view"),
                               ("h", "0", f"ahead 1.5-2x, no enemy army unit in view ({halves} of the games: the later ones)"),
                               ("h", "1", "ahead 1.5-2x, an enemy army unit in view (the same games)"),
                               ("0", "0", "not ahead (under 1.5x; the earlier games: under 2x), no enemy army unit in view"),
                               ("0", "1", "not ahead, an enemy army unit in view")):
        steps = late.get(("steps", ahead, seen))
        print(f"   -- {title}: {int(steps[0]) if steps is not None else 0} side-steps")
        for cl, cname in (("1", "army"), ("2", "hero")):
            rows = [(w, late.get((ahead, seen, cl, w))) for w in ("0", "1", "2")]
            allv = sum((v for _, v in rows if v is not None), np.zeros(W2))
            for wname, v in [("at home", rows[0][1]), ("at the enemy's base", rows[1][1]), ("elsewhere", rows[2][1]),
                             ("ALL", allv)]:
                if v is None or v[0] < 1:
                    continue
                n = v[0]
                line = f"      {cname:<5}{wname:<20} n={int(n):6d}  " + "  ".join(
                    f"{names6[k]} {v[1 + k] / n:.3f}/{v[7 + k] / n:.3f}" for k in (0, 1, 2, 3))
                if v[15] > 0:
                    a, p_ = v[18:24] / max(v[16], 1e-9), v[24:30] / max(v[17], 1e-9)
                    line += (f"\n           attack-moves to: own base {a[0]:.2f}/{p_[0]:.2f}  enemy base {a[1]:.2f}/{p_[1]:.2f}  "
                             f"elsewhere {a[2]:.2f}/{p_[2]:.2f}  [enemy building {a[3]:.2f}/{p_[3]:.2f}; outside "
                             f"{a[4]:.2f}/{p_[4]:.2f}]  (points read for {int(v[15])} unit-steps); attack-moves home a "
                             f"unit-step: {v[18] / v[15]:.4f}/{v[24] / v[15]:.4f}, to the enemy's base: "
                             f"{v[19] / v[15]:.4f}/{v[25] / v[15]:.4f}")
                if v[56] > 0:  # (the games since the moves' places were read)
                    a, p_ = v[44:50] / max(v[42], 1e-9), v[50:56] / max(v[43], 1e-9)
                    line += (f"\n           moves to: own base {a[0]:.2f}/{p_[0]:.2f}  enemy base {a[1]:.2f}/{p_[1]:.2f}  "
                             f"elsewhere {a[2]:.2f}/{p_[2]:.2f}  [enemy building {a[3]:.2f}/{p_[3]:.2f}; outside "
                             f"{a[4]:.2f}/{p_[4]:.2f}]  (for {int(v[56])} unit-steps: move {v[42] / v[56]:.3f}/{v[43] / v[56]:.3f}); "
                             f"moves home a unit-step: {v[44] / v[56]:.4f}/{v[50] / v[56]:.4f}, to the enemy's base: "
                             f"{v[45] / v[56]:.4f}/{v[51] / v[56]:.4f}")
                print(line)
    # ---- 2b
    steps = mid.get(("steps",))
    print(f"\n== 2b. minutes 4-12 with 6+ army units and a hero: {int(steps[0]) if steps is not None else 0} side-steps; "
          "the anchor's chance / the policy's, a unit-step")
    for cl, cname in (("1", "army"), ("2", "hero")):
        v = sum((mid[(cl, w)] for w in ("0", "1", "2") if (cl, w) in mid), np.zeros(W2))
        if v[0] < 1:
            continue
        n = v[0]
        a, p_ = v[18:24] / max(v[16], 1e-9), v[24:30] / max(v[17], 1e-9)
        print(f"   {cname:<5} n={int(n):6d}  " + "  ".join(f"{names6[k]} {v[1 + k] / n:.3f}/{v[7 + k] / n:.3f}" for k in (0, 1, 2, 3))
              + f"\n         an attack on a creep in view (attack unit x its pointer on a creep): {v[13] / n:.4f}/{v[14] / n:.4f}"
              + f"\n         attack-moves to: a cell within 600 of a living creep {a[5]:.2f}/{p_[5]:.2f}  own base "
                f"{a[0]:.2f}/{p_[0]:.2f}  enemy base {a[1]:.2f}/{p_[1]:.2f}  (points read for {int(v[15])} unit-steps)"
              + f"\n         an attack-move toward a creep, a unit-step: {v[23] / max(v[15], 1):.4f}/{v[29] / max(v[15], 1):.4f}"
              + (f"; a move toward a creep: {v[49] / v[56]:.4f}/{v[55] / v[56]:.4f}; moves home a unit-step: "
                 f"{v[44] / v[56]:.4f}/{v[50] / v[56]:.4f}" if v[56] > 0 else ""))
    for race in HALLS:
        v = mid.get(("expand", race))
        if v is None or v[0] < 1:
            continue
        n = v[0]
        line = (f"   expansion, {race}: {int(n)} side-steps, the hall's order open to a worker in {int(v[1])} "
                f"({100 * v[1] / n:.0f}%); hall orders a side-step {v[2] / n:.5f}/{v[3] / n:.5f}; on the open steps "
                f"{v[2] / max(v[1], 1):.5f}/{v[3] / max(v[1], 1):.5f}; one or more: {v[4] / max(v[1], 1):.5f}/{v[5] / max(v[1], 1):.5f}")
        if v[9] > 0:
            line += (f"; the likeliest builder's hall: at home {v[12] / max(v[10], 1e-12):.2f}/{v[14] / max(v[11], 1e-12):.2f}, "
                     f"at another mine {v[13] / max(v[10], 1e-12):.2f}/{v[15] / max(v[11], 1e-12):.2f}")
        if race == "undead":
            line += (f"; haunt a mine: open in {int(v[6])} side-steps, orders a side-step {v[7] / n:.5f}/{v[8] / n:.5f}")
        print(line)
    print("   the hall's order on EVERY evaluated step, by race and phase: side-steps; the bank covers its cost; the mask "
          "leaves it open; hall orders a side-step on the open steps, the anchor's / the policy's; the bank's mean gold")
    for race in HALLS:
        for ph in range(4):
            v = exp.get(("expand", race, str(ph)))
            if v is None or v[0] < 1:
                continue
            n = v[0]
            line = (f"      {race:<9}{PHASES[ph]:<6} {int(n):5d} side-steps; bank covers {100 * v[18] / n:4.1f}%; open "
                    f"{100 * v[1] / n:4.1f}%; hall orders on the open steps {v[2] / max(v[1], 1):.4f}/{v[3] / max(v[1], 1):.4f} "
                    f"(one or more {v[4] / max(v[1], 1):.4f}/{v[5] / max(v[1], 1):.4f}); mean gold {v[16] / n:.0f}")
            if v[9] > 0:
                line += (f"; its place: at home {v[12] / max(v[10], 1e-12):.2f}/{v[14] / max(v[11], 1e-12):.2f}, at another mine "
                         f"{v[13] / max(v[10], 1e-12):.2f}/{v[15] / max(v[11], 1e-12):.2f}")
            if race == "undead":
                line += f"; haunt a mine: open {100 * v[6] / n:.1f}%, orders a side-step {v[7] / n:.5f}/{v[8] / n:.5f}"
            print(line)
    # the same orders over all the evaluated steps, by phase (from 1's sums)
    print("   over ALL evaluated steps, by phase: the anchor's / the policy's chance a unit-step")
    for name, cl, pred in (("a worker's hall order", 0, lambda l: l.startswith("build ") and l[6:] in HALLS.values()),
                           ("a worker's haunt-a-mine order", 0, lambda l: l == "ugol"),
                           ("an army unit's attack-move", 1, lambda l: l == "attack (point)"),
                           ("an army unit's move", 1, lambda l: l == "move (point)"),
                           ("an army unit's no order", 1, lambda l: l == "none"),
                           ("a hero's attack-move", 2, lambda l: l == "attack (point)"),
                           ("a hero's attack on a unit", 2, lambda l: l == "attack (unit)")):
        cols = [i for i, l in enumerate(labels) if pred(l)]
        line = f"      {name:<30}"
        for ph in range(4):
            sel = [c for k, c in cells.items() if k[0] == ph and k[1] == cl]
            n = max(sum(c[0] for c in sel), 1)
            line += f"  {PHASES[ph]}: {sum(float(c[2][cols].sum()) for c in sel) / n:.5f}/{sum(float(c[3][cols].sum()) for c in sel) / n:.5f}"
        print(line)

    # ---- 3
    print("\n== 3. the pull's direction: the top cells (the anchor's top order, class, phase) by KL")
    print("   cell: % of the total KL (order head's part), n; the anchor's chance of its top order A -> the policy's; the "
          "order the policy takes most there B: the policy's chance <- the anchor's")
    ranked = sorted(cells.items(), key=lambda kv: -float(kv[1][1].sum()))
    for (ph, cl, top), c in ranked[:14]:
        n = c[0]
        b = int(np.argmax(c[3]))
        more = "the same order" if b == top else ("anchor LESS on B" if c[2][b] < c[3][b] else "anchor MORE on B")
        terms = "; ".join(f"{labels[i]} {100 * c[4][i] / max(c[1][0], 1e-12):.0f}% ({c[2][i] / n:.3f} -> {c[3][i] / n:.3f})"
                          for i in np.argsort(-c[4])[:3])
        print(f"   {labels[top]:<22} {CLASSES[cl]:<8} {PHASES[ph]:<6} {100 * c[1].sum() / tot:5.1f}% ({100 * c[1][0] / tot:4.1f}) "
              f"n={n:6d}  A: anchor {c[2][top] / n:.2f} -> policy {c[3][top] / n:.2f} | B = {labels[b]}: policy {c[3][b] / n:.2f} "
              f"<- anchor {c[2][b] / n:.2f}  [{more}]\n        the order KL's largest terms (the order: its share of the "
              f"cell's order KL, the anchor's chance -> the policy's): {terms}")
    print("   the same by the order's kind (the anchor's top order's kind, class, phase):")
    gc: dict[tuple, list] = {}
    for (ph, cl, top), c in cells.items():
        g = gc.setdefault((groups[top], cl, ph), [0, np.zeros(4), np.zeros(C), np.zeros(C)])
        g[0] += c[0]
        g[1] += c[1]
        g[2] += c[2]
        g[3] += c[3]
    gi: dict[str, list] = defaultdict(list)
    for i, g in enumerate(groups):
        gi[g].append(i)
    for (g, cl, ph), c in sorted(gc.items(), key=lambda kv: -float(kv[1][1].sum()))[:12]:
        n = c[0]
        pa = {k: float(c[2][v].sum()) / n for k, v in gi.items()}
        pp = {k: float(c[3][v].sum()) / n for k, v in gi.items()}
        b = max(pp, key=pp.get)
        more = "the same kind" if b == g else ("anchor LESS on B" if pa[b] < pp[b] else "anchor MORE on B")
        print(f"   {g:<22} {CLASSES[cl]:<8} {PHASES[ph]:<6} {100 * c[1].sum() / tot:5.1f}% ({100 * c[1][0] / tot:4.1f}) n={n:6d}  "
              f"A: anchor {pa[g]:.2f} -> policy {pp[g]:.2f} | B = {b}: policy {pp[b]:.2f} <- anchor {pa[b]:.2f}  [{more}]")
    print("   the net pull on the orders (the forward KL's gradient on an order's logit is policy - anchor): orders a "
          "1,000 unit-steps the anchor gives MORE than the policy (+) or FEWER (-), by class; [the order KL's terms' share]")
    for cl in range(4):
        sel = [c for k, c in cells.items() if k[1] == cl]
        n = max(sum(c[0] for c in sel), 1)
        A, P, T = sum(c[2] for c in sel), sum(c[3] for c in sel), sum(c[4] for c in sel)
        d = {g: (1000 * float((A - P)[v].sum()) / n, 1000 * float(A[v].sum()) / n, 1000 * float(P[v].sum()) / n,
                 float(T[v].sum())) for g, v in gi.items()}
        kl_o = sum(float(c[1][0]) for c in sel)
        print(f"   {CLASSES[cl]} ({n} unit-steps, order KL {kl_o / n:.4f}): " + "; ".join(
            f"{g} {v[0]:+.1f} ({v[1]:.1f} vs {v[2]:.1f}) [{100 * v[3] / max(kl_o, 1e-9):.0f}%]"
            for g, v in sorted(d.items(), key=lambda kv: -abs(kv[1][0]))[:8]))
        idx = np.argsort(-np.abs(A - P))[:6]
        print("        single orders: " + "; ".join(
            f"{labels[i]} {1000 * (A - P)[i] / n:+.1f} ({1000 * A[i] / n:.1f} vs {1000 * P[i] / n:.1f})" for i in idx))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", nargs="+", type=Path, help="no game: the tables from these games' files")
    ap.add_argument("--weights", type=Path, help="the policy (it plays both sides)")
    ap.add_argument("--anchor", type=Path, default=None, help="the anchor (default: the run's --init)")
    ap.add_argument("--run", type=Path, help="the run whose settings the game is played with (run.json, vocab.json)")
    ap.add_argument("--races", nargs=2, default=["orc", "orc"])
    ap.add_argument("--steps", type=int, default=4800)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--every", type=int, default=4, help="one sequence of 16 steps in this many is evaluated")
    ap.add_argument("--points", type=int, default=12, help="army units a step whose attack-move points are read")
    ap.add_argument("--set", nargs="*", default=[], help="settings changed for this game: key=value")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--name", default="anchorprobe", help="the game instance's name")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--shim", type=Path, default=None)
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    if a.report:
        return report(a.report)
    if a.weights is None or a.run is None or a.out is None:
        ap.error("--weights, --run and --out (or --report FILES)")
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    import torch
    torch.set_num_threads(a.threads)
    import ghoul_probe as gp
    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame import selfplay as sp
    from warcraftsim.fullgame import strategy as strat
    from warcraftsim.fullgame.model import load
    from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup, exact_cuts
    run_json = json.loads((a.run / "run.json").read_text())["config"]
    vocab = json.loads((a.run / "vocab.json").read_text())
    over = dict(kv.split("=", 1) for kv in a.set)
    over["z_share"] = 0.0
    over["max_steps"] = a.steps
    cfg = gp.run_cfg(a.run, vocab, over)
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
    bounds = fx.map_bounds(stock_map_path(cfg["map"]))
    if cfg["order_masks"] & fx.EDGE_ORDERS:
        fx.set_bounds(bounds)
    net, _ = load(a.weights, "cpu", vocab=vocab)
    anchor = a.anchor or Path(run_json["init"])
    if not anchor.is_absolute():
        anchor = a.run.resolve().parent.parent / anchor
    ref, ck_ref = load(anchor, "cpu", True, vocab=vocab)  # (selfplay.main: the run's memory setting, its vocabulary)
    if ck_ref["vocab"]["orders"] != vocab["orders"]:
        raise SystemExit("the anchor's orders are not the run's")
    ref.allowed.copy_(net.allowed)  # (ppo_update: the anchor's allowed orders are the learner's)
    fx.set_z(net.config.get("z", 0))
    for n_ in (net, ref):
        n_.eval()
        for p_ in n_.parameters():
            p_.requires_grad_(False)
    print(f"policy {a.weights} ({net.config['G']} global numbers, omask {net.omask}); anchor {anchor} "
          f"({ref.config['G']}, omask {ref.omask}, memory {ref.memory}); bins {fx.BINS}; KL mode "
          f"{run_json.get('ref_kl_mode')} x{run_json.get('ref_kl')}", flush=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    races = list(a.races)
    p = Probe(net, ref, vocab, cfg, a.seed, a.every, a.points, bounds)
    install(p, races)
    slots = [Agent(r, handicap=cfg["handicap"], difficulty="normal") for r in races]
    setup = GameSetup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"], max_game_seconds=cfg["max_minutes"] * 60,
                      victory="decisive", window=(320, 240), draw=False, wait_floor_ms=cfg["wait_floor_ms"],
                      native_obs=True, nice=19, d3d_thread=False, render_threads=0, warm_spare=False,
                      cuts=exact_cuts(cfg.get("game_cuts")) or None, obs_full=bool(cfg.get("obs_full")),
                      **({"shim_dir": a.shim} if a.shim else {}))
    opp = {"name": "self", "kind": "self"}
    t0 = time.time()
    with GameInstance(setup, name=a.name, timeout=600) as g:
        obs = g.start()
        ep = sp.play_one(g, obs, cfg, vocab, races, 0, opp, p, p, 0, record=False, train=False)
    meta = {"races": races, "seed": a.seed, "weights": str(a.weights), "anchor": str(anchor), "run": str(a.run),
            "game_minutes": ep["game_time"] / 60.0, "outcome": ep["outcome"], "length": ep["length"],
            "wall_seconds": round(time.time() - t0), "material_lead": ep.get("material_lead"),
            "expansions": [ep["prod"].get("expansions"), ep["opp_prod"].get("expansions")],
            "creeps": [ep["prod"].get("creeps"), ep["opp_prod"].get("creeps")],
            "kills": [ep["prod"].get("kills"), ep["opp_prod"].get("kills")],
            "attacks": ep.get("attacks"), "left": ep.get("left")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    p.save(a.out, meta)
    print(f"GAME {'/'.join(races)} seed {a.seed}: {meta['game_minutes']:.1f} min, outcome {ep['outcome']}, wall "
          f"{meta['wall_seconds']} s (the evaluation {p.eval_ms / 1000:.0f} s), evaluated {p.evaluated} of {p.k} side-steps, "
          f"{p.unit_steps} unit-steps; check max difference {p.check_max:.2e}; masks differ {p.mask_differs}; "
          f"expansions {meta['expansions']} creeps {meta['creeps']} -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
