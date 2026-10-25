"""Play a behavior-cloned whole-game policy (fullgame/bc.py) against the built-in AI.

    python3 -m warcraftsim.fullgame.play runs/bc/fullgame-1/policy.pt --games 20 --race human --ai-race orc

(the torch Python). The agent takes a computer slot without an AI: every order its units get
comes from the policy. Each step it sees what the demonstrations' players saw (fullgame/features:
its own units, the enemy's and neutral units it can see, resources, supply, time, races,
upgrades), and each own unit gets the order the policy samples (or none): an order at once
(train, research, stop, ...), at a point (move, build: the order is the building's type), on a unit
(attack, harvest, repair, ...), on a tree (harvest: the tree nearest the chosen point) or a hero
skill to learn.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
from collections import Counter
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..protocol import (Build, Buy, Command, EventKind, ImmediateOrder, LearnSkill, Observation, PickUp, PointOrder,
                        TargetDestructable, TargetOrder)
from ..runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup
from . import features as fx
from .collect import Films, series
from .model import load
from .neutral import is_ware, nearest_shop, sells

TYPE_CODE = fx.TYPE_CODE  # orders at or above this are unit / building / upgrade / ability codes


def rawcode_or(order: int) -> str:
    from ..protocol import rawcode
    return rawcode(order) if order and order >= TYPE_CODE else str(order)


def unit_rows(obs: Observation, t: int) -> np.ndarray:
    """The observation's units as rows of collect.UNIT_COLS, then a hero's items and abilities (features.ROW_COLS:
    its 6 items' types, per ability slot its level and cooldown left in 0.1 s; zeros for others)."""
    ua = getattr(obs, "unit_array", None)
    if ua is not None:  # (native observations: the table's first 31 columns are these, after the step)
        out = np.empty((len(ua), fx.ROW_COLS), np.int64)
        out[:, 0] = t
        out[:, 1:] = ua[:, :fx.ROW_COLS - 1]
        return out
    return np.asarray([(t, u.id, u.type_id, u.owner, u.x, u.y, u.facing, u.hp, u.max_hp, u.mana, u.max_mana,
                        u.order, int(u.flags), u.visible_to, u.resource, u.hero_level, u.hero_xp, u.skill_points,
                        *(tuple(u.items) + (0,) * 6)[:6],
                        *(v for lv, cd in (tuple(u.abilities) + ((0, 0.0),) * 4)[:4] for v in (lv, int(round(cd * 10)))))
                       for u in obs.units], np.int64).reshape(-1, fx.ROW_COLS)


REPAIR = 852024  # the order a worker resumes a building with
QUEUE_MAX = 7  # a building's queue: at most this many units and research
LOADED = 32  # (UnitFlags.LOADED: in a burrow or a transport)
RESTORE, RENEW = 852202, 852161  # (undead acolytes' and night elf wisps' repairs: finished buildings only)
# buildings a player needs one of (altars, the upgrade buildings, the shops): a second never sent (each unit samples
# its own order: 2-3 workers had built one in a step; a second altar, war mill or graveyard in 24-52% of short won
# games; audit 8)
ONE_OF = {int.from_bytes(c.encode(), "big") for c in ("halt", "oalt", "uaod", "eate", "hbla", "ofor", "ugrv", "edob",
                                                      "hvlt", "ovln", "utom", "eden", "otto", "usap", "obea")}
ATTACK = 851983
EXPAND, CREEP, MIX, PICKUP = 990001, 990002, 990003, 990004  # (macros.EXPAND, .CREEP, .MIX, .PICKUP)
HAUNT = int.from_bytes(b"ugol", "big")
WISP = int.from_bytes(b"ewsp", "big")
REFUSED_CTX = 40  # (refused orders a game whose context is kept)
MINE_WORKERS, MINE_NEAR = 5, 350.0  # (a haunted or entangled mine's workers at most; how near they work it)
HIDDEN = 16  # (UnitFlags.HIDDEN: a worker inside a gold mine)
MACHINES = {int.from_bytes(c.encode(), "big") for c in ("hgyr", "hmtt", "hrtt", "ocat", "ebal", "umtw")}  # (repairable)
MINED_BY = {"undead": HAUNT, "nightelf": int.from_bytes(b"egol", "big")}  # (their own; humans and orcs: a plain one)
EXPAND_HALLS = {r: int.from_bytes(h.encode(), "big") for r, h in (("human", "htow"), ("orc", "ogre"), ("nightelf", "etol"))}
CANCEL = 851976  # a building's queue: its last unit
ALTARS = [int.from_bytes(c.encode(), "big") for c in ("halt", "oalt", "uaod", "eate")]
ENTANGLE = 852147  # a tree of life's entangle gold mine (tested 2026-10-23: given 840 from the mine, it uproots, walks
# nearer and the mine is entangled ~9 s on)
TREES_OF_LIFE = [int.from_bytes(c.encode(), "big") for c in ("etol", "etoa", "etoe")]
NGOL, EGOL = int.from_bytes(b"ngol", "big"), int.from_bytes(b"egol", "big")
# (as features.isin's sorted arrays: per step)
TREE_CODES, ALTAR_CODES, MINED_CODES = fx.codes(TREES_OF_LIFE), fx.codes(ALTARS), fx.codes(MINED_BY.values())
HARVESTING_CODES = fx.codes(list(fx.HARVESTING) + [fx.SMART])
ENTANGLE_NEAR, ENTANGLE_EVERY = 1200.0, 20  # (a tree this near a mine no one entangled: the order, again after so
# many steps until it is)
TIERS = {int.from_bytes(c.encode(), "big") for c in ("hkee", "hcas", "ostr", "ofrt", "unp1", "unp2", "etoa", "etoe")}
REVIVE = 852039  # an altar's revival of a dead hero: a target order on it (its training order is refused; tested
# 2026-10-22: accepted ~7 s after the death, at 425 x 0.4 gold for a level 1 hero)
BUY_REACH, BUY_ARRIVED = 500.0, 6  # (a buyer this near its shop this many steps: free again, _release_buyers)
SKILL_WAIT_STEPS = 20  # (--macros 3: a skill point unspent this long, 10 s, is spent for the policy)
PICKUP_COMMIT_STEPS = 8  # (a hero sent to pick up an item: 4 s to walk there, macros.PICKUP_NEAR at ~300 a second,
# unless it picks something up before)
BUY_COMMIT_STEPS = 120  # (a unit sent to buy: 60 s to walk there, the harness keeps the purchase pending meanwhile; a
# worker from its start reached the tavern, unseen, in 77 steps)


class BCAgent:
    """The policy playing one player of a live game."""

    def __init__(self, net, vocab: dict, player: int, device, temperature: float = 1.0, order_temperature: float = 1.0,
                 costs=None):
        """`costs` (costs.order_costs): orders the player can't pay for yet are masked."""
        self.net, self.enc, self.player, self.device = net, fx.Encoder(vocab, costs), player, device
        self.temperature = temperature
        self.order_temperature = order_temperature  # sharpens the choice among orders only
        self.orders = [(0, 0)] + [tuple(o) for o in vocab["orders"]]  # class -> (order id, kind)
        self.view = None
        self.trees: dict[int, tuple[int, int]] = {}
        self.issued = 0
        # a worker repairing counts as building too: with the built-in AI advising (collect.Shadow) a
        # builder the AI pulled away is sent back to its building with "repair", and the policy then
        # gave it another order (human buildings started by 2 minutes, finished: 1.2 a game, 4.2 without)
        self.repair_builds = False

    def begin(self, obs: Observation, races: list[int]) -> None:
        rows = unit_rows(obs, 0)
        sign = self.enc.side(rows, self.player)
        self.view = self.enc.view(self.player, sign, races)
        p = obs.players.get(self.player)  # (its start: --macros' expand finds its natural from it)
        self.view.start = (p.start_x, p.start_y) if p is not None and hasattr(p, "start_x") else None
        self.trees = {d.id: (d.x, d.y) for d in (obs.destructables or [])}
        self.h = None  # the memory core's state (FullGameNet.memory)
        self.building_since: dict[int, int] = {}  # worker -> the step it started its building order
        self.buying_since: dict[int, int] = {}  # unit -> the step it was sent to buy (protocol.Buy: on its way)
        self.t = 0

    def accepted(self, cmds: list[Command], results: list[bool]) -> None:
        """The game's answer to the last step's orders: accepted train / research orders queue, and
        accepted orders set what the workers harvest (View.assign)."""
        self.view.record_orders((c.unit, c.order, 0) for c, ok in zip(cmds, results)
                                if ok and isinstance(c, ImmediateOrder))
        # each unit's last order and its target (features.A_LAST), as the recorded orders give them
        self.view.note_orders([(c.unit, getattr(c, "order", getattr(c, "building", 0)), 1, c.x, c.y, 0) if hasattr(c, "x")
                               else (c.unit, c.order, 2, 0, 0, c.target) if hasattr(c, "target")
                               else (c.unit, c.order, 0, 0, 0, 0)
                               for c, ok in zip(cmds, results)
                               if ok and hasattr(c, "unit") and (hasattr(c, "order") or hasattr(c, "building"))],
                              self.t, self.trees)
        for c, ok, res in zip(cmds, results, getattr(self, "sent_resources", [])):
            unit = getattr(c, "unit", None)
            if ok and unit is not None:
                if res is None:
                    self.view.assign.pop(unit, None)
                else:
                    self.view.assign[unit] = res
                    if res == "gold" and hasattr(c, "target"):  # (its mine: another one is a transfer, not redundant)
                        self.view.mine_of[unit] = c.target
        pending = getattr(self.view, "expanding", None)
        if pending is not None and any(not ok and getattr(c, "unit", None) == pending[1]  # (a refused expand: no other
                                       and (getattr(c, "building", None) in EXPAND_HALLS.values() or getattr(c, "order", None) == HAUNT)
                                       for c, ok in zip(cmds, results)):  # blocked for its 60 s; audit 8)
            self.view.expanding = None
        buying = getattr(self, "buying_since", {})
        tally = self.__dict__.setdefault("order_results", {})  # (per kind of order: sent, refused; the episode row)
        ctx = self.__dict__.setdefault("refused_ctx", [])  # (the first refused orders' context: the episode row)
        rows = getattr(self, "rows", None)
        for c, ok in zip(cmds, results):
            o = getattr(c, "order", None) or getattr(c, "building", None) or getattr(c, "item", None) or getattr(c, "ability", 0)
            key = f"{type(c).__name__}:{fx.rawcode(int(o)) if isinstance(o, int) and o >= TYPE_CODE else o}"
            sent, refused = tally.get(key, (0, 0))
            tally[key] = (sent + 1, refused + (not ok))
            if not ok and len(ctx) < REFUSED_CTX and rows is not None:
                ctx.append(self._refusal(c, key, rows))
        for c, ok in zip(cmds, results):  # (a revival the altar took: under way, View.tech_allowed)
            if ok and isinstance(c, TargetOrder) and c.order == REVIVE:
                self.view.__dict__.setdefault("reviving", {})[c.target] = self.t
        picking = getattr(self, "picking_since", {})
        for c, ok in zip(cmds, results):  # (a buy or a pick-up the game refused: the unit is free again)
            if not ok and isinstance(c, Buy):
                buying.pop(c.unit, None)
                getattr(self, "hiring", {}).pop(c.unit, None)  # (its money no longer kept; audit 10)
            elif not ok and isinstance(c, PickUp):
                picking.pop(c.unit, None)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Categorical(logits=logits / self.temperature).sample()

    def observe(self, obs: Observation, t: int, rows: np.ndarray | None = None) -> dict | None:
        """This step's view (features.View.step), or None when the player has no units. `rows`:
        unit_rows(obs, t) when already made (both sides of a self-play game share them)."""
        picking = getattr(self, "picking_since", {})
        rows = unit_rows(obs, t) if rows is None else rows
        self.rows = rows
        for e in obs.events:
            if int(e.kind) == int(EventKind.TREE_DEATH):
                self.trees.pop(e.a, None)
            elif int(e.kind) == int(EventKind.ITEM_PICKUP):
                picking.pop(e.a, None)  # (a hero sent to pick up an item: free again once it holds one)
                if e.c and (rows[:, fx.C_ID] == e.a).any() and int(rows[rows[:, fx.C_ID] == e.a][0, fx.C_OWNER]) == self.player:
                    self.picked_up = getattr(self, "picked_up", 0) + 1  # (an item off the ground: the episode row)
        self.t = t
        p = obs.players.get(self.player)
        me = (np.asarray([t, self.player, p.gold, p.lumber, p.food_used, p.food_cap, p.upkeep, p.gold_gathered,
                          p.lumber_gathered, p.structures, int(p.result)]) if p is not None else None)
        self.me = me
        events = np.asarray([(t, int(e.kind), e.a, e.b, e.c) for e in obs.events], np.int64).reshape(-1, 5)
        q = obs.players.get(1 - self.player)  # (the privileged block: the value only, features.View.priv)
        opp = (np.asarray([t, 1 - self.player, q.gold, q.lumber, q.food_used, q.food_cap, q.upkeep]) if q is not None
               else None)
        self.view.ground = getattr(obs, "ground", ())  # (--macros 2: the items lying on the ground, macros.pickable)
        hiring = getattr(self, "hiring", {})  # (a tavern hero's hire on its way: its money kept, View.hire_class; it
        self.view.hire_class = next((self.enc.order_index.get((oid, fx.UNIT)) for oid in hiring.values()), None)
        self.view.hire_free = bool(set(hiring) & getattr(self, "free_hires", set()))  # (the first hero's: free, its
        # money not kept; its full price had been, audit 11)
        self.view.hiring_heroes = frozenset(hiring.values())  # counts as a hero: the masks' hero count, audit 10)
        st = self.view.step(rows, me, events, t, opp)
        sel = st["sel"][:st["n_own"]]  # the workers busy with a building order (features.building), and since when
        cur = sel[:, fx.C_ORDER]  # (by masks: a unit at a time, numpy's scalars, had been ~1% of an actor's time)
        to_build = (cur >= fx.TYPE_CODE) | ((cur == REPAIR) if self.repair_builds else False)
        busy = set(sel[(sel[:, fx.C_FLAGS] & fx.WORKER > 0) & to_build, fx.C_ID].tolist())
        self.building_since = {u: self.building_since.get(u, t) for u in busy}
        return st if st["n_own"] > 0 else None

    def commands(self, st: dict, order, tgt, bx, by) -> list[Command]:
        """The orders sampled for the view's own units (order class, pointer, x and y bins per unit;
        sequences over the first min(n_own, MAX_OWN) units) as game commands."""
        out: list[Command] = []
        self._release_buyers(st["sel"][:st["n_own"]])
        expanded = False  # (--macros: one expand order a step)
        built: set[int] = set()  # (one building of a kind a step)
        queue: dict[int, int] = {}  # (this step's train and research orders a building)
        me, costs = getattr(self, "me", None), self.enc.costs  # the step's money, spent order by order (each order's
        budget = [int(me[2]), int(me[3])] if me is not None and costs is not None else None  # mask checks it alone:
        # a tier taken, then two builds refused at 212 gold; 32-140 paid orders refused a training game; audit 9)
        if costs is not None and getattr(self.view, "free_hero", False) and self.enc.costs_first is not None:
            costs = self.enc.costs_first  # (the first hero free: the mask's costs, View.step)
        hire = getattr(self.view, "hire_class", None)
        kept = {r[0]: (r[1], r[2]) for r in (getattr(self.view, "reserve", None),) if r is not None}  # (as the mask:
        if hire is not None and costs is not None and not getattr(self.view, "hire_free", False):  # (the target's next
            kept[hire] = (int(costs[hire][0]), int(costs[hire][1]))  # item and a hire on its way keep theirs)
        if budget is not None:
            budget = [budget[0] - sum(v[0] for v in kept.values()), budget[1] - sum(v[1] for v in kept.values())]

        def pay(c: int) -> bool:
            if budget is None or c is None:
                return True
            back = kept.pop(c, (0, 0))  # (a kept item itself: its own money back)
            budget[0], budget[1] = budget[0] + back[0], budget[1] + back[1]
            g, l = int(costs[c][0]), int(costs[c][1])
            # (only for what it costs, as the mask's max(0, ...): a lumber-short reserve put the lumber below 0, and every
            # order was refused, a footman or a worker too: 230 a target game, armies never trained; audit 10)
            if (g > 0 and g > max(0, budget[0])) or (l > 0 and l > max(0, budget[1])):
                self.over_budget = getattr(self, "over_budget", 0) + 1
                return False
            budget[0] -= g
            budget[1] -= l
            return True

        def produce(unit: int, oid: int, c) -> bool:
            """A train, research or tier order at a building: not at a hall upgrading (the game refuses all of it),
            not to a full queue, paid from the step's budget; counted in the step's queue."""
            if unit in frames:  # (a frame trains nothing: hero orders at altar frames refused; audit 11)
                self.frame_orders = getattr(self, "frame_orders", 0) + 1
                return False
            if unit in self.view.own_buildings:
                m = self.view.making.get(unit)
                if m is not None and m[0] == 4:  # (a tier under way: refused, ~1,500 workers a game at a keep; audit 10)
                    self.upgrading_refused = getattr(self, "upgrading_refused", 0) + 1
                    return False
                if self.view.queued.get(unit, 0) + queue.get(unit, 0) >= QUEUE_MAX:  # (a full queue refuses it)
                    self.full_queue = getattr(self, "full_queue", 0) + 1
                    return False
            if not pay(c):
                return False
            queue[unit] = queue.get(unit, 0) + 1
            return True
        self.last_sent: list[tuple[int, int]] = []  # (order id, kind) of what it sent: the video's panel
        self.sent_resources: list[str | None] = []  # per command: what it sends the worker to harvest (accepted())
        sel = st["sel"]
        self.view_n_own = st["n_own"]
        own_sel = sel[:st["n_own"]]
        frames = set(own_sel[(own_sel[:, fx.C_FLAGS] & fx.CONSTRUCTING) > 0, fx.C_ID].tolist())
        n = min(st["n_own"], fx.MAX_OWN, len(order))
        for i in np.flatnonzero(np.asarray(order[:n])).tolist():  # (the units given an order: most are given none)
            c = int(order[i])
            oid, kind = self.orders[c]
            if kind == fx.UNIT:
                ttype = int(sel[int(tgt[i]), fx.C_TYPE])
                res = fx.harvest_resource(oid, kind, fx.GOLD_MINES[0] if oid == fx.HARVEST else ttype)
            else:
                res = fx.harvest_resource(oid, kind, None)
            unit = int(sel[i, fx.C_ID])
            same = True  # (gold from another mine than its own: a transfer, not redundant)
            gold_mine = None
            if res == "gold" and kind == fx.UNIT:
                gold_mine = mine = self._mine(sel, int(tgt[i]), i)  # (its own mine looked for only when it has none)
                same = mine is None or mine == (self.view.mine_of[unit] if unit in self.view.mine_of else self._mine(sel, -1, i))
            current = int(sel[i, fx.C_ORDER])
            if current == fx.SMART and int(sel[i, fx.C_TYPE]) == WISP and self._at_own_mine(sel, i):  # (a wisp
                current = fx.HARVEST  # mining shows "smart": a harvest re-sent was refused, 318 a game; one walking
                # on a right-click elsewhere is not mining: its harvest goes out; audit 6)
            if fx.redundant(oid, kind, current, res, self.view.assign.get(unit), same):
                continue  # it harvests that already
            if self.committed(unit):
                continue  # on its way to build, or building: another order would cancel it
            if int(sel[i, fx.C_FLAGS]) & LOADED:  # (a peon in a burrow, a unit in a transport: the game refuses its orders,
                self.loaded_orders = getattr(self, "loaded_orders", 0) + 1  # moves orc's most refused kind; audit 10)
                continue
            self.last_sent.append((oid, kind))
            if kind == fx.IMMEDIATE and oid == MIX:  # (--macros: the building trains its target's army mix)
                from .macros import mix_choice
                unit_code = mix_choice(getattr(self.view, "mix_target", None) or {}, getattr(self.view, "mix_have", {}),
                                       fx.rawcode(int(sel[i, fx.C_TYPE])), getattr(self.view, "mix_can", lambda u: False))
                if unit_code is not None:  # (as any train order: it had skipped the queue and the budget, ~160-216
                    uid = int.from_bytes(unit_code.encode(), "big")  # refused a target game; audit 10)
                    if produce(unit, uid, self.enc.order_index.get((uid, fx.IMMEDIATE))):
                        out.append(ImmediateOrder(unit, uid))
            elif kind == fx.IMMEDIATE and oid == CREEP:  # (--macros: to the natural's camp, attack-moving; not again to a
                at = getattr(self.view, "creep_at", None)  # unit attack-moving already)
                if at is not None and int(sel[i, fx.C_ORDER]) != ATTACK:
                    out.append(PointOrder(unit, ATTACK, int(at[0]), int(at[1])))
            elif kind == fx.IMMEDIATE and oid == EXPAND:  # (--macros: the natural taken, macros.py; one worker a step)
                where = getattr(self.view, "expand_site", None)
                race = fx.RACES[self.view.races[self.player]]
                if where is not None and not expanded and pay(self.view.enc.expand_via.get(race)):  # (its hall's cost)
                    expanded = True
                    self.view.expanding = (self.t + fx.EXPAND_PENDING_STEPS, unit)
                    mine, _, _, hx, hy = where
                    out.append(TargetOrder(unit, HAUNT, mine) if race == "undead" else Build(unit, EXPAND_HALLS[race], hx, hy))
            elif kind == fx.IMMEDIATE and oid == PICKUP:  # (--macros 2: the nearest item lying near the hero)
                if unit in getattr(self.view, "pickers", ()):
                    from .macros import PICKUP_NEAR
                    out.append(PickUp(unit, int(PICKUP_NEAR)))
                    self.pickup_orders = getattr(self, "pickup_orders", 0) + 1
                    self.__dict__.setdefault("picking_since", {})[unit] = self.t
            elif kind == fx.IMMEDIATE and oid in self.view.dead_heroes:  # (a dead hero's order at an altar: its revival)
                out.append(TargetOrder(unit, REVIVE, self.view.dead_heroes[oid]))
            elif kind == fx.IMMEDIATE and oid in TIERS and unit in self.view.busy and self.view.making.get(unit, (4,))[0] != 4:
                # (a hall's next tier while it trains workers: its queue cancelled first; the game takes no upgrade
                # behind them, so the order was refused while a hall trained workers back to back: human targets
                # missed their keep and won 4 of 26 in the real game; the target builder did this already)
                if pay(c):  # (the tier's cost: it had bypassed the step's budget; audit 10)
                    out += [ImmediateOrder(unit, CANCEL)] * 6 + [ImmediateOrder(unit, oid)]
            elif kind == fx.IMMEDIATE and oid == CANCEL and (int(sel[i, fx.C_FLAGS]) & fx.CONSTRUCTING or not (
                    self.view.queued.get(unit, 0) > 0 or self.view.making.get(unit, (4,))[0] < 4)):
                # (and nothing queued: a cancel there undoes a tier under way, a fortress aborted in a replay; audit 11)
                # (a frame cancelled: a quarter of its cost lost, and the game removes it outright; the policy cancelled
                # its own frames 2-10 times a game, the cancel learned from the demonstrations' training queues; audit 9)
                self.frame_cancels = getattr(self, "frame_cancels", 0) + 1
            elif kind == fx.IMMEDIATE:
                if oid < TYPE_CODE or produce(unit, oid, c):  # (a hero too: the first one's cost 0, the mask's costs)
                    out.append(ImmediateOrder(unit, oid))
            elif kind == fx.SKILL:
                out.append(LearnSkill(unit, oid))
            elif kind == fx.POINT:
                x, y = fx.bins_world(int(bx[i]), int(by[i]), self.view.sign)
                if oid < TYPE_CODE:
                    out.append(PointOrder(unit, oid, x, y))
                elif oid not in built and not (oid in ONE_OF and self._has_building(oid)) and pay(c):  # (each unit
                    built.add(oid)  # samples its own order: 2-3 workers had built one building in a step, and a second
                    out.append(Build(unit, oid, x, y))  # altar, war mill or graveyard came in 24-52% of short won games;
                    # audit 8)
            elif kind == fx.UNIT and oid in self.view.dead_heroes:  # (a dead tavern hero's hire: its revival at an altar of
                altar = self._altar(sel[:st["n_own"]])            # the player's; tested 2026-10-23: 0.4 of its gold, as
                if altar is not None:                             # an altar hero's)
                    out.append(TargetOrder(altar, REVIVE, self.view.dead_heroes[oid]))
            elif kind == fx.UNIT and oid >= TYPE_CODE and is_ware(oid):  # bought at a neutral building (neutral.py):
                shop = sel[int(tgt[i])]                    # the one pointed at, if it sells it, else the nearest that
                if not sells(int(shop[fx.C_TYPE]), oid):   # does (a new order's pointer has no reason to pick the
                    shop = nearest_shop(sel, oid, sel[i])  # tavern: no hire in thousands of games)
                hero = fx.is_hero_code(fx.rawcode(oid))
                hiring = self.__dict__.setdefault("hiring", {})
                if hero and (hiring or oid in built or not pay(c)):  # (one tavern hire at a time: 60% of the games with
                    shop = None  # hires had ordered two or three heroes; audit 10)
                if shop is not None:
                    if hero:  # (paid at the sale, ~33 s later: its money kept meanwhile; 10,475 hire orders had made
                        hiring[unit] = oid  # 308 hires, the money spent during the walk; audit 8)
                        free = self.__dict__.setdefault("free_hires", set())
                        if getattr(self.view, "free_hero", False):
                            free.add(unit)
                        else:
                            free.discard(unit)
                        built.add(oid)
                    out.append(Buy(unit, int(shop[fx.C_ID]), oid))
                    self.__dict__.setdefault("buying_since", {})[unit] = (self.t, float(shop[fx.C_X]), float(shop[fx.C_Y]))
                    self.bought = getattr(self, "bought", 0) + 1
                    buys = self.__dict__.setdefault("buys", {})  # (the episode row: each ware by its buyers' type)
                    key = f"{fx.rawcode(oid)} by {fx.rawcode(int(sel[i, fx.C_TYPE]))}"
                    buys[key] = buys.get(key, 0) + 1
            elif kind == fx.UNIT and oid >= TYPE_CODE:  # a building made on a unit: undead's haunted gold mine, on a
                mine = self._mine(sel, int(tgt[i]), i, plain=True)  # gold mine (it went down the buy path to the goblin
                if mine is not None and pay(c):                     # merchant before: undead never expanded)
                    out.append(TargetOrder(unit, oid, mine))
            elif kind == fx.UNIT and int(tgt[i]) >= st.get("ghost0", len(sel)):  # an enemy it remembers (inputs 3):
                if oid in (ATTACK, fx.SMART):  # where it saw it (attack: attack-move); nothing else reaches it
                    g = sel[int(tgt[i])]
                    out.append(PointOrder(unit, oid, int(g[fx.C_X]), int(g[fx.C_Y])))
            elif kind == fx.UNIT:
                target = sel[int(tgt[i])]
                if oid == fx.HARVEST:  # (a mine the player's workers can mine: the one pointed at, else the nearest;
                    mine = gold_mine if res == "gold" else self._mine(sel, int(tgt[i]), i)  # 77% of the clone's harvest
                    # orders pointed elsewhere; the mine found above, once)
                    if mine is None:
                        continue
                    out.append(TargetOrder(unit, oid, mine))
                elif oid in (REPAIR, RESTORE, RENEW) and (not self._repairable(target) or int(sel[i, fx.C_ORDER]) == oid):  # (nothing
                    # to repair there, or repairing already: the game refuses both,
                    self.noop_repairs = getattr(self, "noop_repairs", 0) + 1  # ~1,000-1,500 a game as human and orc,
                    continue  # 11-35% of the policy's orders since fgself-100; not sent: what it did, as before)
                else:
                    out.append(TargetOrder(unit, oid, int(target[fx.C_ID])))
            elif kind == fx.TREE and self.trees:
                x, y = fx.bins_world(int(bx[i]), int(by[i]), self.view.sign)
                out.append(TargetDestructable(unit, oid, self._nearest_tree(x, y)))
            if len(out) > len(self.sent_resources):  # (aligned with out: accepted() pairs them; a tier's queue cancels
                self.sent_resources += [None] * (len(out) - len(self.sent_resources) - 1) + [res]  # before it: none;
                # one entry for its seven commands had paired the step's later orders with others' resources; audit 4)
        if fx.MACROS and fx.RACES[self.view.races[self.player]] == "nightelf" and getattr(self, "rows", None) is not None:
            out += self._entangle(self.rows)  # (--macros: the expansion's mine; the policy has no entangle order)
        if fx.MACROS >= 3 and getattr(self, "rows", None) is not None:
            out += self._auto_skills(self.rows, out)  # (--macros 3: a skill point the policy left unspent)
        self.sent_resources += [None] * (len(out) - len(self.sent_resources))
        self.issued += len(out)
        return out

    def _nearest_tree(self, x: float, y: float) -> int:
        """min(self.trees, key=its squared distance to (x, y)), the same tree: the distances in numpy first (the dict of
        ~2,000 trees walked in Python had taken 1.4 ms an order), then the few nearest by the Python key itself (numpy's
        square and Python's ** 2 differ in the last bit now and then: a near tie)."""
        if getattr(self, "_tree_of", None) is not self.trees or self._tree_len != len(self.trees):
            ids = list(self.trees)  # (trees are only taken away: a length is a version; the ones left in their order)
            keep = (np.fromiter((k in self.trees for k in self._tree_ids), bool, len(self._tree_ids))
                    if getattr(self, "_tree_of", None) is self.trees else None)
            if keep is not None and int(keep.sum()) == len(ids):
                self._tree_xy = self._tree_xy[keep]  # (made again from the dict: ~0.3 ms each time a tree fell)
            else:
                self._tree_xy = np.array([self.trees[k] for k in ids], float).reshape(-1, 2)
            self._tree_ids, self._tree_of, self._tree_len = ids, self.trees, len(self.trees)
        d = (self._tree_xy[:, 0] - x) ** 2 + (self._tree_xy[:, 1] - y) ** 2
        near = np.flatnonzero(d <= d.min() * (1 + 1e-9) + 1e-6).tolist()
        return min((self._tree_ids[j] for j in near), key=lambda k: (self.trees[k][0] - x) ** 2 + (self.trees[k][1] - y) ** 2)

    def _auto_skills(self, rows: np.ndarray, given: list) -> list[Command]:
        """--macros 3: a hero of the player's with a skill point it has not spent for SKILL_WAIT_STEPS learns the
        ability the built-in AI's order for it names (herodata.next_skill). The policy spent its points on 2.8% of
        the steps it had one: 51 of 128 heroes ended their games with points unspent, and 4 of 42 that reached level
        6 learned their ultimate (the AI's 68 of 68; audit 3)."""
        from .herodata import next_skill
        if rows.shape[1] < fx.ROW_COLS:
            return []
        learning = {c.hero for c in given if isinstance(c, LearnSkill)}
        wait = self.__dict__.setdefault("skill_since", {})
        heroes = rows[rows[:, fx.C_SKILLPTS] > 0]  # (the few with a point first: the other masks over those)
        heroes = heroes[(heroes[:, fx.C_OWNER] == self.player) & (heroes[:, fx.C_FLAGS] & fx.HERO != 0)
                        & (heroes[:, fx.C_FLAGS] & fx.DEAD == 0)]
        out: list[Command] = []
        for r in heroes:
            hid = int(r[fx.C_ID])
            since = wait.setdefault(hid, self.t)
            if hid in learning:
                wait[hid] = self.t
                continue
            if self.t - since < SKILL_WAIT_STEPS:
                continue
            code = next_skill(fx.rawcode(int(r[fx.C_TYPE])), [int(r[fx.C_ABIL0 + 2 * k]) for k in range(4)],
                              int(r[fx.C_HLEVEL]))
            if code is not None:
                out.append(LearnSkill(hid, int.from_bytes(code.encode(), "big")))
                wait[hid] = self.t
                self.auto_skills = getattr(self, "auto_skills", 0) + 1
        for hid in [h for h in wait if h not in set(heroes[:, fx.C_ID].tolist())]:
            del wait[hid]  # (spent, or dead)
        return out

    def _entangle(self, rows: np.ndarray) -> list[Command]:
        """Night elves: a finished tree of life of the player's within ENTANGLE_NEAR of a gold mine no one has
        entangled entangles it (the expand order builds a tree of life at the natural; its mine is mined once
        entangled, and the policy has no order for that)."""
        fl = rows[:, fx.C_FLAGS]
        trees = rows[(rows[:, fx.C_OWNER] == self.player) & fx.isin(rows[:, fx.C_TYPE], TREE_CODES)
                     & (fl & (fx.CONSTRUCTING | fx.DEAD) == 0)]
        mines = rows[(rows[:, fx.C_TYPE] == NGOL) & (fl & fx.DEAD == 0)]
        if not len(trees) or not len(mines):
            return []
        held = rows[(rows[:, fx.C_TYPE] == EGOL) & (fl & fx.DEAD == 0)]
        sent = self.__dict__.setdefault("entangle_sent", {})
        out: list[Command] = []
        for tree in trees:
            d = np.hypot(mines[:, fx.C_X] - tree[fx.C_X], mines[:, fx.C_Y] - tree[fx.C_Y])
            m = mines[int(np.argmin(d))]
            if d.min() > ENTANGLE_NEAR or (np.hypot(held[:, fx.C_X] - m[fx.C_X], held[:, fx.C_Y] - m[fx.C_Y]) < 300).any():
                continue
            tid = int(tree[fx.C_ID])
            if self.t - sent.get(tid, -ENTANGLE_EVERY) >= ENTANGLE_EVERY:
                sent[tid] = self.t
                out.append(TargetOrder(tid, ENTANGLE, int(m[fx.C_ID])))
        return out

    @staticmethod
    def _altar(own: np.ndarray) -> int | None:
        """A finished altar of the player's (`own`: its units' rows), its id, or None."""
        done = own[fx.isin(own[:, fx.C_TYPE], ALTAR_CODES) & (own[:, fx.C_FLAGS] & (fx.CONSTRUCTING | fx.DEAD) == 0)]
        return int(done[0, fx.C_ID]) if len(done) else None

    def _mine(self, sel: np.ndarray, k: int, i: int, plain: bool = False) -> int | None:
        """The gold mine an order pointing at row `k` goes to: that one if the player's workers can mine it, else the
        nearest such to the mine pointed at (or to unit row `i`); its id, None with none in view. Mined: a plain gold
        mine by humans and orcs, the player's own haunted one by undead, its entangled one by night elves (a harvest
        on any other was refused: 195 a game as undead, 434 as night elf; fgself-138). `plain`: a plain gold mine,
        a haunted gold mine's site."""
        race = fx.RACES[self.view.races[self.player]] if self.view is not None else "human"
        want = fx.GOLD_MINES[0] if plain else MINED_BY.get(race, fx.GOLD_MINES[0])
        ok = sel[:, fx.C_TYPE] == want
        if not plain and race in MINED_BY:  # (its own, finished, with room: a haunted or entangled mine takes 5 workers,
            ok &= (sel[:, fx.C_OWNER] == self.player) & (sel[:, fx.C_FLAGS] & fx.CONSTRUCTING == 0)  # the 6th's and an
            for j in np.flatnonzero(ok).tolist():  # unfinished one's harvest refused: 550-741 a game with an expansion
                if self._miners(sel, j, i) >= MINE_WORKERS:  # and 15+ workers; audit 6)
                    ok[j] = False
        if 0 <= k < len(sel) and ok[k]:
            return int(sel[k, fx.C_ID])
        mines = sel[ok]
        if not len(mines):
            return None
        ref = sel[k] if 0 <= k < len(sel) and int(sel[k, fx.C_TYPE]) in fx.GOLD_MINES else sel[i]
        d = (mines[:, fx.C_X] - ref[fx.C_X]) ** 2 + (mines[:, fx.C_Y] - ref[fx.C_Y]) ** 2
        return int(mines[int(np.argmin(d)), fx.C_ID])

    def _refusal(self, c, key: str, rows: np.ndarray) -> list:
        """A refused order's context: [its kind, the unit's type, flags and current order, the target's type, whose
        (own / enemy / neutral), flags and hit points' share, the player's gold and lumber], for what the game refuses
        and why."""
        def row(i):
            k = np.flatnonzero(rows[:, fx.C_ID] == i)
            return rows[k[0]] if len(k) else None
        u = row(getattr(c, "unit", getattr(c, "hero", -1)))
        out = [key] + ([fx.rawcode(int(u[fx.C_TYPE])), int(u[fx.C_FLAGS]), int(u[fx.C_ORDER])] if u is not None else [None] * 3)
        t = row(c.target) if hasattr(c, "target") else None
        if t is not None:
            whose = "own" if int(t[fx.C_OWNER]) == self.player else "enemy" if int(t[fx.C_OWNER]) == 1 - self.player else "neutral"
            out += [fx.rawcode(int(t[fx.C_TYPE])), whose, int(t[fx.C_FLAGS]), round(int(t[fx.C_HP]) / max(int(t[fx.C_MAXHP]), 1), 2)]
        me = getattr(self, "me", None)
        return out + ([int(me[2]), int(me[3])] if me is not None else [])  # (the player's gold and lumber)

    def _miners(self, sel: np.ndarray, j: int, i: int) -> int:
        """Workers of the player's at mine row j (within MINE_NEAR, harvesting, or in it), not counting unit row i.
        (Each mine's, once a step: every gold harvest ordered had counted them again for every mine.)"""
        n = getattr(self, "view_n_own", None)
        memo = self.__dict__.get("_miners_at")
        if memo is None or memo[0] is not sel or memo[1] != n:
            memo = self._miners_at = (sel, n, {})
        ids = memo[2].get(j)
        if ids is None:
            own = sel[:n] if n is not None else sel
            w = own[own[:, fx.C_FLAGS] & fx.WORKER > 0]
            near = np.hypot(w[:, fx.C_X] - sel[j, fx.C_X], w[:, fx.C_Y] - sel[j, fx.C_Y]) < MINE_NEAR
            mining = fx.isin(w[:, fx.C_ORDER], HARVESTING_CODES) | (w[:, fx.C_FLAGS] & HIDDEN > 0)
            ids = memo[2][j] = w[near & mining, fx.C_ID].tolist()
        return len(ids) - ids.count(int(sel[i, fx.C_ID]))

    def _at_own_mine(self, sel: np.ndarray, i: int) -> bool:
        """A unit row within MINE_NEAR of one of the player's entangled or haunted gold mines."""
        m = sel[fx.isin(sel[:, fx.C_TYPE], MINED_CODES) & (sel[:, fx.C_OWNER] == self.player)]
        return bool(len(m)) and bool((np.hypot(m[:, fx.C_X] - sel[i, fx.C_X], m[:, fx.C_Y] - sel[i, fx.C_Y]) < MINE_NEAR).any())

    def _repairable(self, row: np.ndarray) -> bool:
        """A repair's target: a building or a machine of the player's that is damaged or unfinished (a frame: while the
        player has the lumber it costs, or the gold of one that costs none)."""
        fl = int(row[fx.C_FLAGS])
        human = fx.rawcode(int(row[fx.C_TYPE]))[:1] == "h"  # (only a human frame takes helpers: an orc one refused, tested
        return (int(row[fx.C_OWNER]) == self.player and (fl & fx.STRUCTURE or int(row[fx.C_TYPE]) in MACHINES)  # 2026-10-23;
                and ((int(row[fx.C_HP]) < int(row[fx.C_MAXHP]) and not fl & fx.CONSTRUCTING)  # undead and night elf ones
                     or (fl & fx.CONSTRUCTING and human)) and self._can_power_build(row))  # build themselves; a repair
        # costs a share of the building's price as a power build does: refused without its lumber, 80% of the repairs
        # refused; audit 10)

    def _has_building(self, oid: int) -> bool:
        """The player has a building of type `oid` (finished or a frame) or a worker on its way to build one."""
        rows = getattr(self, "rows", None)
        if rows is None:
            return False
        own = rows[(rows[:, fx.C_OWNER] == self.player) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]
        return bool(((own[:, fx.C_TYPE] == oid) | (own[:, fx.C_ORDER] == oid)).any())

    def _can_power_build(self, row: np.ndarray) -> bool:
        """A helper at a frame costs a share of the building's price as it builds (AbilityData Ahrp: 0.15), and the game
        refuses the order with none of the lumber (tested 2026-10-24: 5 gold and no lumber refused, no gold and 5
        lumber or 1 and 1 taken): ~275 refused a game as human, a frame's helpers sent on after the lumber ran out."""
        me, enc = getattr(self, "me", None), self.view.enc
        k = enc.order_index.get((int(row[fx.C_TYPE]), fx.POINT)) or enc.order_index.get((int(row[fx.C_TYPE]), fx.IMMEDIATE))
        if me is None or k is None or enc.costs is None:  # (an upgraded type, a keep: its upgrade order's row; audit 11)
            return True
        gold, lumber = int(enc.costs[k][0]), int(enc.costs[k][1])  # (each it costs, at least 1: a damaged farm refused
        return (gold == 0 or int(me[2]) >= 1) and (lumber == 0 or int(me[3]) >= 1)  # with 1000 gold and no lumber, and
        # with 5 lumber and no gold; audit 11)

    def committed(self, unit: int) -> bool:
        """A worker busy with a building order (features.building) for less than BUILD_COMMIT_STEPS, or a unit
        on its way to buy at a neutral building for less than BUY_COMMIT_STEPS (another order every half second
        turned it back long before it reached the tavern: no tavern hero hired in thousands of games)."""
        since = self.building_since.get(unit)
        bought = getattr(self, "buying_since", {}).get(unit)
        picking = getattr(self, "picking_since", {}).get(unit)
        return ((since is not None and self.t - since < fx.BUILD_COMMIT_STEPS)
                or (bought is not None and self.t - bought[0] < BUY_COMMIT_STEPS)
                or (picking is not None and self.t - picking < PICKUP_COMMIT_STEPS))

    def _release_buyers(self, sel: np.ndarray) -> None:
        """Buyers free again once they have been in reach of their shop BUY_ARRIVED steps (the sale went through or
        will not: the harness keeps trying meanwhile), not the whole BUY_COMMIT_STEPS (a hero in reach stood still)."""
        buying = getattr(self, "buying_since", {})
        if not buying:
            return
        pos = {int(u): (float(x), float(y)) for u, x, y in sel[:, [fx.C_ID, fx.C_X, fx.C_Y]].tolist()}
        near = self.__dict__.setdefault("buy_near", {})
        hiring = getattr(self, "hiring", {})
        for unit, (t0, sx, sy) in list(buying.items()):
            p = pos.get(unit)
            if p is None or self.t - t0 >= BUY_COMMIT_STEPS:
                buying.pop(unit, None)
                near.pop(unit, None)
                hiring.pop(unit, None)
            elif (p[0] - sx) ** 2 + (p[1] - sy) ** 2 < BUY_REACH ** 2:
                near[unit] = near.get(unit, 0) + 1
                if near[unit] >= BUY_ARRIVED:
                    buying.pop(unit, None)
                    near.pop(unit, None)
                    hiring.pop(unit, None)

    def act(self, obs: Observation, t: int) -> list[Command]:
        st = self.observe(obs, t)
        if st is None:
            return []
        n, n_own = st["n"], st["n_own"]
        dev = self.device
        with torch.no_grad():
            ent = torch.as_tensor(st["ent"], device=dev).unsqueeze(0).float()
            typ = torch.as_tensor(st["type"], device=dev).unsqueeze(0).long()
            cur = torch.as_tensor(st["cur"], device=dev).unsqueeze(0).long()
            mask = torch.ones(1, n, dtype=torch.bool, device=dev)
            glob = torch.as_tensor(st["glob"], device=dev).unsqueeze(0)
            g, u = self.net.encode(ent, typ, cur, mask, glob)
            g, self.h = self.net.context(g, self.h)
            O = min(n_own, fx.MAX_OWN)
            avail = torch.as_tensor(st["avail"], device=dev)[None] if st.get("avail") is not None else None
            logits = self.net.order_logits(g, u[:, :O], typ, torch.tensor([O], device=dev), avail=avail, ent=ent)
            order = self._sample(logits)  # [1, O]
            if self.order_temperature != 1.0:  # whether a unit gets an order stays as learned; which one sharpens
                again = self._sample(logits[..., 1:] / self.order_temperature) + 1
                order = torch.where(order > 0, again, order)
            ptr, xl, z = self.net.target_logits(g, u, mask, order)
            tgt, bx_ = self._sample(ptr)[0], self._sample(xl)
            by = self._sample(self.net.y_logits(z, bx_))[0]
        return self.commands(st, order[0].tolist(), tgt.tolist(), bx_[0].tolist(), by.tolist())


def matchup_setup(map_name: str, race: str, ai_race: str, difficulty: str, agent_side: int, handicap: int,
                  max_minutes: float, step_seconds: float, screen: tuple[int, int] = (320, 240)) -> GameSetup:
    agent = Agent(race, handicap=handicap)
    ai = BuiltinAI(ai_race, difficulty, handicap=handicap)
    slots = [agent, ai] if agent_side == 0 else [ai, agent]
    return GameSetup(map=map_name, slots=slots, step_seconds=step_seconds, max_game_seconds=max_minutes * 60,
                     victory="decisive", window=screen,  # a small screen: nothing looks at the pixels
                     d3d_thread=False, render_threads=0,  # (and drawn in the game's own thread)
                     record_ai_orders=True)  # (the AI's orders: shown in the videos' panel)


def play_game(g: GameInstance, obs: Observation, net, vocab: dict, device, agent_side: int,
              temperature: float, order_temperature: float = 1.0, values: dict | None = None, costs=None,
              left_ok: bool = False) -> dict:
    """One game from its first observation `obs` (the game's setup: matchup_setup). `values`
    (unit type values, fullgame.trace.unit_values): also a trace for the video's panel ("trace"). `left_ok` (a
    person at the game's window, versus.py): the game closed before its end is outcome "LEFT", not an error."""
    from ..runtime.instance import GameCrashed
    from .trace import material, trace_step
    slots = g.setup.slots
    races = [fx.RACES.index(s.race) if s.race in fx.RACES else 0 for s in slots]
    bot = BCAgent(net, vocab, agent_side, device, temperature, order_temperature, costs=costs)
    t0 = time.time()
    order_names = {v: k for k, v in (obs.orders or {}).items()}
    bot.begin(obs, races)
    t = 0
    sent = failed = 0
    by_kind: dict[str, list[int]] = {}  # command kind -> [sent, failed]
    names = {v: k for k, v in (obs.orders or {}).items()}
    kinds = {0: fx.IMMEDIATE, 1: fx.POINT, 2: fx.UNIT, 3: fx.SKILL}
    trace: list[dict] = []
    left = False
    held = {0: [], 1: []}  # gold + lumber on hand each step (hoarding: gathered and not spent)
    food_1min: dict[int, int] = {}
    while not obs.game_over:
        for p in (0, 1):
            s_ = obs.players.get(p)
            if s_ is not None:
                held[p].append(s_.gold + s_.lumber)
                if obs.game_time >= 60 and p not in food_1min:
                    food_1min[p] = s_.food_used
        cmds = bot.act(obs, t)
        if values is not None:
            mine = Counter(fx.order_label(o, k, names) for o, k in getattr(bot, "last_sent", []))
            trace.append(trace_step(t, obs, material(obs, values), {agent_side: mine}))
        try:
            obs = g.step(cmds)
        except GameCrashed:
            if not left_ok:
                raise
            left = True  # (the person closed the game)
            break
        if values is not None and trace:
            owner = {u.id: u.owner for u in obs.units}
            ai = Counter(fx.order_label(o.order, kinds.get(o.kind, fx.IMMEDIATE), names) for o in obs.issued
                         if o.order not in fx.DROPPED_ORDERS and owner.get(o.unit) == 1 - agent_side)
            trace[-1]["orders"][str(1 - agent_side)] = dict(ai)
        bot.accepted(cmds, obs.command_results)
        for c, ok in zip(cmds, obs.command_results):  # the game refused the order
            order = getattr(c, "order", None) or getattr(c, "building", None) or 0  # (Build: the building's type)
            name = order_names.get(order, None) or rawcode_or(order)
            k = by_kind.setdefault(f"{type(c).__name__}:{name}", [0, 0])
            k[0] += 1
            k[1] += not ok
        sent += len(cmds)
        failed += sum(not ok for ok in obs.command_results[:len(cmds)])
        t += 1
    results = {p: s.result.name for p, s in obs.players.items()}
    sides = {("agent" if p == agent_side else "ai"): {"gold": s.gold_gathered, "lumber": s.lumber_gathered,
                                                        "food": f"{s.food_used}/{s.food_cap}",
                                                        "structures": s.structures,
                                                        "held": round(sum(held[p]) / max(len(held[p]), 1)),
                                                        "food_1min": food_1min.get(p)}
             for p, s in obs.players.items() if p in (0, 1)}
    outcome = "LEFT" if left else results.get(agent_side, "?")
    ai = slots[1 - agent_side]
    out = {"outcome": outcome, "minutes": round(obs.game_time / 60, 2), "orders": bot.issued, "steps": t,
            "failed": failed, "by_kind": by_kind, "sides": sides,
            "race": slots[agent_side].race, "ai_race": ai.race, "difficulty": ai.difficulty, "side": agent_side,
            "seconds": round(time.time() - t0, 1)}
    if values is not None:
        res = {s: 1.0 if r == "VICTORY" else -1.0 if r == "DEFEAT" else 0.0 for s, r in results.items() if s in (0, 1)}
        out["trace"] = {"steps": trace, "gamma": 0.997, "outcome": {str(s): v for s, v in res.items()},
                        "sides": [{"player": agent_side, "name": f"the clone ({slots[agent_side].race})", "kind": "agent"},
                                  {"player": 1 - agent_side, "name": f"built-in AI {ai.difficulty} ({ai.race})",
                                   "kind": "ai"}]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--map", default="duelrush")
    ap.add_argument("--race", default="human", help="the agent's race, or 'all' (random each game)")
    ap.add_argument("--ai-race", default="orc", help="the built-in AI's race, or 'all'")
    ap.add_argument("--difficulty", default="normal")
    ap.add_argument("--handicap", type=int, default=50)
    ap.add_argument("--max-minutes", type=float, default=4.0)
    ap.add_argument("--step-seconds", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--order-temperature", type=float, default=1.0,
                    help="sharpens which order a unit gets, not whether it gets one")
    ap.add_argument("--games-per-process", type=int, default=4,
                    help="games of one matchup in one running game (restarts reload the map in it)")
    ap.add_argument("--label", help="the evaluation's name on the dashboard")
    ap.add_argument("--videos", type=int, default=2, help="games to film (the first of a launch each)")
    ap.add_argument("--avail-mask", type=int, default=1, help="mask the orders the player can't pay for yet")
    ap.add_argument("--mirror", action="store_true", help="the AI plays the agent's race")
    ap.add_argument("--device", help="default: cuda if available")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="results (.jsonl; default: next to the checkpoint, play.jsonl)")
    ap.add_argument("--wine", default="", help="another Wine's bin folder (e.g. GE-Proton's files/bin, as the "
                    "self-play runs use; runtime.wine.use_wine)")
    ap.add_argument("--fsync", type=int, default=1, help="with --wine (a Proton build): futex-based synchronization")
    args = ap.parse_args(argv)
    if args.wine:  # (before any game: the games read the environment)
        from ..runtime import wine
        wine.use_wine(args.wine, bool(args.fsync))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net, ck = load(args.checkpoint, device)
    fx.set_z(net.config.get("z", 0))  # (a strategy-conditioned network reads a strategy block: none given, zeros)
    from ..data.duelmap import parse_duel_name
    if parse_duel_name(args.map) is None:  # (another map's frame, as selfplay's: Echo Isles' starts and point bins)
        from ..data.mapbuild import stock_map_path
        fx.set_geometry(*fx.map_geometry(stock_map_path(args.map)))
    if net.bins != fx.BINS:
        raise SystemExit(f"{args.checkpoint} has {net.bins} point bins, {args.map}'s frame {fx.BINS}: another map's policy")
    vocab = ck["vocab"]
    rng = random.Random(args.seed)
    plans = [(i, rng.choice(fx.RACES) if args.race == "all" else args.race,
              rng.choice(fx.RACES) if args.ai_race == "all" else args.ai_race, i % 2) for i in range(args.games)]
    if args.mirror:
        plans = [(i, race, race, side) for i, race, _, side in plans]
    out = args.out or args.checkpoint.with_name("play.jsonl")
    eval_id = f"{args.checkpoint.stem}@{int(time.time())}"  # this launch's games (the dashboard groups by it)
    label = args.label or (f"{args.checkpoint.name} (epoch {ck.get('epoch')}) vs {args.difficulty} AI"
                           + (f", temperature {args.temperature:g}" if args.temperature != 1.0 else "")
                           + (f", order temperature {args.order_temperature:g}" if args.order_temperature != 1.0 else "")
                           + (", mirror matchups" if args.mirror else "") + ("" if args.avail_mask else ", no availability mask"))
    names = queue.Queue()
    for k in range(args.parallel):
        names.put(f"bcplay{k}")
    # games of one matchup run one after the other in one process (restarts reload the map)
    by_matchup: dict[tuple, list] = {}
    for p in plans:
        by_matchup.setdefault(p[1:], []).append(p)
    chunks = [ps[s:s + args.games_per_process] for ps in by_matchup.values()
              for s in range(0, len(ps), args.games_per_process)]
    chunks.sort(key=lambda c: c[0][0])
    results: list[dict] = []
    lock = threading.Lock()

    def run(chunk) -> None:
        _, race, ai_race, side = chunk[0]
        setup = matchup_setup(args.map, race, ai_race, args.difficulty, side, args.handicap, args.max_minutes,
                              args.step_seconds)

        def play(g, obs, k, fresh) -> None:
            i = chunk[k][0]
            film = fresh and films.due()
            r = play_game(g, obs, net, vocab, device, side, args.temperature, args.order_temperature,
                          values=values if film else None, costs=costs)
            trace = r.pop("trace", None)
            r.update({"game": i, "checkpoint": str(args.checkpoint), "time": time.time(), "map": args.map,
                      "eval": eval_id, "label": label, "epoch": ck.get("epoch"), "temperature": args.temperature,
                      "order_temperature": args.order_temperature})
            with lock:
                results.append(r)
                with open(out, "a") as f:
                    f.write(json.dumps(r) + "\n")
            if film:
                if trace is not None:
                    trace["title"] = f"{args.checkpoint.parent.name} · {label}"[:70]
                films.film(g, f"{eval_id.replace('@', '-')}-game{i:03d}", trace=trace, row={
                    "episode": i, "eval": eval_id, "label": label,
                    "title": f"game {i}: the clone ({race}) vs the built-in AI ({ai_race}, {args.difficulty})",
                    "outcome": {"VICTORY": 1.0, "DEFEAT": -1.0}.get(r["outcome"], 0.0),
                    "sub": f"{r['outcome'].lower()} after {r['minutes']:.1f} game minutes · {r['orders']} orders, "
                           f"{r['failed']} refused"})
            print(f"game {i}: {race} (side {side}) vs {ai_race} {args.difficulty}: {r['outcome']} after "
                  f"{r['minutes']} min ({r['orders']} orders, {r['failed']} refused {r['by_kind']}; {r['sides']})",
                  flush=True)

        name = names.get()
        try:
            series(setup, name, len(chunk), play)
        finally:
            names.put(name)

    from .costs import order_costs
    costs = order_costs(vocab, args.map) if args.avail_mask else None  # (unaffordable orders masked)
    films = Films(out.parent, 0.0, args.videos, "bcplay_video")  # videos: the fit's Replays on the dashboard
    from .trace import unit_values
    values = unit_values() if args.videos > 0 else None
    with ThreadPoolExecutor(args.parallel) as ex:
        list(ex.map(run, chunks))
    if films.count:
        print(f"rendering {films.count} videos", flush=True)
    films.wait()
    wins = sum(r["outcome"] == "VICTORY" for r in results)
    ties = sum(r["outcome"] == "TIE" for r in results)
    print(f"{args.checkpoint}: {wins} wins, {ties} ties, {len(results) - wins - ties} losses in {len(results)} games "
          f"against the built-in AI ({args.difficulty})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
