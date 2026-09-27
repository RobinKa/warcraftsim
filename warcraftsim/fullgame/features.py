"""Demonstration games (fullgame/collect.py) -> what one player saw and ordered, per step.

Pure numpy (no warcraftsim imports), so the torch trainer can load it by path like rl/.

For player p at step t:
* entities: p's own living units (buildings too) first, then the enemy's and neutral units
  (creeps, gold mines) that p could see (the units' visibility bits), at most MAX_ENT. Positions
  are mirrored so that p's own base is on the left (the duel maps are mirror-symmetric).
* entity features (F floats), each entity's unit type and current order (vocabulary indices).
* global features: resources, supply, game time, both races, p's upgrade levels.
* labels for each of the first MAX_OWN own entities: the order it got during the step (the last
  one if several; class 0: none), as a class of the order vocabulary, i.e. (order id, kind),
  and its target: another entity (a pointer), or a point (x / y bins over the map). Orders on
  trees become kind TREE with the tree's position (at play time the nearest tree there is used).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

MAP_EXTENT = 3072.0  # half the duel map's width
BINS = 128  # point targets: 48 units per bin
MAX_ENT, MAX_OWN = 160, 96
MOVE, AIMOVE = 851986, 851988  # the AI's own move order is played back as a move
# orders that are not decisions: internal ones the AI gives to most units (851974, 852660), and
# the engine's own: resumeharvesting (a worker going back after dropping off: 7.7% of recorded
# orders; a policy that learned it interrupted working harvesters, the game refused two thirds),
# returnresources, and autocasts (recharge: moon wells; ambush: archers at night)
DROPPED_ORDERS = {851974, 852660, 852017, 852020, 852157, 852131}
HARVEST, SMART = 852018, 851971
HARVESTING = {852018, 852017, 852020}  # current orders of a worker in its harvest cycle


def redundant(order: int, kind: int, current: int) -> bool:
    """A harvest order (harvest, or smart on a tree or unit) for a worker already harvesting: the
    AI re-issues these without effect; learned as decisions, a policy re-ordered its workers every
    few seconds to other trees and reset their work (240 gold from 27 peasants in 80 s)."""
    return current in HARVESTING and order in (HARVEST, SMART) and kind in (UNIT, TREE)
IMMEDIATE, POINT, UNIT, SKILL, TREE = range(5)
KIND_NAMES = ("immediate", "point", "unit", "skill", "tree")
RACES = ("human", "orc", "undead", "nightelf")
RESEARCH_FINISH = 9

# unit columns (collect.UNIT_COLS)
C_STEP, C_ID, C_TYPE, C_OWNER, C_X, C_Y, C_FACING, C_HP, C_MAXHP, C_MANA, C_MAXMANA, C_ORDER, C_FLAGS, \
    C_VIS, C_RESOURCE, C_HLEVEL, C_HXP, C_SKILLPTS = range(18)
DEAD, STRUCTURE = 1024, 2
FLAG_BITS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048)  # UnitFlags but DEAD
F = 3 + 2 + 4 + len(FLAG_BITS) + 3 + 2 + 1 + 2  # 28: ... and production (queued, busy)
TYPE_CODE = 0x1000000  # order ids at or above this are unit / building / upgrade codes
# events: a building's production (a = the building)
TRAIN_START, TRAIN_FINISH, TRAIN_CANCEL = 5, 6, 7
PRODUCTION_START = {5, 8, 11}  # train, research, upgrade started
PRODUCTION_END = {6, 7, 9, 10, 12, 13}  # finished or cancelled


def load_game(path: str | Path) -> dict:
    z = np.load(path)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(str(z["meta"]))
    return out


def _step_slices(col: np.ndarray, steps: int) -> np.ndarray:
    """[steps + 1] boundaries of each step's rows (rows are in step order)."""
    return np.searchsorted(col, np.arange(steps + 1), side="left")


def relabel(order: int, kind: int, target_known: bool, target_is_tree: bool) -> tuple[int, int] | None:
    """A recorded order as (order id, kind) of the vocabulary, None: not a decision to learn."""
    if order in DROPPED_ORDERS:
        return None
    if order == AIMOVE:
        order = MOVE
    if kind == 2:
        if target_known:
            return order, UNIT
        if target_is_tree:
            return order, TREE
        return None  # an item or a unit this player didn't see
    return order, {0: IMMEDIATE, 1: POINT, 3: SKILL}[kind]


def build_vocab(paths: list[str | Path], min_count: int = 5) -> dict:
    """Unit types, current orders, order classes and upgrades seen in the games."""
    types, cur, orders, upgrades = Counter(), Counter(), Counter(), Counter()
    names: dict[int, str] = {}
    for p in paths:
        g = load_game(p)
        if not names:  # the game's order ids by name (the same in every game)
            names = {int(v): k for k, v in (g["meta"].get("order_names") or {}).items()}
        u = g["units"]
        types.update(u[:, C_TYPE].tolist())
        cur.update(u[u[:, C_ORDER] != 0, C_ORDER].tolist())
        trees = set(g["trees"][:, 0].tolist())
        ids = set(u[:, C_ID].tolist())
        for r in g["orders"]:
            lab = relabel(int(r[2]), int(r[3]), int(r[6]) in ids, int(r[6]) in trees)
            if lab is not None:
                orders[lab] += 1
        ev = g["events"]
        upgrades.update(ev[ev[:, 1] == RESEARCH_FINISH, 3].tolist())
    keep = lambda c: [k for k, n in c.most_common() if n >= min_count]  # noqa: E731
    return {"types": keep(types), "current_orders": keep(cur)[:255],
            "orders": [list(k) for k in keep(orders)], "upgrades": keep(upgrades),
            "counts": {"orders": [orders[tuple(k)] for k in keep(orders)]},
            "order_names": {str(k): v for k, v in names.items()}}


def rawcode(v: int) -> str:
    return int(v).to_bytes(4, "big").decode("latin-1")


def order_label(order: int, kind: int, names: dict[int, str]) -> str:
    """An order class for people: "attack (unit)", "train hpea", "build hbar (point)", ..."""
    if order >= TYPE_CODE:  # a unit, building or upgrade code: train / build / research / learn
        code = rawcode(order)
        verb = {IMMEDIATE: "research" if code.startswith("R") else "train", POINT: "build",
                SKILL: "learn"}.get(kind, "")  # (upgrade codes start with R)
        return f"{verb} {code}".strip()
    name = names.get(order, str(order))
    return name if kind == IMMEDIATE else f"{name} ({KIND_NAMES[kind]})"


def demo_order_names(runs_dir: str | Path) -> dict[int, str]:
    """The game's order names by id, from a recorded game's meta (runs/fullgame/*/game*.npz)."""
    for p in sorted(Path(runs_dir).glob("fullgame/*/game*.npz"))[:1]:
        try:
            with np.load(p) as z:
                return {int(v): k for k, v in (json.loads(str(z["meta"])).get("order_names") or {}).items()}
        except (OSError, ValueError, KeyError):
            pass
    return {}


def describe_spaces(vocab: dict, order_names: dict[int, str] | None = None, agents: str = "") -> dict:
    """The observation and action spaces for people, in the format the dashboard shows
    (puffer.tasks.describe_spaces): run.json "spaces" of a self-play run."""
    names = order_names or {int(k): v for k, v in (vocab.get("order_names") or {}).items()}
    enc = Encoder(vocab)
    ent = (["own", "enemy", "neutral", "x (mirrored: own base left)", "y", "hp / max hp", "max hp / 1000",
            "mana / max mana", "max mana / 1000"] + [f"flag {b}" for b in FLAG_BITS]
           + ["hero level / 10", "resource / 12500 (mines)", "skill points / 3", "sin facing", "cos facing (mirrored)",
              "has an order", "queued production / 5", "producing"])
    glob = (["gold / 1000", "lumber / 1000", "food used / 100", "food cap / 100", "upkeep / 2", "time (steps / 1800)"]
            + [f"own race: {r}" for r in RACES] + [f"enemy race: {r}" for r in RACES]
            + [f"upgrade {rawcode(u)} (level / 3)" for u in vocab["upgrades"]])
    orders = ["none"] + [order_label(int(o), int(k), names) for o, k in vocab["orders"]]
    return {"observation": {"size": f"{enc.G} + up to {MAX_ENT} × {F}", "blocks": [
                {"name": "global token", "rows": 1, "features": glob},
                {"name": f"entities: own units first (up to {MAX_OWN}), then the enemy's and neutral ones it sees",
                 "rows": MAX_ENT, "features": ent},
                {"name": f"each entity's unit type (one of {enc.n_types}, embedded)", "rows": MAX_ENT,
                 "features": ["type"]},
                {"name": f"each entity's current order (one of {enc.n_cur}, embedded)", "rows": MAX_ENT,
                 "features": ["current order"]}]},
            "actions": {"units": f"up to {MAX_OWN}", "sizes": [enc.n_orders, MAX_ENT, BINS, BINS],
                        "heads": [
                            {"name": "order", "size": enc.n_orders, "options": orders, "used_by": []},
                            {"name": "target unit (pointer)", "size": f"≤ {MAX_ENT}", "options": ["any entity it sees"],
                             "used_by": ["an order on a unit"]},
                            {"name": "point x", "size": BINS, "options": [f"{BINS} bins of {2 * MAP_EXTENT / BINS:.0f}"],
                             "used_by": ["an order at a point or on a tree (the nearest)"]},
                            {"name": "point y (given x)", "size": BINS, "options": [f"{BINS} bins"],
                             "used_by": ["an order at a point or on a tree"]}],
                        "masks": "Orders: those the unit's type got in the demonstrations (none always); "
                                 "the pointer: the entities in view. Harvest orders to workers already harvesting "
                                 "are not sent (the demonstrations' AI re-issued them without effect)."},
            "agents": agents, "reward": "+1 for a win, -1 for a loss, 0 for a tie (the time limit), at the end.",
            "reward_scale": 1}


def _bin(v: np.ndarray) -> np.ndarray:
    return np.clip(((v + MAP_EXTENT) / (2 * MAP_EXTENT) * BINS).astype(np.int64), 0, BINS - 1)


def bin_center(b: np.ndarray | int) -> np.ndarray | float:
    return (np.asarray(b) + 0.5) / BINS * 2 * MAP_EXTENT - MAP_EXTENT


class Encoder:
    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.type_index = {t: i + 1 for i, t in enumerate(vocab["types"])}  # 0: unknown
        self.cur_index = {o: i + 1 for i, o in enumerate(vocab["current_orders"])}
        self.order_index = {tuple(k): i + 1 for i, k in enumerate(vocab["orders"])}  # 0: no order
        self.upgrade_index = {u: i for i, u in enumerate(vocab["upgrades"])}
        self.n_types, self.n_cur = len(self.type_index) + 1, len(self.cur_index) + 1
        self.n_orders = len(self.order_index) + 1
        self.order_kind = np.array([IMMEDIATE] + [k[1] for k in vocab["orders"]], np.int64)
        self.G = 6 + 2 * len(RACES) + len(self.upgrade_index)

    @staticmethod
    def side(units_step0: np.ndarray, player: int) -> float:
        """-1 if the player's base is on the right (its view is mirrored), else 1."""
        own = units_step0[(units_step0[:, C_OWNER] == player) & (units_step0[:, C_FLAGS] & STRUCTURE > 0)]
        return -1.0 if len(own) and own[:, C_X].mean() > 0 else 1.0

    def entities(self, rows: np.ndarray, player: int, sign: float, production=None) -> tuple[np.ndarray, ...]:
        """One step's units -> (rows in entity order, n own, float features, types, current orders)."""
        alive = rows[:, C_FLAGS] & DEAD == 0
        vis = (rows[:, C_VIS] >> player) & 1 == 1
        own = rows[alive & (rows[:, C_OWNER] == player)][:MAX_OWN]
        enemy = rows[alive & (rows[:, C_OWNER] == 1 - player) & vis]
        neutral = rows[alive & (rows[:, C_OWNER] != player) & (rows[:, C_OWNER] != 1 - player) & vis]
        sel = np.concatenate([own, enemy, neutral])[:MAX_ENT]
        n, n_own = len(sel), len(own)
        f = np.zeros((n, F), np.float32)
        f[:n_own, 0] = 1
        f[n_own:n_own + len(enemy), 1] = 1
        f[n_own + len(enemy):, 2] = 1
        f[:, 3] = sign * sel[:, C_X] / MAP_EXTENT
        f[:, 4] = sel[:, C_Y] / MAP_EXTENT
        f[:, 5] = sel[:, C_HP] / np.maximum(sel[:, C_MAXHP], 1)
        f[:, 6] = sel[:, C_MAXHP] / 1000.0
        f[:, 7] = np.where(sel[:, C_MAXMANA] > 0, sel[:, C_MANA] / np.maximum(sel[:, C_MAXMANA], 1), 0)
        f[:, 8] = sel[:, C_MAXMANA] / 1000.0
        for k, bit in enumerate(FLAG_BITS):
            f[:, 9 + k] = (sel[:, C_FLAGS] & bit) > 0
        a = 9 + len(FLAG_BITS)
        f[:, a] = sel[:, C_HLEVEL] / 10.0
        f[:, a + 1] = sel[:, C_RESOURCE] / 12500.0
        f[:, a + 2] = sel[:, C_SKILLPTS] / 3.0
        rad = np.radians(sel[:, C_FACING].astype(np.float32))
        f[:, a + 3] = np.sin(rad)
        f[:, a + 4] = sign * np.cos(rad)
        f[:, a + 5] = sel[:, C_ORDER] != 0
        # production (buildings' current order stays 0 while they train): what the player queued, and
        # whether something is being made (View keeps both from its orders and the events)
        if production is not None:
            queued, busy = production
            f[:, a + 6] = [min(queued.get(int(i), 0), 5) / 5.0 for i in sel[:, C_ID]]
            f[:, a + 7] = [float(int(i) in busy) for i in sel[:, C_ID]]
        types = np.array([self.type_index.get(int(t), 0) for t in sel[:, C_TYPE]], np.int64)
        cur = np.array([self.cur_index.get(int(o), 0) for o in sel[:, C_ORDER]], np.int64)
        return sel, n_own, f, types, cur

    def view(self, player: int, sign: float, races: list[int]) -> "View":
        return View(self, player, sign, races)

    def encode(self, game: dict, player: int) -> dict:
        """The whole game from `player`'s side: arrays over its steps (padded to MAX_ENT)."""
        meta = game["meta"]
        T = meta["steps"] + 1
        u, pl, ev, od = game["units"], game["players"], game["events"], game["orders"]
        us, ps, es, os_ = (_step_slices(a[:, 0], T) for a in (u, pl, ev, od))
        sign = self.side(u[us[0]:us[1]], player)
        trees = {int(r[0]): (int(r[2]), int(r[3])) for r in game["trees"]}
        races = [RACES.index(r) if r in RACES else 0 for r in meta["races"]]
        out = {"ent": np.zeros((T, MAX_ENT, F), np.float16), "type": np.zeros((T, MAX_ENT), np.int16),
               "cur": np.zeros((T, MAX_ENT), np.int16), "mask": np.zeros((T, MAX_ENT), bool),
               "n_own": np.zeros(T, np.int16), "glob": np.zeros((T, self.G), np.float32),
               "y_order": np.zeros((T, MAX_OWN), np.int16), "y_ptr": np.full((T, MAX_OWN), -1, np.int16),
               "y_x": np.full((T, MAX_OWN), -1, np.int16), "y_y": np.full((T, MAX_OWN), -1, np.int16)}
        view = self.view(player, sign, races)
        for t in range(T):
            prow = pl[ps[t]:ps[t + 1]]
            me = prow[prow[:, 1] == player]
            st = view.step(u[us[t]:us[t + 1]], me[0] if len(me) else None, ev[es[t]:es[t + 1]], t)
            n = st["n"]
            out["ent"][t, :n], out["type"][t, :n], out["cur"][t, :n] = st["ent"], st["type"], st["cur"]
            out["mask"][t, :n] = True
            out["n_own"][t] = st["n_own"]
            out["glob"][t] = st["glob"]
            # labels: each own unit's last order in the step
            index = st["index"]
            view.record_orders((int(r[1]), int(r[2]), int(r[3])) for r in od[os_[t]:os_[t + 1]])
            for r in od[os_[t]:os_[t + 1]]:
                k = index.get(int(r[1]))
                if k is None or k >= st["n_own"]:
                    continue
                target = int(r[6])
                lab = relabel(int(r[2]), int(r[3]), target in index, target in trees)
                c = self.order_index.get(lab) if lab is not None else None
                if c is None or redundant(lab[0], lab[1], int(st["sel"][k, C_ORDER])):
                    continue
                out["y_order"][t, k] = c
                out["y_ptr"][t, k] = -1
                out["y_x"][t, k] = out["y_y"][t, k] = -1
                if lab[1] == UNIT:
                    out["y_ptr"][t, k] = index[target]
                elif lab[1] in (POINT, TREE):
                    x, y = (trees[target] if lab[1] == TREE else (int(r[4]), int(r[5])))
                    out["y_x"][t, k] = _bin(np.array(sign * x))
                    out["y_y"][t, k] = _bin(np.array(y))
        return out


class View:
    """What one player sees, step by step (demonstrations, or a live game): the entities, the
    global features, and the upgrades it has researched so far."""

    def __init__(self, enc: Encoder, player: int, sign: float, races: list[int]):
        self.enc, self.player, self.sign, self.races = enc, player, sign, races
        self.upgrades = np.zeros(len(enc.upgrade_index), np.float32)
        self.own_buildings: set[int] = set()
        self.queued: dict[int, int] = {}  # building -> units / research ordered and not yet done
        self.busy: set[int] = set()  # buildings making something

    def record_orders(self, orders) -> None:
        """The player's orders of the last step (unit, order id, kind): train / research orders
        queue at their building (called after step(): they count from the next step on)."""
        for unit, order, kind in orders:
            if kind == 0 and order >= TYPE_CODE and unit in self.own_buildings:
                self.queued[unit] = self.queued.get(unit, 0) + 1

    def step(self, rows: np.ndarray, me: np.ndarray | None, events: np.ndarray, t: int) -> dict:
        """rows: this step's units (UNIT_COLS), me: the player's row (PLAYER_COLS) or None, events:
        this step's (step, kind, a, b, c)."""
        enc, player = self.enc, self.player
        self.own_buildings.update(rows[(rows[:, C_OWNER] == player) & (rows[:, C_FLAGS] & STRUCTURE > 0), C_ID].tolist())
        for r in events:  # production started / ended at a building
            kind, a = int(r[1]), int(r[2])
            if kind in PRODUCTION_START:
                self.busy.add(a)
            elif kind in PRODUCTION_END:
                self.busy.discard(a)
                if self.queued.get(a, 0) > 0:
                    self.queued[a] -= 1
        sel, n_own, f, types, cur = enc.entities(rows, player, self.sign, (self.queued, self.busy))
        for r in events:  # research done (by one of the player's buildings)
            if r[1] == RESEARCH_FINISH and int(r[2]) in self.own_buildings and int(r[3]) in enc.upgrade_index:
                self.upgrades[enc.upgrade_index[int(r[3])]] = r[4] / 3.0
        g = np.zeros(enc.G, np.float32)
        if me is not None:
            _, _, gold, lumber, fu, fc, upkeep = me[:7]
            g[:6] = gold / 1000.0, lumber / 1000.0, fu / 100.0, fc / 100.0, upkeep / 2.0, t / 1800.0
        g[6 + self.races[player]] = 1
        g[6 + len(RACES) + self.races[1 - player]] = 1
        g[6 + 2 * len(RACES):] = self.upgrades
        return {"n": len(sel), "sel": sel, "n_own": n_own, "ent": f, "type": types, "cur": cur, "glob": g,
                "index": {int(i): k for k, i in enumerate(sel[:, C_ID])}}
