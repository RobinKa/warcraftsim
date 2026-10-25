"""The agent's own orders for what the game spreads over many (selfplay --macros 1).

    creep    an army unit or a hero attack-moves to the natural expansion's camp, while the camp stands (and nobody
             holds the natural). The lesson that sent the whole army there every 10 s (fgself-123) lost its games;
             as an order, the policy decides who goes and when.
    mix      a building trains its target's army mix: the unit the target's army has the largest share of more than the
             side's own army, of those the building makes and the player can pay for now (with a target that names
             an army). Taught one unit at a time (fgself-131), the army-mix gap did not move: a handful of lessons a
             game against the hundreds of train orders the policy gives itself.
    expand   a worker takes the natural expansion: the race's hall built where people build it (Echo Isles: ~800
             north of the natural's mine), or for undead the mine haunted. Taught as a raw build order at a point
             (fgself-123/124), the policy learned "build a hall" and built it at home: 26-30% of its games had a
             second hall there, 0.2% an expansion. It may be given while the natural is free (no hall or haunted mine
             there, its camp gone) and the player can pay for the hall; the agent then sends the build order.
    pick up  (--macros 2) a hero picks up the nearest item lying within PICKUP_NEAR (a creep's drop: Echo Isles' camps
             drop one each, a permanent item, a potion or a tome, and the policy had no way to take them): the harness
             reports the items on the ground its player sees (protocol.Observation.ground) and right-clicks the nearest
             (protocol.PickUp). It may be given while a hero of the player's has one that near (and a free slot, but
             for a powerup).

A vocabulary gets the orders of its level once (extend; "macros" marks the level), and a network loaded with --macros
grows to them (grow:
its row in the order head starts rarer than any order the policy gives, its embedding the build orders' mean)."""

from __future__ import annotations

import numpy as np
import torch

from . import features as fx

EXPAND = 990001  # (no game order has these ids: the agent's own)
CREEP = 990002
MIX = 990003
PICKUP = 990004
ORDERS = (EXPAND, CREEP, MIX)  # --macros 1
LEVELS = {1: ORDERS, 2: ORDERS + (PICKUP,)}
PICKUP_NEAR = 1000.0  # (a hero this near an item may pick it up: the harness takes the nearest within it)
HALLS = {"human": "htow", "orc": "ogre", "undead": "unpl", "nightelf": "etol"}
WORKERS = ("hpea", "opeo", "uaco", "ewsp")
MINES = {int.from_bytes(c.encode(), "big") for c in ("ngol", "ugol", "egol")}
HAUNTED = int.from_bytes(b"ugol", "big")
# what holds a natural: a hall of any tier, a haunted or an entangled mine (a keep there had read as a free natural)
HOLDS = [int.from_bytes(c.encode(), "big") for c in
         ("htow", "hkee", "hcas", "ogre", "ostr", "ofrt", "unpl", "unp1", "unp2", "etol", "etoa", "etoe", "ugol", "egol")]
MINE_CODES, HOLD_CODES = fx.codes(MINES), fx.codes(HOLDS)  # (as features.isin's sorted arrays)
NATURAL_FROM_START = 1500.0  # (the main mine is nearer its start)
CAMP_NEAR = 900.0  # creeps this near the natural's mine guard it
TAKEN_NEAR = 1000.0  # a hall or a haunted mine this near it: taken
# where people built their halls at Echo Isles' natural expansions (95 games of 1.29: the mine -> the hall's offset)
EXPAND_SPOTS = {(-4864, -3200): (100, 800), (4352, -3200): (-300, 700)}


def code(c: str) -> int:
    return int.from_bytes(c.encode(), "big")


def orders(level: int | None = None) -> tuple[int, ...]:
    """The macro orders of a --macros level (None: this process's, features.MACROS, at least 1)."""
    level = (fx.MACROS or 1) if level is None else level
    return LEVELS[min(int(level), max(LEVELS))] if level else ()


def extended(vocab: dict, level: int | None = None) -> bool:
    have = {tuple(o) for o in vocab["orders"]}
    return all((o, fx.IMMEDIATE) in have for o in orders(level))


def extend(vocab: dict, level: int | None = None) -> dict:
    """`vocab` with the macro orders of a level (None: this process's) at its end (immediate orders) -> a new
    vocabulary."""
    level = int((fx.MACROS or 1) if level is None else level)
    if extended(vocab, level):  # (its level noted even when it adds no order: --macros 3's skill points; a
        return vocab if int(vocab.get("macros") or 0) >= level else {**vocab, "macros": level}  # checkpoint plays
        # alone at its own level, model.configure_for; audit 5)
    out = [list(o) for o in vocab["orders"]]
    for o in orders(level):  # (in this order: a vocabulary with the first ones gets the later ones after them)
        if [o, fx.IMMEDIATE] not in out:
            out.append([o, fx.IMMEDIATE])
    names = {**(vocab.get("order_names") or {}), str(EXPAND): "expand", str(CREEP): "creep the natural",
             str(MIX): "train the target's mix", str(PICKUP): "pick up an item"}
    counts = dict(vocab.get("counts") or {})
    if "orders" in counts:
        counts["orders"] = list(counts["orders"]) + [0] * (len(out) - len(counts["orders"]))
    return {**vocab, "orders": out, "order_names": names, "counts": counts,
            "macros": max(level, int(vocab.get("macros") or 0))}


@torch.no_grad()
def grow(net, old: dict, new: dict):
    """`net` (of vocabulary `old`) with the orders `new` adds: their order-head rows zero with the bias below every
    order the policy gives now, their embeddings the build orders' mean; workers may give them (allow)."""
    from .model import FullGameNet
    O0, O1 = len(old["orders"]) + 1, len(new["orders"]) + 1
    if O0 == O1:
        return net
    config = {**net.config, "n_orders": O1}
    valid = config.pop("valid", 0)
    out = FullGameNet(**config).to(net.order_kind.device)
    state = {k: v for k, v in net.state_dict().items() if not k.startswith("valid_")}
    w, b = state["order.2.weight"], state["order.2.bias"]
    state["order.2.weight"] = torch.cat([w, w.new_zeros(O1 - O0, w.shape[1])])
    state["order.2.bias"] = torch.cat([b, (torch.quantile(b.float(), 0.05) - 2.0).to(b.dtype).expand(O1 - O0)])
    oe = state["order_emb.weight"]
    builds = [c for c, (o, k) in enumerate(old["orders"], start=1) if o >= fx.TYPE_CODE and k == fx.POINT]
    state["order_emb.weight"] = torch.cat([oe, oe[builds].mean(0).expand(O1 - O0, -1).clone()])
    allowed = state["allowed"]
    state["allowed"] = torch.cat([allowed, allowed.new_zeros(allowed.shape[0], O1 - O0)], 1)
    state["order_kind"] = torch.as_tensor([0] + [k for _, k in new["orders"]], device=oe.device)
    out.load_state_dict(state, strict=False)
    if valid:
        out.set_valid(new)
    allow(out, new)
    return out.train(net.training)


def allow(net, vocab: dict) -> None:
    """Who may give the macro orders (net's allowed table, in place): expand a worker, creep whatever may attack-move
    and is no worker, mix a building that trains army units, pick up a hero."""
    oix = {tuple(o): c for c, o in enumerate(vocab["orders"], start=1)}
    tix = {int(t): i + 1 for i, t in enumerate(vocab["types"])}
    k = oix.get((EXPAND, fx.IMMEDIATE))
    if k is not None:
        for w in WORKERS:
            if code(w) in tix:
                net.allowed[tix[code(w)], k] = True
    k, move = oix.get((CREEP, fx.IMMEDIATE)), oix.get((851983, fx.POINT))
    if k is not None and move is not None:
        workers = [tix[code(w)] for w in WORKERS if code(w) in tix]
        net.allowed[:, k] = net.allowed[:, move]
        net.allowed[workers, k] = False
    k = oix.get((MIX, fx.IMMEDIATE))
    if k is not None:  # (mix: a building that trains army units)
        for b in {m for u, ms in _makers().items() if army(u) for m in ms}:
            if code(b) in tix:
                net.allowed[tix[code(b)], k] = True
    k = oix.get((PICKUP, fx.IMMEDIATE))
    if k is not None:
        for t, i in tix.items():
            if fx.is_hero_code(fx.rawcode(t)):
                net.allowed[i, k] = True


def army(c: str) -> bool:
    """An army unit's code (lower case, no worker; buildings are trained by no one)."""
    return c[:1].islower() and c not in WORKERS


def _makers() -> dict[str, tuple[str, ...]]:
    from .techtree import tree
    return {u: ms for u, ms in tree().makers.items() if army(u) and all(m[:1].islower() for m in ms)
            and not any(u in ups for ups in tree().upgrades.values())}


def mix_choice(target: dict[str, float], have: dict[str, float], building: str, can) -> str | None:
    """mix: the unit `building` trains for its target's army (`target`: each unit's share of it; `have`: the side's
    own army shares): of those it makes and `can(unit)` allows now, the one most lacking; None if none lacks."""
    makes = [u for u, ms in _makers().items() if building in ms and u in target]
    best = max(((target[u] - have.get(u, 0.0), u) for u in makes if can(u)), default=None)
    return best[1] if best is not None and best[0] > 0 else None


def pickable(rows: np.ndarray, player: int, ground) -> list[int]:
    """pick up: the player's heroes (their ids) with an item lying within PICKUP_NEAR that the player sees
    (`ground`: protocol.Observation.ground) and a free slot for it (none needed for a powerup)."""
    bit = 1 << player
    items = [(x, y, pw) for _, x, y, vis, pw in ground if vis & bit]
    if not items:
        return []
    out = []
    heroes = np.flatnonzero((rows[:, fx.C_OWNER] == player) & (rows[:, fx.C_FLAGS] & fx.HERO != 0)
                            & (rows[:, fx.C_FLAGS] & fx.DEAD == 0))
    for i in heroes.tolist():
        free = rows.shape[1] <= fx.C_ITEM0 or bool((rows[i, fx.C_ITEM0:fx.C_ITEM0 + 6] == 0).any())
        hx, hy = float(rows[i, fx.C_X]), float(rows[i, fx.C_Y])
        if any((free or pw) and (x - hx) ** 2 + (y - hy) ** 2 < PICKUP_NEAR ** 2 for x, y, pw in items):
            out.append(int(rows[i, fx.C_ID]))
    return out


def _native(rows: np.ndarray, start: tuple[float, float]) -> list | None:
    """featnative.natural for these rows and start (the natural's row, id, x, y, taken, its camp), or None: the numpy
    code (no library, or two mines as near)."""
    return _fn().natural(rows, start, MINE_CODES, HOLD_CODES, NATURAL_FROM_START, TAKEN_NEAR, CAMP_NEAR)


_FN: list = []


def _fn():
    if not _FN:
        from . import featnative
        _FN.append(featnative)
    return _FN[0]


def natural(rows: np.ndarray, start: tuple[float, float] | None):
    """The player's natural expansion's mine row (the gold mine nearest its start but its main), or None."""
    if start is None:
        return None
    got = _native(rows, start)
    if got is not None:
        return rows[got[0]] if got[0] >= 0 else None
    sx, sy = start
    mines = rows[fx.isin(rows[:, fx.C_TYPE], MINE_CODES) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]
    if not len(mines):
        return None
    d = np.hypot(mines[:, fx.C_X] - sx, mines[:, fx.C_Y] - sy)
    order = np.argsort(d)
    far = order[d[order] > NATURAL_FROM_START]
    return mines[far[0]] if len(far) else None


def camp(rows: np.ndarray, start: tuple[float, float] | None) -> tuple[float, float] | None:
    """The natural's mine (x, y) while its camp stands and nobody holds the natural (creep: where to go), or None."""
    got = _native(rows, start) if start is not None else None
    if got is not None:
        return (float(got[2]), float(got[3])) if got[0] >= 0 and not got[4] and got[5] else None
    nat = natural(rows, start)
    if nat is None:
        return None
    mx, my = float(nat[fx.C_X]), float(nat[fx.C_Y])
    alive = rows[rows[:, fx.C_FLAGS] & fx.DEAD == 0]
    dist = np.hypot(alive[:, fx.C_X] - mx, alive[:, fx.C_Y] - my)
    halls = fx.isin(alive[:, fx.C_TYPE], HOLD_CODES) & (alive[:, fx.C_OWNER] <= 1)
    if (halls & (dist < TAKEN_NEAR)).any() or not ((alive[:, fx.C_OWNER] == 12) & (dist < CAMP_NEAR)).any():
        return None
    return mx, my


def site(rows: np.ndarray, start: tuple[float, float] | None) -> tuple[int, float, float, float, float] | None:
    """The player's natural expansion if it is free (rows: the step's units, start: its start location): (its mine's
    id, the mine's x and y, the hall's x and y), or None (no natural, taken, or its camp stands)."""
    if start is None:
        return None
    sx, sy = start
    got = _native(rows, start)
    if got is not None:  # (in C: the same mine, the same tests)
        if got[0] < 0 or got[4] or got[5]:
            return None
        nat_id, mx, my = got[1], float(got[2]), float(got[3])
    else:
        mines = rows[fx.isin(rows[:, fx.C_TYPE], MINE_CODES) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]
        if not len(mines):
            return None
        d = np.hypot(mines[:, fx.C_X] - sx, mines[:, fx.C_Y] - sy)
        order = np.argsort(d)
        far = order[d[order] > NATURAL_FROM_START]
        if not len(far):
            return None
        nat = mines[far[0]]
        mx, my = float(nat[fx.C_X]), float(nat[fx.C_Y])
        alive = rows[rows[:, fx.C_FLAGS] & fx.DEAD == 0]
        dist = np.hypot(alive[:, fx.C_X] - mx, alive[:, fx.C_Y] - my)
        halls = fx.isin(alive[:, fx.C_TYPE], HOLD_CODES) & (alive[:, fx.C_OWNER] <= 1)
        if (halls & (dist < TAKEN_NEAR)).any() or ((alive[:, fx.C_OWNER] == 12) & (dist < CAMP_NEAR)).any():
            return None
        nat_id = int(nat[fx.C_ID])
    ox, oy = EXPAND_SPOTS.get((int(round(mx)), int(round(my))), (None, None))
    if ox is None:  # (another map's: 750 toward its start)
        n = max(float(np.hypot(sx - mx, sy - my)), 1.0)
        ox, oy = 750.0 * (sx - mx) / n, 750.0 * (sy - my) / n
    return nat_id, mx, my, mx + ox, my + oy
