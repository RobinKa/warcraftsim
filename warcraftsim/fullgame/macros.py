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

A vocabulary gets the order once (extend; "macros" marks it), and a network loaded with --macros grows to it (grow:
its row in the order head starts rarer than any order the policy gives, its embedding the build orders' mean)."""

from __future__ import annotations

import numpy as np
import torch

from . import features as fx

EXPAND = 990001  # (no game order has these ids: the agent's own)
CREEP = 990002
MIX = 990003
ORDERS = (EXPAND, CREEP, MIX)
HALLS = {"human": "htow", "orc": "ogre", "undead": "unpl", "nightelf": "etol"}
WORKERS = ("hpea", "opeo", "uaco", "ewsp")
MINES = {int.from_bytes(c.encode(), "big") for c in ("ngol", "ugol", "egol")}
HAUNTED = int.from_bytes(b"ugol", "big")
NATURAL_FROM_START = 1500.0  # (the main mine is nearer its start)
CAMP_NEAR = 900.0  # creeps this near the natural's mine guard it
TAKEN_NEAR = 1000.0  # a hall or a haunted mine this near it: taken
# where people built their halls at Echo Isles' natural expansions (95 games of 1.29: the mine -> the hall's offset)
EXPAND_SPOTS = {(-4864, -3200): (100, 800), (4352, -3200): (-300, 700)}


def code(c: str) -> int:
    return int.from_bytes(c.encode(), "big")


def extended(vocab: dict) -> bool:
    have = {tuple(o) for o in vocab["orders"]}
    return all((o, fx.IMMEDIATE) in have for o in ORDERS)


def extend(vocab: dict) -> dict:
    """`vocab` with the macro orders at its end (immediate orders) -> a new vocabulary."""
    if extended(vocab):
        return vocab
    orders = [list(o) for o in vocab["orders"]]
    for o in ORDERS:  # (in this order: a vocabulary with the first ones gets the later ones after them)
        if [o, fx.IMMEDIATE] not in orders:
            orders.append([o, fx.IMMEDIATE])
    names = {**(vocab.get("order_names") or {}), str(EXPAND): "expand", str(CREEP): "creep the natural",
             str(MIX): "train the target's mix"}
    counts = dict(vocab.get("counts") or {})
    if "orders" in counts:
        counts["orders"] = list(counts["orders"]) + [0] * (len(orders) - len(counts["orders"]))
    return {**vocab, "orders": orders, "order_names": names, "counts": counts, "macros": 1}


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
    and is no worker."""
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


def natural(rows: np.ndarray, start: tuple[float, float] | None):
    """The player's natural expansion's mine row (the gold mine nearest its start but its main), or None."""
    if start is None:
        return None
    sx, sy = start
    mines = rows[np.isin(rows[:, fx.C_TYPE], list(MINES)) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]
    if not len(mines):
        return None
    d = np.hypot(mines[:, fx.C_X] - sx, mines[:, fx.C_Y] - sy)
    order = np.argsort(d)
    far = order[d[order] > NATURAL_FROM_START]
    return mines[far[0]] if len(far) else None


def camp(rows: np.ndarray, start: tuple[float, float] | None) -> tuple[float, float] | None:
    """The natural's mine (x, y) while its camp stands and nobody holds the natural (creep: where to go), or None."""
    nat = natural(rows, start)
    if nat is None:
        return None
    mx, my = float(nat[fx.C_X]), float(nat[fx.C_Y])
    alive = rows[rows[:, fx.C_FLAGS] & fx.DEAD == 0]
    dist = np.hypot(alive[:, fx.C_X] - mx, alive[:, fx.C_Y] - my)
    halls = np.isin(alive[:, fx.C_TYPE], [code(h) for h in HALLS.values()] + [HAUNTED]) & (alive[:, fx.C_OWNER] <= 1)
    if (halls & (dist < TAKEN_NEAR)).any() or not ((alive[:, fx.C_OWNER] == 12) & (dist < CAMP_NEAR)).any():
        return None
    return mx, my


def site(rows: np.ndarray, start: tuple[float, float] | None) -> tuple[int, float, float, float, float] | None:
    """The player's natural expansion if it is free (rows: the step's units, start: its start location): (its mine's
    id, the mine's x and y, the hall's x and y), or None (no natural, taken, or its camp stands)."""
    if start is None:
        return None
    sx, sy = start
    mines = rows[np.isin(rows[:, fx.C_TYPE], list(MINES)) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]
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
    halls = np.isin(alive[:, fx.C_TYPE], [code(h) for h in HALLS.values()] + [HAUNTED]) & (alive[:, fx.C_OWNER] <= 1)
    if (halls & (dist < TAKEN_NEAR)).any() or ((alive[:, fx.C_OWNER] == 12) & (dist < CAMP_NEAR)).any():
        return None
    ox, oy = EXPAND_SPOTS.get((int(round(mx)), int(round(my))), (None, None))
    if ox is None:  # (another map's: 750 toward its start)
        n = max(float(np.hypot(sx - mx, sy - my)), 1.0)
        ox, oy = 750.0 * (sx - mx) / n, 750.0 * (sy - my) / n
    return int(nat[fx.C_ID]), mx, my, mx + ox, my + oy
