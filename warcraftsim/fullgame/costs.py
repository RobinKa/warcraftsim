"""What each order class costs (gold, lumber, food), for the availability mask: an order the
player can't pay for yet is masked (the clone's train / research / build orders were refused 58%
and 53% of the time, mostly for that).

Costs come from the game's tables (UnitBalance.slk, UpgradeData.slk) scaled by the map's rules
(a duel map's Rules.cost; duelmap.rules_files scales them the same way). Food is the unit's
food use. Orders that aren't a unit, building or upgrade cost nothing. A building upgraded from
another (a keep from a town hall, a guard tower from a scout tower: techtree's "Upgrade") costs the
difference: the table holds both together (a keep 705 gold, its upgrade 320), and until 2026-10-19
the mask held a policy's next tier back until it had both. A research with several levels costs more at each
(UpgradeData goldmod, lumbermod): order_costs holds its first level's, research_steps the step per level, and a
view costs it at the level the player would research next (features.Encoder.at_levels: the mask, the agent's step
budget, the labels' paid orders).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import features as fx


def _num(row: dict, key: str) -> int:
    try:
        return int(float(row.get(key) or 0))
    except ValueError:
        return 0


def order_costs(vocab: dict, map_name: str = "duelrush") -> np.ndarray:
    """[n_orders, 3]: gold, lumber and food of each order class (row 0: no order)."""
    from ..data.duelmap import map_rules
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    from .techtree import tree
    rules = map_rules(map_name)
    base = {to: frm for frm, tos in tree().upgrades.items() for to in tos}  # (an upgrade -> its building)
    with GameArchives() as g:
        units = {r.get("unitBalanceID"): r for r in parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))}
        upgrades = {r.get("upgradeid"): r for r in parse_slk(g.read("Units\\UpgradeData.slk").decode("latin-1"))}
        items = {r.get("itemID"): r for r in parse_slk(g.read("Units\\ItemData.slk").decode("latin-1"))}

    def scaled(v: int) -> int:  # as duelmap.rules_files: max(1, round(v * f)), unchanged at 1
        return v if rules.cost == 1.0 or v <= 0 else max(1, round(v * rules.cost))

    out = np.zeros((len(vocab["orders"]) + 1, 3), np.int64)
    for c, (oid, kind) in enumerate(vocab["orders"], start=1):
        if oid < fx.TYPE_CODE or kind not in (fx.IMMEDIATE, fx.POINT, fx.UNIT):
            continue
        code = fx.rawcode(oid)
        if kind == fx.UNIT:  # (bought at a neutral building, protocol.Buy: a tavern's hero, a mercenary, an item; or
            if code in units:  # built on a unit: a haunted gold mine) a unit's scaled as the map's tables (duelmap:
                r = units[code]  # NEUTRAL_SOLD), an item's not
                out[c] = (scaled(_num(r, "goldcost")), scaled(_num(r, "lumbercost")), _num(r, "fused"))
            elif code in items:
                out[c] = (_num(items[code], "goldcost"), _num(items[code], "lumbercost"), 0)
            continue
        if code in units:
            r = units[code]
            out[c] = (scaled(_num(r, "goldcost")), scaled(_num(r, "lumbercost")), _num(r, "fused"))
            if base.get(code) in units:  # an upgrade: the difference (each scaled, as the map's tables)
                b = units[base[code]]
                out[c, :2] -= (scaled(_num(b, "goldcost")), scaled(_num(b, "lumbercost")))
        elif code in upgrades:  # the first level's cost (later levels cost more: research_steps, Encoder.at_levels)
            r = upgrades[code]
            out[c] = (scaled(_num(r, "goldbase")), scaled(_num(r, "lumberbase")), 0)
    return out


def research_steps(vocab: dict, map_name: str = "duelrush") -> np.ndarray:
    """[n_orders, 2]: how much more gold and lumber each research order's next level costs per level the player has
    of it (UpgradeData goldmod and lumbermod, scaled as the map's tables scale them: duelmap.rules_files), zeros for
    other orders. Level n costs goldbase + (n - 1) * goldmod: iron forged swords 100/50, then 175/175, then 250/300.
    (order_costs holds the first level's; until 2026-10-26 every level was costed so, and the mask let the policy
    order a second level the game then refused for want of money.) Read-only, cached per vocabulary and map."""
    return _research_steps(tuple(tuple(int(x) for x in o) for o in vocab["orders"]), map_name)


@lru_cache(8)
def _research_steps(orders: tuple, map_name: str) -> np.ndarray:
    from ..data.duelmap import map_rules
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    from .techtree import research_levels
    rules = map_rules(map_name)
    with GameArchives() as g:
        upgrades = {r.get("upgradeid"): r for r in parse_slk(g.read("Units\\UpgradeData.slk").decode("latin-1"))}

    def scaled(v: int) -> int:  # (as order_costs' and duelmap.rules_files')
        return v if rules.cost == 1.0 or v <= 0 else max(1, round(v * rules.cost))
    levels = research_levels()
    out = np.zeros((len(orders) + 1, 2), np.int64)
    for c, (oid, kind) in enumerate(orders, start=1):
        if oid < fx.TYPE_CODE or kind != fx.IMMEDIATE:
            continue
        code = fx.rawcode(oid)
        if code in upgrades and levels.get(code, 1) > 1:
            out[c] = (scaled(_num(upgrades[code], "goldmod")), scaled(_num(upgrades[code], "lumbermod")))
    out.flags.writeable = False
    return out


def at_levels(costs: np.ndarray, steps: np.ndarray, rows, levels: dict) -> np.ndarray:
    """`costs` with each research's order (`rows`: its class and research id, those with steps) at the cost of the
    next level the player would research (`levels`: research id -> the level it has) -> `costs` itself when no
    such research has a level yet, else a copy."""
    out = costs
    for k, rid in rows:
        n = levels.get(rid, 0)
        if n > 0:
            if out is costs:
                out = np.array(costs, copy=True)
            out[k, 0] += n * steps[k, 0]
            out[k, 1] += n * steps[k, 1]
    return out


def available(costs: np.ndarray, gold: int, lumber: int, food_used: int, food_cap: int) -> np.ndarray:
    """[n_orders] bool: the order classes the player can pay for now (no order: always)."""
    food_cap = min(food_cap, 100)  # (the game's ceiling: the cap it reports counts farms past it)
    ok = (costs[:, 0] <= gold) & (costs[:, 1] <= lumber) & ((costs[:, 2] == 0) | (food_used + costs[:, 2] <= food_cap))
    ok[0] = True
    return ok
