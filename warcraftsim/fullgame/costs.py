"""What each order class costs (gold, lumber, food), for the availability mask: an order the
player can't pay for yet is masked (the clone's train / research / build orders were refused 58%
and 53% of the time, mostly for that).

Costs come from the game's tables (UnitBalance.slk, UpgradeData.slk) scaled by the map's rules
(a duel map's Rules.cost; duelmap.rules_files scales them the same way). Food is the unit's
food use. Orders that aren't a unit, building or upgrade cost nothing.
"""

from __future__ import annotations

import numpy as np

from . import features as fx


def _num(row: dict, key: str) -> int:
    try:
        return int(float(row.get(key) or 0))
    except ValueError:
        return 0


def order_costs(vocab: dict, map_name: str = "duelrush") -> np.ndarray:
    """[n_orders, 3]: gold, lumber and food of each order class (row 0: no order)."""
    from ..data.duelmap import Rules, parse_duel_name
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    parsed = parse_duel_name(map_name)
    rules = parsed[1] if parsed else Rules()
    with GameArchives() as g:
        units = {r.get("unitBalanceID"): r for r in parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))}
        upgrades = {r.get("upgradeid"): r for r in parse_slk(g.read("Units\\UpgradeData.slk").decode("latin-1"))}

    def scaled(v: int) -> int:  # as duelmap.rules_files: max(1, round(v * f)), unchanged at 1
        return v if rules.cost == 1.0 or v <= 0 else max(1, round(v * rules.cost))

    out = np.zeros((len(vocab["orders"]) + 1, 3), np.int64)
    for c, (oid, kind) in enumerate(vocab["orders"], start=1):
        if oid < fx.TYPE_CODE or kind not in (fx.IMMEDIATE, fx.POINT):
            continue
        code = fx.rawcode(oid)
        if code in units:
            r = units[code]
            out[c] = (scaled(_num(r, "goldcost")), scaled(_num(r, "lumbercost")), _num(r, "fused"))
        elif code in upgrades:  # the first level's cost (later levels cost more: then the game refuses)
            r = upgrades[code]
            out[c] = (scaled(_num(r, "goldbase")), scaled(_num(r, "lumberbase")), 0)
    return out


def available(costs: np.ndarray, gold: int, lumber: int, food_used: int, food_cap: int) -> np.ndarray:
    """[n_orders] bool: the order classes the player can pay for now (no order: always)."""
    ok = (costs[:, 0] <= gold) & (costs[:, 1] <= lumber) & ((costs[:, 2] == 0) | (food_used + costs[:, 2] <= food_cap))
    ok[0] = True
    return ok
