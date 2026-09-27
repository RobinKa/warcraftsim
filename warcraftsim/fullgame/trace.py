"""The per-step trace a whole-game player writes for its video's panel (fullgame.overlay); no
imaging here, so the players can import it without Pillow."""

from __future__ import annotations

from ..protocol import Observation

DEAD_FLAG = 1024  # the unit flags' dead bit (an int test, not an enum operation: per unit per step)


def trace_step(t: int, obs: Observation, material: dict, orders: dict, **policy) -> dict:
    """One step's row: `material` {player: value}, `orders` {player: Counter of order labels},
    `policy`: value / reward / entropy, each {player: float} (the players a policy controls)."""
    res = {str(p): [s.gold, s.lumber, s.food_used, s.food_cap] for p, s in obs.players.items() if p in (0, 1)}
    row = {"t": t, "game_time": round(obs.game_time, 2), "material": {str(p): round(v) for p, v in material.items()},
           "res": res, "orders": {str(p): dict(c) for p, c in orders.items()}}
    for k, v in policy.items():
        row[k] = {str(p): round(float(x), 4) for p, x in v.items()}
    return row


def material(obs, values: dict) -> dict[int, float]:
    """What each player's living units and buildings cost, times their hit points left: {0: .., 1: ..}."""
    out = {0: 0.0, 1: 0.0}
    for u in obs.units:
        if u.owner in (0, 1) and not int(u.flags) & DEAD_FLAG:
            out[u.owner] += values.get(str(u.type_id), 0) * (u.hp / u.max_hp if u.max_hp > 0 else 1.0)
    return out


def unit_values() -> dict[str, int]:
    """What each unit type costs (gold + lumber, the game's own tables): {type id: value}."""
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        rows = parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))
    out = {}
    for r in rows:
        oid = r.get("unitBalanceID", "")
        if len(oid) != 4:
            continue
        try:
            v = int(float(r.get("goldcost") or 0)) + int(float(r.get("lumbercost") or 0))
        except ValueError:
            continue
        if v > 0:
            out[str(int.from_bytes(oid.encode("latin-1"), "big"))] = v
    return out
