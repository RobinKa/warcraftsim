"""Hero and item facts for the policy's hero inputs (features.A_HERO, selfplay --inputs 2) and its mask of
the orders a hero can't carry out now (model.FullGameNet.valid, --valid-mask): each hero type's abilities
per slot (as the harness reports their levels and cooldowns: data.abilities.hero_ability_table), with their
levels, the hero level each ability level needs and their mana costs; and what an item in a hero's inventory
does (usable, heals, restores mana, a town portal: Units\\ItemData.slk's abilities)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from .. import paths

# item abilities (ItemData.slk abilList) by what they do
HEAL_ABILS = {"AIh1", "AIh2", "AIh3", "AIhx", "AIrl", "AIha", "AIre", "AIra", "AIsl", "AIhw", "AIpz",
              *(f"AIp{k}" for k in range(1, 7))}  # (potions, salves, scrolls, wards; replenishment: both)
MANA_ABILS = {"AIm1", "AIm2", "AIpr", "AIpl", "AImr", "AIre", "AIra", "AIrn", *(f"AIp{k}" for k in range(1, 7))}
PORTAL_ABILS = {"AItp"}
SLOTS, ITEM_SLOTS = 4, 6


def fourcc(code: str) -> int:
    return int.from_bytes(code.encode("latin-1"), "big")


@dataclass(frozen=True)
class Ability:
    code: str
    order: str | None  # the cast's order string (None: passive)
    toggles: tuple[str, ...]  # its other order strings (autocast on / off, undo): need it learned only
    levels: int
    req_level: int  # the hero level its first level needs
    level_skip: int  # hero levels between its levels
    mana: tuple[float, ...]  # per level

    def hero_level_for(self, level: int) -> int:
        return self.req_level + (level - 1) * (self.level_skip or 2)


@lru_cache(1)
def hero_table() -> dict[int, tuple[Ability | None, ...]]:
    """Hero type id -> its abilities per slot (None: no ability there), the harness's slots."""
    from ..data.abilities import ability_info, hero_ability_table
    from ..data.objects import ability_orders
    info, orders = ability_info(), ability_orders()
    out = {}
    for hero, codes in hero_ability_table().items():
        slots = []
        for c in (tuple(codes) + (None,) * SLOTS)[:SLOTS]:
            a = info.get(c) if c else None
            if a is None:
                slots.append(None)
                continue
            o = orders.get(c, {})
            cast = a.order if a.castable else None
            slots.append(Ability(c, cast, tuple(v for k, v in sorted(o.items()) if k != "Order" and v != cast),
                                 a.levels, a.req_level, a.level_skip, tuple(a.mana)))
        out[fourcc(hero)] = tuple(slots)
    return out


@lru_cache(1)
def item_table() -> dict[int, tuple[int, int, int, int]]:
    """Item type id -> (usable, heals, restores mana, town portal), 0 or 1 each."""
    cache = paths.CACHE_DIR / "item_kinds.json"
    if cache.exists():
        return {int(k): tuple(v) for k, v in json.loads(cache.read_text()).items()}
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        rows = parse_slk(g.read("Units\\ItemData.slk").decode("latin-1"))
    out = {}
    for r in rows:
        code = r.get("itemID") or ""
        if len(code) != 4:
            continue
        abils = {a.strip() for a in str(r.get("abilList") or "").split(",")}
        out[fourcc(code)] = (int(r.get("usable") == "1"), int(bool(abils & HEAL_ABILS)), int(bool(abils & MANA_ABILS)),
                             int(bool(abils & PORTAL_ABILS)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({str(k): v for k, v in out.items()}))
    return out


def order_needs(vocab: dict) -> dict[tuple[int, int], int]:
    """(unit type index, order class) -> what the order needs of the unit (model.FullGameNet's "valid_need"):
    0..3 the ability in that slot ready (learned, cooled down, the mana), 4..7 learned (its toggles),
    8..11 a skill point it may spend there (learning it), 12..17 a usable item in that inventory slot
    (useslot1..6, any unit type: index -1)."""
    from . import features as fx
    kinds = [0] + [int(k) for _, k in vocab["orders"]]
    names = {v: int(k) for k, v in (vocab.get("order_names") or {}).items()}  # (order string -> id)
    index = {(int(o), int(k)): c for c, (o, k) in enumerate(vocab["orders"], start=1)}
    by_order: dict[int, list[int]] = {}
    for (o, k), c in index.items():
        by_order.setdefault(o, []).append(c)
    out: dict[tuple[int, int], int] = {}
    heroes = hero_table()
    for t, code in enumerate(vocab["types"], start=1):
        for k, a in enumerate(heroes.get(int(code), ())):
            if a is None:
                continue
            c = index.get((fourcc(a.code), fx.SKILL))
            if c is not None:
                out[(t, c)] = 8 + k
            for s, need in [(a.order, k)] + [(s, 4 + k) for s in a.toggles]:
                for c in by_order.get(names.get(s, -1), ()) if s else ():
                    if kinds[c] != fx.SKILL:
                        out[(t, c)] = need
    for k in range(ITEM_SLOTS):
        for c in by_order.get(852008 + k, ()):
            out[(-1, c)] = 12 + k
    return out
