"""Hero abilities: which abilities each hero type has, and how each is cast.

The numbers (mana, cooldown, range, area per level; levels and the hero level they need) come
from AbilityData.slk, the order strings from the *AbilityFunc.txt files. How an ability is cast
is not in the data (it follows from the ability's base code), so it is listed here for the
melee heroes:

    cast  "unit" (a target unit), "point" (a target point), "instant" (no target) or "passive"
    side  whom it is meant for: "enemy", "ally" (own units, the hero included) or "self";
          "summon" and "utility" abilities work but are left out of random skill builds
          (summons add units mid-episode; far sight, blink, sacrifices do not fight)

Slot k of a hero is the k-th entry of its heroAbilList (UnitAbilities.slk); the harness reports
each slot's level and remaining cooldown for heroes in that order (HERO_ABILITY_SLOTS slots).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache

from .. import paths
from ..protocol import HERO_ABILITY_SLOTS
from .mpq import GameArchives
from .objects import ability_orders, parse_slk

_CAST: dict[str, tuple[str, str]] = {
    # Paladin: Holy Light heals a friendly living unit (it also hurts undead enemies)
    "AHhb": ("unit", "ally"), "AHds": ("instant", "self"), "AHre": ("instant", "ally"), "AHad": ("passive", ""),
    # Archmage
    "AHbz": ("point", "enemy"), "AHab": ("passive", ""), "AHwe": ("instant", "summon"), "AHmt": ("unit", "utility"),
    # Mountain King
    "AHtc": ("instant", "enemy"), "AHtb": ("unit", "enemy"), "AHbh": ("passive", ""), "AHav": ("instant", "self"),
    # Blood Mage
    "AHfs": ("point", "enemy"), "AHbn": ("unit", "enemy"), "AHdr": ("unit", "enemy"), "AHpx": ("instant", "summon"),
    # Blademaster
    "AOwk": ("instant", "self"), "AOcr": ("passive", ""), "AOmi": ("instant", "summon"), "AOww": ("instant", "enemy"),
    # Far Seer
    "AOfs": ("point", "utility"), "AOsf": ("instant", "summon"), "AOcl": ("unit", "enemy"), "AOeq": ("point", "enemy"),
    # Tauren Chieftain
    "AOsh": ("point", "enemy"), "AOae": ("passive", ""), "AOre": ("passive", ""), "AOws": ("instant", "enemy"),
    # Shadow Hunter
    "AOhw": ("unit", "ally"), "AOhx": ("unit", "enemy"), "AOsw": ("point", "summon"), "AOvd": ("instant", "ally"),
    # Death Knight: Death Coil hurts a living enemy (it also heals undead allies)
    "AUdc": ("unit", "enemy"), "AUdp": ("unit", "utility"), "AUau": ("passive", ""), "AUan": ("instant", "summon"),
    # Lich
    "AUfn": ("unit", "enemy"), "AUfu": ("unit", "ally"), "AUdr": ("unit", "utility"), "AUdd": ("point", "enemy"),
    # Dreadlord
    "AUav": ("passive", ""), "AUsl": ("unit", "enemy"), "AUcs": ("point", "enemy"), "AUin": ("point", "summon"),
    # Crypt Lord
    "AUim": ("point", "enemy"), "AUts": ("passive", ""), "AUcb": ("instant", "summon"), "AUls": ("instant", "enemy"),
    # Keeper of the Grove
    "AEer": ("unit", "enemy"), "AEfn": ("point", "summon"), "AEah": ("passive", ""), "AEtq": ("instant", "ally"),
    # Priestess of the Moon: Searing Arrows cast on a unit is one burning attack
    "AHfa": ("unit", "enemy"), "AEst": ("instant", "summon"), "AEar": ("passive", ""), "AEsf": ("instant", "enemy"),
    # Demon Hunter: Immolation toggles on
    "AEmb": ("unit", "enemy"), "AEim": ("instant", "enemy"), "AEev": ("passive", ""), "AEme": ("instant", "self"),
    # Warden
    "AEbl": ("point", "utility"), "AEfk": ("instant", "enemy"), "AEsh": ("unit", "enemy"), "AEsv": ("instant", "summon"),
}
# order strings the func files do not give
_ORDER_OVERRIDES = {"AHdr": "drain"}
# heroes whose abilities are described (the melee heroes)
MELEE_HEROES = ("Hpal", "Hamg", "Hmkg", "Hblm", "Obla", "Ofar", "Otch", "Oshd",
                "Udea", "Ulic", "Udre", "Ucrl", "Ekee", "Emoo", "Edem", "Ewar")


@dataclass(frozen=True)
class AbilityInfo:
    code: str
    name: str
    order: str | None
    cast: str  # "unit", "point", "instant", "passive"
    side: str  # "enemy", "ally", "self", "summon", "utility" ("" when passive)
    levels: int
    req_level: int  # hero level for ability level 1
    level_skip: int  # hero levels between ability levels
    mana: tuple[float, ...]  # per ability level
    cooldown: tuple[float, ...]
    range: tuple[float, ...]
    area: tuple[float, ...]

    @property
    def castable(self) -> bool:
        return self.cast != "passive" and bool(self.order)

    @property
    def in_builds(self) -> bool:
        """Chosen in random skill builds (see the module doc)."""
        return self.side not in ("summon", "utility")

    def hero_level_for(self, level: int) -> int:
        """The hero level ability level `level` needs."""
        return self.req_level + (level - 1) * (self.level_skip or 2)

    def at(self, values: tuple[float, ...], level: int) -> float:
        return values[min(max(level, 1), len(values)) - 1] if values else 0.0


def _num(v) -> float:
    try:
        return float(str(v).strip())
    except ValueError:
        return 0.0


@lru_cache(maxsize=1)
def _tables() -> dict:
    cache = paths.CACHE_DIR / "hero_abilities-2.json"
    if cache.exists():
        return json.loads(cache.read_text())
    names: dict[str, str] = {}
    with GameArchives() as g:
        units = parse_slk(g.read("Units\\UnitAbilities.slk").decode("latin-1"))
        abils = parse_slk(g.read("Units\\AbilityData.slk").decode("latin-1"))
        for fname in sorted(g.names("Units\\*AbilityStrings.txt"), key=str.lower):
            current = None
            for line in g.read(fname).decode("latin-1").splitlines():
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    current = line[1:-1]
                elif current and line.startswith("Name=") and current not in names:
                    names[current] = line[5:].strip().strip('"')
    heroes = {}
    for r in units:
        uid = r.get("unitAbilID") or ""
        codes = [c.strip() for c in str(r.get("heroAbilList") or "").split(",") if c.strip() and c.strip() != "_"]
        if uid[:1].isupper() and codes:
            heroes[uid] = codes[:HERO_ABILITY_SLOTS]
    wanted = {c for codes in heroes.values() for c in codes}
    rows = {}
    for r in abils:
        code = r.get("alias") or ""
        if code not in wanted:
            continue
        levels = max(int(_num(r.get("levels"))), 1)
        rows[code] = {
            "name": names.get(code, code), "levels": levels, "req_level": int(_num(r.get("reqLevel"))) or 1,
            "level_skip": int(_num(r.get("levelSkip"))),
            **{f: [_num(r.get(f"{col}{k}")) for k in range(1, levels + 1)]
               for f, col in (("mana", "Cost"), ("cooldown", "Cool"), ("range", "Rng"), ("area", "Area"))},
        }
    out = {"heroes": heroes, "abilities": rows}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


@lru_cache(maxsize=1)
def hero_abilities() -> dict[str, tuple[str, ...]]:
    """Hero type -> its ability codes in slot order (heroAbilList; the ultimate is among them)."""
    return {h: tuple(c) for h, c in _tables()["heroes"].items()}


@lru_cache(maxsize=1)
def ability_info() -> dict[str, AbilityInfo]:
    """Ability code -> AbilityInfo for every hero ability (cast "passive" where not described)."""
    orders = ability_orders()
    out = {}
    for code, r in _tables()["abilities"].items():
        cast, side = _CAST.get(code, ("passive", ""))
        order = _ORDER_OVERRIDES.get(code) or orders.get(code, {}).get("Order")
        out[code] = AbilityInfo(code, r["name"], order, cast, side, r["levels"], r["req_level"], r["level_skip"],
                                tuple(r["mana"]), tuple(r["cooldown"]), tuple(r["range"]), tuple(r["area"]))
    return out


def ability_order_strings() -> tuple[str, ...]:
    """Order strings of the described hero abilities (for the harness's order table)."""
    info = ability_info()
    return tuple(sorted({info[c].order for c in _CAST if c in info and info[c].order}))


def hero_ability_table() -> dict[str, tuple[str, ...]]:
    """The hero -> ability slots table the harness gets: the melee heroes (and any other hero
    whose abilities are all described)."""
    return {h: codes for h, codes in hero_abilities().items()
            if h in MELEE_HEROES or all(c in _CAST for c in codes)}


def describe(code: str) -> dict:
    return asdict(ability_info()[code])
