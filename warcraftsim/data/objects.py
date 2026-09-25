"""Game object data from the archives: SLK tables (UnitData, UnitBalance, ...).

Used for the unit-type vocabulary of observation features and for basic unit
facts (race, cost, hit points) without asking the game.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from .. import paths
from .mpq import GameArchives


def parse_slk(text: str) -> list[dict[str, str]]:
    """Parse a SYLK table into row dicts keyed by the header row."""
    cells: dict[tuple[int, int], str] = {}
    x = y = 0
    for line in text.splitlines():
        if not line.startswith("C;"):
            continue
        value = None
        for field in line[2:].split(";"):
            if not field:
                continue
            tag, rest = field[0], field[1:]
            if tag == "X":
                x = int(rest)
            elif tag == "Y":
                y = int(rest)
            elif tag == "K":
                value = rest[1:-1] if rest.startswith('"') and rest.endswith('"') else rest
        if value is not None:
            cells[(y, x)] = value
    if not cells:
        return []
    max_y = max(k[0] for k in cells)
    max_x = max(k[1] for k in cells)
    header = [cells.get((1, c), f"col{c}") for c in range(1, max_x + 1)]
    rows = []
    for r in range(2, max_y + 1):
        row = {header[c - 1]: cells[(r, c)] for c in range(1, max_x + 1) if (r, c) in cells}
        if row:
            rows.append(row)
    return rows


@dataclass(frozen=True)
class UnitInfo:
    id: str
    race: str
    hp: int
    gold: int
    lumber: int
    food: int
    is_building: bool
    name: str = ""


@lru_cache(maxsize=1)
def unit_table() -> dict[str, UnitInfo]:
    cache = paths.CACHE_DIR / "units.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        return {k: UnitInfo(**v) for k, v in data.items()}
    with GameArchives() as g:
        unitdata = parse_slk(g.read("Units\\UnitData.slk").decode("latin-1"))
        balance = {r.get("unitBalanceID"): r for r in parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))}
        ui = {r.get("unitUIID"): r for r in parse_slk(g.read("Units\\unitUI.slk").decode("latin-1"))}
    def num(row: dict, key: str) -> int:
        try:
            return int(float(row.get(key, "0") or 0))
        except ValueError:
            return 0

    table = {}
    for row in unitdata:
        uid = row.get("unitID")
        if not uid or len(uid) != 4:
            continue
        b = balance.get(uid, {})
        table[uid] = UnitInfo(
            id=uid, race=row.get("race", ""), hp=num(b, "HP"), gold=num(b, "goldcost"), lumber=num(b, "lumbercost"),
            food=num(b, "fused"), is_building=(row.get("isbldg") or b.get("isbldg") or "0") == "1",
            name=ui.get(uid, {}).get("name", ""),
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({k: v.__dict__ for k, v in table.items()}))
    return table


@lru_cache(maxsize=1)
@dataclass(frozen=True)
class CombatStats:
    """A unit type's fighting numbers (first weapon; heroes at level 1)."""
    range: float  # attack range
    cooldown: float  # seconds between attacks
    damage: float  # average damage per attack (heroes: including the primary attribute)
    armor: float
    speed: float  # move speed
    hits_air: bool
    is_hero: bool

    @property
    def dps(self) -> float:
        return self.damage / self.cooldown if self.cooldown > 0 else 0.0


@lru_cache(maxsize=None)
def combat_stats() -> dict[str, CombatStats]:
    """Combat numbers per unit type from UnitWeapons/UnitBalance (cached in the cache dir)."""
    cache = paths.CACHE_DIR / "combat.json"
    if not cache.exists():
        with GameArchives() as g:
            rows = parse_slk(g.read("Units\\UnitWeapons.slk").decode("latin-1"))
            key = next(iter(rows[0]))  # the id column (its header cell is not reliable: "serpent")
            weapons = {r.get(key): r for r in rows}
            balance = {r.get("unitBalanceID"): r for r in parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))}

        def num(row: dict, key: str, default: float = 0.0) -> float:
            try:
                return float(str(row.get(key, "")).strip())
            except ValueError:
                return default

        out = {}
        for uid, w in weapons.items():
            b = balance.get(uid, {})
            if not uid:
                continue
            hero = uid[:1].isupper()
            primary = str(b.get("Primary", "")).strip()
            damage = num(w, "dmgplus1") + num(w, "dice1") * (num(w, "sides1") + 1) / 2
            if hero and primary in ("STR", "AGI", "INT"):
                damage += num(b, primary)
            out[uid] = {"range": num(w, "rangeN1"), "cooldown": num(w, "cool1"), "damage": round(damage, 2),
                        "armor": num(b, "def"), "speed": num(b, "spd"),
                        "hits_air": "air" in str(w.get("targs1", "")), "is_hero": hero}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, indent=0))
    return {k: CombatStats(**v) for k, v in json.loads(cache.read_text()).items()}


@lru_cache(maxsize=1)
def unit_names() -> dict[str, str]:
    """Unit type -> its display name ("Mountain King"), from the *UnitStrings.txt files."""
    cache = paths.CACHE_DIR / "unit_names.json"
    if cache.exists():
        return json.loads(cache.read_text())
    names: dict[str, str] = {}
    with GameArchives() as g:
        for fname in sorted(g.names("Units\\*UnitStrings.txt"), key=str.lower):
            current = None
            for line in g.read(fname).decode("latin-1").splitlines():
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    current = line[1:-1]
                elif current and line.startswith("Name=") and current not in names:
                    names[current] = line[5:].strip().strip('"')
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(names))
    return names


def unit_vocabulary() -> dict[str, int]:
    """Unit type code -> index 1..N (0 is reserved for unknown/padding). Stable: sorted by code."""
    return {code: i + 1 for i, code in enumerate(sorted(unit_table()))}


_ORDER_FIELDS = ("Order", "Orderon", "Orderoff", "Unorder")


@lru_cache(maxsize=1)
def ability_orders() -> dict[str, dict[str, str]]:
    """Ability code -> {"Order": ..., "Orderon": ..., ...} from the *AbilityFunc.txt files."""
    cache = paths.CACHE_DIR / "ability_orders.json"
    if cache.exists():
        return json.loads(cache.read_text())
    table: dict[str, dict[str, str]] = {}
    with GameArchives() as g:
        for name in sorted({n for n in g.names("Units\\*AbilityFunc.txt")}, key=str.lower):
            current = None
            for line in g.read(name).decode("latin-1").splitlines():
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    current = line[1:-1]
                elif current and "=" in line:
                    key, value = line.split("=", 1)
                    if key in _ORDER_FIELDS and value.strip():
                        table.setdefault(current, {})[key] = value.strip().lower()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(table, sort_keys=True))
    return table


@lru_cache(maxsize=1)
def order_strings() -> tuple[str, ...]:
    """Every order string used by an ability, sorted."""
    return tuple(sorted({o for fields in ability_orders().values() for o in fields.values()}))
