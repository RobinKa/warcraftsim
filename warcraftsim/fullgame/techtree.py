"""The tech tree, from the game's data (Units\\<Race>UnitFunc.txt and <Race>UpgradeFunc.txt): what each
building trains, researches and upgrades to ("Trains", "Researches", "Upgrade"), and what each unit,
building, hero and research needs first ("Requires", "Requires1", ...: per level, a hero's per its
place among the player's heroes). The target builder (selfplay.TargetBuilder) orders what a target
strategy's next item needs before the item."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

RACE_FILES = {"human": "Human", "orc": "Orc", "undead": "Undead", "nightelf": "NightElf"}
TIERS = {"human": ("htow", "hkee", "hcas"), "orc": ("ogre", "ostr", "ofrt"),
         "undead": ("unpl", "unp1", "unp2"), "nightelf": ("etol", "etoa", "etoe")}
ALTARS = {"human": "halt", "orc": "oalt", "undead": "uaod", "nightelf": "eate"}


@dataclass(frozen=True)
class Tree:
    makers: dict[str, tuple[str, ...]]  # an item -> the building types that train, research or upgrade to it
    needs: dict[str, tuple[tuple[str, ...], ...]]  # an item -> what each level needs (a hero: as the 1st, 2nd, 3rd)
    upgrades: dict[str, tuple[str, ...]]  # a building type -> the types it upgrades to (a hall: its next tier)


def parse(texts: list[str]) -> Tree:
    makers: dict[str, list[str]] = {}
    upgrades: dict[str, tuple[str, ...]] = {}
    needs: dict[str, dict[int, tuple[str, ...]]] = {}
    for text in texts:
        sec = None
        for line in text.splitlines():
            m = re.match(r"\[(\w{4})\]", line)
            if m:
                sec = m.group(1)
                continue
            if sec is None or "=" not in line:
                continue
            k, v = (s.strip() for s in line.split("=", 1))
            codes = tuple(c.strip() for c in v.split(",") if c.strip())
            if k == "Upgrade":
                upgrades[sec] = codes
            if k in ("Trains", "Researches", "Upgrade"):
                for c in codes:
                    if sec not in makers.setdefault(c, []):
                        makers[c].append(sec)
            elif (m := re.fullmatch(r"Requires(\d*)", k)) is not None:
                needs.setdefault(sec, {})[int(m.group(1) or 0)] = codes
    return Tree({c: tuple(v) for c, v in makers.items()},
                {c: tuple(d.get(i, ()) for i in range(max(d) + 1)) for c, d in needs.items()}, upgrades)


@lru_cache(1)
def tree() -> Tree:
    from ..data.mpq import GameArchives
    with GameArchives() as g:
        return parse([g.read(f"Units\\{r}{k}Func.txt").decode("latin-1")
                      for r in RACE_FILES.values() for k in ("Unit", "Upgrade")])


@lru_cache(1)
def research_levels() -> dict[str, int]:
    """Research -> its number of levels (UpgradeData.slk "maxlevel")."""
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        rows = parse_slk(g.read("Units\\UpgradeData.slk").decode("latin-1"))
    out = {}
    for r in rows:
        try:
            out[r["upgradeid"]] = max(1, int(float(r.get("maxlevel") or 1)))
        except (KeyError, ValueError):
            pass
    return out


def is_hero(code: str) -> bool:
    """A hero's id: upper case, as research's ("R...") is too."""
    return code[:1].isupper() and code[0] != "R"


def resolve(code: str, race: str) -> str:
    """A requirement's generic code as the race's: any altar ("TALT"), any hall from tier 2 ("TWN2") or 3."""
    if code == "TALT":
        return ALTARS[race]
    if code in ("TWN1", "TWN2", "TWN3"):
        return TIERS[race][int(code[3]) - 1]
    return code


def counts_as(types, race: str) -> set[str]:
    """What a player's finished buildings and research count as: a hall is each lower tier too."""
    have = set(types)
    for tiers in TIERS.values():
        for i, t in enumerate(tiers):
            if t in have:
                have.update(tiers[:i])
    return have


def missing(item: str, race: str, have: set[str], level: int = 0, t: Tree | None = None) -> list[str]:
    """What `item` (at `level`: a research's next level, a hero's place among the player's) needs and the
    player lacks (`have`: counts_as)."""
    t = t or tree()
    per = t.needs.get(item, ())
    if not per:
        return []
    return [r for r in (resolve(c, race) for c in per[min(level, len(per) - 1)]) if r not in have]


def closure(items, race: str, t: Tree | None = None) -> set[str]:
    """`items` and, in turn, what each needs (a hero: as many heroes as `items` has) and, for one that none
    of `items` makes, the first building type that does: what a player making `items` makes on the way."""
    t = t or tree()
    items = set(items)
    heroes = sum(is_hero(c) for c in items)
    out: set[str] = set()
    todo = list(items)
    while todo:
        c = todo.pop()
        if c in out:
            continue
        out.add(c)
        per = t.needs.get(c, ())
        for need in per[:max(heroes, 1)] if is_hero(c) else per[:1]:
            todo += [resolve(r, race) for r in need]
        makers = t.makers.get(c, ())
        if makers and not items & set(makers):
            todo.append(makers[0])
    return out
