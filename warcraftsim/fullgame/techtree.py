"""The tech tree, from the game's data (Units\\<Race>UnitFunc.txt and <Race>UpgradeFunc.txt): what each
building trains, researches and upgrades to ("Trains", "Researches", "Upgrade"), and what each unit,
building, hero and research needs first ("Requires", "Requires1", ...: per level, a hero's per its
place among the player's heroes). The target builder (selfplay.TargetBuilder) orders what a target
strategy's next item needs before the item."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    # an item -> per level, the research levels its needs ask for beyond the first ("Requiresamount": Mark of the Talon
    # needs Druid of the Talon training at level 2; ignored before, ~10 refused a night elf game; audit 11)
    amounts: dict[str, dict[int, dict[str, int]]] = field(default_factory=dict)


def parse(texts: list[str]) -> Tree:
    makers: dict[str, list[str]] = {}
    upgrades: dict[str, tuple[str, ...]] = {}
    needs: dict[str, dict[int, tuple[str, ...]]] = {}
    amounts: dict[str, dict[int, tuple[str, ...]]] = {}
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
            elif (m := re.fullmatch(r"Requires(\d*)", k)) is not None:  # (a key given twice in a section: both
                # lines count, as in the game: Cloud's Requires=hcas and Requires=hvlt; the last alone had let the
                # mask offer it without a castle, refused every time; audit 13)
                got = needs.setdefault(sec, {})
                lvl = int(m.group(1) or 0)
                got[lvl] = tuple(dict.fromkeys(got.get(lvl, ()) + codes))
            elif (m := re.fullmatch(r"Requiresamount(\d*)", k)) is not None:
                amounts.setdefault(sec, {})[int(m.group(1) or 0)] = codes
    levels: dict[str, dict[int, dict[str, int]]] = {}
    for sec, per in amounts.items():  # (each level's amounts beside its codes; a research among them asked past level 1)
        for lvl, vals in per.items():
            codes = needs.get(sec, {}).get(lvl, ())
            got = {c: int(v) for c, v in zip(codes, vals) if c.startswith("R") and v.isdigit() and int(v) > 1}
            if got:
                levels.setdefault(sec, {})[lvl] = got
    return Tree({c: tuple(v) for c, v in makers.items()},
                {c: tuple(d.get(i, ()) for i in range(max(d) + 1)) for c, d in needs.items()}, upgrades, levels)


@lru_cache(1)
def tree() -> Tree:
    from ..data.mpq import GameArchives
    with GameArchives() as g:
        races = [g.read(f"Units\\{r}{k}Func.txt").decode("latin-1") for r in RACE_FILES.values() for k in ("Unit", "Upgrade")]
        neutral = parse([g.read("Units\\NeutralUnitFunc.txt").decode("latin-1")])
    t = parse(races)
    # the tavern's heroes need what a race's hero does (an altar, then each tier: NeutralUnitFunc's Requires); the
    # neutral buildings that sell them are no makers (a player can't build a tavern)
    heroes = {c: v for c, v in neutral.needs.items() if is_hero(c) and c not in t.needs}
    return Tree(t.makers, {**t.needs, **heroes}, t.upgrades, t.amounts)


@lru_cache(4)
def ability_research(race: str) -> dict[str, tuple[str, int]]:
    """Order string -> the research (and its level) the race's abilities that use it need (its AbilityFunc's Requires: defend, burrow,
    cannibalize, sentinel, ensnare, ...), where every ability of the race with that order needs the same one. (A hand
    list had missed most: 23,575 cannibalizes refused in fgself-146 before ghoul frenzy... before Ruac; audit 10.)"""
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        text = g.read(f"Units\\{RACE_FILES[race]}AbilityFunc.txt").decode("latin-1")
        unit_abils = parse_slk(g.read("Units\\UnitAbilities.slk").decode("latin-1"))
    letter = {"human": "h", "orc": "o", "undead": "u", "nightelf": "e"}[race]
    mine = {c for c in tree().makers if c[0].lower() == letter and c[0] != "R"}  # (the units the race makes: their
    used = {a.strip() for r in unit_abils if r.get("unitAbilID") in mine  # abilities, not a summon's or a creep's
            for a in (r.get("abilList") or "").split(",") if a.strip()}   # variant with the same order)
    names = re.split(r"(?m)^\[(\w{4})\]", text)
    per: dict[str, set] = {}
    for sec, block in zip(names[1::2], names[2::2]):
        if sec not in used:
            continue
        kv = dict((a.strip(), b.strip()) for a, b in (line.split("=", 1) for line in block.splitlines() if "=" in line))
        codes = [c.strip() for c in kv.get("Requires", "").split(",")]
        amounts = [a.strip() for a in kv.get("Requiresamount", "").split(",")]
        req = next(((c, int(amounts[i]) if i < len(amounts) and amounts[i].isdigit() else 1)
                    for i, c in enumerate(codes) if c.startswith("R")), None)  # (its level: Requiresamount, audit 11)
        for k in ("Order", "Orderon", "Orderoff", "Unorder"):
            if kv.get(k):
                per.setdefault(kv[k], set()).add(req)
    return {o: next(iter(r)) for o, r in per.items() if len(r) == 1 and None not in r}


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


def short_levels(item: str, level: int, levels: dict[str, int], t: Tree | None = None) -> bool:
    """Whether `item` (at `level`) needs a research at a higher level than the player has (`levels`: research ->
    level; Requiresamount)."""
    t = t or tree()
    per = t.amounts.get(item)
    if not per:
        return False
    need = per.get(min(level, max(per)), per.get(0, {}))
    return any(levels.get(r, 0) < a for r, a in need.items())


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
