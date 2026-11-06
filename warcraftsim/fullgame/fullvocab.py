"""Every order a player can give in a 1.29 melee game, as the policy's orders (selfplay --full-vocab 1).

The vocabulary holds the orders the demonstrations gave at least 5 times (features.build_vocab): the built-in AI's and the
recorded games'. Whatever the insane AI never does was missing, and spells.py, shopitems.py and neutral.py filled only
parts of it. This module lists every order a player of each race can give, from the game's own tables, and adds the missing
ones to a vocabulary (extend), its network (grow) and the masks:

  universe(race)   from Units\\*UnitFunc.txt (Trains, Builds, Upgrade, Researches, Sellunits, Makeitems), UnitAbilities.slk
                   (each unit type's abilities), AbilityData.slk (mana, cooldowns, the units a morph or a summon makes) and the
                   *AbilityFunc/Strings.txt order strings (BASE_ORDERS: the game's own strings where the files name none or
                   one the game does not know): every unit, building, upgrade and research; every unit's, summon's and hero's
                   ability orders, autocast on/off and morph/undo pairs too; the generic orders (move, attack, stop, patrol,
                   hold position, attack ground, set rally); the heroes' items (buy, use, drop, give, sell, pick up); the
                   neutral buildings of the maps the runs play (MAPS: the tavern, the mercenary camp, the goblin laboratory
                   and merchant) and their units' abilities.
  coverage(vocab)  what of it a vocabulary lacks, per race (scripts/order_coverage.py; docs/reference/order-coverage.md).
  extend(vocab)    the missing orders and unit types at the vocabulary's end ("full_vocab" marks it).
  grow(net, ...)   a network grown to them: each new order's row in the order head is the "no order" row's, its bias less
                   its category's (BIASES), so every old order's logit is as before and a new order is rare at first in any
                   state; a new unit type's embedding the mean of similar old types' (SIMILAR).
  allow(net, ...)  who may give each new order (the unit types the tables say have it; a morph's order its source form, its
                   undo the morphed form), and each new unit type's orders.

What is left out on purpose is EXCLUDED, with the reason. The order ids are the game's (OrderId of each string, read from a
game: warcraftsim/data/order_ids.json, scripts/vocab_probe.py ids), or this module's own for the item orders the game has no
id of its own for (ITEM_ORDERS: protocol.DropItem, PickUpAt)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from . import features as fx

RACES = ("human", "orc", "undead", "nightelf")
WORKERS = {"human": "hpea", "orc": "opeo", "undead": "uaco", "nightelf": "ewsp"}
# the maps the runs train and evaluate on, and the neutral buildings on them that take an order (fountains, waygates and
# creep huts take none; scripts/order_coverage.py --maps reads each map's units)
MAPS = ("(2)EchoIsles", "echofast2", "lastrefugefast2", "amazoniafast2", "terenaslvfast2")
MAP_SHOPS = ("ntav", "ngme", "nmer", "nmrk", "ngad")
IMMEDIATE, POINT, UNIT, SKILL, TREE = fx.IMMEDIATE, fx.POINT, fx.UNIT, fx.SKILL, fx.TREE
ORDER_FIELDS = ("Order", "Orderon", "Orderoff", "Unorder")
UNNAMED_IDS = {"cancel": 851976, **{f"useslot{k}": 852007 + k for k in range(1, 7)}}  # (OrderId knows no string for these)

# the game's own order strings where the files name none, one the game does not know (OrderId 0: sphinxform, absorbmana,
# unloadcorpse), or not the one it takes (the Dreadlord's inferno: the built-in AI's dreadlordinferno; Searing and Cold
# Arrows' cast on a unit, their files' Order being the autocast's), by ability (or its base ability): these fields replace
# the files'. Each checked in a game (scripts/vocab_probe.py orders)
BASE_ORDERS = {
    "Acri": {"Order": "cripple"}, "Afae": {"Order": "faeriefire", "Orderon": "faeriefireon", "Orderoff": "faeriefireoff"},
    "Acyc": {"Order": "cyclone"}, "Arej": {"Order": "rejuvination"}, "Aast": {"Order": "ancestralspirittarget"},
    "Adcn": {"Order": "dispel"}, "Advm": {"Order": "devourmagic"}, "Andt": {"Order": "neutraldetectaoe"},
    "Acoi": {"Order": "coupletarget"}, "Alam": {"Order": "requestsacrifice"},
    "Aave": {"Order": "avengerform", "Unorder": "unavengerform"}, "Aabs": {"Order": "absorb"},
    "Amed": {"Order": "unloadallcorpses"}, "AUin": {"Order": "dreadlordinferno"}, "AIta": {"Order": "detectaoe"},
    "AHfa": {"Order": "flamingarrowstarg", "Orderon": "flamingarrows", "Orderoff": "unflamingarrows", "Unorder": None},
    "ANfa": {"Order": "coldarrowstarg", "Orderon": "coldarrows", "Orderoff": "uncoldarrows"},
    "Ambt": {"Orderon": "rechargeon", "Orderoff": "rechargeoff"},
    # (a zeppelin's unload at a point: unloadall taken, unload refused; the entangled mine's at once likewise)
    "Adro": {"Order": "unloadall"}, "Adri": {"Order": "unloadall"},
}
# orders a vocabulary gives in another form the agent turns into this one (play.BCAgent._militia: a peasant's call to arms
# from militiaconvert on a hall, a militia's back to work from militiaunconvert)
ALIASES = {(852072, 0): (852071, 2), (852073, 0): (852651, 2)}
# the order kind of each castable ability's main order, by ability or its base (its toggles and undo: immediate). A test
# (tests/unit/test_fullvocab.py) checks every castable ability of the universe has one.
KIND = {
    # human
    "Ahea": UNIT, "Ainf": UNIT, "Adis": POINT, "Aivs": UNIT, "Aply": UNIT, "Aslo": UNIT, "Asps": UNIT, "Acmg": UNIT,
    "Afla": POINT, "Aclf": UNIT, "Amls": UNIT, "Adef": IMMEDIATE, "Amic": IMMEDIATE, "Amil": IMMEDIATE, "Arep": UNIT,
    "Ahar": UNIT, "AIta": POINT,
    # orc
    "Ablo": UNIT, "Alsh": UNIT, "Aprg": UNIT, "Aeye": POINT, "Ahwd": POINT, "Asta": POINT, "Acpf": IMMEDIATE,
    "Aspl": UNIT, "Aast": UNIT, "Absk": IMMEDIATE, "Aens": UNIT, "Adev": UNIT, "Auco": UNIT, "Abtl": IMMEDIATE,
    "Astd": IMMEDIATE,
    # undead
    "Arai": IMMEDIATE, "Auhf": UNIT, "Acri": UNIT, "Aams": UNIT, "Acrs": UNIT, "Aps2": UNIT, "Aave": IMMEDIATE,
    "Aabs": UNIT, "Advm": POINT, "Arpl": UNIT, "Arpm": UNIT, "Acan": IMMEDIATE, "Aweb": UNIT, "Abur": IMMEDIATE,
    "Astn": IMMEDIATE, "Afrz": UNIT, "Amel": UNIT, "Amed": IMMEDIATE, "Alam": UNIT, "Asac": UNIT, "Auns": UNIT,
    "Aaha": UNIT, "Arst": UNIT, "Arpb": IMMEDIATE,
    # night elf
    "Acoi": UNIT, "Adec": IMMEDIATE, "Ashm": IMMEDIATE, "Acor": UNIT, "Abrf": IMMEDIATE, "Aroa": IMMEDIATE,
    "Arav": IMMEDIATE, "Aadm": UNIT, "Amfl": IMMEDIATE, "Apsh": IMMEDIATE, "Ambt": UNIT, "Atau": IMMEDIATE,
    "Agra": TREE, "Aesn": TREE, "Aent": UNIT, "Aeat": TREE, "Aroo": POINT, "Adtn": UNIT, "Aren": UNIT, "Awha": UNIT,
    "Afae": UNIT, "Acyc": UNIT, "Arej": UNIT, "Adri": IMMEDIATE, "Aloa": UNIT,
    # neutral units (the goblin laboratory's, the mercenaries', summons': a doom guard's, a quillbeast's, an avatar of
    # vengeance's)
    "Adro": POINT, "Asds": UNIT, "Ahrl": UNIT, "Andt": POINT, "Aawa": IMMEDIATE, "ACrf": POINT, "Afzy": IMMEDIATE,
    "Avng": UNIT, "AOws": IMMEDIATE, "Anhe": UNIT, "ANrf": POINT,
}
# abilities whose UnitID or DataC name what a target turns into (polymorph's sheep, hex's critters), not the caster's units
TRANSFORMS_TARGET = frozenset({"Aply", "ACpy", "AOhx", "ANhx", "AChx"})
# the form each morph turns a unit into (its order given by the source form, its undo by the morphed form): AbilityData's
# DataA1 (the source) and UnitID1 (the form), for these bases
MORPH_BASES = ("Abrf", "Arav", "Acpf", "Aave", "Astn", "Abur", "Aetf")
# units an ability makes that its data names nowhere (a shade: an acolyte sacrificed at the pit)
MADE = {"Asac": ("ushd",), "Alam": ("ushd",)}
# abilities with an order string but no order a player gives: a worker's harvest is the vocabulary's own (harvest on a
# mine or a tree), returning is the engine's (features.DROPPED_ORDERS), a shop's selection and sale are protocol.Buy's
SKIP = frozenset({"Argl", "Arlm", "Artn", "Apit", "Aneu", "Ane2", "Asid", "Asud", "Abds", "Abdl", "Abgs", "Abgl", "Aall",
                  "Avul", "Aawa"})
# abilities (or their base) and order strings left out on purpose: EXCLUDED's reason (in the universe, never extended)
EXCLUDED_ABILITIES = {"Afrz": "passive breaths", "Acor": "passive breaths", "AHta": "arcane reveal", "AIta": "arcane reveal",
                      "Aroo": "uproot", "Aro1": "uproot", "Aro2": "uproot", "Aeat": "uproot", "Amel": "corpses",
                      "Aast": "corpses"}
EXCLUDED_NAMES = {"undivineshield": "undivineshield", "unavengerform": "undestroyer", "returnresources": "returnresources",
                  "buy tgrh": "tiny great hall"}
# a morph the game charges for (scripts/vocab_probe.py: an obsidian statue into a destroyer took the units' difference in
# gold, lumber and food): its order's cost, costs.order_costs
MORPH_COSTS = {"avengerform": ("uobs", "ubsp")}
# the item orders the game has no order id of its own for (this module's ids, as macros' are the agent's own): drop the item
# of slot k at a point, give it to a unit of the player's, sell it to a shop (the race's own shop or a neutral one that buys:
# the goblin merchant, the marketplace), and pick up the item lying nearest a point
DROP0, GIVE0, SELL0, PICKUP_AT = 990101, 990111, 990121, 990131
ITEM_ORDERS = tuple([(DROP0 + k, POINT) for k in range(6)] + [(GIVE0 + k, UNIT) for k in range(6)]
                    + [(SELL0 + k, UNIT) for k in range(6)] + [(PICKUP_AT, POINT)])
ITEM_NAMES = {**{DROP0 + k: f"drop item {k + 1}" for k in range(6)}, **{GIVE0 + k: f"give item {k + 1}" for k in range(6)},
              **{SELL0 + k: f"sell item {k + 1}" for k in range(6)}, PICKUP_AT: "pick up an item near a point"}
EXTRA_ITEMS = {"hvlt": ("mcri",), "ovln": ("tgrh",)}  # (the race shops' items shopitems.py leaves out: mechanical critter,
# tiny great hall)
# what is left out on purpose, and why (docs/reference/order-coverage.md shows these)
EXCLUDED = {
    "moveslot": "moving an item between inventory slots: no effect on play (an item acts from any slot; the policy reads and "
                "uses all six)",
    "returnresources": "the engine's own (a worker returning its load): left out of every vocabulary on purpose "
                       "(features.DROPPED_ORDERS: a policy that learned it interrupted working harvesters)",
    "marketplace": "the marketplace's stock is random (blizzard.j PerformStockUpdates: an item of a level and class creeps "
                   "have dropped, every 30 s) and 1.29's JASS can't read a shop's stock, so no mask could tell what it sells",
    "backpack": "a unit's backpack (Rhpm, Ropm, Rupm, Repm research): the harness reports inventories of heroes only",
    "awaken": "a tavern's instant revival of a dead hero: an altar revives it (the hero's train order), and the tavern's "
              "button picks no hero the policy can point at (dead heroes are no entities)",
    "passive breaths": "a frost wyrm's freezing breath and a chimaera's corrosive breath: their files give an order, the "
                       "game refuses it on a visible enemy building (scripts/vocab_probe.py); they act through the "
                       "attack once researched",
    "arcane reveal": "the arcane tower's reveal (AHta): refused in every form tried (detectaoe and neutraldetectaoe at a "
                     "point, detectaoe at once), its research done",
    "undivineshield": "a paladin's divine shield cut short: refused before, during and after the shield",
    "undestroyer": "a destroyer back to an obsidian statue: the files' unsphinxform is no order of the game's, its "
                   "unavengerform refused",
    "uproot": "an ancient's unroot, root and eat tree: the probe uprooted an ancient but could not root it again (at its "
              "own place, on open ground 500 away, at once), and an uprooted ancient trains and researches nothing: left "
              "out so that the policy can't strand its buildings",
    "corpses": "a meat wagon's get corpse and a spirit walker's ancestral spirit: the harness reports a unit's death and "
               "then drops it, so a corpse is no entity to point at, and a harness order on the nearest corpse was "
               "refused for both (a necromancer's raise dead there too)",
    "tiny great hall": "the orc shop's tiny great hall: sold, then gone from the buyer within 5 s (the engine uses it by "
                       "itself for a computer slot; shopitems.py's probe saw it stand once, vanish, and be used unasked)",
}
# the category of each new order: its bias below "no order" (each new order exp(-bias) as likely as giving none, in any
# state, before the order temperature). spells.py's NEW_BIAS 8 left its spells never sampled (9 casts in ~13,800 games);
# shopitems' 7 is a hero's paid order with ~8 open at once. Here by how many units may give it: an order every unit has
# (patrol, hold position) at 9 is ~1 in 8,100 of a unit's steps, a few a game over ~40 units and ~1,500 steps; a worker's
# new building at 9 likewise (it costs too); a building's new unit, upgrade or research (one or two buildings, paid) and a
# morph at 6; a unit's ability (its casters only, its mana and research masked) at 5, ~1 in 150 of a caster's open steps.
BIASES = {"generic": 9.0, "build": 9.0, "ware": 9.0, "train": 6.0, "upgrade": 6.0, "research": 6.0, "morph": 6.0,
          "unmorph": 6.0, "toggle": 7.0, "ability": 5.0, "spell": 5.0, "skill": 5.0, "item": 7.0, "neutral": 8.0}
# a new unit type's embedding: the mean of these old types' (else of its kind's: _like_types)
SIMILAR = {"ospm": ("oshm", "odoc"), "ospw": ("oshm", "odoc"), "ehip": ("hgry", "owyv", "ugar"),
           "ehpr": ("earc", "hgry", "owyv"), "hmtt": ("hmtm", "ocat", "umtw"), "hrtt": ("hmtm", "ocat", "umtw"),
           "ubsp": ("uobs", "ufro"), "ushd": ("uaco", "ugho"), "edtm": ("edot",), "edcm": ("edoc",),
           "hatw": ("hgtw",), "hctw": ("hgtw",), "owtw": ("hgtw", "otrb"), "uzg2": ("uzg1",), "etrp": ("eaom", "eaow"),
           "otbk": ("ohun",), "nzep": ("hgyr",), "ngir": ("hpea", "opeo"), "ngsp": ("ogru", "ohun"), "ngad": ("ngme", "ntav")}


def code(c: str) -> int:
    return int.from_bytes(c.encode("latin-1"), "big")


def rawcode(v: int) -> str:
    return fx.rawcode(int(v))


@lru_cache(1)
def order_ids() -> dict[str, int]:
    """The game's order id of each order string (OrderId in a game: warcraftsim/data/order_ids.json, written by
    scripts/vocab_probe.py ids), with UNNAMED_IDS."""
    from importlib import resources
    got = json.loads(resources.files("warcraftsim.data").joinpath("order_ids.json").read_text())
    return {**{k: int(v) for k, v in got.items()}, **UNNAMED_IDS}


def order_name(oid: int) -> str:
    """An order id's string (or the item orders' names, or a unit / research code)."""
    if oid in ITEM_NAMES:
        return ITEM_NAMES[oid]
    if oid >= fx.TYPE_CODE:
        return rawcode(oid)
    return _names_by_id().get(int(oid), str(oid))


@lru_cache(1)
def _names_by_id() -> dict[int, str]:
    out: dict[int, str] = {}
    for k, v in sorted(order_ids().items()):
        out.setdefault(v, k)
    return out


@lru_cache(1)
def tables() -> dict:
    """The game's tables this module reads, once (a cache file beside the others: actors read it, not the archives):
    "units" {type: {abil, hero_abil, gold, lumber, food, building, weapon, speed}}, "abilities" {code: {base, orders,
    requires, amounts, targs, cost, cool, made (UnitID per level), data_a, data_c, hero}}, "funcs" {type: {Trains, Builds,
    Upgrade, Researches, Sellunits, Sellitems, Makeitems}}, "items" {item: {class, abil, gold, pawnable, droppable}}."""
    from .. import paths
    path = paths.CACHE_DIR / "full_vocab-3.json"
    if path.exists():
        return json.loads(path.read_text())
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        ua = {r.get("unitAbilID"): r for r in parse_slk(g.read("Units\\UnitAbilities.slk").decode("latin-1"))}
        ub = {r.get("unitBalanceID"): r for r in parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))}
        weap = parse_slk(g.read("Units\\UnitWeapons.slk").decode("latin-1"))
        uw = {r.get(next(iter(r))): r for r in weap}  # (its first column, the unit's id)
        ad = {r.get("alias"): r for r in parse_slk(g.read("Units\\AbilityData.slk").decode("latin-1"))}
        idata = {r.get("itemID"): r for r in parse_slk(g.read("Units\\ItemData.slk").decode("latin-1"))}
        texts = [g.read(n).decode("latin-1") for n in sorted(g.names("Units\\*Func.txt"), key=str.lower)]
        texts += [g.read(n).decode("latin-1") for n in sorted(g.names("Units\\*AbilityStrings.txt"), key=str.lower)]
    sections: dict[str, dict[str, str]] = {}
    keys = (*ORDER_FIELDS, "Requires", "Requiresamount", "Trains", "Builds", "Upgrade", "Researches", "Sellunits",
            "Sellitems", "Makeitems")
    for text in texts:
        sec = None
        for line in text.splitlines():
            line = line.strip()
            m = re.fullmatch(r"\[(\w{4})\]", line)
            if m:
                sec = m.group(1)
                continue
            if sec is None or "=" not in line:
                continue
            k, v = (s.strip() for s in line.split("=", 1))
            if v and k in keys:
                sections.setdefault(sec, {}).setdefault(k, v)

    def num(v) -> float:
        try:
            return float(str(v).strip())
        except (TypeError, ValueError):
            return 0.0

    def split(v) -> list[str]:
        return [c.strip() for c in str(v or "").split(",") if c.strip() and c.strip() != "_"]
    units = {}
    for c, r in ua.items():
        if not c or len(c) != 4:
            continue
        b, w = ub.get(c) or {}, uw.get(c) or {}
        units[c] = {"abil": split(r.get("abilList")), "hero_abil": split(r.get("heroAbilList")),
                    "gold": int(num(b.get("goldcost"))), "lumber": int(num(b.get("lumbercost"))),
                    "food": int(num(b.get("fused"))), "building": str(b.get("isbldg") or "0").strip() == "1",
                    "weapon": str(w.get("weapTp1") or "").strip() if str(w.get("weapsOn") or "0").strip() not in ("0", "")
                    else "", "speed": num(b.get("spd")), "range": num(w.get("rangeN1"))}
    abilities = {}
    for c, r in ad.items():
        if not c or len(c) != 4:
            continue
        s = sections.get(c, {})
        n = max(1, int(num(r.get("levels"))))
        abilities[c] = {"base": r.get("code") or c, "orders": {k: s[k].lower() for k in ORDER_FIELDS if k in s},
                        "requires": split(s.get("Requires")), "amounts": split(s.get("Requiresamount")),
                        "targs": str(r.get("targs1") or ""), "cost": [num(r.get(f"Cost{k}")) for k in range(1, n + 1)],
                        "cool": [num(r.get(f"Cool{k}")) for k in range(1, n + 1)],
                        "made": [str(r.get(f"UnitID{k}") or "").strip() for k in range(1, n + 1)],
                        "data_a": str(r.get("DataA1") or "").strip(), "data_c": str(r.get("DataC1") or "").strip(),
                        "hero": str(r.get("hero") or "0").strip() == "1"}
    funcs = {c: {k: split(v) for k, v in s.items() if k in ("Trains", "Builds", "Upgrade", "Researches", "Sellunits",
                                                             "Sellitems", "Makeitems")}
             for c, s in sections.items()}
    funcs = {c: f for c, f in funcs.items() if f}
    items = {c: {"class": str(r.get("class") or ""), "abil": split(r.get("abilList")), "gold": int(num(r.get("goldcost"))),
                 "pawnable": str(r.get("pawnable") or "1").strip() != "0",
                 "droppable": str(r.get("droppable") or "1").strip() != "0"}
             for c, r in idata.items() if c and len(c) == 4}
    out = {"units": units, "abilities": abilities, "funcs": funcs, "items": items}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, sort_keys=True))
    return out


@dataclass
class Entry:
    """One order of the universe: (order id, kind), what it is ("train", "build", "upgrade", "research", "skill", "spell",
    "ability", "toggle", "morph", "unmorph", "generic", "item", "ware", "neutral"), who gives it (unit types), and what it
    needs: `needs` [(research or building, level)] (any one ability with the order: all of them must agree, else none),
    `mana` and `cool` per giver type (the ability's first level)."""
    oid: int
    kind: int
    name: str
    what: str
    givers: set = field(default_factory=set)
    ability: str = ""
    needs: list | None = None
    mana: dict = field(default_factory=dict)
    cool: dict = field(default_factory=dict)
    state: tuple | None = None  # (an autocast's on / off, a root and its undo: (ability, the state it sets))


def _ability_orders(a: str) -> dict[str, str]:
    """An ability's order strings by field: its files' that the game knows, BASE_ORDERS' fields (its own entry, else its
    base ability's) in their place (None: none)."""
    t = tables()["abilities"].get(a, {})
    ids = order_ids()
    got = {k: v for k, v in (t.get("orders") or {}).items() if v in ids}
    got.update(BASE_ORDERS.get(a) or BASE_ORDERS.get(t.get("base", ""), {}))
    return {k: v for k, v in got.items() if v}


def _made(a: str) -> list[str]:
    """The unit types ability `a` makes (a summon's, a morph's form, the corpses it raises), known unit types only."""
    t = tables()
    ab, units = t["abilities"].get(a, {}), t["units"]
    if a in TRANSFORMS_TARGET or ab.get("base") in TRANSFORMS_TARGET:
        return []
    out = [m for m in ab.get("made", []) for m in m.split(",") if m in units]
    out += [m for m in ab.get("data_c", "").split(",") if m in units]
    if ab.get("base") == "ANef":
        out += [m for m in ab.get("data_a", "").split(",") if m in units]
    return list(dict.fromkeys(out + list(MADE.get(a, ()))))


def _needs(a: str) -> list[tuple[str, int]]:
    ab = tables()["abilities"].get(a, {})
    req, amt = ab.get("requires", []), ab.get("amounts", [])
    return [(r, int(amt[i]) if i < len(amt) and amt[i].isdigit() else (1 if r[:1] == "R" else 0))
            for i, r in enumerate(req) if r]


def is_hero(c: str) -> bool:
    return fx.is_hero_code(c)


@lru_cache(8)
def _universe(race: str) -> tuple[dict, frozenset]:
    from ..data.abilities import ability_info
    from . import techtree as tt
    from .neutral import MERCHANT_ITEMS, TAVERN_HEROES
    from .shopitems import RACE_SHOPS, STOCK
    t = tables()
    U, F, A = t["units"], t["funcs"], t["abilities"]
    ids = order_ids()
    out: dict[tuple[int, int], Entry] = {}

    def add(oid, kind, name, what, giver, ability="", needs=None, mana=0.0, cool=0.0, state=None):
        if oid is None:
            return
        e = out.get((oid, kind))
        if e is None:
            e = out[(oid, kind)] = Entry(int(oid), int(kind), name, what, ability=ability, needs=needs, state=state)
        elif e.needs != needs:  # (abilities with one order and other needs: none masked by tech)
            e.needs = None
        e.givers.add(giver)
        if mana:
            e.mana[giver] = mana
        if cool:
            e.cool[giver] = cool
    # the race's units: from its hall and worker, what each makes, builds, upgrades to, summons or turns into; heroes at
    # its altar and the tavern; the neutral buildings' units on the maps
    shops = {s: F.get(s, {}) for s in MAP_SHOPS}
    todo = [tt.TIERS[race][0], WORKERS[race], *TAVERN_HEROES, *(c for s in ("nmer", "ngad") for c in shops[s].get("Sellunits", ()))]
    types: set[str] = set()
    while todo:
        u = todo.pop()
        if u in types or u not in U:
            continue
        types.add(u)
        f = F.get(u, {})
        todo += f.get("Trains", []) + f.get("Builds", []) + f.get("Upgrade", [])
        for a in U[u]["abil"] + U[u]["hero_abil"]:
            todo += _made(a)
    types |= {s for s in MAP_SHOPS}
    # train, build, upgrade, research
    for u in sorted(types):
        f = F.get(u, {})
        if u in MAP_SHOPS:
            continue
        for c in f.get("Trains", []):  # (known unit types only: a peon's Builds lists orbr, which is none)
            if c in U:
                add(code(c), IMMEDIATE, f"train {c}", "train", u)
        for c in f.get("Builds", []):
            if c in U:
                add(code(c), UNIT if c == "ugol" else POINT, f"build {c}", "build", u)
        for c in f.get("Upgrade", []):
            if c in U:
                add(code(c), IMMEDIATE, f"upgrade to {c}", "upgrade", u)
        for c in f.get("Researches", []):
            add(code(c), IMMEDIATE, f"research {c}", "research", u)
    # abilities: heroes' (learning, casting, toggles), units' (casting, toggles, morphs)
    info = ability_info()
    for u in sorted(types):
        for a in U[u]["hero_abil"]:
            add(code(a), SKILL, f"learn {a}", "skill", u)
            i = info.get(a)
            orders = _ability_orders(a)
            kind = {"unit": UNIT, "point": POINT, "instant": IMMEDIATE}.get(i.cast if i is not None else "", None)
            main = orders.get("Order") or (i.order if i is not None and i.castable else None)
            if main and kind is not None:
                add(ids.get(main), kind, main, "spell", u, a)
            for fld, name in orders.items():
                if fld != "Order" and name != main:
                    add(ids.get(name), IMMEDIATE, name, "toggle", u, a, state=(a, fld == "Orderon"))
        if u in MAP_SHOPS:  # (a neutral building's own: given by the player's units, below)
            continue
        for a in U[u]["abil"]:
            ab = A.get(a)
            if ab is None or a in SKIP or ab["base"] in SKIP or ab.get("hero"):
                continue
            orders = _ability_orders(a)
            if not orders:
                continue
            base = ab["base"]
            kind = KIND.get(a, KIND.get(base))
            if kind is None:
                kind = -1  # (no kind known: test_fullvocab names it)
            src, form = ab.get("data_a"), (ab.get("made") or [""])[0]
            morph = base in MORPH_BASES and src in U and form in U
            mana, cool = (ab["cost"] or [0])[0], (ab["cool"] or [0])[0]
            needs = _needs(a)
            for fld, name in orders.items():
                if fld == "Order":
                    if morph and u != src:
                        continue
                    pair = (base, True) if "Unorder" in orders and not morph else None  # (a root, its unroot)
                    add(ids.get(name), kind, name, "morph" if morph else "ability", u, a, needs, mana, cool, pair)
                elif fld == "Unorder" and morph:
                    if u == form:
                        add(ids.get(name), IMMEDIATE, name, "unmorph", u, a, None, 0.0, cool)
                else:
                    add(ids.get(name), IMMEDIATE, name, "toggle", u, a, needs, state=(base, fld == "Orderon"))
    # generic orders: a unit's moves and attacks, attack ground for artillery, a building's rally point and cancel
    g = ids
    for u in sorted(types):
        x = U[u]
        if u in MAP_SHOPS:
            continue
        mobile = not x["building"] and x["speed"] > 0
        armed = bool(x["weapon"]) and x["weapon"] not in ("_", "none")
        if mobile:
            for name, kind in (("move", POINT), ("smart", POINT), ("smart", UNIT), ("stop", IMMEDIATE),
                               ("patrol", POINT), ("holdposition", IMMEDIATE)):
                add(g[name], kind, name, "generic", u)
            if armed:
                add(g["attack"], UNIT, "attack", "generic", u)
                add(g["attack"], POINT, "attack", "generic", u)
            if x["weapon"] in ("artillery", "aline"):
                add(g["attackground"], POINT, "attackground", "generic", u)
        elif x["building"]:
            f = F.get(u, {})
            if armed:
                add(g["attack"], UNIT, "attack", "generic", u)
                add(g["stop"], IMMEDIATE, "stop", "generic", u)
            if f.get("Trains"):
                add(g["setrally"], POINT, "setrally", "generic", u)
                add(g["setrally"], UNIT, "setrally", "generic", u)
            if f.get("Trains") or f.get("Researches") or f.get("Upgrade"):
                add(g["cancel"], IMMEDIATE, "cancel", "generic", u)
    # the heroes' items: bought at the race's shop (with what shopitems leaves out) or the goblin merchant, used from a
    # slot in each form, dropped, given, sold, picked up
    heroes = sorted(u for u in types if is_hero(u))
    shop = RACE_SHOPS[race]
    wares = list(STOCK[shop]) + list(EXTRA_ITEMS.get(shop, ())) + list(MERCHANT_ITEMS)
    for h in heroes:
        for c in wares:
            add(code(c), UNIT, f"buy {c}", "item", h)
        for k in range(6):
            for kind in (IMMEDIATE, UNIT, POINT):
                add(ids[f"useslot{k + 1}"], kind, f"useslot{k + 1}", "item", h)
        for oid, kind in ITEM_ORDERS:
            add(oid, kind, ITEM_NAMES[oid], "item", h)
    # the neutral buildings' wares (a tavern's heroes, a camp's mercenaries, the laboratory's units: a hero or a worker
    # buys them) and the laboratory's reveal
    buyers = heroes + [WORKERS[race]]
    for s in ("ntav", "nmer", "ngad"):
        for c in shops[s].get("Sellunits", ()):
            for b in buyers:
                add(code(c), UNIT, f"buy {c}", "ware", b)
    for b in buyers:
        add(ids.get("neutraldetectaoe"), POINT, "neutraldetectaoe", "neutral", b, "Andt")
    return out, frozenset(types)


def universe(race: str) -> dict[tuple[int, int], Entry]:
    """Every order a player of `race` can give: (order id, kind) -> Entry."""
    return _universe(race)[0]


def unit_types(race: str) -> frozenset[str]:
    """The unit types a player of `race` may own (its units, buildings, heroes, summons, morphed forms) or order (the map
    shops), and the neutral units it may buy."""
    return _universe(race)[1]


BACKPACK = ("Rhpm", "Ropm", "Rupm", "Repm")


def excluded(e: Entry) -> str | None:
    """Why an order of the universe is left out on purpose (EXCLUDED's key), or None."""
    if e.name in EXCLUDED_NAMES:
        return EXCLUDED_NAMES[e.name]
    if e.name.startswith("moveslot"):
        return "moveslot"
    if e.what == "research" and rawcode(e.oid) in BACKPACK:
        return "backpack"
    ab = tables()["abilities"].get(e.ability, {}) if e.ability else {}
    return EXCLUDED_ABILITIES.get(e.ability) or EXCLUDED_ABILITIES.get(ab.get("base", ""))


def covered(e: Entry, kinds: set) -> bool:
    """Whether a vocabulary with the order kinds `kinds` of e's order id (and `have`'s aliases) gives e: in its kind, or a
    spell or unit ability aimed at a place by a unit (a unit's place: the built-in AI's blizzard, earthquake, stasis trap,
    dispel...) or an immediate one with a target the AI gave it (raise dead, carrion scarabs on a corpse)."""
    if e.kind in kinds:
        return True
    if e.what in ("spell", "ability"):
        if e.kind in (POINT, UNIT) and kinds & {POINT, UNIT}:
            return True
        if e.kind == IMMEDIATE and UNIT in kinds:
            return True
    return False


def missing(vocab: dict, race: str) -> list[Entry]:
    """The orders of `race`'s universe a vocabulary lacks (none of EXCLUDED), in a fixed order (what, name, id, kind)."""
    have: dict[int, set] = {}
    for o, k in vocab["orders"]:
        have.setdefault(int(o), set()).add(int(k))
    out = []
    for key, e in sorted(universe(race).items(), key=lambda kv: (kv[1].what, kv[1].name, kv[0])):
        if excluded(e) is not None or covered(e, have.get(e.oid, set())):
            continue
        alias = ALIASES.get(key)
        if alias is not None and alias[1] in have.get(alias[0], set()):
            continue
        out.append(e)
    return out


def coverage(vocab: dict) -> dict:
    """Per race: the universe's orders and unit types `vocab` lacks -> {race: {"orders": n, "have": n, "missing": [Entry],
    "excluded": [EXCLUDED's keys], "types": n, "missing_types": [code]}}."""
    types = {rawcode(t) for t in vocab["types"]}
    out = {}
    for race in RACES:
        u = universe(race)
        keep = [e for e in u.values() if excluded(e) is None]
        miss = missing(vocab, race)
        ut = unit_types(race)
        out[race] = {"orders": len(keep), "have": len(keep) - len(miss), "missing": miss,
                     "excluded": sorted({excluded(e) for e in u.values()} - {None}),
                     "types": len(ut), "missing_types": sorted(ut - types)}
    return out


# ---- the vocabulary and the network ----------------------------------------------------------------------------------

LAB_UNITS = ("ngsp", "nzep", "ngir")  # (the goblin laboratory's: protocol.Buy at the "ngad" on the map)
# the workshop's siege engine: hmtt before the barrage research, hrtt after it (each refused otherwise, scripts/vocab_probe.py;
# "!": the player has not that research)
SIEGE = {"hmtt": (("!Rhrt", 1),), "hrtt": (("Rhrt", 1),)}
EXTRA_NEEDS = {"tgrh": "ofrt"}  # (Units\\ItemFunc.txt Requires of EXTRA_ITEMS)
# how a hero uses EXTRA_ITEMS (shopitems.USES' forms): the critter at once, the tiny great hall at a point
EXTRA_USES = {"mcri": (IMMEDIATE,), "tgrh": (POINT,)}
FREE_BIT, HELD0, MANA0 = 19, 20, 26  # (model.FullGameNet.unit_valid's bits: a free slot, slot k holds an item, mana >= a step)
PICKUP_RADIUS = 400  # (pick up: the item lying nearest the point within this, protocol.PickUpAt)


def extended(vocab: dict) -> bool:
    return bool(vocab.get("full_vocab"))


@lru_cache(1)
def merged() -> dict[tuple[int, int], Entry]:
    """The four races' universes in one: each order's givers of every race (needs: TechMask reads its race's own)."""
    out: dict[tuple[int, int], Entry] = {}
    for race in RACES:
        for k, e in universe(race).items():
            m = out.get(k)
            if m is None:
                out[k] = Entry(e.oid, e.kind, e.name, e.what, set(e.givers), e.ability, e.needs, dict(e.mana), dict(e.cool),
                               e.state)
            else:
                m.givers |= e.givers
                m.mana.update(e.mana)
                m.cool.update(e.cool)
    return out


def entry(oid: int, kind: int) -> Entry | None:
    """The universe's entry of an order class, or that of its order id in another target form a vocabulary has it in
    (covered: a spell aimed at a unit's place, raise dead on a corpse)."""
    m = merged()
    e = m.get((int(oid), int(kind)))
    if e is None:
        e = next((m[(int(oid), k)] for k in (POINT, UNIT, IMMEDIATE) if (int(oid), k) in m
                  and m[(int(oid), k)].what in ("spell", "ability")), None)
    return e


def extend(vocab: dict) -> dict:
    """`vocab` with every order of the four races' universes it lacks (missing: none EXCLUDED) and their unit types at its
    end, each race's in turn -> a new vocabulary ("full_vocab": 1; "full_vocab_from": the old one's sizes, allow's)."""
    if extended(vocab):
        return vocab
    orders = [list(o) for o in vocab["orders"]]
    have = {tuple(o) for o in orders}
    names = dict(vocab.get("order_names") or {})
    for race in RACES:
        for e in missing({**vocab, "orders": orders}, race):
            if (e.oid, e.kind) in have:
                continue
            have.add((e.oid, e.kind))
            orders.append([e.oid, e.kind])
            if e.oid < fx.TYPE_CODE:
                names.setdefault(str(e.oid), ITEM_NAMES.get(e.oid, e.name))
    types = list(vocab["types"])
    have_t = {rawcode(t) for t in types}
    for race in RACES:
        for c in sorted(unit_types(race)):
            if c not in have_t:
                have_t.add(c)
                types.append(code(c))
    counts = dict(vocab.get("counts") or {})
    if "orders" in counts:
        counts["orders"] = list(counts["orders"]) + [0] * (len(orders) - len(counts["orders"]))
    return {**vocab, "types": types, "orders": orders, "order_names": names, "counts": counts, "full_vocab": 1,
            "full_vocab_from": {"orders": len(vocab["orders"]), "types": len(vocab["types"])}}


def bias_of(oid: int, kind: int) -> float:
    """A new order's bias below "no order" (BIASES by its category; one a worker gives at least the build's: ~15 workers
    a side, a toggle at 7 had been ~20 orders a game each)."""
    e = entry(oid, kind)
    b = BIASES.get(e.what if e is not None else "", 9.0)
    if e is not None and set(e.givers) & set(WORKERS.values()):
        b = max(b, BIASES["build"])
    return b


GENERIC_IDS = frozenset((851971, 851972, 851983, 851986, 851988, 851990, 851993, 851984, 851980, 851976, 852018, 852024,
                         852202, 852161))


def _like_orders(old: dict, oid: int, kind: int) -> list[int]:
    """The old vocabulary's order classes a new order's embedding is the mean of: its kind's (trains, builds, research,
    a neutral building's wares, a slot's uses, the unit abilities of its target kind...)."""
    e = entry(oid, kind)
    what = e.what if e is not None else ""
    oc = [(c, int(o), int(k)) for c, (o, k) in enumerate(old["orders"], start=1)]
    T = fx.TYPE_CODE
    slots = range(fx.USESLOT, fx.USESLOT + 6)
    if what == "generic":
        like = {851990: (851986, POINT), 851993: (851972, IMMEDIATE), 851984: (851983, POINT),
                851980: (851986, POINT) if kind == POINT else (851971, UNIT)}.get(oid)
        return [c for c, o, k in oc if (o, k) == like]
    if what in ("train", "upgrade"):
        return [c for c, o, k in oc if o >= T and k == IMMEDIATE and rawcode(o)[0] != "R" and not is_hero(rawcode(o))]
    if what == "research":
        return [c for c, o, k in oc if o >= T and k == IMMEDIATE and rawcode(o)[0] == "R"]
    if what == "build":
        return [c for c, o, k in oc if o >= T and k == POINT]
    if what == "ware":
        from .neutral import MERCENARIES, TAVERN_HEROES
        return [c for c, o, k in oc if o >= T and k == UNIT and rawcode(o) in (*MERCENARIES, *TAVERN_HEROES)]
    if what == "item":
        if oid >= T:
            from .neutral import MERCHANT_ITEMS
            return [c for c, o, k in oc if o >= T and k == UNIT and rawcode(o) in MERCHANT_ITEMS]
        if oid == PICKUP_AT:
            return [c for c, o, k in oc if (o, k) == (990004, IMMEDIATE)] or [c for c, o, k in oc if o in slots]
        return [c for c, o, k in oc if o in slots and k == kind] or [c for c, o, k in oc if o in slots]
    if what == "skill":
        return [c for c, o, k in oc if k == SKILL]
    ab = [c for c, o, k in oc if o < T and o not in GENERIC_IDS and o not in slots and k != SKILL and not 990000 <= o < 991000]
    if what in ("toggle", "unmorph"):
        return [c for c in ab if oc[c - 1][2] == IMMEDIATE]
    return [c for c in ab if oc[c - 1][2] == kind] or ab


def _like_types(old: dict, c: str) -> list[int]:
    """The old vocabulary's unit types a new type's embedding is the mean of: SIMILAR's, else its kind's (heroes, buildings,
    the other units)."""
    tix = {rawcode(t): i for i, t in enumerate(old["types"], start=1)}
    got = [tix[s] for s in SIMILAR.get(c, ()) if s in tix]
    if got:
        return got
    U = tables()["units"]
    if is_hero(c):
        return [i for s, i in tix.items() if is_hero(s)]
    building = U.get(c, {}).get("building", False)
    return [i for s, i in tix.items() if not is_hero(s) and U.get(s, {}).get("building", False) == building]


@torch.no_grad()
def grow(net, old: dict, new: dict):
    """`net` (of vocabulary `old`) grown to `new` (extend's): a new order's row in the order head is the "no order" row's
    with its bias less bias_of's (its logit no order's less that, in any state: every old logit is as before), its
    embedding the mean of its kind's old orders (_like_orders: the target head aims it as it aims those); a new unit
    type's embedding the mean of similar types' (_like_types); who gives them: allow."""
    from .model import FullGameNet
    T0, T1 = len(old["types"]) + 1, len(new["types"]) + 1
    O0, O1 = len(old["orders"]) + 1, len(new["orders"]) + 1
    if (T0, O0) == (T1, O1):
        return net
    if [list(o) for o in new["orders"][:O0 - 1]] != [list(o) for o in old["orders"]] or \
            list(new["types"][:T0 - 1]) != list(old["types"]):
        raise ValueError("fullvocab.grow: the new vocabulary must list the old one's orders and types first, in their order")
    config = {**net.config, "n_types": T1, "n_orders": O1}
    valid = config.pop("valid", 0)
    out = FullGameNet(**config).to(net.order_kind.device)
    state = {k: v for k, v in net.state_dict().items() if not k.startswith("valid_")}
    te = state["type_emb.weight"]
    rows = []
    for t in new["types"][T0 - 1:]:
        like = _like_types(old, rawcode(t))
        rows.append(te[like].mean(0) if like else te[1:].mean(0))
    if rows:
        state["type_emb.weight"] = torch.cat([te, torch.stack(rows)])
    w, b = state["order.2.weight"], state["order.2.bias"]
    added = [(int(o), int(k)) for o, k in new["orders"][O0 - 1:]]
    if added:
        state["order.2.weight"] = torch.cat([w, w[:1].expand(len(added), -1).clone()])
        state["order.2.bias"] = torch.cat([b, torch.stack([b[0] - bias_of(o, k) for o, k in added]).to(b.dtype)])
        oe = state["order_emb.weight"]
        emb = []
        for o, k in added:
            like = _like_orders(old, o, k)
            emb.append(oe[like].mean(0) if like else oe[1:].mean(0))
        state["order_emb.weight"] = torch.cat([oe, torch.stack(emb)])
    allowed = torch.zeros(T1, O1, dtype=torch.bool, device=te.device)
    allowed[:T0, :O0] = state["allowed"]
    state["allowed"] = allowed
    state["order_kind"] = torch.as_tensor([0] + [int(k) for _, k in new["orders"]], device=te.device)
    out.load_state_dict(state, strict=False)
    if valid:
        out.set_valid(new)
    allow(out, new)
    return out.train(net.training)


@torch.no_grad()
def allow(net, vocab: dict) -> None:
    """Who may give what (net's allowed table, in place), for the orders and unit types extend added only: a new order its
    givers (the universe's unit types with it: a morph its source form, its undo the morphed form), a new unit type the
    old orders it has. The old types' old orders as they were."""
    frm = vocab.get("full_vocab_from") or {}
    O0 = int(frm.get("orders", len(vocab["orders"]))) + 1
    T0 = int(frm.get("types", len(vocab["types"]))) + 1
    T, O = net.allowed.shape
    tix = {rawcode(t): i for i, t in enumerate(vocab["types"], start=1)}
    for c, (o, k) in enumerate(vocab["orders"], start=1):
        if c >= O:
            break
        e = entry(o, k)
        if c >= O0:
            net.allowed[:, c] = False
        if e is None:
            continue
        for g in e.givers:
            i = tix.get(g)
            if i is not None and i < T and (c >= O0 or i >= T0):
                net.allowed[i, c] = True


# ---- the masks ---------------------------------------------------------------------------------------------------------

@lru_cache(1)
def mana_steps() -> tuple[float, ...]:
    """The mana costs of the universe's unit abilities, each a bit of model.FullGameNet.unit_valid (MANA0 on)."""
    return tuple(sorted({float(v) for e in merged().values() for v in e.mana.values() if v > 0}))


def valid_needs(vocab: dict) -> dict[tuple[int, int], int]:
    """herodata.order_needs' entries for extend's orders: (unit type index, order class) -> the bit unit_valid checks: a
    unit ability's mana (MANA0 + its step in mana_steps), an item's drop, give or sale its slot holding one (HELD0 + k), a
    pick up a free slot (FREE_BIT); type -1: any unit type."""
    frm = vocab.get("full_vocab_from") or {}
    O0 = int(frm.get("orders", len(vocab["orders"]))) + 1
    tix = {rawcode(t): i for i, t in enumerate(vocab["types"], start=1)}
    steps = mana_steps()
    out: dict[tuple[int, int], int] = {}
    for c, (o, k) in enumerate(vocab["orders"], start=1):
        if c < O0:
            continue
        o = int(o)
        if DROP0 <= o < SELL0 + 6:
            out[(-1, c)] = HELD0 + (o - DROP0) % 10
            continue
        if o == PICKUP_AT:
            out[(-1, c)] = FREE_BIT
            continue
        e = entry(o, k)
        if e is None or e.what not in ("ability", "morph"):
            continue
        for g, m in e.mana.items():
            if m > 0 and g in tix:
                out[(tix[g], c)] = MANA0 + steps.index(float(m))
    return out


def tech_needs(vocab: dict, race: str) -> dict[int, list[tuple[str, int]]]:
    """features.TechMask's entries for extend's orders a player of `race` gives: order class -> [(research, level) or
    (building, 0)] it needs (every ability with the order agreeing), and EXTRA_ITEMS' race shop and needs ("-" never:
    another race's shop)."""
    from .shopitems import RACE_SHOPS
    frm = vocab.get("full_vocab_from") or {}
    O0 = int(frm.get("orders", len(vocab["orders"]))) + 1
    u = universe(race)
    out: dict[int, list[tuple[str, int]]] = {}
    for c, (o, k) in enumerate(vocab["orders"], start=1):
        if c < O0:
            continue
        e = u.get((int(o), int(k)))
        if e is not None and e.needs:
            out[c] = list(e.needs)
        r = rawcode(o) if int(o) >= fx.TYPE_CODE else ""
        if r in SIEGE and int(k) == IMMEDIATE:
            out[c] = list(SIEGE[r])
        for shop, items in EXTRA_ITEMS.items():
            if r in items and int(k) == UNIT:
                out[c] = ([(shop, 0)] + ([(EXTRA_NEEDS[r], 0)] if r in EXTRA_NEEDS else [])
                          if RACE_SHOPS[race] == shop else [("-", 0)])
    return out


@lru_cache(1)
def lab_stock() -> dict[str, float]:
    """The goblin laboratory's units' first game second (UnitBalance stockStart, as neutral.stock_times' wares)."""
    from .. import paths
    path = paths.CACHE_DIR / "full_vocab-stock.json"
    if path.exists():
        return {k: float(v) for k, v in json.loads(path.read_text()).items()}
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        ub = {r.get("unitBalanceID"): r for r in parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))}
    out = {}
    for c in LAB_UNITS:
        try:
            out[c] = float((ub.get(c) or {}).get("stockStart") or 0)
        except ValueError:
            out[c] = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


RACE_SHOP_CODES = tuple(code(c) for c in ("hvlt", "ovln", "utom", "eden"))
PAWN_SHOPS = tuple(code(c) for c in ("ngme", "nmrk"))  # (the neutral shops that buy items: Apit)
LAB = code("ngad")
GOLD_MINE = code("ngol")
# a partner the order needs, by order string: (who gives it, whom it takes): a hippogryph rider's two halves, an acolyte
# and the sacrificial pit, a spirit walker and its dead tauren, a tree of life and a gold mine
PARTNERS = {"coupletarget": {"earc": "ehip", "ehip": "earc"}, "sacrifice": {"usap": "uaco"},
            "requestsacrifice": {"uaco": "usap"}, "entangle": {"etol": "ngol", "etoa": "ngol", "etoe": "ngol"}}
# the autocasts a computer player's units start with on (scripts/vocab_probe.py: "off" taken on a fresh unit), and the
# rooted ancients; every other on / off pair starts off. A toggle the game refuses is in the state it asked for already
DEFAULT_ON = frozenset({"Ahea", "Anhe", "Aslo", "Asps", "Acrs", "Aweb", "Afae", "Apsh", "Aadm", "Avng", "Aroo"})
NIGHT_ONLY = ("ambush",)  # (shadow meld: refused by day)
ATTACK_GROUND = 851984  # (refused beyond the unit's range: not sent then)


class Checks:
    """What extend's orders need of the player's units now: View.step's mask (any unit of the player's can give it now),
    play.BCAgent's per unit (not sent when that one can't): a unit ability's mana and cooldown (the cooldown from the
    order's accepted time: View.cooling), a partner (PARTNERS), a hero's item in the slot, a shop to sell to, an item on the
    ground, the goblin laboratory on the map."""

    def __init__(self, vocab: dict):
        frm = vocab.get("full_vocab_from") or {}
        O0 = int(frm.get("orders", len(vocab["orders"]))) + 1
        self.timed: dict[int, dict[int, tuple[float, float]]] = {}  # class -> {type id: (mana, cooldown s)}
        self.partner: dict[int, dict[int, int]] = {}  # class -> {giver type id: partner type id}
        self.item: dict[int, tuple[str, int]] = {}  # class -> (drop / give / sell, slot)
        self.pickup = self.reveal = None
        self.lab: list[int] = []
        self.toggle: dict[int, tuple[str, bool, frozenset]] = {}  # class -> (ability, the state it sets, giver type ids)
        self.night: list[int] = []
        self.ground = None  # (attack ground's class: its unit's range)
        self.range = {code(u): float(x.get("range") or 0) for u, x in tables()["units"].items()}
        self.extra_items = frozenset(c for items in EXTRA_ITEMS.values() for c in items)
        names = {int(k): v for k, v in (vocab.get("order_names") or {}).items()}
        self.names = {c: fx.order_label(int(o), int(k), names) for c, (o, k) in enumerate(vocab["orders"], start=1)
                      if c >= O0}
        for c, (o, k) in enumerate(vocab["orders"], start=1):
            if c < O0:
                continue
            o, k = int(o), int(k)
            if DROP0 <= o < SELL0 + 6:
                self.item[c] = (("drop", "give", "sell")[(o - DROP0) // 10], (o - DROP0) % 10)
                continue
            if o == PICKUP_AT:
                self.pickup = c
                continue
            if o >= fx.TYPE_CODE and rawcode(o) in LAB_UNITS:
                self.lab.append(c)
                continue
            e = entry(o, k)
            if e is None:
                continue
            if e.what == "neutral":
                self.reveal = c
            if e.name in PARTNERS:
                self.partner[c] = {code(g): code(p) for g, p in PARTNERS[e.name].items()}
            if e.name in NIGHT_ONLY:
                self.night.append(c)
            if o == ATTACK_GROUND:
                self.ground = c
            if e.state is not None:
                self.toggle[c] = (e.state[0], bool(e.state[1]), frozenset(code(g) for g in e.givers))
            if e.what in ("ability", "morph", "unmorph") and (e.mana or e.cool):
                self.timed[c] = {code(g): (float(e.mana.get(g, 0.0)), float(e.cool.get(g, 0.0)))
                                 for g in set(e.mana) | set(e.cool)}
        # every on / off order's (old ones too: a vocabulary's healon, immolation): the state it leaves its unit in
        self.toggle_ids = {e.oid: (e.state[0], bool(e.state[1])) for e in merged().values() if e.state is not None}

    def unit_ok(self, c: int, row, t: int, cooling: dict, states: dict | None = None) -> bool:
        """Whether the unit of `row` can give extend's order class `c` now: its mana and cooldown (`cooling`: (unit, class)
        -> the step it is ready again), a toggle not in the state it asks for already (`states`: (unit, ability) -> on)."""
        tg = self.toggle.get(c)
        if tg is not None and states is not None and self.state(states, int(row[fx.C_ID]), tg[0]) == tg[1]:
            return False
        need = self.timed.get(c, {}).get(int(row[fx.C_TYPE]))
        if need is None:
            return True
        return row[fx.C_MANA] >= need[0] and cooling.get((int(row[fx.C_ID]), c), -1) <= t

    @staticmethod
    def state(states: dict, unit: int, ability: str) -> bool:
        """A unit's autocast (or rooted) state: as the agent's last toggle set it, else DEFAULT_ON's."""
        got = states.get((unit, ability))
        return ability in DEFAULT_ON if got is None else got

    @staticmethod
    def is_night(t: int, step_s: float) -> bool:
        """Night in the game at step t (features' clock: a day 480 s, dawn at 6, dusk at 18, the game at 8)."""
        hour = (fx.START_HOUR + t * step_s / fx.DAY_SECONDS * 24.0) % 24.0
        return hour >= fx.DUSK or hour < fx.DAWN

    def mask(self, view, rows, avail, t: int) -> None:
        """`avail` (in place) without extend's orders no unit of the view's player can give now."""
        own = fx.own_alive(rows, view.player)
        cooling = getattr(view, "cooling", {})
        states = getattr(view, "autocast", {})
        types = own[:, fx.C_TYPE]
        # (the unit types the player has: most classes are another race's, no giver of theirs here, and one np.isin per
        # class had been ~0.7 ms a side-step, more than all the rest of the actor's Python per step)
        present = set(types.tolist())
        for c, need in self.timed.items():
            if c >= len(avail) or not avail[c]:
                continue
            if present.isdisjoint(need):
                avail[c] = False
                continue
            rows_c = own[np.isin(types, list(need))]
            avail[c] = any(self.unit_ok(c, r, t, cooling, states) for r in rows_c)
        for c, (ability, on, givers) in self.toggle.items():  # (a toggle while a giver is in the other state)
            if c >= len(avail) or not avail[c] or c in self.timed:
                continue
            if present.isdisjoint(givers):
                avail[c] = False
                continue
            ids = own[np.isin(types, list(givers)), fx.C_ID].tolist()
            avail[c] = any(self.state(states, int(u), ability) != on for u in ids)
        if self.night and not self.is_night(t, getattr(view, "step_seconds", 0.5)):
            for c in self.night:
                if c < len(avail):
                    avail[c] = False
        for c, pairs in self.partner.items():
            if c >= len(avail) or not avail[c]:
                continue
            ok = False
            for g, p in pairs.items():
                if not (types == g).any():
                    continue
                if p == GOLD_MINE:  # (a gold mine anyone may entangle: one not entangled yet, on the map)
                    ok = ok or bool((rows[:, fx.C_TYPE] == GOLD_MINE).any())
                elif p == code("otau"):  # (a dead tauren of the player's lying there)
                    ok = ok or bool(((rows[:, fx.C_OWNER] == view.player) & (rows[:, fx.C_TYPE] == p)
                                     & (rows[:, fx.C_FLAGS] & fx.DEAD > 0)).any())
                else:
                    ok = ok or bool(((types == p) & (own[:, fx.C_FLAGS] & fx.CONSTRUCTING == 0)).any())
            avail[c] = ok
        heroes = own[own[:, fx.C_FLAGS] & fx.HERO > 0] if own.shape[1] > fx.C_ITEM0 + 5 else own[:0]
        if self.item:
            shop = bool(np.isin(rows[:, fx.C_TYPE], PAWN_SHOPS).any()) or bool(
                (np.isin(types, RACE_SHOP_CODES) & (own[:, fx.C_FLAGS] & fx.CONSTRUCTING == 0)).any())
            for c, (verb, k) in self.item.items():
                if c >= len(avail) or not avail[c]:
                    continue
                held = bool((heroes[:, fx.C_ITEM0 + k] != 0).any()) if len(heroes) else False
                avail[c] = held and (verb != "give" or len(heroes) > 1) and (verb != "sell" or shop)
        if self.pickup is not None and self.pickup < len(avail) and avail[self.pickup]:
            bit = 1 << view.player
            seen = any(vis & bit for _, _, _, vis, _ in (getattr(view, "ground", ()) or ()))
            avail[self.pickup] = seen and len(heroes) > 0 and bool((heroes[:, fx.C_ITEM0:fx.C_ITEM0 + 6] == 0).any())
        lab = bool((rows[:, fx.C_TYPE] == LAB).any())
        for c in self.lab + ([self.reveal] if self.reveal is not None else []):
            if c < len(avail):
                avail[c] = avail[c] and lab
