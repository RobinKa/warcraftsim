"""--order-masks (features.ORDER_MASKS): orders the game refuses that no other mask catches (bug finder 10's refusals in
fgself-224/225's episode rows; "refused" and "refused_ctx" stay the measure). Off (0) nothing here runs.

Bit 1, spell targets: the pointer (the entity a unit-target order aims at) only on the entities the order's abilities
let it take, by AbilityData.slk's targs1 (the union of every ability with that order string: an entity is refused only
when all of them refuse it), and the order only while some entity in view takes it. From the entities' features and
types alone (model.FullGameNet "omask": acting, the native loop's graphs and the update compute the same mask):

    side      own units when the targs name a side but none of friend, player, ally, self; the enemy's when not
              "enemy"; the neutral entities (creeps and critters, shops, mines) when neither "enemy" nor "neutral"
    itself    the caster when "notself" (Holy Light, Death Coil, Siphon Mana), or when own units are refused (Storm
              Bolt, Mana Burn, Frost Nova on the hero itself)
    class     a building without "structure", a flying unit without "air", a ground unit without "ground" (none
              when the targs name no class, or "ward": a ward is a ground unit to the features)
    kind      a machine with "organic" (and not "mechanical"), a hero with "nonhero", an ancient with "nonancient"
              (by unit type: herodata.unit_kinds; an unknown type is none of them)
    alive     every unit, when the targs take the dead only (raise dead, carrion scarabs, cannibalize aimed at a
              unit: corpses are no pointer targets)
    unseen    an enemy a privileged view has out of the player's sight (opsd's teacher: the order mask reads only what
              the player sees, as the policy's own)

Left out on purpose (not sure the game refuses, or not in the features): a unit hidden in a gold mine or loaded in a
burrow (scripts/order_masks_probe.py: the game took frost nova on a burrowed peon and on one in a mine, refused storm
bolt on another in the mine, a miner inside ~1 s at a time; fgself-225's refusal contexts had ~5,500 frost novas on
burrowed peons refused), inputs 3's remembered enemies, invulnerable and magic immune targets (some abilities take
them), wards, sappers, a target at full health or mana (Heal was refused on hurt units as often), the generic orders
(attack, smart, move), harvest and the item orders.

Bit 2, toggles against their state (View.toggles, the player's avail): an "on" order only while a unit of the
player's that has the ability is off (or its state unknown), the "off" only while one is on: immolation and mana
shield, from obs-full's buff events (101 a unit's state anew, its buffs in the 102s after it: teacher.Teacher.note_buffs'
reading; a unit with no event yet is unknown, never masked; the probe saw BEim and BNms come and go with them). A demon
hunter and a sea witch are one a player (heroes are unique): their masks are exact. And by form (model.load: the allowed
table): the tinker's robo-goblin from the robo-goblin, its way back from the tinker. Not defend: a defending footman
shows no buff (the probe), and its research (Rhde) is not in the tech mask for that old order.

Bit 4, learns: a skill is learned only by a hero type with it in a slot (herodata.order_needs gives every other type
the slot's skill-point bit, never set off a hero's own slots). The allowed table lets an unknown unit type give any
order (bc: row 0), and the maps' creep types obs-full added copied that row: a summoned clockwerk goblin and a hexed
hero (a critter) learned skills, ~1,900 refused in fgself-225's first 25,441 games.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import features as fx

# a pointer rule's bits (model.FullGameNet's "ptr_rule", per order class): the entities with a property the order refuses
OWN, ENEMY, NEUTRAL, SELF, STRUCT, AIR, GROUND, HERO, MECH, ANCIENT, UNSEEN, ALIVE = (1 << k for k in range(12))
BIT_NAMES = ("own", "enemy", "neutral", "itself", "building", "air", "ground", "hero", "machine", "ancient", "unseen",
             "alive")
SIDES = frozenset({"enemy", "friend", "player", "ally", "neutral"})
# unit-target orders left out: the generic ones and those with masks of their own or targets the targs don't describe
# (a gold mine, a tree, a dead hero at an altar, a hall for the militia)
SKIP_NAMES = frozenset({"smart", "attack", "attackonce", "move", "patrol", "harvest", "entangle", "revive", "board",
                        "militiaconvert", "militiaunconvert", "setrally", "acolyteharvest", "returnresources",
                        "detonate"})  # (detonate: a wisp's blast, --econ-masks 8)
# the entity features the rules read (features.entities: the side one-hot, then the flags)
CH_OWN, CH_ENEMY, CH_NEUTRAL = 0, 1, 2
CH_HERO, CH_STRUCT, CH_FLYING = (9 + fx.FLAG_BITS.index(b) for b in (1, 2, 2048))


def targs_rule(targs: str) -> int:
    """One ability's targs1 -> the bits of the entities it surely refuses (0: no rule, it takes any)."""
    t = {x.strip().lower() for x in str(targs or "").split(",") if x.strip()} - {"_"}
    if not t or t == {"none"}:
        return 0
    r = UNSEEN
    side = t & SIDES
    if side:
        if not side & {"friend", "player", "ally"} and "self" not in t:
            r |= OWN
        if "enemy" not in side:
            r |= ENEMY
        if not side & {"enemy", "neutral"}:  # (the features' neutral: creeps, which are enemies, and critters, shops)
            r |= NEUTRAL
    if "notself" in t or r & OWN:
        r |= SELF
    cls = t & {"air", "ground", "structure"}
    if cls:
        if "structure" not in cls:
            r |= STRUCT
        if "air" not in cls:
            r |= AIR
        if "ground" not in cls and "ward" not in t:
            r |= GROUND
    if "organic" in t and "mechanical" not in t:
        r |= MECH
    if "nonhero" in t:
        r |= HERO
    if "nonancient" in t:
        r |= ANCIENT
    if "dead" in t and "alive" not in t:
        r |= ALIVE
    return r


@lru_cache(1)
def _abilities_by_order() -> dict[str, list[str]]:
    """Order string -> the abilities (codes) whose cast it is: the game's files' (fullvocab.tables, with BASE_ORDERS'
    fields) and the hero abilities' (data.abilities: the orders the files do not give)."""
    from ..data.abilities import ability_info
    from . import fullvocab
    out: dict[str, list[str]] = {}
    for a in fullvocab.tables()["abilities"]:
        o = fullvocab._ability_orders(a).get("Order")
        if o:
            out.setdefault(o, []).append(a)
    for a, i in ability_info().items():
        if i.order and a not in out.get(i.order, ()):
            out.setdefault(i.order, []).append(a)
    return out


def order_rule(name: str) -> int:
    """An order string's rule: the entities every ability cast with it refuses (bits AND-ed: 0 when one takes any)."""
    if not name or name in SKIP_NAMES:
        return 0
    from . import fullvocab
    abil = fullvocab.tables()["abilities"]
    codes = _abilities_by_order().get(name, ())
    if not codes:
        return 0
    r = -1
    for a in codes:
        r &= targs_rule(abil.get(a, {}).get("targs", ""))
    return max(r, 0)


def pointer_rules(vocab: dict) -> list[int]:
    """Per order class (0: none) its pointer rule: the unit-target orders of spells and abilities, else 0."""
    from . import fullvocab
    names = {int(k): v for k, v in (vocab.get("order_names") or {}).items() if v}
    by_id = fullvocab._names_by_id()
    universe = fullvocab.merged()
    out = [0]
    for o, k in vocab["orders"]:
        o, k = int(o), int(k)
        e = universe.get((o, k)) if k == fx.UNIT and o < fx.TYPE_CODE else None
        # (only a spell or ability cast on a unit: an area spell's unit form, blizzard or impale aimed at a unit's place,
        # takes the place and any unit there; a vocabulary's order the universe has in another form is such a one)
        out.append(order_rule(by_id.get(o) or names.get(o, "")) if e is not None and e.what in ("spell", "ability")
                   else 0)
    return out


def type_kinds(vocab: dict) -> list[int]:
    """Per unit type index (0: unknown, none) its kind bits: MECH, ANCIENT (Units\\UnitBalance.slk's type column)."""
    from .herodata import unit_kinds
    k = unit_kinds()
    out = [0]
    for t in vocab["types"]:
        t = int(t)
        out.append((MECH if t in k["mechanical"] else 0) | (ANCIENT if t in k["ancient"] else 0))
    return out


def rule_names(rule: int) -> list[str]:
    return [n for b, n in enumerate(BIT_NAMES) if rule >> b & 1]


# ---- bit 2: toggles against their state ------------------------------------------------------------------------------

# (the ability, its buff while on, the order that turns it on, the one that turns it off)
TOGGLE_BUFFS = (("AEim", "BEim", "immolation", "unimmolation"), ("ANms", "BNms", "manashieldon", "manashieldoff"))
# a form's order the game refuses (its type -> the order string): the robo-goblin morphs again, the tinker turns back
FORM_REFUSED = {"Nrob": ("robogoblin",), "Ntin": ("unrobogoblin",)}
EV_STATE, EV_BUFF = 101, 102  # (obsfull's events)


def fourcc(c: str) -> int:
    return int.from_bytes(c.encode("latin-1"), "big")


@lru_cache(1)
def _toggle_givers() -> dict[str, frozenset[int]]:
    """Ability -> the unit types (ids) that have it (the game's tables: the hero, its morphed forms)."""
    from . import fullvocab
    U = fullvocab.tables()["units"]
    out = {}
    for a, _, _, _ in TOGGLE_BUFFS:
        out[a] = frozenset(fourcc(u) for u, x in U.items() if a in x.get("abil", ()) or a in x.get("hero_abil", ()))
    return out


class Toggles:
    """Bit 2's per-player state: each unit's buffs as obs-full's events give them, and the toggles' order classes."""

    def __init__(self, enc):
        oix, ids = enc.order_index, _order_ids()
        self.classes = []  # [(giver type ids, buff code, the on class or None, the off class or None)]
        givers = _toggle_givers()
        for a, buff, on, off in TOGGLE_BUFFS:
            c_on, c_off = (oix.get((ids.get(n, -1), fx.IMMEDIATE)) for n in (on, off))
            if c_on is not None or c_off is not None:
                self.classes.append((givers[a], fourcc(buff), c_on, c_off))
        self.watch = frozenset().union(*(g for g, _, _, _ in self.classes)) if self.classes else frozenset()
        self.buffs: dict[int, set] = {}  # (unit id -> its buffs' codes; none: unknown)
        self.masked = 0  # (order classes this mask turned off, summed over the steps: the episode row)

    def step(self, rows: np.ndarray, player: int, events: np.ndarray, avail: np.ndarray) -> None:
        """A step: the buffs of the player's units that have a toggle, from the step's events, then the mask."""
        if not self.classes:
            return
        units = [(u[fx.O_ID], u[fx.O_TYPE]) for u in fx.own_list(rows, player) if u[fx.O_TYPE] in self.watch]
        if units and len(events):
            self.note(events, [i for i, _ in units])
        self.mask(units, avail)

    def note(self, events, ids: list) -> None:
        """A step's events (View.step's [t, kind, a, b, c] rows, as Python's lists there) for the units `ids`: a unit's
        state anew (101) clears its buffs, each 102 after it adds one (Teacher.note_buffs' reading)."""
        if isinstance(events, np.ndarray):
            events = events.tolist()
        mine, buffs = set(ids), self.buffs
        for r in events:
            k, a = r[1], r[2]
            if (k == EV_STATE or k == EV_BUFF) and a in mine:
                if k == EV_STATE:
                    buffs[a] = set()
                else:
                    got = buffs.get(a)
                    if got is not None:
                        got.add(r[3])

    def mask(self, units: list, avail: np.ndarray) -> None:
        """`avail` (in place) without an on order no unit of the player's with its ability (`units`: [(id, type)]) can
        take now (none, or all known to be on), nor an off order (none, or all known to be off)."""
        buffs = self.buffs
        for givers, buff, c_on, c_off in self.classes:
            can_on = can_off = False
            for i, ty in units:
                if ty not in givers:
                    continue
                got = buffs.get(i)
                if got is None:  # (unknown: either)
                    can_on = can_off = True
                    break
                if buff in got:
                    can_off = True
                else:
                    can_on = True
                if can_on and can_off:
                    break
            for c, ok in ((c_on, can_on), (c_off, can_off)):
                if c is not None and avail[c] and not ok:
                    avail[c] = False
                    self.masked += 1


@lru_cache(1)
def _order_ids() -> dict[str, int]:
    from .fullvocab import order_ids
    return order_ids()


def form_refused(vocab: dict) -> list[tuple[int, int]]:
    """Bit 2's (unit type index, order class) pairs the game refuses by form (FORM_REFUSED), in `vocab`."""
    ids = _order_ids()
    tix = {fx.rawcode(int(t)): i for i, t in enumerate(vocab["types"], start=1)}
    oix = {(int(o), int(k)): c for c, (o, k) in enumerate(vocab["orders"], start=1)}
    out = []
    for form, names in FORM_REFUSED.items():
        i = tix.get(form)
        for n in names:
            c = oix.get((ids.get(n, -1), fx.IMMEDIATE))
            if i is not None and c is not None:
                out.append((i, c))
    return out
