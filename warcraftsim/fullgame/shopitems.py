"""The race shops' items as the policy's orders (selfplay --shop-items 1; needs --neutral).

A vocabulary held only the goblin merchant's 11 items, so the policy built its race's shop in ~42% of its games and could
never buy what that shop sells; people have a shop in 89-100% of their games and click an item 7-15 times a game. With
this flag each item of the race shops' stock (STOCK, the game's Units\\*UnitFunc.txt Makeitems) that can also be used
becomes a kind UNIT order, at the vocabulary's end, as neutral.py's are (extend): a hero's only (allow), bought at its own
side's finished shop of that kind (pick_shop; a hero walks there, protocol.Buy), or at the goblin merchant for what it
sells too. The orders a hero's item needs (its slot's immediate, on a unit or on a point) are the slot orders; the
vocabulary lacked the sixth slot's and every point form, so it gets them here.

What the game takes (scripts/shop_items_probe.py, the results at its top; every item on echofast2, 2026-11-01):

  * IssueNeutralImmediateOrderById works for a player's own shop too, so the harness's Buy (op 8) needs no change.
  * ITEM_NEEDS (Units\\ItemFunc.txt Requires) is enforced at a race shop; a shop's stockStart is not: a healing potion
    (stockStart 440) sold at second 32 once the tier stood.
  * A full inventory: the sale goes through, the gold is gone, the item lands on the ground. Masked (herodata 19).
  * UnitUseItem without a target answers True for items that need one and does nothing (a town portal), so the answer
    says little; UnitUseItemPoint answers False and acts (a portal teleports, an ivory tower is built). USES holds what
    each item took.
  * An agent's slot is a computer slot, and the engine has its heroes use some items by themselves in a fight (lesser
    clarity, mana potion, scroll of regeneration, salve). items_gone in the episode row counts those too; item_uses only
    the policy's orders.

A network grows to them (grow): their rows in the order head are the "no order" row's less a bias, as spells.py's (a
constant low bias had taken up to 7% of a hero's orders in odd states of fgself-153's network). Every old order's logit
stays as it was. Left out: the mechanical critter (mcri): it makes a critter of a type the vocabulary has not, no unit
type's orders reach it, so it could be bought and not used; and the tiny great hall (tgrh, 600 gold and 185 lumber at a
fortress): used by a point it stood once, in another game the item was gone with no hall anywhere, and in a third the engine
used it by itself, so it can't be told to work (people: 0.21 a game).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from . import features as fx

RACE_SHOPS = {"human": "hvlt", "orc": "ovln", "undead": "utom", "nightelf": "eden"}
MERCHANT = "ngme"
# what a race's shop sells as orders (Units\*UnitFunc.txt Makeitems, the mechanical critter and the tiny great hall left out)
STOCK = {"hvlt": ("sreg", "plcl", "phea", "pman", "stwp", "tsct", "ofir", "ssan"),
         "ovln": ("shas", "hslv", "plcl", "phea", "pman", "stwp", "oli2"),
         "utom": ("rnec", "dust", "skul", "phea", "pman", "stwp", "ocor", "shea"),
         "eden": ("moon", "plcl", "dust", "phea", "pman", "stwp", "spre", "oven", "pams")}
# what an item needs at a race shop (Units\ItemFunc.txt Requires; TWN2, TWN3: any hall of that tier, techtree.resolve)
ITEM_NEEDS = {"phea": "TWN2", "pman": "TWN2", "stwp": "TWN2", "tsct": "hkee", "ofir": "hcas", "ssan": "hcas",
              "oli2": "ofrt", "skul": "ugrv", "ocor": "unp2", "shea": "unp2", "spre": "etoa",
              "oven": "etoe", "pams": "etoe"}
NOW, UNIT, POINT = fx.IMMEDIATE, fx.UNIT, fx.POINT
# the order kinds an item's use takes (none: passive, it acts by being held). Immediate: the hero's own use, on itself.
USES: dict[str, tuple[int, ...]] = {
    "sreg": (NOW,), "plcl": (NOW,), "phea": (NOW,), "pman": (NOW,), "shas": (NOW,), "rnec": (NOW,), "dust": (NOW,),
    "shea": (NOW,), "moon": (NOW,), "pams": (NOW,), "hslv": (NOW, UNIT),  # (a salve: on itself, or on another unit)
    "stwp": (POINT,), "tsct": (POINT,), "skul": (POINT,), "ssan": (UNIT,), "spre": (UNIT,), "stel": (UNIT,),  # (the staff of
    # teleportation: the merchant's, on a building of the player's)
    "ofir": (), "oli2": (), "ocor": (), "oven": (),  # (orbs)
    "tret": (NOW,), "pnvl": (NOW,), "spro": (NOW,), "pinv": (NOW,), "bspd": (), "prvt": (), "cnob": ()}  # (the merchant's)
HEALS, RESTORES = ("phea", "hslv", "sreg", "shea"), ("plcl", "pman")  # (no use at full hit points; at full mana)
ITEMS = tuple(dict.fromkeys(c for s in STOCK.values() for c in s))  # (in their order in a vocabulary, after what it has)
SLOTS = tuple(range(fx.USESLOT, fx.USESLOT + 6))
SLOT_KINDS = (NOW, UNIT, POINT)
BUY_BIAS, USE_BIAS = 7.0, 7.0  # (an item's order, a new slot order: "no order"'s logit less this. A hero with a shop has
# some 8 items open at any step, and each costs money: 7 is about one order every 70 s from a rich hero. Not spells.NEW_BIAS'
# 3, which is 1/20 of no order: every hero starts with a town portal in its first slot, so that would spend it in seconds)
PORTAL_NEAR = 1100.0  # (a town portal lands near one of the player's structures: AItp's area)
PORTAL_STEPS = 11  # (its cast is 5 s: the policy's next order would cancel it)


def code(c: str) -> int:
    return int.from_bytes(c.encode("latin-1"), "big")


_USE = {code(c): k for c, k in USES.items()}
_HEAL, _MANA = frozenset(map(code, HEALS)), frozenset(map(code, RESTORES))
ITEM_NAMES = frozenset(ITEMS) | frozenset(USES)  # (the orders of the merchant's items as well)
ITEM_CODES = frozenset(map(code, ITEM_NAMES))
_PORTAL, _STAFF = code("stwp"), code("stel")


def extended(vocab: dict) -> bool:
    return bool(vocab.get("shop_items"))


def _orders() -> list[tuple[int, int]]:
    return [*((code(c), UNIT) for c in ITEMS), *((s, k) for s in SLOTS for k in SLOT_KINDS)]


def extend(vocab: dict) -> dict:
    """`vocab` with the race shops' items and the item slots' missing forms at its end -> a new vocabulary."""
    if extended(vocab):
        return vocab
    out = [list(o) for o in vocab["orders"]]
    have = {tuple(o) for o in out}
    for o in _orders():
        if o not in have:
            have.add(o)
            out.append(list(o))
    counts = dict(vocab.get("counts") or {})
    if "orders" in counts:
        counts["orders"] = list(counts["orders"]) + [0] * (len(out) - len(counts["orders"]))
    return {**vocab, "orders": out, "counts": counts, "shop_items": 1}


@torch.no_grad()
def grow(net, old: dict, new: dict):
    """`net` (of vocabulary `old`) with the orders `new` adds after old's: their order-head rows the "no order" row's less
    BUY_BIAS or USE_BIAS, an item's embedding the mean of the merchant's items', a slot form's the mean of what the slot
    orders have (a point form's that of the heroes' point spells, which the target head aims); who gives them: allow."""
    from .model import FullGameNet
    from .neutral import MERCHANT_ITEMS
    from .spells import _hero_spells
    O0, O1 = len(old["orders"]) + 1, len(new["orders"]) + 1
    if O0 == O1:
        return net
    if [list(o) for o in new["orders"][:O0 - 1]] != [list(o) for o in old["orders"]]:
        raise ValueError("shopitems.grow: the new vocabulary must list the old one's orders first, in their order")
    config = {**net.config, "n_orders": O1}
    valid = config.pop("valid", 0)
    out = FullGameNet(**config).to(net.order_kind.device)
    state = {k: v for k, v in net.state_dict().items() if not k.startswith("valid_")}
    w, b = state["order.2.weight"], state["order.2.bias"]
    added = [tuple(o) for o in new["orders"][O0 - 1:]]
    state["order.2.weight"] = torch.cat([w, w[:1].expand(O1 - O0, -1).clone()])
    state["order.2.bias"] = torch.cat([b, torch.stack([b[0] - (USE_BIAS if o in SLOTS else BUY_BIAS) for o, _ in added])])
    oe = state["order_emb.weight"]
    at = {tuple(o): c for c, o in enumerate(old["orders"], start=1)}
    wares = [at[(code(c), UNIT)] for c in MERCHANT_ITEMS if (code(c), UNIT) in at]
    slots = [c for (o, _), c in at.items() if o in SLOTS]
    spells = _hero_spells(old)

    def row(o: int, k: int) -> torch.Tensor:
        same = (slots if k != POINT else spells.get(POINT) or slots) if o in SLOTS else wares
        return oe[same].mean(0) if same else oe[1:].mean(0)
    state["order_emb.weight"] = torch.cat([oe, torch.stack([row(o, k) for o, k in added])])
    allowed = state["allowed"]
    state["allowed"] = torch.cat([allowed, allowed.new_zeros(allowed.shape[0], O1 - O0)], 1)
    state["order_kind"] = torch.as_tensor([0] + [k for _, k in new["orders"]], device=oe.device)
    out.load_state_dict(state, strict=False)
    if valid:
        out.set_valid(new)
    allow(out, new)
    return out.train(net.training)


@torch.no_grad()
def allow(net, vocab: dict) -> None:
    """Who may give them (net's allowed table, in place): a hero its items and every form of its slots' uses, no other
    unit type an item (a worker has no inventory)."""
    oix = {tuple(o): c for c, o in enumerate(vocab["orders"], start=1)}
    items = [oix[(code(c), UNIT)] for c in ITEMS if (code(c), UNIT) in oix]
    slots = [oix[(s, k)] for s in SLOTS for k in SLOT_KINDS if (s, k) in oix]
    heroes = [i for i, t in enumerate(vocab["types"], start=1) if fx.is_hero_code(fx.rawcode(int(t)))]
    items = [c for c in items if c < net.allowed.shape[1]]
    net.allowed[:, items] = False
    for i in heroes:
        if i < net.allowed.shape[0]:
            net.allowed[i, items] = True
            net.allowed[i, [c for c in slots if c < net.allowed.shape[1]]] = True


def sold(c: str, race: str, have: set, now: float, stock: dict) -> bool:
    """Whether the player can buy item `c` now: its race's shop stands and has what the item needs (`have`:
    techtree.counts_as, the player's finished buildings and research), or the goblin merchant sells it and has it in
    stock by `now`. (TechMask, for a vocabulary with these orders.)"""
    from . import techtree as tt
    from .neutral import MERCHANT_ITEMS
    shop = RACE_SHOPS[race]
    if shop in have and c in STOCK[shop]:
        need = ITEM_NEEDS.get(c)
        if need is None or tt.resolve(need, race) in have:
            return True
    return c in MERCHANT_ITEMS and stock.get(c, 0.0) <= now


def pick_shop(sel: np.ndarray, item: int, i: int, k: int, n_own: int, race: str, now: float):
    """The row of the shop hero row `i` buys `item` at: the one pointed at (row `k`) if it can, else the nearest that
    can: the player's own finished shop of its race (with what the item needs), or the goblin merchant if it sells it
    and has it in stock by `now` (no one else's: an enemy's tomb of relics is in `sel` too) -> a row, or None."""
    from . import techtree as tt
    from .neutral import MERCHANT_ITEMS, stock_times
    c = fx.rawcode(item)
    own = sel[:n_own]
    done = own[(own[:, fx.C_FLAGS] & fx.STRUCTURE > 0) & (own[:, fx.C_FLAGS] & (fx.CONSTRUCTING | fx.DEAD) == 0)]
    have = tt.counts_as({fx.rawcode(int(t)) for t in done[:, fx.C_TYPE].tolist()}, race)
    shop = RACE_SHOPS[race]
    rows = []
    if c in STOCK[shop] and (ITEM_NEEDS.get(c) is None or tt.resolve(ITEM_NEEDS[c], race) in have):
        rows.append(done[done[:, fx.C_TYPE] == code(shop)])
    if c in MERCHANT_ITEMS and stock_times().get(c, 0.0) <= now:
        rows.append(sel[sel[:, fx.C_TYPE] == code(MERCHANT)])
    can = np.concatenate(rows) if rows else sel[:0]
    if not len(can):
        return None
    if 0 <= k < len(sel) and (can[:, fx.C_ID] == sel[k, fx.C_ID]).any():
        return sel[k]
    d = (can[:, fx.C_X] - sel[i, fx.C_X]) ** 2 + (can[:, fx.C_Y] - sel[i, fx.C_Y]) ** 2
    return can[int(d.argmin())]


def full(hero: np.ndarray) -> bool:
    """A hero with no free inventory slot: a sale to it would go through and the item land on the ground."""
    return len(hero) > fx.C_ITEM0 + 5 and bool((hero[fx.C_ITEM0:fx.C_ITEM0 + 6] != 0).all())


def use_ok(item: int, kind: int, hero: np.ndarray) -> bool:
    """Whether a hero holding `item` in a slot may use it by that order kind: the forms it takes (USES; one this module
    does not list: as features.item_use_now without the flag, immediate if it is usable and not TARGETED_ITEMS), and an
    immediate heal not at full hit points, a mana potion not at full mana (taken, and wasted)."""
    if not item:
        return False
    kinds = _USE.get(item)
    if kinds is None and fx.FULL_VOCAB:  # (--full-vocab: the mechanical critter's and the tiny great hall's forms too)
        from .fullvocab import EXTRA_USES
        kinds = {code(c): k for c, k in EXTRA_USES.items()}.get(item)
    if kinds is None:
        return (kind == NOW and item not in fx.TARGETED_ITEMS
                and bool(fx._mod("herodata").item_table().get(item, (0, 0, 0, 0))[0]))
    if kind not in kinds:
        return False
    if kind == NOW:
        if item in _HEAL and hero[fx.C_HP] >= hero[fx.C_MAXHP]:
            return False
        if item in _MANA and hero[fx.C_MANA] >= hero[fx.C_MAXMANA]:
            return False
    return True


def use_target(sel: np.ndarray, item: int, kind: int, i: int, k: int, xy: tuple[float, float], n_own: int):
    """How hero row `i` uses `item` by an order of `kind` (one use_ok allows) with pointer `k` and point `xy` -> ("now",),
    ("unit", id), ("point", x, y), or None when nothing takes it (no other unit of the player's for a staff that moves one
    home)."""
    if kind == NOW:
        return ("now",)
    own = sel[:n_own]
    pointed = sel[k] if 0 <= k < n_own else None  # (a unit target's: a unit of the player's)
    if item == _PORTAL:  # (a point near a structure of the player's; none near: its nearest one)
        b = own[(own[:, fx.C_FLAGS] & fx.STRUCTURE > 0) & (own[:, fx.C_FLAGS] & fx.DEAD == 0)]
        if not len(b):
            return None
        d = np.hypot(b[:, fx.C_X] - xy[0], b[:, fx.C_Y] - xy[1])
        if d.min() > PORTAL_NEAR:
            xy = (float(b[int(d.argmin()), fx.C_X]), float(b[int(d.argmin()), fx.C_Y]))
        return ("point", *xy)
    if kind == POINT:
        return ("point", *xy)
    if item == _STAFF:  # (a building of the player's: the one pointed at, else its nearest)
        b = own[(own[:, fx.C_FLAGS] & fx.STRUCTURE > 0) & (own[:, fx.C_FLAGS] & fx.DEAD == 0)]
        if pointed is not None and int(pointed[fx.C_FLAGS]) & fx.STRUCTURE and not int(pointed[fx.C_FLAGS]) & fx.DEAD:
            return ("unit", int(pointed[fx.C_ID]))
        if not len(b):
            return None
        return ("unit", int(b[int(np.hypot(b[:, fx.C_X] - sel[i, fx.C_X], b[:, fx.C_Y] - sel[i, fx.C_Y]).argmin()), fx.C_ID]))
    mine = (own[:, fx.C_FLAGS] & (fx.STRUCTURE | fx.DEAD) == 0)  # (a unit of the player's: on any other the game refuses)
    if item == code("hslv"):  # (a salve: on the unit pointed at, else on itself)
        if pointed is not None and mine[k]:
            return ("unit", int(pointed[fx.C_ID]))
        return ("unit", int(sel[i, fx.C_ID]))
    mine[i] = False  # (a staff of sanctuary or preservation: refused on the hero itself)
    if pointed is not None and mine[k]:
        return ("unit", int(pointed[fx.C_ID]))
    if not mine.any():
        return None
    c = own[mine]
    return ("unit", int(c[int(np.hypot(c[:, fx.C_X] - sel[i, fx.C_X], c[:, fx.C_Y] - sel[i, fx.C_Y]).argmin()), fx.C_ID]))


@lru_cache(1)
def costs_of() -> dict[str, tuple[int, int]]:
    """Item -> gold and lumber (Units\\ItemData.slk: a probe's and a test's look-up)."""
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        rows = parse_slk(g.read("Units\\ItemData.slk").decode("latin-1"))
    return {r.get("itemID"): (int(r.get("goldcost") or 0), int(r.get("lumbercost") or 0)) for r in rows}
