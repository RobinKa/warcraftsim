"""The melee heroes' spells the vocabulary lacks (selfplay --spells 1).

The vocabulary's orders are those the demonstrations gave at least 5 times (features.build_vocab): the built-in
AI's and the recorded games' orders. Five spells of the melee heroes were never among them (learning them was: the
skill orders are in), so the policy could learn War Stomp, Hex, Banish, Far Sight and Scout and never cast them:

    War Stomp   Tauren Chieftain   AOws  stomp     852127  no target
    Hex         Shadow Hunter      AOhx  hex       852502  a unit
    Banish      Blood Mage         AHbn  banish    852486  a unit
    Far Sight   Far Seer           AOfs  farsight  852122  a point
    Scout       Priestess of the   AEst  scout     852181  no target (the owl appears beside her)
                Moon

(Order ids: the game's order table, a recorded game's meta "order_names"; how each is cast: data.abilities._CAST.)

A vocabulary gets them once, at its end in this order (extend; "spells" marks it), and a network loaded with
--spells, or for a run whose vocabulary has them (model.load's `vocab`), grows to them (grow): their rows in the order
head the "no order" row's with its bias less NEW_BIAS (each new spell exp(-8) = 1/2981 as likely as no order for its
hero, in any state: a constant low bias, as neutral.grow's and macros.grow's, took up to 7% of a hero's orders in odd
states of fgself-153's network), their embeddings the mean of the old vocabulary's hero spells of the same target kind
and side (War Stomp: thunder clap's, ...); only the hero with the ability may give one (allow), and with --valid-mask
only while it is learned, cooled down and paid for in mana (herodata.order_needs, by the vocabulary's order names).
Every logit of the old orders is as before, and so are the other units' order distributions."""

from __future__ import annotations

import torch

from . import features as fx

# (ability, order string, order id, target kind) in the order a vocabulary gets them
SPELLS = (("AOws", "stomp", 852127, fx.IMMEDIATE), ("AOhx", "hex", 852502, fx.UNIT),
          ("AHbn", "banish", 852486, fx.UNIT), ("AOfs", "farsight", 852122, fx.POINT),
          ("AEst", "scout", 852181, fx.IMMEDIATE))
HEROES = {"AOws": "Otch", "AOhx": "Oshd", "AHbn": "Hblm", "AOfs": "Ofar", "AEst": "Emoo"}
NEW_BIAS = 3.0  # (a new spell's logit: no order's less this, before the order temperature; 8 had left them
# never sampled, so never learned: 9 casts in ~13,800 games; audit 15. --spells-bias resets a grown network's)


def code(c: str) -> int:
    return int.from_bytes(c.encode("latin-1"), "big")


def extended(vocab: dict) -> bool:
    have = {tuple(o) for o in vocab["orders"]}
    return all((oid, kind) in have for _, _, oid, kind in SPELLS)


def extend(vocab: dict) -> dict:
    """`vocab` with the spells at its end (those it lacks, in SPELLS' order) -> a new vocabulary."""
    if extended(vocab):
        return vocab if vocab.get("spells") else {**vocab, "spells": 1}
    out = [list(o) for o in vocab["orders"]]
    have = {tuple(o) for o in out}
    names = dict(vocab.get("order_names") or {})
    for _, name, oid, kind in SPELLS:
        names.setdefault(str(oid), name)
        if (oid, kind) not in have:
            out.append([oid, kind])
    counts = dict(vocab.get("counts") or {})
    if "orders" in counts:
        counts["orders"] = list(counts["orders"]) + [0] * (len(out) - len(counts["orders"]))
    return {**vocab, "orders": out, "order_names": names, "counts": counts, "spells": 1}


def _hero_spells(vocab: dict) -> dict:
    """(target kind, side) and target kind -> the order classes of the melee heroes' spells in `vocab` (their cast
    orders, not toggles): War Stomp's embedding is the other no-target enemy spells' mean (thunder clap, ...)."""
    from ..data.abilities import ability_info, hero_ability_table
    names = {v: int(k) for k, v in (vocab.get("order_names") or {}).items()}
    info = ability_info()
    side = {names[info[a].order]: info[a].side for codes in hero_ability_table().values() for a in codes
            if a in info and info[a].castable and info[a].order in names}
    out: dict = {}
    for c, (o, k) in enumerate(vocab["orders"], start=1):
        if o in side:
            out.setdefault((int(k), side[o]), []).append(c)
            out.setdefault(int(k), []).append(c)
    return out


@torch.no_grad()
def grow(net, old: dict, new: dict):
    """`net` (of vocabulary `old`) with the orders `new` adds after old's: their order-head rows the "no order" row's,
    its bias less NEW_BIAS, their embeddings the mean of old's hero spells of the same target kind and side (the target
    head then aims them as it aims those); the heroes allowed (allow)."""
    from .model import FullGameNet
    O0, O1 = len(old["orders"]) + 1, len(new["orders"]) + 1
    if O0 == O1:
        return net
    if [list(o) for o in new["orders"][:O0 - 1]] != [list(o) for o in old["orders"]]:
        raise ValueError("spells.grow: the new vocabulary must list the old one's orders first, in their order")
    config = {**net.config, "n_orders": O1}
    valid = config.pop("valid", 0)
    out = FullGameNet(**config).to(net.order_kind.device)
    state = {k: v for k, v in net.state_dict().items() if not k.startswith("valid_")}
    w, b = state["order.2.weight"], state["order.2.bias"]
    state["order.2.weight"] = torch.cat([w, w[:1].expand(O1 - O0, -1).clone()])
    state["order.2.bias"] = torch.cat([b, (b[:1] - NEW_BIAS).expand(O1 - O0).clone()])
    oe = state["order_emb.weight"]
    like, sides = _hero_spells(old), {oid: a for a, _, oid, _ in SPELLS}
    from ..data.abilities import ability_info
    rows = []
    for o, k in new["orders"][O0 - 1:]:
        side = ability_info()[sides[o]].side if o in sides else None
        same = (like.get((int(k), side)) or like.get(int(k))
                or [c for key, v in like.items() if isinstance(key, int) for c in v])
        rows.append(oe[same].mean(0) if same else oe[1:].mean(0))
    state["order_emb.weight"] = torch.cat([oe, torch.stack(rows)])
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
    """Who may give the spells (net's allowed table, in place): the hero with the ability, no other unit type."""
    oix = {tuple(o): c for c, o in enumerate(vocab["orders"], start=1)}
    tix = {int(t): i + 1 for i, t in enumerate(vocab["types"])}
    for abil, _, oid, kind in SPELLS:
        k, t = oix.get((oid, kind)), tix.get(code(HEROES[abil]))
        if k is None or k >= net.allowed.shape[1]:
            continue
        net.allowed[:, k] = False
        if t is not None and t < net.allowed.shape[0]:
            net.allowed[t, k] = True
