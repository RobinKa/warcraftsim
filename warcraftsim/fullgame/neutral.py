"""The neutral buildings' wares as the policy's orders (the user's question, 2026-10-20: what can it not do yet):
a tavern's heroes, a mercenary camp's mercenaries and the goblin merchant's items, each an order of a hero or a
worker on the building (a unit target: protocol.Buy), and the tavern heroes' skills and spells. selfplay --neutral 1:
every network the run loads gets the extended vocabulary (extend_vocab) and grows to it (grow, in model.load): the
new unit types start as their kind's mean embedding, the new orders rare (their logits' bias low) until the reward
or a teacher raises them.

Echo Isles (checked in the game, scratch probes of 2026-10-20): the tavern at the middle sells its heroes after
135 s (each needs an altar, then each tier, as a race's own), a mercenary camp at each side after 220 s (guarded
by creeps), the goblin merchant south of the middle its items (a town portal after 440 s)."""

from __future__ import annotations

import torch

TAVERN, MERCHANT, CAMP, MARKET = "ntav", "ngme", "nmer", "nmrk"
SHOPS = (TAVERN, MERCHANT, CAMP, MARKET)
TAVERN_HEROES = ("Nalc", "Ntin", "Nfir", "Nngs", "Nbrn", "Npbm", "Nplh", "Nbst")
MERCENARIES = ("nftb", "nfsp", "nogm", "ngrk")
MERCHANT_ITEMS = ("stwp", "bspd", "dust", "tret", "prvt", "cnob", "stel", "pnvl", "shea", "spro", "pinv")
# the game's order ids of the tavern heroes' spells (a game's order table, 2026-10-20)
SPELL_IDS = {"acidbomb": 852662, "blackarrow": 852577, "blackarrowoff": 852579, "blackarrowon": 852578,
             "breathoffire": 852580, "charm": 852581, "chemicalrage": 852663, "clusterrockets": 852652,
             "coldarrowstarg": 852243, "doom": 852583, "drain": 852487, "drunkenhaze": 852585, "elementalfury": 852586,
             "forkedlightning": 852587, "healingspray": 852664, "howlofterror": 852588, "lavamonster": 852667,
             "manashieldoff": 852590, "manashieldon": 852589, "rainoffire": 852238, "robogoblin": 852656,
             "silence": 852592, "soulburn": 852668, "stampede": 852593, "summonfactory": 852658,
             "summongrizzly": 852594, "summonquillbeast": 852595, "summonwareagle": 852596, "tornado": 852597,
             "transmute": 852665, "unrobogoblin": 852657, "volcano": 852669}
# the orders any unit gives: move, attack, smart, stop, hold position, patrol; a hero's items too
GENERIC = (851986, 851983, 851971, 851972, 851993, 851990)
USESLOTS = tuple(range(852008, 852014))
WORKERS = ("hpea", "opeo", "uaco", "ewsp")
MELEE_HEROES = ("Hpal", "Hamg", "Hmkg", "Hblm", "Obla", "Ofar", "Otch", "Oshd",
                "Udea", "Ulic", "Udre", "Ucrl", "Ekee", "Emoo", "Edem", "Ewar")
_KIND = {"unit": 2, "point": 1, "instant": 0}  # (features.UNIT, POINT, IMMEDIATE)


def fourcc(code: str) -> int:
    return int.from_bytes(code.encode("latin-1"), "big")


def extended(vocab: dict) -> bool:
    return bool(vocab.get("neutral"))


def extend_vocab(vocab: dict) -> dict:
    """`vocab` with the neutral buildings, the tavern heroes and the mercenaries as unit types, and as orders: each
    ware bought at its building (a unit target), each tavern hero's skills (learning) and spells -> a new vocabulary
    (the old one's entries first, in their places)."""
    from ..data.abilities import ability_info, hero_ability_table
    from ..data.objects import ability_orders
    if extended(vocab):
        return vocab
    types = list(vocab["types"])
    for c in (*SHOPS, *TAVERN_HEROES, *MERCENARIES):
        if fourcc(c) not in types:
            types.append(fourcc(c))
    orders = [list(o) for o in vocab["orders"]]
    have = {tuple(o) for o in orders}
    names = dict(vocab.get("order_names") or {})

    def add(oid: int, kind: int) -> None:
        if (oid, kind) not in have:
            have.add((oid, kind))
            orders.append([oid, kind])
    for c in (*TAVERN_HEROES, *MERCENARIES, *MERCHANT_ITEMS):
        add(fourcc(c), 2)  # (bought at the building the pointer picks)
    info, func = ability_info(), ability_orders()
    for h in TAVERN_HEROES:
        for code in hero_ability_table()[h]:
            add(fourcc(code), 3)  # (learning it)
            a = info[code]
            for name in {a.order, *func.get(code, {}).values()} - {None}:
                oid = SPELL_IDS.get(name)
                if oid is None:
                    continue
                names.setdefault(str(oid), name)
                add(oid, _KIND.get(a.cast, 0) if name == a.order else 0)  # (a toggle: no target)
    counts = dict(vocab.get("counts") or {})
    if "orders" in counts:
        counts["orders"] = list(counts["orders"]) + [0] * (len(orders) - len(counts["orders"]))
    return {**vocab, "types": types, "orders": orders, "order_names": names, "counts": counts, "neutral": 1}


@torch.no_grad()
def grow(net, old: dict, new: dict):
    """`net` (of vocabulary `old`) grown to `new` (extend_vocab's): unit types' embeddings (a tavern hero's the
    melee heroes' mean, a mercenary's the army units', a neutral building's the neutral units'), the order head's
    new rows (zero weights, the bias at the lowest 5% of the old ones'), their embeddings (the train orders' mean),
    which unit types may give them (heroes and workers buy; a tavern hero its own skills and spells and what any
    unit gives) -> a new network."""
    from ..data.abilities import ability_info, hero_ability_table
    from ..data.objects import ability_orders
    from . import features as fx
    from .model import FullGameNet
    T0, T1, O0, O1 = len(old["types"]) + 1, len(new["types"]) + 1, len(old["orders"]) + 1, len(new["orders"]) + 1
    if (T0, O0) == (T1, O1):
        return net
    config = {**net.config, "n_types": T1, "n_orders": O1}
    valid = config.pop("valid", 0)
    out = FullGameNet(**config).to(net.order_kind.device)
    state = {k: v for k, v in net.state_dict().items() if not k.startswith("valid_")}
    tix = {t: i + 1 for i, t in enumerate(new["types"])}
    code = {i + 1: int(t).to_bytes(4, "big").decode("latin-1") for i, t in enumerate(new["types"])}

    def mean_of(codes) -> torch.Tensor:
        rows = [tix[fourcc(c)] for c in codes if fourcc(c) in tix and tix[fourcc(c)] < T0]
        return state["type_emb.weight"][rows].mean(0) if rows else state["type_emb.weight"].mean(0)
    army = [code[i] for i in range(1, T0) if code[i][:1].islower() and code[i] not in WORKERS and code[i][0] in "houe"]
    neutral = [code[i] for i in range(1, T0) if code[i][:1] == "n"]
    emb = torch.cat([state["type_emb.weight"], state["type_emb.weight"].new_zeros(T1 - T0, net.d)])
    for i in range(T0, T1):
        c = code[i]
        emb[i] = mean_of(MELEE_HEROES if c in TAVERN_HEROES else army if c in MERCENARIES else neutral)
    state["type_emb.weight"] = emb
    w, b = state["order.2.weight"], state["order.2.bias"]
    low = (torch.quantile(b.float(), 0.05) - 2.0).to(b.dtype)  # (rarer than any order the policy gives now)
    state["order.2.weight"] = torch.cat([w, w.new_zeros(O1 - O0, w.shape[1])])
    state["order.2.bias"] = torch.cat([b, low.expand(O1 - O0)])
    oe = state["order_emb.weight"]
    trains = [c for c, (o, k) in enumerate(old["orders"], start=1) if o >= fx.TYPE_CODE and k == fx.IMMEDIATE]
    state["order_emb.weight"] = torch.cat([oe, oe[trains].mean(0).expand(O1 - O0, -1).clone()])
    allowed = torch.zeros(T1, O1, dtype=torch.bool, device=oe.device)
    allowed[:T0, :O0] = state["allowed"]
    oix = {tuple(o): c for c, o in enumerate(new["orders"], start=1)}
    buys = [oix[(fourcc(c), 2)] for c in (*TAVERN_HEROES, *MERCENARIES, *MERCHANT_ITEMS)]
    generic = [c for (o, k), c in oix.items() if o in GENERIC + USESLOTS and k in (0, 1, 2)]
    for i in range(1, T1):
        c = code[i]
        if c[:1].isupper() and c[0] != "R" or c in WORKERS:  # (a hero or a worker: it buys)
            allowed[i, buys] = True
        if c in TAVERN_HEROES:
            allowed[i, generic] = True
        if c in MERCENARIES:
            allowed[i, [g for g in generic if new["orders"][g - 1][0] not in USESLOTS]] = True
    for h in TAVERN_HEROES:  # (its skills and spells)
        mine = {fourcc(a) for a in hero_ability_table()[h]}
        spells = {SPELL_IDS[n] for a in hero_ability_table()[h]
                  for n in (ability_info()[a].order, *ability_orders().get(a, {}).values()) if n in SPELL_IDS}
        for (o, k), c in oix.items():
            if (k == 3 and o in mine) or (k != 3 and o in spells):
                allowed[tix[fourcc(h)], c] = True
    state["allowed"] = allowed
    state["order_kind"] = torch.as_tensor([0] + [k for _, k in new["orders"]], device=oe.device)
    out.load_state_dict(state, strict=False)
    if valid:
        out.set_valid(new)
    return out.train(net.training)


def stock_times() -> dict[str, float]:
    """A ware's first game second at its building (UnitBalance's / ItemData's stockStart)."""
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        units = {r.get("unitBalanceID"): r for r in parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))}
        items = {r.get("itemID"): r for r in parse_slk(g.read("Units\\ItemData.slk").decode("latin-1"))}
    out = {}
    for c in (*TAVERN_HEROES, *MERCENARIES, *MERCHANT_ITEMS):
        r = units.get(c) or items.get(c) or {}
        try:
            out[c] = float(r.get("stockStart") or 0)
        except ValueError:
            out[c] = 0.0
    return out


def is_shop(type_id: int) -> bool:
    return int(type_id) in {fourcc(c) for c in SHOPS}


def nearest_shop(sel, ware: int, unit):
    """The row of the building in view nearest `unit` that sells `ware` (a tavern its heroes, a camp its
    mercenaries, the goblin merchant its items), or None."""
    from . import features as fx
    code = int(ware).to_bytes(4, "big").decode("latin-1")
    kind = TAVERN if code in TAVERN_HEROES else CAMP if code in MERCENARIES else MERCHANT
    shops = sel[sel[:, fx.C_TYPE] == fourcc(kind)]
    if not len(shops):
        return None
    d = (shops[:, fx.C_X] - unit[fx.C_X]) ** 2 + (shops[:, fx.C_Y] - unit[fx.C_Y]) ** 2
    return shops[int(d.argmin())]
