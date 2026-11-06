"""--obs-full 1: what a person playing sees that the observation lacked (docs/reference/observation-coverage.md), as a
block of entity channels at the end of the features (features.A_OBS) and three entity kinds after the units.

From the harness (W3S_CFG_OBS_FULL) and the shim (shim/units.c unit_extras), as events (protocol.Event; kinds 101-109):
  101  a unit a seer sees: its classifications (stunned, snared, poisoned, polymorphed, sleeping, ethereal, magic immune,
       plagued), its positive and negative magic buffs, its buffs' count, whether its buff list was read; its move speed
       and armor now. Written when they changed (the shim keeps their hash per unit): this module keeps the last ones.
  102  a buff on it (the code the ability's BuffID spells: Unholy Frenzy's BUhf), one event each, after its 101
  104  a spell's point (the caster, x, y), then 109: the ability and the seers that see the point
  105  a hero's items' charges (10 bits a slot)
  106  a corpse (a dead unit, not a hero, building, summon or image): its place at death; 107 gone; 108 the seers that
       see its place
  and from what the observation had already: the research every player finishes (events 9: the enemy's levels as a
  visible unit's tooltip shows them), production and construction starts (5, 8, 11, 2: their progress against the
  tables' build times), the items lying on the ground (the G records, play.BCAgent's View.ground).

The block (OBS_F channels, offsets K_*):
  K_KIND    3   the entity kinds: a corpse, an item on the ground, an area spell in progress. The model gives each kind an
                attention bias (FullGameNet "kind_bias", -30 at first: a network from before plays as it did) and keeps
                them out of the target pointer (no order of the vocabulary takes them: the pick-up order is a point's)
  K_CLASS   8   the classifications (event 101)
  K_MAGIC   2   positive, negative magic buffs / 3
  K_ARMOR       armor now / 10
  K_SPEED       move speed now / 400
  K_SPEED_R     move speed / its type's (1: as its type; 0 for a building)
  K_HAVE        its state is known (an event 101 came for it)
  K_BUFF    BUFF_D  its buffs' codes, summed (a fixed random code per buff id: code()); an area spell's ability's code
  K_UPG     2   its owner's weapon and armor upgrade levels for its type / 3 (own units: the player's levels; an
                enemy's: the levels its finished research reached, what the unit's tooltip shows)
  K_ITEM    ITEM_D  a hero's items' codes, summed; an item on the ground: its code
  K_CHARGE  6   a hero's items' charges / 5, per slot
  K_PROD    3   own building: the current item's progress (0-1) and its seconds left / 60; any building in sight under
                construction: its progress (0-1)
  K_AGE         a corpse: seconds since its death / 90; an area: seconds left / 30; an item: a powerup (1)
  K_RADIUS      an area spell: its radius / 1000
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np

from . import features as fx

EV_STATE, EV_BUFF, EV_SPELL_AT, EV_CHARGES, EV_CORPSE, EV_CORPSE_GONE, EV_CORPSE_SEEN, EV_SPELL_SEEN = (
    101, 102, 104, 105, 106, 107, 108, 109)
KINDS = ("corpse", "item", "area")
BUFF_D, ITEM_D = 24, 16
K_KIND = 0
K_CLASS = K_KIND + len(KINDS)
K_MAGIC = K_CLASS + 8
K_ARMOR = K_MAGIC + 2
K_SPEED = K_ARMOR + 1
K_SPEED_R = K_SPEED + 1
K_HAVE = K_SPEED_R + 1
K_BUFF = K_HAVE + 1
K_UPG = K_BUFF + BUFF_D
K_ITEM = K_UPG + 2
K_CHARGE = K_ITEM + ITEM_D
K_PROD = K_CHARGE + 6
K_AGE = K_PROD + 3
K_RADIUS = K_AGE + 1
OBS_F = K_RADIUS + 1  # 70
MAX_CORPSES, MAX_ITEMS, MAX_AREAS = 8, 6, 4  # (a fight leaves ~15-20 corpses in sight: the 8 nearest own units)
CORPSE_SECONDS = 95.0  # (a corpse never reported gone: forgotten after this; they decay after ~92 s, scripts/obs_probe.py)
CLASSES = ("stunned", "snared", "poisoned", "polymorphed", "sleeping", "ethereal", "magic immune", "plagued")
# the area spells in progress a person sees on the ground (their AbilityData: Area1, Dur1 / HeroDur1, or waves DataA1
# x Cast1); the channeled ones end with their caster's death
AREA_SPELLS = ("AHbz", "AHfs", "AUdd", "AOeq", "ANrf", "ANvc", "AEsf", "AEtq", "ANst", "AUls", "Aclf", "ANmo", "ACbz",
               "ACrf")
CHANNELED = ("AHbz", "AUdd", "AOeq", "ANrf", "ANvc", "AEsf", "AEtq", "ANst", "ANmo", "ACbz", "ACrf")
TIME_PRIOR = 1.0  # (the map's build times over the tables' until a finished item tells: State.scale)


def channel_names() -> list[str]:
    """The block's channels for people (features.describe_spaces: run.json's "spaces")."""
    out = [f"kind: {k}" for k in ("a corpse", "an item on the ground", "an area spell in progress")]
    out += [f"state: {c}" for c in CLASSES] + ["positive magic buffs / 3", "negative magic buffs / 3", "armor now / 10",
                                               "move speed now / 400", "move speed / its type's", "its state is known"]
    out += [f"buffs' code {k}" for k in range(BUFF_D)] + ["weapon upgrade level / 3", "armor upgrade level / 3"]
    out += [f"items' code {k}" for k in range(ITEM_D)] + [f"item {k + 1}'s charges / 5" for k in range(6)]
    out += ["own building: progress of what it makes", "own building: its seconds left / 60", "construction progress",
            "corpse: seconds since its death / 90; area: seconds left / 30; item: a powerup", "area: radius / 1000"]
    assert len(out) == OBS_F
    return out


def code(v: int, d: int) -> np.ndarray:
    """A fixed code for an id (a buff's, an item's, an ability's): d signs / sqrt(d), the same in every process."""
    rng = np.random.default_rng((int(v) & 0xffffffff) * 2654435761 % (2 ** 32) + d)
    return (rng.integers(0, 2, d) * 2 - 1).astype(np.float32) / math.sqrt(d)


_CODES: dict = {}


def code_of(v: int, d: int) -> np.ndarray:
    k = (int(v), d)
    c = _CODES.get(k)
    if c is None:
        c = _CODES[k] = code(v, d)
    return c


def fourcc(c: str) -> int:
    return int.from_bytes(c.encode("latin-1"), "big")


@lru_cache(1)
def tables() -> dict:
    """The game's tables this module reads (a cache file beside the others): "speed" {type: move speed}, "upgrades"
    {type: ([weapon upgrades], [armor upgrades])}, "areas" {ability: (radius, seconds)}, "build" {type or research:
    seconds}, "timemod" {research: seconds more a level}."""
    from .. import paths
    path = paths.CACHE_DIR / "obs_full-1.json"
    if path.exists():
        return json.loads(path.read_text())
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        ub = parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))
        ud = parse_slk(g.read("Units\\UpgradeData.slk").decode("latin-1"))
        ad = {r.get("alias"): r for r in parse_slk(g.read("Units\\AbilityData.slk").decode("latin-1"))}

    def num(v, default=0.0) -> float:
        try:
            return float(str(v).strip())
        except (TypeError, ValueError):
            return default
    kind = {}
    for r in ud:
        effects = {str(r.get(f"effect{k}") or "").strip() for k in range(1, 5)}
        kind[r.get("upgradeid")] = "weapon" if "ratd" in effects or "ratx" in effects else "armor" if "rarm" in effects else ""
    speed, ups, build = {}, {}, {}
    for r in ub:
        c = r.get("unitBalanceID")
        if not c or len(c) != 4:
            continue
        speed[c] = num(r.get("spd"))
        build[c] = num(r.get("bldtm"))
        names = [u.strip() for u in str(r.get("upgrades") or "").split(",") if u.strip() and u.strip() != "_"]
        w, a = [u for u in names if kind.get(u) == "weapon"], [u for u in names if kind.get(u) == "armor"]
        if w or a:
            ups[c] = (w, a)
    timemod = {}
    for r in ud:
        build[r.get("upgradeid")] = num(r.get("timebase"))
        timemod[r.get("upgradeid")] = num(r.get("timemod"))
    areas = {}
    for a in AREA_SPELLS:
        r = ad.get(a) or {}
        secs = max(num(r.get("Dur1")), num(r.get("HeroDur1")))
        if secs <= 0:
            secs = num(r.get("DataA1"), 1.0) * max(num(r.get("Cast1"), 1.0), 1.0)
        areas[a] = (num(r.get("Area1"), 200.0), secs)
    out = {"speed": speed, "upgrades": ups, "areas": areas, "build": build, "timemod": timemod, "creeps": map_creeps()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, sort_keys=True))
    return out


CRITTERS = ("npig", "nrac", "nshe", "nfro", "ndog", "ncrb", "nder", "nfbr", "nhmc", "nskk", "nsno", "nalb", "nech",
            "npng", "nrat", "ndwm", "nvul", "nsea")


def map_creeps() -> list[str]:
    """The creep types the maps the runs play place (fullvocab.MAPS' scripts: CreateUnit of a neutral unit that is no
    building and no critter)."""
    import re
    from ..data.duelmap import parse_echo_name
    from ..data.mapbuild import stock_map_path
    from ..data.mpq import MpqArchive
    from . import fullvocab as fv
    U = fv.tables()["units"]
    out = set()
    for name in fv.MAPS:
        parsed = parse_echo_name(name)
        try:
            text = MpqArchive(stock_map_path(parsed[0] if parsed else name)).read("war3map.j").decode("latin-1")
        except Exception:  # noqa: BLE001  (a map not here: its creeps not added)
            continue
        for c in re.findall(r"CreateUnit\(\s*\w+\s*,\s*'(\w{4})'", text):
            u = U.get(c)
            if c[0] == "n" and u is not None and not u["building"] and c not in CRITTERS:
                out.add(c)
    return sorted(out)


def extended(vocab: dict) -> bool:
    return bool(vocab.get("obs_full"))


def extend(vocab: dict) -> dict:
    """`vocab` with the maps' creep types it lacks at its types' end ("obs_full": 1; "obs_full_from": the old count): a
    creep it did not know read as unknown (index 0); 7 of Echo Isles' 15 were (docs/reference/observation-coverage.md)."""
    if extended(vocab):
        return vocab
    types = list(vocab["types"])
    have = {fx.rawcode(int(t)) for t in types}
    for c in tables()["creeps"]:
        if c not in have:
            types.append(fourcc(c))
    return {**vocab, "types": types, "obs_full": 1, "obs_full_from": {"types": len(vocab["types"])}}


def grow(net, old: dict, new: dict):
    """`net` (of vocabulary `old`) grown to `new` (extend's): each new type's embedding, allowed and valid rows are
    unknown's (row 0), so every entity's token is what it was when its type read as unknown."""
    import torch
    from .model import FullGameNet
    T0, T1 = len(old["types"]) + 1, len(new["types"]) + 1
    if T0 == T1:
        return net
    if list(new["types"][:T0 - 1]) != list(old["types"]):
        raise ValueError("obsfull.grow: the new vocabulary must list the old one's types first, in their order")
    out = FullGameNet(**{**net.config, "n_types": T1}).to(net.order_kind.device)
    state = dict(net.state_dict())
    for k in ("type_emb.weight", "allowed", "valid_need"):
        v = state.get(k)
        if v is not None:
            state[k] = torch.cat([v, v[:1].expand(T1 - T0, *v.shape[1:]).clone()])
    out.load_state_dict(state, strict=True)
    return out.train(net.training)


class State:
    """One player's view of the obs-full events (features.View.obs), kept across steps."""

    def __init__(self):
        self.extras: dict[int, list] = {}  # unit -> [bits, speed/armor word, [buffs], charges (6,) or None]
        self.corpses: dict[int, list] = {}  # unit -> [type, owner, x, y, step of death, seers' bits]
        self.areas: list[list] = []  # [ability, x, y, radius, step it ends, seers' bits, caster, owner]
        self.enemy_levels: dict[int, int] = {}  # research -> level (the other players' finished research)
        self.since: dict[int, tuple] = {}  # building -> (step it started, seconds the item takes in the tables)
        self.built: dict[int, tuple] = {}  # structure under construction -> (step, its build seconds)
        self.ratios: list[float] = []  # (finished items: their seconds over the tables', the map's pace)
        self.scale = TIME_PRIOR
        self._spell: dict[int, tuple] = {}  # caster -> its last spell point (104), until its 109
        self._rows: dict[int, np.ndarray] = {}  # (row(): each unit's block, made again after an event about it)
        self._dirty: set[int] = set()

    def seconds(self, c: int, levels: dict | None = None) -> float:
        t = tables()
        k = fx.rawcode(int(c))
        s = float(t["build"].get(k, 0.0))
        if k[:1] == "R" and levels is not None:
            s += float(t["timemod"].get(k, 0.0)) * max(int(levels.get(int(c), 0)), 0)
        return s

    def events(self, events, rows: np.ndarray, t: int, player: int, own: set, levels: dict, step_seconds: float) -> None:
        """This step's events (step, kind, a, b, c)."""
        deaths: dict[int, int] = {}
        for r in events:
            kind, a, b, c = int(r[1]), int(r[2]), int(r[3]), int(r[4])
            if kind in (EV_STATE, EV_BUFF, EV_CHARGES):
                self._dirty.add(a)
            if kind == EV_STATE:
                self.extras[a] = [b, c, [], None]
            elif kind == EV_BUFF:
                e = self.extras.get(a)
                if e is not None:
                    e[2].append(b)
            elif kind == EV_CHARGES:
                e = self.extras.get(a)
                if e is not None:
                    e[3] = [((b if i < 3 else c) >> (10 * (i % 3))) & 1023 for i in range(6)]
            elif kind == 1:  # (a death: a corpse's type; an area's caster)
                deaths[a] = c
                self.areas = [ar for ar in self.areas if not (ar[6] == a and fx.rawcode(ar[0]) in CHANNELED)]
                self.since.pop(a, None)
                self.built.pop(a, None)
            elif kind == EV_CORPSE:
                owner = -1
                hit = rows[rows[:, fx.C_ID] == a] if len(rows) else rows
                typ = deaths.get(a, int(hit[0, fx.C_TYPE]) if len(hit) else 0)
                if len(hit):
                    owner = int(hit[0, fx.C_OWNER])
                self.corpses[a] = [typ, owner, b, c, t, 0]
            elif kind == EV_CORPSE_GONE:
                self.corpses.pop(a, None)
            elif kind == EV_CORPSE_SEEN:
                if a in self.corpses:
                    self.corpses[a][5] = b
            elif kind == EV_SPELL_AT:
                self._spell[a] = (b, c)
            elif kind == EV_SPELL_SEEN:
                at = self._spell.pop(a, None)
                ab = fx.rawcode(b)
                area = tables()["areas"].get(ab)
                if at is not None and area is not None and (at[0] or at[1]):
                    hit = rows[rows[:, fx.C_ID] == a] if len(rows) else rows
                    owner = int(hit[0, fx.C_OWNER]) if len(hit) else -1
                    self.areas.append([b, at[0], at[1], area[0], t + area[1] / step_seconds, c, a, owner])
            elif kind == fx.RESEARCH_FINISH and a not in own:
                self.enemy_levels[b] = max(self.enemy_levels.get(b, 0), c)
            elif kind in fx.PRODUCTION_START:  # (train 5, research 8, a tier 11: its start, and the item's time)
                if kind == 11:
                    nxt = fx_next_tier(b)
                    s = self.seconds(fourcc(nxt)) if nxt else 0.0
                else:
                    s = self.seconds(b, levels if kind == 8 else None)
                self.since[a] = (t, s)
            elif kind in fx.PRODUCTION_END:
                got = self.since.pop(a, None)
                if got is not None and kind in (6, 9, 12) and got[1] > 0:
                    self._pace((t - got[0]) * step_seconds / got[1])
            elif kind == 2:  # construction started
                self.built[a] = (t, self.seconds(b))
            elif kind in (3, 4):
                got = self.built.pop(a, None)
                if got is not None and kind == 3 and got[1] > 0:
                    self._pace((t - got[0]) * step_seconds / got[1])
        for k in [k for k, v in self.corpses.items() if (t - v[4]) * step_seconds > CORPSE_SECONDS]:
            del self.corpses[k]
        if self.corpses and len(rows):  # (a corpse's handle id a living unit's after its death's step: the corpse is
            # gone, decayed or used, and the game gave its id to a new unit, or it was raised again; the harness's 107
            # had never come for a removed corpse: a real game kept a ghoul's corpse 5 s after a farm took its id, 466 of
            # 707 steps had such a pair, and the views' index took the unit's id to the corpse's row: AI labels on it)
            alive = rows[(rows[:, fx.C_FLAGS] & fx.DEAD) == 0, fx.C_ID]
            ids = np.fromiter(self.corpses, np.int64, len(self.corpses))
            for k in ids[np.isin(ids, alive)].tolist():
                if self.corpses[k][4] < t:
                    del self.corpses[k]
        self.areas = [ar for ar in self.areas if ar[4] > t]

    def row(self, uid: int) -> np.ndarray | None:
        """A unit's block from K_CLASS to K_PROD (its extras), made again only after an event about it."""
        r = self._rows.get(uid)
        if r is not None and uid not in self._dirty:
            return r
        e = self.extras.get(uid)
        if e is None:
            return None
        self._dirty.discard(uid)
        bits, word, buffs, charges = e
        r = np.zeros(K_PROD - K_CLASS, np.float32)
        for c in range(8):
            if bits >> c & 1:
                r[c] = 1.0
        r[K_MAGIC - K_CLASS] = (bits >> 8 & 15) / 3.0
        r[K_MAGIC - K_CLASS + 1] = (bits >> 12 & 15) / 3.0
        r[K_ARMOR - K_CLASS] = ((word & 4095) - 2048) / 100.0
        r[K_SPEED - K_CLASS] = (word >> 12) / 400.0
        r[K_HAVE - K_CLASS] = 1.0
        for bf in buffs:
            r[K_BUFF - K_CLASS:K_BUFF - K_CLASS + BUFF_D] += code_of(bf, BUFF_D)
        if charges is not None:
            r[K_CHARGE - K_CLASS:K_CHARGE - K_CLASS + 6] = np.minimum(np.asarray(charges, np.float32) / 5.0, 2.0)
        self._rows[uid] = r
        return r

    def _pace(self, r: float) -> None:
        if 0.05 < r < 3.0:  # (a queue's or a paused item's: off)
            self.ratios = (self.ratios + [r])[-15:]
            self.scale = float(np.median(self.ratios))


@lru_cache(None)
def fx_next_tier(cur: int) -> str | None:
    """A building's next tier (its Upgrade in the *UnitFunc.txt files), for an upgrade's time."""
    from .fullvocab import tables as fvt
    up = fvt()["funcs"].get(fx.rawcode(int(cur)), {}).get("Upgrade") or []
    return up[0] if up else None


_SPEEDS: list = []


def _lookup_speed(types: np.ndarray) -> np.ndarray:
    """Each unit type's move speed in the tables (0: none, a building's)."""
    if not _SPEEDS:
        sp = tables()["speed"]
        keys = np.array(sorted(fourcc(c) for c in sp), np.int64)
        _SPEEDS.extend([keys, np.array([sp[fx.rawcode(int(k))] for k in keys], np.float32)])
    keys, vals = _SPEEDS
    j = np.minimum(np.searchsorted(keys, types), len(keys) - 1)
    return np.where(keys[j] == types, vals[j], 0.0).astype(np.float32)


@lru_cache(None)
def _type_upgrades(typ: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    w, a = tables()["upgrades"].get(fx.rawcode(int(typ)), ([], []))
    return tuple(fourcc(u) for u in w), tuple(fourcc(u) for u in a)


def apply(view, rows: np.ndarray, events, t: int, sel: np.ndarray, n_own: int, f: np.ndarray, types: np.ndarray,
          cur: np.ndarray) -> tuple:
    """The block for this step's entities, and the new kinds' rows after them -> (sel, f, types, cur)."""
    st: State = view.__dict__.get("obs")
    if st is None:
        st = view.obs = State()
    step_s = float(getattr(view, "step_seconds", 0.5))
    st.events(events, rows, t, view.player, view.own_buildings, view.levels, step_s)
    A = fx.A_OBS
    tb = tables()
    n = len(sel)
    if n:  # (Python's lists: the arrays are ~50 rows, numpy's calls on them had cost more than the loops)
        idl, tyl, owl = sel[:, fx.C_ID].tolist(), sel[:, fx.C_TYPE].tolist(), sel[:, fx.C_OWNER].tolist()
        if st.extras:  # (the units' extras: each unit's block row, kept until an event changes it)
            hit, got, spd = [], [], []
            sp = tb["speed"]
            for k, uid in enumerate(idl):
                r = st.row(uid)
                if r is not None:
                    hit.append(k)
                    got.append(r)
                    base = sp.get(fx.rawcode(tyl[k]), 0.0)
                    spd.append(float(r[K_SPEED - K_CLASS]) * 400.0 / base if base > 0 else 0.0)
            if hit:
                f[hit, A + K_CLASS:A + K_PROD] = got
                f[hit, A + K_SPEED_R] = spd
        own = view.player
        memo: dict = {}
        ups, upk = [], []
        for k in range(n):  # (the upgrade levels: per type and owner of the players' units in view)
            o = owl[k]
            if o != own and o != 1 - own:
                continue
            key = (tyl[k], o)
            v = memo.get(key)
            if v is None:
                w, a = _type_upgrades(tyl[k])
                lv = view.levels if o == own else st.enemy_levels
                v = memo[key] = ((max(lv.get(c, 0) for c in w) / 3.0 if w else 0.0),
                                 (max(lv.get(c, 0) for c in a) / 3.0 if a else 0.0))
            if v[0] or v[1]:
                upk.append(k)
                ups.append(v)
        if upk:
            f[upk, A + K_UPG:A + K_UPG + 2] = ups
        if sel.shape[1] >= fx.ROW_COLS:
            fl = sel[:, fx.C_FLAGS]
            for k in np.flatnonzero(fl & fx.HERO).tolist():
                for it in sel[k, fx.C_ITEM0:fx.C_ITEM0 + 6].tolist():
                    if it:
                        f[k, A + K_ITEM:A + K_ITEM + ITEM_D] += code_of(it, ITEM_D)
        if st.since or st.built:
            where = dict(zip(idl, range(n)))
            for d, col, mine in ((st.since, K_PROD, True), (st.built, K_PROD + 2, False)):
                for uid, (t0, secs) in d.items():
                    k = where.get(uid)
                    if k is None or secs <= 0 or (mine and k >= n_own):
                        continue
                    if not mine and not sel[k, fx.C_FLAGS] & fx.CONSTRUCTING:
                        continue
                    total, done = secs * st.scale, (t - t0) * step_s
                    f[k, A + col] = min(done / total, 1.0)
                    if mine:
                        f[k, A + col + 1] = max(total - done, 0.0) / 60.0
    # the new kinds: corpses, items on the ground and area spells the player sees, after the entities in view
    room = fx.MAX_ENT - n
    if room <= 0:
        return sel, f, types, cur
    p = view.player
    new: list[tuple] = []  # (kind, id, type, owner, x, y, the block's values: {offset: value or array})
    cs = [(cid, v) for cid, v in st.corpses.items() if v[5] >> p & 1]
    if len(cs) > MAX_CORPSES and n_own:  # (the nearest to the player's own units: what its casters can reach)
        mine = sel[:n_own][:, [fx.C_X, fx.C_Y]].astype(np.float64)
        xy = np.array([[v[2], v[3]] for _, v in cs], np.float64)
        d = ((xy[:, None, :] - mine[None, :, :]) ** 2).sum(-1).min(1)
        cs = [cs[i] for i in np.argsort(d, kind="stable")[:MAX_CORPSES].tolist()]
    for ci, (cid, (typ, owner, x, y, t0, _)) in enumerate(cs[:MAX_CORPSES]):  # (an id of its own, as the items' and
        # areas': its unit's handle id goes to a new unit once the corpse is gone, and two rows with one id had the
        # views' index take the unit's id to the corpse's row, which the target pointer masks)
        new.append((0, -(3000 + ci), typ, owner, x, y, {K_AGE: min((t - t0) * step_s / 90.0, 2.0)}))
    gi = 0
    for g in getattr(view, "ground", ()) or ():
        if gi >= MAX_ITEMS:
            break
        it, x, y, vis, up = (int(v) for v in g[:5])
        if not vis >> p & 1:
            continue
        new.append((1, -(1000 + gi), 0, -1, x, y, {K_ITEM: code_of(it, ITEM_D), K_AGE: float(up)}))
        gi += 1
    ai = 0
    for ab, x, y, radius, t_end, vis, caster, owner in st.areas:
        if ai >= MAX_AREAS:
            break
        if not vis >> p & 1:
            continue
        new.append((2, -(2000 + ai), 0, owner, x, y, {K_BUFF: code_of(ab, BUFF_D), K_AGE: min((t_end - t) * step_s / 30.0, 2.0),
                                                      K_RADIUS: radius / 1000.0}))
        ai += 1
    new = new[:room]
    if not new:
        return sel, f, types, cur
    m = len(new)
    s2 = np.zeros((m, sel.shape[1]), sel.dtype)
    f2 = np.zeros((m, f.shape[1]), np.float32)
    t2 = np.zeros(m, types.dtype)
    xs = np.zeros(m)
    ys = np.zeros(m)
    for k, (kind, nid, typ, owner, x, y, vals) in enumerate(new):
        s2[k, fx.C_ID], s2[k, fx.C_TYPE], s2[k, fx.C_OWNER], s2[k, fx.C_X], s2[k, fx.C_Y] = nid, typ, owner, x, y
        f2[k, 0 if owner == p else 1 if owner == 1 - p else 2] = 1.0
        f2[k, A + K_KIND + kind] = 1.0
        for off, v in vals.items():
            if isinstance(v, np.ndarray):
                f2[k, A + off:A + off + len(v)] = v
            else:
                f2[k, A + off] = v
        xs[k], ys[k] = x, y
    f2[:, 3], f2[:, 4] = fx.frame_xy(xs, ys, view.sign)
    t2[:] = fx._lookup(s2[:, fx.C_TYPE], view.enc._type_keys, view.enc._type_vals)  # (a corpse: its unit's type)
    for i, kk in enumerate(fx.POS_FREQS if getattr(view, "active_f", fx.F) > fx.A_POS else ()):  # (as the units'
        px, py = math.pi * kk * f2[:, 3], math.pi * kk * f2[:, 4]
        f2[:, fx.A_POS + 4 * i:fx.A_POS + 4 * i + 4] = np.stack([np.sin(px), np.cos(px), np.sin(py), np.cos(py)], 1)
    return (np.concatenate([sel, s2]), np.concatenate([f, f2]), np.concatenate([types, t2]),
            np.concatenate([cur, np.zeros(m, cur.dtype)]))
