"""The per-step trace a whole-game player writes for its video's panel (fullgame.overlay); no
imaging here, so the players can import it without Pillow."""

from __future__ import annotations

from ..protocol import Observation

DEAD_FLAG = 1024  # the unit flags' dead bit (an int test, not an enum operation: per unit per step)
HERO_FLAG = 1  # (protocol.UnitFlags.HERO)
_FX = None  # (fullgame.features, once imported: _features())


def _features():
    """fullgame.features, imported on first use (an import statement in a function had cost ~5k instructions a call)."""
    global _FX
    if _FX is None:
        from . import features
        _FX = features
    return _FX


def heroes(obs: Observation) -> dict[str, list]:
    """Each player's heroes: [type, level, alive, [each ability slot's level (data.abilities order)]]."""
    from .features import rawcode
    out: dict[str, list] = {"0": [], "1": []}
    ua = getattr(obs, "unit_array", None)
    if ua is not None:  # (native observations: columns of native.py; the abilities' levels at 23, 25, 27, 29)
        for r in ua[(ua[:, 11] & HERO_FLAG) > 0].tolist():
            if r[2] in (0, 1):
                out[str(r[2])].append([rawcode(r[1]), r[14], int(not r[11] & DEAD_FLAG), r[23:31:2]])
        return out
    for u in obs.units:
        if u.is_hero and u.owner in (0, 1):
            out[str(u.owner)].append([u.type, u.hero_level, int(u.alive), [lv for lv, _ in u.abilities]])
    return out


def trace_step(t: int, obs: Observation, material: dict, orders: dict, **policy) -> dict:
    """One step's row: `material` {player: value}, `orders` {player: Counter of order labels},
    `policy`: value / reward / entropy, each {player: float} (the players a policy controls)."""
    res = {str(p): [s.gold, s.lumber, s.food_used, s.food_cap] for p, s in obs.players.items() if p in (0, 1)}
    row = {"t": t, "game_time": round(obs.game_time, 2), "material": {str(p): round(v) for p, v in material.items()},
           "res": res, "orders": {str(p): dict(c) for p, c in orders.items()}, "heroes": heroes(obs)}
    for k, v in policy.items():
        row[k] = {str(p): round(float(x), 4) for p, x in v.items()}
    return row


WORKER_IDS = {int.from_bytes(c.encode(), "big") for c in ("hpea", "opeo", "uaco", "ewsp")}  # (type ids)
_VALUES: dict[int, tuple] = {}  # id(values) -> (values, its size, the type ids seen (sorted), their values, and
# w3obs_material's description of them: an int64 array and its address)
_NATIVE = None  # (_native()'s, made on first use)
_UA: tuple = (None, 0)  # (the last unit table given to the C helpers and its address: a step's material, Production and
# unit_rows share it. One tuple, replaced whole: an actor's game threads share it)


def _native():
    """The C helpers for native observations (warcraftsim.native's library: material, Production's scan, play's
    unit_rows) as a namespace, or False without them (no compiler: the numpy code). Imported on first use: the
    players import this module without them."""
    global _NATIVE
    if _NATIVE is None:
        import ctypes
        import subprocess
        from types import SimpleNamespace

        import numpy as np
        try:
            from .. import native
            L = native.lib()
        except (ImportError, OSError, RuntimeError, subprocess.CalledProcessError):
            _NATIVE = False
            return _NATIVE
        workers = np.array(sorted(WORKER_IDS), np.int64)
        hired = np.array(sorted(HIRED), np.int64)
        _NATIVE = SimpleNamespace(lib=L, addr=native.addr, ucols=native.UCOLS, i32=np.dtype(np.int32), np=np,
                                  workers=workers, workers_at=native.addr(workers), hired=hired,
                                  hired_at=native.addr(hired), doubles=ctypes.c_double * 2, int32s=ctypes.c_int32,
                                  addressof=ctypes.addressof, from_buffer=ctypes.c_char.from_buffer)
    return _NATIVE


def _table_at(ua) -> int:
    """A unit table's address for the C helpers, or 0: not native.py's kind (int32, its columns, C order, writable),
    empty, or no library."""
    global _UA
    got = _UA
    if got[0] is ua:
        return got[1]
    nat = _NATIVE if _NATIVE is not None else _native()
    at = 0
    if nat and ua.dtype is nat.i32 and ua.ndim == 2 and ua.shape[1] == nat.ucols:
        try:  # (a ctypes char over its buffer: C order and writable, else an error)
            at = nat.addressof(nat.from_buffer(ua))
        except (TypeError, ValueError, BufferError):
            at = 0
    _UA = (ua, at)
    return at


def fill_rows(ua, t: int, out, cols: int) -> bool:
    """play.unit_rows' rows of a native unit table in C (w3obs_unit_rows): `out` [len(ua), cols] int64 (contiguous)
    gets the step t, then the table's first cols - 1 columns; False when the C helpers cannot (the numpy code then)."""
    at = _table_at(ua) if type(t) is int and len(ua) else 0
    if not at or cols > _NATIVE.ucols + 1:
        return False
    _NATIVE.lib.w3obs_unit_rows(at, len(ua), t, _NATIVE.addr(out), cols)
    return True


def _type_values(values: dict, types):
    """Each unit's value by its type id (values[str(id)], 0 for none), as float64: a sorted table of the types seen
    so far, made again when one is new (np.unique with its inverse, each step, had been ~85 us of an actor's step).
    The table is replaced whole: an actor's game threads share it (keys and values set apart, a thread had read new
    keys with old values: wrong values or an IndexError; audit 7)."""
    import numpy as np
    tab = _VALUES.get(id(values))
    if tab is None or tab[0] is not values or tab[1] != len(values):
        tab = (values, len(values), np.zeros(0, np.int64), np.zeros(0, np.float64))
    keys, vals = tab[2], tab[3]
    j = np.searchsorted(keys, types)
    if not len(keys) or not (keys[np.minimum(j, len(keys) - 1)] == types).all():
        keys = np.union1d(keys, types).astype(np.int64)
        vals = np.array([values.get(str(t), 0) for t in keys.tolist()], float)
        nat = _NATIVE if _NATIVE is not None else _native()
        desc = (np.array([nat.addr(keys), nat.addr(vals), len(keys), nat.workers_at, len(nat.workers)], np.int64)
                if nat and len(keys) else None)
        _VALUES[id(values)] = (values, len(values), keys, vals, desc, nat.addr(desc) if desc is not None else 0)
        j = np.searchsorted(keys, types)
    return vals[j]


def _material_c(ua, values: dict, hero_level, worker_cap):
    """material() of a native unit table in C (w3obs_material: the same numbers; numpy's ~10 calls had been ~190k
    instructions a step), or None: no library, another table, a type new to the values' table (the numpy code adds
    it), a hero level not a float (an int's products are int32 in numpy)."""
    if hero_level and type(hero_level) is not float:
        return None
    at = _table_at(ua)
    tab = _VALUES.get(id(values))
    if not at or tab is None or tab[0] is not values or tab[1] != len(values) or not tab[5]:
        return None
    io = _NATIVE.doubles(hero_level or 0.0, worker_cap or 0.0)  # (in: the two settings; out: the two sums)
    if _NATIVE.lib.w3obs_material(at, len(ua), tab[5], io):
        return None
    return {0: io[0], 1: io[1]}


def material(obs, values: dict, hero_level: float = 0.0, worker_cap: int = 0) -> dict[int, float]:
    """What each player's living units and buildings cost, times their hit points left: {0: .., 1: ..}.
    `hero_level`: a hero also counts this much for each level above the first (its cost alone does not
    grow as it levels; a dead hero counts nothing until it is revived). `worker_cap`: a player's workers
    count only as much as that many of them (their value times cap / their number when more): in long
    losses the learner made ~33 workers to the AI's ~19, and every one raised the shaping's potential."""
    out = {0: 0.0, 1: 0.0}
    ua = getattr(obs, "unit_array", None)
    if ua is not None and len(ua):  # native observations: in C, else vectorized
        fast = _material_c(ua, values, hero_level, worker_cap)
        if fast is not None:
            return fast
        import numpy as np
        live = ((ua[:, 11] & DEAD_FLAG) == 0) & ((ua[:, 2] == 0) | (ua[:, 2] == 1))
        a = ua[live]
        v = _type_values(values, a[:, 1])
        if hero_level:  # (column 14: the hero's level, 0 for other units)
            v = v + hero_level * np.maximum(a[:, 14] - 1, 0)
        hp = np.where(a[:, 7] > 0, a[:, 6] / np.maximum(a[:, 7], 1), 1.0)
        if worker_cap:
            work = np.isin(a[:, 1], list(WORKER_IDS))
            for p in (0, 1):
                n = int((work & (a[:, 2] == p)).sum())
                if n > worker_cap:
                    v = np.where(work & (a[:, 2] == p), v * worker_cap / n, v)
        vh = v * hp
        for p in (0, 1):
            out[p] = float(vh[a[:, 2] == p].sum())
        return out
    live = [u for u in obs.units if u.owner in (0, 1) and not int(u.flags) & DEAD_FLAG]
    n = {p: sum(u.type_id in WORKER_IDS for u in live if u.owner == p) for p in (0, 1)}
    for u in live:
        v = values.get(str(u.type_id), 0) + hero_level * max(int(getattr(u, "hero_level", 0) or 0) - 1, 0)
        if worker_cap and u.type_id in WORKER_IDS and n[u.owner] > worker_cap:
            v *= worker_cap / n[u.owner]
        out[u.owner] += v * (u.hp / u.max_hp if u.max_hp > 0 else 1.0)
    return out


def unit_values() -> dict[str, int]:
    """What each unit type costs (gold + lumber, the game's own tables): {type id: value}."""
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        rows = parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))
    out = {}
    for r in rows:
        oid = r.get("unitBalanceID", "")
        if len(oid) != 4:
            continue
        try:
            v = int(float(r.get("goldcost") or 0)) + int(float(r.get("lumbercost") or 0))
        except ValueError:
            continue
        if v > 0:
            out[str(int.from_bytes(oid.encode("latin-1"), "big"))] = v
    return out


def material_steps(units, steps: int, values: dict) -> "np.ndarray":
    """A demonstration's material (as material()) at each step: [steps, 2]. `units`: its unit rows
    (fullgame.features.load_game: step, id, type, owner, x, y, facing, hp, max hp, ..., flags)."""
    import numpy as np
    live = ((units[:, 12] & DEAD_FLAG) == 0) & ((units[:, 3] == 0) | (units[:, 3] == 1))
    a = units[live]
    types, inv = np.unique(a[:, 2], return_inverse=True)
    v = np.array([values.get(str(t), 0) for t in types.tolist()], float)[inv]
    hp = np.where(a[:, 8] > 0, a[:, 7] / np.maximum(a[:, 8], 1), 1.0)
    return np.bincount(a[:, 0] * 2 + a[:, 3], weights=v * hp, minlength=2 * steps)[:2 * steps].reshape(steps, 2)


WORKERS = {"hpea", "opeo", "uaco", "ewsp"}
HALLS = {"htow", "ogre", "unpl", "etol"}  # (a race's first-tier hall: one more is an expansion if at another mine)
MINES = {"ngol", "ugol", "egol"}  # (a gold mine, haunted or entangled)
EXPANSION_FROM_START = 1500.0  # a hall this far from its player's start, and
EXPANSION_AT_MINE = 1000.0  # this near a gold mine: an expansion (a second hall at home is none)


HIRED = {int.from_bytes(c.encode(), "big") for c in
         ("Nalc", "Ntin", "Nfir", "Nngs", "Nbrn", "Npbm", "Nplh", "Nbst", "nftb", "nfsp", "nogm", "ngrk")}


class _Scan:
    """Production's look at a native unit table in C (w3obs_prod_scan): the hired units' rows, and the (id, owner) pairs
    new or changed since its last look with the owners (it mirrors what it handed Python: Production.owners took each
    pair; the whole table's ids and owners as lists, zipped into the dict, had been ~100k instructions a step). A copy
    starts empty (it then hands every pair again: the same dict)."""

    def __init__(self, nat):
        self.nat, self.cap = nat, 0
        self.p = nat.lib.w3obs_owners_new(nat.hired_at, len(nat.hired))
        if not self.p:
            raise MemoryError("w3obs_owners_new")

    def __del__(self):
        p, self.p = getattr(self, "p", None), None
        if p:
            self.nat.lib.w3obs_owners_free(p)

    def __reduce__(self):
        return _new_scan, ()

    def clear(self) -> None:
        self.nat.lib.w3obs_owners_clear(self.p)

    def scan(self, ua, at: int, owners: bool) -> tuple[list, list]:
        """(the hired units' ids, types and owners, flat; the pairs' ids and owners, flat). `at`: ua's address."""
        n = len(ua)
        if 2 + 5 * n > self.cap:  # (this game's own buffer: a game's thread steps its Production)
            nat = self.nat
            self.cap = max(2 + 5 * n, 2 * self.cap, 1024)
            self.buf = (nat.int32s * self.cap)()
            self.at = nat.addressof(self.buf)
            self.ints = memoryview(self.buf).cast("B").cast("i")
        if self.nat.lib.w3obs_prod_scan(self.p, at, n, owners, self.at):
            raise MemoryError("w3obs_prod_scan")
        ints = self.ints
        h, q = 3 * ints[0], 2 * ints[1]
        if not h and not q:
            return [], []
        v = ints[2:2 + h + q].tolist()
        return v[:h], v[h:]


def _new_scan() -> _Scan:
    return _Scan(_native())


class Production:
    """What each player makes, loses and holds over a game, from the observations' events: units
    trained and buildings finished by type, research, kills and losses, food at one minute,
    resources on hand. The episode rows' "prod" / "opp_prod" (the dashboard's Production tab)."""

    def __init__(self):
        from collections import Counter
        self.trained = {0: Counter(), 1: Counter()}
        self.built = {0: Counter(), 1: Counter()}
        self.research: dict[int, dict[str, int]] = {0: {}, 1: {}}
        self.kills, self.lost = {0: 0, 1: 0}, {0: 0, 1: 0}
        self.held, self.food_1min, self.max_food = {0: 0.0, 1: 0.0}, {}, {0: 0, 1: 0}
        self.steps = 0
        # gathered so far at the game's start: a scripted reset (games without the built-in AI) keeps
        # the game's counters, so without this they added up over a launch's games (185k gold)
        self.base: dict[int, tuple[int, int]] = {}
        self.hired: set[int] = set()  # (tavern heroes and mercenaries seen: hired, no training event)
        self.mines: list[tuple[float, float]] | None = None  # the gold mines (their first step's)
        self.expansions: dict[int, list[float]] = {0: [], 1: []}  # each expansion's game second
        self.halls_home = {0: 0, 1: 0}  # more halls that are no expansion (at home, or at no mine)
        self.creeps = {0: 0, 1: 0}  # neutral units killed (creeping)
        # each unit's owner as last seen (kept after it is gone: a dead unit is mostly removed from the step that
        # reports its death, and the step's own table had missed ~half of the deaths, kills and creeps; audit 8)
        self.owners: dict[int, int] = {}
        from .features import codes
        self._hired_codes = codes(HIRED)  # (features.isin's)
        self._scan: _Scan | None = None  # (a native table's look in C: made on the first)

    def _hires(self, obs, rows=None) -> None:
        """A tavern hero or a mercenary of a player's seen for the first time: trained (bought, protocol.Buy).
        `rows`: their (id, type, owner), when the caller has them (_Scan's)."""
        rawcode = (_FX or _features()).rawcode
        if rows is None:
            isin = _FX.isin
            ua = getattr(obs, "unit_array", None)  # (their rows by a mask: each row in Python had been ~2% of a step)
            rows = ([(int(i), int(t), int(o)) for i, t, o in ua[isin(ua[:, 1], self._hired_codes), :3].tolist()]
                    if ua is not None else [(u.id, u.type_id, u.owner) for u in obs.units if u.type_id in HIRED])
        for i, t, o in rows:
            if i not in self.hired and o in (0, 1):
                self.hired.add(i)
                if self.steps > 1:  # (not one the map had at its start)
                    self.trained[o][rawcode(t)] += 1

    def _scanned(self, ua, owners: bool):
        """_Scan.scan of a native unit table, or None (no library, another table: the numpy code)."""
        at = _table_at(ua)
        if not at:
            return None
        if self._scan is None:
            self._scan = _Scan(_NATIVE)
        return self._scan.scan(ua, at, owners)

    def step(self, obs) -> None:
        rawcode = (_FX or _features()).rawcode
        self.steps += 1
        for p in (0, 1):
            s = obs.players.get(p)
            if s is None:
                continue
            self.base.setdefault(p, (s.gold_gathered, s.lumber_gathered))
            self.held[p] += s.gold + s.lumber
            self.max_food[p] = max(self.max_food[p], s.food_used)
            if obs.game_time >= 60 and p not in self.food_1min:
                self.food_1min[p] = s.food_used
        # (the owners every 4 steps and on any step with events: a summon's is event 17; audit 10)
        owners = bool(obs.events) or self.steps % 4 == 1
        ua = getattr(obs, "unit_array", None)
        scan = self._scanned(ua, owners) if ua is not None else None  # (in C: the hires' rows, the owners changed)
        if scan is None:
            self._hires(obs)
        else:
            it = iter(scan[0])
            self._hires(obs, zip(it, it, it))
        if self.mines is None:
            self.mines = [(u.x, u.y) for u in obs.units if rawcode(u.type_id) in MINES]
        events = [e for e in obs.events if int(e.kind) in (1, 3, 6, 9, 12)]
        if not owners:
            return
        owner = self.owners
        if scan is not None:  # (only the pairs new or changed: the same dict)
            it = iter(scan[1])
            owner.update(zip(it, it))
        else:
            owner.update(zip(ua[:, 0].tolist(), ua[:, 2].tolist()) if ua is not None and len(ua)
                         else ((u.id, u.owner) for u in obs.units))
            if self._scan is not None:  # (its mirror is no longer the dict's: it hands every pair again)
                self._scan.clear()
        for e in events:
            k = int(e.kind)
            if k == 6:  # a unit trained: a=building, b=the unit, c=its type
                p = owner.get(e.a)
                if p in (0, 1):
                    self.trained[p][rawcode(e.c)] += 1
            elif k in (3, 12):  # a building finished (b=type), or upgraded (b=the new type)
                p = owner.get(e.a)
                if p in (0, 1):
                    self.built[p][rawcode(e.b)] += 1
                    if k == 3 and (rawcode(e.b) in HALLS or rawcode(e.b) == "ugol"):  # (undead's: a haunted mine)
                        self._hall(p, e.a, obs)
            elif k == 9:  # research done: b=tech, c=level
                p = owner.get(e.a)
                if p in (0, 1):
                    code = rawcode(e.b)
                    self.research[p][code] = max(self.research[p].get(code, 0), e.c)
            elif k == 1:  # a death: a=the unit, b=its killer
                dead, killer = owner.get(e.a), owner.get(e.b)
                if dead in (0, 1):
                    self.lost[dead] += 1
                    if killer in (0, 1) and killer != dead:
                        self.kills[killer] += 1
                elif dead is not None and killer in (0, 1):  # (a neutral unit: a creep)
                    self.creeps[killer] += 1

    def _hall(self, p: int, unit: int, obs) -> None:
        """A hall of player p finished: an expansion when far from its start and at a gold mine."""
        u = next((x for x in obs.units if x.id == unit), None)
        s = obs.players.get(p)
        if u is None or s is None:
            return
        home = ((u.x - s.start_x) ** 2 + (u.y - s.start_y) ** 2) ** 0.5
        mine = min((((u.x - x) ** 2 + (u.y - y) ** 2) ** 0.5 for x, y in self.mines or ()), default=1e9)
        if home > EXPANSION_FROM_START and mine < EXPANSION_AT_MINE:
            self.expansions[p].append(round(obs.game_time))
        else:
            self.halls_home[p] += 1

    def row(self, p: int, obs) -> dict:
        s = obs.players.get(p)
        t = self.trained[p]
        return {"trained": dict(t), "built": dict(self.built[p]), "research": self.research[p],
                "workers": sum(v for k, v in t.items() if k in WORKERS),
                "heroes": sum(v for k, v in t.items() if k[:1].isupper()),
                "army": sum(v for k, v in t.items() if k not in WORKERS and not k[:1].isupper()),
                "kills": self.kills[p], "lost": self.lost[p], "food_1min": self.food_1min.get(p),
                "max_food": self.max_food[p], "held": round(self.held[p] / max(self.steps, 1)),
                "gold": s.gold_gathered - self.base.get(p, (0, 0))[0] if s else 0,
                "lumber": s.lumber_gathered - self.base.get(p, (0, 0))[1] if s else 0,
                "expansions": len(self.expansions[p]), **({"expansion_t": self.expansions[p][0]} if self.expansions[p] else {}),
                "halls_home": self.halls_home[p], "creeps": self.creeps[p]}
