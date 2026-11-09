"""--obs-full's block and new kinds in C (obsfull._apply_c, native/w3feat.c w3f_obs_a) against the Python they replaced
(master's obsfull.State and apply, kept below as text and run in obsfull's names): random games' steps of every kind of
event the block reads (units' state, buffs, charges, deaths, corpses seen and gone, area spells, research, production and
construction), heroes with items, items on the ground, more corpses in sight than are kept, a full entity table, the
store grown mid-game. Every array View.step gives (features, selection, types, masks) and the views' obs state the same
bit for bit; also the Python path (no library) against the same reference. scripts/bench_actor.py replay --compare checks
whole recorded games (commands, chunks, episode rows, view steps)."""

import numpy as np
import pytest

from warcraftsim.fullgame import featnative, features as fx

REF = r'''
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
    # (the entities in sight: inputs 3's remembered enemies after them are rows as last seen, and their state now, their
    # owner's upgrade levels now and their construction's progress now are what the player does not see)
    seen = n if getattr(view, "privileged_act", False) else max(0, min(int(getattr(view.enc, "ghost0", n)), n))
    if n:  # (Python's lists: the arrays are ~50 rows, numpy's calls on them had cost more than the loops)
        idl, tyl, owl = sel[:, fx.C_ID].tolist(), sel[:, fx.C_TYPE].tolist(), sel[:, fx.C_OWNER].tolist()
        if st.extras:  # (the units' extras: each unit's block row, kept until an event changes it)
            hit, got, spd = [], [], []
            sp = tb["speed"]
            for k, uid in enumerate(idl[:seen]):
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
        for k in range(seen):  # (the upgrade levels: per type and owner of the players' units in view)
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
                    if k is None or secs <= 0 or k >= seen or (mine and k >= n_own):
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
'''


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


TYPES = ("hfoo", "hbar", "Hamg", "ogru", "obar", "ugho", "hpea", "Hpal", "Obla", "nmrl", "ewsp", "uaco", "hkni", "otau")
VOCAB = {"types": [code(c) for c in TYPES[:9]], "current_orders": [], "upgrades": [],
         "orders": [[851983, fx.UNIT], [851986, fx.POINT], [851972, fx.IMMEDIATE]],
         "order_names": {"851983": "attack", "851986": "move", "851972": "stop"}}
BUFFS = [code(c) for c in ("Binf", "Beng", "Bslo", "BHbd", "BOac", "BUhf", "Bpoi", "Bcri", "Bfro", "BUau")]
ITEMS = [code(c) for c in ("phea", "pman", "tdex", "ratc", "stwp", "bspd", "ofir", "rde1")]
RESEARCH = [code(c) for c in ("Rhme", "Rhar", "Rome", "Rora", "Rume", "Rura", "Rhla")]
SPELLS = [code(c) for c in ("AHbz", "AHfs", "AUdd", "AOeq", "ANrf", "AEsf", "AHtb", "ANst")]


@pytest.fixture()
def ref():
    try:
        from warcraftsim.fullgame import obsfull
        obsfull.tables()
    except Exception as e:  # noqa: BLE001 (the game's tables: not on every machine)
        pytest.skip(f"no game data: {e}")
    fx.set_inputs(2)
    fx.set_obs_full(1)
    ns = dict(obsfull.__dict__)
    exec(compile(REF, "<master's obsfull>", "exec"), ns)
    yield ns["apply"]
    fx.set_obs_full(0)
    fx.set_inputs(1)


class Game:
    """A random game's steps: its units' rows (some heroes with items, buildings, constructions, the dead), events of
    every kind obs-full reads, items on the ground."""

    def __init__(self, seed: int, units: int = 60):
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.units = {}
        for uid in range(1, units + 1):
            self.units[uid] = self._unit(uid)
        self.next_id = units + 1
        self.corpses: list = []
        self.casters: list = []

    def _unit(self, uid: int) -> np.ndarray:
        r = self.rng
        u = np.zeros(fx.ROW_COLS, np.int64)
        typ = TYPES[int(r.integers(len(TYPES)))]
        u[fx.C_ID], u[fx.C_TYPE] = uid, code(typ)
        u[fx.C_OWNER] = (0, 0, 1, 1, 12)[int(r.integers(5))]
        u[fx.C_X], u[fx.C_Y] = int(r.integers(-5000, 5000)), int(r.integers(-4000, 4000))
        u[fx.C_HP] = u[fx.C_MAXHP] = 400
        flags = 0
        if typ[0].isupper():
            flags |= fx.HERO
            for s in range(6):
                if r.random() < 0.5:
                    u[fx.C_ITEM0 + s] = ITEMS[int(r.integers(len(ITEMS)))] if r.random() < 0.8 else int(r.integers(1, 2 ** 40))
        if typ in ("hbar", "obar"):
            flags |= fx.STRUCTURE | (fx.CONSTRUCTING if r.random() < 0.3 else 0)
        u[fx.C_FLAGS], u[fx.C_VIS] = flags, int(r.integers(0, 4))
        return u

    def step(self, t: int) -> tuple:
        r = self.rng
        ids = list(self.units)
        ev = []

        def pick():
            return int(ids[int(r.integers(len(ids)))]) if r.random() < 0.9 else int(r.integers(1, 3 * len(ids)))
        for _ in range(int(r.integers(0, 6))):  # (units' state, then buffs and charges: as the harness sends them)
            u = pick()
            ev.append((t, 101, u, int(r.integers(0, 2 ** 16)), (int(r.integers(0, 523)) << 12) | int(r.integers(0, 4096))))
            for _ in range(int(r.integers(0, 4))):
                b = BUFFS[int(r.integers(len(BUFFS)))] if r.random() < 0.9 else int(r.integers(1, 2 ** 32))
                ev.append((t, 102, u, b, int(r.integers(0, 2))))
            if r.random() < 0.3:
                ev.append((t, 105, u, int(r.integers(0, 2 ** 30)), int(r.integers(0, 2 ** 30))))
        if r.random() < 0.2:  # (a buff or charges for one without a state now and then)
            ev.append((t, (102, 105)[int(r.integers(2))], pick(), BUFFS[0], 3))
        if r.random() < 0.4 and len(self.units) > 20:  # a death: its corpse, seen
            u = pick()
            if u in self.units:
                row = self.units.pop(u)
                ev += [(t, 1, u, 0, int(row[fx.C_TYPE])), (t, 106, u, int(row[fx.C_X]), int(row[fx.C_Y])),
                       (t, 108, u, int(r.integers(0, 4)), 0)]
                self.corpses.append(u)
                if r.random() < 0.3:  # (its id taken by a new unit at once: the corpse's handle reused)
                    self.units[u] = self._unit(u)
        if self.corpses and r.random() < 0.05:
            ev.append((t, 107, self.corpses.pop(int(r.integers(len(self.corpses)))), 0, 0))
        if r.random() < 0.15:  # a spell's point, then the ability and the seers
            u = pick()
            ev += [(t, 104, u, int(r.integers(-4000, 4000)), int(r.integers(-3000, 3000))),
                   (t, 109, u, SPELLS[int(r.integers(len(SPELLS)))], int(r.integers(0, 4)))]
        if r.random() < 0.1:
            ev.append((t, fx.RESEARCH_FINISH, pick(), RESEARCH[int(r.integers(len(RESEARCH)))], int(r.integers(1, 4))))
        if r.random() < 0.2:
            k = (5, 8, 11)[int(r.integers(3))]
            b = {5: code("hfoo"), 8: RESEARCH[int(r.integers(len(RESEARCH)))], 11: code("htow")}[k]
            ev.append((t, k, pick(), b, 0))
        if r.random() < 0.15:
            ev.append((t, (6, 7, 9, 10, 12, 13)[int(r.integers(6))], pick(), RESEARCH[0], int(r.integers(1, 3))))
        if r.random() < 0.1:
            ev.append((t, 2, pick(), code("hbar"), 0))
        if r.random() < 0.1:
            ev.append((t, (3, 4)[int(r.integers(2))], pick(), 0, 0))
        for u in list(self.units):  # (moves; new units)
            if r.random() < 0.3:
                self.units[u][fx.C_X] += int(r.integers(-100, 100))
        if r.random() < 0.3:
            self.units[self.next_id] = self._unit(self.next_id)
            self.next_id += 1
        rows = np.stack(list(self.units.values()))
        ground = [(ITEMS[int(r.integers(len(ITEMS)))], int(r.integers(-4000, 4000)), int(r.integers(-3000, 3000)),
                   int(r.integers(0, 4)), int(r.integers(0, 2))) for _ in range(int(r.integers(0, 9)))]
        return rows, np.asarray(ev, np.int64).reshape(-1, 5), ground


def _play(seed: int, player: int, steps: int, apply_fn=None, monkeypatch=None, burst: int = 0) -> list:
    """A game's views' steps (each step's arrays, and the obs state at the end) with apply_fn as obsfull.apply."""
    from warcraftsim.fullgame import obsfull
    enc = fx.Encoder(VOCAB)
    view = enc.view(player, 1.0 if player == 0 else -1.0, [0, 1])
    game = Game(seed)
    out = []
    if apply_fn is not None:
        monkeypatch.setattr(obsfull, "apply", apply_fn)
    try:
        for t in range(steps):
            rows, ev, ground = game.step(t)
            if burst and t == 7:  # (many units' states at once: the store grows)
                ev = np.concatenate([ev, np.asarray([(t, 101, 5000 + k, k, (100 << 12) | 2050) for k in range(burst)],
                                                    np.int64)])
            view.ground = ground
            st = view.step(rows, None, ev, t)
            out.append({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()})
        o = view.obs
        out.append({"corpses": dict(o.corpses), "areas": [list(a) for a in o.areas], "enemy": dict(o.enemy_levels),
                    "since": dict(o.since), "built": dict(o.built), "scale": o.scale, "levels": dict(view.levels)})
    finally:
        if apply_fn is not None:
            monkeypatch.undo()
    return out


def _diff(a: list, b: list) -> list:
    bad = []
    assert len(a) == len(b)
    for t, (x, y) in enumerate(zip(a, b)):
        assert x.keys() == y.keys(), t
        for k in x:
            u, v = x[k], y[k]
            if isinstance(u, np.ndarray):
                if not (isinstance(v, np.ndarray) and u.dtype == v.dtype and u.shape == v.shape and np.array_equal(u, v)):
                    bad.append((t, k))
            elif u != v:
                bad.append((t, k))
    return bad


def _new_kinds(steps: list) -> np.ndarray:
    A = fx.A_OBS
    return np.array([st["ent"][:, A:A + 3].sum(0) for st in steps[:-1]])


@pytest.mark.parametrize("seed,player", [(0, 0), (1, 1), (2, 0), (3, 1)])
def test_c_path_as_master(ref, monkeypatch, seed, player):
    want = _play(seed, player, 160, ref, monkeypatch, burst=600)
    got = _play(seed, player, 160, burst=600)
    assert _diff(want, got) == []
    kinds = _new_kinds(got)
    assert (kinds[:, 0] == 8).any() and kinds[:, 1].any() and kinds[:, 2].any()  # (corpses capped, items, areas seen)
    A = fx.A_OBS
    blocks = np.concatenate([st["ent"][:, A:] for st in got[:-1]])
    from warcraftsim.fullgame import obsfull as of
    for lo, hi in ((of.K_CLASS, of.K_SPEED_R), (of.K_SPEED_R, of.K_HAVE), (of.K_BUFF, of.K_UPG), (of.K_UPG, of.K_ITEM),
                   (of.K_ITEM, of.K_CHARGE), (of.K_CHARGE, of.K_PROD), (of.K_PROD, of.K_AGE)):
        assert (blocks[:, lo:hi] != 0).any(), (lo, hi)  # (every part of the block exercised)


def test_c_path_with_a_full_table(ref, monkeypatch):
    monkeypatch.setattr(fx, "MAX_ENT", 40)  # (the entities near the cap: the new kinds' rows cut at the room left)
    want = _play(5, 0, 80, ref, monkeypatch)
    monkeypatch.setattr(fx, "MAX_ENT", 40)
    got = _play(5, 0, 80)
    assert _diff(want, got) == []
    assert max(st["n"] for st in got[:-1]) == 40


def test_python_path_as_master(ref, monkeypatch):
    monkeypatch.setattr(featnative, "_lib", False)  # (no library: the numpy features, State.extras, _apply_py)
    want = _play(4, 1, 120, ref, monkeypatch)
    monkeypatch.setattr(featnative, "_lib", False)
    got = _play(4, 1, 120)
    assert _diff(want, got) == []


def test_c_path_is_used(ref):
    from warcraftsim.fullgame import obsfull
    enc = fx.Encoder(VOCAB)
    view = enc.view(0, 1.0, [0, 1])
    rows, ev, ground = Game(0).step(0)
    view.ground = ground
    view.step(rows, None, ev, 0)
    assert view.obs.native and not view.obs.extras and obsfull._native(view) is not None


@pytest.mark.parametrize("obs_full", [0, 1])
def test_collate_entities_as_before(obs_full):
    """The learner's minibatch entities: collated as float16 (the chunks' dtype) and made float32 after the copy, the
    same float32 tensor as master's float32 fill; a step of another dtype keeps the float32 buffer."""
    import torch

    from warcraftsim.fullgame.selfplay import collate_seq
    fx.set_inputs(2)
    fx.set_obs_full(obs_full)
    try:
        rng = np.random.default_rng(obs_full)
        G = 40

        def step(n, dt=np.float16):
            e = (rng.standard_normal((n, fx.F)) * 3).astype(np.float16)
            e[rng.random((n, fx.F)) < 0.6] = 0  # (zeros, as most channels)
            e[0, 0] = np.float16(np.inf) if n > 30 else np.float16(6e-8)  # (the extremes float16 has)
            if dt != np.float16:  # (numbers float16 can't hold)
                e = (rng.standard_normal((n, fx.F)) / 3).astype(dt)
            return {"ent": e.astype(dt), "type": rng.integers(0, 9, n).astype(np.int16),
                    "cur": rng.integers(0, 3, n).astype(np.int16), "glob": rng.standard_normal(G).astype(np.float32),
                    "n": n, "n_own": min(n, 5), "logp": np.zeros(min(n, 5), np.float32), "adv": 0.5, "ret": 0.1,
                    "avail": None, **{k: np.zeros(min(n, 5), np.int16) for k in ("order", "tgt", "bx", "by")}}
        for mixed in (False, True):
            seqs = [[step(int(rng.integers(8, 70))) for _ in range(int(rng.integers(1, 8)))] for _ in range(6)]
            if mixed:
                seqs[2][0] = step(20, np.float32)
            mb = collate_seq(seqs, torch.device("cpu"), width=80, B=7, T=8)
            T = 8
            flat = [q[t] if t < len(q) else None for q in seqs + [[]] for t in range(T)]
            want = np.zeros((len(flat), 80, fx.F), np.float32)  # (master's: a float32 buffer the steps fill)
            for i, s in enumerate(flat):
                if s is not None:
                    want[i, :s["n"]] = s["ent"]
            got = mb["ent"]
            assert got.dtype == torch.float32 and got.shape == want.shape
            assert np.array_equal(got.numpy().view(np.uint32), want.view(np.uint32))  # (bit for bit)
    finally:
        fx.set_obs_full(0)
        fx.set_inputs(1)
