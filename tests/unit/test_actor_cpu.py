"""The actors' per-step Python made cheaper (2026-11-06): View.step's last-order features, the --econ-masks masks
(econ_allowed, chores, futile with its spells and mines in C, item_slots) and --full-vocab's Checks.mask, each against the
code it replaced (master's, kept below as the reference and run in its module's names) on random sides of every race:
the same masks, features and side attributes, bit for bit. scripts/bench_actor.py replay --compare checks whole
recorded games (every step's commands, chunks, episode rows and view steps)."""

import copy

import numpy as np
import pytest

from warcraftsim.fullgame import featnative, features as fx, herodata


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


REF_FEATURES = 'def econ_allowed(self, rows: np.ndarray) -> np.ndarray:\n    """--econ-masks: the order classes ECON_MASK leaves the player: a tech building\'s build order while a copy it has\n    is idle (finished and making nothing: View.busy; under construction; or a worker on its way to build one),\n    and its worker\'s train order once its workers, alive and in training, reach worker_cap for its mines."""\n    enc = self.enc\n    race = RACES[self.races[self.player]]\n    cls = self.__dict__.get("_econ_cls")\n    if cls is None or cls[0] is not enc:  # (the race\'s classes, once: [(type code, build class)], worker, mines)\n        w, mines, _, _ = ECON_WORKERS[race]\n        codes = [int.from_bytes(x.encode(), "big") for x in ECON_TECH[race]]\n        tech = [(c, enc.order_index[(c, POINT)]) for c in codes if (c, POINT) in enc.order_index]\n        wc = int.from_bytes(w.encode(), "big")\n        cls = self._econ_cls = (enc, tech, wc, enc.order_index.get((wc, IMMEDIATE)),\n                                np.array([int.from_bytes(m.encode(), "big") for m in mines], np.int64))\n    _, tech, wc, wk, mines = cls\n    ok = np.ones(enc.n_orders, bool)\n    r = rows[(rows[:, C_OWNER] == self.player) & (rows[:, C_FLAGS] & DEAD == 0)]\n    types, fl = r[:, C_TYPE], r[:, C_FLAGS]\n    built = fl & STRUCTURE > 0\n    coming = set(r[(fl & WORKER > 0) & (r[:, C_ORDER] >= TYPE_CODE), C_ORDER].tolist())  # (on their way to build)\n    for c, k in tech:\n        copies = r[built & (types == c)][:, [C_ID, C_FLAGS]].tolist()\n        if c in coming or any(i not in self.busy or f & CONSTRUCTING for i, f in copies):  # (a copy idle or building)\n            ok[k] = False\n    if wk is not None:\n        n = int((types == wc).sum()) + sum(1 for m in self.making.values() if m[0] == 0 and m[1] == wc)\n        if n >= worker_cap(race, int((built & np.isin(types, mines)).sum())):\n            ok[wk] = False\n    return ok\n\ndef chores(self, rows: np.ndarray, avail: np.ndarray) -> None:\n    """--econ-masks bits 8, 16, 32 (DETONATE_MASK, REPAIR_MASK, MILITIA_MASK): the orders that can do nothing now out\n    of `avail` (in place), from the player\'s units and the enemy units it sees; REPAIR_MASK also sets repair_ids,\n    the ids of the buildings and machines worth a repair (play.BCAgent sends repairs to those only)."""\n    oix = self.enc.order_index\n    race = RACES[self.races[self.player]]\n    p = self.player\n    fl = rows[:, C_FLAGS]\n    live = fl & DEAD == 0\n    own = rows[live & (rows[:, C_OWNER] == p)]\n    seen = live & (rows[:, C_OWNER] == 1 - p) & ((rows[:, C_VIS] >> p) & 1 == 1)\n    enemy = rows[seen & (fl & STRUCTURE == 0)]\n    ex, ey = enemy[:, C_X].astype(np.float64), enemy[:, C_Y].astype(np.float64)\n\n    def near(units: np.ndarray, d: float, among=None) -> np.ndarray:\n        """Per unit row: an enemy unit (of `among`\'s, else any it sees) within d of it."""\n        ax, ay = (ex, ey) if among is None else (among[:, C_X].astype(np.float64), among[:, C_Y].astype(np.float64))\n        if not len(units) or not len(ax):\n            return np.zeros(len(units), bool)\n        ux, uy = units[:, C_X].astype(np.float64), units[:, C_Y].astype(np.float64)\n        return ((ux[:, None] - ax[None]) ** 2 + (uy[:, None] - ay[None]) ** 2 < d * d).any(1)\n\n    def mask(keys, ok: bool) -> None:\n        if not ok:\n            for k in keys:\n                c = oix.get(k)\n                if c is not None:\n                    avail[c] = False\n    if DETONATE_MASK and race == "nightelf":\n        summons = enemy[enemy[:, C_FLAGS] & SUMMONED > 0]\n        wisps = own[own[:, C_TYPE] == WISP_T]\n        mask(((DETONATE, POINT), (DETONATE, UNIT)), bool(near(wisps, DETONATE_NEAR, summons).any()))\n    if REPAIR_MASK:\n        of = own[:, C_FLAGS]\n        cand = own[((of & STRUCTURE > 0) | np.isin(own[:, C_TYPE], list(REPAIR_MACHINES))) & (of & CONSTRUCTING == 0)\n                   & (own[:, C_HP] < own[:, C_MAXHP])]\n        low = cand[:, C_HP] < REPAIR_BELOW * cand[:, C_MAXHP]\n        need = cand[low | near(cand, UNDER_ATTACK_NEAR)]\n        self.repair_ids = set(need[:, C_ID].tolist())\n        mask(((REPAIR, UNIT), (REPAIR, POINT)), len(need) > 0)\n    if MILITIA_MASK and race == "human":\n        halls = own[np.isin(own[:, C_TYPE], HUMAN_HALLS) & (own[:, C_FLAGS] & CONSTRUCTING == 0)]\n        militia = bool((own[:, C_TYPE] == MILITIA_T).any())\n        mask(((TOWNBELL_ON, IMMEDIATE), (MILITIA_CONVERT, UNIT), (MILITIA_CALL, IMMEDIATE)),\n             bool(near(halls, MILITIA_NEAR).any()))\n        mask(((TOWNBELL_OFF, IMMEDIATE), (MILITIA_UNCONVERT, UNIT), (MILITIA_OFF, IMMEDIATE)), militia)\n    if BURROW_MASK and race == "orc":\n        of = own[:, C_FLAGS]\n        burrows = own[(own[:, C_TYPE] == BURROW_T) & (of & CONSTRUCTING == 0)]\n        halls = own[np.isin(own[:, C_TYPE], ORC_HALLS) & (of & CONSTRUCTING == 0)]\n        mask(((BATTLESTATIONS, IMMEDIATE), (BOARD, UNIT)),\n             bool(near(np.concatenate([burrows, halls]), MILITIA_NEAR).any()))\n        self.burrows_safe = set(burrows[~near(burrows, MILITIA_NEAR), C_ID].tolist())  # (play.BCAgent\'s stand-down)\n\ndef futile(self, rows: np.ndarray, avail: np.ndarray, t: int) -> None:\n    """--econ-masks bit 64 (FUTILE_MASK): the orders no unit of the player\'s can give now with the game taking it,\n    out of `avail` (in place) at step t: a harvest while its haunted or entangled mines are full, Death Coil and Holy\n    Light with no unit it sees to take them, an item slot\'s immediate use with no hero holding an item there that it\n    can use so (the per-unit and per-target checks: play.BCAgent)."""\n    oix = self.enc.order_index\n    race = RACES[self.races[self.player]]\n    p = self.player\n    own = own_alive(rows, p)\n\n    def off(keys) -> None:\n        for k in keys:\n            c = oix.get(k)\n            if c is not None:\n                avail[c] = False\n    mined = MINED_MINES.get(race)\n    if mined is not None and avail[oix.get((HARVEST, UNIT), 0)]:\n        mines = own[(own[:, C_TYPE] == mined) & (own[:, C_FLAGS] & CONSTRUCTING == 0)]\n        full = self.__dict__.get("full_mines") or {}  # (play.BCAgent.accepted: a mine\'s refused harvest, its step)\n        if not any(len(mine_workers(own, m)) < MINE_WORKERS and not 0 <= t - full.get(int(m[C_ID]), -MINE_FULL_STEPS)\n                   < MINE_FULL_STEPS for m in mines):\n            off(((HARVEST, UNIT),))\n    for o in FUTILE_SPELLS:\n        c = oix.get((o, UNIT))\n        if c is not None and avail[c] and not (spell_targets(o, rows, p) & ~((rows[:, C_TYPE] == SPELL_CASTERS[o])\n                                                                           & (rows[:, C_OWNER] == p))).any():\n            avail[c] = False  # (the caster itself no target: its kind\'s units of the player\'s left out)\n    heroes = own[own[:, C_FLAGS] & HERO > 0] if own.shape[1] > C_ITEM0 + 5 else own[:0]\n    for k in range(6):\n        c = oix.get((USESLOT + k, IMMEDIATE))\n        if c is not None and avail[c] and not any(item_use_now(int(h[C_ITEM0 + k]), h) for h in heroes):\n            avail[c] = False\n\ndef item_slots(self, rows: np.ndarray, avail: np.ndarray) -> None:\n    """--shop-items: each slot\'s use (immediate, on a unit, on a point) out of `avail` (in place) while no hero of the\n    player\'s holds an item in that slot that takes it (shopitems.use_ok; the per-hero check: play.BCAgent)."""\n    own = own_alive(rows, self.player)\n    heroes = own[own[:, C_FLAGS] & HERO > 0] if own.shape[1] > C_ITEM0 + 5 else own[:0]\n    use_ok = _mod("shopitems").use_ok\n    oix = self.enc.order_index\n    for k in range(6):\n        for kind in (IMMEDIATE, UNIT, POINT):\n            c = oix.get((USESLOT + k, kind))\n            if c is not None and avail[c] and not any(use_ok(int(h[C_ITEM0 + k]), kind, h) for h in heroes):\n                avail[c] = False\n\n'

REF_MASK = 'def mask(self, view, rows, avail, t: int) -> None:\n    """`avail` (in place) without extend\'s orders no unit of the view\'s player can give now."""\n    own = fx.own_alive(rows, view.player)\n    cooling = getattr(view, "cooling", {})\n    states = getattr(view, "autocast", {})\n    types = own[:, fx.C_TYPE]\n    # (the unit types the player has: most classes are another race\'s, no giver of theirs here, and one np.isin per\n    # class had been ~0.7 ms a side-step, more than all the rest of the actor\'s Python per step)\n    present = set(types.tolist())\n    for c, need in self.timed.items():\n        if c >= len(avail) or not avail[c]:\n            continue\n        if present.isdisjoint(need):\n            avail[c] = False\n            continue\n        rows_c = own[np.isin(types, list(need))]\n        avail[c] = any(self.unit_ok(c, r, t, cooling, states) for r in rows_c)\n    for c, (ability, on, givers) in self.toggle.items():  # (a toggle while a giver is in the other state)\n        if c >= len(avail) or not avail[c] or c in self.timed:\n            continue\n        if present.isdisjoint(givers):\n            avail[c] = False\n            continue\n        ids = own[np.isin(types, list(givers)), fx.C_ID].tolist()\n        avail[c] = any(self.state(states, int(u), ability) != on for u in ids)\n    if self.night and not self.is_night(t, getattr(view, "step_seconds", 0.5)):\n        for c in self.night:\n            if c < len(avail):\n                avail[c] = False\n    for c, pairs in self.partner.items():\n        if c >= len(avail) or not avail[c]:\n            continue\n        ok = False\n        for g, p in pairs.items():\n            if not (types == g).any():\n                continue\n            if p == GOLD_MINE:  # (a gold mine anyone may entangle: one not entangled yet, on the map)\n                ok = ok or bool((rows[:, fx.C_TYPE] == GOLD_MINE).any())\n            elif p == code("otau"):  # (a dead tauren of the player\'s lying there)\n                ok = ok or bool(((rows[:, fx.C_OWNER] == view.player) & (rows[:, fx.C_TYPE] == p)\n                                 & (rows[:, fx.C_FLAGS] & fx.DEAD > 0)).any())\n            else:\n                ok = ok or bool(((types == p) & (own[:, fx.C_FLAGS] & fx.CONSTRUCTING == 0)).any())\n        avail[c] = ok\n    heroes = own[own[:, fx.C_FLAGS] & fx.HERO > 0] if own.shape[1] > fx.C_ITEM0 + 5 else own[:0]\n    if self.item:\n        shop = bool(np.isin(rows[:, fx.C_TYPE], PAWN_SHOPS).any()) or bool(\n            (np.isin(types, RACE_SHOP_CODES) & (own[:, fx.C_FLAGS] & fx.CONSTRUCTING == 0)).any())\n        for c, (verb, k) in self.item.items():\n            if c >= len(avail) or not avail[c]:\n                continue\n            held = bool((heroes[:, fx.C_ITEM0 + k] != 0).any()) if len(heroes) else False\n            avail[c] = held and (verb != "give" or len(heroes) > 1) and (verb != "sell" or shop)\n    if self.pickup is not None and self.pickup < len(avail) and avail[self.pickup]:\n        bit = 1 << view.player\n        seen = any(vis & bit for _, _, _, vis, _ in (getattr(view, "ground", ()) or ()))\n        avail[self.pickup] = seen and len(heroes) > 0 and bool((heroes[:, fx.C_ITEM0:fx.C_ITEM0 + 6] == 0).any())\n    lab = bool((rows[:, fx.C_TYPE] == LAB).any())\n    for c in self.lab + ([self.reveal] if self.reveal is not None else []):\n        if c < len(avail):\n            avail[c] = avail[c] and lab\n'

REF_LAST = 'def last_loop(self, f, sel, ids, n_own, t):\n    """View.step\'s last-order loop as it was (a unit at a time, numpy\'s scalars and np.clip)."""\n    index = {u: k for k, u in enumerate(ids)} if self.active_f > A_LAST else {}\n    if self.active_f > A_LAST:  # (a per-unit loop: skipped when the run does not use them, --features)\n        f[:n_own, A_LAST + 4] = 1.0  # (never ordered: long ago)\n    for k in range(n_own if self.active_f > A_LAST else 0):\n        lo = self.last_order.get(ids[k])\n        if lo is None:\n            continue\n        kind, x, y, target, t0 = lo\n        f[k, A_LAST + 4] = min(1.0, (t - t0) / LAST_AGE)\n        if kind == 2 and target in index:  # (a target out of sight: its kind only)\n            x, y = float(sel[index[target], C_X]), float(sel[index[target], C_Y])\n        if kind == 1 or (kind == 2 and target in index):\n            f[k, A_LAST] = np.clip(self.sign * (x - sel[k, C_X]) / LAST_SCALE, -2.0, 2.0)\n            f[k, A_LAST + 1] = np.clip((y - sel[k, C_Y]) / LAST_SCALE, -2.0, 2.0)\n        if kind:\n            f[k, A_LAST + 1 + kind] = 1.0\n'


def _ref(src: str, module) -> dict:
    """The reference functions, run in the names of the module they came from."""
    ns = dict(vars(module))
    exec(src, ns)  # noqa: S102 (this file's own reference code)
    return ns


UNDEAD = ("uaco", "ugol", "unpl", "Udea", "ugho", "ucry", "uabo", "ugrv", "uslh", "uaod", "utom", "usep", "unp1")
MECH = ("ugol", "unpl", "hgyr", "hmtt", "ocat", "umtw", "ebal")
ANCIENT = ("etol", "eaom", "edob", "etoa", "eaoe")
ITEMS = ("dust", "stwp", "phea", "pman", "rnec", "pghe", "pgma", "shea", "ankh", "wneg")
TYPES = sorted({*UNDEAD, *MECH, *ANCIENT, "hpea", "htow", "hkee", "hcas", "hbar", "halt", "hars", "harm", "hgra", "hbla",
                "hvlt", "hfoo", "hmil", "Hpal", "Hamg", "opeo", "ogre", "ostr", "ofrt", "otrb", "oalt", "osld", "otto",
                "obea", "ovln", "ogru", "Obla", "egol", "ewsp", "eaoe", "eaow", "edos", "eate", "eden", "earc", "Edem",
                "ngol", "nmrk", "ngme", "ngad", "edry", "hspt", "efdr", "ubsp", "nfrl", "Npbm", "otau"})
WORK_ORDERS = sorted(fx.HARVESTING | {fx.SMART, 851986, 851983, 0, 0, code("hbar"), code("halt"), code("usep")})


@pytest.fixture()
def kinds(monkeypatch):
    """The game's unit kinds and items as the masks read them (no game data needed)."""
    monkeypatch.setattr(herodata, "unit_kinds", lambda: {"undead": frozenset(map(code, UNDEAD)),
                                                         "mechanical": frozenset(map(code, MECH)),
                                                         "ancient": frozenset(map(code, ANCIENT))})
    table = {code(c): (1, 0, 0, int(c == "stwp")) for c in ITEMS}
    monkeypatch.setattr(herodata, "item_table", lambda: table)
    monkeypatch.setattr(fx, "_KINDS", None)
    yield
    fx.set_econ_masks(0)
    fx.set_shop_items(0)


def _rows(rng, n: int, player: int) -> np.ndarray:
    """A step's random rows: units of both players and neutrals, near each other and far, any flags."""
    r = np.zeros((n, fx.ROW_COLS), np.int64)
    r[:, fx.C_ID] = rng.permutation(np.arange(1, 4 * n + 1))[:n]
    r[:, fx.C_TYPE] = [code(TYPES[k]) for k in rng.integers(len(TYPES), size=n)]
    r[:, fx.C_OWNER] = rng.choice([player, player, 1 - player, 12, 15], n)
    r[:, fx.C_X], r[:, fx.C_Y] = rng.integers(-2500, 2500, n), rng.integers(-2500, 2500, n)
    r[:, fx.C_MAXHP] = rng.integers(100, 1500, n)
    r[:, fx.C_HP] = np.where(rng.random(n) < 0.5, r[:, fx.C_MAXHP], (r[:, fx.C_MAXHP] * rng.random(n)).astype(np.int64))
    r[:, fx.C_MAXMANA] = rng.integers(0, 600, n)
    r[:, fx.C_MANA] = (r[:, fx.C_MAXMANA] * rng.random(n)).astype(np.int64)
    r[:, fx.C_ORDER] = rng.choice(WORK_ORDERS, n)
    bits = (fx.STRUCTURE, fx.WORKER, fx.HERO, fx.DEAD, fx.CONSTRUCTING, fx.SUMMONED, fx.HIDDEN_FLAG)
    p = (0.3, 0.3, 0.15, 0.08, 0.1, 0.1, 0.1)
    r[:, fx.C_FLAGS] = sum(np.where(rng.random(n) < q, b, 0) for b, q in zip(bits, p))
    r[:, fx.C_VIS] = rng.integers(0, 4, n)
    items = rng.choice([0, 0, 0] + [code(c) for c in ITEMS], (n, 6))
    r[:, fx.C_ITEM0:fx.C_ITEM0 + 6] = items
    return r


def _view(race: int, player: int, rng):
    orders = ([[fx.HARVEST, fx.UNIT], [fx.DEATH_COIL, fx.UNIT], [fx.HOLY_LIGHT, fx.UNIT], [fx.REPAIR, fx.UNIT],
               [fx.REPAIR, fx.POINT], [fx.DETONATE, fx.POINT], [fx.DETONATE, fx.UNIT], [fx.BATTLESTATIONS, fx.IMMEDIATE],
               [fx.BOARD, fx.UNIT]]
              + [[o, fx.IMMEDIATE] for o in (fx.TOWNBELL_ON, fx.TOWNBELL_OFF, fx.MILITIA_CALL, fx.MILITIA_OFF)]
              + [[fx.MILITIA_CONVERT, fx.UNIT], [fx.MILITIA_UNCONVERT, fx.UNIT]]
              + [[fx.USESLOT + k, kind] for k in range(6) for kind in (fx.IMMEDIATE, fx.UNIT, fx.POINT)]
              + [[code(c), fx.POINT] for techs in fx.ECON_TECH.values() for c in techs]
              + [[code(w), fx.IMMEDIATE] for w, *_ in fx.ECON_WORKERS.values()])
    vocab = {"types": [code(c) for c in TYPES], "current_orders": [], "upgrades": [], "orders": orders}
    enc = fx.Encoder(vocab, np.zeros((len(orders) + 1, 3), np.int64))
    view = enc.view(player, 1.0, [race, 1 - race if race < 2 else 0])
    rows = _rows(rng, int(rng.integers(1, 120)), player)
    view.busy = set(rng.choice(rows[:, fx.C_ID], int(rng.integers(0, len(rows) + 1))).tolist())
    view.making = {int(i): (int(rng.integers(0, 5)), int(rng.choice(rows[:, fx.C_TYPE]))) for i in rng.choice(rows[:, fx.C_ID], 4)}
    view.full_mines = {int(i): int(rng.integers(0, 120)) for i in rng.choice(rows[:, fx.C_ID], 3)}
    return view, rows, len(orders) + 1


@pytest.mark.parametrize("shop_items", [0, 1])
def test_econ_chores_futile_slots_as_before(kinds, shop_items):
    fx.set_econ_masks(255)
    fx.set_shop_items(shop_items)
    ref = _ref(REF_FEATURES, fx)
    rng = np.random.default_rng(shop_items)
    seen = {"repair": 0, "burrows": 0, "spells": 0, "harvest": 0}
    for trial in range(600):
        race, player = trial % 4, int(rng.integers(2))
        view, rows, n = _view(race, player, rng)
        t = int(rng.integers(0, 200))
        start = rng.random(n) < 0.85
        start[0] = True
        old, new = copy.deepcopy(view), copy.deepcopy(view)
        assert np.array_equal(ref["econ_allowed"](old, rows), new.econ_allowed(rows)), trial
        old.__dict__.pop("_econ_cls", None)
        a, b = start.copy(), start.copy()
        ref["chores"](old, rows, a)
        new.chores(rows, b)
        assert np.array_equal(a, b), trial
        assert getattr(old, "repair_ids", None) == getattr(new, "repair_ids", None), trial
        assert getattr(old, "burrows_safe", None) == getattr(new, "burrows_safe", None), trial
        ref["futile"](old, rows, a, t)
        new.futile(rows, b, t)
        assert np.array_equal(a, b), trial
        ref["item_slots"](old, rows, a)
        new.item_slots(rows, b)
        assert np.array_equal(a, b), trial
        seen["repair"] += bool(getattr(new, "repair_ids", None))
        seen["burrows"] += bool(getattr(new, "burrows_safe", None))
        seen["spells"] += int(not b[2]) + int(not b[3])
        seen["harvest"] += int(not b[1])
    assert all(v > 10 for v in seen.values()), seen  # (each branch taken, and not always)


def test_spells_and_mines_in_c_and_python(kinds):
    """featnative.spell_any and mine_counts against spell_targets and mine_workers (and spell_any's Python pass)."""
    if featnative.lib() is None:
        pytest.skip("no C library")
    rng = np.random.default_rng(7)
    mined = code("ugol")
    for trial in range(400):
        player = int(rng.integers(2))
        rows = _rows(rng, int(rng.integers(1, 200)), player)
        owner = type("Owner", (), {})()
        got = fx.spell_any(rows, player, owner)
        assert got == fx.spell_any(rows, player), trial
        for o in fx.FUTILE_SPELLS:
            want = bool((fx.spell_targets(o, rows, player) & ~((rows[:, fx.C_TYPE] == fx.SPELL_CASTERS[o])
                                                               & (rows[:, fx.C_OWNER] == player))).any())
            assert got[o] == want, (trial, o)
        own = fx.own_alive(rows, player)
        mines = own[(own[:, fx.C_TYPE] == mined) & (own[:, fx.C_FLAGS] & fx.CONSTRUCTING == 0)]
        want = [(int(m[fx.C_ID]), len(fx.mine_workers(own, m))) for m in mines]
        assert featnative.mine_counts(rows, player, mined, fx._MINING, fx.MINE_NEAR, owner) == want, trial


def test_last_order_features_as_before():
    ref = _ref(REF_LAST, fx)
    rng = np.random.default_rng(3)
    for trial in range(400):
        n = int(rng.integers(1, 90))
        n_own = int(rng.integers(0, n + 1))
        sel = np.zeros((n, fx.ROW_COLS), np.int64)
        sel[:, fx.C_ID] = rng.permutation(np.arange(1, 3 * n + 1))[:n]
        sel[:, fx.C_X], sel[:, fx.C_Y] = rng.integers(-6000, 6000, n), rng.integers(-6000, 6000, n)
        ids = sel[:, fx.C_ID].tolist()
        view = type("View", (), {})()
        view.active_f, view.sign = fx.A_LAST + 5, float(rng.choice([-1.0, 1.0]))
        t = int(rng.integers(0, 3000))
        view.last_order = {}
        for u in rng.choice(np.arange(1, 3 * n + 1), int(rng.integers(0, 2 * n + 1))).tolist():
            kind = int(rng.integers(3))
            target = int(rng.choice(ids)) if kind == 2 and rng.random() < 0.7 else int(rng.integers(1, 9 * n))
            x, y = (float(rng.integers(-9000, 9000)), float(rng.integers(-9000, 9000))) if kind == 1 else (0.0, 0.0)
            view.last_order[u] = (kind, x, y, target if kind == 2 else 0, int(rng.integers(max(0, t - 80), t + 1)))
        f0 = rng.standard_normal((n, fx.A_LAST + 8)).astype(np.float32)
        a, b, c = f0.copy(), f0.copy(), f0.copy()
        ref["last_loop"](view, a, sel, ids, n_own, t)
        fx.View.last_features(view, b, sel, ids, n_own, t)  # (no records: the Python)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), trial
        view.last_packed = {u: fx.LAST_PACK.pack(u, *v) for u, v in view.last_order.items()}  # (as note_orders keeps)
        fx.View.last_features(view, c, sel, ids, n_own, t)  # (in C, when the library is there)
        assert np.array_equal(a.view(np.uint32), c.view(np.uint32)), trial


@pytest.fixture()
def game_tables():
    try:
        from warcraftsim.fullgame import fullvocab
        from warcraftsim.fullgame.herodata import hero_table, item_table
        from warcraftsim.fullgame.techtree import tree
        hero_table(), item_table(), tree(), fullvocab.tables(), fullvocab.universe("human")
    except Exception as e:  # noqa: BLE001 (the game's tables: not on every machine)
        pytest.skip(f"no game data: {e}")
    fx.set_neutral(1)
    fx.set_spells(1)
    fx.set_shop_items(1)
    fx.set_full_vocab(1)
    yield
    fx.set_neutral(0)
    fx.set_spells(0)
    fx.set_shop_items(0)
    fx.set_full_vocab(0)


def test_full_vocab_mask_as_before(game_tables):
    """Checks.mask whole (timed, toggles, night, partners, items and the shop, pick-up, the laboratory) against master's
    on random sides: types of every class's givers and partners, mana, cooldowns, autocast states, heroes' items."""
    from warcraftsim.fullgame import fullvocab
    ref = _ref(REF_MASK, fullvocab)
    vocab = {"types": [code("Hpal")], "current_orders": [], "upgrades": [],
             "orders": [[852092, fx.UNIT], [851983, fx.UNIT], [851986, fx.POINT], [851972, fx.IMMEDIATE]]}
    v = fullvocab.extend(vocab)
    ch = fullvocab.Checks(v)
    pool = sorted({g for c in ch.timed for g in ch.timed[c]} | {g for c in ch.toggle for g in ch.toggle[c][2]}
                  | {g for c in ch.partner for g in ch.partner[c]} | {p for c in ch.partner for p in ch.partner[c].values()}
                  | set(map(code, ("hfoo", "hpea", "ogru", "ewsp", "Hpal", "Hamg", "ngad", "ngme", "nmrk", "hvlt", "ngol"))))
    items = [0, 0, 0] + sorted(code(c) for c in ITEMS)
    rng = np.random.default_rng(11)
    n_orders = len(v["orders"]) + 1
    timed, toggles = sorted(ch.timed), sorted(ch.toggle)

    class View:
        step_seconds = 0.5

    for trial in range(400):
        view = View()
        view.player = player = int(rng.integers(2))
        n = int(rng.integers(1, 60))
        rows = np.zeros((n, fx.ROW_COLS), np.int64)
        rows[:, fx.C_ID] = np.arange(1, n + 1)
        rows[:, fx.C_TYPE] = rng.choice(pool, n)
        rows[:, fx.C_OWNER] = rng.choice([player, player, 1 - player, 15], n)
        rows[:, fx.C_FLAGS] = (np.where(rng.random(n) < 0.1, fx.DEAD, 0) | np.where(rng.random(n) < 0.3, fx.HERO, 0)
                               | np.where(rng.random(n) < 0.15, fx.CONSTRUCTING, 0))
        rows[:, fx.C_MANA] = rng.integers(0, 400, n)
        rows[:, fx.C_ITEM0:fx.C_ITEM0 + 6] = rng.choice(items, (n, 6))
        view.cooling = {(int(rng.integers(1, n + 1)), timed[int(rng.integers(len(timed)))]): int(rng.integers(0, 40))
                        for _ in range(int(rng.integers(0, 8)))}
        view.autocast = {(int(rng.integers(1, n + 1)), ch.toggle[int(c)][0]): bool(rng.integers(2))
                         for c in rng.choice(toggles, int(rng.integers(0, 10)))}
        view.ground = [(0, 0, 0, int(rng.integers(0, 4)), 0) for _ in range(int(rng.integers(0, 3)))]
        start = rng.random(n_orders) < 0.85
        t = int(rng.integers(0, 2000))
        a, b = start.copy(), start.copy()
        ref["mask"](ch, view, rows, a, t)
        ch.mask(view, rows, b, t)
        assert np.array_equal(a, b), trial


# ---- the remote games' wire: trajectories as pack_steps' arrays, float16 by F16C ----------------------------------------


def _chunks(rng, n_chunks: int = 6) -> list:
    """Trajectory's chunks as play_one makes them (raw arrays as the view and the network gave them, a memory state or
    none, labels on some steps, a game's end), through a fake queue."""
    from warcraftsim.fullgame.selfplay import Trajectory

    class Q:
        def __init__(self):
            self.got = []

        def put(self, msg):
            self.got.append(msg)

    q = Q()
    cfg = {"chunk": 64, "gamma": 0.9997, "lam": 0.95, "upgo": 1.0}
    G, NO, d, F = 40, 203, 16, fx.F
    for game in range(n_chunks // 2):
        tr = Trajectory(cfg, q, {"worker": 3, "opponent": "self"})
        h = None
        for t in range(int(rng.integers(70, 160))):
            n = int(rng.integers(5, 90))
            n_own = int(rng.integers(1, n + 1))
            o = min(n_own, fx.MAX_OWN)
            st = {"n": n, "n_own": n_own, "ent": rng.standard_normal((n, F)).astype(np.float32),
                  "type": rng.integers(0, 300, n), "cur": rng.integers(0, 400, n),
                  "glob": rng.standard_normal(G).astype(np.float32), "avail": rng.random(NO) < 0.5, "h": h}
            mem = np.asarray(rng.standard_normal(d), np.dtype("float32").newbyteorder("="))
            res = {"order": rng.integers(0, NO, o), "tgt": rng.integers(0, n, o), "bx": rng.integers(0, 22, o),
                   "by": rng.integers(0, 22, o), "logp": -rng.random(o).astype(np.float32), "value": float(rng.random()),
                   "version": int(rng.integers(0, 9)), "h": mem}
            tr.add(st, res)
            h = mem
            tr.steps[-1]["reward"] += float(rng.standard_normal())
            if rng.random() < 0.1:  # (a lesson's labels on the step: more keys, another layout)
                tr.steps[-1].update(y_order=rng.integers(0, NO, o), y_w=np.float32(0.5))
        tr.end(1.0)
    return [m for m in q.got if "steps" in m]


def _same(a, b, path="") -> list:
    """Where two learner-side values differ: types, dtypes, shapes, bytes, writability and order as well."""
    if type(a) is not type(b):
        return [f"{path}: {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, np.ndarray):
        ok = (a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
              and a.flags.writeable == b.flags.writeable and a.flags.c_contiguous == b.flags.c_contiguous)
        return [] if ok else [path]
    if isinstance(a, dict):
        if list(a) != list(b):
            return [f"{path}: keys"]
        return [x for k in a for x in _same(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in _same(u, v, f"{path}[{i}]")]
    return [] if (a == b or (a != a and b != b)) else [f"{path}: {a!r} vs {b!r}"]


def test_wire_chunks_reach_the_learner_the_same():
    """A remote actor's chunk on the wire as pack_steps' arrays (wire.PackedChunk at the gateway, pickled on as
    PackedSteps pickles): the learner gets the same steps as the step-by-step encoding gave it (and a local actor's)."""
    import pickle
    from multiprocessing.reduction import ForkingPickler

    from warcraftsim.fullgame import wire
    from warcraftsim.fullgame.selfplay import PackedSteps
    rng = np.random.default_rng(5)
    for m in _chunks(rng):
        ps = m["steps"]
        old = PackedSteps([dict(s) for s in list.__iter__(ps)], raw=ps.raw)
        local = PackedSteps([dict(s) for s in list.__iter__(ps)], raw=ps.raw)
        got = wire.unpack_value(wire.pack_value({"steps": ps, "info": m["info"]}))
        assert isinstance(got["steps"], wire.PackedChunk) and len(got["steps"]) == len(ps)
        new = pickle.loads(bytes(ForkingPickler.dumps(got)))
        before = pickle.loads(bytes(ForkingPickler.dumps(wire.unpack_value(wire.pack_value({"steps": list(old),
                                                                                           "info": m["info"]})))))
        assert type(new["steps"]) is list and not _same(before, new), _same(before, new)[:5]
        mine = pickle.loads(bytes(ForkingPickler.dumps({"steps": local})))["steps"]  # (a local actor's: dtypes alike)
        assert len(mine) == len(new["steps"]) and all(
            list(a) == list(b) and all(np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) else a[k] == b[k]
                                       for k in a) for a, b in zip(mine, new["steps"]))


def test_wire_chunk_refuses_what_does_not_add_up():
    from warcraftsim.fullgame import wire
    from warcraftsim.fullgame.selfplay import pack_steps
    rng = np.random.default_rng(6)
    m = _chunks(rng, 2)[0]
    layouts, which, cols = m["steps"].wire_packed()
    wire.PackedChunk((layouts, which, cols))
    bad = dict(cols)
    blob, lens, dt, tr = bad["ent"]
    bad["ent"] = (blob[:-2], lens, dt, tr)
    with pytest.raises(wire.WireError):
        wire.PackedChunk((layouts, which, bad))
    bad = dict(cols)
    bad["type"] = (bad["type"][0], bad["type"][1], "O", bad["type"][3])
    with pytest.raises(wire.WireError):
        wire.PackedChunk((layouts, which, bad))
    bad = dict(cols)
    del bad["n"]
    with pytest.raises(wire.WireError):
        wire.PackedChunk((layouts, which, bad))
    with pytest.raises(wire.WireError):
        wire.PackedChunk((layouts, bytes([len(layouts)]) + which[1:], cols))
    assert pack_steps  # (the packer the chunk came from)


def test_float16_both_ways_as_numpy():
    """featnative.to_f32_into (the gateway's shared rows) on every float16, and wire.pack_request's F16C float16: numpy's
    bits."""
    from warcraftsim.fullgame import wire
    if featnative.lib() is None:
        pytest.skip("no C library")
    h = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16)
    out = np.empty(65536, np.float32)
    assert featnative.to_f32_into(h, out)
    assert np.array_equal(out.view(np.uint32), h.astype(np.float32).view(np.uint32))
    for n in (1, 7, 9, 23):
        o = np.empty((n, 1), np.float32)
        assert featnative.to_f32_into(h[3000:3000 + n].reshape(n, 1).copy(), o)
        assert np.array_equal(o.view(np.uint32), h[3000:3000 + n].reshape(n, 1).astype(np.float32).view(np.uint32))
    rng = np.random.default_rng(8)
    st = {"n": 37, "n_own": 9, "ent": (rng.standard_normal((40, fx.F)) * 300).astype(np.float32),
          "type": rng.integers(0, 150, 40), "cur": rng.integers(0, 400, 40), "glob": rng.standard_normal(30).astype(np.float32),
          "avail": rng.random(99) < 0.5, "h": rng.standard_normal(16).astype(np.float32)}
    st["ent"][0, :4] = [np.inf, -np.inf, np.nan, 1e6]
    got = wire.unpack_request(wire.pack_request([("current", st)]))[0][1]["ent"]
    assert np.array_equal(got.view(np.uint16), st["ent"][:37].astype(np.float16).view(np.uint16))

