"""The entity features in C (native/w3feat.c, loaded with ctypes): features.Encoder.entities' selection and columns,
and View.priv's per-side block, in one pass over a step's rows. The numpy code stays the reference (and the path
for the remembered enemies of --inputs 3 and the privileged view); WARCRAFTSIM_PY_FEATURES=1 makes it the only one.
scripts/bench_view.py --compare checks both give the same features."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

from .. import native, paths

SRC = paths.REPO_ROOT / "native" / "w3feat.c"
LIB = native.lib_path(SRC, "w3feat")
_lib = None


def lib():
    """The library (built when missing: its name holds its source's hash), or None (no compiler, or turned off)."""
    global _lib
    if _lib is None:
        if os.environ.get("WARCRAFTSIM_PY_FEATURES") == "1":
            _lib = False
            return None
        try:
            if not LIB.exists():
                native.build_lib(SRC, LIB, ["-lm"])
            L = ctypes.CDLL(str(LIB))
        except (OSError, subprocess.CalledProcessError):
            _lib = False
            return None
        L.w3f_entities.restype = ctypes.c_int
        P, I, D, Q = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int64  # (raw pointers: ndpointer's
        L.w3f_entities.argtypes = [P, I, I, I, D, P, P, I, P, I, I, I, I, I, I,  # checks per argument had cost more than the C)
                                   P, P, I, P, I, P, I, P, P, I, P, P, I, P, P, P, P, P]
        L.w3f_priv.restype = None
        L.w3f_priv.argtypes = [P, I, I, I, D, P, P]
        L.w3f_to_f16.restype = I
        L.w3f_to_f16.argtypes = [P, P, I]
        L.w3f_view.restype = I
        L.w3f_view.argtypes = [P, I, I, I, D, P, P, I, I, I, I, P, P, I, I, P, P, P, P]
        L.w3f_side.restype = I
        L.w3f_side.argtypes = [P, I, I, I, P]
        L.w3f_entangle.restype = I
        L.w3f_entangle.argtypes = [P, I, I, I, P, I, Q, Q, D, D, P]
        L.w3f_natural.restype = None
        L.w3f_natural.argtypes = [P, I, I, D, D, P, I, P, I, D, D, D, P]
        L.w3f_afford.restype = None
        L.w3f_afford.argtypes = [P, I, Q, Q, Q, Q, I, Q, Q, Q, Q, P]
        _lib = L
    return _lib or None


_from_buffer, _addressof = ctypes.c_char.from_buffer, ctypes.addressof


def addr(a: np.ndarray) -> int:
    """A C-contiguous array's address: of a ctypes char over its buffer when it is writable and not empty (numpy's
    .ctypes.data made a Python object of its own each time, ~3x the instructions), else .ctypes.data."""
    try:
        return _addressof(_from_buffer(a))
    except (TypeError, ValueError, BufferError):
        return a.ctypes.data


_ROWS: tuple = (None, 0)  # (the last rows given and their address: a step's calls share them. One tuple, replaced
# whole: an actor's game threads share it)


def _rows_addr(rows: np.ndarray) -> int:
    global _ROWS
    got = _ROWS
    if got[0] is not rows:
        got = _ROWS = (rows, addr(rows))
    return got[1]


T_N = 29  # (w3f_view's tables: the T_ entries)


def view_tables(enc, fx) -> tuple | None:
    """w3f_view's fixed tables for this encoder and the features' module as they are (kept on the encoder with their
    addresses; made again when the map's geometry or the inputs change): None when a hero ability has more mana
    costs than the table holds (the Python hero block then)."""
    key = (fx.STARTS, fx.F, fx.NEUTRAL, fx.MAX_OWN, fx.MAX_ENT)
    got = getattr(enc, "_view_tables", None)
    if got is not None and got[0] == key:
        return got[1]
    arrs = [np.ascontiguousarray(np.sort(fx.PASSIVE_IDS).astype(np.int64)),
            np.ascontiguousarray(np.sort(fx.SHOP_IDS).astype(np.int64)),
            np.ascontiguousarray(enc._type_keys, np.int64), np.ascontiguousarray(enc._type_vals, np.int64),
            np.ascontiguousarray(enc._cur_keys, np.int64), np.ascontiguousarray(enc._cur_vals, np.int64),
            geometry(fx)]
    do_hero = fx.F > fx.F_BASE
    htypes, hslots, itemk, itemv, maxl = np.zeros(1, np.int64), np.zeros(1), np.zeros(1, np.int64), np.zeros(4), 1
    if do_hero:
        from .herodata import hero_table, item_table
        table, items = hero_table(), item_table()
        maxl = max([len(a.mana) for slots in table.values() for a in slots if a is not None] + [1])
        if maxl > 64:
            enc._view_tables = (key, None)
            return None
        ht = sorted(table)
        hs = np.zeros((max(len(ht), 1), 4, 6 + maxl))
        for i, t in enumerate(ht):
            for s, a in enumerate(tuple(table[t])[:4]):
                if a is not None:
                    hs[i, s, :6] = (1, 1 if a.order else 0, a.levels, a.req_level, a.level_skip, len(a.mana))
                    hs[i, s, 6:6 + len(a.mana)] = a.mana
        htypes = np.asarray(ht or [0], np.int64)
        hslots = np.ascontiguousarray(hs.reshape(-1))
        ik = sorted(items)
        itemk = np.asarray(ik or [0], np.int64)
        itemv = np.ascontiguousarray(np.asarray([items[k] for k in ik] or [(0, 0, 0, 0)], np.float64).reshape(-1))
    arrs += [htypes, hslots, itemk, itemv]
    p = [a.ctypes.data for a in arrs]
    t = np.zeros(T_N + 3, np.int64)
    t[:T_N] = [p[0], len(arrs[0]), p[1], len(arrs[1]), p[2], p[3], len(arrs[2]), p[4], p[5], len(arrs[4]), p[6],
               fx.MAX_OWN, fx.MAX_ENT, fx.F, fx.A_POS, fx.F_BASE, fx.A_HPD, fx.A_HERO, fx.A_PROD,
               p[7], len(htypes) if do_hero and len(hero_table()) else 0, p[8], p[9],
               len(itemk) if do_hero and len(item_table()) else 0, p[10], int(bool(fx.NEUTRAL)), int(do_hero),
               fx.ROW_COLS, maxl]
    got = (arrs, t, t.ctypes.data)
    enc._view_tables = (key, got)
    return got


def view_step(enc, fx, view, rows: np.ndarray, player: int, sign: float, production, lumber, priv: bool):
    """w3f_view for View.step (`view`: its hp_last, the last step's sorted ids and hit points, kept with their addresses
    in view._hp_addr): (sel, n_own, f, types, cur, the packed list (a Python list), privileged numbers (float32 [PRIV],
    the per-side block filled) or None); None without its tables. view.hp_last becomes this step's."""
    tabs = view_tables(enc, fx)
    if tabs is None:
        return None
    L = lib()
    rows = np.ascontiguousarray(rows, np.int64)
    nrows, ncols = rows.shape
    queued, busy, making = production
    q = sorted(queued.items())
    b, lm = sorted(busy), sorted(lumber)
    m = sorted((k, v[0] + 1) for k, v in making.items()) if making and fx.F > fx.F_BASE else ()
    keys = np.array([*(k for k, _ in q), *(v for _, v in q), *b, *lm, *(k for k, _ in m), *(v for _, v in m), 0],
                    np.int64)
    ka = addr(keys)
    last = view.hp_last
    got = getattr(view, "_hp_addr", None)
    if got is not None and got[0] is last:
        pa, ha = got[1], got[2]
    else:  # (made by another path: contiguous copies, kept with it)
        pv, ph = np.ascontiguousarray(last[0], np.int64), np.ascontiguousarray(last[1], np.int64)
        view._hp_keep = (pv, ph)
        pa, ha = (addr(pv), addr(ph)) if len(pv) else (0, 0)
    E = fx.MAX_ENT
    cap = min(E, nrows)
    out = np.empty(8 * E + 16, np.int64)
    f = np.zeros((cap + 1, fx.F), np.float32)  # (its last row: the privileged numbers)
    sel = np.empty((max(cap, 1), ncols), np.int64)
    fa, oa = addr(f), addr(out)
    nq, nb, nl, nm = len(q), len(b), len(lm), len(m)
    k = L.w3f_view(_rows_addr(rows), nrows, ncols, int(player), float(sign), tabs[2], ka, nq, nb, nl, nm, pa, ha,
                   len(last[0]), int(view.active_f), oa, fa, addr(sel), fa + 4 * cap * fx.F if priv else 0)
    if k < 0:
        raise MemoryError("w3f_view")
    pk = out[5 * E:5 * E + k].tolist()
    n = pk[0]
    view.hp_last = hp = (out[3 * E:3 * E + n], out[4 * E:4 * E + n])
    view._hp_addr = (hp, oa + 24 * E, oa + 32 * E)
    return sel[:n], pk[1], f[:n], out[E:E + n], out[2 * E:2 * E + n], pk, f[cap, :fx.PRIV] if priv else None


def side(rows: np.ndarray, player: int, fresh: bool = False) -> list:
    """w3f_side: one player's masks of a step's rows, as a list: 8 counts, then the segments they count (its living
    buildings' ids, its living units' ids, its dead heroes' (type, id, level), its heroes' types, its finished
    buildings' types, (its army's size), its plain living units' types, the rows of its living heroes with a skill
    point). Read only. Kept per player with the rows
    asked: the next ask of the same rows gets it, unless `fresh` (View.step's: a step's rows are new, but a caller
    may change rows in place between steps; the target builder asks after it, on the step's rows)."""
    got = _SIDE.get(player)
    if not fresh and got is not None and got[0] is rows:
        return got[1]
    given = rows
    rows = np.ascontiguousarray(rows, np.int64)
    out = np.empty(9 + 8 * len(rows), np.int64)
    k = lib().w3f_side(_rows_addr(rows), len(rows), rows.shape[1], int(player), addr(out))
    res = out[:k].tolist()
    _SIDE[player] = (given, res)  # (a tuple replaced whole: an actor's game threads share it)
    return res


_SIDE: dict = {}  # player -> (the last rows asked, side's list)


_NAT: tuple = (None, None, None)  # (the last rows and start asked, and natural's answer: View.step's site and camp and
# the target builder's ask on a step's rows; one tuple, replaced whole: an actor's game threads share it)
_CODES: dict = {}


def natural(rows: np.ndarray, start: tuple, mines: np.ndarray, holds: np.ndarray, from_start: float, taken_near: float,
            camp_near: float, fresh: bool = False) -> list | None:
    """w3f_natural: [the natural's row (-1 none, -2 a tie: None is returned, the numpy code decides), its id, x, y,
    whether a hall stands at it, whether creeps do], kept with the rows and start asked for the next ask of the same
    (unless `fresh`: View.step's, as side's). None without the library."""
    global _NAT
    got = _NAT
    if not fresh and got[0] is rows and got[1] == start:
        return got[2]
    L = lib()
    if L is None:
        return None
    rows = np.ascontiguousarray(rows, np.int64)
    c = _CODES.get(id(mines))
    if c is None or c[0] is not mines or c[2] is not holds:
        m, h = np.ascontiguousarray(mines, np.int64), np.ascontiguousarray(holds, np.int64)
        c = _CODES[id(mines)] = (mines, (m, h, m.ctypes.data, h.ctypes.data), holds)
    m, h, ma, ha = c[1]
    out = np.empty(6, np.int64)
    L.w3f_natural(_rows_addr(rows), len(rows), rows.shape[1], float(start[0]), float(start[1]), ma, len(m), ha, len(h),
                  float(from_start), float(taken_near), float(camp_near), addr(out))
    res = out.tolist()
    res = None if res[0] == -2 else res
    _NAT = (rows, start, res)
    return res


def entangle(rows: np.ndarray, player: int, trees: np.ndarray, ngol: int, egol: int, near: float,
             held_near: float) -> list | None:
    """w3f_entangle: [(tree id, mine id)] (play.BCAgent._entangle's distances, its trees in order; none without a
    mine)."""
    rows = np.ascontiguousarray(rows, np.int64)
    out = np.empty(2 * len(rows) + 2, np.int64)
    c = _CODES.get(id(trees))
    if c is None or c[0] is not trees:
        t = np.ascontiguousarray(trees, np.int64)
        c = _CODES[id(trees)] = (trees, (t, t.ctypes.data), None)
    t, ta = c[1]
    n = lib().w3f_entangle(_rows_addr(rows), len(rows), rows.shape[1], int(player), ta, len(t), int(ngol), int(egol),
                           float(near), float(held_near), addr(out))
    if n < 0:  # (no mine)
        return []
    v = out[:2 * n].tolist()
    return list(zip(v[0::2], v[1::2]))


def afford(costs: np.ndarray, gold: int, lumber: int, fu: int, fc: int, kept: list) -> np.ndarray | None:
    """features.affordable in C (costs: int64 [n, 3], C-contiguous; kept: its (class, gold, lumber)s), None when it
    can't (other costs, a kept class out of range)."""
    L = lib()
    n = len(costs)
    if (L is None or costs.dtype != np.int64 or costs.ndim != 2 or costs.shape[1] != 3 or not costs.flags.c_contiguous
            or not n or any(not 0 <= r[0] < n for r in kept) or len(kept) > 2):
        return None
    out = np.empty(n, np.bool_)
    k1 = kept[0][0] if kept else 0
    k2 = kept[1][0] if len(kept) > 1 else 0
    rg, rl = (sum(r[1] for r in kept), sum(r[2] for r in kept)) if kept else (0, 0)
    L.w3f_afford(addr(costs), n, gold, lumber, fu, fc, len(kept), k1, k2, rg, rl, addr(out))
    return out


def _keyed(queued, busy, lumber) -> tuple[np.ndarray, int, int, int]:
    """The production's queued counts (a dict; a set: 1 each), its busy buildings and the lumber harvesters (sets or
    dicts' keys) as one int64 array: queued's sorted keys, their values, busy's sorted, lumber's sorted (and a 0, so
    that it is never empty); their lengths. (An array, a sort and a pointer each had been ~30 us a step.)"""
    q = sorted(queued.items()) if isinstance(queued, dict) else [(k, 1) for k in sorted(queued)]
    b, lm = sorted(busy), sorted(lumber)
    return np.array([*(k for k, _ in q), *(v for _, v in q), *b, *lm, 0], np.int64), len(q), len(b), len(lm)


def geometry(fx) -> np.ndarray:
    (ax, ay), (bx, by) = fx.STARTS
    return np.array([ax, ay, bx, by, fx.DUEL_D, fx.NEAR, fx.SCALE, fx.ANCHOR_X], np.float64)


def entities(enc, fx, rows: np.ndarray, player: int, sign: float, production, lumber):
    """Encoder.entities' (sel, n_own, f, types, cur) without the hero and production-kind blocks (the caller's)."""
    L = lib()
    rows = np.ascontiguousarray(rows, np.int64)
    queued, busy = (production[0], production[1]) if production is not None else ({}, set())
    keys, nq, nb, nl = _keyed(queued, busy, lumber)
    qk = keys.ctypes.data
    qv, bk, lk = qk + 8 * nq, qk + 16 * nq, qk + 8 * (2 * nq + nb)
    if getattr(enc, "_native_key", None) != (fx.STARTS, fx.F):  # (the tables, kept with their pointers)
        arrs = (np.ascontiguousarray(np.sort(fx.PASSIVE_IDS).astype(np.int64)),
                np.ascontiguousarray(np.sort(fx.SHOP_IDS).astype(np.int64)),
                np.ascontiguousarray(enc._type_keys, np.int64), np.ascontiguousarray(enc._type_vals, np.int64),
                np.ascontiguousarray(enc._cur_keys, np.int64), np.ascontiguousarray(enc._cur_vals, np.int64),
                geometry(fx))
        enc._native_tables = (arrs, [a.ctypes.data for a in arrs])
        enc._native_key = (fx.STARTS, fx.F)
    arrs, (passive, shops, tk, tv, ck, cv, geo) = enc._native_tables
    E = fx.MAX_ENT
    out = np.zeros(3 * E + 5, np.int64)  # (the indices, types, current orders and counts: one array, one pointer)
    o = out.ctypes.data
    f = np.zeros((E, fx.F), np.float32)
    n = L.w3f_entities(rows.ctypes.data, len(rows), rows.shape[1], int(player), float(sign), geo,
                       passive, len(arrs[0]), shops, len(arrs[1]), int(bool(fx.NEUTRAL)),
                       fx.MAX_OWN, E, fx.F, fx.A_POS,
                       qk, qv, nq, bk, nb, lk, nl,
                       tk, tv, len(arrs[2]), ck, cv, len(arrs[4]),
                       o, o + 24 * E, f.ctypes.data, o + 8 * E, o + 16 * E)
    if n < 0:
        raise MemoryError("w3f_entities")
    n, n_own, n_enemy, n_neutral, own_all = out[3 * E:].tolist()
    idx, types, cur = out[:E], out[E:2 * E], out[2 * E:3 * E]
    sel = rows[idx[:n]]
    enc.ghost0 = min(n, n_own + n_enemy + n_neutral)
    enc.counts = (own_all, n_own + n_enemy + n_neutral)
    return sel, n_own, f[:n], types[:n], cur[:n]


_GEO: list = [None, None, None]  # priv_sides' geometry: [what it was made from, the array, its pointer]


def priv_sides(fx, rows: np.ndarray, player: int, sign: float, out: np.ndarray) -> None:
    """View.priv's per-side block into `out` (float32, PRIV long)."""
    rows = np.ascontiguousarray(rows, np.int64)
    key = (fx.STARTS, fx.DUEL_D, fx.NEAR, fx.SCALE, fx.ANCHOR_X)
    if _GEO[0] != key:  # (made again only when the map's geometry changes: an array a step before)
        geo = geometry(fx)
        _GEO[:] = [key, geo, geo.ctypes.data]
    lib().w3f_priv(rows.ctypes.data, len(rows), rows.shape[1], int(player), float(sign), _GEO[2], out.ctypes.data)


def to_f16(a: np.ndarray) -> np.ndarray:
    """A float32 array as float16 by the CPU's F16C instruction, the same values as numpy's astype (whose conversion
    is in software: ~40 us a step's features); numpy's without the library or the instruction."""
    L = lib()
    a = np.ascontiguousarray(a, np.float32)
    out = np.empty(a.shape, np.float16)
    if L is None or not a.size or not L.w3f_to_f16(a.ctypes.data, out.ctypes.data, a.size):
        return a.astype(np.float16)
    return out
