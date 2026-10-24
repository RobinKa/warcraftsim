"""The entity features in C (native/w3feat.c, loaded with ctypes): features.Encoder.entities' selection and columns,
and View.priv's per-side block, in one pass over a step's rows. The numpy code stays the reference (and the path
for the remembered enemies of --inputs 3 and the privileged view); WARCRAFTSIM_PY_FEATURES=1 makes it the only one.
scripts/bench_view.py --compare checks both give the same features."""

from __future__ import annotations

import ctypes
import fcntl
import os
import subprocess

import numpy as np

from .. import paths

SRC = paths.REPO_ROOT / "native" / "w3feat.c"
LIB = paths.REPO_ROOT / "build" / "native" / "libw3feat.so"
_lib = None


def _build() -> None:
    LIB.parent.mkdir(parents=True, exist_ok=True)
    with open(LIB.parent / ".build.lock", "w") as lock:  # (one build at a time: actor processes start together)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if LIB.exists() and LIB.stat().st_mtime >= SRC.stat().st_mtime:
            return
        tmp = LIB.with_suffix(".so.tmp")
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", str(tmp), str(SRC), "-lm"], check=True)
        tmp.replace(LIB)


def lib():
    """The library (built when missing or older than its source), or None (no compiler, or turned off)."""
    global _lib
    if _lib is None:
        if os.environ.get("WARCRAFTSIM_PY_FEATURES") == "1":
            _lib = False
            return None
        try:
            if not LIB.exists() or LIB.stat().st_mtime < SRC.stat().st_mtime:
                _build()
            L = ctypes.CDLL(str(LIB))
        except (OSError, subprocess.CalledProcessError):
            _lib = False
            return None
        L.w3f_entities.restype = ctypes.c_int
        P, I, D = ctypes.c_void_p, ctypes.c_int, ctypes.c_double  # (raw pointers: ndpointer's checks per argument had
        L.w3f_entities.argtypes = [P, I, I, I, D, P, P, I, P, I, I, I, I, I, I,  # cost more than the C itself)
                                   P, P, I, P, I, P, I, P, P, I, P, P, I, P, P, P, P, P]
        L.w3f_priv.restype = None
        L.w3f_priv.argtypes = [P, I, I, I, D, P, P]
        L.w3f_to_f16.restype = I
        L.w3f_to_f16.argtypes = [P, P, I]
        _lib = L
    return _lib or None


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
