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
        _lib = L
    return _lib or None


def _keys(d) -> tuple[np.ndarray, np.ndarray, int]:
    """A dict (or a set: values 1) as sorted int64 keys and values."""
    if not d:
        z = np.zeros(1, np.int64)
        return z, z, 0
    if isinstance(d, dict):
        k = np.fromiter(d.keys(), np.int64, len(d))
        v = np.fromiter(d.values(), np.int64, len(d))
    else:
        k = np.fromiter(d, np.int64, len(d))
        v = np.ones(len(d), np.int64)
    o = np.argsort(k)
    return np.ascontiguousarray(k[o]), np.ascontiguousarray(v[o]), len(d)


def geometry(fx) -> np.ndarray:
    (ax, ay), (bx, by) = fx.STARTS
    return np.array([ax, ay, bx, by, fx.DUEL_D, fx.NEAR, fx.SCALE, fx.ANCHOR_X], np.float64)


def entities(enc, fx, rows: np.ndarray, player: int, sign: float, production, lumber):
    """Encoder.entities' (sel, n_own, f, types, cur) without the hero and production-kind blocks (the caller's)."""
    L = lib()
    rows = np.ascontiguousarray(rows, np.int64)
    queued, busy = (production[0], production[1]) if production is not None else ({}, set())
    qk, qv, nq = _keys(queued)
    bk, _, nb = _keys(busy)
    lk, _, nl = _keys(lumber)
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
    idx = np.empty(E, np.int64)
    counts = np.zeros(5, np.int64)
    f = np.zeros((E, fx.F), np.float32)
    types = np.zeros(E, np.int64)
    cur = np.zeros(E, np.int64)
    n = L.w3f_entities(rows.ctypes.data, len(rows), rows.shape[1], int(player), float(sign), geo,
                       passive, len(arrs[0]), shops, len(arrs[1]), int(bool(fx.NEUTRAL)),
                       fx.MAX_OWN, E, fx.F, fx.A_POS,
                       qk.ctypes.data, qv.ctypes.data, nq, bk.ctypes.data, nb, lk.ctypes.data, nl,
                       tk, tv, len(arrs[2]), ck, cv, len(arrs[4]),
                       idx.ctypes.data, counts.ctypes.data, f.ctypes.data, types.ctypes.data, cur.ctypes.data)
    if n < 0:
        raise MemoryError("w3f_entities")
    n, n_own, n_enemy, n_neutral, own_all = (int(x) for x in counts)
    sel = rows[idx[:n]]
    enc.ghost0 = min(n, n_own + n_enemy + n_neutral)
    enc.counts = (own_all, n_own + n_enemy + n_neutral)
    return sel, n_own, f[:n], types[:n], cur[:n]


def priv_sides(fx, rows: np.ndarray, player: int, sign: float, out: np.ndarray) -> None:
    """View.priv's per-side block into `out` (float32, PRIV long)."""
    rows = np.ascontiguousarray(rows, np.int64)
    geo = geometry(fx)
    lib().w3f_priv(rows.ctypes.data, len(rows), rows.shape[1], int(player), float(sign), geo.ctypes.data, out.ctypes.data)
