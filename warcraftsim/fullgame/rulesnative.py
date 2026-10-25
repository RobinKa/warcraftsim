"""The rules' per-step facts in C (native/w3rules.c, loaded with ctypes): what the heuristic teacher
(teacher.Teacher.commands, its expansion lesson) and the strategy modes' tracker (modes.Tracker.step) read from a
step's rows, in a pass or two over them (their numpy calls had cost ~700k instructions a side-step of an actor's
~2.5M). The numpy code in those modules stays the reference and the path without the library; as featnative's,
WARCRAFTSIM_PY_FEATURES=1 makes it the only one."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

from .. import native, paths
from .featnative import _rows_addr, addr

SRC = paths.REPO_ROOT / "native" / "w3rules.c"
LIB = native.lib_path(SRC, "w3rules")
_lib = None
# type classes (w3rules.c's K_*): bits of a type code in a classes() table
K_ALTAR, K_TIER2, K_TIER3, K_WORKER, K_MINE, K_HOLD, K_OWN_MINE = 1, 2, 4, 8, 16, 32, 64
TH, EH, MH = 12, 19, 8  # (the outputs' heads: w3r_teach's, w3r_expansion's, w3r_modes')


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
        P, I, D, Q = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int64
        L.w3r_teach.restype = I
        L.w3r_teach.argtypes = [P, I, I, I, Q, Q, P, P, I, P]
        L.w3r_expansion.restype = I
        L.w3r_expansion.argtypes = [P, I, I, I, D, D, P, P, I, D, D, D, D, P]
        L.w3r_modes.restype = I
        L.w3r_modes.argtypes = [P, I, I, I, D, D, D, D, I, D, D, D, Q, P, I, P, I, P, I, P, P, I, I, P]
        _lib = L
    return _lib or None


def table(keys, vals=None) -> tuple:
    """Sorted int64 keys (and their values) for the C's lookups, kept with their addresses: (keys, values, the keys'
    address, the values', how many). `vals`: a dict's values for its keys (None: `keys` is a {code: bits} dict)."""
    items = sorted(keys.items()) if vals is None else sorted(zip(keys, vals))
    k = np.ascontiguousarray([c for c, _ in items] or [0], np.int64)
    v = np.ascontiguousarray([b for _, b in items] or [0], np.int64)
    return k, v, k.ctypes.data, v.ctypes.data, len(items)


def _buf(holder, n: int) -> tuple:
    """`holder`'s output buffer of n int64 at least, with its address (one per teacher or tracker: a game's thread
    alone uses it)."""
    b = holder.__dict__.get("_rules_buf")
    if b is None or len(b[0]) < n:
        a = np.empty(max(n, 1024), np.int64)
        b = holder._rules_buf = (a, addr(a))
    return b


def _rows(rows: np.ndarray) -> np.ndarray:
    return rows if rows.dtype == np.int64 and rows.flags.c_contiguous else np.ascontiguousarray(rows, np.int64)


def teach(holder, rows: np.ndarray, side: int, tavern: int, merchant: int, cls: tuple) -> list | None:
    """w3r_teach: [n_heroes, n_live, tavern row, merchant row, n_done, n_army, done classes, n_altars, finished altar
    row, worker nearest the tavern row, n_alive, 0] then the heroes' types, the living heroes' rows, the finished
    buildings' (type, id)s, the army's types; None without the library."""
    L = lib()
    if L is None:
        return None
    rows = _rows(rows)
    n, nc = rows.shape
    buf, ba = _buf(holder, TH + 5 * n)
    k = L.w3r_teach(_rows_addr(rows), n, nc, side, tavern, merchant, cls[2], cls[3], cls[4], ba)
    return buf[:k].tolist()


def expansion(holder, rows: np.ndarray, side: int, sx: float, sy: float, cls: tuple, from_start: float,
              taken_near: float, worker_near: float, camp_near: float) -> list | None:
    """w3r_expansion's facts (its comment), None without the library."""
    L = lib()
    if L is None:
        return None
    rows = _rows(rows)
    n, nc = rows.shape
    buf, ba = _buf(holder, EH + n)
    k = L.w3r_expansion(_rows_addr(rows), n, nc, side, sx, sy, cls[2], cls[3], cls[4], from_start, taken_near,
                        worker_near, camp_near, ba)
    return buf[:k].tolist()


def modes(holder, rows: np.ndarray, side: int, enemy: tuple, near2: float, scout2: float, own: tuple | None,
          far2: float, at2: int, halls: tuple, mines: tuple, workers: tuple, food: tuple, need: int) -> list | None:
    """w3r_modes' measures (its comment), None without the library."""
    L = lib()
    if L is None:
        return None
    rows = _rows(rows)
    n, nc = rows.shape
    buf, ba = _buf(holder, MH + 2 * n)
    ox, oy = own if own is not None else (0.0, 0.0)
    k = L.w3r_modes(_rows_addr(rows), n, nc, side, enemy[0], enemy[1], near2, scout2, own is not None, ox, oy, far2,
                    at2, halls[2], halls[4], mines[2], mines[4], workers[2], workers[4], food[2], food[3], food[4],
                    need, ba)
    if k < 0:
        raise MemoryError("w3r_modes")
    return buf[:k].tolist()
