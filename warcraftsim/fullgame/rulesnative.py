"""The rules' per-step facts in C (native/w3rules.c, loaded with ctypes): what the heuristic teacher
(teacher.Teacher.commands, its expansion lesson) and the strategy modes' tracker (modes.Tracker.step) read from a
step's rows, in a pass or two over them (their numpy calls had cost ~700k instructions a side-step of an actor's
~2.5M). The numpy code in those modules stays the reference and the path without the library; as featnative's,
WARCRAFTSIM_PY_FEATURES=1 makes it the only one.

Each C function takes one argument, an int64 array of its caller's (a ctypes argument had cost ~1,600
instructions): the head (A_* slots) the address of the rows' descriptor (their address and shape: made once per
rows object), the caller's constants and its tables' addresses, then the output (array.array: an item or a short
slice's list cheaper than numpy's)."""

from __future__ import annotations

import array
import ctypes
import os
import subprocess

import numpy as np

from .. import native, paths
from .featnative import addr

SRC = paths.REPO_ROOT / "native" / "w3rules.c"
LIB = native.lib_path(SRC, "w3rules")
_lib = None
# type classes (w3rules.c's K_*): bits of a type code in a table()
K_ALTAR, K_TIER2, K_TIER3, K_WORKER, K_MINE, K_HOLD, K_OWN_MINE = 1, 2, 4, 8, 16, 32, 64
TH, EH, MH = 12, 19, 8  # (the outputs' heads: w3r_teach's, w3r_expansion's, w3r_modes')
# the argument's head (w3rules.c's A_*): the rows' descriptor, the side, a table (keys, values, length), the doubles'
# address, then the function's own slots
A_ROWS, A_SIDE, A_KEYS, A_VALS, A_NKEYS, A_DOUBLES = 0, 3, 4, 5, 6, 7
A_X = tuple(range(8, 18))
HEAD = 24


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
        for f in (L.w3r_teach, L.w3r_expansion, L.w3r_modes):
            f.restype, f.argtypes = ctypes.c_int, [ctypes.c_void_p]
        _lib = L
    return _lib or None


def table(keys, vals=None) -> tuple:
    """Sorted int64 keys (and their values) for the C's lookups, kept with their addresses: (keys, values, the keys'
    address, the values', how many). `vals`: the values of `keys` (None: `keys` is a {code: bits} dict)."""
    items = sorted(keys.items()) if vals is None else sorted(zip(keys, vals))
    k = np.ascontiguousarray([c for c, _ in items] or [0], np.int64)
    v = np.ascontiguousarray([b for _, b in items] or [0], np.int64)
    return k, v, k.ctypes.data, v.ctypes.data, len(items)


_I64 = np.dtype(np.int64)
_LAST: tuple = (None, 0, 0, None, None)  # (the last rows given, their descriptor's address, how many, the int64
# C-contiguous rows and the descriptor kept: one tuple, replaced whole; an actor's game threads share it)


def _rows(rows: np.ndarray) -> tuple:
    global _LAST
    got = _LAST
    if got[0] is rows:
        return got
    r = rows if rows.dtype is _I64 and rows.flags.c_contiguous else np.ascontiguousarray(rows, np.int64)
    n, nc = r.shape
    desc = array.array("q", (addr(r), n, nc))
    got = _LAST = (rows, desc.buffer_info()[0], n, r, desc)
    return got


def _arg(cap: int, slots: dict, doubles=(), keep=None) -> tuple:
    """A caller's argument array: its head's constants (`slots`), the doubles, room for `cap` int64 of output:
    (the array, its address, its room, `keep` (what the addresses point into), the doubles)."""
    cap = max(cap, 1024)
    a = array.array("q", bytes(8 * (HEAD + cap)))
    d = array.array("d", [float(x) for x in doubles] + [0.0] * 8)
    for k, v in slots.items():
        a[k] = v
    a[A_DOUBLES] = d.buffer_info()[0]
    return a, a.buffer_info()[0], cap, keep, d


HF = 10  # (w3r_teach's fields of a living hero: its row, id, hit points, maximum, 6 items)


def teach(holder, rows: np.ndarray, side: int, tavern: int, merchant: int, cls: tuple, mix: tuple) -> list | None:
    """w3r_teach: [n_heroes, n_live, tavern row, merchant row, n_done, n_army, done classes, n_altars, finished altar
    row, worker nearest the tavern row, n_alive, n_mix], then the heroes' types, the living heroes' HF fields, each
    finished building type's first (type, id), the army's count of each of the mix's types (`mix`: (an int64 array of
    them, its address, how many)); None without the library."""
    L = _lib or lib()
    if L is None:
        return None
    r = _rows(rows)
    need = TH + (HF + 4) * r[2] + mix[2]
    c = holder.__dict__.get("_rules_teach")
    if (c is None or c[2] < need or c[3][0] is not cls or c[3][1] is not mix
            or c[3][2:] != (side, tavern, merchant)):  # (cls and mix: the tables, kept, and their addresses)
        c = holder._rules_teach = _arg(need, {A_SIDE: side, A_X[0]: tavern, A_X[1]: merchant, A_X[2]: mix[1],
                                              A_X[3]: mix[2], A_KEYS: cls[2], A_VALS: cls[3], A_NKEYS: cls[4]},
                                       keep=(cls, mix, side, tavern, merchant))
    a = c[0]
    a[A_ROWS] = r[1]
    k = L.w3r_teach(c[1])
    return a[HEAD:HEAD + k].tolist()


def expansion(holder, rows: np.ndarray, side: int, sx: float, sy: float, cls: tuple, from_start: float,
              taken_near: float, worker_near: float, camp_near: float, nat: int = -1) -> list | None:
    """w3r_expansion's facts (its comment; `nat`: the natural's row, the caller's choice, or -1), None without the
    library."""
    L = _lib or lib()
    if L is None:
        return None
    r = _rows(rows)
    key = (side, cls[2], cls[3], cls[4], from_start, taken_near, worker_near, camp_near)
    c = holder.__dict__.get("_rules_expansion")
    if c is None or c[2] < EH + 3 * r[2] or c[3] != key:
        c = holder._rules_expansion = _arg(EH + 3 * r[2], {A_SIDE: side, A_KEYS: cls[2], A_VALS: cls[3], A_NKEYS: cls[4]},
                                           (0.0, 0.0, from_start, taken_near, worker_near, camp_near), key)
    a, d = c[0], c[4]
    a[A_ROWS], a[A_X[0]] = r[1], nat
    d[0], d[1] = sx, sy
    k = L.w3r_expansion(c[1])
    if k < 0:
        raise IndexError(f"w3r_expansion: row {nat}")
    return a[HEAD:HEAD + k].tolist()


def modes(holder, rows: np.ndarray, need: int, setup) -> list | None:
    """w3r_modes' measures (its comment), None without the library. `setup()`: the tracker's constants, (side, the
    enemy's start, near2, scout2, its own start or None, far2, at2, the tables of the halls, gold mines, workers and
    food), asked once (they are fixed)."""
    L = _lib or lib()
    if L is None:
        return None
    r = _rows(rows)
    c = holder.__dict__.get("_rules_modes")
    if c is None or c[2] < MH + 2 * r[2]:
        side, enemy, near2, scout2, own, far2, at2, halls, mines, workers, food = setup()
        ox, oy = own if own is not None else (0.0, 0.0)
        c = holder._rules_modes = _arg(
            MH + 2 * r[2], {A_SIDE: side, A_X[1]: int(own is not None), A_X[2]: at2, A_X[3]: halls[2], A_X[4]: halls[4],
                            A_X[5]: mines[2], A_X[6]: mines[4], A_X[7]: workers[2], A_X[8]: workers[4],
                            A_KEYS: food[2], A_VALS: food[3], A_NKEYS: food[4]},
            (enemy[0], enemy[1], near2, scout2, ox, oy, far2), (halls, mines, workers, food))
    a = c[0]
    a[A_ROWS], a[A_X[0]] = r[1], need
    k = L.w3r_modes(c[1])
    if k < 0:
        raise MemoryError("w3r_modes")
    return a[HEAD:HEAD + k].tolist()
