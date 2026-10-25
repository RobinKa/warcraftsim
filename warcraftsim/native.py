"""Observation parsing in C (native/w3obs.c, loaded with ctypes): the harness's token stream
parsed and merged into a unit table without Python objects per unit, and without the GIL (ctypes
releases it during the call, so the game threads of one process parse at the same time).

GameInstance uses it with GameSetup.native_obs: the Observation it returns then has
`unit_array` (int32 [n, UCOLS]: the columns below, the merged table in first-seen order) and a
lazy `units` that makes Unit objects only when something reads them. The rules are those of
protocol.parse_tokens and merge_observation (tests/unit/test_native.py compares the two).

Built on first use (gcc), into build/native/.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import paths
from .protocol import (HERO_ABILITY_SLOTS, PROTOCOL_VERSION, Destructable, Event, EventKind, IssuedOrder,
                       Observation, PlayerState, ProtocolError, Race, Result, Unit, UnitFlags)

SRC = paths.REPO_ROOT / "native" / "w3obs.c"


def lib_path(src, name: str):
    """A native library's file: its name holds its source's hash (a time check had kept a stale one where a copy
    keeps the files' times: rsync -a, git archive; audit 7)."""
    return paths.REPO_ROOT / "build" / "native" / f"lib{name}-{hashlib.sha1(src.read_bytes()).hexdigest()[:12]}.so"


def build_lib(src, lib, flags=()) -> None:
    """Builds `lib` from `src` (one build at a time: actor processes start together); the older builds go."""
    lib.parent.mkdir(parents=True, exist_ok=True)
    with open(lib.parent / ".build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if lib.exists():
            return
        tmp = lib.with_suffix(".so.tmp")
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", str(tmp), str(src), *flags], check=True)
        tmp.replace(lib)
        for old in lib.parent.glob(lib.name.rsplit("-", 1)[0] + "*.so"):  # (mapped by a live process: it keeps it)
            if old != lib:
                old.unlink(missing_ok=True)


LIB = lib_path(SRC, "w3obs")
# unit_array columns: the U record's fields, then the hero fields (0 for others), then whether they're set
(C_ID, C_TYPE, C_OWNER, C_X, C_Y, C_FACING, C_HP, C_MAXHP, C_MANA, C_MAXMANA, C_ORDER, C_FLAGS, C_VIS, C_RESOURCE,
 C_HLEVEL, C_HXP, C_SKILLPTS) = range(17)
C_ITEMS, C_ABILITIES, C_HERO = 17, 23, 31
UCOLS = 32
_H_SEQ, _H_MS, _H_OVER, _H_FULL, _H_VERSION, _H_DAMAGED, _H_ENDED, _H_CAMERA, _H_CAM_X, _H_CAM_Y, _H_NUNITS = range(11)
# w3obs_rows' tables' row widths: players, events, command results, issued orders, order ids, destructables, removed
# units, unit deltas, ground items (checked against the library's)
_WIDTHS = (14, 4, 1, 6, 1, 5, 1, UCOLS, 5)
_TABLES = 25  # (w3obs_export: the header's 16 ints and the 9 tables' row counts, then their rows)
_lib = None


def lib():
    global _lib
    if _lib is None:
        if not LIB.exists():
            build_lib(SRC, LIB)
        L = ctypes.CDLL(str(LIB))  # (CDLL: the GIL is released during calls)
        L.w3obs_new.restype = ctypes.c_void_p
        L.w3obs_free.argtypes = [ctypes.c_void_p]
        L.w3obs_parse.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        L.w3obs_merge.argtypes = [ctypes.c_void_p]
        L.w3obs_table.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        L.w3obs_header.restype = ctypes.POINTER(ctypes.c_int32)
        L.w3obs_header.argtypes = [ctypes.c_void_p]
        L.w3obs_rows.restype = ctypes.c_void_p
        L.w3obs_rows.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        L.w3obs_width.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.w3obs_export.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        L.w3obs_parse_export.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        if L.w3obs_ucols() != UCOLS:
            raise RuntimeError(f"{LIB}: {L.w3obs_ucols()} unit columns, expected {UCOLS}")
        s = L.w3obs_new()
        widths = tuple(L.w3obs_width(s, k) for k in range(9))
        L.w3obs_free(s)
        if widths != _WIDTHS:
            raise RuntimeError(f"{LIB}: table widths {widths}, expected {_WIDTHS}")
        _lib = L
    return _lib


def addr(a: np.ndarray) -> int:
    """A C-contiguous array's address: of a ctypes char over its buffer when it is writable and not empty (numpy's
    .ctypes.data made a Python object of its own each time, ~3x the instructions), else .ctypes.data."""
    try:
        return ctypes.addressof(ctypes.c_char.from_buffer(a))
    except (TypeError, ValueError, BufferError):
        return a.ctypes.data


_FLAGS: dict[int, UnitFlags] = {}  # (each flags' value as UnitFlags, made once: the enum's call per unit was slow)


def unit_from_row(r) -> Unit:
    f = _FLAGS.get(r[C_FLAGS])
    if f is None:
        f = _FLAGS[r[C_FLAGS]] = UnitFlags(r[C_FLAGS])
    u = Unit(r[C_ID], r[C_TYPE], r[C_OWNER], r[C_X], r[C_Y], r[C_FACING], r[C_HP], r[C_MAXHP], r[C_MANA], r[C_MAXMANA],
             r[C_ORDER], f, r[C_VIS], r[C_RESOURCE])
    if r[C_HERO]:
        u.hero_level, u.hero_xp, u.skill_points = r[C_HLEVEL], r[C_HXP], r[C_SKILLPTS]
        u.items = tuple(r[C_ITEMS:C_ITEMS + 6])
        u.abilities = tuple((r[C_ABILITIES + 2 * k], r[C_ABILITIES + 1 + 2 * k] / 10.0) for k in range(HERO_ABILITY_SLOTS))
    return u


class LazyUnits(Sequence):
    """Observation.units from the unit table: Unit objects made on first use."""

    def __init__(self, rows: np.ndarray):
        self.rows = rows
        self._units: list[Unit] | None = None

    def _all(self) -> list[Unit]:
        if self._units is None:
            self._units = [unit_from_row(r) for r in self.rows.tolist()]
        return self._units

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        return self._all()[i]

    def __iter__(self):
        return iter(self._all())

    def __eq__(self, other) -> bool:
        return list(self) == list(other)


class NativeObs:
    """One game's parser and unit table."""

    def __init__(self):
        self._lib = lib()
        self._s = self._lib.w3obs_new()
        if not self._s:
            raise MemoryError("w3obs_new")
        self._grow(4096)

    def __del__(self):
        s, self._s = getattr(self, "_s", None), None
        if s:
            self._lib.w3obs_free(s)

    def _grow(self, n: int) -> None:
        """The buffer w3obs_export fills (this game's own: a game's thread parses its observations), at least n ints."""
        self._cap = max(n, 2 * getattr(self, "_cap", 0))
        self._buf = (ctypes.c_int32 * self._cap)()
        self._addr = ctypes.addressof(self._buf)

    def _rows(self, which: int) -> np.ndarray:
        """Table `which` of the last observation parsed (w3obs_rows' order), as an array."""
        n, w = ctypes.c_int(), ctypes.c_int()
        ptr = self._lib.w3obs_rows(self._s, which, ctypes.byref(n), ctypes.byref(w))
        out = np.empty((n.value, w.value), np.int32)
        if len(out):
            ctypes.memmove(out.ctypes.data, ptr, out.nbytes)
        return out

    def parse(self, payload: bytes, order_names: Sequence[str]) -> Observation:
        """The observation without its units (merge() adds them): as protocol.parse_token_lines. (Its tables come in
        one copy, w3obs_export, as one list of ints: each table as an array of its own had cost ~100k instructions.)"""
        n = self._lib.w3obs_parse_export(self._s, payload, len(payload), self._addr, self._cap)
        if n == -2:
            raise MemoryError("w3obs_parse")
        if n > self._cap:
            self._grow(n)
            n = self._lib.w3obs_export(self._s, self._addr, self._cap)
        v = self._buf[:n]
        if not v[_H_ENDED]:
            raise ProtocolError("observation has no end marker (partial write?)")
        if v[_H_VERSION] != PROTOCOL_VERSION:
            raise ProtocolError(f"harness protocol {v[_H_VERSION]}, expected {PROTOCOL_VERSION}")
        n_p, n_e, n_c, n_i, n_o, n_d, n_r, _, n_g = v[16:_TABLES]
        a, players = _TABLES, {}
        if n_p:
            races, results_ = Race._value2member_map_, Result._value2member_map_
            for k in range(a, a + 14 * n_p, 14):
                pid, race, agent, gold, lumber, fu, fc, upkeep, gg, lg, structures, res, sx, sy = v[k:k + 14]
                r = results_.get(res)
                players[pid] = PlayerState(pid, races.get(race, Race.UNKNOWN), bool(agent), gold, lumber, fu, fc, upkeep,
                                           gg, lg, structures, Result(res) if r is None else r, sx, sy)
            a += 14 * n_p
        if n_e:
            kinds = EventKind._value2member_map_  # (EventKind(k), or k unknown: without the call and its exception)
            it = iter(v[a:a + 4 * n_e])
            events = [Event(kinds.get(k, k), x, y, z) for k, x, y, z in zip(it, it, it, it)]
            a += 4 * n_e
        else:
            events = []
        results = list(map(bool, v[a:a + n_c]))
        a += n_c
        if n_i:
            it = iter(v[a:a + 6 * n_i])
            issued = list(map(IssuedOrder, it, it, it, it, it, it))
            a += 6 * n_i
        else:
            issued = []
        order_map = {name: oid for name, oid in zip(order_names, v[a:a + n_o]) if oid} if n_o else None
        a += n_o
        if n_d:
            it = iter(v[a:a + 5 * n_d])
            dests = list(map(Destructable, it, it, it, it, it))
            a += 5 * n_d
        else:
            dests = None
        removed = v[a:a + n_r]
        a += n_r
        it = iter(v[a:a + 5 * n_g])
        return Observation(v[_H_SEQ], v[_H_MS], bool(v[_H_OVER]), players, [], events, results, dests, order_map,
                           v[_H_VERSION], v[_H_DAMAGED], bool(v[_H_FULL]), removed,
                           (float(v[_H_CAM_X]), float(v[_H_CAM_Y])) if v[_H_CAMERA] else None, issued,
                           list(zip(it, it, it, it, it)))

    def merge(self, obs: Observation) -> Observation:
        """Apply the last parsed observation to the unit table: obs.units / unit_array hold all units."""
        n = self._lib.w3obs_merge(self._s)
        if n < 0:
            raise MemoryError("w3obs_merge")
        rows = np.empty((n, UCOLS), np.int32)
        if n:
            self._lib.w3obs_table(self._s, addr(rows), n)
        obs.unit_array = rows
        obs.units = LazyUnits(rows)
        obs.full = True
        return obs
