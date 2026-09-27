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
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import paths
from .protocol import (HERO_ABILITY_SLOTS, PROTOCOL_VERSION, Destructable, Event, EventKind, IssuedOrder,
                       Observation, PlayerState, ProtocolError, Race, Result, Unit, UnitFlags)

SRC = paths.REPO_ROOT / "native" / "w3obs.c"
LIB = paths.REPO_ROOT / "build" / "native" / "libw3obs.so"
# unit_array columns: the U record's fields, then the hero fields (0 for others), then whether they're set
(C_ID, C_TYPE, C_OWNER, C_X, C_Y, C_FACING, C_HP, C_MAXHP, C_MANA, C_MAXMANA, C_ORDER, C_FLAGS, C_VIS, C_RESOURCE,
 C_HLEVEL, C_HXP, C_SKILLPTS) = range(17)
C_ITEMS, C_ABILITIES, C_HERO = 17, 23, 31
UCOLS = 32
_H_SEQ, _H_MS, _H_OVER, _H_FULL, _H_VERSION, _H_DAMAGED, _H_ENDED, _H_CAMERA, _H_CAM_X, _H_CAM_Y, _H_NUNITS = range(11)
_lib = None


def _build() -> None:
    LIB.parent.mkdir(parents=True, exist_ok=True)
    with open(LIB.parent / ".build.lock", "w") as lock:  # one build at a time (actor processes start together)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if LIB.exists() and LIB.stat().st_mtime >= SRC.stat().st_mtime:
            return
        tmp = LIB.with_suffix(".so.tmp")
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", str(tmp), str(SRC)], check=True)
        tmp.replace(LIB)


def lib():
    global _lib
    if _lib is None:
        if not LIB.exists() or LIB.stat().st_mtime < SRC.stat().st_mtime:
            _build()
        L = ctypes.CDLL(str(LIB))  # (CDLL: the GIL is released during calls)
        L.w3obs_new.restype = ctypes.c_void_p
        L.w3obs_free.argtypes = [ctypes.c_void_p]
        L.w3obs_parse.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        L.w3obs_merge.argtypes = [ctypes.c_void_p]
        L.w3obs_table.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        L.w3obs_header.restype = ctypes.POINTER(ctypes.c_int32)
        L.w3obs_header.argtypes = [ctypes.c_void_p]
        L.w3obs_rows.restype = ctypes.POINTER(ctypes.c_int32)
        L.w3obs_rows.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        if L.w3obs_ucols() != UCOLS:
            raise RuntimeError(f"{LIB}: {L.w3obs_ucols()} unit columns, expected {UCOLS}")
        _lib = L
    return _lib


def unit_from_row(r) -> Unit:
    u = Unit(r[C_ID], r[C_TYPE], r[C_OWNER], r[C_X], r[C_Y], r[C_FACING], r[C_HP], r[C_MAXHP], r[C_MANA], r[C_MAXMANA],
             r[C_ORDER], UnitFlags(r[C_FLAGS]), r[C_VIS], r[C_RESOURCE])
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

    def __del__(self):
        s, self._s = getattr(self, "_s", None), None
        if s:
            self._lib.w3obs_free(s)

    def _rows(self, which: int) -> np.ndarray:
        n, w = ctypes.c_int(), ctypes.c_int()
        ptr = self._lib.w3obs_rows(self._s, which, ctypes.byref(n), ctypes.byref(w))
        if n.value == 0:
            return np.zeros((0, w.value), np.int32)
        return np.ctypeslib.as_array(ptr, shape=(n.value, w.value)).copy()

    def parse(self, payload: bytes, order_names: Sequence[str]) -> Observation:
        """The observation without its units (merge() adds them): as protocol.parse_token_lines."""
        rc = self._lib.w3obs_parse(self._s, payload, len(payload))
        if rc == -2:
            raise MemoryError("w3obs_parse")
        h = self._lib.w3obs_header(self._s)
        if rc == -1:
            raise ProtocolError("observation has no end marker (partial write?)")
        if h[_H_VERSION] != PROTOCOL_VERSION:
            raise ProtocolError(f"harness protocol {h[_H_VERSION]}, expected {PROTOCOL_VERSION}")
        players = {}
        for v in self._rows(0).tolist():
            players[v[0]] = PlayerState(v[0], Race(v[1]) if v[1] in Race._value2member_map_ else Race.UNKNOWN,
                                        bool(v[2]), v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], Result(v[11]),
                                        v[12], v[13])
        events = []
        for k, a, b, c in self._rows(1).tolist():
            try:
                kind: EventKind | int = EventKind(k)
            except ValueError:
                kind = k
            events.append(Event(kind, a, b, c))
        results = [bool(v[0]) for v in self._rows(2).tolist()]
        issued = [IssuedOrder(*v[:6]) for v in self._rows(3).tolist()]
        orders = self._rows(4)
        dests = self._rows(5)
        order_map = ({name: oid for name, oid in zip(order_names, orders[:, 0].tolist()) if oid}
                     if len(orders) else None)
        obs = Observation(h[_H_SEQ], h[_H_MS], bool(h[_H_OVER]), players, [], events, results,
                          [Destructable(*v) for v in dests.tolist()] if len(dests) else None, order_map, h[_H_VERSION])
        obs.damaged_records = h[_H_DAMAGED]
        obs.full = bool(h[_H_FULL])
        obs.removed = self._rows(6)[:, 0].tolist()
        obs.issued = issued
        obs.camera = (float(h[_H_CAM_X]), float(h[_H_CAM_Y])) if h[_H_CAMERA] else None
        return obs

    def merge(self, obs: Observation) -> Observation:
        """Apply the last parsed observation to the unit table: obs.units / unit_array hold all units."""
        n = self._lib.w3obs_merge(self._s)
        if n < 0:
            raise MemoryError("w3obs_merge")
        rows = np.empty((n, UCOLS), np.int32)
        if n:
            self._lib.w3obs_table(self._s, rows.ctypes.data, n)
        obs.unit_array = rows
        obs.units = LazyUnits(rows)
        obs.full = True
        return obs
