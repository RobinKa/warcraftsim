"""A flat, empty version of a stock map, for scenarios.

Keeps the source map's size, players, start locations and script (the scenario harness removes
its units anyway) but rewrites the terrain to one level plane of the first ground texture with
no water, cliffs, ramps, shadows or doodads (trees), and marks every cell inside the map border
as open land in the pathing grid.
"""

from __future__ import annotations

import os
import shutil
import struct
import threading
from pathlib import Path

import numpy as np

from .. import paths
from .mpq import MpqArchive

LAND = 0x40  # walkable, flyable, buildable, not water
_lock = threading.Lock()


def flatten_w3e(data: bytes) -> bytes:
    if data[:4] != b"W3E!":
        raise ValueError("not a w3e file")
    pos = 4 + 9
    (n_ground,) = struct.unpack_from("<I", data, pos)
    pos += 4 + 4 * n_ground
    (n_cliff,) = struct.unpack_from("<I", data, pos)
    pos += 4 + 4 * n_cliff
    width, height = struct.unpack_from("<II", data, pos)
    pos += 16
    corners = np.frombuffer(data, dtype=np.uint8, count=width * height * 7, offset=pos).reshape(-1, 7).copy()
    corners[:, 0:2] = np.frombuffer(struct.pack("<h", 0x2000), dtype=np.uint8)  # ground level 0
    corners[:, 4] &= 0x80  # keep the boundary flag; no ramp/blight/water; ground texture 0
    corners[:, 5] = 0  # texture variation
    corners[:, 6] = 0x02  # cliff texture 0, layer height 2 (the base level)
    return data[:pos] + corners.tobytes() + data[pos + corners.size:]


def flatten_wpm(data: bytes) -> bytes:
    if data[:4] != b"MP3W":
        raise ValueError("not a wpm file")
    _version, width, height = struct.unpack_from("<III", data, 4)
    cells = np.frombuffer(data, dtype=np.uint8, count=width * height, offset=16).copy()
    border = (cells & 0x80) != 0
    cells[~border] = LAND
    return data[:16] + cells.tobytes() + data[16 + cells.size:]


def empty_doodads(data: bytes) -> bytes:
    if data[:4] != b"W3do":
        raise ValueError("not a doo file")
    version, subversion = struct.unpack_from("<II", data, 4)
    # no doodads, then an empty special-doodad section (version 0, count 0)
    return b"W3do" + struct.pack("<IIIII", version, subversion, 0, 0, 0)


def make_flat_map(source: str | Path, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    shutil.copyfile(source, tmp)
    try:
        with MpqArchive(tmp, writable=True) as m:
            m.write("war3map.w3e", flatten_w3e(m.read("war3map.w3e")))
            m.write("war3map.wpm", flatten_wpm(m.read("war3map.wpm")))
            m.write("war3map.shd", bytes(len(m.read("war3map.shd"))))
            m.write("war3map.doo", empty_doodads(m.read("war3map.doo")))
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def flat_map_path(base: str = "(2)EchoIsles") -> Path:
    """The cached flat version of a stock map (built on first use)."""
    from .mapbuild import stock_map_path

    src = stock_map_path(base)
    out = paths.CACHE_DIR / "maps" / f"flat_{src.stem}.w3x"
    with _lock:
        if not out.exists():
            make_flat_map(src, out)
    return out
