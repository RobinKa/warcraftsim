"""Flat, empty maps for scenarios.

``make_flat_map(source, dest)`` keeps the source map's size, players and script (the scenario
harness removes its units anyway) but rewrites the terrain to one level plane of the first ground
texture with no water, cliffs, ramps, shadows or doodads, and marks every cell inside the border
as open land.

``make_flat_map(source, dest, size=N)`` builds a new N x N-tile playable area (N*128 world units a
side) centered at (0, 0), keeping the source's tileset, border widths and players. The two start
locations are at (-N*32, 0) and (N*32, 0).

Map names (see mapbuild.stock_map_path): "flat" = 32 x 32 tiles, "flatN" = N x N tiles,
"flat:<stock map>" = full-size flat version of a stock map.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import threading
from pathlib import Path

import numpy as np

from .. import paths
from .mpq import MpqArchive
from .w3i import parse_w3i

LAND = 0x40  # walkable, flyable, buildable, not water
BORDER = 0xCE  # the value of border cells in stock maps (unwalkable, unflyable, unbuildable)
BOUNDARY_FLAG = 0x4000  # in the w3e water-level field: corner outside the playable area
CAMERA_MARGIN = (512.0, 256.0)  # default camera margins (x, y), see GetCameraMargin
DEFAULT_SIZE = 32
_lock = threading.Lock()


def _w3e_header(data: bytes) -> tuple[int, int, int]:
    """(position of width field, corner width, corner height)."""
    if data[:4] != b"W3E!":
        raise ValueError("not a w3e file")
    pos = 4 + 9
    (n_ground,) = struct.unpack_from("<I", data, pos)
    pos += 4 + 4 * n_ground
    (n_cliff,) = struct.unpack_from("<I", data, pos)
    pos += 4 + 4 * n_cliff
    width, height = struct.unpack_from("<II", data, pos)
    return pos, width, height


def _border_cells(width_tiles: int, height_tiles: int, complements: list[int]) -> np.ndarray:
    """Pathing-cell mask (4 cells per tile) of the border around the playable area."""
    left, right, bottom, top = complements
    mask = np.zeros((height_tiles * 4, width_tiles * 4), dtype=bool)
    mask[:, :left * 4] = mask[:, (width_tiles - right) * 4:] = True
    mask[:bottom * 4, :] = mask[(height_tiles - top) * 4:, :] = True
    return mask


def _flat_corners(width: int, height: int, complements: list[int], water_level: int) -> np.ndarray:
    left, right, bottom, top = complements
    wt, ht = width - 1, height - 1
    corners = np.zeros((height, width, 7), dtype=np.uint8)
    corners[:, :, 0:2] = np.frombuffer(struct.pack("<h", 0x2000), dtype=np.uint8)  # ground level 0
    cols, rows = np.arange(width), np.arange(height)
    border = ((cols < left) | ((cols >= wt - right) & (cols < wt)))[None, :] | \
             ((rows < bottom) | ((rows >= ht - top) & (rows < ht)))[:, None]
    water = np.where(border, water_level | BOUNDARY_FLAG, water_level).astype("<u2")
    corners[:, :, 2:4] = water.view(np.uint8).reshape(height, width, 2)
    corners[:, :, 6] = 0x02  # cliff texture 0, layer height 2 (the base level)
    return corners


def flatten_w3e(data: bytes, complements: list[int]) -> bytes:
    pos, width, height = _w3e_header(data)
    start = pos + 16
    old = np.frombuffer(data, dtype=np.uint8, count=width * height * 7, offset=start).reshape(height, width, 7)
    water_level = int(old[0, 0, 2:4].copy().view("<u2")[0] & 0x3FFF)
    corners = _flat_corners(width, height, complements, water_level)
    return data[:start] + corners.tobytes() + data[start + corners.size:]


def flatten_wpm(data: bytes, complements: list[int]) -> bytes:
    if data[:4] != b"MP3W":
        raise ValueError("not a wpm file")
    _version, width, height = struct.unpack_from("<III", data, 4)
    border = _border_cells(width // 4, height // 4, complements)
    cells = np.where(border, BORDER, LAND).astype(np.uint8)
    return data[:16] + cells.tobytes() + data[16 + cells.size:]


def empty_doodads(data: bytes) -> bytes:
    if data[:4] != b"W3do":
        raise ValueError("not a doo file")
    version, subversion = struct.unpack_from("<II", data, 4)
    # no doodads, then an empty special-doodad section (version 0, count 0)
    return b"W3do" + struct.pack("<IIIII", version, subversion, 0, 0, 0)


def empty_units(data: bytes) -> bytes:
    if data[:4] != b"W3do":
        raise ValueError("not a units.doo file")
    version, subversion = struct.unpack_from("<II", data, 4)
    return b"W3do" + struct.pack("<III", version, subversion, 0)


def _resized(m: MpqArchive, size: int) -> dict[str, bytes]:
    """All files that change when the playable area becomes size x size tiles around (0, 0)."""
    info = parse_w3i(m.read("war3map.w3i"))
    left, right, bottom, top = info.complements
    wt, ht = size + left + right, size + bottom + top
    off_x, off_y = -(size / 2 + left) * 128.0, -(size / 2 + bottom) * 128.0
    half = size * 64.0
    mx, my = CAMERA_MARGIN
    info.camera_bounds = [-half + mx, -half + my, half - mx, half - my, -half + mx, half - my, half - mx, -half + my]
    info.playable_width = info.playable_height = size
    starts = [(-size * 32.0, 0.0), (size * 32.0, 0.0)]
    for i, p in enumerate(info.players):
        p.start_x, p.start_y = starts[i % 2]
    # terrain: same header (tilesets), new size and offset
    w3e = m.read("war3map.w3e")
    pos, width, height = _w3e_header(w3e)
    old = np.frombuffer(w3e, dtype=np.uint8, count=7, offset=pos + 16)
    water_level = int(old[2:4].copy().view("<u2")[0] & 0x3FFF)
    corners = _flat_corners(wt + 1, ht + 1, info.complements, water_level)
    new_w3e = w3e[:pos] + struct.pack("<IIff", wt + 1, ht + 1, off_x, off_y) + corners.tobytes()
    wpm = m.read("war3map.wpm")
    cells = np.where(_border_cells(wt, ht, info.complements), BORDER, LAND).astype(np.uint8)
    new_wpm = wpm[:8] + struct.pack("<II", wt * 4, ht * 4) + cells.tobytes()
    # script: camera bounds, weather area, start locations; no pre-placed units
    script = m.read("war3map.j").decode("latin-1")
    lo_x, lo_y, hi_x, hi_y = -half, -half, half, half
    script = re.sub(
        r"call SetCameraBounds\(.*\)",
        (f"call SetCameraBounds( {lo_x:.1f} + GetCameraMargin(CAMERA_MARGIN_LEFT), {lo_y:.1f} + "
         f"GetCameraMargin(CAMERA_MARGIN_BOTTOM), {hi_x:.1f} - GetCameraMargin(CAMERA_MARGIN_RIGHT), {hi_y:.1f} - "
         f"GetCameraMargin(CAMERA_MARGIN_TOP), {lo_x:.1f} + GetCameraMargin(CAMERA_MARGIN_LEFT), {hi_y:.1f} - "
         f"GetCameraMargin(CAMERA_MARGIN_TOP), {hi_x:.1f} - GetCameraMargin(CAMERA_MARGIN_RIGHT), {lo_y:.1f} + "
         f"GetCameraMargin(CAMERA_MARGIN_BOTTOM) )"), script, count=1)
    script = re.sub(r"Rect\(\s*-?[\d.]+\s*,\s*-?[\d.]+\s*,\s*-?[\d.]+\s*,\s*-?[\d.]+\s*\)",
                    f"Rect({off_x:.1f},{off_y:.1f},{off_x + wt * 128:.1f},{off_y + ht * 128:.1f})", script)
    for i, (sx, sy) in enumerate(starts):
        script = re.sub(rf"call DefineStartLocation\( {i}, -?[\d.]+, -?[\d.]+ \)",
                        f"call DefineStartLocation( {i}, {sx:.1f}, {sy:.1f} )", script)
    script = re.sub(r"^\s*call CreateAllUnits\(\s*\)\s*$", "", script, flags=re.M)
    return {
        "war3map.w3i": info.pack(),
        "war3map.w3e": new_w3e,
        "war3map.wpm": new_wpm,
        "war3map.shd": bytes(wt * ht * 16),
        "war3map.j": script.encode("latin-1"),
        "war3mapUnits.doo": empty_units(m.read("war3mapUnits.doo")),
        "war3map.mmp": bytes(8),  # no minimap icons
    }


def make_flat_map(source: str | Path, dest: str | Path, size: int | None = None) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    shutil.copyfile(source, tmp)
    try:
        with MpqArchive(tmp, writable=True) as m:
            complements = parse_w3i(m.read("war3map.w3i")).complements
            if size is None:
                files = {
                    "war3map.w3e": flatten_w3e(m.read("war3map.w3e"), complements),
                    "war3map.wpm": flatten_wpm(m.read("war3map.wpm"), complements),
                    "war3map.shd": bytes(len(m.read("war3map.shd"))),
                }
            else:
                files = _resized(m, size)
            files["war3map.doo"] = empty_doodads(m.read("war3map.doo"))
            for name, data in files.items():
                m.write(name, data)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def parse_flat_name(name: str) -> tuple[str, int | None] | None:
    """"flat" -> (Echo Isles, 32), "flat48" -> (Echo Isles, 48), "flat:<map>" -> (<map>, None)."""
    if name.startswith("flat:"):
        return name[5:] or "(2)EchoIsles", None
    m = re.fullmatch(r"flat(\d*)", name)
    if not m:
        return None
    size = int(m.group(1)) if m.group(1) else DEFAULT_SIZE
    if size < 8 or size > 256 or size % 2:
        raise ValueError("flat map size must be an even number of tiles between 8 and 256")
    return "(2)EchoIsles", size


def flat_map_path(name: str) -> Path:
    """The cached flat map for a "flat..." name (built on first use)."""
    from .mapbuild import stock_map_path

    base, size = parse_flat_name(name)
    src = stock_map_path(base)
    suffix = f"{size}" if size else "full"
    out = paths.CACHE_DIR / "maps" / f"flat{suffix}_{src.stem}.w3x"
    with _lock:
        if not out.exists():
            make_flat_map(src, out, size)
    return out
