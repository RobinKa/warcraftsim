"""Static map data: terrain header (war3map.w3e) and pathing grid (war3map.wpm).

World coordinates: the w3e header gives the map's bottom-left corner. Terrain
tiles are 128x128 world units; pathing cells are 32x32 (4x4 per tile).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mpq import MpqArchive

PATH_NO_WALK = 0x02
PATH_NO_FLY = 0x04
PATH_NO_BUILD = 0x08
PATH_BLIGHT = 0x20
PATH_NO_WATER = 0x40  # set on land; clear means water


@dataclass
class Terrain:
    offset_x: float  # world x of the bottom-left corner
    offset_y: float
    width: int  # tile corners (tiles + 1)
    height: int
    tileset: str
    corner_height: np.ndarray  # (height, width) float, world units
    corner_water: np.ndarray  # (height, width) bool
    pathing: np.ndarray  # (4*(height-1), 4*(width-1)) uint8 flags, row 0 = bottom

    @property
    def cell_size(self) -> int:
        return 32

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((y - self.offset_y) // 32), int((x - self.offset_x) // 32)

    def world(self, row: int, col: int) -> tuple[float, float]:
        return self.offset_x + (col + 0.5) * 32, self.offset_y + (row + 0.5) * 32

    @property
    def walkable(self) -> np.ndarray:
        return (self.pathing & PATH_NO_WALK) == 0

    @property
    def buildable(self) -> np.ndarray:
        return (self.pathing & (PATH_NO_BUILD | PATH_NO_WALK)) == 0


def parse_w3e(data: bytes) -> dict:
    if data[:4] != b"W3E!":
        raise ValueError("not a w3e file")
    pos = 4
    version, tileset, custom = struct.unpack_from("<IcI", data, pos)
    pos += 9
    (n_ground,) = struct.unpack_from("<I", data, pos)
    pos += 4 + 4 * n_ground
    (n_cliff,) = struct.unpack_from("<I", data, pos)
    pos += 4 + 4 * n_cliff
    width, height, off_x, off_y = struct.unpack_from("<IIff", data, pos)
    pos += 16
    corners = np.frombuffer(data, dtype=np.uint8, count=width * height * 7, offset=pos).reshape(height, width, 7)
    ground = corners[:, :, 0:2].copy().view("<i2")[:, :, 0].astype(np.float32)
    water_raw = corners[:, :, 2:4].copy().view("<u2")[:, :, 0]
    flags = corners[:, :, 4] >> 4
    layer = (corners[:, :, 6] & 0x0F).astype(np.float32)
    # final height = (ground - 0x2000) / 4 + (layer - 2) * 128
    heights = (ground - 0x2000) / 4.0 + (layer - 2) * 128.0
    water = (flags & 0x4) != 0
    return {"version": version, "tileset": tileset.decode("latin-1"), "width": width, "height": height,
            "offset_x": off_x, "offset_y": off_y, "height_map": heights, "water": water,
            "water_level": water_raw & 0x3FFF}


def parse_wpm(data: bytes) -> np.ndarray:
    if data[:4] != b"MP3W":
        raise ValueError("not a wpm file")
    _version, width, height = struct.unpack_from("<III", data, 4)
    return np.frombuffer(data, dtype=np.uint8, count=width * height, offset=16).reshape(height, width).copy()


def load_terrain(map_path: str | Path) -> Terrain:
    with MpqArchive(map_path) as m:
        w3e = parse_w3e(m.read("war3map.w3e"))
        wpm = parse_wpm(m.read("war3map.wpm"))
    return Terrain(w3e["offset_x"], w3e["offset_y"], w3e["width"], w3e["height"], w3e["tileset"],
                   w3e["height_map"], w3e["water"], wpm)


def open_area_center(terrain: Terrain, bounds: tuple[float, float, float, float] | None = None,
                     ignore: np.ndarray | None = None) -> tuple[float, float, float]:
    """The walkable point farthest from any unwalkable cell: (x, y, clearance in world units).

    `ignore` marks cells to treat as walkable anyway (e.g. under trees that will be removed).
    `bounds` = (min_x, min_y, max_x, max_y) limits the search (e.g. the playable area).
    """
    free = terrain.walkable.copy()
    if ignore is not None:
        free |= ignore
    if bounds is not None:
        r0, c0 = terrain.cell(bounds[0], bounds[1])
        r1, c1 = terrain.cell(bounds[2], bounds[3])
        mask = np.zeros_like(free)
        mask[max(r0, 0):r1, max(c0, 0):c1] = True
        free &= mask
    dist = _chebyshev_distance(free)
    r, c = np.unravel_index(int(np.argmax(dist)), dist.shape)
    x, y = terrain.world(r, c)
    return x, y, float(dist[r, c]) * 32.0


def _chebyshev_distance(free: np.ndarray) -> np.ndarray:
    """Cells to the nearest blocked cell (chessboard metric), by repeated erosion."""
    dist = np.zeros(free.shape, dtype=np.int32)
    cur = free.copy()
    level = 0
    while cur.any():
        level += 1
        dist[cur] = level
        nxt = cur.copy()
        nxt[1:, :] &= cur[:-1, :]
        nxt[:-1, :] &= cur[1:, :]
        nxt[:, 1:] &= cur[:, :-1]
        nxt[:, :-1] &= cur[:, 1:]
        nxt[1:, 1:] &= cur[:-1, :-1]
        nxt[:-1, :-1] &= cur[1:, 1:]
        nxt[1:, :-1] &= cur[:-1, 1:]
        nxt[:-1, 1:] &= cur[1:, :-1]
        nxt[0, :] = nxt[-1, :] = False
        nxt[:, 0] = nxt[:, -1] = False
        cur = nxt
    return dist
