"""war3map.w3i (map info, format version 25 / TFT): the parts needed to resize a map.

Parsed: camera bounds, camera complements (border tiles), playable size and the player records
(start positions). Everything else is kept as raw bytes and written back unchanged.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


@dataclass
class W3iPlayer:
    id: int
    type: int
    race: int
    fixed_start: int
    name: bytes
    start_x: float
    start_y: float
    ally_low: int
    ally_high: int

    def pack(self) -> bytes:
        return (struct.pack("<4I", self.id, self.type, self.race, self.fixed_start) + self.name + b"\0"
                + struct.pack("<2f2I", self.start_x, self.start_y, self.ally_low, self.ally_high))


@dataclass
class W3i:
    head: bytes  # version .. players-recommended string
    camera_bounds: list[float]  # 8 floats
    complements: list[int]  # left, right, bottom, top border tiles
    playable_width: int
    playable_height: int
    middle: bytes  # flags .. water tint
    players: list[W3iPlayer] = field(default_factory=list)
    tail: bytes = b""  # forces, upgrades, tech, random tables

    def pack(self) -> bytes:
        out = bytearray(self.head)
        out += struct.pack("<8f", *self.camera_bounds)
        out += struct.pack("<4I", *self.complements)
        out += struct.pack("<2I", self.playable_width, self.playable_height)
        out += self.middle
        out += struct.pack("<I", len(self.players))
        for p in self.players:
            out += p.pack()
        out += self.tail
        return bytes(out)


def parse_w3i(data: bytes) -> W3i:
    pos = 0

    def skip_str() -> None:
        nonlocal pos
        pos = data.index(b"\0", pos) + 1

    (version,) = struct.unpack_from("<I", data, 0)
    if version != 25:
        raise ValueError(f"unsupported w3i version {version}")
    pos = 12  # version, saves, editor version
    for _ in range(4):  # name, author, description, players recommended
        skip_str()
    head = data[:pos]
    camera = list(struct.unpack_from("<8f", data, pos))
    pos += 32
    complements = list(struct.unpack_from("<4I", data, pos))
    pos += 16
    pw, ph = struct.unpack_from("<2I", data, pos)
    pos += 8
    mid_start = pos
    pos += 4 + 1  # flags, ground tileset
    pos += 4  # loading screen number
    for _ in range(4):  # loading screen model, text, title, subtitle
        skip_str()
    pos += 4  # game data set
    for _ in range(4):  # prologue path, text, title, subtitle
        skip_str()
    pos += 4 + 12 + 4  # fog style, start z, end z, density, color
    pos += 4  # weather id
    skip_str()  # sound environment
    pos += 1 + 4  # light environment tileset, water tint
    middle = data[mid_start:pos]
    (n_players,) = struct.unpack_from("<I", data, pos)
    pos += 4
    players = []
    for _ in range(n_players):
        pid, ptype, race, fixed = struct.unpack_from("<4I", data, pos)
        pos += 16
        end = data.index(b"\0", pos)
        name = data[pos:end]
        pos = end + 1
        sx, sy, al, ah = struct.unpack_from("<2f2I", data, pos)
        pos += 16
        players.append(W3iPlayer(pid, ptype, race, fixed, name, sx, sy, al, ah))
    return W3i(head, camera, complements, pw, ph, middle, players, data[pos:])
