"""Warcraft III Game Configuration (.wgc) files.

A .wgc passed via ``-loadfile`` starts a local game directly: map, slots
(user / computer / observer, race, team, AI difficulty) and an integer game
speed multiplier (1 = normal, 0 = paused, no documented upper bound).

Layout (all ints are uint32 little-endian, strings are NUL-terminated;
classic builds read ANSI paths relative to the game root):

    version=1, flags, game_speed, map_path, player_count,
    player_count x (slot, team, race, color, handicap, controller_flags,
                    ai_difficulty, ai_script_path)

Reference: Luashine/wc3-file-formats specs/wgc/WGC_format-v1.txt.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field

FLAG_NO_FOG = 0x01
FLAG_NO_VICTORY = 0x02


class Race(enum.IntEnum):
    HUMAN = 0x01
    ORC = 0x02
    NIGHTELF = 0x04
    UNDEAD = 0x08
    RANDOM = 0x20

    @classmethod
    def parse(cls, value: "str | Race") -> "Race":
        if isinstance(value, Race):
            return value
        aliases = {"elf": "NIGHTELF", "night_elf": "NIGHTELF", "nightelf": "NIGHTELF", "ud": "UNDEAD", "hu": "HUMAN"}
        key = aliases.get(value.lower(), value.upper())
        return cls[key]


class Difficulty(enum.IntEnum):
    EASY = 0
    NORMAL = 1
    INSANE = 2  # "hard" in the file-format docs

    @classmethod
    def parse(cls, value: "str | int | Difficulty") -> "Difficulty":
        if isinstance(value, int):
            return cls(value)
        return cls[{"hard": "INSANE"}.get(value.lower(), value.upper())]


# controller flag bits
CTRL_USER = 0x1
CTRL_OBSERVER = 0x2
CTRL_CUSTOM_AI = 0x4
CTRL_AI_RELATIVE = 0x8


@dataclass
class WgcSlot:
    slot: int
    team: int = 0
    race: Race = Race.HUMAN
    color: int = 0
    handicap: int = 100
    controller: int = 0
    ai_difficulty: int = Difficulty.NORMAL
    ai_script: str = ""

    @classmethod
    def user(cls, slot: int, race: Race | str, team: int | None = None, color: int | None = None) -> "WgcSlot":
        return cls(slot, team if team is not None else slot, Race.parse(race),
                   slot if color is None else color, 100, CTRL_USER, 1)

    @classmethod
    def observer(cls, slot: int) -> "WgcSlot":
        # users/observers must have the difficulty LSB set or the map does not load
        return cls(slot, 0, Race.HUMAN, 0, 100, CTRL_USER | CTRL_OBSERVER, 1)

    @classmethod
    def computer(cls, slot: int, race: Race | str, difficulty: Difficulty | str | int = Difficulty.NORMAL,
                 team: int | None = None, color: int | None = None, ai_script: str = "",
                 ai_script_relative: bool = True) -> "WgcSlot":
        ctrl = 0
        if ai_script:
            ctrl |= CTRL_CUSTOM_AI | (CTRL_AI_RELATIVE if ai_script_relative else 0)
        return cls(slot, team if team is not None else slot, Race.parse(race),
                   slot if color is None else color, 100, ctrl, Difficulty.parse(difficulty), ai_script)

    @property
    def is_user(self) -> bool:
        return bool(self.controller & CTRL_USER)

    @property
    def is_observer(self) -> bool:
        return bool(self.controller & CTRL_OBSERVER)


@dataclass
class Wgc:
    map_path: str
    slots: list[WgcSlot] = field(default_factory=list)
    game_speed: int = 1
    flags: int = 0
    version: int = 1
    encoding: str = "latin-1"  # classic (1.29) reads ANSI paths

    def to_bytes(self) -> bytes:
        out = bytearray(struct.pack("<III", self.version, self.flags, self.game_speed))
        out += self.map_path.encode(self.encoding) + b"\0"
        out += struct.pack("<I", len(self.slots))
        for s in self.slots:
            out += struct.pack("<7I", s.slot, s.team, int(s.race), s.color, s.handicap,
                               s.controller, int(s.ai_difficulty))
            out += s.ai_script.encode(self.encoding) + b"\0"
        return bytes(out)

    @classmethod
    def from_bytes(cls, data: bytes, encoding: str = "latin-1") -> "Wgc":
        pos = 0

        def u32() -> int:
            nonlocal pos
            (v,) = struct.unpack_from("<I", data, pos)
            pos += 4
            return v

        def strz() -> str:
            nonlocal pos
            end = data.index(b"\0", pos)
            s = data[pos:end].decode(encoding)
            pos = end + 1
            return s

        version, flags, speed = u32(), u32(), u32()
        map_path = strz()
        slots = []
        for _ in range(u32()):
            slot, team, race, color, handicap, ctrl, diff = (u32() for _ in range(7))
            slots.append(WgcSlot(slot, team, Race(race), color, handicap, ctrl, diff, strz()))
        if pos != len(data):
            raise ValueError(f"trailing bytes in .wgc: {len(data) - pos}")
        return cls(map_path, slots, speed, flags, version, encoding)

    def write(self, path) -> None:
        with open(path, "wb") as f:
            f.write(self.to_bytes())
