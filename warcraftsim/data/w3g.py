"""Warcraft III replays (.w3g) of any version, in pure Python (zlib only).

    r = read_replay("game.w3g")
    r.header.patch, r.header.build, r.map_path, r.players   # "1.29", 6060, "Maps\\...", {id: Player}
    for a in r.actions: ...                                  # Action: time (ms), player, kind (action id), ...
    s = summarize(r)                                         # per player: build order, APM, leave; the winner

Read: the header (classic: 16-bit block sizes; Reforged: 32-bit; also inside the GRBN container that
Reforged 2.0 writes and warcraft3.info's downloads use: protobuf metadata, then a classic replay), the
compressed blocks, the game info (host, game name, the encoded settings string: speed, visibility,
map size, the map's checksums and path), the player and slot records (team, colour, race, computer
and its level, handicap), and the action stream: time slots with each player's actions (orders with
and without targets, selections, hotkey groups, queue cancels, ...), leave records (reason, result),
chat. Unknown actions skip the rest of their block (counted in Replay.unknown_actions).

Formats: the w3g notes by Blue and Nagger (w3g_format.txt / w3g_actions.txt) for the classic layout
(and its older variants: 1-byte ability flags before 1.13, no extra 8 bytes before 1.07, the 1-byte
subgroup before 1.14b), the Reforged changes as w3gjs (MIT) reads them.

Order targets, an order's source (the item used) and selections name units by their object ids (a
pair of dwords: a slot in the engine's table of objects and the count of objects made before it; not
JASS handles): the replay never says a unit's type, except the subgroup action (0x19), which names
the active subgroup's type and one of its units. Hence a train click's building is known only by
the selection, and what an order achieved (accepted, refused for lack of gold, a full queue) is not
in the replay at all: summarize() reports the clicks, cleaned of the obvious repeats (see there).
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

MAGIC = b"Warcraft III recorded game\x1a\x00"
GRBN = b"GRBN"

RACE_FLAGS = {0x01: "human", 0x02: "orc", 0x04: "nightelf", 0x08: "undead", 0x20: "random"}
AI_LEVELS = {0: "easy", 1: "normal", 2: "insane"}
SPEEDS = {0: "slow", 1: "normal", 2: "fast"}
# leave records: why the connection closed, and the result it reports (w3g_format.txt 5.0)
LEAVE_REASONS = {0x01: "remote", 0x0C: "local", 0x0E: "remote"}
# (0x0D: in NetEase's replays (the GRBN container) every player's leave, as "left")
LEAVE_RESULTS = {0x01: "disconnect", 0x07: "left", 0x08: "lost", 0x09: "won", 0x0A: "draw", 0x0B: "observer",
                 0x0D: "left"}

# action ids (the ones a build order and the APM use; every id of the format is parsed)
PAUSE, RESUME = 0x01, 0x02
ABILITY, ABILITY_POINT, ABILITY_TARGET, GIVE_ITEM, ABILITY_TWO_POINTS = 0x10, 0x11, 0x12, 0x13, 0x14
SELECT, ASSIGN_GROUP, SELECT_GROUP, SUBGROUP, PRE_SUBSELECT = 0x16, 0x17, 0x18, 0x19, 0x1A
SELECT_GROUND_ITEM, CANCEL_REVIVE, DEQUEUE = 0x1C, 0x1D, 0x1E
CHAT_COMMAND, ESC, SCENARIO_TRIGGER, SKILL_MENU, BUILD_MENU, PING = 0x60, 0x61, 0x62, 0x66, 0x67, 0x68
ORDERS = (ABILITY, ABILITY_POINT, ABILITY_TARGET, GIVE_ITEM, ABILITY_TWO_POINTS)
TYPE_CODE = 0x01000000  # an item/order id at or above it is a type's fourcc ('hpea'), below an order id
NO_OBJECT = (0xFFFFFFFF, 0xFFFFFFFF)  # an object id that names none


class ReplayError(ValueError):
    pass


def fourcc(v: int) -> str:
    """'hpea' from its integer (as JASS writes it: the first character in the high byte)."""
    return v.to_bytes(4, "big").decode("latin-1")


def item_name(v: int) -> str:
    """An action's item/order id as text: a type's fourcc, else the order id in decimal ("851971")."""
    return fourcc(v) if v >= TYPE_CODE and all(32 < b < 127 for b in v.to_bytes(4, "big")) else str(v)


@dataclass
class Header:
    container: str  # "w3g", or "GRBN" (a classic replay inside Reforged 2.0's container)
    header_version: int  # 0: Reign of Chaos before 1.07, 1: later
    expansion: str  # "W3XP" (The Frozen Throne) or "WAR3"
    version: int  # the patch's number: 29 for 1.29 (Reforged: 10032 for 1.32, 10200 for 2.0)
    build: int
    multiplayer: bool
    length_ms: int
    blocks: int
    wide_blocks: bool  # 32-bit block sizes (Reforged)

    @property
    def patch(self) -> str:
        """"1.29"; Reforged counts from 10000: 10036 is 1.36, 10200 is 2.0."""
        v = self.version - 10000 if self.version >= 10000 else self.version
        return f"{v // 100}.{v % 100}" if v >= 100 else f"1.{v}"


@dataclass
class Slot:
    index: int
    player: int  # player id (0: a computer or nobody)
    status: str  # empty / closed / used
    computer: bool
    team: int
    color: int
    race_flags: int
    ai: str  # a computer's level
    handicap: int

    @property
    def race(self) -> str:
        return next((n for bit, n in RACE_FLAGS.items() if self.race_flags & bit), "")


@dataclass
class Player:
    id: int  # the id actions and leave records use
    name: str
    host: bool = False
    slot: int = -1  # its slot record's index (the map's player number with fixed start spots)
    team: int = -1
    color: int = -1
    race: str = ""  # the race played: random stays "random" (the replay does not say what it became)
    random: bool = False
    computer: bool = False
    ai: str = ""
    handicap: int = 100
    observer: bool = False


@dataclass(slots=True)
class Action:
    time: int  # game time (ms) of the time slot it came in
    player: int
    kind: int  # the action id (ABILITY, SELECT, ...)
    item: int = 0  # orders, SUBGROUP, DEQUEUE: the order id or the type's fourcc (an integer, see item_name)
    flags: int = 0  # orders: the ability flags; SELECT: 1 add, 2 remove; groups: the group number; DEQUEUE: the slot
    x: float = 0.0
    y: float = 0.0
    target: tuple[int, int] | None = None  # ABILITY_TARGET (None: on the ground), GIVE_ITEM (the receiver), SUBGROUP...
    item2: int = 0  # ABILITY_TWO_POINTS: the second order; GIVE_ITEM: the item's object id (first dword)
    x2: float = 0.0
    y2: float = 0.0
    units: tuple[tuple[int, int], ...] = ()  # SELECT, ASSIGN_GROUP: object ids
    source: tuple[int, int] | None = None  # orders: the object the order comes from (the item used: "useslot")
    text: str = ""  # CHAT_COMMAND: the text; SyncStored*: "file|key|key"

    @property
    def order(self) -> str:
        return item_name(self.item)


@dataclass
class Leave:
    time: int
    player: int
    reason: str
    result: str
    reason_code: int
    result_code: int


@dataclass
class Chat:
    time: int
    player: int
    mode: int  # 0 all, 1 allies, 2 observers, 3+ to a player (3 + their slot)
    text: str


@dataclass
class Replay:
    header: Header
    game_name: str = ""
    map_path: str = ""
    map_checksum: int | None = None  # the classic (xoro) one; 0xFFFFFFFF in Reforged
    map_sha1: str | None = None  # 1.29 and later
    map_size: tuple[int, int] = (0, 0)  # playable width, height (tiles)
    creator: str = ""
    settings: dict = field(default_factory=dict)
    game_type: int = 0  # 0x01 ladder 1v1 / FFA, 0x09 custom, 0x20 ladder team, ...
    private: bool = False
    language: int = 0
    players: dict[int, Player] = field(default_factory=dict)
    slots: list[Slot] = field(default_factory=list)
    random_seed: int = 0
    select_mode: int = 0  # 0x00 teams and races selectable, ..., 0xCC automatic matchmaking
    start_spots: int = 0
    actions: list[Action] = field(default_factory=list)
    leaves: list[Leave] = field(default_factory=list)
    chat: list[Chat] = field(default_factory=list)
    length_ms: int = 0  # the sum of the time slots' increments
    unknown_actions: int = 0  # blocks cut short at an action id this parser does not know
    unknown_ids: dict = field(default_factory=dict)  # those ids: how often (-1: a block cut short inside an action)
    # for rewriting the game info (rewrite_map): the decompressed data and the encoded string's span
    data: bytes = field(default=b"", repr=False)
    settings_span: tuple[int, int] = (0, 0)

    @property
    def playing(self) -> list[Player]:
        """The players who play (not observers), by slot."""
        return sorted((p for p in self.players.values() if not p.observer), key=lambda p: p.slot)

    @property
    def matchup(self) -> str:
        """"human vs orc" (teams apart by "vs", sorted) for 1v1 and team games."""
        teams: dict[int, list[str]] = {}
        for p in self.playing:
            teams.setdefault(p.team, []).append(p.race or "?")
        return " vs ".join(sorted("+".join(sorted(t)) for t in teams.values()))


# ---- containers and blocks ----------------------------------------------------------------------

def _classic_offset(data: bytes) -> tuple[int, str]:
    if data.startswith(MAGIC):
        return 0, "w3g"
    if data.startswith(GRBN):
        # (a 128-byte header whose dword at 12 is the size of the zlib'd metadata that follows)
        guess = 128 + struct.unpack_from("<I", data, 12)[0] if len(data) >= 16 else 0
        if data[guess:guess + len(MAGIC)] == MAGIC:
            return guess, "GRBN"
        at = data.find(MAGIC, 0, 8 << 20)
        if at >= 0:
            return at, "GRBN"
    raise ReplayError("not a Warcraft III replay")


def read_header(data: bytes) -> tuple[Header, int]:
    """The header and the offset of the first compressed block."""
    base, container = _classic_offset(data)
    if len(data) < base + 64:
        raise ReplayError("truncated header")
    header_size, _compressed, header_version, _decompressed, blocks = struct.unpack_from("<5I", data, base + 28)
    if header_version == 0:  # (RoC before 1.07: a word before the version)
        _, version, build, flags, length = struct.unpack_from("<4HI", data, base + 48)
        expansion = "WAR3"
    else:
        ident, version, build, flags, length = struct.unpack_from("<4sIHHI", data, base + 48)
        expansion = ident[::-1].decode("latin-1")
    first = base + header_size
    # Reforged's blocks have 32-bit sizes: the high half of the compressed size reads as a zero
    # decompressed size in the classic layout (a classic block always decompresses to 8 KB)
    wide = len(data) >= first + 4 and struct.unpack_from("<H", data, first + 2)[0] == 0
    return Header(container, header_version, expansion, version, build, bool(flags & 0x8000), length, blocks,
                  wide), first


def decompress(data: bytes) -> tuple[Header, bytes]:
    """The header and the decompressed replay data (all blocks)."""
    header, pos = read_header(data)
    out = bytearray()
    for _ in range(header.blocks):
        if header.wide_blocks:
            if pos + 12 > len(data):
                break
            size, _full = struct.unpack_from("<II", data, pos)
            pos += 12
        else:
            if pos + 8 > len(data):
                break
            size, _full = struct.unpack_from("<HH", data, pos)
            pos += 8
        chunk = data[pos:pos + size]
        pos += size
        try:
            # (a block's stream is cut at a flush, without its end: decompressobj, not decompress)
            out += zlib.decompressobj().decompress(chunk)
        except zlib.error as e:
            if not out:
                raise ReplayError(f"block not zlib: {e}") from None
            break  # a damaged tail: keep what came before
    return header, bytes(out)


def decode_settings(enc: bytes) -> bytes:
    """The encoded settings string: in each group of 8 bytes the first holds the low bits of the
    other 7 (each was made odd, so that no byte is 0)."""
    out = bytearray()
    for i in range(0, len(enc), 8):
        mask = enc[i]
        for j, b in enumerate(enc[i + 1:i + 8], start=1):
            out.append(b if mask & (1 << j) else b - 1)
    return bytes(out)


def encode_settings(raw: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(raw), 7):
        group = raw[i:i + 7]
        mask = 1
        body = bytearray()
        for j, b in enumerate(group, start=1):
            if b % 2:
                mask |= 1 << j
                body.append(b)
            else:
                body.append(b + 1)
        out.append(mask)
        out += body
    return bytes(out)


# ---- the game info --------------------------------------------------------------------------------

class _Reader:
    __slots__ = ("b", "p")

    def __init__(self, b: bytes, p: int = 0):
        self.b, self.p = b, p

    def u8(self) -> int:
        v = self.b[self.p]
        self.p += 1
        return v

    def u16(self) -> int:
        v, = struct.unpack_from("<H", self.b, self.p)
        self.p += 2
        return v

    def u32(self) -> int:
        v, = struct.unpack_from("<I", self.b, self.p)
        self.p += 4
        return v

    def f32(self) -> float:
        v, = struct.unpack_from("<f", self.b, self.p)
        self.p += 4
        return v

    def cstr(self) -> str:
        end = self.b.index(b"\0", self.p)
        s = self.b[self.p:end].decode("utf-8", "replace")
        self.p = end + 1
        return s

    def skip(self, n: int) -> None:
        self.p += n


def _player_record(r: _Reader) -> Player:
    kind = r.u8()  # 0x00 the host, 0x16 another player
    pid = r.u8()
    name = r.cstr()
    r.skip(r.u8())  # 1: custom game (a zero), 8: ladder (runtime, race flags)
    return Player(pid, name, host=kind == 0)


def _parse_settings(rep: Replay, raw: bytes) -> None:
    r = _Reader(raw)
    if len(raw) < 13:
        return
    s0, s1, s2, s3 = raw[0], raw[1], raw[2], raw[3]
    rep.settings = {
        "speed": SPEEDS.get(s0 & 3, str(s0 & 3)),
        "visibility": {1: "hide terrain", 2: "explored", 4: "always visible", 8: "default"}.get(s1 & 0x0F, str(s1 & 0x0F)),
        "observers": {0: "none", 2: "on defeat", 3: "full", 1: "referees"}.get((s1 >> 4) & 3, ""),
        "teams_together": bool(s1 & 0x40), "lock_teams": (s2 & 0x06) == 0x06, "shared_units": bool(s3 & 1),
        "random_hero": bool(s3 & 2), "random_races": bool(s3 & 4), "referees": bool(s3 & 0x40)}
    r.skip(5)
    rep.map_size = (r.u16(), r.u16())
    rep.map_checksum = r.u32()
    rep.map_path = r.cstr()
    rep.creator = r.cstr()
    r.skip(1)
    if len(raw) - r.p >= 20:  # (1.29 on: the map's SHA-1 too)
        rep.map_sha1 = raw[r.p:r.p + 20].hex()


def _parse_info(rep: Replay, d: bytes) -> int:
    """The game info at the start of the data; returns where the replay's records begin."""
    r = _Reader(d, 4)  # (a dword: 0x110)
    host = _player_record(r)
    rep.players[host.id] = host
    rep.game_name = r.cstr()
    r.skip(1)
    start = r.p
    end = d.index(b"\0", start)
    rep.settings_span = (start, end)
    _parse_settings(rep, decode_settings(d[start:end]))
    r.p = end + 1
    r.u32()  # player count (slots)
    gt = r.u32()
    rep.game_type, rep.private = gt & 0xFF, bool(gt & 0xFF00)
    rep.language = r.u32()
    while r.p < len(d):
        b = d[r.p]
        if b == 0x16:
            p = _player_record(r)
            rep.players[p.id] = p
            r.skip(4)
        elif b in (0x38, 0x39):  # Reforged: protobuf metadata (players' accounts, skins: subtype, size, bytes)
            r.skip(2)
            r.skip(r.u32())
        else:
            break
    if r.u8() != 0x19:
        raise ReplayError("no game start record")
    size = r.u16()
    end = r.p + size
    n = r.u8()
    per = (size - 7) // n if n else 9  # 9 bytes from 1.07 on (7: 1.00, 8: 1.03)
    for i in range(n):
        b = d[r.p:r.p + per]
        r.skip(per)
        slot = Slot(i, b[0], {0: "empty", 1: "closed", 2: "used"}.get(b[2], str(b[2])), b[3] == 1, b[4], b[5], b[6],
                    AI_LEVELS.get(b[7], str(b[7])) if per >= 8 else "", b[8] if per >= 9 else 100)
        rep.slots.append(slot)
    rep.random_seed = r.u32()
    rep.select_mode = r.u8()
    rep.start_spots = r.u8()
    observer_team = 24 if n > 12 or any(s.team > 12 for s in rep.slots) else 12
    computers = 0
    for s in rep.slots:
        if s.status != "used":
            continue
        if s.computer:  # (computers have no player id: numbered after the people)
            computers += 1
            p = Player(-computers, f"Computer ({s.ai})", computer=True, ai=s.ai)
            rep.players[p.id] = p
        else:
            p = rep.players.setdefault(s.player, Player(s.player, f"player {s.player}"))
        p.slot, p.team, p.color, p.handicap = s.index, s.team, s.color, s.handicap
        p.race, p.random = s.race, bool(s.race_flags & 0x20)
        p.observer = s.team >= observer_team
    for pid in [k for k, p in rep.players.items() if p.slot < 0]:  # (a record without a slot: left the lobby)
        rep.players[pid].observer = True
    return end


# ---- the action stream -------------------------------------------------------------------------

def _actions(rep: Replay, block: bytes, pid: int, t: int, version: int, out: list[Action], base: int = 0,
             ids: list[int] | None = None) -> None:
    """One player's actions of one time slot; `ids`: also where each object id (a pair of dwords) is,
    as offsets in the data (`base`: the block's)."""
    r = _Reader(block)
    at = (lambda: ids.append(base + r.p)) if ids is not None else (lambda: None)  # noqa: E731
    fl = 2 if version >= 13 else 1  # ability flags: a word from 1.13 on
    unk = 8 if version >= 7 else 0  # two dwords after the order from 1.07 on
    n = len(block)
    try:
        while r.p < n:
            k = r.u8()
            if ABILITY <= k <= ABILITY_TWO_POINTS:
                flags = r.u16() if fl == 2 else r.u8()
                item = r.u32()
                a = Action(t, pid, k, item, flags)
                if unk:  # (the order's source object: the item of a "useslot", else none)
                    at()
                    src = (r.u32(), r.u32())
                    a.source = src if src != NO_OBJECT else None
                if k >= ABILITY_POINT:
                    a.x, a.y = r.f32(), r.f32()
                if k == ABILITY_TARGET or k == GIVE_ITEM:  # (none: a point on the ground)
                    at()
                    tgt = (r.u32(), r.u32())
                    a.target = tgt if tgt != NO_OBJECT else None
                if k == GIVE_ITEM:
                    at()
                    a.item2 = r.u32()
                    r.skip(4)
                if k == ABILITY_TWO_POINTS:
                    a.item2 = r.u32()
                    r.skip(9)
                    a.x2, a.y2 = r.f32(), r.f32()
                out.append(a)
            elif k == SELECT or k == ASSIGN_GROUP:
                mode = r.u8()
                count = r.u16()
                if ids is not None:
                    ids.extend(base + r.p + 8 * j for j in range(count))
                units = tuple(struct.unpack_from(f"<{2 * count}I", block, r.p))
                r.skip(8 * count)
                out.append(Action(t, pid, k, flags=mode, units=tuple(zip(units[::2], units[1::2]))))
            elif k == SELECT_GROUP:
                out.append(Action(t, pid, k, flags=r.u8()))
                r.skip(1)
            elif k == SUBGROUP:
                if version >= 14:
                    item = r.u32()
                    at()
                    out.append(Action(t, pid, k, item, target=(r.u32(), r.u32())))
                else:
                    out.append(Action(t, pid, k, flags=r.u8()))
            elif k in (PRE_SUBSELECT, ESC, SKILL_MENU, BUILD_MENU, 0x65, PAUSE, RESUME, 0x04, 0x05):
                out.append(Action(t, pid, k))
            elif k in (0x1B, SELECT_GROUND_ITEM):
                r.skip(1)
                at()
                out.append(Action(t, pid, k, target=(r.u32(), r.u32())))
            elif k == CANCEL_REVIVE:
                at()
                out.append(Action(t, pid, k, target=(r.u32(), r.u32())))
            elif k == DEQUEUE:
                slot = r.u8()
                out.append(Action(t, pid, k, r.u32(), slot))
            elif k == 0x03:  # game speed
                out.append(Action(t, pid, k, flags=r.u8()))
            elif k == 0x06:  # save game
                out.append(Action(t, pid, k, text=r.cstr()))
            elif k == 0x07:
                r.skip(4)
            elif k == 0x21:
                r.skip(8)
            elif k in (0x20, 0x22, 0x23, 0x24, 0x25, 0x26, 0x29, 0x2A, 0x2B, 0x2C, 0x2F, 0x30, 0x31, 0x32):
                out.append(Action(t, pid, k))  # single-player cheats
            elif k in (0x27, 0x28, 0x2D):
                r.skip(5)
                out.append(Action(t, pid, k))
            elif k == 0x2E:
                r.skip(4)
                out.append(Action(t, pid, k))
            elif k == 0x50:  # ally options: slot, flags
                slot = r.u8()
                out.append(Action(t, pid, k, r.u32(), slot))
            elif k == 0x51:  # transfer resources: slot, gold, lumber
                slot = r.u8()
                out.append(Action(t, pid, k, r.u32(), slot, item2=r.u32()))
            elif k == CHAT_COMMAND:
                r.skip(8)
                out.append(Action(t, pid, k, text=r.cstr()))
            elif k == SCENARIO_TRIGGER:
                r.skip(12 if version >= 7 else 8)
            elif k == PING:
                x, y = r.f32(), r.f32()
                r.skip(4)
                out.append(Action(t, pid, k, x=x, y=y))
            elif k in (0x69, 0x6A):  # continue game
                r.skip(16)
            elif k == 0x6B:  # SyncStoredInteger: file, mission key, key, value
                text = "|".join((r.cstr(), r.cstr(), r.cstr()))
                out.append(Action(t, pid, k, r.u32(), text=text))
            elif k == 0x6C:  # SyncStoredReal / Boolean (as w3gjs reads them): three strings and a dword
                text = "|".join((r.cstr(), r.cstr(), r.cstr()))
                out.append(Action(t, pid, k, r.u32(), text=text))
            elif k in (0x70, 0x71, 0x72, 0x73):  # SyncStored* without a value (as w3gjs reads them)
                out.append(Action(t, pid, k, text="|".join((r.cstr(), r.cstr(), r.cstr()))))
            elif k == 0x75:  # arrow key
                out.append(Action(t, pid, k, flags=r.u8()))
            elif k == 0x76:  # Reforged: mouse event (event, x, y, button)
                r.skip(1)
                x, y = r.f32(), r.f32()
                r.skip(1)
                out.append(Action(t, pid, k, x=x, y=y))
            elif k == 0x77:  # Reforged: BlzSendSyncData (prefix, data, a dword)
                out.append(Action(t, pid, k, text=r.cstr() + "|" + r.cstr()))
                r.skip(4)
            elif k == 0x78:  # Reforged: keyboard event
                r.skip(16)
            elif k == 0x7A:  # Reforged 2.0, before an order: the unit, its ability and the order it gives
                at()
                obj = (r.u32(), r.u32())
                ability = r.u32()
                out.append(Action(t, pid, k, r.u32(), target=obj, item2=ability))
            elif k in (0x92, 0x94):  # (NetEase's client: a dword, once per player at the start; meaning unknown)
                r.skip(4)
            else:
                rep.unknown_actions += 1
                rep.unknown_ids[k] = rep.unknown_ids.get(k, 0) + 1
                return
    except (IndexError, struct.error, ValueError):
        rep.unknown_actions += 1
        rep.unknown_ids[-1] = rep.unknown_ids.get(-1, 0) + 1


def parse_replay(data: bytes, actions: bool = True) -> Replay:
    """A replay from its file's bytes. `actions`: also the action stream (False: the header, the
    game info, players and slots only)."""
    header, d = decompress(data)
    rep = Replay(header, data=d)
    _records(rep, d, _parse_info(rep, d), actions)
    return rep


def _records(rep: Replay, d: bytes, pos: int, actions: bool, ids: list[int] | None = None) -> None:
    """The replay's records from `pos` on: time slots (their actions), leaves, chat. `ids`: also the
    offsets of every object id in the actions (_actions)."""
    t = 0
    version = rep.header.version if rep.header.version < 10000 else 99
    n = len(d)
    while pos < n:
        k = d[pos]
        if k == 0x1F or k == 0x1E:  # time slot: size, increment, then (player, size, actions)*
            size, inc = struct.unpack_from("<HH", d, pos + 1)
            t += inc
            p, end = pos + 5, pos + 3 + size
            if actions:
                while p + 3 <= end:
                    pid, length = d[p], struct.unpack_from("<H", d, p + 1)[0]
                    _actions(rep, d[p + 3:p + 3 + length], pid, t, version, rep.actions, p + 3, ids)
                    p += 3 + length
            pos = end
        elif k == 0x17:  # a player left
            reason, pid, result = struct.unpack_from("<IBI", d, pos + 1)
            rep.leaves.append(Leave(t, pid, LEAVE_REASONS.get(reason, hex(reason)),
                                    LEAVE_RESULTS.get(result, hex(result)), reason, result))
            pos += 14
        elif k in (0x1A, 0x1B, 0x1C):  # start of the game
            pos += 5
        elif k == 0x20:  # chat: player, size, flags, [mode], text
            pid, size = struct.unpack_from("<BH", d, pos + 1)
            flags = d[pos + 4]
            body = d[pos + 5:pos + 4 + size]
            mode = 0
            if flags == 0x20 and len(body) >= 4:
                mode, = struct.unpack_from("<I", body)
                body = body[4:]
            rep.chat.append(Chat(t, pid, mode, body.split(b"\0", 1)[0].decode("utf-8", "replace")))
            pos += 4 + size
        elif k == 0x22:  # checksum: size, bytes
            pos += 2 + d[pos + 1]
        elif k == 0x23:
            pos += 11
        elif k == 0x2F:  # forced game end countdown
            pos += 9
        elif k == 0:
            break  # (padding to the block's end)
        else:
            raise ReplayError(f"unknown record 0x{k:02x} at {pos}")
    rep.length_ms = t


def read_replay(path: str | Path, actions: bool = True) -> Replay:
    return parse_replay(Path(path).read_bytes(), actions)


def _fold(crc: int) -> int:
    return (crc ^ (crc >> 16)) & 0xFFFF


def rewrite(data: bytes, map_path: str | None = None, checksum: int | None = None, sha1: bytes | None = None,
            objects=None) -> bytes:
    """The replay file's bytes with the game info's map path and/or checksums replaced and/or every
    object id in the actions mapped by `objects((id1, id2)) -> (id1, id2)`, as a classic .w3g (a GRBN
    container's metadata is dropped). The blocks are compressed anew, with the checksums the game
    checks on loading: the header's CRC-32 (over the header with it zeroed), and each block's: the
    CRC-32s of its 8-byte header (checksum zeroed) and of its compressed bytes, each folded to 16
    bits (low and high word)."""
    header, d = decompress(data)
    if header.wide_blocks:
        raise ReplayError("rewriting Reforged replays is not supported")
    rep = Replay(header)
    records = _parse_info(rep, d)
    start, end = rep.settings_span
    raw = bytearray(decode_settings(d[start:end]))
    path_start = 13  # (settings 4, a zero, width 2, height 2, checksum 4)
    path_end = raw.index(b"\0", path_start)
    if checksum is not None:
        raw[9:13] = struct.pack("<I", checksum)
    tail = bytes(raw[path_end:])  # the path's terminator, the creator, a zero, the SHA-1
    if sha1 is not None:
        at = tail.index(b"\0", 1) + 2
        if len(tail) < at + 20 or len(sha1) != 20:
            raise ReplayError("the game info has no SHA-1 to replace")
        tail = tail[:at] + sha1 + tail[at + 20:]
    body = bytearray(d)
    if objects is not None:
        ids: list[int] = []
        _records(Replay(header), d, records, True, ids)
        for o in ids:
            tag = struct.unpack_from("<II", d, o)
            if tag != NO_OBJECT:
                struct.pack_into("<II", body, o, *objects(tag))
    path = map_path.encode("utf-8") if map_path is not None else bytes(raw[path_start:path_end])
    new = bytes(body[:start]) + encode_settings(bytes(raw[:path_start]) + path + tail) + bytes(body[end:])
    base, _ = _classic_offset(data)
    head = bytearray(data[base:base + (68 if header.header_version else 64)])
    size = struct.unpack_from("<I", head, 40)[0] + len(new) - len(d)  # (the data without the last block's padding)
    new = new[:size]
    blocks = []
    for i in range(0, len(new), 8192):
        comp = zlib.compress(new[i:i + 8192].ljust(8192, b"\0"))
        bh = struct.pack("<HHI", len(comp), 8192, 0)
        crc = _fold(zlib.crc32(bh)) | _fold(zlib.crc32(comp)) << 16
        blocks.append(bh[:4] + struct.pack("<I", crc) + comp)
    out = b"".join(blocks)
    struct.pack_into("<I", head, 32, len(head) + len(out))
    struct.pack_into("<I", head, 40, size)
    struct.pack_into("<I", head, 44, len(blocks))
    struct.pack_into("<I", head, len(head) - 4, 0)
    struct.pack_into("<I", head, len(head) - 4, zlib.crc32(bytes(head)))
    return bytes(head) + out


# files of a map that its checksums cover, in order (the first script found of the two)
CHECKSUM_FILES = ("war3map.j", "scripts\\war3map.j", "war3map.w3e", "war3map.wpm", "war3map.doo", "war3map.w3u",
                  "war3map.w3b", "war3map.w3d", "war3map.w3a", "war3map.w3q")


def _rotl3(v: int) -> int:
    return ((v << 3) | (v >> 29)) & 0xFFFFFFFF


def _xoro(data: bytes) -> int:
    v, i, n = 0, 0, len(data)
    if n > 3:  # (dwords, then the bytes left)
        for w in struct.unpack_from(f"<{n // 4}I", data):
            v = _rotl3(v ^ w)
        i = n // 4 * 4
    for b in data[i:]:
        v = _rotl3(v ^ b)
    return v


def map_checksums(files: dict[str, bytes], common_j: bytes, blizzard_j: bytes) -> tuple[int, bytes]:
    """The map's checksums as the game computes them and a replay's game info stores them (the
    classic one and the SHA-1; GHost++'s map.cpp has the algorithm): over the game's common.j and
    Blizzard.j (or the map's own copies), a constant, and the map's CHECKSUM_FILES that it has
    (`files`: name -> bytes). A replay plays back only on a map with its checksums."""
    import hashlib
    v = _rotl3(_rotl3(_xoro(common_j) ^ _xoro(blizzard_j)) ^ 0x03F1379E)
    sha = hashlib.sha1(common_j + blizzard_j + b"\x9e\x37\xf1\x03")
    script = False
    for name in CHECKSUM_FILES:
        d = files.get(name)
        if d is None or (script and name == "scripts\\war3map.j"):
            continue
        script = script or name.endswith("war3map.j")
        v = _rotl3(v ^ _xoro(d))
        sha.update(d)
    return v, sha.digest()


# ---- object types (what an order's fourcc is) -------------------------------------------------------

@lru_cache(maxsize=1)
def object_kinds() -> dict[str, tuple[str, str]]:
    """fourcc -> (kind, name) for the melee game's types: unit, hero, building, upgrade (research),
    item, skill (a hero ability). From the 1.29 game data (w3ids.json, scripts/replay_stats.py
    --ids writes it); the same codes in other versions, a few types aside."""
    try:
        text = resources.files("warcraftsim.data").joinpath("w3ids.json").read_text()
    except (FileNotFoundError, OSError):
        return {}
    return {k: tuple(v) for k, v in json.loads(text).items()}


def object_kind(code: str) -> str:
    """The kind of a type's fourcc (object_kinds), else a guess from its shape: heroes start with a
    capital and then a small letter (Hpal), hero abilities with "A" and a capital (AHbz), upgrades
    with "R"; small letters: a unit (or an item: not told apart)."""
    hit = object_kinds().get(code)
    if hit:
        return hit[0]
    if len(code) != 4:
        return ""
    if code[0] == "A" and code[1].isupper():
        return "skill"
    if code[0] == "R":
        return "upgrade"
    if code[0].isupper() and code[1].islower():
        return "hero"
    return "unit" if code[0].islower() else ""


# ---- summaries: build orders, APM, leaves, the winner -----------------------------------------------

# actions an APM counts: orders, selecting (adding: a click is a remove and an add), hotkeys, queue
# cancels, ESC, the skill and build menus. Not counted: the client's own (pre-)subselections, pings,
# syncs, removing from the selection.
APM_KINDS = frozenset((*ORDERS, ASSIGN_GROUP, SELECT_GROUP, DEQUEUE, ESC, SKILL_MENU, BUILD_MENU, 0x65))
REPEAT_MS = 1500  # the same order again within this: one click (a double click, a spammed key)
# the starting units' types: the race a random player got
RACE_OF_START = {"htow": "human", "hpea": "human", "ogre": "orc", "opeo": "orc", "unpl": "undead", "uaco": "undead",
                 "ugho": "undead", "etol": "nightelf", "ewsp": "nightelf"}


def _winner(rep: Replay) -> tuple[int | None, str]:
    """The winning team and how it was told: a "won" leave record, else the last team standing when
    every other playing player left (lost, or simply left: a surrender), else unknown."""
    playing = {p.id: p for p in rep.playing if not p.computer}
    for lv in rep.leaves:
        if lv.player in playing and lv.result == "won":
            return playing[lv.player].team, "won record"
    teams = {p.team for p in rep.playing}
    gone: list[int] = []
    for lv in rep.leaves:
        p = playing.get(lv.player)
        if p is None:
            continue
        if lv.result in ("lost", "left", "disconnect"):
            gone.append(p.id)
        left_teams = {q.team for q in playing.values() if q.id not in gone} | \
            {q.team for q in rep.playing if q.computer}
        if len(teams) > 1 and len(left_teams) == 1:
            return next(iter(left_teams)), "the others left"
    return None, "unknown"


def summarize(rep: Replay) -> dict:
    """Per playing player: the race it played (a random one's from its starting units' types), the
    build order [(ms, kind, code, x, y)] (kind: build with the spot,
    train, hero, upgrade (a building's: keep, towers), research, skill (a hero's), item (bought),
    cancel (taken from a queue)), its counts, the APM and when it left; and the game's winner.

    Cleaned of the obvious noise: the same order again within REPEAT_MS (a double click, a spammed
    key) counts once; a building placed again at the same spot (a worker re-ordered) counts once.
    Not told apart (the replay does not say): a train click refused (no gold, food, or a full
    queue) from one that trained, a hero's revival from a second hero of the type ("hero" rows
    after the first of a type), a research's next level from a repeat that the game refused."""
    kinds = object_kinds()
    per: dict[int, dict] = {}
    left = {lv.player: lv for lv in rep.leaves}
    for p in rep.playing:
        if p.computer:
            continue
        per[p.id] = {"player": p.id, "name": p.name, "slot": p.slot, "team": p.team, "race": p.race,
                     "random": p.random, "handicap": p.handicap, "build": [], "_apm": 0, "_last": {},
                     "_placed": set(),
                     "left_ms": left[p.id].time if p.id in left else None,
                     "leave": left[p.id].result if p.id in left else None}
    for a in rep.actions:
        s = per.get(a.player)
        if s is None:
            continue
        if a.kind in APM_KINDS or (a.kind == SELECT and a.flags == 1):
            s["_apm"] += 1
        if a.kind == SUBGROUP and "race_played" not in s and a.item >= TYPE_CODE:  # (a random race: its hall's, workers')
            played = RACE_OF_START.get(fourcc(a.item))
            if played:
                s["race_played"] = played
        if a.kind not in (ABILITY, ABILITY_POINT, ABILITY_TARGET, DEQUEUE) or a.item < TYPE_CODE:
            continue
        code = fourcc(a.item)
        kind = (kinds.get(code) or (object_kind(code), ""))[0]
        if a.kind == DEQUEUE:
            s["build"].append((a.time, "cancel", code, 0, 0))
            continue
        if kind == "building":
            if a.kind == ABILITY:
                row = "upgrade"  # (a building's own morph: town hall to keep, scout tower to guard tower)
            else:
                spot = (code, round(a.x / 64), round(a.y / 64))
                if spot in s["_placed"]:
                    continue
                s["_placed"].add(spot)
                s["build"].append((a.time, "build", code, int(a.x), int(a.y)))
                continue
        else:
            row = {"unit": "train", "hero": "hero", "upgrade": "research", "skill": "skill",
                   "item": "item"}.get(kind)
            if row is None or a.kind != ABILITY:
                continue
        last = s["_last"].get(code)
        s["_last"][code] = a.time
        if last is not None and a.time - last < REPEAT_MS:
            continue
        s["build"].append((a.time, row, code, 0, 0))
    length = rep.length_ms or rep.header.length_ms
    team, how = _winner(rep)
    out = []
    for s in per.values():
        minutes = ((s["left_ms"] if s["left_ms"] is not None else length) or 1) / 60000
        counts: dict[str, dict[str, int]] = {}
        for _, kind, code, *_ in s["build"]:
            counts.setdefault(kind, {})
            counts[kind][code] = counts[kind].get(code, 0) + 1
        s["apm"] = round(s.pop("_apm") / minutes, 1) if minutes > 0 else 0.0
        for k in ("_last", "_placed"):
            s.pop(k)
        s["counts"] = counts
        s.setdefault("race_played", s["race"] if s["race"] != "random" else None)
        s["won"] = None if team is None else s["team"] == team
        out.append(s)
    return {"length_ms": length, "winner_team": team, "winner_how": how, "players": out}


def strategy(player: dict, n: int = 20, until_ms: int | None = None) -> dict:
    """A player's strategy statistics from its summary (summarize()["players"][k]), as AlphaStar
    conditioned on a game's: the build order (the first `n` buildings placed, units and heroes
    trained, building upgrades and researches, workers left out) and the bag of everything made
    (counts per code: buildings, units, heroes, researches; up to `until_ms` if given)."""
    rows = [r for r in player["build"] if until_ms is None or r[0] <= until_ms]
    workers = {"hpea", "opeo", "uaco", "ewsp"}
    order = [code for _, kind, code, *_ in rows if kind in ("build", "train", "hero", "upgrade", "research")
             and code not in workers][:n]
    bag: dict[str, int] = {}
    for _, kind, code, *_ in rows:
        if kind in ("build", "train", "hero", "upgrade", "research"):
            bag[code] = bag.get(code, 0) + 1
    return {"build_order": order, "made": bag}
