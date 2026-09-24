"""Wire protocol between the Python controller and the in-map JASS harness.

Observation: the harness writes one ``Preload("<token>")`` line per token to
``CustomMapData/w3sim/obs.txt``. Tokens are integers or single-letter tags. Every
record except V and X ends with a checksum token (int32 sum of its fields)::

    V version
    T seq game_ms over full
    P pid race is_agent gold lumber food_used food_cap upkeep gold_gathered
      lumber_gathered structures result start_x start_y
    O order_id ...                         (first observation of an episode)
    D hid type x y life                    (destructables, first observation)
    U hid type owner x y facing hp max_hp mana max_mana order flags vis resource
      [hero: level xp skill_points item0..item5]
    R hid                                  (unit left the game)

Unit records are deltas: a unit is only written when it changed since it was
last written, except in full snapshots (full=1: the first observation of an
episode, or after a Snapshot command). ``Observation.units`` of a parsed delta
holds only the changed units; ``merge_observation`` applies it to a unit table.
    E kind a b c                           (events since the previous observation)
    C ok                                   (one per command of the previous step)
    X                                      (end)

The game records its own resource loads into the same buffer, from another thread,
so stray entries can appear anywhere; non-numeric ones are dropped, numeric ones are
caught by the checksums (a record with one stray token is repaired, otherwise skipped).

Commands: a flat list of integers stored in the harness mailbox by the action
file (``SetPlayerTechMaxAllowed(Player(PLAYER_NEUTRAL_PASSIVE), 1048576 + i, v)``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Iterable, Sequence

PROTOCOL_VERSION = 3
MAILBOX_BASE = 1048576
COORD_BIAS = 65536
MAX_COMMAND_INTS = 8000

# Order strings resolved to ids by the harness (OrderId) in the first observation, in this order.
ORDER_NAMES: tuple[str, ...] = (
    "smart", "stop", "attack", "move", "patrol", "holdposition", "attackground", "harvest",
    "returnresources", "resumeharvesting", "repair", "setrally", "board", "unload",
    "unloadall", "load", "autoharvestgold", "autoharvestlumber",
)

def all_order_names() -> tuple[str, ...]:
    """ORDER_NAMES followed by every ability order string in the game data (needs the game files)."""
    from .data.objects import order_strings

    return ORDER_NAMES + tuple(o for o in order_strings() if o not in ORDER_NAMES)


_TOKEN_RE = re.compile(r'call Preload\( "(.*?)" \)')
_KEEP_RE = re.compile(r"-?\d+|[A-Z]")


def rawcode(value: int) -> str:
    """Integer object id -> four-character code ('hfoo')."""
    if value <= 0:
        return ""
    try:
        return value.to_bytes(4, "big").decode("latin-1")
    except OverflowError:
        return str(value)


def fourcc(code: str | int) -> int:
    """Four-character code -> integer object id."""
    if isinstance(code, int):
        return code
    if len(code) != 4:
        raise ValueError(f"not a four-character object code: {code!r}")
    return int.from_bytes(code.encode("latin-1"), "big")


class Race(IntEnum):
    UNKNOWN = 0
    HUMAN = 1
    ORC = 2
    UNDEAD = 3
    NIGHTELF = 4


class Result(IntEnum):
    PLAYING = 0
    VICTORY = 1
    DEFEAT = 2
    TIE = 3


class UnitFlags(IntFlag):
    HERO = 1
    STRUCTURE = 2
    WORKER = 4
    CONSTRUCTING = 8
    HIDDEN = 16
    LOADED = 32
    SLEEPING = 64
    PAUSED = 128
    SUMMONED = 256
    ILLUSION = 512
    DEAD = 1024
    FLYING = 2048


class EventKind(IntEnum):
    DEATH = 1  # a=dying unit, b=killer (0 if none), c=dying type
    CONSTRUCT_START = 2  # a=structure, b=type
    CONSTRUCT_FINISH = 3  # a=structure, b=type
    CONSTRUCT_CANCEL = 4  # a=structure
    TRAIN_START = 5  # a=building, b=unit type
    TRAIN_FINISH = 6  # a=building, b=trained unit, c=type
    TRAIN_CANCEL = 7  # a=building, b=unit type
    RESEARCH_START = 8  # a=building, b=tech
    RESEARCH_FINISH = 9  # a=building, b=tech, c=new level
    RESEARCH_CANCEL = 10  # a=building, b=tech
    UPGRADE_START = 11  # a=building, b=type
    UPGRADE_FINISH = 12  # a=building, b=new type
    UPGRADE_CANCEL = 13  # a=building, b=type
    HERO_LEVEL = 14  # a=hero, b=level
    SPELL_EFFECT = 15  # a=caster, b=ability, c=target unit
    TREE_DEATH = 16  # a=destructable
    SUMMON = 17  # a=summoner, b=summoned, c=type
    ITEM_PICKUP = 18  # a=unit, b=item type


@dataclass(slots=True)
class Unit:
    id: int
    type_id: int
    owner: int
    x: int
    y: int
    facing: int
    hp: int
    max_hp: int
    mana: int
    max_mana: int
    order: int
    flags: UnitFlags
    visible_to: int  # bitmask over agent players
    resource: int  # gold left in a mine
    hero_level: int = 0
    hero_xp: int = 0
    skill_points: int = 0
    items: tuple[int, ...] = ()

    @property
    def type(self) -> str:
        return rawcode(self.type_id)

    @property
    def alive(self) -> bool:
        return not (self.flags & UnitFlags.DEAD)

    @property
    def is_hero(self) -> bool:
        return bool(self.flags & UnitFlags.HERO)

    @property
    def is_structure(self) -> bool:
        return bool(self.flags & UnitFlags.STRUCTURE)

    @property
    def is_worker(self) -> bool:
        return bool(self.flags & UnitFlags.WORKER)

    @property
    def idle(self) -> bool:
        return self.order == 0

    def visible(self, player: int) -> bool:
        return bool(self.visible_to >> player & 1)

    def dist(self, x: float, y: float) -> float:
        return ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5


@dataclass(slots=True)
class PlayerState:
    id: int
    race: Race
    is_agent: bool
    gold: int
    lumber: int
    food_used: int
    food_cap: int
    upkeep: int
    gold_gathered: int
    lumber_gathered: int
    structures: int
    result: Result
    start_x: int
    start_y: int


@dataclass(slots=True)
class Destructable:
    id: int
    type_id: int
    x: int
    y: int
    life: int

    @property
    def type(self) -> str:
        return rawcode(self.type_id)


@dataclass(slots=True)
class Event:
    kind: EventKind | int
    a: int
    b: int
    c: int


@dataclass
class Observation:
    seq: int
    game_ms: int
    game_over: bool
    players: dict[int, PlayerState]
    units: list[Unit]
    events: list[Event]
    command_results: list[bool]
    destructables: list[Destructable] | None = None  # first observation of an episode only
    orders: dict[str, int] | None = None  # first observation of an episode only
    version: int = PROTOCOL_VERSION
    damaged_records: int = 0  # records lost to stray entries in the file (normally 0)
    full: bool = True  # False: `units` holds only the units that changed (see merge_observation)
    removed: list[int] = field(default_factory=list)  # units that left the game since the last observation

    @property
    def game_time(self) -> float:
        return self.game_ms / 1000.0

    @property
    def is_first(self) -> bool:
        return self.destructables is not None

    def player(self, pid: int) -> PlayerState:
        return self.players[pid]

    def units_of(self, pid: int, alive: bool = True) -> list[Unit]:
        return [u for u in self.units if u.owner == pid and (u.alive or not alive)]

    def visible_units(self, pid: int) -> list[Unit]:
        """Units the given agent player can currently see (its own included)."""
        return [u for u in self.units if u.alive and (u.owner == pid or u.visible(pid))]

    def enemies_of(self, pid: int, visible_only: bool = True) -> list[Unit]:
        """Living units of other playing players (neutrals excluded)."""
        return [
            u for u in self.units
            if u.alive and u.owner != pid and u.owner in self.players
            and (not visible_only or u.visible(pid))
        ]

    def unit(self, uid: int) -> Unit | None:
        for u in self.units:
            if u.id == uid:
                return u
        return None


class ProtocolError(ValueError):
    pass


def tokens_from_text(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text) if _KEEP_RE.fullmatch(t)]


_FIELDS = {"T": 4, "P": 14, "O": 1, "D": 5, "E": 4, "C": 1, "U": 14, "R": 1}
_HERO_EXTRA = 9


def _i32(v: int) -> int:
    return (v + 2 ** 31) % 2 ** 32 - 2 ** 31


def _record_len(tag: str, fields: list[int]) -> int:
    if tag == "U" and len(fields) > 11 and fields[11] & UnitFlags.HERO:
        return _FIELDS["U"] + _HERO_EXTRA
    return _FIELDS[tag]


def _take_record(tag: str, toks: list[str], i: int) -> tuple[list[int] | None, int]:
    """Read the record starting after its tag at toks[i]: (fields, next index) or (None, i) if damaged.

    Records end with a checksum (sum of the fields, int32). A stray numeric entry inserted by the
    game's own preload recording is removed by trying every position in a one-token-longer window.
    """
    limit = _FIELDS[tag] + (_HERO_EXTRA if tag == "U" else 0) + 2
    vals: list[int] = []
    for t in toks[i:i + limit]:
        try:
            vals.append(int(t))
        except ValueError:
            break  # the next record's tag
    base = _FIELDS[tag]
    if len(vals) > base:
        n = _record_len(tag, vals)
        if len(vals) > n and _i32(sum(vals[:n])) == vals[n]:
            return vals[:n], i + n + 1
    for k in range(len(vals)):
        cand = vals[:k] + vals[k + 1:]
        if len(cand) <= base:
            continue
        n = _record_len(tag, cand)
        if len(cand) > n and _i32(sum(cand[:n])) == cand[n] and k <= n:
            return cand[:n], i + n + 2
    return None, i


def parse_observation(text: str, order_names: Sequence[str] = ORDER_NAMES) -> Observation:
    """Parse an observation file (the script PreloadGenEnd writes)."""
    return parse_tokens(tokens_from_text(text), order_names)


def parse_token_lines(data: bytes, order_names: Sequence[str] = ORDER_NAMES) -> Observation:
    """Parse an observation sent by the w3shim DLL: the Preload tokens, one per line."""
    return parse_tokens([t for t in data.decode("latin-1").split("\n") if _KEEP_RE.fullmatch(t)], order_names)


def parse_tokens(toks: list[str], order_names: Sequence[str] = ORDER_NAMES) -> Observation:
    n = len(toks)
    i = 0
    seq = game_ms = 0
    over = False
    version = -1
    players: dict[int, PlayerState] = {}
    units: list[Unit] = []
    events: list[Event] = []
    results: list[bool] = []
    dests: list[Destructable] | None = None
    orders: list[int] | None = None
    ended = False
    damaged = 0
    full = True
    removed: list[int] = []

    while i < n:
        tag = toks[i]
        i += 1
        if tag == "X":
            ended = True
            break
        if tag == "V":
            version = int(toks[i]) if i < n and toks[i].lstrip("-").isdigit() else -1
            i += 1
            continue
        if tag not in _FIELDS:
            continue  # stray entry between records
        v, j = _take_record(tag, toks, i)
        if v is None:
            damaged += 1
            while i < n and not toks[i].isalpha():
                i += 1
            continue
        i = j
        if tag == "U":
            u = Unit(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], UnitFlags(v[11]), v[12],
                     v[13])
            if len(v) > _FIELDS["U"]:
                u.hero_level, u.hero_xp, u.skill_points = v[14], v[15], v[16]
                u.items = tuple(v[17:23])
            units.append(u)
        elif tag == "D":
            if dests is None:
                dests = []
            dests.append(Destructable(*v))
        elif tag == "E":
            try:
                kind: EventKind | int = EventKind(v[0])
            except ValueError:
                kind = v[0]
            events.append(Event(kind, v[1], v[2], v[3]))
        elif tag == "C":
            results.append(bool(v[0]))
        elif tag == "R":
            removed.append(v[0])
        elif tag == "P":
            players[v[0]] = PlayerState(v[0], Race(v[1]) if v[1] in Race._value2member_map_ else Race.UNKNOWN,
                                        bool(v[2]), v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], Result(v[11]),
                                        v[12], v[13])
        elif tag == "O":
            if orders is None:
                orders = []
            orders.append(v[0])
        elif tag == "T":
            seq, game_ms, over, full = v[0], v[1], bool(v[2]), bool(v[3])
    if not ended:
        raise ProtocolError("observation has no end marker (partial write?)")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"harness protocol {version}, expected {PROTOCOL_VERSION}")
    order_map = None
    if orders is not None:
        order_map = {name: oid for name, oid in zip(order_names, orders) if oid}
    obs = Observation(seq, game_ms, over, players, units, events, results, dests, order_map, version)
    obs.damaged_records = damaged
    obs.full = full
    obs.removed = removed
    return obs


def merge_observation(table: dict[int, Unit], obs: Observation) -> Observation:
    """Apply a (delta) observation to `table` (unit id -> Unit) and return it with all units.

    Units are kept in the order they were first seen, so feature arrays stay stable.
    """
    if obs.full:
        table.clear()
    for u in obs.units:
        table[u.id] = u
    for uid in obs.removed:
        table.pop(uid, None)
    obs.units = list(table.values())
    obs.full = True
    return obs


def read_observation(path: str | os.PathLike, order_names: Sequence[str] = ORDER_NAMES) -> Observation:
    with open(path, encoding="latin-1") as f:
        return parse_observation(f.read(), order_names)


# ---- commands ---------------------------------------------------------------------------

class Op(IntEnum):
    POINT = 1
    TARGET = 2
    IMMEDIATE = 3
    BUILD = 4
    LEARN = 5
    TARGET_DESTRUCTABLE = 6
    ITEM = 7
    SET_RESOURCES = 90
    SPAWN = 91
    QUEUE_SPAWN = 92
    CAMERA = 96
    END_GAME = 97
    SNAPSHOT = 98
    RESTART = 99


def _coord(v: float) -> int:
    c = int(round(v)) + COORD_BIAS
    if c < 0:
        raise ValueError(f"coordinate {v} out of range")
    return c


@dataclass(frozen=True)
class Command:
    """Base class: every command encodes to a list of integers."""

    def encode(self) -> list[int]:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class PointOrder(Command):
    unit: int
    order: int
    x: float
    y: float

    def encode(self) -> list[int]:
        return [Op.POINT, self.unit, self.order, _coord(self.x), _coord(self.y)]


@dataclass(frozen=True)
class TargetOrder(Command):
    unit: int
    order: int
    target: int

    def encode(self) -> list[int]:
        return [Op.TARGET, self.unit, self.order, self.target]


@dataclass(frozen=True)
class ImmediateOrder(Command):
    """Orders without a target: stop, hold position, and train / research / upgrade by object id."""
    unit: int
    order: int

    def encode(self) -> list[int]:
        return [Op.IMMEDIATE, self.unit, self.order]


@dataclass(frozen=True)
class Build(Command):
    unit: int
    building: int
    x: float
    y: float

    def encode(self) -> list[int]:
        return [Op.BUILD, self.unit, fourcc(self.building), _coord(self.x), _coord(self.y)]


@dataclass(frozen=True)
class LearnSkill(Command):
    hero: int
    ability: int

    def encode(self) -> list[int]:
        return [Op.LEARN, self.hero, fourcc(self.ability)]


@dataclass(frozen=True)
class TargetDestructable(Command):
    unit: int
    order: int
    destructable: int

    def encode(self) -> list[int]:
        return [Op.TARGET_DESTRUCTABLE, self.unit, self.order, self.destructable]


@dataclass(frozen=True)
class UseItem(Command):
    unit: int
    slot: int
    x: float | None = None
    y: float | None = None
    target: int = 0

    def encode(self) -> list[int]:
        px = _coord(self.x) if self.x is not None else 0
        py = _coord(self.y) if self.y is not None else 0
        return [Op.ITEM, self.unit, 0, self.slot, px, py, self.target]


@dataclass(frozen=True)
class SetResources(Command):
    """Debug: set a player's gold and lumber."""
    player: int
    gold: int
    lumber: int

    def encode(self) -> list[int]:
        return [Op.SET_RESOURCES, self.player, self.gold, self.lumber]


@dataclass(frozen=True)
class Spawn(Command):
    """Debug: create a unit."""
    player: int
    unit_type: int | str
    x: float
    y: float

    def encode(self) -> list[int]:
        return [Op.SPAWN, self.player, fourcc(self.unit_type), _coord(self.x), _coord(self.y)]


@dataclass(frozen=True)
class QueueSpawn(Command):
    """Scenarios: a unit for the next Restart (spawned after the old units are removed). `hp_permille`
    > 0 scales its hit points to that share of its own maximum; heroes start at `hero_level`."""
    player: int
    unit_type: str
    x: float
    y: float
    facing: float = 0.0
    hp_permille: int = 0
    hero_level: int = 1

    def encode(self) -> list[int]:
        return [Op.QUEUE_SPAWN, self.player, fourcc(self.unit_type), _coord(self.x), _coord(self.y),
                int(self.facing) % 360, int(self.hp_permille), int(self.hero_level)]


@dataclass(frozen=True)
class Camera(Command):
    """Move the local camera (watching / screenshots only; no effect on the game state)."""
    x: float
    y: float

    def encode(self) -> list[int]:
        return [Op.CAMERA, _coord(self.x), _coord(self.y)]


@dataclass(frozen=True)
class EndGame(Command):
    """End the game normally (the engine then writes Replay/LastReplay.w3g); the harness stops."""

    def encode(self) -> list[int]:
        return [Op.END_GAME]


@dataclass(frozen=True)
class Snapshot(Command):
    """Make the next observation a full snapshot of all units."""

    def encode(self) -> list[int]:
        return [Op.SNAPSHOT]


@dataclass(frozen=True)
class Restart(Command):
    """Scenario maps: remove all units and respawn the scenario (a new episode in the same game)."""

    def encode(self) -> list[int]:
        return [Op.RESTART]


def encode_commands(commands: Iterable[Command]) -> list[int]:
    out: list[int] = []
    for c in commands:
        out.extend(int(v) for v in c.encode())
    if len(out) > MAX_COMMAND_INTS:
        raise ValueError(f"too many commands for one step ({len(out)} > {MAX_COMMAND_INTS} integers)")
    return out


_OP_LENGTHS = {Op.POINT: 5, Op.TARGET: 4, Op.IMMEDIATE: 3, Op.BUILD: 5, Op.LEARN: 3, Op.TARGET_DESTRUCTABLE: 4,
               Op.ITEM: 7, Op.SET_RESOURCES: 4, Op.SPAWN: 5, Op.QUEUE_SPAWN: 8, Op.CAMERA: 3, Op.END_GAME: 1,
               Op.SNAPSHOT: 1, Op.RESTART: 1}


def decode_commands(ints: Sequence[int]) -> list[Command]:
    """Unit orders (PointOrder, TargetOrder, ImmediateOrder) and cameras in encoded commands, e.g.
    a step of a replay's command log; other commands are skipped."""
    out: list[Command] = []
    i = 0
    while i < len(ints):
        op = int(ints[i])
        n = _OP_LENGTHS.get(op)
        if n is None:
            raise ProtocolError(f"unknown command op {op} at {i}")
        a = [int(v) for v in ints[i + 1:i + n]]
        if op == Op.POINT:
            out.append(PointOrder(a[0], a[1], a[2] - COORD_BIAS, a[3] - COORD_BIAS))
        elif op == Op.TARGET:
            out.append(TargetOrder(a[0], a[1], a[2]))
        elif op == Op.IMMEDIATE:
            out.append(ImmediateOrder(a[0], a[1]))
        elif op == Op.CAMERA:
            out.append(Camera(a[0] - COORD_BIAS, a[1] - COORD_BIAS))
        i += n
    return out


def action_file_text(ints: Sequence[int]) -> str:
    """The Preloader script that stores the command integers in the harness mailbox."""
    lines = ["function PreloadFiles takes nothing returns nothing", ""]
    values = [len(ints), *ints]
    for i, v in enumerate(values):
        lines.append(f"\tcall SetPlayerTechMaxAllowed(Player(PLAYER_NEUTRAL_PASSIVE), {MAILBOX_BASE + i}, {int(v)})")
    lines += ["", "endfunction", ""]
    return "\r\n".join(lines)


def write_action_file(path: str | os.PathLike, commands: Iterable[Command] = (),
                      ints: Sequence[int] | None = None) -> None:
    """Write the action file for `commands` (or for already encoded `ints`)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="latin-1", newline="") as f:
        f.write(action_file_text(list(ints) if ints is not None else encode_commands(commands)))
    os.replace(tmp, path)
