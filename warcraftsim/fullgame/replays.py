"""Demonstrations from people's replays: a Warcraft III 1.29 replay (.w3g, build 6060) of a 1v1 on
Echo Isles played back in our client, observed every 0.5 s of game time, and written as the arrays of
the built-in AI's demonstrations (collect.py: units, heroes, players, events, orders, trees, meta; one
.npz per game, games.jsonl, collect.json), so that bc.py reads them unchanged.

    python -m warcraftsim.fullgame.replays --replays runs/fullgame/human-replays-129 --out runs/fullgame/demos-human-129 --parallel 2

How (docs/reference/human-replays.md): a replay is re-simulated from the map and the recorded
actions, and the actions name units by the engine's object ids, which count every object made. The
map gets an observing harness (harness/w3obs.j) that changes nothing and makes all of its objects at
the end of main, between two probe groups; a first, short playback reads their ids (the w3shim replay
build: shim/replay.c), and the replay's later ids are shifted by their count (data.w3g.rewrite, with
the map's checksums). The second playback is the game as it was played. The game would fire no
event trigger without making objects, so the harness has none: the events and the orders (the
labels) are derived here, from the units' states (the engine's own construction, production and
upgrade flags) and from the replay's actions (each player's selection, mapped to units by their
object ids).

Checked per game ("sync" in the meta and in games.jsonl): every unit a player's subgroup action
names (its type and object ids) is a unit of that type in the playback, and the buildings the replay
places appear where it placed them.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import paths
from ..data import w3g
from ..protocol import Observation
from ..runtime import wine
from ..runtime.instance import GameError, GameInstance, GameSetup
from .collect import HERO_COLS, PLAYER_COLS, UNIT_COLS, claim_slot, game_row, write_info

STEP_SECONDS = 0.5
REPLAY_SHIM = paths.REPO_ROOT / "build" / "shim-replay"
GE_WINE = Path.home() / "wc3" / "ge" / "GE-Proton10-34" / "files" / "bin"  # (the training's Wine, if there)
ECHO = "(2)EchoIsles"
RACES = {1: "human", 2: "orc", 3: "undead", 4: "nightelf"}
# the replay build's flags (shim/replay.c): under construction is the harness's own bit (8)
PRODUCING, UPGRADING = 4096, 8192
# orders every selected unit takes (the others go to the active subgroup's units)
GENERIC_ORDERS = {851971: "smart", 851986: "move", 851983: "attack", 851990: "patrol", 851993: "holdposition",
                  851972: "stop", 851984: "attackground"}


class ReplayGame(GameInstance):
    """A game process that plays back a stock-map replay: the observer map where the replay looks for
    its map (Documents\\Warcraft III\\<its path>), the replay build of w3shim."""

    def __init__(self, setup: GameSetup, map_file: Path, map_path: str, name: str, timeout: float = 120.0):
        super().__init__(setup, name=name, timeout=timeout)
        self.map_file, self.map_path = map_file, map_path

    def _prepare(self) -> None:
        self._lock_name()
        wine.ensure_template()
        self.prefix = wine.clone_prefix(self.name)
        wine.kill_prefix(self.prefix)
        self.ipc_dir.mkdir(parents=True, exist_ok=True)
        for f in self.ipc_dir.iterdir():
            f.unlink()
        docs = wine.documents_dir(self.prefix)
        cmd_dir = docs / "CustomMapData"
        cmd_dir.mkdir(parents=True, exist_ok=True)
        link = cmd_dir / "w3sim"
        if link.is_symlink() or link.exists():
            link.unlink() if link.is_symlink() else shutil.rmtree(link)
        link.symlink_to(self.ipc_dir, target_is_directory=True)
        work = self.prefix / "drive_c" / wine.WORK_DIR
        doc_link = docs / wine.WORK_DIR
        if not doc_link.is_symlink():
            doc_link.symlink_to(work, target_is_directory=True)
        # a replay resolves its map from the Documents folder (and finds it only with its checksums)
        dest = docs.joinpath(*re.split(r"[\\/]", self.map_path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.map_file, dest)
        self._set_window_size()

    def play(self, data: bytes) -> Observation:
        """Launch the replay `data` (its first observation, at game time 0)."""
        self._playback = {}
        self._stop_process()
        if self.prefix is None:
            self._prepare()
        if not self._display:
            from ..runtime.display import Xvfb
            self._own_display = Xvfb(*self.setup.window)
            self._display = self._own_display.display
        (self.prefix / "drive_c" / wine.WORK_DIR / "replay.w3g").write_bytes(data)
        tags = self.inst_dir / "tags.txt"
        tags.unlink(missing_ok=True)
        self._loadfile = f"C:\\{wine.WORK_DIR}\\replay.w3g"
        try:
            return self._launch(attempts=1)
        finally:
            self._loadfile = None

    def tags(self) -> list[list[str]]:
        """The replay build's lines so far (shim/replay.c: P probes, D trees, T units)."""
        path = self.inst_dir / "tags.txt"
        return [line.split() for line in path.read_text().splitlines()] if path.exists() else []


# ---- which replays -------------------------------------------------------------------------------

def stock_checksums(name: str = ECHO) -> tuple[int, bytes]:
    from ..data.mapbuild import stock_map_path
    return file_checksums(stock_map_path(name))


def stock_maps(spec: str = "echo") -> dict[str, tuple[int, bytes]]:
    """The stock maps to convert and their checksums: "echo", "all" (every melee map of the game's
    Maps/FrozenThrone, Community included) or names ("(2)LastRefuge-1.3,(2)Amazonia")."""
    if spec in ("echo", "EchoIsles", ECHO):
        return {ECHO: stock_checksums()}
    if spec == "all":
        root = paths.GAME_DIR / "Maps" / "FrozenThrone"
        names = sorted(f.stem for f in [*root.glob("*.w3x"), *(root / "Community").glob("*.w3x")])
    else:
        names = [n.strip() for n in spec.split(",") if n.strip()]
    return {n: stock_checksums(n) for n in names}


def match_map(rep: w3g.Replay, stock: dict[str, tuple[int, bytes]], any_version: bool = False) -> str | None:
    """The stock map the replay was played on: its checksums (the SHA-1 when the replay has one; some
    replays carry ffffffff as the classic checksum), or with `any_version` its file name only (another
    game version's checksums cover its own common.j and Blizzard.j; another version of the map:
    NetEase's)."""
    if any_version:
        bare = lambda s: re.sub(r"^\(\d+\)", "", s.lower())  # noqa: E731 (1.26's downloads: "EchoIsles.w3x")
        name = bare(re.split(r"[\\/]", rep.map_path)[-1])
        return next((n for n in stock if bare(n + ".w3x") == name), None)
    for n, (xoro, sha1) in stock.items():
        if rep.map_sha1:
            if rep.map_sha1 == sha1.hex() and rep.map_checksum in (xoro, 0xFFFFFFFF):
                return n
        elif rep.map_checksum == xoro:
            return n
    return None


def file_checksums(path: Path) -> tuple[int, bytes]:
    """A map file's checksums as a replay stores them (w3g.map_checksums)."""
    from ..data.mapbuild import game_scripts
    from ..data.mpq import MpqArchive
    with MpqArchive(path) as m:
        files = {n: m.read(n) for n in w3g.CHECKSUM_FILES if n in m}
    common, blizzard = (p.read_bytes() for p in game_scripts())
    return w3g.map_checksums(files, common, blizzard)


def skip_reason(rep: w3g.Replay, stock: dict[str, tuple[int, bytes]] | tuple[int, bytes],
                any_version: bool = False, other_maps: bool = False) -> str | None:
    """Why a replay can't be converted (None: it can): our client plays back only its own build (unless
    `any_version`: a test), and only on a stock map of its own (`stock`: name -> checksums; a pair:
    Echo Isles') that the replay's checksums match (or with `other_maps` any version of it: by its
    file name)."""
    if isinstance(stock, tuple):
        stock = {ECHO: stock}
    h = rep.header
    if not any_version and (h.version >= 10000 or h.version != 29 or h.build != 6060):
        return f"version {h.patch} build {h.build} (the client is 1.29 build 6060)"
    if ".." in rep.map_path or ":" in rep.map_path:
        return "map path outside the maps folder"
    if match_map(rep, stock, any_version or other_maps) is None:
        name = re.split(r"[\\/]", rep.map_path)[-1]
        if any((n + ".w3x").lower() == name.lower() for n in stock):
            return f"another version of {name} (map checksum)"
        return f"map {rep.map_path}"
    playing = rep.playing
    if len(playing) != 2 or len({p.team for p in playing}) != 2:
        return f"not a 1v1 ({len(playing)} players)"
    if any(p.computer for p in playing):
        return "a computer player"
    return None


def observer_map(name: str = ECHO, step_seconds: float = STEP_SECONDS) -> Path:
    """A stock map (Echo Isles) with the observing harness (cached by the harness's source)."""
    import hashlib

    from ..data.mapbuild import HarnessConfig, build_map, observer_source
    key = hashlib.sha1((observer_source() + f"{step_seconds}").encode()).hexdigest()[:12]
    slug = "echo" if name == ECHO else re.sub(r"[^A-Za-z0-9_.-]", "", name)
    path = paths.CACHE_DIR / "maps" / f"{slug}-obs-{key}.w3x"
    if not path.exists():
        build_map(name, path, HarnessConfig(step_seconds=step_seconds), observer=True)
    return path


def shift_of(probes: list[tuple[int, int]]):
    """The object-id map from the recorded game to the playback, from the harness's probe groups (the
    first and the last object it made): every object made after them took ids one count further per
    harness object, and a slot that many further if it took a new slot (the harness's took new slots:
    checked)."""
    (i0, s0), (i1, s1) = probes
    n = s1 - s0 + 1
    if n <= 0 or i1 - i0 != s1 - s0:
        raise ValueError(f"the harness's objects did not take consecutive ids: {probes}")
    return (lambda t: (t[0] + n if t[0] >= i0 else t[0], t[1] + n if t[1] >= s0 else t[1])), n


HALLS = {int.from_bytes(c.encode(), "big") for c in ("htow", "ogre", "unpl", "etol")}


def preplaced_offsets(rep: w3g.Replay, neutral: list[tuple[float, float, int]], near: float = 32.0) -> dict:
    """How often each offset (our id - the replay's) names a preplaced unit: every object the replay's
    orders target (the first time; an order carries its target's position) matched to the one neutral
    unit of ours there (`neutral`: x, y, object id in the first observation; mines, shops, creeps)."""
    seen, offsets = set(), {}
    for a in rep.actions:
        if a.kind != w3g.ABILITY_TARGET or not a.target or a.target == w3g.NO_OBJECT or a.target[0] != a.target[1] \
                or a.target[0] in seen:
            continue
        seen.add(a.target[0])
        there = [i for x, y, i in neutral if abs(x - a.x) <= near and abs(y - a.y) <= near]
        if len(there) == 1:
            offsets[there[0] - a.target[0]] = offsets.get(there[0] - a.target[0], 0) + 1
    return offsets


def id_map(first_unit: int, preplaced_off: int, first_player: int, player_off: int):
    """The object-id map from a replay's game to ours (`first_unit`, `first_player`: our first preplaced
    unit's and first starting unit's ids; the offsets: ours - the replay's): the map's trees (made
    before its units) keep their ids, its preplaced units move by one offset, everything from the
    starting units on by the other. None when both are 0."""
    if preplaced_off == 0 and player_off == 0:
        return None
    a, b = first_unit - preplaced_off, first_player - player_off  # (in the replay's numbering)

    def f(v: int) -> int:
        return v + player_off if v >= b else v + preplaced_off if v >= a else v
    return lambda t: (f(t[0]), f(t[1]))


def version_shift(rep: w3g.Replay, obs, tags: list[list[str]]):
    """For another version's replay (--any-version) or one on another version of the map
    (--other-map-versions): the object-id map from its game to ours, when the games made a different
    number of objects before the map's preplaced units (NetEase's map versions: 3 or 5 more) or before
    the starting units (1.29 makes more: 24 players; NetEase's Echo Isles, Turtle Rock and Twisted
    Meadows ~96 more). The preplaced units' offset: the objects the replay's orders target matched to
    ours by position (one offset, or the map's units differ: an error); the starting units': a
    starting hall the replay selects (its ids there, ours in the first observation). The trees keep
    their ids. Returns the map (None when there is nothing to shift) and the two offsets."""
    ids = {}
    for t in tags:
        if t[0] == "T":
            ids.setdefault(int(t[2]), (int(t[3]), int(t[4])))
    ours, neutral = {}, []
    first_unit = first_player = 1 << 31
    for u in obs.units:
        if u.id in ids:
            first_unit = min(first_unit, ids[u.id][0])
            if u.owner >= 24:  # (neutral: creeps, mines, shops)
                if ids[u.id][0] == ids[u.id][1]:
                    neutral.append((u.x, u.y, ids[u.id][0]))
            else:
                first_player = min(first_player, ids[u.id][0])
                if u.type_id in HALLS:
                    ours[(u.owner, u.type_id)] = ids[u.id]
    offsets = set()
    for a in rep.actions:
        if a.kind == w3g.SUBGROUP and a.item in HALLS and a.target and a.time < 120000:
            mine = ours.get((rep.players[a.player].slot, a.item))
            if mine and mine[0] == mine[1] and a.target[0] == a.target[1]:
                offsets.add(mine[0] - a.target[0])
    if not offsets:
        return None, None
    if len(offsets) > 1:
        raise GameError(f"the starting halls' object ids are not one offset: {sorted(offsets)}")
    off = offsets.pop()
    pre = preplaced_offsets(rep, neutral)
    pre_off, n = max(pre.items(), key=lambda kv: kv[1], default=(0, 0))
    if n and n < 0.8 * sum(pre.values()):
        raise GameError(f"the preplaced units' object ids are not one offset (ours - the replay's: count): {pre}")
    return id_map(first_unit, pre_off, first_player, off), (pre_off, off)


# ---- the playback ------------------------------------------------------------------------------

def play(rep: w3g.Replay, data: bytes, name: str, obs_map: Path, checksums: tuple[int, bytes],
         timeout: float = 30.0, nice: int = 10, log=print, map_name: str = ECHO,
         version: tuple[int, int] | None = None, other_map: bool = False) -> dict:
    """Both playbacks of one replay; returns the raw recording: per step the units, heroes, players,
    the harness's research events, the units' object ids, the trees and theirs, the order names.
    `version` (another game version's replay) or `other_map` (on another version of the map): its
    object ids mapped to ours (version_shift)."""
    setup = GameSetup(map=map_name, step_seconds=STEP_SECONDS, nice=nice, launch_timeout=90, window=(320, 240),
                      warm_spare=False, draw=False, d3d_thread=False, render_threads=0, shim_dir=str(REPLAY_SHIM))
    g = ReplayGame(setup, obs_map, rep.map_path, name=name, timeout=timeout)
    try:
        t0 = time.time()
        obs0 = g.play(w3g.rewrite(data, checksum=checksums[0], sha1=checksums[1], version=version))
        probes = [(int(p[2]), int(p[3])) for p in g.tags() if p[0] == "P"]
        vfn, voff = version_shift(rep, obs0, g.tags()) if version is not None or other_map else (None, None)
        g._stop_process()
        if len(probes) != 2:
            raise GameError(f"the harness reported {len(probes)} probes")
        fn, n = shift_of(probes)
        if vfn is not None:
            hfn = fn
            fn = lambda t: hfn(vfn(t))  # noqa: E731
        shifted = w3g.rewrite(data, checksum=checksums[0], sha1=checksums[1], objects=fn, version=version)
        obs = g.play(shifted)
        log(f"{name}: harness objects {n} (probe {probes}){f'; ids shifted {voff}' if vfn else ''}; "
            f"playing ({time.time() - t0:.0f}s)")
        rec = {"units": [], "heroes": [], "players": [], "events": [], "ms": [],
               "trees": [(d.id, d.type_id, d.x, d.y, d.life) for d in (obs.destructables or [])]}
        rec["order_names"] = dict(obs.orders or {})
        rec["races"] = {p: RACES.get(int(s.race), "unknown") for p, s in obs.players.items()}  # (random: resolved)
        end_ms = rep.length_ms
        # the game may end before the replay does: a side's defeat ends it for the viewer (no more steps),
        # and the leave records come up to a minute or two later
        left = [lv.time for lv in rep.leaves if lv.player in {p.id for p in rep.playing}]
        over_ms = min(left, default=end_ms) - 120000
        t = 0
        while True:
            rec["ms"].append(obs.game_ms)
            for u in obs.units:
                rec["units"].append((t, u.id, u.type_id, u.owner, u.x, u.y, u.facing, u.hp, u.max_hp, u.mana,
                                     u.max_mana, u.order, int(u.flags), u.visible_to, u.resource, u.hero_level,
                                     u.hero_xp, u.skill_points))
                if u.is_hero:
                    items = (tuple(u.items) + (0,) * 6)[:6]
                    ab = (tuple(u.abilities) + ((0, 0.0),) * 4)[:4]
                    rec["heroes"].append((t, u.id, *items, *(a[0] for a in ab),
                                          *(int(round(a[1] * 10)) for a in ab)))
            for p, s in obs.players.items():
                rec["players"].append((t, p, s.gold, s.lumber, s.food_used, s.food_cap, s.upkeep, s.gold_gathered,
                                       s.lumber_gathered, s.structures, 0))
            for e in obs.events:
                rec["events"].append((t, int(e.kind), e.a, e.b, e.c))
            if obs.game_ms + STEP_SECONDS * 1000 > end_ms:
                break
            try:
                obs = g.step([])
            except GameError as e:
                if obs.game_ms > over_ms:
                    break
                raise GameError(f"{e} at {obs.game_ms / 1000:.0f} s of {end_ms / 1000:.0f}") from None
            t += 1
        rec["end_ms"] = obs.game_ms
        rec["tags"] = g.tags()
        rec["probes"], rec["shift"], rec["version_shift"] = probes, n, voff
        rec["seconds"] = time.time() - t0
        rec["shifted"] = shifted
        return rec
    finally:
        g.close()


# ---- from the recording to the demonstration ---------------------------------------------------

def remap_players(rec: dict, real: list[int]) -> None:
    """The two players as 0 and 1 (`real`: their ids in the game, in slot order), in owners, player
    rows and visibility bits (the demonstrations' players are 0 and 1; a replay's are its slots)."""
    to = {r: k for k, r in enumerate(real)}
    swap = lambda p: to.get(p, p if p >= 12 else p + 100)  # noqa: E731 (others than the two: out of the way)

    def bits(v: int) -> int:
        out = 0
        for r, k in to.items():
            if v >> r & 1:
                out |= 1 << k
        return out

    rec["units"] = [(r[0], r[1], r[2], swap(r[3]), *r[4:13], bits(r[13]), *r[14:]) for r in rec["units"]]
    rec["players"] = [(r[0], to[r[1]], *r[2:]) for r in rec["players"] if r[1] in to]
    rec["events"] = [(t, k, -1 - to[-1 - a] if k == 9 and a < 0 and -1 - a in to else a, b, c)
                     for t, k, a, b, c in rec["events"]]


def unit_tags(rec: dict) -> tuple[dict, dict]:
    """(object ids -> [(from step, unit id)]) for units (a handle's ids change when it is reused) and
    for trees (object ids -> tree id)."""
    units: dict[tuple[int, int], list] = {}
    trees = {}
    for p in rec["tags"]:
        if p[0] == "T":
            units.setdefault((int(p[3]), int(p[4])), []).append((int(p[1]) - 1, int(p[2])))
        elif p[0] == "D":
            trees[(int(p[2]), int(p[3]))] = int(p[1])
    return units, trees


def derive_orders(rep: w3g.Replay, players: list[int], ms: list[int], tag_unit: dict, tag_tree: dict,
                  units_at, kinds: dict, trees: list = ()) -> list[tuple]:
    """The labels: each player's orders as rows (step, unit, order, kind, x, y, target), as the game
    reports the built-in AI's (Observation.issued). A replay has each order once for the selection:
    orders every unit takes (smart, move, attack, ...) go to all selected units of the player; a
    build, train, research or upgrade order to one unit of the active subgroup (a worker: the one
    that is on its way to that building next step, else the first; a building: the first that is
    not producing yet, else the first); other abilities to the active subgroup's units; a hero
    skill learned to the subgroup's hero (kind 3). Items bought (an order on a shop) are no label.
    Targets: units and trees by their object ids (a tree out of sight by its type and spot, among
    `trees`: the game's (id, type, x, y, life)), else none (an item, or out of the playback)."""
    out = []
    sel: dict[int, list] = {p: [] for p in players}
    groups: dict[int, dict] = {p: {} for p in players}
    active: dict[int, int] = {}  # the active subgroup's type

    def unit_of(tag, t):
        for start, uid in reversed(tag_unit.get(tag, ())):
            if start <= t:
                return uid
        return None

    for a in rep.actions:
        p = a.player
        if p not in sel:
            continue
        if a.kind == w3g.SELECT:
            if a.flags == 1:
                sel[p] += [u for u in a.units if u not in sel[p]]
            elif a.flags == 2:
                sel[p] = [u for u in sel[p] if u not in a.units]
            continue
        if a.kind == w3g.ASSIGN_GROUP:
            groups[p][a.flags] = list(a.units)
            continue
        if a.kind == w3g.SELECT_GROUP:
            sel[p] = list(groups[p].get(a.flags, ()))
            continue
        if a.kind == w3g.SUBGROUP:
            active[p] = a.item
            continue
        if a.kind not in w3g.ORDERS:
            continue
        t = max(0, bisect.bisect_right(ms, a.time) - 1)
        here = units_at(t)  # unit id -> (type, owner, x, y, flags, order) at step t
        nxt = units_at(min(t + 1, len(ms) - 1))
        cands = [u for u in (unit_of(tag, t) for tag in sel[p]) if u is not None and u in here
                 and here[u][1] == players.index(p)]
        if not cands:
            continue
        typed = [u for u in cands if here[u][0] == active.get(p)] or cands
        order, kind, x, y, target = a.item, 0, int(a.x), int(a.y), 0
        code_kind = kinds.get(w3g.fourcc(order), ("",))[0] if order >= w3g.TYPE_CODE else ""
        if a.kind == w3g.ABILITY:
            if code_kind == "item":
                continue  # (bought: the shop's order)
            kind, x, y = (3 if code_kind == "skill" else 0), 0, 0
        elif a.kind == w3g.ABILITY_POINT:
            kind = 1
        elif a.kind in (w3g.ABILITY_TARGET, w3g.GIVE_ITEM):
            if a.target is None:
                kind = 1
            else:
                kind = 2
                target = unit_of(a.target, t) or tag_tree.get(a.target, 0)
        elif a.kind == w3g.ABILITY_TWO_POINTS:  # (on a tree out of sight: its type and spot; else the point)
            target = nearest_tree(trees, a.x2, a.y2, a.item2)
            kind, x, y = (2, int(a.x2), int(a.y2)) if target else (1, x, y)
        if code_kind in ("building", "unit", "hero", "upgrade") or kind == 3:
            if code_kind == "building" and a.kind != w3g.ABILITY:  # a worker builds: the one on its way
                going = [u for u in typed if u in nxt and nxt[u][5] == order]
                who = going[:1] or typed[:1]
            elif kind == 3:
                who = [u for u in typed if here[u][4] & 1][:1] or typed[:1]
            else:  # train, research, upgrade: one building
                idle = [u for u in typed if not here[u][4] & PRODUCING]
                who = idle[:1] or typed[:1]
        elif order in GENERIC_ORDERS:
            who = cands
        else:
            who = typed
        for u in who:
            out.append((t, u, order, kind, x, y, target))
    return out


def nearest_tree(trees, x: float, y: float, type_id: int) -> int:
    """The tree of `type_id` nearest to (x, y) of `trees` ((id, type, x, y, life)), within 128, or 0."""
    best, bd = 0, 128.0 ** 2
    for tid, typ, tx, ty, _ in trees:
        if typ == type_id:
            d = (tx - x) ** 2 + (ty - y) ** 2
            if d < bd:
                best, bd = tid, d
    return best


def derive_events(units: np.ndarray, heroes: np.ndarray, steps: int, research: list[tuple], orders: list[tuple],
                  new_at: set) -> list[tuple]:
    """The events (step, kind, a, b, c) of collect.py from the units' states step by step (the game's
    events would have needed triggers: w3obs.j): construction started (a new structure under
    construction), finished, cancelled (gone unfinished, not dead); production started and finished at
    a building, from its producing flag and the units appearing beside it (a train finished when a
    unit of its owner appears beside a producing building; started when the flag rises or a unit came
    out and it goes on; its type the building's last train or research order, as the labels have it;
    the flag falling with no unit: a cancel); upgrades started, finished (the type changed), cancelled;
    deaths (the killer unknown: 0), hero levels, summons (the summoner unknown: 0), items picked up;
    research done from the harness's polled levels (`research`: (step, 9, -1 - player, upgrade,
    level)) at the player's building last ordered to research it. `new_at`: (step, unit) of the
    units first seen at a step (the replay build's object ids: a handle reused is a new unit)."""
    C = {c: i for i, c in enumerate(UNIT_COLS)}
    F, OWN, TYP, X, Y = C["flags"], C["owner"], C["type"], C["x"], C["y"]
    ev: list[tuple] = []
    clicks: dict[int, list] = {}  # building -> [(step, order)] of its train / research orders
    for t, u, order, kind, *_ in orders:
        if kind == 0 and order >= w3g.TYPE_CODE:
            clicks.setdefault(u, []).append((t, order))
    research_at: dict[int, list] = {}
    for t, _, a, b, c in research:
        research_at.setdefault(t, []).append((-1 - a, b, c))
    by_step = np.searchsorted(units[:, 0], np.arange(steps + 2), side="left")
    h_step = np.searchsorted(heroes[:, 0], np.arange(steps + 2), side="left")

    def clicked(b: int, t: int) -> int:
        """The last train / research order the building got up to step t (0: none)."""
        best = 0
        for s, order in clicks.get(b, ()):
            if s > t:
                break
            best = order
        return best

    def kind_of(order: int, train: int, research_: int) -> int:
        return research_ if order and w3g.object_kind(w3g.fourcc(order)) == "upgrade" else train

    prev: dict[int, np.ndarray] = {}
    items: dict[int, set] = {}
    for t in range(steps + 1):
        cur = {int(r[C["id"]]): r for r in units[by_step[t]:by_step[t + 1]]}
        fresh = {i for i in cur if i not in prev or (t, i) in new_at}
        producing = {i for i, r in prev.items() if r[F] & 2 and r[F] & PRODUCING and i not in fresh}
        for i, r in cur.items():
            f, typ = int(r[F]), int(r[TYP])
            old = None if i in fresh else prev[i]
            of = int(old[F]) if old is not None else 0
            if old is None:
                if f & 2 and f & 8:
                    ev.append((t, 2, i, typ, 0))
                elif f & 256 and not f & 2:
                    ev.append((t, 17, 0, i, typ))
            else:
                if f & 2 and of & 8 and not f & 8 and not f & 1024:
                    ev.append((t, 3, i, typ, 0))
                if f & 2 and f & UPGRADING and not of & UPGRADING:
                    ev.append((t, 11, i, typ, 0))
                if f & 2 and of & UPGRADING and typ != int(old[TYP]):
                    ev.append((t, 12, i, typ, 0))
                elif f & 2 and of & UPGRADING and not f & UPGRADING:
                    ev.append((t, 13, i, typ, 0))
                if f & 2 and f & PRODUCING and not of & PRODUCING:
                    what = clicked(i, t)
                    ev.append((t, kind_of(what, 5, 8), i, what, 0))
                if f & 1 and int(r[C["hero_level"]]) > int(old[C["hero_level"]]):
                    ev.append((t, 14, i, int(r[C["hero_level"]]), 0))
            if f & 1024 and not of & 1024:
                ev.append((t, 1, i, 0, typ))
        for i, r in prev.items():  # gone unfinished, not dead: cancelled
            if i not in cur and r[F] & 2 and r[F] & 8 and not r[F] & 1024:
                ev.append((t, 4, i, 0, 0))
        ended = {i for i in producing if i in cur and not cur[i][F] & PRODUCING}
        for i in sorted(fresh):  # units out of a producing building
            r = cur[i]
            owner = int(r[OWN])
            if r[F] & (2 | 256 | 1024) or owner > 1:
                continue
            near = [b for b in producing if int(prev[b][OWN]) == owner]
            if not near:
                continue
            b = min(near, key=lambda b: (int(prev[b][X]) - int(r[X])) ** 2 + (int(prev[b][Y]) - int(r[Y])) ** 2)
            ev.append((t, 6, b, i, int(r[TYP])))
            ended.discard(b)
            if b in cur and cur[b][F] & PRODUCING:
                ev.append((t, 5, b, clicked(b, t), 0))
        for player, upgrade, level in research_at.get(t, ()):
            mine = [b for b, r in (prev | cur).items() if int(r[OWN]) == player and r[F] & 2]
            b = next((b for b in mine if clicked(b, t) == upgrade), next((b for b in mine if b in producing), 0))
            ev.append((t, 9, b, upgrade, level))
            ended.discard(b)
        for b in sorted(ended):
            what = clicked(b, t)
            ev.append((t, kind_of(what, 7, 10), b, what, 0))
        for r in heroes[h_step[t]:h_step[t + 1]]:  # items picked up
            hid = int(r[1])
            have = {int(v) for v in r[2:8] if v}
            for item in sorted(have - items.get(hid, set())):
                if hid in items:  # (a hero's first record: what it came with)
                    ev.append((t, 18, hid, item, 0))
            items[hid] = have
        prev = cur
    return ev


def type_at(units: np.ndarray) -> dict:
    """Unit id -> [(step, type)] when it changed (a handle reused, a building upgraded)."""
    C = {c: i for i, c in enumerate(UNIT_COLS)}
    out: dict[int, list] = {}
    for r in units:
        i, typ = int(r[C["id"]]), int(r[C["type"]])
        lst = out.setdefault(i, [])
        if not lst or lst[-1][1] != typ:
            lst.append((int(r[0]), typ))
    return out


def sync_check(rep: w3g.Replay, shifted: w3g.Replay, players: list[int], ms: list[int], tag_unit: dict,
               units: np.ndarray) -> dict:
    """How well the playback follows the replay: of the units the players' subgroup actions name
    (their type and object ids), how many are a unit of that type in the playback then; of the
    buildings the replay places (type and spot), how many came up within 64 of the spot (an order
    the game refused, for want of gold or room, places nothing: not all)."""
    C = {c: i for i, c in enumerate(UNIT_COLS)}
    types = type_at(units)
    named = ok = 0
    first_miss = None
    seen = set()
    for a in shifted.actions:
        if a.kind != w3g.SUBGROUP or a.player not in players or a.target is None or a.target in seen:
            continue
        seen.add(a.target)
        typ = w3g.fourcc(a.item)
        if w3g.object_kind(typ) not in ("unit", "hero", "building"):
            continue
        named += 1
        t = max(0, bisect.bisect_right(ms, a.time) - 1)
        uid = next((u for s, u in reversed(tag_unit.get(a.target, ())) if s <= t + 1), None)
        known = [ty for s, ty in types.get(uid, ()) if s <= t + 1]
        if known and a.item in (known[-1], *known[-2:]):  # (an upgrade's subgroup may still name the old type)
            ok += 1
        elif first_miss is None:
            first_miss = {"ms": a.time, "type": typ, "ids": list(a.target)}
    placed = built = 0
    structures = units[(units[:, C["flags"]] & 2) > 0]
    for a in rep.actions:
        if a.kind != w3g.ABILITY_POINT or a.player not in players or a.item < w3g.TYPE_CODE:
            continue
        if w3g.object_kind(w3g.fourcc(a.item)) != "building":
            continue
        placed += 1
        t = bisect.bisect_right(ms, a.time) - 1
        m = structures[(structures[:, C["type"]] == a.item) & (np.abs(structures[:, C["x"]] - a.x) <= 64)
                       & (np.abs(structures[:, C["y"]] - a.y) <= 64) & (structures[:, 0] >= t)]
        built += bool(len(m))
    return {"subgroup_units": named, "subgroup_matched": ok, "first_miss": first_miss,
            "buildings_placed": placed, "buildings_found": built}


def convert(path: Path, rep: w3g.Replay, data: bytes, name: str, obs_map: Path, checksums: tuple[int, bytes],
            log=print, nice: int = 10, map_name: str = ECHO, version: tuple[int, int] | None = None,
            other_map: bool = False
            ) -> tuple[dict, dict]:
    """One replay to the demonstration's arrays and its meta."""
    rec = play(rep, data, name, obs_map, checksums, nice=nice, log=log, map_name=map_name, version=version,
               other_map=other_map)
    playing = rep.playing
    real = [p.slot for p in playing]  # (a melee player's id in the game is its slot)
    remap_players(rec, real)
    as_array = lambda rows, n: np.asarray(rows, np.int64).reshape(-1, n)  # noqa: E731
    units = as_array(rec["units"], len(UNIT_COLS))
    heroes = as_array(rec["heroes"], len(HERO_COLS))
    steps = len(rec["ms"]) - 1
    tag_unit, tag_tree = unit_tags(rec)
    C = {c: i for i, c in enumerate(UNIT_COLS)}
    bounds = np.searchsorted(units[:, 0], np.arange(steps + 2), side="left")
    state = [{int(r[C["id"]]): (int(r[C["type"]]), int(r[C["owner"]]), int(r[C["x"]]), int(r[C["y"]]),
                                 int(r[C["flags"]]), int(r[C["order"]])) for r in units[bounds[t]:bounds[t + 1]]}
             for t in range(steps + 1)]
    pids = [p.id for p in playing]
    shifted = w3g.parse_replay(rec["shifted"])
    orders = derive_orders(shifted, pids, rec["ms"], tag_unit, tag_tree, lambda t: state[t], w3g.object_kinds(),
                           rec["trees"])
    new_at = {(s, u) for lst in tag_unit.values() for s, u in lst}
    events = derive_events(units, heroes, steps, [e for e in rec["events"] if e[1] == 9], orders, new_at)
    sync = sync_check(rep, shifted, pids, rec["ms"], tag_unit, units)
    units[:, C["flags"]] &= ~(PRODUCING | UPGRADING)  # (the replay build's own bits: not the demonstrations')
    summary = w3g.summarize(rep)
    team = summary["winner_team"]
    result = {str(k): ("TIE" if team is None else "VICTORY" if p.team == team else "DEFEAT")
              for k, p in enumerate(playing)}
    apm = {s["player"]: s["apm"] for s in summary["players"]}
    meta = {"races": [rec["races"].get(r, p.race) for r, p in zip(real, playing)],
            "difficulties": ["human", "human"], "handicap": playing[0].handicap, "map": map_name, "map_file": map_name,
            "victory": "melee", "step_seconds": STEP_SECONDS, "order_names": rec["order_names"],
            "result": result, "steps": steps, "game_seconds": rec["ms"][-1] / 1000.0,
            "source": {"replay": path.name, "patch": rep.header.patch, "build": rep.header.build,
                       "map_checksum": f"{rep.map_checksum or 0:08x}",
                       "winner": summary["winner_how"], "length_ms": rep.length_ms, "played_ms": rec["end_ms"],
                       "apm": [apm.get(p.id) for p in playing], "slots": real,
                       "harness_objects": rec["shift"], "probes": rec["probes"],
                       "version_shift": rec["version_shift"]},
            "derived": "orders from the replay's actions and selections; events from the units' states",
            "sync": sync, "seconds": round(rec["seconds"], 1)}
    players = as_array(rec["players"], len(PLAYER_COLS))
    last = players[:, 0] == steps  # (the game's result on its last step, as the harness reports it)
    players[last, -1] = [{"VICTORY": 1, "DEFEAT": 2, "TIE": 3}[result[str(int(p))]] for p in players[last, 1]]
    arrays = {"units": units, "heroes": heroes, "players": players,
              "events": as_array(events, 5), "orders": as_array(orders, 7), "trees": as_array(rec["trees"], 5)}
    return {k: v.astype(np.int32) for k, v in arrays.items()}, meta


# ---- the command line ----------------------------------------------------------------------------

def game_number(path: Path, taken: set) -> int:
    """The game's number (its file game<number>.npz): the replay's id when its name is one (the
    downloads': <id>.w3g), else the next free one from 1,000,000."""
    if path.stem.isdigit() and int(path.stem) not in taken:
        return int(path.stem)
    n = 1_000_000
    while n in taken:
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", type=Path, required=True, help="a directory of .w3g files (or one)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--map", default="echo",
                    help='the stock maps to convert: "echo", "all" or names ("(2)LastRefuge-1.3,(2)Amazonia")')
    ap.add_argument("--any-version", action="store_true",
                    help="a test: replays of other game versions too, their header set to 1.29 build 6060 and "
                         "their map found by its file name (does the playback stay in step?)")
    ap.add_argument("--other-map-versions", action="store_true",
                    help="1.29 replays on another version of a stock map too (NetEase's: found by its file name), "
                         "their object ids mapped to the stock map's (version_shift)")
    ap.add_argument("--parallel", type=int, default=2, help="games played back at once")
    ap.add_argument("--limit", type=int, default=0, help="at most this many conversions (0: all)")
    ap.add_argument("--min-sync", type=float, default=1.0,
                    help="games whose share of matched subgroup units is below this go to <out>/desync")
    ap.add_argument("--nice", type=int, default=10, help="the games' niceness (the training's are at 0)")
    ap.add_argument("--wine", default=os.environ.get("WARCRAFTSIM_WINE") or (str(GE_WINE) if GE_WINE.exists() else ""),
                    help="the Wine to use (default: the training's GE-Proton, if there)")
    ap.add_argument("--fsync", type=int, default=1, help="with a Proton Wine: its fsync")
    ap.add_argument("--retry-failed", action="store_true", help="convert replays that failed before again")
    args = ap.parse_args(argv)
    if args.wine:
        wine.use_wine(args.wine, fsync=bool(args.fsync))
    if not (REPLAY_SHIM / "w3shim.dll").exists():
        ap.error(f"the replay build of w3shim is missing: make -C shim replay (-> {REPLAY_SHIM})")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "desync").mkdir(exist_ok=True)
    files = sorted(args.replays.glob("*.w3g")) if args.replays.is_dir() else [args.replays]
    done: dict[str, dict] = {}  # replay -> its row (converted, skipped or failed)
    for name in ("games.jsonl", "skipped.jsonl"):
        if (out / name).exists():
            for line in (out / name).read_text().splitlines():
                r = json.loads(line)
                done[r.get("replay", "")] = r
    if args.retry_failed:  # (their rows go: a failure again writes a new one)
        done = {k: r for k, r in done.items() if not str(r.get("reason", "")).startswith("failed")}
        if (out / "skipped.jsonl").exists():
            rows = [line for line in (out / "skipped.jsonl").read_text().splitlines()
                    if not str(json.loads(line).get("reason", "")).startswith("failed")]
            (out / "skipped.jsonl").write_text("".join(r + "\n" for r in rows))
    info = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "time": time.time(),
            "status": "converting", "pid": os.getpid(), "step_seconds": STEP_SECONDS,
            "command": "python -m warcraftsim.fullgame.replays " + " ".join(sys.argv[1:] if argv is None else argv)}
    write_info(out, info)
    stock = stock_maps(args.map)
    observers: dict[str, tuple[Path, tuple[int, bytes]]] = {}  # map -> its observer map and that's checksums
    version = (29, 6060) if args.any_version else None
    lock = threading.Lock()
    taken = {int(p.stem[4:]) for p in out.rglob("game*.npz") if p.stem[4:].isdigit()}
    todo = []
    for path in files:
        if path.name in done:
            continue
        try:
            head = w3g.parse_replay(path.read_bytes(), actions=False)
            reason = skip_reason(head, stock, args.any_version, args.other_map_versions)
        except (w3g.ReplayError, ValueError, IndexError, OSError) as e:
            reason = f"unreadable: {e}"
        if reason:
            with open(out / "skipped.jsonl", "a") as f:
                f.write(json.dumps({"replay": path.name, "reason": reason}) + "\n")
            continue
        n = game_number(path, taken)
        taken.add(n)
        map_name = match_map(head, stock, args.any_version or args.other_map_versions)
        if map_name not in observers:
            obs = observer_map(map_name)
            observers[map_name] = obs, file_checksums(obs)
        todo.append((path, n, map_name))
        if args.limit and len(todo) >= args.limit:
            break
    print(f"{len(files)} replays: {len(todo)} to convert ({len(done)} done before)", flush=True)
    slot, slot_lock = claim_slot(out.parent, kind="replayconv")
    names = queue.Queue()
    for k in range(args.parallel):
        names.put(f"replayconv{slot}_{k}")
    stats = {"n": 0, "ok": 0, "t0": time.time()}

    def run(job) -> None:
        path, n, map_name = job
        obs_map, checksums = observers[map_name]
        name = names.get()
        t0 = time.time()
        try:
            data = path.read_bytes()
            rep = w3g.parse_replay(data)
            arrays, meta = convert(path, rep, data, name, obs_map, checksums, nice=args.nice,
                                   log=lambda m: print(m, flush=True), map_name=map_name, version=version,
                                   other_map=args.other_map_versions)
            sync = meta["sync"]
            share = sync["subgroup_matched"] / max(1, sync["subgroup_units"])
            dest = out / (f"game{n:05d}.npz" if share >= args.min_sync else f"desync/game{n:05d}.npz")
            tmp = dest.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
            tmp.replace(dest)
            row = game_row(n, meta, arrays)
            row.update(replay=path.name, seconds=round(time.time() - t0, 1), sync=sync,
                       desync=share < args.min_sync, apm=meta["source"]["apm"])
            with lock:
                with open(out / "games.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
                stats["ok"] += share >= args.min_sync
            print(f"{path.name} -> {dest.name} ({map_name}): {meta['races'][0]} vs {meta['races'][1]}, {meta['result']}, "
                  f"{meta['game_seconds'] / 60:.1f} min, {len(arrays['orders'])} orders, sync "
                  f"{sync['subgroup_matched']}/{sync['subgroup_units']} units, {sync['buildings_found']}/"
                  f"{sync['buildings_placed']} buildings ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001 (one game fewer; the rest go on)
            with lock, open(out / "skipped.jsonl", "a") as f:
                f.write(json.dumps({"replay": path.name, "reason": f"failed: {type(e).__name__}: {e}"}) + "\n")
            print(f"{path.name}: failed: {type(e).__name__}: {e}", flush=True)
        finally:
            names.put(name)
            with lock:
                stats["n"] += 1

    try:
        with ThreadPoolExecutor(args.parallel) as ex:
            list(ex.map(run, todo))
        info["status"] = "finished"
    except BaseException:
        info["status"] = "stopped"
        raise
    finally:
        info["finished"] = time.time()
        info["converted"] = stats["ok"]
        write_info(out, info)
        slot_lock.close()
    print(f"converted {stats['ok']} of {stats['n']} in {(time.time() - stats['t0']) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
