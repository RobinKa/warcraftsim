"""Behavior cloning of the built-in AI on whole games (demonstrations from fullgame/collect.py).

    python3 -m warcraftsim.fullgame.bc --data runs/fullgame/demos-1 runs/fullgame/demos-2 --name fullgame-1

(the torch Python). Writes runs/bc/<name>/: bc.json, vocab.json, fit.jsonl (one row per epoch,
shown by the dashboard) and policy.pt. Games are encoded on the fly by loader workers (both
players' sides of every game; ~0.1 s per side), a few games are held out for validation.

With --memory (the network's minGRU core) batches are chunks of consecutive steps instead: lanes each
walk through whole game sides, --seq-len steps at a time, and each lane's state carries over from
its last chunk (not differentiated through: truncated backpropagation through time), so the core
learns from states like those it has when playing, built up from the game's start.

The value head learns each step's return under self-play's rewards (fullgame/selfplay.py: the
outcome, a tie's material tie-break, the material-lead shaping), from the recorded games: the
value of the built-in AI's play, a start for self-play's (which otherwise warms it up from nothing).

People's games on several maps, for a run's KL anchor (docs/reference/human-replays.md, "Cloning people"):

    python3 -m warcraftsim.fullgame.bc --like-run runs/fgself-158 --data runs/fullgame/demos-human-129 \\
        runs/fullgame/demos-human-129-maps --name human-anchor-1 --val-pct 5

(the run's settings, vocabulary and network from its --init; each game in its own frame, game_frame; each side's own
target as the strategy input, own_target; labels the run's masks rule out left out, drop_labels.)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

from . import features as fx
from . import strategy
from .model import NEG, FullGameNet, fine_head, point_parts

KEYS = ("ent", "type", "cur", "mask", "n_own", "glob", "y_order", "y_ptr", "y_x", "y_y", "ret", "avail")
REWARD = {"gamma": 0.997, "shaping": 1.0, "shaping_scale": 2000.0, "tie_break": 0.5}  # selfplay's defaults
IGNORE = -100  # y_order of a label left out (the step's mask rules its order out): cross_entropy's ignore_index


@dataclass
class Prep:
    """How each demonstration game is read for a run's policy (--like-run; people's replays on several maps):
    `frames`: each game in its own frame (game_frame: the two starts its players used, turned level), the points'
    bins over `bin_range` (the network's, fixed: every map's points share its bins); `items` (the run's --z-pool
    item slots per race): each side's own target as the strategy input (own_target), given to a share `z_share` of
    the sides (the rest zeros: self-play's games without a target), in a block of `zdim` (its modes' and mix parts
    after strategy.ZDIM: zeros); labels whose order the step's mask rules out (the run's masks, --econ-masks ...)
    or in `drop` (--neutral 2: mercenaries) are left out (IGNORE), and with `allowed` (the run's policy's table of
    the orders each unit type may get, [n_types, n_orders]) those it rules out; `smart`: people's right clicks as
    the orders the game makes of them (resolve_smart)."""
    frames: bool = False
    bin_range: float = 0.0
    items: dict | None = None
    z_share: float = 1.0
    zdim: int = 0
    drop: tuple = ()
    reward: dict = field(default_factory=lambda: dict(REWARD))
    orderable: frozenset | None = None
    mercenaries: frozenset = frozenset()
    allowed: np.ndarray | None = None
    smart: bool = False


SMART, MOVE, ATTACK = 851971, 851986, 851983
NEUTRAL_HOSTILE = 12  # (the creeps' player; 15: the neutral buildings, mines and critters)


def resolve_smart(game: dict) -> dict:
    """People's right clicks (smart: 30% of a human game's orders, which the built-in AI never gives, so a run's
    policy may not give them but to workers) as the orders the game makes of them: a unit's (not a building's) on a
    point a move, on an enemy's or a creep's unit an attack; the others (a building's rally, on its own units, trees,
    mines, shops) as they were."""
    od = game["orders"]
    at = np.flatnonzero(od[:, 2] == SMART) if len(od) else []
    if not len(at):
        return game
    u, T = game["units"], int(game["meta"]["steps"]) + 1
    sl = np.searchsorted(u[:, fx.C_STEP], np.arange(T + 1), side="left")
    od, seen = od.copy(), {}
    for i in at:
        t, unit, kind, target = int(od[i, 0]), int(od[i, 1]), int(od[i, 3]), int(od[i, 6])
        if t >= T:
            continue
        rows = seen.get(t)
        if rows is None:
            r = u[sl[t]:sl[t + 1]]
            rows = seen[t] = dict(zip(r[:, fx.C_ID].tolist(), zip(r[:, fx.C_OWNER].tolist(), r[:, fx.C_FLAGS].tolist())))
        me = rows.get(unit)
        if me is None or me[0] not in (0, 1) or me[1] & fx.STRUCTURE:
            continue
        if kind == 1:
            od[i, 2] = MOVE
        elif kind == 2 and rows.get(target, (None,))[0] in (1 - me[0], NEUTRAL_HOSTILE):
            od[i, 2] = ATTACK
    return {**game, "orders": od}


@lru_cache(None)
def map_starts(map_name: str) -> tuple[tuple[float, float], ...]:
    """All start locations of a map (its w3i's players)."""
    from ..data.mapbuild import stock_map_path
    from ..data.mpq import MpqArchive
    from ..data.w3i import parse_w3i
    with MpqArchive(stock_map_path(map_name)) as m:
        info = parse_w3i(m.read("war3map.w3i"))
    return tuple((float(p.start_x), float(p.start_y)) for p in info.players)


def used_starts(game: dict, starts) -> tuple | None:
    """The start locations players 0 and 1 began at (on a 4-player map two of four): the one nearest each player's
    buildings at the game's first step; None when a player has none or both are nearest the same."""
    u = game["units"]
    first = u[u[:, fx.C_STEP] == u[0, fx.C_STEP]] if len(u) else u
    picks = []
    for p in (0, 1):
        own = first[(first[:, fx.C_OWNER] == p) & (first[:, fx.C_FLAGS] & fx.STRUCTURE > 0)]
        if not len(own):
            return None
        x, y = float(own[:, fx.C_X].mean()), float(own[:, fx.C_Y].mean())
        picks.append(min(range(len(starts)), key=lambda i: (starts[i][0] - x) ** 2 + (starts[i][1] - y) ** 2))
    return None if picks[0] == picks[1] else (tuple(starts[picks[0]]), tuple(starts[picks[1]]))


def level_frame(game: dict, a, b) -> tuple[dict, tuple]:
    """`game` with every position (units, point and target orders, trees) and facing turned about the midpoint of
    starts a and b so that they lie on a level line, left and right (features' frame mirrors x only: it reads a map
    whose bases are left and right, as Echo Isles'), and those two starts. Level already: unchanged."""
    (ax, ay), (bx, by) = sorted([tuple(map(float, a)), tuple(map(float, b))])
    ang = math.atan2(by - ay, bx - ax)
    if abs(ang) < 1e-9:
        return game, ((ax, ay), (bx, by))
    c, s = math.cos(-ang), math.sin(-ang)
    mx, my = (ax + bx) / 2, (ay + by) / 2

    def turn(arr, cx, cy):
        x, y = arr[:, cx].astype(np.float64) - mx, arr[:, cy].astype(np.float64) - my
        arr[:, cx], arr[:, cy] = np.rint(mx + c * x - s * y), np.rint(my + s * x + c * y)

    units = game["units"].copy()
    turn(units, fx.C_X, fx.C_Y)
    units[:, fx.C_FACING] = np.mod(np.rint(units[:, fx.C_FACING] - math.degrees(ang)), 360)
    orders = game["orders"].copy()
    at = np.flatnonzero(np.isin(orders[:, 3], (1, 2)))  # (point orders; a target order's point: the target's place)
    if len(at):
        sub = orders[at]
        turn(sub, 4, 5)
        orders[at] = sub
    trees = game["trees"].copy()
    if len(trees):
        turn(trees, 2, 3)
    D = math.hypot(bx - ax, by - ay)
    return {**game, "units": units, "orders": orders, "trees": trees}, ((mx - D / 2, my), (mx + D / 2, my))


def game_frame(game: dict, bin_range: float) -> dict | None:
    """The game in its own frame (features.set_geometry: the two starts its players used, level_frame; the bins over
    `bin_range`), for encoding it next; None when its map or starts can't be told (or its bases are closer than the
    frame's squeeze allows)."""
    name = game["meta"].get("map_file") or game["meta"].get("map")
    try:
        starts = map_starts(name)
    except Exception:  # (a map not on this machine)
        return None
    used = used_starts(game, starts) if len(starts) > 2 else tuple(starts[:2])
    if used is None or len(used) < 2:
        return None
    game, (a, b) = level_frame(game, *used)
    if b[0] - a[0] <= 2 * fx.NEAR + 100:
        return None
    fx.set_geometry((a, b), bin_range)
    return game


def own_target(strategy_: dict, items: list[str], orderable_=None, mercenaries=frozenset()) -> dict | None:
    """A side's own strategy (strategy.from_demos) as the run's pool holds its strategies (strategy.main --narrow):
    its items cut to the slots and the horizon (build_pool; at least 5), narrowed to its heroes and main army
    (strategy.narrow); None when the pool would have left it out."""
    from .techtree import tree
    t = {c: round(v, 1) for c, v in strategy_["t"].items()
         if v <= strategy.HORIZON and c in items and strategy.of_race(c, strategy_["race"]) and c not in mercenaries
         and (orderable_ is None or c in orderable_)}
    if len(t) < 5:
        return None
    return strategy.narrow({**strategy_, "t": t}, tree())


RESEARCH_MAP = "(2)EchoIsles"  # (the research levels' costs of the demonstrations' maps: main sets it with the costs)
_STEPS: dict = {}


def research_levels(vocab: dict, costs) -> np.ndarray | None:
    """costs.research_steps for the encoder, as the actors give theirs (a research's next level at its own cost: without
    it the mask checked levels 2-3 at level 1's cost; audit 18); none without the costs."""
    if costs is None:
        return None
    key = (id(vocab), RESEARCH_MAP)
    if key not in _STEPS:
        from .costs import research_steps
        _STEPS[key] = research_steps(vocab, RESEARCH_MAP)
    return _STEPS[key]


def drop_labels(out: dict, drop=(), allowed: np.ndarray | None = None) -> int:
    """Labels of orders the step's mask rules out (avail; as self-play's masks, which the recorded play didn't
    follow), in `drop`, or (`allowed`, the run's policy's [n_types, n_orders]) its unit's type may not get: left out
    (y_order IGNORE, no target), in place; how many. (Self-play's anchor plays with the policy's table: an order
    outside it would only take weight from the others.)"""
    y = out["y_order"].astype(np.int64)
    T = len(y)
    bad = (y > 0) & ~out["avail"][np.arange(T)[:, None], np.clip(y, 0, None)]
    if len(drop):
        bad |= np.isin(y, np.asarray(drop, np.int64))
    if allowed is not None:  # (own units first: the labels' columns are the first entities')
        typ = np.clip(out["type"][:, :y.shape[1]].astype(np.int64), 0, len(allowed) - 1)
        bad |= (y > 0) & ~allowed[typ, np.clip(y, 0, allowed.shape[1] - 1)]
    out["y_order"][bad] = IGNORE
    for k in ("y_ptr", "y_x", "y_y"):
        out[k][bad] = -1
    # a point past the frame's range is clipped to the edge bin: its point label taught "the edge" (on the 4-player maps
    # with neighbouring starts, up to 14% of a side's points; audit 18): the order kept, its point left out
    last = fx.BINS * max(int(getattr(fx, "FINE", 1)), 1) - 1
    edge = ((out["y_x"] == 0) | (out["y_x"] == last) | (out["y_y"] == 0) | (out["y_y"] == last)) & (out["y_x"] >= 0)
    out["y_x"][edge] = -1
    out["y_y"][edge] = -1
    return int(bad.sum())


def game_sides(path: Path, enc: fx.Encoder, values: dict, prep: Prep, rng: random.Random, winners: bool = False,
               max_minutes: float = 0.0) -> list[dict]:
    """A demonstration game's sides as `prep` reads them (side_data each): none when its frame can't be told."""
    game = fx.load_game(path)
    if prep.frames:
        game = game_frame(game, prep.bin_range)
        if game is None:
            return []
    if prep.smart:
        game = resolve_smart(game)
    own = {}
    if prep.items is not None:
        own = {s["player"]: own_target(s, prep.items[s["race"]], prep.orderable, prep.mercenaries)
               for s in strategy.from_demos(Path(path)) if s["race"] in prep.items}
    out = []
    for player in sides_of(game, winners, max_minutes):
        z = None
        if prep.items is not None:
            z = np.zeros(prep.zdim, np.float32)
            st = own.get(player)
            if st is not None and rng.random() < prep.z_share:
                z[:strategy.ZDIM] = strategy.vector(st, prep.items[st["race"]])
        out.append(side_data(enc, game, player, values, z, prep))
    return out


def returns(game: dict, player: int, values: dict, reward: dict = REWARD) -> np.ndarray:
    """[steps]: the discounted return from each step of `player`'s side, with self-play's rewards:
    the shaping gamma * phi(s') - phi(s) (phi: the material lead over shaping_scale), and at the last
    step the outcome (a tie: tie_break * tanh(2 * lead)) minus phi (the potential back to 0)."""
    from .trace import material_steps
    T = game["meta"]["steps"] + 1
    m = material_steps(game["units"], T, values)
    phi = (m[:, player] - m[:, 1 - player]) / reward["shaping_scale"]
    res = (game["meta"].get("result") or {}).get(str(player), "TIE")
    outcome = 1.0 if res == "VICTORY" else -1.0 if res == "DEFEAT" else reward["tie_break"] * np.tanh(2.0 * phi[-1])
    gamma, k = reward["gamma"], reward["shaping"]
    r = k * (gamma * phi[1:] - phi[:-1])  # steps 0 .. T-2: into the next state
    r = np.append(r, 0.0)
    r[-1] += outcome - k * phi[-1]  # (the last step, the end: as self-play's last acted step)
    out = np.zeros(T, np.float32)
    g = 0.0
    for t in range(T - 1, -1, -1):
        g = r[t] + gamma * g
        out[t] = g
    return out


def side_data(enc: fx.Encoder, game: dict, player: int, values: dict, z: np.ndarray | None = None,
              prep: Prep | None = None) -> dict:
    """A game from `player`'s side: the encoded steps and their returns. In a takeover game
    (collect.py --policy) the side a policy played until the built-in AI took it over starts at
    the takeover: the steps before it have the policy's orders, not the AI's (the encoder still runs
    through them, for what the player queued and researched). `prep` (a run's): its rewards, and the labels
    its masks rule out left out (drop_labels)."""
    out = enc.encode(game, player, z)
    out["ret"] = returns(game, player, values, prep.reward if prep is not None else REWARD)
    if prep is not None:
        drop_labels(out, prep.drop, prep.allowed)
    tk = game["meta"].get("takeover")
    if tk and tk["player"] == player and tk["step"] > 0:
        out = {k: v[tk["step"]:] for k, v in out.items()}
    return out


def sides_of(game: dict, winners: bool = False, max_minutes: float = 0.0) -> tuple[int, ...]:
    """The sides of a game to learn from: both; with `winners` only the side that won (none for a
    tie); with `max_minutes` only games decided within that many minutes of the start, or of the
    takeover in a takeover game. (As AlphaStar fine-tuned its supervised policy on winning replays:
    on duelfast the policy tied 80-95% of its games and the built-in AI's short wins show it
    finishing them.) The side the built-in AI advised (collect.py --shadow) always counts: its labels
    are the teacher's in the policy's own states, whoever won."""
    if not winners and not max_minutes:
        return (0, 1)
    meta = game["meta"]
    sh = meta.get("shadow")
    if sh:
        return tuple(sorted({sh["player"], *sides_of({"meta": {k: v for k, v in meta.items() if k != "shadow"}},
                                                      winners, max_minutes)}))
    if max_minutes:
        seconds = meta.get("game_seconds") or 0.0
        tk = meta.get("takeover")
        if tk and tk["step"] < (meta.get("steps") or 0):
            seconds -= tk["step"] * meta.get("step_seconds", 0.5)
        if seconds > 60.0 * max_minutes:
            return ()
    if not winners:
        return (0, 1)
    res = meta.get("result") or {}
    return tuple(p for p in (0, 1) if res.get(str(p)) == "VICTORY")


def as_is(batch):
    """A loader's collate_fn for batches that are complete already (Steps with arrays)."""
    return batch


class Steps(torch.utils.data.IterableDataset):
    """Shuffled batches of steps from the games (both sides), encoded by the loader workers."""

    def __init__(self, paths: list[Path], vocab: dict, batch: int, values: dict, costs=None, buffer_steps: int = 8192,
                 seed: int = 0, arrays: bool = False, winners: bool = False, max_minutes: float = 0.0,
                 strategies: dict | None = None, prep: Prep | None = None):
        """`arrays`: numpy batches (with a loader's collate_fn=as_is they cross to its process as bytes:
        a tensor crosses as a shared-memory file of its own, ~15 ms for a batch's twelve). `strategies`
        (a strategy pool, fullgame/strategy.py): each side's own strategy as the input (strategy.hindsight;
        the sides without one left out). `prep`: the games read for a run's policy instead (game_sides)."""
        self.paths, self.vocab, self.batch, self.buffer_steps, self.seed = paths, vocab, batch, buffer_steps, seed
        self.values, self.costs, self.arrays = values, costs, arrays
        self.strategies, self.prep = strategies, prep
        self.zdim = fx.ZDIM  # (the loader's workers: a process started anew reads the strategy block's size here)
        self.winners, self.max_minutes = winners, max_minutes  # (sides_of)
        self.epoch = 0

    def __iter__(self):
        """A shuffle buffer of `buffer_steps` steps (~18 games), kept full: a side read pays out as many
        steps as it brings, each step once. (Filling a buffer of 16 games and then emptying it made no
        batch for seconds at a time, and the loader takes its workers' batches in turn: self-play's
        learner waited for them 26% of its update.)"""
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1000 + self.epoch * 100 + wid)
        paths = self.paths[wid::nw]
        rng.shuffle(paths)
        fx.set_z(self.zdim)
        enc = fx.Encoder(self.vocab, self.costs, research_levels(self.vocab, self.costs))
        pick = np.random.default_rng(rng.randrange(1 << 30))
        cap = max(self.buffer_steps, self.batch)
        buf: dict[str, np.ndarray] = {}
        used = np.zeros(cap, bool)

        def batch():
            idx = pick.choice(np.flatnonzero(used), self.batch, replace=False)
            used[idx] = False
            return {k: buf[k][idx] if self.arrays else torch.from_numpy(buf[k][idx]) for k in KEYS}

        def sides(p):
            if self.prep is not None:
                yield from game_sides(p, enc, self.values, self.prep, rng, self.winners, self.max_minutes)
                return
            game = fx.load_game(p)
            own = strategy.hindsight(self.strategies, p, game["meta"]) if self.strategies is not None else None
            for player in sides_of(game, self.winners, self.max_minutes):
                if own is not None and player not in own:
                    continue
                z = strategy.vector(own[player], self.strategies["items"][own[player]["race"]]) if own else None
                yield side_data(enc, game, player, self.values, z)

        for p in paths:
            for out in sides(p):
                if not buf:
                    buf = {k: np.zeros((cap,) + out[k].shape[1:], out[k].dtype) for k in KEYS}
                n, at = len(out["n_own"]), 0
                while at < n:
                    free = np.flatnonzero(~used)[:n - at]
                    if not len(free):
                        yield batch()
                        continue
                    for k in KEYS:
                        buf[k][free] = out[k][at:at + len(free)]
                    used[free] = True
                    at += len(free)
        while used.sum() >= self.batch:
            yield batch()


class Sequences(torch.utils.data.IterableDataset):
    """Chunks of consecutive steps: `lanes` game sides at a time, `seq_len` steps of each per batch
    (lane-major: [lanes * seq_len] steps), with "starts" [lanes, seq_len] (a side's first step: the
    state resets) and "lane" [lanes] (ids across the loader workers, for the carried states). A lane
    goes on with a new side where its side ends."""

    def __init__(self, paths: list[Path], vocab: dict, lanes: int, seq_len: int, values: dict, costs=None, seed: int = 0,
                 winners: bool = False, max_minutes: float = 0.0, prep: Prep | None = None):
        self.paths, self.vocab, self.lanes, self.seq_len, self.seed = paths, vocab, lanes, seq_len, seed
        self.values, self.costs = values, costs
        self.winners, self.max_minutes = winners, max_minutes
        self.prep = prep
        self.zdim = fx.ZDIM
        self.epoch = 0

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1000 + self.epoch * 100 + wid)
        paths = self.paths[wid::nw]
        rng.shuffle(paths)
        fx.set_z(self.zdim)
        enc = fx.Encoder(self.vocab, self.costs, research_levels(self.vocab, self.costs))
        games, pending = iter(paths), []

        def side():  # the next game side's arrays, or None when the games are used up
            while not pending:  # (a game can give no side: a tie, with --winners-only)
                p = next(games, None)
                if p is None:
                    return None
                if self.prep is not None:
                    pending.extend(game_sides(p, enc, self.values, self.prep, rng, self.winners, self.max_minutes))
                    continue
                g = fx.load_game(p)
                for player in sides_of(g, self.winners, self.max_minutes):
                    pending.append(side_data(enc, g, player, self.values))
            return pending.pop(0)

        L, T = self.lanes, self.seq_len
        cur = [side() for _ in range(L)]
        pos = [0] * L
        while all(c is not None for c in cur):
            parts: dict[str, list] = {k: [] for k in KEYS}
            starts = np.zeros((L, T), bool)
            for i in range(L):
                t = 0
                while t < T:
                    if pos[i] >= len(cur[i]["n_own"]):
                        cur[i], pos[i] = side(), 0
                        if cur[i] is None:
                            return
                    if pos[i] == 0:
                        starts[i, t] = True
                    n = min(T - t, len(cur[i]["n_own"]) - pos[i])
                    for k in KEYS:
                        parts[k].append(cur[i][k][pos[i]:pos[i] + n])
                    pos[i] += n
                    t += n
            b = {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}
            b["starts"] = torch.from_numpy(starts)
            b["lane"] = torch.arange(wid * L, (wid + 1) * L)
            yield b


def whole_sides(paths: list[Path], vocab: dict, values: dict, costs=None, sides: int = 4, winners: bool = False,
                max_minutes: float = 0.0, prep: Prep | None = None):
    """Validation: every step of every game side, `sides` sides per batch as sequences padded to the
    longest ("valid" marks the steps; lane-major as Sequences, each lane starting at its side's start).
    The same batches for every model, with memory or without. `prep`: as game_sides reads them (each side
    given its target: the same inputs every epoch)."""
    enc = fx.Encoder(vocab, costs, research_levels(vocab, costs))
    todo = []
    for p in paths:
        if prep is not None:
            todo += game_sides(p, enc, values, Prep(**{**prep.__dict__, "z_share": 1.0}), random.Random(0), winners,
                               max_minutes)
            continue
        g = fx.load_game(p)
        todo += [side_data(enc, g, player, values) for player in sides_of(g, winners, max_minutes)]
    for a in range(0, len(todo), sides):
        group = todo[a:a + sides]
        T = max(len(x["n_own"]) for x in group)
        parts: dict[str, list] = {k: [] for k in KEYS}
        valid = np.zeros((len(group), T), bool)
        for i, x in enumerate(group):
            n = len(x["n_own"])
            valid[i, :n] = True
            for k in KEYS:
                pad = np.zeros((T - n,) + x[k].shape[1:], x[k].dtype)
                if k in ("y_ptr", "y_x", "y_y"):
                    pad[:] = -1
                elif k == "mask":
                    pad[:, 0] = True  # (a padding step: one entity, or attention over nothing gives NaNs)
                parts[k].append(np.concatenate([x[k], pad]))
        b = {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}
        b["starts"] = torch.from_numpy(np.eye(1, T, dtype=bool).repeat(len(group), 0))
        b["lane"] = torch.arange(len(group))
        b["valid"] = torch.from_numpy(valid.reshape(-1))
        yield b


def allowed_orders(paths: list[Path], vocab: dict) -> np.ndarray:
    """[n_types, n_orders]: the order classes each unit type got in the demonstrations."""
    enc = fx.Encoder(vocab)
    allowed = np.zeros((enc.n_types, enc.n_orders), bool)
    allowed[0] = True
    allowed[:, 0] = True
    for p in paths:
        g = fx.load_game(p)
        types = {int(r[1]): enc.type_index.get(int(r[2]), 0) for r in g["units"]}
        trees = set(g["trees"][:, 0].tolist())
        ids = set(types)
        for r in g["orders"]:
            lab = fx.relabel(int(r[2]), int(r[3]), int(r[6]) in ids, int(r[6]) in trees)
            c = enc.order_index.get(lab) if lab is not None else None
            if c is not None:
                allowed[types.get(int(r[1]), 0), c] = True
    return allowed


def losses(net: FullGameNet, b: dict, device, states: torch.Tensor | None = None,
           value_coef: float = 0.5, avail_mask: bool = True, stats: bool = True, width: int | None = None,
           acting_only: bool = False, ref: FullGameNet | None = None, ref_states: torch.Tensor | None = None,
           hold_start: float = 0.0, hold_all: float = 0.0) -> tuple[torch.Tensor, dict]:
    """The loss of a batch and its statistics (`stats` False: none, and a dozen waits for the GPU
    fewer). With memory, `states` [lanes, d] holds each lane's state after its last chunk (read, and
    updated with the batch's). `width`: the entities to look at (default: the batch's widest view;
    a CUDA graph needs it given: finding it on the GPU is a wait). `acting_only` (--acting-only): the order's
    loss only for the units that got one, over the orders without "none" (class 0 out of the softmax: no
    gradient to it), so what to order is learned and whether to (people order a unit in ~5% of its steps)
    is left as it was; the targets' losses as before (they had labels only with an order).
    `ref` (a fine-tune's start network, frozen; with memory its lanes' states in `ref_states`): `hold_start` times
    KL(start || network) of the act/no-act decision per own unit (Bernoulli on P(none), the orders as played: the
    unit type's table), `hold_all` times the forward KL over every head (model.anchor_kl's forward-all: the orders,
    and the target unit, x and y given the label's order and x for the units that got one), on the batch's states."""
    E = width or int(b["mask"].any(0).nonzero().max()) + 1  # the batch's widest view (before the copy: no wait for the GPU)
    b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    O = min(fx.MAX_OWN, E)
    ent, typ, cur, mask = b["ent"][:, :E].float(), b["type"][:, :E].long(), b["cur"][:, :E].long(), b["mask"][:, :E]
    y_order, y_ptr = b["y_order"][:, :O].long(), b["y_ptr"][:, :O].long()
    # (--fine-points: the labels are point indices, the bins and the sub-cell; y_f None without, or for a network
    # without a fine head of their size)
    y_x, y_y, y_f = point_parts(b["y_x"][:, :O].long(), b["y_y"][:, :O].long())
    y_f = y_f if y_f is not None and fine_head(net) else None
    n_own = b["n_own"].long()
    g, u = net.encode(ent, typ, cur, mask, b["glob"].float())
    if net.memory and "lane" in b:
        L = len(b["lane"])
        c, h = net.context_seq(g.view(L, -1, g.shape[-1]), states[b["lane"]], b["starts"])
        states[b["lane"]] = h[:, -1].detach().to(states.dtype)
        g = c.reshape(g.shape)
    elif net.memory:  # shuffled steps (self-play's auxiliary cloning loss): each as if a game's first
        g = net.context(g, None)[0]
    avail = b["avail"] if avail_mask else None  # (as when playing: what the player can pay for)
    logits = net.order_logits(g, u[:, :O], typ, n_own, by_type=False, avail=avail)  # own units come first
    own = (torch.arange(O, device=device)[None] < n_own[:, None]) & (y_order != IGNORE)  # (IGNORE: drop_labels')
    if not stats:  # (self-play's auxiliary loss: masked means, not boolean indexing; each was a GPU sync)
        w = own.float()
        if net.training:  # the orders each unit type gets (padding and "none" set class 0: always allowed)
            y_flat = y_order.clamp(min=0).reshape(-1)  # (a GPU tensor of True: a Python True is a copy from the CPU, not capturable)
            net.allowed.index_put_((typ[:, :O].reshape(-1), y_flat), torch.ones_like(y_flat, dtype=torch.bool))
        ptr, xl, z = net.target_logits(g, u, mask, y_order.clamp(min=0))
        yl = net.y_logits(z, y_x)
        l_pt = _masked_ce(xl, y_x, w) + _masked_ce(yl, y_y, w)
        if y_f is not None:  # (the sub-cell given the true bins)
            l_pt = l_pt + _masked_ce(net.fine_logits(z, y_x, y_y), y_f, w)
        l_order = (_masked_ce(acting(logits), y_order, w * (y_order > 0).float()) if acting_only
                   else _masked_ce(logits, y_order, w))
        loss = l_order + _masked_ce(ptr, y_ptr, w) + 0.5 * l_pt
        if value_coef:
            value, ret = net.value(g, b["glob"].float()).float(), b["ret"].float()
            if "valid" in b:
                v = b["valid"].float()
                loss = loss + value_coef * 0.5 * (((value - ret) ** 2) * v).sum() / v.sum().clamp(min=1)
            else:
                loss = loss + value_coef * 0.5 * ((value - ret) ** 2).mean()
        return loss, {}
    if acting_only:  # (the orders given one: "none" labels and the none class left out)
        given = own & (y_order > 0)
        l_order = F_.cross_entropy(acting(logits)[given], y_order[given]) if given.any() else logits.sum() * 0
    else:
        l_order = F_.cross_entropy(logits[own], y_order[own], ignore_index=IGNORE)  # (IGNORE: a label left out, drop_labels)
    if net.training:  # the orders each unit type gets (the mask for playing)
        issued_ = own & (y_order > 0)
        net.allowed[typ[:, :O][issued_], y_order[issued_]] = True
    ptr, xl, z = net.target_logits(g, u, mask, y_order.clamp(min=0))
    yl = net.y_logits(z, y_x)  # given the true x (teacher forcing)
    has_ptr, has_pt = (y_ptr >= 0) & own, (y_x >= 0) & own
    l_ptr = F_.cross_entropy(ptr[has_ptr], y_ptr[has_ptr]) if has_ptr.any() else logits.sum() * 0
    l_pt = (F_.cross_entropy(xl[has_pt], y_x[has_pt]) + F_.cross_entropy(yl[has_pt], y_y[has_pt])
            if has_pt.any() else logits.sum() * 0)
    if y_f is not None and has_pt.any():  # (--fine-points: the sub-cell given the true bins)
        l_pt = l_pt + F_.cross_entropy(net.fine_logits(z, y_x, y_y)[has_pt], y_f[has_pt])
    value, ret = net.value(g, b["glob"].float()).float(), b["ret"].float()
    if "valid" in b:  # (padded steps: no value target)
        value, ret = value[b["valid"]], ret[b["valid"]]
    l_value = 0.5 * ((value - ret) ** 2).mean()
    play = played(net, logits, typ[:, :O])  # (as the policy plays: the unit type's orders)
    l_hold, hold = logits.sum() * 0, {}
    if ref is not None and (hold_start or hold_all):  # (the start network's view of the same states, frozen)
        with torch.no_grad():
            g_r, u_r = ref.encode(ent, typ, cur, mask, b["glob"].float())
            if ref.memory and "lane" in b:
                c_r, h_r = ref.context_seq(g_r.view(len(b["lane"]), -1, g_r.shape[-1]), ref_states[b["lane"]],
                                           b["starts"])
                ref_states[b["lane"]] = h_r[:, -1].to(ref_states.dtype)
                g_r = c_r.reshape(g_r.shape)
            elif ref.memory:
                g_r = ref.context(g_r, None)[0]
            play_r = played(net, ref.order_logits(g_r, u_r[:, :O], typ, n_own, by_type=False, avail=avail), typ[:, :O])
        units = (torch.arange(O, device=device)[None] < n_own[:, None]).float()  # (every own unit, labelled or not)
        n_u = units.sum().clamp(min=1)
        k_act = (act_kl(play_r, play) * units).sum() / n_u
        k_all = k_act * 0
        if hold_all:
            from .model import anchor_kl
            with torch.no_grad():
                ptr_r, xl_r, z_r = ref.target_logits(g_r, u_r, mask, y_order.clamp(min=0))
                ev_r = {"logits": play_r, "ptr": ptr_r, "xl": xl_r, "yl": ref.y_logits(z_r, y_x)}
                if y_f is not None and fine_head(ref):
                    ev_r["fl"] = ref.fine_logits(z_r, y_x, y_y)
            ev = {"logits": play, "ptr": ptr, "xl": xl, "yl": yl, "uses_ptr": has_ptr.float(), "uses_pt": has_pt.float()}
            if "fl" in ev_r:
                ev["fl"] = net.fine_logits(z, y_x, y_y)
            k_all = (anchor_kl(ev, ev_r, "forward-all") * units).sum() / n_u
        l_hold = hold_start * k_act + hold_all * k_all
        hold = {"hold_n": 1, "kl_act": float(k_act.detach()), "kl_all": float(k_all.detach())}
    with torch.no_grad():
        issued = own & (y_order > 0)
        p_order = 1 - logits.softmax(-1)[..., 0]  # the chance of any order
        pred = logits[..., 1:].argmax(-1) + 1  # the most likely order, given one
        stats = {"n": int(own.sum()), "issued": int(issued.sum()), "p_issued": float(p_order[own].sum()),
                 "p_played": float((1 - play.softmax(-1)[..., 0])[own].sum()), **hold,
                 "issued_hit": int((issued & (pred == y_order)).sum()),
                 "none_hit": 0,
                 "ptr_n": int(has_ptr.sum()), "ptr_hit": int((has_ptr & (ptr.argmax(-1) == y_ptr)).sum()),
                 "pt_n": int(has_pt.sum()),
                 "pt_err": float(((xl.argmax(-1) - y_x).abs() + (yl.argmax(-1) - y_y).abs())[has_pt].float().sum()),
                 "l_order": float(l_order), "l_ptr": float(l_ptr), "l_pt": float(l_pt), "l_value": float(l_value),
                 "ret_n": len(ret), "ret_sum": float(ret.sum()), "ret_sq": float((ret ** 2).sum()),
                 "err_sq": float(((value - ret) ** 2).sum())}
    return l_order + l_ptr + 0.5 * l_pt + value_coef * l_value + l_hold, stats


def played(net: FullGameNet, logits: torch.Tensor, typ: torch.Tensor) -> torch.Tensor:
    """Order logits [N, O, C] as the policy plays them: only the orders the unit's type may give (the network's table,
    model.order_logits' by_type; "none" always)."""
    ok = net.allowed[typ.long()].clone()
    ok[..., 0] = True
    return logits.masked_fill(~ok, NEG)


def act_kl(ref_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """KL(ref || current) of the act/no-act decision per unit [...] from order logits [..., C]: the Bernoulli on
    P(none) (class 0) against P(any order) (--hold-start: a fine-tune keeps its start's rate of giving orders)."""
    lq, lp = torch.log_softmax(ref_logits.float(), -1), torch.log_softmax(logits.float(), -1)
    q0, p0 = lq[..., 0], lp[..., 0]
    q1, p1 = torch.logsumexp(lq[..., 1:], -1), torch.logsumexp(lp[..., 1:], -1)
    return q0.exp() * (q0 - p0) + q1.exp() * (q1 - p1)


HEADS = ("order.", "order_emb.", "cond.", "query.", "key.", "point_x.", "x_emb.", "point_y.", "fine_x.", "fine_y.",
         "point_f.", "value_head.", "priv_proj.")  # (--train-heads: the parameters after the trunk)


def train_heads(net: FullGameNet) -> int:
    """Only the heads trainable (--train-heads: the orders', the targets', the value's; the entity and global
    encoders, the transformer and the memory core frozen) -> the number of trainable parameters."""
    for name, p in net.named_parameters():
        p.requires_grad_(name.startswith(HEADS))
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def acting(logits: torch.Tensor) -> torch.Tensor:
    """Order logits [..., C] with "none" (class 0) ruled out: their softmax is the distribution over the orders given
    one (--acting-only)."""
    return logits.masked_fill(torch.arange(logits.shape[-1], device=logits.device) == 0, NEG)


def _masked_ce(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of [N, O, K] logits against labels [N, O] (-1: none), averaged over the labelled
    entries with weight > 0 (0 when there are none), with no GPU sync."""
    has = (y >= 0).float() * weight
    ce = F_.cross_entropy(logits.float().flatten(0, 1), y.clamp(min=0).flatten(), reduction="none")
    return (ce * has.flatten()).sum() / has.sum().clamp(min=1)


def summarize(stats: list[dict]) -> dict:
    s = {k: sum(x[k] for x in stats) for k in stats[0]}
    n_b = len(stats)
    ret_var = s["ret_sq"] / s["ret_n"] - (s["ret_sum"] / s["ret_n"]) ** 2
    return {"loss_order": s["l_order"] / n_b, "loss_target": s["l_ptr"] / n_b, "loss_point": s["l_pt"] / n_b,
            "loss_value": s["l_value"] / n_b,
            "value_explained_variance": 1 - s["err_sq"] / s["ret_n"] / max(ret_var, 1e-8),
            "order_acc": s["issued_hit"] / max(s["issued"], 1),  # which order, among the units that got one
            "order_rate_pred": s["p_issued"] / max(s["n"], 1),  # vs orders_per_unit_step: calibration
            "target_acc": s["ptr_hit"] / max(s["ptr_n"], 1),
            "point_error_bins": s["pt_err"] / max(s["pt_n"], 1) / 2,
            "orders_per_unit_step": s["issued"] / max(s["n"], 1),
            **({"order_rate_played": s["p_played"] / max(s["n"], 1)} if "p_played" in s else {}),
            **({"kl_act": s["kl_act"] / s["hold_n"], "kl_all": s["kl_all"] / s["hold_n"]} if "hold_n" in s else {})}


def run_config(path: Path) -> tuple[dict, Path]:
    """A self-play run's settings (its run.json's "config") and the directory its paths are relative to (the
    project's): `path` the run's directory or its run.json."""
    path = Path(path)
    run_json = path / "run.json" if path.is_dir() else path
    cfg = json.loads(run_json.read_text())["config"]
    root = Path(cfg["runs"]).parent if cfg.get("runs") else run_json.parent.parent.parent
    return cfg, root


def _int(v) -> int:
    return int(float(v)) if v not in (None, "", "None") else 0


def _float(v) -> float:
    return float(v) if v not in (None, "", "None") else 0.0


def apply_run(cfg: dict) -> None:
    """The features' settings of a self-play run as its learner sets them (selfplay.main, before it loads --init):
    the unit channels in use, the inputs, the masks, the neutral wares, macros and spells, the strategy block's size
    (its modes' and mix parts), the run's map's frame."""
    import os
    from . import modes as zmodes
    fx.ACTIVE_F = _int(cfg.get("features")) or fx.F
    os.environ["WARCRAFTSIM_FEATURES"] = str(fx.ACTIVE_F)
    fx.set_inputs(_int(cfg.get("inputs")) or 1, _float(cfg.get("opsd_share")) > 0 or _float(cfg.get("opsd_coef")) > 0)
    fx.set_masks(_int(cfg.get("tech_mask")), _int(cfg.get("valid_mask")))
    fx.set_neutral(_int(cfg.get("neutral")))
    fx.set_macros(_int(cfg.get("macros")))
    fx.set_spells(_int(cfg.get("spells")))
    fx.set_fine(_int(cfg.get("fine_points")))
    fx.set_macros_off(str(cfg.get("macros_off") or "").split(","))
    fx.set_econ_masks(_int(cfg.get("econ_masks")))
    z_modes, z_comp, z_mix = _float(cfg.get("z_modes")), _float(cfg.get("z_comp")), _int(cfg.get("z_mix"))
    modes_part = z_modes > 0 or z_comp > 0
    if modes_part:
        fx.set_z(strategy.ZDIM + zmodes.MDIM)
    if z_mix:
        fx.set_z(strategy.ZDIM + (zmodes.MDIM if modes_part else 0) + strategy.SLOTS)
    if z_comp > 0:
        fx.set_z(strategy.ZDIM + zmodes.MDIM + (strategy.SLOTS if z_mix else 0) + zmodes.CDIM)
    from ..data.duelmap import parse_duel_name
    if parse_duel_name(cfg["map"]) is None:
        from ..data.mapbuild import stock_map_path
        fx.set_geometry(*fx.map_geometry(stock_map_path(cfg["map"])))


def set_dropout(net: torch.nn.Module, p: float) -> None:
    """Every dropout of the network at p (as selfplay.set_dropout), saved with its config."""
    for m in net.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p
        elif isinstance(m, torch.nn.MultiheadAttention):
            m.dropout = p
    net.config["dropout"] = p


@torch.no_grad()
def zero_z_extras(net: FullGameNet) -> int:
    """The strategy block's parts after strategy.ZDIM (the modes', the target army's mix) read with zero weights: the
    cloning data has them zero, so they stay zero, and a run that fills them sees the network as without them."""
    c = net.config
    extra = c.get("z", 0) - strategy.ZDIM
    if extra <= 0:
        return 0
    end = c["G"] - c.get("priv", 0) - c.get("gx", 0)  # (the policy's inputs: [known | strategy block | clock])
    net.glob[0].weight[:, end - extra:end] = 0
    return extra


def held_out(paths: list[Path], pct: float) -> list[Path]:
    """The games kept for validation by a hash of their names (the same ones while a directory grows)."""
    return [p for p in paths if zlib.crc32(f"{p.parent.name}/{p.name}".encode()) % 1000 < pct * 10]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+", required=True,
                    help="directories of demonstration games (game*.npz)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[2] / "runs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-games", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--memory", action="store_true", help="the minGRU core (trained on chunks of consecutive steps)")
    ap.add_argument("--winners-only", action="store_true", help="learn only from the side that won each game (bc.sides_of)")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="learn only from games decided within this many minutes")
    ap.add_argument("--init", type=Path, help="start from this fit's policy (its network and vocabulary; e.g. to add "
                                             "takeover games to a fit)")
    ap.add_argument("--seq-len", type=int, default=32, help="with --memory: steps per chunk")
    ap.add_argument("--value-coef", type=float, default=0.5, help="weight of the value head's loss (0: untrained)")
    ap.add_argument("--avail-mask", type=int, default=1,
                    help="orders the player couldn't pay for: no labels, and masked (as when playing)")
    ap.add_argument("--device", help="default: cuda if available (asking opens the GPU driver)")
    ap.add_argument("--like-run", type=Path, help="a self-play run (its directory or run.json) whose anchor the clone "
                    "is to be: its settings (inputs, masks, --neutral/--macros/--spells, the strategy block, the map's "
                    "frame and bins), its vocabulary, its rewards for the value; --init defaults to its --init; each "
                    "side's own target from its --z-pool's slots (own_target) as the strategy input; each game in its "
                    "own frame (game_frame)")
    ap.add_argument("--run-orders", type=int, default=1, help="with --like-run: labels of orders the run's policy's "
                    "table rules out for the unit's type left out (drop_labels; the anchor plays with that table)")
    ap.add_argument("--resolve-smart", type=int, default=1, help="with --like-run: people's right clicks as the move "
                    "or attack the game makes of them (resolve_smart)")
    ap.add_argument("--fresh", action="store_true", help="with --init: its architecture and tables, new weights")
    ap.add_argument("--z-share", type=float, help="with --like-run: the share of sides given their own target "
                    "(default: the run's --z-share)")
    ap.add_argument("--val-pct", type=float, default=0.0, help="hold out this percentage of the games by a hash of "
                    "their names (instead of --val-games: the same games while a directory grows)")
    ap.add_argument("--max-batches", type=int, default=0, help="stop after this many batches (0: --epochs)")
    ap.add_argument("--acting-only", action="store_true", help="the order learned only given one: no loss for the "
                    "'none' labels and 'none' out of the softmax (bc.losses), so a fine-tune keeps its network's rate "
                    "of giving orders (people's: ~5%% of a unit's steps) and learns what people order; targets as before")
    ap.add_argument("--hold-start", type=float, default=0.0, help="with --init: this times KL(start || network) of "
                    "the act/no-act decision per own unit on the demonstrations' states (Bernoulli on P(none), the "
                    "start network frozen): a fine-tune keeps its start's rate of giving orders (e.g. 1-10)")
    ap.add_argument("--hold-all", type=float, default=0.0, help="with --init: this times the forward KL from the start "
                    "over every head (the orders, and the target unit and point given the label's order; self-play's "
                    "forward-all anchor) on the demonstrations' states: a trust region around the start")
    ap.add_argument("--train-heads", action="store_true", help="with --init: train only the heads (the orders', "
                    "the targets', the value's; bc.HEADS), the trunk (encoders, transformer, memory core) frozen")
    ap.add_argument("--z-extras", choices=("auto", "zero", "keep"), default="auto", help="with --like-run: the "
                    "strategy block's modes' and mix parts' weights (the demonstrations have them zero): zero (the clone "
                    "reads them as without them), keep (as --init reads them: a fine-tune of a policy that learned "
                    "them), auto: keep when --init's own network has them (a run's policy with --z-mix/--z-modes)")
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    cfg, root = run_config(args.like_run) if args.like_run else (None, None)
    if cfg is not None:
        apply_run(cfg)
        args.init = args.init or root / cfg["init"]
        args.memory = bool(_int(cfg.get("memory")))
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(p for d in args.data for p in d.glob("game*.npz") if ".tmp" not in p.name)  # (not one being written)
    if args.max_games:
        paths = paths[:args.max_games]
    random.Random(0).shuffle(paths)
    if args.val_pct > 0:
        val = held_out(paths, args.val_pct)
        held = set(val)
        train = [p for p in paths if p not in held]
    else:
        val, train = paths[:args.val_games], paths[args.val_games:]
    out = args.runs / "bc" / args.name
    out.mkdir(parents=True, exist_ok=True)
    init = torch.load(args.init, map_location="cpu", weights_only=False) if args.init else None
    run_vocab = json.loads((Path(cfg["runs"]) / cfg["name"] / "vocab.json").read_text()) if cfg is not None else None
    loaded = None
    if init and cfg is not None:  # (the run's way: model.load extends the vocabulary as the run's and puts its orders
        from .model import load  # in the run's order; the network grown to the run's inputs)
        loaded = load(args.init, "cpu", args.memory, vocab=run_vocab)
        vocab = loaded[1]["vocab"]
        if vocab["orders"] != run_vocab["orders"] or vocab["types"] != run_vocab["types"]:
            raise SystemExit(f"{args.init} under {args.like_run}'s settings: another vocabulary than the run's")
    else:
        vocab = init["vocab"] if init else fx.build_vocab(train)  # (from --init: its vocabulary, unknown types -> 0)
    (out / "vocab.json").write_text(json.dumps(vocab))
    enc = fx.Encoder(vocab)
    from .trace import unit_values
    values = unit_values()
    map_name = fx.load_game(train[0])["meta"].get("map", "duelrush")
    from ..data.duelmap import parse_duel_name
    frames = any(parse_duel_name(json.loads(str(np.load(p)["meta"])).get("map", "duelrush")) is None for p in paths[:20])
    if frames and cfg is None:  # another map than the duel maps: the first game's map's bins (the loader workers
        from ..data.mapbuild import stock_map_path  # fork from here and inherit it); each game its own frame
        fx.set_geometry(*fx.map_geometry(stock_map_path(map_name)))
    if frames:
        print(f"each game in its own frame (game_frame); point bins over +-{fx.BIN_RANGE:g} ({fx.BINS})", flush=True)
    prep = Prep(frames=frames, bin_range=fx.BIN_RANGE) if frames or cfg is not None else None
    if cfg is not None:
        prep.reward = {"gamma": _float(cfg["gamma"]), "shaping": _float(cfg["shaping"]),
                       "shaping_scale": _float(cfg["shaping_scale"]), "tie_break": _float(cfg["tie_break"])}
        if fx.NEUTRAL == 2:  # (no mercenaries: never labelled, never in a target)
            from .neutral import MERCENARIES, fourcc
            oix = {tuple(o): k for k, o in enumerate(vocab["orders"], start=1)}
            prep.drop = tuple(oix[(fourcc(m), fx.UNIT)] for m in MERCENARIES if (fourcc(m), fx.UNIT) in oix)
            prep.mercenaries = frozenset(MERCENARIES)
        if cfg.get("z_pool") and fx.ZDIM:
            pool = json.loads((root / cfg["z_pool"]).read_text())
            prep.items, prep.zdim = pool["items"], fx.ZDIM
            prep.z_share = args.z_share if args.z_share is not None else _float(cfg.get("z_share") or 1.0)
            prep.orderable = frozenset(strategy.orderable(vocab))
        if args.run_orders:  # (the run's policy's table of orders by unit type: its last checkpoint's, as load gives it)
            from .model import load
            own_ck = sorted((Path(cfg["runs"]) / cfg["name"] / "checkpoints").glob("[0-9]*.pt"))
            prep.allowed = load(own_ck[-1] if own_ck else args.init, "cpu", args.memory,
                                vocab=run_vocab)[0].allowed.numpy().copy()
        prep.smart = bool(args.resolve_smart)
    costs = None
    if args.avail_mask:
        from .costs import order_costs
        costs = order_costs(vocab, map_name)
        global RESEARCH_MAP
        RESEARCH_MAP = map_name
    info = {"kind": "bc", "name": args.name, "created": time.time(), "task": "fullgame",
            "policy": "recorded play" if cfg is not None else "built-in AI",
            **({"like_run": str(args.like_run), "z_share": prep.z_share if prep.items else 0.0} if cfg is not None else {}),
            "status": "fitting", "data": " ".join(map(str, args.data)), "games": {"train": len(train), "val": len(val)},
            "vocab": {"types": enc.n_types, "orders": enc.n_orders, "upgrades": len(enc.upgrade_index)},
            "args": {k: " ".join(map(str, v)) if isinstance(v, list) else str(v) for k, v in vars(args).items()},
            "command": "python3 -m warcraftsim.fullgame.bc " + " ".join(sys.argv[1:] if argv is None else argv),
            "value_reward": (prep.reward if prep is not None else REWARD) if args.value_coef > 0 else None,
            **({"init_from": str(args.init)} if args.init else {})}
    (out / "bc.json").write_text(json.dumps(info, indent=1))
    if args.note:
        (out / "notes.md").write_text(args.note + "\n")
    device = torch.device(args.device)
    if init:
        from .model import load
        net = loaded[0].to(device) if loaded else load(args.init, device)[0]
        if cfg is not None and fx.ZDIM and net.config.get("z", 0) != fx.ZDIM:
            raise SystemExit(f"{args.init}: a strategy block of {net.config.get('z', 0)}, the run's {fx.ZDIM}")
        if cfg is not None and not fx.ZDIM:
            fx.set_z(net.config.get("z", 0))
        if args.fresh:  # (its architecture and its tables from the game's data, new weights)
            from .model import DERIVED
            new = FullGameNet(**net.config).to(device)
            keep = {k: v for k, v in net.state_dict().items() if k in DERIVED or k == "order_kind"}
            new.load_state_dict({**new.state_dict(), **keep})
            net = new
        if cfg is not None:
            set_dropout(net, args.dropout)
            own_z = init["config"].get("z", 0) >= net.config.get("z", 0)  # (--init's own weights for the extras)
            if args.z_extras == "zero" or (args.z_extras == "auto" and not own_z):
                zero_z_extras(net)
            elif net.config.get("z", 0) > strategy.ZDIM:
                print(f"the strategy block's modes' and mix parts read as {args.init} reads them (--z-extras "
                      f"{args.z_extras})", flush=True)
            if prep.allowed is not None:  # (the run's table: the labels outside it are left out)
                net.allowed.copy_(torch.as_tensor(prep.allowed, device=net.allowed.device))
        args.d, args.memory = net.config["d"], net.memory
        if net.bins != fx.BINS:
            raise SystemExit(f"{args.init} has {net.bins} point bins, this map's frame {fx.BINS}: scripts/grow_bins.py")
        fx.set_fine(net.fine)  # (a network with a fine head: its labels as point indices; before the loaders)
    else:
        net = FullGameNet(enc.n_types, enc.n_cur, enc.n_orders, enc.G, d=args.d, layers=args.layers, priv=fx.PRIV,
                          dropout=args.dropout, memory=args.memory, bins=fx.BINS).to(device)
        net.allowed[0] = True  # unknown unit types: any order
        net.order_kind.copy_(torch.as_tensor(enc.order_kind, device=device))
    hold = args.hold_start > 0 or args.hold_all > 0
    if (hold or args.train_heads) and (not init or args.fresh):
        raise SystemExit("--hold-start, --hold-all and --train-heads fine-tune --init's network (not with --fresh)")
    ref = None
    if hold:  # the start, frozen (eval: no dropout), for the KL that holds the fine-tune near it
        import copy
        ref = copy.deepcopy(net).eval()
        ref.requires_grad_(False)
    if args.train_heads:
        n_train = train_heads(net)
        print(f"--train-heads: {n_train:,} of {sum(p.numel() for p in net.parameters()):,} parameters trained "
              f"(the heads), the trunk frozen", flush=True)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    kw = {"value_coef": args.value_coef, "avail_mask": bool(args.avail_mask), "acting_only": args.acting_only,
          "ref": ref, "hold_start": args.hold_start, "hold_all": args.hold_all}
    if args.memory:  # args.batch steps: lanes of seq_len steps
        lanes = max(1, args.batch // args.seq_len)
        data = Sequences(train, vocab, lanes, args.seq_len, values, costs, winners=args.winners_only,
                         max_minutes=args.max_minutes, prep=prep)
        states = torch.zeros(max(args.workers, 1) * lanes, args.d, device=device)
        ref_states = torch.zeros_like(states) if ref is not None else None  # (the start's own lanes' states)
    else:
        data = Steps(train, vocab, args.batch, values, costs, winners=args.winners_only, max_minutes=args.max_minutes,
                     prep=prep)
        states = ref_states = None
    val_batches = list(whole_sides(val, vocab, values, costs, winners=args.winners_only, max_minutes=args.max_minutes,
                                   prep=prep))
    print(f"{len(train)} training games, {len(val)} validation ({len(val_batches)} batches); "
          f"{enc.n_types} unit types, {enc.n_orders} order classes", flush=True)
    start_val = None
    if init and not args.fresh and (args.acting_only or hold or args.train_heads):  # (the start's validation: its rate
        net.eval()                                                  # of giving orders, which the fine-tune should keep)
        with torch.no_grad():
            vs = torch.zeros(len(val_batches[0]["lane"]), args.d, device=device) if args.memory else None
            start_val = summarize([losses(net, b, device, vs, **{**kw, "ref": None})[1] for b in val_batches])
        info["start_val"] = start_val
        (out / "bc.json").write_text(json.dumps(info, indent=1))
        print(f"start: order acc {start_val['order_acc']:.3f} rate {start_val['order_rate_pred']:.3f} (as played "
              f"{start_val['order_rate_played']:.3f}; people's {start_val['orders_per_unit_step']:.3f}) target "
              f"{start_val['target_acc']:.3f}", flush=True)
    t0 = time.time()
    total_batches = args.max_batches or None
    step = 0
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        data.epoch = epoch
        loader = torch.utils.data.DataLoader(data, batch_size=None, num_workers=args.workers, persistent_workers=False,
                                             pin_memory=device.type == "cuda",  # (workers forked, as before Python 3.14)
                                             multiprocessing_context="fork" if args.workers else None)
        net.train()
        train_stats, n = [], 0
        for b in loader:
            if args.max_batches and step >= args.max_batches:
                break
            if total_batches:  # cosine decay over the run
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * 0.5 * (1 + np.cos(np.pi * min(step / total_batches, 1.0)))
            loss, st = losses(net, b, device, states, ref_states=ref_states, **kw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            train_stats.append(st)
            n += 1
            step += 1
            if n % args.log_every == 0:  # (an epoch of whole games can take an hour: how far it is)
                tr_ = summarize(train_stats[-args.log_every:])
                print(f"epoch {epoch}: {n} batches, {time.time() - t0:.0f} s, order loss {tr_['loss_order']:.3f} "
                      f"acc {tr_['order_acc']:.3f} rate {tr_['order_rate_played']:.3f}"
                      + (f" kl act {tr_['kl_act']:.4f} all {tr_['kl_all']:.4f}" if ref is not None else ""), flush=True)
        if not train_stats:
            raise SystemExit("no batch in an epoch: with --memory each loader worker needs more game sides than lanes "
                             "(--batch / --seq-len); fewer --workers or a smaller --batch")
        total_batches = total_batches or n * args.epochs
        net.eval()
        with torch.no_grad():
            vs = torch.zeros(len(val_batches[0]["lane"]), args.d, device=device) if args.memory else None
            vs_ref = torch.zeros_like(vs) if args.memory and ref is not None else None
            val_stats = [losses(net, b, device, vs, ref_states=vs_ref, **kw)[1] for b in val_batches]
        tr, va = summarize(train_stats), summarize(val_stats)
        row = {"epoch": epoch, "time": time.time(), "seconds": round(time.time() - t0, 1),
               "lr": opt.param_groups[0]["lr"], "train_loss": tr["loss_order"] + tr["loss_target"] + 0.5 * tr["loss_point"],
               "val_loss": va["loss_order"] + va["loss_target"] + 0.5 * va["loss_point"],
               "acc": {"order (given one)": va["order_acc"], "target": va["target_acc"],
                       "value (explained variance)": va["value_explained_variance"]},
               "val": va, "train": tr}
        print(f"epoch {epoch}: train {row['train_loss']:.3f} val {row['val_loss']:.3f} | order acc "
              f"{va['order_acc']:.3f} rate {va['order_rate_played']:.3f} as played, {va['order_rate_pred']:.3f} "
              f"without the type's table (actual {va['orders_per_unit_step']:.3f}) "
              f"target {va['target_acc']:.3f} point err {va['point_error_bins']:.1f} bins value ev "
              f"{va['value_explained_variance']:.3f} ({n} batches, "
              f"{time.time() - t0:.0f}s)", flush=True)
        if start_val is not None:  # (the drift from the start: the rate as played, kept by --hold-start)
            row["drift"] = {"rate": va["order_rate_pred"] - start_val["order_rate_pred"],
                            "rate_played": va["order_rate_played"] - start_val["order_rate_played"]}
            print(f"  drift from the start: rate {row['drift']['rate']:+.3f}, as played "
                  f"{row['drift']['rate_played']:+.3f} ({start_val['order_rate_played']:.3f} -> "
                  f"{va['order_rate_played']:.3f})" + (f"; KL act {va['kl_act']:.4f} all {va['kl_all']:.4f}"
                                                       if ref is not None else ""), flush=True)
        with open(out / "fit.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        ck = {"model": net.state_dict(), "config": net.config, "vocab": vocab, "epoch": epoch, "val_loss": row["val_loss"],
              "value_reward": info["value_reward"]}
        torch.save(ck, out / "last.pt")
        if row["val_loss"] < best:  # policy.pt: the epoch with the lowest validation loss
            best = row["val_loss"]
            torch.save(ck, out / "policy.pt")
            info["best_epoch"] = epoch
            (out / "bc.json").write_text(json.dumps(info, indent=1))  # (the dashboard: which epoch policy.pt is)
        if args.max_batches and step >= args.max_batches:
            break
    info["status"] = "fitted"
    (out / "bc.json").write_text(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
