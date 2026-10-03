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
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

from . import features as fx
from .model import FullGameNet

KEYS = ("ent", "type", "cur", "mask", "n_own", "glob", "y_order", "y_ptr", "y_x", "y_y", "ret", "avail")
REWARD = {"gamma": 0.997, "shaping": 1.0, "shaping_scale": 2000.0, "tie_break": 0.5}  # selfplay's defaults


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


def side_data(enc: fx.Encoder, game: dict, player: int, values: dict) -> dict:
    """A game from `player`'s side: the encoded steps and their returns. In a takeover game
    (collect.py --policy) the side a policy played until the built-in AI took it over starts at
    the takeover: the steps before it have the policy's orders, not the AI's (the encoder still runs
    through them, for what the player queued and researched)."""
    out = enc.encode(game, player)
    out["ret"] = returns(game, player, values)
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
                 seed: int = 0, arrays: bool = False, winners: bool = False, max_minutes: float = 0.0):
        """`arrays`: numpy batches (with a loader's collate_fn=as_is they cross to its process as bytes:
        a tensor crosses as a shared-memory file of its own, ~15 ms for a batch's twelve)."""
        self.paths, self.vocab, self.batch, self.buffer_steps, self.seed = paths, vocab, batch, buffer_steps, seed
        self.values, self.costs, self.arrays = values, costs, arrays
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
        enc = fx.Encoder(self.vocab, self.costs)
        pick = np.random.default_rng(rng.randrange(1 << 30))
        cap = max(self.buffer_steps, self.batch)
        buf: dict[str, np.ndarray] = {}
        used = np.zeros(cap, bool)

        def batch():
            idx = pick.choice(np.flatnonzero(used), self.batch, replace=False)
            used[idx] = False
            return {k: buf[k][idx] if self.arrays else torch.from_numpy(buf[k][idx]) for k in KEYS}

        for p in paths:
            game = fx.load_game(p)
            for player in sides_of(game, self.winners, self.max_minutes):
                out = side_data(enc, game, player, self.values)
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
                 winners: bool = False, max_minutes: float = 0.0):
        self.paths, self.vocab, self.lanes, self.seq_len, self.seed = paths, vocab, lanes, seq_len, seed
        self.values, self.costs = values, costs
        self.winners, self.max_minutes = winners, max_minutes
        self.epoch = 0

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1000 + self.epoch * 100 + wid)
        paths = self.paths[wid::nw]
        rng.shuffle(paths)
        enc = fx.Encoder(self.vocab, self.costs)
        games, pending = iter(paths), []

        def side():  # the next game side's arrays, or None when the games are used up
            if not pending:
                p = next(games, None)
                if p is None:
                    return None
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
                max_minutes: float = 0.0):
    """Validation: every step of every game side, `sides` sides per batch as sequences padded to the
    longest ("valid" marks the steps; lane-major as Sequences, each lane starting at its side's start).
    The same batches for every model, with memory or without."""
    enc = fx.Encoder(vocab, costs)
    todo = []
    for p in paths:
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
           value_coef: float = 0.5, avail_mask: bool = True, stats: bool = True) -> tuple[torch.Tensor, dict]:
    """The loss of a batch and its statistics (`stats` False: none, and a dozen waits for the GPU
    fewer). With memory, `states` [lanes, d] holds each lane's state after its last chunk (read, and
    updated with the batch's)."""
    E = int(b["mask"].any(0).nonzero().max()) + 1  # the batch's widest view (before the copy: no wait for the GPU)
    b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    O = min(fx.MAX_OWN, E)
    ent, typ, cur, mask = b["ent"][:, :E].float(), b["type"][:, :E].long(), b["cur"][:, :E].long(), b["mask"][:, :E]
    y_order, y_ptr = b["y_order"][:, :O].long(), b["y_ptr"][:, :O].long()
    y_x, y_y = b["y_x"][:, :O].long(), b["y_y"][:, :O].long()
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
    own = torch.arange(O, device=device)[None] < n_own[:, None]
    if not stats:  # (self-play's auxiliary loss: masked means, not boolean indexing; each was a GPU sync)
        w = own.float()
        if net.training:  # the orders each unit type gets (padding and "none" set class 0: always allowed)
            net.allowed[typ[:, :O].reshape(-1), y_order.reshape(-1)] = True
        ptr, xl, z = net.target_logits(g, u, mask, y_order)
        yl = net.y_logits(z, y_x)
        loss = (_masked_ce(logits, y_order, w) + _masked_ce(ptr, y_ptr, w)
                + 0.5 * (_masked_ce(xl, y_x, w) + _masked_ce(yl, y_y, w)))
        if value_coef:
            value, ret = net.value(g).float(), b["ret"].float()
            if "valid" in b:
                v = b["valid"].float()
                loss = loss + value_coef * 0.5 * (((value - ret) ** 2) * v).sum() / v.sum().clamp(min=1)
            else:
                loss = loss + value_coef * 0.5 * ((value - ret) ** 2).mean()
        return loss, {}
    l_order = F_.cross_entropy(logits[own], y_order[own])
    if net.training:  # the orders each unit type gets (the mask for playing)
        issued_ = own & (y_order > 0)
        net.allowed[typ[:, :O][issued_], y_order[issued_]] = True
    ptr, xl, z = net.target_logits(g, u, mask, y_order)
    yl = net.y_logits(z, y_x)  # given the true x (teacher forcing)
    has_ptr, has_pt = (y_ptr >= 0) & own, (y_x >= 0) & own
    l_ptr = F_.cross_entropy(ptr[has_ptr], y_ptr[has_ptr]) if has_ptr.any() else logits.sum() * 0
    l_pt = (F_.cross_entropy(xl[has_pt], y_x[has_pt]) + F_.cross_entropy(yl[has_pt], y_y[has_pt])
            if has_pt.any() else logits.sum() * 0)
    value, ret = net.value(g).float(), b["ret"].float()
    if "valid" in b:  # (padded steps: no value target)
        value, ret = value[b["valid"]], ret[b["valid"]]
    l_value = 0.5 * ((value - ret) ** 2).mean()
    with torch.no_grad():
        issued = own & (y_order > 0)
        p_order = 1 - logits.softmax(-1)[..., 0]  # the chance of any order
        pred = logits[..., 1:].argmax(-1) + 1  # the most likely order, given one
        stats = {"n": int(own.sum()), "issued": int(issued.sum()), "p_issued": float(p_order[own].sum()),
                 "issued_hit": int((issued & (pred == y_order)).sum()),
                 "none_hit": 0,
                 "ptr_n": int(has_ptr.sum()), "ptr_hit": int((has_ptr & (ptr.argmax(-1) == y_ptr)).sum()),
                 "pt_n": int(has_pt.sum()),
                 "pt_err": float(((xl.argmax(-1) - y_x).abs() + (yl.argmax(-1) - y_y).abs())[has_pt].float().sum()),
                 "l_order": float(l_order), "l_ptr": float(l_ptr), "l_pt": float(l_pt), "l_value": float(l_value),
                 "ret_n": len(ret), "ret_sum": float(ret.sum()), "ret_sq": float((ret ** 2).sum()),
                 "err_sq": float(((value - ret) ** 2).sum())}
    return l_order + l_ptr + 0.5 * l_pt + value_coef * l_value, stats


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
            "orders_per_unit_step": s["issued"] / max(s["n"], 1)}


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
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(p for d in args.data for p in d.glob("game*.npz"))
    if args.max_games:
        paths = paths[:args.max_games]
    random.Random(0).shuffle(paths)
    val, train = paths[:args.val_games], paths[args.val_games:]
    out = args.runs / "bc" / args.name
    out.mkdir(parents=True, exist_ok=True)
    init = torch.load(args.init, map_location="cpu", weights_only=False) if args.init else None
    vocab = init["vocab"] if init else fx.build_vocab(train)  # (from --init: its vocabulary, unknown types -> 0)
    (out / "vocab.json").write_text(json.dumps(vocab))
    enc = fx.Encoder(vocab)
    from .trace import unit_values
    values = unit_values()
    costs = None
    if args.avail_mask:
        from .costs import order_costs
        costs = order_costs(vocab, fx.load_game(train[0])["meta"].get("map", "duelrush"))
    info = {"kind": "bc", "name": args.name, "created": time.time(), "task": "fullgame", "policy": "built-in AI",
            "status": "fitting", "data": " ".join(map(str, args.data)), "games": {"train": len(train), "val": len(val)},
            "vocab": {"types": enc.n_types, "orders": enc.n_orders, "upgrades": len(enc.upgrade_index)},
            "args": {k: " ".join(map(str, v)) if isinstance(v, list) else str(v) for k, v in vars(args).items()},
            "command": "python3 -m warcraftsim.fullgame.bc " + " ".join(sys.argv[1:] if argv is None else argv),
            "value_reward": REWARD if args.value_coef > 0 else None,
            **({"init_from": str(args.init)} if args.init else {})}
    (out / "bc.json").write_text(json.dumps(info, indent=1))
    if args.note:
        (out / "notes.md").write_text(args.note + "\n")
    device = torch.device(args.device)
    if init:
        from .model import load
        net, _ = load(args.init, device)
        args.d, args.memory = net.config["d"], net.memory
    else:
        net = FullGameNet(enc.n_types, enc.n_cur, enc.n_orders, enc.G, d=args.d, layers=args.layers,
                          dropout=args.dropout, memory=args.memory).to(device)
        net.allowed[0] = True  # unknown unit types: any order
        net.order_kind.copy_(torch.as_tensor(enc.order_kind, device=device))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.memory:  # args.batch steps: lanes of seq_len steps
        lanes = max(1, args.batch // args.seq_len)
        data = Sequences(train, vocab, lanes, args.seq_len, values, costs, winners=args.winners_only, max_minutes=args.max_minutes)
        states = torch.zeros(args.workers * lanes, args.d, device=device)
    else:
        data = Steps(train, vocab, args.batch, values, costs, winners=args.winners_only, max_minutes=args.max_minutes)
        states = None
    val_batches = list(whole_sides(val, vocab, values, costs, winners=args.winners_only, max_minutes=args.max_minutes))
    print(f"{len(train)} training games, {len(val)} validation ({len(val_batches)} batches); "
          f"{enc.n_types} unit types, {enc.n_orders} order classes", flush=True)
    t0 = time.time()
    total_batches = None
    step = 0
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        data.epoch = epoch
        loader = torch.utils.data.DataLoader(data, batch_size=None, num_workers=args.workers, persistent_workers=False,
                                             pin_memory=device.type == "cuda")
        net.train()
        train_stats, n = [], 0
        for b in loader:
            if total_batches:  # cosine decay over the run
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * 0.5 * (1 + np.cos(np.pi * min(step / total_batches, 1.0)))
            loss, st = losses(net, b, device, states, args.value_coef, bool(args.avail_mask))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            train_stats.append(st)
            n += 1
            step += 1
        total_batches = total_batches or n * args.epochs
        net.eval()
        with torch.no_grad():
            vs = torch.zeros(len(val_batches[0]["lane"]), args.d, device=device) if args.memory else None
            val_stats = [losses(net, b, device, vs, args.value_coef, bool(args.avail_mask))[1] for b in val_batches]
        tr, va = summarize(train_stats), summarize(val_stats)
        row = {"epoch": epoch, "time": time.time(), "seconds": round(time.time() - t0, 1),
               "lr": opt.param_groups[0]["lr"], "train_loss": tr["loss_order"] + tr["loss_target"] + 0.5 * tr["loss_point"],
               "val_loss": va["loss_order"] + va["loss_target"] + 0.5 * va["loss_point"],
               "acc": {"order (given one)": va["order_acc"], "target": va["target_acc"],
                       "value (explained variance)": va["value_explained_variance"]},
               "val": va, "train": tr}
        with open(out / "fit.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch}: train {row['train_loss']:.3f} val {row['val_loss']:.3f} | order acc "
              f"{va['order_acc']:.3f} rate {va['order_rate_pred']:.3f} (actual {va['orders_per_unit_step']:.3f}) "
              f"target {va['target_acc']:.3f} point err {va['point_error_bins']:.1f} bins value ev "
              f"{va['value_explained_variance']:.3f} ({n} batches, "
              f"{time.time() - t0:.0f}s)", flush=True)
        ck = {"model": net.state_dict(), "config": net.config, "vocab": vocab, "epoch": epoch, "val_loss": row["val_loss"],
              "value_reward": info["value_reward"]}
        torch.save(ck, out / "last.pt")
        if row["val_loss"] < best:  # policy.pt: the epoch with the lowest validation loss
            best = row["val_loss"]
            torch.save(ck, out / "policy.pt")
            info["best_epoch"] = epoch
            (out / "bc.json").write_text(json.dumps(info, indent=1))  # (the dashboard: which epoch policy.pt is)
    info["status"] = "fitted"
    (out / "bc.json").write_text(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
