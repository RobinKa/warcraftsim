"""Behavior cloning of the built-in AI on whole games (demonstrations from fullgame/collect.py).

    python3 -m warcraftsim.fullgame.bc --data runs/fullgame/demos-1 --name fullgame-1 --epochs 20

(the torch Python). Writes runs/bc/<name>/: bc.json, vocab.json, fit.jsonl (one row per epoch,
shown by the dashboard) and policy.pt. Games are encoded on the fly by loader workers (both
players' sides of every game; ~0.25 s per side), a few games are held out for validation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

from . import features as fx
from .model import FullGameNet

KEYS = ("ent", "type", "cur", "mask", "n_own", "glob", "y_order", "y_ptr", "y_x", "y_y")


class Steps(torch.utils.data.IterableDataset):
    """Shuffled batches of steps from the games (both sides), encoded by the loader workers."""

    def __init__(self, paths: list[Path], vocab: dict, batch: int, buffer_games: int = 16, seed: int = 0):
        self.paths, self.vocab, self.batch, self.buffer_games, self.seed = paths, vocab, batch, buffer_games, seed
        self.epoch = 0

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1000 + self.epoch * 100 + wid)
        paths = self.paths[wid::nw]
        rng.shuffle(paths)
        enc = fx.Encoder(self.vocab)
        buf: dict[str, list] = {k: [] for k in KEYS}
        for i, p in enumerate(paths):
            game = fx.load_game(p)
            for player in (0, 1):
                out = enc.encode(game, player)
                for k in KEYS:
                    buf[k].append(out[k])
            if len(buf["n_own"]) >= 2 * self.buffer_games or i == len(paths) - 1:
                cat = {k: np.concatenate(v) for k, v in buf.items()}
                order = np.random.default_rng(rng.randrange(1 << 30)).permutation(len(cat["n_own"]))
                for s in range(0, len(order) - self.batch + 1, self.batch):
                    idx = order[s:s + self.batch]
                    yield {k: torch.from_numpy(cat[k][idx]) for k in KEYS}
                buf = {k: [] for k in KEYS}


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


def losses(net: FullGameNet, b: dict, device) -> tuple[torch.Tensor, dict]:
    b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    E = int(b["mask"].any(0).nonzero().max()) + 1  # the batch's widest view
    O = min(fx.MAX_OWN, E)
    ent, typ, cur, mask = b["ent"][:, :E].float(), b["type"][:, :E].long(), b["cur"][:, :E].long(), b["mask"][:, :E]
    y_order, y_ptr = b["y_order"][:, :O].long(), b["y_ptr"][:, :O].long()
    y_x, y_y = b["y_x"][:, :O].long(), b["y_y"][:, :O].long()
    n_own = b["n_own"].long()
    g, u = net.encode(ent, typ, cur, mask, b["glob"].float())
    logits = net.order_logits(g, u, typ, n_own, by_type=False)
    own = torch.arange(O, device=device)[None] < n_own[:, None]
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
                 "l_order": float(l_order), "l_ptr": float(l_ptr), "l_pt": float(l_pt)}
    return l_order + l_ptr + 0.5 * l_pt, stats


def summarize(stats: list[dict]) -> dict:
    s = {k: sum(x[k] for x in stats) for k in stats[0]}
    n_b = len(stats)
    return {"loss_order": s["l_order"] / n_b, "loss_target": s["l_ptr"] / n_b, "loss_point": s["l_pt"] / n_b,
            "order_acc": s["issued_hit"] / max(s["issued"], 1),  # which order, among the units that got one
            "order_rate_pred": s["p_issued"] / max(s["n"], 1),  # vs orders_per_unit_step: calibration
            "target_acc": s["ptr_hit"] / max(s["ptr_n"], 1),
            "point_error_bins": s["pt_err"] / max(s["pt_n"], 1) / 2,
            "orders_per_unit_step": s["issued"] / max(s["n"], 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="a directory of demonstration games (game*.npz)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[2] / "runs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--val-games", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    paths = sorted(args.data.glob("game*.npz"))
    if args.max_games:
        paths = paths[:args.max_games]
    random.Random(0).shuffle(paths)
    val, train = paths[:args.val_games], paths[args.val_games:]
    out = args.runs / "bc" / args.name
    out.mkdir(parents=True, exist_ok=True)
    vocab = fx.build_vocab(train)
    (out / "vocab.json").write_text(json.dumps(vocab))
    enc = fx.Encoder(vocab)
    info = {"kind": "bc", "name": args.name, "created": time.time(), "task": "fullgame", "policy": "built-in AI",
            "status": "fitting", "data": str(args.data), "games": {"train": len(train), "val": len(val)},
            "vocab": {"types": enc.n_types, "orders": enc.n_orders, "upgrades": len(enc.upgrade_index)},
            "args": {k: str(v) for k, v in vars(args).items()}}
    (out / "bc.json").write_text(json.dumps(info, indent=1))
    if args.note:
        (out / "notes.md").write_text(args.note + "\n")
    device = torch.device(args.device)
    net = FullGameNet(enc.n_types, enc.n_cur, enc.n_orders, enc.G, d=args.d, layers=args.layers).to(device)
    net.allowed[0] = True  # unknown unit types: any order
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    data = Steps(train, vocab, args.batch)
    val_batches = list(Steps(val, vocab, args.batch))
    print(f"{len(train)} training games, {len(val)} validation ({len(val_batches)} batches); "
          f"{enc.n_types} unit types, {enc.n_orders} order classes", flush=True)
    t0 = time.time()
    total_batches = None
    step = 0
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
            loss, st = losses(net, b, device)
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
            val_stats = [losses(net, b, device)[1] for b in val_batches]
        tr, va = summarize(train_stats), summarize(val_stats)
        row = {"epoch": epoch, "time": time.time(), "seconds": round(time.time() - t0, 1),
               "lr": opt.param_groups[0]["lr"], "train_loss": tr["loss_order"] + tr["loss_target"] + 0.5 * tr["loss_point"],
               "val_loss": va["loss_order"] + va["loss_target"] + 0.5 * va["loss_point"],
               "acc": {"order (given one)": va["order_acc"], "target": va["target_acc"]},
               "val": va, "train": tr}
        with open(out / "fit.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch}: train {row['train_loss']:.3f} val {row['val_loss']:.3f} | order acc "
              f"{va['order_acc']:.3f} rate {va['order_rate_pred']:.3f} (actual {va['orders_per_unit_step']:.3f}) "
              f"target {va['target_acc']:.3f} point err {va['point_error_bins']:.1f} bins ({n} batches, "
              f"{time.time() - t0:.0f}s)", flush=True)
        torch.save({"model": net.state_dict(), "config": net.config, "vocab": vocab, "epoch": epoch}, out / "policy.pt")
    info["status"] = "fitted"
    (out / "bc.json").write_text(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
