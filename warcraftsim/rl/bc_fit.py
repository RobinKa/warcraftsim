"""Behavior cloning for EntityNet: fit it to recorded episodes (warcraftsim.puffer.bc collect on a
general-order task) and write a checkpoint the torch trainer starts from (--init-from).

    python warcraftsim/rl/bc_fit.py runs/bc/mirror_mix_gen_hp400-pull35 --spec spec.json

The loss is the negative log-likelihood of the recorded orders (only the heads each order uses
count, under the recorded masks), minus an entropy bonus that keeps the choices the script never
makes possible for PPO (like label smoothing), plus a small value loss on the discounted return.
Each epoch goes to fit.jsonl next to the output, as for PufferLib's network (bc_train.py), so the
dashboard shows the fit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "puffer"))
from bc_train import load  # noqa: E402  (the dataset reader of the PufferLib fit)
from model import KINDS, EntityNet  # noqa: E402


def batches(episodes, idx, size, device):
    for i in range(0, len(idx), size):
        chunk = [episodes[j] for j in idx[i:i + size]]
        T = max(len(e[0]) for e in chunk)
        B = len(chunk)
        obs = np.zeros((T, B, chunk[0][0].shape[1]), np.float32)
        act = np.zeros((T, B, chunk[0][1].shape[1]), np.int64)
        ret = np.zeros((T, B), np.float32)
        valid = np.zeros((T, B), np.float32)
        masks = np.ones((T, B, chunk[0][4].shape[1]), np.uint8)
        for b, (o, a, r, _, m) in enumerate(chunk):
            n = len(o)
            obs[:n, b], act[:n, b], ret[:n, b], valid[:n, b], masks[:n, b] = o, a, r, 1, m
        starts = np.zeros((T, B), np.float32)
        starts[0] = 1
        yield tuple(torch.from_numpy(x).to(device) for x in (obs, act, ret, valid, masks, starts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="default: <data>/policy.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32, help="episodes per update")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--vf-coef", type=float, default=0.005)
    ap.add_argument("--entropy", type=float, default=0.01, help="entropy bonus (keeps unused choices possible)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = json.loads(args.spec.read_text())
    meta, episodes = load(args.data, args.gamma)
    net = EntityNet(spec).to(device)
    k = net.k
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, len(episodes) // 10)
    val, train = order[:n_val], order[n_val:]
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    out = args.out or args.data / "policy.pt"
    log = open(out.parent / "fit.jsonl", "w")
    print(f"{len(episodes)} episodes of {meta['policy']} on {meta['task']}; EntityNet "
          f"{sum(p.numel() for p in net.parameters()) / 1e3:.0f}k parameters; {device}", flush=True)
    t0 = time.time()

    def run(obs, act, ret, valid, masks, starts):
        T, B = obs.shape[:2]
        logp, ent, v = net.evaluate(obs, masks, act.view(T, B, k, -1), net.initial_state(B, device), starts)
        n = valid.sum().clamp(min=1)
        nll = -(logp * valid).sum() / n
        ent_m = (ent * valid).sum() / n
        vmse = (((v - ret) ** 2) * valid).sum() / n
        return nll, ent_m, vmse

    for epoch in range(args.epochs):
        net.train()
        rng.shuffle(train)
        tl, tn = 0.0, 0
        for b in batches(episodes, train, args.batch, device):
            nll, ent, vmse = run(*b)
            loss = nll - args.entropy * ent + args.vf_coef * vmse
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl, tn = tl + loss.item(), tn + 1
        lr_now = sched.get_last_lr()[0]
        sched.step()
        # validation: loss, and how often the most likely order kind is the script's
        net.eval()
        vl = vv = vn = 0.0
        tp = np.zeros(len(KINDS))
        pred_n = np.zeros(len(KINDS))
        true_n = np.zeros(len(KINDS))
        with torch.no_grad():
            for obs, act, ret, valid, masks, starts in batches(episodes, val, args.batch, device):
                nll, ent, vmse = run(obs, act, ret, valid, masks, starts)
                vl, vv, vn = vl + nll.item(), vv + vmse.item(), vn + 1
                T, B = obs.shape[:2]
                u, x = net.encode(obs.view(T * B, -1))
                x = x.view(T, B, -1)
                h, hs = net.initial_state(B, device), []
                for t in range(T):
                    h = net.core(x[t], h * (1 - starts[t]).unsqueeze(-1))
                    hs.append(h)
                acts, *_ = net.heads(obs.view(T * B, -1), u, torch.stack(hs).view(T * B, -1),
                                     masks.view(T * B, -1), greedy=True)
                pk = acts[..., 0].view(T, B, k)
                tk = act.view(T, B, k, -1)[..., 0]
                live = (masks.view(T, B, k, -1)[..., :len(KINDS)].sum(-1) > 1) & (valid.unsqueeze(-1) > 0)
                for c in range(len(KINDS)):
                    tp[c] += ((pk == c) & (tk == c) & live).sum().item()
                    pred_n[c] += ((pk == c) & live).sum().item()
                    true_n[c] += ((tk == c) & live).sum().item()
        acc = tp.sum() / max(true_n.sum(), 1)
        row = {"epoch": epoch + 1, "time": time.time(), "seconds": round(time.time() - t0, 1), "lr": lr_now,
               "train_loss": tl / max(tn, 1), "val_loss": vl / max(vn, 1), "value_mse": vv / max(vn, 1),
               "acc": {"first": acc},
               "choices": {KINDS[c]: {"precision": tp[c] / max(pred_n[c], 1), "recall": tp[c] / max(true_n[c], 1),
                                      "share": true_n[c] / max(true_n.sum(), 1)} for c in range(len(KINDS))
                           if true_n[c] or pred_n[c]}}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if epoch % 5 == 4 or epoch == args.epochs - 1:
            recall = "  ".join(f"{k_} r {v['recall']:.2f}" for k_, v in row["choices"].items())
            print(f"epoch {epoch + 1:3d}  val nll {row['val_loss']:.4f}  order acc {acc:.3f}  {recall}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    log.close()
    torch.save({"model": net.state_dict(), "config": net.config, "steps": 0, "spec": spec,
                "bc": {"data": str(args.data), "epochs": args.epochs}}, out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
