"""Fit PufferLib's default network to recorded episodes (warcraftsim.puffer.bc collect) and write a
checkpoint the trainer loads (train.py --init-from).

The network is the trainer's (puffercpu.c PufferNet): a linear encoder, MinGRU layers with a
highway gate, a linear decoder whose last output is the value; no biases. Episodes start from a
zero recurrent state, as in the trainer (zeroed after a terminal). Loss: cross-entropy per action
head (label-smoothed, so choices the script never makes stay possible for PPO to try), where a
head that only details some choices of the unit's first head (the move direction, the attack
target) counts only on steps with that choice; plus the value fitted to the discounted return
(scaled rewards).

Standalone (numpy + torch), run by bc.py with a Python that has torch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PufferNet(nn.Module):
    def __init__(self, obs_size: int, act_sizes: list[int], hidden: int, layers: int):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Linear(obs_size, hidden, bias=False)
        self.gru = nn.ModuleList(nn.Linear(hidden, 3 * hidden, bias=False) for _ in range(layers))
        self.decoder = nn.Linear(hidden, sum(act_sizes) + 1, bias=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [B, T, obs_size] -> decoder outputs [B, T, sum(act_sizes) + 1], from zero states."""
        x = self.encoder(obs)
        for lin in self.gru:
            hid, gate, hw = lin(x).chunk(3, dim=-1)  # the gates depend only on the input: one matmul
            h_tilde = torch.where(hid >= 0, hid + 0.5, torch.sigmoid(hid))
            z = torch.sigmoid(gate)
            state = torch.zeros_like(x[:, 0])
            outs = []
            for t in range(x.shape[1]):
                state = state + z[:, t] * (h_tilde[:, t] - state)
                outs.append(state)
            out = torch.stack(outs, dim=1)
            s = torch.sigmoid(hw)
            x = s * out + (1 - s) * x
        return self.decoder(x)

    def export(self, path: Path) -> None:
        """The trainer's weight file: encoder, decoder, MinGRU layers; each padded to 8 floats."""
        parts = []
        for w in (self.encoder.weight, self.decoder.weight, *(lin.weight for lin in self.gru)):
            a = w.detach().float().cpu().numpy().ravel()
            parts += [a, np.zeros((-len(a)) % 8, np.float32)]
        np.concatenate(parts).astype(np.float32).tofile(path)


def load(data: Path, gamma: float):
    meta = json.loads((data / "meta.json").read_text())
    episodes = []
    for f in sorted(data.glob("game*.npz")):
        # each d[key] decompresses the whole array again: read every array once per file (per
        # episode, each slice kept its own copy of the file's array alive: 41 GB for 2000 episodes)
        with np.load(f) as d:
            obs, act, rew, ends = d["obs"], d["act"].astype(np.int64), d["rew"], d["ends"]
            lives = d["live"] if "live" in d else None
            masks = d["masks"] if "masks" in d else None
        start = 0
        for end in ends:
            r = rew[start:end]
            ret = np.zeros_like(r)
            acc = 0.0
            for t in range(len(r) - 1, -1, -1):
                acc = r[t] + gamma * acc
                ret[t] = acc
            live = lives[start:end] if lives is not None else np.ones((end - start, 1), bool)
            mask = masks[start:end] if masks is not None else np.ones((end - start, 1), np.uint8)
            episodes.append((obs[start:end], act[start:end], ret, live, mask))
            start = end
    return meta, episodes


def batches(episodes, idx, size, device):
    for k in range(0, len(idx), size):
        chunk = [episodes[i] for i in idx[k:k + size]]
        T = max(len(e[0]) for e in chunk)
        obs = np.zeros((len(chunk), T, chunk[0][0].shape[1]), np.float32)
        act = np.zeros((len(chunk), T, chunk[0][1].shape[1]), np.int64)
        ret = np.zeros((len(chunk), T), np.float32)
        mask = np.zeros((len(chunk), T), np.float32)
        live = np.zeros((len(chunk), T, chunk[0][3].shape[1]), np.float32)  # per unit group
        legal = np.ones((len(chunk), T, chunk[0][4].shape[1]), np.float32)  # action masks (1 column: none)
        for b, (o, a, r, lv, m) in enumerate(chunk):
            n = len(o)
            obs[b, :n], act[b, :n], ret[b, :n], mask[b, :n], live[b, :n], legal[b, :n] = o, a, r, 1, lv, m
        yield tuple(torch.from_numpy(x).to(device) for x in (obs, act, ret, mask, live, legal))


def losses(net, meta, obs, act, ret, mask, live, legal=None, vf_coef: float = 0.005, smoothing: float = 0.0):
    """Mean cross-entropy over the counted head-steps + vf_coef * value MSE (returns are scaled:
    ~±20, so a small weight keeps the value from dominating the shared layers). A head counts on
    steps where its unit is alive; a detail head only when the unit's first head chose what it
    details. Also returns per-choice counts of the first heads (for precision and recall)."""
    out = net(obs)
    sizes = meta["act_sizes"]
    group = meta.get("group_size", 1)
    detail = {int(k): (v if isinstance(v, list) else [v]) for k, v in meta.get("detail_heads", {}).items()}
    # first-head value -> the head offsets that detail it (e.g. cast -> ability slot and target)
    ce_sum, n_sum, stats = 0.0, 0.0, {}
    at = 0
    for h, n in enumerate(sizes):
        logits = out[..., at:at + n]
        at += n
        target = act[..., h]
        unit = h // group
        w = mask * (live[..., unit] if live.shape[-1] > unit else 1.0)
        offset = h % group
        if offset and detail:
            first = act[..., h - offset]
            chose = torch.zeros_like(mask, dtype=torch.bool)
            for v, offs in detail.items():
                if offset in offs:
                    chose |= first == v
            w = w * chose.float()
        if legal is not None and legal.shape[-1] > 1:  # action masks: only the legal options count,
            ok = legal[..., at - n:at] > 0              # smoothing spreads over them alone
            ok = ok | ~ok.any(dim=-1, keepdim=True)
            # a recorded choice the mask forbids (e.g. a script choosing retreat during a committed
            # retreat, which the env carries on regardless) teaches nothing: left out
            w = w * ok.gather(-1, target.unsqueeze(-1)).squeeze(-1).float()
            logits = logits.masked_fill(~ok, -1e9)
            logp = torch.log_softmax(logits, dim=-1)
            nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            spread = -(logp * ok).sum(-1) / ok.sum(-1)
            ce = (1 - smoothing) * nll + smoothing * spread
        else:
            ce = F.cross_entropy(logits.reshape(-1, n), target.reshape(-1), reduction="none",
                                 label_smoothing=smoothing).reshape(target.shape)
        ce_sum = ce_sum + (ce * w).sum()
        n_sum += float(w.sum())
        pred = logits.argmax(-1)
        key = "first" if offset == 0 else f"detail{offset}"
        s = stats.setdefault(key, {"hit": 0.0, "n": 0.0, "tp": np.zeros(n), "pred": np.zeros(n), "true": np.zeros(n)})
        s["hit"] += ((pred == target).float() * w).sum().item()
        s["n"] += float(w.sum())
        if offset == 0:
            for c in range(n):
                s["tp"][c] += (((pred == c) & (target == c)).float() * w).sum().item()
                s["pred"][c] += ((pred == c).float() * w).sum().item()
                s["true"][c] += ((target == c).float() * w).sum().item()
    vloss = (((out[..., -1] - ret) ** 2) * mask).sum() / mask.sum()
    loss = ce_sum / max(n_sum, 1.0) + vf_coef * vloss
    return loss, vloss.item(), stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--batch", type=int, default=64, help="episodes per update")
    ap.add_argument("--seed", type=int, default=0)
    # the returns are noisy (the outcome dominates): a larger weight let the value take over the
    # shared layers and the retreat decisions were not learned (recall 0.11 at 0.05, 0.82 at 0.005)
    ap.add_argument("--vf-coef", type=float, default=0.005)
    # a script uses few of the choices; fitted exactly, the others get ~0 probability and PPO
    # never tries them (fine-tuning pull35 stayed at 70% with every attack on "weakest")
    ap.add_argument("--smoothing", type=float, default=0.1, help="label smoothing: keeps all choices possible")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta, episodes = load(args.data, args.gamma)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, len(episodes) // 10)
    val, train = order[:n_val], order[n_val:]
    net = PufferNet(meta["obs_size"], meta["act_sizes"], args.hidden, args.layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"{len(episodes)} episodes ({sum(len(e[0]) for e in episodes)} steps) of {meta['policy']} on "
          f"{meta['task']} (win {meta['win_rate']:.0%}); train {len(train)}, validate {len(val)}; {device}",
          flush=True)
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        rng.shuffle(train)
        for obs, act, ret, mask, live, legal in batches(episodes, train, args.batch, device):
            loss, _, _ = losses(net, meta, obs, act, ret, mask, live, legal, args.vf_coef, args.smoothing)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        if epoch % 5 == 4 or epoch == args.epochs - 1:
            net.eval()
            tot, vl, nb, acc = 0.0, 0.0, 0, {}
            with torch.no_grad():
                for obs, act, ret, mask, live, legal in batches(episodes, val, args.batch, device):
                    loss, v, st = losses(net, meta, obs, act, ret, mask, live, legal, args.vf_coef, args.smoothing)
                    tot, vl, nb = tot + loss.item(), vl + v, nb + 1
                    for k, s in st.items():
                        a = acc.setdefault(k, {kk: 0 * vv if isinstance(vv, np.ndarray) else 0.0 for kk, vv in s.items()})
                        for kk, vv in s.items():
                            a[kk] = a[kk] + vv
            parts = []
            for k, a in sorted(acc.items()):
                if not a["n"]:
                    continue
                parts.append(f"{k} acc {a['hit'] / a['n']:.3f}")
                if k == "first":  # per choice that occurs: precision / recall
                    parts += [f"[{c}] p {a['tp'][c] / max(a['pred'][c], 1):.2f} r {a['tp'][c] / a['true'][c]:.2f}"
                              for c in range(len(a["true"])) if a["true"][c]]
            print(f"epoch {epoch + 1:3d}  val loss {tot / nb:.4f}  value mse {vl / nb:.2f}  {'  '.join(parts)}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    net.export(args.out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
