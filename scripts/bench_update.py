"""The whole-game learner's update (selfplay.ppo_update) on realistic steps: seconds per update of
8,192 steps and, with --profile, where the time goes (CPU and CUDA time by op).

    python3 scripts/bench_update.py --checkpoint runs/fgself-12/checkpoints/<steps>.pt \\
        --ref runs/bc/fullgame-fast-win6/policy.pt --data runs/fullgame/demos-fast-takeover-12 [--profile]

The steps are the demonstrations' (features, availability, the AI's labels for a share of the chunks as
if advised), with actions sampled by the checkpoint; advantages and returns are random. The update's
settings are fgself-12's (two epochs, minibatches of 256 steps in sequences of 16, bf16, the clone's
KL term, the cloning loss on 64-step batches, the distillation loss). It needs the GPU to itself.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warcraftsim.fullgame import bc, features as fx  # noqa: E402
from warcraftsim.fullgame.costs import order_costs  # noqa: E402
from warcraftsim.fullgame.model import act, load  # noqa: E402
from warcraftsim.fullgame.selfplay import ppo_update  # noqa: E402
from warcraftsim.fullgame.trace import unit_values  # noqa: E402


def make_steps(paths, vocab, costs, net, device, n_steps: int, chunk: int = 64, labelled: float = 0.5, seed: int = 0):
    """Chunks of `chunk` consecutive steps of game sides, n_steps in all, as the actors send them."""
    enc, rng = fx.Encoder(vocab, costs), np.random.default_rng(seed)
    chunks, total = [], 0
    for p in paths:
        g = fx.load_game(p)
        for side in (0, 1):
            d = bc.side_data(enc, g, side, unit_values())
            T = len(d["n_own"])
            for a in range(0, T - chunk + 1, chunk):
                lab = rng.random() < labelled
                ch = []
                for t in range(a, a + chunk):
                    n = max(1, int(d["mask"][t].sum()))
                    o = min(int(d["n_own"][t]), fx.MAX_OWN)
                    st = {"ent": d["ent"][t, :n].astype(np.float16), "type": d["type"][t, :n].astype(np.int16),
                          "cur": d["cur"][t, :n].astype(np.int16), "glob": d["glob"][t].astype(np.float32), "n": n,
                          "n_own": o, "avail": d["avail"][t], "h": None, "version": 0, "reward": 0.0, "done": False}
                    if lab:
                        st.update({k: d[k][t, :o].astype(np.int16) for k in ("y_order", "y_ptr", "y_x", "y_y")})
                    ch.append(st)
                chunks.append(ch)
                total += chunk
                if total >= n_steps:
                    break
            if total >= n_steps:
                break
        if total >= n_steps:
            break
    steps = [s for c in chunks for s in c]
    with torch.no_grad():  # the actions, as the actors sampled them
        for i in range(0, len(steps), 256):
            part = steps[i:i + 256]
            E = max(s["n"] for s in part)
            ent = torch.zeros(len(part), E, fx.F); typ = torch.zeros(len(part), E, dtype=torch.long)
            cur = torch.zeros(len(part), E, dtype=torch.long); mask = torch.zeros(len(part), E, dtype=torch.bool)
            for j, s in enumerate(part):
                ent[j, :s["n"]] = torch.from_numpy(s["ent"].astype(np.float32)); typ[j, :s["n"]] = torch.from_numpy(s["type"].astype(np.int64))
                cur[j, :s["n"]] = torch.from_numpy(s["cur"].astype(np.int64)); mask[j, :s["n"]] = True
            glob = torch.from_numpy(np.stack([s["glob"] for s in part]))
            n_own = torch.tensor([s["n_own"] for s in part])
            avail = torch.from_numpy(np.stack([s["avail"] for s in part]))
            a = act(net, ent.to(device), typ.to(device), cur.to(device), mask.to(device), glob.to(device), n_own.to(device),
                    avail=avail.to(device))
            for j, s in enumerate(part):
                o = s["n_own"]
                for k in ("order", "tgt", "bx", "by"):
                    s[k] = a[k][j, :o].cpu().numpy().astype(np.int16)
                s["logp"] = a["logp"][j, :o].float().cpu().numpy()
                s["value"] = float(a["value"][j])
                s["adv"], s["ret"] = float(rng.normal()), float(rng.normal())
    return steps, chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--ref", type=Path, help="the KL term's clone")
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="demonstrations: the steps and the cloning batches")
    ap.add_argument("--steps", type=int, default=8192)
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", action="append", default=[], help="an update setting, e.g. --set minibatch=512")
    ap.add_argument("--graphs", action="store_true", help="the CUDA-graph update (selfplay.GraphedUpdate)")
    ap.add_argument("--no-dropout", action="store_true", help="dropout off (to compare the eager and graphed updates)")
    a = ap.parse_args()
    device = torch.device(a.device)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    net, ck = load(a.checkpoint, device)
    vocab = ck["vocab"]
    ref = load(a.ref, device, memory=False)[0] if a.ref else None
    if ref is not None:
        ref.eval()
    costs = order_costs(vocab, "duelfast")
    paths = sorted(p for d in a.data for p in d.glob("game*.npz"))
    t0 = time.time()
    steps, chunks = make_steps(paths, vocab, costs, net, device, a.steps)
    print(f"{len(steps)} steps in {len(chunks)} chunks ({time.time() - t0:.0f} s)", flush=True)
    args = types.SimpleNamespace(epochs=2, minibatch=256, pad_groups=8, seq_len=16, bf16=1, clip=0.2, vf_coef=0.5, ent_coef=0.0,
                                 ref_kl=0.2, bc_coef=0.02, max_grad_norm=1.0, opd_coef=0.05)
    for kv in a.set:
        k, v = kv.split("=", 1)
        setattr(args, k.replace("-", "_"), type(getattr(args, k.replace("-", "_"), 0.0))(v))
    bc_set = bc.Steps(paths, vocab, 64, unit_values(), costs, arrays=True)

    def bc_cycle():
        while True:
            for b in torch.utils.data.DataLoader(bc_set, batch_size=None, num_workers=2, collate_fn=bc.as_is):
                yield {k: torch.from_numpy(v) for k, v in b.items()}
    bc_iter = bc_cycle()
    if a.no_dropout:
        for m in net.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
            if hasattr(m, "dropout") and isinstance(getattr(m, "dropout"), float):
                m.dropout = 0.0
    opt = torch.optim.Adam(net.parameters(), lr=3e-5, eps=1e-5, capturable=a.graphs)
    from warcraftsim.fullgame.selfplay import GraphedUpdate
    updater = GraphedUpdate(net, ref, opt, args, device) if a.graphs else None
    times = []
    for u in range(a.updates + 1):
        torch.cuda.synchronize() if device.type == "cuda" else None
        t = time.time()
        if a.profile and u == a.updates:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                out = updater.update(steps, chunks, False, bc_iter) if updater else ppo_update(net, ref, opt, steps, args, False, device, chunks, bc_iter)
                torch.cuda.synchronize()
            ka = prof.key_averages()
            cuda_total = sum(e.self_device_time_total for e in ka) / 1e6
            print(f"profiled update: {time.time() - t:.2f} s wall, {cuda_total:.2f} s of CUDA kernels; "
                  f"{sum(e.count for e in ka if e.key.startswith('cuda') and 'Launch' in e.key)} kernel launches")
            print(ka.table(sort_by="self_cpu_time_total", row_limit=25))
            print(ka.table(sort_by="self_device_time_total", row_limit=15))
            break
        out = updater.update(steps, chunks, False, bc_iter) if updater else ppo_update(net, ref, opt, steps, args, False, device, chunks, bc_iter)
        torch.cuda.synchronize() if device.type == "cuda" else None
        times.append(time.time() - t)
        print(f"update {u}: {times[-1]:.2f} s  " + " ".join(f"{k} {v:.4f}" for k, v in sorted(out.items())), flush=True)
    if len(times) > 1:
        print(f"seconds per update (after the first): {np.mean(times[1:]):.2f}")
    print("weights checksum:", float(sum(p.detach().double().sum() for p in net.parameters())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
