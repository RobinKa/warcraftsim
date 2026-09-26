"""What an EntityNet checkpoint thought during a video episode (for warcraftsim.overlay).

    python warcraftsim/rl/overlay_eval.py CHECKPOINT.pt TRACE.steps.npz OUT.npz

The trace holds the episode's observations, masks and actions per agent. For every step: the
value, and each action head's distribution given the orders chosen before it (as the policy
samples them), in the task's flat head order. Run by the bridge with the torch Python.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import EntityNet  # noqa: E402


def main() -> None:
    ckpt, trace_path, out = sys.argv[1:4]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = EntityNet(ck["spec"], **ck.get("config", {}))
    net.load_state_dict(ck["model"])
    net.eval()
    net._keep_dists = True
    trace = np.load(trace_path)
    obs = torch.as_tensor(trace["obs"], dtype=torch.float32)  # [T, A, obs]
    T, A = obs.shape[:2]
    masks = torch.as_tensor(trace["masks"]) if "masks" in trace.files else torch.ones(T, A, net.k * net.per, dtype=torch.uint8)
    acts = torch.as_tensor(trace["actions"], dtype=torch.long).view(T, A, net.k, -1)
    values = np.zeros((T, A))
    heads = len(net.sizes)
    probs = [np.zeros((T, A, n)) for _ in range(net.k) for n in net.sizes]
    ent = np.zeros((T, A, net.k * heads))
    with torch.no_grad():
        for a in range(A):
            h = net.initial_state(1, "cpu")
            for t in range(T):
                u, x = net.encode(obs[t, a:a + 1])
                h = net.gru(x, h)
                *_, v = net.heads(obs[t, a:a + 1], u, h, masks[t, a:a + 1], actions=acts[t, a:a + 1])
                values[t, a] = float(v)
                for i in range(net.k):
                    for hd in range(heads):
                        p = net.last_dists[hd][0, i].numpy()
                        probs[i * heads + hd][t, a] = p
                        ent[t, a, i * heads + hd] = -(p * np.log(np.maximum(p, 1e-12))).sum()
    # the trainer learns values of rewards times reward_scale (by default 1 / the task's, which the
    # bridge applied): back to the bridge's scale, which the video's returns are in
    scale = (ck.get("args") or {}).get("reward_scale") or 1.0 / ck["spec"].get("reward_scale", 1.0)
    np.savez(out, values=values / scale, entropy=ent, **{f"probs{j}": p for j, p in enumerate(probs)})


if __name__ == "__main__":
    main()
