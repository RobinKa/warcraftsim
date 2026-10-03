"""How likely policies are to give the orders the built-in AI gave, at the AI's own decisions in
demonstration games: e.g. the heroes (the default), or any set of order codes.

    python3 scripts/order_probability.py runs/fgself-12/checkpoints/<steps>.pt runs/bc/fullgame-fast-1/policy.pt \\
        --data runs/fullgame/demos-fast-1 --map duelfast [--orders Hpal,Hamg,...]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame.costs import order_costs  # noqa: E402
from warcraftsim.fullgame.model import load  # noqa: E402

HEROES = "Hpal,Hamg,Hmkg,Hblm,Obla,Ofar,Otch,Oshd,Udea,Ulic,Udre,Ucrl,Edem,Ekee,Emoo,Ewar"


def code(o: int) -> str:
    return int(o).to_bytes(4, "big").decode("latin-1", "replace")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("policies", type=Path, nargs="+")
    ap.add_argument("--data", type=Path, default=Path("runs/fullgame/demos-fast-1"))
    ap.add_argument("--map", default="duelfast")
    ap.add_argument("--orders", default=HEROES, help="order codes (comma-separated)")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--per-game", type=int, default=3, help="decisions a game side at most")
    args = ap.parse_args()
    want = set(args.orders.split(","))
    paths = sorted(glob.glob(str(args.data / "game*.npz")))[:args.games]
    for path in args.policies:
        net, ck = load(path, "cpu")
        vocab = ck["vocab"]
        enc = fx.Encoder(vocab, order_costs(vocab, args.map))
        classes = [c + 1 for c, (o, k) in enumerate(vocab["orders"]) if code(o) in want]
        ps, by = [], {}
        for p in paths:
            g = fx.load_game(p)
            races = g["meta"].get("races") if isinstance(g["meta"], dict) else None
            for player in (0, 1):
                d = enc.encode(g, player)
                for t, u in np.argwhere(np.isin(d["y_order"], classes))[:args.per_game]:
                    n = int(d["mask"][t].sum())
                    with torch.no_grad():
                        f = lambda k, dt: torch.from_numpy(d[k][t:t + 1, :n].astype(dt))  # noqa: E731
                        g_, uu = net.encode(f("ent", np.float32), f("type", np.int64), f("cur", np.int64),
                                            torch.ones(1, n, dtype=torch.bool), torch.from_numpy(d["glob"][t:t + 1]))
                        g_, _ = net.context(g_, None)
                        lo = net.order_logits(g_, uu[:, :min(fx.MAX_OWN, n)], f("type", np.int64),
                                              torch.tensor([int(d["n_own"][t])]), avail=torch.from_numpy(d["avail"][t:t + 1]))
                    pr = torch.softmax(lo[0, u].float(), -1)
                    ps.append((pr[int(d["y_order"][t, u])].item(), pr[0].item()))
                    if races:
                        by.setdefault(races[player], []).append(ps[-1][0])
        a = np.array(ps)
        print(f"{path}: {len(a)} decisions, p(the AI's order) {a[:, 0].mean():.4f}, p(no order) {a[:, 1].mean():.3f}",
              {k: round(float(np.mean(v)), 4) for k, v in sorted(by.items())})


if __name__ == "__main__":
    main()
