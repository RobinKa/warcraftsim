"""The actors' per-step view (features.View.step: the entities' features, the masks, the global vector) on recorded
games, as a run sets it up: microseconds a step, and with --profile where the time goes; with --save, every step's
outputs, to compare another implementation's against (--compare).

    python scripts/bench_view.py --run fgself-139 --games 4 [--profile] [--save runs/bench/view-ref.npz]
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warcraftsim.fullgame import features as fx  # noqa: E402


def setup(run: str):
    """The run's settings (as selfplay.main sets them for its actors) -> (vocab, costs, map)."""
    c = json.loads((Path("runs") / run / "run.json").read_text())["config"]
    num = lambda k, d=0: int(float(c.get(k) or d))  # noqa: E731
    fx.set_inputs(num("inputs", 1), False)
    fx.set_masks(num("tech_mask"), num("valid_mask"))
    fx.set_neutral(num("neutral"))
    fx.set_macros(num("macros"))
    vocab = json.loads((Path("runs") / run / "vocab.json").read_text())
    import torch
    ck = torch.load(Path("runs") / run / "current.pt", map_location="cpu", weights_only=False)
    fx.set_z(ck["config"].get("z", 0))
    from warcraftsim.data.mapbuild import stock_map_path
    fx.set_geometry(*fx.map_geometry(stock_map_path(c.get("eval_map") or "(2)EchoIsles")))
    from warcraftsim.fullgame.costs import order_costs
    return vocab, order_costs(vocab, c["map"]), c["map"]


def steps(vocab, costs, games: list[Path]):
    """Each recorded game's sides, step by step: (view, rows, me, events, t, opp) -> yields View.step's arguments."""
    enc = fx.Encoder(vocab, costs)
    for path in games:
        g = fx.load_game(path)
        meta = g["meta"]
        T = meta["steps"] + 1
        u, pl, ev = g["units"], g["players"], g["events"]
        if fx.F > fx.F_BASE and "heroes" in g:
            u = fx.hero_rows(u, g["heroes"])
        us, ps, es = (fx._step_slices(a[:, 0], T) for a in (u, pl, ev))
        races = [fx.RACES.index(r) if r in fx.RACES else 0 for r in meta["races"]]
        for player in (0, 1):
            sign = enc.side(u[us[0]:us[1]], player)
            view = enc.view(player, sign, races)
            view.step_seconds = float(meta.get("step_seconds", 0.5))
            view.start = None
            for t in range(T):
                prow = pl[ps[t]:ps[t + 1]]
                me, opp = prow[prow[:, 1] == player], prow[prow[:, 1] == 1 - player]
                yield view, u[us[t]:us[t + 1]], me[0] if len(me) else None, ev[es[t]:es[t + 1]], t, \
                    opp[0] if len(opp) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="fgself-139")
    ap.add_argument("--demos", type=Path, default=Path("runs/fullgame/demos-echofast-normal"))
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--save", type=Path, help="every step's outputs (ent, type, cur, glob, avail, n_own) to this .npz")
    ap.add_argument("--compare", type=Path, help="a --save file: each step's outputs compared with it")
    a = ap.parse_args()
    vocab, costs, _ = setup(a.run)
    games = sorted(a.demos.glob("game*.npz"))[:a.games]
    args = list(steps(vocab, costs, games))
    print(f"{len(games)} games, {len(args)} side-steps")
    outs = []
    prof = cProfile.Profile() if a.profile else None
    t0 = time.perf_counter()
    if prof:
        prof.enable()
    for view, rows, me, ev, t, opp in args:
        st = view.step(rows, me, ev, t, opp)
        if a.save or a.compare:
            outs.append((st["ent"].copy(), st["type"].copy(), st["cur"].copy(), st["glob"].copy(),
                         None if st["avail"] is None else st["avail"].copy(), st["n_own"]))
    if prof:
        prof.disable()
    dt = time.perf_counter() - t0
    print(f"View.step: {1e6 * dt / len(args):.0f} us a side-step ({dt:.1f} s)")
    if prof:
        pstats.Stats(prof).sort_stats("cumulative").print_stats(28)
        pstats.Stats(prof).sort_stats("tottime").print_stats(22)
    if a.save:
        a.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.save, **{f"{k}_{i}": v for i, o in enumerate(outs)
                                        for k, v in zip(("ent", "type", "cur", "glob", "avail", "n_own"), o) if v is not None})
        print("saved", a.save)
    if a.compare:
        ref = np.load(a.compare)
        bad = 0
        for i, o in enumerate(outs):
            for k, v in zip(("ent", "type", "cur", "glob", "avail", "n_own"), o):
                key = f"{k}_{i}"
                if v is None and key not in ref.files:
                    continue
                same = key in ref.files and (np.allclose(np.asarray(v), ref[key], rtol=0, atol=1e-6) if k == "ent"
                                             and np.shape(v) == ref[key].shape else np.array_equal(np.asarray(v), ref[key]))
                if not same:  # (ent: float32 sines and cosines may differ in their last bit between numpy and C)
                    bad += 1
                    if bad <= 5:
                        print("differs:", key)
        print(f"compared {len(outs)} side-steps: {bad} outputs differ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
