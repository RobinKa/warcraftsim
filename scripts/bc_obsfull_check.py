"""Check people's demonstrations with --obs-full events (fullgame/replays.py --obs-full) as bc.py --like-run reads them:
the encoder's features (features.Encoder.encode, the cloning's) against a live game's feeding of the same steps
(bc.live_encode: the units through play.unit_rows, the events in an observation's order, the items on the ground), and
what the --obs-full block holds (the units' state known, buffs, corpses, items lying, areas, upgrade levels, progress).

    .venv/bin/python scripts/bc_obsfull_check.py runs/scratch-human-obsfull/demos-obsfull --like-run runs/fgself-212

A demonstration without the events (an old one) is checked too: its state, corpse, item and area parts must be empty.
CPU only (no network is loaded); ~5-10 s a game side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warcraftsim.fullgame import bc  # noqa: E402
from warcraftsim.fullgame import features as fx  # noqa: E402

KEYS = ("ent", "type", "cur", "mask", "n_own", "glob", "avail")


def block_stats(out: dict) -> dict:
    """What the --obs-full block holds over a side's steps."""
    from warcraftsim.fullgame import obsfull as of
    A = fx.A_OBS
    f = out["ent"].astype(np.float32)
    m = out["mask"]
    kinds = f[..., A:A + 3]
    units = m & (kinds.sum(-1) == 0)
    own = np.zeros_like(m)
    for t, n in enumerate(out["n_own"].tolist()):
        own[t, :n] = True
    have = f[..., A + of.K_HAVE] > 0
    buffs = np.abs(f[..., A + of.K_BUFF:A + of.K_BUFF + of.BUFF_D]).sum(-1) > 0
    classes = f[..., A + of.K_CLASS:A + of.K_CLASS + 8].sum(-1) > 0
    sr = f[..., A + of.K_SPEED_R][units & have & (f[..., A + of.K_SPEED_R] > 0)]
    T = len(m)
    return {"steps": T, "unit_rows": int(units.sum()),
            "state_known": round(float(have[units].mean()), 3) if units.any() else 0.0,
            "state_known_own": round(float(have[units & own].mean()), 3) if (units & own).any() else 0.0,
            "with_buffs": int((buffs & units).sum()), "with_classes": int((classes & units).sum()),
            "speed_ratio_median": round(float(np.median(sr)), 3) if len(sr) else None,
            "upgrade_levels": int((np.abs(f[..., A + of.K_UPG:A + of.K_UPG + 2]).sum(-1) > 0)[units].sum()),
            "charges": int((f[..., A + of.K_CHARGE:A + of.K_CHARGE + 6].sum(-1) > 0)[units].sum()),
            "progress": int((f[..., A + of.K_PROD:A + of.K_PROD + 3].sum(-1) > 0)[units].sum()),
            "corpses_per_step": round(float(kinds[..., 0].sum() / T), 2),
            "items_per_step": round(float(kinds[..., 1].sum() / T), 3),
            "area_steps": int((kinds[..., 2].sum(-1) > 0).sum())}


def empty_without_events(out: dict) -> bool:
    """A side without --obs-full events: no state, buffs, charges, corpses, items or areas."""
    from warcraftsim.fullgame import obsfull as of
    f = out["ent"].astype(np.float32)[..., fx.A_OBS:]
    return not (f[..., :of.K_UPG].any() or f[..., of.K_CHARGE:of.K_CHARGE + 6].any() or f[..., of.K_AGE:].any())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", type=Path, nargs="+", help="game*.npz files or directories of them")
    ap.add_argument("--like-run", type=Path, default=Path("runs/fgself-212"))
    ap.add_argument("--max-games", type=int, default=4)
    args = ap.parse_args(argv)
    cfg, root = bc.run_config(args.like_run)
    bc.apply_run(cfg)
    vocab = json.loads((Path(cfg["runs"]) / cfg["name"] / "vocab.json").read_text())
    paths = [q for p in args.demos for q in (sorted(p.glob("game*.npz")) if p.is_dir() else [p])][:args.max_games]
    from warcraftsim.fullgame.costs import order_costs
    map_name = fx.load_game(paths[0])["meta"].get("map", "(2)EchoIsles")
    costs = order_costs(vocab, map_name)
    bc.RESEARCH_MAP = map_name
    steps = bc.research_levels(vocab, costs)
    print(f"{args.like_run}: obs-full {fx.OBS_FULL}, inputs {fx.INPUTS}, features {fx.ACTIVE_F} of {fx.F} (block at "
          f"{fx.A_OBS}), {len(vocab['types'])} types, {len(vocab['orders'])} orders", flush=True)
    bad = 0
    for p in paths:
        game = fx.load_game(p)
        has = "obsfull" in game
        game = bc.game_frame(game, fx.BIN_RANGE)
        if game is None:
            print(f"{p.name}: no frame (its map or starts)")
            continue
        game = bc.resolve_smart(game)
        for player in (0, 1):
            a = fx.Encoder(vocab, costs, steps).encode(game, player)
            b = bc.live_encode(fx.Encoder(vocab, costs, steps), game, player)
            diff = {k: int((a[k] != b[k]).reshape(len(a[k]), -1).any(1).sum()) for k in KEYS}
            same = not any(diff.values())
            bad += not same
            line = f"{p.name} side {player} ({game['meta']['races'][player]}): encode == live {same}"
            if not same:
                line += f" (steps differing per key: {diff})"
            if has:
                line += f"; block {block_stats(a)}"
            else:
                ok = empty_without_events(a)
                bad += not ok
                line += f"; no events: block's state, corpse, item and area parts empty {ok}"
            print(line, flush=True)
    print("OK" if not bad else f"{bad} problems")
    return int(bool(bad))


if __name__ == "__main__":
    raise SystemExit(main())
