"""--obs-full's cost in the observation path, on the CPU: a recorded game's observations (scripts/obs_probe.py cost
--save) through play.BCAgent.observe (features.View.step, the masks, the C features) with the flag off and on.

    python3 scripts/obs_full_bench.py OBS.pkl [--vocab runs/fgself-194/vocab.json] [--repeat 3]

Printed per setting: µs a step (median of the repeats), the entities a step (mean, max) and of them the new kinds
(corpses, items on the ground, area spells), and how often each part of the block is non-zero.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def setup(full: int, vocab_path: str):
    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame import features as fx
    fx.set_inputs(2)
    fx.set_masks(True, True)
    fx.set_neutral(2)
    fx.set_macros(3)
    fx.set_spells(1)
    fx.set_shop_items(1)
    fx.set_econ_masks(255)
    fx.set_obs_full(full)
    fx.ACTIVE_F = 29
    fx.set_geometry(*fx.map_geometry(stock_map_path("(2)EchoIsles")))
    vocab = json.loads(Path(vocab_path).read_text())
    from warcraftsim.fullgame import macros, shopitems, spells
    from warcraftsim.fullgame import neutral
    for mod in (neutral, macros, spells, shopitems):  # (the run's vocabulary as model.load extends it)
        ext = getattr(mod, "extend_vocab", None) or mod.extend
        done = getattr(mod, "extended")
        if not done(vocab):
            vocab = ext(vocab)
    if full:
        from warcraftsim.fullgame import obsfull
        vocab = obsfull.extend(vocab)
    return fx, vocab


def run(obs_list, full: int, vocab_path: str, repeat: int) -> dict:
    fx, vocab = setup(full, vocab_path)
    from warcraftsim.fullgame.costs import order_costs, research_steps
    from warcraftsim.fullgame.play import BCAgent
    costs, steps = order_costs(vocab, "echofast2"), research_steps(vocab, "echofast2")
    times, ents, kinds, nz = [], [], np.zeros(3), None
    for rep in range(repeat):
        bots = [BCAgent(None, vocab, p, None, costs=costs, steps=steps) for p in (0, 1)]
        races = [fx.RACES.index("human"), fx.RACES.index("undead")]
        for p, b in enumerate(bots):
            b.begin(obs_list[0], races if p == 0 else races[::-1])
        t0 = time.perf_counter()
        for t, obs in enumerate(obs_list):
            for b in bots:
                st = b.observe(obs, t)
                if rep == 0 and st is not None:
                    n = st["n"]
                    ents.append(n)
                    if full:
                        blk = st["ent"][:n, fx.A_OBS:]
                        kinds += blk[:, :3].sum(0)
                        nz = (blk != 0).sum(0) if nz is None else nz + (blk != 0).sum(0)
        times.append((time.perf_counter() - t0) / (2 * len(obs_list)) * 1e6)
    out = {"us_per_side_step": float(np.median(times)), "entities_mean": float(np.mean(ents)), "entities_max": int(max(ents)),
           "F": fx.F}
    if full:
        from warcraftsim.fullgame import obsfull as of
        parts = {"kinds (corpse, item, area) a step": (kinds / len(ents)).round(3).tolist()}
        for name, a, b in (("classes", of.K_CLASS, of.K_MAGIC), ("magic buffs", of.K_MAGIC, of.K_ARMOR),
                           ("armor", of.K_ARMOR, of.K_SPEED), ("speed", of.K_SPEED, of.K_HAVE), ("have state", of.K_HAVE, of.K_BUFF),
                           ("buff code", of.K_BUFF, of.K_UPG), ("upgrades", of.K_UPG, of.K_ITEM), ("item code", of.K_ITEM, of.K_CHARGE),
                           ("charges", of.K_CHARGE, of.K_PROD), ("progress", of.K_PROD, of.K_AGE), ("age", of.K_AGE, of.K_RADIUS),
                           ("radius", of.K_RADIUS, of.OBS_F)):
            parts[name] = round(float(nz[a:b].max()) / max(sum(ents), 1), 4)
        out["non-zero share of entity rows"] = parts
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("obs")
    ap.add_argument("--vocab", default="/home/tora/warcraftsim/runs/fgself-194/vocab.json")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--full", type=int, default=-1, help="only this setting (-1: both, in two processes' worth of state)")
    a = ap.parse_args()
    with open(a.obs, "rb") as f:
        obs_list = pickle.load(f)
    for full in ((0, 1) if a.full < 0 else (a.full,)):
        print(f"RESULT obs_full={full}: " + json.dumps(run(obs_list, full, a.vocab, a.repeat)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
