"""--map-pool in the real game: selfplay.game_loop's training games on a mix of maps within one process (its threads), a
run's policy on the CPU, the run's rewards, targets, teachers and scenarios. No learner: the steps are looked at and dropped.

    python scripts/map_pool_probe.py --run runs/fgself-173 --pool echofast2,lastrefugefast2 [--games 4] [--slots 2]
        [--minutes 4] [--opponent self|ai] [--scenarios natural,base2]

Printed per game: its map, races, length, what the game refused, expansions, and the frame's check: at each trained side's
first step its own hall's position in the features (columns 3 and 4) where the frame says a base stands. Instance names: fgsp<slot>.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["DISPLAY"] = ":65535"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="runs/fgself-173")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--pool", default="echofast2,lastrefugefast2")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--opponent", default="self", choices=("self", "ai"))
    ap.add_argument("--scenarios", default="", help="scenario names, drawn always (default: none)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--z-share", type=float, default=0.75)
    ap.add_argument("--races", default="")
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    import numpy as np
    import torch

    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame import mapframe, selfplay
    from warcraftsim.fullgame.collect import claim_slot
    from warcraftsim.fullgame.costs import order_costs
    from warcraftsim.fullgame.evalworker import run_config
    from warcraftsim.fullgame.model import load
    from warcraftsim.fullgame.trace import unit_values
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True, False)
    wine.ensure_template()
    run = Path(a.run).resolve()
    c = json.loads((run / "run.json").read_text())["config"]
    root = run.parent.parent
    ck = Path(a.checkpoint) if a.checkpoint else sorted((run / "checkpoints").glob("[0-9]*.pt"))[-1]
    print(f"checkpoint {ck}", flush=True)
    out = Path(tempfile.mkdtemp(prefix="mappoolprobe"))
    cfg, _ = run_config(run, "probe", 1, 0)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg.update(run_dir=str(out), map=c["map"], z_pool=str(root / c["z_pool"]), z_share=a.z_share, z_reward=float(c["z_reward"]),
               z_mix_reward=float(c.get("z_mix_reward", 0.0)), z_mode_reward=float(c.get("z_mode_reward", 0.0)),
               z_build=int(c.get("z_build", 0)), z_workers=int(c.get("z_workers", 0)), z_modes=float(c.get("z_modes", 0.0)),
               z_build_labels=int(c.get("z_build_labels", 0)), teach=int(c.get("teach", 0)),
               win_balance=0.0, scenarios=[s for s in a.scenarios.split(",") if s], scenario_share=1.0 if a.scenarios else 0.0,
               late_states=[], late_z=0, eval_checkpoints=[], eval_games_per_actor=0, games_per_actor=a.slots, actors=1,
               games_per_process=1, agent_games_factor=1, game_cuts=int(c["game_cuts"]) or None,
               max_minutes=a.minutes, max_steps=int(a.minutes * 60 / cfg["step_seconds"]) + 20, device="cpu",
               values=unit_values(), order_names={}, wait_floor_ms=int(c.get("wait_floor_ms", 5)), slot=0, game_nice=19,
               eval_vs=0.0, eval_heuristic=0.0, eval_z=0, eval_fallback=0, seed=int(time.time()),
               map_pool=[list(p) for p in mapframe.parse_pool(a.pool, c["map"])])
    if a.races:
        cfg["races"] = a.races.split(",")
    cfg["slot"], slot_lock = claim_slot(root / "runs")
    fx.ACTIVE_F = cfg["features"]
    os.environ["WARCRAFTSIM_FEATURES"] = str(cfg["features"])
    fx.set_inputs(cfg["inputs"], bool(cfg["privileged"]))
    fx.set_masks(cfg["tech_mask"], cfg["valid_mask"])
    fx.set_neutral(cfg["neutral"])
    fx.set_macros(int(cfg.get("macros", 0) or 0))
    fx.set_spells(int(cfg.get("spells", 0) or 0))
    fx.set_fine(int(cfg.get("fine_points", 0) or 0))
    fx.set_macros_off(str(cfg.get("macros_off") or "").split(","))
    fx.set_econ_masks(cfg.get("econ_masks", 0))
    fx.set_geometry(*fx.map_geometry(stock_map_path(cfg["map"])))
    print(f"frame {fx.STARTS}, bins {fx.BINS}, cuts {cfg['game_cuts']}, pool {cfg['map_pool']}", flush=True)
    for m, _ in cfg["map_pool"]:
        f = mapframe.frame_for(m)
        print(f"  {m}: {'turned %.1f' % f.deg if f else 'home'}", flush=True)
    cfg["map_geometry"] = [fx.STARTS, fx.BIN_RANGE]
    checkpoint = torch.load(ck, map_location="cpu", weights_only=False)
    cfg["z_dim"] = checkpoint["config"].get("z", 0)
    fx.set_z(cfg["z_dim"])
    version = int(checkpoint.get("update", checkpoint.get("version", 0)) or 0)
    del checkpoint
    vocab = json.loads(Path(cfg["vocab"]).read_text())
    shutil.copy(cfg["vocab"], out / "vocab.json")
    cfg["vocab"] = str(out / "vocab.json")
    selfplay.publish(load(ck, "cpu", vocab=vocab if fx.NEUTRAL else None)[0], version, out)
    cfg["costs"] = order_costs(vocab, cfg["map"]).tolist() if cfg["avail_mask"] else None
    cfg["learner_pid"] = os.getppid()
    (out / "league_spec.json").write_text(json.dumps({
        "launch": [{"kind": "agents", "p": 1.0} if a.opponent == "self" else {"kind": "ai", "difficulty": "insane", "p": 1.0}],
        "agents": [{"kind": "self", "name": "self", "p": 1.0}]}))
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    infer = selfplay.Inference(selfplay.Nets(cfg, device), device, a.slots * 2, compile_=False, buckets=(1, 2, 4),
                               entities=tuple(cfg["infer_entities"]))

    first: dict = {}
    add = selfplay.Trajectory.add

    def add_(self, st, res):
        k = id(self)
        if k not in first:
            f, sel = st["ent"], st["sel"]
            own = sel[:st["n_own"]]
            hall = np.flatnonzero(own[:, fx.C_FLAGS] & fx.STRUCTURE > 0)
            first[k] = True
            if len(hall):
                i = int(hall[0])
                print(f"  first step: own hall world-as-seen ({own[i, fx.C_X]},{own[i, fx.C_Y]}), feature position "
                      f"({float(f[i, 3]):.3f}, {float(f[i, 4]):.3f}), expected ({fx.ANCHOR_X:.3f}, 0.000)", flush=True)
        add(self, st, res)
    selfplay.Trajectory.add = add_

    class Out:
        def __init__(self):
            self.done, self.steps, self.rows = 0, 0, []

        def put(self, msg):
            e = msg.get("episode")
            if e is None:
                self.steps += 1
                return
            self.done += 1
            self.rows.append(e)
            p = e["prod"]
            print(f"game {self.done}: {e.get('map')} {e['races']} side {e['side']} {e['game_time']:.0f} s outcome {e['outcome']} "
                  f"scenario {e.get('scenario')} z {e.get('z')} gold {p['gold']} expansions {p['expansions']} halls_home {p['halls_home']} "
                  f"refused {e.get('refused_share')} {dict(Counter(e.get('refused') or {}).most_common(6))} "
                  f"built {p['built']} frame_orders {e.get('frame_orders')} step_wall {e.get('step_wall_ms')} ms", flush=True)
            for k in ("taught", "z_build", "z_refusals"):
                if e.get(k):
                    print(f"    {k}: {str(e[k])[:400]}", flush=True)

    o, stop = Out(), threading.Event()
    threads = [threading.Thread(target=selfplay.game_loop, args=(0, k, cfg, infer, o, stop, None), daemon=True) for k in range(a.slots)]
    for th in threads:
        th.start()
        time.sleep(3)
    t0 = time.time()
    while o.done < a.games and time.time() - t0 < 3600 and any(th.is_alive() for th in threads):
        time.sleep(2)
    stop.set()
    for th in threads:
        th.join(timeout=240)
    print(f"{o.done} games, {o.steps} trajectory messages, maps {Counter(r.get('map') for r in o.rows)}", flush=True)
    shutil.rmtree(out, ignore_errors=True)
    slot_lock.close()
    return 0 if o.done else 1


if __name__ == "__main__":
    raise SystemExit(main())
