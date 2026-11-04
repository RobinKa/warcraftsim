"""Check A (docs/proposals/next-experiments.md, run 5: an EMA anchor, selfplay --ref-ema): blends of the KL anchor (the
run's --init) and a run's checkpoint, the anchor's weights moved 0, 25, 50, 75 and 100% of the way to the checkpoint's
(selfplay.EmaAnchor's step at that share; the anchor's buffers, the policy's allowed orders, as the learner copies them):
where an EMA anchor passes on its way. For each blend:

(a) its KL to each parent: the run's KL term (model.anchor_kl, forward-all: KL(parent || blend) over the orders, and the
    targets and points of the orders that use them) per own unit, on real steps: windows of 16 steps (the update's
    sequences, from the memory the playing policy had) out of the games below, all the blends' games pooled;
(b) games against the untaxed insane AI on the run's evaluation map, real rules, on the CPU (selfplay.play_one, an
    evaluation game: no target, no tax, a person's income): won, length, and whether play looks broken (supply at 5
    and 10 minutes, workers and army trained, gold gathered).

    python3 scripts/blend_check.py play --out DIR [--games 8] [--parallel 2] [--blends 0,25,50,75,100]
    python3 scripts/blend_check.py report DIR

Every blend plays the same matchups (races and sides drawn once), in turn. Instance names start with "blendcheck". Nice
19, no GPU, no learner; games.jsonl gets a row per game as it ends, kl.json the KL table after the last.
"""

from __future__ import annotations

import argparse
import copy
import json
import queue
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from computer_slot_probe import run_settings  # noqa: E402 (also: no GPU, the games' own Xvfbs)

import numpy as np  # noqa: E402

from warcraftsim.fullgame import features as fx  # noqa: E402

ROOT = Path("/home/tora/warcraftsim")
SEQ, EVERY = 16, 8  # (a window of SEQ steps out of every EVERY: the KL's sample)
MINUTES = (5, 10)  # (our side's supply noted at these game minutes)


class Watch:
    """A game instance's stand-in for play_one: our side's supply at MINUTES."""

    def __init__(self, g, side: int):
        self.g, self.side, self.food = g, side, {}

    def __getattr__(self, name):
        return getattr(self.g, name)

    def step(self, cmds=None):
        obs = self.g.step(cmds)
        p = obs.players.get(self.side)
        for m in MINUTES:
            if m not in self.food and obs.game_time >= 60 * m and p is not None:
                self.food[m] = [int(p.food_used), int(p.food_cap)]
        return obs


class Recorder:
    """The shared Inference as one game sees it: its policy's steps kept, a window of SEQ out of every EVERY (the
    learner's step dicts, Trajectory.add: the inputs, the memory going in, the orders chosen)."""

    def __init__(self, infer):
        self.infer, self.t, self.windows = infer, 0, {}

    def __getattr__(self, name):
        return getattr(self.infer, name)

    def request(self, items):
        out = self.infer.request(items)
        for (_, st), res in zip(items, out):
            w = self.t // SEQ
            self.t += 1
            if res is None or w % EVERY:
                continue
            n, o = st["n"], min(st["n_own"], fx.MAX_OWN)
            h = st.get("h")
            self.windows.setdefault(w, []).append({
                "ent": np.array(st["ent"][:n], np.float16), "type": np.array(st["type"][:n], np.int16),
                "cur": np.array(st["cur"][:n], np.int16), "glob": np.array(st["glob"], np.float32), "n": n, "n_own": o,
                "avail": None if st.get("avail") is None else np.array(st["avail"]),
                "h": None if h is None else np.array(h.cpu() if hasattr(h, "cpu") else h, np.float32).reshape(-1),
                **{k: np.array(res[k][:o], np.int16) for k in ("order", "tgt", "bx", "by")},
                "logp": np.zeros(o, np.float32), "adv": 0.0, "ret": 0.0})
        return out


def blends(init: Path, ck: Path, vocab: dict, shares: list[int]) -> dict:
    """{share: the anchor moved share% of the way to the checkpoint}, grown to the run's vocabulary as the run grows them."""
    from warcraftsim.fullgame.model import arch_key, load
    from warcraftsim.fullgame.selfplay import EmaAnchor
    anchor, _ = load(init, "cpu", True, vocab=vocab if fx.NEUTRAL else None)
    last, _ = load(ck, "cpu", True, vocab=vocab if fx.NEUTRAL else None)
    if arch_key(anchor.config) != arch_key(last.config):
        raise SystemExit("the anchor and the checkpoint differ in shape")
    out = {}
    for s in shares:
        b = copy.deepcopy(anchor)
        b.allowed.copy_(last.allowed)  # (the learner copies the policy's into the anchor every update)
        e = EmaAnchor(b, last, s / 100.0)
        assert e.left == 0
        e.step()
        out[s] = b.eval()
        for p in b.parameters():
            p.requires_grad_(False)
    return out


def kl_table(nets: dict, windows: list, lo: int, hi: int, batch: int = 16) -> dict:
    """Each blend's KL to the two parents (blends `lo` and `hi`) per own unit: the run's (forward-all) and the orders' alone."""
    import torch

    from warcraftsim.fullgame.model import anchor_kl, evaluate
    from warcraftsim.fullgame.selfplay import collate_seq
    sums = {s: {k: 0.0 for k in ("anchor", "last", "anchor_orders", "last_orders")} for s in nets}
    units = 0.0
    with torch.no_grad():
        for a in range(0, len(windows), batch):
            mb = collate_seq(windows[a:a + batch], torch.device("cpu"), T=SEQ)
            own = mb["own"].float()
            units += float(own.sum())
            ev = {s: evaluate(n, mb["ent"], mb["type"], mb["cur"], mb["mask"], mb["glob"], mb["n_own"], mb["order"],
                              mb["tgt"], mb["bx"], mb["by"], mb["avail"], seq=mb["seq"]) for s, n in nets.items()}
            for s in nets:
                for k, p in (("anchor", lo), ("last", hi)):
                    sums[s][k] += float((anchor_kl(ev[s], ev[p], "forward-all") * own).sum())
                    sums[s][k + "_orders"] += float((anchor_kl(ev[s], ev[p], "forward") * own).sum())
    return {"units": units, "steps": sum(len(w) for w in windows), "windows": len(windows),
            "kl": {str(s): {k: v / max(units, 1.0) for k, v in d.items()} for s, d in sums.items()}}


def play(a) -> int:
    import torch

    from warcraftsim.fullgame import selfplay
    from warcraftsim.runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup, exact_cuts
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="blendcheck"))
    cfg, vocab, ck = run_settings(a)
    cfg["run_dir"] = str(tmp)
    shares = [int(s) for s in a.blends.split(",")]
    nets = blends(Path(a.init), Path(ck), vocab, shares)
    n = max(1, a.parallel)
    infer = selfplay.Inference(selfplay.Nets(cfg, torch.device("cpu")), torch.device("cpu"), n, compile_=False,
                               buckets=tuple(range(1, n + 1)), entities=tuple(cfg["infer_entities"]))
    for s, net in nets.items():  # (each blend served as a past snapshot is, by its key: Nets.get)
        infer.nets.snapshots[f"blend{s}"] = net
    cfg["max_past"] = max(cfg.get("max_past", 6), len(nets))
    rng = random.Random(a.seed)
    races = list(fx.RACES)
    matchups = [([rng.choice(races), rng.choice(races)], rng.randrange(2)) for _ in range(a.games)]
    jobs: queue.Queue = queue.Queue()
    for k, (rs, side) in enumerate(matchups):
        for s in shares:
            jobs.put((s, k, rs, side))
    windows, lock = [], threading.Lock()

    class Rows:
        def put(self, msg):
            pass

    def worker(w: int) -> None:
        while True:
            try:
                s, k, rs, side = jobs.get_nowait()
            except queue.Empty:
                return
            slots = [None, None]
            slots[side] = Agent(rs[side], handicap=cfg["handicap"], difficulty="normal")  # (a person's income)
            slots[1 - side] = BuiltinAI(rs[1 - side], "insane", handicap=cfg["handicap"])
            setup = GameSetup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"],
                              max_game_seconds=a.minutes * 60, victory="decisive", window=(320, 240), draw=False,
                              wait_floor_ms=cfg.get("wait_floor_ms", 5), native_obs=bool(cfg.get("native_obs", True)),
                              nice=19, d3d_thread=False, render_threads=0, cuts=exact_cuts(cfg.get("game_cuts")) or None,
                              warm_spare=False, shim_dir=a.shim)
            rec = Recorder(infer)
            t0 = time.time()
            try:
                # (a name of its own per game, as computer_slot_probe's: a second game under a name just closed lost
                # its pipe at once)
                with GameInstance(setup, name=f"blendcheck{s}g{k}", timeout=600) as g:
                    obs = g.start()
                    watch = Watch(g, side)
                    ep = selfplay.play_one(watch, obs, cfg, vocab, rs, side,
                                           {"name": "script:ai-insane (real)", "kind": "ai", "eval_key": f"blend{s}"},
                                           rec, Rows(), w, train=False)
            except Exception as e:  # (a game that failed: said, the others go on)
                print(f"FAILED blend {s} game {k}: {e!r}", flush=True)
                continue
            prod = ep.get("prod") or {}
            row = {"blend": s, "game": k, "races": rs, "side": side, "outcome": ep.get("outcome"),
                   "minutes": round(ep.get("game_time", 0) / 60.0, 2), "wall": round(time.time() - t0),
                   "food": {str(m): v for m, v in watch.food.items()},
                   **{key: prod.get(key) for key in ("workers", "army", "heroes", "max_food", "gold", "lumber", "kills",
                                                      "lost", "expansions")},
                   "opp_max_food": (ep.get("opp_prod") or {}).get("max_food"), "refused_share": ep.get("refused_share")}
            with lock:
                windows.extend(rec.windows.values())
                with open(out / "games.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
            print(f"blend {s} game {k} ({rs[side]} vs {rs[1 - side]}): {row['outcome']} at {row['minutes']} min, "
                  f"supply {row['food']}, workers {row['workers']}, army {row['army']}, wall {row['wall']} s", flush=True)

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(n)]
    for th in threads:
        th.start()
        time.sleep(5)
    for th in threads:
        th.join()
    torch.save(windows, out / "windows.pt")
    table = kl_table(nets, windows, min(shares), max(shares))
    table.update(init=str(a.init), checkpoint=str(ck), shares=shares)
    (out / "kl.json").write_text(json.dumps(table, indent=1))
    return report(a)


def report(a) -> int:
    out = Path(a.out)
    rows = [json.loads(x) for x in open(out / "games.jsonl")] if (out / "games.jsonl").exists() else []
    kl = json.loads((out / "kl.json").read_text()) if (out / "kl.json").exists() else {"kl": {}}
    shares = sorted({r["blend"] for r in rows} | {int(s) for s in kl["kl"]})
    print(f"KL per own unit on {kl.get('steps', 0)} steps ({kl.get('windows', 0)} windows) of these games\n")
    print("| Blend | KL to anchor | KL to last | (orders: anchor / last) | Won | Ties | Minutes | Supply 5 / 10 min "
          "| Workers | Army | Gold |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    for s in shares:
        rs = [r for r in rows if r["blend"] == s]
        k = kl["kl"].get(str(s), {})
        f5 = [r["food"]["5"][0] for r in rs if "5" in r["food"]]
        f10 = [r["food"]["10"][0] for r in rs if "10" in r["food"]]
        print(f"| {s}% | {k.get('anchor', float('nan')):.3f} | {k.get('last', float('nan')):.3f} "
              f"| {k.get('anchor_orders', float('nan')):.3f} / {k.get('last_orders', float('nan')):.3f} "
              f"| {sum(r['outcome'] > 0 for r in rs)}/{len(rs)} | {sum(r['outcome'] == 0 for r in rs)} "
              f"| {mean([r['minutes'] for r in rs]):.1f} | {mean(f5):.0f} / {mean(f10):.0f} "
              f"| {mean([r['workers'] or 0 for r in rs]):.1f} | {mean([r['army'] or 0 for r in rs]):.1f} "
              f"| {mean([r['gold'] or 0 for r in rs]):,.0f} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("play", "report"))
    ap.add_argument("out", nargs="?", default=None, help="report: the directory (play: --out)")
    ap.add_argument("--out", dest="out_dir", default=".")
    ap.add_argument("--run", default=str(ROOT / "runs/fgself-182"))
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/fgself-182/checkpoints/0000000050150172.pt"))
    ap.add_argument("--init", default=str(ROOT / "runs/fgself-100/checkpoints/0000000013649982.pt"),
                    help="the anchor (the run's --init)")
    ap.add_argument("--blends", default="0,25,50,75,100", help="the shares of the way to the checkpoint, in %%")
    ap.add_argument("--games", type=int, default=8, help="games per blend")
    ap.add_argument("--parallel", type=int, default=2, help="games at a time (one process, threads)")
    ap.add_argument("--minutes", type=float, default=40.0, help="a game's limit (the run's evaluation games': 40)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--shim", default=str(Path(__file__).resolve().parents[1] / "build" / "shim"))
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    a.out = a.out or a.out_dir
    if a.what == "report":
        return report(a)
    import os
    os.nice(19)
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    return play(a)


if __name__ == "__main__":
    raise SystemExit(main())
