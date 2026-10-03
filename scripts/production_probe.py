"""Why a whole-game policy holds its gold and lumber: plays games against the built-in AI (on the
CPU) and, every step, looks at the policy's own production buildings (those whose orders include
training a unit): busy or idle, and when idle with something affordable, how likely the policy
is to train there. Also the player's gold, lumber and supply.

    python3 scripts/production_probe.py runs/fgself-12/checkpoints/<steps>.pt --map duelfast --games 6
"""

from __future__ import annotations

import argparse
import random
import statistics as st_
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame.costs import order_costs
from warcraftsim.fullgame.model import load
from warcraftsim.fullgame.play import BCAgent, matchup_setup, unit_rows
from warcraftsim.runtime.instance import GameInstance

A = 9 + len(fx.FLAG_BITS)  # the entity features after the flags (features.Encoder.entities)
QUEUED, BUSY = A + 6, A + 7
STRUCTURE_COL = 9 + fx.FLAG_BITS.index(2)  # UnitFlags.STRUCTURE


class Probe(BCAgent):
    """BCAgent that also records, per step, its production buildings and its chance to train there."""

    def setup_probe(self, vocab: dict) -> None:
        self.rows: list[dict] = []
        orders = [(0, 0)] + [tuple(o) for o in vocab["orders"]]
        code = lambda o: int(o).to_bytes(4, "big").decode("latin-1", "replace")  # noqa: E731
        self.train = torch.tensor([i for i, (o, k) in enumerate(orders)
                                   if k == fx.IMMEDIATE and o >= 0x41000000 and not code(o).startswith("R")])
        self.research = torch.tensor([i for i, (o, k) in enumerate(orders)
                                      if k == fx.IMMEDIATE and o >= 0x41000000 and code(o).startswith("R")])

    def act(self, obs, t):
        st = self.observe(obs, t)
        if st is None:
            return []
        n, n_own = st["n"], st["n_own"]
        with torch.no_grad():
            ent = torch.as_tensor(st["ent"]).unsqueeze(0).float()
            typ = torch.as_tensor(st["type"]).unsqueeze(0).long()
            cur = torch.as_tensor(st["cur"]).unsqueeze(0).long()
            mask = torch.ones(1, n, dtype=torch.bool)
            g, u = self.net.encode(ent, typ, cur, mask, torch.as_tensor(st["glob"]).unsqueeze(0))
            g, self.h = self.net.context(g, self.h)
            O = min(n_own, fx.MAX_OWN)
            avail = torch.as_tensor(st["avail"])[None] if st.get("avail") is not None else None
            logits = self.net.order_logits(g, u[:, :O], typ, torch.tensor([O]), avail=avail)
            p = torch.softmax(logits[0].float(), -1)  # [O, classes]
            allowed = logits[0] > -1e8
            producers = allowed[:, self.train].any(-1)  # units that can train something now (affordable, their type's)
            can_ever = self.net.allowed[typ[0, :O]][:, self.train].any(-1)
            f = st["ent"]
            for i in range(O):
                if not bool(can_ever[i]) or f[i, STRUCTURE_COL] < 0.5:
                    continue
                self.rows.append({"t": t, "type": int(st["sel"][i, fx.C_TYPE]), "busy": bool(f[i, BUSY] > 0.5), "affordable": bool(producers[i]),
                                  "p_train": float(p[i, self.train].sum()), "p_research": float(p[i, self.research].sum()),
                                  "p_none": float(p[i, 0])})
            order = self._sample(logits)
            ptr, xl, z = self.net.target_logits(g, u, mask, order)
            tgt, bx_ = self._sample(ptr)[0], self._sample(xl)
            by = self._sample(self.net.y_logits(z, bx_))[0]
        pl = obs.players.get(self.player)
        if pl is not None:
            self.rows.append({"t": t, "player": True, "gold": pl.gold, "lumber": pl.lumber, "food": pl.food_used,
                              "cap": pl.food_cap})
        return self.commands(st, order[0].tolist(), tgt.tolist(), bx_[0].tolist(), by.tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--map", default="duelfast")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--difficulty", default="normal")
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    net, ck = load(args.checkpoint, "cpu")
    vocab, costs = ck["vocab"], order_costs(ck["vocab"], args.map)
    rng = random.Random(args.seed)
    plans = [(i, rng.choice(fx.RACES), rng.choice(fx.RACES), i % 2) for i in range(args.games)]
    lock, games = threading.Lock(), []

    def run(plan):
        i, race, ai_race, side = plan
        setup = matchup_setup(args.map, race, ai_race, args.difficulty, side, 50, args.max_minutes, 0.5)
        bot = Probe(net, vocab, side, torch.device("cpu"), costs=costs)
        bot.setup_probe(vocab)
        with GameInstance(setup, name=f"probe{i}", timeout=300) as g:
            obs = g.start()
            bot.begin(obs, [fx.RACES.index(s.race) for s in setup.slots])
            t = 0
            minutes = []  # per minute, both sides: food, held, gold gathered, heroes, their levels (as the AI analysis)
            while not obs.game_over:
                if t % 120 == 0 and t > 0:
                    rows = unit_rows(obs, t)
                    alive = rows[(rows[:, fx.C_FLAGS] & 1024) == 0]
                    for p_ in (side, 1 - side):
                        pl_ = obs.players.get(p_)
                        hs = alive[(alive[:, fx.C_OWNER] == p_) & ((alive[:, fx.C_FLAGS] & 1) > 0)]
                        if pl_ is not None:
                            minutes.append({"minute": t // 120, "learner": p_ == side, "food": pl_.food_used,
                                            "held": pl_.gold + pl_.lumber, "gold_gathered": pl_.gold_gathered,
                                            "heroes": len(hs), "hero_levels": int(hs[:, fx.C_HLEVEL].sum()) if len(hs) else 0})
                cmds = bot.act(obs, t)
                obs = g.step(cmds)
                bot.accepted(cmds, obs.command_results)
                t += 1
            result = obs.players[side].result.name if side in obs.players else "?"
        with lock:
            games.append({"race": race, "ai": ai_race, "result": result, "minutes": round(obs.game_time / 60, 1),
                          "rows": bot.rows, "per_minute": minutes})
            print(f"game {i}: {race} vs {ai_race}: {result} after {obs.game_time / 60:.1f} min", flush=True)

    with ThreadPoolExecutor(min(args.games, 8)) as ex:
        list(ex.map(run, plans))
    print("per minute, the learner vs the AI (mean over the games still going): food | held | gold gathered | heroes | hero levels")
    for m in range(1, 7):
        pm = [r for g in games for r in g["per_minute"] if r["minute"] == m]
        if not pm:
            continue
        cell = lambda k: "%7.1f vs %-7.1f" % (st_.mean(r[k] for r in pm if r["learner"]), st_.mean(r[k] for r in pm if not r["learner"]))  # noqa: E731
        print(f"  {m} | " + " | ".join(cell(k) for k in ("food", "held", "gold_gathered", "heroes", "hero_levels")) + f"  (n {len(pm) // 2})")
    by_type: dict[int, list] = {}
    for g in games:
        for r in g["rows"]:
            if "busy" in r:
                by_type.setdefault(r["type"], []).append(r)
    code = lambda o: int(o).to_bytes(4, "big").decode("latin-1", "replace")  # noqa: E731
    print("by building type: building-steps, busy, idle and affordable, p(train) there")
    for ty, rs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        ia = [r for r in rs if not r["busy"] and r["affordable"]]
        print(f"  {code(ty)} {len(rs):6d}  busy {100 * sum(r['busy'] for r in rs) / len(rs):3.0f}%  idle+affordable "
              f"{100 * len(ia) / len(rs):3.0f}%  p(train) {st_.mean(r['p_train'] for r in ia) if ia else 0:.3f}")
    for g in games:
        b = [r for r in g["rows"] if "busy" in r]
        pl = [r for r in g["rows"] if r.get("player")]
        late = [r for r in pl if r["t"] > 240]  # after two minutes
        idle = [r for r in b if not r["busy"]]
        idle_afford = [r for r in idle if r["affordable"]]
        print(f"{g['race']:9s} vs {g['ai']:9s} {g['result']:8s} {g['minutes']:4.1f} min | building-steps {len(b)}: "
              f"busy {100 * sum(r['busy'] for r in b) / max(len(b), 1):.0f}%, idle+affordable "
              f"{100 * len(idle_afford) / max(len(b), 1):.0f}% with p(train) {st_.mean(r['p_train'] for r in idle_afford) if idle_afford else 0:.3f} "
              f"p(research) {st_.mean(r['p_research'] for r in idle_afford) if idle_afford else 0:.3f} "
              f"p(none) {st_.mean(r['p_none'] for r in idle_afford) if idle_afford else 0:.2f} | after 2 min: gold "
              f"{st_.mean(r['gold'] for r in late) if late else 0:.0f} lumber {st_.mean(r['lumber'] for r in late) if late else 0:.0f} "
              f"supply-blocked {100 * sum(r['food'] >= r['cap'] - 1 for r in late) / max(len(late), 1):.0f}%")


if __name__ == "__main__":
    main()
