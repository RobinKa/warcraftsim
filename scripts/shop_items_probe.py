"""--shop-items in the real game (fullgame/shopitems.py): each race shop's items bought and used by scripted orders through
the policy's own path (play.BCAgent.observe, its mask, commands, accepted; no network), with the gold, the item and the
effect read from the game's state.

    python3 scripts/shop_items_probe.py items --race human [--map echofast2] [--vocab runs/fgself-173/vocab.json]
    python3 scripts/shop_items_probe.py merchant                # the goblin merchant's items, the staff's forms
    python3 scripts/shop_items_probe.py play --race orc --run runs/fgself-173 [--checkpoint PT] [--minutes 3]

items: an altar and the race's shop built by workers, the hall upgraded tier by tier (the other prerequisites spawned), a
hero trained at the altar; every item of the shop's stock in turn, before the tier it needs (masked), then bought with the
pointer elsewhere (its own shop by the fallback), used in its form. play: a game through selfplay.play_one with a run's
checkpoint on the CPU, the flag on: the network grows and the game runs.

Instance names start with "shopprobe".

Results (2026-11-01, echofast2, each race's shop and hall built in the game; the money each sale took was the item's price,
within a step's income; before/after in the game's state):
  healing potion  human 134 -> 275, orc 143 -> 238, undead 173 -> 300, night elf 133 -> 250 hit points at once
  mana potion     mana 86 -> 237, 61 -> 213, 75 -> 226, 149 -> 285 at once
  lesser clarity  mana +18 in 4 s (human 68 -> 86, orc 117 -> 135, night elf 149 -> 167: over time)
  regeneration    human hp 132 -> 164 in 5 s; salve: orc hp 85 -> 126 in 5 s, charges left; scroll of healing: undead 164 -> 300
  scroll of speed orc 296 -> 373 a second
  rod             undead: refused with no corpse; two peasants dead ~2 s, 2 skeletons (5 games of 6, the other refused)
  skull           undead: a ziggurat at the point refused before, taken after (blight); the hero need not walk near
  ivory tower     human: a watch tower at the point; town portal, each race: a jump of 1,270-1,620 in a step toward the hall
  staffs          sanctuary: a footman 2,500 from home to the hall, staff kept; preservation: an archer home, staff kept; both
                  refused on the hero; the merchant's teleportation: the hero 2,275 in a step to a building, staff kept
  orbs            sold, held; hit points off a grunt in 8 s without / with: fire 135 / 161, lightning 194 / 216, venom 163 / 282,
                  corruption 323 / 312 (armour: not seen so; all noisy, one try each)
  moonstone, anti-magic, dust, the merchant's tome, invulnerability, protection, invisibility: taken and used up (dust:
                  a charge a use), the effect not in what the harness reports. Boots, ring, cloak: sold; ring and cloak's stats seen
  a tier-2 item before second 440 sells at a race shop (the merchant's, placed at second 48: not at 451, at 521); masks: each
  race's list at tier 1, 2, 3 as ITEM_NEEDS; full inventory: no order sent, gold kept.
  tiny great hall (not offered): a hall at the point in one game, gone with no hall in another, gone by itself in a third.
  play: fgself-173's network, 493 -> 518 orders, old biases unchanged; three games ran (with --bias 2 undead bought 4 skulls).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.pop("WAYLAND_DISPLAY", None)  # (games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"

import numpy as np  # noqa: E402

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame import neutral, shopitems as si, techtree as tt  # noqa: E402
from warcraftsim.fullgame.play import BCAgent  # noqa: E402
from warcraftsim.protocol import (Build, Buy, ImmediateOrder, PointOrder, RemoveUnit, SetResources, Spawn, TargetOrder,  # noqa: E402
                                  UseItem, fourcc, rawcode)
from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup  # noqa: E402

MOVE, ATTACK, STOP = 851986, 851983, 851972
HEROES = {"human": ("Hblm", "AHfs", 852488, "point"), "orc": ("Ofar", "AOcl", 852119, "unit"),
          "undead": ("Udre", "AUcs", 852218, "point"), "nightelf": ("Ekee", "AEer", 852171, "unit")}
FRIEND = {"human": "hfoo", "orc": "ogru", "undead": "ugho", "nightelf": "earc"}


class Probe:
    def __init__(self, g, race: str, vocab: dict, map_name: str):
        from warcraftsim.fullgame.costs import order_costs, research_steps
        self.g, self.race = g, race
        self.obs = g.start()
        self.bot = BCAgent(None, vocab, 0, None, costs=order_costs(vocab, map_name), steps=research_steps(vocab, map_name))
        self.bot.begin(self.obs, [fx.RACES.index(race), 0 if race != "human" else 1])
        self.oix = {tuple(o): c for c, o in enumerate(vocab["orders"], start=1)}
        self.t = 0
        self.st = None
        self.tick()

    def tick(self, scripted=(), orders=None):
        """A step: the view (its masks), the policy's path for `orders` ({unit id: (class, pointed unit id or None, point
        or None)}), the game's answers -> the answers to the scripted commands and those to the policy's."""
        bot = self.bot
        st = self.st = bot.observe(self.obs, self.t)
        cmds = []
        if st is not None:
            n = min(st["n_own"], fx.MAX_OWN)
            sel = st["sel"]
            order, tgt, bx, by = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
            at = {int(u): i for i, u in enumerate(sel[:, fx.C_ID].tolist())}
            for u, (c, on, xy) in (orders or {}).items():
                if u in at and at[u] < n:
                    order[at[u]] = c
                    tgt[at[u]] = at.get(on, at[u])
                    if xy is not None:
                        wx, wy = fx.world_bins(np.float64(xy[0]), np.float64(xy[1]), self.bot.view.sign)
                        bx[at[u]], by[at[u]] = int(wx), int(wy)
            cmds = bot.commands(st, order.tolist(), tgt.tolist(), bx.tolist(), by.tolist())
        self.sent = cmds
        scripted = list(scripted)
        self.obs = self.g.step(scripted + cmds)
        res = list(self.obs.command_results)
        if st is not None:
            bot.accepted(cmds, res[len(scripted):])
        self.t += 1
        return res[:len(scripted)], res[len(scripted):]

    def wait(self, n: int, cond=None) -> int:
        for k in range(n):
            if cond is not None and cond():
                return k
            self.tick()
        return n

    def own(self, kind: str, p: int = 0):
        return [u for u in self.obs.units if u.owner == p and u.type == kind and u.alive]

    def gold(self) -> tuple[int, int]:
        q = self.obs.players[0]
        return q.gold, q.lumber

    def hero(self):
        return self.obs.unit(self.hero_id)

    def show(self) -> str:
        h = self.hero()
        return (f"hp {h.hp}/{h.max_hp} mp {h.mana}/{h.max_mana} at {h.x},{h.y} items "
                f"{[rawcode(k) if k else '-' for k in h.items]}")

    def avail(self, c: int) -> bool:
        return bool(self.st["avail"][c])

    def mask_of(self, item: str) -> bool:
        self.tick()
        return self.avail(self.oix[(fourcc(item), fx.UNIT)])

    def buy(self, item: str):
        """The order through the policy's path, its pointer on the hero itself -> (its slot or None, the sale's gold and
        lumber drop, the steps it took). The money is set high each time, so the drop is the price less a step's income."""
        c = self.oix[(fourcc(item), fx.UNIT)]
        before = list(self.hero().items)
        prev = self.gold()
        drop = None
        self.tick(orders={self.hero_id: (c, self.hero_id, None)})
        sent = list(self.sent)
        steps = 0
        for steps in range(120):
            now = self.gold()
            d = prev[0] - now[0], prev[1] - now[1]
            if d[0] > 0 or d[1] > 0:
                drop = d
            prev = now
            have = list(self.hero().items)
            if any(have[k] and have[k] != (before[k] if k < len(before) else 0) for k in range(len(have))):
                break
            self.tick()
        after = list(self.hero().items)
        new = [k for k in range(len(after)) if after[k] and after[k] != (before[k] if k < len(before) else 0)]
        return new[0] if new else None, drop, steps, sent

    def use(self, slot: int, kind: int, on=None, xy=None):
        c = self.oix[(fx.USESLOT + slot, kind)]
        self.tick()  # (the mask a step after the sale)
        masked = not self.avail(c)
        _, res = self.tick(orders={self.hero_id: (c, on, xy)})
        return list(self.sent), res, masked

    def hurt(self, drain: bool = True) -> None:
        """Hit points down to ~60% (two grunts on the hero, then taken away; the engine has a hero near death drink a
        potion by itself, so not lower: first back to 80% if it is low), mana down (its spell)."""
        self.wait(300, lambda: self.hero().hp >= 0.8 * self.hero().max_hp)
        h = self.hero()
        self.tick([Spawn(1, "ogru", h.x + 80, h.y + 80), Spawn(1, "ogru", h.x - 80, h.y + 80)])
        self.tick()
        en = [u.id for u in self.own("ogru", 1)]
        for _ in range(100):
            self.tick([TargetOrder(x, ATTACK, self.hero_id) for x in en])
            if self.hero().hp < 0.6 * self.hero().max_hp:
                break
        self.tick([RemoveUnit(x) for x in en] + [ImmediateOrder(self.hero_id, STOP)])
        if not drain:
            return
        _, abil, order, how = HEROES[self.race]
        for _ in range(3):
            h = self.hero()
            if how == "point":
                self.tick([PointOrder(self.hero_id, order, h.x + 300, h.y)])
            else:
                self.tick([Spawn(1, "hfoo", h.x + 120, h.y)])
                self.tick()
                foes = self.own("hfoo", 1)
                if foes:
                    self.tick([TargetOrder(self.hero_id, order, foes[-1].id)])
                    self.wait(3)
                    self.tick([RemoveUnit(foes[-1].id)])
            self.wait(3)
        self.tick([ImmediateOrder(self.hero_id, STOP)])


def setup_base(p: Probe) -> None:
    """The altar and the shop by workers, the hall's tiers by its own orders, the rest of what those need spawned."""
    race, obs = p.race, p.obs
    hall = p.own(tt.TIERS[race][0])[0]
    sx, sy = (1 if hall.x < 0 else -1), (1 if hall.y < 0 else -1)
    p.base = (hall.x, hall.y)
    altar, shop = tt.ALTARS[race], si.RACE_SHOPS[race]
    t2, t3 = tt.TIERS[race][1], tt.TIERS[race][2]
    pre = tt.closure({altar, shop, t3}, race) - {altar, shop, *tt.TIERS[race]} - {c for c in tt.closure({t3}, race) if c[0] == "R"}
    pre = sorted(c for c in pre if c[0] != "R" and not c[:1].isupper() and c not in neutral.WORKERS)
    pre += [n for c in si.STOCK[shop] for n in [si.ITEM_NEEDS.get(c)]  # (a graveyard: the sacrificial skull's)
            if n and n not in tt.TIERS[race] and not n.startswith("TWN") and n not in pre]
    spots = [(hall.x + sx * (300 + 350 * k), hall.y - sy * 900) for k in range(len(pre))]
    p.tick([SetResources(0, 50000, 50000)] + [Spawn(0, c, x, y) for c, (x, y) in zip(pre, spots)])
    print(f"spawned {pre}", flush=True)
    w = [u for u in obs.units if u.owner == 0 and u.is_worker and u.alive]
    a1, _ = p.tick([Build(w[0].id, fourcc(altar), hall.x + sx * 750, hall.y + sy * 150),
                    Build(w[1].id, fourcc(shop), hall.x + sx * 150, hall.y + sy * 750)])
    print(f"builds of {altar} and {shop}: {a1}", flush=True)

    def done(kind):
        return lambda: any(not int(u.flags) & fx.CONSTRUCTING for u in p.own(kind))
    n = p.wait(600, lambda: done(altar)() and done(shop)())
    print(f"t {p.obs.game_time:.0f}: both finished after {n} steps", flush=True)
    p.altar, p.shop = p.own(altar)[0].id, p.own(shop)[0].id
    hero = HEROES[race][0]
    print(f"hero {hero}: {p.tick([ImmediateOrder(p.altar, fourcc(hero))])[0]}", flush=True)
    p.wait(400, lambda: bool(p.own(hero)))
    p.hero_id = p.own(hero)[0].id
    from warcraftsim.protocol import LearnSkill
    p.tick([LearnSkill(p.hero_id, fourcc(HEROES[race][1]))])
    print(f"t {p.obs.game_time:.0f}: {p.show()}", flush=True)
    p.t2, p.t3 = t2, t3


def upgrade(p: Probe, to: str) -> None:
    hall = next(u for u in p.obs.units if u.owner == 0 and u.alive and u.type in tt.TIERS[p.race])
    res, _ = p.tick([ImmediateOrder(hall.id, fourcc(to))])
    n = p.wait(600, lambda: bool(p.own(to)))
    print(f"t {p.obs.game_time:.0f}: hall to {to}: answer {res}, finished after {n} steps", flush=True)


def line(item: str, cost, sold, form: str, effect: str, refused="") -> str:
    return f"RESULT {item:5s} cost {cost[0]:>3}/{cost[1]:<3} sale drop {str(sold):>12s} use {form:8s} {effect} {refused}"


def attack_rate(p: Probe, n: int = 16) -> int:
    """The hit points the hero takes off a grunt in n steps (an orb's effect), the hero whole first."""
    p.wait(600, lambda: p.hero().hp >= 0.9 * p.hero().max_hp)
    h = p.hero()
    p.tick([Spawn(1, "ogru", h.x + 90, h.y)])
    p.tick()
    foe = p.own("ogru", 1)[-1].id
    first = p.obs.unit(foe).hp
    for _ in range(n):
        p.tick([TargetOrder(p.hero_id, ATTACK, foe)])
        if p.obs.unit(foe) is None or not p.obs.unit(foe).alive:
            break
    left = p.obs.unit(foe).hp if p.obs.unit(foe) is not None else 0
    p.tick([RemoveUnit(foe), ImmediateOrder(p.hero_id, STOP)])
    return first - left


def items(a) -> int:
    race = a.race
    vocab = si.extend(neutral.extend_vocab(json.loads(Path(a.vocab).read_text())))
    slots = [Agent(race, handicap=100, difficulty="normal"), Agent("orc" if race != "orc" else "human", handicap=100, difficulty="normal")]
    gs = GameSetup(map=a.map, slots=slots, step_seconds=0.5, draw=False, native_obs=False, d3d_thread=False, render_threads=0,
                   nice=19, warm_spare=False, window=(320, 240), max_game_seconds=0, victory="decisive", shim_dir=a.shim)
    g = GameInstance(gs, name=f"shopprobe{a.name}", timeout=600)
    p = Probe(g, race, vocab, a.map)
    setup_base(p)
    shop = si.RACE_SHOPS[race]
    stock = sorted(si.STOCK[shop], key=lambda c: c not in ("phea", "shea"))  # (those first, the hero whole)
    costs = si.costs_of()
    results = []
    near = lambda: math.hypot(p.hero().x - p.obs.unit(p.shop).x, p.hero().y - p.obs.unit(p.shop).y) < 400  # noqa: E731
    needs_t2 = [c for c in stock if si.ITEM_NEEDS.get(c) == "TWN2"]
    free = [c for c in stock if c not in si.ITEM_NEEDS]
    print(f"== tier 1, t {p.obs.game_time:.0f}: masks: "
          f"{ {c: p.mask_of(c) for c in stock} }", flush=True)
    upgrade(p, p.t2)
    print(f"== tier 2, t {p.obs.game_time:.0f} (below a stockStart of 440 on {needs_t2}): masks: {({c: p.mask_of(c) for c in stock})}", flush=True)
    upgrade(p, p.t3)
    print(f"== tier 3, t {p.obs.game_time:.0f}: masks: {({c: p.mask_of(c) for c in stock})}", flush=True)
    p.tick([SetResources(0, 50000, 50000)])

    def bought(c: str):
        slot, drop, steps, sent = p.buy(c)
        print(f"-- {c}: sent {sent}, slot {slot} after {steps} steps, money drop {drop}, cost {costs[c]}; {p.show()}", flush=True)
        return slot, drop

    def report(c, drop, form, effect, used=None):
        print(results.append(line(c, costs[c], drop, form, effect, f"[sent {used}]" if used is not None else "")) or results[-1], flush=True)

    only = set(a.only.split(",")) if a.only else None
    for c in stock:
        if only is not None and c not in only:
            continue
        if c in ("pman", "hslv", "plcl", "sreg"):  # (down first: the engine uses these by itself in a fight)
            p.hurt(True)
            print(f"   hurt: {p.show()}", flush=True)
            p.tick([PointOrder(p.hero_id, MOVE, p.obs.unit(p.shop).x + 200, p.obs.unit(p.shop).y + 200)])
            p.wait(40, near)
        elif c in ("phea", "shea"):  # (hurt, it drinks a potion the moment it gets one: bought at full health first)
            p.wait(800, lambda: p.hero().hp >= 0.97 * p.hero().max_hp)
        without = attack_rate(p) if c in ("ofir", "oli2", "ocor", "oven") else None
        if without is not None:
            p.tick([PointOrder(p.hero_id, MOVE, p.obs.unit(p.shop).x + 200, p.obs.unit(p.shop).y + 200)])
            p.wait(40, near)
        slot, drop = bought(c)
        if slot is None:
            report(c, drop, "-", f"NOT HELD after the sale (hurt: the engine used it at once): {p.show()}")
            continue
        if c in ("phea", "shea"):
            p.hurt(False)
            print(f"   hurt: {p.show()}", flush=True)
        h0 = p.hero()
        if c in ("ofir", "oli2", "ocor", "oven"):
            report(c, drop, "passive", f"held; hit points taken off a grunt in 8 s: {without} before, {attack_rate(p)} holding it")
            continue
        if c in ("phea", "hslv", "shea", "sreg"):
            hp0 = p.hero().hp
            sent, res, masked = p.use(slot, fx.IMMEDIATE)
            p.wait(10 if c in ("sreg", "hslv") else 1)
            report(c, drop, "on itself", f"hp {hp0} -> {p.hero().hp} of {p.hero().max_hp}{' in 5 s' if c in ('sreg', 'hslv') else ''}, masked {masked}, answer {res}, slot now {rawcode(p.hero().items[slot]) if p.hero().items[slot] else None}", sent)
        elif c in ("pman", "plcl"):
            mp0 = p.hero().mana
            sent, res, masked = p.use(slot, fx.IMMEDIATE)
            p.wait(8 if c == "plcl" else 1)
            report(c, drop, "on itself", f"mp {mp0} -> {p.hero().mana} of {p.hero().max_mana}{' in 4 s' if c == 'plcl' else ''}, masked {masked}, answer {res}, slot now {rawcode(p.hero().items[slot]) if p.hero().items[slot] else None}", sent)
        elif c == "shas":
            def speed():
                h = p.hero()
                x0, y0 = h.x, h.y
                p.tick([PointOrder(p.hero_id, MOVE, h.x + (1500 if h.x < 0 else -1500), h.y)])
                p.wait(5)
                h = p.hero()
                return int(math.hypot(h.x - x0, h.y - y0) / 3)
            v0 = speed()
            sent, res, masked = p.use(slot, fx.IMMEDIATE)
            p.wait(1)
            v1 = speed()
            report(c, drop, "at once", f"speed per second {v0} -> {v1}, answer {res}, slot now {rawcode(p.hero().items[slot]) if p.hero().items[slot] else None}", sent)
        elif c == "rnec":
            sent0, res0, masked0 = p.use(slot, fx.IMMEDIATE)
            print(f"   no corpse: answer {res0}", flush=True)
            h = p.hero()
            p.tick([Spawn(1, "hpea", h.x + 90, h.y + 40), Spawn(1, "hpea", h.x + 90, h.y - 40), Spawn(0, FRIEND[race], h.x + 150, h.y)])
            p.tick()
            for u in p.own("hpea", 1):
                for _ in range(40):
                    if p.obs.unit(u.id) is None or not p.obs.unit(u.id).alive:
                        break
                    p.tick([TargetOrder(p.hero_id, ATTACK, u.id)] + [TargetOrder(f.id, ATTACK, u.id) for f in p.own(FRIEND[race])])
            p.tick([ImmediateOrder(p.hero_id, STOP)])
            p.wait(4)  # (a body is no corpse before its death is over: refused, and nothing raised)
            print(f"   peasants alive {len(p.own('hpea', 1))}, hero at {p.hero().x},{p.hero().y}", flush=True)
            n0 = {u.id for u in p.obs.units}
            sent, res, masked = p.use(slot, fx.IMMEDIATE)
            p.wait(6)
            new = [u.type for u in p.obs.units if u.id not in n0 and u.owner == 0]
            report(c, drop, "at once", f"with no corpse answer {res0}; two peasants killed by its side, then answer {res}: new units {new}, slot now {rawcode(p.hero().items[slot]) if p.hero().items[slot] else None}", sent)
        elif c == "dust":
            sent, res, masked = p.use(slot, fx.IMMEDIATE)
            p.wait(4)
            first = rawcode(p.hero().items[slot]) if p.hero().items[slot] else None
            sent2, res2, _ = p.use(slot, fx.IMMEDIATE)
            p.wait(4)
            report(c, drop, "at once", f"two charges: answer {res}, slot {first}; answer {res2}, slot {rawcode(p.hero().items[slot]) if p.hero().items[slot] else None} (reveals: nothing to see here)", sent)
        elif c in ("moon", "pams"):
            sent, res, masked = p.use(slot, fx.IMMEDIATE)
            p.wait(2)
            report(c, drop, "at once", f"answer {res}, slot now {rawcode(p.hero().items[slot]) if p.hero().items[slot] else None} (its effect not in the game's reports to the harness)", sent)
        elif c == "tsct":
            h = p.hero()
            dx = 1 if p.base[0] < 0 else -1
            at = (h.x + dx * 700, h.y)
            n0 = {u.id for u in p.obs.units}
            sent, res, masked = p.use(slot, fx.POINT, xy=at)
            p.wait(60, lambda: bool([u for u in p.obs.units if u.id not in n0 and u.owner == 0 and u.is_structure]))
            p.wait(4)
            new = [f"{u.type} at {u.x},{u.y} hp {u.hp}/{u.max_hp}" for u in p.obs.units if u.id not in n0 and u.owner == 0]
            print(f"   hero {p.show()}; halls {[(u.type, u.x, u.y) for u in p.obs.units if u.owner == 0 and u.type in tt.TIERS[race]]}", flush=True)
            report(c, drop, "a point", f"asked {at}, answer {res}: new {new}, slot now {rawcode(p.hero().items[slot]) if p.hero().items[slot] else None}", sent)
        elif c == "skul":
            hx, hy = p.base
            dx = 1 if hx < 0 else -1
            at = (hx + dx * 2600, hy)  # (off the base's blight)
            worker = next(u for u in p.obs.units if u.owner == 0 and u.is_worker and u.alive)
            p.tick([PointOrder(p.hero_id, MOVE, at[0] - dx * 200, at[1])])
            p.wait(30, lambda: math.hypot(p.hero().x - at[0], p.hero().y - at[1]) < 350)
            before, _ = p.tick([Build(worker.id, fourcc("uzig"), at[0], at[1])])
            order0 = rawcode(p.obs.unit(worker.id).order) if p.obs.unit(worker.id).order >= fx.TYPE_CODE else p.obs.unit(worker.id).order
            p.tick([ImmediateOrder(worker.id, STOP)])
            sent, res, masked = p.use(slot, fx.POINT, xy=at)
            p.wait(8)
            after, _ = p.tick([Build(worker.id, fourcc("uzig"), at[0], at[1])])
            order1 = rawcode(p.obs.unit(worker.id).order) if p.obs.unit(worker.id).order >= fx.TYPE_CODE else p.obs.unit(worker.id).order
            p.wait(6)
            report(c, drop, "a point", f"a ziggurat there before: answer {before}, order {order0}; after: answer {after}, order {order1}; answer {res}, slot now {rawcode(p.hero().items[slot]) if p.hero().items[slot] else None}", sent)
            p.tick([ImmediateOrder(worker.id, STOP), PointOrder(p.hero_id, MOVE, p.base[0], p.base[1])])
            p.wait(30, near)
        elif c in ("ssan", "spre"):
            h = p.hero()
            dx = 1 if p.base[0] < 0 else -1
            p.tick([PointOrder(p.hero_id, MOVE, p.base[0] + dx * 2200, p.base[1])])
            p.wait(30, lambda: math.hypot(p.hero().x - p.base[0], p.hero().y - p.base[1]) > 2000)
            h = p.hero()
            p.tick([Spawn(0, FRIEND[race], h.x + 150, h.y + 40)])
            p.tick()
            f = p.own(FRIEND[race])[-1]
            at0 = (f.x, f.y)
            sent0, res0, _ = p.use(slot, fx.UNIT, on=p.hero_id)  # (pointed at itself: the unit nearest, in commands)
            p.wait(14)
            moved = p.obs.unit(f.id)
            report(c, drop, "a unit", f"answer {res0}; {FRIEND[race]} {at0} -> {moved.x},{moved.y} (hall at {p.base}), staff kept {rawcode(p.hero().items[slot]) if p.hero().items[slot] else None}", sent0)
        elif c == "stwp":
            dx = 1 if p.base[0] < 0 else -1
            p.tick([PointOrder(p.hero_id, MOVE, p.base[0] + dx * 2200, p.base[1])])
            p.wait(30, lambda: math.hypot(p.hero().x - p.base[0], p.hero().y - p.base[1]) > 2000)
            h = p.hero()
            at0 = (h.x, h.y)
            sent, res, masked = p.use(slot, fx.POINT, xy=(h.x + dx * 500, h.y + 300))  # (near no building: its nearest)
            orders = []
            jump = 0.0
            for k in range(16):  # (a move ordered every step: not sent while it casts, BCAgent.committed)
                x0, y0 = p.hero().x, p.hero().y
                p.tick(orders={p.hero_id: (p.oix[(851986, fx.POINT)], None, (h.x, h.y))} if k < 11 else None)
                orders.append(p.hero().order)
                jump = max(jump, math.hypot(p.hero().x - x0, p.hero().y - y0))
            h1 = p.hero()
            report(c, drop, "a point", f"hero {at0} -> {h1.x},{h1.y} (hall at {p.base}), longest move in one step {jump:.0f}, answer {res}, its orders while casting {sorted(set(orders))}, slot now {rawcode(h1.items[slot]) if slot < len(h1.items) and h1.items[slot] else None}", sent)
        p.wait(2)
        p.tick([PointOrder(p.hero_id, MOVE, p.obs.unit(p.shop).x + 200, p.obs.unit(p.shop).y + 200)])
        p.wait(40, near)
    print("== a full inventory", flush=True)
    filled = 0
    for c in free * 3:
        if si.full(np.array([*[0] * fx.C_ITEM0, *p.hero().items, *[0] * 6])):
            break
        p.tick(orders={p.hero_id: (p.oix[(fourcc(c), fx.UNIT)], p.hero_id, None)})
        p.wait(8)
        filled += 1
    print(f"after {filled} buys: {p.show()}", flush=True)
    if len([k for k in p.hero().items if k]) == 6:
        g0 = p.gold()
        p.tick(orders={p.hero_id: (p.oix[(fourcc(free[0]), fx.UNIT)], p.hero_id, None)})
        print(f"full: the order to buy {free[0]} sent {p.sent}, gold {g0} -> {p.gold()}, futile orders {getattr(p.bot, 'futile_orders', 0)}", flush=True)
    print("== the episode row's counts", flush=True)
    print({k: getattr(p.bot, k, {}) for k in ("shop_bought", "merchant_bought", "item_uses", "items_gone")}, flush=True)
    print(f"order_results {getattr(p.bot, 'order_results', {})}", flush=True)
    print(f"refused {getattr(p.bot, 'refused_ctx', [])[:6]}", flush=True)
    print("\n".join(results), flush=True)
    g.close()
    return 0


def merchant(a) -> int:
    """The merchant's items (a merchant placed beside the base, past its stock time): which form each takes."""
    race = "human"
    vocab = si.extend(neutral.extend_vocab(json.loads(Path(a.vocab).read_text())))
    slots = [Agent(race, handicap=100, difficulty="normal"), Agent("orc", handicap=100, difficulty="normal")]
    gs = GameSetup(map=a.map, slots=slots, step_seconds=0.5, draw=False, native_obs=False, d3d_thread=False, render_threads=0,
                   nice=19, warm_spare=False, window=(320, 240), max_game_seconds=0, victory="decisive", shim_dir=a.shim)
    g = GameInstance(gs, name=f"shopprobe{a.name}m", timeout=600)
    p = Probe(g, race, vocab, a.map)
    setup_base(p)
    hall = p.own("htow")[0]
    dx = 1 if hall.x < 0 else -1
    p.tick([Spawn(15, "ngme", hall.x + dx * 300, hall.y - 300), Spawn(0, "hfoo", hall.x + dx * 800, hall.y + 300)])
    p.wait(2)
    m = sorted(p.own("ngme", 15), key=lambda u: math.hypot(u.x - hall.x, u.y - hall.y))  # (the placed one, not the map's)
    print(f"merchant {[(u.id, u.x, u.y) for u in m]}", flush=True)
    p.tick([PointOrder(p.hero_id, MOVE, m[0].x + 100, m[0].y + 100)])
    while p.obs.game_time < 520:  # (its stock counts from its placing: 440 s on)
        p.tick()
    p.tick([SetResources(0, 50000, 50000)])
    foot = p.own("hfoo")[0].id
    print(f"t {p.obs.game_time:.0f}: {p.show()}", flush=True)
    for c in ("tret", "pnvl", "spro", "pinv", "bspd", "prvt", "cnob", "stel"):
        before = list(p.hero().items)
        g0 = p.gold()
        res, _ = p.tick([Buy(p.hero_id, m[0].id, fourcc(c))])
        p.wait(6)
        after = list(p.hero().items)
        new = [k for k in range(len(after)) if after[k] and after[k] != (before[k] if k < len(before) else 0)]
        print(f"-- {c}: {res}, {g0} -> {p.gold()}, {p.show()}", flush=True)
        if not new:
            continue
        s = new[0]
        if c in ("bspd", "prvt", "cnob"):
            continue
        forms = [("none", {}), ("building", {"target": p.altar}), ("friend", {"target": foot})]
        h = p.hero()
        if c == "stel":  # (far from the altar first: a jump, not a walk)
            p.tick([PointOrder(p.hero_id, MOVE, hall.x + dx * 3000, hall.y)])
            p.wait(30, lambda: math.hypot(p.hero().x - hall.x, p.hero().y - hall.y) > 2600)
            at0 = (p.hero().x, p.hero().y)
            sent, res, masked = p.use(s, fx.UNIT, on=p.altar)
            jump = 0.0
            for _ in range(10):
                x0, y0 = p.hero().x, p.hero().y
                p.tick()
                jump = max(jump, math.hypot(p.hero().x - x0, p.hero().y - y0))
            print(f"RESULT stel the policy's path: sent {sent} answer {res} masked {masked}, hero {at0} -> {p.hero().x},{p.hero().y} (altar {p.obs.unit(p.altar).x},{p.obs.unit(p.altar).y}), longest move in a step {jump:.0f}, staff kept {rawcode(p.hero().items[s]) if p.hero().items[s] else None}", flush=True)
            continue
        for name, kw in forms + [("point", {"x": hall.x + dx * 500, "y": hall.y + 200})]:
            h = p.hero()
            at0, sl0 = (h.x, h.y), h.items[s]
            res, _ = p.tick([UseItem(p.hero_id, s, **kw)])
            p.wait(12)
            h = p.hero()
            print(f"   {name}: answer {res}, hero {at0} -> {h.x},{h.y}, slot {rawcode(sl0)} -> {rawcode(h.items[s]) if h.items[s] else None}, {p.show()}", flush=True)
            if not h.items[s]:
                break
    g.close()
    return 0


def play(a) -> int:
    """A game through selfplay.play_one with a run's checkpoint on the CPU: the network grows, the masks and the row work."""
    import torch

    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame import selfplay
    from warcraftsim.fullgame.costs import order_costs
    from warcraftsim.fullgame.evalworker import run_config
    from warcraftsim.fullgame.model import load
    from warcraftsim.fullgame.trace import unit_values
    run = Path(a.run).resolve()
    c = json.loads((run / "run.json").read_text())["config"]
    root = run.parent.parent
    ck = Path(a.checkpoint) if a.checkpoint else sorted((run / "checkpoints").glob("[0-9]*.pt"))[-2]  # (not one being written)
    out = Path(tempfile.mkdtemp(prefix="shopprobe", dir=a.tmp or None))
    cfg, _ = run_config(run, "probe", 1, 0)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg.update(run_dir=str(out), vocab=str(out / "vocab.json"), map=a.map, shop_items=1, z_pool=str(root / c["z_pool"]), z_share=1.0,
               z_reward=float(c["z_reward"]), z_mix_reward=float(c.get("z_mix_reward", 0.0)),
               z_mode_reward=float(c.get("z_mode_reward", 0.0)), z_build=int(c.get("z_build", 0)), z_workers=int(c.get("z_workers", 0)),
               z_modes=float(c.get("z_modes", 0.0)), z_build_labels=int(c.get("z_build_labels", 0)), teach=int(c.get("teach", 0)),
               win_balance=float(c.get("win_balance", 0.0)), scenarios=[], scenario_share=0.0, late_states=[], late_z=0,
               eval_checkpoints=[], max_minutes=a.minutes, max_steps=int(a.minutes * 60 / cfg["step_seconds"]) + 20, device="cpu",
               values=unit_values(), order_names={})
    fx.ACTIVE_F = cfg["features"]
    os.environ["WARCRAFTSIM_FEATURES"] = str(cfg["features"])
    fx.set_inputs(cfg["inputs"], bool(cfg["privileged"]))
    fx.set_masks(cfg["tech_mask"], cfg["valid_mask"])
    fx.set_neutral(cfg["neutral"])
    fx.set_macros(int(cfg.get("macros", 0) or 0))
    fx.set_spells(int(cfg.get("spells", 0) or 0))
    fx.set_shop_items(1)
    fx.set_macros_off(str(cfg.get("macros_off") or "").split(","))
    fx.set_econ_masks(cfg.get("econ_masks", 0))
    fx.set_geometry(*fx.map_geometry(stock_map_path(cfg["map"])))
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    cfg["z_dim"] = raw["config"].get("z", 0)
    fx.set_z(cfg["z_dim"])
    version = int(raw.get("update", raw.get("version", 0)) or 0)
    orders0, types0, allowed0 = len(raw["vocab"]["orders"]), len(raw["vocab"]["types"]), raw["model"]["allowed"].shape
    head0 = raw["model"]["order.2.bias"].clone()
    del raw
    net, got = load(ck, "cpu")
    vocab = got["vocab"]
    print(f"{ck}: orders {orders0} -> {len(vocab['orders'])}, types {types0} -> {len(vocab['types'])}, allowed {tuple(allowed0)} -> "
          f"{tuple(net.allowed.shape)}, free-slot needs {int((net.valid_need == 19).sum())}, shop_items {vocab.get('shop_items')}, old orders' biases the same: "
          f"{bool(torch.equal(head0, net.order[2].bias[:len(head0)]))}", flush=True)
    if a.bias > 0:  # (to see the policy's own orders in a short game: the new rows far less rare than a run starts them)
        with torch.no_grad():
            b = net.order[2].bias
            b[orders0 + 1:] = b[0] - a.bias
        print(f"the new rows' biases: no order's less {a.bias}", flush=True)
    Path(cfg["vocab"]).write_text(json.dumps(vocab))
    selfplay.publish(net, version, out)
    cfg["costs"] = order_costs(vocab, cfg["map"]).tolist() if cfg["avail_mask"] else None
    cfg["costs_array"] = np.asarray(cfg["costs"], np.int64) if cfg["avail_mask"] else None
    cfg["learner_pid"] = os.getpid()
    torch.set_num_threads(a.threads)
    infer = selfplay.Inference(selfplay.Nets(cfg, torch.device("cpu")), torch.device("cpu"), 2, compile_=False, buckets=(1, 2),
                               entities=tuple(cfg["infer_entities"]))

    class Rows:
        def __init__(self):
            self.ep = None

        def put(self, msg):
            if "episode" in msg:
                self.ep = msg["episode"]
    rows = Rows()
    races = [a.race, a.foe]
    slots = [Agent(a.race, handicap=100, difficulty="normal"), Agent(a.foe, handicap=100, difficulty="normal")]
    gs = GameSetup(map=cfg["map"], slots=slots, step_seconds=0.5, draw=False, native_obs=bool(cfg["native_obs"]), d3d_thread=False,
                   render_threads=0, nice=19, warm_spare=False, window=(320, 240), max_game_seconds=a.minutes * 60,
                   victory="decisive", shim_dir=a.shim)
    opp = {"name": "self", "kind": "self"}
    try:
        with GameInstance(gs, name=f"shopprobe{a.name}p", timeout=600) as g:
            obs = g.start()
            ep = selfplay.play_one(g, obs, cfg, vocab, races, 0, opp, infer, rows, 0)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    print(f"RESULT over at {ep['game_time']:.0f} s, outcome {ep['outcome']}, refused share {ep.get('refused_share')}, gold {ep.get('gold')}", flush=True)
    for name in ("buys", "buy_orders", "shop_bought", "merchant_bought", "item_uses", "items_gone", "futile_orders"):
        print(f"  {name}: {ep.get(name)}", flush=True)
    print(f"  built: {(ep.get('prod') or {}).get('built')}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("items", "merchant", "play"))
    ap.add_argument("--race", default="human")
    ap.add_argument("--foe", default="orc")
    ap.add_argument("--map", default="echofast2")
    ap.add_argument("--vocab", default="/home/tora/warcraftsim/runs/fgself-173/vocab.json")
    ap.add_argument("--run", default="/home/tora/warcraftsim/runs/fgself-173")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--name", default="")
    ap.add_argument("--only", default="", help="items: only these items, comma separated")
    ap.add_argument("--bias", type=float, default=0.0, help="play: the new orders at no order's logit less this (0: as grown)")
    ap.add_argument("--tmp", default="")
    ap.add_argument("--shim", default=str(Path(__file__).resolve().parents[1] / "build" / "shim"))
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    fx.set_inputs(2)
    fx.set_masks(True, True)
    fx.set_neutral(2)
    fx.set_macros(3)
    fx.set_spells(1)
    fx.set_shop_items(1)
    fx.set_econ_masks(255)
    if a.what != "play":
        from warcraftsim.data.mapbuild import stock_map_path
        fx.set_geometry(*fx.map_geometry(stock_map_path(a.map)))
    return {"items": items, "merchant": merchant, "play": play}[a.what](a)


if __name__ == "__main__":
    raise SystemExit(main())
