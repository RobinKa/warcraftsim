"""--full-vocab in the real game (fullgame/fullvocab.py): every order the vocabulary gained, given through the policy's own path
(play.BCAgent.observe, its masks, commands, accepted; no network), with the game's answer and what it did read from the game's
state.

    python3 scripts/vocab_probe.py ids                          # the game's order id of every order string (OrderId)
    python3 scripts/vocab_probe.py orders --race human [--only polymorph,patrol]
    python3 scripts/vocab_probe.py items                        # drop, give, sell, pick up; the critter, the tiny hall
    python3 scripts/vocab_probe.py neutral                      # the goblin laboratory: its units, load and unload, reveal

ids: one game whose harness resolves every order string of the data files and of the game's executable (its order string
table) and writes warcraftsim/data/order_ids.json, which fullvocab.order_ids reads. orders: each of the race's new orders
(fullvocab.extend of fgself-194's vocabulary) given once by a unit of a type that has it, placed for it with full mana
(protocol.Place), the race's research set (SetResearch), enemies placed near; printed per order: the mask before it
(View.step's avail), what the agent sent (or "not sent"), the game's answer, and within 4 s what happened (the giver's
order and type, a new unit, an event, the money). items and neutral likewise for the heroes' item orders and the goblin
laboratory (lastrefugefast2). Instance names start with "vocabprobe"; one game at a time.

Results (2026-11-06; echofast2, lastrefugefast2 for the laboratory; one game at a time, each mode once per race):
  ids         347 of 379 strings resolved. The files' sphinxform, unsphinxform, absorbmana, unloadcorpse,
              unstableconcoctionon/off are no order of the game's (OrderId 0): fullvocab.BASE_ORDERS has the game's own.
  taken       trains hmtt (castle, before the barrage), hrtt (after it only), ospm, ehip; upgrades hatw, hctw, uzg2; builds
              owtw, etrp; research Rhfl, Rhrt, Robf, Robk, Rotr, Rowt, Ruex, Reht, Renb, Repb (each its event); patrol, hold
              position, set rally (point and unit); attack ground 300 and 600 away (mortar, catapult, meat wagon, glaive
              thrower; refused ~1,000 away: beyond range, the agent sends none there); flare, magic leash (an air unit),
              polymorph (the footman gone: a sheep), berserk, sentry ward (a ward appears), spirit link, dispel by a spirit
              walker (point or unit: Disenchant's real order; "disenchant" refused), devour magic, absorb (from a
              necromancer: mana drained), possession (the banshee gone into the footman), unsummon (a ziggurat: gone),
              destroyer form (uobs -> ubsp, the difference in gold, lumber and +2 food taken: fullvocab.MORPH_COSTS), bear
              form and back, crow form and back, uncorporeal form (ospw -> ospm), the hippogryph's pick up of an archer
              (both gone, a rider made), decouple (rider -> archer and hippogryph), cyclone, recharge (a hurt archer
              healed), searing arrows on a unit (flamingarrowstarg), entangle, the acolyte's requestsacrifice on the pit and
              the pit's sacrifice on an acolyte (a shade each), drop an item at a point, pick it up near the point with
              another hero, give an item to another hero (the pointer on itself: the agent's pick), sell to the race's
              shop (+325 gold for a town portal) and to the goblin merchant (walking), a zeppelin bought at the goblin
              laboratory, its reveal, the zeppelin's load and its unloadall (at a point and at once).
  toggles     each on or off is refused when the unit is in that state already: a computer player's units start with
              heal, slow, spell steal, curse, web, faerie fire, phase shift, abolish magic, vengeance on (their "off"
              taken on a fresh unit), inner fire, repair, frost armor, raise dead, carrion scarabs, restoration, recharge,
              renew, replenish off. The agent tracks each unit's state (fullvocab.DEFAULT_ON, then its own orders; a
              refusal means the state was so already).
  refused     corporeal form on a fresh ospm: the engine turns a computer slot's spirit walkers corporeal by itself
              within 2 s; a shadow meld by day (night only); the zeppelin's "unload" at a point; the meat wagon's drop all
              corpses taken, its get corpse and a spirit walker's ancestral spirit never (a corpse is no entity, and a
              harness order on the nearest corpse found none: left out); the arcane tower's reveal in every form (left
              out); freezing and corrosive breath on a visible building (passive: left out); undivineshield before,
              during and after the shield (left out); unavengerform (left out); an ancient uprooted, eaten a tree, but
              never rooted again at its place, 500 away or at once (unroot, root, eat tree left out); the tiny great hall
              bought (600 gold) and gone from the buyer within 5 s (left out); the laboratory's sapper and shredder not
              sold at ~450 s on lastrefugefast2 (masked: stock).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("WAYLAND_DISPLAY", None)  # (games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXE = Path.home() / "wc3/legacy-1.29/Warcraft III.exe"
VOCAB = "/home/tora/warcraftsim/runs/fgself-194/vocab.json"
ARMY = {"human": "hfoo", "orc": "ogru", "undead": "ugho", "nightelf": "earc"}
HEROES = {"human": ("Hpal", "Hamg"), "orc": ("Ofar", "Oshd"), "undead": ("Ulic", "Ucrl"), "nightelf": ("Emoo", "Ekee")}
FOES = ("hfoo", "hfoo", "owyv", "hbar")  # (player 1's, near: ground and air units to aim at, a building)


def exe_order_strings(exe: Path = EXE) -> list[str]:
    """The game's order strings, as its executable lists them (the block from "attackground" to "neutralspell")."""
    data = exe.read_bytes()
    words = [w.decode("latin-1") for w in re.findall(rb"[\x20-\x7e]{3,}", data)]
    a = words.index("attackground")
    b = words.index("neutralspell", a)
    return [w for w in words[a:b + 1] if re.fullmatch(r"[A-Za-z_0-9]+", w)]


def game(a, race: str, foe: str, map_name: str | None = None):
    from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup
    slots = [Agent(race, handicap=100, difficulty="normal"), Agent(foe, handicap=100, difficulty="normal")]
    gs = GameSetup(map=map_name or a.map, slots=slots, step_seconds=0.5, draw=False, native_obs=False, d3d_thread=False,
                   render_threads=0, nice=19, warm_spare=False, window=(320, 240), max_game_seconds=0, victory="none",
                   shim_dir=a.shim)
    return GameInstance(gs, name=f"vocabprobe{a.name}", timeout=900)


def ids(a) -> int:
    from warcraftsim.data import mapbuild
    from warcraftsim.protocol import all_order_names
    extra = ["smart", "stop", "cancel", "instant", "stunned", "setrally", "patrol", "holdposition", "AImove",
             *(f"useslot{k}" for k in range(1, 7)), *(f"moveslot{k}" for k in range(1, 7)), *exe_order_strings()]
    names = tuple(dict.fromkeys(all_order_names() + tuple(extra)))  # (the data files' strings, then the executable's)
    mapbuild.HarnessConfig.resolved_order_names = lambda self: names  # (this process only: its own map build)
    with game(a, "human", "orc") as g:
        obs = g.start()
        got = dict(sorted((k, int(v)) for k, v in (obs.orders or {}).items()))
    missing = [n for n in names if n not in got]
    print(f"RESULT {len(got)} of {len(names)} order strings resolved; none for {missing}", flush=True)
    out = ROOT / "warcraftsim" / "data" / "order_ids.json"
    out.write_text(json.dumps(got, indent=0, sort_keys=True) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


class Probe:
    """A game driven through the agent's own path (as scripts/shop_items_probe.py's): BCAgent's view, masks, commands."""

    def __init__(self, g, race: str, vocab: dict, map_name: str):
        from warcraftsim.fullgame import features as fx
        from warcraftsim.fullgame.costs import order_costs, research_steps
        from warcraftsim.fullgame.play import BCAgent
        self.fx, self.g, self.race = fx, g, race
        self.obs = g.start()
        self.bot = BCAgent(None, vocab, 0, None, costs=order_costs(vocab, map_name), steps=research_steps(vocab, map_name))
        self.bot.begin(self.obs, [fx.RACES.index(race), (fx.RACES.index(race) + 1) % 4])
        self.t, self.st, self.events, self.sent = 0, None, [], []
        p = self.obs.players[0]
        self.base = (p.start_x, p.start_y)
        self.dx = 1 if p.start_x < 0 else -1
        self.tick()

    def tick(self, scripted=(), orders=None):
        fx, bot = self.fx, self.bot
        st = self.st = bot.observe(self.obs, self.t)
        cmds = []
        if st is not None:
            n = min(st["n_own"], fx.MAX_OWN)
            sel = st["sel"]
            order, tgt, bx, by = (np.zeros(n, np.int64) for _ in range(4))
            at = {int(u): i for i, u in enumerate(sel[:, fx.C_ID].tolist())}
            for u, (c, on, xy) in (orders or {}).items():
                if u in at and at[u] < n:
                    order[at[u]] = c
                    tgt[at[u]] = at.get(on, at[u]) if on is not None else at[u]
                    if xy is not None:
                        wx, wy = fx.world_bins(np.float64(xy[0]), np.float64(xy[1]), bot.view.sign)
                        bx[at[u]], by[at[u]] = int(wx), int(wy)
            cmds = bot.commands(st, order.tolist(), tgt.tolist(), bx.tolist(), by.tolist())
        self.sent = cmds
        scripted = list(scripted)
        self.obs = self.g.step(scripted + cmds)
        self.events += list(self.obs.events)
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

    def place(self, player: int, kind: str, x: float, y: float, **kw):
        from warcraftsim.protocol import Place
        before = {u.id for u in self.obs.units}
        self.tick([Place(player, kind, x, y, **kw)])
        new = [u for u in self.obs.units if u.id not in before and u.owner == player and u.type == kind]
        return new[0].id if new else None


def research_all(p, race: str):
    """Every research of the race at its last level (SetResearch), money for everything."""
    from warcraftsim.fullgame import techtree as tt
    from warcraftsim.protocol import SetResearch, SetResources
    letter = race[0] if race != "nightelf" else "e"
    codes = sorted(c for c, makers in tt.tree().makers.items() if c[0] == "R" and any(m[0] == letter for m in makers))
    lv = tt.research_levels()
    p.tick([SetResources(0, 90000, 90000), SetResources(1, 90000, 90000)]
           + [SetResearch(0, c, lv.get(c, 1)) for c in codes])
    for c in codes:  # (SetResearch sends no event: the view's levels, which its tech mask reads, told here)
        p.bot.view.levels[int.from_bytes(c.encode(), "big")] = lv.get(c, 1)
    return codes


def snapshot(p, uid):
    u = p.obs.unit(uid) if uid is not None else None
    return None if u is None else (u.type, u.hp, u.mana, u.order, u.x, u.y, u.alive)


def try_order(p, c: int, giver: int, tgt=None, xy=None, steps: int = 8, label: str = "") -> dict:
    """One order through the agent's path -> what the game did."""
    from warcraftsim.fullgame import features as fx
    p.tick()
    avail = bool(p.st["avail"][c]) if p.st is not None and c < len(p.st["avail"]) else None
    g0, t0 = snapshot(p, giver), snapshot(p, tgt)
    own0 = {u.id: u.type for u in p.obs.units if u.owner == 0 and u.alive}
    money0 = (p.obs.players[0].gold, p.obs.players[0].lumber, p.obs.players[0].food_used)
    p.events = []
    _, res = p.tick(orders={giver: (c, tgt, xy)})
    sent = [type(x).__name__ + str(tuple(v for k, v in vars(x).items() if k != "unit")) for x in p.sent]
    order1 = snapshot(p, giver)
    p.wait(steps)
    g1, t1 = snapshot(p, giver), snapshot(p, tgt)
    new = Counter(u.type for u in p.obs.units if u.owner == 0 and u.alive and u.id not in own0)
    gone = Counter(t for i, t in own0.items() if p.obs.unit(i) is None or not p.obs.unit(i).alive)
    money1 = (p.obs.players[0].gold, p.obs.players[0].lumber, p.obs.players[0].food_used)
    ev = Counter(int(e.kind) for e in p.events if giver in (e.a, e.b) or (tgt is not None and tgt in (e.a, e.b, e.c)))
    o, k = p.bot.orders[c]
    name = p.bot.full.names.get(c, str(c)) if p.bot.full is not None else str(c)
    out = {"order": name, "label": label, "mask": avail, "sent": sent, "answer": res, "giver": g0 and g0[0],
           "giver_order_after": order1 and order1[3], "giver_after": g1 and (g1[0], g1[2], g1[6]), "mana": (g0 and g0[2], g1 and g1[2]),
           "target": t0 and (t0[0], t0[1]), "target_after": t1 and (t1[0], t1[1], t1[6]), "new": dict(new), "gone": dict(gone),
           "money": tuple(b - a for a, b in zip(money0, money1)), "events": dict(ev)}
    print("RESULT " + json.dumps(out), flush=True)
    return out


def orders(a) -> int:
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame import fullvocab as fv
    from warcraftsim.protocol import ImmediateOrder, SetResources
    race = a.race
    base = json.loads(Path(a.vocab).read_text())
    vocab = fv.extend(base)
    O0 = len(base["orders"]) + 1
    types = fv.unit_types(race)
    letter = {"human": "h", "orc": "o", "undead": "u", "nightelf": "e"}[race]
    tests = []
    for c, (o, k) in enumerate(vocab["orders"], start=1):
        if c < O0:
            continue
        e = fv.entry(o, k)
        if e is None or e.what in ("item", "ware", "neutral"):
            continue
        givers = sorted(g for g in e.givers if g in types and (g[0] == letter or g in HEROES[race]))
        if givers:
            tests.append((c, e, givers))
    only = set(a.only.split(",")) if a.only else None
    if only:
        tests = [t for t in tests if t[1].name in only]
    print(f"{race}: {len(tests)} orders to try: {[t[1].name for t in tests]}", flush=True)
    foe = "orc" if race != "orc" else "human"
    from warcraftsim.data.abilities import hero_abilities
    from warcraftsim.fullgame import techtree as tt
    U = fv.tables()["units"]
    first = ("train", "research", "upgrade", "build")  # (before the research is set: a research at its last level is masked)
    tests.sort(key=lambda t: (t[1].what not in first, t[0]))
    with game(a, race, foe) as g:
        p = Probe(g, race, vocab, a.map)
        bx, by = p.base
        dx = p.dx
        before = {t[1].name: bool(p.st["avail"][t[0]]) for t in tests if t[1].needs}
        print(f"masks with no research (False: masked): {before}", flush=True)
        p.tick([SetResources(0, 90000, 90000)])
        blds = sorted(u for u in types if U.get(u, {}).get("building") and u[0] == letter and u not in ("ugol", "egol")
                      and u not in tt.TIERS[race][:2])
        placed: dict[str, int] = {}
        for i, b in enumerate(blds):  # (the race's every building, finished, in a grid behind the base)
            placed[b] = p.place(0, b, bx - dx * (700 + 260 * (i % 5)), by - 900 + 260 * (i // 5))
        p.wait(2)
        print(f"buildings placed {sorted(k for k, v in placed.items() if v)}; not: {sorted(k for k, v in placed.items() if not v)}", flush=True)
        foes = [p.place(1, f, bx + dx * (900 + 120 * i), by + 300) for i, f in enumerate(FOES)]
        own_army = p.place(0, ARMY[race], bx + dx * 500, by - 200)
        heroes = {}
        for i, h in enumerate(HEROES[race]):
            ab = hero_abilities()[h]
            heroes[h] = p.place(0, h, bx + dx * 400, by - 400 + 120 * i, hero_level=10,
                                skills=tuple(s for s in ab[:3] for _ in range(3)) + tuple(ab[3:4]))
        print(f"foes {foes} army {own_army} heroes {heroes}", flush=True)
        A = fv.tables()["abilities"]
        results = []
        researched = False
        for c, e, givers in tests:
            if e.what not in first and not researched:
                research_all(p, race)
                researched = True
                p.wait(2)
            gtype = givers[0]
            gid = heroes.get(gtype) or placed.get(gtype)
            if gid is None or p.obs.unit(gid) is None or not p.obs.unit(gid).alive:
                spot = (bx + dx * (300 + 40 * (len(placed) % 8)), by + 500 + 40 * (len(placed) // 8))
                gid = placed[gtype] = p.place(0, gtype, *spot)
            if gid is None:
                print(f"RESULT {json.dumps({'order': e.name, 'error': f'could not place {gtype}'})}", flush=True)
                continue
            gu = p.obs.unit(gid)
            targs = A.get(e.ability, {}).get("targs", "") if e.ability else ""
            tgt, xy = None, None
            alive_foes = [f for f in foes if f is not None and p.obs.unit(f) is not None and p.obs.unit(f).alive]
            if e.kind == fx.UNIT:
                if "enemy" in targs and "air" in targs and "ground" not in targs:
                    tgt = next((f for f in alive_foes if p.obs.unit(f).type == "owyv"), None)
                elif "enemy" in targs and "structure" in targs and "ground" not in targs:
                    tgt = next((f for f in alive_foes if p.obs.unit(f).type == "hbar"), None)
                elif "enemy" in targs or e.what == "generic":
                    tgt = next((f for f in alive_foes if p.obs.unit(f).type == "hfoo"), None)
                else:
                    tgt = own_army
            elif e.kind in (fx.POINT, fx.TREE):
                if e.kind == fx.TREE:
                    trees = sorted(p.bot.trees.items(), key=lambda kv: (kv[1][0] - gu.x) ** 2 + (kv[1][1] - gu.y) ** 2)
                    xy = trees[0][1] if trees else (gu.x + 200, gu.y)
                elif "enemy" in targs and alive_foes:
                    f = p.obs.unit(alive_foes[0])
                    xy = (f.x, f.y)
                elif e.what == "build":
                    xy = (bx + dx * 900, by - 900)
                else:
                    xy = (gu.x + dx * 250, gu.y)
            results.append(try_order(p, c, gid, tgt, xy, label=gtype))
            if e.what in ("train", "research", "upgrade"):  # (its queue emptied again)
                p.tick([ImmediateOrder(gid, 851976)] * 3)
            p.tick([SetResources(0, 90000, 90000)])
        took = [r["order"] for r in results if r.get("answer") and any(r["answer"])]
        print(f"SUMMARY {race}: {len(took)} of {len(results)} taken; refused or not sent: "
              f"{[(r['order'], r['sent'][:1]) for r in results if not (r.get('answer') and any(r['answer']))]}", flush=True)
    return 0


def variants(a) -> int:
    """Scripted commands (not the agent's) for what the orders run left open: each race's forms of the refused orders
    (attack ground near, the arcane tower's reveal, the destroyer's absorb, the acolyte's sacrifice both ways, corpses,
    the hippogryph's two halves, an ancient uprooted, autocast states), printed as the game's answer and the units'
    (type, hp, mana, order, flags) before and after."""
    from warcraftsim.data.abilities import hero_abilities
    from warcraftsim.fullgame import fullvocab as fv
    from warcraftsim.protocol import (ImmediateOrder, LearnSkill, Place, PointOrder, SetResearch, SetResources,
                                      TargetDestructable, TargetOrder)
    race = a.race
    ids = fv.order_ids()
    vocab = fv.extend(json.loads(Path(a.vocab).read_text()))
    foe = "orc" if race != "orc" else "human"

    def snap(p, u):
        x = p.obs.unit(u) if u is not None else None
        return None if x is None else (x.type, x.hp, x.mana, x.order, int(x.flags), x.alive)

    with game(a, race, foe) as g:
        p = Probe(g, race, vocab, a.map)
        bx, by = p.base
        dx = p.dx
        research_all(p, race)

        def put(player, kind, ox, oy, **kw):
            return p.place(player, kind, bx + dx * ox, by + oy, **kw)

        def do(label, cmds, watch, steps=8):
            before = [snap(p, u) for u in watch]
            res, _ = p.tick(cmds)
            first = [snap(p, u) for u in watch]
            p.wait(steps)
            after = [snap(p, u) for u in watch]
            print("RESULT " + json.dumps({"label": label, "answer": res, "before": before, "1 step": first, "after": after}),
                  flush=True)
            return res

        def foe_near(kind, ox, oy):
            return put(1, kind, ox, oy)
        siege = {"human": "hmtm", "orc": "ocat", "undead": "umtw", "nightelf": "ebal"}[race]
        s = put(0, siege, 300, 600)
        su = p.obs.unit(s)
        do("attackground 300 away", [PointOrder(s, ids["attackground"], su.x + dx * 300, su.y)], [s])
        do("attackground 600 away", [PointOrder(s, ids["attackground"], su.x + dx * 600, su.y)], [s])
        if race == "human":
            t = put(0, "hatw", 200, 900)
            tu = p.obs.unit(t)
            for nm in ("detectaoe", "neutraldetectaoe"):
                do(f"arcane tower {nm} at a point", [PointOrder(t, ids[nm], tu.x + 600, tu.y)], [t])
            do("arcane tower detectaoe immediate", [ImmediateOrder(t, ids["detectaoe"])], [t])
            w = put(0, "harm", -600, -600)
            do("train hrtt with Rhrt", [ImmediateOrder(w, fv.code("hrtt"))], [w])
            do("train hmtt with Rhrt", [ImmediateOrder(w, fv.code("hmtt"))], [w])
            pal = put(0, "Hpal", 300, -300, hero_level=6, skills=("AHhb", "AHds", "AHds", "AHad", "AHre"))
            do("undivineshield before", [ImmediateOrder(pal, ids["undivineshield"])], [pal], 2)
            do("divineshield", [ImmediateOrder(pal, ids["divineshield"])], [pal], 2)
            do("undivineshield during", [ImmediateOrder(pal, ids["undivineshield"])], [pal], 2)
            pr = put(0, "hmpr", 350, -200)
            do("innerfireon", [ImmediateOrder(pr, ids["innerfireon"])], [pr], 2)
            do("innerfireoff after on", [ImmediateOrder(pr, ids["innerfireoff"])], [pr], 2)
        if race == "orc":
            sw = put(0, "ospm", 300, -300)
            first = snap(p, sw)
            p.wait(4)
            print("placed ospm:", first, "2 s later:", snap(p, sw), flush=True)
            do("corporealform", [ImmediateOrder(sw, ids["corporealform"])], [sw])
            do("uncorporealform", [ImmediateOrder(sw, ids["uncorporealform"])], [sw])
            f1 = foe_near("hfoo", 450, -300)
            fu = p.obs.unit(f1)
            for nm in ("disenchant", "dispel"):
                do(f"{nm} at a point", [PointOrder(sw, ids[nm], fu.x, fu.y)], [sw])
                do(f"{nm} on a unit", [TargetOrder(sw, ids[nm], f1)], [sw])
            tau = put(0, "otau", 420, -280, hp_permille=5)
            kill = [TargetOrder(f1, ids["attack"], tau)]
            for _ in range(30):
                p.tick(kill)
                x = p.obs.unit(tau)
                if x is None or not x.alive:
                    break
            print("tauren", snap(p, tau), flush=True)
            for nm in ("ancestralspirittarget", "ancestralspirit"):
                do(f"{nm} on the dead tauren", [TargetOrder(sw, ids[nm], tau)], [sw, tau], 6)
            do("ancestralspirit immediate", [ImmediateOrder(sw, ids["ancestralspirit"])], [sw, tau], 6)
        if race == "undead":
            d = put(0, "ubsp", 300, -300)
            ne = put(0, "unec", 350, -300)
            do("absorb from a necromancer", [TargetOrder(d, ids["absorb"], ne)], [d, ne])
            ob = put(0, "uobs", 300, -450)
            for nm in ("replenishlifeoff", "replenishlifeon", "replenishmanaoff", "replenishmanaon"):
                do(nm, [ImmediateOrder(ob, ids[nm])], [ob], 2)
            do("unavengerform", [ImmediateOrder(d, ids["unavengerform"])], [d])
            pit = put(0, "usap", -500, 500)
            ac = put(0, "uaco", -420, 450)
            ac2 = put(0, "uaco", -420, 500)
            do("acolyte sacrifice on the pit", [TargetOrder(ac, ids["sacrifice"], pit)], [ac, pit], 30)
            do("acolyte requestsacrifice on the pit", [TargetOrder(ac2, ids["requestsacrifice"], pit)], [ac2, pit], 30)
            ac3 = put(0, "uaco", -420, 550)
            do("pit sacrifice on an acolyte", [TargetOrder(pit, ids["sacrifice"], ac3)], [ac3, pit], 30)
            print("shades", [u.id for u in p.own("ushd")], flush=True)
            z = put(0, "uzig", -700, 300)
            ac4 = put(0, "uaco", -650, 300)
            do("unsummon a ziggurat", [TargetOrder(ac4, ids["unsummon"], z)], [ac4, z], 20)
            fb = foe_near("hbar", 600, -500)
            fw = put(0, "ufro", 450, -500)
            do("freezingbreath on a visible enemy building", [TargetOrder(fw, ids["freezingbreath"], fb)], [fw, fb])
            bn = put(0, "uban", 400, -200)
            f2 = foe_near("hfoo", 550, -200)
            do("possession of a visible footman", [TargetOrder(bn, ids["possession"], f2)], [bn, f2], 10)
            mw = put(0, "umtw", 300, 300)
            f3 = foe_near("hfoo", 380, 300)
            for _ in range(40):
                p.tick([TargetOrder(mw, ids["attack"], f3)])
                x = p.obs.unit(f3)
                if x is None or not x.alive:
                    break
            print("the footman", snap(p, f3), flush=True)
            do("loadcorpse on the dead footman", [TargetOrder(mw, ids["loadcorpse"], f3)], [mw, f3], 10)
        if race == "nightelf":
            ar = put(0, "earc", 300, -300)
            hp_ = put(0, "ehip", 360, -300)
            do("archer coupletarget on a hippogryph", [TargetOrder(ar, ids["coupletarget"], hp_)], [ar, hp_], 10)
            print("riders", [u.id for u in p.own("ehpr")], flush=True)
            ar2 = put(0, "earc", 300, -400)
            hp2 = put(0, "ehip", 360, -400)
            do("hippogryph coupletarget on an archer", [TargetOrder(hp2, ids["coupletarget"], ar2)], [ar2, hp2], 10)
            dt = put(0, "edot", 400, -200)
            f1 = foe_near("hfoo", 550, -200)
            do("cyclone on a visible footman", [TargetOrder(dt, ids["cyclone"], f1)], [dt, f1])
            fb = foe_near("hbar", 600, -500)
            ch = put(0, "echm", 450, -500)
            do("corrosivebreath on a visible enemy building", [TargetOrder(ch, ids["corrosivebreath"], fb)], [ch, fb])
            an = put(0, "eaom", -700, 300)
            do("unroot", [ImmediateOrder(an, ids["unroot"])], [an], 12)
            au = p.obs.unit(an)
            trees = sorted(p.bot.trees.items(), key=lambda kv: (kv[1][0] - au.x) ** 2 + (kv[1][1] - au.y) ** 2)
            do("eattree uprooted", [TargetDestructable(an, ids["eattree"], trees[0][0])], [an], 12)
            do("root at a point", [PointOrder(an, ids["root"], au.x + 300, au.y)], [an], 30)
            mw = put(0, "emow", -500, 600)
            ww = put(0, "earc", -450, 600, hp_permille=300)
            do("recharge a hurt archer", [TargetOrder(mw, ids["recharge"], ww)], [mw, ww])
            pm = put(0, "Emoo", 200, -100, hero_level=6, skills=("AHfa", "AHfa", "AEst", "AEar", "AEsf"))
            f4 = foe_near("hfoo", 400, -100)
            do("flamingarrowstarg on a visible footman", [TargetOrder(pm, ids["flamingarrowstarg"], f4)], [pm, f4], 3)
            do("flamingarrows (autocast on)", [ImmediateOrder(pm, ids["flamingarrows"])], [pm], 2)
            do("unflamingarrows (off)", [ImmediateOrder(pm, ids["unflamingarrows"])], [pm], 2)
            do("ambush (time of day %.0f)" % p.obs.game_time, [ImmediateOrder(pm, ids["ambush"])], [pm], 2)
            mine = next((u.id for u in p.own("egol")), None)
            do("entangled mine unload", [ImmediateOrder(mine, ids["unload"])] if mine else [], [mine], 4)
            wi = put(0, "ewsp", -300, 300)
            for nm in ("renewon", "renewoff"):
                do(nm, [ImmediateOrder(wi, ids[nm])], [wi], 2)
    return 0


def corpses(a) -> int:
    """The orders on a corpse through protocol.CorpseOrder (the harness reports a unit's death, then no more of it): a meat
    wagon's get corpse, a spirit walker's ancestral spirit; a siege engine with the barrage researched; an ancient's root."""
    from warcraftsim.fullgame import fullvocab as fv
    from warcraftsim.protocol import CorpseOrder, ImmediateOrder, PointOrder, TargetOrder
    race = a.race
    ids = fv.order_ids()
    vocab = fv.extend(json.loads(Path(a.vocab).read_text()))
    with game(a, race, "orc" if race != "orc" else "human") as g:
        p = Probe(g, race, vocab, a.map)
        bx, by = p.base
        dx = p.dx
        research_all(p, race)

        def put(player, kind, ox, oy, **kw):
            return p.place(player, kind, bx + dx * ox, by + oy, **kw)

        def snap(u):
            x = p.obs.unit(u) if u is not None else None
            return None if x is None else (x.type, x.hp, x.mana, x.order, int(x.flags), x.alive)

        def do(label, cmds, watch, steps=8):
            before = [snap(u) for u in watch]
            res, _ = p.tick(cmds)
            p.wait(steps)
            new = Counter(u.type for u in p.obs.units if u.owner == 0 and u.alive and u.id not in seen)
            print("RESULT " + json.dumps({"label": label, "answer": res, "before": before, "after": [snap(u) for u in watch],
                                          "new own units": dict(new)}), flush=True)

        def kill(victim, killer):
            for _ in range(60):
                p.tick([TargetOrder(killer, ids["attack"], victim)])
                x = p.obs.unit(victim)
                if x is None or not x.alive:
                    return True
            return False
        seen = {u.id for u in p.obs.units}
        if race == "undead":
            mw = put(0, "umtw", 300, 300)
            gh = put(0, "ugho", 360, 300)
            f = put(1, "hfoo", 380, 300, hp_permille=50)
            print("footman killed by a ghoul (a corpse: an artillery kill explodes)", kill(f, gh), flush=True)
            p.wait(4)
            seen = {u.id for u in p.obs.units}
            do("meat wagon get corpse (CorpseOrder)", [CorpseOrder(mw, ids["loadcorpse"], 700)], [mw], 12)
            do("meat wagon drop all corpses", [ImmediateOrder(mw, ids["unloadallcorpses"])], [mw], 6)
            f2 = put(1, "hfoo", 380, 360, hp_permille=50)
            print("another footman killed by the ghoul", kill(f2, gh), flush=True)
            ne = put(0, "unec", 320, 360)
            do("necromancer raise dead at once (immediate: a corpse there?)", [ImmediateOrder(ne, ids["raisedead"])], [ne], 6)
        if race == "orc":
            sw = put(0, "ospw", 300, -300)
            tau = put(0, "otau", 420, -280, hp_permille=20)
            f = put(1, "hfoo", 480, -280)
            print("tauren killed", kill(tau, f), flush=True)
            p.wait(2)
            seen = {u.id for u in p.obs.units}
            do("ancestral spirit on the nearest dead tauren of its own (CorpseOrder)",
               [CorpseOrder(sw, ids["ancestralspirittarget"], 700, fv.code("otau"), True)], [sw], 12)
            do("ancestral spirit, any corpse near (CorpseOrder)", [CorpseOrder(sw, ids["ancestralspirittarget"], 700)], [sw], 12)
            do("ancestralspirit (the other string) on the nearest corpse",
               [CorpseOrder(sw, ids["ancestralspirit"], 700)], [sw], 12)
        if race == "human":
            put(0, "hcas", -900, -900)
            w = put(0, "harm", -600, -600)
            p.wait(2)
            do("train hrtt with Rhrt and a castle", [ImmediateOrder(w, fv.code("hrtt"))], [w], 2)
            do("train hmtt with Rhrt and a castle", [ImmediateOrder(w, fv.code("hmtt"))], [w], 2)
            p.wait(120)
            print("siege engines now", [(u.type, u.id) for u in p.obs.units if u.owner == 0 and u.type in ("hmtt", "hrtt")],
                  flush=True)
        if race == "nightelf":
            an = put(0, "eaom", -700, 300)
            do("unroot", [ImmediateOrder(an, ids["unroot"])], [an], 12)
            au = p.obs.unit(an)
            do("root at its own place", [PointOrder(an, ids["root"], au.x, au.y)], [an], 30)
            do("unroot again", [ImmediateOrder(an, ids["unroot"])], [an], 12)
            au = p.obs.unit(an)
            do("root immediate", [ImmediateOrder(an, ids["root"])], [an], 30)
            do("root 500 away, open ground", [PointOrder(an, ids["root"], au.x + dx * 500, au.y + 500)], [an], 40)
    return 0


def items(a) -> int:
    """The heroes' item orders: drop at a point, give to another hero, sell to the race's shop and the goblin merchant,
    pick up near a point; the mechanical critter (human) and the tiny great hall (orc) bought and used."""
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame import fullvocab as fv
    from warcraftsim.protocol import SetResources
    race = a.race
    base = json.loads(Path(a.vocab).read_text())
    vocab = fv.extend(base)
    oix = {tuple(o): c for c, o in enumerate(vocab["orders"], start=1)}
    foe = "orc" if race != "orc" else "human"
    shop = {"human": "hvlt", "orc": "ovln", "undead": "utom", "nightelf": "eden"}[race]
    with game(a, race, foe) as g:
        p = Probe(g, race, vocab, a.map)
        bx, by = p.base
        dx = p.dx
        research_all(p, race)
        h1 = p.place(0, HEROES[race][0], bx + dx * 400, by - 300, hero_level=3, items=("phea", "pman", "stwp"))
        h2 = p.place(0, HEROES[race][1], bx + dx * 500, by - 300, hero_level=3)
        s = p.place(0, shop, bx + dx * 600, by - 600)
        if race == "orc":
            p.place(0, "ofrt", bx + dx * 1100, by + 600)
        print(f"heroes {h1} {h2}, shop {s}; items {[fv.rawcode(i) if i else '-' for i in p.obs.unit(h1).items]}", flush=True)
        drop, give, sell = (oix[(fv.DROP0, fx.POINT)], oix[(fv.GIVE0 + 1, fx.UNIT)], oix[(fv.SELL0 + 2, fx.UNIT)])
        h = p.obs.unit(h1)
        at = (h.x + dx * 200, h.y + 100)
        r = try_order(p, drop, h1, None, at, label="drop slot 1 (phea) at a point")
        print(f"   ground now {[(fv.rawcode(t), x, y) for t, x, y, v, pw in p.obs.ground]}", flush=True)
        try_order(p, oix[(fv.PICKUP_AT, fx.POINT)], h2, None, at, steps=12, label="the other hero picks it up near the point")
        print(f"   hero 2 items {[fv.rawcode(i) if i else '-' for i in p.obs.unit(h2).items]}", flush=True)
        try_order(p, give, h1, h1, None, steps=12, label="give slot 2 (pman), pointer on itself: the other hero by the fix-up")
        print(f"   items {[fv.rawcode(i) if i else '-' for i in p.obs.unit(h1).items]} / {[fv.rawcode(i) if i else '-' for i in p.obs.unit(h2).items]}", flush=True)
        g0 = p.obs.players[0].gold
        try_order(p, sell, h1, s, None, steps=30, label="sell slot 3 (stwp) to its race's shop")
        print(f"   gold {g0} -> {p.obs.players[0].gold}; items {[fv.rawcode(i) if i else '-' for i in p.obs.unit(h1).items]}", flush=True)
        merchant = next((u.id for u in p.obs.units if u.type == "ngme"), None)
        if merchant is not None:
            p.tick([SetResources(0, 5000, 5000)])
            h = p.obs.unit(h2)
            k = next((k for k, i in enumerate(h.items) if i), None)
            if k is not None:
                g0 = p.obs.players[0].gold
                try_order(p, oix[(fv.SELL0 + k, fx.UNIT)], h2, merchant, None, steps=120, label=f"sell slot {k + 1} to the goblin merchant (walking)")
                print(f"   gold {g0} -> {p.obs.players[0].gold}", flush=True)
        for c_ in fv.EXTRA_ITEMS.get(shop, ()):
            p.tick([SetResources(0, 5000, 5000)])
            k = oix.get((fv.code(c_), fx.UNIT))
            hh = p.place(0, HEROES[race][0], p.obs.unit(s).x + 150, p.obs.unit(s).y, hero_level=1)
            r = try_order(p, k, hh, s, None, steps=10, label=f"buy {c_} at {shop}")
            hu = p.obs.unit(hh)
            slot = next((j for j, i in enumerate(hu.items) if i == fv.code(c_)), None)
            print(f"   {c_} held in slot {slot}", flush=True)
            if slot is not None:
                use = oix[(fx.USESLOT + slot, fx.IMMEDIATE if c_ == "mcri" else fx.POINT)]
                at = (bx + dx * 1300, by - 1300)
                n0 = {u.id for u in p.obs.units}
                try_order(p, use, hh, None, at, steps=40, label=f"use {c_}")
                print(f"   new: {[(u.type, u.owner, u.x, u.y) for u in p.obs.units if u.id not in n0]}", flush=True)
    return 0


def neutral(a) -> int:
    """The goblin laboratory (lastrefugefast2): a zeppelin, a shredder and a sapper bought by a hero beside it, its reveal;
    the zeppelin loads a footman and unloads it at a point, the sapper blows up at an enemy building, the shredder chops."""
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame import fullvocab as fv
    from warcraftsim.protocol import Place, SetResources
    race = a.race
    base = json.loads(Path(a.vocab).read_text())
    vocab = fv.extend(base)
    oix = {tuple(o): c for c, o in enumerate(vocab["orders"], start=1)}
    with game(a, race, "orc" if race != "orc" else "human", map_name=a.lab_map) as g:
        p = Probe(g, race, vocab, a.lab_map)
        lab = min((u for u in p.obs.units if u.type == "ngad"), key=lambda u: u.dist(*p.base))
        print(f"lab {lab.id} at {lab.x},{lab.y}; base {p.base}", flush=True)
        from warcraftsim.protocol import ClearCreeps, ImmediateOrder, PointOrder
        p.tick([SetResources(0, 9000, 9000), ClearCreeps(lab.x, lab.y, 1500)])  # (its guards: they had killed the buyer)
        p.wait(int(450 / 0.5) - p.t)  # (its units in stock from 440 s: UnitBalance stockStart)
        p.tick([ClearCreeps(lab.x, lab.y, 1500)])
        hero = p.place(0, HEROES[race][0], lab.x + 150, lab.y + 150, hero_level=1)
        worker = p.place(0, {"human": "hpea", "orc": "opeo", "undead": "uaco", "nightelf": "ewsp"}[race], lab.x + 180, lab.y + 100)
        p.tick([SetResources(0, 9000, 9000)])
        for c_ in fv.LAB_UNITS:
            try_order(p, oix[(fv.code(c_), fx.UNIT)], hero, lab.id, None, steps=6, label=f"buy {c_}")
            p.tick([SetResources(0, 9000, 9000)])
        try_order(p, oix[(fv.code("ngsp"), fx.UNIT)], worker, lab.id, None, steps=6, label="a worker buys a sapper")
        at = (lab.x + 600, lab.y)
        g0 = p.obs.players[0].gold
        try_order(p, oix[(fv.order_ids()["neutraldetectaoe"], fx.POINT)], hero, None, (p.base[0], p.base[1] + 3000), steps=4,
                  label="the lab's reveal far away")
        print(f"   gold {g0} -> {p.obs.players[0].gold}", flush=True)
        zep = next((u.id for u in p.own("nzep")), None)
        ids = fv.order_ids()
        for k, (label, cmd) in enumerate((("unload at a point", lambda z, x, y: PointOrder(z, ids["unload"], x, y)),
                                          ("unloadall at a point", lambda z, x, y: PointOrder(z, ids["unloadall"], x, y)),
                                          ("unloadall at once", lambda z, x, y: ImmediateOrder(z, ids["unloadall"])))):
            foot = p.place(0, "hfoo", lab.x + 100, lab.y - 150 - 60 * k)
            if zep is None or foot is None:
                break
            try_order(p, oix[(ids["load"], fx.UNIT)], zep, foot, None, steps=12, label="zeppelin loads a footman")
            z = p.obs.unit(zep)
            res, _ = p.tick([cmd(zep, z.x + 300, z.y)])
            p.wait(16)
            print("RESULT " + json.dumps({"label": f"zeppelin {label}", "answer": res,
                                          "footman": snapshot(p, foot)}), flush=True)
        sap = next((u.id for u in p.own("ngsp")), None)
        foe_b = None
        p.tick([Place(1, "hbar", lab.x - 500, lab.y - 300)])
        foe_b = next((u.id for u in p.own("hbar", 1)), None)
        if sap is not None and foe_b is not None:
            try_order(p, oix[(fv.order_ids()["selfdestruct"], fx.UNIT)], sap, foe_b, None, steps=20, label="sapper at a building")
        shr = next((u.id for u in p.own("ngir")), None)
        if shr is not None:
            trees = sorted(p.bot.trees.items(), key=lambda kv: (kv[1][0] - lab.x) ** 2 + (kv[1][1] - lab.y) ** 2)
            if trees:
                try_order(p, oix[(852018, fx.TREE)], shr, None, trees[0][1], steps=40, label="shredder harvests")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("ids", "orders", "variants", "corpses", "items", "neutral"))
    ap.add_argument("--race", default="human")
    ap.add_argument("--map", default="echofast2")
    ap.add_argument("--lab-map", default="lastrefugefast2")
    ap.add_argument("--vocab", default=VOCAB)
    ap.add_argument("--only", default="", help="orders: only these order names, comma separated")
    ap.add_argument("--name", default="")
    ap.add_argument("--shim", default=str(ROOT / "build" / "shim"))
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    if a.what != "ids":
        from warcraftsim.data.mapbuild import stock_map_path
        from warcraftsim.fullgame import features as fx
        fx.set_inputs(2)
        fx.set_masks(True, True)
        fx.set_neutral(2)
        fx.set_macros(3)
        fx.set_spells(1)
        fx.set_shop_items(1)
        fx.set_full_vocab(1)
        fx.set_econ_masks(255)
        fx.set_geometry(*fx.map_geometry(stock_map_path(a.lab_map if a.what == "neutral" else a.map)))
    return {"ids": ids, "orders": orders, "variants": variants, "corpses": corpses, "items": items, "neutral": neutral}[a.what](a)


if __name__ == "__main__":
    raise SystemExit(main())
