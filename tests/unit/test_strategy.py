import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import strategy
from warcraftsim.fullgame.model import FullGameNet


def _pool():
    s = [{"race": "human", "opp": "orc", "src": "human", "won": True, "t": {"hbar": 60.0, "halt": 90.0, "hkee": 300.0}},
         {"race": "human", "opp": "undead", "src": "ai", "won": False, "t": {"hbar": 80.0, "hpea": 5.0, "hhou": 30.0,
                                                                            "hlum": 200.0, "hkee": 2000.0}},
         {"race": "orc", "opp": "human", "src": "ai", "won": True, "t": {"obar": 50.0}}]
    return strategy.build_pool([{**x, "t": strategy._first_times(x["t"].items())} for x in s], min_items=1)


def test_pool_vector_and_mismatch():
    """The pool: each race's items by how many strategies make them (workers out), times past the horizon
    cut; the policy's input per slot (made, time / horizon); the reward's count of items made by now
    by one of the player and the strategy, not both."""
    pool = _pool()
    items = pool["items"]["human"]
    assert items[0] == "hbar" and set(items) == {"hbar", "halt", "hkee", "hhou", "hlum"}  # (workers out)
    won = next(s for s in pool["strategies"] if s["won"] and s["race"] == "human")
    lost = next(s for s in pool["strategies"] if not s["won"] and s["race"] == "human")
    assert "hkee" not in lost["t"]  # (made after the horizon)
    z = strategy.vector(won, items)
    k = items.index("hkee")
    assert z.shape == (strategy.ZDIM,) and z[k] == 1.0 and z[strategy.SLOTS + k] == pytest.approx(300 / strategy.HORIZON)
    assert z[items.index("hlum")] == 0.0 and z[strategy.SLOTS + items.index("hlum")] == 1.0
    assert z[strategy.SLOTS + len(items)] == 0.0 and not strategy.vector(None, items).any()
    assert strategy.mismatch({}, won, items, 100.0) == 2  # (barracks and altar not made yet)
    assert strategy.mismatch({"hbar": 50.0, "halt": 95.0, "hlum": 99.0}, won, items, 100.0) == 1  # (a lumber mill extra)
    assert strategy.mismatch({"hbar": 50.0, "halt": 95.0}, won, items, 92.0) == 1  # (the altar late)
    assert strategy.mismatch({"hbar": 50.0, "halt": 80.0, "hkee": 290.0}, won, items, 5000.0) == 0


def test_enforced_target_allows_only_its_items():
    """--z-enforce: train / build / research orders of the race's listed items only when the target makes
    them (workers and supply always); other orders and unlisted items stay; the times scale."""
    code = lambda s: int.from_bytes(s.encode(), "big")  # noqa: E731
    pool = _pool()
    won = next(s for s in pool["strategies"] if s["won"] and s["race"] == "human")  # barracks, altar, keep
    orders = [[851983, fx.UNIT], [code("hbar"), fx.POINT], [code("hlum"), fx.POINT], [code("hpea"), fx.IMMEDIATE],
              [code("hhou"), fx.POINT], [code("hkee"), fx.IMMEDIATE], [code("hgry"), fx.IMMEDIATE]]
    ok = strategy.allowed(won, pool["items"]["human"], orders)
    assert ok.tolist() == [True, True, True, False, True, True, True, True]  # (no lumber mill; gryphons unlisted)
    assert strategy.scaled(won, 0.5)["t"]["hkee"] == 150.0 and strategy.scaled(won, 1.0) is won


def test_sample_prefers_the_matchup_and_winners():
    pool = _pool()
    rng = random.Random(0)
    assert strategy.sample(pool, "human", "orc", rng)["won"]
    assert strategy.sample(pool, "human", "undead", rng)["won"]  # (a winner over the matchup)
    assert not strategy.sample(pool, "human", "undead", rng, winners=False)["won"]
    assert strategy.sample(pool, "undead", "orc", rng) is None
    assert strategy.sample(pool, "human", "undead", rng, src="ai")["src"] == "ai"  # (only the AI's: its loser)
    assert strategy.sample(pool, "human", "undead", rng, src="human")["opp"] == "orc"
    assert strategy.sample(pool, "orc", "human", rng, src="human") is None


def test_strategy_block_in_the_global_vector():
    """The strategy block sits between what the player knows and the privileged block; a network without
    one reads its own part of a wider row (config G less the privileged block) and the value's block from
    the end, and every global number is named."""
    code = lambda s: int.from_bytes(s.encode(), "big")  # noqa: E731
    vocab = {"types": [code("hpea")], "current_orders": [851986], "upgrades": [code("Rhma")],
             "orders": [[851983, fx.UNIT], [code("hpea"), fx.IMMEDIATE]]}
    old = fx.ZDIM
    try:
        fx.set_z(0)
        G0 = fx.Encoder(vocab).G
        fx.set_z(strategy.ZDIM)
        enc = fx.Encoder(vocab)
        assert enc.G == G0 + strategy.ZDIM and len(fx.describe_spaces(vocab)["observation"]["blocks"][0]["features"]) == enc.G
    finally:
        fx.set_z(old)
    torch.manual_seed(0)
    net = FullGameNet(n_types=5, n_cur=5, n_orders=7, G=G0, d=16, heads=2, layers=1, priv=fx.PRIV).eval()
    ent, typ, cur = torch.randn(2, 4, fx.F), torch.randint(0, 5, (2, 4)), torch.randint(0, 5, (2, 4))
    mask, glob = torch.ones(2, 4, dtype=torch.bool), torch.randn(2, G0)
    wide = torch.cat([glob[:, :G0 - fx.PRIV], torch.rand(2, strategy.ZDIM), glob[:, G0 - fx.PRIV:]], 1)
    with torch.no_grad():
        g0, _ = net.encode(ent, typ, cur, mask, glob)
        g1, _ = net.encode(ent, typ, cur, mask, wide)
        assert torch.allclose(g0, g1) and torch.allclose(net.value(g0, glob), net.value(g1, wide))


def test_grow_strategy_plays_as_before():
    """scripts/grow_strategy.py: the global layer's new columns zero (any strategy, the same outputs) and
    Adam's moments widened alike."""
    spec = importlib.util.spec_from_file_location("grow_strategy", Path(__file__).parents[2] / "scripts" / "grow_strategy.py")
    gs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gs)
    torch.manual_seed(0)
    G0 = 30 + fx.PRIV
    net = FullGameNet(n_types=5, n_cur=5, n_orders=7, G=G0, d=16, heads=2, layers=1, priv=fx.PRIV, memory=True).eval()
    opt = torch.optim.Adam(net.parameters())
    sum(p.sum() for p in net.parameters()).backward()
    opt.step()
    big = gs.grow(net, 8)
    assert big.config["G"] == G0 + 8 and big.config["z"] == 8
    ent, typ, cur = torch.randn(2, 4, fx.F), torch.randint(0, 5, (2, 4)), torch.randint(0, 5, (2, 4))
    mask, glob = torch.ones(2, 4, dtype=torch.bool), torch.randn(2, G0)
    wide = torch.cat([glob[:, :30], torch.rand(2, 8), glob[:, 30:]], 1)
    with torch.no_grad():
        assert torch.allclose(net.encode(ent, typ, cur, mask, glob)[0], big.encode(ent, typ, cur, mask, wide)[0], atol=1e-6)
    state = gs.grow_adam(opt.state_dict(), net, 8)
    opt2 = torch.optim.Adam(big.parameters())
    opt2.load_state_dict(state)
    m = opt2.state[big.glob[0].weight]["exp_avg"]
    assert m.shape == big.glob[0].weight.shape and not m[:, 30:].any()
    assert np.allclose(m[:, :30].numpy(), opt.state[net.glob[0].weight]["exp_avg"].numpy())


def test_hindsight_finds_each_sides_own_strategy():
    """A demonstration game's sides' own strategies from the pool: the built-in AI's by the game's file and
    side, people's by the replay and the players' slots (a converted replay's meta "source")."""
    pool = {"items": {}, "strategies": [
        {"race": "orc", "t": {"obar": 50.0}, "game": "demos-x/game00001.npz", "player": 1},
        {"race": "human", "t": {"hbar": 60.0}, "replay": "123.w3g", "slot": 4},
        {"race": "orc", "t": {"oalt": 70.0}, "replay": "123.w3g", "slot": 1}]}
    ai = strategy.hindsight(pool, Path("runs/fullgame/demos-x/game00001.npz"), {})
    assert list(ai) == [1] and ai[1]["t"] == {"obar": 50.0}
    people = strategy.hindsight(pool, Path("runs/fullgame/demos-h/game00000.npz"),
                                {"source": {"replay": "123.w3g", "slots": [1, 4]}})
    assert people[0]["race"] == "orc" and people[1]["race"] == "human"


TREE = """[htow]
Upgrade=hkee
Trains=hpea
[hkee]
Upgrade=hcas
Trains=hpea
[halt]
Trains=Hamg,Hpal
[hbar]
Trains=hfoo,hkni
Researches=Rhde
[hbla]
Requires=htow
[hkni]
Requires=hlum,hcas,hbla
[Hamg]
Requirescount=3
Requires=
Requires1=hkee
Requires2=hcas
[Rhde]
Requires=hbla
"""


def _builder(monkeypatch, st, orders):
    import random as _random
    from warcraftsim.fullgame import selfplay as sp
    from warcraftsim.fullgame import techtree
    monkeypatch.setattr(techtree, "tree", lambda: techtree.parse([TREE]))
    vocab = {"orders": [[_code(c), k] for c, k in orders]}
    cfg = {"costs_array": np.asarray([[0, 0, 0]] + [[100, 50, 0]] * len(orders))}
    return sp.TargetBuilder(cfg, vocab, 0, {"race": "human", **st}, _random.Random(0))


def _code(s):
    return int.from_bytes(s.encode(), "big")


def _obs(units, t=65.0, events=(), gold=1000):
    from types import SimpleNamespace as NS
    from warcraftsim.protocol import UnitFlags
    us = [NS(id=i, owner=0, alive=True, type_id=_code(tid), type=tid, x=x, y=y, order=0,
             flags=(UnitFlags.STRUCTURE if tid[0] == "h" and tid != "hpea" else 0) | (UnitFlags.WORKER if tid == "hpea" else 0),
             is_worker=tid == "hpea", is_structure=tid[0] == "h" and tid != "hpea") for i, tid, x, y in units]
    return NS(players={0: NS(gold=gold, lumber=500, food_used=10, food_cap=20)}, units=us, game_time=t,
              events=[NS(kind=k, a=a, b=_code(b) if isinstance(b, str) else b, c=0) for k, a, b in events])


def test_techtree_parses_makers_needs_and_upgrades():
    from warcraftsim.fullgame import techtree
    t = techtree.parse([TREE])
    assert t.makers["hfoo"] == ("hbar",) and t.makers["hkee"] == ("htow",) and t.upgrades["htow"] == ("hkee",)
    assert t.needs["Hamg"] == ((), ("hkee",), ("hcas",))
    have = techtree.counts_as({"hcas", "hbar"}, "human")
    assert {"htow", "hkee"} <= have
    assert techtree.missing("hkni", "human", have, t=t) == ["hlum", "hbla"]
    assert techtree.missing("Hamg", "human", techtree.counts_as({"htow"}, "human"), level=1, t=t) == ["hkee"]
    assert techtree.resolve("TALT", "orc") == "oalt" and techtree.resolve("TWN2", "undead") == "unp1"
    # what a target with knights makes on the way (the --z-enforce mask lets those through too)
    assert techtree.closure({"hkni"}, "human", t) == {"hkni", "hlum", "hcas", "hbla", "hbar", "hkee", "htow"}
    assert techtree.closure({"Hamg"}, "human", t) == {"Hamg", "halt"}
    assert techtree.closure({"Hamg", "Hpal"}, "human", t) == {"Hamg", "Hpal", "halt", "hkee", "htow"}


def test_target_builder_starts_due_items_at_their_makers(monkeypatch):
    """--z-build: a due, affordable target building goes to the worker nearest the town hall, a hero to the
    altar, a unit to its building; items made, not yet due or tried in the last 20 steps are left alone."""
    P, I = fx.POINT, fx.IMMEDIATE
    b = _builder(monkeypatch, {"t": {"hbar": 50.0, "Hamg": 60.0, "hfoo": 62.0, "hlum": 500.0}},
                 [("hbar", P), ("Hamg", I), ("hfoo", I), ("hlum", P)])
    units = [(1, "htow", 0, 0), (2, "halt", 300, 0), (3, "hpea", 900, 0), (4, "hpea", 100, 0)]
    c = b.commands(_obs(units), {}, 10, 0.5)
    assert len(c) == 1 and c[0].unit == 4 and c[0].building == _code("hbar")
    c = b.commands(_obs(units), {}, 11, 0.5)  # (the barracks tried: the hero next)
    assert len(c) == 1 and c[0].unit == 2 and c[0].order == _code("Hamg")
    assert b.commands(_obs(units), {}, 12, 0.5) == []  # (no barracks for the footman; the lumber mill not due)
    c = b.commands(_obs(units + [(5, "hbar", 500, 0)]), {"hbar": 1.0, "Hamg": 2.0}, 40, 0.5)
    assert len(c) == 1 and c[0].unit == 5 and c[0].order == _code("hfoo")
    assert b.commands(_obs(units, gold=50), {}, 60, 0.5) == []  # (nothing it can pay for)


def test_target_builder_starts_what_an_item_needs_first(monkeypatch):
    """A due knight without a castle, blacksmith or lumber mill: the builder starts its needs (a lumber mill
    first, the keep the castle comes from), not what is under way, and the knight once it can."""
    P, I = fx.POINT, fx.IMMEDIATE
    b = _builder(monkeypatch, {"t": {"hkni": 60.0}},
                 [("hlum", P), ("hbla", P), ("hkee", I), ("hcas", I), ("hkni", I)])
    units = [(1, "htow", 0, 0), (2, "hbar", 300, 0), (3, "hpea", 100, 0)]
    c = b.commands(_obs(units), {}, 10, 0.5)
    assert c[0].building == _code("hlum")
    c = b.commands(_obs(units), {}, 11, 0.5)  # (the lumber mill tried: the keep the castle comes from)
    assert c[0].unit == 1 and c[0].order == _code("hkee")
    c = b.commands(_obs(units), {}, 12, 0.5)
    assert c[0].building == _code("hbla")
    # the keep under way (an upgrade's start names the type upgraded from): the town hall busy, nothing more
    assert b.commands(_obs(units, events=[(11, 1, "htow")]), {}, 13, 0.5) == []
    assert "hkee" in b.started and 1 in b.busy
    b.commands(_obs(units, events=[(12, 1, "hkee")]), {}, 14, 0.5)  # (finished)
    assert "hkee" not in b.started and 1 not in b.busy
    done = [(1, "hcas", 0, 0), (2, "hbar", 300, 0), (3, "hpea", 100, 0), (4, "hlum", 0, 600), (5, "hbla", 600, 0)]
    c = b.commands(_obs(done), {"hkee": 1.0, "hcas": 2.0, "hlum": 3.0, "hbla": 4.0}, 50, 0.5)
    assert c[0].unit == 2 and c[0].order == _code("hkni")


def test_rescale_people_horizon_and_orderable():
    """People's times onto the AI's rules: per race the ratio of the medians, no item earlier than the AI's
    10% quantile, the game's end scaled too; a strategy covers only its game (horizon); items the
    vocabulary cannot order are dropped."""
    ai = [{"race": "human", "src": "ai", "t": {"hbar": 40.0 + i, "halt": 30.0 + i}} for i in range(20)]
    hu = [{"race": "human", "src": "human", "t": {"hbar": 100.0, "halt": 20.0, "hhou": 5.0}, "end": 600.0} for _ in range(20)]
    f = strategy.rescale_people(ai + hu, 0.1)
    assert 0.4 < f["human"] < 2.0
    assert hu[0]["t"]["hbar"] == pytest.approx(100.0 * f["human"])
    assert hu[0]["t"]["halt"] >= 30.0 + 1.9  # (clamped: the AI's 10% quantile)
    assert hu[0]["end"] == pytest.approx(600.0 * f["human"], abs=0.1)
    assert strategy.horizon(hu[0]) == hu[0]["end"] and strategy.horizon({"t": {}}) == strategy.HORIZON
    assert strategy.mismatch({"hbar": 1.0}, {"t": {}, "end": 0.5}, ["hbar"], 100.0) == 0  # (past its game)
    assert strategy.scaled({"t": {"hbar": 100.0}}, 0.5)["end"] == strategy.HORIZON * 0.5
    code = lambda s: int.from_bytes(s.encode(), "big")  # noqa: E731
    assert strategy.orderable({"orders": [[code("hbar"), fx.POINT], [851983, fx.POINT]]}) == {"hbar"}


def test_consistent_puts_items_after_their_needs(monkeypatch):
    from warcraftsim.fullgame import techtree
    t = techtree.parse([TREE])
    st = {"race": "human", "t": {"hbar": 90.0, "hfoo": 60.0, "Hpal": 50.0, "Hamg": 70.0, "hkee": 100.0, "htow": 0.0}}
    strategy.consistent(st, t)
    assert st["t"]["hfoo"] == 90.0  # (after its barracks)
    assert st["t"]["Hpal"] == 50.0 and st["t"]["Hamg"] == 100.0  # (the second hero: after the keep)


def test_owners_at_the_event_step_with_reused_ids():
    """The game reuses a dead unit's id: an event's owner is the one at its step, not the id's last."""
    units = np.zeros((5, 18), np.int64)
    units[:, :4] = [[0, 7, 0, 0], [5, 7, 0, 0], [20, 7, 0, 1], [30, 7, 0, 1], [3, 8, 0, 1]]
    events = np.asarray([[6, 3, 7, 0, 0], [25, 3, 7, 0, 0], [3, 3, 8, 0, 0], [3, 3, 9, 0, 0]])
    assert strategy.owners_at(units, events) == [0, 1, 1, -1]
    assert strategy.of_race("hbar", "human") and strategy.of_race("Rhme", "human") and not strategy.of_race("hhou", "undead")
    assert strategy.of_race("Udea", "undead") and not strategy.of_race("Rome", "nightelf")


def test_research_is_not_a_hero(monkeypatch):
    """Research ids are upper case too: a research under way must not count as a hero (the second hero then
    asked for the third tier)."""
    from warcraftsim.fullgame import techtree
    assert techtree.is_hero("Hamg") and techtree.is_hero("Ekee") and not techtree.is_hero("Rhde")
    P, I = fx.POINT, fx.IMMEDIATE
    b = _builder(monkeypatch, {"t": {"Hamg": 10.0}}, [("Hamg", I), ("hkee", I)])
    b.started["Rhde"] = 0.0
    c = b.commands(_obs([(1, "htow", 0, 0), (2, "halt", 300, 0)], t=20.0), {"Hpal": 5.0}, 10, 0.5)
    assert c[0].unit == 1 and c[0].order == _code("hkee")  # (the second hero needs the keep, not the castle)


def test_target_builder_queues_skips_expansions_and_gives_up(monkeypatch):
    """A busy town hall (training workers) still takes its next tier, queued; a second town hall (an
    expansion) is the agent's; an item the game refused three times is left for the game."""
    P, I = fx.POINT, fx.IMMEDIATE
    b = _builder(monkeypatch, {"t": {"htow": 5.0, "hkee": 10.0, "hbar": 20.0}},
                 [("htow", P), ("hkee", I), ("hbar", P)])
    units = [(1, "htow", 0, 0), (3, "hpea", 100, 0)]
    b.busy[1] = ("hpea",)
    c = b.commands(_obs(units), {}, 10, 0.5)
    assert c[0].unit == 1 and c[0].order == _code("hkee")
    b.refused["hbar"] = 3
    assert b.commands(_obs(units), {}, 11, 0.5) == []  # (no expansion; the barracks given up)
