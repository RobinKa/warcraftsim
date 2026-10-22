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
    assert b.commands(_obs(units, gold=50), {"Hamg": 2.0}, 60, 0.5) == []  # (nothing it can pay for)
    c = b.commands(_obs(units, gold=50), {}, 80, 0.5)  # (but a first hero: free, its food only)
    assert len(c) == 1 and c[0].order == _code("Hamg")


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
    """A unit or research queues at a busy building; an upgrade (the game refuses it behind others) at a
    hall training workers cancels their queue first; a second town hall (an expansion) is the agent's; an
    item the game refused three times is left for the game."""
    from warcraftsim.fullgame.selfplay import CANCEL
    P, I = fx.POINT, fx.IMMEDIATE
    b = _builder(monkeypatch, {"t": {"htow": 5.0, "hkee": 10.0, "hfoo": 12.0, "hbar": 20.0}},
                 [("htow", P), ("hkee", I), ("hfoo", I), ("hbar", P)])
    units = [(1, "htow", 0, 0), (2, "hbar", 300, 0), (3, "hpea", 100, 0)]
    b.busy[1], b.busy[2] = ("hpea",), ("hfoo",)
    c = b.commands(_obs(units), {"hbar": 1.0}, 10, 0.5)
    assert [x.order for x in c[:-1]] == [CANCEL] * 6 and c[-1].unit == 1 and c[-1].order == _code("hkee")
    c = b.commands(_obs(units), {"hbar": 1.0}, 11, 0.5)
    assert c == [c[0]] and c[0].unit == 2 and c[0].order == _code("hfoo")  # (queued)
    b.teach = True  # (taught: the upgrade alone, the hall's decision)
    b.tried.pop("hkee")
    assert b.commands(_obs(units), {"hbar": 1.0}, 12, 0.5) == [type(c[0])(1, _code("hkee"))]
    b.refused["hbar"] = 3
    done = {"hkee": 1.0, "hfoo": 2.0}
    assert b.commands(_obs([(1, "htow", 0, 0), (3, "hpea", 100, 0)]), done, 40, 0.5) == []  # (no expansion; the barracks given up)


def test_upgrade_costs_the_difference():
    """The game's table holds an upgraded building's cost with its base's (a keep: 705 gold); the order
    costs the difference (320), which the availability mask and the builder must use."""
    from warcraftsim.fullgame.costs import order_costs
    vocab = {"orders": [[_code("htow"), fx.POINT], [_code("hkee"), fx.IMMEDIATE], [_code("hcas"), fx.IMMEDIATE],
                        [_code("hgtw"), fx.IMMEDIATE]]}
    c = order_costs(vocab, "(2)EchoIsles")
    assert c[1].tolist() == [385, 205, 0] and c[2].tolist() == [320, 210, 0] and c[3].tolist() == [360, 210, 0]
    assert c[4].tolist() == [70, 50, 0]


def test_target_builder_builds_supply_when_blocked(monkeypatch):
    """Out of supply a unit ordered waits in its building's queue and blocks a next tier: a farm first."""
    P, I = fx.POINT, fx.IMMEDIATE
    b = _builder(monkeypatch, {"t": {"hkee": 10.0}}, [("hhou", P), ("hkee", I)])
    obs = _obs([(1, "htow", 0, 0), (3, "hpea", 100, 0)])
    obs.players[0].food_used = 20
    c = b.commands(obs, {}, 10, 0.5)
    assert c[0].building == _code("hhou")
    obs.players[0].food_used = 10
    assert b.commands(obs, {}, 11, 0.5)[0].order == _code("hkee")


def test_target_builder_resumes_a_stalled_frame(monkeypatch):
    """A human building whose worker was taken away stays an unfinished frame: after 20 steps without hit
    points gained the nearest free worker resumes it (the repair order)."""
    from warcraftsim.fullgame.play import REPAIR
    from warcraftsim.protocol import UnitFlags
    b = _builder(monkeypatch, {"t": {}}, [("hbar", fx.POINT)])
    obs = _obs([(1, "htow", 0, 0), (2, "halt", 500, 0), (3, "hpea", 100, 0), (4, "hpea", 450, 0)])
    altar = obs.units[1]
    altar.flags |= UnitFlags.CONSTRUCTING
    altar.hp = 100
    assert b.commands(obs, {}, 0, 0.5) == []
    assert b.commands(obs, {}, 19, 0.5) == []
    c = b.commands(obs, {}, 20, 0.5)
    assert c[0].order == REPAIR and c[0].target == 2 and c[0].unit == 4
    altar.hp = 150  # (progress again: nothing)
    assert b.commands(obs, {}, 45, 0.5) == []


def test_target_builder_saves_for_a_tavern_hero(monkeypatch):
    """--neutral: a due tavern hero is bought by the worker nearest the tavern, and short of its cost the
    builder keeps that cost from the agent's other orders (its reserve: the hire's class, as for any item)."""
    b = _builder(monkeypatch, {"t": {"Npbm": 60.0}}, [("hfoo", fx.IMMEDIATE), ("Npbm", fx.UNIT)])
    units = [(1, "htow", 0, 0), (2, "halt", 300, 0), (3, "hpea", 900, 0), (4, "hpea", 100, 0), (9, "ntav", 1200, 0)]
    assert b.commands(_obs(units, gold=50), {"Hamg": 1.0}, 10, 0.5) == [] and b.reserve == (2, 100, 50)
    c = b.commands(_obs(units), {"Hamg": 1.0}, 40, 0.5)
    assert len(c) == 1 and (c[0].unit, c[0].shop, c[0].item) == (3, 9, _code("Npbm"))
    c = b.commands(_obs(units, gold=50), {}, 80, 0.5)  # (the first hero: free)
    assert len(c) == 1 and c[0].item == _code("Npbm") and b.reserve is None


def test_first_hero_is_free():
    """A player's first hero costs only its food (the game's hero token): at an altar or a tavern; not research."""
    vocab = {"orders": [[_code("hfoo"), fx.IMMEDIATE], [_code("Hamg"), fx.IMMEDIATE], [_code("Npbm"), fx.UNIT],
                        [_code("Rhde"), fx.IMMEDIATE]]}
    costs = np.asarray([[0, 0, 0], [135, 0, 2], [425, 100, 5], [425, 135, 5], [100, 50, 0]])
    free = fx.first_hero_free(costs, vocab)
    assert free.tolist() == [[0, 0, 0], [135, 0, 2], [0, 0, 5], [0, 0, 5], [100, 50, 0]]
    assert fx.affordable(free, 50, 20, 5, 12).tolist() == [True, False, True, True, False]


def test_affordable_keeps_the_reserve():
    """--z-build: the target's next item's cost is kept from the other orders; the item and free orders stay."""
    costs = np.asarray([[0, 0, 0], [0, 0, 0], [80, 30, 0], [160, 105, 0], [70, 0, 2]])  # none, move, barracks, keep, footman
    assert fx.affordable(costs, 200, 120, 10, 20).tolist() == [True, True, True, True, True]
    assert fx.affordable(costs, 200, 120, 10, 20, (3, 160, 105)).tolist() == [True, True, False, True, False]
    assert fx.affordable(costs, 100, 50, 10, 20, (3, 160, 105)).tolist() == [True, True, False, False, False]


def test_classifier_free_guidance_of_the_order():
    """--eval-guide (model.act): weight 1 is no guidance; weight 0 samples as the network without its target
    (its strategy block zeroed); the same noise for both."""
    from warcraftsim.fullgame.model import act
    torch.manual_seed(0)
    Z, P = 8, fx.PRIV
    G = 30 + Z + P
    net = FullGameNet(n_types=5, n_cur=5, n_orders=7, G=G, d=16, heads=2, layers=1, priv=P, z=Z).eval()
    for p_ in net.parameters():
        p_.data.normal_(0, 0.5)
    N, E = 6, 5
    ent, typ, cur = torch.randn(N, E, fx.F), torch.randint(1, 5, (N, E)), torch.randint(0, 5, (N, E))
    mask, glob, n_own = torch.ones(N, E, dtype=torch.bool), torch.randn(N, G), torch.full((N,), 3)
    net.allowed.fill_(True)
    blank = glob.clone()
    blank[:, G - P - Z:G - P] = 0.0

    def run(g, guide):
        gen = torch.Generator().manual_seed(7)
        with torch.no_grad():
            return act(net, ent, typ, cur, mask, g, n_own, generator=gen, guide=guide)["order"]
    assert torch.equal(run(glob, None), run(glob, torch.ones(N)))
    assert torch.equal(run(glob, torch.zeros(N)), run(blank, None))
    assert not torch.equal(run(glob, torch.full((N,), 8.0)), run(glob, None))


def test_narrow_strategy_keeps_heroes_main_army_and_their_needs():
    """A narrow strategy (the user's idea): the game's heroes, its main army (unit types with a fifth of its army's
    food) and what they need; its other buildings and research are free, other heroes and army units are not (the
    reward, the enforcement); drawn evenly per signature."""
    pytest.importorskip("warcraftsim.data.objects")
    try:
        from warcraftsim.fullgame.techtree import tree
        tree()
    except Exception as e:  # noqa: BLE001 (the game's tables)
        pytest.skip(f"no game data: {e}")
    t = {"hbar": 40.0, "halt": 50.0, "Hamg": 80.0, "hfoo": 90.0, "hlum": 150.0, "hbla": 160.0, "hrif": 200.0,
         "hars": 300.0, "hsor": 350.0, "Rhme": 400.0, "hkee": 420.0}
    s = {"race": "human", "opp": "orc", "src": "human", "won": True, "t": t,
         "n": {"hpea": 20, "hfoo": 10, "hrif": 8, "hsor": 1}}
    n = strategy.narrow(s)
    assert n["narrow"] and n["sig"] == "Hamg|hfoo+hrif"  # (the sorceress: a tenth of the food)
    assert set(n["t"]) == {"hbar", "halt", "Hamg", "hfoo", "hrif", "hbla"}  # (riflemen need the blacksmith)
    items = ["hbar", "halt", "Hamg", "hfoo", "hlum", "hbla", "hrif", "hars", "hsor", "Rhme", "hkee", "Hmkg", "hkni"]
    made = {"hbar": 30.0, "halt": 40.0, "Hamg": 70.0, "hfoo": 80.0, "hbla": 150.0, "hrif": 190.0}
    assert strategy.mismatch(made, n, items, 600.0) == 0
    assert strategy.mismatch({**made, "hlum": 99.0, "hars": 200.0, "Rhme": 300.0}, n, items, 600.0) == 0  # (free)
    assert strategy.mismatch({**made, "hsor": 300.0, "Hmkg": 400.0}, n, items, 600.0) == 2  # (not its army, hero)
    code = lambda c: int.from_bytes(c.encode(), "big")  # noqa: E731
    orders = [[code(c), fx.IMMEDIATE] for c in ("hfoo", "hsor", "Hmkg", "Rhme", "hkee")] + [[code("hars"), fx.POINT]]
    assert strategy.allowed(n, items, orders).tolist() == [True, True, False, False, True, True, True]
    assert strategy.narrow({**s, "n": {"hpea": 20}}) is None  # (no army)
    assert strategy.narrow({**s, "t": {c: v for c, v in t.items() if c != "Hamg"}}) is None  # (no hero)
    pool = {"strategies": [n] * 9 + [{**n, "sig": "rare", "t": {"hbar": 1.0}}]}
    rng = random.Random(1)
    assert 0.3 < sum(strategy.sample(pool, "human", "orc", rng)["sig"] == "rare" for _ in range(400)) / 400 < 0.7


def test_natural_scenario_clears_both_naturals():
    """--scenarios natural: the camp at each start's nearest gold mine past its main is removed (ClearCreeps)."""
    import random as _random
    from types import SimpleNamespace as NS

    from warcraftsim.fullgame import scenarios
    from warcraftsim.protocol import ClearCreeps
    players = {0: NS(start_x=-5184.0, start_y=2944.0), 1: NS(start_x=4672.0, start_y=2944.0)}
    units = [NS(type="ngol", x=-5632.0, y=3584.0), NS(type="ngol", x=-4864.0, y=-3200.0), NS(type="ngol", x=5120.0, y=3584.0),
             NS(type="ngol", x=4352.0, y=-3200.0)]
    out = scenarios.commands("natural", NS(players=players, units=units), 0, ["human", "orc"], _random.Random(0))
    assert out == [ClearCreeps(-4864.0, -3200.0, 900), ClearCreeps(4352.0, -3200.0, 900)]
    assert ClearCreeps(-4864.0, -3200.0, 900).encode()[0] == 88


def test_army_mix_input_and_gap():
    """--z-mix: a strategy's army as shares of its army units (no workers or heroes); the gap of what a side trained
    to it, their total variation (0 until 5 army units)."""
    from warcraftsim.fullgame import strategy as st
    s = {"t": {"hfoo": 60.0}, "n": {"hpea": 20, "hfoo": 6, "hrif": 2, "Hamg": 1}}
    z = st.mix(s, ["hfoo", "hrif", "hkni"])
    assert z[:3].tolist() == [0.75, 0.25, 0.0] and z.shape == (st.SLOTS,)
    assert st.mix(None, ["hfoo"]).sum() == 0.0
    assert st.mix_gap({"hfoo": 3}, s) == 0.0  # (too few yet)
    assert st.mix_gap({"hfoo": 6, "hrif": 2, "hpea": 9}, s) == pytest.approx(0.0)
    assert st.mix_gap({"hfoo": 8}, s) == pytest.approx(0.25)
    assert st.mix_gap({"hkni": 8}, s) == pytest.approx(1.0)
