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


def test_target_builder_starts_due_buildings_and_heroes():
    """--z-build: a due, affordable target building goes to the worker nearest the town hall, a due hero to
    the altar; items made, not yet due or tried in the last 20 steps are left alone."""
    import random as _random
    from types import SimpleNamespace as NS
    from warcraftsim.fullgame import selfplay as sp
    code = lambda s: int.from_bytes(s.encode(), "big")  # noqa: E731
    vocab = {"orders": [[code("hbar"), fx.POINT], [code("Hamg"), fx.IMMEDIATE], [code("hfoo"), fx.IMMEDIATE]]}
    cfg = {"costs_array": np.asarray([[0, 0, 0], [160, 60, 0], [425, 100, 5], [135, 0, 2]])}
    st = {"t": {"hbar": 50.0, "Hamg": 60.0, "hfoo": 70.0, "hlum": 500.0}}
    b = sp.TargetBuilder(cfg, vocab, 0, st, _random.Random(0))
    unit = lambda i, tid, x, y, worker=False, struct=False: NS(id=i, owner=0, alive=True, type_id=code(tid), x=x, y=y,  # noqa: E731
                                                              is_worker=worker, is_structure=struct)
    units = [unit(1, "htow", 0, 0, struct=True), unit(2, "halt", 300, 0, struct=True),
             unit(3, "hpea", 900, 0, worker=True), unit(4, "hpea", 100, 0, worker=True)]
    obs = NS(players={0: NS(gold=1000, lumber=500, food_used=10, food_cap=20)}, units=units, game_time=65.0)
    c = b.commands(obs, {}, 10, 0.5)
    assert len(c) == 1 and c[0].unit == 4 and c[0].building == code("hbar")
    c = b.commands(obs, {}, 11, 0.5)  # (the barracks tried: the hero next)
    assert len(c) == 1 and c[0].unit == 2 and c[0].order == code("Hamg")
    assert b.commands(obs, {}, 12, 0.5) == []  # (a footman is not a building or a hero; the lumber mill not due)
    assert b.commands(obs, {"hbar": 1.0, "Hamg": 2.0}, 40, 0.5) == []
