"""--z-shop (strategy.py, selfplay.py): race-shop items bought as part of the strategy targets. The items chosen from
people's replays, the strategies' first purchases (people's; the built-in AI's have none), the strategy block's width
with and without the flag, an old network grown to it plays as before, the first purchase as the item made, the reward
and the builder's buy, the flag refused without --shop-items, and a run without the flag as before."""

import json
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame import modes, strategy  # noqa: E402


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


def _player(slot: int, race: str, items=(), won=False) -> dict:
    build = [[30000, "build", {"orc": "obar", "undead": "usep", "human": "hbar", "nightelf": "eaom"}[race], 0, 0],
             [60000, "train", {"orc": "ogru", "undead": "ugho", "human": "hfoo", "nightelf": "earc"}[race], 0, 0],
             [61000, "train", {"orc": "ogru", "undead": "ugho", "human": "hfoo", "nightelf": "earc"}[race], 0, 0]]
    build += [[t * 1000, "item", c, 0, 0] for c, t in items]
    build += [[t * 1000 + 50000, "item", c, 0, 0] for c, t in items]  # (bought again: the first counts)
    build.sort(key=lambda r: r[0])
    return {"slot": slot, "race_played": race, "won": won, "build": build}


def _stats(tmp_path):
    """Ten orc against undead 1v1s (half on Echo Isles): orcs buy salves (all), speed scrolls (8), clarity (6), the orb
    (3), town portals (9) and the merchant's potion (9, not a race shop's); undead rods (all), skulls (7), dust (5), the
    orb once; a 2v2 and a game with an unknown race are left out."""
    rows = []
    for g in range(10):
        orc = [("hslv", 100 + g), ("stwp", 300)] + ([("shas", 150)] if g < 8 else []) + ([("plcl", 400)] if g < 6 else [])
        orc += ([("oli2", 500 + g)] if g < 3 else []) + ([("pnvl", 200)] if g < 9 else [])
        orc = orc if g < 9 else [c for c in orc if c[0] != "stwp"]
        und = [("rnec", 80)] + ([("skul", 300)] if g < 7 else []) + ([("dust", 250)] if g < 5 else [])
        und += [("ocor", 600)] if g == 0 else []
        rows.append({"file": f"{g}.w3g", "type": "1v1", "patch": "1.29", "length_ms": 900000,
                     "map": "Maps\\FrozenThrone\\(2)EchoIsles.w3x" if g < 5 else "Maps/1.29\\(2)Amazonia.w3x",
                     "build_orders": [_player(0, "orc", orc, won=g % 2 == 0), _player(1, "undead", und, won=g % 2 == 1)]})
    rows.append({"file": "x.w3g", "type": "2v2", "build_orders": [_player(0, "orc", [("plcl", 10)])] * 4})
    rows.append({"file": "y.w3g", "type": "1v1", "build_orders": [_player(0, "orc", [("plcl", 10)]),
                                                                  {**_player(1, "human"), "race_played": None}]})
    path = tmp_path / "stats.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_items_chosen_from_replays(tmp_path, capsys):
    """Per race the K race-shop items most people buy (a town portal and other shops' wares left out), the race's orb
    in the last place when enough of them buy one; the shares and median first purchases per map."""
    stats = _stats(tmp_path)
    table = strategy.shop_table([stats])
    orc, und = table["all"]["orc"], table["all"]["undead"]
    assert orc["games"] == 10 and und["games"] == 10 and table["echo"]["orc"]["games"] == 5
    assert orc["items"]["hslv"] == (1.0, 104.5) and orc["items"]["oli2"] == pytest.approx((0.3, 501.0))
    assert "pnvl" not in orc["items"] and orc["items"]["stwp"][0] == 0.9
    assert table["echo"]["orc"]["items"]["plcl"][0] == 1.0  # (the first five games: all bought clarity)
    shop = strategy.shop_choice(table, 3)
    assert shop["orc"] == ["hslv", "shas", "oli2"]  # (the orb in clarity's place: 30% buy it)
    assert shop["undead"] == ["rnec", "skul", "dust"]  # (the orb: 10%, too few)
    assert shop["human"] == [] and strategy.shop_choice(table, 2)["orc"] == ["hslv", "oli2"]
    strategy.print_shop(table, shop)
    assert "| hslv (chosen) | 100%, 104 s | 100%, 102 s |" in capsys.readouterr().out
    # people's strategies with their first purchases (the click: a sale is at once), "shop" set; without: as before
    plain = strategy.from_replays(stats, "fixed")
    told = strategy.from_replays(stats, "fixed", shop=shop)
    assert len(plain) == len(told) == 20 and all("shop" not in s for s in plain)
    assert all(s["shop"] for s in told)
    for a, b in zip(plain, told):
        assert {k: v for k, v in b["t"].items() if k not in shop[b["race"]]} == a["t"]
    assert told[0]["t"]["hslv"] == 100.0 and told[0]["t"]["oli2"] == 500.0 and "plcl" not in told[0]["t"]
    assert told[1]["t"]["rnec"] == 80.0 and "ocor" not in told[1]["t"]
    # an existing pool's people's strategies with the items added, at its pace (its "end": orc's x0.5, undead's as
    # recorded); the built-in AI's untouched
    ai = {"race": "orc", "opp": "undead", "src": "ai", "won": True, "t": {"obar": 30.0}, "end": 600.0}
    pool = {"items": {}, "strategies": [{**s, "t": dict(s["t"]), "end": s["end"] * (0.5 if s["race"] == "orc" else 1)}
                                        for s in plain] + [ai]}
    assert strategy.add_shop(pool, [stats], shop) == {"orc": 0.5, "undead": 1.0}
    got = pool["strategies"]
    assert got[0]["t"]["hslv"] == 50.0 and got[0]["t"]["oli2"] == 250.0 and got[1]["t"]["rnec"] == 80.0
    assert all(s["shop"] for s in got[:-1]) and "shop" not in got[-1] and got[-1]["t"] == {"obar": 30.0}
    assert pool["shop_items"] == shop


def test_pool_keeps_the_items_bought(tmp_path):
    """build_pool: the items bought kept in people's strategies (none in the built-in AI's), neither among the race's
    item lists nor counted toward the least items; the pool without them as before."""
    shop = {"human": ["plcl", "sreg", "phea"], "orc": ["hslv", "shas", "oli2"], "undead": [], "nightelf": []}
    people = {"race": "orc", "opp": "human", "src": "human", "won": True, "shop": True,
              "t": {"obar": 40.0, "oalt": 50.0, "ogru": 60.0, "hslv": 120.0, "oli2": 2000.0}}
    ai = {"race": "orc", "opp": "human", "src": "ai", "won": True, "t": {"obar": 30.0, "oalt": 45.0, "ogru": 50.0}}
    pool = strategy.build_pool([people, ai], min_items=3, shop=shop)
    before = strategy.build_pool([{**people, "t": {c: v for c, v in people["t"].items() if c not in shop["orc"]}}, ai],
                                 min_items=3)
    assert pool["items"] == before["items"] and "hslv" not in pool["items"]["orc"]
    assert pool["shop_items"] == shop and "shop_items" not in before
    assert pool["strategies"][0]["t"] == {"obar": 40.0, "oalt": 50.0, "ogru": 60.0, "hslv": 120.0}  # (past the horizon: cut)
    assert pool["strategies"][1]["t"] == ai["t"]
    assert len(strategy.build_pool([people], min_items=4, shop=shop)["strategies"]) == 0  # (3 items and a salve)


def _shop_pool():
    shop = {"human": ["plcl", "sreg", "phea"], "orc": ["hslv", "shas", "oli2"], "undead": ["rnec", "ocor", "dust"],
            "nightelf": ["spre", "oven", "plcl"]}
    people = {"race": "orc", "opp": "human", "src": "human", "won": True, "shop": True,
              "t": {"obar": 40.0, "oalt": 50.0, "ogru": 60.0, "hslv": 90.0, "oli2": 450.0}}
    ai = {"race": "orc", "opp": "human", "src": "ai", "won": True, "t": {"obar": 30.0, "oalt": 45.0, "ogru": 50.0}}
    return strategy.build_pool([people, ai], min_items=3, shop=shop), shop


def test_items_bought_as_input_and_reward():
    """The items' input (per item bought and when; never: 0, 1; the built-in AI's: zeros, it does not tell) and the
    reward's items (the race's list and the items bought, for a strategy that tells them: one list per race)."""
    pool, shop = _shop_pool()
    people, ai = pool["strategies"]
    z = strategy.shop_vector(people, shop["orc"])
    assert z.tolist() == pytest.approx([1, 0, 1, 90 / strategy.HORIZON, 1, 450 / strategy.HORIZON])
    assert not strategy.shop_vector(ai, shop["orc"]).any() and not strategy.shop_vector(None, shop["orc"]).any()
    items = pool["items"]["orc"]
    assert strategy.scored(pool, "orc", ai) is items
    scored = strategy.scored(pool, "orc", people)
    assert scored == items + shop["orc"] and strategy.scored(pool, "orc", people) is scored
    made = {"obar": 35.0, "oalt": 50.0, "ogru": 55.0}
    assert strategy.mismatch(made, people, scored, 100.0) == 1  # (no salve yet)
    assert strategy.mismatch({**made, "hslv": 95.0}, people, scored, 100.0) == 0
    assert strategy.mismatch({**made, "hslv": 95.0, "shas": 99.0}, people, scored, 100.0) == 1  # (speed: not its)
    assert strategy.mismatch({**made, "hslv": 95.0, "shas": 99.0}, ai, strategy.scored(pool, "orc", ai), 100.0) == 0


def test_narrow_strategy_keeps_its_items_bought():
    pytest.importorskip("warcraftsim.data.objects")
    try:
        from warcraftsim.fullgame.techtree import tree
        tree()
    except Exception as e:  # noqa: BLE001 (the game's tables)
        pytest.skip(f"no game data: {e}")
    t = {"hbar": 40.0, "halt": 50.0, "Hamg": 80.0, "hfoo": 90.0, "hlum": 150.0, "plcl": 200.0, "sreg": 300.0}
    s = {"race": "human", "opp": "orc", "src": "human", "won": True, "t": t, "n": {"hfoo": 10}}
    assert set(strategy.narrow({**s, "shop": True})["t"]) == {"hbar", "halt", "Hamg", "hfoo", "plcl", "sreg"}
    assert set(strategy.narrow(s)["t"]) == {"hbar", "halt", "Hamg", "hfoo"}
    n = strategy.narrow({**s, "shop": True})
    items = ["hbar", "halt", "Hamg", "hfoo", "hlum", "plcl", "sreg", "phea"]
    made = {"hbar": 40.0, "halt": 50.0, "Hamg": 80.0, "hfoo": 90.0}
    assert strategy.mismatch(made, n, items, 400.0) == 2  # (no clarity potion, no regeneration scroll)
    assert strategy.mismatch({**made, "plcl": 210.0, "sreg": 290.0, "phea": 350.0}, n, items, 400.0) == 0  # (a potion: free)


def test_strategy_block_width_with_and_without_the_flag():
    """The block's width the flags ask for (selfplay's and bc's): 2 per item bought at the end; without --z-shop as
    before. selfplay.target: the items' input last, the rest as without the flag."""
    from warcraftsim.fullgame.selfplay import target
    Z, M, S, C = strategy.ZDIM, modes.MDIM, strategy.SLOTS, modes.CDIM
    assert strategy.block_width() == 0 and strategy.block_width(0.5) == Z + M and strategy.block_width(z_mix=1) == Z + S
    assert strategy.block_width(0.5, 1) == Z + M + S and strategy.block_width(0.0, 1, 0.3) == Z + M + S + C
    assert strategy.block_width(0.5, 1, 0.0, 3) == Z + M + S + 6 and strategy.block_width(z_shop=3) == Z + 6
    pool, shop = _shop_pool()
    people = pool["strategies"][0]
    vocab = {"types": [code("ogru")], "current_orders": [], "upgrades": [], "orders": [[code("ogru"), fx.IMMEDIATE]]}
    try:
        fx.set_z(Z + M + S)
        G = fx.Encoder(vocab).G
        before = NS(z=None)
        target({"z_mix": 1, "step_seconds": 0.5}, before, people, pool, "orc", vocab, {})
        assert len(before.z) == Z + M + S
        fx.set_z(strategy.block_width(0.5, 1, 0.0, 3))
        assert fx.Encoder(vocab).G == G + 6
        view = NS(z=None)
        target({"z_mix": 1, "z_shop": 3, "step_seconds": 0.5}, view, people, pool, "orc", vocab, {})
        assert len(view.z) == fx.ZDIM and np.array_equal(view.z[:-6], before.z)
        assert np.array_equal(view.z[-6:], strategy.shop_vector(people, shop["orc"]))
        fx.set_z(strategy.block_width(0.5, 1, 0.3, 3))  # (the composition kinds' block, then the items)
        view = NS(z=None)
        target({"z_mix": 1, "z_shop": 3, "step_seconds": 0.5}, view, people, pool, "orc", vocab, {"variety": 2.0})
        assert len(view.z) == fx.ZDIM and np.array_equal(view.z[-6 - C:-6], modes.comp_vector({"variety": 2.0}))
        fx.set_z(strategy.block_width(z_shop=3))  # (no modes nor mix: the build target, then the items)
        view = NS(z=None)
        target({"z_shop": 3, "step_seconds": 0.5}, view, people, pool, "orc", vocab, {})
        assert len(view.z) == Z + 6 and np.array_equal(view.z[:Z], strategy.vector(people, pool["items"]["orc"]))
    finally:
        fx.set_z(0)


def test_old_network_grows_and_plays_as_before(tmp_path):
    """A checkpoint of today's block (target, modes, mix) loaded with --z-shop's block: the items' columns zero, its
    player token and order logits as before whatever the items' inputs hold."""
    from warcraftsim.fullgame.model import FullGameNet, load
    z_old = strategy.ZDIM + modes.MDIM + strategy.SLOTS
    vocab = {"types": [code("Hpal"), code("ogru")], "current_orders": [], "upgrades": [code("Rome")],
             "orders": [[851983, fx.UNIT], [code("ogru"), fx.IMMEDIATE], [851986, fx.POINT]]}
    try:
        fx.set_z(z_old)
        enc = fx.Encoder(vocab)
        torch.manual_seed(0)
        old = FullGameNet(n_types=3, n_cur=2, n_orders=4, G=enc.G, d=32, layers=1, priv=fx.PRIV, z=z_old, gx=fx.GX).eval()
        old.allowed[:] = True
        torch.save({"config": old.config, "model": old.state_dict(), "vocab": vocab}, tmp_path / "old.pt")
        fx.set_z(strategy.block_width(0.5, 1, 0.0, 3))
        net, _ = load(tmp_path / "old.pt")
        net.eval()
        assert net.config["z"] == z_old + 6 and net.config["G"] == enc.G + 6
        at = enc.G0 + z_old
        assert not net.glob[0].weight[:, at:at + 6].any()
        glob_old = torch.randn(2, enc.G)
        glob = torch.cat([glob_old[:, :at], torch.rand(2, 6), glob_old[:, at:]], 1)
        ent, typ = torch.randn(2, 3, fx.F), torch.tensor([[1, 2, 2], [2, 1, 0]])
        cur, mask, n_own = torch.zeros(2, 3, dtype=torch.long), torch.tensor([[True] * 3, [True, True, False]]), torch.tensor([2, 1])
        with torch.no_grad():
            g0, u0 = old.encode(ent, typ, cur, mask, glob_old)
            g1, u1 = net.encode(ent, typ, cur, mask, glob)
            assert torch.allclose(g0, g1, atol=1e-6) and torch.allclose(u0, u1, atol=1e-6)
            # (by_type off: load prunes the orders a unit type can't give from the game's data, the same with or without)
            assert torch.allclose(old.order_logits(g0, u0, typ, n_own, by_type=False),
                                  net.order_logits(g1, u1, typ, n_own, by_type=False), atol=1e-5)
            assert torch.allclose(old.value(g0, glob_old), net.value(g1, glob), atol=1e-6)
    finally:
        fx.set_z(0)


def test_first_purchase_is_the_item_made():
    """play.BCAgent counts each item's first purchase at its game second (the episode's shop_bought's tracking): the
    target's item made then; a second one changes nothing."""
    try:
        from warcraftsim.fullgame.herodata import item_table
        item_table()
    except Exception as e:  # noqa: BLE001 (the game's tables)
        pytest.skip(f"no game data: {e}")
    from warcraftsim.fullgame.play import BCAgent
    from warcraftsim.protocol import EventKind
    vocab = {"types": [code("Ofar")], "current_orders": [], "upgrades": [], "orders": [[code("hslv"), fx.UNIT]]}
    bot = BCAgent(None, vocab, 0, None)
    row = np.zeros(fx.ROW_COLS, np.int64)
    row[fx.C_ID], row[fx.C_TYPE], row[fx.C_FLAGS], row[fx.C_HP], row[fx.C_MAXHP] = 7, code("Ofar"), fx.HERO, 500, 500
    rows = np.stack([row])
    sale = NS(kind=EventKind.ITEM_PICKUP, a=7, b=code("hslv"), c=0)
    bot.buying_at = {7: (code("hslv"), "ovln")}
    bot._count_items(rows, [], 50.0)
    assert "first_bought" not in bot.__dict__
    bot._count_items(rows, [sale], 61.5)
    bot.buying_at = {7: (code("hslv"), "ovln")}
    bot._count_items(rows, [sale], 130.0)
    assert bot.first_bought == {"hslv": 61.5} and bot.shop_bought == {"hslv": 2}
    pool, shop = _shop_pool()
    people = pool["strategies"][0]
    made = {"obar": 35.0, "oalt": 50.0, "ogru": 55.0}
    made.update({c: t for c, t in bot.first_bought.items() if c in shop["orc"]})  # (selfplay.play_one's)
    assert strategy.mismatch(made, people, strategy.scored(pool, "orc", people), 100.0) == 0


TREE = """[htow]
Upgrade=hkee
Trains=hpea
[halt]
Trains=Hamg
"""


def _builder(monkeypatch, st, cfg=None):
    from warcraftsim.fullgame import selfplay as sp
    from warcraftsim.fullgame import techtree
    monkeypatch.setattr(techtree, "tree", lambda: techtree.parse([TREE]))
    orders = [("hpea", fx.IMMEDIATE), ("hvlt", fx.POINT), ("plcl", fx.UNIT), ("phea", fx.UNIT), ("hkee", fx.IMMEDIATE)]
    vocab = {"orders": [[code(c), k] for c, k in orders]}
    cfg = {"costs_array": np.asarray([[0, 0, 0]] + [[70, 0, 0]] * len(orders)), "z_shop": 3, **(cfg or {})}
    return sp.TargetBuilder(cfg, vocab, 0, {"race": "human", **st}, random.Random(0))


def _obs(units, t=300.0, gold=1000):
    from warcraftsim.protocol import UnitFlags
    us = []
    for i, tid, x, y in units:
        hero, building = tid[0].isupper(), tid[0] == "h" and tid != "hpea"
        us.append(NS(id=i, owner=0, alive=True, type_id=code(tid), type=tid, x=x, y=y, order=0, is_hero=hero,
                     flags=(UnitFlags.STRUCTURE if building else 0) | (UnitFlags.WORKER if tid == "hpea" else 0),
                     is_worker=tid == "hpea", is_structure=building))
    return NS(players={0: NS(gold=gold, lumber=500, food_used=10, food_cap=20)}, units=us, game_time=t, events=[])


def test_target_builder_buys_a_due_item_near_the_shop(monkeypatch):
    """--z-shop with --z-build: a due item needs its race shop (built first, as any need) and what the shop asks (a
    potion of healing: a tier-2 hall); it is bought by the hero near the shop, never one far from it; once bought, not
    again. Its order is a Buy, as the buy labels' rows (selfplay.label_rows) take it."""
    from warcraftsim.fullgame.selfplay import label_rows
    from warcraftsim.protocol import Buy
    b = _builder(monkeypatch, {"t": {"plcl": 200.0}})
    base = [(1, "htow", 0, 0), (2, "hpea", 300, 0)]
    c = b.commands(_obs(base), {}, 10, 0.5)
    assert len(c) == 1 and c[0].building == code("hvlt")  # (no shop: the shop first)
    b = _builder(monkeypatch, {"t": {"plcl": 200.0}})
    far = base + [(3, "hvlt", 600, 0), (4, "Hamg", 4000, 0)]
    assert b.commands(_obs(far), {}, 10, 0.5) == [] and b.blocked["plcl"]["no hero near"] == 1
    near = base + [(3, "hvlt", 600, 0), (4, "Hamg", 900, 300)]
    c = b.commands(_obs(near), {}, 40, 0.5)
    assert len(c) == 1 and isinstance(c[0], Buy) and (c[0].unit, c[0].shop, c[0].item) == (4, 3, code("plcl"))
    assert label_rows(40, c).tolist() == [[40, 4, code("plcl"), 2, 0, 0, 3]]
    assert b.commands(_obs(near), {"plcl": 41.0}, 80, 0.5) == []  # (bought: not again)
    b = _builder(monkeypatch, {"t": {"phea": 250.0}})
    c = b.commands(_obs(near), {}, 80, 0.5)  # (a healing potion needs a keep: the hall's next tier first)
    assert len(c) == 1 and c[0].order == code("hkee")
    keep = [(1, "hkee", 0, 0)] + near[1:]
    assert b.commands(_obs(keep, gold=10), {}, 120, 0.5) == [] and b.blocked["phea"]["gold"] == 1
    c = b.commands(_obs(keep), {}, 160, 0.5)
    assert len(c) == 1 and (c[0].unit, c[0].shop, c[0].item) == (4, 3, code("phea"))
    # without --z-shop the builder knows no item (its pool holds none: selfplay.strategy_pool)
    assert _builder(monkeypatch, {"t": {}}, {"z_shop": 0}).shop_items == frozenset()


def test_run_without_the_flag_is_as_before(tmp_path, monkeypatch):
    """A pool with items bought in a run without --z-shop reads as the pool without them: the same strategies, inputs and
    rewards; with --z-shop it keeps them."""
    from warcraftsim.fullgame import selfplay as sp
    monkeypatch.setattr(sp, "trainable_units", lambda: frozenset({"ogru"}))
    pool, shop = _shop_pool()
    bare = {**pool, "strategies": [{k: v for k, v in s.items() if k != "shop"} | {"t": {c: t for c, t in s["t"].items()
                                    if c not in shop[s["race"]]}} for s in pool["strategies"]]}
    bare.pop("shop_items")
    (tmp_path / "with.json").write_text(json.dumps(pool))
    (tmp_path / "bare.json").write_text(json.dumps(bare))
    a = sp.strategy_pool({"z_pool": str(tmp_path / "with.json")})
    b = sp.strategy_pool({"z_pool": str(tmp_path / "bare.json")})
    assert "shop_items" not in a and [s["t"] for s in a["strategies"]] == [s["t"] for s in b["strategies"]]
    vocab = {"types": [], "current_orders": [], "upgrades": [], "orders": [[code("ogru"), fx.IMMEDIATE]]}
    made = {"obar": 35.0, "ogru": 55.0, "hslv": 60.0}
    try:
        fx.set_z(strategy.block_width(0.5, 1))
        for sa, sb in zip(a["strategies"], b["strategies"]):
            va, vb = NS(z=None), NS(z=None)
            sp.target({"z_mix": 1, "step_seconds": 0.5}, va, sa, a, "orc", vocab, {})
            sp.target({"z_mix": 1, "step_seconds": 0.5}, vb, sb, b, "orc", vocab, {})
            assert np.array_equal(va.z, vb.z)
            assert strategy.scored(a, "orc", sa) is a["items"]["orc"]
            assert strategy.mismatch(made, sa, strategy.scored(a, "orc", sa), 300.0) == \
                strategy.mismatch(made, sb, strategy.scored(b, "orc", sb), 300.0)
    finally:
        fx.set_z(0)
    (tmp_path / "with2.json").write_text(json.dumps(pool))
    kept = sp.strategy_pool({"z_pool": str(tmp_path / "with2.json"), "z_shop": 3})
    assert kept["shop_items"] == shop and kept["strategies"][0]["t"]["hslv"] == 90.0


def test_flag_refused_without_shop_items(tmp_path, capsys):
    """--z-shop needs --shop-items (the items as orders) and a pool with that many items bought per race."""
    from warcraftsim.fullgame import selfplay as sp
    pool, _ = _shop_pool()
    (tmp_path / "p.json").write_text(json.dumps(pool))
    (tmp_path / "old.json").write_text(json.dumps({k: v for k, v in pool.items() if k != "shop_items"}))
    base = ["--name", "zshop-args-test", "--init", str(tmp_path / "none.pt"), "--runs", str(tmp_path / "runs"),
            "--neutral", "1", "--z-shop", "3"]
    try:
        for extra, why in ((["--z-pool", str(tmp_path / "p.json")], "--z-shop needs --shop-items"),
                           (["--shop-items", "1", "--z-pool", str(tmp_path / "old.json")], "has no 3 items bought"),
                           (["--shop-items", "1", "--z-pool", str(tmp_path / "p.json"), "--z-shop", "2"], "has no 2 items")):
            with pytest.raises(SystemExit) as e:
                sp.main(base + extra)
            assert e.value.code == 2 and why in capsys.readouterr().err
    finally:
        fx.set_inputs(1)
        fx.set_masks(False, False)
        fx.set_neutral(0)
        fx.set_shop_items(0)
        fx.set_macros(0)
        fx.set_spells(0)
    assert not (tmp_path / "runs").exists()
