"""The heroes' spells the vocabulary lacked (selfplay --spells, fullgame/spells.py) and research costed at the level the
player would research next (costs.research_steps, features.Encoder.at_levels)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from warcraftsim.fullgame import features as fx  # noqa: E402


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


HEROES = ("Otch", "Oshd", "Hblm", "Ofar", "Emoo", "Hpal")
VOCAB = {"types": [code(h) for h in HEROES] + [code("hfoo"), code("hbla")], "current_orders": [], "upgrades": [],
         "orders": [[851983, fx.UNIT], [852096, fx.IMMEDIATE], [852095, fx.UNIT], [852092, fx.UNIT],
                    [code("AOws"), fx.SKILL], [code("hfoo"), fx.IMMEDIATE], [852119, fx.UNIT]],
         "order_names": {"851983": "attack", "852096": "thunderclap", "852095": "thunderbolt", "852092": "holybolt",
                         "852119": "chainlightning", "852127": "stomp"},
         "counts": {"orders": [9, 8, 7, 6, 5, 4, 3]}}


@pytest.fixture()
def game_tables():
    try:
        from warcraftsim.data.abilities import ability_info, hero_ability_table
        from warcraftsim.fullgame.herodata import hero_table
        ability_info(), hero_ability_table(), hero_table()
    except Exception as e:  # noqa: BLE001 (the game's tables: not on every machine)
        pytest.skip(f"no game data: {e}")


def test_spells_extend_at_the_end_in_a_fixed_order():
    from warcraftsim.fullgame import spells
    v = spells.extend(VOCAB)
    new = [(852127, fx.IMMEDIATE), (852502, fx.UNIT), (852486, fx.UNIT), (852122, fx.POINT), (852181, fx.IMMEDIATE)]
    assert v["orders"][:len(VOCAB["orders"])] == VOCAB["orders"] and [tuple(o) for o in v["orders"][-5:]] == new
    assert spells.extended(v) and not spells.extended(VOCAB) and v["spells"] == 1
    assert spells.extend(v) is v and len(v["counts"]["orders"]) == len(v["orders"])
    assert v["order_names"]["852122"] == "farsight" and v["order_names"]["852127"] == "stomp"
    # (one listed already keeps its place; the others after, in their order)
    w = spells.extend({**VOCAB, "orders": VOCAB["orders"] + [[852502, fx.UNIT]]})
    assert [tuple(o) for o in w["orders"][-5:]] == [(852502, fx.UNIT), (852127, fx.IMMEDIATE), (852486, fx.UNIT),
                                                    (852122, fx.POINT), (852181, fx.IMMEDIATE)]
    # (the same in another process with another hash seed: the vocab-order bug of fgself-113)
    prog = ("import json; from warcraftsim.fullgame.spells import extend; "
            f"print(json.dumps(extend(json.loads({json.dumps(json.dumps(VOCAB))}))['orders']))")
    for seed in ("1", "2"):
        out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, check=True,
                             env={**os.environ, "PYTHONHASHSEED": seed})
        assert json.loads(out.stdout) == v["orders"]


def test_spells_kinds_match_the_game_tables(game_tables):
    from warcraftsim.data.abilities import ability_info, hero_ability_table
    from warcraftsim.fullgame import spells
    info, table = ability_info(), hero_ability_table()
    kind = {"instant": fx.IMMEDIATE, "unit": fx.UNIT, "point": fx.POINT}
    for abil, name, oid, k in spells.SPELLS:
        assert info[abil].order == name and kind[info[abil].cast] == k and abil in table[spells.HEROES[abil]]


def test_spells_grow_keeps_the_old_orders(game_tables):
    from warcraftsim.fullgame import spells
    from warcraftsim.fullgame.herodata import order_needs
    from warcraftsim.fullgame.model import FullGameNet
    torch.manual_seed(0)
    old = VOCAB
    net = FullGameNet(n_types=len(old["types"]) + 1, n_orders=len(old["orders"]) + 1, n_cur=2, G=fx.Encoder(old).G,
                      d=16, layers=1).eval()
    with torch.no_grad():
        net.order[2].bias.normal_()
        net.order_emb.weight.normal_()
        net.allowed[:] = True
    v = spells.extend(old)
    grown = spells.grow(net, old, v).eval()
    O0, O1 = len(old["orders"]) + 1, len(v["orders"]) + 1
    assert grown.config["n_orders"] == O1 and grown.order_kind[O0:].tolist() == [0, 2, 2, 1, 0]
    assert torch.equal(grown.order[2].weight[:O0], net.order[2].weight)  # (a new spell: "no order"'s row, less 8)
    assert torch.equal(grown.order[2].weight[O0:], net.order[2].weight[:1].expand(O1 - O0, -1))
    assert torch.equal(grown.order[2].bias[O0:], (net.order[2].bias[0] - spells.NEW_BIAS).expand(O1 - O0))
    assert torch.equal(grown.allowed[:, :O0], net.allowed)
    tix = {t: i + 1 for i, t in enumerate(v["types"])}
    for j, (abil, _, _, _) in enumerate(spells.SPELLS):  # (only the hero with the ability)
        assert grown.allowed[:, O0 + j].nonzero().flatten().tolist() == [tix[code(spells.HEROES[abil])]]
    # (War Stomp aims as thunder clap, the old vocabulary's one no-target enemy spell; Hex and Banish as its enemy
    # unit spells: storm bolt, chain lightning, holy light's harm)
    assert torch.equal(grown.order_emb.weight[O0], net.order_emb.weight[2])
    assert torch.allclose(grown.order_emb.weight[O0 + 1], net.order_emb.weight[[3, 7]].mean(0))
    # the old orders' logits as before (to the matrix product's rounding: the head has more rows), and where a new
    # order is not allowed the same distribution
    typ = torch.tensor([[tix[code("hfoo")]] * 8, [tix[code("Hpal")]] * 8, [tix[code("Otch")]] * 8])
    g, u, n_own = torch.randn(3, 16), torch.randn(3, 8, 16), torch.tensor([8, 8, 8])
    with torch.no_grad():
        a, b = net.order_logits(g, u, typ, n_own), grown.order_logits(g, u, typ, n_own)
    assert torch.allclose(a, b[..., :O0], rtol=0, atol=1e-5)
    pa, pb = torch.softmax(a, -1), torch.softmax(b, -1)
    assert torch.allclose(pa[:2], pb[:2, :, :O0], rtol=0, atol=1e-6) and not pb[:2, :, O0:].any()
    # (the chieftain's War Stomp: 1/2981 as likely as no order, in any state, until trained)
    assert torch.allclose(pb[2, :, O0] / pb[2, :, 0], torch.tensor(np.exp(-spells.NEW_BIAS), dtype=torch.float32))
    assert not pb[2, :, O0 + 1:].any()
    # (--valid-mask: War Stomp only while it is learned, ready and paid for: the chieftain's fourth slot; Far Sight
    # the far seer's first)
    need = order_needs(v)
    assert need[(tix[code("Otch")], O0)] == 3 and need[(tix[code("Ofar")], O0 + 3)] == 0


def test_research_costs_at_the_next_level(game_tables):
    from warcraftsim.fullgame.costs import at_levels, order_costs, research_steps
    from warcraftsim.fullgame.play import BCAgent
    from warcraftsim.protocol import ImmediateOrder
    try:
        vocab = {"types": [code("hbla"), code("hfoo")], "current_orders": [], "upgrades": [],
                 "orders": [[code("Rhme"), fx.IMMEDIATE], [code("hfoo"), fx.IMMEDIATE], [code("Rhan"), fx.IMMEDIATE]]}
        costs, steps = order_costs(vocab, "EchoIsles"), research_steps(vocab, "EchoIsles")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no game data: {e}")
    rhme, rhan = code("Rhme"), code("Rhan")
    assert costs[1].tolist() == [100, 50, 0] and steps[1].tolist() == [75, 125] and not steps[2:].any()
    assert at_levels(costs, steps, [(1, rhme)], {}) is costs
    assert at_levels(costs, steps, [(1, rhme)], {rhme: 1})[1].tolist() == [175, 175, 0]  # (iron forged swords)
    assert at_levels(costs, steps, [(1, rhme)], {rhme: 2})[1].tolist() == [250, 300, 0]  # (steel)
    fast = research_steps(vocab, "echofast")  # (the map's tables halve them as they halve the first level's)
    assert order_costs(vocab, "echofast")[1].tolist()[:2] == [50, 25] and fast[1].tolist() == [round(37.5), round(62.5)]
    # the mask and the agent's budget: the next level's cost once the first is done
    bot = BCAgent(None, vocab, 0, None, costs=costs, steps=steps)
    bot.view = view = bot.enc.view(0, 1.0, [0, 1])
    bot.trees, bot.building_since, bot.t = {}, {}, 0
    forge = np.zeros(fx.ROW_COLS, np.int64)
    forge[fx.C_ID], forge[fx.C_TYPE], forge[fx.C_HP], forge[fx.C_MAXHP] = 7, code("hbla"), 500, 500
    forge[fx.C_FLAGS], forge[fx.C_VIS] = fx.STRUCTURE, 3
    rows, none = forge[None], np.zeros((0, 5), np.int64)
    me = np.asarray([0, 0, 150, 100, 5, 12, 0])
    assert view.step(rows, me, none, 0)["avail"][1]
    done = np.asarray([[1, fx.RESEARCH_FINISH, 7, rhme, 1]], np.int64)
    st = view.step(rows, me, done, 1)
    assert view.levels == {rhme: 1} and not st["avail"][1] and st["costs"][1].tolist() == [175, 175, 0]
    assert view.step(rows, np.asarray([0, 0, 200, 200, 5, 12, 0]), none, 2)["avail"][1]
    bot.me = np.asarray([0, 0, 180, 170, 5, 12, 0])  # (the step's money: short of the lumber at level 2)
    st = view.step(rows, bot.me, none, 3)
    assert bot.commands(st, [1], [0], [0], [0]) == []
    bot.me = np.asarray([0, 0, 180, 180, 5, 12, 0])
    st = view.step(rows, bot.me, none, 4)
    assert bot.commands(st, [1], [0], [0], [0]) == [ImmediateOrder(7, rhme)]
