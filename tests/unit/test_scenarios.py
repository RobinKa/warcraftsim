"""Scenarios (scenarios.py): the first step's commands set up a situation, the same value for both sides."""

import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import late, macros, scenarios, strategy as strat
from warcraftsim.protocol import ClearCreeps, Place, SetMineGold, SetResources, Spawn, SpawnHero, TargetOrder, UnitFlags, fourcc


def _obs():
    p = lambda x: SimpleNamespace(start_x=x, start_y=0, gold=500, lumber=150)  # noqa: E731
    return SimpleNamespace(players={0: p(-5000), 1: p(5000)})


def test_scenarios_commands():
    try:
        from warcraftsim.data.objects import unit_table
        unit_table()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no game data: {e}")
    rng = random.Random(0)
    rich = scenarios.commands("rich", _obs(), 0, ["human", "orc"], rng)
    assert rich == [SetResources(0, 2000, 850), SetResources(1, 2000, 850)]
    clash = scenarios.commands("clash", _obs(), 0, ["human", "orc"], rng)
    heroes = [c for c in clash if isinstance(c, SpawnHero)]
    assert [h.player for h in heroes] == [0, 1] and all(h.level == 3 for h in heroes)
    assert heroes[0].x < 0 < heroes[1].x  # (each a third of the way from its own base)
    units = [c for c in clash if isinstance(c, Spawn)]
    assert {u.player for u in units} == {0, 1} and all(u.unit_type in scenarios.UNITS["human"] for u in units if u.player == 0)
    raid = scenarios.commands("raid", _obs(), 1, ["human", "orc"], rng)
    xs = {p: [u.x for u in raid if u.player == p] for p in (0, 1)}
    assert all(2500 < x < 4600 for x in xs[0]) and all(3500 < x < 4700 for x in xs[1])  # (both at the learner's base)


def test_weight_averaging():
    import importlib.util
    from pathlib import Path

    import torch
    spec = importlib.util.spec_from_file_location("average_weights", Path(__file__).parents[2] / "scripts" / "average_weights.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    base = {"w": torch.zeros(2), "mask": torch.tensor([True, False])}
    a, b = {"w": torch.tensor([2.0, 0.0]), "mask": base["mask"]}, {"w": torch.tensor([0.0, 4.0]), "mask": base["mask"]}
    assert m.merge([a, b])["w"].tolist() == [1.0, 2.0]  # (a soup)
    assert m.merge([a, b], base, 0.5)["w"].tolist() == [1.0, 2.0] and m.merge([a, b], base, 1.0)["w"].tolist() == [2.0, 4.0]
    assert m.merge([a, b])["mask"].tolist() == [True, False]


L, R, NAT_L, NAT_R = (-5184, 2944), (4672, 2944), (-4864, -3200), (4352, -3200)  # (Echo Isles')
MAIN_L, MAIN_R = (-5632, 3584), (5120, 3584)
STRUCT, CONSTRUCTING = int(UnitFlags.STRUCTURE), int(UnitFlags.CONSTRUCTING)


def _first(races) -> SimpleNamespace:
    """A game's first observation: each main's mine of its race, the plain naturals."""
    u = lambda t, xy, owner, gold: SimpleNamespace(type=t, x=xy[0], y=xy[1], owner=owner, resource=gold, alive=True)  # noqa: E731
    units = [u("ngol", NAT_L, 15, 6250), u("ngol", NAT_R, 15, 6250)]
    for s, xy in ((0, MAIN_L), (1, MAIN_R)):
        units.append(u(late.OWN_MINE.get(races[s], "ngol"), xy, s if races[s] in late.OWN_MINE else 15, 6750))
    players = {s: SimpleNamespace(start_x=xy[0], start_y=xy[1]) for s, xy in ((0, L), (1, R))}
    return SimpleNamespace(units=units, players=players)


def _rows(obs, more=()) -> np.ndarray:
    """The observation's mines as unit rows, and more units (type, owner, x, y, flags, resource)."""
    out = []
    for k, (t, owner, x, y, flags, res) in enumerate([*((u.type, u.owner, u.x, u.y, STRUCT, u.resource) for u in obs.units), *more]):
        r = np.zeros(fx.ROW_COLS, np.int64)
        r[[fx.C_ID, fx.C_TYPE, fx.C_OWNER, fx.C_X, fx.C_Y, fx.C_HP, fx.C_MAXHP, fx.C_FLAGS, fx.C_RESOURCE]] = (
            k + 1, fourcc(t), owner, x, y, 100, 100, flags, res)
        out.append(r)
    return np.stack(out)


@pytest.mark.parametrize("race,hall", [("human", "htow"), ("orc", "ogre"), ("nightelf", "etol")])
def test_base2_a_hall_and_workers_at_the_natural(race, hall):
    obs = _first([race, "human"])
    cmds, state = scenarios.base2(obs, [0], [race, "human"])
    assert cmds[0] == ClearCreeps(*NAT_L, scenarios.CAMP_RADIUS)
    placed = [c for c in cmds if isinstance(c, Place)]
    assert all(c.player == 0 for c in placed)  # (only the sides given: the other side's natural keeps its camp too)
    assert sum(isinstance(c, ClearCreeps) for c in cmds) == 1
    assert [c.unit_type for c in placed] == [hall] + [scenarios.WORKERS[race]] * scenarios.BASE2_WORKERS
    h = placed[0]
    if race == "nightelf":  # (the tree where its instant entangle takes: the box of the melee script's)
        assert abs(h.x - NAT_L[0]) <= scenarios.TREE_BOX and abs(h.y - NAT_L[1]) <= scenarios.TREE_BOX
        assert h.y > NAT_L[1]  # (still on the side people build it)
    else:
        assert (h.x, h.y) == (NAT_L[0] + 100, NAT_L[1] + 800)  # (macros.EXPAND_SPOTS)
    assert all(math.hypot(c.x - NAT_L[0], c.y - NAT_L[1]) < 800 for c in placed[1:])
    assert state["sides"][1]["units"] == []
    assert sum(u[5] == "g" for u in state["sides"][0]["units"]) == scenarios.BASE2_WORKERS


def test_base2_both_agents_each_at_its_own_natural():
    obs = _first(["orc", "human"])
    cmds, _ = scenarios.base2(obs, [0, 1], ["orc", "human"])
    assert [c for c in cmds if isinstance(c, ClearCreeps)] == [ClearCreeps(*NAT_L, scenarios.CAMP_RADIUS),
                                                                ClearCreeps(*NAT_R, scenarios.CAMP_RADIUS)]
    halls = {c.player: c for c in cmds if isinstance(c, Place) and c.unit_type in late.HALLS}
    assert (halls[0].unit_type, halls[0].x, halls[0].y) == ("ogre", NAT_L[0] + 100, NAT_L[1] + 800)
    assert (halls[1].unit_type, halls[1].x, halls[1].y) == ("htow", NAT_R[0] - 300, NAT_R[1] + 700)


def test_base2_no_natural():
    obs = _first(["human", "orc"])
    obs.units = [u for u in obs.units if u.type != "ngol" or (u.x, u.y) == MAIN_L]
    assert scenarios.base2(obs, [0], ["human", "orc"]) is None


def test_base2_undead_haunts_the_mine_on_the_maps_scale():
    """No hall: the mine haunted at step 0, its gold the map's at once (the game's placing gives it 12,500), the other mines
    left alone (late.again takes a mine it does not know for one mined out)."""
    races = ["undead", "orc"]
    obs = _first(races)
    cmds, state = scenarios.base2(obs, [0], races)
    placed = [c for c in cmds if isinstance(c, Place)]
    assert [c.unit_type for c in placed] == ["ugol"] + ["uaco"] * 5 and (placed[0].x, placed[0].y) == NAT_L
    rows = _rows(obs, [("ugol", 0, *NAT_L, STRUCT, 12500)])
    assert late.again(state, rows) == [SetMineGold(int(rows[-1, fx.C_ID]), 6250)]  # (nothing for the other mines)
    workers = [("uaco", 0, c.x, c.y, 0, 0) for c in placed[1:]]
    given = late.jobs(state, _rows(obs, [("ugol", 0, *NAT_L, STRUCT, 6250), *workers]), 0, "undead", {})
    assert len(given) == 5 and all(isinstance(c, TargetOrder) and job == "gold" for c, _, job, _ in given)
    assert {tg for *_, tg in given} == {len(obs.units) + 1}  # (the natural's haunted mine, not the main's)


def test_base2_night_elf_entangled_at_step_1_complete_at_step_2():
    races = ["nightelf", "orc"]
    obs = _first(races)
    cmds, state = scenarios.base2(obs, [0], races)
    assert not any(isinstance(c, Place) and c.unit_type == "egol" for c in cmds)  # (its tree first)
    tree = next(c for c in cmds if isinstance(c, Place) and c.unit_type == "etol")
    step1 = late.again(state, _rows(obs, [("etol", 0, tree.x, tree.y, STRUCT, 0)]))
    assert step1 == [Place(0, "egol", *NAT_L)]
    rows = _rows(obs, [("etol", 0, tree.x, tree.y, STRUCT, 0), ("egol", 0, *NAT_L, STRUCT | CONSTRUCTING, 6250)])
    assert late.again(state, rows) == [SetMineGold(int(rows[-1, fx.C_ID]), 6250)]
    wisps = [("ewsp", 0, c.x, c.y, 0, 0) for c in cmds if isinstance(c, Place) and c.unit_type == "ewsp"]
    given = late.jobs(state, _rows(obs, [("egol", 0, *NAT_L, STRUCT, 6250), *wisps]), 0, "nightelf", {})
    assert len(given) == 5 and {tg for *_, tg in given} == {len(obs.units) + 1}


def test_base2_a_targets_expansion_is_done_no_reward_either_way():
    """Made and due alike: the mismatch stays what it was at every time (no gift, no debt), the builder finds it made."""
    target = {"t": {"htow": 300.0, "hbar": 60.0}, "race": "human"}
    made = scenarios.base2_made(target)
    assert made == {"htow": 300.0}
    items = ["htow", "hbar", "hkee"]
    for now in (0.0, 59.0, 299.0, 301.0, 2000.0):
        assert strat.mismatch(dict(made), target, items, now) == strat.mismatch({}, {"t": {"hbar": 60.0}}, items, now)
    assert scenarios.base2_made({"t": {"hbar": 60.0}}) == {}
    assert scenarios.base2_made({"t": {"ugol": 200.0, "unpl": 400.0}}) == {"ugol": 200.0, "unpl": 400.0}  # (undead names either)


def test_hall_spot():
    assert macros.hall_spot(*NAT_L, *L) == (NAT_L[0] + 100, NAT_L[1] + 800)
    x, y = macros.hall_spot(0.0, 0.0, 3000.0, 4000.0)  # (another map's: toward its start)
    assert (round(x), round(y)) == (450, 600)


@pytest.mark.parametrize("races", [("human", "orc"), ("undead", "nightelf")])
def test_dry_mains_low_and_naturals_cleared(races):
    """dry: each start's main mine (its own race's kind) gets one share of its gold for both sides, both naturals' camps
    are cleared, and nothing else is touched."""
    import random as _random
    from warcraftsim.protocol import ClearCreeps, SetMineGold
    obs = _first(races)
    for k, u in enumerate(obs.units, start=1):
        u.id = 100 + k
    got = scenarios.commands("dry", obs, 0, list(races), _random.Random(3))
    gold = {c.unit: c.gold for c in got if isinstance(c, SetMineGold)}
    mains = {u.id: u.resource for u in obs.units if (u.x, u.y) in (MAIN_L, MAIN_R)}
    assert set(gold) == set(mains)
    shares = {round(gold[i] / mains[i], 3) for i in gold}
    assert len(shares) == 1 and scenarios.DRY_SHARE[0] <= shares.pop() <= scenarios.DRY_SHARE[1]
    cleared = sorted((c.x, c.y) for c in got if isinstance(c, ClearCreeps))
    assert cleared == sorted([NAT_L, NAT_R])
    assert len(got) == 4
