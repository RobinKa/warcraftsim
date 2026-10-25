"""The hand-written player (fullgame/heuristic.py): its decisions on made-up observations, no game."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from warcraftsim.fullgame import heuristic as hz
from warcraftsim.protocol import (Destructable, ImmediateOrder, Observation, PlayerState, PointOrder, Race, Result,
                                  TargetDestructable, TargetOrder, Unit, UnitFlags)

HOME, ENEMY = (-5000.0, 3000.0), (5000.0, 3000.0)
MINE = (-5400.0, 3600.0)


@pytest.fixture(autouse=True)
def _game_data():
    try:
        hz.unit_data()
        hz.research_costs()
    except Exception as e:  # noqa: BLE001 (the game's tables)
        pytest.skip(f"no game data: {e}")


def _unit(i, code, owner, x, y, hp=None, flags=0, order=0, mana=0, level=0, abilities=()):
    mhp = hz.unit_data().get(code, (400,))[0] or 400
    return Unit(id=i, type_id=hz.fourcc(code), owner=owner, x=int(x), y=int(y), facing=0,
                hp=mhp if hp is None else hp, max_hp=mhp, mana=mana, max_mana=max(mana, 1), order=order,
                flags=UnitFlags(flags), visible_to=0b11, resource=12000 if code == "ngol" else 0, hero_level=level,
                abilities=abilities)


def _player(pid, x, gold=500, lumber=150, food=5, cap=10, race=Race.ORC):
    return PlayerState(id=pid, race=race, is_agent=True, gold=gold, lumber=lumber, food_used=food, food_cap=cap, upkeep=0,
                       gold_gathered=0, lumber_gathered=0, structures=1, result=Result.PLAYING, start_x=int(x),
                       start_y=3000)


def _obs(units, t=1.0, gold=500, lumber=150, food=5, cap=10, events=(), first=False):
    trees = [Destructable(9000 + k, hz.fourcc("LTlt"), int(-5900 + 40 * k), 2400, 50) for k in range(30)]
    return Observation(seq=1, game_ms=int(t * 1000), game_over=False,
                       players={0: _player(0, HOME[0], gold, lumber, food, cap), 1: _player(1, ENEMY[0])},
                       units=list(units), events=list(events), command_results=[],
                       destructables=trees if first else None, orders={} if first else None)


def _agent(units, **kw):
    """An agent for side 0 begun on `units` (with its hall, the main mine and the enemy's hall), no terrain."""
    base = [_unit(1, "ogre", 0, *HOME, flags=UnitFlags.STRUCTURE), _unit(2, "ngol", 15, *MINE, flags=UnitFlags.STRUCTURE),
            _unit(3, "ogre", 1, *ENEMY, flags=UnitFlags.STRUCTURE)]
    a = hz.HeuristicAgent(0, "orc", map_name="no-such-map", log=False, **kw)
    a.begin(_obs(base + list(units), t=0.0, first=True))
    return a, base


def _step(a, base, units, t=60.0, **kw):
    obs = _obs(base + list(units), t=t, **kw)
    a.observe(obs, int(t * 2))
    return a.commands()


def test_camps_and_power():
    creeps = [_unit(10, "nftt", 12, 0, 0), _unit(11, "nftt", 12, 300, 0), _unit(12, "nomg", 12, 600, 100),
              _unit(13, "ngna", 12, 4000, 4000)]
    camps = hz.camps_of(creeps)
    assert sorted(len(c["ids"]) for c in camps) == [1, 3]
    grunts = [_unit(20 + k, "ogru", 0, 0, 0) for k in range(4)]
    assert hz.power(grunts) == pytest.approx(4 * hz.power(grunts[:2]))  # (Lanchester: twice the units, 4x)


def test_placer_finds_a_free_spot_outside_the_lane():
    terrain = SimpleNamespace(offset_x=0.0, offset_y=0.0, buildable=np.ones((64, 64), bool),
                              cell=lambda x, y: (int(y // 32), int(x // 32)))
    p = hz.Placer.__new__(hz.Placer)
    p.terrain, p.base, p.trees, p.taken = terrain, terrain.buildable.copy(), {1: (1024.0, 1024.0)}, []
    spot = p.find("obar", (1024.0, 1024.0), [], 0.0)  # (a tree at the centre: the barracks beside it)
    assert spot is not None and math.hypot(spot[0] - 1024, spot[1] - 1024) > 150
    again = p.find("obar", (1024.0, 1024.0), [], 1.0)  # (the first spot is taken now)
    assert again is not None and again != spot
    avoid = p.find("obar", (1024.0, 1024.0), [], 2.0, avoid=[(1024.0, 1024.0, 600.0)])
    assert avoid is not None and math.hypot(avoid[0] - 1024, avoid[1] - 1024) > 600


def test_idle_workers_go_to_the_mine_then_the_trees():
    miners = [_unit(30 + k, "opeo", 0, MINE[0] + 50, MINE[1] - 100, flags=UnitFlags.WORKER, order=hz.HARVEST)
              for k in range(5)]
    idle = _unit(40, "opeo", 0, HOME[0] + 100, HOME[1], flags=UnitFlags.WORKER)
    a, base = _agent(miners + [idle])
    cmds = _step(a, base, miners + [idle], t=20.0, gold=0, lumber=0)  # (nothing to build: the idle one works)
    sent = [c for c in cmds if getattr(c, "unit", None) == 40]
    assert len(sent) == 1 and isinstance(sent[0], TargetDestructable) and sent[0].order == hz.HARVEST  # (gold full)
    assert all(a.assign[m.id] == "gold" for m in miners)


def test_focus_fire_without_overkill():
    """Each grunt attacks the enemy in reach with the least hit points left after what the others aim at it: the
    nearly dead one draws two, the third goes to the other."""
    mine = [_unit(50 + k, "ogru", 0, 0 + 40 * k, 0) for k in range(3)]
    weak, strong = _unit(60, "ogru", 1, 120, 0, hp=20), _unit(61, "ogru", 1, 140, 60)
    a, base = _agent(mine)
    a.mode = "attack"
    cmds = _step(a, base, mine + [weak, strong], t=300.0)
    targets = sorted((c.unit, c.target) for c in cmds if isinstance(c, TargetOrder) and c.order == hz.ATTACK)
    assert [t for _, t in targets].count(60) == 2 and [t for _, t in targets].count(61) == 1


def test_retreat_when_outnumbered_away_from_home():
    mine = [_unit(70 + k, "ogru", 0, 0 + 30 * k, 0) for k in range(2)]
    theirs = [_unit(80 + k, "ogru", 1, 400 + 30 * k, 0) for k in range(8)]
    a, base = _agent(mine)
    a.mode = "attack"
    cmds = _step(a, base, mine + theirs, t=400.0)
    assert a.mode == "gather" and a.t < a.retreat_until
    moves = [c for c in cmds if isinstance(c, PointOrder) and c.order == hz.MOVE and c.unit in (70, 71)]
    assert len(moves) == 2 and all(math.hypot(c.x - a.rally[0], c.y - a.rally[1]) < 1 for c in moves)


def test_production_waits_for_the_unit_its_mix_lacks():
    """Two grunts and no headhunter (a 1:1 mix): the barracks waits for a headhunter's lumber rather than training a
    third grunt."""
    bar = _unit(90, "obar", 0, HOME[0] + 500, HOME[1], flags=UnitFlags.STRUCTURE)
    mill = _unit(95, "ofor", 0, HOME[0] + 700, HOME[1], flags=UnitFlags.STRUCTURE)  # (headhunters need a war mill)
    alt = _unit(96, "oalt", 0, HOME[0] + 900, HOME[1], flags=UnitFlags.STRUCTURE)
    grunts = [_unit(91 + k, "ogru", 0, HOME[0] + 600, HOME[1]) for k in range(2)]
    a, base = _agent([bar, mill, alt] + grunts)
    cmds = _step(a, base, [bar, mill, alt] + grunts, t=200.0, gold=1000, lumber=0, food=20, cap=60)
    assert not any(isinstance(c, ImmediateOrder) and c.unit == 90 for c in cmds)
    cmds = _step(a, base, [bar, mill, alt] + grunts, t=200.5, gold=1000, lumber=100, food=20, cap=60)
    assert [c.order for c in cmds if isinstance(c, ImmediateOrder) and c.unit == 90] == [hz.fourcc("ohun")]


def test_kept_lumber_does_not_block_what_costs_none():
    """A due building short of lumber keeps its money; a grunt (no lumber) is still trained with the gold left."""
    bar = _unit(90, "obar", 0, HOME[0] + 500, HOME[1], flags=UnitFlags.STRUCTURE)
    a, base = _agent([bar])
    cmds = _step(a, base, [bar], t=200.0, gold=1000, lumber=0, food=20, cap=60)
    assert [c.order for c in cmds if isinstance(c, ImmediateOrder) and c.unit == 90] == [hz.fourcc("ogru")]


def test_peons_out_of_burrows_it_did_not_order():
    """The game sends a computer slot's peons into its burrows when enemies come near; the agent stands them down
    (in burrows they gather nothing) unless it called battle stations itself."""
    burrow = _unit(100, "otrb", 0, HOME[0] + 300, HOME[1] - 300, flags=UnitFlags.STRUCTURE)
    boarding = _unit(101, "opeo", 0, HOME[0] + 200, HOME[1], flags=UnitFlags.WORKER, order=hz.BOARD)
    a, base = _agent([burrow, boarding])
    cmds = _step(a, base, [burrow, boarding], t=200.0, food=10, cap=20)
    assert ImmediateOrder(100, hz.STANDDOWN) in cmds
    a.in_burrows = True  # (its own call: kept, while it holds)
    a.burrow_since = 420
    cmds = _step(a, base, [burrow, boarding], t=210.0, food=10, cap=20)
    assert ImmediateOrder(100, hz.STANDDOWN) not in cmds


def test_costs_follow_the_maps_rules():
    real = hz.HeuristicAgent(0, "orc", map_name="(2)EchoIsles", log=False)
    fast = hz.HeuristicAgent(0, "orc", map_name="echofast", log=False)
    assert real._cost("ogru") == (200, 0) and fast._cost("ogru") == (100, 0)
    assert real._cost("ostr") == (315, 190)  # (an upgrade: the difference, as costs.order_costs)


def test_variants_switch_options():
    import importlib
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(hz.__file__).resolve().parents[2] / "scripts"))
    hm = importlib.import_module("heuristic_match")
    cls = hm.variant_class("heuristic:HeuristicAgent:sticky=0,pull=1")
    a = cls(0, "orc", map_name="no-such-map", log=False)
    assert a.opt["sticky"] == 0 and a.opt["pull"] == 1 and a.opt["hero_behind"] == 1
    assert hm.variant_class("") is None
