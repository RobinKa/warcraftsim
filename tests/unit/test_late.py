"""Late-game starts (fullgame/late.py): a game's state captured, its sides swapped, the first step's commands that set
it up, the step-1 harvest orders; the new harness ops' lengths alike in protocol.py, the harness and the shim."""

import json
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import late
from warcraftsim.protocol import (_OP_LENGTHS, Op, Place, RemoveUnit, SetMineGold, SetResearch, SetResources,
                                  TargetDestructable, TargetOrder, UnitFlags, command_ops, encode_commands, fourcc)

ROOT = Path(__file__).resolve().parents[2]
HERO, STRUCT, WORKER, HIDDEN, DEAD, SUMMONED = (int(UnitFlags.HERO), int(UnitFlags.STRUCTURE), int(UnitFlags.WORKER),
                                                int(UnitFlags.HIDDEN), int(UnitFlags.DEAD), int(UnitFlags.SUMMONED))
L, R = (-5184, 2944), (4672, 2944)  # (Echo Isles' start locations)


def row(uid, t, owner, x, y, hp=100, maxhp=100, flags=0, order=0, resource=0, level=0, xp=0, abil=(), items=(),
        mana=0, maxmana=0):
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[[fx.C_ID, fx.C_TYPE, fx.C_OWNER, fx.C_X, fx.C_Y, fx.C_HP, fx.C_MAXHP, fx.C_MANA, fx.C_MAXMANA, fx.C_ORDER,
       fx.C_FLAGS, fx.C_RESOURCE, fx.C_HLEVEL, fx.C_HXP]] = (uid, fourcc(t), owner, x, y, hp, maxhp, mana, maxmana,
                                                              order, flags, resource, level, xp)
    for k, v in enumerate(items):
        r[fx.C_ITEM0 + k] = fourcc(v)
    for k, v in enumerate(abil):
        r[fx.C_ABIL0 + 2 * k] = v
    return r


def players(gold=(500, 400), lumber=(150, 100), starts=(L, R)):
    return {s: SimpleNamespace(gold=gold[s], lumber=lumber[s], start_x=starts[s][0], start_y=starts[s][1])
            for s in (0, 1)}


@pytest.fixture
def heroes(monkeypatch):
    import warcraftsim.data.abilities as ab
    monkeypatch.setattr(ab, "hero_abilities", lambda: {"Hamg": ("AHbz", "AHab", "AHwe", "AHmt"),
                                                       "Udea": ("AUdc", "AUdp", "AUau", "AUan")})


def midgame() -> np.ndarray:
    """Human (left) at tier 2 with an expansion against undead (right) with a haunted expansion."""
    return np.stack([
        row(1, "hkee", 0, *L, 1800, 2000, STRUCT),
        row(2, "hbar", 0, L[0] + 400, L[1] - 300, flags=STRUCT),
        row(3, "htow", 0, -4764, -2400, flags=STRUCT),  # (the expansion's hall)
        row(4, "hpea", 0, -5600, 3500, flags=WORKER | HIDDEN, order=fx.HARVEST),  # (in the mine)
        row(5, "hpea", 0, -5900, 2600, flags=WORKER, order=fx.HARVEST),  # (at a tree)
        row(6, "hmil", 0, L[0], L[1] - 200, flags=WORKER),  # (a militia: the peasant it was)
        row(7, "hfoo", 0, -3000, 0, 300, 420),
        row(8, "Hamg", 0, -3000, 50, 500, 600, HERO, level=4, xp=900, abil=(2, 1, 1, 0), items=("stwp",),
            mana=150, maxmana=300),
        row(9, "hwat", 0, -3000, 80, flags=SUMMONED),  # (summoned: not captured)
        row(10, "hgtw", 0, -4600, 2200, flags=STRUCT | int(UnitFlags.CONSTRUCTING)),  # (being built: not captured)
        row(20, "unpl", 1, *R, flags=STRUCT),
        row(21, "ugol", 1, 5120, 3584, flags=STRUCT, resource=3000),
        row(22, "ugol", 1, 4352, -3200, flags=STRUCT, resource=4550),  # (a haunted expansion: the mine hidden under it)
        row(23, "uaco", 1, 5000, 3300, flags=WORKER, order=fx.HARVEST),
        row(24, "ugho", 1, 5300, 2500, order=fx.HARVEST),
        row(25, "Udea", 1, 4000, 2000, 0, 700, HERO | DEAD, level=3, abil=(2, 1, 0, 0)),
        row(30, "ngol", 15, -5632, 3584, flags=STRUCT, resource=4000),
        row(31, "ngol", 15, 4352, -3200, flags=STRUCT | HIDDEN, resource=6250),
        row(40, "nmrr", late.NEUTRAL_HOSTILE, 0, -3000),
    ])


def fresh() -> np.ndarray:
    """A new game's first step: the starts' units, the map's mines and creeps."""
    return np.stack([
        row(101, "htow", 0, *L, 1500, 1500, STRUCT),
        *[row(102 + k, "hpea", 0, -5400 + 50 * k, 3300, flags=WORKER) for k in range(5)],
        row(120, "unpl", 1, *R, flags=STRUCT),
        row(121, "ugol", 1, 5120, 3584, flags=STRUCT, resource=6250),
        *[row(122 + k, "uaco", 1, 5000 + 50 * k, 3300, flags=WORKER) for k in range(3)],
        row(125, "ugho", 1, 4900, 2900),
        row(130, "ngol", 15, -5632, 3584, flags=STRUCT, resource=6250),
        row(131, "ngol", 15, 4352, -3200, flags=STRUCT, resource=6250),
        row(132, "ngol", 15, -4864, -3200, flags=STRUCT, resource=6250),  # (mined out in the captured game)
        row(140, "nmrr", late.NEUTRAL_HOSTILE, 50, -3050),  # (still standing)
        row(141, "nmrm", late.NEUTRAL_HOSTILE, 200, -3000),  # (killed in the captured game)
    ])


def test_op_lengths_agree():
    """Each new op's integer count is the same in protocol.py, the harness (W3S_ApplyOne's n) and the shim (jass_len):
    a count off by one in the shim had its step's later commands read from the wrong integers."""
    harness = (ROOT / "warcraftsim" / "harness" / "w3sim.j").read_text()
    apply = harness[harness.index("function W3S_ApplyOne"):harness.index("function W3S_ApplyChunk")]
    jass = {int(op): int(n) for op, n in re.findall(r"op == (\d+) then\s*(?://[^\n]*\n\s*)*set n = (\d+)", apply)}
    shim = (ROOT / "shim" / "units.c").read_text()
    table = shim[shim.index("static int jass_len"):shim.index("static int owned")]
    c = {int(op): int(n) for line in re.findall(r"((?:case \d+: )+)return (\d+);", table)
         for op, n in [(o, line[1]) for o in re.findall(r"case (\d+):", line[0])]}
    for op in (Op.REMOVE_UNIT, Op.SET_MINE_GOLD, Op.SET_RESEARCH, Op.PLACE, Op.PLACED):
        assert _OP_LENGTHS[op] == jass[int(op)] == c[int(op)], op
    assert all(_OP_LENGTHS[op] == n for op, n in jass.items() if op in _OP_LENGTHS)  # (and the others too)


def test_place_encoding():
    p = Place(1, "Hamg", -3000, 50, 833, 500, 4, ("AHbz", "AHbz", "AHab"), ("stwp",), dead=True, hero_xp=900)
    ints = encode_commands([p])
    assert command_ops(ints) == [Op.PLACE, Op.PLACED, Op.PLACED, Op.PLACED, Op.PLACED, Op.PLACED]
    assert ints[:9] == [Op.PLACE, 1, fourcc("Hamg"), -3000 + 65536, 50 + 65536, 833, 500, 4, 900]
    assert ints[-3:] == [Op.PLACED, 3, 0]  # (the death last: after its skills and items)
    assert command_ops(encode_commands([RemoveUnit(5), SetMineGold(6, 0), SetResearch(0, "Rhme", 2)])) == [
        Op.REMOVE_UNIT, Op.SET_MINE_GOLD, Op.SET_RESEARCH]


def test_capture(heroes):
    seen = {0: {}, 1: {}}
    rows = midgame()
    late.note_research(seen, [SimpleNamespace(kind=fx.RESEARCH_FINISH, a=2, b=fourcc("Rhme"), c=2),
                              SimpleNamespace(kind=fx.RESEARCH_FINISH, a=20, b=fourcc("Rume"), c=1)], rows)
    st = late.capture(rows, players(), ["human", "undead"], "echofast2", seen, 600.0,
                      jobs={23: "gold"}, mat={0: 3000.0, 1: 2000.0})
    assert st["minute"] == 10.0 and st["starts"] == [list(L), list(R)] and st["races"] == ["human", "undead"]
    h, u = st["sides"]
    assert h["research"] == {"Rhme": 2} and u["research"] == {"Rume": 1}
    assert (h["gold"], h["lumber"]) == (500, 150)
    by = {k[0]: k for k in h["units"]}
    assert [k[0] for k in h["units"]].count("hpea") == 3  # (the militia as a peasant; not the summoned, nor the frame)
    assert "hwat" not in by and "hgtw" not in by and by["hkee"][3] == 900 and by["hbar"][5] == "s"
    assert sorted(k[5] for k in h["units"] if k[0] == "hpea") == ["", "g", "l"]  # (in the mine: gold; at a tree: lumber)
    assert h["heroes"] == [["Hamg", -3000, 50, 833, 500, 4, 900, ["AHbz", "AHab", "AHwe", "AHbz"], ["stwp"], 0]]
    assert u["heroes"][0][-1] == 1 and u["heroes"][0][-3] == ["AUdc", "AUdp", "AUdc"]  # (dead; its skills kept)
    assert {k[0]: k[5] for k in u["units"] if k[0] in ("uaco", "ugho")} == {"uaco": "g", "ugho": "l"}
    assert ["ngol", 4352, -3200, 6250] in st["mines"] and ["ugol", 4352, -3200, 4550] in st["mines"]
    assert st["creeps"] == [["nmrr", 0, -3000]]
    json.dumps(st)  # (a line of late-states.jsonl)
    # decided games give none: a side without a town hall, or far behind in material
    assert late.capture(rows[rows[:, fx.C_ID] != 20], players(), ["human", "undead"], "echofast2", seen, 600.0) is None
    assert late.capture(rows, players(), ["human", "undead"], "echofast2", seen, 600.0, mat={0: 100.0, 1: 2000.0}) is None


def test_fit_and_swap(heroes):
    st = late.capture(midgame(), players(), ["human", "undead"], "echofast2", {0: {}, 1: {}}, 600.0)
    st.update(winner=0, learner=1)
    assert late.fit(st, "echofast2", ["human", "undead"], [L, R]) is st
    sw = late.fit(st, "echofast2", ["undead", "human"], [R, L])  # (the other player has each side)
    assert sw["races"] == ["undead", "human"] and sw["sides"][1] is st["sides"][0] and sw["starts"] == [list(R), list(L)]
    assert (sw["winner"], sw["learner"]) == (1, 0)
    assert late.fit(st, "echofast2", ["human", "undead"], [R, L]) is None  # (the start locations the other way: no
    assert late.fit(st, "echofast2", ["undead", "human"], [L, R]) is None  # mirror image of a state)
    assert late.fit(st, "(2)EchoIsles", ["human", "undead"], [L, R]) is None
    assert late.fit(st, "echofast2", ["human", "orc"], [L, R]) is None


def test_commands(heroes):
    seen = {0: {"Rhme": 2}, 1: {}}
    st = late.capture(midgame(), players(), ["human", "undead"], "echofast2", seen, 600.0)
    cmds = late.commands(st, fresh())
    removed = {c.unit for c in cmds if isinstance(c, RemoveUnit)}
    # the start's workers, ghoul and the tier-1 hall a keep replaces; the killed creep; undead's necropolis kept
    assert removed == {101, *range(102, 107), 122, 123, 124, 125, 141}
    gold = {c.unit: c.gold for c in cmds if isinstance(c, SetMineGold)}
    assert gold == {121: 3000, 130: 4000, 131: 4550, 132: 0}  # (the haunted expansion's mine: the haunted one's gold)
    placed = [c for c in cmds if isinstance(c, Place)]
    h = [c.unit_type for c in placed if c.player == 0]
    assert h[:2] == ["hkee", "htow"] and h.count("hpea") == 3 and "Hamg" == h[-1]  # (halls first, heroes last)
    u = [c.unit_type for c in placed if c.player == 1]
    assert "unpl" not in u and u.count("ugol") == 1  # (the start's hall and haunted mine kept; the expansion haunted)
    assert u.index("ugol") < u.index("uaco")
    mag = next(c for c in placed if c.unit_type == "Hamg")
    assert (mag.hero_level, mag.hero_xp, mag.skills, mag.items, mag.dead) == (4, 900, ("AHbz", "AHab", "AHwe", "AHbz"),
                                                                              ("stwp",), False)
    assert next(c for c in placed if c.unit_type == "Udea").dead
    assert [c for c in cmds if isinstance(c, SetResearch)] == [SetResearch(0, "Rhme", 2)]
    assert cmds.index(SetResearch(0, "Rhme", 2)) < cmds.index(next(c for c in placed if c.player == 0))
    assert [c for c in cmds if isinstance(c, SetResources)] == [SetResources(0, 500, 150), SetResources(1, 400, 100)]
    assert late.levels(st, 0) == {fourcc("Rhme"): 2}


def test_again(heroes):
    """Steps 1 and 2: the gold mine hidden under a placed haunted mine (the type's own gold), an uncovered mine the
    captured game had mined out, a night elf's entangled mine entangled again and then marked complete."""
    st = late.capture(midgame(), players(), ["human", "undead"], "echofast2", {0: {}, 1: {}}, 600.0)
    st["sides"][0]["units"].append(["egol", -5632, 3584, 1000, 1000, "s"])  # (as if the left side's mine were entangled)
    step1 = np.stack([row(301, "ngol", 15, 4352, -3200, flags=STRUCT | HIDDEN, resource=12500),
                      row(302, "ugol", 1, 4352, -3200, flags=STRUCT, resource=4550),
                      row(303, "ngol", 15, -4864, -3200, flags=STRUCT, resource=6250)])
    assert late.again(st, step1) == [SetMineGold(301, 6250), SetMineGold(303, 0), Place(0, "egol", -5632, 3584)]
    step2 = np.stack([row(304, "egol", 0, -5632, 3584, flags=STRUCT | int(UnitFlags.CONSTRUCTING), resource=4000),
                      row(302, "ugol", 1, 4352, -3200, flags=STRUCT, resource=4550)])
    st["mines"].append(["egol", -5632, 3584, 4000])
    assert late.again(st, step2) == [SetMineGold(304, 4000)]  # (its gold as it is, marked complete)


def test_jobs(heroes):
    st = late.capture(midgame(), players(), ["human", "undead"], "echofast2", {0: {}, 1: {}}, 600.0)
    rows = np.stack([*fresh()[fresh()[:, fx.C_OWNER] == 15],
                     row(201, "hpea", 0, -5600, 3500, flags=WORKER), row(202, "hpea", 0, -5890, 2610, flags=WORKER),
                     row(203, "hpea", 0, *L, flags=WORKER), row(210, "ugol", 1, 5120, 3584, flags=STRUCT),
                     row(211, "uaco", 1, 5000, 3300, flags=WORKER), row(212, "ugho", 1, 5300, 2500)])
    trees = {900: (-6000, 2600), 901: (5400, 2400), 902: (0, 0)}
    h = late.jobs(st, rows, 0, "human", trees)
    assert sorted((type(c).__name__, u, j, tg) for c, u, j, tg in h) == [
        ("TargetDestructable", 202, "lumber", 900), ("TargetOrder", 201, "gold", 130)]
    assert all(c.order == fx.HARVEST for c, *_ in h)
    u = late.jobs(st, rows, 1, "undead", trees)
    assert {(u_, j, tg) for _, u_, j, tg in u} == {(211, "gold", 210), (212, "lumber", 901)}  # (its own haunted mine)
    view = SimpleNamespace(assign={}, mine_of={}, added=lambda unit: None)
    assert late.noted(view, h, [True, False] if isinstance(h[0][0], TargetOrder) else [False, True]) == 1
    assert view.assign == {201: "gold"} and view.mine_of == {201: 130}
    assert isinstance(TargetDestructable(1, 2, 3), TargetDestructable)


def test_states_file(tmp_path, heroes):
    st = late.capture(midgame(), players(), ["human", "undead"], "echofast2", {0: {}, 1: {}}, 600.0)
    f = tmp_path / "late-states.jsonl"
    late.append(f, st)
    late.append(f, dict(st, map="(2)EchoIsles"))
    with open(f, "a") as fh:
        fh.write('{"map": "echofast2", "races"')  # (a line being written: read next time)
    s = late.States([str(f)], refresh=0.0)
    rng = random.Random(0)
    assert s.pick("echofast2", ["human", "undead"], [L, R], rng)["sides"][0]["gold"] == 500
    assert s.pick("echofast2", ["undead", "human"], [R, L], rng)["sides"][1]["gold"] == 500  # (swapped)
    assert s.pick("echofast2", ["human", "undead"], [R, L], rng) is None
    assert s.pick("(2)EchoIsles", ["human", "undead"], [L, R], rng)["map"] == "(2)EchoIsles"
    assert s.count == 2
    with open(f, "a") as fh:
        fh.write(':["orc","orc"]}\n')  # (the line completed: indexed now, not a state of these races)
    s.pick("echofast2", ["human", "undead"], [L, R], rng)
    assert s.count == 2 and s.done[0] == f.stat().st_size
