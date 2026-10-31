"""Late-game starts (fullgame/late.py): a game's state captured, its sides swapped, the first step's commands that set
it up, the step-1 harvest orders; the new harness ops' lengths alike in protocol.py, the harness and the shim."""

import copy
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


def test_weighted_files(tmp_path, heroes):
    """--late-states PATH:W: a file's states are a share W of the late starts however many the others hold; the files
    without a weight share the rest alike; without weights every state alike, as before."""
    assert late.weighted("runs/a/late-states.jsonl") == ("runs/a/late-states.jsonl", None)
    assert late.weighted("people.jsonl:0.5") == ("people.jsonl", 0.5)
    assert late.weighted("dir:x/a.jsonl") == ("dir:x/a.jsonl", None) and late.weighted(":0.5") == (":0.5", None)
    with pytest.raises(ValueError):
        late.weighted("people.jsonl:-1")
    st = late.capture(midgame(), players(), ["human", "undead"], "echofast2", {0: {}, 1: {}}, 600.0)
    own, people = tmp_path / "own.jsonl", tmp_path / "people.jsonl"
    for k in range(9):
        late.append(own, dict(st, tag="own"))
    late.append(people, dict(st, tag="people"))
    late.append(people, dict(st, tag="people", races=["orc", "orc"]))  # (another matchup: only people's file has it)

    def share(files, races=("human", "undead"), n=4000):
        s, rng = late.States([str(f) for f in files], refresh=0.0), random.Random(1)
        got = [s.pick("echofast2", list(races), [L, R], rng) for _ in range(n)]
        return sum(g is not None and g["tag"] == "people" for g in got) / n, sum(g is None for g in got) / n

    assert abs(share([own, people])[0] - 0.1) < 0.02  # (no weights: each state alike, 1 of 10)
    assert abs(share([own, f"{people}:0.5"])[0] - 0.5) < 0.03  # (the own file: the rest, 0.5)
    assert abs(share([f"{own}:0.25", f"{people}:0.25"])[0] - 0.5) < 0.03  # (weights only: renormalized)
    assert abs(share([own, f"{people}:0.8"])[0] - 0.8) < 0.03
    assert share([own, f"{people}:0"]) == (0.0, 0.0)  # (weight 0: never)
    assert share([own, f"{people}:1"]) == (1.0, 0.0)  # (the rest 0: the own file never, while people's fits)
    assert share([own, f"{people}:0.5"], races=("orc", "orc")) == (1.0, 0.0)  # (the only file that fits)
    assert share([f"{own}:0.5"], races=("orc", "orc")) == (0.0, 1.0)  # (none fits: a game from the start)


def test_rescaled(heroes):
    """A state of the real Echo Isles as echofast2's (the same layout: their map files differ only in the mines' gold
    and the starting resources): the mines' gold and each side's gold and lumber halved, the rest as it was."""
    st = late.capture(midgame(), players(gold=(800, 401), lumber=(300, 51)), ["human", "undead"], "(2)EchoIsles",
                      {0: {"Rhme": 2}, 1: {}}, 600.0)
    assert late.layout("echofast2") == late.layout("echo50") == late.layout("(2)EchoIsles") == "(2)EchoIsles"
    assert late.layout("duelfast") is None and late.rescaled(st, "duelfast") is None
    assert late.rescaled(st, "(2)LastRefuge-1.3") is None and late.rescaled(st, "(2)EchoIsles") is st
    e = late.rescaled(st, "echofast2")
    assert e["map"] == "echofast2" and e["captured_on"] == "(2)EchoIsles" and st["map"] == "(2)EchoIsles"
    assert [(s["gold"], s["lumber"]) for s in e["sides"]] == [(400, 150), (200, 26)]  # (rounded to even: 200.5, 25.5)
    assert ["ngol", 4352, -3200, 3125] in e["mines"] and ["ugol", 4352, -3200, 2275] in e["mines"]
    assert e["sides"][0]["units"] == st["sides"][0]["units"] and e["sides"][0]["research"] == {"Rhme": 2}
    assert e["sides"][1]["heroes"] == st["sides"][1]["heroes"] and e["creeps"] == st["creeps"]
    assert late.fit(e, "echofast2", ["undead", "human"], [R, L])["sides"][1]["gold"] == 400
    back = late.rescaled(e, "(2)EchoIsles")
    assert back["captured_on"] == "(2)EchoIsles" and back["sides"][0]["gold"] == 800


def demo(steps: int = 12) -> tuple[dict, dict]:
    """A demonstration's arrays (collect.py's; a step a minute): the fresh game at step 0, midgame() after, the human's
    barracks researching Rhme (levels 1, 2, 3 at steps 3, 5, 11)."""
    mid, first = midgame(), fresh()
    units = np.concatenate([np.c_[np.full(len(first), 0), first[:, 1:18]]]
                           + [np.c_[np.full(len(mid), t), mid[:, 1:18]] for t in range(1, steps + 1)])
    hero = mid[(mid[:, fx.C_FLAGS] & HERO) > 0]
    heroes = np.asarray([[t, int(h[fx.C_ID]), *h[fx.C_ITEM0:fx.C_ITEM0 + 6], *h[fx.C_ABIL0:fx.C_ABIL0 + 8:2], 0, 0, 0, 0]
                         for t in range(1, steps + 1) for h in hero], np.int64)
    players_ = np.asarray([[t, s, 500 + 10 * t, 100 + s, 0, 0, 0, 0, 0, 0, 0] for t in range(steps + 1) for s in (0, 1)])
    events = np.asarray([[t, fx.RESEARCH_FINISH, 2, fourcc("Rhme"), lv] for t, lv in ((3, 1), (5, 2), (11, 3))])
    meta = {"races": ["human", "undead"], "map": "(2)EchoIsles", "steps": steps, "step_seconds": 60.0,
            "result": {"0": "DEFEAT", "1": "VICTORY"}, "source": {"replay": "10900.w3g"},
            "sync": {"subgroup_units": 5, "subgroup_matched": 5, "first_miss": None}}
    return {"units": units, "heroes": heroes, "players": players_, "events": events}, meta


@pytest.fixture
def plans(heroes, monkeypatch):
    """Plans without the game's data: the build times, research levels' times, the tech tree (none between the items)."""
    from warcraftsim.fullgame import strategy as strat, techtree
    monkeypatch.setattr(strat, "build_seconds", lambda: {"hbar": 60.0, "hcas": 140.0, "Rhme": 60.0, "hsor": 30.0})
    monkeypatch.setattr(strat, "research_time_steps", lambda: {"Rhme": 15.0})
    monkeypatch.setattr(strat, "consistent", lambda st, tree=None: st)
    monkeypatch.setattr(techtree, "counts_as", lambda types, race: set(types) | ({"htow"} if "hkee" in types else set()))
    monkeypatch.setattr(strat, "restricted", lambda c: c[:1].isupper() or c in ("hfoo", "hsor", "hrif", "hkni"))


def planned_demo():
    """demo() with what the human goes on to make after minute 4 (a step a minute): footmen (held) and sorceresses (new),
    a castle, Rhme's next levels, a second barracks (held); a new expansion's hall first seen at minute 8; a tavern hero at 9;
    the undead side makes one ghoul only."""
    arrays, meta = demo()
    extra = [[t, 3, 2, fourcc(c), 0] for t, c in ((6, "hbar"),)] + [[7, 12, 1, fourcc("hcas"), 0]]
    extra += [[t, 6, 2, 900 + t, fourcc(c)] for t, c in ((5, "hfoo"), (6, "hfoo"), (7, "hsor"), (8, "hsor"), (9, "hsor"),
                                                         (3, "hrif"))]  # (the rifleman: before the state)
    extra += [[5, 6, 20, 950, fourcc("ugho")]]
    arrays["events"] = np.asarray(sorted(extra + arrays["events"].tolist()), np.int64)
    units = arrays["units"]
    hall = row(60, "htow", 0, 4352, -3200, flags=STRUCT)  # (at the fresh map's mine, far from its start)
    tav = row(61, "Nalc", 0, -3000, 100, flags=HERO)
    new = [np.c_[np.full(1, t), hall[None, 1:18]] for t in range(8, 13)] + [np.c_[np.full(1, t), tav[None, 1:18]]
                                                                             for t in range(9, 13)]
    arrays["units"] = np.concatenate([units, *new])
    arrays["units"] = arrays["units"][np.argsort(arrays["units"][:, 0], kind="stable")]
    return arrays, meta


def test_plan_from_demo(plans):
    """A person's plan: what the side made after the state, timed from it, without what it holds; its next research
    level and an expansion all the same; a side that made little has none; none far from the end."""
    arrays, meta = planned_demo()
    st = late.from_demo(arrays, meta, [4], [R, L], plans=True)[0]
    p = st["sides"][0]["plan"]
    assert set(p["t"]) == {"hsor", "hcas", "Rhme", "htow", "Nalc"}  # (not hfoo, hbar, Hamg: the state holds them)
    assert p["t"]["hsor"] == 180.0  # (at minute 7, from minute 4)
    assert p["t"]["hcas"] == 180.0 and p["t"]["Nalc"] == 300.0 and p["t"]["htow"] == 240.0
    assert p["t"]["Rhme"] == 80.0  # (level 2 finished a minute later, but research under way is not in the state: its time)
    assert p["n"]["hsor"] == 3 and p["n"]["hfoo"] == 2 and "hrif" not in p["n"]  # (the units trained after the state only)
    assert p["n"]["hpea"] == 0  # (3 peasants held, under a race's 5)
    assert {"hfoo", "hbar", "Hamg", "hkee", "Rhme"} <= set(p["held"]) and "htow" not in p["held"]  # (its hall at no mine)
    assert (p["end"], p["won"]) == (480.0, False)
    assert "plan" not in st["sides"][1]  # (one ghoul made: little)
    assert all("plan" not in s for s in late.from_demo(arrays, meta, [10], [R, L], plans=True)[0]["sides"])  # (one item)
    # a hall at a mine is an expansion the state holds; the plan's own expansion comes all the same
    with_mine = dict(st, mines=[*st["mines"], ["ngol", -4864, -3200, 6250]])
    assert "htow" in late.holds(with_mine, 0)
    assert "htow" not in late.holds(st, 0) and "unpl" not in late.holds(st, 1)  # (the start's hall: no item)
    # the old states' format without --plans
    assert all("plan" not in s for s in late.from_demo(arrays, meta, [4], [R, L])[0]["sides"])
    json.dumps(st)


def test_plan_moves_with_the_side_and_the_rules(plans):
    """A swapped state swaps the plans; on echofast2 the plan's times go at the people's pace, by the side's race."""
    arrays, meta = planned_demo()
    st = late.from_demo(arrays, meta, [4], [R, L], plans=True)[0]
    sw = late.fit(st, "(2)EchoIsles", ["undead", "human"], [R, L])
    assert sw["sides"][1]["plan"] is st["sides"][0]["plan"] and "plan" not in sw["sides"][0]
    assert late.plan_target(sw, 1, ["hsor"])["race"] == "human" and late.plan_target(sw, 0, ["hsor"]) is None
    assert late.plan_target(sw, 1, ["hsor"])["opp"] == "undead"
    e = late.rescaled(st, "echofast2")
    f = late.PEOPLE_PACE["human"]
    assert e["sides"][0]["plan"]["t"]["hsor"] == round(180.0 * f, 1) and e["sides"][0]["plan"]["end"] == round(480.0 * f, 1)
    assert st["sides"][0]["plan"]["t"]["hsor"] == 180.0  # (the state itself as it was)
    sw2 = late.fit(e, "echofast2", ["undead", "human"], [R, L])
    assert sw2["sides"][1]["plan"]["t"] == e["sides"][0]["plan"]["t"]
    assert late.rescaled(e, "(2)EchoIsles")["sides"][0]["plan"]["t"]["hsor"] == pytest.approx(180.0, abs=0.2)
    assert late.pace("(2)EchoIsles", "orc") == 1.0 and 0.0 < late.pace("echo50", "orc") - 0.728 < 0.272


def test_plan_as_a_target_no_windfall(plans):
    """As a target: only the pool's items, what it holds no windfall and no penalty; a plan's item made later credited,
    and the input in range (made, at 0, for what it holds)."""
    from warcraftsim.fullgame import strategy as strat
    arrays, meta = planned_demo()
    st = late.from_demo(arrays, meta, [4], [R, L], plans=True)[0]
    items = ["hfoo", "hrif", "hsor", "hkee", "Rhme", "Hamg", "hbar", "Nalc", "htow", "hkni"]
    z = late.plan_target(st, 0, items, drop=("Nalc",), trainable={"hsor", "hfoo"})
    assert z["narrow"] and z["src"] == "late" and z["race"] == "human" and "Nalc" not in z["t"]
    assert z["n"].get("hrif") is None and z["n"]["hsor"] == 3  # (the army's mix: units trained)
    made: dict = {}
    assert strat.mismatch(made, z, items, 1.5) == 0  # (the first steps: nothing due, nothing counted)
    assert strat.mix_gap({}, z) == 0.0
    assert strat.mismatch(made, z, items, 79.0) == 0
    assert strat.mismatch(made, z, items, 81.0) == 1  # (Rhme's next level due)
    made["Rhme"] = 81.0
    assert strat.mismatch(made, z, items, 81.0) == 0  # (made: credited)
    made["hfoo"] = 20.0  # (more of what it holds: neither made nor missed)
    assert strat.mismatch(made, z, items, 81.0) == 0
    made["hkni"] = 30.0  # (an army unit neither planned nor held: counts, as any narrow target's)
    assert strat.mismatch(made, z, items, 81.0) == 1
    v = strat.vector(z, items)
    assert v.min() >= 0.0 and v.max() <= 1.0
    k = {c: i for i, c in enumerate(items)}
    assert (v[k["hfoo"]], v[strat.SLOTS + k["hfoo"]]) == (1.0, 0.0)  # (held: made, at 0)
    assert v[k["hsor"]] == 1.0 and v[strat.SLOTS + k["hsor"]] == pytest.approx(180.0 / strat.HORIZON)
    assert (v[k["hrif"]], v[strat.SLOTS + k["hrif"]]) == (0.0, 1.0)  # (not in it)
    assert late.plan_target(st, 1, items) is None  # (no plan: no target)
    own = late.from_demo(arrays, meta, [4], [R, L])[0]
    assert late.plan_target(own, 0, items) is None  # (a state without plans: an own game's)
    assert late.plan_target(st, 0, ["hrif"]) is not None  # (its army's mix alone)
    won = copy.deepcopy(st)
    won["sides"][0]["plan"]["won"] = False
    assert late.plan_target(won, 0, ["hrif"]) is not None and late.plan_target(won, 0, ["hrif"], won_only=True) is None
    won["sides"][0]["plan"]["won"] = True
    assert late.plan_target(won, 0, ["hrif"], won_only=True) is not None  # (--late-z 2: the winner's plan only)


def test_late_z_keeps_defaults():
    """Without "held" the strategy's functions read as before."""
    from warcraftsim.fullgame import strategy as strat
    s = {"t": {"hfoo": 90.0}, "race": "human"}
    v = strat.vector(s, ["hfoo", "hrif"])
    assert (v[0], v[strat.SLOTS], v[1], v[strat.SLOTS + 1]) == (1.0, 0.1, 0.0, 1.0)
    assert strat.mismatch({"hrif": 10.0}, s, ["hfoo", "hrif"], 100.0) == 2


def test_from_demo(heroes):
    """People's states from a demonstration: capture's at each minute it reached (not in its last one), the research
    levels by then, each side's start location the one nearest its town hall at step 0, how the game ended."""
    arrays, meta = demo()
    got = late.from_demo(arrays, meta, [4, 10, 12, 14], [R, L])  # (12: the game's last minute; 14: past its end)
    assert [s["minute"] for s in got] == [4.0, 10.0]
    a, b = got
    assert a["starts"] == [list(L), list(R)] and a["races"] == ["human", "undead"] and a["map"] == "(2)EchoIsles"
    assert a["sides"][0]["research"] == {"Rhme": 1} and b["sides"][0]["research"] == {"Rhme": 2}
    assert (a["sides"][0]["gold"], a["sides"][1]["lumber"]) == (540, 101)
    assert (b["winner"], b["end_minute"], b["replay"]) == (1, 12.0, "10900.w3g")
    mag = a["sides"][0]["heroes"][0]
    assert mag[0] == "Hamg" and mag[5:9] == [4, 900, ["AHbz", "AHab", "AHwe", "AHbz"], ["stwp"]]
    want = late.capture(fx.hero_rows(arrays["units"][arrays["units"][:, 0] == 10],
                                     arrays["heroes"][arrays["heroes"][:, 0] == 10]),
                        players(gold=(600, 600), lumber=(100, 101)), ["human", "undead"], "(2)EchoIsles",
                        {0: {"Rhme": 2}, 1: {}}, 600.0)
    assert b["sides"][0]["units"] == want["sides"][0]["units"] and b["mines"] == want["mines"]
    assert late.from_demo(arrays, meta, [10], [R, L], before_end=3.0) == []
    off = dict(meta, sync={"subgroup_units": 5, "subgroup_matched": 4, "first_miss": None})
    assert late.from_demo(arrays, off, [4, 10], [R, L]) == []  # (out of step with its replay)
    values = {str(fourcc("hkee")): 1000, str(fourcc("unpl")): 10}  # (the undead under a quarter: decided)
    assert late.from_demo(arrays, meta, [4, 10], [R, L], values) == []
    assert len(late.from_demo(arrays, meta, [4, 10], [R, L], {str(fourcc("hkee")): 1000, str(fourcc("unpl")): 900})) == 2
