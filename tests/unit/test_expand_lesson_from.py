"""--expand-lesson-from S (teacher.py): in a game without a target the expansion lessons from game second S, once the
side has a living hero and 6 army units and no enemy army in sight near its halls, then for the rest of the game: the
camp (the heroes and the army near them, twice its strength), the hall (one builder), the workers (idle ones, then
choppers; the main keeps five on its gold). Off: nothing of it, and the teacher's draws as before."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import teacher as tm
from warcraftsim.fullgame.teacher import ATTACK, HARVEST, Teacher
from warcraftsim.protocol import Build, ImmediateOrder, PointOrder, TargetOrder

S, W = fx.STRUCTURE, fx.WORKER
START = (-5184.0, 2944.0)  # (Echo Isles' left start: its natural's hall where people put it, teacher.EXPAND_SPOTS)
MAIN, NAT = (-5632, 3584), (-4864, -3200)


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


def _unit(i: int, owner: int, x: int, y: int, typ: str, flags: int = 0, vis: int = 3, hp: int = 400, maxhp: int = 400,
          level: int = 0, order: int = 0) -> np.ndarray:
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = i, code(typ), owner, x, y
    r[fx.C_HP], r[fx.C_MAXHP], r[fx.C_FLAGS], r[fx.C_VIS], r[fx.C_HLEVEL], r[fx.C_ORDER] = hp, maxhp, flags, vis, level, order
    return r


def _player(gold=1000, lumber=500):
    return SimpleNamespace(gold=gold, lumber=lumber, food_used=30, food_cap=60, start_x=START[0], start_y=START[1])


def _rows(race="human", army=6, hero_hp=450, camp=True, camp_hp=400, workers=None, extra=()):
    hall, worker, unit, hero = {"human": ("htow", "hpea", "hfoo", "Hamg"), "undead": ("unpl", "uaco", "ugho", "Udea"),
                                "nightelf": ("etol", "ewsp", "earc", "Edem"), "orc": ("ogre", "opeo", "ogru", "Ofar")}[race]
    rows = [_unit(1, 0, int(START[0]), int(START[1]), hall, S), _unit(50, 15, *MAIN, "ngol", S), _unit(51, 15, *NAT, "ngol", S)]
    if workers is None:  # (five on the main's gold)
        workers = [_unit(10 + k, 0, -5400, 3300 + 10 * k, worker, W, order=HARVEST) for k in range(5)]
    rows += workers
    rows += [_unit(30 + k, 0, -4800, 2600 + 40 * k, unit) for k in range(army)]
    rows.append(_unit(40, 0, -4800, 2500, hero, fx.HERO, hp=hero_hp, maxhp=450, level=1))
    if camp:
        rows += [_unit(60 + k, 12, NAT[0] + 100 + 100 * k, NAT[1] + 300, "nftt", hp=camp_hp, maxhp=camp_hp) for k in range(2)]
    return np.stack(rows + list(extra))


def _teacher(race="human", **kw):
    tc = Teacher(0, race, random.Random(0), kw.pop("target", None), **{"expand_from": 360.0, **kw})
    return tc


def _lessons(tc, rows, now=400.0, t=1000, can=None, player=None):
    return [c for c in tc.commands(rows, player or _player(), now, t, can)
            if not (isinstance(c, Build) and c.building in tm.ALTARS)]


def test_off_by_default_and_the_draws_as_before():
    rows = _rows()
    off = Teacher(0, "human", random.Random(0), None)
    assert off.expand_from == 0.0 and _lessons(off, rows) == [] and "expansion: its camp" not in off.taught
    a, b = random.Random(5), random.Random(5)
    Teacher(0, "human", a, None)
    Teacher(0, "human", b, None, expand_from=360.0)  # (every side: no draw)
    assert a.random() == b.random()
    a, b = random.Random(5), random.Random(5)
    Teacher(0, "human", a, None)
    Teacher(0, "human", b, None, expand_share=0.5)  # (a share without the lesson: no draw either)
    assert a.random() == b.random()


def test_a_target_s_game_has_none():
    tc = _teacher(target={"t": {"hbar": 60.0}, "n": {}})
    assert tc.expand_from == 0.0 and _lessons(tc, _rows()) == []


def test_the_share():
    on = sum(Teacher(0, "human", random.Random(k), None, expand_from=360.0, expand_share=0.3).expand_from > 0 for k in range(400))
    assert 80 < on < 160
    assert Teacher(0, "human", random.Random(1), None, expand_from=360.0, expand_share=0.0).expand_from == 0.0


def test_the_camp_from_s_on_with_a_hero_and_six():
    rows = _rows()
    assert _lessons(_teacher(), rows, now=350.0) == []  # (before S)
    assert _lessons(_teacher(pace=0.5), rows, now=170.0) == []  # (faster rules: S x 0.5)
    tc = _teacher(pace=0.5)
    out = _lessons(tc, rows, now=185.0)
    cx, cy = float(NAT[0] + 150), float(NAT[1] + 300)
    assert out == [PointOrder(i, ATTACK, cx, cy) for i in [40] + [30 + k for k in range(6)]]  # (the hero and the army near it)
    assert tc.taught == {"expansion: its camp": 1} and tc.expand_on
    assert _lessons(tc, rows, now=186.0, t=1001) == []  # (a lesson every GAP steps)
    assert len(_lessons(tc, rows, now=196.0, t=1000 + tm.GAP)) == 7
    assert _lessons(_teacher(), _rows(army=5)) == []  # (five army units)
    no_hero = _rows()
    no_hero = no_hero[no_hero[:, fx.C_ID] != 40]
    assert _lessons(_teacher(), no_hero) == []
    raid = [_unit(91, 1, -5000, 2600, "ogru", vis=1)]  # (an enemy army unit it sees near its hall)
    assert _lessons(_teacher(), _rows(extra=raid)) == []
    unseen = [_unit(91, 1, -5000, 2600, "ogru", vis=2)]
    assert len(_lessons(_teacher(), _rows(extra=unseen))) == 7


def test_the_camp_only_when_strong_and_healthy():
    assert _lessons(_teacher(), _rows(hero_hp=200)) == []  # (a hero under CREEP_HP of its hit points)
    assert _lessons(_teacher(), _rows(camp_hp=20000)) == []  # (a camp it is not EARLY_MARGIN times as strong as)
    far = _rows()
    far[np.isin(far[:, fx.C_ID], [30, 31, 32, 33, 34, 35]), fx.C_X] = 4000  # (the army away from the hero: it alone is too weak)
    assert _lessons(_teacher(), far) == []


def test_the_fight_at_the_camp_goes_on():
    """Its heroes at the camp: the lesson goes on with a hero down to EARLY_FIGHT_HP and a group as strong as what is left
    of the camp (on the way there: above CREEP_HP and twice the camp)."""
    def at_camp(hero_hp, camp_hp=400, army=6):
        rows = _rows(hero_hp=hero_hp, camp_hp=camp_hp, army=army)
        mine = np.isin(rows[:, fx.C_ID], [40] + [30 + k for k in range(army)])
        rows[mine, fx.C_X], rows[mine, fx.C_Y] = NAT[0] + 200, NAT[1] + 600  # (300 from the creeps' middle)
        return rows
    tc = _teacher()
    assert len(_lessons(tc, _rows())) == 7  # (on)
    assert len(_lessons(tc, at_camp(200), t=1100)) == 7  # (44% of its hit points: fighting on)
    assert _lessons(tc, at_camp(120), t=1200) == []  # (27%: no more)
    assert _lessons(tc, _rows(hero_hp=200), t=1300) == []  # (at home with 44%: not sent)
    strong = at_camp(450, camp_hp=2400)  # (the group's strength 1 to 2 times the camp's: enough in the fight)
    ratio = tm._power(strong[np.isin(strong[:, fx.C_ID], [40] + [30 + k for k in range(6)])]) / tm._power(strong[strong[:, fx.C_OWNER] == 12])
    assert 1.0 < ratio < 2.0
    assert len(_lessons(tc, strong, t=1400)) == 7
    home = _rows(camp_hp=2400)
    assert _lessons(tc, home, t=1500) == []  # (the same camp from home: under twice)
    assert _lessons(tc, at_camp(450, camp_hp=6000), t=1600) == []  # (a camp stronger than the group: not in the fight either)


def test_on_for_the_rest_of_the_game_then_the_hall_by_one_builder():
    tc = _teacher()
    assert len(_lessons(tc, _rows())) == 7
    cleared = _rows(army=5, camp=False)  # (the camp cost a unit: five left)
    out = _lessons(tc, cleared, now=470.0, t=1200)
    assert len(out) == 1 and isinstance(out[0], Build) and out[0].building == code("htow")
    assert (out[0].x, out[0].y) == (-4764.0, -2400.0) and out[0].unit in range(10, 15)
    fresh = _teacher()
    assert _lessons(fresh, cleared, now=470.0, t=1200) == [] and not fresh.expand_on  # (never had six: not on)
    tc = _teacher()
    _lessons(tc, _rows())
    walking = cleared.copy()
    walking[walking[:, fx.C_ID] == 12, fx.C_ORDER] = code("htow")  # (a worker on its way to build it)
    assert _lessons(tc, walking, now=470.0, t=1200) == []
    assert _lessons(tc, cleared, now=470.0, t=1200, can=lambda c, k=0: False) == []  # (the hall not affordable: the mask)
    raid = np.concatenate([cleared, _unit(91, 1, -5000, 2600, "ogru", vis=1)[None]])
    assert _lessons(tc, raid, now=470.0, t=1204) == []  # (an enemy army near its hall: not now)
    assert len(_lessons(tc, cleared, now=480.0, t=1220)) == 1


def test_the_builder_is_no_gold_worker_when_another_can_go():
    """The hall's builder: a worker without work or a chopper before one of the main's gold workers, however near."""
    gold = [_unit(10 + k, 0, -5000, 2000 + 10 * k, "hpea", W, order=HARVEST) for k in range(5)]  # (the nearest to the natural)
    chopper, idle = _unit(20, 0, -5600, 3600, "hpea", W, order=HARVEST), _unit(21, 0, -5700, 3700, "hpea", W)
    tc = _teacher()
    _lessons(tc, _rows())
    tc.economy = (False, {20}, {})
    out = _lessons(tc, _rows(army=5, camp=False, workers=gold + [chopper, idle]), now=470.0, t=1200)
    assert len(out) == 1 and isinstance(out[0], Build) and out[0].unit == 20  # (the chopper, nearer than the idle one)
    tc = _teacher()
    _lessons(tc, _rows())
    out = _lessons(tc, _rows(army=5, camp=False, workers=gold + [idle]), now=470.0, t=1200)
    assert out[0].unit == 21
    ud = _teacher("undead")
    _lessons(ud, _rows("undead"))
    aco = [_unit(10 + k, 0, -5000, 2000 + 10 * k, "uaco", W, order=HARVEST) for k in range(5)]
    spare = _unit(21, 0, -5700, 3700, "uaco", W)
    out = _lessons(ud, _rows("undead", camp=False, workers=aco + [spare]), now=470.0, t=1200)
    assert out == [TargetOrder(21, code("ugol"), 51)]


def test_the_hall_s_cost_is_kept_while_it_is_due():
    """--expand-lesson-reserve N: with the early hall due (the camp gone, the natural free, no builder on its way) the
    view's reserve is the hall's class and cost, for N steps a game; not before, not while a builder walks, not off."""
    costs = np.zeros((4, 3), np.int64)
    costs[2] = (385, 205, 0)
    enc = SimpleNamespace(order_index={(code("htow"), fx.POINT): 2, (code("ugol"), fx.UNIT): 3}, costs=costs)
    tc = _teacher(expand_reserve=3)
    assert tc.hall_reserve(enc) is None  # (nothing due yet)
    _lessons(tc, _rows())
    assert not tc.hall_due and tc.hall_reserve(enc) is None  # (its camp stands)
    cleared = _rows(army=5, camp=False)
    poor = lambda c, k=0: False  # noqa: E731 (the policy could not pay: no lesson, but due)
    assert _lessons(tc, cleared, now=470.0, t=1200, can=poor) == [] and tc.hall_due
    assert tc.hall_reserve(enc) == (2, 385, 205)
    walking = cleared.copy()
    walking[walking[:, fx.C_ID] == 12, fx.C_ORDER] = code("htow")
    _lessons(tc, walking, now=471.0, t=1202, can=poor)
    assert not tc.hall_due and tc.hall_reserve(enc) is None  # (a builder on its way: nothing kept during the walk)
    _lessons(tc, cleared, now=472.0, t=1204, can=poor)
    assert tc.hall_reserve(enc) == (2, 385, 205) and tc.hall_reserve(enc) == (2, 385, 205)
    assert tc.hall_reserve(enc) is None and tc.reserved == 3  # (its three steps of the game used up)
    off = _teacher()
    _lessons(off, _rows())
    _lessons(off, cleared, now=470.0, t=1200, can=poor)
    assert off.hall_due and off.hall_reserve(enc) is None  # (no --expand-lesson-reserve)
    raid = np.concatenate([cleared, _unit(91, 1, -5000, 2600, "ogru", vis=1)[None]])
    tc2 = _teacher(expand_reserve=100)
    _lessons(tc2, _rows())
    _lessons(tc2, raid, now=470.0, t=1200, can=poor)
    assert not tc2.hall_due  # (an enemy army near its halls: not due, nothing kept)


def test_a_side_without_lessons_keeps_the_cost_too():
    """Teacher.saving (an evaluation game's side, a past opponent's): hall_due as commands() sets it, nothing taught."""
    costs = np.zeros((4, 3), np.int64)
    costs[2] = (385, 205, 0)
    enc = SimpleNamespace(order_index={(code("htow"), fx.POINT): 2}, costs=costs)
    cleared = _rows(army=5, camp=False)
    sv = _teacher(expand_reserve=100)
    sv.saving(cleared, _player(), 470.0, 1200)
    assert not sv.hall_due and not sv.expand_on  # (never had a hero and six)
    sv.saving(_rows(), _player(), 350.0, 1190)
    assert not sv.expand_on  # (before S)
    sv.saving(_rows(), _player(), 400.0, 1200)
    assert sv.expand_on and not sv.hall_due  # (on; its camp stands)
    sv.saving(cleared, _player(), 470.0, 1300)
    assert sv.hall_due and sv.hall_reserve(enc) == (2, 385, 205)
    raid = np.concatenate([cleared, _unit(91, 1, -5000, 2600, "ogru", vis=1)[None]])
    sv.saving(raid, _player(), 480.0, 1310)
    assert not sv.hall_due and sv.hall_reserve(enc) is None
    done = np.concatenate([cleared, _unit(70, 0, -4764, -2400, "htow", S)[None]])
    sv.saving(done, _player(), 490.0, 1320)
    assert not sv.hall_due  # (taken)
    target = _teacher(expand_reserve=100, target={"t": {"hbar": 60.0}, "n": {}})
    target.saving(_rows(), _player(), 400.0, 1200)
    target.saving(cleared, _player(), 470.0, 1300)
    assert not target.hall_due  # (a target's side: its builder's business)
    off = Teacher(0, "human", random.Random(0), None, expand_reserve=100)
    off.saving(_rows(), _player(), 400.0, 1200)
    off.saving(cleared, _player(), 470.0, 1300)
    assert not off.hall_due  # (no --expand-lesson-from)


def test_undead_haunts_with_one_acolyte():
    tc = _teacher("undead")
    assert len(_lessons(tc, _rows("undead"))) == 7
    cleared = _rows("undead", camp=False)
    out = _lessons(tc, cleared, now=470.0, t=1200)
    assert len(out) == 1 and isinstance(out[0], TargetOrder) and (out[0].order, out[0].target) == (code("ugol"), 51)
    walking = cleared.copy()
    walking[walking[:, fx.C_ID] == 12, fx.C_ORDER] = code("ugol")
    assert _lessons(tc, walking, now=480.0, t=1220) == []
    unseen = cleared.copy()
    unseen[unseen[:, fx.C_ID] == 51, fx.C_VIS] = 2  # (the mine out of its sight: no pointer could name it: walked there
    out = _lessons(tc, unseen, now=480.0, t=1220)   # first, as a target's lesson has it; a move's lesson, floored by
    assert out == [PointOrder(out[0].unit, fx.MOVE, float(NAT[0]), float(NAT[1]))] and tc.hall_due  # the "workers" group)
    target = Teacher(0, "undead", random.Random(0), {"t": {"ugol": 300.0}})
    out = target.commands(unseen, _player(), 480.0, 1220)
    assert len(out) == 1 and isinstance(out[0], PointOrder) and out[0].order == fx.MOVE  # (a target's: as before)
    spare = np.concatenate([unseen, _unit(25, 0, -5700, 3700, "uaco", W)[None]])  # (an acolyte without work walks, not a miner)
    assert _lessons(tc, spare, now=490.0, t=1240) == [PointOrder(25, fx.MOVE, float(NAT[0]), float(NAT[1]))]


def _taken(workers, extra=()):
    hall = [_unit(70, 0, -4764, -2400, "htow", S)]
    return _rows(camp=False, workers=workers, extra=hall + list(extra))


def test_the_workers_idle_first_then_choppers_and_the_main_keeps_five():
    gold = [_unit(10 + k, 0, -5400, 3300 + 10 * k, "hpea", W, order=HARVEST) for k in range(5)]
    chop = [_unit(20 + k, 0, -5000 + 30 * k, 2500, "hpea", W, order=HARVEST) for k in range(4)]
    builder = [_unit(29, 0, -4700, -2500, "hpea", W)]  # (beside the new hall, no order)
    tc = _teacher()
    _lessons(tc, _rows())
    tc.economy = (False, {20, 21, 22, 23}, {})
    out = _lessons(tc, _taken(gold + chop + builder), now=600.0, t=1500)
    # (the builder, and the choppers past the two that stay, the nearest first; two short still: a worker more, at the new hall)
    train = ImmediateOrder(70, code("hpea"))
    assert out == [TargetOrder(29, HARVEST, 51), TargetOrder(23, HARVEST, 51), TargetOrder(22, HARVEST, 51), train]
    assert tc.taught["expansion: workers"] == 1 and tc.taught["expansion: a worker"] == 1
    # (carried out: the builder mines, two on their way read "harvest" and are the view's choppers no more: the gold
    # workers past the main's five; the last two choppers stay, one more is trained)
    sent = [_unit(29, 0, -4800, -3000, "hpea", W, order=HARVEST)] + [_unit(22 + k, 0, -4900, 1000, "hpea", W, order=HARVEST) for k in range(2)]
    tc.economy = (False, {20, 21}, {})
    out = _lessons(tc, _taken(gold + sent + chop[:2]), now=610.0, t=1520)
    assert out == [TargetOrder(22, HARVEST, 51), TargetOrder(23, HARVEST, 51), train]
    # (five work its mine: none)
    there = [_unit(20 + k, 0, -4800, -3000 + 20 * k, "hpea", W, order=HARVEST) for k in range(5)]
    tc.economy = (False, set(), {})
    assert _lessons(tc, _taken(gold + there), now=620.0, t=1540) == []
    # (no idle worker, no chopper: the main's five stay, a worker is trained)
    assert _lessons(tc, _taken(gold), now=630.0, t=1560) == [train]
    assert _lessons(tc, _taken(gold), now=631.0, t=1562) == []  # (one every GAP steps)
    # (seven on the main's gold: the two past five go)
    more = gold + [_unit(15 + k, 0, -5450, 3350 + 10 * k, "hpea", W, order=HARVEST) for k in range(2)]
    out = _lessons(tc, _taken(more), now=640.0, t=1580)
    assert len(out) == 3 and all(isinstance(c, TargetOrder) and c.order == HARVEST and c.target == 51 for c in out[:2])
    assert out[2] == train
    # (five stand beside the new hall with no order, trained there: within 900 and not at work, so they are sent. The
    # target's lesson counts whoever is within 900 and would stop)
    standing = [_unit(20 + k, 0, -4700, -2500 + 20 * k, "hpea", W) for k in range(5)]
    out = _lessons(tc, _taken(gold + standing), now=650.0, t=1600)
    assert out == [TargetOrder(20 + k, HARVEST, 51) for k in range(3)]
    # (the mine out of its sight: no pointer can name it; the nearest of them walks to it first)
    dark = _taken(gold + standing)
    dark[dark[:, fx.C_ID] == 51, fx.C_VIS] = 2
    assert _lessons(tc, dark, now=660.0, t=1620) == [PointOrder(20, fx.MOVE, float(NAT[0]), float(NAT[1]))]


def test_a_worker_more_only_when_the_policy_could_and_the_hall_is_free():
    gold = [_unit(10 + k, 0, -5400, 3300 + 10 * k, "hpea", W, order=HARVEST) for k in range(5)]

    def taught(can=None, busy=(), race="human", rows=None):
        tc = _teacher(race)
        _lessons(tc, _rows(race))
        tc.busy = set(busy)
        return _lessons(tc, _taken(gold) if rows is None else rows, now=600.0, t=1500, can=can)
    assert taught() == [ImmediateOrder(70, code("hpea"))]
    seen = []
    assert taught(can=lambda c, k=fx.IMMEDIATE: seen.append((c, k)) or False) == [] and seen == [("hpea", fx.IMMEDIATE)]
    assert taught(busy={70}) == []  # (the new hall trains one already)
    building = _taken(gold)
    building[building[:, fx.C_ID] == 70, fx.C_FLAGS] |= fx.CONSTRUCTING
    assert taught(rows=building) == []  # (the hall not finished: the target's rule, nothing yet)
    # (undead: a haunted mine has no hall; the necropolis at home trains the acolyte)
    aco = [_unit(10 + k, 0, -5400, 3300 + 10 * k, "uaco", W, order=HARVEST) for k in range(5)]
    haunted = _rows("undead", camp=False, workers=aco, extra=[_unit(71, 0, *NAT, "ugol", S)])
    assert taught(race="undead", rows=haunted) == [ImmediateOrder(1, code("uaco"))]


def test_wisps_in_the_entangled_mine_count_and_burrowed_peons_do_not():
    """A night elf's five in its natural's entangled mine (hidden or loaded): the mine has its workers, no lesson. An
    orc's peons in a burrow are none of it."""
    tc = _teacher("nightelf")
    _lessons(tc, _rows("nightelf"))
    home = [_unit(10 + k, 0, -5400, 3300, "ewsp", W | tm.HIDDEN) for k in range(5)]
    tree, held = _unit(70, 0, -4736, -2368, "etol", S), _unit(71, 0, *NAT, "egol", S)
    for flags in (tm.HIDDEN, tm.LOADED, tm.HIDDEN | tm.LOADED):
        there = [_unit(20 + k, 0, NAT[0], NAT[1], "ewsp", W | flags) for k in range(5)]
        idle = [_unit(30 + k, 0, -4700, -2600, "ewsp", W) for k in range(2)]
        rows = _rows("nightelf", army=0, camp=False, workers=home + there + idle, extra=[tree, held])
        assert _lessons(tc, rows, now=700.0, t=2000 + 40 * flags) == []
    orc = _teacher("orc")
    _lessons(orc, _rows("orc"))
    gold = [_unit(10 + k, 0, -5400, 3300, "opeo", W, order=HARVEST) for k in range(5)]
    burrowed = [_unit(20 + k, 0, -5000, 2800, "opeo", W | tm.LOADED) for k in range(3)]
    hall = _unit(70, 0, -4764, -2400, "ogre", S)
    out = _lessons(orc, _rows("orc", camp=False, workers=gold + burrowed, extra=[hall]), now=700.0, t=2000)
    assert out == [ImmediateOrder(70, code("opeo"))]  # (none to send: the burrow's peons are no surplus on gold)


def test_the_target_s_lessons_are_as_they_were():
    """A target's due expansion: the three nearest workers whatever they do, and a second builder's label (unchanged)."""
    tc = Teacher(0, "human", random.Random(0), {"t": {"htow": 300.0}}, expand_from=360.0)
    cleared = _rows(camp=False)
    walking = cleared.copy()
    walking[walking[:, fx.C_ID] == 12, fx.C_ORDER] = code("htow")
    out = tc.commands(walking, _player(), 470.0, 1200)
    assert len(out) == 1 and isinstance(out[0], Build)  # (another worker: the target's lesson has no such test)
    gold = [_unit(10 + k, 0, -5400, 3300 + 10 * k, "hpea", W, order=HARVEST) for k in range(5)]
    out = tc.commands(_taken(gold), _player(), 600.0, 1500)
    assert len(out) == 3 and all(c.unit in range(10, 15) for c in out)  # (the main's own gold workers)


@pytest.mark.parametrize("race", ["human", "orc", "undead", "nightelf"])
def test_every_race_gets_its_camp_lesson(race):
    assert len(_lessons(_teacher(race), _rows(race))) == 7


def test_until():
    """--expand-lesson-until S: no camp or hall lesson and nothing kept after S game seconds (times the pace); off: as
    before; a hall taken still gets its workers (the workers' lesson is above the limit's check)."""
    costs = np.zeros((4, 3), np.int64)
    costs[2] = (385, 205, 0)
    enc = SimpleNamespace(order_index={(code("htow"), fx.POINT): 2}, costs=costs)
    tc = _teacher(expand_until=600.0, expand_reserve=10)
    assert len(_lessons(tc, _rows())) == 7  # (the camp, before S)
    late = _teacher(expand_until=600.0, expand_reserve=10)
    late.expand_on = True
    assert _lessons(late, _rows(), now=700.0, t=1000) == []  # (the camp, after S: none)
    cleared = _rows(army=5, camp=False)
    assert len(_lessons(tc, cleared, now=470.0, t=1200)) == 1 and tc.hall_due  # (the hall, before S)
    assert tc.hall_reserve(enc) == (2, 385, 205)
    tc2 = _teacher(expand_until=600.0, expand_reserve=10)
    _lessons(tc2, _rows())
    assert _lessons(tc2, cleared, now=650.0, t=1300) == [] and not tc2.hall_due and tc2.hall_reserve(enc) is None
    off = _teacher(expand_reserve=10)
    _lessons(off, _rows())
    assert len(_lessons(off, cleared, now=650.0, t=1300)) == 1  # (no limit: as before)
