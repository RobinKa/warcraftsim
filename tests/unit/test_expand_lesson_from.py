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
from warcraftsim.protocol import Build, PointOrder, TargetOrder

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


def test_undead_haunts_with_one_acolyte():
    tc = _teacher("undead")
    assert len(_lessons(tc, _rows("undead"))) == 7
    cleared = _rows("undead", camp=False)
    out = _lessons(tc, cleared, now=470.0, t=1200)
    assert len(out) == 1 and isinstance(out[0], TargetOrder) and (out[0].order, out[0].target) == (code("ugol"), 51)
    walking = cleared.copy()
    walking[walking[:, fx.C_ID] == 12, fx.C_ORDER] = code("ugol")
    assert _lessons(tc, walking, now=480.0, t=1220) == []


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
    assert out == [TargetOrder(29, HARVEST, 51), TargetOrder(23, HARVEST, 51), TargetOrder(22, HARVEST, 51)]  # (the nearest)
    assert tc.taught["expansion: workers"] == 1
    # (carried out: three on their way read "harvest", no choppers of the view's any more; two more make five)
    sent = [_unit(29, 0, -4800, -3000, "hpea", W, order=HARVEST)] + [_unit(22 + k, 0, -4900, 1000, "hpea", W, order=HARVEST) for k in range(2)]
    tc.economy = (False, {20, 21}, {})
    out = _lessons(tc, _taken(gold + sent + chop[:2]), now=610.0, t=1520)
    assert out == [TargetOrder(21, HARVEST, 51), TargetOrder(20, HARVEST, 51)]
    # (five work its mine: none)
    there = [_unit(20 + k, 0, -4800, -3000 + 20 * k, "hpea", W, order=HARVEST) for k in range(5)]
    tc.economy = (False, set(), {})
    assert _lessons(tc, _taken(gold + there), now=620.0, t=1540) == []
    # (no idle worker, no chopper: the main's five stay)
    assert _lessons(tc, _taken(gold), now=630.0, t=1560) == []
    # (seven on the main's gold: the two past five go)
    more = gold + [_unit(15 + k, 0, -5450, 3350 + 10 * k, "hpea", W, order=HARVEST) for k in range(2)]
    out = _lessons(tc, _taken(more), now=640.0, t=1580)
    assert len(out) == 2 and all(isinstance(c, TargetOrder) and c.order == HARVEST and c.target == 51 for c in out)


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
