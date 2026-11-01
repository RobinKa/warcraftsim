"""--hunt-lesson (teacher.py): an army, the enemy beaten as far as it can tell and no building of it in view: attack-move to
a building it remembers, else to a gold mine it has not had in sight lately. Only what the player can know."""

import random
from types import SimpleNamespace

import numpy as np

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import teacher as tm
from warcraftsim.fullgame.selfplay import label_rows
from warcraftsim.fullgame.teacher import ATTACK, Teacher
from warcraftsim.protocol import PointOrder


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


def _unit(i: int, owner: int, x: int, y: int, typ: str, flags: int = 0, vis: int = 3) -> np.ndarray:
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = i, code(typ), owner, x, y
    r[fx.C_HP], r[fx.C_MAXHP], r[fx.C_FLAGS], r[fx.C_VIS] = 400, 400, flags, vis
    return r


S = fx.STRUCTURE
HOME, ENEMY_MAIN, ENEMY_NATURAL, OWN_NATURAL = (-5000, 3000), (5000, 3000), (4400, -3200), (-4800, -3200)
MINES = [_unit(100 + k, 15, x, y, "ngol", S, vis=0) for k, (x, y) in enumerate((HOME, OWN_NATURAL, ENEMY_MAIN, ENEMY_NATURAL))]
PLAYER = SimpleNamespace(gold=0, lumber=0, food_used=30, food_cap=60, start_x=-5000.0, start_y=3000.0)


def _rows(at=ENEMY_MAIN, n=8, extra=()):
    """Side 0's hall at home, its army of n at `at`, the map's mines, `extra`."""
    army = [_unit(30 + k, 0, at[0] + 20 * k, at[1], "hfoo") for k in range(n)]
    return np.stack([_unit(1, 0, *HOME, "htow", S), *army, *MINES, *extra])


def _main(k: int, vis: int = 1, flags: int = S):
    return _unit(200 + k, 1, ENEMY_MAIN[0] + 100 * k, ENEMY_MAIN[1], "ogre" if k == 0 else "obar", flags, vis=vis)


def _hunter() -> Teacher:
    tc = Teacher(0, "human", random.Random(0), hunt=1)
    tc.first, tc.pick = None, None  # (no other lessons in the way)
    return tc


def _beaten(tc: Teacher, buildings: int = 3) -> int:
    """It saw `buildings` of the enemy's at its main and saw them gone: its first look at them, then their deaths, an
    enemy unit in view at both (the last contact: 302 s)."""
    grunt = _unit(400, 1, ENEMY_MAIN[0], ENEMY_MAIN[1] - 300, "ogru", vis=1)
    assert tc.commands(_rows(extra=[grunt, *(_main(k) for k in range(buildings))]), PLAYER, 300.0, 100) == []
    assert tc.commands(_rows(extra=[grunt, *(_main(k, flags=S | fx.DEAD) for k in range(buildings))]), PLAYER, 302.0, 104) == []
    assert tc.hunt_known == {} and tc.hunt_gone == buildings
    return 108


def _hunts(out: list) -> list:
    return [c for c in out if isinstance(c, PointOrder) and c.order == ATTACK]


def test_off_by_default_nothing_changes():
    tc = Teacher(0, "human", random.Random(0))
    assert tc.hunt == 0
    other = Teacher(0, "human", random.Random(0), hunt=0)
    other.first, other.pick, tc.first, tc.pick = None, None, None, None
    rows = _rows()
    for t, now in zip(range(100, 400, 4), range(300, 1000, 50)):
        assert tc.commands(rows, PLAYER, float(now), t) == []
    assert tc.hunt_lessons == 0 and tc.hunt_mines is None and tc.hunt_gone == 0  # (no record kept)
    assert "look hunt" not in tc.last


def test_teaches_the_nearest_mine_it_has_not_seen():
    tc = _hunter()
    t = _beaten(tc)
    assert tc.commands(_rows(), PLAYER, 421.0, t) == []  # (119 s since an enemy army unit was in view)
    out = _hunts(tc.commands(_rows(), PLAYER, 423.0, t + 4))
    assert out and {(c.x, c.y) for c in out} == {tuple(map(float, ENEMY_NATURAL))}  # (its main in sight, its own far)
    assert {c.unit for c in out} == set(range(30, 36))  # (HUNT_ARMY of the 8, half of them being fewer)
    big = _hunter()
    t = _beaten(big)
    big.hunt_contact = 0.0
    assert len(_hunts(big.commands(_rows(n=20), PLAYER, 500.0, t))) == 10  # (half)
    assert tc.hunt_lessons == 1 and tc.taught["hunt"] == 1
    assert tc.hunt_found_at is None


def test_never_while_an_enemy_building_is_in_view():
    tc = _hunter()
    t = _beaten(tc)
    tc.hunt_contact = 0.0
    rows = _rows(extra=[_unit(300, 1, *ENEMY_NATURAL, "ogre", S, vis=1)])  # (the natural's hall in its view)
    for k in range(40):
        assert _hunts(tc.commands(rows, PLAYER, 500.0 + 10 * k, t + 100 * k)) == []
    assert tc.hunt_lessons == 0


def test_a_building_out_of_view_tells_it_nothing():
    """The natural's hall stands, the rows have it, and the side has not seen it: as if it were not there (vis bit 2)."""
    tc = _hunter()
    t = _beaten(tc)
    tc.hunt_contact = 0.0
    hidden = _unit(300, 1, 1000, -1000, "ogre", S, vis=2)  # (seen by the enemy only: at no mine)
    out = _hunts(tc.commands(_rows(extra=[hidden]), PLAYER, 500.0, t))
    assert out and {(c.x, c.y) for c in out} == {tuple(map(float, ENEMY_NATURAL))}
    assert 300 not in tc.hunt_known
    # (and with it gone: the same)
    again = _hunter()
    t = _beaten(again)
    again.hunt_contact = 0.0
    assert _hunts(again.commands(_rows(), PLAYER, 500.0, t)) == out


def test_a_building_it_remembers_comes_first():
    tc = _hunter()
    t = _beaten(tc)
    tc.hunt_contact = 0.0
    far = _unit(300, 1, 1000, -1000, "hhou", S, vis=1)
    assert tc.commands(_rows(at=(1000, -1100), extra=[far]), PLAYER, 400.0, t) == []  # (in view)
    out = _hunts(tc.commands(_rows(at=ENEMY_MAIN, extra=[_unit(300, 1, 1000, -1000, "hhou", S, vis=2)]), PLAYER, 500.0, t + 4))
    assert {(c.x, c.y) for c in out} == {(1000.0, -1000.0)}  # (out of view, its place too: remembered)
    # (its place in sight and nothing there: it is gone, and the mines' turn)
    assert tc.commands(_rows(at=(1000, -1100), extra=[_unit(300, 1, 1000, -1000, "hhou", S, vis=2)]), PLAYER, 510.0, t + 8) == []
    assert 300 not in tc.hunt_known and tc.hunt_gone == 4


def test_not_when_it_saw_too_little_or_the_enemy_army_is_about():
    t = 100
    tc = _hunter()  # (seen and never gone: a wave gone home is no beaten enemy)
    assert tc.commands(_rows(extra=[_main(k) for k in range(3)]), PLAYER, 300.0, t) == []
    assert _hunts(tc.commands(_rows(at=HOME, extra=[_main(k, vis=2) for k in range(3)]), PLAYER, 900.0, t + 400)) == []
    tc = _hunter()
    t = _beaten(tc, 2)  # (two gone)
    tc.hunt_contact = 0.0
    assert _hunts(tc.commands(_rows(), PLAYER, 900.0, t)) == []
    tc = _hunter()
    t = _beaten(tc)
    tc.hunt_contact = 0.0
    grunt = _unit(400, 1, 4000, 2000, "ogru", vis=1)
    assert _hunts(tc.commands(_rows(extra=[grunt]), PLAYER, 900.0, t)) == []  # (an army unit in view)
    assert tc.hunt_contact == 900.0
    peon = _unit(401, 1, 4000, 2000, "opeo", fx.WORKER, vis=1)
    tc.hunt_contact = 0.0
    assert _hunts(tc.commands(_rows(extra=[peon]), PLAYER, 900.0, t + 4))  # (a worker is no army)
    tc = _hunter()
    t = _beaten(tc)
    tc.hunt_contact = 0.0
    assert _hunts(tc.commands(_rows(n=5), PLAYER, 900.0, t)) == []  # (no army to speak of)


def test_tries_each_mine_in_turn_and_every_so_often():
    tc = _hunter()
    t = _beaten(tc)
    tc.hunt_contact = 0.0
    out = _hunts(tc.commands(_rows(), PLAYER, 500.0, t))
    assert {(c.x, c.y) for c in out} == {tuple(map(float, ENEMY_NATURAL))}
    assert _hunts(tc.commands(_rows(), PLAYER, 502.0, t + 4)) == []  # (HUNT_GAP)
    t += tm.HUNT_GAP
    # (the army at that mine, nothing there: its own natural next, the one it has not looked at since the game began)
    out = _hunts(tc.commands(_rows(at=ENEMY_NATURAL), PLAYER, 600.0, t))
    assert {(c.x, c.y) for c in out} == {tuple(map(float, OWN_NATURAL))}
    t += tm.HUNT_GAP
    assert _hunts(tc.commands(_rows(at=OWN_NATURAL), PLAYER, 700.0, t)) == []  # (each looked at within HUNT_STALE)
    assert [p for p, at in tc.hunt_mines.items() if at < 0] == []
    t += tm.HUNT_GAP
    out = _hunts(tc.commands(_rows(at=OWN_NATURAL), PLAYER, 800.0, t))  # (its main's mine last seen at 502 s: stale)
    assert {(c.x, c.y) for c in out} == {tuple(map(float, ENEMY_MAIN))}


def test_mines_from_the_first_look_only():
    """What becomes of a mine later (mined out, haunted: its row's type, its going) tells it nothing; two rows on one place
    (an entangled mine on its mine) are one place."""
    tc = _hunter()
    entangled = _unit(150, 1, ENEMY_NATURAL[0] + 5, ENEMY_NATURAL[1], "egol", S, vis=2)
    t = 100
    tc.commands(_rows(extra=[entangled]), PLAYER, 10.0, t)
    assert len(tc.hunt_mines) == 4
    first = dict(tc.hunt_mines)
    rows = np.stack([r for r in _rows() if r[fx.C_ID] != 103])  # (the natural's mine gone from the rows)
    tc.commands(rows, PLAYER, 20.0, t + 4)
    assert set(tc.hunt_mines) == set(first)


def test_episode_counts_a_base_found_near_a_taught_point():
    tc = _hunter()
    t = _beaten(tc)
    tc.hunt_contact = 0.0
    assert _hunts(tc.commands(_rows(), PLAYER, 500.0, t))
    # (a barracks there is no base; the hall is)
    bar = _unit(300, 1, ENEMY_NATURAL[0], ENEMY_NATURAL[1] + 500, "obar", S, vis=1)
    tc.commands(_rows(at=(4400, -2600), extra=[bar]), PLAYER, 540.0, t + 4)
    assert tc.hunt_found_at is None
    hall = _unit(301, 1, ENEMY_NATURAL[0], ENEMY_NATURAL[1] + 800, "ogre", S, vis=1)
    tc.commands(_rows(at=(4400, -2600), extra=[bar, hall]), PLAYER, 560.0, t + 8)
    assert tc.hunt_found_at == 560
    tc.commands(_rows(at=(4400, -2600), extra=[bar, hall]), PLAYER, 600.0, t + 12)
    assert tc.hunt_found_at == 560  # (the first time)


def test_no_lesson_no_base_found():
    tc = _hunter()
    tc.commands(_rows(extra=[_unit(300, 1, *ENEMY_NATURAL, "ogre", S, vis=1)]), PLAYER, 100.0, 100)
    assert tc.hunt_found_at is None and tc.hunt_lessons == 0


def test_label_rows_of_the_lesson():
    rows = label_rows(7, [PointOrder(30, ATTACK, 4400.0, -3200.0)])
    assert rows.tolist() == [[7, 30, ATTACK, 1, 4400, -3200, 0]]


def test_flag_default_is_off_and_every_row_field_with_it():
    import inspect

    from warcraftsim.fullgame import selfplay
    src = inspect.getsource(selfplay)
    assert 'add_argument("--hunt-lesson", type=int, default=0' in src
    assert 'if side in teachers and cfg.get("hunt_lesson") else {}' in src
