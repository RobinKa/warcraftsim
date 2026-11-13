"""--attack-lesson (teacher.Teacher._attacks): an attack on a unit where the engine's attack-move picks last: an enemy
worker in reach of the side's army with no enemy army unit near it, or a hurt enemy hero in reach (first); the nearest 3
army units labelled; only what the side sees, only when the mask allows it, looked at every ATTACK_LOOK steps."""

import random

import numpy as np
from test_spell_lesson import PLAYER, _unit, code

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import teacher as tm
from warcraftsim.fullgame.teacher import ATTACK, Teacher
from warcraftsim.protocol import TargetOrder

ARMY = [_unit(10 + k, 0, 100 * k, 0, "ogru") for k in range(5)]  # (5 grunts along x 0-400)


def _peon(i, x, y=0, owner=1, vis=3):
    return _unit(i, owner, x, y, "uaco", fx.WORKER, vis=vis, hp=220)


def _lessons(rows, t=100, share=1.0, ok=True, tc=None):
    if tc is None:
        tc = Teacher(0, "orc", random.Random(0), attacks=share)
    tc.order_ok = lambda oid, k: ok and (oid, k) == (ATTACK, fx.UNIT)
    out = [c for c in tc.commands(np.stack(rows), PLAYER, 300.0, t) if getattr(c, "order", None) == ATTACK
           and isinstance(c, TargetOrder)]
    return tc, out


def test_a_lone_worker_in_reach():
    tc, out = _lessons(ARMY + [_peon(90, 900), _peon(91, 2000)])
    assert {c.target for c in out} == {90}  # (the nearest worker in reach; 91 is out of reach)
    assert sorted(c.unit for c in out) == [12, 13, 14]  # (the 3 grunts nearest it)
    assert tc.taught == {"attack worker": 1}


def test_not_when_the_enemy_army_guards_it():
    guard = _unit(95, 1, 1100, 0, "ugho")  # (a ghoul 200 from the acolyte)
    assert _lessons(ARMY + [_peon(90, 900), guard])[1] == []
    far = _unit(95, 1, 2500, 0, "ugho")
    assert {c.target for c in _lessons(ARMY + [_peon(90, 900), far])[1]} == {90}


def test_a_hurt_hero_first():
    hero = _unit(97, 1, 700, 0, "Udea", fx.HERO, hp=1000)
    hero[fx.C_HP] = 300  # (30% of its hit points)
    out = _lessons(ARMY + [_peon(90, 500), hero])[1]
    assert {c.target for c in out} == {97}
    hero[fx.C_HP] = 600  # (60%: not hurt enough; the acolyte is guarded by the hero, an army unit)
    assert _lessons(ARMY + [_peon(90, 500), hero])[1] == []


def test_only_what_it_sees_and_the_mask_allows():
    assert _lessons(ARMY + [_peon(90, 900, vis=2)])[1] == []  # (seen by side 1 only)
    assert _lessons(ARMY + [_peon(90, 900)], ok=False)[1] == []
    assert _lessons(ARMY + [_unit(90, 1, 900, 0, "unpl", fx.STRUCTURE)])[1] == []  # (a building: not a worker)


def test_look_gap_share_and_off():
    tc, out = _lessons(ARMY + [_peon(90, 900)], t=100)
    assert out
    assert _lessons(ARMY + [_peon(90, 900)], t=100 + tm.ATTACK_LOOK - 1, tc=tc)[1] == []
    assert _lessons(ARMY + [_peon(90, 900)], t=100 + tm.ATTACK_LOOK, tc=tc)[1]
    assert _lessons(ARMY + [_peon(90, 900)], share=0.0)[1] == []
    off = Teacher(0, "orc", random.Random(0))
    assert off.attacks == 0
    off.order_ok = lambda oid, k: True
    assert not [c for c in off.commands(np.stack(ARMY + [_peon(90, 900)]), PLAYER, 300.0, 100) if isinstance(c, TargetOrder)
                and c.order == ATTACK]


def test_lesson_labels_attack_on_the_unit():
    """The lesson as the encoder's label (taught) at a grunt: attack (unit), its pointer the acolyte."""
    from warcraftsim.fullgame.selfplay import label_rows
    types = ("ogru", "uaco")
    vocab = {"types": [code(c) for c in types], "current_orders": [], "upgrades": [], "orders": [[ATTACK, fx.UNIT]]}
    enc = fx.Encoder(vocab, np.zeros((len(vocab["orders"]) + 1, 3), np.int64))
    view = enc.view(0, 1.0, [0, 1])
    rows = np.stack(ARMY + [_peon(90, 900)])
    me = np.asarray([0, 0, 1000, 500, 30, 60, 0])
    st = view.step(rows, me, np.zeros((0, 5), np.int64), 0)
    out = _lessons(list(rows))[1]
    y, _ = enc.step_labels(view, st, label_rows(0, out), me, np.zeros((0, 5), np.int64), {}, taught=True)
    g = st["index"][14]
    assert int(y["y_order"][g]) == 1 and int(y["y_ptr"][g]) == st["index"][90]


def test_episode_counts_attacks():
    from warcraftsim.fullgame.selfplay import attack_counts
    assert attack_counts({f"TargetOrder:{ATTACK}": (3, 1), f"PointOrder:{ATTACK}": (90, 0), "PointOrder:851986": (5, 0)}) \
        == {"unit": 3, "point": 90}
    assert attack_counts({"PointOrder:851986": (5, 0)}) == {}
