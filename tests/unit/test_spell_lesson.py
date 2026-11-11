"""--spell-lesson (teacher.py): a hero's area spell, nova or immolation, taught when a person would cast it."""

import random
from types import SimpleNamespace

import numpy as np

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import obsfull
from warcraftsim.fullgame import teacher as tm
from warcraftsim.fullgame.teacher import Teacher
from warcraftsim.protocol import ImmediateOrder, PointOrder, TargetOrder

BLIZZARD, THUNDERCLAP, IMMOLATION, UNIMMOLATION, BREATH = 852089, 852096, 852177, 852178, 852580
MANA_BURN, SLEEP = 852179, 852227
PLAYER = SimpleNamespace(gold=1000, lumber=500, food_used=30, food_cap=60, start_x=-2000.0, start_y=0.0)


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


def _unit(i: int, owner: int, x: int, y: int, typ: str, flags: int = 0, vis: int = 3, hp: int = 400, mana: int = 0,
          abilities=()) -> np.ndarray:
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = i, code(typ), owner, x, y
    r[fx.C_HP], r[fx.C_MAXHP], r[fx.C_FLAGS], r[fx.C_VIS], r[fx.C_MANA], r[fx.C_MAXMANA] = hp, hp, flags, vis, mana, 500
    for k, (level, cooldown) in enumerate(abilities):  # (per slot: its level, its cooldown left in 0.1 s)
        r[fx.C_ABIL0 + 2 * k], r[fx.C_ABIL0 + 2 * k + 1] = level, cooldown
    return r


def _hero(typ: str, slot: int, level: int = 1, cooldown: int = 0, mana: int = 300, x: int = 0, y: int = 0):
    abilities = [(0, 0)] * 4
    abilities[slot] = (level, cooldown)
    return _unit(50, 0, x, y, typ, fx.HERO, hp=700, mana=mana, abilities=abilities)


def _enemies(points, typ: str = "hfoo", vis: int = 3, owner: int = 1):
    return [_unit(80 + k, owner, x, y, typ, vis=vis) for k, (x, y) in enumerate(points)]


CLUMP = [(600, 0), (650, 60), (620, -60)]  # (within 200 of each other, 600-650 from the hero)


def _lessons(rows, forms=((BLIZZARD, fx.UNIT),), t=100, share=1.0, buff=None, tc=None):
    """The spell lessons of side 0 at step t: `forms` the (order id, kind) the policy's vocabulary and mask allow,
    `buff` the hero's immolation (None: unknown)."""
    if tc is None:
        tc = Teacher(0, "human", random.Random(0), spells=share)
    tc.order_ok = lambda oid, k: (oid, k) in forms
    tc.buff_on = lambda uid, b: buff
    out = [c for c in tc.commands(np.stack(rows), PLAYER, 300.0, t) if getattr(c, "order", None) in
           (BLIZZARD, THUNDERCLAP, IMMOLATION, UNIMMOLATION, BREATH, MANA_BURN, SLEEP)]
    return tc, out


def test_area_spell_at_the_clump():
    rows = [_hero("Hamg", 0)] + _enemies(CLUMP)
    tc, out = _lessons(rows)  # (the vocabulary's blizzard is on a unit, as the built-in AI casts it)
    assert out == [TargetOrder(50, BLIZZARD, 80)] and tc.taught["spell AHbz"] == 1  # (the unit nearest the centre)
    _, out = _lessons(rows, forms=((BLIZZARD, fx.POINT), (BLIZZARD, fx.UNIT)))  # (a point form: the clump's centre)
    assert out == [PointOrder(50, BLIZZARD, 623, 0)]
    _, out = _lessons([_hero("Npbm", 0)] + _enemies([(400, 0), (450, 60), (420, -60)]), forms=((BREATH, fx.POINT),))
    assert out == [PointOrder(50, BREATH, 423, 0)]  # (breath of fire: range 375, the clump 400-450 away is in reach)
    assert _lessons([_hero("Hamg", 0)] + _enemies(CLUMP, typ="nftr", owner=12))[1] == [TargetOrder(50, BLIZZARD, 80)]
    assert _lessons([_hero("Hamg", 0)] + _enemies(CLUMP, owner=2))[1] == []  # (creeps yes; another player's not)


def test_area_spell_not_when_a_person_would_not():
    assert _lessons([_hero("Hamg", 0, cooldown=30)] + _enemies(CLUMP))[1] == []  # (cooling down)
    assert _lessons([_hero("Hamg", 0, mana=70)] + _enemies(CLUMP))[1] == []  # (75 mana)
    assert _lessons([_hero("Hamg", 0, level=0)] + _enemies(CLUMP))[1] == []  # (not learned)
    assert _lessons([_hero("Hamg", 0)] + _enemies(CLUMP[:2]))[1] == []  # (two units)
    assert _lessons([_hero("Hamg", 0)] + _enemies([(600, 0), (600, 450), (600, -450)]))[1] == []  # (spread out)
    assert _lessons([_hero("Hamg", 0)] + _enemies([(x + 900, y) for x, y in CLUMP]))[1] == []  # (out of reach)
    assert _lessons([_hero("Hamg", 0)] + _enemies(CLUMP, vis=2))[1] == []  # (not seen by side 0)
    assert _lessons([_hero("Hamg", 0)] + _enemies(CLUMP, typ="hspt"))[1] == []  # (spell breakers: spell immune)
    assert _lessons([_hero("Hamg", 0)] + _enemies(CLUMP), forms=())[1] == []  # (the mask: not now)
    channel = _hero("Hamg", 0)
    channel[fx.C_ORDER] = BLIZZARD
    assert _lessons([channel] + _enemies(CLUMP))[1] == []  # (channelling it already)


def test_nova_around_the_hero():
    forms = ((THUNDERCLAP, fx.IMMEDIATE),)
    near = [(100, 0), (-150, 50), (0, 200)]  # (thunder clap's 300)
    assert _lessons([_hero("Hmkg", 0)] + _enemies(near), forms=forms)[1] == [ImmediateOrder(50, THUNDERCLAP)]
    assert _lessons([_hero("Hmkg", 0)] + _enemies(near[:2]), forms=forms)[1] == []
    assert _lessons([_hero("Hmkg", 0)] + _enemies(CLUMP), forms=forms)[1] == []  # (600 away)


def test_on_the_nearest_enemy_hero():
    forms = ((MANA_BURN, fx.UNIT), (SLEEP, fx.UNIT))
    lich = _unit(90, 1, 350, 0, "Ulic", fx.HERO, mana=200)
    far = _unit(91, 1, 380, 300, "Udea", fx.HERO, mana=200)  # (480 away)
    dh = _hero("Edem", 0)  # (mana burn: its 300 and a step, in a fight: 3 enemy units within 900)
    assert _lessons([dh, lich, far] + _enemies(CLUMP), forms=forms)[1] == [TargetOrder(50, MANA_BURN, 90)]
    dry = lich.copy()
    dry[fx.C_MANA] = 20
    assert _lessons([dh, dry] + _enemies(CLUMP), forms=forms)[1] == []  # (no mana to burn)
    assert _lessons([dh] + _enemies(CLUMP), forms=forms)[1] == []  # (no hero: not on its army)
    assert _lessons([dh, far] + _enemies(CLUMP), forms=forms)[1] == []  # (out of reach)
    assert _lessons([dh, lich, far], forms=forms)[1] == []  # (two heroes, no fight)
    dread = _hero("Udre", 1)  # (sleep: 800 and a step; the enemy's dead hero no target)
    paladin = _unit(92, 1, 850, 0, "Hpal", fx.HERO, mana=200)
    assert _lessons([dread, paladin] + _enemies(CLUMP), forms=forms)[1] == [TargetOrder(50, SLEEP, 92)]
    paladin[fx.C_FLAGS] |= fx.DEAD
    assert _lessons([dread, paladin] + _enemies(CLUMP), forms=forms)[1] == []


def test_immolation_on_in_a_fight_off_after():
    forms = ((IMMOLATION, fx.IMMEDIATE), (UNIMMOLATION, fx.IMMEDIATE))
    dh = _hero("Edem", 1)
    fight = _enemies([(150, 0), (0, 200)])
    assert _lessons([dh] + fight, forms=forms, buff=False)[1] == [ImmediateOrder(50, IMMOLATION)]
    assert _lessons([dh] + fight, forms=forms, buff=True)[1] == []  # (on already)
    assert _lessons([dh] + fight, forms=forms, buff=None)[1] == []  # (its state unknown: no obs-full block)
    assert _lessons([_hero("Edem", 1, mana=20)] + fight, forms=forms, buff=False)[1] == []  # (25 mana)
    assert _lessons([dh] + _enemies([(150, 0)]), forms=forms, buff=False)[1] == []  # (one unit)
    assert _lessons([dh] + _enemies([(1200, 0)]), forms=forms, buff=True)[1] == [ImmediateOrder(50, UNIMMOLATION)]
    assert _lessons([dh] + _enemies([(700, 0)]), forms=forms, buff=True)[1] == []  # (an enemy within 900: not yet)


def test_gap_share_and_off():
    rows = [_hero("Hamg", 0)] + _enemies(CLUMP)
    tc, out = _lessons(rows, t=100)
    assert len(out) == 1
    assert _lessons(rows, t=101, tc=tc)[1] == []  # (one look a hero's spell every SPELL_GAP steps)
    assert len(_lessons(rows, t=100 + tm.SPELL_GAP, tc=tc)[1]) == 1
    tc, out = _lessons(rows, share=1e-9)  # (looked at, given with the chance: none, and no look again before the gap)
    assert out == [] and "spell AHbz" not in tc.taught and tc.last["spell 50 AHbz"] == 100
    off = Teacher(0, "human", random.Random(0))  # (off by default)
    assert off.spells == 0.0 and _lessons(rows, tc=off)[1] == []


def test_buff_on_reads_the_obs_full_block(monkeypatch):
    monkeypatch.setattr(fx, "OBS_FULL", 1)
    monkeypatch.setattr(fx, "A_OBS", 7)
    row = np.zeros(7 + obsfull.OBS_F, np.float32)
    a = 7 + obsfull.K_BUFF
    imm, other = code("BEim"), code("BHbz")
    assert obsfull.buff_on(row, imm) is None  # (no event 101 for it yet)
    row[7 + obsfull.K_HAVE] = 1.0
    assert obsfull.buff_on(row, imm) is False
    row[a:a + obsfull.BUFF_D] = obsfull.code_of(other, obsfull.BUFF_D)
    assert obsfull.buff_on(row, imm) is False
    row[a:a + obsfull.BUFF_D] += obsfull.code_of(imm, obsfull.BUFF_D)
    assert obsfull.buff_on(row, imm) is True
    monkeypatch.setattr(fx, "OBS_FULL", 0)
    assert obsfull.buff_on(row, imm) is None


def test_lesson_labels_the_hero():
    """The lesson as the encoder's label (taught: never given) at the hero, its target the clump's unit."""
    from warcraftsim.fullgame.selfplay import label_rows
    types = ("Hamg", "hfoo")
    vocab = {"types": [code(c) for c in types], "current_orders": [], "upgrades": [],
             "orders": [[BLIZZARD, fx.UNIT], [BREATH, fx.POINT]]}
    enc = fx.Encoder(vocab, np.zeros((len(vocab["orders"]) + 1, 3), np.int64))
    view = enc.view(0, 1.0, [0, 1])
    rows = np.stack([_hero("Hamg", 0)] + _enemies(CLUMP))
    me = np.asarray([0, 0, 1000, 500, 30, 60, 0])
    st = view.step(rows, me, np.zeros((0, 5), np.int64), 0)
    out = _lessons(list(rows))[1]
    y, _ = enc.step_labels(view, st, label_rows(0, out), me, np.zeros((0, 5), np.int64), {}, taught=True)
    h = st["index"][50]
    assert int(y["y_order"][h]) == 1 and int(y["y_ptr"][h]) == st["index"][80]


def test_episode_counts_the_learners_casts():
    from warcraftsim.fullgame.selfplay import lesson_casts
    results = {f"TargetOrder:{BLIZZARD}": (3, 1), f"ImmediateOrder:{IMMOLATION}": (2, 0), f"PointOrder:{BREATH}": (1, 0),
               "TargetOrder:852222": (9, 0), "ImmediateOrder:hfoo": (4, 0)}  # (death coil and a footman: not taught)
    assert lesson_casts(results) == {"blizzard": 3, "immolation": 2, "breathoffire": 1}
