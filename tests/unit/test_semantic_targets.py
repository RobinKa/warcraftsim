from types import SimpleNamespace

import warcraftsim.env as env_mod
from warcraftsim.data.objects import CombatStats
from warcraftsim.env import SEMANTIC_TARGETS, MicroEnv
from warcraftsim.protocol import Unit, UnitFlags, fourcc

STATS = {
    "hfoo": CombatStats(range=90, cooldown=1.35, damage=12.5, armor=2, speed=270, hits_air=False, is_hero=False),
    "hrif": CombatStats(range=400, cooldown=1.5, damage=21, armor=0, speed=270, hits_air=True, is_hero=False),
    "Hpal": CombatStats(range=100, cooldown=2.2, damage=32, armor=4, speed=270, hits_air=False, is_hero=True),
}


def _unit(uid: int, kind: str, x: int, hp: int, owner: int = 1) -> Unit:
    return Unit(id=uid, type_id=fourcc(kind), owner=owner, x=x, y=0, facing=0, hp=hp, max_hp=500, mana=0,
                max_mana=0, order=0, flags=UnitFlags(0), visible_to=1, resource=0)


def _env(monkeypatch, enemy):
    monkeypatch.setattr(env_mod, "combat_stats", lambda: STATS)
    env = MicroEnv.__new__(MicroEnv)  # no game: only the targeting state
    env.targeting, env.move_distance, env._enemy, env._attacking = "semantic", 250.0, enemy, {}
    moves = []
    env.game = SimpleNamespace(move=lambda u, x, y: moves.append((u.id, round(x), round(y))))
    return env, moves


def test_semantic_targets_resolve_to_enemy_slots(monkeypatch):
    # the footman at x=0: a footman in reach (x=150, 300 hp), a weaker rifleman out of reach
    # (x=600, 100 hp), a paladin further away (x=900); slot 1 is dead
    enemy = [_unit(10, "hfoo", 150, 300), None, _unit(12, "hrif", 600, 100), _unit(13, "Hpal", 900, 400)]
    env, _ = _env(monkeypatch, enemy)
    me = _unit(1, "hfoo", 0, 500, owner=0)
    slot = {rule: env.target_slot(me, i) for i, rule in enumerate(SEMANTIC_TARGETS)}
    assert slot == {"weak_in_range": 0, "nearest": 0, "weakest": 2, "hero": 3, "threat": 0}
    rifle = _unit(2, "hrif", 250, 500, owner=0)  # reaches the footman and the rifleman
    assert env.target_slot(rifle, SEMANTIC_TARGETS.index("weak_in_range")) == 2
    assert env.target_slot(rifle, SEMANTIC_TARGETS.index("threat")) == 2  # 14 dps / 100 hp beats 9 / 300


def test_slot_targets_and_retreat(monkeypatch):
    enemy = [_unit(10, "hfoo", 150, 300), None]
    env, moves = _env(monkeypatch, enemy)
    me = _unit(1, "hfoo", 0, 500, owner=0)
    env._retreat(me)
    assert moves == [(1, -250, 0)]  # straight away from the nearest enemy
    env.targeting = "slot"
    assert env.target_slot(me, 0) == 0 and env.target_slot(me, 1) is None


def _hero(uid, kind, x, hp, mana, abilities, owner=0):
    u = _unit(uid, kind, x, hp, owner=owner)
    u.flags = UnitFlags.HERO
    u.max_mana, u.mana, u.abilities = 300, mana, abilities
    return u


def _abilities(monkeypatch):
    from warcraftsim.data import abilities as ab
    from warcraftsim.data.abilities import AbilityInfo

    def info(code, name, order, cast, side, mana, cd, rng, area):
        return AbilityInfo(code, name, order, cast, side, 3, 1, 0, (mana,) * 3, (cd,) * 3, (rng,) * 3, (area,) * 3)

    table = {"AHtb": info("AHtb", "Storm Bolt", "thunderbolt", "unit", "enemy", 75, 9, 600, 0),
             "AHtc": info("AHtc", "Thunder Clap", "thunderclap", "instant", "enemy", 90, 6, 0, 300),
             "AHhb": info("AHhb", "Holy Light", "holybolt", "unit", "ally", 65, 5, 800, 0),
             "AHbh": info("AHbh", "Bash", "bash", "passive", "", 0, 0, 0, 0)}
    monkeypatch.setattr(ab, "ability_info", lambda: table)
    monkeypatch.setattr(ab, "hero_abilities", lambda: {"Hmkg": ("AHtc", "AHtb", "AHbh", "AHav"),
                                                        "Hpal": ("AHhb", "AHds", "AHre", "AHad")})


def test_cast_commands(monkeypatch):
    from warcraftsim.protocol import ImmediateOrder, TargetOrder

    _abilities(monkeypatch)
    monkeypatch.setitem(STATS, "Hmkg", STATS["Hpal"])
    enemy = [_unit(10, "hfoo", 400, 300), None, _unit(12, "hrif", 650, 100), _unit(13, "hfoo", 900, 50)]
    env, _ = _env(monkeypatch, enemy)
    env.game.order_id = {"thunderbolt": 1, "thunderclap": 2, "holybolt": 3}.__getitem__
    weak = SEMANTIC_TARGETS.index("weak_in_range")
    # Storm Bolt (slot 1): the weakest enemy within its cast range (600 + reach), not the weakest overall
    mk = _hero(1, "Hmkg", 0, 500, 200, ((1, 0.0), (1, 0.0), (0, 0.0), (0, 0.0)))
    assert env.cast_command(mk, 1, weak, [mk], enemy) == TargetOrder(1, 1, 12)
    assert env.cast_command(mk, 0, weak, [mk], enemy) == ImmediateOrder(1, 2)  # Thunder Clap: instant
    assert env.cast_command(mk, 2, weak, [mk], enemy) is None  # not learned (and passive)
    cooling = _hero(1, "Hmkg", 0, 500, 200, ((1, 0.0), (1, 4.5), (0, 0.0), (0, 0.0)))
    assert env.cast_command(cooling, 1, weak, [cooling], enemy) is None
    drained = _hero(1, "Hmkg", 0, 500, 50, ((1, 0.0), (1, 0.0), (0, 0.0), (0, 0.0)))
    assert env.cast_command(drained, 1, weak, [drained], enemy) is None
    far = _hero(1, "Hmkg", -2000, 500, 200, ((1, 0.0), (1, 0.0), (0, 0.0), (0, 0.0)))
    assert env.cast_command(far, 1, weak, [far], enemy) is None  # nobody in cast range
    # the scripted caster: Thunder Clap only with an enemy within its area; Storm Bolt in range
    assert env.scripted_cast(mk, [mk], enemy) == (1, weak)
    near = [_unit(10, "hfoo", 200, 300)]
    assert env.scripted_cast(mk, [mk], near) == (0, weak)
    # Holy Light: the most hurt own unit in range (share of hit points), the hero included
    pal = _hero(2, "Hpal", 0, 480, 200, ((1, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)))
    own = [pal, _unit(3, "hfoo", 100, 250, owner=0), _unit(4, "hfoo", 1500, 20, owner=0)]
    assert env.cast_command(pal, 0, weak, own, enemy) == TargetOrder(2, 3, 3)
    assert env.scripted_cast(pal, own, enemy) == (0, weak)  # 250/500 < 70%
