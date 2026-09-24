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
