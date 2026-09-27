import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import warcraftsim.env as env_mod
from warcraftsim.env import GENERAL_KINDS, MicroEnv
from warcraftsim.protocol import Unit, UnitFlags, fourcc

K = {k: i for i, k in enumerate(GENERAL_KINDS)}


def _unit(uid, x, hp, owner):
    return Unit(id=uid, type_id=fourcc("hfoo"), owner=owner, x=x, y=0, facing=0, hp=hp, max_hp=500, mana=0,
                max_mana=0, order=0, flags=UnitFlags(0), visible_to=1, resource=0)


def _env():
    env = MicroEnv.__new__(MicroEnv)  # no game: the order logic only
    env.targeting, env.max_own, env.max_enemy, env.group, env.abilities = "general", 2, 2, 5, False
    env.opponent_casts, env._attacking = False, {}
    env._own = [_unit(1, 0, 400, 0), None]
    env._enemy = [None, _unit(11, 300, 200, 1)]
    calls = []
    env.game = SimpleNamespace(
        attack=lambda u, t: (calls.append(("attack", u.id, t.id)), setattr(u, "order", 99)),  # now attacking
        move=lambda u, x, y: calls.append(("move", u.id, round(x), round(y))),
        attack_move=lambda u, x, y: calls.append(("attack_move", u.id, round(x), round(y))),
        stop=lambda u: calls.append(("stop", u.id)), hold=lambda u: calls.append(("hold", u.id)),
        order_id=lambda name: 99)
    env.action_space = SimpleNamespace(nvec=np.tile([7, 16, 3, 4, 4], (2, 1)))
    return env, calls


def test_general_orders_are_plain_game_orders():
    env, calls = _env()
    act = lambda *row: np.array([row, (0, 0, 0, 0, 0)])  # noqa: E731
    env._commands(act(K["attack"], 0, 0, 3, 0))  # target 3 = enemy slot 1
    env._commands(act(K["attack"], 0, 0, 3, 0))  # the same attack again: not re-issued
    env._commands(act(K["move"], 8, 1, 0, 0))    # direction 8 of 16 = west, distance 350
    env._commands(act(K["attack_move"], 0, 0, 0, 0))
    env._commands(act(K["hold"], 0, 0, 0, 0))
    env._commands(act(K["attack"], 0, 0, 0, 0))  # an own slot: no attack
    assert calls == [("attack", 1, 11), ("move", 1, -350, 0), ("attack_move", 1, 150, 0), ("hold", 1)]


def test_general_mask_only_rules_out_the_impossible():
    from warcraftsim.puffer.tasks import _micro_mask

    env, _ = _env()
    (m,) = _micro_mask(env)
    per = 7 + 16 + 3 + 4 + 4
    assert m[:7].tolist() == [1, 1, 1, 1, 1, 1, 0]           # everything but cast (no abilities)
    assert m[7:7 + 16].all() and m[23:26].all()
    assert m[26:30].tolist() == [1, 0, 0, 1]                  # alive units: own slot 0, enemy slot 1
    assert m[per:per + 7].tolist() == [1, 0, 0, 0, 0, 0, 0]   # an empty own slot: noop only
    env._enemy = [None, None]
    (m,) = _micro_mask(env)
    assert m[K["attack"]] == 0                                # nothing left to attack


def test_league_scripts_and_pfsp():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "warcraftsim" / "rl"))
    from league import KINDS, League, Scripts, pfsp_weight

    from warcraftsim.puffer.tasks import _micro_layout, _micro_sizes

    lay = _micro_layout(5, general=True)  # the layout of mirror_mix_gen (whose Task needs the game's data)
    obs_size, act = _micro_sizes(5, general=True)
    spec = {"obs_size": obs_size, "spaces": {
        "observation": {"blocks": [{"name": n, "rows": r, "features": list(f)} for n, r, f in lay["obs_layout"]]},
        "actions": {"heads": [{"name": n, "size": z} for n, z in zip(lay["head_names"], act[:5])]}}}
    s = Scripts(spec)
    k, F = s.k, s.F
    obs = np.zeros((1, spec["obs_size"]), np.float32)
    own = obs[0, :k * F].reshape(k, F)
    enemy = obs[0, k * F + k:2 * k * F + k].reshape(k, F)
    obs[0, k * F:k * F + 3] = 1
    obs[0, 2 * k * F + k:2 * k * F + k + 3] = 1
    for i, (hp, lost) in enumerate([(0.9, 0.0), (0.2, -0.1), (0.3, 0.0)]):
        own[i, s.i_x], own[i, s.i_hp], own[i, s.i_maxhp], own[i, s.i_lost] = -0.2, hp, 0.5, lost
    for j, hp in enumerate([0.8, 0.4, 0.9]):
        enemy[j, s.i_x], enemy[j, s.i_hp], enemy[j, s.i_maxhp] = 0.2, hp, 0.5
    focus = s.act("focus", obs).reshape(k, -1)
    assert [KINDS[r[0]] for r in focus[:3]] == ["attack"] * 3 and set(focus[:3, 3]) == {k + 1}  # the weakest
    pull = s.act("pull35", obs).reshape(k, -1)
    assert KINDS[pull[1, 0]] == "move" and pull[1, 1] == 8   # hurt and losing hit points: away (west)
    assert KINDS[pull[2, 0]] == "attack"                     # hurt but not being hit: keeps fighting
    assert pfsp_weight(0.9) < pfsp_weight(0.5) < pfsp_weight(0.1) and pfsp_weight(None) == 1.0


    # with abilities: "cast<script>" casts like the game's scripted opponent
    lay = _micro_layout(5, abilities=True, general=True)
    obs_size, act = _micro_sizes(5, abilities=True, general=True)
    spec = {"obs_size": obs_size, "spaces": {
        "observation": {"blocks": [{"name": n, "rows": r, "features": list(f)} for n, r, f in lay["obs_layout"]]},
        "actions": {"heads": [{"name": n, "size": z} for n, z in zip(lay["head_names"], act[:5])]}}}
    s = Scripts(spec)
    k, F = s.k, s.F
    feat = spec["spaces"]["observation"]["blocks"][0]["features"]
    obs = np.zeros((1, obs_size), np.float32)
    own = obs[0, :k * F].reshape(k, F)
    enemy = obs[0, k * F + k:2 * k * F + k].reshape(k, F)
    obs[0, k * F:k * F + 2] = 1
    obs[0, 2 * k * F + k:2 * k * F + k + 2] = 1
    for i in range(2):
        own[i, s.i_x], own[i, s.i_hp], own[i, s.i_maxhp] = -0.2, 0.9, 0.5
    for j, hp in enumerate([0.8, 0.4]):
        enemy[j, s.i_x], enemy[j, s.i_hp], enemy[j, s.i_maxhp] = 0.2, hp, 0.5  # 600 apart
    f = lambda name: feat.index(name)  # noqa: E731
    own[0, f("ability 2: ready to cast")] = 1  # a bolt on enemies, range 600
    own[0, f("ability 2: cast on a unit")] = own[0, f("ability 2: for enemies")] = 1
    own[0, f("ability 2: cast range (/1000)")] = 0.6
    a = s.act("castamove", obs).reshape(k, -1)
    assert KINDS[a[0, 0]] == "cast" and a[0, 4] == 1 and a[0, 3] == k + 1  # the weakest enemy in range
    assert KINDS[a[1, 0]] != "cast"
    own[0, f("ability 2: cast range (/1000)")] = 0.3  # out of range
    assert KINDS[s.act("castamove", obs).reshape(k, -1)[0, 0]] != "cast"
    league = League(Path("/tmp"), ["noop"])
    a, b = league.add_snapshot("a.pt", 0), league.add_snapshot("b.pt", 10)
    for _ in range(20):
        a.record(1.0)  # the learner always beats a
    picks = [league.sample_past().name for _ in range(200)]
    assert picks.count("past:10") > 150
