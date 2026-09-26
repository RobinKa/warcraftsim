"""RL tasks served to the PufferLib 5.0 trainer.

A task wraps a warcraftsim environment with fixed-size float observations and a flat
multi-discrete action vector per agent (PufferLib's native format). Sizes are compile-time
constants on the C side, so the task also generates the C header (see build.py).

Single-agent tasks wrap a Gymnasium env; self-play tasks (num_agents=2) wrap MicroSelfPlayEnv and
serve both sides of every game to the trainer (both controlled by the policy being trained, or
by historical checkpoints when PufferLib's self-play pool is enabled).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from ..env import (ABILITY_FEATURE_NAMES, ABILITY_FEATURES, GENERAL_DIRECTIONS, GENERAL_DISTANCES, GENERAL_KINDS,
                   RELATIONAL_FEATURE_NAMES, RELATIONAL_FEATURES, SEMANTIC_TARGETS, UNIT_FEATURE_NAMES, UNIT_FEATURES,
                   MicroEnv, MicroSelfPlayEnv, MirrorSelfPlayEnv, NavigateEnv)
from ..protocol import HERO_ABILITY_SLOTS
from ..scenario import Scenario


@dataclass
class Task:
    name: str
    obs_size: int
    act_sizes: tuple[int, ...]
    make_env: Callable[[str], Any]  # instance name -> env
    flatten: Callable[[Any], np.ndarray]  # one agent's env observation -> float32[obs_size]
    to_action: Callable[[np.ndarray], Any]  # float32[num_atns] -> one agent's env action
    outcome: Callable[[Any, dict], float] = lambda env, info: 0.0  # agent 0: +1 win, -1 loss, 0 other
    description: str = ""
    scenario: Scenario | None = None
    num_agents: int = 1
    # PufferLib 5 uses raw advantages: small dense rewards are scaled up so they are not drowned
    # out by the entropy bonus (episode returns in the logs are scaled too)
    reward_scale: float = 1.0
    # for the replay overlay: value names per action head, heads per controlled unit (one row
    # each), and which head details a choice of the unit's first head (e.g. move -> direction)
    head_labels: tuple[tuple[str, ...], ...] = ()
    group_size: int = 1
    detail_heads: dict[int, int] = field(default_factory=dict)
    # counts per step (env before the step, actions of all agents), summed over an episode
    action_stats: Callable[[Any, list[np.ndarray]], Counter] | None = None
    # env -> per agent, one byte per option of every action head (0: not possible now); PufferLib
    # samples and trains with them (None: everything allowed)
    action_mask: Callable[[Any], list[np.ndarray]] | None = None
    # train.py settings found by sweeps for this task (used where the command line does not set
    # them): its option names (horizon, lr, minibatch, ...), "extra": PufferLib --section.key=value
    train_defaults: dict[str, Any] = field(default_factory=dict)
    # for people (describe_spaces, the dashboard): the observation's blocks (name, rows, the
    # feature names of a row) in flattened order, the names of a unit's action heads, what the
    # action masks allow, and the reward
    obs_layout: tuple[tuple[str, int, tuple[str, ...]], ...] = ()
    head_names: tuple[str, ...] = ()
    mask_info: str = ""
    reward_info: str = ""

    @property
    def num_atns(self) -> int:
        return len(self.act_sizes)

    # ---- uniform multi-agent view used by the bridge -------------------------------------

    def reset(self, env, options: dict | None = None) -> tuple[list[np.ndarray], dict]:
        obs, info = env.reset(options=options)
        info = info[0] if self.num_agents > 1 else info
        return self._obs_list(obs), info

    def step(self, env, actions: list[np.ndarray]):
        """-> (obs per agent, reward per agent, done, info, outcome per agent)"""
        if self.num_agents == 1:
            obs, reward, terminated, truncated, info = env.step(self.to_action(actions[0]))
            done = terminated or truncated
            outcomes = [self.outcome(env, info) if done else 0.0]
            return [self.flatten(obs)], [float(reward) * self.reward_scale], done, info, outcomes
        obs, rewards, terminated, truncated, infos = env.step({a: self.to_action(actions[a])
                                                               for a in range(self.num_agents)})
        done = any(terminated.values()) or any(truncated.values())
        info = infos[0]
        outcomes = [0.0] * self.num_agents
        if done:
            o = info["obs"]
            for a in range(self.num_agents):
                r = o.players[a].result.name
                outcomes[a] = 1.0 if r == "VICTORY" else -1.0 if r == "DEFEAT" else 0.0
        return (self._obs_list(obs), [float(rewards[a]) * self.reward_scale for a in range(self.num_agents)], done,
                info, outcomes)

    def _obs_list(self, obs) -> list[np.ndarray]:
        if self.num_agents == 1:
            return [self.flatten(obs)]
        return [self.flatten(obs[a]) for a in range(self.num_agents)]


# ---- navigate: one footman must reach a point --------------------------------------------------

_DIRS = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")


def _nav_task(distance: float = 1200.0) -> Task:
    sc = Scenario.move_to_target("hfoo", distance=distance, max_game_seconds=30)

    def outcome(env, info) -> float:
        return 1.0 if info.get("distance", 1e9) <= sc.target[2] else -1.0

    return Task(
        name="nav", obs_size=6, act_sizes=(9,),
        make_env=lambda name: NavigateEnv(sc, name=name),
        flatten=lambda obs: np.asarray(obs, dtype=np.float32),
        to_action=lambda a: int(a[0]),
        outcome=outcome, scenario=sc, head_labels=(("stop", *_DIRS),),
        description="Move a footman 1200 units to a target (stop or 8 directions); reward = progress.",
        obs_layout=(("the footman and its target", 1, ("target dx (/1000)", "target dy (/1000)", "facing sin",
                                                        "facing cos", "moving", "time (share of the limit)")),),
        head_names=("move",), reward_info="Per step: the distance gained toward the target (/1000); "
                                          "success within the target's radius.",
    )


# ---- micro: small fights ------------------------------------------------------------------------

_FEAT = UNIT_FEATURES


def _micro_sizes(max_units: int, targets: int | None = None, abilities: bool = False,
                 relational: bool = False, general: bool = False) -> tuple[int, tuple[int, ...]]:
    k = max_units
    feat = _FEAT + (ABILITY_FEATURES * HERO_ABILITY_SLOTS if abilities else 0) + (RELATIONAL_FEATURES if relational else 0)
    heads = (5, 8, targets or k, HERO_ABILITY_SLOTS) if abilities else (4, 8, targets or k)
    if general:
        heads = (len(GENERAL_KINDS), GENERAL_DIRECTIONS, len(GENERAL_DISTANCES), 2 * k, HERO_ABILITY_SLOTS)
    return k * feat + k + k * feat + k + 1, heads * k


def _micro_layout(max_units: int, abilities: bool = False, relational: bool = False, general: bool = False) -> dict:
    """obs_layout and head_names of the micro tasks (the order of _micro_flatten)."""
    feat = (*UNIT_FEATURE_NAMES,
            *(f"ability {k + 1}: {n}" for k in range(HERO_ABILITY_SLOTS if abilities else 0) for n in ABILITY_FEATURE_NAMES),
            *(RELATIONAL_FEATURE_NAMES if relational else ()))
    return dict(obs_layout=(("own units (slots A0..)", max_units, feat), ("own slot alive", max_units, ("alive",)),
                            ("enemy units (slots E0..)", max_units, feat), ("enemy slot alive", max_units, ("alive",)),
                            ("time", 1, ("episode time (share of the limit)",))),
                head_names=("order", "direction", "distance", "target", "ability") if general else
                ("order", "direction", "target", "ability")[:4 if abilities else 3])


_MICRO_REWARD = ("Per step: (enemy hit points lost − own hit points lost) / the sides' initial hit points; "
                 "+1 win, −1 loss at the end.")


def _micro_masks(targeting: str = "slot", abilities: bool = False, tactical: bool = False,
                 rejoin: bool = False) -> str:
    if targeting == "general":
        return ("orders for empty slots (noop only), attacks with no enemy left, casts without a ready ability and "
                "targets in empty slots are masked; nothing else (out of range: the unit walks there first).")
    rules = ["attacks on empty enemy slots are masked" if targeting == "slot" else
             "targets are rules (always possible; resolved when ordered)"]
    if abilities:
        rules.append("cast only for a hero that can cast something now, and only its castable ability slots")
    if tactical:
        rules.append(f"no plain moves; retreat only below {TACTICAL_RETREAT_HP:.0%} hit points while losing them; "
                     "during a committed retreat only noop (it carries on, up to 3 s while still losing hit points)")
    if rejoin:
        rules.append("after a pull-back, noop attack-moves the unit back into the fight")
    return "; ".join(rules) + "."


def _micro_flatten(obs: dict) -> np.ndarray:
    return np.concatenate([obs["own"].ravel(), obs["own_mask"].astype(np.float32), obs["enemy"].ravel(),
                           obs["enemy_mask"].astype(np.float32), obs["time"]]).astype(np.float32)


_KINDS = ("noop", "stop", "move", "attack")
_SEMANTIC_KINDS = ("noop", "retreat", "move", "attack")
_ABILITY_SLOTS = tuple(f"ability {k + 1}" for k in range(HERO_ABILITY_SLOTS))


_DIRS16 = tuple(f"{round(k * 360 / GENERAL_DIRECTIONS)}°" for k in range(GENERAL_DIRECTIONS))


def _micro_labels(max_units: int, semantic: bool = False, abilities: bool = False, general: bool = False) -> dict:
    if general:  # move / attack-move: direction and distance; attack: target; cast: ability and target
        targets = tuple(f"A{i}" for i in range(max_units)) + tuple(f"E{i}" for i in range(max_units))
        return dict(head_labels=(GENERAL_KINDS, _DIRS16, tuple(f"{d:.0f}" for d in GENERAL_DISTANCES), targets,
                                 _ABILITY_SLOTS) * max_units,
                    group_size=5, detail_heads={3: (1, 2), 4: 3, 5: (1, 2), 6: (4, 3)},
                    action_stats=_micro_action_stats, action_mask=_micro_mask)
    targets = SEMANTIC_TARGETS if semantic else tuple(f"E{i}" for i in range(max_units))
    if abilities:  # cast (kind 4) is detailed by the ability slot and the target rule
        return dict(head_labels=(_SEMANTIC_KINDS + ("cast",), _DIRS, targets, _ABILITY_SLOTS) * max_units,
                    group_size=4, detail_heads={2: 1, 3: 2, 4: (3, 2)}, action_stats=_micro_action_stats,
                    action_mask=_micro_mask)
    return dict(head_labels=(_SEMANTIC_KINDS if semantic else _KINDS, _DIRS, targets) * max_units,
                group_size=3, detail_heads={2: 1, 3: 2}, action_stats=_micro_action_stats, action_mask=_micro_mask)


# tactical masks: a retreat is possible below this share of hit points (while losing them); at any
# hit points, random retreats of healthy units cost more than the useful ones teach
TACTICAL_RETREAT_HP = 0.5


def _general_mask(env) -> np.ndarray:
    """General orders: only what is impossible is masked: orders for empty slots (noop only),
    attacks with no enemy left, casts without a ready ability, targets in empty slots."""
    heads = [int(n) for n in env.action_space.nvec[0]]
    per, offs = sum(heads), np.cumsum([0, *heads])
    k, e = env.max_own, env.max_enemy
    own = [i < len(env._own) and env._own[i] is not None for i in range(k)]
    enemy = [j < len(env._enemy) and env._enemy[j] is not None for j in range(e)]
    m = np.zeros(k * per, np.uint8)
    for i in range(k):
        base = i * per
        m[base + offs[:-1]] = 1  # the first option of every head: some option stays possible
        u = env._own[i] if own[i] else None
        if u is None:
            continue
        ready = env.ready_abilities(u)
        m[base:base + heads[0]] = [1, 1, 1, 1, any(enemy), 1, any(ready)]
        m[base + offs[1]:base + offs[3]] = 1
        m[base + offs[3]:base + offs[4]] = own + enemy
        if any(ready):
            m[base + offs[4]:base + offs[5]] = ready
    return m


def _micro_mask(env) -> list[np.ndarray]:
    """MicroEnv action masks: attacks on empty enemy slots (slot targeting); casts only by units
    with an ability they can cast now, and only those ability slots."""
    if env.targeting == "general":
        return [_general_mask(env)]
    heads = [int(n) for n in env.action_space.nvec[0]]
    per = sum(heads)
    offs = np.cumsum([0, *heads])
    m = np.ones(env.max_own * per, np.uint8)
    live = [e is not None for e in env._enemy[:env.max_enemy]]
    live += [False] * (env.max_enemy - len(live))
    weak = SEMANTIC_TARGETS.index("weak_in_range")
    for i, u in enumerate(env._own[:env.max_own]):
        if u is None:  # an empty slot's actions are ignored
            continue
        base = i * per
        if env.targeting == "slot" and any(live):
            m[base + offs[2]:base + offs[3]] = live
        if env.abilities:
            ok = [u.is_hero and env.cast_command(u, k, weak, env._own, env._enemy) is not None
                  for k in range(heads[3])]
            if any(ok):
                m[base + offs[3]:base + offs[4]] = ok
            else:
                m[base + 4] = 0  # kind "cast"
        if getattr(env, "tactical", False) and env.retreating(u):
            m[base:base + heads[0]] = 0  # a committed retreat goes on: only noop (= continue)
            m[base] = 1
            continue
        if getattr(env, "tactical", False):
            m[base + 2] = 0  # no plain moves
            hist = env.encoder._hp.get(u.id) if env.encoder is not None else None
            if not hist or u.hp >= hist[0] or u.hp > TACTICAL_RETREAT_HP * u.max_hp:
                m[base + 1] = 0  # retreat: only while losing hit points (over HP_HISTORY steps) and hurt
    return [m]


def _micro_action_stats(env, actions: list[np.ndarray]) -> Counter:
    """Per live unit and step: the order kind; for attacks, the target (the enemy slot a semantic
    target resolves to)."""
    c: Counter = Counter()
    kinds = getattr(env, "kind_names", _KINDS)
    for agent, action in enumerate(actions):
        side = env.sides[agent] if getattr(env, "sides", None) else env  # MirrorSelfPlayEnv: per player
        own, enemy = env._units[agent] if hasattr(env, "_units") else (side._own, side._enemy)
        resolve = None if hasattr(env, "_units") else getattr(side, "target_slot", None)
        a = np.asarray(action, int).reshape(-1, getattr(env, "group", 3))
        general = getattr(side, "targeting", "") == "general"
        attack = kinds.index("attack")
        targets = []
        for i in range(min(len(own), len(a))):
            if own[i] is None:
                continue
            kind, target = a[i][0], a[i][3 if general else 2]
            c["unit_steps"] += 1
            c[kinds[kind]] += 1
            if kinds[kind] == "cast":
                cmd = (side.general_cast(own[i], int(a[i][4]), int(target)) if general else
                       side.cast_command(own[i], int(a[i][3]), int(target), own, enemy))
                if cmd is None:
                    c["cast_invalid"] += 1  # not learned, cooling down, no mana or no target: nothing happens
            if kind == attack and general:
                j = int(target) - side.max_own
                if 0 <= j < len(enemy) and enemy[j] is not None:
                    targets.append(j)
                else:
                    c["attack_invalid"] += 1
            elif kind == attack:
                if resolve is not None:
                    slot = resolve(own[i], int(target))
                else:
                    slot = int(target) if target < len(enemy) and enemy[target] is not None else None
                if slot is not None:
                    targets.append(slot)
                else:
                    c["attack_invalid"] += 1
        live = [j for j in range(len(enemy)) if enemy[j] is not None]
        if targets and live:
            weakest = min(live, key=lambda j: enemy[j].hp)
            c["attack_valid"] += len(targets)
            c["attack_weakest"] += sum(t == weakest for t in targets)
            if len(targets) >= 2:
                c["attack_grouped"] += len(targets)
                c["attack_focus"] += Counter(targets).most_common(1)[0][1]
    return c


def action_summary(c: Counter) -> dict[str, float]:
    """Episode action statistics as fractions (for the episode log and dashboard)."""
    out: dict[str, float] = {}
    if c.get("unit_steps"):
        for k in dict.fromkeys(_KINDS + _SEMANTIC_KINDS + ("cast",)):
            if k in _KINDS or k in c:
                out[k] = c.get(k, 0) / c["unit_steps"]
    if c.get("cast"):
        out["cast_invalid"] = c.get("cast_invalid", 0) / c["cast"]
    if c.get("attack"):
        out["attack_invalid"] = c.get("attack_invalid", 0) / c["attack"]
    if c.get("attack_valid"):
        out["attack_weakest"] = c.get("attack_weakest", 0) / c["attack_valid"]
    if c.get("attack_grouped"):
        out["focus_fire"] = c.get("attack_focus", 0) / c["attack_grouped"]
    return {k: round(v, 4) for k, v in out.items()}


def _micro_outcome(env, info) -> float:
    o = info.get("obs")
    if o is None or not o.game_over:
        return 0.0
    r = o.players[0].result
    return 1.0 if r.name == "VICTORY" else -1.0 if r.name == "DEFEAT" else 0.0


def _micro_task(own: tuple[str, ...] = ("hfoo",) * 4, enemy: tuple[str, ...] = ("ogru",) * 3,
                max_units: int = 6, name: str = "micro", max_hp: int = 0, max_game_seconds: float = 90,
                enemy_max_hp: int | None = None) -> Task:
    sc = Scenario.skirmish(list(own), list(enemy), max_game_seconds=max_game_seconds, max_hp=max_hp,
                           enemy_max_hp=enemy_max_hp)
    obs_size, act_sizes = _micro_sizes(max_units)
    return Task(
        name=name, obs_size=obs_size, act_sizes=act_sizes,
        make_env=lambda inst: MicroEnv(sc, max_own=max_units, max_enemy=max_units, name=inst),
        flatten=_micro_flatten, to_action=lambda a: np.asarray(a, dtype=np.int64).reshape(max_units, 3),
        outcome=_micro_outcome, scenario=sc, reward_scale=10.0, **_micro_labels(max_units),
        description=f"{len(own)} {own[0]} vs {len(enemy)} {enemy[0]} (scripted); per unit: noop/stop/move/attack.",
        **_micro_layout(max_units), mask_info=_micro_masks(), reward_info=_MICRO_REWARD,
    )


def _selfplay_task(units: tuple[str, ...] = ("hfoo",) * 4, max_units: int = 6,
                   name: str = "selfplay_micro") -> Task:
    sc = Scenario.skirmish(list(units), list(units), max_game_seconds=90)
    obs_size, act_sizes = _micro_sizes(max_units)
    return Task(
        name=name, obs_size=obs_size, act_sizes=act_sizes, num_agents=2,
        make_env=lambda inst: MicroSelfPlayEnv(sc, max_units=max_units, name=inst),
        flatten=_micro_flatten, to_action=lambda a: np.asarray(a, dtype=np.int64).reshape(max_units, 3),
        scenario=sc, reward_scale=10.0, **{**_micro_labels(max_units), "action_mask": None},  # masks: MicroEnv only
        description=f"Self-play: {len(units)} {units[0]} vs {len(units)} {units[0]}, both sides are agents.",
        **_micro_layout(max_units), mask_info="None.", reward_info=_MICRO_REWARD + " Zero-sum between the sides.",
    )


TASKS: dict[str, Callable[[], Task]] = {
    "nav": _nav_task,
    "micro": _micro_task,
    "micro_mirror": lambda: _micro_task(("hfoo",) * 4, ("hfoo",) * 4, name="micro_mirror"),
    # the smallest fight worth learning: pull a damaged footman back so the enemies switch targets
    "footmen2": lambda: replace(_micro_task(("hfoo",) * 2, ("hfoo",) * 2, max_units=2, name="footmen2", max_hp=100,
                                            max_game_seconds=40),
                                # sweeps f2-*: 95% wins after ~0.2M steps
                                train_defaults=dict(step_seconds=0.5, lr=0.01, minibatch=192, replay_ratio=4.0)),
    "selfplay_micro": _selfplay_task,
    "mirror_mix": lambda: _mirror_task(),  # defined below
}


# ---- random mirror matches: a hero and some units, the same for both sides, new every episode --

MIRROR_UNITS = ("hfoo", "hrif", "hkni", "ogru", "ohun", "otau", "ugho", "ucry", "uabo", "earc", "esen")
MIRROR_HEROES = ("Hpal", "Hamg", "Hmkg", "Hblm", "Obla", "Ofar", "Otch", "Oshd", "Udea", "Ulic", "Udre", "Ucrl",
                 "Ekee", "Emoo", "Edem", "Ewar")


def skill_build(hero: str, level: int, rng) -> tuple[str, ...]:
    """A random way to spend a hero's `level` skill points: one ability level per point among its
    abilities that fight (data.abilities: no summons or utility), within the hero level rules
    (ability level k needs hero level 1 + 2(k-1); ultimates 6)."""
    from ..data.abilities import ability_info, hero_abilities

    info = ability_info()
    learned: Counter = Counter()
    out = []
    for _ in range(level):
        options = [c for c in hero_abilities().get(hero, ()) if (a := info.get(c)) is not None and a.in_builds
                   and learned[c] < a.levels and a.hero_level_for(learned[c] + 1) <= level]
        if not options:
            break
        code = options[int(rng.integers(len(options)))]
        learned[code] += 1
        out.append(code)
    return tuple(out)


def mirror_spawner(units: tuple[int, int] = (2, 4), heroes: int = 1, hero_levels: tuple[int, int] = (1, 3),
                   hp_permille: int = 250, pool=MIRROR_UNITS, hero_pool=MIRROR_HEROES, skills: bool = False):
    """rng -> QueueSpawn list: `heroes` random heroes and 2-4 random units per side, the same for both
    sides (mirrored positions); melee in front, ranged behind. Hit points at hp_permille/1000 of
    normal keep fights short. With `skills` heroes spend their skill points (skill_build, the same
    build on both sides); otherwise they have no abilities."""
    from ..data.objects import combat_stats
    from ..protocol import QueueSpawn

    stats = combat_stats()

    def spawn(rng) -> list:
        comp = [(str(rng.choice(hero_pool)), int(rng.integers(hero_levels[0], hero_levels[1] + 1)))
                for _ in range(heroes)]
        comp += [(str(rng.choice(pool)), 1) for _ in range(int(rng.integers(units[0], units[1] + 1)))]
        rows: dict[bool, list] = {False: [], True: []}
        for code, level in comp:
            build = skill_build(code, level, rng) if skills and code[:1].isupper() else ()
            rows[stats[code].range > 200].append((code, level, build))
        out = []
        for player, side in ((0, -1), (1, 1)):
            for ranged, row in rows.items():
                for i, (code, level, build) in enumerate(row):
                    x = side * (350 + (150 if ranged else 0))
                    y = (i - (len(row) - 1) / 2) * 110
                    out.append(QueueSpawn(player, code, x, y, 0 if side < 0 else 180, hp_permille, level, build))
        return out

    return spawn


def _mirror_task(name: str = "mirror_mix", max_units: int = 5, hp_permille: int = 250,
                 targeting: str = "slot", abilities: bool = False, relational: bool = False,
                 tactical: bool = False, selfplay: bool = False, kill_reward: float = 0.0,
                 rejoin: bool = False, draw_reward: float = 0.0) -> Task:
    # fights last longer with more hit points: 45 s at 25%, 70 s at 50%
    sc = Scenario(units=(), victory="elimination", max_game_seconds=round(20 + hp_permille / 10), name=name)
    semantic, general = targeting == "semantic", targeting == "general"
    obs_size, act_sizes = _micro_sizes(max_units, len(SEMANTIC_TARGETS) if semantic else None, abilities, relational,
                                       general)
    spawner = mirror_spawner(units=(2, max_units - 1), hp_permille=hp_permille, skills=abilities)
    group = 5 if general else 4 if abilities else 3

    def make_env(inst: str):
        env = (MirrorSelfPlayEnv if selfplay else MicroEnv)(
            sc, max_own=max_units, max_enemy=max_units, name=inst, targeting=targeting, abilities=abilities,
            relational=relational, tactical=tactical, kill_reward=kill_reward, rejoin=rejoin, draw_reward=draw_reward)
        env.spawner = spawner
        return env

    extra = ""
    if abilities:
        extra = (" Heroes have random skill builds (the same on both sides) and cast: a cast kind with an"
                 " ability slot head and the target rules; the scripted opponent casts too.")
    elif semantic:
        extra = " Attack targets are rules (weakest in range, nearest, weakest, hero, threat); stop is retreat."
    if general:
        extra += (" General orders per unit: noop, stop, hold, move and attack-move (16 directions x 3 distances),"
                  " attack and cast (a pointer at any unit slot).")
    if relational:
        extra += " Units also see relational features (nearest opponent, in range, threatened, weakest, time to die)."
    if tactical or rejoin:
        extra += " Tactical masks: retreat only while losing hit points, no plain moves."
    if rejoin:
        extra += " After a pull-back, a unit told nothing attack-moves back into the fight."
    if draw_reward:
        extra += f" A draw (the time runs out) is worth {draw_reward:+g}, for both sides."
    if kill_reward:
        extra += f" Reward {kill_reward:+g} per enemy killed, {-kill_reward:+g} per own unit lost."
    labels = _micro_labels(max_units, semantic, abilities, general)
    if selfplay:
        extra += " Self-play: the policy plays both sides (two agents per game; the win rate is side 0's)."
        labels["action_mask"] = lambda env: [_micro_mask(side)[0] for side in env.sides.values()]
    return Task(
        name=name, obs_size=obs_size, act_sizes=act_sizes, make_env=make_env,
        flatten=_micro_flatten, to_action=lambda a: np.asarray(a, dtype=np.int64).reshape(max_units, group),
        outcome=_micro_outcome, scenario=sc, reward_scale=10.0, num_agents=2 if selfplay else 1, **labels,
        # sweeps abil6*, from scratch with action masks: 30% wins after 0.1M steps (was 0.66M)
        train_defaults=dict(step_seconds=0.5, horizon=16, lr=0.003, minibatch=192, replay_ratio=4.0,
                            extra=["--train.gae_lambda=0.8", "--train.clip_coef=0.3"]) if abilities else {},
        description=f"Mirror match, a new composition every episode: a hero (level 1-3) and 2-{max_units - 1} "
                    f"units from all races, {hp_permille / 10:.0f}% hit points, vs the scripted opponent." + extra,
        **_micro_layout(max_units, abilities, relational, general),
        mask_info=_micro_masks(targeting, abilities, tactical, rejoin),
        reward_info=_MICRO_REWARD + (f" {kill_reward:+g} per enemy unit killed, {-kill_reward:+g} per own unit lost."
                                     if kill_reward else "") + (f" A draw (time limit) is worth {draw_reward:+g}."
                                                                 if draw_reward else "")
                    + (" Zero-sum between the sides." if selfplay and not draw_reward else ""),
    )


@dataclass
class UnitAgentsTask(Task):
    """A single-agent micro task played by one agent per unit, all with the same policy (parameter
    sharing: every unit's experience trains the same behavior). Agent i sees its unit's features
    and whether it is alive, then the whole observation of the base task; it gives its unit's
    action heads; every agent gets the team's reward and outcome. A dead or missing unit's agent
    can only noop. `display_task` (the base task) labels videos, which show the team's orders."""
    base: Task | None = None
    display_task: Task | None = None

    def _agents_obs(self, env, obs) -> list[np.ndarray]:
        flat = self.base.flatten(obs)
        return [np.concatenate([obs["own"][i], obs["own_mask"][i:i + 1].astype(np.float32), flat]).astype(np.float32)
                for i in range(self.num_agents)]

    def reset(self, env, options: dict | None = None):
        obs, info = env.reset(options=options)
        return self._agents_obs(env, obs), info

    def step(self, env, actions: list[np.ndarray]):
        team = np.concatenate([np.asarray(a, dtype=np.int64).ravel() for a in actions])
        obs, reward, terminated, truncated, info = env.step(self.base.to_action(team))
        done = terminated or truncated
        outcome = self.base.outcome(env, info) if done else 0.0
        n = self.num_agents
        return self._agents_obs(env, obs), [float(reward) * self.reward_scale] * n, done, info, [outcome] * n

    def combine(self, per_agent: list[np.ndarray]) -> np.ndarray:
        """The team's action (or mask) in the base task's layout."""
        return np.concatenate([np.asarray(a).ravel() for a in per_agent])


def describe_spaces(task: Task) -> dict:
    """The observation and action spaces for people (run.json "spaces"; the dashboard shows it)."""
    blocks = [{"name": n, "rows": r, "features": list(f)} for n, r, f in task.obs_layout]
    if sum(b["rows"] * len(b["features"]) for b in blocks) != task.obs_size:
        blocks = []  # no layout (or an outdated one): only the size
    g = max(task.group_size, 1)
    heads = []
    for h in range(g):
        labels = task.head_labels[h] if h < len(task.head_labels) else ()
        used_by = []
        if h and task.head_labels:
            for kind, offs in task.detail_heads.items():
                if h in (offs if isinstance(offs, (tuple, list)) else (offs,)) and kind < len(task.head_labels[0]):
                    used_by.append(task.head_labels[0][kind])
        heads.append({"name": task.head_names[h] if h < len(task.head_names) else f"head {h + 1}",
                      "size": task.act_sizes[h], "options": list(labels) or [str(i) for i in range(task.act_sizes[h])],
                      "used_by": used_by})
    return {"observation": {"size": task.obs_size, "blocks": blocks},
            "actions": {"units": task.num_atns // g, "heads": heads, "sizes": list(task.act_sizes),
                        "masks": task.mask_info if task.action_mask is not None else "None."},
            "agents": task.num_agents, "reward": task.reward_info, "reward_scale": task.reward_scale}


def task_spec(task: Task) -> dict:
    """What a trainer outside this package needs to know about a task (warcraftsim/rl: spec.json)."""
    return {"task": task.name, "obs_size": task.obs_size, "num_atns": task.num_atns, "act_sizes": list(task.act_sizes),
            "group_size": task.group_size, "agents": task.num_agents, "reward_scale": task.reward_scale,
            "spaces": describe_spaces(task)}


def _unit_agents(base: Task, name: str) -> UnitAgentsTask:
    heads = base.act_sizes[:base.group_size]
    per = sum(heads)
    k = base.num_atns // base.group_size
    unit_feat = (base.obs_size - 1) // (2 * k) - 1  # (feat + mask) per unit, own and enemy halves

    def mask(env):
        m = base.action_mask(env)[0].reshape(k, per).copy()
        for i in range(k):
            if i >= len(env._own) or env._own[i] is None:  # no unit: noop only
                m[i, :heads[0]] = 0
                m[i, 0] = 1
        return list(m)

    def stats(env, actions):
        return base.action_stats(env, [np.concatenate([np.asarray(a).ravel() for a in actions])])

    return UnitAgentsTask(
        name=name, obs_size=base.obs_size + unit_feat + 1, act_sizes=heads, make_env=base.make_env,
        flatten=base.flatten, to_action=lambda a: np.asarray(a, dtype=np.int64).reshape(-1, base.group_size),
        outcome=base.outcome, description=base.description + f" One agent per unit ({k}), one shared policy.",
        scenario=base.scenario, num_agents=k, reward_scale=base.reward_scale,
        head_labels=base.head_labels[:base.group_size], group_size=base.group_size, detail_heads=base.detail_heads,
        action_stats=stats, action_mask=mask, train_defaults=base.train_defaults, base=base, display_task=base,
        obs_layout=(("this agent's unit", 1, base.obs_layout[0][2] if base.obs_layout else ()),
                    ("its slot alive", 1, ("alive",)), *base.obs_layout) if base.obs_layout else (),
        head_names=base.head_names,
        mask_info=base.mask_info + " A dead or missing unit's agent can only noop.",
        reward_info=base.reward_info + " Every agent gets the team's reward.")


def _footmen_task(name: str) -> Task | None:
    """footmen{N}v{M}[_hp{HP}][_ehp{EHP}]: N agent footmen against M scripted ones, HP hit points
    each (default 100), the enemies EHP (default: HP; a handicap)."""
    m = re.fullmatch(r"footmen(\d+)v(\d+)(?:_hp(\d+))?(?:_ehp(\d+))?", name)
    if not m:
        return None
    n, e, hp = int(m[1]), int(m[2]), int(m[3] or 100)
    ehp = int(m[4]) if m[4] else None
    return _micro_task(("hfoo",) * n, ("hfoo",) * e, max_units=max(n, e), name=name, max_hp=hp,
                       max_game_seconds=40 + 5 * max(n, e), enemy_max_hp=ehp)


def _mirror_variant(name: str) -> Task | None:
    if name.endswith("_units") and name != "_units":  # one agent per unit on top of any variant
        base = get_task(name[:-len("_units")])
        return _unit_agents(base, name)
    """mirror_mix[_sem][_abil][_rel][_tac][_rejoin][_kill][_self][_hp{P}]: mirror_mix with semantic
    attack targets (MicroEnv targeting="semantic"), hero abilities (skill builds and casting; implies
    semantic targets), relational unit features, tactical action masks (and units rejoining the
    fight after a pull-back: implies tactical), kill rewards, self-play (MirrorSelfPlayEnv: the
    policy plays both sides), and/or P permille of the units' hit points (default 250)."""
    m = re.fullmatch(r"mirror_mix(?P<gen>_gen)?(?P<sem>_sem)?(?P<abil>_abil)?(?P<rel>_rel)?(?P<tac>_tac)?"
                     r"(?P<rejoin>_rejoin)?(?P<kill>_kill)?(?P<nodraw>_nodraw)?(?P<self>_self)?(?:_hp(?P<hp>\d+))?", name)
    if not m or not any(m.groups()):
        return None
    tactical = bool(m["tac"] or m["rejoin"])
    if m["gen"] and (m["sem"] or tactical):
        raise KeyError(f"{name}: _gen (general orders) has no semantic targets or tactical mode")
    return _mirror_task(name, hp_permille=int(m["hp"] or 250),
                        targeting="general" if m["gen"] else
                        "semantic" if m["sem"] or m["abil"] or tactical or m["self"] else "slot",
                        abilities=bool(m["abil"]), relational=bool(m["rel"]), tactical=tactical,
                        selfplay=bool(m["self"]), kill_reward=0.2 if m["kill"] else 0.0, rejoin=bool(m["rejoin"]),
                        draw_reward=-1.0 if m["nodraw"] else 0.0)


def get_task(name: str) -> Task:
    if name in TASKS:
        return TASKS[name]()
    task = _footmen_task(name) or _mirror_variant(name)
    if task is None:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}, footmen<N>v<M>[_hp<HP>][_ehp<EHP>] "
                       f"and mirror_mix[_gen][_sem][_abil][_rel][_tac][_rejoin][_kill][_nodraw][_self][_hp<permille>]")
    return task
