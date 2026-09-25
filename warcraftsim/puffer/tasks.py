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

from ..env import (ABILITY_FEATURES, RELATIONAL_FEATURES, SEMANTIC_TARGETS, UNIT_FEATURES, MicroEnv,
                   MicroSelfPlayEnv, MirrorSelfPlayEnv, NavigateEnv)
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
    )


# ---- micro: small fights ------------------------------------------------------------------------

_FEAT = UNIT_FEATURES


def _micro_sizes(max_units: int, targets: int | None = None, abilities: bool = False,
                 relational: bool = False) -> tuple[int, tuple[int, ...]]:
    k = max_units
    feat = _FEAT + (ABILITY_FEATURES * HERO_ABILITY_SLOTS if abilities else 0) + (RELATIONAL_FEATURES if relational else 0)
    heads = (5, 8, targets or k, HERO_ABILITY_SLOTS) if abilities else (4, 8, targets or k)
    return k * feat + k + k * feat + k + 1, heads * k


def _micro_flatten(obs: dict) -> np.ndarray:
    return np.concatenate([obs["own"].ravel(), obs["own_mask"].astype(np.float32), obs["enemy"].ravel(),
                           obs["enemy_mask"].astype(np.float32), obs["time"]]).astype(np.float32)


_KINDS = ("noop", "stop", "move", "attack")
_SEMANTIC_KINDS = ("noop", "retreat", "move", "attack")
_ABILITY_SLOTS = tuple(f"ability {k + 1}" for k in range(HERO_ABILITY_SLOTS))


def _micro_labels(max_units: int, semantic: bool = False, abilities: bool = False) -> dict:
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


def _micro_mask(env) -> list[np.ndarray]:
    """MicroEnv action masks: attacks on empty enemy slots (slot targeting); casts only by units
    with an ability they can cast now, and only those ability slots."""
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
        targets = []
        for i in range(min(len(own), len(a))):
            if own[i] is None:
                continue
            kind, _, target = a[i][:3]
            c["unit_steps"] += 1
            c[kinds[kind]] += 1
            if kinds[kind] == "cast" and side.cast_command(own[i], int(a[i][3]), int(target), own, enemy) is None:
                c["cast_invalid"] += 1  # not learned, cooling down, no mana or no target: nothing happens
            if kind == 3:
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
                 tactical: bool = False, selfplay: bool = False) -> Task:
    # fights last longer with more hit points: 45 s at 25%, 70 s at 50%
    sc = Scenario(units=(), victory="elimination", max_game_seconds=round(20 + hp_permille / 10), name=name)
    semantic = targeting == "semantic"
    obs_size, act_sizes = _micro_sizes(max_units, len(SEMANTIC_TARGETS) if semantic else None, abilities, relational)
    spawner = mirror_spawner(units=(2, max_units - 1), hp_permille=hp_permille, skills=abilities)
    group = 4 if abilities else 3

    def make_env(inst: str):
        env = (MirrorSelfPlayEnv if selfplay else MicroEnv)(
            sc, max_own=max_units, max_enemy=max_units, name=inst, targeting=targeting, abilities=abilities,
            relational=relational, tactical=tactical)
        env.spawner = spawner
        return env

    extra = ""
    if abilities:
        extra = (" Heroes have random skill builds (the same on both sides) and cast: a cast kind with an"
                 " ability slot head and the target rules; the scripted opponent casts too.")
    elif semantic:
        extra = " Attack targets are rules (weakest in range, nearest, weakest, hero, threat); stop is retreat."
    if relational:
        extra += " Units also see relational features (nearest opponent, in range, threatened, weakest, time to die)."
    if tactical:
        extra += " Tactical masks: retreat only while losing hit points, no plain moves."
    labels = _micro_labels(max_units, semantic, abilities)
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
        action_stats=stats, action_mask=mask, train_defaults=base.train_defaults, base=base, display_task=base)


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
    """mirror_mix[_sem][_abil][_rel][_tac][_self][_hp{P}]: mirror_mix with semantic attack targets
    (MicroEnv targeting="semantic"), hero abilities (skill builds and casting; implies semantic
    targets), relational unit features, tactical action masks, self-play (MirrorSelfPlayEnv: the
    policy plays both sides), and/or P permille of the units' hit points (default 250)."""
    m = re.fullmatch(r"mirror_mix(_sem)?(_abil)?(_rel)?(_tac)?(_self)?(?:_hp(\d+))?", name)
    if not m or not any(m.groups()):
        return None
    return _mirror_task(name, hp_permille=int(m[6] or 250),
                        targeting="semantic" if m[1] or m[2] or m[4] or m[5] else "slot",
                        abilities=bool(m[2]), relational=bool(m[3]), tactical=bool(m[4]), selfplay=bool(m[5]))


def get_task(name: str) -> Task:
    if name in TASKS:
        return TASKS[name]()
    task = _footmen_task(name) or _mirror_variant(name)
    if task is None:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}, footmen<N>v<M>[_hp<HP>][_ehp<EHP>] "
                       f"and mirror_mix[_sem][_abil][_rel][_tac][_self][_hp<permille>]")
    return task
