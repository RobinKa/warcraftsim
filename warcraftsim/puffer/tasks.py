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
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..env import UNIT_FEATURES, MicroEnv, MicroSelfPlayEnv, NavigateEnv
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


def _micro_sizes(max_units: int) -> tuple[int, tuple[int, ...]]:
    k = max_units
    return k * _FEAT + k + k * _FEAT + k + 1, (4, 8, k) * k


def _micro_flatten(obs: dict) -> np.ndarray:
    return np.concatenate([obs["own"].ravel(), obs["own_mask"].astype(np.float32), obs["enemy"].ravel(),
                           obs["enemy_mask"].astype(np.float32), obs["time"]]).astype(np.float32)


_KINDS = ("noop", "stop", "move", "attack")


def _micro_labels(max_units: int) -> dict:
    return dict(head_labels=(_KINDS, _DIRS, tuple(f"E{i}" for i in range(max_units))) * max_units,
                group_size=3, detail_heads={2: 1, 3: 2}, action_stats=_micro_action_stats)


def _micro_action_stats(env, actions: list[np.ndarray]) -> Counter:
    """Per live unit and step: the order kind; for attacks, the target choice."""
    c: Counter = Counter()
    for agent, action in enumerate(actions):
        own, enemy = env._units[agent] if hasattr(env, "_units") else (env._own, env._enemy)
        a = np.asarray(action, int).reshape(-1, 3)
        targets = []
        for i in range(min(len(own), len(a))):
            kind, _, target = a[i]
            c["unit_steps"] += 1
            c[_KINDS[kind]] += 1
            if kind == 3:
                if target < len(enemy):
                    targets.append(int(target))
                else:
                    c["attack_invalid"] += 1
        if targets:
            weakest = min(range(len(enemy)), key=lambda j: enemy[j].hp)
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
        for k in _KINDS:
            out[k] = c.get(k, 0) / c["unit_steps"]
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
                max_units: int = 6, name: str = "micro", max_hp: int = 0, max_game_seconds: float = 90) -> Task:
    sc = Scenario.skirmish(list(own), list(enemy), max_game_seconds=max_game_seconds, max_hp=max_hp)
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
        scenario=sc, reward_scale=10.0, **_micro_labels(max_units),
        description=f"Self-play: {len(units)} {units[0]} vs {len(units)} {units[0]}, both sides are agents.",
    )


TASKS: dict[str, Callable[[], Task]] = {
    "nav": _nav_task,
    "micro": _micro_task,
    "micro_mirror": lambda: _micro_task(("hfoo",) * 4, ("hfoo",) * 4, name="micro_mirror"),
    # the smallest fight worth learning: pull a damaged footman back so the enemies switch targets
    "footmen2": lambda: _micro_task(("hfoo",) * 2, ("hfoo",) * 2, max_units=2, name="footmen2", max_hp=100,
                                    max_game_seconds=40),
    "selfplay_micro": _selfplay_task,
}


def _footmen_task(name: str) -> Task | None:
    """footmen{N}v{M}[_hp{HP}]: N agent footmen against M scripted ones, HP each (default 100)."""
    m = re.fullmatch(r"footmen(\d+)v(\d+)(?:_hp(\d+))?", name)
    if not m:
        return None
    n, e, hp = int(m[1]), int(m[2]), int(m[3] or 100)
    return _micro_task(("hfoo",) * n, ("hfoo",) * e, max_units=max(n, e), name=name, max_hp=hp,
                       max_game_seconds=40 + 5 * max(n, e))


def get_task(name: str) -> Task:
    if name in TASKS:
        return TASKS[name]()
    task = _footmen_task(name)
    if task is None:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)} and footmen<N>v<M>[_hp<HP>]")
    return task
