"""RL tasks served to the PufferLib 5.0 trainer.

A task wraps a warcraftsim environment with fixed-size float observations and a flat
multi-discrete action vector per agent (PufferLib's native format). Sizes are compile-time
constants on the C side, so the task also generates the C header (see build.py).

Single-agent tasks wrap a Gymnasium env; self-play tasks (num_agents=2) wrap MicroSelfPlayEnv and
serve both sides of every game to the trainer (both controlled by the policy being trained, or
by historical checkpoints when PufferLib's self-play pool is enabled).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..env import MicroEnv, MicroSelfPlayEnv, NavigateEnv
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
            return [self.flatten(obs)], [float(reward)], done, info, outcomes
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
        return self._obs_list(obs), [float(rewards[a]) for a in range(self.num_agents)], done, info, outcomes

    def _obs_list(self, obs) -> list[np.ndarray]:
        if self.num_agents == 1:
            return [self.flatten(obs)]
        return [self.flatten(obs[a]) for a in range(self.num_agents)]


# ---- navigate: one footman must reach a point --------------------------------------------------

def _nav_task(distance: float = 1200.0) -> Task:
    sc = Scenario.move_to_target("hfoo", distance=distance, max_game_seconds=30)

    def outcome(env, info) -> float:
        return 1.0 if info.get("distance", 1e9) <= sc.target[2] else -1.0

    return Task(
        name="nav", obs_size=6, act_sizes=(9,),
        make_env=lambda name: NavigateEnv(sc, name=name),
        flatten=lambda obs: np.asarray(obs, dtype=np.float32),
        to_action=lambda a: int(a[0]),
        outcome=outcome, scenario=sc,
        description="Move a footman 1200 units to a target (stop or 8 directions); reward = progress.",
    )


# ---- micro: small fights ------------------------------------------------------------------------

_FEAT = 24


def _micro_sizes(max_units: int) -> tuple[int, tuple[int, ...]]:
    k = max_units
    return k * _FEAT + k + k * _FEAT + k + 1, (4, 8, k) * k


def _micro_flatten(obs: dict) -> np.ndarray:
    return np.concatenate([obs["own"].ravel(), obs["own_mask"].astype(np.float32), obs["enemy"].ravel(),
                           obs["enemy_mask"].astype(np.float32), obs["time"]]).astype(np.float32)


def _micro_outcome(env, info) -> float:
    o = info.get("obs")
    if o is None or not o.game_over:
        return 0.0
    r = o.players[0].result
    return 1.0 if r.name == "VICTORY" else -1.0 if r.name == "DEFEAT" else 0.0


def _micro_task(own: tuple[str, ...] = ("hfoo",) * 4, enemy: tuple[str, ...] = ("ogru",) * 3,
                max_units: int = 6, name: str = "micro") -> Task:
    sc = Scenario.skirmish(list(own), list(enemy), max_game_seconds=90)
    obs_size, act_sizes = _micro_sizes(max_units)
    return Task(
        name=name, obs_size=obs_size, act_sizes=act_sizes,
        make_env=lambda inst: MicroEnv(sc, max_own=max_units, max_enemy=max_units, name=inst),
        flatten=_micro_flatten, to_action=lambda a: np.asarray(a, dtype=np.int64).reshape(max_units, 3),
        outcome=_micro_outcome, scenario=sc,
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
        scenario=sc,
        description=f"Self-play: {len(units)} {units[0]} vs {len(units)} {units[0]}, both sides are agents.",
    )


TASKS: dict[str, Callable[[], Task]] = {
    "nav": _nav_task,
    "micro": _micro_task,
    "micro_mirror": lambda: _micro_task(("hfoo",) * 4, ("hfoo",) * 4, name="micro_mirror"),
    "selfplay_micro": _selfplay_task,
}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]()
    except KeyError:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}") from None
