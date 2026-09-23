"""RL tasks served to the PufferLib 5.0 trainer.

A task wraps one of the Gymnasium environments with fixed-size float observations and a flat
multi-discrete action vector (PufferLib's native format). Sizes are compile-time constants on the
C side, so the task also generates the C header (see build.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..env import MicroEnv, NavigateEnv
from ..scenario import Scenario


@dataclass
class Task:
    name: str
    obs_size: int
    act_sizes: tuple[int, ...]
    make_env: Callable[[str], Any]  # instance name -> env
    flatten: Callable[[Any], np.ndarray]  # env observation -> float32[obs_size]
    to_action: Callable[[np.ndarray], Any]  # float32[num_atns] -> env action
    outcome: Callable[[Any, dict], float] = lambda env, info: 0.0  # +1 win, -1 loss, 0 other
    description: str = ""
    scenario: Scenario | None = None

    @property
    def num_atns(self) -> int:
        return len(self.act_sizes)


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


# ---- micro: small fights against the scripted opponent -----------------------------------------

def _micro_task(own: tuple[str, ...] = ("hfoo",) * 4, enemy: tuple[str, ...] = ("ogru",) * 3,
                max_units: int = 6, name: str = "micro") -> Task:
    sc = Scenario.skirmish(list(own), list(enemy), max_game_seconds=90)
    k = e = max_units
    feat = 24
    obs_size = k * feat + k + e * feat + e + 1

    def flatten(obs: dict) -> np.ndarray:
        return np.concatenate([obs["own"].ravel(), obs["own_mask"].astype(np.float32), obs["enemy"].ravel(),
                               obs["enemy_mask"].astype(np.float32), obs["time"]]).astype(np.float32)

    def to_action(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=np.int64).reshape(k, 3)

    def outcome(env, info) -> float:
        o = info.get("obs")
        if o is None or not o.game_over:
            return 0.0
        r = o.players[0].result
        return 1.0 if r.name == "VICTORY" else -1.0 if r.name == "DEFEAT" else 0.0

    return Task(
        name=name, obs_size=obs_size, act_sizes=(4, 8, e) * k,
        make_env=lambda inst: MicroEnv(sc, max_own=k, max_enemy=e, name=inst),
        flatten=flatten, to_action=to_action, outcome=outcome, scenario=sc,
        description=f"{len(own)} {own[0]} vs {len(enemy)} {enemy[0]} (scripted); per unit: noop/stop/move/attack.",
    )


TASKS: dict[str, Callable[[], Task]] = {
    "nav": _nav_task,
    "micro": _micro_task,
    "micro_mirror": lambda: _micro_task(("hfoo",) * 4, ("hfoo",) * 4, name="micro_mirror"),
}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]()
    except KeyError:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}") from None
