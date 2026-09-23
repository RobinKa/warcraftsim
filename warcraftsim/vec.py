"""Run several games in parallel.

Every game is its own Wine process; the Python side mostly waits on sockets, so
a thread per environment is enough to keep all games busy.

    envs = Wc3VecEnv([lambda i=i: MicroEnv(name=f"micro{i}") for i in range(8)])
    obs, infos = envs.reset()
    obs, rewards, terminated, truncated, infos = envs.step(actions)

Finished environments reset automatically inside step(): the returned observation
is the first one of the new episode and the last one of the old episode is in
``infos["final_obs"][i]`` (with ``infos["final_info"][i]``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array


class Wc3VecEnv(gym.vector.VectorEnv):
    def __init__(self, env_fns: Sequence[Callable[[], gym.Env]], start_parallel: bool = True):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        first = self.envs[0]
        self.single_observation_space = first.observation_space
        self.single_action_space = first.action_space
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        try:
            self.action_space = batch_space(self.single_action_space, self.num_envs)
        except (TypeError, ValueError):
            self.action_space = self.single_action_space  # command lists: pass one list per env
        self._pool = ThreadPoolExecutor(max_workers=self.num_envs)
        self._start_parallel = start_parallel
        self.metadata = getattr(first, "metadata", {})

    def _batch(self, observations: list[Any]) -> Any:
        out = create_empty_array(self.single_observation_space, n=self.num_envs)
        return concatenate(self.single_observation_space, observations, out)

    def reset(self, *, seed: int | Sequence[int] | None = None, options: dict | None = None):
        seeds = seed if isinstance(seed, (list, tuple)) else [seed] * self.num_envs
        if self._start_parallel:
            results = list(self._pool.map(lambda args: args[0].reset(seed=args[1], options=options),
                                          zip(self.envs, seeds)))
        else:
            results = [env.reset(seed=s, options=options) for env, s in zip(self.envs, seeds)]
        obs, infos = zip(*results)
        return self._batch(list(obs)), {"infos": list(infos)}

    def _step_one(self, env: gym.Env, action: Any):
        obs, reward, terminated, truncated, info = env.step(action)
        final_obs = final_info = None
        if terminated or truncated:
            final_obs, final_info = obs, info
            obs, info = env.reset()
        return obs, reward, terminated, truncated, info, final_obs, final_info

    def step(self, actions: Any):
        if isinstance(actions, np.ndarray) or (isinstance(actions, (list, tuple)) and len(actions) == self.num_envs):
            per_env = [actions[i] for i in range(self.num_envs)]
        else:
            raise ValueError(f"expected one action per environment ({self.num_envs})")
        results = list(self._pool.map(lambda args: self._step_one(*args), zip(self.envs, per_env)))
        obs, rewards, terminated, truncated, infos, final_obs, final_info = zip(*results)
        batch_infos = {"infos": list(infos), "final_obs": list(final_obs), "final_info": list(final_info)}
        return (self._batch(list(obs)), np.array(rewards, dtype=np.float32), np.array(terminated, dtype=bool),
                np.array(truncated, dtype=bool), batch_infos)

    def close_extras(self, **kwargs):
        list(self._pool.map(lambda e: e.close(), self.envs))
        self._pool.shutdown(wait=False)
