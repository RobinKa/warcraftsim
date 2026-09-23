"""Gymnasium environments.

* ``Wc3Env``: the general interface. Observations are fixed-size feature arrays of
  the units the agent can see; actions are lists of ``protocol.Command`` objects
  (build them with the helpers on ``env.game``, a ``Wc3Game``). Reward: +1 win,
  -1 loss, 0 otherwise.
* ``MicroEnv``: scenario fights with a factored discrete action per own unit
  (noop / stop / move in one of 8 directions / attack an enemy slot), reward =
  damage balance + outcome.
* ``NavigateEnv``: move one unit to a target point as fast as possible.
* ``MicroSelfPlayEnv``: two policies control the two sides of a scenario (dict API).

Every env exposes the raw ``Observation`` as ``info["obs"]``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import Wc3Game
from .data.objects import unit_vocabulary
from .protocol import Command, Observation, Result, Unit, UnitFlags
from .runtime.instance import GameSetup
from .scenario import Scenario

UNIT_FEATURES = 24
PLAYER_FEATURES = 8
_N_FLAGS = 12


def _order_class(order: int, orders: dict[str, int]) -> int:
    if order == 0:
        return 0
    if order == orders.get("move") or order == orders.get("smart"):
        return 1
    if order == orders.get("attack"):
        return 2
    if order in (orders.get("harvest"), orders.get("resumeharvesting"), orders.get("returnresources")):
        return 3
    return 4


class UnitEncoder:
    """Units -> float32 feature rows. Coordinates are normalised by `extent` around `origin`."""

    def __init__(self, player: int, origin: tuple[float, float], extent: float):
        self.player = player
        self.origin = origin
        self.extent = extent
        self.vocab = unit_vocabulary()

    def encode(self, units: Sequence[Unit], obs: Observation, orders: dict[str, int], max_units: int,
               allies: set[int] = frozenset()) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        feats = np.zeros((max_units, UNIT_FEATURES), dtype=np.float32)
        types = np.zeros(max_units, dtype=np.int64)
        ids = np.zeros(max_units, dtype=np.int64)
        mask = np.zeros(max_units, dtype=bool)
        ox, oy = self.origin
        for i, u in enumerate(units[:max_units]):
            if u.owner == self.player:
                rel = 0
            elif u.owner in allies:
                rel = 1
            elif u.owner in obs.players:
                rel = 2
            else:
                rel = 3
            f = feats[i]
            f[rel] = 1.0
            f[4] = (u.x - ox) / self.extent
            f[5] = (u.y - oy) / self.extent
            f[6] = u.hp / max(u.max_hp, 1)
            f[7] = u.max_hp / 1000.0
            f[8] = u.mana / max(u.max_mana, 1) if u.max_mana else 0.0
            f[9] = u.max_mana / 1000.0
            rad = math.radians(u.facing)
            f[10], f[11] = math.sin(rad), math.cos(rad)
            for b in range(_N_FLAGS - 1):  # all flags but DEAD (dead units are filtered out)
                f[12 + b] = float(u.flags >> b & 1)
            f[23] = _order_class(u.order, orders) / 4.0
            types[i] = self.vocab.get(u.type, 0)
            ids[i] = u.id
            mask[i] = True
        return feats, types, ids, mask


class CommandListSpace(spaces.Space):
    """Actions for Wc3Env: a list of protocol.Command objects."""

    def __init__(self):
        super().__init__(shape=None, dtype=None)

    def contains(self, x: Any) -> bool:
        return isinstance(x, (list, tuple)) and all(isinstance(c, Command) for c in x)

    def sample(self, mask: Any = None, probability: Any = None) -> list[Command]:
        return []

    @property
    def is_np_flattenable(self) -> bool:
        return False


class Wc3Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, setup: GameSetup | None = None, player: int | None = None, max_units: int = 256,
                 name: str = "env0", **instance_kw):
        self.setup = setup or GameSetup()
        self.game = Wc3Game(self.setup, player=player, name=name, **instance_kw)
        self.player = self.game.player
        self.max_units = max_units
        self.encoder: UnitEncoder | None = None
        self.observation_space = spaces.Dict({
            "units": spaces.Box(-np.inf, np.inf, (max_units, UNIT_FEATURES), np.float32),
            "unit_types": spaces.Box(0, 10_000, (max_units,), np.int64),
            "unit_ids": spaces.Box(0, np.iinfo(np.int64).max, (max_units,), np.int64),
            "unit_mask": spaces.MultiBinary(max_units),
            "player": spaces.Box(-np.inf, np.inf, (PLAYER_FEATURES,), np.float32),
        })
        self.action_space = CommandListSpace()

    # ---- observation ----------------------------------------------------------------------

    def _visible(self, obs: Observation) -> list[Unit]:
        if self.setup.fog_enabled:
            units = obs.visible_units(self.player)
        else:
            units = [u for u in obs.units if u.alive]
        # own units first, then by distance to the own start location
        me = obs.players.get(self.player)
        sx, sy = (me.start_x, me.start_y) if me else (0, 0)
        return sorted(units, key=lambda u: (u.owner != self.player, u.dist(sx, sy)))

    def _encode(self, obs: Observation) -> dict[str, np.ndarray]:
        if self.encoder is None:
            me = obs.players.get(self.player)
            self.encoder = UnitEncoder(self.player, (me.start_x, me.start_y) if me else (0.0, 0.0), 8192.0)
        feats, types, ids, mask = self.encoder.encode(self._visible(obs), obs, self.game._orders, self.max_units)
        p = obs.players.get(self.player)
        pv = np.zeros(PLAYER_FEATURES, dtype=np.float32)
        if p:
            pv[:] = [p.gold / 1000, p.lumber / 1000, p.food_used / 100, p.food_cap / 100, p.upkeep / 100,
                     obs.game_time / 600, p.structures / 20, len(obs.units_of(self.player)) / 100]
        return {"units": feats, "unit_types": types, "unit_ids": ids, "unit_mask": mask, "player": pv}

    def _outcome(self, obs: Observation) -> tuple[float, bool, bool]:
        p = obs.players.get(self.player)
        if not obs.game_over or p is None:
            return 0.0, False, False
        if p.result == Result.VICTORY:
            return 1.0, True, False
        if p.result == Result.DEFEAT:
            return -1.0, True, False
        return 0.0, False, True  # tie = time limit

    # ---- gym API ----------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """options: {"relaunch": True} starts the episode in a fresh game process."""
        super().reset(seed=seed)
        obs = self.game.reset(relaunch=bool((options or {}).get("relaunch")))
        self._on_reset(obs)
        return self._encode(obs), {"obs": obs}

    def _on_reset(self, obs: Observation) -> None:
        pass

    def step(self, action):
        commands = list(action) if action is not None else []
        obs = self.game.step(commands)
        reward, terminated, truncated = self._outcome(obs)
        return self._encode(obs), reward, terminated, truncated, {"obs": obs}

    def render(self):
        path = f"/dev/shm/warcraftsim/{self.game.instance.name}/frame.png"
        self.game.instance.screenshot(path)
        try:
            import imageio.v3 as iio  # optional
            return iio.imread(path)
        except ImportError:
            return path

    def close(self):
        self.game.close()


class MicroEnv(Wc3Env):
    """Scenario fights. Action per own-unit slot: [kind, direction, target].

    kind 0 noop, 1 stop, 2 move `move_distance` in direction (8-way), 3 attack enemy slot `target`.
    Reward: (enemy hp lost - own hp lost) / initial total hp per step, +1/-1 for win/loss.
    """

    def __init__(self, scenario: Scenario | None = None, max_own: int = 12, max_enemy: int = 12,
                 move_distance: float = 250.0, opponent: str = "scripted", name: str = "micro0", **kw):
        from .runtime.instance import Agent, Idle, Scripted

        self.scenario = scenario or Scenario.skirmish(["hfoo"] * 4, ["hfoo"] * 4)
        opp = {"scripted": Scripted, "idle": Idle, "agent": Agent}[opponent]
        setup = kw.pop("setup", None) or GameSetup(slots=[Agent("human"), opp("orc")], scenario=self.scenario)
        super().__init__(setup, player=0, max_units=max_own + max_enemy, name=name, **kw)
        self.max_own, self.max_enemy = max_own, max_enemy
        self.move_distance = move_distance
        self.observation_space = spaces.Dict({
            "own": spaces.Box(-np.inf, np.inf, (max_own, UNIT_FEATURES), np.float32),
            "own_types": spaces.Box(0, 10_000, (max_own,), np.int64),
            "own_mask": spaces.MultiBinary(max_own),
            "enemy": spaces.Box(-np.inf, np.inf, (max_enemy, UNIT_FEATURES), np.float32),
            "enemy_types": spaces.Box(0, 10_000, (max_enemy,), np.int64),
            "enemy_mask": spaces.MultiBinary(max_enemy),
            "time": spaces.Box(0, np.inf, (1,), np.float32),
        })
        self.action_space = spaces.MultiDiscrete(np.tile([4, 8, max_enemy], (max_own, 1)))
        self._own: list[Unit] = []
        self._enemy: list[Unit] = []
        self._hp0 = (1.0, 1.0)
        self._hp_prev = (0.0, 0.0)

    def _split(self, obs: Observation) -> tuple[list[Unit], list[Unit]]:
        own = sorted((u for u in obs.units if u.alive and u.owner == self.player), key=lambda u: u.id)
        enemy = sorted((u for u in obs.units if u.alive and u.owner != self.player and u.owner in obs.players),
                       key=lambda u: u.id)
        return own[:self.max_own], enemy[:self.max_enemy]

    def _on_reset(self, obs: Observation) -> None:
        cx, cy = self.scenario.resolved_center()
        self.encoder = UnitEncoder(self.player, (cx, cy), 1500.0)
        own, enemy = self._split(obs)
        self._hp0 = (max(sum(u.max_hp for u in own), 1), max(sum(u.max_hp for u in enemy), 1))
        self._hp_prev = (sum(u.hp for u in own), sum(u.hp for u in enemy))

    def _encode(self, obs: Observation) -> dict[str, np.ndarray]:
        self._own, self._enemy = self._split(obs)
        orders = self.game._orders
        of, ot, _, om = self.encoder.encode(self._own, obs, orders, self.max_own)
        ef, et, _, em = self.encoder.encode(self._enemy, obs, orders, self.max_enemy)
        return {"own": of, "own_types": ot, "own_mask": om, "enemy": ef, "enemy_types": et, "enemy_mask": em,
                "time": np.array([obs.game_time / max(self.scenario.max_game_seconds, 1)], np.float32)}

    def _commands(self, action: np.ndarray) -> list[Command]:
        action = np.asarray(action).reshape(self.max_own, 3)
        for i, unit in enumerate(self._own):
            kind, direction, target = (int(v) for v in action[i])
            if kind == 1:
                self.game.stop(unit)
            elif kind == 2:
                a = direction * math.pi / 4
                self.game.move(unit, unit.x + self.move_distance * math.cos(a), unit.y + self.move_distance * math.sin(a))
            elif kind == 3 and target < len(self._enemy):
                self.game.attack(unit, self._enemy[target])
        return []

    def step(self, action):
        self._commands(action)
        obs = self.game.step()
        own_hp = sum(u.hp for u in obs.units if u.alive and u.owner == self.player)
        enemy_hp = sum(u.hp for u in obs.units if u.alive and u.owner != self.player and u.owner in obs.players)
        dealt = (self._hp_prev[1] - enemy_hp) / self._hp0[1]
        taken = (self._hp_prev[0] - own_hp) / self._hp0[0]
        self._hp_prev = (own_hp, enemy_hp)
        outcome, terminated, truncated = self._outcome(obs)
        return self._encode(obs), float(dealt - taken + outcome), terminated, truncated, {"obs": obs}


class NavigateEnv(Wc3Env):
    """Move one unit to the scenario's target. Action: Discrete(9) = stop or move 8-way.

    Reward: progress towards the target / initial distance, -0.01 per step, +1 on arrival.
    """

    def __init__(self, scenario: Scenario | None = None, move_distance: float = 300.0, name: str = "nav0", **kw):
        from .runtime.instance import Agent

        self.scenario = scenario or Scenario.move_to_target("hfoo", distance=1200.0)
        if self.scenario.target is None:
            raise ValueError("NavigateEnv needs a scenario with a target")
        setup = GameSetup(slots=[Agent("human")], scenario=self.scenario)
        super().__init__(setup, player=0, max_units=1, name=name, **kw)
        self.move_distance = move_distance
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)
        self.action_space = spaces.Discrete(9)
        self._dist0 = 1.0
        self._dist_prev = 0.0
        self._unit: Unit | None = None

    def _target(self) -> tuple[float, float, float]:
        return self.scenario.absolute_target()

    def _encode(self, obs: Observation) -> np.ndarray:
        units = obs.units_of(self.player)
        self._unit = units[0] if units else None
        tx, ty, _ = self._target()
        if self._unit is None:
            return np.zeros(6, np.float32)
        u = self._unit
        rad = math.radians(u.facing)
        return np.array([(tx - u.x) / 1000, (ty - u.y) / 1000, math.sin(rad), math.cos(rad),
                         float(u.order != 0), obs.game_time / self.scenario.max_game_seconds], np.float32)

    def _on_reset(self, obs: Observation) -> None:
        tx, ty, _ = self._target()
        u = obs.units_of(self.player)[0]
        self._dist0 = max(u.dist(tx, ty), 1.0)
        self._dist_prev = u.dist(tx, ty)

    def step(self, action):
        a = int(action)
        if self._unit is not None:
            if a == 0:
                self.game.stop(self._unit)
            else:
                ang = (a - 1) * math.pi / 4
                self.game.move(self._unit, self._unit.x + self.move_distance * math.cos(ang),
                               self._unit.y + self.move_distance * math.sin(ang))
        obs = self.game.step()
        tx, ty, radius = self._target()
        units = obs.units_of(self.player)
        d = units[0].dist(tx, ty) if units else self._dist_prev
        reward = (self._dist_prev - d) / self._dist0 - 0.01
        self._dist_prev = d
        terminated = d <= radius
        if terminated:
            reward += 1.0
        truncated = not terminated and obs.game_over  # time limit (the harness reports a tie)
        return self._encode(obs), float(reward), terminated, truncated, {"obs": obs, "distance": d}


class MicroSelfPlayEnv:
    """Two policies fight each other in a scenario (a PettingZoo-style parallel API).

        env = MicroSelfPlayEnv(Scenario.skirmish(["hfoo"] * 4, ["hfoo"] * 4))
        obs, infos = env.reset()                          # {0: obs0, 1: obs1}
        obs, rewards, terminated, truncated, infos = env.step({0: action0, 1: action1})

    Observations, actions and rewards per player are those of ``MicroEnv`` from that player's
    perspective (own units first, the other side as enemies). Rewards are zero-sum.
    """

    possible_agents = (0, 1)

    def __init__(self, scenario: Scenario | None = None, max_units: int = 12, move_distance: float = 250.0,
                 name: str = "selfplay0", **instance_kw):
        from .runtime.instance import Agent

        self.scenario = scenario or Scenario.skirmish(["hfoo"] * 4, ["hfoo"] * 4)
        setup = GameSetup(slots=[Agent("human"), Agent("orc")], scenario=self.scenario)
        self.setup = setup
        self.game = Wc3Game(setup, player=0, name=name, **instance_kw)
        self.views = {p: None for p in self.possible_agents}
        self.max_units = max_units
        self.move_distance = move_distance
        self.encoders: dict[int, UnitEncoder] = {}
        self.single_observation_space = spaces.Dict({
            "own": spaces.Box(-np.inf, np.inf, (max_units, UNIT_FEATURES), np.float32),
            "own_types": spaces.Box(0, 10_000, (max_units,), np.int64),
            "own_mask": spaces.MultiBinary(max_units),
            "enemy": spaces.Box(-np.inf, np.inf, (max_units, UNIT_FEATURES), np.float32),
            "enemy_types": spaces.Box(0, 10_000, (max_units,), np.int64),
            "enemy_mask": spaces.MultiBinary(max_units),
            "time": spaces.Box(0, np.inf, (1,), np.float32),
        })
        self.single_action_space = spaces.MultiDiscrete(np.tile([4, 8, max_units], (max_units, 1)))
        self._units: dict[int, tuple[list[Unit], list[Unit]]] = {}
        self._hp0: dict[int, float] = {}
        self._hp_prev: dict[int, float] = {}

    def observation_space(self, agent: int) -> spaces.Space:
        return self.single_observation_space

    def action_space(self, agent: int) -> spaces.Space:
        return self.single_action_space

    def _side(self, obs: Observation, player: int) -> tuple[list[Unit], list[Unit]]:
        own = sorted((u for u in obs.units if u.alive and u.owner == player), key=lambda u: u.id)
        enemy = sorted((u for u in obs.units if u.alive and u.owner != player and u.owner in obs.players),
                       key=lambda u: u.id)
        return own[:self.max_units], enemy[:self.max_units]

    def _hp(self, obs: Observation, player: int) -> float:
        return float(sum(u.hp for u in obs.units if u.alive and u.owner == player))

    def _encode(self, obs: Observation) -> dict[int, dict[str, np.ndarray]]:
        out = {}
        for p in self.possible_agents:
            own, enemy = self._units[p] = self._side(obs, p)
            enc = self.encoders[p]
            of, ot, _, om = enc.encode(own, obs, self.game._orders, self.max_units)
            ef, et, _, em = enc.encode(enemy, obs, self.game._orders, self.max_units)
            out[p] = {"own": of, "own_types": ot, "own_mask": om, "enemy": ef, "enemy_types": et,
                      "enemy_mask": em,
                      "time": np.array([obs.game_time / max(self.scenario.max_game_seconds, 1)], np.float32)}
        return out

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs = self.game.reset(relaunch=bool((options or {}).get("relaunch")))
        cx, cy = self.scenario.resolved_center()
        self.encoders = {p: UnitEncoder(p, (cx, cy), 1500.0) for p in self.possible_agents}
        for p in self.possible_agents:
            self.views[p] = self.game.as_player(p)
            self._hp0[p] = max(sum(u.max_hp for u in obs.units if u.owner == p), 1)
            self._hp_prev[p] = self._hp(obs, p)
        return self._encode(obs), {p: {"obs": obs} for p in self.possible_agents}

    def step(self, actions: dict[int, np.ndarray]):
        for p, action in actions.items():
            view = self.views[p]
            own, enemy = self._units[p]
            action = np.asarray(action).reshape(self.max_units, 3)
            for i, unit in enumerate(own):
                kind, direction, target = (int(v) for v in action[i])
                if kind == 1:
                    view.stop(unit)
                elif kind == 2:
                    a = direction * math.pi / 4
                    view.move(unit, unit.x + self.move_distance * math.cos(a),
                              unit.y + self.move_distance * math.sin(a))
                elif kind == 3 and target < len(enemy):
                    view.attack(unit, enemy[target])
        obs = self.game.step()
        hp = {p: self._hp(obs, p) for p in self.possible_agents}
        loss = {p: (self._hp_prev[p] - hp[p]) / self._hp0[p] for p in self.possible_agents}
        self._hp_prev = hp
        rewards = {0: loss[1] - loss[0], 1: loss[0] - loss[1]}
        terminated = {p: False for p in self.possible_agents}
        truncated = {p: False for p in self.possible_agents}
        if obs.game_over:
            for p in self.possible_agents:
                result = obs.players[p].result
                if result == Result.VICTORY:
                    rewards[p] += 1.0
                    terminated[p] = True
                elif result == Result.DEFEAT:
                    rewards[p] -= 1.0
                    terminated[p] = True
                else:
                    truncated[p] = True
        return self._encode(obs), rewards, terminated, truncated, {p: {"obs": obs} for p in self.possible_agents}

    def close(self) -> None:
        self.game.close()
