"""Gymnasium environments.

* ``Wc3Env``: the general interface. Observations are fixed-size feature arrays of
  the units the agent can see; actions are lists of ``protocol.Command`` objects
  (build them with the helpers on ``env.game``, a ``Wc3Game``). Reward: +1 win,
  -1 loss, 0 otherwise.
* ``MicroEnv``: scenario fights with a factored discrete action per own unit
  (noop / stop / move in one of 8 directions / attack an enemy slot), reward =
  damage balance + outcome. With ``targeting="semantic"`` the target is a rule (the
  weakest enemy in range, the nearest, ...) and stop becomes retreat; with ``abilities``
  heroes also cast (an ability slot head, targets by the same rules).
* ``NavigateEnv``: move one unit to a target point as fast as possible.
* ``MicroSelfPlayEnv``: two policies control the two sides of a scenario (dict API).

Every env exposes the raw ``Observation`` as ``info["obs"]``.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import Wc3Game
from .data.objects import combat_stats, unit_vocabulary
from .protocol import (HERO_ABILITY_SLOTS, Command, ImmediateOrder, Observation, PointOrder, Result, TargetOrder,
                       Unit, UnitFlags)
from .runtime.instance import GameSetup
from .scenario import Scenario

UNIT_FEATURES = 32
ABILITY_FEATURES = 10  # per hero ability slot, after the UNIT_FEATURES (UnitEncoder ability_slots)
RELATIONAL_FEATURES = 5  # UnitEncoder relational: the unit against the opposing side (last)
PLAYER_FEATURES = 8
_N_FLAGS = 12
# what UnitEncoder puts in each column (shown with the runs on the dashboard)
UNIT_FEATURE_NAMES = (
    "is own", "is ally", "is enemy", "is neutral", "x (/1500 from the centre)", "y (/1500 from the centre)",
    "hit points (share)", "max hit points (/1000)", "mana (share)", "max mana (/1000)", "facing sin", "facing cos",
    *(f"flag {f.name.lower()}" for f in UnitFlags if f is not UnitFlags.DEAD),
    "current order (none/move/attack/harvest/other, /4)", "hit points lost last step (share)",
    "hit points lost over the last 4 steps (share)", "type: attack range (/1000)", "type: damage per second (/40)",
    "type: armor (/10)", "type: speed (/500)", "type: attack cooldown (/3 s)", "type: hits air")
ABILITY_FEATURE_NAMES = ("level (/3)", "ready to cast", "cooldown left (/20 s)", "cast on a unit", "cast on a point",
                         "cast instantly", "for enemies", "for allies", "cast range (/1000)", "area (/500)")
RELATIONAL_FEATURE_NAMES = ("nearest opponent distance (/1000)", "opponents within its reach (/5)",
                            "opponents that reach it (/5)", "is its side's weakest", "seconds it survives their damage (/20)")


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

    HP_HISTORY = 4  # steps of hit point history per unit (1 s at 0.25 s steps)

    def __init__(self, player: int, origin: tuple[float, float], extent: float, ability_slots: int = 0,
                 relational: bool = False):
        self.player = player
        self.origin = origin
        self.extent = extent
        self.vocab = unit_vocabulary()
        self.stats = combat_stats()
        self._hp: dict[int, list[int]] = {}  # unit -> hit points of the last HP_HISTORY encodes
        # heroes' abilities: per slot ABILITY_FEATURES after the unit features
        self.ability_slots = ability_slots
        # relational features: what an MLP otherwise has to work out from positions and types
        self.relational = relational
        self.features = UNIT_FEATURES + ABILITY_FEATURES * ability_slots + (RELATIONAL_FEATURES if relational else 0)

    def encode(self, units: Sequence[Unit], obs: Observation, orders: dict[str, int], max_units: int,
               allies: set[int] = frozenset(), opponents: Sequence[Unit | None] = ()
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """`opponents`: the other side's units, for the relational features."""
        feats = np.zeros((max_units, self.features), dtype=np.float32)
        types = np.zeros(max_units, dtype=np.int64)
        ids = np.zeros(max_units, dtype=np.int64)
        mask = np.zeros(max_units, dtype=bool)
        ox, oy = self.origin
        for i, u in enumerate(units[:max_units]):
            if u is None:  # an empty slot (the unit died): masked, all zero
                continue
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
            # recent damage (who is being attacked right now): hit point change over the last step
            # and over the last HP_HISTORY steps, as a fraction of the maximum
            hist = self._hp.get(u.id)
            if hist is not None:
                f[24] = (u.hp - hist[-1]) / max(u.max_hp, 1)
                f[25] = (u.hp - hist[0]) / max(u.max_hp, 1)
            st = self.stats.get(u.type)  # what the unit type fights like
            if st is not None:
                f[26] = st.range / 1000.0
                f[27] = st.dps / 40.0
                f[28] = st.armor / 10.0
                f[29] = st.speed / 500.0
                f[30] = st.cooldown / 3.0
                f[31] = float(st.hits_air)
            if self.ability_slots and u.abilities:
                self._encode_abilities(u, f)
            if self.relational:
                self._encode_relations(u, units, opponents, f)
            types[i] = self.vocab.get(u.type, 0)
            ids[i] = u.id
            mask[i] = True
        for u in units[:max_units]:
            if u is None:
                continue
            hist = self._hp.setdefault(u.id, [u.hp] * self.HP_HISTORY)
            hist.append(u.hp)
            del hist[0]
        return feats, types, ids, mask

    def _encode_relations(self, u: Unit, side: Sequence[Unit | None], opponents: Sequence[Unit | None],
                          f: np.ndarray) -> None:
        """Distance to the nearest opponent, opponents within the unit's attack range, opponents
        that have it within theirs, whether it is its side's weakest unit (focus fire picks it),
        and the seconds it would survive the damage of those opponents (the reason to pull back)."""
        o = self.features - RELATIONAL_FEATURES
        foes = [e for e in opponents if e is not None]
        if not foes:
            return
        st = self.stats.get(u.type)
        reach = (st.range if st else 100.0) + REACH
        dist = [e.dist(u.x, u.y) for e in foes]
        threats = [e for e, d in zip(foes, dist)
                   if d <= ((s.range if (s := self.stats.get(e.type)) else 100.0) + REACH)]
        incoming = sum(s.dps for e in threats if (s := self.stats.get(e.type)) is not None)
        f[o] = min(min(dist) / 1000.0, 2.0)
        f[o + 1] = sum(d <= reach for d in dist) / 5.0
        f[o + 2] = len(threats) / 5.0
        f[o + 3] = float(u.hp <= min(v.hp for v in side if v is not None))
        f[o + 4] = min(u.hp / incoming / 20.0, 1.0) if incoming > 0 else 1.0

    def _encode_abilities(self, u: Unit, f: np.ndarray) -> None:
        """Per learned ability slot: level, ready to cast, cooldown left, how it is cast (unit /
        point / instant; passive: none), for enemies or allies, range and area."""
        from .data.abilities import ability_info, hero_abilities

        info_by_code = ability_info()
        for k, code in enumerate(hero_abilities().get(u.type, ())[:self.ability_slots]):
            if k >= len(u.abilities):
                break
            level, cooldown = u.abilities[k]
            info = info_by_code.get(code)
            if not level or info is None:
                continue
            o = UNIT_FEATURES + k * ABILITY_FEATURES
            f[o] = level / 3.0
            f[o + 1] = float(ability_ready(u, k, info))
            f[o + 2] = min(cooldown / 20.0, 1.0)
            if info.cast in ("unit", "point", "instant"):
                f[o + 3 + ("unit", "point", "instant").index(info.cast)] = 1.0
            f[o + 6] = float(info.side == "enemy")
            f[o + 7] = float(info.side in ("ally", "self"))
            f[o + 8] = info.at(info.range, level) / 1000.0
            f[o + 9] = info.at(info.area, level) / 500.0


def ability_ready(u: Unit, slot: int, info) -> bool:
    """Whether hero `u` can cast its ability in `slot` now: learned, castable, off cooldown, mana."""
    if info is None or not info.castable or slot >= len(u.abilities):
        return False
    level, cooldown = u.abilities[slot]
    return level > 0 and cooldown <= 0 and u.mana >= info.at(info.mana, level)


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

    spawner = None  # scenarios: rng -> protocol.QueueSpawn list, the units of each new episode

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """options: {"relaunch": True} starts the episode in a fresh game process."""
        super().reset(seed=seed)
        spawns = self.spawner(self.np_random) if self.spawner is not None else ()
        obs = self.game.reset(relaunch=bool((options or {}).get("relaunch")), spawns=spawns)
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


def _slot_units(obs: Observation, player: int, slots: tuple[list[int], list[int]], max_own: int,
                max_enemy: int) -> tuple[list[Unit | None], list[Unit | None]]:
    """Own and enemy units by slot. A unit keeps the slot it got when it first appeared (by id at
    the start of the episode) for the whole episode; a dead unit leaves its slot empty (None).
    Compacting the living units instead would shift everyone behind a dead unit into another
    slot, so "attack enemy 1" would suddenly mean a different unit. `slots` is updated."""
    own_ids, enemy_ids = slots
    alive = {u.id: u for u in obs.units if u.alive}
    for ids, cap, mine in ((own_ids, max_own, True), (enemy_ids, max_enemy, False)):
        new = sorted(u.id for u in alive.values() if (u.owner == player) == mine
                     and (mine or u.owner in obs.players) and u.id not in ids)
        ids.extend(new[:max(0, cap - len(ids))])
    return [alive.get(i) for i in own_ids], [alive.get(i) for i in enemy_ids]


def _issue(view, attacking: dict[int, int], unit: Unit, kind: int, direction: int, target: Unit | None,
           move_distance: float) -> None:
    """One unit's micro action: 0 noop, 1 stop, 2 move `move_distance` in 8-way `direction`,
    3 attack `target`. An attack on the unit the unit is already attacking is not issued again:
    a new order restarts the approach and the attack wind-up, so a unit re-ordered every step
    (0.25 s) could stand still or never land a hit."""
    if kind == 1:
        view.stop(unit)
    elif kind == 2:
        a = direction * math.pi / 4
        view.move(unit, unit.x + move_distance * math.cos(a), unit.y + move_distance * math.sin(a))
    elif kind == 3 and target is not None:
        if attacking.get(unit.id) == target.id and unit.order == view.order_id("attack"):
            return
        view.attack(unit, target)
        attacking[unit.id] = target.id
        return
    if kind in (1, 2):
        attacking.pop(unit.id, None)


# Attack targets as rules (MicroEnv targeting="semantic"), for the unit given the order. "In range":
# within its attack range (+ REACH); rules without a candidate fall back to the nearest enemy.
SEMANTIC_TARGETS = ("weak_in_range", "nearest", "weakest", "hero", "threat")
REACH = 90.0  # attack ranges count from the attacker's edge, positions are centers
# tactical mode: a retreat goes on (re-ordered every step) while the unit keeps losing hit points,
# for at most this long (game seconds): pulling a unit out pays only if it stays out until the
# enemies switch targets, which step-by-step exploration rarely does. (A fixed 1.5 s commitment
# kept units out too long: the pull-back script dropped from 68% to 51%.)
RETREAT_MAX = 3.0


class MicroEnv(Wc3Env):
    """Scenario fights. Action per own-unit slot: [kind, direction, target].

    kind 0 noop, 1 stop, 2 move `move_distance` in direction (8-way), 3 attack enemy slot `target`.
    With targeting="semantic": kind 1 retreats (moves `move_distance` straight away from the
    nearest enemy) and `target` picks a rule from SEMANTIC_TARGETS: the weakest enemy in range,
    the nearest, the weakest overall, a hero, or the biggest threat in range (the highest DPS
    per hit point left: killing it removes the most damage soonest).
    With abilities (semantic targeting only): [kind, direction, target, ability]; kind 4 casts
    the hero's ability in slot `ability` (data.abilities.hero_abilities). An ability for enemies
    goes to the enemy the target rule picks among those within its cast range; one for allies
    to an own unit by the same rules (weakest = lowest share of hit points; nearest = another
    unit; threat = the one with the most DPS); instant ones need no target. A cast that is not
    possible (not learned, cooling down, too little mana, no target) does nothing. Unit features
    get ABILITY_FEATURES per hero ability slot. With `opponent_casts` the scripted opponent's
    heroes cast too (scripted_cast).
    Reward: (enemy hp lost - own hp lost) / initial total hp per step, +1/-1 for win/loss.
    """

    def __init__(self, scenario: Scenario | None = None, max_own: int = 12, max_enemy: int = 12,
                 move_distance: float = 250.0, opponent: str = "scripted", name: str = "micro0",
                 targeting: str = "slot", abilities: bool = False, opponent_casts: bool | None = None,
                 relational: bool = False, tactical: bool = False, kill_reward: float = 0.0,
                 rejoin: bool = False, **kw):
        from .runtime.instance import Agent, Idle, Scripted

        self.scenario = scenario or Scenario.skirmish(["hfoo"] * 4, ["hfoo"] * 4)
        opp = {"scripted": Scripted, "idle": Idle, "agent": Agent}[opponent]
        setup = kw.pop("setup", None) or GameSetup(slots=[Agent("human"), opp("orc")], scenario=self.scenario)
        super().__init__(setup, player=0, max_units=max_own + max_enemy, name=name, **kw)
        self.max_own, self.max_enemy = max_own, max_enemy
        self.move_distance = move_distance
        if targeting not in ("slot", "semantic"):
            raise ValueError(f"targeting must be 'slot' or 'semantic', not {targeting!r}")
        self.targeting = targeting
        if abilities and targeting != "semantic":
            raise ValueError("abilities need targeting='semantic' (cast targets are rules)")
        self.abilities = abilities
        self.opponent_casts = abilities if opponent_casts is None else opponent_casts
        self.kind_names = ("noop", "retreat", "move", "attack") if targeting == "semantic" else \
            ("noop", "stop", "move", "attack")
        if abilities:
            self.kind_names += ("cast",)
        self.group = 4 if abilities else 3  # action heads per unit
        self.relational = relational
        # added to the reward per enemy unit killed and subtracted per own unit lost: a unit pulled
        # out of a fight pays off as a death that doesn't happen, sooner than the outcome
        self.kill_reward = kill_reward
        # tactical action masks (semantic targeting): retreat only for a hurt unit (below half its
        # hit points) that is losing hit points, no plain moves; exploring either elsewhere costs
        # a lot and teaches little
        self.tactical = tactical or rejoin
        # tactical: a unit whose pull-back ended and that is told nothing (noop) attack-moves back
        # into the fight; otherwise it stands where the retreat left it, out of reach, and pulling
        # back only pays when an attack order follows (without focus fire: 4% wins against 59%)
        self.rejoin = rejoin
        feat = (UNIT_FEATURES + (ABILITY_FEATURES * HERO_ABILITY_SLOTS if abilities else 0)
                + (RELATIONAL_FEATURES if relational else 0))
        self.observation_space = spaces.Dict({
            "own": spaces.Box(-np.inf, np.inf, (max_own, feat), np.float32),
            "own_types": spaces.Box(0, 10_000, (max_own,), np.int64),
            "own_mask": spaces.MultiBinary(max_own),
            "enemy": spaces.Box(-np.inf, np.inf, (max_enemy, feat), np.float32),
            "enemy_types": spaces.Box(0, 10_000, (max_enemy,), np.int64),
            "enemy_mask": spaces.MultiBinary(max_enemy),
            "time": spaces.Box(0, np.inf, (1,), np.float32),
        })
        n_targets = len(SEMANTIC_TARGETS) if targeting == "semantic" else max_enemy
        heads = [len(self.kind_names), 8, n_targets] + ([HERO_ABILITY_SLOTS] if abilities else [])
        self.action_space = spaces.MultiDiscrete(np.tile(heads, (max_own, 1)))
        self._own: list[Unit] = []
        self._enemy: list[Unit] = []
        self._hp0 = (1.0, 1.0)
        self._hp_prev = (0.0, 0.0)
        self._attacking: dict[int, int] = {}  # unit -> enemy it was last ordered to attack
        self._slots: tuple[list[int], list[int]] = ([], [])  # unit ids by slot (see _slot_units)
        self._cast_orders: set[int] = set()  # order ids of hero abilities (a unit casting)

    def _split(self, obs: Observation) -> tuple[list[Unit | None], list[Unit | None]]:
        return _slot_units(obs, self.player, self._slots, self.max_own, self.max_enemy)

    def _on_reset(self, obs: Observation) -> None:
        cx, cy = self.scenario.resolved_center()
        self.encoder = UnitEncoder(self.player, (cx, cy), 1500.0,
                                   ability_slots=HERO_ABILITY_SLOTS if self.abilities else 0,
                                   relational=self.relational)
        if self.abilities:
            from .data.abilities import ability_order_strings
            self._cast_orders = {self.game._orders[o] for o in ability_order_strings() if o in self.game._orders}
        self._attacking = {}
        self._retreat_until: dict[int, float] = {}  # tactical: unit -> game time its retreat ends
        self._slots = ([], [])
        own, enemy = self._split(obs)
        own, enemy = [u for u in own if u], [u for u in enemy if u]
        self._hp0 = (max(sum(u.max_hp for u in own), 1), max(sum(u.max_hp for u in enemy), 1))
        self._hp_prev = (sum(u.hp for u in own), sum(u.hp for u in enemy))
        self._alive_prev = (len(own), len(enemy))

    def _encode(self, obs: Observation) -> dict[str, np.ndarray]:
        self._own, self._enemy = self._split(obs)
        orders = self.game._orders
        of, ot, _, om = self.encoder.encode(self._own, obs, orders, self.max_own, opponents=self._enemy)
        ef, et, _, em = self.encoder.encode(self._enemy, obs, orders, self.max_enemy, opponents=self._own)
        return {"own": of, "own_types": ot, "own_mask": om, "enemy": ef, "enemy_types": et, "enemy_mask": em,
                "time": np.array([obs.game_time / max(self.scenario.max_game_seconds, 1)], np.float32)}

    def target_slot(self, unit: Unit, target: int) -> int | None:
        """The enemy slot an attack with head value `target` goes to (None: no such enemy)."""
        if self.targeting == "slot":
            live = [j for j, e in enumerate(self._enemy) if e is not None]
            return target if target in live else None
        st = combat_stats().get(unit.type)
        return self._rule_target(unit, target, self._enemy, (st.range if st else 100.0) + REACH)

    @staticmethod
    def _rule_target(unit: Unit, rule_index: int, pool: Sequence[Unit | None], reach: float) -> int | None:
        """The index in `pool` (enemies) that target rule SEMANTIC_TARGETS[rule_index] picks for
        `unit`; "in range" is within `reach`. Rules without a candidate: the nearest enemy."""
        live = [j for j, e in enumerate(pool) if e is not None]
        if not live:
            return None
        stats = combat_stats()
        dist = {j: pool[j].dist(unit.x, unit.y) for j in live}
        nearest = min(live, key=dist.__getitem__)
        in_range = [j for j in live if dist[j] <= reach]
        rule = SEMANTIC_TARGETS[rule_index]
        if rule == "weak_in_range":
            return min(in_range, key=lambda j: pool[j].hp) if in_range else nearest
        if rule == "weakest":
            return min(live, key=lambda j: pool[j].hp)
        if rule == "hero":
            heroes = [j for j in live if (s := stats.get(pool[j].type)) is not None and s.is_hero]
            return min(heroes, key=dist.__getitem__) if heroes else nearest
        if rule == "threat":
            def threat(j: int) -> float:
                s = stats.get(pool[j].type)
                return (s.dps if s else 0.0) / max(pool[j].hp, 1)
            return max(in_range, key=threat) if in_range else nearest
        return nearest

    @staticmethod
    def _ally_target(unit: Unit, rule_index: int, pool: Sequence[Unit | None], reach: float) -> int | None:
        """The index in `pool` (the caster's side, itself included) an ability for allies goes to."""
        live = [j for j, a in enumerate(pool) if a is not None and a.dist(unit.x, unit.y) <= reach]
        if not live:
            return None
        me = next((j for j in live if pool[j].id == unit.id), live[0])
        frac = {j: pool[j].hp / max(pool[j].max_hp, 1) for j in live}
        rule = SEMANTIC_TARGETS[rule_index]
        if rule in ("weak_in_range", "weakest"):
            return min(live, key=frac.__getitem__)
        if rule == "hero":
            heroes = [j for j in live if pool[j].is_hero]
            return min(heroes, key=lambda j: pool[j].dist(unit.x, unit.y)) if heroes else me
        if rule == "threat":
            stats = combat_stats()
            return max(live, key=lambda j: (s.dps if (s := stats.get(pool[j].type)) else 0.0))
        others = [j for j in live if j != me]
        return min(others, key=lambda j: pool[j].dist(unit.x, unit.y)) if others else me

    def cast_command(self, unit: Unit, slot: int, rule: int, own: Sequence[Unit | None],
                     enemy: Sequence[Unit | None]) -> Command | None:
        """The order for hero `unit` to cast its ability in `slot` with target rule `rule` (`own`
        and `enemy`: units by slot as seen from the caster's side), None if it cannot."""
        from .data.abilities import ability_info, hero_abilities

        codes = hero_abilities().get(unit.type, ())
        info = ability_info().get(codes[slot]) if slot < len(codes) else None
        if not ability_ready(unit, slot, info):
            return None
        try:
            oid = self.game.order_id(info.order)
        except KeyError:
            return None
        level = unit.abilities[slot][0]
        if info.cast == "instant":
            return ImmediateOrder(unit.id, oid)
        reach = info.at(info.range, level) + REACH
        if info.side == "enemy":  # the rule picks among the enemies within cast range
            in_reach = [e if e is not None and e.dist(unit.x, unit.y) <= reach else None for e in enemy]
            j = self._rule_target(unit, rule, in_reach, reach)
            target = in_reach[j] if j is not None else None
        else:
            j = self._ally_target(unit, rule, own, reach)
            target = own[j] if j is not None else None
        if target is None:
            return None
        if info.cast == "unit":
            return TargetOrder(unit.id, oid, target.id)
        return PointOrder(unit.id, oid, target.x, target.y)

    def scripted_cast(self, unit: Unit, own: Sequence[Unit | None], enemy: Sequence[Unit | None],
                      smart: bool = False) -> tuple[int, int] | None:
        """A simple caster: (ability slot, target rule) of the first ready ability in slot order
        with a use now, else None. For enemies: the weakest enemy in cast range (instant ones:
        an enemy within their area); for allies: the most hurt own unit in range below 70% hit
        points; self buffs: below half hit points with an enemy within 500. Used by the scripted
        opponent and scripts. `smart`: instant area spells only with two enemies in the area,
        targeted ones on the biggest threat (DPS per hit point left), heals below half."""
        from .data.abilities import ability_info, hero_abilities

        info_by_code = ability_info()
        weak = SEMANTIC_TARGETS.index("weak_in_range")
        rule = SEMANTIC_TARGETS.index("threat") if smart else weak
        foes = [e for e in enemy if e is not None]
        for slot, code in enumerate(hero_abilities().get(unit.type, ())):
            info = info_by_code.get(code)
            if not ability_ready(unit, slot, info) or not info.in_builds:
                continue
            level = unit.abilities[slot][0]

            def near(d: float, n: int = 1) -> bool:
                return sum(e.dist(unit.x, unit.y) <= d for e in foes) >= n

            if info.cast == "instant":
                if info.side == "enemy" and near(max(info.at(info.area, level), 250.0), 2 if smart else 1):
                    return slot, weak
                # self buffs (Divine Shield) when hurt in a fight
                if info.side != "enemy" and near(500.0) and unit.hp < 0.5 * unit.max_hp:
                    return slot, weak
            elif info.side == "enemy":
                if near(info.at(info.range, level) + REACH):
                    return slot, rule
            elif info.side == "ally":
                reach = info.at(info.range, level) + REACH
                low = 0.5 if smart else 0.7
                if any(a is not None and a.dist(unit.x, unit.y) <= reach and a.hp < low * a.max_hp for a in own):
                    return slot, weak
        return None

    def _commands(self, action: np.ndarray) -> list[Command]:
        action = np.asarray(action).reshape(self.max_own, self.group)
        for i, unit in enumerate(self._own):
            if unit is None:
                continue
            kind, direction, target = (int(v) for v in action[i][:3])
            if self.tactical and self.retreating(unit):
                self._retreat(unit)  # a committed retreat goes on (the mask allows only noop)
                continue
            if self.rejoin and self._retreat_until.pop(unit.id, None) is not None and kind == 0:
                self._rejoin(unit)  # the pull-back is over
                continue
            if kind == 1 and self.targeting == "semantic":
                if self.tactical:
                    self._retreat_until[unit.id] = self._now() + RETREAT_MAX
                self._retreat(unit)
                continue
            if kind == 4:
                cmd = self.cast_command(unit, int(action[i][3]), target, self._own, self._enemy)
                if cmd is not None:
                    self.game.issue(cmd)
                    self._attacking.pop(unit.id, None)
                continue
            slot = self.target_slot(unit, target) if kind == 3 else None
            _issue(self.game, self._attacking, unit, kind, direction,
                   self._enemy[slot] if slot is not None else None, self.move_distance)
        if self.opponent_casts:
            self._opponent_casts()
        return []

    def _opponent_casts(self) -> None:
        """The scripted opponent's heroes cast what scripted_cast picks (the harness only
        attack-moves them); a hero already casting is left alone."""
        for unit in self._enemy:
            if unit is None or not unit.is_hero or unit.order in self._cast_orders:
                continue
            choice = self.scripted_cast(unit, self._enemy, self._own)
            if choice is not None:
                cmd = self.cast_command(unit, choice[0], choice[1], self._enemy, self._own)
                if cmd is not None:
                    self.game.issue(cmd)

    def _now(self) -> float:
        obs = self.game.obs
        return obs.game_time if obs is not None else 0.0

    def retreating(self, unit: Unit) -> bool:
        """Tactical mode: whether `unit` is in a committed retreat: it chose to retreat less than
        RETREAT_MAX ago and it is still losing hit points (over the last two steps)."""
        until = getattr(self, "_retreat_until", {}).get(unit.id)
        if until is None or self._now() >= until:
            return False
        hist = self.encoder._hp.get(unit.id) if self.encoder is not None else None
        return bool(hist) and len(hist) >= 3 and unit.hp < hist[-3]

    def _rejoin(self, unit: Unit) -> None:
        live = [e for e in self._enemy if e is not None]
        if live:
            e = min(live, key=lambda e: e.dist(unit.x, unit.y))
            self.game.attack_move(unit, e.x, e.y)
            self._attacking.pop(unit.id, None)

    def _retreat(self, unit: Unit) -> None:
        live = [e for e in self._enemy if e is not None]
        if not live:
            return
        e = min(live, key=lambda e: e.dist(unit.x, unit.y))
        d = max(e.dist(unit.x, unit.y), 1.0)
        self.game.move(unit, unit.x + (unit.x - e.x) / d * self.move_distance,
                       unit.y + (unit.y - e.y) / d * self.move_distance)
        self._attacking.pop(unit.id, None)

    def step(self, action):
        self._commands(action)
        obs = self.game.step()
        reward, terminated, truncated = self._reward(obs)
        return self._encode(obs), reward, terminated, truncated, {"obs": obs}

    def _reward(self, obs: Observation) -> tuple[float, bool, bool]:
        own = [u for u in obs.units if u.alive and u.owner == self.player]
        enemy = [u for u in obs.units if u.alive and u.owner != self.player and u.owner in obs.players]
        own_hp, enemy_hp = sum(u.hp for u in own), sum(u.hp for u in enemy)
        dealt = (self._hp_prev[1] - enemy_hp) / self._hp0[1]
        taken = (self._hp_prev[0] - own_hp) / self._hp0[0]
        self._hp_prev = (own_hp, enemy_hp)
        kills = 0.0
        if self.kill_reward:
            prev = getattr(self, "_alive_prev", (len(own), len(enemy)))
            kills = self.kill_reward * ((prev[1] - len(enemy)) - (prev[0] - len(own)))
            self._alive_prev = (len(own), len(enemy))
        outcome, terminated, truncated = self._outcome(obs)
        return float(dealt - taken + kills + outcome), terminated, truncated


class MirrorSelfPlayEnv(MicroEnv):
    """Self-play with MicroEnv's full action set (semantic targets, abilities, masks, ...): this
    env plays player 0, `sides[1]` (a copy bound to player 1's view of the same game, with its own
    unit slots, encoder and order state) plays player 1. Dict API like MicroSelfPlayEnv; both see
    the fight from their own side and the rewards are zero-sum. The observations and actions are
    those of the single-agent MicroEnv, so one policy plays both sides and also the scripted
    opponent."""

    possible_agents = (0, 1)

    def __init__(self, scenario: Scenario | None = None, **kw):
        from .runtime.instance import Agent

        scenario = scenario or Scenario.skirmish(["hfoo"] * 4, ["hfoo"] * 4)
        setup = GameSetup(slots=[Agent("human"), Agent("orc")], scenario=scenario)
        super().__init__(scenario, setup=setup, opponent_casts=False, **kw)
        self.sides: dict[int, MicroEnv] = {0: self}

    def _side(self, player: int) -> MicroEnv:
        side = copy.copy(self)
        side.player = player
        side.game = self.game.as_player(player)
        side._slots, side._attacking, side.encoder, side.sides = ([], []), {}, None, None
        return side

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs0, info = super().reset(seed=seed, options=options)
        side = self.sides[1] = self._side(1)
        side._on_reset(info["obs"])
        return {0: obs0, 1: side._encode(info["obs"])}, {0: info, 1: info}

    def step(self, actions: dict[int, np.ndarray]):
        for p, side in self.sides.items():
            side._commands(actions[p])
        obs = self.game.step()
        encoded, rewards, terminated, truncated = {}, {}, {}, {}
        for p, side in self.sides.items():
            rewards[p], terminated[p], truncated[p] = side._reward(obs)
            encoded[p] = side._encode(obs)
        return encoded, rewards, terminated, truncated, {p: {"obs": obs} for p in self.sides}


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
        self._attacking: dict[int, int] = {}  # unit -> enemy it was last ordered to attack
        self._slots: dict[int, tuple[list[int], list[int]]] = {}  # per player: unit ids by slot

    def observation_space(self, agent: int) -> spaces.Space:
        return self.single_observation_space

    def action_space(self, agent: int) -> spaces.Space:
        return self.single_action_space

    def _side(self, obs: Observation, player: int) -> tuple[list[Unit | None], list[Unit | None]]:
        return _slot_units(obs, player, self._slots.setdefault(player, ([], [])), self.max_units, self.max_units)

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
        self._attacking = {}
        self._slots = {}
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
                if unit is None:
                    continue
                kind, direction, target = (int(v) for v in action[i])
                _issue(view, self._attacking, unit, kind, direction,
                       enemy[target] if kind == 3 and target < len(enemy) else None, self.move_distance)
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
