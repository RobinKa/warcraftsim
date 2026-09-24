"""High-level Python API for controlling a Warcraft III game.

    from warcraftsim import Wc3Game, GameSetup, Agent, BuiltinAI

    with Wc3Game(GameSetup(slots=[Agent("human"), BuiltinAI("orc", "insane")])) as game:
        obs = game.reset()
        while not obs.game_over:
            for worker in game.idle_workers():
                game.harvest(worker, game.nearest_mine(worker))
            obs = game.step()

Orders are queued with the helper methods (or passed as Command objects to step())
and sent with the next step. Unit, player and order arguments accept objects or ids.
"""

from __future__ import annotations

import copy
from typing import Sequence, Iterable

from .protocol import (Build, Command, ImmediateOrder, LearnSkill, Observation, PointOrder, SetResources, Spawn,
                       TargetDestructable, TargetOrder, Unit, UnitFlags, UseItem, fourcc)
from .runtime.instance import GameInstance, GameSetup


def _uid(u: Unit | int) -> int:
    return u.id if isinstance(u, Unit) else int(u)


class Wc3Game:
    def __init__(self, setup: GameSetup | None = None, player: int | None = None, name: str = "game0",
                 **instance_kw):
        self.setup = setup or GameSetup()
        agents = self.setup.agent_players
        self.player = player if player is not None else (agents[0] if agents else 0)
        self.instance = GameInstance(self.setup, name=name, **instance_kw)
        self._shared: dict = {"obs": None, "first": None}
        self._queue: list[Command] = []
        self._orders: dict[str, int] = {}

    # ---- episode control ------------------------------------------------------------------

    def reset(self, relaunch: bool = False, spawns: Sequence[Command] = ()) -> Observation:
        """Start the game, or begin a new episode (relaunch=True: in a fresh game process).
        Scenarios: `spawns` (protocol.QueueSpawn) are the new episode's units, on top of the
        scenario's own (e.g. a random composition per episode)."""
        self._queue.clear()
        if self.instance.proc is None:
            obs = self.instance.start()
        else:
            obs = self.instance.restart(relaunch=relaunch)
        if spawns:
            obs = self.instance.respawn(spawns)
        self._shared["first"] = obs
        if obs.orders:
            self._orders.clear()
            self._orders.update(obs.orders)
        self.obs = obs
        return obs

    @property
    def obs(self) -> Observation | None:
        return self._shared["obs"]

    @obs.setter
    def obs(self, value: Observation | None) -> None:
        self._shared["obs"] = value

    @property
    def first_obs(self) -> Observation | None:
        return self._shared["first"]

    def step(self, commands: Iterable[Command] = ()) -> Observation:
        """Send queued orders plus `commands`, advance one step, return the new observation."""
        batch = self._queue + list(commands)
        self._queue.clear()
        self.obs = self.instance.step(batch)
        self._shared["obs"] = self.obs
        return self.obs

    def as_player(self, player: int) -> "Wc3Game":
        """A handle acting as another agent player (self-play): helpers and queries use `player`,
        orders go into the same queue and are sent by the next step() on any handle."""
        if player not in self.setup.agent_players:
            raise ValueError(f"player {player} is not an agent slot {self.setup.agent_players}")
        view = copy.copy(self)  # shares instance, queue, orders and the shared observation holder
        view.player = player
        return view

    def close(self) -> None:
        self.instance.close()

    def __enter__(self) -> "Wc3Game":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def game_time(self) -> float:
        return self.obs.game_time if self.obs else 0.0

    def set_speed(self, speed: float) -> None:
        self.instance.set_speed(speed)

    # ---- queries --------------------------------------------------------------------------

    def order_id(self, name: str | int) -> int:
        """Order id for an order string ("move", "stormbolt") or an ability code ("AHtb")."""
        if isinstance(name, int):
            return name
        if name in self._orders:
            return self._orders[name]
        if len(name) == 4:
            from .data.objects import ability_orders

            order = ability_orders().get(name, {}).get("Order")
            if order in self._orders:
                return self._orders[order]
        raise KeyError(f"unknown order {name!r} ({len(self._orders)} known)")

    def my_units(self, unit_type: str | None = None) -> list[Unit]:
        units = self.obs.units_of(self.player)
        return [u for u in units if unit_type is None or u.type == unit_type]

    def idle_workers(self) -> list[Unit]:
        return [u for u in self.my_units() if u.is_worker and u.idle and not u.flags & UnitFlags.HIDDEN]

    def mines(self) -> list[Unit]:
        return [u for u in self.obs.units if u.type == "ngol" and u.alive]

    def nearest_mine(self, unit: Unit) -> Unit | None:
        mines = self.mines()
        return min(mines, key=lambda m: m.dist(unit.x, unit.y)) if mines else None

    def enemies(self, visible_only: bool = True) -> list[Unit]:
        return self.obs.enemies_of(self.player, visible_only)

    @property
    def me(self):
        return self.obs.players[self.player]

    # ---- orders (queued until the next step) ----------------------------------------------

    def order(self, unit: Unit | int, order: str | int) -> None:
        self._queue.append(ImmediateOrder(_uid(unit), self.order_id(order)))

    def order_point(self, unit: Unit | int, order: str | int, x: float, y: float) -> None:
        self._queue.append(PointOrder(_uid(unit), self.order_id(order), x, y))

    def order_target(self, unit: Unit | int, order: str | int, target: Unit | int) -> None:
        self._queue.append(TargetOrder(_uid(unit), self.order_id(order), _uid(target)))

    def move(self, unit: Unit | int, x: float, y: float) -> None:
        self.order_point(unit, "move", x, y)

    def attack_move(self, unit: Unit | int, x: float, y: float) -> None:
        self.order_point(unit, "attack", x, y)

    def attack(self, unit: Unit | int, target: Unit | int) -> None:
        self.order_target(unit, "attack", target)

    def smart(self, unit: Unit | int, target: Unit | int) -> None:
        """Right-click on a unit."""
        self.order_target(unit, "smart", target)

    def stop(self, unit: Unit | int) -> None:
        self.order(unit, "stop")

    def hold(self, unit: Unit | int) -> None:
        self.order(unit, "holdposition")

    def harvest(self, worker: Unit | int, target: Unit | int) -> None:
        """Harvest from a gold mine (unit) - use harvest_tree for lumber."""
        self.order_target(worker, "harvest", target)

    def harvest_tree(self, worker: Unit | int, tree_id: int) -> None:
        self._queue.append(TargetDestructable(_uid(worker), self.order_id("harvest"), tree_id))

    def train(self, building: Unit | int, unit_type: str) -> None:
        """Train a unit / research a tech / upgrade a building: all are immediate orders by object id."""
        self._queue.append(ImmediateOrder(_uid(building), fourcc(unit_type)))

    research = train
    upgrade = train

    def build(self, worker: Unit | int, building: str, x: float, y: float) -> None:
        self._queue.append(Build(_uid(worker), fourcc(building), x, y))

    def learn(self, hero: Unit | int, ability: str) -> None:
        self._queue.append(LearnSkill(_uid(hero), fourcc(ability)))

    def cast(self, unit: Unit | int, order: str | int, target: Unit | int | None = None,
             x: float | None = None, y: float | None = None) -> None:
        """Cast an ability by order string ("thunderbolt" is Storm Bolt, "thunderclap"; see data.abilities)."""
        if target is not None:
            self.order_target(unit, order, target)
        elif x is not None and y is not None:
            self.order_point(unit, order, x, y)
        else:
            self.order(unit, order)

    def use_item(self, unit: Unit | int, slot: int, target: Unit | int | None = None,
                 x: float | None = None, y: float | None = None) -> None:
        self._queue.append(UseItem(_uid(unit), slot, x, y, _uid(target) if target is not None else 0))

    # ---- debug ----------------------------------------------------------------------------

    def set_resources(self, player: int, gold: int, lumber: int) -> None:
        self._queue.append(SetResources(player, gold, lumber))

    def spawn(self, player: int, unit_type: str, x: float, y: float) -> None:
        self._queue.append(Spawn(player, unit_type, x, y))
