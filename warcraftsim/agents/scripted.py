"""A simple rule-based Human bot, used to exercise the melee pipeline end to end.

Economy: peasants mine gold (5) and cut trees (rest), the town hall trains peasants
up to a target, farms keep supply ahead, barracks train footmen, and the army
attack-moves to the enemy start location once it is large enough.
"""

from __future__ import annotations

import math
from collections import Counter

from ..client import Wc3Game
from ..protocol import EventKind, Observation, Unit


class HumanRushBot:
    GOLD_WORKERS = 5
    MAX_WORKERS = 14
    ATTACK_AT = 8

    def __init__(self, game: Wc3Game):
        self.game = game
        self.trees: dict[int, tuple[int, int]] = {}
        self.tree_type: int | None = None
        self.build_slots: list[tuple[float, float]] = []
        self.pending_builds: dict[int, float] = {}  # worker id -> game time of the order
        self.attacking = False

    def on_reset(self, obs: Observation) -> None:
        self.trees.clear()
        if obs.destructables:
            counts = Counter(d.type_id for d in obs.destructables)
            self.tree_type = counts.most_common(1)[0][0]
            self.trees = {d.id: (d.x, d.y) for d in obs.destructables if d.type_id == self.tree_type and d.life > 0}
        me = obs.players[self.game.player]
        hall = next(u for u in obs.units_of(self.game.player) if u.is_structure)
        mine = self.game.nearest_mine(hall)
        # candidate build spots on rings around the hall, away from the mine side
        away = math.atan2(hall.y - mine.y, hall.x - mine.x) if mine else 0.0
        self.build_slots = []
        for radius in (600, 850, 1100):
            for k in range(-4, 5):
                a = away + k * math.pi / 8
                self.build_slots.append((me.start_x + radius * math.cos(a), me.start_y + radius * math.sin(a)))
        self.pending_builds.clear()
        self.attacking = False

    def _nearest_tree(self, u: Unit) -> int | None:
        if not self.trees:
            return None
        return min(self.trees, key=lambda t: (self.trees[t][0] - u.x) ** 2 + (self.trees[t][1] - u.y) ** 2)

    def act(self, obs: Observation) -> None:
        g = self.game
        for e in obs.events:
            if e.kind == EventKind.TREE_DEATH:
                self.trees.pop(e.a, None)
        me = obs.players[g.player]
        units = obs.units_of(g.player)
        workers = [u for u in units if u.is_worker]
        halls = [u for u in units if u.type in ("htow", "hkee", "hcas")]
        barracks = [u for u in units if u.type == "hbar"]
        farms_building = sum(1 for u in units if u.type == "hhou" and u.flags & u.flags.CONSTRUCTING)
        footmen = [u for u in units if u.type == "hfoo"]
        gold, lumber = me.gold, me.lumber
        gold_order = g.order_id("harvest")
        miners = [w for w in workers if w.order in (gold_order, g.order_id("resumeharvesting"),
                                                     g.order_id("returnresources"))]
        # workers: gold first, then lumber
        n_gold = sum(1 for w in miners if w.flags & w.flags.HIDDEN) + len(miners)
        for w in workers:
            if not w.idle or w.id in self.pending_builds:
                continue
            if n_gold < self.GOLD_WORKERS * 2 and halls:
                mine = g.nearest_mine(w)
                if mine:
                    g.harvest(w, mine)
                    n_gold += 2
                    continue
            tree = self._nearest_tree(w)
            if tree is not None:
                g.harvest_tree(w, tree)
        # expire build orders that never started
        for wid, t in list(self.pending_builds.items()):
            if obs.game_time - t > 20:
                del self.pending_builds[wid]
        builder = next((w for w in workers if w.id not in self.pending_builds and not w.flags & w.flags.HIDDEN), None)
        # supply
        if builder and me.food_cap - me.food_used <= 4 and me.food_cap < 100 and farms_building == 0 \
                and gold >= 80 and lumber >= 20 and self.build_slots:
            x, y = self.build_slots.pop(0)
            g.build(builder, "hhou", x, y)
            self.pending_builds[builder.id] = obs.game_time
            gold -= 80
            lumber -= 20
            builder = None
        # barracks
        if builder and len(barracks) < 2 and gold >= 160 and lumber >= 60 and self.build_slots \
                and (len(barracks) == 0 or len(workers) >= 10):
            x, y = self.build_slots.pop(0)
            g.build(builder, "hbar", x, y)
            self.pending_builds[builder.id] = obs.game_time
            gold -= 160
            lumber -= 60
        # production
        for h in halls:
            if len(workers) < self.MAX_WORKERS and h.idle and gold >= 75 and me.food_cap - me.food_used >= 1:
                g.train(h, "hpea")
                gold -= 75
        for b in barracks:
            if b.idle and not b.flags & b.flags.CONSTRUCTING and gold >= 135 and me.food_cap - me.food_used >= 2:
                g.train(b, "hfoo")
                gold -= 135
        # army
        enemy = next((p for pid, p in obs.players.items() if pid != g.player), None)
        if len(footmen) >= self.ATTACK_AT:
            self.attacking = True
        if self.attacking and enemy:
            targets = [u for u in obs.enemies_of(g.player, visible_only=False)]
            for f in footmen:
                if f.idle:
                    if targets:
                        t = min(targets, key=lambda u: u.dist(f.x, f.y))
                        g.attack_move(f, t.x, t.y)
                    else:
                        g.attack_move(f, enemy.start_x, enemy.start_y)
