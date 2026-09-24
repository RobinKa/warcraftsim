"""Scripted micro policies for MicroEnv tasks: baselines, and demonstrations for behavior cloning.

Policies (per own unit, every step):
    noop      units fight on their own (auto-acquire)
    focus     everyone attacks the weakest enemy
    range     each unit attacks the weakest enemy within its own attack range (none: fights on
              its own), so melee units do not walk past the front line to reach a target
    sticky    range, but a unit keeps its target while that is alive and in reach (a new attack
              order restarts the attack wind-up)
    [base]pull<L>
              base (focus if omitted), but a unit below L% hit points that is being hit (lost
              hit points in the last second) and is not the healthiest walks away from the
              enemies until it is no longer being hit, then rejoins (the enemies switch to
              another target meanwhile); e.g. pull35, rangepull35, nooppull35

With semantic targeting (MicroEnv targeting="semantic") the same decisions are expressed with the
env's rules: focus = "weakest", range = "weak_in_range", walking away = retreat.
"""

from __future__ import annotations

import math

import numpy as np

from ..data.objects import combat_stats
from ..env import REACH, SEMANTIC_TARGETS

POLICIES = ("noop", "focus", "range", "sticky", "pull35", "pull50", "rangepull35", "nooppull35")


def micro_action(policy: str, env, state: dict, max_units: int) -> np.ndarray:
    """One step's action [max_units, 3] (kind, direction, target) for the env's current units.
    `state` carries memory between steps of an episode (start each episode with {})."""
    own, enemy = env._own, env._enemy  # by slot; None: dead
    semantic = getattr(env, "targeting", "slot") == "semantic"
    a = np.zeros((max_units, 3), np.int64)
    live = [i for i, e in enumerate(enemy) if e is not None]
    if not live or policy == "noop":
        return a
    base, pull, low = policy.partition("pull")
    base = base or "focus"
    stats = combat_stats()
    weakest = min(live, key=lambda i: enemy[i].hp)
    ex = sum(enemy[i].x for i in live) / len(live)
    ey = sum(enemy[i].y for i in live) / len(live)
    healthiest = max((u.hp for u in own if u is not None), default=0)
    for i, u in enumerate(own[:max_units]):
        if u is None:
            continue
        if base == "focus":
            a[i] = (3, 0, SEMANTIC_TARGETS.index("weakest") if semantic else weakest)
        elif base in ("range", "sticky"):
            st = stats.get(u.type)
            reach = (st.range if st else 100) + REACH
            near = [j for j in live if enemy[j].dist(u.x, u.y) <= reach]
            if near and semantic:
                a[i] = (3, 0, SEMANTIC_TARGETS.index("weak_in_range"))
            elif near:
                targets = state.setdefault("targets", {})
                kept = [j for j in near if enemy[j].id == targets.get(u.id)] if base == "sticky" else []
                j = kept[0] if kept else min(near, key=lambda j: enemy[j].hp)
                a[i] = (3, 0, j)
                targets[u.id] = enemy[j].id
        if not pull:
            continue
        hist = state.setdefault(("hp", u.id), [u.hp] * 4)
        hit = u.hp < hist[0]
        hist.append(u.hp)
        del hist[0]
        if hit and u.hp < int(low) / 100 * u.max_hp and u.hp < healthiest:
            if semantic:
                a[i] = (1, 0, 0)  # retreat
            else:
                ang = math.atan2(u.y - ey, u.x - ex)
                a[i] = (2, round(ang / (math.pi / 4)) % 8, 0)
    return a
