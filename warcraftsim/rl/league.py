"""A league of opponents for self-play (after AlphaStar's league and OpenAI Five's past selves).

In a self-play task every game has two sides; the learner always plays side 0. Games are split
into groups for the whole run:

    self    the learner also plays side 1: both sides' experience trains it
    past    side 1 is a past snapshot of the learner, chosen per episode by prioritized fictitious
            self-play (PFSP): snapshots the learner beats less often are chosen more often
    script  side 1 is a fixed scripted policy (noop, focus, pull35, amove: anchors that do not drift
            with the league, and yardsticks across runs; amove attack-moves at the nearest enemy, as the
            game's scripted opponent does: without such a chaser, self-play learned to run away)

Scripts decide from side 1's own observations (numpy), with general orders.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KINDS = ("noop", "stop", "hold", "move", "attack", "attack_move", "cast")


@dataclass
class Member:
    name: str
    path: str | None = None  # a snapshot's checkpoint; None for scripts
    steps: int = 0
    wins: float = 0.0
    losses: float = 0.0
    draws: float = 0.0
    recent: list = field(default_factory=list)  # the learner's last results against it (1 / 0.5 / 0)

    @property
    def games(self) -> float:
        return self.wins + self.losses + self.draws

    def win_rate(self, window: int = 50) -> float | None:
        r = self.recent[-window:]
        return sum(r) / len(r) if r else None

    def record(self, outcome: float) -> None:
        if outcome > 0.5:
            self.wins += 1
        elif outcome < -0.5:
            self.losses += 1
        else:
            self.draws += 1
        self.recent.append(1.0 if outcome > 0.5 else 0.0 if outcome < -0.5 else 0.5)
        del self.recent[:-200]


def pfsp_weight(p: float | None, mode: str = "hard") -> float:
    """AlphaStar's weighting of an opponent by the learner's win rate p against it."""
    if p is None:
        return 1.0  # not played yet: as likely as a hard one
    if mode == "hard":
        return (1 - p) ** 2
    if mode == "variance":
        return p * (1 - p)
    return 1.0


class League:
    def __init__(self, run_dir: Path, scripts: list[str], pfsp: str = "hard", max_past: int = 50):
        self.run_dir = Path(run_dir)
        self.scripts = {s: Member("script:" + s) for s in scripts}
        self.past: list[Member] = []
        self.self_member = Member("self")
        self.pfsp = pfsp
        self.max_past = max_past

    def add_snapshot(self, path: str, steps: int) -> Member:
        m = Member(f"past:{steps}", path=path, steps=steps)
        self.past.append(m)
        if len(self.past) > self.max_past:  # keep the first (a fixed early anchor) and the newest
            self.past.pop(1)
        return m

    def sample_past(self) -> Member | None:
        if not self.past:
            return None
        w = [max(pfsp_weight(m.win_rate(), self.pfsp), 1e-3) for m in self.past]
        return random.choices(self.past, weights=w)[0]

    def sample_script(self) -> Member:
        """Scripts are chosen by PFSP too: the ones the learner still loses to come up more."""
        members = list(self.scripts.values())
        w = [max(pfsp_weight(m.win_rate(), self.pfsp), 0.05) for m in members]  # every anchor stays in play
        return random.choices(members, weights=w)[0]

    def summary(self) -> dict:
        def row(m: Member) -> dict:
            p = m.win_rate()
            return {"name": m.name, "steps": m.steps, "games": m.games, "wins": m.wins, "losses": m.losses,
                    "draws": m.draws, "win_rate": p, "weight": pfsp_weight(p, self.pfsp) if m.path else None}
        return {"pfsp": self.pfsp, "members": [row(m) for m in [*self.scripts.values(), *self.past]],
                "self": row(self.self_member)}

    def save(self) -> None:
        tmp = self.run_dir / "league.json.tmp"
        tmp.write_text(json.dumps(self.summary(), indent=1))
        tmp.replace(self.run_dir / "league.json")


# ---- scripted opponents from observations -------------------------------------------------------
class Scripts:
    """General-order scripts computed from a side's observation vector (the micro layout)."""

    def __init__(self, spec: dict):
        blocks = spec["spaces"]["observation"]["blocks"]
        self.k = blocks[0]["rows"]
        feat = blocks[0]["features"]
        self.F = len(feat)
        self.i_x, self.i_y = feat.index("x (/1500 from the centre)"), feat.index("y (/1500 from the centre)")
        self.i_hp, self.i_maxhp = feat.index("hit points (share)"), feat.index("max hit points (/1000)")
        self.i_lost = feat.index("hit points lost over the last 4 steps (share)")
        self.i_order = feat.index("current order (none/move/attack/harvest/other, /4)")
        heads = spec["spaces"]["actions"]["heads"]
        self.n_dir = heads[1]["size"]
        self.group = len(heads)

    def act(self, name: str, obs: np.ndarray) -> np.ndarray:
        """obs [N, obs] (side 1's own view) -> actions [N, k * group]."""
        k, F = self.k, self.F
        N = obs.shape[0]
        own = obs[:, :k * F].reshape(N, k, F)
        own_alive = obs[:, k * F:k * F + k] > 0.5
        at = k * F + k
        enemy = obs[:, at:at + k * F].reshape(N, k, F)
        enemy_alive = obs[:, at + k * F:at + k * F + k] > 0.5
        a = np.zeros((N, k, self.group), np.int64)
        if name == "noop":
            return a.reshape(N, -1)
        if name == "amove":  # as the game's scripted opponent: an idle unit attack-moves at the nearest enemy
            dx = enemy[:, None, :, self.i_x] - own[..., None, self.i_x]
            dy = enemy[:, None, :, self.i_y] - own[..., None, self.i_y]
            dist = np.where(enemy_alive[:, None, :], dx ** 2 + dy ** 2, np.inf)
            j = dist.argmin(-1)
            ax = np.take_along_axis(dx, j[..., None], -1)[..., 0]
            ay = np.take_along_axis(dy, j[..., None], -1)[..., 0]
            d = np.round(np.arctan2(ay, ax) / (2 * math.pi / self.n_dir)).astype(np.int64) % self.n_dir
            # only idle units: a new order would cancel an attack in progress (re-ordering every unit
            # every step, the first version of this script hardly landed a hit)
            idle = own_alive & enemy_alive.any(-1)[:, None] & (own[..., self.i_order] < 0.125)
            a[..., 0] = np.where(idle, KINDS.index("attack_move"), 0)
            a[..., 1] = np.where(idle, d, 0)
            a[..., 2] = np.where(idle, 2, 0)  # the farthest step (700): toward the enemy, as far as it goes
            return a.reshape(N, -1)
        ehp = np.where(enemy_alive, enemy[..., self.i_hp] * enemy[..., self.i_maxhp], np.inf)
        weakest = ehp.argmin(-1)  # [N]
        any_enemy = enemy_alive.any(-1)
        attack = np.zeros_like(a)
        attack[..., 0] = KINDS.index("attack")
        attack[..., 3] = (k + weakest)[:, None]
        a = np.where((own_alive & any_enemy[:, None])[..., None], attack, a)
        if name.startswith("pull"):
            low = int(name[4:]) / 100 if name[4:].isdigit() else 0.35
            hp = own[..., self.i_hp]
            abs_hp = np.where(own_alive, hp * own[..., self.i_maxhp], -np.inf)
            hurt = own_alive & (hp < low) & (own[..., self.i_lost] < 0) & (abs_hp < abs_hp.max(-1, keepdims=True))
            # away from the nearest enemy
            dx = own[..., None, self.i_x] - enemy[:, None, :, self.i_x]
            dy = own[..., None, self.i_y] - enemy[:, None, :, self.i_y]
            dist = np.where(enemy_alive[:, None, :], dx ** 2 + dy ** 2, np.inf)
            j = dist.argmin(-1)
            ax = np.take_along_axis(dx, j[..., None], -1)[..., 0]
            ay = np.take_along_axis(dy, j[..., None], -1)[..., 0]
            d = np.round(np.arctan2(ay, ax) / (2 * math.pi / self.n_dir)).astype(np.int64) % self.n_dir
            move = np.zeros_like(a)
            move[..., 0] = KINDS.index("move")
            move[..., 1] = d
            move[..., 2] = 1
            a = np.where((hurt & any_enemy[:, None])[..., None], move, a)
        return a.reshape(N, -1)
