"""Games between agents held even by who is expected to win (selfplay --win-balance; the user's idea, 2026-10-23:
"a tax on self play depending on which side is expected to win"; the races' balance, League.balance, held a level
per pair of races, blind to the targets).

A logistic model of a game's outcome for side A against side B (selfplay's episode rows of games between agents:
the learner's self-play and its games against past snapshots): from the two sides' races, whether each plays toward a
target and what the target holds (an expansion, a third tier, a tavern hero), whether B is a past snapshot, and the
share of A's income taken (< 0: of B's). The features are A's minus B's (no intercept: an outcome's chances do not
depend on which side is called A). Fitted on the newest WINDOW games (Newton's method, a Gaussian prior: the
features' weights at 0, the tax's at PRIOR_BETA, about what fgself-85's AI bonus did). A new game's tax: the share
whose effect cancels the side's expected edge, -(w . x) / beta, with a little noise (JITTER: the tax's effect stays
measurable), at most --win-balance; the favoured side is the taxed one. Before MIN_GAMES the races' balance stays.

In a game between two targets, a target's own pull (one that costs a policy its games: 35% won with a target in
fgself-134's self-play) is evened out: the policy learns to win while following it, not to drop it."""

from __future__ import annotations

from collections import deque

import numpy as np

RACES = ("human", "orc", "undead", "nightelf")
EXPANSIONS = {"htow", "ogre", "unpl", "etol", "ugol"}  # (a target's second hall or haunted mine: an expansion)
TIER3 = {"hcas", "ofrt", "unp2", "etoe"}
NAMES = ("human", "orc", "undead", "night elf", "a target", "a target's expansion", "a target's tier 3",
         "a target's tavern hero", "against a past snapshot")
WINDOW, MIN_GAMES, JITTER = 3000, 150, 0.05
PRIOR_W, PRIOR_BETA, PRIOR_BETA_SD = 1.0, -3.0, 1.5


def side_features(race: str, target: dict | None) -> np.ndarray:
    """One side's: its race (one-hot) and its target (none: zeros)."""
    f = np.zeros(8)
    if race in RACES:
        f[RACES.index(race)] = 1.0
    if target:
        codes = set(target.get("t") or {})
        f[4] = 1.0
        f[5] = float(bool(codes & EXPANSIONS))  # (a hall in its items: a second one, the first stands from the start)
        f[6] = float(bool(codes & TIER3))
        f[7] = float(any(c[:1] == "N" for c in codes))
    return f


def features(race_a: str, race_b: str, target_a: dict | None, target_b: dict | None, past_b: bool) -> np.ndarray:
    """A game's features for side A: A's side features minus B's, then whether B is a past snapshot."""
    return np.concatenate([side_features(race_a, target_a) - side_features(race_b, target_b), [float(past_b)]])


def tax(coef: dict | None, x, rng) -> float:
    """The share of A's income to take in a game with features `x` (< 0: of B's), from the model in the league's
    spec (`coef`: Balance.coef); None without a model yet."""
    if not coef:
        return None
    w, beta = np.asarray(coef["w"]), float(coef["beta"])
    if beta > -0.1:  # (no measurable effect of the tax yet)
        return None
    t = -float(np.dot(w, x)) / beta + rng.uniform(-coef.get("jitter", JITTER), coef.get("jitter", JITTER))
    return float(np.clip(t, -coef["max"], coef["max"]))


class Balance:
    """The model, refitted from the newest games between agents (the learner's process: League)."""

    def __init__(self, maximum: float, window: int = WINDOW):
        self.maximum = maximum
        self.rows: deque = deque(maxlen=window)  # (features, A's tax, A's score: 1 a win, 0.5 a tie, 0 a loss)
        self.w = np.zeros(len(NAMES))
        self.beta = PRIOR_BETA
        self.fitted = 0  # (games at the last fit)

    def add(self, x, tax_a: float, outcome: float) -> None:
        self.rows.append((np.asarray(x, float), float(tax_a), (float(outcome) + 1.0) / 2.0))
        if len(self.rows) >= MIN_GAMES and len(self.rows) - self.fitted >= 25:
            self.fit()

    def fit(self, iters: int = 12) -> None:
        """Newton's method on the prior-weighted log likelihood (scores as soft labels)."""
        X = np.array([np.append(x, t) for x, t, _ in self.rows])
        y = np.array([s for _, _, s in self.rows])
        k = X.shape[1]
        prior_mean = np.zeros(k)
        prior_mean[-1] = PRIOR_BETA
        prec = np.full(k, 1.0 / PRIOR_W ** 2)
        prec[-1] = 1.0 / PRIOR_BETA_SD ** 2
        theta = np.append(self.w, self.beta)
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(X @ theta, -30, 30)))
            grad = X.T @ (p - y) + prec * (theta - prior_mean)
            hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(prec)
            step = np.linalg.solve(hess, grad)
            theta = theta - step
            if np.abs(step).max() < 1e-6:
                break
        self.w, self.beta = theta[:-1], float(theta[-1])
        self.fitted = len(self.rows)

    def coef(self) -> dict | None:
        """The model for the actors (the league's spec), once fitted."""
        if self.fitted < MIN_GAMES:
            return None
        return {"w": [round(float(v), 4) for v in self.w], "beta": round(self.beta, 4), "max": self.maximum,
                "jitter": JITTER, "games": self.fitted}

    def summary(self) -> dict:
        """Each feature's weight (log-odds for A), the tax's, the games: the league.json and the train rows."""
        return {"weights": {n: round(float(v), 3) for n, v in zip(NAMES, self.w)}, "beta": round(self.beta, 3),
                "games": len(self.rows)}

    def restore(self, rows: list) -> None:
        for x, t, s in rows:
            self.rows.append((np.asarray(x, float), float(t), float(s)))
        if len(self.rows) >= MIN_GAMES:
            self.fit()
