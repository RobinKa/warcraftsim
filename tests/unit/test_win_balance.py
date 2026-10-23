"""--win-balance (fullgame/balance.py): the logistic model of who wins a game between agents, fitted from games with
known edges, and the tax it gives (the favoured side's, the share that evens the game out)."""

import random

import numpy as np

from warcraftsim.fullgame.balance import MIN_GAMES, NAMES, Balance, features, side_features, tax


def test_features_are_a_side_minus_the_other():
    t = {"t": {"halt": 30.0, "hkee": 200.0, "htow": 400.0, "Npbm": 150.0}}
    assert side_features("human", t).tolist() == [1, 0, 0, 0, 1, 1, 0, 1]
    assert side_features("orc", None).tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    x = features("human", "orc", t, None, True)
    assert len(x) == len(NAMES) and x.tolist() == [1, -1, 0, 0, 1, 1, 0, 1, 1]
    assert (features("orc", "human", None, t, False)[:-1] == -x[:-1]).all()  # (the other side's: the opposite)


def test_fit_recovers_an_edge_and_the_tax_evens_it():
    rng = np.random.default_rng(0)
    true_w = np.zeros(len(NAMES))
    true_w[0], true_w[4] = -0.8, -1.0  # (human weaker; a target costs its side)
    true_beta = -4.0
    b = Balance(0.5)
    races = ("human", "orc", "undead", "nightelf")
    for _ in range(2500):
        ra, rb = rng.choice(races), rng.choice(races)
        za = {"t": {"x": 1.0}} if rng.random() < 0.5 else None
        zb = {"t": {"x": 1.0}} if rng.random() < 0.5 else None
        x = features(ra, rb, za, zb, False)
        t = rng.uniform(-0.3, 0.3)
        p = 1 / (1 + np.exp(-(true_w @ x + true_beta * t)))
        b.add(x, t, 1.0 if rng.random() < p else -1.0)
    assert b.fitted >= MIN_GAMES
    assert abs(b.w[0] - b.w[1] - (-0.8)) < 0.35 and abs(b.w[4] - (-1.0)) < 0.3 and abs(b.beta - true_beta) < 1.0
    coef = b.coef()
    x = features("human", "orc", {"t": {"x": 1.0}}, None, False)  # (A weaker by 1.8 log-odds: B is taxed)
    t = tax({**coef, "jitter": 0.0}, x, random.Random(0))
    assert -0.5 <= t < -0.2
    assert tax({**coef, "jitter": 0.0}, -x, random.Random(0)) == -t
    assert tax(None, x, random.Random(0)) is None and Balance(0.5).coef() is None  # (not fitted: the races' balance)
