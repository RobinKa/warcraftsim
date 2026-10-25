"""The actors' per-step work in C (round 4: featnative.view_step and side, the affordable mask, macros' natural
expansion, night elves' entangle) against the code it replaced, on random steps: the same views, masks and answers
to the bit (the hit points' change, the hero block and what each building makes included: an evaluation game's view
keeps every feature)."""

import numpy as np
import pytest

from warcraftsim.fullgame import featnative
from warcraftsim.fullgame import features as fx

pytestmark = pytest.mark.skipif(featnative.lib() is None, reason="no C compiler")


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


TYPES = [code(c) for c in ("hfoo", "hpea", "htow", "hbar", "halt", "Hpal", "Hamg", "ogru", "ntav", "nkob", "ngol",
                           "egol", "etol", "ugol", "uaco")]


def _rows(rng, ids: np.ndarray, t: int) -> np.ndarray:
    n = len(ids)
    r = np.zeros((n, fx.ROW_COLS), np.int64)
    r[:, fx.C_STEP] = t
    r[:, fx.C_ID] = ids
    r[:, fx.C_TYPE] = rng.choice(TYPES, n)
    r[:, fx.C_OWNER] = rng.choice([0, 1, 12, 15], n)
    r[:, fx.C_X], r[:, fx.C_Y] = rng.integers(-6000, 6000, n), rng.integers(-6000, 6000, n)
    r[:, fx.C_FACING] = rng.integers(0, 360, n)
    r[:, fx.C_MAXHP] = rng.integers(0, 2000, n)
    r[:, fx.C_HP] = rng.integers(0, 2000, n)
    r[:, fx.C_MAXMANA] = rng.integers(0, 3, n) * 100
    r[:, fx.C_MANA] = rng.integers(0, 300, n)
    r[:, fx.C_ORDER] = rng.choice([0, 851983, 852018, 852024, code("hbar")], n)
    r[:, fx.C_FLAGS] = rng.choice([0, 1, 2, 4, 8, 16, 256, 512, 1024, 2 | 8, 4 | 16, 1 | 1024, 1 | 256, 2 | 1024], n)
    r[:, fx.C_VIS] = rng.integers(0, 4, n)
    r[:, fx.C_RESOURCE] = rng.integers(0, 12500, n)
    r[:, fx.C_HLEVEL] = rng.integers(0, 10, n)
    r[:, fx.C_SKILLPTS] = rng.integers(0, 3, n)
    heroes = (r[:, fx.C_FLAGS] & fx.HERO) > 0
    r[heroes, fx.C_ITEM0:fx.C_ITEM0 + 6] = rng.choice([0, 0, code("phea"), code("stwp"), code("pman"), 77], (heroes.sum(), 6))
    r[heroes, fx.C_ABIL0:fx.C_ABIL0 + 8] = rng.integers(0, 4, (heroes.sum(), 8)) * rng.integers(0, 2, (heroes.sum(), 8))
    return r


@pytest.fixture(params=[1, 2])
def inputs(request, monkeypatch):
    if request.param == 2:
        try:
            from warcraftsim.fullgame.herodata import hero_table, item_table
            hero_table(), item_table()
        except Exception as e:  # noqa: BLE001 (the game's tables: not on every machine)
            pytest.skip(f"no game data: {e}")
    monkeypatch.setattr(fx, "A_HID", -1)  # (another test's privileged channel: set_inputs keeps it)
    fx.set_inputs(request.param, False)
    yield request.param
    fx.set_inputs(1, False)


@pytest.mark.parametrize("active", ["all", 29])
def test_view_step_matches_the_old_path(inputs, active, monkeypatch):
    vocab = {"types": TYPES[:8], "current_orders": [851983, 852018], "upgrades": [],
             "orders": [[851983, fx.POINT], [code("hfoo"), fx.IMMEDIATE], [code("Hpal"), fx.IMMEDIATE],
                        [code("hbar"), fx.POINT]]}
    costs = np.array([[0, 0, 0], [0, 0, 0], [135, 0, 2], [425, 100, 5], [160, 60, 0]], np.int64)
    rng = np.random.default_rng(inputs)
    for game in range(6):
        player, sign = game % 2, (1.0, -1.0)[game % 2]
        views = [fx.Encoder(vocab, costs).view(player, sign, [0, 1]) for _ in range(2)]
        for v in views:
            v.active_f = fx.F if active == "all" else active
        ids = np.arange(1, 120) + 1000 * game
        for t in range(30):
            alive = ids[rng.random(len(ids)) < 0.85]
            rows = _rows(rng, alive, t)
            k = len(rows)
            events = np.array([(t, int(rng.choice([5, 6, 8, 9, 10, 11, 12])), int(rows[int(rng.integers(0, k)), fx.C_ID])
                                if k else 0, int(rng.choice([code("hfoo"), code("Hpal"), code("Rhde")])),
                                int(rng.integers(0, 3))) for _ in range(int(rng.integers(0, 4)))], np.int64).reshape(-1, 5)
            me = np.array([t, player, int(rng.integers(0, 600)), int(rng.integers(0, 200)), 10, 30, 0, 0, 0, 0, 0])
            opp = np.array([t, 1 - player, 100, 50, 5, 10, 1])
            for v in views:  # (the same production and harvest state in both)
                for b in rows[(rows[:, fx.C_OWNER] == player) & (rows[:, fx.C_FLAGS] & fx.STRUCTURE > 0), fx.C_ID][:3].tolist():
                    v.queued.setdefault(b, 1)
                for u in rows[(rows[:, fx.C_OWNER] == player), fx.C_ID][:5].tolist():
                    v.assign.setdefault(u, "lumber" if u % 2 else "gold")
                v.reserve = (2, 135, 0) if t % 3 == 0 else None
            fast = views[0].step(rows, me, events, t, opp)
            assert views[0]._marks is not None  # (the C path taken)
            monkeypatch.setattr(fx, "PRIVILEGED", True)  # (View.step's numpy masks and hit points; the entities' C)
            old = views[1].step(rows, me, events, t, opp)
            monkeypatch.setattr(fx, "PRIVILEGED", False)
            for key in ("n", "n_own", "first_free", "index", "ghost0"):
                assert fast[key] == old[key], key
            for key in ("sel", "ent", "type", "cur", "glob", "avail"):
                a, b = fast[key], old[key]
                assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), (key, t)
            a, b = views
            assert (a.own_buildings, a.queued, a.busy, a.making, a.dead_heroes, a.assign, a.levels, a.free_hero) == \
                (b.own_buildings, b.queued, b.busy, b.making, b.dead_heroes, b.assign, b.levels, b.free_hero)
            assert all(np.array_equal(x, y) for x, y in zip(a.hp_last, b.hp_last))
            # play.BCAgent's marks: the own entities' workers on a building order or repairing, their frames
            own = old["sel"][:old["n_own"]]
            w = own[own[:, fx.C_FLAGS] & fx.WORKER > 0]
            assert a._marks == (w[w[:, fx.C_ORDER] >= fx.TYPE_CODE, fx.C_ID].tolist(),
                                w[w[:, fx.C_ORDER] == 852024, fx.C_ID].tolist(),
                                own[own[:, fx.C_FLAGS] & fx.CONSTRUCTING > 0, fx.C_ID].tolist())


def test_affordable_in_c_matches_numpy():
    rng = np.random.default_rng(1)
    for trial in range(300):
        n = int(rng.integers(1, 40))
        costs = np.ascontiguousarray(rng.integers(0, 400, (n, 3)) * rng.integers(0, 2, (n, 3)), np.int64)
        gold, lumber, fu, fc = (int(v) for v in rng.integers(0, 500, 4))
        kept = [(int(rng.integers(0, n)), int(rng.integers(0, 300)), int(rng.integers(0, 300)))
                for _ in range(int(rng.integers(0, 3)))]
        want = fx.affordable(costs, gold, lumber, fu, fc, *(kept + [None, None])[:2])
        got = featnative.afford(costs, gold, lumber, fu, fc, kept)
        assert got.dtype == want.dtype and np.array_equal(got, want), (trial, kept)


def test_natural_in_c_matches_numpy(monkeypatch):
    from warcraftsim.fullgame import macros
    rng = np.random.default_rng(2)
    for trial in range(200):
        rows = _rows(rng, np.arange(1, int(rng.integers(1, 120))), 0)
        rows[rng.random(len(rows)) < 0.15, fx.C_TYPE] = code("ngol")
        rows[rng.random(len(rows)) < 0.1, fx.C_TYPE] = code("htow")
        start = (float(rng.integers(-3000, 3000)), float(rng.integers(-3000, 3000)))
        got = (macros.site(rows, start, True), macros.camp(rows, start, True))
        nat = macros.natural(rows, start)
        monkeypatch.setattr(featnative, "_lib", False)
        want = (macros.site(rows, start, True), macros.camp(rows, start, True))
        want_nat = macros.natural(rows, start)
        monkeypatch.setattr(featnative, "_lib", None)
        assert got == want, trial
        assert (nat is None and want_nat is None) or np.array_equal(nat, want_nat)


def test_entangle_in_c_matches_numpy(monkeypatch):
    from warcraftsim.fullgame import play
    rng = np.random.default_rng(3)
    for trial in range(200):
        rows = _rows(rng, np.arange(1, int(rng.integers(1, 80))), 0)
        bot = play.BCAgent(None, {"types": [], "current_orders": [], "upgrades": [], "orders": []}, 0, None)
        bot.t = int(rng.integers(0, 50))
        bot.entangle_sent = {int(i): int(rng.integers(0, 50)) for i in rows[:3, fx.C_ID]}
        sent = dict(bot.entangle_sent)
        got = bot._entangle(rows)
        got_sent, bot.entangle_sent = bot.entangle_sent, sent
        monkeypatch.setattr(featnative, "_lib", False)
        want = bot._entangle(rows)
        monkeypatch.setattr(featnative, "_lib", None)
        assert got == want and got_sent == bot.entangle_sent, trial
