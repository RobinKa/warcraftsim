"""--obs-full for cloning people (fullgame/replays.py --obs-full, bc.py --like-run): a demonstration's events 101-109 and
items lying on the ground reach the View as a live game gives them (Encoder.encode's features are bc.live_encode's: the
units through play.unit_rows, the events in an observation's order, Observation.ground); a demonstration without them
leaves the block's state, corpse, item and area parts empty; bc.level_frame turns their places; bc.apply_run sets the
run's --obs-full."""
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from warcraftsim.fullgame import bc  # noqa: E402
from warcraftsim.fullgame import features as fx  # noqa: E402


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


VOCAB = {"types": [code(c) for c in ("hfoo", "hbar", "Hamg", "ogru", "obar", "ugho")], "current_orders": [],
         "upgrades": [], "orders": [[851983, fx.UNIT], [851986, fx.POINT], [851972, fx.IMMEDIATE]],
         "order_names": {"851983": "attack", "851986": "move", "851972": "stop"}}


@pytest.fixture()
def obs_on():
    try:
        from warcraftsim.fullgame import obsfull
        obsfull.tables()
    except Exception as e:  # noqa: BLE001 (the game's tables: not on every machine)
        pytest.skip(f"no game data: {e}")
    fx.set_inputs(2)
    fx.set_obs_full(1)
    yield
    fx.set_obs_full(0)
    fx.set_inputs(1)


def _unit(t, uid, typ, owner, x, y, flags=0, vis=3, mana=0, level=0):
    return [t, uid, code(typ), owner, x, y, 0, 400, 400, mana, 300 if mana else 0, 0, flags, vis, 0, level, 0, 0]


def _game(with_obs: bool = True) -> dict:
    """Four steps: a footman, an archmage and a barracks of player 0's, a grunt and a ghoul of player 1's (the ghoul dies
    at step 2: its corpse lies, then is gone at step 3), an item on the ground, a blizzard at step 2."""
    units, heroes, players = [], [], []
    for t in range(4):
        units += [_unit(t, 1, "hfoo", 0, -4000, 2900), _unit(t, 2, "hbar", 0, -4200, 2600, flags=fx.STRUCTURE),
                  _unit(t, 3, "Hamg", 0, -4100, 2900, flags=fx.HERO, mana=200, level=3),
                  _unit(t, 4, "ogru", 1, -3800, 2900), _unit(t, 5, "obar", 1, 3000, 2900, flags=fx.STRUCTURE, vis=2)]
        if t <= 2:
            units.append(_unit(t, 6, "ugho", 1, -3900, 2950, flags=fx.DEAD if t == 2 else 0))
        heroes.append([t, 3, code("phea"), code("pman"), 0, 0, 0, 0, 1, 0, 0, 0, 25 if t >= 2 else 0, 0, 0, 0])
        players += [[t, 0, 500, 200, 10, 30, 0, 0, 0, 2, 0], [t, 1, 400, 100, 8, 30, 0, 0, 0, 1, 0]]
    events = [(1, 5, 2, code("hfoo"), 0), (2, 1, 6, 0, code("ugho"))]
    game = {"units": np.array(units, np.int32), "heroes": np.array(heroes, np.int32),
            "players": np.array(players, np.int32), "events": np.array(events, np.int32).reshape(-1, 5),
            "orders": np.zeros((0, 7), np.int32), "trees": np.zeros((0, 5), np.int32),
            "meta": {"steps": 3, "races": ["human", "orc"], "step_seconds": 0.5}}
    if with_obs:
        speed_armor = (188 << 12) | (130 + 2048)
        obs = [(0, 101, 1, 2 | (1 << 8) | (2 << 16) | (1 << 20), speed_armor), (0, 102, 1, code("Binf"), 0),
               (0, 102, 1, code("Beng"), 1), (0, 101, 4, 8, (125 << 12) | 2118),
               (1, 101, 3, 0, (320 << 12) | 2093), (1, 105, 3, 1 | (2 << 10), 0),
               (2, 106, 6, -3900, 2950), (2, 108, 6, 3, 0), (2, 104, 3, -3850, 2900), (2, 109, 3, code("AHbz"), 3),
               (3, 107, 6, 0, 0)]
        game["obsfull"] = np.array(obs, np.int32)
        game["ground"] = np.array([(1, code("tdex"), -3950, 2800, 1, 1), (2, code("tdex"), -3950, 2800, 3, 1),
                                   (2, code("ratc"), 3000, 2800, 2, 0)], np.int32)
    return game


KEYS = ("ent", "type", "cur", "mask", "n_own", "glob", "avail")


@pytest.mark.parametrize("player", (0, 1))
def test_encode_feeds_the_view_as_a_live_game(obs_on, player):
    from warcraftsim.fullgame import obsfull as of
    enc = fx.Encoder(VOCAB)
    game = _game()
    a, b = enc.encode(game, player), bc.live_encode(fx.Encoder(VOCAB), game, player)
    for k in KEYS:
        assert np.array_equal(a[k], b[k]), k
    A, f = fx.A_OBS, a["ent"].astype(np.float32)
    kinds = f[..., A:A + 3]
    assert kinds[2, :, 0].sum() == 1  # (the corpse: both see its place, 108's bits 3)
    assert kinds[3, :, 0].sum() == 0  # (gone: 107)
    assert kinds[1, :, 1].sum() == (1 if player == 0 else 0)  # (the item lying: player 0 sees it at step 1)
    assert kinds[2, :, 1].sum() == (1 if player == 0 else 2)  # (step 2: both see the tome, player 1 its own ring)
    assert kinds[2, :, 2].sum() == 1  # (the blizzard)
    have = f[0, :, A + of.K_HAVE]
    assert have.sum() == 2  # (the units' state known: the footman's and the grunt's, both in sight of both)
    assert f[0, :, A + of.K_BUFF:A + of.K_BUFF + of.BUFF_D].any()  # (the footman's buffs)


def test_without_the_events_the_block_stays_empty(obs_on):
    from warcraftsim.fullgame import obsfull as of
    enc = fx.Encoder(VOCAB)
    game = _game(with_obs=False)
    for player in (0, 1):
        out = enc.encode(game, player)
        f = out["ent"].astype(np.float32)[..., fx.A_OBS:]
        assert not f[..., :of.K_UPG].any()  # (the kinds, classes, buffs' counts, armor, speed, known, buffs)
        assert not f[..., of.K_CHARGE:of.K_CHARGE + 6].any()  # (the charges)
        assert not f[..., of.K_AGE:].any()  # (a corpse's, an area's, an item's age; an area's radius)
        n = out["mask"].sum(1)
        assert (n <= 6).all()  # (the units only: no corpse, item or area rows)
        assert np.array_equal(out["ent"], bc.live_encode(fx.Encoder(VOCAB), game, player)["ent"])
    # what the old demonstrations have does fill its channels as a live game's: the hero's items (K_ITEM), its own
    # barracks' production after its train start (K_PROD)
    f = enc.encode(game, 0)["ent"].astype(np.float32)
    assert f[0, :, fx.A_OBS + of.K_ITEM:fx.A_OBS + of.K_ITEM + of.ITEM_D].any()
    assert f[2, :, fx.A_OBS + of.K_PROD + 1].any()


def test_level_frame_turns_the_corpses_spells_and_items():
    a, b = (-3000.0, -3000.0), (3000.0, 3000.0)
    rows = [[t, 100 + p, 1, p, x, y, 90, 100, 100, 0, 0, 0, fx.STRUCTURE, 3] + [0] * 4
            for t in range(2) for p, (x, y) in enumerate((a, b))]
    game = {"units": np.array(rows, np.int64), "orders": np.zeros((0, 7), np.int64), "trees": np.zeros((0, 5), np.int64),
            "meta": {"steps": 1},
            "obsfull": np.array([(0, 106, 9, -3000, -3000), (0, 104, 3, 3000, 3000), (0, 101, 3, 5, 7)], np.int64),
            "ground": np.array([(1, 77, -3000, -3000, 3, 0)], np.int64)}
    out, (na, nb) = bc.level_frame(game, a, b)
    xo, gr = out["obsfull"], out["ground"]
    assert (xo[0, 3], xo[0, 4]) == (round(na[0]), round(na[1]))  # (the corpse where player 0's hall is)
    assert (xo[1, 3], xo[1, 4]) == (round(nb[0]), round(nb[1]))  # (the spell's point at player 1's)
    assert xo[2].tolist() == [0, 101, 3, 5, 7]  # (a unit's state: no place)
    assert (gr[0, 2], gr[0, 3]) == (round(na[0]), round(na[1])) and gr[0, 4] == 3
    assert math.isclose(nb[0] - na[0], math.hypot(6000, 6000), rel_tol=1e-6)
    assert game["obsfull"][0, 3] == -3000  # (the game itself unchanged)


def test_apply_run_sets_the_runs_obs_full():
    saved_env = dict(os.environ)
    saved = {k: v for k, v in vars(fx).items() if k.isupper()}
    try:
        bc.apply_run({"features": "55", "inputs": "2", "obs_full": "1", "neutral": "2", "map": "duelrush",
                      "z_modes": "0", "z_mix": "0", "z_comp": "0", "z_shop": "0"})
        assert fx.OBS_FULL == 1 and fx.ACTIVE_F == 55 and fx.A_OBS == fx.F - fx.OBS_F and fx.INPUTS == 2
        assert os.environ["WARCRAFTSIM_OBS_FULL"] == "1"
        bc.apply_run({"features": "55", "inputs": "2", "map": "duelrush"})  # (a run from before: none)
        assert fx.OBS_FULL == 0 and fx.A_OBS == -1
    finally:
        vars(fx).update(saved)
        os.environ.clear()
        os.environ.update(saved_env)
