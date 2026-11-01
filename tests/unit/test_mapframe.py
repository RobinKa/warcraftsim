"""Other maps in one process (fullgame/mapframe.py): their positions in the home map's frame, at the game's door."""

import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import mapframe as mf
from warcraftsim.protocol import Build, Destructable, IssuedOrder, PointOrder, TargetOrder, UseItem

ECHO = ((-5184.0, 2944.0), (4672.0, 2944.0))
LAST_REFUGE = ((-3840.0, -3968.0), (3840.0, 3264.0))  # (the stock maps' start locations)
AMAZONIA = ((3328.0, 3072.0), (-3392.0, -3584.0))
TERENAS = ((-4096.0, 3968.0), (4096.0, -4608.0))


@pytest.fixture
def echo_frame():
    old = fx.STARTS, fx.BIN_RANGE
    fx.set_geometry(ECHO, 2.75)
    yield
    fx.set_geometry(*old)


@pytest.mark.parametrize("starts", [LAST_REFUGE, AMAZONIA, TERENAS])
def test_bases_land_on_the_home_bases_and_the_frame_reads_them(starts, echo_frame):
    f = mf.MapFrame(starts, ECHO)
    (ax, ay), (bx, by) = sorted(starts)
    for (x, y), home in zip(((ax, ay), (bx, by)), ECHO):
        assert np.allclose(f.to_home(x, y), home, atol=1e-6)
    # each player's own base where the duel maps' left one is, the enemy's on the other side, on the same line
    for sign, own, other in ((1.0, (ax, ay), (bx, by)), (-1.0, (bx, by), (ax, ay))):
        assert np.allclose(fx.frame_xy(*f.to_home(*own), sign), (fx.ANCHOR_X, 0.0), atol=1e-9)
        assert np.allclose(fx.frame_xy(*f.to_home(*other), sign), (-fx.ANCHOR_X, 0.0), atol=1e-9)


@pytest.mark.parametrize("starts", [LAST_REFUGE, AMAZONIA, TERENAS])
def test_round_trip_and_near_a_base_distances_kept(starts):
    f = mf.MapFrame(starts, ECHO)
    rng = random.Random(0)
    for _ in range(200):
        x, y = rng.uniform(-6000, 6000), rng.uniform(-6000, 6000)
        assert np.allclose(f.to_world1(*f.to_home(x, y)), (x, y), atol=1e-6)
    (ax, ay), _ = sorted(starts)
    for _ in range(50):  # (within mf.RIGID of a base along the line: a turn and a shift, its own distances)
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2900)
        x, y = ax + r * math.cos(a), ay + r * math.sin(a)
        hx, hy = f.to_home(x, y)
        assert abs(math.hypot(hx - ECHO[0][0], hy - ECHO[0][1]) - r) < 1e-6


def test_left_stays_left_across_the_turn():
    """A half-turn symmetric map: the lower x start is the frame's left, whichever slot the w3i lists first."""
    assert np.allclose(mf.MapFrame(AMAZONIA, ECHO).to_home(-3392, -3584), ECHO[0], atol=1e-6)
    assert np.allclose(mf.MapFrame(tuple(reversed(AMAZONIA)), ECHO).to_home(3328, 3072), ECHO[1], atol=1e-6)


def test_too_close_bases_refused():
    with pytest.raises(ValueError):
        mf.MapFrame(((-2000.0, 0.0), (2000.0, 0.0)), ECHO)


def test_observation_in_place_and_commands_back(echo_frame):
    f = mf.MapFrame(LAST_REFUGE, ECHO)
    ua = np.zeros((2, 32), np.int32)
    ua[0, [mf.UA_X, mf.UA_Y, mf.UA_FACING]] = (-3840, -3968, 90)
    ua[1, [mf.UA_X, mf.UA_Y, mf.UA_FACING]] = (3840, 3264, 10)
    players = {0: SimpleNamespace(start_x=-3840, start_y=-3968), 1: SimpleNamespace(start_x=3840, start_y=3264)}
    obs = SimpleNamespace(unit_array=ua, units=None, players=players, destructables=[Destructable(7, 0, -3840, -3968, 50)],
                          ground=[(1, 3840, 3264, 3, 0)], issued=[IssuedOrder(1, 2, 1, -3840, -3968, 0), IssuedOrder(1, 2, 0, 0, 0, 0)])
    assert f.observation(obs) is obs
    assert ua[0, mf.UA_X] == -5184 and ua[0, mf.UA_Y] == 2944 and ua[1, mf.UA_X] == 4672
    assert ua[0, mf.UA_FACING] == round(90 - f.deg) % 360 and ua[1, mf.UA_FACING] == round(10 - f.deg) % 360
    assert (players[0].start_x, players[0].start_y, players[1].start_x) == (-5184, 2944, 4672)
    d = obs.destructables[0]
    assert (d.x, d.y) == (-5184, 2944) and obs.ground[0][1:3] == (4672, 2944)
    assert (obs.issued[0].x, obs.issued[0].y) == (-5184, 2944) and (obs.issued[1].x, obs.issued[1].y) == (0, 0)
    sent = [PointOrder(1, 2, -5184.0, 2944.0), Build(1, 2, 4672.0, 2944.0), TargetOrder(1, 2, 3), UseItem(1, 0),
            UseItem(1, 0, -5184.0, 2944.0)]
    back = f.commands(sent)
    assert np.allclose((back[0].x, back[0].y), (-3840, -3968)) and np.allclose((back[1].x, back[1].y), (3840, 3264))
    assert back[2] is sent[2] and back[3] is sent[3] and np.allclose((back[4].x, back[4].y), (-3840, -3968))
    assert sent[0].x == -5184.0  # (the caller's list keeps home's: the views read them)


def test_framed_game_turns_both_ways():
    calls = {}

    class Game:
        setup = "the setup"
        game_wall_total = 1.5

        def _obs(self):
            return SimpleNamespace(unit_array=np.zeros((0, 32), np.int32), units=[], players={}, destructables=None,
                                   ground=[], issued=[])

        def start(self):
            return self._obs()

        def restart(self, relaunch=False):
            calls["relaunch"] = relaunch
            return self._obs()

        def step(self, commands=()):
            calls["step"] = list(commands)
            return self._obs()

    f = mf.MapFrame(AMAZONIA, ECHO)
    g = mf.FramedGame(Game(), f)
    assert g.setup == "the setup" and g.game_wall_total == 1.5 and hasattr(g, "game_wall_total")
    g.start()
    g.step([PointOrder(1, 2, -5184.0, 2944.0)])
    assert np.allclose((calls["step"][0].x, calls["step"][0].y), (-3392, -3584))
    g.restart(relaunch=True)
    assert calls["relaunch"] is True


def test_pool_form_and_draw():
    pool = mf.parse_pool("echofast2,lastrefugefast2:0.25, amazoniafast2:0.5")
    assert pool == [("echofast2", 1.0), ("lastrefugefast2", 0.25), ("amazoniafast2", 0.5)]
    rng = random.Random(1)
    n = [mf.draw(rng, pool) for _ in range(7000)]
    assert abs(n.count("echofast2") / 7000 - 1 / 1.75) < 0.03 and abs(n.count("lastrefugefast2") / 7000 - 0.25 / 1.75) < 0.03
    for bad in ("", "a:0", "a:-1"):
        with pytest.raises(ValueError):
            mf.parse_pool(bad)
