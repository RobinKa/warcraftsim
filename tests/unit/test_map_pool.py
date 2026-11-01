"""--map-pool: each launch's map, the fast copies' names, the halls' places; without it, the launches as before."""

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from warcraftsim.data.duelmap import ECON_FAST, FAST_MAPS, map_rules, parse_echo_name
from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import late, macros, mapframe as mf, selfplay, teacher
from warcraftsim.fullgame.costs import order_costs

ECHO = ((-5184.0, 2944.0), (4672.0, 2944.0))
LAST_REFUGE = ((-3840.0, -3968.0), (3840.0, 3264.0))


@pytest.mark.parametrize("key,stock", FAST_MAPS.items())
def test_fast_names(key, stock):
    assert parse_echo_name(f"{key}fast2") == (stock, ECON_FAST)
    assert map_rules(f"{key}fast2") == map_rules("echofast2")
    assert late.layout(f"{key}fast2") == stock  # (no state of another map fits)
    assert selfplay.mine_scale(f"{key}fast2") == selfplay.mine_scale("echofast2") == 0.5
    assert late.pace(f"{key}fast2", "human") == late.pace("echofast2", "human")


def test_names_of_before_read_as_before():
    assert parse_echo_name("echofast2")[0] == "(2)EchoIsles" and parse_echo_name("echo100")[1] == map_rules("(2)EchoIsles")
    for other in ("(2)EchoIsles", "(2)LastRefuge-1.3", "duelfast", "xfast2", "fast2"):
        assert parse_echo_name(other) is None


def test_pool_checks_the_rules():
    assert mf.parse_pool("echofast2,lastrefugefast2:0.5", "echofast2") == [("echofast2", 1.0), ("lastrefugefast2", 0.5)]
    with pytest.raises(ValueError):
        mf.parse_pool("echofast2,(2)Amazonia", "echofast2")  # (the real game's rules: other costs)


def test_one_costs_table_serves_the_pool():
    vocab = {"types": [], "current_orders": [], "upgrades": [],
             "orders": [[int.from_bytes(b"hfoo", "big"), fx.IMMEDIATE], [int.from_bytes(b"hbar", "big"), fx.POINT]]}
    try:
        a = order_costs(vocab, "echofast2")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no game data: {e}")
    for key in FAST_MAPS:
        assert np.array_equal(a, order_costs(vocab, f"{key}fast2"))


def test_expansions_in_home_coordinates():
    keep = dict(macros.EXPAND_SPOTS), dict(teacher.EXPAND_SPOTS)
    try:
        f = mf.MapFrame(LAST_REFUGE, ECHO)
        mf.register_expansions("lastrefugefast2", f)
        mx, my = 1728, -4224
        ox, oy = mf.EXPANSIONS["(2)LastRefuge-1.3"][(mx, my)]
        hx, hy = f.to_home1(mx, my)
        sx, sy = macros.hall_spot(hx, hy, *ECHO[0])
        assert np.allclose(f.to_world1(sx, sy), (mx + ox, my + oy), atol=2.5)  # (where people build it, in the world)
        assert teacher.EXPAND_SPOTS[(hx, hy)] == macros.EXPAND_SPOTS[(hx, hy)]
        # Echo Isles' own places stay, and a map without a table falls back to 750 toward the start
        assert macros.hall_spot(-4864, -3200, *ECHO[0]) == (-4764, -2400)
    finally:
        macros.EXPAND_SPOTS.clear(), macros.EXPAND_SPOTS.update(keep[0])
        teacher.EXPAND_SPOTS.clear(), teacher.EXPAND_SPOTS.update(keep[1])


def _rows(rng, n):
    r = np.zeros((n, fx.ROW_COLS), np.int64)
    r[:, fx.C_ID] = np.arange(1, n + 1) + 1000
    r[:, fx.C_TYPE] = int.from_bytes(b"hfoo", "big")
    r[:, fx.C_OWNER] = rng.choice([0, 1], n)
    r[:, fx.C_X], r[:, fx.C_Y] = rng.integers(-6000, 6000, n), rng.integers(-5000, 5000, n)
    r[:, fx.C_FACING], r[:, fx.C_MAXHP], r[:, fx.C_HP] = rng.integers(0, 360, n), 400, 300
    r[:, fx.C_VIS] = 3
    return r


def test_a_turned_game_encodes_as_the_home_map_would(monkeypatch):
    """The same units, one game on the home map and one on a turned map through its frame: within a base's rigid stretch
    the same features, to the rounding of a position."""
    old = fx.STARTS, fx.BIN_RANGE
    fx.set_geometry(ECHO, 2.75)
    try:
        enc = fx.Encoder({"types": [int.from_bytes(b"hfoo", "big")], "current_orders": [], "upgrades": [], "orders": [[851983, fx.POINT]]})
        rng = np.random.default_rng(0)
        home = _rows(rng, 40)
        home[:, fx.C_X] = ECHO[0][0] + rng.integers(-2500, 2500, 40)
        home[:, fx.C_Y] = ECHO[0][1] + rng.integers(-2500, 2500, 40)
        f = mf.MapFrame(LAST_REFUGE, ECHO)
        ua = np.zeros((40, 32), np.int32)
        ua[:, mf.UA_FACING] = np.mod(home[:, fx.C_FACING] + round(f.deg), 360)
        world = [f.to_world1(x, y) for x, y in home[:, [fx.C_X, fx.C_Y]].tolist()]
        ua[:, mf.UA_X], ua[:, mf.UA_Y] = np.rint([w[0] for w in world]), np.rint([w[1] for w in world])
        f.rows(ua)
        turned = home.copy()
        turned[:, fx.C_X], turned[:, fx.C_Y], turned[:, fx.C_FACING] = ua[:, mf.UA_X], ua[:, mf.UA_Y], ua[:, mf.UA_FACING]
        assert np.abs(turned[:, [fx.C_X, fx.C_Y]] - home[:, [fx.C_X, fx.C_Y]]).max() <= 2
        a = enc.entities(home, 0, 1.0, ({}, set(), {}), set())[2]
        b = enc.entities(turned, 0, 1.0, ({}, set(), {}), set())[2]
        assert np.allclose(a[:, 3:5], b[:, 3:5], atol=2.5 / fx.SCALE)
    finally:
        fx.set_geometry(*old)


class _Game:
    """GameInstance's stand-in: records the launches' setups."""
    launches: list = []

    def __init__(self, setup, name="", timeout=0):
        self.setup = setup
        _Game.launches.append(setup)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def start(self):
        return SimpleNamespace(unit_array=np.zeros((0, 32), np.int32), units=[], players={}, destructables=None, ground=[], issued=[])

    def restart(self, relaunch=False):
        return self.start()


def _loop(tmp_path, monkeypatch, n_launches, **extra):
    """game_loop's chosen launches, play_one stubbed: [(the setup's map, the game object, cfg map, races, side)]."""
    from warcraftsim.runtime import instance
    _Game.launches = []
    (tmp_path / "vocab.json").write_text(json.dumps({"orders": []}))
    (tmp_path / "league_spec.json").write_text(json.dumps({"launch": [{"kind": "agents", "p": 1.0}],
                                                            "agents": [{"kind": "self", "name": "self", "p": 1.0}]}))
    cfg = {"seed": 0, "start_update": 0, "video_every": 0.0, "actors": 1, "vocab": str(tmp_path / "vocab.json"),
           "run_dir": str(tmp_path), "slot": 0, "games_per_actor": 1, "learner_pid": __import__("os").getppid(),
           "races": ["human", "orc", "undead", "nightelf"], "mirror": False, "map": "echofast2", "handicap": 100,
           "step_seconds": 0.5, "max_minutes": 1.0, "wait_floor_ms": 5, "scripted_reset": False, "native_obs": True,
           "games_per_process": 1, "agent_games_factor": 1, **extra}
    seen, stop = [], threading.Event()

    def play_one(g, obs, cfg_, vocab, races, side, opp, infer, out_q, wid, record=False, train=True, grpo=None):
        seen.append((g.setup.map, g, cfg_["map"], list(races), side))
        if len(seen) >= n_launches:
            stop.set()
        return {}

    monkeypatch.setattr(instance, "GameInstance", _Game)
    monkeypatch.setattr(selfplay, "play_one", play_one)
    monkeypatch.setattr(mf, "map_starts", lambda name: LAST_REFUGE if "lastrefuge" in name else ECHO)
    mf._frame.cache_clear()
    old = fx.STARTS, fx.BIN_RANGE
    fx.set_geometry(ECHO, 2.75)
    try:
        selfplay.game_loop(0, 0, cfg, None, None, stop, None)
    finally:
        fx.set_geometry(*old)
        mf._frame.cache_clear()
    return seen


def test_without_the_flag_launches_as_before(tmp_path, monkeypatch):
    seen = _loop(tmp_path, monkeypatch, 5)
    assert all(m == "echofast2" and cm == "echofast2" and type(g) is _Game for m, g, cm, *_ in seen)
    # (the draws as without the code: the races and side first from the game's generator)
    rng = selfplay.game_rng({"seed": 0, "start_update": 0, "run_dir": str(tmp_path)}, 0, 0)
    for _, _, _, races, side in seen:
        want = [rng.choice(["human", "orc", "undead", "nightelf"]), rng.choice(["human", "orc", "undead", "nightelf"])]
        assert races == want and side == rng.randrange(2)
        rng.random()  # (_choose's launch)
        rng.random()  # (_choose's opponent)
        rng.random()  # (opp["advise"]'s)


def test_with_the_flag_each_launch_draws_its_map_and_is_framed(tmp_path, monkeypatch):
    seen = _loop(tmp_path, monkeypatch, 24, map_pool=[["echofast2", 1.0], ["lastrefugefast2", 1.0]])
    maps = {m for m, *_ in seen}
    assert maps == {"echofast2", "lastrefugefast2"}
    for m, g, cm, *_ in seen:
        assert cm == m
        assert isinstance(g, mf.FramedGame) == (m == "lastrefugefast2")


def test_a_row_says_its_map_only_with_the_flag():
    src = open(selfplay.__file__).read()
    assert "cfg.get(\"map_pool\") or cfg.get(\"eval_map_pool\")" in src
