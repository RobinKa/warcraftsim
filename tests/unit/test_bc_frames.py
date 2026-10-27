"""Cloning people's games on several maps (fullgame/bc.py --like-run): each game in its own frame, labels the run's
masks rule out left out, the strategy block's extra parts read with zero weights."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from warcraftsim.fullgame import bc  # noqa: E402
from warcraftsim.fullgame import features as fx  # noqa: E402


def _game(starts, extra_units=()):
    """Two players' halls at `starts` (step 0 and 1), a point order, a target order, an immediate one and a tree."""
    rows = []
    for t in range(2):
        for p, (x, y) in enumerate(starts):
            rows.append([t, 100 + p, 1, p, x, y, 90, 100, 100, 0, 0, 0, fx.STRUCTURE, 3] + [0] * 4)
        for uid, p, x, y in extra_units:
            rows.append([t, uid, 2, p, x, y, 0, 100, 100, 0, 0, 0, 0, 3] + [0] * 4)
    orders = [[0, 100, 5, 1, 1000, 2000, 0], [0, 100, 6, 2, 300, 400, 7], [1, 100, 8, 0, 0, 0, 0]]
    return {"units": np.array(rows, np.int64), "orders": np.array(orders, np.int64),
            "trees": np.array([[7, 0, 300, 400, 50]], np.int64), "meta": {"steps": 1}}


def test_level_frame_turns_the_starts_level():
    a, b = (-3000.0, -3000.0), (3000.0, 3000.0)  # (a diagonal map: Last Refuge, Amazonia)
    g = _game([a, b])
    out, (na, nb) = bc.level_frame(g, a, b)
    assert na[1] == pytest.approx(nb[1]) and nb[0] - na[0] == pytest.approx(math.hypot(6000, 6000))
    u = out["units"]
    # the halls sit at the new starts, the facing turned with the map (by -45 degrees)
    assert (u[0, fx.C_X], u[0, fx.C_Y]) == (round(na[0]), round(na[1]))
    assert (u[1, fx.C_X], u[1, fx.C_Y]) == (round(nb[0]), round(nb[1]))
    assert u[0, fx.C_FACING] == 45
    # point and target orders' points and the trees turned alike (distances kept); an immediate order untouched
    o = out["orders"]
    d0 = math.hypot(1000 - a[0], 2000 - a[1])
    assert math.hypot(o[0, 4] - na[0], o[0, 5] - na[1]) == pytest.approx(d0, abs=1.5)
    assert (o[1, 4], o[1, 5]) == tuple(out["trees"][0, 2:4])
    assert o[2, 4:6].tolist() == [0, 0]
    assert g["units"][0, fx.C_X] == -3000  # (the game itself unchanged)


def test_level_frame_keeps_a_level_map():
    g = _game([(-5184, 2944), (4672, 2944)])  # (Echo Isles)
    out, starts = bc.level_frame(g, (4672, 2944), (-5184, 2944))
    assert out is g and starts == ((-5184.0, 2944.0), (4672.0, 2944.0))


def test_used_starts_of_a_four_player_map():
    starts = ((-4544.0, -4032.0), (-3584.0, 4480.0), (4544.0, 3584.0), (3904.0, -5056.0))  # (Twisted Meadows)
    g = _game([(4400, 3500), (-3500, 4300)])
    assert bc.used_starts(g, starts) == (starts[2], starts[1])
    assert bc.used_starts(_game([(4400, 3500), (4500, 3600)]), starts) is None  # (both nearest the same)


def test_drop_labels_the_mask_rules_out():
    T, O, C = 2, 3, 5
    out = {"y_order": np.array([[1, 2, 0], [3, 0, 4]], np.int16), "y_ptr": np.full((T, O), 2, np.int16),
           "y_x": np.full((T, O), 5, np.int16), "y_y": np.full((T, O), 5, np.int16), "avail": np.ones((T, C), bool)}
    out["avail"][0, 2] = False  # (step 0: order 2 masked)
    n = bc.drop_labels(out, drop=(4,))  # (order 4: never, e.g. a mercenary with --neutral 2)
    assert n == 2
    assert out["y_order"].tolist() == [[1, bc.IGNORE, 0], [3, 0, bc.IGNORE]]
    assert out["y_ptr"][0, 1] == -1 and out["y_x"][1, 2] == -1 and out["y_ptr"][0, 0] == 2


@pytest.mark.parametrize("stats", [True, False])
def test_losses_skip_ignored_labels(stats):
    from warcraftsim.fullgame.model import FullGameNet
    torch.manual_seed(0)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=32, layers=1).train()
    net.allowed[:] = True
    rng = np.random.default_rng(0)
    B, E, O = 4, 6, 3
    y = rng.integers(1, 5, (B, E))
    y[0, 0] = y[1, 2] = bc.IGNORE
    b = {"ent": torch.from_numpy(rng.normal(size=(B, E, fx.F)).astype(np.float32)),
         "type": torch.from_numpy(rng.integers(0, 20, (B, E))), "cur": torch.from_numpy(rng.integers(0, 10, (B, E))),
         "mask": torch.ones(B, E, dtype=torch.bool), "glob": torch.from_numpy(rng.normal(size=(B, 30)).astype(np.float32)),
         "n_own": torch.full((B,), O), "y_order": torch.from_numpy(y), "y_ptr": torch.full((B, E), -1),
         "y_x": torch.from_numpy(rng.integers(0, fx.BINS, (B, E))), "y_y": torch.from_numpy(rng.integers(0, fx.BINS, (B, E))),
         "avail": torch.ones(B, 30, dtype=torch.bool), "ret": torch.zeros(B)}
    loss, _ = bc.losses(net, b, torch.device("cpu"), None, value_coef=0.5, stats=stats)
    loss.backward()
    assert torch.isfinite(loss)


def test_zero_z_extras_reads_the_modes_and_mix_with_zero_weights():
    from warcraftsim.fullgame.model import FullGameNet
    from warcraftsim.fullgame.strategy import ZDIM
    z, gx, priv = ZDIM + 20 + 64, 3, 5
    G = 10 + z + gx + priv
    net = FullGameNet(n_types=4, n_cur=3, n_orders=5, G=G, d=16, layers=1, priv=priv, z=z, gx=gx)
    assert bc.zero_z_extras(net) == 84
    w = net.glob[0].weight
    end = G - priv - gx
    assert (w[:, end - 84:end] == 0).all() and (w[:, end - z:end - 84] != 0).any() and (w[:, end:] != 0).any()


def test_held_out_games_stay_as_the_directory_grows(tmp_path):
    from pathlib import Path
    paths = [Path(f"d/game{i:05d}.npz") for i in range(400)]
    val = bc.held_out(paths, 5.0)
    assert 5 < len(val) < 40 and bc.held_out(paths[:200], 5.0) == [p for p in val if p in paths[:200]]
