"""The gradient noise scale's running means are kept in the run: a resume (or a fork, from its parent's) goes on
from them, not from scratch (every restart had made the estimate jump)."""

from warcraftsim.fullgame.selfplay import NoiseScale


def test_means_survive_a_restart(tmp_path):
    path = tmp_path / "noise_scale.json"
    a = NoiseScale(path=path)
    for g2, tr in ((0.01, 100.0), (0.02, 300.0), (0.015, 200.0)):
        out = a.add({"all": (g2, tr)}, 256)
    b = NoiseScale(path=path)  # (a resume)
    assert b.mean == a.mean
    c = NoiseScale(path=tmp_path / "fork.json", parent=path)  # (a fork: its parent's)
    assert c.mean == a.mean
    assert c.add({"all": (0.015, 200.0)}, 256)["grad/noise_scale"] > 0
    assert abs(out["grad/noise_scale"] - a.mean["all"][2] / a.mean["all"][1]) < 1e-6
