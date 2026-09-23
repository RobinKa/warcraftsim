"""Policy evaluation from a PufferLib checkpoint, and the replay video overlay."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from warcraftsim.protocol import ORDER_NAMES, PointOrder, TargetOrder, read_observation
from warcraftsim.puffer.policy import PufferPolicy
from warcraftsim.puffer.tasks import get_task
from warcraftsim.runtime.instance import Slot

FIXTURE = Path(__file__).parent.parent / "fixtures" / "obs_echoisles.txt"


def _checkpoint(path: Path, obs: int, acts: tuple, hidden: int, layers: int) -> None:
    n, pad = 0, lambda k: (k + 7) & ~7  # noqa: E731
    for size in [hidden * obs, (sum(acts) + 1) * hidden] + [3 * hidden * hidden] * layers:
        n = pad(n + size)
    np.random.default_rng(0).normal(0, 0.3, n).astype(np.float32).tofile(path)


def test_policy_checkpoint(tmp_path):
    ck = tmp_path / "0000000000001000.bin"
    _checkpoint(ck, obs=5, acts=(3, 4), hidden=8, layers=2)
    pol = PufferPolicy(ck, 5, (3, 4), hidden=8, layers=2)
    out = pol.run(np.random.default_rng(1).normal(size=(6, 5)))
    assert out.values.shape == (6,) and [p.shape for p in out.probs] == [(6, 3), (6, 4)]
    assert np.allclose([p.sum(axis=1) for p in out.probs], 1.0)
    assert out.entropy.shape == (6, 2) and (out.logp(np.zeros((6, 2))) <= 0).all()
    with pytest.raises(ValueError):
        PufferPolicy(ck, 6, (3, 4), hidden=8, layers=2)  # a different architecture does not fit


def test_overlay_frames():
    from warcraftsim.overlay import EpisodeOverlay

    task = get_task("micro_mirror")
    obs = read_observation(FIXTURE)
    T = 4
    trace = {"obs": np.zeros((T, 1, task.obs_size), np.float32),
             "actions": np.tile(np.array([3, 0, 1] * 6, np.float32), (T, 1, 1)),
             "rewards": np.full((T, 1), 0.1, np.float32), "outcomes": np.array([1.0])}
    fake = SimpleNamespace(values=np.linspace(0, 1, T), entropy=np.ones((T, task.num_atns)),
                           probs=[np.full((T, n), 1.0 / n) for n in task.act_sizes])
    ov = EpisodeOverlay(task, trace, [fake], title="test", policy_step=123456)
    setup = SimpleNamespace(scenario=None, slots=[Slot("agent", "human"), Slot("scripted", "orc")])
    ov.begin(setup, ORDER_NAMES, 480, 540)
    unit = next(u for u in obs.units if u.owner == 0)
    cmds = [PointOrder(unit.id, ORDER_NAMES.index("move"), 0, 0), TargetOrder(unit.id, 2, unit.id)]
    frames = ov.render_step([bytes(480 * 540 * 4)] * 3, 1, obs, obs, cmds)
    assert len(frames) == 3 and all(len(f) == (480 + ov.PANEL_W) * 540 * 3 for f in frames)
    assert len(ov.render_end(bytes(480 * 540 * 4), fps=10)) == 20
    assert ov.returns[0, 0] > ov.returns[-1, 0] > 0  # discounted return-to-go
