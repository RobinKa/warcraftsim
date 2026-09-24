import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from warcraftsim.puffer import bc
from warcraftsim.puffer.policy import PufferPolicy

PUFFER_DIR = Path(bc.__file__).parent


def test_fit_exports_the_trainer_network(tmp_path):
    try:
        py = bc._torch_python()
    except SystemExit:
        pytest.skip("no Python with torch")
    # a tiny dataset: 2 unit groups of (kind, direction, target) heads, 6 episodes
    rng = np.random.default_rng(0)
    ends = np.cumsum(rng.integers(5, 12, 6))
    n = int(ends[-1])
    np.savez(tmp_path / "game0.npz", obs=rng.normal(size=(n, 20)).astype(np.float32),
             act=rng.integers(0, 4, (n, 6)).astype(np.int16), rew=rng.normal(size=n).astype(np.float32),
             live=np.ones((n, 2), bool), ends=ends.astype(np.int32))
    (tmp_path / "meta.json").write_text(json.dumps({
        "task": "t", "policy": "p", "win_rate": 0.5, "obs_size": 20, "act_sizes": [4, 8, 5, 4, 8, 5],
        "group_size": 3, "detail_heads": {"2": 1, "3": 2}}))
    out = tmp_path / "policy.bin"
    subprocess.run([py, str(PUFFER_DIR / "bc_train.py"), str(tmp_path), str(out), "--epochs=2", "--hidden=16"],
                   check=True, capture_output=True)
    # the exported weights evaluated by the trainer's layout (numpy PufferPolicy) = the torch net
    obs = rng.normal(size=(9, 20)).astype(np.float32)
    np.save(tmp_path / "obs.npy", obs)
    script = f"""
import sys, numpy as np, torch
sys.path.insert(0, {str(PUFFER_DIR)!r})
import bc_train
net = bc_train.PufferNet(20, [4, 8, 5, 4, 8, 5], 16, 2)
w = np.fromfile({str(out)!r}, np.float32)
pos = 0
for p in (net.encoder.weight, net.decoder.weight, *(l.weight for l in net.gru)):
    k = p.numel()
    p.data = torch.from_numpy(w[pos:pos + k].reshape(p.shape).copy())
    pos = (pos + k + 7) & ~7
assert pos == len(w), (pos, len(w))
with torch.no_grad():
    np.save({str(tmp_path / 'want.npy')!r}, net(torch.from_numpy(np.load({str(tmp_path / 'obs.npy')!r}))[None])[0].numpy())
"""
    subprocess.run([py, "-c", script], check=True, capture_output=True)
    pol = PufferPolicy(out, 20, (4, 8, 5, 4, 8, 5), hidden=16, layers=2)
    state = pol.initial_state()
    got = np.stack([pol.step(o, state) for o in obs])
    np.testing.assert_allclose(got, np.load(tmp_path / "want.npy"), rtol=1e-4, atol=1e-4)
