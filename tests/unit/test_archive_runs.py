"""scripts/archive_runs.py: what stays here: a run's last checkpoint, those a later run names; only old runs move."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("archive_runs", ROOT / "scripts" / "archive_runs.py")
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def _run(root: Path, n: int, steps=(1, 2, 3)) -> None:
    d = root / f"fgself-{n}" / "checkpoints"
    d.mkdir(parents=True)
    for s in steps:
        (d / f"{s:016d}.pt").write_bytes(b"x")
        (d / f"{s:016d}.adam").write_bytes(b"x")


def test_plan_keeps_what_is_named_and_each_last(tmp_path):
    root = tmp_path / "runs"
    _run(root, 5)
    _run(root, 9)
    (root / "fgself-5" / "videos").mkdir()
    (root / "fgself-5" / "videos" / "a.mp4").write_bytes(b"x")
    (root / "fgself-5" / "videos" / "a.info.json").write_text("{}")
    named_ck = "runs/fgself-5/checkpoints/0000000000000001.pt"
    (root / "fgself-9" / "run.json").write_text(json.dumps({"config": {"init": named_ck}}))
    keep = archive.named(root, 5)
    assert keep == {named_ck}
    moved = archive.plan(root, 5, keep)
    assert named_ck not in moved
    assert "runs/fgself-5/checkpoints/0000000000000003.pt" not in moved  # (its last)
    assert "runs/fgself-5/checkpoints/0000000000000002.pt" in moved
    assert "runs/fgself-5/checkpoints/0000000000000003.adam" in moved
    assert "runs/fgself-5/videos/a.mp4" in moved
    assert "runs/fgself-5/videos/a.info.json" not in moved
    assert not any("fgself-9" in f for f in moved)  # (past --upto)
